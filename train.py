import os
import sys
import torch
import datetime
import argparse
import contextlib
import io
import math
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)

from tensorboardX import SummaryWriter

import tinycudann as tcnn

from configs import get_args
from model import TCNNModel
from dataset import TextureDataset
from Common.Material.loss import MaterialLoss
from Common.Material.reconstruction import canonical_gt_mip
from configs import Config, write_resolved_config
from Common.Comparison.material import (log_comparison_storage, _compute_group_metrics,
                                        run_material_comparison_pipeline, evaluate_materials, primary_method,
                                        complete_material_evaluation)
from Common.Material.compression import MaterialCompression
from Common.Comparison.pbr_scene import run_pbr_comparison_pipeline
from dataset import material_size
from Core.Feature_Grid.training import FeatureGridTraining
from Core.Feature_Grid.optimization import make_scheduler


class PhaseTrainingControl:
    STATE_KEYS = ('phase', 'start', 'threshold', 'best_score', 'best_iteration', 'best_state',
                  'previous_score', 'bad_windows', 'lr_reductions')

    def state_dict(self):
        return {key: getattr(self, key) for key in self.STATE_KEYS}

    def load_state_dict(self, state):
        for key in self.STATE_KEYS:
            setattr(self, key, state[key])

    def __init__(self, phase, start, threshold=0.01):
        self.phase, self.start = phase, start
        self.threshold = threshold
        self.best_score = -math.inf
        self.best_iteration = None
        self.best_state = None
        self.previous_score = None
        self.bad_windows = 0
        self.lr_reductions = 0

    def observe(self, model, score, iteration, allow_stop=True, adjust_lr=True):
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite {self.phase} validation score: {score}")
        if score > self.best_score:
            self.best_score, self.best_iteration = score, iteration
            self.best_state = {key: value.detach().cpu().clone()
                               for key, value in model.state_dict().items()}
        if not adjust_lr:
            return False
        previous = self.previous_score
        self.previous_score = score
        if previous is None or score - previous >= self.threshold:
            self.bad_windows = 0
            return False
        if self.phase == "decoder":
            group = model.optimizer.param_groups[0]
            reduced = min(group["lr"], max(0.0001, group["lr"] * 0.25))
            if reduced < group["lr"] * (1 - 1e-8):
                group["lr"] = reduced
                self.lr_reductions += 1
                self.bad_windows = 0
                print(f"[Decoder LR] iter {iteration}: lr={reduced:.7g}", flush=True)
                return False
        self.bad_windows += 1
        return allow_stop and self.bad_windows >= 2

    def restore(self, model):
        if self.best_state is None:
            return False
        model.load_state_dict(self.best_state)
        model._qat_param_cache.clear()
        print(f"[Best Model] {self.phase}: restored iter {self.best_iteration}, "
              f"PSNR={self.best_score:.6f}", flush=True)
        return True

    def summary(self):
        return {"phase": self.phase, "start_iteration": self.start,
                "best_iteration": self.best_iteration,
                "best_psnr": self.best_score if self.best_iteration is not None else None,
                "lr_reductions": self.lr_reductions}


class Trainer(FeatureGridTraining):

    def __init__(self, params: argparse.Namespace, config_override: Config = None,
                 deployment_config: Config = None, checkpoint=None) -> None:

        # initialize required runtime fieldsvariables, e.g. dataset, configs and models
        if config_override is not None:
            configs = config_override
        else:
            configs = Config(params)
        from checkpoint import read_resume, TrainingProgress, data_contract, training_recipe
        if checkpoint is None:
            checkpoint = read_resume(configs)
        self._training_recipe = training_recipe(deployment_config or configs)
        self.progress = TrainingProgress(**checkpoint.state['progress']) if checkpoint else TrainingProgress()
        self._checkpoint_data = checkpoint.state['data'] if checkpoint else data_contract(configs.data_dir)
        startup_log = io.StringIO()
        try:
            with contextlib.redirect_stdout(startup_log):
                dataset = TextureDataset(configs)
                material_compression = MaterialCompression(configs.material_compression_formats,
                                                           configs.astcenc_path, configs.astcenc_quality,
                                                           tile_size=configs.fntc_tile_size)
                material_compression.validate_dataset(dataset)
                print(f"[NormalEncoding] mode={dataset.normal_encoding}")
                configs.num_mips = dataset.num_mips
                # Tile size is measured in decoder/material texels, independent of GT resolution.
                if not configs.pos_encoding_reference_edge:
                    configs.pos_encoding_reference_edge = max(dataset.material_reconstruction_size)
                configs.material_reconstruction_size = list(dataset.material_reconstruction_size)
                configs.material_cnn_inference_size = list(material_size(configs,dataset.texture_height,dataset.texture_width,
                                  resolution=configs.material_compare_resolution))
                if (deployment_config or configs).material_cnn_enable:
                    from Core.Tile_CNN.training import validate_training_size
                    validate_training_size((dataset.texture_height, dataset.texture_width),
                                           configs.material_cnn_inference_size)
                model = TCNNModel(configs, checkpoint=checkpoint.model_path if checkpoint else None)
                model.configure_direct_diffuse_from_dataset(dataset)
                configs.material_reconstruction_size = list(material_size(model, dataset.texture_height, dataset.texture_width))
                configs.num_channels = model.num_channels
            log_comparison_storage(model, dataset, material_compression, configs,
                                   deployment_config=deployment_config)
        finally:
            print(startup_log.getvalue(), end='')

        # initialize required runtime fields
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.max_iter = configs.max_iter
        self.trained_iter = 0
        self.eval_interval = configs.eval_interval
        self.save_interval = configs.save_interval
        self.quantize = configs.quantize

        # save and log
        self.save_dir = configs.save_dir
        self.start_time = datetime.datetime.now()
        self.last_checkpoint_time = self.start_time
        self.end_time = 0
        self.duration_time = 0
        self.save_path = os.path.join(self.save_dir, self.start_time.strftime(r"%y_%m_%d_%H_%M_%S_%f"))
        self.log_path = os.path.join(self.save_path, "tensorboard")
        self.model_path = os.path.join(self.save_path, "models")
        self.media_path = os.path.join(self.save_path, "media")
        self.writer = SummaryWriter(log_dir=self.log_path)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.media_path, exist_ok=True)
        self._write_resolved_config(configs)
        dataset.save_texture_packing(os.path.join(self.save_path, 'texture_packing.json'))
        if configs.load_iter != 0:
            self.infer_path = os.path.join(configs.load_dir, "infer")
            os.makedirs(self.infer_path, exist_ok=True)

        # data config
        self.num_mips = dataset.num_mips
        self.texture_height = dataset.texture_height
        self.texture_width = dataset.texture_width
        self.mip0_only = bool(getattr(configs, "mip0_only", False))
        native_h, native_w = configs.material_reconstruction_size
        infer_h, infer_w = material_size(model,self.texture_height,self.texture_width,
            resolution=configs.material_compare_resolution)
        print(f'[Material Reconstruction] training={native_w}x{native_h}, inference={infer_w}x{infer_h}, '
              f'GT={self.texture_width}x{self.texture_height}, '
              f'resize_to_gt={(infer_h, infer_w) != (self.texture_height, self.texture_width)}, texel_centers=True')
        self.subpixel_sampling_enable = bool(getattr(configs, "subpixel_sampling_enable", True))
        self.subpixel_sampling_ratio = float(getattr(configs, "subpixel_sampling_ratio", 0.25))
        self.subpixel_jitter = float(getattr(configs, "subpixel_jitter", 0.5))

        # Canonical per-channel weights; eval keeps diffuse weight, training loss may zero direct diffuse.
        self.eval_weights = dataset.get_canonical_loss_weights(
            config_weights=configs.texture_weights
        )
        self.output_loss_weights = list(configs.output_loss_weights or self.eval_weights)
        if getattr(model, "direct_diffuse_enabled", False) and "diffuse" in dataset.available_textures:
            s, e = dataset.canonical_channel_slices["diffuse"]
            self.output_loss_weights[s:e] = [0.0] * (e - s)
        self._training_scalars = {}

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(return_full_image=True).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

        # early stopping (PSNR-based)
        self.early_stop = configs.early_stop
        self.early_stop_interval = configs.early_stop_interval
        self.early_stop_psnr_threshold = configs.early_stop_psnr_threshold
        self._early_stop_phase_start_iter = 0
        self._phase_control = None
        self.last_training_stats = {}

        self.configs = configs
        self.model = model
        self.dataset = dataset
        self.material_compression = material_compression
        self.material_compare_interval = configs.material_compare_interval
        self.enable_material_compare = bool(configs.enable_material_compare)
        self.astcenc_path = str(configs.astcenc_path)
        self.astcenc_quality = str(getattr(configs, "astcenc_quality", "medium"))
        self.astc_block = str(getattr(configs, "astc_block", "6x6"))
        self.material_compare_resolution = getattr(configs, "material_compare_resolution", None)
        self.enable_pbr_compare = bool(getattr(configs, "enable_pbr_compare", False))
        self.pbr_compare_interval = int(getattr(configs, "pbr_compare_interval", 5000))
        self.pbr_render_resolution = int(getattr(configs, "pbr_render_resolution", 512))
        self.pbr_mip_lod_bias = float(getattr(configs, "pbr_mip_lod_bias", -1.0))
        self.pbr_loss_enable = bool(getattr(configs, "pbr_loss_enable", False))
        self.pbr_loss_weight = float(getattr(configs, "pbr_loss_weight", 0.1))
        self.pbr_lighting = {
            "directional_enable": configs.pbr_directional_light_enable,
            "directional_direction": configs.pbr_directional_light_direction,
            "directional_intensity": configs.pbr_directional_light_intensity,
            "directional_color": configs.pbr_directional_light_color,
            "point_lights": configs.pbr_point_lights,
        }
        self.normal_encoding = str(getattr(configs, "normal_encoding", "xyz")).lower()
        self.super_resolution_enable = bool(getattr(configs, "super_resolution_enable", False))
        try:
            self._initialize_grid_compression(dataset, configs)
            if checkpoint is not None:
                from checkpoint import restore_training_state
                restore_training_state(self, checkpoint.state)
        except Exception:
            self.writer.close()
            raise

    def save_checkpoint(self, iteration, completed=False):
        from checkpoint import save_training_state
        if completed and self.progress.status == 'running':
            self.progress.status = 'stage_complete'
        self.model.save(iteration, self.model_path)
        save_training_state(self, iteration)

    def _log_checkpoint_interval(self, curr_iter: int) -> None:
        now = datetime.datetime.now()
        interval_time = now - self.last_checkpoint_time
        self.last_checkpoint_time = now
        print(f"[Checkpoint] Iter {curr_iter}: interval elapsed time = {interval_time}")

    def _log_total_training_time(self) -> None:
        self.end_time = datetime.datetime.now()
        self.duration_time = self.end_time - self.start_time
        print(f"Total training time: {self.duration_time}")

    def _compute_pbr_render_loss(self, gt, pred):
        return self._material_objective().pbr(gt, pred)

    def _material_objective(self):
        if not hasattr(self, '_objective') or self._objective.dataset is not self.dataset:
            self._objective = MaterialLoss(self.dataset, self.normal_encoding)
        return self._objective

    def _compute_reconstruction_loss(self, gt_texture, predict_texture, loss_weights, curr_iter=None):
        loss, stats = self._material_objective().training_reconstruction(
            gt_texture, predict_texture, loss_weights, statistics=curr_iter is not None)
        if curr_iter is not None:
            for group, values in stats.items():
                if values is not None:
                    for prefix, key in (('Loss', 'loss'), ('LossRaw', 'raw_loss'), ('LossWeight', 'weight')):
                        self._record_training_scalar(f'{prefix}/{group}', values[key])
        return loss

    def _record_training_scalar(self, name, value):
        self._training_scalars[name] = value.detach().float()

    def _flush_training_scalars(self, iteration):
        """Read all step diagnostics in one device-to-host transfer."""
        names = tuple(self._training_scalars)
        values = torch.stack(tuple(self._training_scalars.values())).cpu().tolist()
        self._training_scalars = dict(zip(names, values))
        for name, value in self._training_scalars.items():
            self.writer.add_scalar(name, value, iteration)
        return self._training_scalars

    def _loss_stats_str(self) -> str:
        labels = {
            "diffuse": "Weighted diffuse loss",
            "normal": "Weighted normal MSE loss",
            "romsd": "Weighted ROMSD loss",
        }
        parts = []
        for group in ("diffuse", "normal", "romsd"):
            loss = self._training_scalars.get(f'Loss/{group}')
            value = "-" if loss is None else f"{loss:.6f}"
            parts.append(f"{labels[group]}:{value}")
        return ", ".join(parts)

    def _reset_training_phase(self, phase, start):
        self._metrics_phase = phase
        self._phase_control = PhaseTrainingControl(
            phase, start,
            threshold=self.early_stop_psnr_threshold,
        )
        self._early_stop_phase_start_iter = start

    def _validation_score(self, curr_iter, eval_psnr=None):
        score = float(eval_psnr if eval_psnr is not None else self.deployment_metrics(psnr_only=True)['average'][0])
        self.writer.add_scalar('EarlyStop/recompressed_PSNR' if self.configs.material_recompression_enable
                               else 'EarlyStop/uncompressed_PSNR', score, curr_iter)
        return score

    def material_evaluation(self, *, baseline=True, material_override=None, primary_only=False):
        return evaluate_materials(self.model, self.dataset, self.configs, self.material_compression,
                                  self.astc_codec, baseline=baseline, material_override=material_override,
                                  primary_only=primary_only)

    @property
    def primary_method(self):
        return primary_method(self.material_compression,self.dataset,self.configs.material_recompression_enable)

    @torch.no_grad()
    def deployment_metrics(self, *, evaluation=None, psnr_only=False):
        evaluation = evaluation or self.material_evaluation(baseline=False, primary_only=True)
        prediction = evaluation.images[self.primary_method]
        gt = canonical_gt_mip(self.dataset, 0)
        return self._compute_material_metrics(prediction, gt, psnr_only=psnr_only)

    def run_comparison_reports(self, iteration, *, material=True, pbr=True, output_root=None, evaluation=None):
        material = material and self.enable_material_compare
        pbr = pbr and self.enable_pbr_compare
        if not material and not pbr:
            return None, None
        evaluation = (complete_material_evaluation(evaluation,self.dataset,self.material_compression)
                      if evaluation is not None else self.material_evaluation())
        material_metrics = self.run_material_comparison(curr_iter=iteration, output_root=output_root,
                                                        evaluation=evaluation) if material else None
        pbr_metrics = self.run_pbr_comparison(curr_iter=iteration, output_root=output_root,
                                             evaluation=evaluation, return_metrics=True)[1] if pbr else None
        return material_metrics, pbr_metrics

    def _observe_training_score(self, score, iteration, **kwargs):
        previous_best = self._phase_control.best_score
        stop = self._phase_control.observe(self.model, score, iteration, **kwargs)
        if self._phase_control.best_score > previous_best:
            # Persist the raw best state atomically as well as retaining its CPU
            # copy. Final exported checkpoints still use the standard save path.
            filename = f"best_{self._phase_control.phase}.pth"
            path = os.path.join(self.model_path, filename)
            from checkpoint import atomic_torch_save
            atomic_torch_save(self._phase_control.best_state, path)
        return stop

    def _complete_training_phase(self, eval_psnr=None, score=None, final_report=True, *, evaluation=None):
        last_iter = self.model.current_iter
        # Include the final updated model even when its iteration is not a check point.
        if score is None:
            if eval_psnr is None:
                evaluation = evaluation or self.material_evaluation(baseline=False, primary_only=True)
                eval_psnr = self.eval(last_iter, evaluation=evaluation)
            score = self._validation_score(last_iter, eval_psnr)
        self._observe_training_score(score, last_iter, allow_stop=False, adjust_lr=False)
        final_lr = self.model.optimizer.param_groups[0]["lr"]
        restored_earlier = self._phase_control.best_iteration != last_iter
        self._phase_control.restore(self.model)
        self._grid_compression_optimizer = None
        self.last_training_stats = dict(self._phase_control.summary(),
                                        completed_iteration=last_iter, final_decoder_lr=final_lr)
        # Refresh terminal media and metrics after best-state restoration.
        if restored_earlier:
            evaluation = self.material_evaluation(baseline=False, primary_only=True)
            restored_psnr = self.eval(last_iter, evaluation=evaluation)
            if final_report:
                self._validation_score(last_iter, restored_psnr)
        if final_report:
            self.run_comparison_reports(last_iter, evaluation=evaluation)
            self.save_checkpoint(last_iter, completed=True)
            if not self.enable_material_compare:
                self.save_compressed_materials(self.media_path, last_iter)
            self._log_checkpoint_interval(last_iter)
        self._log_total_training_time()

    def train(self, final_report=True) -> None:
        if self.progress.complete:
            print('[Resume] This training stage is already complete')
            return
        loss_weights = torch.tensor(
            self.output_loss_weights, device=self.device, dtype=torch.float32
        )
        if self._phase_control is None:
            codec_start = int(self.max_iter * self.configs.codec_start_frac)
            phase = (
                "warmup"
                if self.configs.codec_training_enable and self.trained_iter < codec_start
                else "codec" if self.configs.codec_training_enable else "feature"
            )
            self._reset_training_phase(phase, self.trained_iter)
        eval_psnr = current_psnr = None
        evaluation = None  # Keep a raster only for this step (or final completion).
        for curr_iter in range(self.trained_iter, self.max_iter):
            self._training_scalars.clear()
            codec_due = self._begin_codec_step(curr_iter)
            if codec_due and self._phase_control.phase == "warmup":
                self._reset_training_phase("codec", curr_iter)
                # Reset plateau history when the input distribution changes.
                if isinstance(self.model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.model.scheduler = make_scheduler(
                        self.model.optimizer, self.configs.lr_scheduler, self.max_iter - curr_iter)
            self.model.optimizer.zero_grad()

            if self.quantize and curr_iter % self.eval_interval == 0:
                self.writer.add_scalar("QAT/noise_mult", self.model._qat_noise_multiplier(), curr_iter)

            # generate random sample indices
            ys = torch.randint(0, self.texture_height, [self.batch_size, 1]).to(self.device)
            xs = torch.randint(0, self.texture_width, [self.batch_size, 1]).to(self.device)
            # Get data; expand to canonical 11 channels, missing texture positions filled with 0
            gt_texture = self.dataset.mip_cache[0][ys[:, 0], xs[:, 0]]
            gt_texture = self.dataset.expand_to_canonical(gt_texture).to(
                torch.float32 if self.native_reconstruction else torch.float16)
            superres_base = None
            if self.super_resolution_enable:
                superres_base = self.dataset.expand_to_canonical(
                    self.dataset.superres_base_cache[0][ys[:, 0], xs[:, 0]]
                ).to(torch.float16)

            us = (xs.float() + 0.5) / self.texture_width
            vs = (ys.float() + 0.5) / self.texture_height
            batch_input = torch.cat([us, vs, torch.zeros_like(us)], dim=1)
            if superres_base is not None:
                batch_input = torch.cat([batch_input, superres_base], dim=1)
            total_loss = self._training_batch_loss(
                batch_input, gt_texture, superres_base, loss_weights, curr_iter, codec_due
            )

            self._record_training_scalar('Loss/train', total_loss)
            # optimize
            total_loss.backward()
            if self.pbr_loss_enable:
                # The PBR proxy contains a specular reciprocal term; cap its
                # occasional large batch gradient before the optimizer step.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self._step_codec_parameters(codec_due)
            self.model.optimizer.step()
            loss_value = self._flush_training_scalars(curr_iter)['Loss/train']
            if self.model.scheduler is not None:
                if isinstance(self.model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.model.scheduler.step(metrics=loss_value)
                else:
                    self.model.scheduler.step()

            # print(self.model.optimizer.param_groups[0]['lr'], self.model.optimizer.param_groups[1]['lr'])

            self.model.clamp_value()
            self._finish_codec_step(codec_due)

            eval_psnr = current_psnr = None

            # eval
            if (curr_iter - self._early_stop_phase_start_iter) % self.eval_interval == 0:
                evaluation = self.material_evaluation(baseline=False, primary_only=True)
                eval_psnr = self.eval(curr_iter, evaluation=evaluation)

                if (curr_iter > self._early_stop_phase_start_iter
                        and (curr_iter - self._early_stop_phase_start_iter) % self.early_stop_interval == 0):
                    current_psnr = self._validation_score(curr_iter, eval_psnr)
                    self.writer.add_scalar('LR/decoder', self.model.optimizer.param_groups[0]["lr"], curr_iter)
                    stop = self._observe_training_score(
                        current_psnr, curr_iter,
                        allow_stop=self.early_stop and self._phase_control.phase != "warmup",
                    )
                    print(f"[EarlyStopCheck] {self._phase_control.phase} iter {curr_iter}: "
                          f"PSNR={current_psnr:.4f}, stagnant_windows={self._phase_control.bad_windows}")
                    if stop:
                        self._complete_training_phase(eval_psnr, current_psnr, final_report, evaluation=evaluation)
                        return

                # print(self.model.optimizer.param_groups[0]['lr'], self.model.optimizer.param_groups[1]['lr'])

            # Report cadence is independent of saving, validation windows and training stages.
            periodic = 0 < curr_iter < self.max_iter - 1
            material_due = periodic and curr_iter % self.material_compare_interval == 0
            pbr_due = periodic and curr_iter % self.pbr_compare_interval == 0
            self.run_comparison_reports(curr_iter, material=material_due, pbr=pbr_due, evaluation=evaluation)
            if curr_iter + 1 < self.max_iter:
                evaluation = None
            if curr_iter > 0 and curr_iter + 1 < self.max_iter and curr_iter % self.save_interval == 0:
                self.save_checkpoint(curr_iter)
                self._log_checkpoint_interval(curr_iter)

            if curr_iter > 0 and curr_iter % 10000 == 0:
                torch.cuda.empty_cache()
                tcnn.free_temporary_memory()

        self._complete_training_phase(eval_psnr, current_psnr, final_report, evaluation=evaluation)

    def fit_material_cnn(self, steps=50000, validation_interval=1000, resume_state=None):
        from Core.Tile_CNN.training import fit_material_cnn
        return fit_material_cnn(self, steps, validation_interval, resume_state)

    def _compute_material_metrics(
        self, pred: torch.Tensor, gt: torch.Tensor, *, psnr_only=False
    ) -> dict:
        """Compute per-material groups plus the evaluation-weighted overall score."""
        return _compute_group_metrics(
            pred,
            gt,
            self.dataset,
            self.psnr,
            self.ssim,
            self.lpips,
            eval_weights=self.eval_weights,
            psnr_only=psnr_only,
        )

    @torch.no_grad()
    def save_compressed_materials(self, output_root, iteration):
        # Standalone screenshots share the same methods and annotation path as comparisons.
        from pathlib import Path
        if getattr(self, '_saved_material_report', None) != (str(Path(output_root).resolve()),iteration):
            self.run_material_comparison(curr_iter=iteration, output_root=output_root, verbose=False)

    def _metrics_context(self, iteration=None):
        checkpoint = (getattr(self.model, 'loaded_checkpoint', None) if self.configs.mode == 'infer'
                      else os.path.join(self.model_path, f'train_result_{iteration}', 'model.pth'))
        selected = None
        checkpoint = checkpoint if checkpoint and os.path.isfile(checkpoint) else None
        checkpoint_iteration = (int(os.path.basename(os.path.dirname(checkpoint)).removeprefix('train_result_'))
                                if checkpoint else None)
        if self.configs.mode != 'infer' and self.last_training_stats.get('completed_iteration') == iteration:
            selected = self.last_training_stats.get('best_iteration')
        return dict(phase=getattr(self, '_metrics_phase', self.configs.mode),
                    mode=self.configs.mode,
                    checkpoint=checkpoint,
                    checkpoint_iteration=checkpoint_iteration,
                    selected_iteration=selected)

    @torch.no_grad()
    def eval(self, curr_iter, *, evaluation=None) -> float:
        metrics = self.deployment_metrics(evaluation=evaluation, psnr_only=True)
        for group, scores in metrics.items():
            self.writer.add_scalar(f'Material/{self.primary_method}/{group}/PSNR/train', scores[0], curr_iter)
        score = float(metrics['average'][0])
        print(f'Iter:{curr_iter}, {self.primary_method} MaterialPSNR:{score:.4f}. {self._loss_stats_str()}')
        return score

    @torch.no_grad()
    def infer(self) -> None:
        # Inference always writes material metrics and standalone screenshots.
        evaluation = self.material_evaluation()
        self.run_material_comparison(output_root=self.infer_path, evaluation=evaluation)
        if self.enable_pbr_compare:
            self.run_pbr_comparison(output_root=self.infer_path, evaluation=evaluation)

    @torch.no_grad()
    def run_material_comparison(self, curr_iter=None, output_root=None, verbose=True,
                                report_context=None, save_outputs=True, evaluation=None):
        from pathlib import Path
        output_root = output_root or (self.infer_path if self.configs.mode == 'infer' else self.media_path)
        metrics = run_material_comparison_pipeline(
            self, evaluation=evaluation, curr_iter=curr_iter,
            output_root=output_root,
            verbose=verbose, save_outputs=save_outputs,
            report_context={**self._metrics_context(curr_iter), **(report_context or {})})
        if curr_iter is not None:
            for group,scores in metrics[self.primary_method].items():
                for name,value in zip(('PSNR', 'SSIM', 'LPIPS'),scores):
                    self.writer.add_scalar(f'Material/{self.primary_method}/{group}/{name}/train',value,curr_iter)
        if save_outputs:
            self._saved_material_report = (str(Path(output_root).resolve()),curr_iter)
        return metrics

    @torch.no_grad()
    def run_pbr_comparison(self, curr_iter=None, output_root=None, return_metrics=False,
                           report_context=None, save_outputs=True, include_detail_metrics=False,
                           material_override=None, evaluation=None):
        output_root = output_root or (self.infer_path if self.configs.mode == 'infer' else self.media_path)
        evaluation = evaluation or self.material_evaluation(material_override=material_override)
        path, metrics = run_pbr_comparison_pipeline(
            self, evaluation, output_root, curr_iter=curr_iter,
            report_context={**self._metrics_context(curr_iter), **(report_context or {})},
            save_outputs=save_outputs, include_detail_metrics=include_detail_metrics)
        if curr_iter is not None:
            for view, methods in metrics.items():
                for method, values in methods.items():
                    for name, value in zip(('PSNR', 'SSIM', 'LPIPS'), values):
                        self.writer.add_scalar(f'PBRCompare/{view}/{method}/{name}', value, curr_iter)
        return (path, metrics) if return_metrics else path

    def _write_resolved_config(self, configs):
        write_resolved_config(configs, self.save_path)


def run(params, configs):
    """Dispatch one run and release its log writer even when training fails."""
    if configs.multi_stage_training_enable and params.mode == 'train':
        from Core.Multi_Stage_Training.multi_stage_training import run_multi_stage_training
        return run_multi_stage_training(params, configs)
    trainer = Trainer(params, config_override=configs)
    try:
        return trainer.train() if params.mode == 'train' else trainer.infer()
    finally:
        trainer.writer.close()


if __name__ == "__main__":

    params = get_args()

    configs = Config(params)

    if configs.groups_batching:
        # GroupsBatching mode: iterate over all texture groups sequentially
        print(f"[GroupsBatching] Starting batch training for {len(configs.groups_list)} groups...")
        failed_groups = []
        for idx, group_name in enumerate(configs.groups_list):
            print(f"\n{'='*60}")
            print(f"[GroupsBatching] Training group {idx+1}/{len(configs.groups_list)}: {group_name}")
            print(f"{'='*60}")
            group_cfg = configs.make_group_config(group_name)
            try:
                run(params, group_cfg)
            except Exception as e:
                print(f"[GroupsBatching] ERROR on group '{group_name}': {e}")
                import traceback
                traceback.print_exc()
                failed_groups.append(group_name)
                continue
            finally:
                try:
                    if torch.cuda.is_available():
                        # Free GPU memory between groups
                        torch.cuda.empty_cache()
                        tcnn.free_temporary_memory()
                except Exception:
                    pass  # empty_cache can fail after CUDA OOM
            print(f"[GroupsBatching] Finished group '{group_name}'")
        print(f"\n[GroupsBatching] All groups completed.")
        if failed_groups:
            print(f"[GroupsBatching] Exiting with error; failed groups: {', '.join(failed_groups)}", file=sys.stderr)
            sys.exit(1)
    else:
        run(params, configs)
