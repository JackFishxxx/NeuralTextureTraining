import os
import sys
import torch
import datetime
import argparse
import copy
import contextlib
import io
import math
from typing import List, Tuple
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)

from tensorboardX import SummaryWriter

import tinycudann as tcnn

from configs import get_args, inference_mips, mip_sampling_probabilities
from model import TCNNModel
from dataset import TextureDataset
from Common.material_loss import MaterialLoss
from Common.material_reconstruction import (canonical_gt_mip, make_deployment_model, render_material_mip,
                                          postprocess_material, save_material_visuals,
                                          comparison_material_size)
from configs import Config, write_resolved_config
from Common.comparison_astc import (
    ASTCENC_DEFAULT_PATH,
    cached_astc_base,
    log_comparison_storage,
    _compute_group_metrics,
    run_astc_comparison_pipeline,
    method_label, metric_records, write_material_metrics,
)
from Common.comparison_pbr_scene import run_pbr_comparison_pipeline
from dataset import material_size
from Core.Feature_Grid.training import FeatureGridTraining
from Core.Feature_Grid.optimization import make_scheduler
from Core.Feature_Grid.compression import compressed_method


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
                print(f"[NormalEncoding] mode={dataset.normal_encoding}")
                configs.num_mips = dataset.num_mips
                # Tile size is measured in decoder/material texels, independent of GT resolution.
                configs.pos_encoding_reference_edge = max(dataset.material_reconstruction_size)
                configs.material_reconstruction_size = list(dataset.material_reconstruction_size)
                configs.material_cnn_inference_size = list(comparison_material_size(
                    configs, dataset.texture_height, dataset.texture_width, configs.ref_astc_resolution))
                if (deployment_config or configs).material_cnn_enable:
                    from Core.Tile_CNN.training import validate_training_size
                    validate_training_size((dataset.texture_height, dataset.texture_width),
                                           configs.material_cnn_inference_size)
                model = TCNNModel(configs, checkpoint=checkpoint.model_path if checkpoint else None)
                model.configure_direct_diffuse_from_dataset(dataset)
                configs.material_reconstruction_size = list(material_size(model, dataset.texture_height, dataset.texture_width))
                configs.num_channels = model.num_channels
            log_comparison_storage(model, dataset, configs.astc_block, configs.ref_astc_resolution,
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
        self.quantize_bits = configs.quantize_bits
        self.save_bits = configs.save_bits

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
        infer_h, infer_w = comparison_material_size(
            model, self.texture_height, self.texture_width, configs.ref_astc_resolution)
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
        self._last_loss_stats = {}

        # visualization configs for eval/infer (data-driven, not hardcoded)
        self.vis_configs = dataset.get_vis_configs()

        self.sample_probabilities = self.generate_probabilities()

        # losses
        self.L2_loss = torch.nn.MSELoss(reduction="none")

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
        self.eval_inference_tile = max(0, int(configs.eval_inference_tile))
        self.enable_astc_compare = bool(configs.enable_astc_compare)
        self.astcenc_path = str(getattr(configs, "astcenc_path", ASTCENC_DEFAULT_PATH))
        self.astcenc_quality = str(getattr(configs, "astcenc_quality", "medium"))
        self.astc_block = str(getattr(configs, "astc_block", "6x6"))
        self.ref_astc_resolution = getattr(configs, "ref_astc_resolution", None)
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
            self._initialize_astc_aware(dataset, configs)
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
        loss, stats = self._material_objective().reconstruction(
            gt_texture, predict_texture, loss_weights, statistics=curr_iter is not None)
        if curr_iter is not None:
            self._last_loss_stats = stats
            for group, values in stats.items():
                if values is not None:
                    name = 'normal_angular' if group == 'normal' else group
                    for prefix, key in (('Loss', 'loss'), ('LossRaw', 'raw_loss'), ('LossWeight', 'weight')):
                        self.writer.add_scalar(f'{prefix}/{name}', values[key], curr_iter)
        return loss

    def _loss_stats_str(self) -> str:
        labels = {
            "diffuse": "Weighted diffuse loss",
            "normal": "Weighted normal angular loss",
            "romsd": "Weighted ROMSD loss",
        }
        parts = []
        for group in ("diffuse", "normal", "romsd"):
            stat = self._last_loss_stats.get(group)
            value = "-" if stat is None else f"{stat['loss']:.6f}"
            parts.append(f"{labels[group]}:{value}")
        return ", ".join(parts)

    def _reset_training_phase(self, phase, start):
        self._metrics_phase = phase
        self._phase_control = PhaseTrainingControl(
            phase, start,
            threshold=self.early_stop_psnr_threshold,
        )
        self._early_stop_phase_start_iter = start

    def _validation_score(self, curr_iter, eval_psnr, verbose=True):
        score = float(self.deployment_metrics()['average'][0])
        if self.enable_astc_compare:
            self.run_astc_comparison(curr_iter=curr_iter, output_root=self.media_path, verbose=verbose)
        self.writer.add_scalar('EarlyStop/compressed_PSNR', score, curr_iter)
        return score

    @torch.no_grad()
    def deployment_metrics(self):
        decoded = make_deployment_model(self.model)
        prediction, gt, _, _ = self._render_mip_pair(0, decoded)
        return self._compute_material_metrics(prediction, gt)

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

    def _complete_training_phase(self, eval_psnr=None, score=None, final_report=True):
        last_iter = self.model.current_iter
        # Include the final updated model even when its iteration is not a check point.
        if score is None:
            if eval_psnr is None:
                eval_psnr = self.eval(last_iter)
            score = self._validation_score(last_iter, eval_psnr, verbose=final_report)
        self._observe_training_score(score, last_iter, allow_stop=False, adjust_lr=False)
        final_lr = self.model.optimizer.param_groups[0]["lr"]
        restored_earlier = self._phase_control.best_iteration != last_iter
        self._phase_control.restore(self.model)
        self._astc_block_optimizer = None
        self.last_training_stats = dict(self._phase_control.summary(),
                                        completed_iteration=last_iter, final_decoder_lr=final_lr)
        # Refresh terminal media and metrics after best-state restoration.
        if restored_earlier:
            restored_psnr = self.eval(last_iter)
            if final_report:
                self._validation_score(last_iter, restored_psnr)
        if final_report:
            if self.enable_pbr_compare:
                self.run_pbr_comparison(curr_iter=last_iter, output_root=self.media_path)
            self.save_checkpoint(last_iter, completed=True)
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
        for curr_iter in range(self.trained_iter, self.max_iter):

            codec_due = self._begin_astc_step(curr_iter)
            if codec_due and self._phase_control.phase == "warmup":
                self._reset_training_phase("codec", curr_iter)
                # Reset plateau history when the input distribution changes.
                if isinstance(self.model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.model.scheduler = make_scheduler(
                        self.model.optimizer, self.configs.lr_scheduler, self.max_iter - curr_iter)
            self.model.optimizer.zero_grad()

            # update model's current iteration for noise annealing
            self.model.current_iter = curr_iter
            if self.quantize and curr_iter % self.eval_interval == 0:
                self.writer.add_scalar("QAT/noise_mult", self.model._qat_noise_multiplier(), curr_iter)

            # generate random sample indices
            ys = torch.randint(0, self.texture_height, [self.batch_size, 1]).to(self.device)
            xs = torch.randint(0, self.texture_width, [self.batch_size, 1]).to(self.device)
            # sample Mips with a coarse-to-fine prior
            # mips = torch.randint(0, self.num_mips, [self.batch_size, 1]).to(self.device)
            mips = self.sample_probabilities.multinomial(num_samples=self.batch_size, replacement=True)
            #mips = torch.zeros(mips.shape).to(self.device).to(torch.int32)
            mips = mips.unsqueeze(1)
            # mips = torch.randint(0, 1, size=(self.batch_size, 1)).to(self.device)
            batch_index = torch.cat([ys, xs, mips], dim=1)

            # Get data; expand to canonical 11 channels, missing texture positions filled with 0
            gt_texture = (self.dataset.mip_cache[0][ys[:, 0], xs[:, 0]]
                          if self.mip0_only else self.dataset(batch_index))
            gt_texture = self.dataset.expand_to_canonical(gt_texture).to(
                torch.float32 if self.native_reconstruction else torch.float16)
            superres_base = None
            if self.super_resolution_enable:
                superres_base = self.dataset.expand_to_canonical(
                    self.dataset.superres_base_cache[0][ys[:, 0], xs[:, 0]]
                    if self.mip0_only else self.dataset.get_superres_base(batch_index)
                ).to(torch.float16)

            # Use texel centers in the selected mip plane. The dataset indexes
            # discrete targets after integer downscaling, so floor the base
            # coordinates before converting them to mip-local UVs.
            if self.mip0_only:
                us = (xs.float() + 0.5) / self.texture_width
                vs = (ys.float() + 0.5) / self.texture_height
            else:
                mip_scale = torch.pow(2.0, mips.float())
                mip_x = torch.floor(xs.float() / mip_scale)
                mip_y = torch.floor(ys.float() / mip_scale)
                mip_width = self.texture_width / mip_scale
                mip_height = self.texture_height / mip_scale
                us = (mip_x + 0.5) / mip_width
                vs = (mip_y + 0.5) / mip_height
            mips = mips.float() / (self.num_mips - 1) if self.num_mips > 1 else torch.zeros_like(mips, dtype=torch.float32)
            batch_input = torch.cat([us, vs, mips], dim=1)
            if superres_base is not None:
                batch_input = torch.cat([batch_input, superres_base], dim=1)
            total_loss = self._training_batch_loss(
                batch_input, gt_texture, superres_base, loss_weights, curr_iter, codec_due
            )

            loss_value = total_loss.item()
            self.writer.add_scalar('Loss/train', loss_value, curr_iter)
            # optimize
            total_loss.backward()
            if self.pbr_loss_enable:
                # The PBR proxy contains a specular reciprocal term; cap its
                # occasional large batch gradient before the optimizer step.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self._step_astc_parameters(codec_due)
            self.model.optimizer.step()
            if self.model.scheduler is not None:
                if isinstance(self.model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.model.scheduler.step(metrics=loss_value)
                else:
                    self.model.scheduler.step()

            # print(self.model.optimizer.param_groups[0]['lr'], self.model.optimizer.param_groups[1]['lr'])

            self.model.clamp_value()
            self._finish_astc_step(codec_due)

            # track whether ASTC comparison was already run this iteration (avoid duplicate runs)
            _astc_already_ran = False
            eval_psnr = current_psnr = None

            # eval
            if (curr_iter - self._early_stop_phase_start_iter) % self.eval_interval == 0:
                eval_psnr = self.eval(curr_iter)

                if (curr_iter > self._early_stop_phase_start_iter
                        and (curr_iter - self._early_stop_phase_start_iter) % self.early_stop_interval == 0):
                    current_psnr = self._validation_score(
                        curr_iter, eval_psnr, verbose=final_report)
                    _astc_already_ran = self.enable_astc_compare
                    self.writer.add_scalar('LR/decoder', self.model.optimizer.param_groups[0]["lr"], curr_iter)
                    stop = self._observe_training_score(
                        current_psnr, curr_iter,
                        allow_stop=self.early_stop and self._phase_control.phase != "warmup",
                    )
                    print(f"[EarlyStopCheck] {self._phase_control.phase} iter {curr_iter}: "
                          f"PSNR={current_psnr:.4f}, stagnant_windows={self._phase_control.bad_windows}")
                    if stop:
                        self._complete_training_phase(eval_psnr, current_psnr, final_report)
                        return

                # print(self.model.optimizer.param_groups[0]['lr'], self.model.optimizer.param_groups[1]['lr'])

            if curr_iter > 0 and curr_iter + 1 < self.max_iter and curr_iter % self.save_interval == 0:
                if self.enable_astc_compare and not _astc_already_ran:
                    self.run_astc_comparison(curr_iter=curr_iter, output_root=self.media_path)
                if self.enable_pbr_compare:
                    self.run_pbr_comparison(curr_iter=curr_iter, output_root=self.media_path)
                self.save_checkpoint(curr_iter)
                self._log_checkpoint_interval(curr_iter)

            if (self.enable_pbr_compare and curr_iter > 0 and curr_iter + 1 < self.max_iter
                    and curr_iter % self.pbr_compare_interval == 0
                    and curr_iter % self.save_interval != 0):
                self.run_pbr_comparison(curr_iter=curr_iter, output_root=self.media_path)

            if curr_iter > 0 and curr_iter % 10000 == 0:
                torch.cuda.empty_cache()
                tcnn.free_temporary_memory()

        self._complete_training_phase(eval_psnr, current_psnr, final_report)

    def fit_material_cnn(self, steps=50000, validation_interval=1000, resume_state=None):
        from Core.Tile_CNN.training import fit_material_cnn
        return fit_material_cnn(self, steps, validation_interval, resume_state)

    def _cuda_trim_eval_mem(self, synchronize: bool = False) -> None:
        if self.device != "cuda":
            return
        if synchronize:
            torch.cuda.synchronize()
        tcnn.free_temporary_memory()
        torch.cuda.empty_cache()

    def _backup_feature_grid_state(self):
        return [{k: v.detach().clone() for k, v in g.state_dict().items()} for g in self.model.feature_grids]

    def _restore_feature_grid_state(self, backups) -> None:
        for g, bak in zip(self.model.feature_grids, backups):
            g.load_state_dict(bak)

    @torch.no_grad()
    def _fill_predicted_mip(self, mip, mip_height, mip_width, model=None):
        model = self.model if model is None else model
        base = None
        if mip == 0 and self.super_resolution_enable and (model._decoder_adaptation_features is not None
                                                         or getattr(model, '_compressed_material', False)):
            size = comparison_material_size(model, mip_height, mip_width, self.ref_astc_resolution)
            base = cached_astc_base(self.dataset, self.astc_codec,
                                    self.configs.super_resolution_base_resolution, size)
        return render_material_mip(
            model, getattr(self, 'dataset', None), mip, mip_height, mip_width, num_mips=self.num_mips, device=self.device,
            inference_tile=self.eval_inference_tile, ref_astc_resolution=getattr(self, 'ref_astc_resolution', None),
            superres_base_override=base)

    def _mip_size(self, mip: int) -> Tuple[int, int]:
        return self.texture_height // (2 ** mip), self.texture_width // (2 ** mip)

    @torch.no_grad()
    def _render_mip_pair(self, mip, model=None):
        height, width = self._mip_size(mip)
        pred = self._fill_predicted_mip(mip, height, width, model).clamp(0, 1).permute(2, 0, 1)[None]
        return pred, canonical_gt_mip(self.dataset, mip), height, width

    def _compute_material_metrics(
        self, pred: torch.Tensor, gt: torch.Tensor
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
        )

    def _save_mip_visuals(self, pred, gt, output_root, filename):
        save_material_visuals(pred, gt, output_root, filename, self.vis_configs, self.normal_encoding)

    @torch.no_grad()
    def save_compressed_materials(self, output_root, iteration):
        decoded = make_deployment_model(self.model)
        for mip in self._infer_mips():
            pred, gt, _, _ = self._render_mip_pair(mip, decoded)
            self._save_mip_visuals(pred, gt, output_root, f'{iteration}_{mip}.png')

    @staticmethod
    def _mean_metric(values: List[float]) -> float:
        return math.fsum(values) / len(values) if values else 0.0

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
    def eval(self, curr_iter) -> float:
        self._cuda_trim_eval_mem(synchronize=True)

        psnr_list: List[float] = []
        ssim_list: List[float] = []
        lpips_list: List[float] = []
        material_group_order = ("diffuse", "normal", "romsd", "average")
        should_save_visuals = curr_iter % self.save_interval == 0

        feature_grid_backup = self._backup_feature_grid_state()
        was_training = self.model.training
        try:
            self.model.eval()
            self.model.simulate_quantize()

            eval_mips = range(1) if self.mip0_only else range(self.num_mips)
            for mip in eval_mips:
                pred, gt, mip_height, mip_width = self._render_mip_pair(mip)
                material_metrics = self._compute_material_metrics(pred, gt)
                psnr_value, ssim_value, lpips_value = material_metrics["average"]

                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
                lpips_list.append(lpips_value)
                for group_name in material_group_order:
                    group_metric = material_metrics.get(group_name)
                    if group_metric is None:
                        continue
                    group_psnr, group_ssim, group_lpips = group_metric
                    self.writer.add_scalar(
                        f'Material/{group_name}/PSNR_Mip{mip}/train', group_psnr, curr_iter
                    )
                    self.writer.add_scalar(
                        f'Material/{group_name}/SSIM_Mip{mip}/train', group_ssim, curr_iter
                    )
                    self.writer.add_scalar(
                        f'Material/{group_name}/LPIPS_Mip{mip}/train', group_lpips, curr_iter
                    )

        finally:
            self._restore_feature_grid_state(feature_grid_backup)
            self.model.train(was_training)
            self._cuda_trim_eval_mem(synchronize=False)

        if should_save_visuals:
            self.save_compressed_materials(self.media_path, curr_iter)

        psnr_avg = self._mean_metric(psnr_list)
        ssim_avg = self._mean_metric(ssim_list)
        lpips_avg = self._mean_metric(lpips_list)
        self.writer.add_scalar('Material/average/PSNR/train', psnr_avg, curr_iter)
        self.writer.add_scalar('Material/average/SSIM/train', ssim_avg, curr_iter)
        self.writer.add_scalar('Material/average/LPIPS/train', lpips_avg, curr_iter)

        print(f"Iter:{curr_iter}, MaterialPSNR:{psnr_avg:.4f}. {self._loss_stats_str()}")
        return psnr_avg

    def _postprocess_for_vis(self, image, vis_mode):
        return postprocess_material(image, vis_mode, self.normal_encoding)

    def _infer_mips(self) -> range:
        return inference_mips(self.num_mips, self.mip0_only)

    @torch.no_grad()
    def infer(self) -> None:
        records = []
        storage = log_comparison_storage(self.model, self.dataset, self.astc_block,
                                         self.ref_astc_resolution, verbose=False)
        decoded = make_deployment_model(self.model)
        for mip in self._infer_mips():
            pred, gt, mip_height, mip_width = self._render_mip_pair(mip, decoded)
            material_metrics = self._compute_material_metrics(pred, gt)
            records.extend(metric_records(material_metrics, method_label(compressed_method(self.model)),
                                          mip=mip, storage_bytes=storage[compressed_method(self.model)],
                                          lpips_available=min(mip_height, mip_width) >= 128))
            self._save_mip_visuals(pred, gt, self.infer_path, f"Mip_{mip}.png")
        del decoded
        averages = [row for row in records if row['group'] == 'average']
        if len(averages) > 1:
            values = tuple(self._mean_metric([row[key] for row in averages])
                           if all(row[key] is not None for row in averages) else None
                           for key in ('psnr', 'ssim', 'lpips'))
            summary = metric_records({'average': values}, averages[0]['method'], mip=None,
                                     storage_bytes=averages[0]['storage_bytes'],
                                     lpips_available=values[2] is not None)[0]
            summary['psnr_type'] = 'mean_across_mips'
            records.append(summary)
        write_material_metrics(
            self.infer_path, records, storage, self.model, self.dataset, self.ref_astc_resolution,
            self.eval_weights, report_context=self._metrics_context())

        if self.enable_astc_compare:
            self.run_astc_comparison(output_root=self.infer_path)
        if self.enable_pbr_compare:
            self.run_pbr_comparison(output_root=self.infer_path)

    @torch.no_grad()
    def run_astc_comparison(self, curr_iter: int = None, output_root: str = None, verbose=True,
                            report_context=None, save_outputs=True) -> dict:
        if output_root is None:
            output_root = self.infer_path if hasattr(self, "infer_path") else self.media_path
        return run_astc_comparison_pipeline(
            model=self.model,
            dataset=self.dataset,
            astc_codec=self.astc_codec,
            psnr_metric=self.psnr,
            ssim_metric=self.ssim,
            lpips_metric=self.lpips,
            vis_configs=self.vis_configs,
            postprocess_for_vis=self._postprocess_for_vis,
            output_root=output_root,
            texture_height=self.texture_height,
            texture_width=self.texture_width,
            num_mips=self.num_mips,
            device=self.device,
            curr_iter=curr_iter,
            ref_astc_resolution=self.ref_astc_resolution,
            eval_weights=self.eval_weights,
            inference_tile=self.eval_inference_tile,
            verbose=verbose,
            report_context={**self._metrics_context(curr_iter), **(report_context or {})},
            save_outputs=save_outputs,
        )

    @torch.no_grad()
    def run_pbr_comparison(self, curr_iter=None, output_root=None, return_metrics=False,
                           report_context=None, save_outputs=True, include_detail_metrics=False,
                           material_override=None):
        output_root = output_root or getattr(self, 'infer_path', self.media_path)
        path, metrics = run_pbr_comparison_pipeline(
            self.model, self.dataset, self.astc_codec, self.configs, output_root, curr_iter=curr_iter,
            ssim_metric=self.ssim, lpips_metric=self.lpips, lighting=self.pbr_lighting,
            report_context={**self._metrics_context(curr_iter), **(report_context or {})}, save_outputs=save_outputs,
            include_detail_metrics=include_detail_metrics, material_override=material_override)
        if curr_iter is not None:
            for method, values in metrics.items():
                for name, value in zip(('PSNR', 'SSIM', 'LPIPS'), values):
                    self.writer.add_scalar(f'PBRCompare/{method}_{name}', value, curr_iter)
        return (path, metrics) if return_metrics else path

    def _write_resolved_config(self, configs):
        write_resolved_config(configs, self.save_path)

    def generate_probabilities(self) -> torch.Tensor:
        # Generate a coarse-to-fine Mip sampling distribution, or isolate Mip0.
        return mip_sampling_probabilities(self.num_mips, self.mip0_only, self.device)


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
