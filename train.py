import os
import sys
import torch
import torch.nn.functional as F
import datetime
import argparse
import json
import copy
from typing import List, Optional, Tuple
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)

from tensorboardX import SummaryWriter
import torchvision.transforms.functional as TF

import tinycudann as tcnn

from configs import get_args, inference_mips, mip_sampling_probabilities
from model import TCNNModel
from dataset import TextureDataset
from normal_encoding import decode_normal, normal_angular_loss, normal_angular_psnr, normal_vectors
from configs import Config
from Comparison_ASTC import (
    ASTCENC_DEFAULT_PATH,
    _astc_roundtrip_superres_base_mip0,
    _render_gt_mip0,
    _render_model_mip0,
    _ref_astc_hw_from_side,
    _traditional_baseline_resampled,
    _compute_group_metrics,
    apply_astc_to_feature_grids,
    run_astc_comparison_pipeline,
)
from Comparison_PBRScene import save_pbr_comparison
from Core.ASTC_Aware.trainer import ASTCAwareTrainer


class Trainer(ASTCAwareTrainer):

    def __init__(self, params: argparse.Namespace, config_override: Config = None) -> None:

        # initialize required runtime fieldsvariables, e.g. dataset, configs and models
        if config_override is not None:
            configs = config_override
        else:
            configs = Config(params)
        dataset = TextureDataset(configs)
        print(f"[NormalEncoding] mode={dataset.normal_encoding}")
        configs.num_mips = dataset.num_mips
        # Real texture edge so pos_encoding_tile_size (texels) maps correctly to UV space.
        configs.pos_encoding_reference_edge = max(dataset.texture_height, dataset.texture_width)
        model = TCNNModel(configs)
        model.configure_direct_diffuse_from_dataset(dataset)
        # Network output is fixed to 11 channels (aligned with diffuse->displacement); missing filled by dataset with 0
        configs.num_channels = model.num_channels

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
        self.save_path = os.path.join(self.save_dir, self.start_time.strftime(r"%y_%m_%d_%H_%M_%S"))
        self.log_path = os.path.join(self.save_path, "tensorboard")
        self.model_path = os.path.join(self.save_path, "models")
        self.media_path = os.path.join(self.save_path, "media")
        self.writer = SummaryWriter(log_dir=self.log_path)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.media_path, exist_ok=True)
        self._write_resolved_config(configs)
        if configs.load_iter != 0:
            self.infer_path = os.path.join(configs.load_dir, "infer")
            os.makedirs(self.infer_path, exist_ok=True)

        # data config
        self.num_mips = dataset.num_mips
        self.texture_height = dataset.texture_height
        self.texture_width = dataset.texture_width
        self.mip0_only = bool(getattr(configs, "mip0_only", False))
        self.subpixel_sampling_enable = bool(getattr(configs, "subpixel_sampling_enable", True))
        self.subpixel_sampling_ratio = float(getattr(configs, "subpixel_sampling_ratio", 0.25))
        self.subpixel_jitter = float(getattr(configs, "subpixel_jitter", 0.5))

        # Canonical per-channel weights; eval keeps diffuse weight, training loss may zero direct diffuse.
        self.eval_weights = dataset.get_canonical_loss_weights(
            config_weights=getattr(configs, 'texture_weights', getattr(configs, 'texture_loss_weights', None))
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
        self._early_stop_eval_count = 0
        self._early_stop_avg_psnr = 0
        self._early_stop_prev_psnr = 0   # PSNR of the previous segment
        self._early_stop_phase_start_iter = 0

        self.configs = configs
        self.model = model
        self.dataset = dataset
        self.eval_inference_tile = max(0, int(configs.eval_inference_tile))
        self.eval_metrics_max_edge = int(configs.eval_metrics_max_edge)
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
        self._initialize_astc_aware(dataset, configs)

    def _log_checkpoint_interval(self, curr_iter: int) -> None:
        now = datetime.datetime.now()
        interval_time = now - self.last_checkpoint_time
        self.last_checkpoint_time = now
        print(f"[Checkpoint] Iter {curr_iter}: interval elapsed time = {interval_time}")

    def _log_total_training_time(self) -> None:
        self.end_time = datetime.datetime.now()
        self.duration_time = self.end_time - self.start_time
        print(f"Total training time: {self.duration_time}")

    def _group_indices(self, textures: List[str], device: torch.device) -> torch.Tensor:
        key = (tuple(textures), device)
        if not hasattr(self, "_loss_group_indices"):
            self._loss_group_indices = {}
        if key in self._loss_group_indices:
            return self._loss_group_indices[key]
        idx = []
        for tex in textures:
            if tex not in self.dataset.available_textures:
                continue
            s, e = self.dataset.canonical_channel_slices[tex]
            idx.extend(range(s, e))
        indices = torch.tensor(idx, device=device, dtype=torch.long)
        self._loss_group_indices[key] = indices
        return indices

    def _compute_pbr_render_loss(self, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Differentiable shaded-RGB loss for PBR-sensitive material channels."""
        slices = self.dataset.canonical_channel_slices
        def get_pair(name, channels=1, default=0.0):
            if name not in self.dataset.available_textures:
                shape = (gt.shape[0], channels)
                value = torch.full(shape, default, device=gt.device, dtype=gt.dtype)
                return value, value
            s, e = slices[name]
            return (torch.nan_to_num(gt[:, s:e].float(), nan=default, posinf=default, neginf=default).clamp(0, 1),
                    torch.nan_to_num(pred[:, s:e].float(), nan=default, posinf=default, neginf=default).clamp(0, 1))

        gt_diff, pr_diff = get_pair("diffuse", 3, 0.5)
        gt_norm, pr_norm = get_pair("normal", getattr(self.dataset, "normal_encoding", "xyz") == "xyz" and 3 or 2, 0.5)
        gt_rough, pr_rough = get_pair("roughness", 1, 0.5)
        gt_metal, pr_metal = get_pair("metallic", 1, 0.0)
        gt_spec, pr_spec = get_pair("specular", 1, 0.5)

        gt_n = normal_vectors(gt_norm, self.normal_encoding)
        pr_n = normal_vectors(pr_norm, self.normal_encoding)
        # Use a fixed tangent-space light with positive Z so a flat normal map
        # receives direct light and all PBR channels get useful gradients.
        lighting = getattr(self, "_pbr_lighting", None)
        if lighting is None or lighting[0].device != pred.device or lighting[0].dtype != pred.dtype:
            light = torch.tensor([-0.35, 0.45, 0.82], device=pred.device, dtype=pred.dtype)
            light = light / torch.clamp(torch.linalg.vector_norm(light), min=1e-6)
            view = torch.tensor([0.0, 0.0, 1.0], device=pred.device, dtype=pred.dtype)
            half = (light + view) / torch.clamp(torch.linalg.vector_norm(light + view), min=1e-6)
            self._pbr_lighting = (light, view, half)
        light, view, half = self._pbr_lighting
        def shade(diff, normal, rough, metal, spec):
            diff = torch.clamp(diff, 0, 1)
            ndotl = torch.clamp((normal * light[None]).sum(1, keepdim=True), 0, 1)
            ndoth = torch.clamp((normal * half[None]).sum(1, keepdim=True), 0, 1)
            vdoth = torch.clamp((view[None] * half[None]).sum(1, keepdim=True), 0, 1)
            # Avoid the singular GGX lobe at roughness=0.045 during training.
            rough = torch.clamp(rough, 0.12, 1.0)
            alpha = rough * rough
            d_den = (ndoth * ndoth * (alpha * alpha - 1) + 1) ** 2
            d = alpha * alpha / (torch.pi * torch.clamp(d_den, min=0.04))
            d = torch.clamp(d, max=8.0)
            f0 = (0.02 + 0.14 * spec) * (1 - metal) + diff * metal
            fresnel = torch.clamp(f0 + (1 - f0) * (1 - vdoth) ** 5, 0.0, 1.0)
            shaded = torch.clamp(diff * (1 - metal) * (1 - fresnel) * ndotl / torch.pi
                                 + d * fresnel * ndotl * 0.25 + diff * 0.03, 0, 1)
            return torch.nan_to_num(shaded, nan=0.0, posinf=1.0, neginf=0.0)
        pred_shaded = shade(pr_diff, pr_n, pr_rough, pr_metal, pr_spec)
        gt_shaded = shade(gt_diff, gt_n, gt_rough, gt_metal, gt_spec).detach()
        return torch.nan_to_num(torch.nn.functional.smooth_l1_loss(pred_shaded, gt_shaded),
                                nan=0.0, posinf=1.0, neginf=0.0)

    def _compute_reconstruction_loss(
        self,
        gt_texture: torch.Tensor,
        predict_texture: torch.Tensor,
        loss_weights: torch.Tensor,
        curr_iter: Optional[int] = None,
    ) -> torch.Tensor:
        groups = {
            "diffuse": ["diffuse"],
            "normal": ["normal"],
            "romsd": ["roughness", "occlusion", "metallic", "specular", "displacement"],
        }
        total_loss = torch.tensor(0.0, device=gt_texture.device, dtype=torch.float32)
        stats = {}

        for group, textures in groups.items():
            idx = self._group_indices(textures, gt_texture.device)
            if idx.numel() == 0:
                stats[group] = None
                continue

            pred = torch.index_select(predict_texture.float(), dim=1, index=idx)
            gt = torch.index_select(gt_texture.float(), dim=1, index=idx)
            weights = loss_weights.index_select(0, idx).float()
            weight = weights.mean()

            if group == "normal":
                raw_loss = normal_angular_loss(pred, gt, self.normal_encoding)
                weighted_loss = raw_loss * weights.sum()
            else:
                mse = (gt - pred).pow(2).mean(dim=0)
                raw_loss = mse.mean()
                weighted_loss = (mse * weights).sum()

            total_loss = total_loss + weighted_loss
            stats[group] = {
                "loss": float(weighted_loss.detach().item()),
                "raw_loss": float(raw_loss.detach().item()),
                "weight": float(weight.detach().item()),
            }

            if curr_iter is not None:
                loss_name = "normal_angular" if group == "normal" else group
                self.writer.add_scalar(f'Loss/{loss_name}', stats[group]["loss"], curr_iter)
                self.writer.add_scalar(f'LossRaw/{loss_name}', stats[group]["raw_loss"], curr_iter)
                self.writer.add_scalar(f'LossWeight/{loss_name}', stats[group]["weight"], curr_iter)

        if curr_iter is not None:
            self._last_loss_stats = stats
        return total_loss

    def _loss_stats_str(self) -> str:
        residual_stat = self._last_loss_stats.get("superres_residual")
        if residual_stat is not None:
            terms = []
            for group, label in (("diffuse", "diffuse"), ("normal", "normal"),
                                 ("romsd", "ROMSD")):
                stat = residual_stat.get(group)
                if stat is not None:
                    terms.append(f"{label} loss:{stat['loss']:.6f}")
            decomposition = " + ".join(terms)
            return f"Weighted super-resolution residual loss:{residual_stat['loss']:.6f} = {decomposition}"
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

    def _compute_superres_residual_loss(
        self, target_residual: torch.Tensor, predicted_residual: torch.Tensor, loss_weights: torch.Tensor,
        curr_iter: Optional[int] = None,
    ) -> torch.Tensor:
        """Weighted channel MSE for signed residuals, including encoded normal channels."""
        mse = (target_residual.float() - predicted_residual.float()).pow(2).mean(dim=0)
        loss = (mse * loss_weights.float()).sum()
        if curr_iter is not None:
            group_losses = {}
            for group, textures in {
                "diffuse": ["diffuse"],
                "normal": ["normal"],
                "romsd": ["roughness", "occlusion", "metallic", "specular", "displacement"],
            }.items():
                idx = self._group_indices(textures, target_residual.device)
                if idx.numel() == 0:
                    continue
                group_loss = (mse.index_select(0, idx) * loss_weights.index_select(0, idx).float()).sum()
                group_losses[group] = group_loss.detach()
            values = torch.stack([loss.detach(), *group_losses.values()]).tolist()
            self.writer.add_scalar('Loss/superres_residual', values[0], curr_iter)
            stats = {"loss": values[0]}
            for group, value in zip(group_losses, values[1:]):
                stats[group] = {"loss": value}
                self.writer.add_scalar(f'Loss/superres_residual_{group}', stats[group]["loss"], curr_iter)
            self._last_loss_stats = {"superres_residual": stats}
        return loss


    def train(self) -> None:

        loss_weights = torch.tensor(self.output_loss_weights, device=self.device, dtype=torch.float32)
        for curr_iter in range(self.trained_iter, self.max_iter):

            codec_due = self._begin_astc_step(curr_iter)
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
            gt_texture = self.dataset.expand_to_canonical(gt_texture).to(torch.float16)
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
            self.model.scheduler.step(metrics=loss_value)

            # print(self.model.optimizer.param_groups[0]['lr'], self.model.optimizer.param_groups[1]['lr'])

            self.model.clamp_value()
            self._finish_astc_step(codec_due)

            # track whether ASTC comparison was already run this iteration (avoid duplicate runs)
            _astc_already_ran = False
            _pbr_already_ran = False

            # eval
            if (curr_iter - self._early_stop_phase_start_iter) % self.eval_interval == 0:
                eval_psnr = self.eval(curr_iter)

                # early stopping check (PSNR-based, evaluated at early_stop_interval)
                if self.early_stop:
                    self._early_stop_avg_psnr += eval_psnr
                    self._early_stop_eval_count += 1
                    if (curr_iter > self._early_stop_phase_start_iter
                            and (curr_iter - self._early_stop_phase_start_iter) % self.early_stop_interval == 0):
                        if self.enable_astc_compare:
                            # Use fntc_astc_{block} average PSNR as early stopping metric
                            astc_metrics = self.run_astc_comparison(curr_iter=curr_iter, output_root=self.media_path)
                            _astc_already_ran = True
                            fntc_astc_name = f"fntc_astc_{self.astc_block}"
                            fntc_astc_avg = astc_metrics.get(fntc_astc_name, {}).get("average")
                            if fntc_astc_avg is not None:
                                current_psnr = fntc_astc_avg[0]  # (psnr, ssim, lpips)
                                self.writer.add_scalar(f'EarlyStop/{fntc_astc_name}_avg_PSNR', current_psnr, curr_iter)
                            else:
                                # Fallback to eval PSNR if ASTC metrics unavailable
                                current_psnr = self._early_stop_avg_psnr / self._early_stop_eval_count
                            psnr_improvement = current_psnr - self._early_stop_prev_psnr
                        else:
                            self._early_stop_avg_psnr /= self._early_stop_eval_count
                            current_psnr = self._early_stop_avg_psnr
                            psnr_improvement = current_psnr - self._early_stop_prev_psnr
                        if self.enable_pbr_compare:
                            self.run_pbr_comparison(curr_iter=curr_iter, output_root=self.media_path)
                            _pbr_already_ran = True
                        label = f"{fntc_astc_name} avg PSNR" if self.enable_astc_compare else "PSNR"
                        print(f"[EarlyStopCheck] Iter {curr_iter}: {label} = {current_psnr:.4f} dB, "
                              f"improvement = {psnr_improvement:.4f} dB, threshold = {self.early_stop_psnr_threshold:.4f} dB")
                        if psnr_improvement < self.early_stop_psnr_threshold:
                            print(f"[EarlyStopCheck] PSNR improvement ({psnr_improvement:.4f} dB) < threshold "
                                  f"({self.early_stop_psnr_threshold:.4f} dB). Stopping training at iter {curr_iter}.")
                            # save model after all terminal comparisons
                            self.model.save(curr_iter, self.model_path)
                            self._log_checkpoint_interval(curr_iter)
                            # When enable_astc_compare is True, ASTC comparison was already run above for the metric check.
                            self._log_total_training_time()
                            return
                        else:
                            self._early_stop_prev_psnr = current_psnr
                            self._early_stop_avg_psnr = 0
                            self._early_stop_eval_count = 0


                # print(self.model.optimizer.param_groups[0]['lr'], self.model.optimizer.param_groups[1]['lr'])

            if curr_iter > 0 and curr_iter % self.save_interval == 0:
                if self.enable_astc_compare and not _astc_already_ran:
                    self.run_astc_comparison(curr_iter=curr_iter, output_root=self.media_path)
                if self.enable_pbr_compare and not _pbr_already_ran:
                    self.run_pbr_comparison(curr_iter=curr_iter, output_root=self.media_path)
                self.model.save(curr_iter, self.model_path)
                self._log_checkpoint_interval(curr_iter)

            if (self.enable_pbr_compare and curr_iter > 0
                    and curr_iter % self.pbr_compare_interval == 0
                    and curr_iter % self.save_interval != 0):
                self.run_pbr_comparison(curr_iter=curr_iter, output_root=self.media_path)

            if curr_iter > 0 and curr_iter % 10000 == 0:
                torch.cuda.empty_cache()
                tcnn.free_temporary_memory()

        self._log_total_training_time()

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

    @staticmethod
    def _mip_plane_tile(
        h0: int, h1: int, w0: int, w1: int, plane_h: int, plane_w: int, mip_f: float, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[N,3] batch and (row,col) indices; UV=(col+0.5)/W, (row+0.5)/H."""
        rr, cc = torch.meshgrid(
            torch.arange(h0, h1, device=device),
            torch.arange(w0, w1, device=device),
            indexing="ij",
        )
        z = torch.full_like(rr, mip_f)
        inp = torch.stack(((cc + 0.5) / plane_w, (rr + 0.5) / plane_h, z), dim=-1).reshape(-1, 3)
        return inp, rr.reshape(-1).long(), cc.reshape(-1).long()

    @torch.no_grad()
    def _fill_predicted_mip(self, mip: int, mip_height: int, mip_width: int) -> torch.Tensor:
        """Tiled full-plane forward (caps tcnn batch size; scatter by row/col to avoid layout bugs)."""
        device = self.device
        c = self.model.num_channels
        out = torch.empty(mip_height, mip_width, c, device=device, dtype=torch.float32)
        step = max(1, self.eval_inference_tile or max(mip_height, mip_width))
        mip_f = 0.0 if self.num_mips <= 1 else float(mip) / float(self.num_mips - 1)
        for h0 in range(0, mip_height, step):
            h1 = min(h0 + step, mip_height)
            for w0 in range(0, mip_width, step):
                w1 = min(w0 + step, mip_width)
                inp, fr, fc = self._mip_plane_tile(h0, h1, w0, w1, mip_height, mip_width, mip_f, device)
                if self.super_resolution_enable:
                    local_indices = torch.stack([fr, fc, torch.full_like(fr, mip)], dim=1)
                    local_indices[:, :2] *= 2 ** mip
                    base = self.dataset.get_superres_base(local_indices)
                    base = self.dataset.expand_to_canonical(base).float()
                    residual = self.model(torch.cat([inp, base], dim=1)).float()
                    out[fr, fc, :] = (base + residual).clamp(0.0, 1.0)
                else:
                    out[fr, fc, :] = self.model(inp).float()
        if getattr(self.model, "direct_diffuse_enabled", False) and "diffuse" in self.dataset.available_textures:
            s, e = self.dataset.canonical_channel_slices["diffuse"]
            direct = self.model.render_direct_diffuse_mip(mip, mip_height, mip_width)
            out[:, :, s:e] = direct.squeeze(0).permute(1, 2, 0).to(out.dtype)
        return out

    def _downsample_for_metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Area-downsample for PSNR/SSIM/LPIPS when the plane is large."""
        max_edge = self.eval_metrics_max_edge
        if max_edge <= 0:
            return pred, gt
        _, _, h, w = pred.shape
        edge = max(h, w)
        if edge <= max_edge:
            return pred, gt
        scale = max_edge / float(edge)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        return (
            F.interpolate(pred, size=(nh, nw), mode="area"),
            F.interpolate(gt, size=(nh, nw), mode="area"),
        )

    def _mip_size(self, mip: int) -> Tuple[int, int]:
        return self.texture_height // (2 ** mip), self.texture_width // (2 ** mip)

    def _ensure_visual_dirs(self, root: str) -> None:
        for vc in self.vis_configs:
            os.makedirs(os.path.join(root, vc['display_name']), exist_ok=True)

    def _canonical_gt_mip(self, mip: int, mip_height: int, mip_width: int) -> torch.Tensor:
        gt_slice = self.dataset.mip_cache[mip]
        gt_canonical = self.dataset.expand_to_canonical(
            gt_slice.reshape(-1, gt_slice.shape[-1])
        ).reshape(mip_height, mip_width, -1)
        return gt_canonical.permute(2, 0, 1)[None, ...]

    @torch.no_grad()
    def _render_mip_pair(self, mip: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        mip_height, mip_width = self._mip_size(mip)
        pred_hwc = self._fill_predicted_mip(mip, mip_height, mip_width).clamp(0, 1)
        pred = pred_hwc.permute(2, 0, 1)[None, ...]
        gt = self._canonical_gt_mip(mip, mip_height, mip_width)
        return pred, gt, mip_height, mip_width

    def _metric_rgb_pair(self, pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ms, me = self._get_metrics_slice()
        pred_rgb = pred[:, ms:me, :, :]
        pred_rgb = torch.nan_to_num(pred_rgb.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        gt_rgb = gt[:, ms:me, :, :]
        return self._downsample_for_metrics(pred_rgb, gt_rgb)

    def _metric_texture_name(self) -> str:
        return "diffuse" if "diffuse" in self.dataset.available_textures else self.dataset.available_textures[0]

    def _metric_visual_pair(self, pred_ref: torch.Tensor, gt_ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._metric_texture_name() == "normal":
            pred_ref = decode_normal(pred_ref, self.normal_encoding)
            gt_ref = decode_normal(gt_ref, self.normal_encoding)
        return pred_ref, gt_ref

    def _compute_mip_metrics(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mip_height: int,
        mip_width: int,
    ) -> Tuple[float, float, Optional[float]]:
        pred_ref, gt_ref = self._metric_rgb_pair(pred, gt)
        if self._metric_texture_name() == "normal":
            psnr_value = float(normal_angular_psnr(pred_ref, gt_ref, self.normal_encoding).item())
        else:
            psnr_value = float(self.psnr(pred_ref, gt_ref).item())
        visual_pred, visual_gt = self._metric_visual_pair(pred_ref, gt_ref)
        ssim_value, _ = self.ssim(visual_pred, visual_gt)
        ssim_value = float(ssim_value.item())
        if mip_height >= 128 and mip_width >= 128:
            if visual_pred.shape[1] == 1:
                visual_pred = visual_pred.repeat(1, 3, 1, 1)
                visual_gt = visual_gt.repeat(1, 3, 1, 1)
            elif visual_pred.shape[1] == 2:
                visual_pred = visual_pred[:, :1].repeat(1, 3, 1, 1)
                visual_gt = visual_gt[:, :1].repeat(1, 3, 1, 1)
            return psnr_value, ssim_value, float(self.lpips(visual_pred, visual_gt).item())
        return psnr_value, ssim_value, None

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

    def _save_mip_visuals(self, pred: torch.Tensor, gt: torch.Tensor, output_root: str, filename: str) -> None:
        save_image = torch.cat([pred, gt], dim=3).squeeze()
        for vc in self.vis_configs:
            s, e = vc['canonical_channel_slice']
            tex_image = self._postprocess_for_vis(save_image[s:e, ...], vc['vis_mode'])
            save_path = os.path.join(output_root, vc['display_name'], filename)
            TF.to_pil_image(tex_image).save(save_path)

    @staticmethod
    def _mean_metric(values: List[float]) -> float:
        return float(torch.tensor(values).mean().item()) if values else 0.0

    @torch.no_grad()
    def eval(self, curr_iter) -> float:
        self._ensure_visual_dirs(self.media_path)
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

                if should_save_visuals:
                    self._save_mip_visuals(pred, gt, self.media_path, f"{curr_iter}_{mip}.png")

        finally:
            self._restore_feature_grid_state(feature_grid_backup)
            self.model.train(was_training)
            self._cuda_trim_eval_mem(synchronize=False)

        psnr_avg = self._mean_metric(psnr_list)
        ssim_avg = self._mean_metric(ssim_list)
        lpips_avg = self._mean_metric(lpips_list)
        self.writer.add_scalar('Material/average/PSNR/train', psnr_avg, curr_iter)
        self.writer.add_scalar('Material/average/SSIM/train', ssim_avg, curr_iter)
        self.writer.add_scalar('Material/average/LPIPS/train', lpips_avg, curr_iter)

        print(f"Iter:{curr_iter}, MaterialPSNR:{psnr_avg:.4f}. {self._loss_stats_str()}")
        return psnr_avg

    def _postprocess_for_vis(self, image: torch.Tensor, vis_mode: str) -> torch.Tensor:
        """Apply visualization post-processing based on vis_mode.
        Args:
            image: [C, H, W] tensor in [0, 1]
            vis_mode: 'srgb' | 'normal' | 'linear'
        """
        image = torch.clamp(image, 0.0, 1.0)
        if vis_mode == 'srgb':
            image = torch.pow(image, 1.0 / 2.2)
        elif vis_mode == 'normal':
            n = image * 2.0 - 1.0
            norm = torch.sqrt(torch.clamp((n ** 2).sum(dim=0, keepdim=True), min=1e-8))
            n = n / norm
            image = torch.clamp((n + 1.0) * 0.5, 0.0, 1.0)
        elif vis_mode == 'normal_encoded':
            image = decode_normal(image, self.normal_encoding)
        # 'linear' -> no transform
        return image

    def _get_metrics_slice(self):
        """Return (start, end) channel indices in canonical 11-channel space for metrics.
        Prefers 'diffuse' (0:3); falls back to the first available texture's canonical slice.
        """
        canon = self.dataset.canonical_channel_slices
        if 'diffuse' in self.dataset.available_textures:
            return canon['diffuse']
        first = self.dataset.available_textures[0]
        return canon[first]

    def _infer_mips(self) -> range:
        return inference_mips(self.num_mips, self.mip0_only)

    @torch.no_grad()
    def infer(self) -> None:
        self._ensure_visual_dirs(self.infer_path)

        psnr_list: List[float] = []
        ssim_list: List[float] = []
        lpips_list: List[float] = []
        if self._metric_texture_name() == "normal":
            metric_lines = ["Mip NormalAngularPSNR NormalAngularSSIM NormalAngularLPIPS\n"]
        else:
            metric_lines = ["Mip PSNR SSIM LPIPS\n"]
        material_metric_lines = ["Mip Group PSNR SSIM LPIPS\n"]
        material_group_order = ("diffuse", "normal", "romsd", "average")

        was_training = self.model.training
        try:
            self.model.eval()
            for mip in self._infer_mips():
                pred, gt, mip_height, mip_width = self._render_mip_pair(mip)
                material_metrics = self._compute_material_metrics(pred, gt)
                psnr_value, ssim_value, lpips_value = material_metrics["average"]

                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
                if lpips_value is not None:
                    lpips_list.append(lpips_value)
                lpips_for_line = 0.0 if lpips_value is None else lpips_value

                self._save_mip_visuals(pred, gt, self.infer_path, f"Mip_{mip}.png")
                metric_lines.append(f"Mip_{mip} {psnr_value:.4f} {ssim_value:.4f} {lpips_for_line:.4f}\n")
                for group_name in material_group_order:
                    group_metric = material_metrics.get(group_name)
                    if group_metric is None:
                        continue
                    material_metric_lines.append(
                        f"Mip_{mip} {group_name} {group_metric[0]:.4f} "
                        f"{group_metric[1]:.4f} {group_metric[2]:.4f}\n"
                    )
        finally:
            self.model.train(was_training)

        psnr_avg = self._mean_metric(psnr_list)
        ssim_avg = self._mean_metric(ssim_list)
        lpips_avg = self._mean_metric(lpips_list)
        metric_lines.append(f"AVER {psnr_avg} {ssim_avg} {lpips_avg}\n")
        with open(os.path.join(self.infer_path, "metrics.txt"), "w+", encoding="utf-8") as file:
            file.writelines(metric_lines)
        material_metric_lines.append(
            f"AVER average {psnr_avg:.4f} {ssim_avg:.4f} {lpips_avg:.4f}\n"
        )
        with open(os.path.join(self.infer_path, "metrics_material.txt"), "w+", encoding="utf-8") as file:
            file.writelines(material_metric_lines)

        if self.enable_astc_compare:
            self.run_astc_comparison(output_root=self.infer_path)
        if self.enable_pbr_compare:
            self.run_pbr_comparison(output_root=self.infer_path)

    @torch.no_grad()
    def run_astc_comparison(self, curr_iter: int = None, output_root: str = None) -> dict:
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
        )

    @torch.no_grad()
    def run_pbr_comparison(self, curr_iter: int = None, output_root: str = None) -> str:
        """Render the inferred and source Mip0 texture sets in one fixed PBR scene."""
        if output_root is None:
            output_root = self.infer_path if hasattr(self, "infer_path") else self.media_path
        was_training = self.model.training
        self.model.eval()
        try:
            gt = _render_gt_mip0(self.dataset, self.texture_height, self.texture_width)
            quant_model = copy.deepcopy(self.model)
            quant_model.simulate_quantize()
            quant_model.eval()
            fntc_astc_model = copy.deepcopy(quant_model)
            apply_astc_to_feature_grids(fntc_astc_model, self.astc_codec.roundtrip_rgba)
            fntc_astc_model.eval()
            astc_base = None
            if self.super_resolution_enable:
                astc_base = _astc_roundtrip_superres_base_mip0(
                    self.dataset,
                    self.astc_codec.roundtrip_rgba,
                    int(self.configs.super_resolution_base_resolution),
                )
            fntc_astc = _render_model_mip0(
                fntc_astc_model,
                self.dataset,
                self.texture_height,
                self.texture_width,
                self.num_mips,
                self.device,
                superres_base_override=astc_base,
            )
            ref_h, ref_w = _ref_astc_hw_from_side(
                self.ref_astc_resolution,
                self.texture_height,
                self.texture_width,
            )
            traditional_astc = _traditional_baseline_resampled(
                gt_mip0=gt,
                dataset=self.dataset,
                roundtrip_rgba=self.astc_codec.roundtrip_rgba,
                ref_h=ref_h,
                ref_w=ref_w,
            )
            path, metrics = save_pbr_comparison(
                fntc_astc_texture=fntc_astc,
                traditional_astc_texture=traditional_astc,
                gt_texture=gt,
                dataset=self.dataset,
                output_root=output_root,
                curr_iter=curr_iter,
                resolution=self.pbr_render_resolution,
                astc_block=self.astc_block,
                ssim_metric=self.ssim,
                lpips_metric=self.lpips,
                mip_lod_bias=self.pbr_mip_lod_bias,
                lighting=self.pbr_lighting,
            )
            if curr_iter is not None:
                for method, (psnr, ssim, lpips) in metrics.items():
                    self.writer.add_scalar(f"PBRCompare/{method}_PSNR", psnr, curr_iter)
                    self.writer.add_scalar(f"PBRCompare/{method}_SSIM", ssim, curr_iter)
                    self.writer.add_scalar(f"PBRCompare/{method}_LPIPS", lpips, curr_iter)
            block_tag = self.astc_block.replace("x", "_").lower()
            fntc_metrics = metrics.get("fntc", (0.0, 0.0, 0.0))
            astc_metrics = metrics.get("astc", (0.0, 0.0, 0.0))
            print(
                f"[PBR Test] Iter {curr_iter}: "
                f"FNTC_{block_tag}_compressed "
                f"PSNR={fntc_metrics[0]:.4f}, SSIM={fntc_metrics[1]:.4f}, LPIPS={fntc_metrics[2]:.4f}; "
                f"ASTC_{block_tag}_compressed "
                f"PSNR={astc_metrics[0]:.4f}, SSIM={astc_metrics[1]:.4f}, LPIPS={astc_metrics[2]:.4f}; "
                f"output={path}"
            )
            return path
        finally:
            self.model.train(was_training)

    def _write_resolved_config(self, configs: Config) -> None:
        """Persist scalar/list/dict config values so log analyzers can recover run settings."""
        out = {}
        for key, value in vars(configs).items():
            if key.startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
            elif isinstance(value, (list, tuple, dict)):
                try:
                    json.dumps(value)
                    out[key] = value
                except TypeError:
                    out[key] = str(value)
        path = os.path.join(self.save_path, "resolved_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"resolved_config: {path}")

    def generate_probabilities(self) -> torch.Tensor:
        # Generate a coarse-to-fine Mip sampling distribution, or isolate Mip0.
        return mip_sampling_probabilities(self.num_mips, self.mip0_only, self.device)


if __name__ == "__main__":

    params = get_args()

    configs = Config(params)

    if configs.two_stage_finetune_enable and params.mode == "train" and not configs.groups_batching:
        from two_stage_finetune import train_two_stage
        train_two_stage(params, configs)
        sys.exit(0)

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
                if params.mode == "train" and group_cfg.two_stage_finetune_enable:
                    from two_stage_finetune import train_two_stage
                    train_two_stage(params, group_cfg)
                else:
                    trainer = Trainer(params, config_override=group_cfg)
                    if params.mode == "train":
                        trainer.train()
                    elif params.mode == "infer":
                        trainer.infer()
                    else:
                        raise ValueError("Error mode.")
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
        # Single-group mode (original behavior)
        trainer = Trainer(params)

        if params.mode == "train":
            trainer.train()
        elif params.mode == "infer":
            trainer.infer()
        else:
            raise ValueError("Error mode.")
