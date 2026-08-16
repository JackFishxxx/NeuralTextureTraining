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
from normal_encoding import decode_normal, normal_angular_loss, normal_angular_psnr
from configs import Config
from Comparison_ASTC import (
    ASTCCodec,
    ASTCENC_DEFAULT_PATH,
    _astc_roundtrip_superres_base_mip0,
    _render_gt_mip0,
    _render_model_mip0,
    _ref_astc_hw_from_side,
    _traditional_baseline_resampled,
    apply_astc_to_feature_grids,
    run_astc_comparison_pipeline,
)
from Comparison_PBRScene import save_pbr_comparison


class Trainer:

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
        self.astc_codec_loss_weight = float(getattr(configs, "astc_codec_loss_weight", 0.7))
        self.astc_clean_loss_weight = float(getattr(configs, "astc_clean_loss_weight", 0.3))
        self.astc_consistency_weight = float(getattr(configs, "astc_consistency_weight", 0.1))
        self.astc_codec_mip0_prob = float(getattr(configs, "astc_codec_mip0_prob", 0.5))
        self.astc_codec_start_frac = float(getattr(configs, "astc_codec_start_frac", 0.0))
        self.astc_codec_cache_iter = None
        self.astc_codec = ASTCCodec(
            astcenc_path=self.astcenc_path,
            astcenc_quality=self.astcenc_quality,
            astc_block=self.astc_block,
        )
        if (self.enable_astc_compare or self.enable_pbr_compare
                or getattr(configs, "astc_codec_in_loop_enable", False)):
            self.astc_codec.ensure_executable()
        self.astc_superres_base_mip0 = None
        if getattr(configs, "astc_codec_in_loop_enable", False) and self.super_resolution_enable:
            astc_base = _astc_roundtrip_superres_base_mip0(
                dataset, self.astc_codec.roundtrip_rgba, int(configs.super_resolution_base_resolution)
            )
            self.astc_superres_base_mip0 = astc_base[0].permute(1, 2, 0).contiguous()

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
        idx = []
        for tex in textures:
            if tex not in self.dataset.available_textures:
                continue
            s, e = self.dataset.canonical_channel_slices[tex]
            idx.extend(range(s, e))
        return torch.tensor(idx, device=device, dtype=torch.long)

    def _compute_pbr_render_loss(self, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Differentiable shaded-RGB loss for PBR-sensitive material channels."""
        slices = self.dataset.canonical_channel_slices
        def safe_decode_normal(encoded: torch.Tensor) -> torch.Tensor:
            """Decode tangent normals without sqrt-at-zero NaN gradients."""
            enc = self.normal_encoding.lower().replace('-', '_')
            if enc in {'xy', 'hemi_oct', 'hemi_octa'}:
                xy = encoded[:, :2] * 2.0 - 1.0
                radius = torch.sqrt(torch.clamp((xy * xy).sum(1, keepdim=True), min=1e-12))
                # Keep the square-root argument strictly positive. This is
                # essential because d(sqrt(x))/dx is infinite at x=0.
                xy = xy * torch.clamp(0.999 / radius, max=1.0)
                if enc == 'xy':
                    z = torch.sqrt(torch.clamp(1.0 - (xy * xy).sum(1, keepdim=True), min=1e-4))
                    return torch.cat([xy, z], dim=1)
                px = (xy[:, :1] + xy[:, 1:2]) * 0.5
                py = (xy[:, :1] - xy[:, 1:2]) * 0.5
                pz = 1.0 - px.abs() - py.abs()
                oct_n = torch.cat([px, py, pz], dim=1)
                return oct_n / torch.sqrt(torch.clamp((oct_n * oct_n).sum(1, keepdim=True), min=1e-4))
            # XYZ encoding has no square-root boundary; normalize with epsilon.
            xyz = encoded[:, :3] * 2.0 - 1.0
            return xyz / torch.sqrt(torch.clamp((xyz * xyz).sum(1, keepdim=True), min=1e-4))
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

        gt_n = safe_decode_normal(gt_norm)
        pr_n = safe_decode_normal(pr_norm)
        # Use a fixed tangent-space light with positive Z so a flat normal map
        # receives direct light and all PBR channels get useful gradients.
        light = torch.tensor([-0.35, 0.45, 0.82], device=pred.device, dtype=pred.dtype)
        light = light / torch.clamp(torch.linalg.vector_norm(light), min=1e-6)
        view = torch.tensor([0.0, 0.0, 1.0], device=pred.device, dtype=pred.dtype)
        half = (light + view) / torch.clamp(torch.linalg.vector_norm(light + view), min=1e-6)
        def shade(diff, normal, rough, metal, spec):
            diff = torch.clamp(diff, 0, 1) ** 2.2
            nvec = normal * 2.0 - 1.0
            normal = nvec / torch.sqrt(torch.clamp((nvec * nvec).sum(1, keepdim=True), min=1e-4))
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
            "romd": ["roughness", "occlusion", "metallic", "displacement"],
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
            for group, label in (("diffuse", "diffuse"), ("normal", "normal"), ("romd", "ROMD")):
                stat = residual_stat.get(group)
                if stat is not None:
                    terms.append(f"{label} loss:{stat['loss']:.6f}")
            decomposition = " + ".join(terms)
            return f"Weighted super-resolution residual loss:{residual_stat['loss']:.6f} = {decomposition}"
        labels = {
            "diffuse": "Weighted diffuse loss",
            "normal": "Weighted normal angular loss",
            "romd": "Weighted ROMD loss",
        }
        parts = []
        for group in ("diffuse", "normal", "romd"):
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
            self.writer.add_scalar('Loss/superres_residual', loss.item(), curr_iter)
            stats = {"loss": float(loss.detach().item())}
            for group, textures in {
                "diffuse": ["diffuse"],
                "normal": ["normal"],
                "romd": ["roughness", "occlusion", "metallic", "displacement"],
            }.items():
                idx = self._group_indices(textures, target_residual.device)
                if idx.numel() == 0:
                    continue
                group_loss = (mse.index_select(0, idx) * loss_weights.index_select(0, idx).float()).sum()
                stats[group] = {"loss": float(group_loss.detach().item())}
                self.writer.add_scalar(f'Loss/superres_residual_{group}', stats[group]["loss"], curr_iter)
            self._last_loss_stats = {"superres_residual": stats}
        return loss

    def train(self) -> None:

        for curr_iter in range(self.trained_iter, self.max_iter):

            # Enable the ASTC codec branch only after its configured warm-up fraction.
            codec_phase_active = (
                bool(getattr(self.configs, "astc_codec_in_loop_enable", False))
                and curr_iter >= int(self.max_iter * self.astc_codec_start_frac)
            )
            codec_start_iter = int(self.max_iter * self.astc_codec_start_frac)
            codec_interval = int(self.configs.astc_codec_in_loop_interval)
            calibration_due = (
                codec_phase_active
                and curr_iter > 0
                and (
                    self.astc_codec_cache_iter is None
                    or (curr_iter - codec_start_iter) % codec_interval == 0
                )
            )
            if calibration_due:
                self._calibrate_astc_aware_latent(curr_iter)

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
            gt_texture = self.dataset(batch_index)  # [batch_size, num_channels]
            gt_texture = self.dataset.expand_to_canonical(gt_texture).to(torch.float16)
            superres_base = None
            if self.super_resolution_enable:
                superres_base = self.dataset.expand_to_canonical(
                    self.dataset.get_superres_base(batch_index)
                ).to(torch.float16)

            # xys -> uvs
            # shift the sample position from [0, 1, ..., 1023] -> [0.5, 1.5, ..., 1023.5]
            # uvs = ((xys + 0.5) / mip_scale) / (texture_weight / mip_scale)
            us = (xs + 0.5) / self.texture_width
            vs = (ys + 0.5) / self.texture_height
            mips = mips.float() / (self.num_mips - 1) if self.num_mips > 1 else torch.zeros_like(mips, dtype=torch.float32)
            batch_input = torch.cat([us, vs, mips], dim=1)
            if superres_base is not None:
                batch_input = torch.cat([batch_input, superres_base], dim=1)
            # Clean QAT branch remains active to preserve the uncompressed latent solution.
            self.model.astc_codec_branch_enabled = False
            predict_texture = self.model(batch_input)  # [batch_size, num_channels]

            # base reconstruction loss; normal uses angular loss, others use channel MSE
            loss_weights = torch.tensor(self.output_loss_weights, device=self.device, dtype=torch.float32)
            if superres_base is not None:
                target_residual = gt_texture - superres_base
                base_loss = self._compute_superres_residual_loss(
                    target_residual, predict_texture, loss_weights, curr_iter
                )
            else:
                base_loss = self._compute_reconstruction_loss(gt_texture, predict_texture, loss_weights, curr_iter)

            total_loss = base_loss
            pbr_clean_loss = None
            pbr_codec_loss = None
            if self.pbr_loss_enable and self.pbr_loss_weight > 0.0:
                pbr_predict = ((superres_base + predict_texture).clamp(0.0, 1.0)
                               if superres_base is not None else predict_texture.clamp(0.0, 1.0))
                pbr_clean_loss = self._compute_pbr_render_loss(gt_texture.float(), pbr_predict.float())
            if self.subpixel_sampling_enable and self.subpixel_sampling_ratio > 0.0:
                n_sub = max(1, int(round(self.batch_size * self.subpixel_sampling_ratio)))
                sub_mips = self.sample_probabilities.multinomial(n_sub, replacement=True)
                sub_xy = torch.stack([
                    torch.randint(0, self.texture_height, (n_sub,), device=self.device).float(),
                    torch.randint(0, self.texture_width, (n_sub,), device=self.device).float(),
                ], dim=1)
                sub_xy += (torch.rand((n_sub, 2), device=self.device) * 2.0 - 1.0) * self.subpixel_jitter
                sub_gt = self.dataset.expand_to_canonical(self.dataset.sample_continuous(sub_xy, sub_mips)).to(torch.float16)
                sub_uv = torch.stack([(sub_xy[:, 1] + 0.5) / self.texture_width,
                                      (sub_xy[:, 0] + 0.5) / self.texture_height], dim=1)
                sub_mip_uv = sub_mips.float()[:, None] / (self.num_mips - 1) if self.num_mips > 1 else torch.zeros((n_sub, 1), device=self.device)
                sub_input = torch.cat([sub_uv, sub_mip_uv], dim=1)
                if self.super_resolution_enable:
                    sub_base = self.dataset.expand_to_canonical(self.dataset.get_superres_base_continuous(sub_xy, sub_mips)).to(torch.float16)
                    sub_input = torch.cat([sub_input, sub_base], dim=1)
                    sub_pred = self.model(sub_input)
                    sub_loss = self._compute_superres_residual_loss(sub_gt - sub_base, sub_pred, loss_weights)
                else:
                    sub_pred = self.model(sub_input)
                    sub_loss = self._compute_reconstruction_loss(sub_gt, sub_pred, loss_weights)
                total_loss = total_loss + self.subpixel_sampling_ratio * sub_loss
            codec_loss = None
            consistency_loss = None
            mip0 = batch_index[:, 2] == 0
            codec_select = mip0
            if codec_phase_active and self.astc_codec_mip0_prob < 1.0:
                codec_select = codec_select & (
                    torch.rand(batch_index.shape[0], device=self.device) < self.astc_codec_mip0_prob
                )
            if codec_phase_active and bool(codec_select.any()):
                codec_base = superres_base[codec_select] if superres_base is not None else None
                codec_input = batch_input[codec_select]
                codec_gt = gt_texture[codec_select]
                if codec_base is not None and self.astc_superres_base_mip0 is not None:
                    codec_base = self.astc_superres_base_mip0[ys[codec_select, 0], xs[codec_select, 0]].to(superres_base.dtype)
                    codec_input = torch.cat([batch_input[codec_select, :3], codec_base], dim=1)
                self.model.astc_codec_branch_enabled = True
                try:
                    codec_predict = self.model(codec_input)
                finally:
                    self.model.astc_codec_branch_enabled = False
                if codec_base is not None:
                    codec_target = codec_gt - codec_base
                    codec_loss = self._compute_superres_residual_loss(
                        codec_target, codec_predict, loss_weights, curr_iter=None
                    )
                else:
                    codec_loss = self._compute_reconstruction_loss(
                        codec_gt, codec_predict, loss_weights, curr_iter=None
                    )
                # The two branches intentionally use different bases.  Comparing
                # final colors therefore penalizes unavoidable ASTC base error.
                # Compare predicted residuals instead, which isolates decoder
                # consistency and does not fight the codec reconstruction loss.
                consistency_loss = (codec_predict.float() - predict_texture[codec_select].float().detach()).pow(2).mean()
                total_loss = (self.astc_clean_loss_weight * base_loss
                              + self.astc_codec_loss_weight * codec_loss
                              + self.astc_consistency_weight * consistency_loss)
                if self.pbr_loss_enable and self.pbr_loss_weight > 0.0:
                    codec_final = (codec_base + codec_predict).clamp(0.0, 1.0) if codec_base is not None else codec_predict
                    pbr_codec_loss = self._compute_pbr_render_loss(codec_gt.float(), codec_final.float())

            if pbr_clean_loss is not None:
                effective_pbr_loss = pbr_clean_loss
                if pbr_codec_loss is not None:
                    effective_pbr_loss = (self.astc_clean_loss_weight * pbr_clean_loss
                                          + self.astc_codec_loss_weight * pbr_codec_loss)
                total_loss = total_loss + self.pbr_loss_weight * effective_pbr_loss
                self.writer.add_scalar('Loss/pbr_render', effective_pbr_loss.item(), curr_iter)
                self.writer.add_scalar('Loss/pbr_render_clean', pbr_clean_loss.item(), curr_iter)
                if pbr_codec_loss is not None:
                    self.writer.add_scalar('Loss/pbr_render_astc', pbr_codec_loss.item(), curr_iter)
                self.writer.add_scalar('LossWeighted/pbr_render',
                                       self.pbr_loss_weight * effective_pbr_loss.item(), curr_iter)

            self.writer.add_scalar('Loss/train', total_loss.item(), curr_iter)
            self.writer.add_scalar('Loss/base', base_loss.item(), curr_iter)
            if codec_loss is not None:
                self.writer.add_scalar('Loss/astc_codec', codec_loss.item(), curr_iter)
                self.writer.add_scalar('Loss/astc_consistency', consistency_loss.item(), curr_iter)
                self.writer.add_scalar('ASTCInLoop/mip0_fraction', codec_select.float().mean().item(), curr_iter)
                if self.astc_codec_cache_iter is not None:
                    self.writer.add_scalar('ASTCInLoop/cache_age', float(curr_iter - self.astc_codec_cache_iter), curr_iter)

            # optimize
            total_loss.backward()
            if self.pbr_loss_enable:
                # The PBR proxy contains a specular reciprocal term; cap its
                # occasional large batch gradient before the optimizer step.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.model.optimizer.step()
            self.model.scheduler.step(metrics=total_loss.item())

            # print(self.model.optimizer.param_groups[0]['lr'], self.model.optimizer.param_groups[1]['lr'])

            self.model.clamp_value()

            # track whether ASTC comparison was already run this iteration (avoid duplicate runs)
            _astc_already_ran = False

            # eval
            if curr_iter % self.eval_interval == 0:
                eval_psnr = self.eval(curr_iter)

                # early stopping check (PSNR-based, evaluated at early_stop_interval)
                if self.early_stop:
                    self._early_stop_avg_psnr += eval_psnr
                    self._early_stop_eval_count += 1
                    if curr_iter > 0 and curr_iter % self.early_stop_interval == 0:
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
                            print(f"[EarlyStopCheck] Iter {curr_iter}: {fntc_astc_name} avg PSNR = {current_psnr:.4f} dB, "
                                  f"improvement = {psnr_improvement:.4f} dB, threshold = {self.early_stop_psnr_threshold:.4f} dB")
                        else:
                            self._early_stop_avg_psnr /= self._early_stop_eval_count
                            current_psnr = self._early_stop_avg_psnr
                            psnr_improvement = current_psnr - self._early_stop_prev_psnr
                            print(f"[EarlyStopCheck] Iter {curr_iter}: PSNR = {current_psnr:.4f} dB, "
                                  f"improvement = {psnr_improvement:.4f} dB, threshold = {self.early_stop_psnr_threshold:.4f} dB")
                        if psnr_improvement < self.early_stop_psnr_threshold:
                            print(f"[EarlyStopCheck] PSNR improvement ({psnr_improvement:.4f} dB) < threshold "
                                  f"({self.early_stop_psnr_threshold:.4f} dB). Stopping training at iter {curr_iter}.")
                            # save model before stopping
                            self.model.save(curr_iter, self.model_path)
                            self._log_checkpoint_interval(curr_iter)
                            # When enable_astc_compare is True, ASTC comparison was already run above for the metric check
                            self._log_total_training_time()
                            return
                        else:
                            self._early_stop_prev_psnr = current_psnr
                            self._early_stop_avg_psnr = 0
                            self._early_stop_eval_count = 0


                # print(self.model.optimizer.param_groups[0]['lr'], self.model.optimizer.param_groups[1]['lr'])

            if curr_iter > 0 and curr_iter % self.save_interval == 0:
                self.model.save(curr_iter, self.model_path)
                if self.enable_astc_compare and not _astc_already_ran:
                    self.run_astc_comparison(curr_iter=curr_iter, output_root=self.media_path)
                self._log_checkpoint_interval(curr_iter)

            if (self.enable_pbr_compare and curr_iter > 0
                    and curr_iter % self.pbr_compare_interval == 0):
                self.run_pbr_comparison(curr_iter=curr_iter, output_root=self.media_path)

            if curr_iter > 0 and curr_iter % 10000 == 0:
                torch.cuda.empty_cache()
                tcnn.free_temporary_memory()

        self._log_total_training_time()

    @torch.no_grad()
    def _calibrate_astc_aware_latent(self, curr_iter: int) -> None:
        """Measure real astcenc latent error and update the training proxy amplitude.

        The codec call is outside autograd. Subsequent training iterations use the
        measured RMS code error in the differentiable block-correlated proxy.
        """
        probe = copy.deepcopy(self.model).eval()
        probe.simulate_quantize()
        before = []
        for grid in probe.feature_grids:
            before.append(grid.state_dict()["params"].detach().clone())
        apply_astc_to_feature_grids(probe, self.astc_codec.roundtrip_rgba)
        sq_sum = 0.0
        count = 0
        for grid_idx, (old, grid, spec) in enumerate(zip(
                before, probe.feature_grids, probe.feature_grid_specs)):
            new = grid.state_dict()["params"].detach()
            offset, level_count, resolution = spec.highest_level_slice()
            delta = new[offset:offset + level_count] - old[offset:offset + level_count]
            decoded_texture = new[offset:offset + level_count].reshape(
                resolution, resolution, spec.n_features_per_level
            )
            source_texture = old[offset:offset + level_count].reshape(
                resolution, resolution, spec.n_features_per_level
            )
            self.model.set_astc_codec_decoded_texture(grid_idx, decoded_texture, source_texture)
            delta_codes = delta * float(spec.quant_step_count)
            sq_sum += float(delta_codes.float().pow(2).sum().item())
            count += delta_codes.numel()
            channel_rms = delta_codes.reshape(-1, spec.n_features_per_level).float().pow(2).mean(0).sqrt()
            for channel, value in enumerate(channel_rms.tolist()):
                self.writer.add_scalar(
                    f"ASTCInLoop/grid{grid_idx}_channel{channel}_rms_codes", value, curr_iter
                )
        rms_codes = (sq_sum / max(1, count)) ** 0.5
        proxy_scale = min(8.0, rms_codes * (12.0 ** 0.5))
        self.model.set_astc_codec_noise_scale(proxy_scale)
        self.astc_codec_cache_iter = curr_iter
        self.writer.add_scalar("ASTCInLoop/latent_rms_codes", rms_codes, curr_iter)
        self.writer.add_scalar("ASTCInLoop/proxy_noise_scale", proxy_scale, curr_iter)
        print(f"[ASTCInLoop] Iter {curr_iter}: latent RMS={rms_codes:.4f} codes, proxy scale={proxy_scale:.4f}")

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
                    base = self.dataset.superres_base_cache[mip, fr, fc, :]
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
        gt_slice = self.dataset.mip_cache[mip, :mip_height, :mip_width, :]
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
            return psnr_value, ssim_value, float(self.lpips(visual_pred, visual_gt).item())
        return psnr_value, ssim_value, None

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
        should_save_visuals = curr_iter % self.save_interval == 0

        feature_grid_backup = self._backup_feature_grid_state()
        was_training = self.model.training
        try:
            self.model.eval()
            self.model.simulate_quantize()

            # Training evaluation intentionally checks Mip0 only to keep the loop responsive.
            for mip in [0]:
                pred, gt, mip_height, mip_width = self._render_mip_pair(mip)
                psnr_value, ssim_value, lpips_value = self._compute_mip_metrics(pred, gt, mip_height, mip_width)

                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
                metric_prefix = "NormalAngular" if self._metric_texture_name() == "normal" else ""
                self.writer.add_scalar(f'{metric_prefix}PSNR_Mip{mip}/train', psnr_value, curr_iter)
                self.writer.add_scalar(f'{metric_prefix}SSIM_Mip{mip}/train', ssim_value, curr_iter)
                if lpips_value is not None:
                    lpips_list.append(lpips_value)
                    self.writer.add_scalar(f'{metric_prefix}LPIPS_Mip{mip}/train', lpips_value, curr_iter)

                if should_save_visuals:
                    self._save_mip_visuals(pred, gt, self.media_path, f"{curr_iter}_{mip}.png")

        finally:
            self._restore_feature_grid_state(feature_grid_backup)
            self.model.train(was_training)
            self._cuda_trim_eval_mem(synchronize=False)

        psnr_avg = self._mean_metric(psnr_list)
        ssim_avg = self._mean_metric(ssim_list)
        lpips_avg = self._mean_metric(lpips_list)
        metric_prefix = "NormalAngular" if self._metric_texture_name() == "normal" else ""
        self.writer.add_scalar(f'{metric_prefix}PSNR/train', psnr_avg, curr_iter)
        self.writer.add_scalar(f'{metric_prefix}SSIM/train', ssim_avg, curr_iter)
        self.writer.add_scalar(f'{metric_prefix}LPIPS/train', lpips_avg, curr_iter)

        psnr_label = "NormalAngularPSNR" if self._metric_texture_name() == "normal" else "DiffusePSNR"
        print(f"Iter:{curr_iter}, {psnr_label}:{psnr_avg:.4f}. {self._loss_stats_str()}")
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
        # Skip the smallest four mip levels during inference, but always keep
        # Mip0 so tiny datasets still produce an output and metrics file.
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

        was_training = self.model.training
        try:
            self.model.eval()
            for mip in self._infer_mips():
                pred, gt, mip_height, mip_width = self._render_mip_pair(mip)
                psnr_value, ssim_value, lpips_value = self._compute_mip_metrics(pred, gt, mip_height, mip_width)

                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
                if lpips_value is not None:
                    lpips_list.append(lpips_value)
                lpips_for_line = 0.0 if lpips_value is None else lpips_value

                self._save_mip_visuals(pred, gt, self.infer_path, f"Mip_{mip}.png")
                metric_lines.append(f"Mip_{mip} {psnr_value:.4f} {ssim_value:.4f} {lpips_for_line:.4f}\n")
        finally:
            self.model.train(was_training)

        psnr_avg = self._mean_metric(psnr_list)
        ssim_avg = self._mean_metric(ssim_list)
        lpips_avg = self._mean_metric(lpips_list)
        metric_lines.append(f"AVER {psnr_avg} {ssim_avg} {lpips_avg}\n")
        with open(os.path.join(self.infer_path, "metrics.txt"), "w+", encoding="utf-8") as file:
            file.writelines(metric_lines)

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
