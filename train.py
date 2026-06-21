import os
import sys
import torch
import torch.nn.functional as F
import datetime
import argparse
from typing import List, Optional, Tuple
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)

from tensorboardX import SummaryWriter
import torchvision.transforms.functional as TF

import tinycudann as tcnn

from configs import get_args
from model import TCNNModel
from dataset import TextureDataset
from normal_encoding import decode_normal, normal_angular_loss, normal_angular_psnr
from configs import Config
from Comparison_ASTC import ASTCCodec, ASTCENC_DEFAULT_PATH, run_astc_comparison_pipeline
from astc_residual_model import ASTCResidualNoiseModel


class Trainer:

    def __init__(self, params: argparse.Namespace, config_override: Config = None) -> None:

        # initialize required runtime fieldsvariables, e.g. dataset, configs and models
        if config_override is not None:
            configs = config_override
        else:
            configs = Config(params)
        dataset = TextureDataset(configs)
        print(f"[NormalEncoding] mode={dataset.normal_encoding}")
        configs.num_lods = dataset.num_lods
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
        if configs.load_iter != 0:
            self.infer_path = os.path.join(configs.load_dir, "infer")
            os.makedirs(self.infer_path, exist_ok=True)

        # data config
        self.num_lods = dataset.num_lods
        self.texture_height = dataset.texture_height
        self.texture_width = dataset.texture_width

        # 11-channel loss weights; channels for missing textures are 0
        self.output_loss_weights = configs.output_loss_weights or dataset.get_canonical_loss_weights(
            config_weights=getattr(configs, 'texture_loss_weights', None)
        )
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
        self.normal_encoding = str(getattr(configs, "normal_encoding", "xyz")).lower()
        self.astc_codec = ASTCCodec(
            astcenc_path=self.astcenc_path,
            astcenc_quality=self.astcenc_quality,
            astc_block=self.astc_block,
        )
        if self.enable_astc_compare:
            self.astc_codec.ensure_executable()

        # ASTC residual-aware robust training (train-only; inference unchanged)
        self.astc_aware_train = bool(getattr(configs, "astc_aware_train", False))
        self.astc_curriculum_enable = bool(getattr(configs, "astc_curriculum_enable", False))
        self.consistency_loss_enable = bool(getattr(configs, "consistency_loss_enable", False))
        self.consistency_lambda = float(getattr(configs, "consistency_lambda", 0.0))
        self.consistency_start_ratio = float(getattr(configs, "consistency_start_ratio", 0.45))
        self.consistency_end_ratio = float(getattr(configs, "consistency_end_ratio", 1.0))
        self.consistency_loss_type = str(getattr(configs, "consistency_loss_type", "l1")).lower()

        self.sensitive_mask_enable = bool(getattr(configs, "sensitive_mask_enable", False))
        self.sensitive_mask_percentile = float(getattr(configs, "sensitive_mask_percentile", 0.85))
        self.sensitive_mask_detach = bool(getattr(configs, "sensitive_mask_detach", True))

        self.astc_noise_model = ASTCResidualNoiseModel(
            block_size=int(getattr(configs, "astc_block_size", 6)),
            noise_std=float(getattr(configs, "astc_noise_std", 0.02)),
            channel_corr=float(getattr(configs, "astc_noise_channel_corr", 0.5)),
            noise_prob=float(getattr(configs, "astc_noise_prob", 0.0)),
            warmup_ratio=float(getattr(configs, "astc_curriculum_warmup_ratio", 0.2)),
            peak_ratio=float(getattr(configs, "astc_curriculum_peak_ratio", 0.7)),
            max_iter=int(self.max_iter),
        ).to(self.device)

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

    def train(self) -> None:

        for curr_iter in range(self.trained_iter, self.max_iter):

            self.model.optimizer.zero_grad()

            # update model's current iteration for noise annealing
            self.model.current_iter = curr_iter
            if self.quantize and curr_iter % self.eval_interval == 0:
                self.writer.add_scalar("QAT/noise_mult", self.model._qat_noise_multiplier(), curr_iter)

            # generate random sample indices
            ys = torch.randint(0, self.texture_height, [self.batch_size, 1]).to(self.device)
            xs = torch.randint(0, self.texture_width, [self.batch_size, 1]).to(self.device)
            # sample LODs with a coarse-to-fine prior
            # lods = torch.randint(0, self.num_lods, [self.batch_size, 1]).to(self.device)
            lods = self.sample_probabilities.multinomial(num_samples=self.batch_size, replacement=True)
            #lods = torch.zeros(lods.shape).to(self.device).to(torch.int32)
            lods = lods.unsqueeze(1)
            # lods = torch.randint(0, 1, size=(self.batch_size, 1)).to(self.device)
            batch_index = torch.cat([ys, xs, lods], dim=1)

            # Get data; expand to canonical 11 channels, missing texture positions filled with 0
            gt_texture = self.dataset(batch_index)  # [batch_size, num_channels]
            gt_texture = self.dataset.expand_to_canonical(gt_texture).to(torch.float16)

            # xys -> uvs
            # shift the sample position from [0, 1, ..., 1023] -> [0.5, 1.5, ..., 1023.5]
            # uvs = ((xys + 0.5) / lod_scale) / (texture_weight / lod_scale)
            us = (xs + 0.5) / self.texture_width
            vs = (ys + 0.5) / self.texture_height
            lods = lods.float() / (self.num_lods - 1) if self.num_lods > 0 else torch.zeros_like(lods, dtype=torch.float32)
            batch_input = torch.cat([us, vs, lods], dim=1)
            # predict (clean branch)
            predict_texture = self.model(batch_input)  # [batch_size, num_channels]

            # base reconstruction loss; normal uses angular loss, others use channel MSE
            loss_weights = torch.tensor(self.output_loss_weights, device=self.device, dtype=torch.float32)
            base_loss = self._compute_reconstruction_loss(gt_texture, predict_texture, loss_weights, curr_iter)

            # ASTC-aware robust branch (noise-injected output + consistency)
            robust_loss = torch.tensor(0.0, device=self.device)
            consistency_loss = torch.tensor(0.0, device=self.device)
            total_loss = base_loss

            if self.astc_aware_train:
                noisy_input, applied = self.astc_noise_model.perturb_uvlod_input(batch_input, curr_iter, enable=self.astc_curriculum_enable)
                if applied:
                    predict_noisy = self.model(noisy_input)
                    robust_loss = self._compute_reconstruction_loss(gt_texture, predict_noisy, loss_weights, None)
                    total_loss = 0.5 * base_loss + 0.5 * robust_loss

                    prog = float(curr_iter) / max(1.0, float(self.max_iter - 1))
                    in_consistency = self.consistency_start_ratio <= prog <= self.consistency_end_ratio
                    if self.consistency_loss_enable and in_consistency and self.consistency_lambda > 0.0:
                        diff = predict_texture - predict_noisy
                        if self.consistency_loss_type == "mse":
                            c_map = diff.pow(2)
                        else:
                            c_map = diff.abs()

                        if self.sensitive_mask_enable:
                            sens_mask = self.astc_noise_model.build_sensitive_mask(
                                gt_texture,
                                percentile=self.sensitive_mask_percentile,
                                detach_mask=self.sensitive_mask_detach,
                            )
                            c_map = c_map * sens_mask

                        consistency_loss = c_map.mean() * self.consistency_lambda
                        total_loss = total_loss + consistency_loss

            self.writer.add_scalar('Loss/train', total_loss.item(), curr_iter)
            self.writer.add_scalar('Loss/base', base_loss.item(), curr_iter)
            if self.astc_aware_train:
                self.writer.add_scalar('Loss/robust', robust_loss.item(), curr_iter)
                self.writer.add_scalar('Loss/consistency', consistency_loss.item(), curr_iter)

            # optimize
            total_loss.backward()
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
    def _lod_plane_tile(
        h0: int, h1: int, w0: int, w1: int, plane_h: int, plane_w: int, lod_f: float, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[N,3] batch and (row,col) indices; UV=(col+0.5)/W, (row+0.5)/H."""
        rr, cc = torch.meshgrid(
            torch.arange(h0, h1, device=device),
            torch.arange(w0, w1, device=device),
            indexing="ij",
        )
        z = torch.full_like(rr, lod_f)
        inp = torch.stack(((cc + 0.5) / plane_w, (rr + 0.5) / plane_h, z), dim=-1).reshape(-1, 3)
        return inp, rr.reshape(-1).long(), cc.reshape(-1).long()

    @torch.no_grad()
    def _fill_predicted_lod(self, lod: int, lod_height: int, lod_width: int) -> torch.Tensor:
        """Tiled full-plane forward (caps tcnn batch size; scatter by row/col to avoid layout bugs)."""
        device = self.device
        c = self.model.num_channels
        out = torch.empty(lod_height, lod_width, c, device=device, dtype=torch.float32)
        step = max(1, self.eval_inference_tile or max(lod_height, lod_width))
        lod_f = 0.0 if self.num_lods <= 1 else float(lod) / float(self.num_lods - 1)
        for h0 in range(0, lod_height, step):
            h1 = min(h0 + step, lod_height)
            for w0 in range(0, lod_width, step):
                w1 = min(w0 + step, lod_width)
                inp, fr, fc = self._lod_plane_tile(h0, h1, w0, w1, lod_height, lod_width, lod_f, device)
                out[fr, fc, :] = self.model(inp).float()
        if getattr(self.model, "direct_diffuse_enabled", False) and "diffuse" in self.dataset.available_textures:
            s, e = self.dataset.canonical_channel_slices["diffuse"]
            direct = self.model.render_direct_diffuse_lod(lod, lod_height, lod_width)
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

    def _lod_size(self, lod: int) -> Tuple[int, int]:
        return self.texture_height // (2 ** lod), self.texture_width // (2 ** lod)

    def _ensure_visual_dirs(self, root: str) -> None:
        for vc in self.vis_configs:
            os.makedirs(os.path.join(root, vc['display_name']), exist_ok=True)

    def _canonical_gt_lod(self, lod: int, lod_height: int, lod_width: int) -> torch.Tensor:
        gt_slice = self.dataset.lod_cache[lod, :lod_height, :lod_width, :]
        gt_canonical = self.dataset.expand_to_canonical(
            gt_slice.reshape(-1, gt_slice.shape[-1])
        ).reshape(lod_height, lod_width, -1)
        return gt_canonical.permute(2, 0, 1)[None, ...]

    @torch.no_grad()
    def _render_lod_pair(self, lod: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        lod_height, lod_width = self._lod_size(lod)
        pred_hwc = self._fill_predicted_lod(lod, lod_height, lod_width).clamp(0, 1)
        pred = pred_hwc.permute(2, 0, 1)[None, ...]
        gt = self._canonical_gt_lod(lod, lod_height, lod_width)
        return pred, gt, lod_height, lod_width

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

    def _compute_lod_metrics(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        lod_height: int,
        lod_width: int,
    ) -> Tuple[float, float, Optional[float]]:
        pred_ref, gt_ref = self._metric_rgb_pair(pred, gt)
        if self._metric_texture_name() == "normal":
            psnr_value = float(normal_angular_psnr(pred_ref, gt_ref, self.normal_encoding).item())
        else:
            psnr_value = float(self.psnr(pred_ref, gt_ref).item())
        visual_pred, visual_gt = self._metric_visual_pair(pred_ref, gt_ref)
        ssim_value, _ = self.ssim(visual_pred, visual_gt)
        ssim_value = float(ssim_value.item())
        if lod_height >= 128 and lod_width >= 128:
            return psnr_value, ssim_value, float(self.lpips(visual_pred, visual_gt).item())
        return psnr_value, ssim_value, None

    def _save_lod_visuals(self, pred: torch.Tensor, gt: torch.Tensor, output_root: str, filename: str) -> None:
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

            # Training evaluation intentionally checks LOD0 only to keep the loop responsive.
            for lod in [0]:
                pred, gt, lod_height, lod_width = self._render_lod_pair(lod)
                psnr_value, ssim_value, lpips_value = self._compute_lod_metrics(pred, gt, lod_height, lod_width)

                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
                metric_prefix = "NormalAngular" if self._metric_texture_name() == "normal" else ""
                self.writer.add_scalar(f'{metric_prefix}PSNR_LOD{lod}/train', psnr_value, curr_iter)
                self.writer.add_scalar(f'{metric_prefix}SSIM_LOD{lod}/train', ssim_value, curr_iter)
                if lpips_value is not None:
                    lpips_list.append(lpips_value)
                    self.writer.add_scalar(f'{metric_prefix}LPIPS_LOD{lod}/train', lpips_value, curr_iter)

                if should_save_visuals:
                    self._save_lod_visuals(pred, gt, self.media_path, f"{curr_iter}_{lod}.png")

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

    def _infer_lods(self) -> range:
        # Skip the smallest four mip levels during inference, but always keep
        # LOD0 so tiny datasets still produce an output and metrics file.
        return range(max(1, self.num_lods - 4))

    @torch.no_grad()
    def infer(self) -> None:
        self._ensure_visual_dirs(self.infer_path)

        psnr_list: List[float] = []
        ssim_list: List[float] = []
        lpips_list: List[float] = []
        if self._metric_texture_name() == "normal":
            metric_lines = ["LOD NormalAngularPSNR NormalAngularSSIM NormalAngularLPIPS\n"]
        else:
            metric_lines = ["LOD PSNR SSIM LPIPS\n"]

        was_training = self.model.training
        try:
            self.model.eval()
            for lod in self._infer_lods():
                pred, gt, lod_height, lod_width = self._render_lod_pair(lod)
                psnr_value, ssim_value, lpips_value = self._compute_lod_metrics(pred, gt, lod_height, lod_width)

                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
                if lpips_value is not None:
                    lpips_list.append(lpips_value)
                lpips_for_line = 0.0 if lpips_value is None else lpips_value

                self._save_lod_visuals(pred, gt, self.infer_path, f"LOD_{lod}.png")
                metric_lines.append(f"LOD_{lod} {psnr_value:.4f} {ssim_value:.4f} {lpips_for_line:.4f}\n")
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
            num_lods=self.num_lods,
            device=self.device,
            curr_iter=curr_iter,
            ref_astc_resolution=self.ref_astc_resolution,
        )

    @torch.no_grad()
    def generate_probabilities(self) -> torch.Tensor:

        probabilities = []

        # Generate a coarse-to-fine LOD sampling distribution.
        current_prob = 1.0
        for i in range(0, self.num_lods):
            prob = current_prob / 4.0  # original is current_prob / 2**2
            probabilities.append(max(prob, 0.05))  # keep every LOD sampleable
            current_prob = prob
        probabilities = torch.tensor(probabilities).to(self.device)
        probabilities /= probabilities.sum()
        # print(probabilities)

        return probabilities


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
