import os
import math
import torch
import copy
import numpy as np
import tinycudann as tcnn
import torch.nn.functional as F
from torchtyping import TensorType

from configs import Config

from torch.optim.lr_scheduler import ReduceLROnPlateau


from utils import write_dds_r8g8b8a8
from feature_grid import FeatureGridSpec, feature_tensor_to_int, feature_to_unorm, quantize_feature_tensor, unorm_to_feature
from normal_encoding import normal_encoding_channels

# Fixed channel order aligned with: diffuse, normal, roughness, occlusion, metallic, specular, displacement
# Fixed texture order; normal can be 3 channels (xyz) or 2 channels (xy/hemi_oct).
CANONICAL_CHANNEL_ORDER = ["diffuse", "normal", "roughness", "occlusion", "metallic", "specular", "displacement"]


def canonical_channel_counts(normal_encoding: str = "xyz"):
    return [3, normal_encoding_channels(normal_encoding), 1, 1, 1, 1, 1]


def canonical_num_channels(normal_encoding: str = "xyz") -> int:
    return sum(canonical_channel_counts(normal_encoding))


class TCNNModel(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.device = config.device
        self.quantize = config.quantize
        self.quantize_bits = config.quantize_bits
        self.save_bits = config.save_bits

        self.N_k = 2 ** self.quantize_bits
        self.Q_k = 1 / self.N_k
        self.min_quantize_range = - (self.N_k - 1) / 2 * self.Q_k
        # self.N_k / 2 * self.Q_k = 1 / 2
        self.max_quantize_range = 1 / 2
        self.noise_range = 1 / 2 * self.Q_k

        self.max_iter = max(1, int(config.max_iter))
        self._qat_noise_schedule = str(getattr(config, "qat_noise_schedule", "none")).strip().lower()
        self._qat_noise_mult_start = float(getattr(config, "qat_noise_mult_start", 1.0))
        self._qat_noise_mult_end = float(getattr(config, "qat_noise_mult_end", 0.25))
        self._qat_noise_warmup_frac = float(getattr(config, "qat_noise_warmup_frac", 0.0))

        self.current_iter = 0

        self.num_lods = config.num_lods
        self.n_frequencies = config.n_frequencies
        self.n_neurons = config.n_neurons
        self.n_hidden_layers = config.n_hidden_layers
        self.output_activation = getattr(config, "output_activation", "hard_swish")
        # Network output matches the selected canonical texture layout.
        self.normal_encoding = str(getattr(config, "normal_encoding", "xyz")).lower()
        self.num_channels = canonical_num_channels(self.normal_encoding)
        self.direct_diffuse_infer_mode = str(getattr(config, "direct_diffuse_infer_mode", "disable")).lower()
        if self.direct_diffuse_infer_mode not in {"disable", "rgb", "ycocg"}:
            raise ValueError("direct_diffuse_infer_mode must be one of: disable, rgb, ycocg")
        self.direct_diffuse_enabled = self.direct_diffuse_infer_mode != "disable"
        if self.direct_diffuse_enabled:
            self.register_buffer("_direct_diffuse_values", torch.empty(0), persistent=False)
            self.register_buffer("_direct_diffuse_mask", torch.empty(0, dtype=torch.bool), persistent=False)

        # Make feature grids tile seamlessly under wrap/repeat sampling.
        # Can be disabled or softened from config if needed.
        self.wrap_boundary_constraint = bool(getattr(config, 'wrap_boundary_constraint', True))
        self.wrap_boundary_strength = float(getattr(config, 'wrap_boundary_strength', 1.0))
        # Performance knobs for boundary constraint.
        # Apply every N iterations (1 = every iter).
        self.wrap_boundary_interval = max(1, int(getattr(config, 'wrap_boundary_interval', 16)))
        # Constrain only the highest K levels (1 = highest level only).
        self.wrap_boundary_levels = max(1, int(getattr(config, 'wrap_boundary_levels', 1)))

        self.feature_grid_learning_rates = []
        for grid_idx, grid_cfg in enumerate(config.feature_grid_configs):
            self.feature_grid_learning_rates.append(grid_cfg.get('learning_rate', config.learning_rate))

        self.init_model(config)
        if config.load_dir is not None:
            self.load_ckpt(config)

    def init_model(self, config: Config) -> None:

        triangle_wave_config = {
            "n_dims_to_encode": 2,
            "otype": "TriangleWave",
            "n_frequencies": config.n_frequencies
        }
        self.triangle_wave = tcnn.Encoding(
            n_input_dims=2,
            encoding_config=triangle_wave_config
        )

        # how many lods to sample per query
        self.num_sampled_lods = 1

        # Support heterogeneous feature grid configs from configs.py.
        self.feature_grids = torch.nn.ModuleList()
        self.feature_grid_base_res = []       # per-grid base_resolution
        self.feature_grid_n_levels = []       # per-grid n_levels
        self.feature_grid_n_features_per_level = []  # per-grid n_features_per_level
        self.feature_grid_quantize_bits = []  # per-grid quantize bits (fallback to global)
        self.feature_grid_save_bits = []      # per-grid save bits (fallback to global)
        self.feature_grid_specs = []

        for grid_idx, grid_cfg in enumerate(config.feature_grid_configs):
            spec = FeatureGridSpec.from_config(grid_cfg, self.quantize_bits, self.save_bits, config.learning_rate)
            if self.direct_diffuse_enabled and grid_idx == 0:
                spec = FeatureGridSpec(**{**spec.__dict__, "interpolation": "Nearest"})
            self.feature_grid_specs.append(spec)
            max_res = spec.max_resolution
            n_levels = spec.n_levels
            qbits = spec.quantize_bits
            sbits = spec.save_bits
            n_feature_per_level = spec.n_features_per_level
            base_res = spec.base_resolution

            feature_grid_config = {
                "otype": "Grid",
                "type": "Dense",
                "n_levels": n_levels,
                "n_features_per_level": n_feature_per_level,
                "base_resolution": base_res,
                "per_level_scale": 2.0,
                "interpolation": spec.interpolation,
            }
            feature_grid = tcnn.Encoding(
                n_input_dims=2,
                encoding_config=feature_grid_config
            )
            self.feature_grids.append(feature_grid)
            self.feature_grid_base_res.append(int(base_res))
            self.feature_grid_n_levels.append(int(n_levels))
            self.feature_grid_n_features_per_level.append(int(n_feature_per_level))
            self.feature_grid_quantize_bits.append(int(qbits))
            self.feature_grid_save_bits.append(int(sbits))

            print(f"Initialized feature grid: max_res={max_res}, base_res={base_res}, n_levels={n_levels}, n_features_per_level={n_feature_per_level}, quantize_bits={qbits}, save_bits={sbits}, interpolation={spec.interpolation}")

        # Hidden layers still rely on tiny-cuda-nn's native activation.
        # Output activation is applied manually in forward() and can be configured as:
        # hard_swish (default), hard_gelu, or leaky_relu.
        hidden_activation = "LeakyReLU" if config.n_hidden_layers > 0 else "None"
        network_config = {
            #"otype": "FullyFusedMLP",
            "otype": "CutlassMLP",
            "activation": hidden_activation,
            "output_activation": "None",
            "n_neurons": config.n_neurons,
            "n_hidden_layers": config.n_hidden_layers
        }
        total_grid_features = sum(self.feature_grid_n_features_per_level) * self.num_sampled_lods
        if self.direct_diffuse_enabled:
            if not self.feature_grid_n_features_per_level or self.feature_grid_n_features_per_level[0] < 3:
                raise ValueError("direct_diffuse_infer_mode requires feature grid 0 to have at least 3 channels")
            total_grid_features -= 3 * self.num_sampled_lods
        n_input_dims = config.n_frequencies * 2 + total_grid_features + 1
        print(f"total_grid_features={total_grid_features}, n_input_dims={n_input_dims}")
        self.network = tcnn.Network(
            n_input_dims=n_input_dims,
            n_output_dims=self.num_channels,
            network_config=network_config,
        )

        # add optimizer params config
        optimizer_params = [{'params': self.network.parameters(), 'lr': getattr(config, 'network_learning_rate', 0.002)}]
        for idx, feature_grid in enumerate(self.feature_grids):
            lr = self.feature_grid_learning_rates[idx]
            optimizer_params.append({'params': feature_grid.parameters(), 'lr': lr})
        self.optimizer = torch.optim.Adam(optimizer_params)
        # the CosineAnnealingLR is too slow
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_iter, eta_min=0.0)
        # gamma: PSNR, 0.85: 28.1, 0.9: 28.25, 0.95: 28.4, 0.99: too slow
        # self.scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.90)
        # seem to be the best: factor=0.9, patience=2000, PSNR=28.63, time=10min
        # factor=0.95, patience=1000, PSNR=28.55, time=10min
        # factor=0.95, patience=2000, PSNR=28.7, time=20min
        # Squareplus factor=0.95, patience=2000, PSNR=28.57, LPIPS=0., time=20min
        # Squareplus factor=0.85, patience=2000, PSNR=28.60, LPIPS=0.2002, time=20min
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.85, patience=2000)

    def _hard_swish(self, x: torch.Tensor) -> torch.Tensor:
        """Hard-Swish activation without trig/exp.

        Formula: y = x * clamp(x + 3, 0, 6) / 6
        """
        return x * torch.clamp(x + 3.0, min=0.0, max=6.0) * (1.0 / 6.0)

    def _hard_gelu(self, x: torch.Tensor) -> torch.Tensor:
        """Hard-GELU approximation without trig/exp.

        Uses a clamped linear gate in place of GELU's smooth CDF.
        """
        return x * torch.clamp((x + 1.5) * (1.0 / 3.0), min=0.0, max=1.0)

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "hard_swish":
            return self._hard_swish(x)
        if self.output_activation == "hard_gelu":
            return self._hard_gelu(x)
        if self.output_activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=0.01)
        raise ValueError(
            f"Unsupported output activation '{self.output_activation}'. "
            "Expected one of: hard_swish, hard_gelu, leaky_relu."
        )

    def _qat_noise_multiplier(self) -> float:
        """Scales uniform additive noise in fake-quant forward; eval / non-quant unchanged."""
        if self._qat_noise_schedule == "none":
            return 1.0
        T = max(1, self.max_iter - 1)
        t = min(max(int(self.current_iter), 0), T)
        w = int(self._qat_noise_warmup_frac * T)
        if t <= w:
            return self._qat_noise_mult_start
        p = (t - w) / max(1, T - w)
        c = 0.5 * (1.0 + math.cos(math.pi * p))
        return self._qat_noise_mult_end + (self._qat_noise_mult_start - self._qat_noise_mult_end) * c

    def load_ckpt(self, config: Config) -> None:

        # TODO
        ckpt_iter = config.load_iter
        ckpt_dir = os.path.join(config.load_dir, "models")
        if ckpt_iter == -1:
            files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
            if not files:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

            iters = [int(os.path.splitext(f)[0]) for f in files if os.path.splitext(f)[0].isdigit()]
            if not iters:
                raise ValueError(f"No valid iter checkpoints found in {ckpt_dir}")

            ckpt_iter = max(iters)

        ckpt_path = os.path.join(ckpt_dir, f"{ckpt_iter}.pth")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.load_state_dict(ckpt)

    @staticmethod
    def _rgb_to_ycocg(rgb: torch.Tensor) -> torch.Tensor:
        r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
        return torch.cat([0.25 * r + 0.5 * g + 0.25 * b,
                          0.5 * r - 0.5 * b + 0.5,
                          -0.25 * r + 0.5 * g - 0.25 * b + 0.5], dim=-1).clamp(0, 1)

    @staticmethod
    def _ycocg_to_rgb(v: torch.Tensor) -> torch.Tensor:
        y, co, cg = v[..., 0:1], v[..., 1:2] - 0.5, v[..., 2:3] - 0.5
        return torch.cat([y + co - cg, y + cg, y - co - cg], dim=-1).clamp(0, 1)

    def _feature_to_unorm(self, x: torch.Tensor, grid_idx: int) -> torch.Tensor:
        return feature_to_unorm(x, self.feature_grid_specs[grid_idx])

    def _unorm_to_feature(self, x: torch.Tensor, grid_idx: int) -> torch.Tensor:
        return unorm_to_feature(x, self.feature_grid_specs[grid_idx])

    @torch.no_grad()
    def configure_direct_diffuse_from_dataset(self, dataset) -> None:
        if not self.direct_diffuse_enabled:
            return
        if "diffuse" not in getattr(dataset, "available_textures", []):
            raise ValueError("direct_diffuse_infer_mode requires diffuse texture")

        s, e = dataset.channel_slices["diffuse"]
        diffuse = dataset.textures[:, :, s:e].to(self.device).float().permute(2, 0, 1)[None]
        params = self._get_grid_params_tensor(self.feature_grids[0])
        base_res, n_levels = self.feature_grid_base_res[0], self.feature_grid_n_levels[0]
        n_fpl = self.feature_grid_n_features_per_level[0]
        values, mask, offset = torch.zeros_like(params), torch.zeros_like(params, dtype=torch.bool), 0

        for level in range(n_levels):
            res = base_res * (2 ** level)
            count = res * res * n_fpl
            rgb = F.interpolate(diffuse, size=(res, res), mode="bilinear", align_corners=False)
            payload = rgb.squeeze(0).permute(1, 2, 0)
            if self.direct_diffuse_infer_mode == "ycocg":
                payload = self._rgb_to_ycocg(payload)
            encoded = self._unorm_to_feature(payload, 0)
            params.data[offset:offset + count].view(res, res, n_fpl)[:, :, :3] = encoded
            values[offset:offset + count].view(res, res, n_fpl)[:, :, :3] = encoded
            mask[offset:offset + count].view(res, res, n_fpl)[:, :, :3] = True
            offset += count

        self._direct_diffuse_values = values
        self._direct_diffuse_mask = mask
        params.register_hook(lambda grad: torch.where(self._direct_diffuse_mask.to(grad.device), torch.zeros_like(grad), grad))
        print(f"[DirectDiffuse] mode={self.direct_diffuse_infer_mode}")

    @torch.no_grad()
    def _restore_direct_diffuse(self) -> None:
        if self.direct_diffuse_enabled and self._direct_diffuse_mask.numel():
            p = self._get_grid_params_tensor(self.feature_grids[0])
            m = self._direct_diffuse_mask.to(p.device)
            p.data[m] = self._direct_diffuse_values.to(p.device)[m]

    def _decode_direct_diffuse_features(self, features: torch.Tensor) -> torch.Tensor:
        payload = self._feature_to_unorm(features[:, :3], 0)
        return self._ycocg_to_rgb(payload) if self.direct_diffuse_infer_mode == "ycocg" else payload

    @torch.no_grad()
    def render_direct_diffuse_lod(self, lod: int, out_h: int, out_w: int, resize_fn=None) -> torch.Tensor:
        """Render direct diffuse as low-res feature payload, then bilinear upsample to output size."""
        if not self.direct_diffuse_enabled:
            raise RuntimeError("render_direct_diffuse_lod requires direct_diffuse_infer_mode != disable")

        grid_levels = self.feature_grid_n_levels[0]
        grid_fpl = self.feature_grid_n_features_per_level[0]
        base_res = self.feature_grid_base_res[0]
        lod_f = 0.0 if self.num_lods <= 1 else float(lod) / float(self.num_lods - 1)
        selected_level = int(round(grid_levels - self.num_sampled_lods - min(
            lod_f * (self.num_lods - self.num_sampled_lods),
            grid_levels - self.num_sampled_lods,
        )))
        selected_level = max(0, min(grid_levels - 1, selected_level))
        res = base_res * (2 ** selected_level)

        offset = 0
        for level in range(selected_level):
            level_res = base_res * (2 ** level)
            offset += level_res * level_res * grid_fpl
        params = self._get_grid_params_tensor(self.feature_grids[0])
        sampled = params[offset: offset + res * res * grid_fpl].view(-1, grid_fpl)
        tex = self._decode_direct_diffuse_features(sampled).reshape(res, res, 3)
        tex = tex.permute(2, 0, 1)[None].contiguous()
        if (res, res) != (int(out_h), int(out_w)):
            resize_fn = resize_fn or (lambda image, size: F.interpolate(image, size=size, mode="bilinear", align_corners=False))
            tex = resize_fn(tex, size=(int(out_h), int(out_w)))
        return tex.clamp(0, 1)

    def _texel_aligned_grid_uv(self, uvs: torch.Tensor, selected_level, base_res: int) -> torch.Tensor:
        """Map texel-center UVs to tiny-cuda-nn dense-grid point coordinates."""
        if not torch.is_tensor(selected_level):
            selected_level = torch.full((uvs.shape[0], 1), float(selected_level), device=uvs.device, dtype=uvs.dtype)
        level_int = torch.round(selected_level).to(torch.int64)
        res = int(base_res) * torch.pow(
            torch.full_like(selected_level, 2.0),
            level_int.to(dtype=selected_level.dtype),
        )
        if bool(torch.all(res <= 1)):
            return torch.zeros_like(uvs)
        return torch.clamp((uvs * res - 0.5) / torch.clamp(res - 1.0, min=1.0), 0.0, 1.0)

    def forward(self, x: TensorType["batch_size", 3]) -> TensorType["batch_size", "num_channels"]:

        # Explicit wrap to [0,1), so train/eval behavior matches repeat sampling at inference.
        uvs = torch.remainder(x[:, 0:2], 1.0)
        lod_encodings = x[:, [2]]

        # Select the feature-grid level corresponding to the normalized LOD.
        num_sampled_lods = self.num_sampled_lods
        mips = lod_encodings * (self.num_lods - num_sampled_lods)

        positional_encodings = self.triangle_wave(uvs)
        # positional_encodings = self.triangle_wave(xys)

        # get learned feature grid values
        features = []
        direct_diffuse = None
        for idx, feature_grid in enumerate(self.feature_grids):
            grid_levels = self.feature_grid_n_levels[idx]
            grid_fpl = self.feature_grid_n_features_per_level[idx]
            qbits = self.feature_grid_quantize_bits[idx]

            clipped_mips = torch.clamp(mips, max=grid_levels - num_sampled_lods)
            selected_level_f = grid_levels - num_sampled_lods - clipped_mips
            grid_uvs = self._texel_aligned_grid_uv(uvs, selected_level_f, self.feature_grid_base_res[idx])
            all_features = feature_grid(grid_uvs)  # [B, grid_levels * grid_fpl]
            cols = (grid_levels - num_sampled_lods - clipped_mips) * grid_fpl + torch.arange(grid_fpl * num_sampled_lods).to(self.device)
            sampled_features = torch.gather(all_features, 1, cols.to(torch.int64))

            if self.quantize and self.training:
                # Per-grid quantization step size
                N_k = 2 ** qbits
                Q_k = 1.0 / N_k

                # STE (Straight-Through Estimator) quantization simulation:
                # Quantize in forward pass, but let gradients pass through unchanged.
                noise_range = 0.5 * Q_k
                mult = self._qat_noise_multiplier()
                noise = (torch.rand_like(sampled_features) * 2 - 1) * noise_range * mult
                sampled_features_noisy = sampled_features + noise
                # Quantize (round to grid) then use STE
                quantized = torch.round(sampled_features_noisy / Q_k) * Q_k
                # Straight-through: forward uses quantized, backward uses sampled_features
                sampled_features = sampled_features + (quantized - sampled_features).detach()

            if self.direct_diffuse_enabled and idx == 0:
                direct_diffuse = self._decode_direct_diffuse_features(sampled_features)
                sampled_features = sampled_features[:, 3:]

            features.append(sampled_features)
        features = torch.cat(features, dim=1)

        inputs = torch.cat([positional_encodings, features, lod_encodings], dim=1)

        outputs = self.network(inputs)
        outputs = self._apply_output_activation(outputs)
        if direct_diffuse is not None:
            outputs = outputs.clone()
            outputs[:, :3] = direct_diffuse

        return outputs

    def simulate_quantize(self):
        # Quantize only learned feature-grid parameters; the MLP remains full precision.
        for idx, feature_grid in enumerate(self.feature_grids):
            state_dict = feature_grid.state_dict()

            state_dict['params'] = quantize_feature_tensor(state_dict['params'], self.feature_grid_specs[idx])
            feature_grid.load_state_dict(state_dict)

    def clamp_value(self):

        apply_wrap = (
            self.wrap_boundary_constraint
            and self.wrap_boundary_strength > 0.0
            and (self.current_iter % self.wrap_boundary_interval == 0)
        )

        with torch.no_grad():
            for idx, feature_grid in enumerate(self.feature_grids):
                qbits = self.feature_grid_quantize_bits[idx]
                N_k = 2 ** qbits
                Q_k = 1.0 / N_k
                min_q = -(N_k - 1) / 2 * Q_k
                max_q = 0.5

                params = self._get_grid_params_tensor(feature_grid)
                params.clamp_(min=min_q, max=max_q)

                if apply_wrap:
                    self._enforce_wrap_boundary_constraint_inplace(
                        params,
                        base_res=int(self.feature_grid_base_res[idx]),
                        n_levels=int(self.feature_grid_n_levels[idx]),
                        n_features_per_level=int(self.feature_grid_n_features_per_level[idx]),
                        strength=self.wrap_boundary_strength,
                        highest_k_levels=self.wrap_boundary_levels,
                    )
        self._restore_direct_diffuse()

    def _get_grid_params_tensor(self, feature_grid: torch.nn.Module) -> torch.nn.Parameter:
        # tiny-cuda-nn grid usually exposes a single parameter named "params".
        for name, p in feature_grid.named_parameters():
            if name == 'params':
                return p
        return next(feature_grid.parameters())

    def _enforce_wrap_boundary_constraint_inplace(
        self,
        flat_params: torch.Tensor,
        base_res: int,
        n_levels: int,
        n_features_per_level: int,
        strength: float = 1.0,
        highest_k_levels: int = 1,
    ) -> None:
        """Tie opposite borders of each dense grid level for seamless repeat sampling.

        This makes left/right and top/bottom borders consistent, which removes
        visible seams when feature textures are sampled with wrap mode at runtime.
        """
        strength = float(max(0.0, min(1.0, strength)))
        if strength <= 0.0:
            return

        highest_k_levels = max(1, int(highest_k_levels))
        start_level = max(0, n_levels - highest_k_levels)

        offset = 0
        one_minus = 1.0 - strength

        for level in range(n_levels):
            res = base_res * (2 ** level)
            level_count = res * res * n_features_per_level
            if level < start_level:
                offset += level_count
                continue

            level_flat = flat_params[offset: offset + level_count]
            level_tex = level_flat.view(res, res, n_features_per_level)

            # Match left/right edges
            left = level_tex[:, 0, :].clone()
            right = level_tex[:, -1, :].clone()
            lr_avg = 0.5 * (left + right)
            level_tex[:, 0, :] = left * one_minus + lr_avg * strength
            level_tex[:, -1, :] = right * one_minus + lr_avg * strength

            # Match top/bottom edges
            top = level_tex[0, :, :].clone()
            bottom = level_tex[-1, :, :].clone()
            tb_avg = 0.5 * (top + bottom)
            level_tex[0, :, :] = top * one_minus + tb_avg * strength
            level_tex[-1, :, :] = bottom * one_minus + tb_avg * strength

            # Keep four corners fully consistent to avoid corner pinching.
            c00 = level_tex[0, 0, :].clone()
            c01 = level_tex[0, -1, :].clone()
            c10 = level_tex[-1, 0, :].clone()
            c11 = level_tex[-1, -1, :].clone()
            corner_avg = 0.25 * (c00 + c01 + c10 + c11)
            level_tex[0, 0, :] = c00 * one_minus + corner_avg * strength
            level_tex[0, -1, :] = c01 * one_minus + corner_avg * strength
            level_tex[-1, 0, :] = c10 * one_minus + corner_avg * strength
            level_tex[-1, -1, :] = c11 * one_minus + corner_avg * strength

            offset += level_count

    def get_model_info(self) -> np.array:

        info = []
        # include number of feature grids
        num_grids = len(self.feature_grids)
        info.append(int(num_grids))
        # for each grid: base_resolution, n_levels, n_features_per_level, quantize_bits, save_bits
        for i in range(num_grids):
            info.append(int(self.feature_grid_base_res[i]))
            info.append(int(self.feature_grid_n_levels[i]))
            info.append(int(self.feature_grid_n_features_per_level[i]))
            info.append(int(self.feature_grid_quantize_bits[i]))
            info.append(int(self.feature_grid_save_bits[i]))

        # keep network summary info
        info.append(int(self.n_frequencies))
        info.append(int(self.n_neurons))
        info.append(int(self.n_hidden_layers))
        info.append(int((self.num_channels // 16 + 1) * 16))

        info = np.array(info, dtype=np.int32)
        #print(info) # debug npz header

        return info

    @torch.no_grad()
    def save(self, curr_iter: int, model_path: str) -> None:

        save_model = copy.deepcopy(self).cpu()

        save_path = os.path.join(model_path, f"train_result_{curr_iter}")
        os.makedirs(save_path, exist_ok=True)

        model_path = os.path.join(save_path, f"model.pth")
        torch.save(self.state_dict(), model_path)

        if self.quantize:

            # in tcnn the input dim would be padded to the nearest multiple of 16
            # https://github.com/NVlabs/tiny-cuda-nn/issues/6

            save_model.simulate_quantize()

            # ----------------------------------------------------------------------------------
            # - process the range of network weights
            # ----------------------------------------------------------------------------------

            # generate model infos
            info = self.get_model_info()

            save_kwargs = {
                'info': info,
                'network': save_model.network.state_dict()["params"].numpy()
            }

            network_data_path = os.path.join(save_path, f"network_data.npz")
            # Save parameters to npz (features moved to DDS files)
            np.savez(network_data_path, **save_kwargs)

            # ----------------------------------------------------------------------------------
            # - process the range of feature textures
            # ----------------------------------------------------------------------------------

            # Collect per-grid quantized integer features (one entry per feature).
            quant_ints_list = []
            matching_indices = []
            for i, feature_grid in enumerate(save_model.feature_grids):
                qbits = int(save_model.feature_grid_quantize_bits[i])
                sbits = int(save_model.feature_grid_save_bits[i])
                features = feature_grid.state_dict()["params"]
                ints = feature_tensor_to_int(features, save_model.feature_grid_specs[i])
                quant_ints_list.append(ints)
                matching_indices.append(i)

            # Convert packed features to uint8 for R8G8B8A8 format
            # Packed features are in the range [0, 2^quantize_bits - 1]
            # We need to scale them to [0, 255] for uint8
            scale_factor = 255.0 / (2 ** save_model.quantize_bits - 1)

            # For each matching grid, extract highest-resolution level and form a single R8G8B8A8 DDS
            for idx, ints in enumerate(quant_ints_list):
                orig_i = matching_indices[idx]
                ints_np = ints.cpu().numpy()
                base_res = int(self.feature_grid_base_res[orig_i])
                n_levels = int(self.feature_grid_n_levels[orig_i])
                n_fpl = int(self.feature_grid_n_features_per_level[orig_i])

                # compute offset to the highest resolution level (in feature units)
                offset = 0
                for level in range(n_levels - 1):
                    feature_size = base_res * (2 ** level)
                    offset += n_fpl * feature_size ** 2

                max_res_size = base_res * (2 ** (n_levels - 1))
                max_res_features = n_fpl * max_res_size ** 2

                level_features = ints_np[offset: offset + max_res_features]
                # reshape into [H, W, n_fpl]
                try:
                    level_features = level_features.reshape(max_res_size, max_res_size, n_fpl)
                except Exception:
                    raise ValueError(f"Cannot reshape grid {orig_i} features of length {level_features.size} into ({max_res_size},{max_res_size},{n_fpl})")

                # Build RGBA channels from the first up to 4 feature channels
                channels = []
                for c in range(4):
                    if c < n_fpl:
                        ch = (level_features[:, :, c] * scale_factor).astype(np.uint8)
                    else:
                        ch = np.zeros((max_res_size, max_res_size), dtype=np.uint8)
                    channels.append(ch)

                rgba = np.stack(channels, axis=2)
                dds_path = os.path.join(save_path, f"feature_grid_{orig_i}.dds")
                write_dds_r8g8b8a8(dds_path, max_res_size, max_res_size, rgba)

