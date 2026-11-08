import os
import math
import torch
import copy
import numpy as np
import tinycudann as tcnn
import torch.nn.functional as F
from torchtyping import TensorType

from configs import Config
from Core.Positional_Encoding.positional_encoding import TiledPositionalEncoding
from Core.Feature_Grid.feature_gradients import sample_feature_inputs, sample_feature_centers, gradient_directions
from Core.Tile_CNN.material_cnn import MaterialCNN
from Core.Feature_Grid.sampling import resize_bilinear_repeat
from Common.Material.normal_encoding import NORMAL_XY_MIN_Z
from Common.Material.compression import compressed_normal_encoding

from Core.Feature_Grid.optimization import make_scheduler


from Core.Feature_Grid.feature_grid import (
    FeatureGridSpec, LearnedMipPyramid, enforce_wrap_boundary_constraint_inplace,
    feature_to_unorm, qat_noise_multiplier, quantize_feature_ste, quantize_feature_tensor,
    unorm_to_feature,
)
from dataset import get_canonical_channel_slices_static


def _feature_input_channels(config):
    return [grid['n_features_per_level'] * 3 // 4
            if recipe.format == 'bc1' and not getattr(config, 'bcn_alpha_input_enable', False)
            else grid['n_features_per_level']
            for grid, recipe in zip(config.feature_grid_configs, config.grid_compressions)]


class TCNNModel(torch.nn.Module):

    def __getstate__(self):
        state = super().__getstate__()
        state["_qat_param_cache"] = {}
        state["optimizer"] = None
        state["scheduler"] = None
        return state

    def __init__(self, config: Config, *, checkpoint=None):
        super().__init__()

        self.device = config.device
        self.quantize = config.quantize
        self.max_iter = max(1, int(config.max_iter))
        self._qat_noise_schedule = str(getattr(config, "qat_noise_schedule", "none")).strip().lower()
        self._qat_noise_mult_start = float(getattr(config, "qat_noise_mult_start", 1.0))
        self._qat_noise_mult_end = float(getattr(config, "qat_noise_mult_end", 0.25))
        self._qat_noise_warmup_frac = float(getattr(config, "qat_noise_warmup_frac", 0.0))
        self._qat_param_cache = {}
        self._qat_param_cache_iter = None
        self._pending_grid_sync = set()

        self.current_iter = 0

        self.num_mips = config.num_mips
        self.mip0_only = bool(getattr(config, 'mip0_only', False))
        self.inference_mode = getattr(config, 'inference_mode', 'on_load')
        self.material_lod_reference_size = tuple(getattr(config, 'material_lod_reference_size', ()))
        if not self.material_lod_reference_size:
            self.material_lod_reference_size = (
                tuple(getattr(config, 'material_reconstruction_size', ())) or
                (max(g['max_resolution'] for g in config.feature_grid_configs),) * 2
                if self.inference_mode == 'on_sample' else (2 ** (config.num_mips - 1),) * 2)
        self.n_frequencies = config.n_frequencies
        self.n_neurons = config.n_neurons
        self.n_hidden_layers = config.n_hidden_layers
        self.output_activation = getattr(config, "output_activation", "hard_swish")
        self.grid_compressions = config.grid_compressions
        self.feature_input_channels = _feature_input_channels(config)
        self.astcenc_path = config.astcenc_path
        self.astcenc_quality = config.astcenc_quality
        self.fntc_tile_size = config.fntc_tile_size
        self.astc_codec_backend = config.astc_codec_backend
        self.codec_branch_enabled = False
        self._decoder_adaptation_features = None
        self.grid_proxies = torch.nn.ModuleList()
        # Network output matches the selected canonical texture layout.
        self.normal_encoding = str(getattr(config, "normal_encoding", "xyz")).lower()
        self.channel_slices = get_canonical_channel_slices_static(self.normal_encoding, getattr(config, 'texture_types', None))
        self.num_channels = max(end for _, end in self.channel_slices.values())
        self.super_resolution_enable = bool(getattr(config, "super_resolution_enable", False))
        self.super_resolution_base_resolution = config.super_resolution_base_resolution
        self.material_reconstruction_size = tuple(getattr(config, 'material_reconstruction_size', ()))
        self.inference_size = tuple(getattr(config, 'material_cnn_inference_size', self.material_reconstruction_size))
        self.sr_base_codec = None
        if self.super_resolution_enable:
            formats = config.material_compression_formats
            self.sr_base_codec = dict(formats=formats.copy(), tile_size=config.fntc_tile_size,
                                     normal_encoding=compressed_normal_encoding(self.normal_encoding, formats['normal']))
            if any(name.startswith('ASTC') for name in formats.values()):
                self.sr_base_codec['quality'] = config.astcenc_quality
        self.direct_diffuse_infer_mode = str(
            getattr(config, "direct_diffuse_infer_mode", "disable")
        ).lower()
        if self.direct_diffuse_infer_mode not in {"disable", "rgb", "ycocg"}:
            raise ValueError("direct_diffuse_infer_mode must be one of: disable, rgb, ycocg")
        self.direct_diffuse_enabled = self.direct_diffuse_infer_mode != "disable"
        if self.direct_diffuse_enabled:
            self.register_buffer("_direct_diffuse_values", torch.empty(0), persistent=False)
            self.register_buffer(
                "_direct_diffuse_mask", torch.empty(0, dtype=torch.bool), persistent=False
            )

        # Make feature grids tile seamlessly under wrap/repeat sampling.
        # Can be disabled or softened from config if needed.
        self.wrap_boundary_constraint = bool(getattr(config, "wrap_boundary_constraint", True))
        self.wrap_boundary_strength = float(getattr(config, "wrap_boundary_strength", 1.0))
        # Performance knobs for boundary constraint.
        # Apply every N iterations (1 = every iter).
        self.wrap_boundary_interval = max(1, int(getattr(config, "wrap_boundary_interval", 16)))
        # Constrain only the highest K levels (1 = highest level only).
        self.wrap_boundary_levels = max(1, int(getattr(config, "wrap_boundary_levels", 1)))

        self.init_model(config)
        from Core.Feature_Grid.serialization import refresh_inference_contract
        refresh_inference_contract(self)
        if checkpoint is not None or config.load_dir is not None:
            self.load_ckpt(config, checkpoint)

    def init_model(self, config: Config) -> None:

        # Tiled positional encoding with configurable tile size (texels).
        # Reference edge: texture max edge when set by the trainer (train.py),
        # otherwise the feature-grid max resolution.
        reference_edge = int(getattr(config, "pos_encoding_reference_edge", 0)) or max(
            int(g.get("max_resolution", 1024)) for g in config.feature_grid_configs
        )
        tile_size = max(1, int(getattr(config, "pos_encoding_tile_size", 8)))
        tile_uv = float(tile_size) / float(max(1, reference_edge))
        self.pos_encoding_tile_uv = tile_uv
        self.pos_encoding_type = "tiled_sincos"
        self.pos_encoding_tile_size = tile_size
        self.pos_encoding_reference_size = tuple(getattr(config, "material_reconstruction_size",
                                                        (reference_edge, reference_edge)))
        self.positional_encoding = TiledPositionalEncoding(
            n_frequencies=config.n_frequencies, tile_uv=tile_uv, device=self.device)

        # how many mips to sample per query
        self.num_sampled_mips = 1

        # Support heterogeneous feature grid configs from configs.py.
        self.feature_grids = torch.nn.ModuleList()
        self.feature_grid_specs = []

        for idx, grid_cfg in enumerate(config.feature_grid_configs):
            spec = FeatureGridSpec.from_config(grid_cfg, config.quantize_bits, config.learning_rate)
            self.feature_grid_specs.append(spec)
            self.register_buffer(f'_center_shifts_{idx}', torch.tensor(
                ((0, 0), *gradient_directions(spec.feature_gradient_count)),
                device=self.device, dtype=torch.long), persistent=False)
            max_res = spec.max_resolution
            n_levels = spec.n_levels
            qbits = spec.quantize_bits
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
            feature_grid = LearnedMipPyramid(spec, self.device) if n_levels > 1 else tcnn.Encoding(
                n_input_dims=2,
                encoding_config=feature_grid_config
            )
            self.feature_grids.append(feature_grid)

            print(f"Initialized feature grid: max_res={max_res}, base_res={base_res}, n_levels={n_levels}, n_features_per_level={n_feature_per_level}, quantize_bits={qbits}, interpolation={spec.interpolation}")

        self.feature_grid_lod_offsets = tuple(
            math.log2(spec.max_resolution / max(self.material_lod_reference_size))
            if self.inference_mode == 'on_sample' else 0. for spec in self.feature_grid_specs)

        self.material_cnn = None
        if getattr(config, 'material_cnn_enable', False):
            if not self.mip0_only:
                raise ValueError('Material CNN requires Mip0')
            size = getattr(config, 'material_cnn_inference_size', None)
            self.material_cnn = MaterialCNN(self.num_channels, inference_size=size,
                                            tile_size=config.fntc_tile_size).to(self.device)
        if self.direct_diffuse_enabled:
            if self.feature_grid_specs[0].n_features_per_level < 3:
                raise ValueError("direct_diffuse_infer_mode requires feature grid 0 to have at least 3 channels")
        # tiny-cuda-nn CutlassMLP exposes one flat weight tensor and ignores
        # use_bias. Append a constant input explicitly to provide a real bias.
        self.network = self._create_decoder(config, self.positional_encoding.n_output_dims)
        print(f"n_input_dims={self.network.n_input_dims}")
        # CutlassMLP ignores `use_bias` in this tiny-cuda-nn build. The constant
        # input supplies first-layer bias; this parameter supplies output bias.
        self.decoder_output_bias = torch.nn.Parameter(
            torch.zeros(self.num_channels, device=self.device, dtype=torch.float32)
        )

        # add optimizer params config
        optimizer_params = [{'params': [*self.network.parameters(), self.decoder_output_bias],
                             'lr': getattr(config, 'network_learning_rate', 0.002)}]
        for feature_grid, spec in zip(self.feature_grids, self.feature_grid_specs):
            optimizer_params.append({'params': feature_grid.parameters(), 'lr': spec.learning_rate})
        self.optimizer = torch.optim.Adam(optimizer_params, betas=(config.momentum, .999),
                                          weight_decay=config.weight_decay)
        self.scheduler = make_scheduler(self.optimizer, config.lr_scheduler, self.max_iter,
                                        on_sample=self.inference_mode == 'on_sample')

    def _create_decoder(self, config, positional_dims):
        features = sum(channels * (1 + grid['feature_gradient_count'])
                       for channels, grid in zip(_feature_input_channels(config), config.feature_grid_configs))
        inputs = positional_dims + features * self.num_sampled_mips + 1
        if config.super_resolution_enable:
            inputs += self.num_channels
        return tcnn.Network(inputs, self.num_channels, {
            'otype': 'CutlassMLP', 'activation': 'LeakyReLU' if config.n_hidden_layers else 'None',
            'output_activation': 'None', 'n_neurons': config.n_neurons,
            'n_hidden_layers': config.n_hidden_layers,
        })

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
        return qat_noise_multiplier(
            self._qat_noise_schedule, self.current_iter, self.max_iter,
            self._qat_noise_warmup_frac, self._qat_noise_mult_start, self._qat_noise_mult_end,
        )

    def _get_qat_grid_params(self, grid_idx: int, feature_grid: torch.nn.Module) -> torch.Tensor:
        """Build one quantized grid per iteration/branch and reuse it across forwards."""
        if self._decoder_adaptation_features is not None:
            return self._decoder_adaptation_features[grid_idx]
        if self._qat_param_cache_iter != self.current_iter:
            self._qat_param_cache.clear()
            self._qat_param_cache_iter = self.current_iter

        if self.codec_branch_enabled and len(self.grid_proxies):
            from Core.Feature_Grid.compression import sample_proxy_grid
            return sample_proxy_grid(self, grid_idx, feature_grid)

        return self._source_qat_grid_params(grid_idx, feature_grid)

    def _source_qat_grid_params(self, grid_idx, feature_grid):
        if grid_idx in self._pending_grid_sync:
            self._sync_pending_grid_sources()
        params = feature_grid.params
        key = (int(grid_idx), 'source')
        cached = self._qat_param_cache.get(key)
        if cached is not None and cached[0] == params._version:
            return cached[1]

        step = 1.0 / self.feature_grid_specs[grid_idx].quant_step_count
        multiplier = self._qat_noise_multiplier()
        noise = (torch.rand_like(params)-.5)*step*multiplier if multiplier else torch.zeros_like(params)
        noisy_params = params + noise
        quantized = quantize_feature_ste(noisy_params, self.feature_grid_specs[grid_idx])
        self._qat_param_cache[key] = (params._version, quantized)
        return quantized

    def _sync_pending_grid_sources(self):
        if self._pending_grid_sync:
            from Core.Feature_Grid.compression import sync_grid_sources
            sync_grid_sources(self, self._pending_grid_sync)

    def clear_codec_cache(self):
        """Refresh codec candidates while retaining this batch's source QAT noise."""
        for key in tuple(self._qat_param_cache):
            if key[1] == 'codec':
                del self._qat_param_cache[key]

    def load_ckpt(self, config: Config, checkpoint=None) -> None:
        from checkpoint import checkpoint_paths
        ckpt_path = str(checkpoint if checkpoint is not None else checkpoint_paths(config.load_dir, config.load_iter)[0])
        ckpt_iter = int(os.path.basename(os.path.dirname(ckpt_path)).removeprefix('train_result_'))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.load_state_dict(ckpt)
        self.loaded_checkpoint = ckpt_path
        self.current_iter = 0 if config.warm_start else ckpt_iter

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
        params = self.feature_grids[0].params
        spec = self.feature_grid_specs[0]
        base_res, n_levels, n_fpl = spec.base_resolution, spec.n_levels, spec.n_features_per_level
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
            p = self.feature_grids[0].params
            m = self._direct_diffuse_mask.to(p.device)
            p.data[m] = self._direct_diffuse_values.to(p.device)[m]

    def _decode_direct_diffuse_features(self, features: torch.Tensor) -> torch.Tensor:
        payload = self._feature_to_unorm(features[:, :3], 0)
        return self._ycocg_to_rgb(payload) if self.direct_diffuse_infer_mode == "ycocg" else payload

    @torch.no_grad()
    def render_direct_diffuse_mip(self, mip: int, out_h: int, out_w: int, resize_fn=None) -> torch.Tensor:
        """Render direct diffuse from feature-grid RGB, then bilinear upsample to output size."""
        if not self.direct_diffuse_enabled:
            raise RuntimeError("render_direct_diffuse_mip requires direct_diffuse_infer_mode != disable")

        spec = self.feature_grid_specs[0]
        grid_levels, grid_fpl = spec.n_levels, spec.n_features_per_level
        mip_f = 0.0 if self.num_mips <= 1 else float(mip) / float(self.num_mips - 1)
        selected_level = int(round(grid_levels - self.num_sampled_mips - min(
            mip_f * (self.num_mips - self.num_sampled_mips),
            grid_levels - self.num_sampled_mips,
        )))
        selected_level = max(0, min(grid_levels - 1, selected_level))
        res = spec.level_resolution(selected_level)
        offset = spec.level_offset(selected_level)
        params = self.feature_grids[0].params
        sampled = params[offset: offset + res * res * grid_fpl].view(-1, grid_fpl)
        tex = self._decode_direct_diffuse_features(sampled).reshape(res, res, 3)
        tex = tex.permute(2, 0, 1)[None].contiguous()
        if (res, res) != (int(out_h), int(out_w)):
            resize_fn = resize_fn or resize_bilinear_repeat
            tex = resize_fn(tex, size=(int(out_h), int(out_w)))
        return tex.clamp(0, 1)

    def forward_native_centers(self, x, size):
        """Use direct feature gathers when the caller supplies native centers."""
        return self(x, native_size=size)

    def _decoder_features(self, features, index):
        if self.feature_input_channels[index] == self.feature_grid_specs[index].n_features_per_level:
            return features
        return features.reshape(len(features), -1, 4)[..., :3].flatten(1)

    def _prepare_decoder_inputs(self, x, *, native_size=None, grid_params=None):

        coordinate_dims = 2 + (not self.mip0_only)
        expected_dims = coordinate_dims + (self.num_channels if self.super_resolution_enable else 0)
        if x.shape[1] != expected_dims:
            raise ValueError(f"Expected model input with {expected_dims} columns, got {x.shape[1]}")
        if not self.training and self._decoder_adaptation_features is None:
            self._sync_pending_grid_sources()

        # Explicit wrap to [0,1), so train/eval behavior matches repeat sampling at inference.
        uvs = torch.remainder(x[:, 0:2], 1.0)
        mip_encodings = x.new_zeros((len(x), 1)) if self.mip0_only else x[:, [2]]

        # Select the feature-grid level corresponding to the normalized Mip.
        num_sampled_mips = self.num_sampled_mips
        mips = mip_encodings * (self.num_mips - num_sampled_mips)

        positional_encodings = self.positional_encoding(uvs)

        # get learned feature grid values
        features = []
        direct_diffuse = None
        for idx, feature_grid in enumerate(self.feature_grids):
            spec = self.feature_grid_specs[idx]
            active_params = None
            if grid_params is not None:
                active_params = grid_params[idx]
            elif self._decoder_adaptation_features is not None:
                active_params = self._decoder_adaptation_features[idx]
            elif self.quantize and self.training:
                active_params = self._get_qat_grid_params(idx, feature_grid)
            params = active_params if active_params is not None else feature_grid.params
            if (native_size == (spec.max_resolution, spec.max_resolution) and spec.n_levels == 1
                    and uvs.dtype == torch.float32 and not uvs.requires_grad):
                sampled_features = sample_feature_centers(
                    params, spec, uvs, shifts=getattr(self, f'_center_shifts_{idx}'))
            else:
                grid_levels = spec.n_levels
                clipped_mips = torch.clamp(mips + self.feature_grid_lod_offsets[idx],
                                          min=0, max=grid_levels - num_sampled_mips)
                selected_level_f = grid_levels - num_sampled_mips - clipped_mips
                sampled_features = sample_feature_inputs(params, spec, uvs, selected_level_f)
            sampled_features = sampled_features.to(feature_grid.dtype)

            if self.direct_diffuse_enabled and idx == 0:
                # Center channels come first, before the feature differences.
                direct_diffuse = self._decode_direct_diffuse_features(sampled_features)

            features.append(self._decoder_features(sampled_features, idx))
        features = torch.cat(features, dim=1)

        input_parts = [positional_encodings, features]
        if self.super_resolution_enable:
            input_parts.append(x[:, coordinate_dims:])
        input_parts.append(torch.ones((len(x), 1), device=x.device, dtype=features.dtype))
        inputs = torch.cat(input_parts, dim=1)

        return inputs, direct_diffuse

    def forward(self, x: torch.Tensor, *, native_size=None) -> TensorType["batch_size", "num_channels"]:
        inputs, direct_diffuse = self._prepare_decoder_inputs(x, native_size=native_size)
        outputs = self.network(inputs) + self.decoder_output_bias.to(dtype=self.network.params.dtype)
        if not self.super_resolution_enable:
            outputs = self._apply_output_activation(outputs)
        if direct_diffuse is not None:
            outputs = outputs.clone()
            outputs[:, :3] = direct_diffuse

        return outputs

    def simulate_quantize(self):
        # Quantize only learned feature-grid parameters; the MLP remains full precision.
        for idx, feature_grid in enumerate(self.feature_grids):
            state_dict = feature_grid.state_dict()

            state_dict["params"] = quantize_feature_tensor(
                state_dict["params"], self.feature_grid_specs[idx]
            )
            feature_grid.load_state_dict(state_dict)

    def clamp_value(self):
        from Core.Feature_Grid.compression import clamp_proxies

        clamp_proxies(self)

        if self._decoder_adaptation_features is not None:
            return

        apply_wrap = (
            self.wrap_boundary_constraint
            and self.wrap_boundary_strength > 0.0
            and (self.current_iter % self.wrap_boundary_interval == 0)
        )

        with torch.no_grad():
            for idx, feature_grid in enumerate(self.feature_grids):
                spec = self.feature_grid_specs[idx]

                params = feature_grid.params
                params.clamp_(min=spec.quant_min, max=spec.quant_max)

                if apply_wrap:
                    enforce_wrap_boundary_constraint_inplace(
                        params,
                        base_res=spec.base_resolution,
                        n_levels=spec.n_levels,
                        n_features_per_level=spec.n_features_per_level,
                        strength=self.wrap_boundary_strength,
                        highest_k_levels=self.wrap_boundary_levels,
                    )
        self._restore_direct_diffuse()


    def get_model_info(self) -> np.array:

        info = []
        # include number of feature grids
        num_grids = len(self.feature_grids)
        info.append(int(num_grids))
        # for each grid: base_resolution, n_levels, n_features_per_level, quantize_bits
        for spec in self.feature_grid_specs:
            info.extend((spec.base_resolution, spec.n_levels, spec.n_features_per_level,
                         spec.quantize_bits))

        # keep network summary info
        info.append(int(self.n_frequencies))
        info.append(int(self.n_neurons))
        info.append(int(self.n_hidden_layers))
        info.append(int((self.num_channels // 16 + 1) * 16))

        info = np.array(info, dtype=np.int32)
        # print(info) # debug npz header

        return info

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = copy.copy(state_dict)
        from Core.Feature_Grid.serialization import validate_contract, restore_proxies

        validate_contract(self, state_dict)
        # Normalize equivalent JSON metadata to the model's serialized layout.
        state_dict['_inference_contract'] = self._inference_contract.detach().clone()
        restore_proxies(self, state_dict)
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._pending_grid_sync.clear()
        return result

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self._sync_pending_grid_sources()
        super()._save_to_state_dict(destination, prefix, keep_vars)
        from Core.Feature_Grid.serialization import proxy_layout_tensor
        destination[prefix+'_grid_proxy_kinds'] = proxy_layout_tensor(self)

    @torch.no_grad()
    def network_data_arrays(self):
        """The exact decoder/CNN arrays and metadata written to network_data.npz."""
        save_kwargs = {
            "info": self.get_model_info(),
            "feature_sampling": np.array("bilinear_repeat_texel_center"),
            "normal_encoding": np.array(self.normal_encoding),
            "normal_output_convention": np.array("opengl"),
            "normal_sampling": np.array("filter_encoded_then_decode"),
            "normal_xy_min_z": np.array(NORMAL_XY_MIN_Z, dtype=np.float32),
            "pos_encoding_type": np.array(self.pos_encoding_type),
            "pos_encoding_tile_size": np.array(self.pos_encoding_tile_size, dtype=np.int32),
            "pos_encoding_reference_size": np.array(
                self.pos_encoding_reference_size, dtype=np.int32
            ),
            "network": self.network.params.detach().cpu().numpy(),
            "decoder_output_bias": self.decoder_output_bias.detach().cpu().numpy(),
        }

        from Core.Feature_Grid.serialization import deployment_arrays

        save_kwargs.update(deployment_arrays(self))
        if self.material_cnn is not None:
            save_kwargs.update(self.material_cnn.export_arrays())
        return save_kwargs

    def network_data_bytes(self):
        import io
        stream = io.BytesIO()
        np.savez(stream, **self.network_data_arrays())
        return stream.tell()

    @torch.no_grad()
    def save(self, curr_iter: int, model_path: str) -> None:

        save_model = copy.deepcopy(self).cpu()

        save_path = os.path.join(model_path, f"train_result_{curr_iter}")
        os.makedirs(save_path, exist_ok=True)

        model_path = os.path.join(save_path, f"model.pth")
        from checkpoint import atomic_torch_save
        atomic_torch_save(self.state_dict(), model_path)

        if self.quantize:

            # in tcnn the input dim would be padded to the nearest multiple of 16
            # https://github.com/NVlabs/tiny-cuda-nn/issues/6

            save_model.simulate_quantize()

            np.savez(os.path.join(save_path, 'network_data.npz'), **self.network_data_arrays())
            from Core.Feature_Grid.compression import export_latents

            export_latents(save_model, save_path)
