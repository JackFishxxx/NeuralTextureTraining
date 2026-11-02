import argparse
import json
import difflib
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from Core.Feature_Grid.feature_grid import FeatureGridSpec
from Core.Material_Encoding.compression import DEFAULT_FORMATS, parse_material_formats


def _normalize_astc_block(value: str) -> tuple[str, tuple[int, int]]:
    from Common.Block_Compression.astc import normalize_astc_block
    block=normalize_astc_block(value)
    return block,tuple(map(int,block.split('x')))


def _load_yaml_defaults(config_path=None):
    """Load YAML config file as a dict of default values.

    Args:
        config_path: Explicit path to a YAML config file. If None, auto-discovers
                     ``config_pc.yaml`` in the project root directory (same dir as this file).

    Returns:
        dict of parameter defaults, or empty dict if file not found.
    """
    if config_path is not None:
        p = Path(config_path)
    else:
        p = Path(__file__).resolve().parent / "config_pc.yaml"

    if not p.exists():
        if config_path is not None:
            raise FileNotFoundError(f'Config file does not exist: {p}')
        return {}

    with open(p, "r", encoding="utf-8") as f:
        class ConfigLoader(yaml.SafeLoader):
            pass

        def unique_mapping(loader, node):
            keys = [loader.construct_object(key) for key, _ in node.value]
            if any(not isinstance(key, str) for key in keys):
                raise ValueError(f'Configuration keys must be strings in {p}')
            if len(keys) != len(set(keys)):
                raise ValueError(f'Duplicate configuration key in {p}')
            return loader.construct_mapping(node)

        ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
        data = yaml.load(f, Loader=ConfigLoader)

    if not isinstance(data, dict):
        raise ValueError(f'Config must contain a YAML mapping: {p}')
    return data


def _validate_yaml_defaults(parser, defaults):
    actions = {action.dest: action for action in parser._actions}
    for name, value in defaults.items():
        if name not in actions or name in ('help', 'config'):
            match = difflib.get_close_matches(str(name), actions, n=1, cutoff=.55)
            hint = f'; {name} -> {match[0]}' if match else ''
            parser.error(f'Unknown configuration key: {name}{hint}')
        action = actions[name]
        valid = True
        if isinstance(action.default, bool):
            valid = isinstance(value, bool)
        elif action.type in (int, float):
            values = value if isinstance(action.nargs, int) and isinstance(value, list) else [value]
            valid = all(isinstance(item, (int, float) if action.type is float else int)
                        and not isinstance(item, bool) and math.isfinite(item) for item in values)
            if isinstance(action.nargs, int):
                valid &= isinstance(value, list) and len(value) == action.nargs
            if value is None:
                valid = action.default is None
        elif action.type is str:
            valid = isinstance(value, str) or (value is None and action.default is None)
        if not valid:
            parser.error(f'Invalid type or nonfinite value for {name}: {value!r}')
        if action.choices is not None and value not in action.choices:
            parser.error(f'Invalid {name}: {value!r}; choose from {action.choices}')
    for name in ('feature_grid_configs', 'pbr_point_lights'):
        if name in defaults:
            value = defaults[name]
            if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
                parser.error(f'{name} must be a list of mappings')
    if 'feature_grid_configs' in defaults and not defaults['feature_grid_configs']:
        parser.error('feature_grid_configs must not be empty')
    if 'texture_loss_weights' in defaults and not isinstance(defaults['texture_loss_weights'], dict):
        parser.error('texture_loss_weights must be a mapping')
    for name in ('pbr_directional_light_direction', 'pbr_directional_light_color'):
        if name in defaults:
            value = defaults[name]
            if (not isinstance(value, list) or len(value) != 3
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and math.isfinite(v) for v in value)):
                parser.error(f'{name} must contain three finite numbers')

class Config():

    def __init__(self, params: argparse.Namespace):

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        ### ---------- experiment configs ---------- ###
        self.data_dir = params.data_dir
        self.save_dir = params.save_dir
        self.load_iter = params.load_iter
        self.load_dir = params.load_dir
        self.mode = params.mode
        self.warm_start = params.warm_start
        if self.load_dir and self.load_iter == 0:
            raise ValueError('load_dir requires load_iter; use -1 for the newest checkpoint')
        if self.load_iter != 0 and not self.load_dir:
            raise ValueError('load_iter requires load_dir')
        if self.mode == "infer":
            if not self.load_dir:
                raise ValueError("infer mode requires --load_dir")
            if self.load_iter == 0:
                raise ValueError("infer mode requires --load_iter (use -1 for the newest checkpoint)")

        ### ---------- groups batching configs ---------- ###
        self.groups_batching = params.groups_batching
        self.groups_batch_dir = params.groups_batch_dir
        self.groups_save_dir = params.groups_save_dir
        self.groups_max_workers = params.groups_max_workers
        self.groups_verbose = params.groups_verbose
        if self.groups_max_workers <= 0:
            raise ValueError("groups_max_workers must be positive")
        if self.groups_batching and self.groups_max_workers != 1:
            raise ValueError(
                "groups_max_workers > 1 is not implemented; use 1 for sequential group training"
            )
        # Auto-discover all texture group subdirectories under groups_batch_dir
        if self.groups_batching:
            if not os.path.isdir(self.groups_batch_dir):
                raise ValueError(f"groups_batch_dir '{self.groups_batch_dir}' does not exist or is not a directory.")
            self.groups_list: List[str] = sorted([
                d for d in os.listdir(self.groups_batch_dir)
                if os.path.isdir(os.path.join(self.groups_batch_dir, d))
            ])
            if len(self.groups_list) == 0:
                raise ValueError(f"No texture group subdirectories found in '{self.groups_batch_dir}'.")
            print(f"[GroupsBatching] Found {len(self.groups_list)} texture groups: {self.groups_list}")
        else:
            self.groups_list: List[str] = []

        ### ---------- quantization configs ---------- ###
        self.quantize = True
        self.quantize_bits = params.quantize_bits
        # Feature-grid QAT: multiplicative noise schedule (ANA-style cosine anneal)
        self.qat_noise_schedule = str(getattr(params, "qat_noise_schedule", "cosine")).strip().lower()
        if self.qat_noise_schedule not in ("none", "cosine"):
            raise ValueError(
                f"qat_noise_schedule must be 'none' or 'cosine', got '{self.qat_noise_schedule}'"
            )
        self.qat_noise_mult_start = float(getattr(params, "qat_noise_mult_start", 1.0))
        self.qat_noise_mult_end = float(getattr(params, "qat_noise_mult_end", 0.25))
        self.qat_noise_warmup_frac = float(getattr(params, "qat_noise_warmup_frac", 0.1))
        self.bcn_usage_weight = params.bcn_usage_weight
        if not math.isfinite(self.bcn_usage_weight) or self.bcn_usage_weight < 0:
            raise ValueError("bcn_usage_weight must be finite and nonnegative")
        self.bcn_aware_enable = params.bcn_aware_enable
        self.bcn_learning_rate = params.bcn_learning_rate
        if not math.isfinite(self.bcn_learning_rate) or self.bcn_learning_rate <= 0:
            raise ValueError("bcn_learning_rate must be finite and positive")
        ### ---------- ASTC aware configs ---------- ###
        backend = params.astc_codec_backend
        if backend not in ("astcenc", "astc_differentiable_proxy"):
            raise ValueError("astc_codec_backend must be astcenc or astc_differentiable_proxy")
        lr = float(params.astc_learning_rate)
        warmup = float(params.astc_codec_start_frac)
        projection = int(params.astc_decoder_projection_interval)
        interval = int(params.astc_codec_in_loop_interval)
        if (
            not all(math.isfinite(value) for value in (lr, warmup))
            or lr <= 0
            or not 0 <= warmup <= 1
            or projection < 0
            or interval < 1
        ):
            raise ValueError("Invalid ASTC aware settings")
        self.astc_aware_enable = bool(params.astc_aware_enable)
        self.astc_codec_backend = backend
        self.astc_learning_rate = lr
        ### ---------- FNTC tile configs ---------- ###
        self.fntc_tile_size = params.fntc_tile_size
        if self.fntc_tile_size < 1:
            raise ValueError('fntc_tile_size must be positive')

        ### ---------- tile cnn configs ---------- ###
        self.material_cnn_enable = params.material_cnn_enable
        if self.material_cnn_enable:
            if not params.mip0_only:
                raise ValueError("Material CNN requires Mip0")
            if params.mode == "train" and not params.multi_stage_training_enable:
                raise ValueError("Material CNN training requires multi_stage_training_enable: true")
        self.material_cnn_steps = params.material_cnn_steps
        if self.material_cnn_steps < 1:
            raise ValueError("material_cnn_steps must be positive")

        ### ---------- trainer configs ---------- ###
        self.multi_stage_training_enable = bool(params.multi_stage_training_enable)
        self.multi_stage_training_decoder_steps = int(params.multi_stage_training_decoder_steps)
        self.multi_stage_training_decoder_lr_multiplier = float(
            params.multi_stage_training_decoder_lr_multiplier
        )
        self.multi_stage_training_layout_refit_steps = int(params.multi_stage_training_layout_refit_steps)
        if self.multi_stage_training_decoder_steps < 1 or self.multi_stage_training_layout_refit_steps < 0:
            raise ValueError("Decoder steps must be positive and layout refit steps nonnegative")
        if not math.isfinite(self.multi_stage_training_decoder_lr_multiplier) or self.multi_stage_training_decoder_lr_multiplier <= 0:
            raise ValueError("Decoder learning-rate multiplier must be finite and positive")

        self.max_iter = params.max_iter
        self.batch_size = params.batch_size
        self.subpixel_sampling_enable = bool(getattr(params, 'subpixel_sampling_enable', True))
        self.subpixel_sampling_ratio = float(getattr(params, 'subpixel_sampling_ratio', 0.25))
        self.subpixel_jitter = float(getattr(params, 'subpixel_jitter', 0.5))
        self.learning_rate = params.learning_rate
        self.momentum = params.momentum
        self.weight_decay = params.weight_decay
        self.lr_scheduler = params.lr_scheduler
        if not math.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError('momentum must be in [0, 1); it controls Adam beta1')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')

        self.eval_interval = params.eval_interval
        self.save_interval = params.save_interval
        self.eval_inference_tile = params.eval_inference_tile
        self.mip0_only = bool(getattr(params, "mip0_only", False))

        ### ---------- early stopping configs ---------- ###
        self.early_stop = params.early_stop
        self.early_stop_interval = params.early_stop_interval
        self.early_stop_psnr_threshold = params.early_stop_psnr_threshold
        for name in ('max_iter', 'batch_size', 'eval_interval', 'save_interval', 'early_stop_interval',
                     'n_neurons', 'pos_encoding_tile_size'):
            if getattr(params, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('n_frequencies', 'n_hidden_layers', 'eval_inference_tile'):
            if getattr(params, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        for name in ('learning_rate', 'qat_noise_mult_start', 'qat_noise_mult_end',
                     'subpixel_sampling_ratio', 'subpixel_jitter', 'early_stop_psnr_threshold'):
            value = getattr(params, name)
            if not math.isfinite(value) or value < 0 or (name == 'learning_rate' and value == 0):
                raise ValueError(f'{name} must be finite and nonnegative (learning_rate must be positive)')
        if not math.isfinite(params.qat_noise_warmup_frac) or not 0 <= params.qat_noise_warmup_frac <= 1:
            raise ValueError('qat_noise_warmup_frac must be in [0, 1]')
        # early_stop_interval must be a multiple of eval_interval so that PSNR is available at check points
        if self.early_stop and self.early_stop_interval % self.eval_interval != 0:
            raise ValueError(
                f"early_stop_interval ({self.early_stop_interval}) must be a multiple of "
                f"eval_interval ({self.eval_interval}) for PSNR-based early stopping."
            )

        ### ---------- model configs ---------- ###
        self.num_channels = 0
        self.num_mips = 1
        self.n_frequencies = int(getattr(params, "n_frequencies", 0))
        self.pos_encoding_tile_size = int(getattr(params, "pos_encoding_tile_size", 32))
        self.pos_encoding_reference_edge = int(getattr(params, "pos_encoding_reference_edge", 0))
        self.n_neurons = int(getattr(params, "n_neurons", 16))
        self.n_hidden_layers = int(getattr(params, "n_hidden_layers", 0))
        raw_output_activation = getattr(params, "output_activation", "hard_swish")
        self.output_activation = str(raw_output_activation).strip().lower().replace("-", "_")
        supported_output_activations = {"hard_swish", "hard_gelu", "leaky_relu"}
        # Avg dataset performance:
        # 1. hard_swish PSNR=40.2070 SSIM=0.9568 LPIPS=0.0805
        # 2. hard_gelu PSNR=40.1565 SSIM=0.9569 LPIPS=0.0805
        # 3. leaky_relu PSNR=40.1032 SSIM=0.9566 LPIPS=0.0803
        if self.output_activation not in supported_output_activations:
            raise ValueError(
                f"Unsupported output_activation '{raw_output_activation}'. "
                f"Supported values: {sorted(supported_output_activations)}"
            )

        self.normal_encoding = str(getattr(params, "normal_encoding", "xyz")).strip().lower().replace("-", "_")
        if self.normal_encoding not in {"xyz", "xy", "hemi_oct"}:
            raise ValueError("normal_encoding must be one of: xyz, xy, hemi_oct")

        self.super_resolution_enable = bool(getattr(params, "super_resolution_enable", False))
        self.super_resolution_base_resolution = int(getattr(params, "super_resolution_base_resolution", 512))
        if self.super_resolution_base_resolution <= 0:
            raise ValueError("super_resolution_base_resolution must be positive")

        self.direct_diffuse_infer_mode = str(params.direct_diffuse_infer_mode).strip().lower()
        if self.direct_diffuse_infer_mode not in {"disable", "rgb", "ycocg"}:
            raise ValueError("direct_diffuse_infer_mode must be one of: disable, rgb, ycocg")
        if self.super_resolution_enable and self.direct_diffuse_infer_mode != "disable":
            print("[SuperResolution] Disabling direct_diffuse_infer_mode for residual prediction")
            self.direct_diffuse_infer_mode = "disable"

        # Learned feature grids.
        default_feature_grids = [
            {"max_resolution": 1024, "format": "astc6x6", "quantize_bits": 8, "learning_rate": 0.005},
        ]
        self.feature_grid_configs = default_feature_grids if params.feature_grid_configs is None else params.feature_grid_configs
        if (not isinstance(self.feature_grid_configs, list) or not self.feature_grid_configs
                or not all(isinstance(g, dict) for g in self.feature_grid_configs)):
            raise ValueError('feature_grid_configs must be a nonempty list of mappings')

        # Per-texture-type weights shared by training loss and evaluation aggregation.
        # If None, weights from dataset.get_texture_config() will be used
        # Keys: texture type names (diffuse, normal, roughness, etc.)
        # Values: per-channel weight
        default_texture_weights = {
            "diffuse": 1.0,
            "normal": 0.5,
            "roughness": 0.5,
            "occlusion": 0.5,
            "metallic": 0.5,
            "specular": 0.5,
            "displacement": 0.5,
        }
        yaml_texture_weights = getattr(params, "texture_loss_weights", None)
        if yaml_texture_weights is not None:
            if not isinstance(yaml_texture_weights, dict) or set(yaml_texture_weights) - set(default_texture_weights):
                raise ValueError('texture_loss_weights must map known material names to weights')
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0
                   for v in yaml_texture_weights.values()):
                raise ValueError('texture_loss_weights values must be finite and nonnegative')
        self.texture_weights: Optional[Dict[str, float]] = (
            {str(k): float(v) for k, v in yaml_texture_weights.items()}
            if isinstance(yaml_texture_weights, dict)
            else default_texture_weights
        )

        # Final per-channel loss weights list (generated from texture_weights and available textures)
        self.output_loss_weights: Optional[List[float]] = None
        self.network_learning_rate = params.learning_rate

        ### ---------- material baseline ---------- ###
        self.enable_material_compare = bool(params.enable_material_compare)
        self.material_compare_interval = params.material_compare_interval
        if self.material_compare_interval < 1:
            raise ValueError('material_compare_interval must be positive')
        self.material_compression_formats = {group: fmt.name for group,fmt in parse_material_formats(params.material_compression_formats).items()}
        self.astcenc_path = params.astcenc_path
        self.astcenc_quality = params.astcenc_quality
        self.astc_block, _ = _normalize_astc_block(params.astc_block)
        self.material_compare_resolution = getattr(params, "material_compare_resolution", None)
        if self.material_compare_resolution is not None and self.material_compare_resolution <= 0:
            raise ValueError('material_compare_resolution must be positive')

        ### ---------- FNTC tile recompression ---------- ###
        self.material_recompression_enable = params.material_recompression_enable

        ### ---------- PBR scene comparison ---------- ###
        self.enable_pbr_compare = bool(getattr(params, "enable_pbr_compare", False))
        self.pbr_compare_interval = int(getattr(params, "pbr_compare_interval", 5000))
        self.pbr_render_resolution = int(getattr(params, "pbr_render_resolution", 512))
        self.pbr_detail_resolution = params.pbr_detail_resolution
        self.pbr_detail_center = list(params.pbr_detail_center)
        self.pbr_detail_zoom = float(params.pbr_detail_zoom)
        if (len(self.pbr_detail_center) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.pbr_detail_center)
                or not math.isfinite(self.pbr_detail_zoom) or self.pbr_detail_zoom <= 1):
            raise ValueError('PBR detail center must be two coordinates in [0,1] and zoom must be finite and > 1')
        self.pbr_mip_lod_bias = float(getattr(params, "pbr_mip_lod_bias", -1.0))
        self.pbr_loss_enable = bool(getattr(params, "pbr_loss_enable", False))
        self.pbr_loss_weight = float(getattr(params, "pbr_loss_weight", 0.1))
        self.pbr_directional_light_enable = bool(getattr(params, "pbr_directional_light_enable", False))
        self.pbr_directional_light_direction = list(getattr(params, "pbr_directional_light_direction", [-0.45, 0.62, -0.72]))
        self.pbr_directional_light_intensity = float(getattr(params, "pbr_directional_light_intensity", 2.4))
        self.pbr_directional_light_color = list(getattr(params, "pbr_directional_light_color", [1.0, 1.0, 1.0]))
        self.pbr_point_lights = list(getattr(params, "pbr_point_lights", [
            {"position": [0.0, 0.82, 0.7], "intensity": 6.5, "color": [1.0, 1.0, 1.0]},
            {"position": [-0.8, 0.45, -0.15], "intensity": 1.2, "color": [1.0, 1.0, 1.0]},
        ]))
        for name in ('pbr_loss_weight', 'pbr_directional_light_intensity'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if self.pbr_compare_interval <= 0:
            raise ValueError("pbr_compare_interval must be positive")
        if self.pbr_render_resolution < 64:
            raise ValueError("pbr_render_resolution must be at least 64")
        if self.pbr_detail_resolution < 64:
            raise ValueError('pbr_detail_resolution must be at least 64')
        if not -8.0 <= self.pbr_mip_lod_bias <= 8.0:
            raise ValueError("pbr_mip_lod_bias must be in [-8, 8]")
        if self.pbr_loss_weight < 0.0:
            raise ValueError("pbr_loss_weight must be non-negative")
        if (
            len(self.pbr_directional_light_direction) != 3
            or len(self.pbr_directional_light_color) != 3
        ):
            raise ValueError("PBR directional light direction and color must contain 3 values")
        if self.pbr_directional_light_intensity < 0.0:
            raise ValueError("pbr_directional_light_intensity must be non-negative")
        for index, light in enumerate(self.pbr_point_lights):
            if isinstance(light, dict) and set(light) - {"position", "intensity", "color"}:
                raise ValueError(f"Unknown pbr_point_lights[{index}] keys")
            if not isinstance(light, dict) or len(light.get("position", [])) != 3:
                raise ValueError(f"pbr_point_lights[{index}] must contain a 3-value position")
            if len(light.get("color", [1.0, 1.0, 1.0])) != 3:
                raise ValueError(f"pbr_point_lights[{index}] color must contain 3 values")
            values = [
                *light["position"],
                *light.get("color", [1.0, 1.0, 1.0]),
                light.get("intensity", 0.0),
            ]
            if any(
                not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                for v in values
            ):
                raise ValueError(f"pbr_point_lights[{index}] must contain finite numbers")
            if float(light.get("intensity", 0.0)) < 0.0:
                raise ValueError(f"pbr_point_lights[{index}] intensity must be non-negative")

        # Parse each grid's explicit compact format.
        from Core.Feature_Grid.formats import GridCompression

        raw_grids = self.feature_grid_configs
        recipes = tuple(
            GridCompression.from_config(g)
            for g in raw_grids
        )
        if not self.mip0_only:
            raise ValueError('Block-compressed deployment currently requires mip0_only: true; '
                             'multi-Mip training/export is not implemented')
        processed = []
        for cfg, recipe in zip(raw_grids, recipes):
            cfg = dict(cfg)
            if recipe.format == "bc1":
                if not self.mip0_only or self.direct_diffuse_infer_mode != "disable":
                    raise ValueError("BC1 grids require mip0_only and direct diffuse disabled")
                cfg.setdefault("n_features_per_level", 4)
            if recipe.format == 'astc':
                cfg.setdefault('n_features_per_level', 4)
            spec = FeatureGridSpec.from_config(
                cfg, self.quantize_bits, self.learning_rate
            )
            if (
                recipe.format == "bc1"
                and spec.n_features_per_level not in (4, 8)
            ):
                raise ValueError("BC1 RGBA needs four or eight features")
            processed.append(
                dict(
                    max_resolution=spec.max_resolution,
                    n_levels=1 if self.mip0_only else spec.n_levels,
                    n_features_per_level=spec.n_features_per_level,
                    quantize_bits=spec.quantize_bits,
                    learning_rate=spec.learning_rate,
                    interpolation=spec.interpolation,
                    feature_gradient_count=spec.feature_gradient_count,
                    **recipe.as_config(),
                )
            )
        self.feature_grid_configs = processed
        self.grid_compressions = recipes
        families = {r.format for r in recipes}
        self.codec_training_enable = self.mode == "train" and (
            ("bc1" in families and self.bcn_aware_enable)
            or ("astc" in families and self.astc_aware_enable)
        )
        from Core.Feature_Grid.formats import GridTraining
        self.grid_training = tuple(
            GridTraining(self.bcn_aware_enable if r.format == 'bc1' else self.astc_aware_enable,
                         0. if r.format == 'bc1' else warmup,
                         1 if r.format == 'bc1' or i in self.native_astc_grids else interval)
            for i, r in enumerate(recipes))
        active = [p for p in self.grid_training if p.enabled]
        self.codec_start_frac = min((p.start_frac for p in active), default=0.)
        self.codec_interval = math.gcd(*(p.interval for p in active)) if active else 1
        self.astc_decoder_projection_interval = (
            int(params.astc_decoder_projection_interval) if self.native_astc_grids else 0
        )
        if ("bc1" in families or len(set(recipes)) > 1) and (
            not self.mip0_only or self.direct_diffuse_infer_mode != "disable"
        ):
            raise ValueError("BC1/mixed grids require mip0_only and direct diffuse disabled")

    @property
    def native_astc_grids(self):
        return tuple(
            i for i, (r, g) in enumerate(zip(self.grid_compressions, self.feature_grid_configs))
            if self.astc_codec_backend == "astc_differentiable_proxy"
            and r.format == "astc" and r.astc_block == "6x6"
            and g["n_features_per_level"] == 4
            and not (i == 0 and self.direct_diffuse_infer_mode != "disable")
        )

    def make_group_config(self, group_name: str) -> "Config":
        """Create a shallow copy of this Config with data_dir and save_dir
        overridden for a specific texture group (used in GroupsBatching mode).

        Args:
            group_name: Name of the texture group subdirectory.

        Returns:
            A new Config object pointing to the specific group's data and save paths.
        """
        import copy

        cfg = copy.copy(self)
        cfg.data_dir = os.path.join(self.groups_batch_dir, group_name)
        cfg.save_dir = os.path.join(self.groups_save_dir, group_name)
        # Reset per-dataset fields so they get re-populated by the dataset
        cfg.num_channels = 0
        cfg.num_mips = 1
        cfg.output_loss_weights = None
        return cfg


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # --config: path to a YAML config file (parsed first, then overridden by CLI args)
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config path (default: config_pc.yaml; mobile: config_mobile.yaml)')

    ### ---------- experiment configs ---------- ###
    parser.add_argument('--data_dir', type=str, default='data/test',
                        help='root directory of dataset')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory of dataset')
    parser.add_argument('--load_iter', type=int, default=0,
                        help='0 -> do not load model, -1 means the newest')
    parser.add_argument('--load_dir', type=str,
                        help='')
    parser.add_argument('--warm_start', action='store_true', default=False,
                        help='load weights into a new run instead of resuming training state')
    parser.add_argument('--mode', type=str, default="train",
                        help='',
                        choices=['train', 'infer'])

    ### ---------- groups batching configs ---------- ###
    parser.add_argument('--groups_batching', action='store_true', default=False,
                        help='enable GroupsBatching mode: train multiple texture groups from groups_batch_dir sequentially')
    parser.add_argument('--groups_batch_dir', type=str, default='data_batch',
                        help='root directory containing multiple texture group subdirectories (each subfolder is one texture group)')
    parser.add_argument('--groups_save_dir', type=str, default='save_batch',
                        help='root directory for saving GroupsBatching results (each group gets its own subfolder)')
    parser.add_argument('--groups_max_workers', type=int, default=1,
                        help='reserved worker count; GroupsBatching currently supports only 1')
    parser.add_argument('--groups_verbose', action=argparse.BooleanOptionalAction, default=False,
                        help='print per-group training details instead of aggregate progress only')

    ### ---------- quantization configs ---------- ###
    parser.add_argument('--quantize_bits', type=int, default=8,
                        help='choose the bits to quantize',
                        choices=[2, 4, 8])
    parser.add_argument('--bcn_usage_weight', dest='bcn_usage_weight', type=float, default=0.01,
                        help='Weak binary usage/confidence prior; 0 disables it')
    parser.add_argument('--bcn_aware_enable', dest='bcn_aware_enable', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--bcn_learning_rate', dest='bcn_learning_rate', type=float, default=0.01,
                        help='Adam learning rate in RGB565/selector code units')
    parser.add_argument(
        '--qat_noise_schedule',
        type=str,
        default='cosine',
        choices=['none', 'cosine'],
        help='QAT additive noise strength schedule over training iterations (feature grid STE branch)',
    )
    parser.add_argument(
        '--qat_noise_mult_start',
        type=float,
        default=1.0,
        help='noise multiplier at warmup end / start of cosine segment',
    )
    parser.add_argument(
        '--qat_noise_mult_end',
        type=float,
        default=0.25,
        help='noise multiplier at end of training (cosine tail)',
    )
    parser.add_argument(
        '--qat_noise_warmup_frac',
        type=float,
        default=0.1,
        help='fraction of (max_iter-1) iterations holding mult_start before cosine decay',
    )

    ### ---------- ASTC aware configs ---------- ###
    group = parser.add_argument_group("ASTC aware")
    group.add_argument("--astc_aware_enable", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--astc_codec_backend", choices=["astcenc", "astc_differentiable_proxy"], default="astc_differentiable_proxy")
    group.add_argument("--astc_codec_start_frac", type=float, default=0.1,
                        help="clean warmup fraction")
    group.add_argument("--astc_learning_rate", type=float, default=0.001,
                        help="proxy endpoint and weight learning rate")
    group.add_argument("--astc_codec_in_loop_interval", type=int, default=100,
                        help="CPU codec interval; proxy runs every step")
    group.add_argument("--astc_decoder_projection_interval", type=int, default=500,
                        help="proxy material projection interval; 0 disables it")

    ### ---------- FNTC tile configs ---------- ###
    parser.add_argument('--fntc_tile_size', type=int, default=128,
                        help='material tile edge in inference texels, shared by CNN fitting/inference and tile compression')

    ### ---------- tile cnn configs ---------- ###
    group = parser.add_argument_group("tile cnn configs")
    group.add_argument('--material_cnn_enable', action=argparse.BooleanOptionalAction, default=False,
                       help='fit material CNN after layout_refit in multi-stage training, or load it for inference')
    group.add_argument('--material_cnn_steps', type=int, default=50000,
                       help='material CNN fitting steps after decoder/layout stages; independent of feature gradients')

    ### ---------- trainer configs ---------- ###
    parser.add_argument('--multi_stage_training_enable', action=argparse.BooleanOptionalAction, default=True,
                        help='run representation learning, decoder fitting and optional layout refinement')
    parser.add_argument('--multi_stage_training_decoder_steps', type=int, default=5000,
                        help='maximum decoder-fitting steps with the learned ASTC representation frozen')
    parser.add_argument('--multi_stage_training_decoder_lr_multiplier', type=float, default=1.0,
                        help='learning-rate multiplier for the newly initialized decoder stage')
    parser.add_argument('--multi_stage_training_layout_refit_steps', type=int, default=1000,
                        help='offline latent refit and block layout selection after decoder fitting; 0 disables')

    parser.add_argument('--max_iter', type=int, default=100000,
                        help='maximum training iteration')
    parser.add_argument('--batch_size', type=int, default=16384,
                        help='batch size')
    parser.add_argument('--subpixel_sampling_enable', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--subpixel_sampling_ratio', type=float, default=0.25)
    parser.add_argument('--subpixel_jitter', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Adam first-moment coefficient (beta1)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='plateau',
                        help='scheduler type',
                        choices=['plateau', 'steplr', 'cosine', 'poly', 'none'])
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='iteration interval for evaluation')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='iteration interval for saving model')
    parser.add_argument('--eval_inference_tile', type=int, default=512,
                        help='eval/infer: tile edge in pixels (0 = one batch over full plane, may OOM)')
    parser.add_argument('--mip0_only', action=argparse.BooleanOptionalAction, default=False,
                        help='restrict training samples, loss, inference metrics, and comparisons to Mip0')
    parser.add_argument('--feature_grid_configs', type=yaml.safe_load, default=None,
                        help='YAML list of feature grids; each accepts feature_gradient_count: 0, 2, 4, 8 or 24')
    parser.add_argument('--texture_loss_weights', type=yaml.safe_load, default=None,
                        help='YAML mapping of per-texture loss/eval weights')

    ### ---------- early stopping configs ---------- ###
    parser.add_argument('--early_stop', action=argparse.BooleanOptionalAction, default=True,
                        help='enable early stopping when PSNR improvement is below threshold (default: enabled)')
    parser.add_argument('--early_stop_interval', type=int, default=5000,
                        help='number of iterations per segment for early stopping PSNR evaluation')
    parser.add_argument('--early_stop_psnr_threshold', type=float, default=0.01,
                        help='minimum PSNR improvement (in dB) between two consecutive segments; training stops if improvement is below this value')

    ### ---------- material baseline configs ---------- ###
    parser.add_argument('--enable_material_compare', action=argparse.BooleanOptionalAction, default=True,
                        help='report FNTC material and conventional per-group compression metrics')
    parser.add_argument('--material_compare_interval', type=int, default=5000,
                        help='iteration interval for material comparisons, independent of checkpoint saving')
    parser.add_argument('--material_compression_formats', type=yaml.safe_load,
                        default=DEFAULT_FORMATS.copy(),
                        help='per-group baseline and FNTC tile formats: diffuse, normal, romsd')
    parser.add_argument('--astcenc_path', type=str, default='tools/astcenc',
                        help='astcenc executable path or directory. If path does not exist, astcenc will be auto-downloaded there')
    parser.add_argument('--astcenc_quality', type=str, default='medium',
                        choices=['fastest', 'fast', 'medium', 'thorough', 'exhaustive'],
                        help='quality preset for ASTC latent, SR base and ASTC material formats')
    parser.add_argument('--astc_block', type=str, default='6x6',
                        help='shared SR base ASTC block size, independent of material and latent formats')
    parser.add_argument('--material_compare_resolution', type=int, default=None,
                        help='shared FNTC tile and baseline source max edge; omit to use the native material size')

    ### ---------- FNTC tile recompression configs ---------- ###
    parser.add_argument('--material_recompression_enable', action=argparse.BooleanOptionalAction, default=True,
                        help='compress inferred FNTC tiles using the same formats as the baseline; selects the core metric')

    ### ---------- PBR scene comparison configs ---------- ###
    parser.add_argument('--enable_pbr_compare', action=argparse.BooleanOptionalAction, default=False,
                        help='render inferred and ground-truth texture sets in a Cornell-box PBR scene')
    parser.add_argument('--pbr_compare_interval', type=int, default=5000,
                        help='training iteration interval for PBR scene comparison images')
    parser.add_argument('--pbr_render_resolution', type=int, default=512,
                        help='height of each directly rendered PBR scene panel')
    parser.add_argument('--pbr_detail_resolution', type=int, default=2048,
                        help='independent PBR detail render width and height')
    parser.add_argument('--pbr_detail_center', type=float, nargs=2, default=[0.76, 0.59],
                        help='detail viewport center in the original square raster, normalized x/y from top-left')
    parser.add_argument('--pbr_detail_zoom', type=float, default=3.5,
                        help='PBR detail viewport magnification, rendered separately; must be > 1')
    parser.add_argument('--pbr_mip_lod_bias', type=float, default=-1.0,
                        help='PBR texture mip LOD bias; negative values select sharper mips')
    parser.add_argument('--pbr_loss_enable', action=argparse.BooleanOptionalAction, default=False,
                        help='add differentiable material-to-PBR shaded RGB loss during training')
    parser.add_argument('--pbr_loss_weight', type=float, default=0.1,
                        help='weight of the differentiable PBR shaded RGB loss')
    parser.add_argument('--pbr_directional_light_enable', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--pbr_directional_light_direction', type=yaml.safe_load, default=[-0.45, 0.62, -0.72])
    parser.add_argument('--pbr_directional_light_intensity', type=float, default=2.4)
    parser.add_argument('--pbr_directional_light_color', type=yaml.safe_load, default=[1.0, 1.0, 1.0])
    parser.add_argument('--pbr_point_lights', type=yaml.safe_load, default=[
        {"position": [0.0, 0.82, 0.7], "intensity": 6.5, "color": [1.0, 1.0, 1.0]},
        {"position": [-0.8, 0.45, -0.15], "intensity": 1.2, "color": [1.0, 1.0, 1.0]},
    ])

    ### ---------- algorithm configs ---------- ###
    parser.add_argument('--n_frequencies', type=int, default=0,
                        help='tiled positional encoding frequency count (0 = disabled); N -> 4*N dims')
    parser.add_argument('--pos_encoding_tile_size', type=int, default=8,
                        help='tile edge in texels for the tiled positional encoding '
                             '(fundamental period; the encoding repeats every tile)')
    parser.add_argument('--pos_encoding_reference_edge', type=int, default=0,
                        help='reference texture edge in texels used to convert tile_size texels to UV; '
                             '0 = auto: feature-grid max resolution (train.py sets the real texture edge)')
    parser.add_argument('--normal_encoding', type=str, default='xyz', choices=['xyz', 'xy', 'hemi_oct'],
                        help='normal target encoding: xyz=3 channels, xy/hemi_oct=2 channels')
    parser.add_argument('--direct_diffuse_infer_mode', type=str, default='disable',
                        choices=['disable', 'rgb', 'ycocg'],
                        help='direct diffuse inference mode: disable, or store diffuse in feature0 RGB as rgb/ycocg')
    parser.add_argument('--super_resolution_enable', action=argparse.BooleanOptionalAction, default=False,
                        help='enable residual super-resolution from a lower-resolution texture mip')
    parser.add_argument('--super_resolution_base_resolution', type=int, default=512,
                        help='base texture edge used for super-resolution residual input')
    parser.add_argument('--n_neurons', type=int, default=16,
                        help='MLP hidden-layer width (keep a multiple of 16 for CutlassMLP)')
    parser.add_argument('--n_hidden_layers', type=int, default=0,
                        help='MLP hidden-layer count (0 = single-layer direct mapping)')
    parser.add_argument('--output_activation', type=str, default='hard_swish',
                        choices=['hard_swish', 'hard_gelu', 'leaky_relu'],
                        help='output activation applied after the MLP')

    # ── Two-stage parsing: YAML defaults → CLI overrides ──
    # Stage 1: extract --config path only
    known, _ = parser.parse_known_args()
    try:
        yaml_defaults = _load_yaml_defaults(known.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    if yaml_defaults:
        _validate_yaml_defaults(parser, yaml_defaults)
        parser.set_defaults(**yaml_defaults)

    # Stage 2: full parse (CLI args override YAML values)
    return parser.parse_args()


def write_resolved_config(configs, directory) -> None:
    """Persist scalar/list/dict config values so log analyzers can recover run settings."""
    out = {}
    for key, value in vars(configs).items():
        if key.startswith("_") or key in ('grid_compressions', 'grid_training'):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple, dict)):
            try:
                json.dumps(value)
                out[key] = value
            except TypeError:
                out[key] = str(value)
    path = os.path.join(directory, "resolved_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"resolved_config: {path}")
