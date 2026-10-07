import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from feature_grid import FeatureGridSpec


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got '{value}'")


def _normalize_astc_block(value: str) -> tuple[str, tuple[int, int]]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", str(value))
    if match is None:
        raise ValueError("astc_block must use the form WxH, such as 6x6")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("astc_block dimensions must be positive")
    return f"{width}x{height}", (width, height)


def mip_sampling_probabilities(num_mips: int, mip0_only: bool, device: str = "cpu") -> torch.Tensor:
    if num_mips < 1:
        raise ValueError("num_mips must be at least 1")
    if mip0_only:
        probabilities = torch.zeros(num_mips, device=device)
        probabilities[0] = 1.0
        return probabilities

    probabilities = []
    current_prob = 1.0
    for _ in range(num_mips):
        current_prob /= 4.0
        probabilities.append(max(current_prob, 0.05))
    result = torch.tensor(probabilities, device=device)
    return result / result.sum()


def inference_mips(num_mips: int, mip0_only: bool) -> range:
    if num_mips < 1:
        raise ValueError("num_mips must be at least 1")
    return range(1) if mip0_only else range(num_mips)


def _load_yaml_defaults(config_path=None):
    """Load YAML config file as a dict of default values.

    Args:
        config_path: Explicit path to a YAML config file. If None, auto-discovers
                     ``config.yaml`` in the project root directory (same dir as this file).

    Returns:
        dict of parameter defaults, or empty dict if file not found.
    """
    if config_path is not None:
        p = Path(config_path)
    else:
        # Auto-discover config.yaml next to this file
        p = Path(__file__).resolve().parent / "config.yaml"

    if not p.exists():
        return {}

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else {}

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
        self.save_bits = params.save_bits
        # Feature-grid QAT: multiplicative noise schedule (ANA-style cosine anneal)
        self.qat_noise_schedule = str(getattr(params, "qat_noise_schedule", "cosine")).strip().lower()
        if self.qat_noise_schedule not in ("none", "cosine"):
            raise ValueError(
                f"qat_noise_schedule must be 'none' or 'cosine', got '{self.qat_noise_schedule}'"
            )
        self.qat_noise_mult_start = float(getattr(params, "qat_noise_mult_start", 1.0))
        self.qat_noise_mult_end = float(getattr(params, "qat_noise_mult_end", 0.25))
        self.qat_noise_warmup_frac = float(getattr(params, "qat_noise_warmup_frac", 0.1))
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
        self.astc_codec_start_frac = warmup
        self.astc_codec_update_latent = (
            bool(params.astc_codec_update_latent) if backend == "astcenc" else False
        )
        # The two-stage trainer temporarily disables codec updates during adaptation.
        self.astc_codec_in_loop_enable = self.astc_aware_enable and params.mode == "train"
        self.astc_codec_in_loop_interval = 1 if backend == "astc_differentiable_proxy" else interval
        self.astc_decoder_projection_interval = projection

        ### ---------- feature gradient configs ---------- ###
        self.feature_gradient_count = int(params.feature_gradient_count)
        if self.feature_gradient_count not in (0, 2, 4, 8, 24):
            raise ValueError("feature_gradient_count must be 0, 2, 4, 8 or 24")

        ### ---------- trainer configs ---------- ###
        self.two_stage_finetune_enable = bool(getattr(params, "two_stage_finetune_enable", True))
        self.two_stage_finetune_max_steps = int(getattr(params, "two_stage_finetune_max_steps", 5000))
        self.two_stage_finetune_lr_multiplier = float(
            getattr(params, "two_stage_finetune_lr_multiplier", 1.0)
        )

        self.max_iter = params.max_iter
        self.batch_size = params.batch_size
        self.subpixel_sampling_enable = bool(getattr(params, 'subpixel_sampling_enable', True))
        self.subpixel_sampling_ratio = float(getattr(params, 'subpixel_sampling_ratio', 0.25))
        self.subpixel_jitter = float(getattr(params, 'subpixel_jitter', 0.5))
        self.learning_rate = params.learning_rate
        self.momentum = params.momentum
        self.weight_decay = params.weight_decay
        self.lr_scheduler = params.lr_scheduler

        self.eval_interval = params.eval_interval
        self.save_interval = params.save_interval
        self.eval_inference_tile = params.eval_inference_tile
        self.eval_metrics_max_edge = params.eval_metrics_max_edge
        self.mip0_only = bool(getattr(params, "mip0_only", False))

        ### ---------- early stopping configs ---------- ###
        self.early_stop = params.early_stop
        self.early_stop_interval = params.early_stop_interval
        self.early_stop_psnr_threshold = params.early_stop_psnr_threshold
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
        if self.normal_encoding == "rgb":
            self.normal_encoding = "xyz"
        if self.normal_encoding == "hemi_octa":
            self.normal_encoding = "hemi_oct"
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
            {"max_resolution": 1024, "quantize_bits": 8, "save_bits": 32, "learning_rate": 0.005},
            #{"max_resolution": 512, "quantize_bits": 8, "save_bits": 32, "learning_rate": 0.005},
        ]
        self.feature_grid_configs: Optional[List[Dict]] = params.feature_grid_configs or default_feature_grids

        # Per-texture-type weights shared by training loss and evaluation aggregation.
        # If None, weights from dataset.get_texture_config() will be used
        # Keys: texture type names (diffuse, normal, roughness, etc.)
        # Values: per-channel weight
        default_texture_weights = {
            "diffuse": 1.0,
            "normal": 0.4,
            "roughness": 0.4,
            "occlusion": 0.4,
            "metallic": 0.4,
            "specular": 0.4,
            "displacement": 0.4,
        }
        yaml_texture_weights = getattr(params, "texture_loss_weights", None)
        self.texture_weights: Optional[Dict[str, float]] = (
            {str(k): float(v) for k, v in yaml_texture_weights.items()}
            if isinstance(yaml_texture_weights, dict)
            else default_texture_weights
        )
        # Backward-compatible alias for older call sites/configs.
        self.texture_loss_weights = self.texture_weights

        # Final per-channel loss weights list (generated from texture_weights and available textures)
        self.output_loss_weights: Optional[List[float]] = None
        self.network_learning_rate = params.learning_rate

        ### ---------- ASTC comparison (vs traditional round-trip) ---------- ###
        self.enable_astc_compare = bool(params.enable_astc_compare)
        self.astcenc_path = params.astcenc_path
        self.astcenc_quality = params.astcenc_quality
        self.astc_block, self.astc_aware_block = _normalize_astc_block(params.astc_block)
        self.ref_astc_resolution = getattr(params, "ref_astc_resolution", None)

        ### ---------- PBR scene comparison ---------- ###
        self.enable_pbr_compare = bool(getattr(params, "enable_pbr_compare", False))
        self.pbr_compare_interval = int(getattr(params, "pbr_compare_interval", 5000))
        self.pbr_render_resolution = int(getattr(params, "pbr_render_resolution", 512))
        self.pbr_mip_lod_bias = float(getattr(params, "pbr_mip_lod_bias", -1.0))
        self.pbr_loss_enable = bool(getattr(params, "pbr_loss_enable", False))
        self.pbr_loss_weight = float(getattr(params, "pbr_loss_weight", 0.1))
        self.pbr_directional_light_enable = bool(getattr(params, "pbr_directional_light_enable", True))
        self.pbr_directional_light_direction = list(getattr(params, "pbr_directional_light_direction", [-0.45, 0.62, -0.72]))
        self.pbr_directional_light_intensity = float(getattr(params, "pbr_directional_light_intensity", 2.4))
        self.pbr_directional_light_color = list(getattr(params, "pbr_directional_light_color", [1.0, 1.0, 1.0]))
        self.pbr_point_lights = list(getattr(params, "pbr_point_lights", [
            {"position": [0.0, 0.82, 0.7], "intensity": 6.5, "color": [1.0, 1.0, 1.0]},
            {"position": [-0.8, 0.45, -0.15], "intensity": 1.2, "color": [1.0, 1.0, 1.0]},
        ]))
        if self.pbr_compare_interval <= 0:
            raise ValueError("pbr_compare_interval must be positive")
        if self.pbr_render_resolution < 64:
            raise ValueError("pbr_render_resolution must be at least 64")
        if not -8.0 <= self.pbr_mip_lod_bias <= 8.0:
            raise ValueError("pbr_mip_lod_bias must be in [-8, 8]")
        if self.pbr_loss_weight < 0.0:
            raise ValueError("pbr_loss_weight must be non-negative")
        if len(self.pbr_directional_light_direction) != 3 or len(self.pbr_directional_light_color) != 3:
            raise ValueError("PBR directional light direction and color must contain 3 values")
        if self.pbr_directional_light_intensity < 0.0:
            raise ValueError("pbr_directional_light_intensity must be non-negative")
        for index, light in enumerate(self.pbr_point_lights):
            if not isinstance(light, dict) or len(light.get("position", [])) != 3:
                raise ValueError(f"pbr_point_lights[{index}] must contain a 3-value position")
            if len(light.get("color", [1.0, 1.0, 1.0])) != 3:
                raise ValueError(f"pbr_point_lights[{index}] color must contain 3 values")
            if float(light.get("intensity", 0.0)) < 0.0:
                raise ValueError(f"pbr_point_lights[{index}] intensity must be non-negative")

        # Normalize configs to the internal format expected by the model
        if self.feature_grid_configs is not None:
            processed: List[Dict] = []
            for cfg in self.feature_grid_configs:
                spec = FeatureGridSpec.from_config(
                    cfg, self.quantize_bits, self.save_bits, self.learning_rate
                )

                processed.append({
                    "max_resolution": spec.max_resolution,
                    "n_levels": spec.n_levels,
                    "quantize_bits": spec.quantize_bits,
                    "save_bits": spec.save_bits,
                    "learning_rate": spec.learning_rate,
                    "interpolation": spec.interpolation,
                })

            self.feature_grid_configs = processed

    def make_group_config(self, group_name: str) -> 'Config':
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
    parser = argparse.ArgumentParser()

    # --config: path to a YAML config file (parsed first, then overridden by CLI args)
    parser.add_argument('--config', type=str, default=None,
                        help='path to YAML config file (overrides argparse defaults, overridden by CLI args)')

    ### ---------- experiment configs ---------- ###
    parser.add_argument('--data_dir', type=str, default='data/test',
                        help='root directory of dataset')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory of dataset')
    parser.add_argument('--load_iter', type=int, default=0,
                        help='0 -> do not load model, -1 means the newest')
    parser.add_argument('--load_dir', type=str,
                        help='')
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
    parser.add_argument('--save_bits', type=int, default=32,
                        help='choose the bits to quantize',
                        choices=[8, 16, 32])
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
    group.add_argument("--astc_codec_update_latent", action=argparse.BooleanOptionalAction, default=False,
                        help="CPU latent identity STE")
    group.add_argument("--astc_decoder_projection_interval", type=int, default=500,
                        help="proxy material projection interval; 0 disables it")

    ### ---------- feature gradient configs ---------- ###
    group = parser.add_argument_group("feature gradient configs")
    group.add_argument('--feature_gradient_count', type=int, choices=[0, 2, 4, 8, 24], default=0,
                       help='feature differences: 0 off, 2 right/down, 4 axial, 8 3x3, 24 5x5; '
                            'two-stage training applies them only in stage two')

    ### ---------- trainer configs ---------- ###
    parser.add_argument('--two_stage_finetune_enable', action=argparse.BooleanOptionalAction, default=True,
                        help='train zero-layer features first, then a new decoder on frozen ASTC features')
    parser.add_argument('--two_stage_finetune_max_steps', type=int, default=5000,
                        help='maximum ASTC decoder-only steps after the zero-layer baseline stops')
    parser.add_argument('--two_stage_finetune_lr_multiplier', type=float, default=1.0,
                        help='learning-rate multiplier for the newly initialized second-stage decoder')

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
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='iteration interval for evaluation')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='iteration interval for saving model')
    parser.add_argument('--eval_inference_tile', type=int, default=512,
                        help='eval/infer: tile edge in pixels (0 = one batch over full plane, may OOM)')
    parser.add_argument('--eval_metrics_max_edge', type=int, default=0,
                        help='area-downsample to this max(H,W) before PSNR/SSIM/LPIPS (0 = full res)')
    parser.add_argument('--mip0_only', action=argparse.BooleanOptionalAction, default=False,
                        help='restrict training samples, loss, inference metrics, and comparisons to Mip0')
    parser.add_argument('--feature_grid_configs', type=yaml.safe_load, default=None,
                        help='YAML list of learned feature-grid configs')
    parser.add_argument('--texture_loss_weights', type=yaml.safe_load, default=None,
                        help='YAML mapping of per-texture loss/eval weights')

    ### ---------- early stopping configs ---------- ###
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help='enable early stopping when PSNR improvement is below threshold (default: enabled)')
    parser.add_argument('--early_stop_interval', type=int, default=5000,
                        help='number of iterations per segment for early stopping PSNR evaluation')
    parser.add_argument('--early_stop_psnr_threshold', type=float, default=0.01,
                        help='minimum PSNR improvement (in dB) between two consecutive segments; training stops if improvement is below this value')

    ### ---------- ASTC comparison configs ---------- ###
    parser.add_argument('--enable_astc_compare', action=argparse.BooleanOptionalAction, default=True,
                        help='enable ASTC comparison (official astcenc roundtrip)')
    parser.add_argument('--astcenc_path', type=str, default='tools/astcenc',
                        help='astcenc executable path or directory. If path does not exist, astcenc will be auto-downloaded there')
    parser.add_argument('--astcenc_quality', type=str, default='medium',
                        choices=['fastest', 'fast', 'medium', 'thorough', 'exhaustive'],
                        help='astcenc quality preset for ASTC comparison')
    parser.add_argument('--astc_block', type=str, default='6x6',
                        help='ASTC block size, e.g. 4x4 / 6x6 / 8x8')
    parser.add_argument('--ref_astc_resolution', type=int, default=1024,
                        help='Traditional ref_astc_* baseline: square edge length (H=W) before astcenc; omit for Mip0 size')

    ### ---------- PBR scene comparison configs ---------- ###
    parser.add_argument('--enable_pbr_compare', action=argparse.BooleanOptionalAction, default=False,
                        help='render inferred and ground-truth texture sets in a Cornell-box PBR scene')
    parser.add_argument('--pbr_compare_interval', type=int, default=5000,
                        help='training iteration interval for PBR scene comparison images')
    parser.add_argument('--pbr_render_resolution', type=int, default=512,
                        help='final cropped scene height of each PBR comparison panel')
    parser.add_argument('--pbr_mip_lod_bias', type=float, default=-1.0,
                        help='PBR texture mip LOD bias; negative values select sharper mips')
    parser.add_argument('--pbr_loss_enable', action=argparse.BooleanOptionalAction, default=False,
                        help='add differentiable material-to-PBR shaded RGB loss during training')
    parser.add_argument('--pbr_loss_weight', type=float, default=0.1,
                        help='weight of the differentiable PBR shaded RGB loss')
    parser.add_argument('--pbr_directional_light_enable', action=argparse.BooleanOptionalAction, default=True)
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
    yaml_defaults = _load_yaml_defaults(known.config)

    if yaml_defaults:
        # Filter to only keys that correspond to valid parser destinations
        valid_dests = {a.dest for a in parser._actions}
        filtered = {k: v for k, v in yaml_defaults.items() if k in valid_dests}
        parser.set_defaults(**filtered)

    # Stage 2: full parse (CLI args override YAML values)
    return parser.parse_args()
