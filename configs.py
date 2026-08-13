import argparse
import os
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml


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
    return range(1) if mip0_only else range(max(1, num_mips - 4))


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

        ### ---------- groups batching configs ---------- ###
        self.groups_batching = params.groups_batching
        self.groups_batch_dir = params.groups_batch_dir
        self.groups_save_dir = params.groups_save_dir
        self.groups_max_workers = params.groups_max_workers
        self.groups_verbose = params.groups_verbose
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
        self.quantize = params.quantize
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
        self.astc_aware_enable = bool(getattr(params, "astc_aware_enable", False))
        self.astc_aware_block = int(getattr(params, "astc_aware_block", 6))
        self.astc_aware_noise_scale = float(getattr(params, "astc_aware_noise_scale", 1.0))
        self.astc_aware_start_frac = float(getattr(params, "astc_aware_start_frac", 0.0))
        self.astc_codec_in_loop_enable = bool(getattr(params, "astc_codec_in_loop_enable", False))
        self.astc_codec_update_latent = bool(getattr(params, "astc_codec_update_latent", False))
        self.astc_codec_in_loop_interval = int(getattr(params, "astc_codec_in_loop_interval", 5000))
        self.astc_codec_loss_weight = float(getattr(params, "astc_codec_loss_weight", 0.7))
        self.astc_clean_loss_weight = float(getattr(params, "astc_clean_loss_weight", 0.3))
        self.astc_consistency_weight = float(getattr(params, "astc_consistency_weight", 0.1))
        self.astc_codec_mip0_prob = float(getattr(params, "astc_codec_mip0_prob", 0.5))
        self.astc_codec_start_frac = float(getattr(params, "astc_codec_start_frac", 0.0))
        if (self.astc_aware_block <= 0 or self.astc_aware_noise_scale < 0.0
                or not 0.0 <= self.astc_aware_start_frac <= 1.0
                or self.astc_codec_in_loop_interval <= 0
                or min(self.astc_codec_loss_weight, self.astc_clean_loss_weight,
                       self.astc_consistency_weight) < 0.0):
            raise ValueError("invalid ASTC-aware latent settings")
        if not 0.0 <= self.astc_codec_mip0_prob <= 1.0:
            raise ValueError("astc_codec_mip0_prob must be in [0, 1]")
        if not 0.0 <= self.astc_codec_start_frac <= 1.0:
            raise ValueError("astc_codec_start_frac must be in [0, 1]")
        total_astc_weight = self.astc_clean_loss_weight + self.astc_codec_loss_weight + self.astc_consistency_weight
        if total_astc_weight <= 0.0:
            raise ValueError("at least one ASTC-aware loss weight must be positive")
        self.astc_clean_loss_weight /= total_astc_weight
        self.astc_codec_loss_weight /= total_astc_weight
        self.astc_consistency_weight /= total_astc_weight

        ### ---------- trainer configs ---------- ###
        self.max_iter = params.max_iter
        self.batch_size = params.batch_size
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
        self.astc_block = params.astc_block
        self.ref_astc_resolution = getattr(params, "ref_astc_resolution", None)

        # Normalize configs to the internal format expected by the model
        if self.feature_grid_configs is not None:
            processed: List[Dict] = []
            for cfg in self.feature_grid_configs:
                max_res = int(cfg.get("max_resolution", 1024))
                # check max_res
                if max_res <= 0:
                    raise ValueError("max_resolution must be a positive integer")

                n_levels = int(cfg.get("n_levels", int(math.log2(max_res>>1))))
                #n_levels = int(cfg.get("n_levels", 1))
                # check n_levels
                if n_levels <= 0:
                    raise ValueError("n_levels must be a positive integer")

                qbits = int(cfg.get("quantize_bits", self.quantize_bits))
                sbits = int(cfg.get("save_bits", self.save_bits))
                # check quantize bits and save bits
                if sbits < qbits:
                    raise ValueError("The save bits should not be less than the quantize bits.")

                lr = cfg.get("learning_rate", self.learning_rate)

                processed.append({
                    "max_resolution": max_res,
                    "n_levels": n_levels,
                    "quantize_bits": qbits,
                    "save_bits": sbits,
                    "learning_rate": lr,
                    "interpolation": str(cfg.get("interpolation", "Linear")),
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
                        help='number of concurrent training workers for GroupsBatching (default 1 = sequential)')
    parser.add_argument('--groups_verbose', action='store_true', default=False,
                        help='print per-group training details instead of aggregate progress only')

    ### ---------- quantization configs ---------- ###
    parser.add_argument('--quantize', type=bool, default=True,
                        help='whether to quantize the model or not')
    parser.add_argument('--quantize_bits', type=int, default=8,
                        help='choose the bits to quantize',
                        choices=[2, 4, 8, 16])
    parser.add_argument('--save_bits', type=int, default=32,
                        help='choose the bits to quantize',
                        choices=[8, 16, 32, 64])
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
    parser.add_argument('--astc_aware_enable', action=argparse.BooleanOptionalAction, default=False,
                        help='inject block-correlated ASTC-like latent noise during training')
    parser.add_argument('--astc_aware_block', type=int, default=6,
                        help='ASTC-aware latent block edge in texels')
    parser.add_argument('--astc_aware_noise_scale', type=float, default=1.0,
                        help='latent ASTC perturbation in 8-bit code steps')
    parser.add_argument('--astc_aware_start_frac', type=float, default=0.0,
                        help='fraction of training held without ASTC-aware perturbation')
    parser.add_argument('--astc_codec_in_loop_enable', action=argparse.BooleanOptionalAction, default=False,
                        help='periodically calibrate ASTC-aware latent noise using real astcenc round-trips')
    parser.add_argument('--astc_codec_update_latent', action=argparse.BooleanOptionalAction, default=False,
                        help='use a straight-through codec gradient to update feature-grid latents (experimental)')
    parser.add_argument('--astc_codec_in_loop_interval', type=int, default=5000,
                        help='iterations between real ASTC latent calibration passes')
    parser.add_argument('--astc_codec_loss_weight', type=float, default=0.7,
                        help='reconstruction loss weight for the real-ASTC training branch')
    parser.add_argument('--astc_clean_loss_weight', type=float, default=0.3,
                        help='reconstruction loss weight for the clean QAT branch')
    parser.add_argument('--astc_consistency_weight', type=float, default=0.1,
                        help='output consistency weight between ASTC and clean branches')
    parser.add_argument('--astc_codec_mip0_prob', type=float, default=0.5,
                        help='probability of sampling Mip0 during ASTC codec-in-loop training')
    parser.add_argument('--astc_codec_start_frac', type=float, default=0.0,
                        help='fraction of training reserved for clean latent pretraining before codec loss')
    parser.add_argument(
        '--qat_noise_warmup_frac',
        type=float,
        default=0.1,
        help='fraction of (max_iter-1) iterations holding mult_start before cosine decay',
    )

    ### ---------- trainer configs ---------- ###
    parser.add_argument('--max_iter', type=int, default=400000,
                        help='maximum training iteration')
    parser.add_argument('--batch_size', type=int, default=16384,
                        help='batch size')
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
                        help='enable early stopping when PSNR improvement is below threshold')
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
