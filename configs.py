import argparse
import os
import math
from typing import Dict, List, Optional

import torch

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
        self.noise_std = params.noise_std
        self.noise_anneal_fraction = params.noise_anneal_fraction

        ### ---------- trainer configs ---------- ###
        self.max_iter = params.max_iter
        self.batch_size = params.batch_size
        self.learning_rate = params.learning_rate
        self.momentum = params.momentum
        self.weight_decay = params.weight_decay
        self.lr_scheduler = params.lr_scheduler

        self.eval_interval = params.eval_interval
        self.save_interval = params.save_interval

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
        self.num_lods = 1
        self.n_frequencies = 0
        self.n_neurons = 16
        self.n_hidden_layers = 0

        # By default, None → use legacy behavior (uniform grids based on n_features_per_level).
        self.hash_grid_configs: Optional[List[Dict]] = getattr(params, 'hash_grid_configs', None)
        # Multi-HashGrid configs
        self.hash_grid_configs = [
            {"max_resolution": 1024, "quantize_bits":8, "save_bits":32, "learning_rate": 0.005},
            {"max_resolution": 512, "quantize_bits":8, "save_bits":32, "learning_rate": 0.005},
        ]
        
        # Per-texture-type loss weights (optional)
        # If None, weights from dataset.get_texture_config() will be used
        # Keys: texture type names (diffuse, normal, roughness, etc.)
        # Values: per-channel weight
        self.texture_loss_weights: Optional[Dict[str, float]] = {
            "diffuse": 1.0,
            "normal": 0.8,
            "roughness": 0.3,
            "occlusion": 0.3,
            "metallic": 0.3,
            "specular": 0.3,
            "displacement": 0.3,
        }
        
        # Final per-channel loss weights list (generated from texture_loss_weights and available textures)
        self.output_loss_weights: Optional[List[float]] = None
        self.network_learning_rate = params.learning_rate

        # Normalize configs to the internal format expected by the model
        if self.hash_grid_configs is not None:
            processed: List[Dict] = []
            for cfg in self.hash_grid_configs:
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
                })

            self.hash_grid_configs = processed

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
        cfg.num_lods = 1
        cfg.output_loss_weights = None
        return cfg

def get_args():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')
    parser.add_argument('--noise_anneal_fraction', type=float, default=0.8,
                        help='fraction of training over which noise is annealed from full to zero (0.0 = no annealing, 1.0 = anneal over entire training)')
        
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
                        help='the interval of iteration for evalation')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='the interval of iteration for saving model')

    ### ---------- early stopping configs ---------- ###
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help='enable early stopping when PSNR improvement is below threshold')
    parser.add_argument('--early_stop_interval', type=int, default=5000,
                        help='number of iterations per segment for early stopping PSNR evaluation')
    parser.add_argument('--early_stop_psnr_threshold', type=float, default=0.01,
                        help='minimum PSNR improvement (in dB) between two consecutive segments; training stops if improvement is below this value')

    return parser.parse_args()
