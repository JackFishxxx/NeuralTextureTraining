import os
import math
import torch
import copy
import numpy as np
import tinycudann as tcnn
from torchtyping import TensorType

from configs import Config

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


from utils import pack_features, unpack_features, write_dds_r8g8b8a8

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

        self.num_lods = config.num_lods
        self.n_frequencies = config.n_frequencies
        self.n_neurons = config.n_neurons
        self.n_hidden_layers = config.n_hidden_layers
        self.num_channels = config.num_channels

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

        # slightly slower than tcnn
        # self.triangle_wave = TriangularPositionalEncoding2D(device=config.device)
        
        # how many lods to sample per query
        self.num_sampled_lods = 1

        # support heterogeneous hash grid configs from configs.py
        # if not provided, fall back to legacy behavior
        self.hash_grids = torch.nn.ModuleList()
        self.hash_grid_base_res = []       # per-grid base_resolution
        self.hash_grid_n_levels = []       # per-grid n_levels
        self.hash_grid_n_features_per_level = []  # per-grid n_features_per_level
        self.hash_grid_quantize_bits = []  # per-grid quantize bits (fallback to global)
        self.hash_grid_save_bits = []      # per-grid save bits (fallback to global)

        for grid_cfg in config.hash_grid_configs:
            max_res = int(grid_cfg.get('max_resolution', 0))
            n_levels = int(grid_cfg.get("n_levels", 0))
            qbits = int(grid_cfg.get('quantize_bits', self.quantize_bits))
            sbits = int(grid_cfg.get('save_bits', self.save_bits))
            n_feature_per_level = sbits // qbits

            # Derive base_resolution
            base_res = int(max_res >> (n_levels - 1))
            if base_res == 0:
                print(f"Warning: max_resolution {max_res} is too small for {n_levels} levels with scale 2. Clamping base_res to 1.")
                base_res = 1

            log2_hashmap_size = int(math.log2(max_res)) * 2

            hash_grid_config = {
                "otype": "Grid",
                "type": "Dense",
                "n_levels": n_levels,
                "n_features_per_level": n_feature_per_level,
                "base_resolution": base_res,
                "per_level_scale": 2.0,
                # "log2_hashmap_size": log2_hashmap_size,
                "interpolation": "Linear",
            }
            hash_grid = tcnn.Encoding(
                n_input_dims=2,
                encoding_config=hash_grid_config
            )
            self.hash_grids.append(hash_grid)
            self.hash_grid_base_res.append(int(base_res))
            self.hash_grid_n_levels.append(int(n_levels))
            self.hash_grid_n_features_per_level.append(int(n_feature_per_level))
            self.hash_grid_quantize_bits.append(int(qbits))
            self.hash_grid_save_bits.append(int(sbits))

            print(f"Initialized hash grid: max_res={max_res}, base_res={base_res}, n_levels={n_levels}, n_features_per_level={n_feature_per_level}, quantize_bits={qbits}, save_bits={sbits}")

        # ReLU: converg slowly
        # LeakyReLU: PSNR: 28.85, LPIPS: 0.2205(last time was 0.19)
        # Squareplus: PSNR: 28.7937, LPIPS: 0.1768
        # Softplus: PSNR: 28.8342, LPIPS: 0.1823
        network_config = {
            #"otype": "FullyFusedMLP",
            "otype": "CutlassMLP",
            "activation": "LeakyReLU",
            "output_activation": "LeakyReLU",
            "n_neurons": config.n_neurons,
            "n_hidden_layers": config.n_hidden_layers
        }
        total_grid_features = sum(self.hash_grid_n_features_per_level) * self.num_sampled_lods
        n_input_dims = config.n_frequencies * 2 + total_grid_features + 1
        print(f"total_grid_features={total_grid_features}, n_input_dims={n_input_dims}")
        self.network = tcnn.Network(
            n_input_dims=n_input_dims,
            n_output_dims=config.num_channels,
            network_config=network_config,
        )

        # add optimizer params config
        optimizer_params = [{'params': self.network.parameters(), 'lr': 0.002}]
        for hash_grid in self.hash_grids:
            optimizer_params.append({'params': hash_grid.parameters(), 'lr': 0.005})
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
   
    def forward(self, x: TensorType["batch_size", 3]) -> TensorType["batch_size", "num_channels"]:

        [batch_size, _] = x.shape
        uvs = x[:, 0:2]
        lod_encodings = x[:, [2]]

        # get required columns by lods
        # TODO check if num_sampled_lods is currect
        num_sampled_lods = self.num_sampled_lods
        mips = lod_encodings * (self.num_lods - num_sampled_lods)

        positional_encodings = self.triangle_wave(uvs)
        # positional_encodings = self.triangle_wave(xys)  

        # get the results from hash grid
        features = []
        for idx, hash_grid in enumerate(self.hash_grids):
            grid_levels = self.hash_grid_n_levels[idx]
            grid_fpl = self.hash_grid_n_features_per_level[idx]

            all_features = hash_grid(uvs)  # [B, grid_levels * grid_fpl]
            clipped_mips = torch.clamp(mips, max=grid_levels - num_sampled_lods)
            cols = (grid_levels - num_sampled_lods - clipped_mips) * grid_fpl + torch.arange(grid_fpl * num_sampled_lods).to(self.device)
            sampled_features = torch.gather(all_features, 1, cols.to(torch.int64))

            # TODO Check if we should add same noise to all features
            if self.quantize:
                # add symmetric noise
                noise = 2 * self.noise_range * torch.rand(1).to(self.device) - self.noise_range
                sampled_features = sampled_features + noise
            
            features.append(sampled_features)
        features = torch.cat(features, dim=1)
        
        inputs = torch.cat([positional_encodings, features, lod_encodings], dim=1)

        outputs = self.network(inputs)

        return outputs
    
    def simulate_quantize(self):
        # only quantize the features
        for hash_grid in self.hash_grids:
            state_dict = hash_grid.state_dict()

            # transfer the params to 4 bit quantized tensor
            # [min_quantize_range, max_quantize_range] -> [0, 2 ** self.quantize_bits - 1]
            params = state_dict['params']
            params = (params + (-self.min_quantize_range)) * (2 ** self.quantize_bits)
            params = torch.round(params)
            # torch.round would round params to an even number
            # so we need to clamp value
            params = torch.clamp(params, min=0., max=2 ** self.quantize_bits - 1)
            params = params / (2 ** self.quantize_bits) + self.min_quantize_range
            state_dict['params'] = params
            hash_grid.load_state_dict(state_dict)
    
    def clamp_value(self):

        for hash_grid in self.hash_grids:
            state_dict = hash_grid.state_dict()
            state_dict['params'] = torch.clamp(state_dict['params'], min=self.min_quantize_range, max=self.max_quantize_range)
            hash_grid.load_state_dict(state_dict)

    def get_model_info(self) -> np.array:
        
        info = []
        # include number of hash grids
        num_grids = len(self.hash_grids)
        info.append(int(num_grids))
        # for each grid: base_resolution, n_levels, n_features_per_level, quantize_bits, save_bits
        for i in range(num_grids):
            info.append(int(self.hash_grid_base_res[i]))
            info.append(int(self.hash_grid_n_levels[i]))
            info.append(int(self.hash_grid_n_features_per_level[i]))
            info.append(int(self.hash_grid_quantize_bits[i]))
            info.append(int(self.hash_grid_save_bits[i]))

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

        # tcnn does not support torch.quantization?
        # TODO need to figure out the grid feature storage
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

            # collect per-grid quantized integer features (one entry per feature)
            # Only grids with quantize_bits==8 and save_bits==32 are supported for DDS export
            quant_ints_list = []
            matching_indices = []
            for i, hash_grid in enumerate(save_model.hash_grids):
                qbits = int(save_model.hash_grid_quantize_bits[i])
                sbits = int(save_model.hash_grid_save_bits[i])
                features = hash_grid.state_dict()["params"]
                # map float quantized features back to integer range [0, 2^qbits - 1]
                N_k = 2 ** qbits
                min_q = - (N_k - 1) / 2 * (1.0 / N_k)
                ints = torch.round((features + (-min_q)) * (2 ** qbits))
                ints = torch.clamp(ints, min=0., max=2 ** qbits - 1).to(torch.int64)
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
                base_res = int(self.hash_grid_base_res[orig_i])
                n_levels = int(self.hash_grid_n_levels[orig_i])
                n_fpl = int(self.hash_grid_n_features_per_level[orig_i])

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
                dds_path = os.path.join(save_path, f"hash_grid_{orig_i}.dds")
                write_dds_r8g8b8a8(dds_path, max_res_size, max_res_size, rgba)
            