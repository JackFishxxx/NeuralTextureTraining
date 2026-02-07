import os
import torch
import copy
import numpy as np
import tinycudann as tcnn
from torchtyping import TensorType

from configs import Config

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from utils import pack_features, unpack_features


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
        self.base_resolution = config.base_resolution

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
        self.hash_grid_feature_sizes = []  # per-grid n_features_per_level
        self.hash_grid_n_levels = []       # per-grid n_levels
        self.hash_grid_base_res = []       # per-grid base_resolution
        self.hash_grid_quantize_bits = []  # per-grid quantize bits (fallback to global)
        self.hash_grid_save_bits = []      # per-grid save bits (fallback to global)

        for grid_cfg in config.hash_grid_configs:
            # Derive base_resolution
            base_res = int(grid_cfg.get('base_resolution', config.base_resolution))

            # Derive n_levels (prefer explicit, else from max_resolution)
            if 'n_levels' in grid_cfg:
                n_lvls = int(grid_cfg['n_levels'])
            else:
                max_res = grid_cfg.get('max_resolution', None)
                if max_res is None:
                    # fallback to global when neither provided
                    n_lvls = int(config.n_levels)
                else:
                    # n_levels = floor(log2(max_res / base_res)) + 1
                    n_lvls = int(torch.floor(torch.log2(torch.tensor(max_res / base_res))) + 1)
                    n_lvls = max(1, n_lvls)

            # Derive n_features_per_level (prefer explicit, else from bits)
            if 'n_features_per_level' in grid_cfg:
                fpl = int(grid_cfg['n_features_per_level'])
            else:
                qbits = int(grid_cfg.get('quantize_bits', self.quantize_bits))
                sbits = int(grid_cfg.get('save_bits', self.save_bits))
                fpl = int(sbits // qbits)
                fpl = max(1, fpl)

            # Per-grid bits info (default to model-level if unspecified)
            qbits = int(grid_cfg.get('quantize_bits', self.quantize_bits))
            sbits = int(grid_cfg.get('save_bits', self.save_bits))

            hash_grid_config = {
                "otype": "Grid",
                "type": "Hash",
                "n_levels": int(n_lvls),
                "n_features_per_level": int(fpl),
                "base_resolution": int(base_res),
            }
            hash_grid = tcnn.Encoding(
                n_input_dims=2,
                encoding_config=hash_grid_config
            )
            self.hash_grids.append(hash_grid)
            self.hash_grid_feature_sizes.append(int(fpl))
            self.hash_grid_n_levels.append(int(n_lvls))
            self.hash_grid_base_res.append(int(base_res))
            self.hash_grid_quantize_bits.append(int(qbits))
            self.hash_grid_save_bits.append(int(sbits))

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
        total_grid_features = sum(self.hash_grid_feature_sizes) * self.num_sampled_lods
        n_input_dims = config.n_frequencies * 2 + total_grid_features + 1
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
            grid_fpl = self.hash_grid_feature_sizes[idx]

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
            info.append(int(self.hash_grid_feature_sizes[i]))
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
        save_path = os.path.join(model_path, f"{curr_iter}.pth")
        torch.save(self.state_dict(), save_path)

        # tcnn does not support torch.quantization?
        # TODO need to figure out the grid feature storage
        if self.quantize:

            # in tcnn the input dim would be padded to the nearest multiple of 16
            # https://github.com/NVlabs/tiny-cuda-nn/issues/6

            save_model.simulate_quantize()

            # process the range of weights and features
            # quant_model_path = os.path.join(self.model_path, f"{curr_iter}_quant.pt")
            quant_model_path = os.path.join(model_path, f"{curr_iter}_quant.npz")

            # pack each grid separately and save per-grid blobs + metadata
            packed_features_list = []
            for hash_grid in save_model.hash_grids:
                features = hash_grid.state_dict()["params"]
                packed_features = pack_features(features, quantize_bits=save_model.quantize_bits, save_bits=save_model.save_bits)
                packed_features_list.append(packed_features)
                # unpacked_features = unpack_features(packed_features, quantize_bits=save_model.quantize_bits, save_bits=save_model.save_bits)
                # print(torch.abs(unpacked_features - features).sum())

            # generate model infos
            info = self.get_model_info()

            # build simple per-grid metadata: [base_res, n_levels, n_features_per_level]
            grid_meta = np.array([
                [int(b), int(l), int(f)]
                for b, l, f in zip(self.hash_grid_base_res, self.hash_grid_n_levels, self.hash_grid_feature_sizes)
            ], dtype=np.int32)

            save_kwargs = {
                'info': info,
                'network': save_model.network.state_dict()["params"].numpy(),
                'grid_meta': grid_meta,
            }
            for i, pf in enumerate(packed_features_list):
                save_kwargs[f'hash_grid_{i}'] = pf.cpu().numpy()

            np.savez(quant_model_path, **save_kwargs)