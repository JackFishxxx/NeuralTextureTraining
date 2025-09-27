import os
import torch
import copy
import numpy as np
import tinycudann as tcnn
from torchtyping import TensorType

from configs import Config
from positional_encoding import TriangularWave
from features import HashGrid
from network import FullyFusedMLP

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
        self.n_levels = config.n_levels
        self.n_features_per_level = config.n_features_per_level
        self.base_resolution = config.base_resolution

        self.n_frequencies = config.n_frequencies
        self.n_neurons = config.n_neurons
        self.n_hidden_layers = config.n_hidden_layers
        self.num_channels = config.num_channels

        self.init_model(config)
        if config.load_iter != 0:
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

        if config.n_features_per_level % 8 != 0:
            raise ValueError("The n_features_per_level should be 1, 2, 4, or multiple of 8.") 
        
        self.num_hash_grids = 1
        if config.n_features_per_level > 8:
            self.num_hash_grids = config.n_features_per_level // 8 

        self.hash_grids = torch.nn.ModuleList()
        for i in range(self.num_hash_grids):
            # split the hash grid into multiple grids
            # e.g. n_features_per_level = 16, then we have 2 hash grids
            # e.g. n_features_per_level = 32, then we have 4 hash grids

            # if config.n_features_per_level > 8 we set the n_features_per_level to 8
            self.hash_grid_features_per_level = min(config.n_features_per_level, 8) 
            hash_grid_config = {
                "otype": "Grid",
                "type": "Hash",
                "n_levels": config.n_levels,
                "n_features_per_level": self.hash_grid_features_per_level,
                #"per_level_scale": 1.0,
                "base_resolution": config.base_resolution,
            }

            hash_grid = tcnn.Encoding(
                n_input_dims=2,
                encoding_config=hash_grid_config
            )
            self.hash_grids.append(hash_grid)

        # ReLU: converg slowly
        # LeakyReLU: PSNR: 28.85, LPIPS: 0.2205(last time was 0.19)
        # Squareplus: PSNR: 28.7937, LPIPS: 0.1768
        # Softplus: PSNR: 28.8342, LPIPS: 0.1823
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "LeakyReLU",
            "output_activation": "None",
            "n_neurons": config.n_neurons,
            "n_hidden_layers": config.n_hidden_layers
        }
        n_input_dims = config.n_frequencies * 2 + config.n_features_per_level + 1
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
        if config.load_iter == -1:
            ckpt_iter = 0

        ckpt_path = os.path.join(config.load_dir, "models", f"{ckpt_iter}_quant.npz")
        ckpt_params = np.load(ckpt_path, allow_pickle=True).item()

        state_dict = self.state_dict()
        for key, value in ckpt_params.items():
            
            device = state_dict[f"{key}.params"].device
            module_param = torch.from_numpy(value["params"]).to(device)

            if "hash_grid_1" in key:
                module_param = unpack_features(module_param, quantize_bits=self.quantize_bits, save_bits=self.save_bits)
            if "hash_grid_2" in key:
                module_param = unpack_features(module_param, quantize_bits=self.quantize_bits, save_bits=self.save_bits)

            state_dict[f"{key}.params"] = module_param

        self.load_state_dict(state_dict)
   
    def forward(self, x: TensorType["batch_size", 3]) -> TensorType["batch_size", "num_channels"]:

        [batch_size, _] = x.shape
        uvs = x[:, 0:2]
        lod_encodings = x[:, [2]]

        # get required columns by lods
        # TODO check if num_sampled_lods is currect
        num_sampled_lods = 1
        mips = lod_encodings * (self.num_lods - num_sampled_lods)
        clipped_mips = torch.clamp(mips, max=self.n_levels - num_sampled_lods)
        cols = (self.n_levels - num_sampled_lods - clipped_mips) * self.hash_grid_features_per_level + torch.arange(self.hash_grid_features_per_level * num_sampled_lods).to(self.device)

        positional_encodings = self.triangle_wave(uvs)
        # positional_encodings = self.triangle_wave(xys)  

        # get the results from hash grid
        features = []
        for hash_grid in self.hash_grids:
            all_features = hash_grid(uvs)
            sampled_features = torch.gather(all_features, num_sampled_lods, cols.to(torch.int64))

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
        info.append(int(pow(2, self.num_lods - 1)))  # image size : e.g. 2**(11 - 1) = 1024
        info.append(int(self.base_resolution * pow(2, self.n_levels - 1)))  # features size : e.g. 4 * 2 ** (7-1) = 256

        info.append(int(self.quantize_bits))  # quantize bits : e.g. 4
        info.append(int(self.save_bits))  # save bits : e.g. 32

        info.append(int(self.base_resolution))  # base resolution : e.g. 4
        info.append(int(self.n_levels))  # base resolution : e.g. 7
        info.append(int(self.num_lods))  # max mip value : e.g. 10

        info.append(int(self.n_features_per_level))  # level
        info.append(int(self.n_frequencies))  # seq len : e.g. 6
        info.append(int(self.n_neurons))  # hidden size : e.g. 16
        info.append(int(self.n_hidden_layers))  # num hidden layes : e.g. 1
        info.append(int((self.num_channels // 16 + 1) * 16))  # output size : e.g. align16(9) -> 16

        info = np.array(info)    
        
        return info

    @torch.no_grad()    
    def save(self, curr_iter: int, model_path: str) -> None:
        
        save_model = copy.deepcopy(self).cpu()
        save_path = os.path.join(model_path, f"{curr_iter}.pth")
        torch.save(save_model, save_path)

        # tcnn does not support torch.quantization?
        # TODO need to figure out the grid feature storage
        if self.quantize:

            # in tcnn the input dim would be padded to the nearest multiple of 16
            # https://github.com/NVlabs/tiny-cuda-nn/issues/6

            save_model.simulate_quantize()

            # process the range of weights and features
            # quant_model_path = os.path.join(self.model_path, f"{curr_iter}_quant.pt")
            quant_model_path = os.path.join(model_path, f"{curr_iter}_quant.npz")

            packed_features_list = []
            for hash_grid in save_model.hash_grids:
                features = hash_grid.state_dict()["params"]
                packed_features = pack_features(features, quantize_bits=save_model.quantize_bits, save_bits=save_model.save_bits)
                packed_features_list.append(packed_features)
                # unpacked_features = unpack_features(packed_features, quantize_bits=save_model.quantize_bits, save_bits=save_model.save_bits)
                # print(torch.abs(unpacked_features - features).sum())

            # generate model infos
            info = self.get_model_info()

            offset = 0
            all_packed_features = []
            for level in range(self.n_levels):
                feature_size = self.base_resolution * 2 ** level
                cur_level_features_list = []

                for packed_features in packed_features_list:
                    # [feature_size * feature_size, 1] -> [feature_size, feature_size, 1]
                    cur_level_features = packed_features[offset: offset + feature_size ** 2].reshape([feature_size, feature_size, 1])  
                    cur_level_features_list.append(cur_level_features)
                cur_level_features = torch.cat(cur_level_features_list, dim=2)  # [feature_size, feature_size, num_hash_grids]

                all_packed_features.append(cur_level_features.flatten())
                offset += feature_size ** 2
            
            all_packed_features = torch.cat(all_packed_features, dim=0)  # [feature_size * feature_size * num_hash_grids, 1]

            # torch.save(quant_model, quant_model_path)  # torch does not support saving uint16 tensor
            # TODO change to a description file and a npy data file
            np.savez(quant_model_path, info=info, network=save_model.network.state_dict()["params"].numpy(), hash_grid=all_packed_features.cpu().numpy())

class TCNNSplitModel(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.device = config.device
        # self.model_config = self.generate_model_config(config)
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
        self.n_levels = config.n_levels
        self.n_features_per_level = config.n_features_per_level
        self.base_resolution = config.base_resolution

        self.max_feature_level = 3

        self.init_model(config)
        if config.load_iter != 0:
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

        self.hash_grids = []

        max_feature_res = 256
        for feature_level in range(self.max_feature_level):
            
            hash_grid_config = {
                "otype": "Grid",
                "type": "Hash",
                "n_levels": 2,
                "n_features_per_level": config.n_features_per_level,
                "base_resolution": (max_feature_res // 2) // (4 ** feature_level)
            }
            hash_grid = tcnn.Encoding(
                n_input_dims=2,
                encoding_config=hash_grid_config
            )
            self.hash_grids.append(hash_grid)

        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "SquarePlus",
            "output_activation": "None",
            "n_neurons": config.n_neurons,
            "n_hidden_layers": config.n_hidden_layers
        }
        n_input_dims = config.n_frequencies * 2 + 2 * config.n_features_per_level + 1
        self.network = tcnn.Network(
            n_input_dims=n_input_dims,
            n_output_dims=config.num_channels,
            network_config=network_config,
        )

        # add optimizer params 
        optim_params = [{'params': self.network.parameters(), 'lr': 0.005}]
        for feature_level in range(self.max_feature_level):
            optim_params.append({'params': self.hash_grids[feature_level].parameters(), 'lr': 0.005})
        self.optimizer = torch.optim.Adam(optim_params)

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.85, patience=2000)

        # level of mipmap to level of feature
        # M0, M1, M2, M3 -> F0
        # M4, M5, M6 -> F1
        # M7, M8, M9, M10 -> F2
        self.mip2feature = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]).to(self.device)
    
    def load_ckpt(self, config: Config) -> None:
        
        # TODO
        ckpt_iter = config.load_iter
        if config.load_iter == -1:
            ckpt_iter = 0

        ckpt_path = os.path.join(config.load_dir, "models", f"{ckpt_iter}_quant.npz")
        ckpt_params = np.load(ckpt_path, allow_pickle=True)

        state_dict = self.state_dict()
        for key, value in ckpt_params.items():

            if "hash_grid" in key:
                continue
            module_param = torch.from_numpy(value).to(self.device)
            state_dict[f"{key}.params"] = module_param

        self.load_state_dict(state_dict)
        
        for hash_grid in self.hash_grids:
            state_dict = hash_grid.state_dict()
            module_param = unpack_features(module_param, quantize_bits=self.quantize_bits, save_bits=self.save_bits)
            state_dict["params"] = module_param
            hash_grid.load_state_dict(state_dict)
          
    def forward(self, x: TensorType["batch_size", 3]) -> TensorType["batch_size", "num_channels"]:

        [batch_size, _] = x.shape
        uvs = x[:, 0:2]
        lods = x[:, [2]]

        # forward
        positional_encodings = self.triangle_wave(uvs)
        
        n_input_dims = self.n_features_per_level * 2
        features = torch.zeros([batch_size, n_input_dims], dtype=torch.float16).to(self.device)
        for feature_level in range(self.max_feature_level):
            current_level_mask = (lods == feature_level).squeeze().to(self.device)
            current_level_uvs = uvs[current_level_mask]
            current_level_features = self.hash_grids[feature_level](current_level_uvs)
            features[current_level_mask] = current_level_features

        # get the results from hash grid

        lod_encodings = lods

        if self.quantize:
            # add symmetric noise
            noise = 2 * self.noise_range * torch.rand(1).to(self.device) - self.noise_range
            features = features + noise

        inputs = torch.cat([positional_encodings, features, lod_encodings], dim=1)
        # inputs = torch.cat([features, lod_encodings], dim=1)

        outputs = self.network(inputs)

        return outputs
    
    def simulate_quantize(self):

        # only quantize the features
        for feature_level in range(self.max_feature_level):
            state_dict = self.hash_grids[feature_level].state_dict()

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
            self.hash_grids[feature_level].load_state_dict(state_dict)
    
    def clamp_value(self):
        for feature_level in range(self.max_feature_level):
            state_dict = self.hash_grids[feature_level].state_dict()
            state_dict['params'] = torch.clamp(state_dict['params'], min=self.min_quantize_range, max=self.max_quantize_range)
            self.hash_grids[feature_level].load_state_dict(state_dict)

    @torch.no_grad()    
    def save(self, curr_iter: int, model_path: str) -> None:
        
        save_model = copy.deepcopy(self).cpu()
        save_path = os.path.join(model_path, f"{curr_iter}.pth")
        torch.save(save_model, save_path)

        # tcnn does not support torch.quantization?
        # TODO need to figure out the grid feature storage
        if self.quantize:

            # in tcnn the input dim would be padded to the nearest multiple of 16
            # https://github.com/NVlabs/tiny-cuda-nn/issues/6

            save_model.simulate_quantize()

            # process the range of weights and features
            # quant_model_path = os.path.join(self.model_path, f"{curr_iter}_quant.pt")
            quant_model_path = os.path.join(model_path, f"{curr_iter}_quant.npz")

            # triangle_wave = save_model.triangle_wave.state_dict()["params"].numpy()
            network = save_model.network.state_dict()["params"].numpy()
            hash_grids = []
            for feature_level in range(4):

                if feature_level < self.max_feature_level:

                    features = save_model.hash_grids[feature_level].state_dict()["params"]
                    packed_features = pack_features(features, quantize_bits=save_model.quantize_bits, save_bits=save_model.save_bits)
                    # unpacked_features = unpack_features(packed_features, quantize_bits=save_model.quantize_bits, save_bits=save_model.save_bits)
                    # print(torch.abs(unpacked_features - features).sum())
                    hash_grids.append(packed_features.cpu().numpy()) 
                
                else:
                    hash_grids.append(np.zeros([1]))

            # torch.save(quant_model, quant_model_path)  # torch does not support saving uint16 tensor
            # TODO change to a description file and a npy data file
            np.savez(quant_model_path, network=network, hash_grid_0=hash_grids[0], hash_grid_1=hash_grids[1], hash_grid_2=hash_grids[2], hash_grid_3=hash_grids[3])

class TorchModel(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.device = config.device
        # self.model_config = self.generate_model_config(config)
        self.quantize = config.quantize
        self.quantize_bits = config.quantize_bits
        self.save_bits = config.save_bits
        
        self.N_k = 2 ** self.quantize_bits
        self.Q_k = 1 / self.N_k
        self.min_quantize_range = - (self.N_k - 1) / 2 * self.Q_k
        # self.N_k / 2 * self.Q_k = 1 / 2
        self.max_quantize_range = 1 / 2
        self.noise_range = 1 / 2 * self.Q_k

        self.init_model(config)
        if config.load_iter != 0:
            self.load_ckpt(config)

    def init_model(self, config: Config) -> None:

        self.triangle_wave = TriangularWave(device=config.device)
        self.hash_grid = HashGrid(device=config.device)
        self.network = FullyFusedMLP(input_dim=77, output_dim=9, device=config.device)
        
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "Squareplus",
            "output_activation": "None",
            "n_neurons": config.n_neurons,
            "n_hidden_layers": config.n_hidden_layers
        }
        n_input_dims = config.n_frequencies * 2 + config.n_levels * config.n_features_per_level * 1 + 1
        # n_input_dims = config.n_frequencies * 2 + 1 * 1 * 1 + 1
        self.network = tcnn.Network(
            n_input_dims=n_input_dims,
            n_output_dims=config.num_channels,
            network_config=network_config,
        )
    
    def load_ckpt(self, config: Config) -> None:
        
        # TODO
        ckpt_iter = config.load_iter
        if config.load_iter == -1:
            ckpt_iter = 0

        ckpt_path = os.path.join(config.load_dir, "models", f"{ckpt_iter}_quant.npy")
        ckpt_params = np.load(ckpt_path, allow_pickle=True).item()

        state_dict = self.state_dict()
        device = state_dict["network.params"].device
        for key, value in ckpt_params.items():
            
            module_param = torch.from_numpy(value["params"]).to(device)

            if "triangle_wave" in key:
                continue

            if "hash_grid" in key:
                module_param = unpack_features(module_param, quantize_bits=self.quantize_bits, save_bits=self.save_bits)
                
                # divide into split feature grid
                # for

            

            state_dict[f"{key}.params"] = module_param

        self.load_state_dict(state_dict)

    
    def forward(self, x: TensorType["batch_size", 3]) -> TensorType["batch_size", "num_channels"]:

        [batch_size, _] = x.shape
        uvs = x[:, 0:2]
        lods = x[:, [2]]

        # forward
        positional_encodings = self.triangle_wave(uvs)
        # positional_encodings = self.triangle_wave(xys)  # slightly slower than tcnn
        features = self.hash_grid(uvs)
        lod_encodings = lods

        if self.quantize:
            # add symmetric noise
            noise = 2 * self.noise_range * torch.rand(1).to(self.device) - self.noise_range
            features = features + noise

        inputs = torch.cat([positional_encodings, features, lod_encodings], dim=1)
        # inputs = torch.cat([features, lod_encodings], dim=1)

        outputs = self.network(inputs)

        return outputs