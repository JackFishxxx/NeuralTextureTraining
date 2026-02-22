from torch.utils.data import Dataset
from configs import Config
from typing import Dict, List, Optional
from torchtyping import TensorType

import torch
import os
import math
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


def get_texture_config() -> List[Dict]:

    keyword_order = ["diffuse", "normal", "roughness", "occlusion", "metallic", "specular", "displacement"]
    texture_keywords = {
        keyword_order[0]: ["diffuse", "albedo", "color", "diff"],
        keyword_order[1]: ["normal", "nor_gl"],
        keyword_order[2]: ["roughness", "rough"],
        keyword_order[3]: ["occlusion", "ao", "ambient"],
        keyword_order[4]: ["metallic", "metalness"],
        keyword_order[5]: ["specular"],
        keyword_order[6]: ["displacement", "disp"],
        # add other texture types and their possible keywords if needed
    }

    # vis_mode: "srgb" = apply gamma for display, "normal" = re-normalize vectors, "linear" = save as-is
    # loss_weight: per-channel loss weight used during training
    # display_name: subfolder name used when saving evaluation images
    texture_configs = {
        keyword_order[0]: {"expected_channels": 3, "color_mode": "RGB", "vis_mode": "srgb",   "loss_weight": 1.0, "display_name": "rgb"},
        keyword_order[1]: {"expected_channels": 3, "color_mode": "RGB", "vis_mode": "normal", "loss_weight": 0.8, "display_name": "normal"},
        keyword_order[2]: {"expected_channels": 1, "color_mode": "L",   "vis_mode": "linear", "loss_weight": 0.3, "display_name": "roughness"},
        keyword_order[3]: {"expected_channels": 1, "color_mode": "L",   "vis_mode": "linear", "loss_weight": 0.3, "display_name": "occlusion"},
        keyword_order[4]: {"expected_channels": 1, "color_mode": "L",   "vis_mode": "linear", "loss_weight": 0.3, "display_name": "metallic"},
        keyword_order[5]: {"expected_channels": 1, "color_mode": "L",   "vis_mode": "linear", "loss_weight": 0.3, "display_name": "specular"},
        keyword_order[6]: {"expected_channels": 1, "color_mode": "L",   "vis_mode": "linear", "loss_weight": 0.3, "display_name": "displacement"},
        # add other texture types if needed
    }

    return keyword_order, texture_keywords, texture_configs


def get_canonical_num_channels() -> int:
    """Fixed total number of channels (11) aligned with model.CANONICAL_CHANNEL_ORDER."""
    keyword_order, _, texture_configs = get_texture_config()
    return sum(texture_configs[t]["expected_channels"] for t in keyword_order)


def get_canonical_channel_slices_static() -> Dict[str, tuple]:
    """Return (start, end) slice in the 11-channel layout for each texture type (aligned with model)."""
    keyword_order, _, texture_configs = get_texture_config()
    out = {}
    idx = 0
    for t in keyword_order:
        n = texture_configs[t]["expected_channels"]
        out[t] = (idx, idx + n)
        idx += n
    return out


class TextureDataset(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        self.device = config.device
        
        self.data_dir = config.data_dir

        self.keyword_order, self.texture_keywords, self.texture_configs = get_texture_config()
        # mapping from texture type to channel slice [start, end) in current loaded data
        self.channel_slices = {}
        # (start, end) in the canonical 11-channel layout for each texture type, aligned with model
        self.canonical_channel_slices = get_canonical_channel_slices_static()
        self.canonical_num_channels = get_canonical_num_channels()
        # ordered available texture types that were found and concatenated
        self.available_textures = []

        self.textures = self.load_data()
        self.texture_height, self.texture_width, self.num_channels = self.textures.shape
        
        self.num_lods = int(min(math.log2(self.texture_height), math.log2(self.texture_width))) + 1

        self.lod_cache = self.generate_lod()
    
    @torch.no_grad()
    def forward(
            self, 
            batch_index: TensorType["batch_size", 3]
        ) -> List[TensorType["batch_size", "num_channels"]]:

        # lod_cache: [self.num_lods, self.texture_height, self.texture_width, self.num_channels]
        # batch_index: [batch_size, 3]

        batch_size = batch_index.shape[0]
        ys = batch_index[:, 0]
        xs = batch_index[:, 1]
        lods = batch_index[:, 2]

        # use lods to scale the pixel position
        lod_scale = 2 ** lods
        scaled_xs = xs // lod_scale
        scaled_ys = ys // lod_scale

        batch_data = self.lod_cache[lods, scaled_ys, scaled_xs, :]

        return batch_data
        

    def load_data(self) -> TensorType["texture_height", "texture_width", "num_channels"]:

        filenames = os.listdir(self.data_dir)
        textures = {}
        for filename in filenames:

            if not filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                continue
            
            filepath = os.path.join(self.data_dir, filename)
            texture_type = self.identify_texture_type(filename)

            print(f"Input Texture: type='{texture_type}' , filename='{filename}'")

            if texture_type not in self.texture_configs:
                raise ValueError(f"Unknown texture type: {texture_type}")
            color_mode = self.texture_configs[texture_type]['color_mode']
            expected_channels = self.texture_configs[texture_type]['expected_channels']

            with Image.open(filepath) as image:
                
                image = image.convert(color_mode)
                tensor = TF.to_tensor(image)   # [C, H, W]
                # tensor = torch.from_numpy(np.array(image))
                # if len(tensor.shape) == 2:
                #     tensor = tensor.unsqueeze(2)
                # tensor = tensor.permute(2, 0, 1)

                # change srgb to linear to fit ue5 when image is not Normal texture.
                if texture_type != "normal":
                    tensor = torch.pow(tensor, 2.2)

                if tensor.shape[0] != expected_channels:
                    raise ValueError(f"Expected {expected_channels} channels, got {tensor.shape[0]}")

            textures[texture_type] = tensor
        
        # Determine target resolution from the first available texture
        # and resize any mismatched textures to ensure consistent spatial dimensions
        target_h, target_w = None, None
        for texture_type in self.keyword_order:
            if texture_type in textures:
                _, h, w = textures[texture_type].shape
                if target_h is None:
                    target_h, target_w = h, w
                elif h != target_h or w != target_w:
                    print(f"Warning: Resizing '{texture_type}' from {h}x{w} to {target_h}x{target_w} "
                          f"to match other textures.")
                    textures[texture_type] = TF.resize(
                        textures[texture_type], [target_h, target_w],
                        interpolation=TF.InterpolationMode.BICUBIC,
                        antialias=True,
                    )
                    textures[texture_type] = torch.clamp(textures[texture_type], 0.0, 1.0)

        textures_ordered = []
        current_index = 0
        self.channel_slices = {}
        self.available_textures = []
        for texture_type in self.keyword_order:
            if texture_type in textures and textures[texture_type] is not None:
                tex = textures[texture_type]
                textures_ordered.append(tex)
                start = current_index
                end = current_index + tex.shape[0]
                self.channel_slices[texture_type] = (start, end)
                self.available_textures.append(texture_type)
                current_index = end
                

        textures_ordered = torch.cat(textures_ordered, dim=0).permute(1, 2, 0).to(self.device)  # [C, H, W] -> [H, W, C]

        return textures_ordered
    
    def generate_lod(self) -> TensorType["num_lods", "lod_height", "lod_width", "num_channels"]:

        lod_cache = torch.zeros(
            [self.num_lods, self.texture_height, self.texture_width, self.num_channels]
        )
        # here is a bug in pytorch while using a tensor on cuda to interpolate
        # from a large size to a small size, e.g. [1024, 1024] -> [8, 8]
        # the bug was not fixed since 2023
        # work on cpu seems not to have this problem
        textures = self.textures.cpu()

        for lod in range(self.num_lods):
            lod_height = self.texture_height // (2 ** lod)
            lod_width = self.texture_width // (2 ** lod)
            # [H, W, C] -> [C, H, W] -> TF.resize -> [C, lod_H, lod_W] -> [lod_H, lod_W, C]
            lod_texture = TF.resize(
                textures.permute(2, 0, 1), [lod_height, lod_width],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True,
            ).permute(1, 2, 0)
            lod_texture = torch.clamp(lod_texture, min=0., max=1.)
            # [lod, H, W, C] <- [lod_H, lod_W, C]
            lod_cache[lod, :lod_height, :lod_width, :] = lod_texture

        lod_cache = lod_cache.to(self.device)

        return lod_cache
    
    def get_output_loss_weights(self, config_weights: Optional[Dict[str, float]] = None) -> List[float]:
        """Generate per-channel loss weights based on the actually loaded textures.
        
        Args:
            config_weights: Optional dict mapping texture_type -> loss_weight from config.
                           If provided, overrides the default weights in texture_configs.
        """
        weights = []
        for tex_type in self.available_textures:
            cfg = self.texture_configs[tex_type]
            n_ch = cfg['expected_channels']
            # Priority: config_weights > texture_configs
            if config_weights and tex_type in config_weights:
                w = config_weights[tex_type]
            else:
                w = cfg.get('loss_weight', 1.0)
            weights.extend([w] * n_ch)
        return weights

    def expand_to_canonical(self, x: TensorType["batch_size", "num_channels"]) -> TensorType["batch_size", 11]:
        """Expand [B, num_channels] to canonical 11 channels; fill missing texture positions with 0."""
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        out = torch.zeros((batch_size, self.canonical_num_channels), device=device, dtype=dtype)
        for tex_type in self.available_textures:
            ds_start, ds_end = self.channel_slices[tex_type]
            canon_start, canon_end = self.canonical_channel_slices[tex_type]
            out[:, canon_start:canon_end] = x[:, ds_start:ds_end]
        return out

    def expand_lod_to_canonical(self, lod_tensor: TensorType["num_lods", "H", "W", "num_channels"]) -> TensorType["num_lods", "H", "W", 11]:
        """Expand lod_cache [num_lods, H, W, num_channels] to [num_lods, H, W, 11]; fill missing with 0."""
        num_lods, h, w, c = lod_tensor.shape
        device = lod_tensor.device
        dtype = lod_tensor.dtype
        out = torch.zeros((num_lods, h, w, self.canonical_num_channels), device=device, dtype=dtype)
        for tex_type in self.available_textures:
            ds_start, ds_end = self.channel_slices[tex_type]
            canon_start, canon_end = self.canonical_channel_slices[tex_type]
            out[:, :, :, canon_start:canon_end] = lod_tensor[:, :, :, ds_start:ds_end]
        return out

    def get_canonical_loss_weights(self, config_weights: Optional[Dict[str, float]] = None) -> List[float]:
        """Return per-channel loss weights of length 11; channels for missing textures are 0."""
        weights = [0.0] * self.canonical_num_channels
        for tex_type in self.keyword_order:
            canon_start, canon_end = self.canonical_channel_slices[tex_type]
            if tex_type not in self.available_textures:
                continue
            cfg = self.texture_configs[tex_type]
            w = config_weights.get(tex_type, cfg.get("loss_weight", 1.0)) if config_weights else cfg.get("loss_weight", 1.0)
            for i in range(canon_start, canon_end):
                weights[i] = w
        return weights

    def get_vis_configs(self) -> List[Dict]:
        """Return a list of visualization configs for all available textures.
        Each entry: texture_type, display_name, vis_mode, channel_slice (dataset order), canonical_channel_slice (0..11).
        """
        vis = []
        for tex_type in self.available_textures:
            cfg = self.texture_configs[tex_type]
            vis.append({
                'texture_type': tex_type,
                'display_name': cfg.get('display_name', tex_type),
                'vis_mode': cfg.get('vis_mode', 'linear'),
                'channel_slice': self.channel_slices[tex_type],
                'canonical_channel_slice': self.canonical_channel_slices[tex_type],
            })
        return vis

    def identify_texture_type(self, filename: str) -> str:
        
        filename_lower = filename.lower()

        # Identify the texture type based on keywords in the filename
        for texture_type, keywords in self.texture_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return texture_type
