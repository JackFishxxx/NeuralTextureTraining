from torch.utils.data import Dataset
from configs import Config
from typing import Dict, List, Optional, Tuple
from torchtyping import TensorType

import torch
import os
import math
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from normal_encoding import (
    decode_normal,
    encode_normal,
    normal_encoding_channels,
    normal_encoding_vis_mode,
    normalize_normal_rgb,
    normal_to_rgb,
)


def get_texture_config(normal_encoding: str = "xyz") -> List[Dict]:
    normal_channels = normal_encoding_channels(normal_encoding)
    normal_vis_mode = normal_encoding_vis_mode(normal_encoding)

    keyword_order = ["diffuse", "normal", "roughness", "occlusion", "metallic", "specular", "displacement"]
    texture_keywords = {
        keyword_order[0]: ["diffuse", "albedo", "color", "diff"],
        keyword_order[1]: ["normal", "nor_gl"],
        keyword_order[2]: ["roughness", "rough"],
        keyword_order[3]: ["occlusion", "ao", "ambient"],
        keyword_order[4]: ["metallic", "metalness"],
        keyword_order[5]: ["specular"],
        keyword_order[6]: ["displacement", "disp", "height"],
        # add other texture types and their possible keywords if needed
    }

    # vis_mode: "srgb" = apply gamma for display, "normal" = re-normalize vectors, "linear" = save as-is
    # loss_weight: per-channel loss weight used during training
    # display_name: subfolder name used when saving evaluation images
    texture_configs = {
        keyword_order[0]: {"expected_channels": 3, "color_mode": "RGB", "vis_mode": "srgb",   "loss_weight": 1.0, "display_name": "rgb"},
        keyword_order[1]: {"expected_channels": normal_channels, "color_mode": "RGB", "vis_mode": normal_vis_mode, "loss_weight": 0.8, "display_name": "normal"},
        keyword_order[2]: {"expected_channels": 1, "color_mode": "L",   "vis_mode": "linear", "loss_weight": 0.3, "display_name": "roughness"},
        keyword_order[3]: {"expected_channels": 1, "color_mode": "L",   "vis_mode": "linear", "loss_weight": 0.3, "display_name": "occlusion"},
        keyword_order[4]: {"expected_channels": 1, "color_mode": "L",   "vis_mode": "linear", "loss_weight": 0.3, "display_name": "metallic"},
        keyword_order[5]: {"expected_channels": 1, "color_mode": "L",   "vis_mode": "linear", "loss_weight": 0.3, "display_name": "specular"},
        keyword_order[6]: {"expected_channels": 1, "color_mode": "L",   "vis_mode": "linear", "loss_weight": 0.3, "display_name": "displacement"},
        # add other texture types if needed
    }

    return keyword_order, texture_keywords, texture_configs


def get_canonical_num_channels(normal_encoding: str = "xyz") -> int:
    """Total canonical channels aligned with the selected normal encoding."""
    keyword_order, _, texture_configs = get_texture_config(normal_encoding)
    return sum(texture_configs[t]["expected_channels"] for t in keyword_order)


def get_canonical_channel_slices_static(normal_encoding: str = "xyz") -> Dict[str, tuple]:
    """Return (start, end) slice in the canonical layout for each texture type."""
    keyword_order, _, texture_configs = get_texture_config(normal_encoding)
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
        self.normal_encoding = str(getattr(config, "normal_encoding", "xyz")).lower()
        self.super_resolution_enable = bool(getattr(config, "super_resolution_enable", False))
        self.super_resolution_base_resolution = int(getattr(config, "super_resolution_base_resolution", 512))
        self.superres_input_max_edge = self.super_resolution_base_resolution

        self.keyword_order, self.texture_keywords, self.texture_configs = get_texture_config(self.normal_encoding)
        # mapping from texture type to channel slice [start, end) in current loaded data
        self.channel_slices = {}
        # (start, end) in the canonical 11-channel layout for each texture type, aligned with model
        self.canonical_channel_slices = get_canonical_channel_slices_static(self.normal_encoding)
        self.canonical_num_channels = get_canonical_num_channels(self.normal_encoding)
        # ordered available texture types that were found and concatenated
        self.available_textures = []

        self.textures = self.load_data()
        self.texture_height, self.texture_width, self.num_channels = self.textures.shape
        
        self.num_mips = int(min(math.log2(self.texture_height), math.log2(self.texture_width))) + 1

        self.mip_cache = self.generate_mip()
        self.superres_base_cache = self.generate_superres_base() if self.super_resolution_enable else None
    
    @torch.no_grad()
    def forward(
            self, 
            batch_index: TensorType["batch_size", 3]
        ) -> List[TensorType["batch_size", "num_channels"]]:

        # mip_cache: [self.num_mips, self.texture_height, self.texture_width, self.num_channels]
        # batch_index: [batch_size, 3]

        batch_size = batch_index.shape[0]
        ys = batch_index[:, 0]
        xs = batch_index[:, 1]
        mips = batch_index[:, 2]

        # use mips to scale the pixel position
        mip_scale = 2 ** mips
        scaled_xs = xs // mip_scale
        scaled_ys = ys // mip_scale

        batch_data = self.mip_cache[mips, scaled_ys, scaled_xs, :]

        return batch_data

    @torch.no_grad()
    def get_superres_base(
            self,
            batch_index: TensorType["batch_size", 3]
        ) -> TensorType["batch_size", "num_channels"]:
        """Sample the feature-grid-relative low-resolution texture at training positions."""
        if self.superres_base_cache is None:
            raise RuntimeError("Super-resolution is not enabled for this dataset")
        ys, xs, mips = batch_index[:, 0], batch_index[:, 1], batch_index[:, 2]
        mip_scale = 2 ** mips
        return self.superres_base_cache[mips, ys // mip_scale, xs // mip_scale, :]
        

    @staticmethod
    def _is_single_channel_content(tensor: torch.Tensor, atol: float = 1.0 / 255.0) -> bool:
        """Return True if a multi-channel scalar texture carries identical color values.

        Opaque alpha introduced by PNG/RGBA files is ignored for this test so RGB+constant-alpha
        scalar maps are still treated as single-channel photos.
        """
        if tensor.shape[0] <= 1:
            return True
        comparable = tensor
        if comparable.shape[0] == 4 and torch.allclose(
            comparable[3], torch.ones_like(comparable[3]), atol=atol, rtol=0.0
        ):
            comparable = comparable[:3]
        reference = comparable[:1]
        return bool(torch.allclose(comparable, reference.expand_as(comparable), atol=atol, rtol=0.0))

    @staticmethod
    def _split_packed_romd_channels(filename: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Map packed scalar texture channels to roughness/occlusion/metallic/displacement when encoded together.

        Common game-engine packed maps (e.g. MetallicSmoothness) often store multiple scalar material
        channels in RGB(A). If the channels are actually identical, keep the image as a single-channel
        texture; otherwise treat available channels as a packed ROMD source.
        """
        name = filename.lower()
        channel_map: List[Tuple[str, int]]

        if "metallicsmoothness" in name or "metallic_smoothness" in name or "metallic-smoothness" in name:
            # Unity-style packed map observed in data_batch: R=metallic, A=smoothness/roughness proxy.
            # Use alpha when present; otherwise fall back to green.
            rough_idx = 3 if tensor.shape[0] >= 4 else min(1, tensor.shape[0] - 1)
            channel_map = [("metallic", 0), ("roughness", rough_idx)]
        else:
            # Generic packed ROMD convention: R=roughness, G=occlusion, B=metallic, A=displacement.
            channel_map = [("roughness", 0), ("occlusion", 1), ("metallic", 2), ("displacement", 3)]

        packed: Dict[str, torch.Tensor] = {}
        for tex_type, channel_idx in channel_map:
            if channel_idx < tensor.shape[0]:
                packed[tex_type] = tensor[channel_idx:channel_idx + 1]
        return packed

    def _load_texture_tensor(self, filepath: str, texture_type: str) -> torch.Tensor:
        with Image.open(filepath) as image:
            if texture_type in {"roughness", "occlusion", "metallic", "displacement"}:
                # Preserve scalar-map packing channels instead of blindly converting to L.
                color_mode = "RGBA" if "A" in image.getbands() else "RGB"
            else:
                color_mode = self.texture_configs[texture_type]['color_mode']
            image = image.convert(color_mode)
            tensor = TF.to_tensor(image)
            if texture_type != "normal":
                tensor = torch.pow(tensor, 2.2)
            else:
                tensor = encode_normal(tensor, self.normal_encoding)
        return tensor

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
            expected_channels = self.texture_configs[texture_type]['expected_channels']
            tensor = self._load_texture_tensor(filepath, texture_type)

            if texture_type in {"roughness", "occlusion", "metallic", "displacement"} and tensor.shape[0] > 1:
                if self._is_single_channel_content(tensor):
                    tensor = tensor[:1]
                else:
                    for packed_type, packed_tensor in self._split_packed_romd_channels(filename, tensor).items():
                        if packed_type not in textures:
                            print(f"Packed ROMD Texture: type='{packed_type}' from filename='{filename}'")
                            textures[packed_type] = packed_tensor
                    continue

            if tensor.shape[0] != expected_channels:
                raise ValueError(f"Expected {expected_channels} channels for {texture_type}, got {tensor.shape[0]}")

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
    
    def _resize_texture_for_mip(self, texture_type: str, tex_chw: torch.Tensor, mip_height: int, mip_width: int) -> torch.Tensor:
        if texture_type == "normal":
            normal_vec = normalize_normal_rgb(decode_normal(tex_chw, self.normal_encoding))
            mip_vec = TF.resize(
                normal_vec, [mip_height, mip_width],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True,
            )
            return encode_normal(normal_to_rgb(mip_vec), self.normal_encoding)

        mip_texture = TF.resize(
            tex_chw, [mip_height, mip_width],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        )
        return torch.clamp(mip_texture, min=0., max=1.)

    def generate_mip(self) -> TensorType["num_mips", "mip_height", "mip_width", "num_channels"]:

        mip_cache = torch.zeros(
            [self.num_mips, self.texture_height, self.texture_width, self.num_channels]
        )
        # here is a bug in pytorch while using a tensor on cuda to interpolate
        # from a large size to a small size, e.g. [1024, 1024] -> [8, 8]
        # the bug was not fixed since 2023
        # work on cpu seems not to have this problem
        textures = self.textures.cpu()

        for mip in range(self.num_mips):
            mip_height = self.texture_height // (2 ** mip)
            mip_width = self.texture_width // (2 ** mip)
            mip_parts = []
            for texture_type in self.available_textures:
                ds_start, ds_end = self.channel_slices[texture_type]
                tex_chw = textures[:, :, ds_start:ds_end].permute(2, 0, 1)
                mip_parts.append(self._resize_texture_for_mip(texture_type, tex_chw, mip_height, mip_width))
            mip_texture = torch.cat(mip_parts, dim=0).permute(1, 2, 0)
            # [mip, H, W, C] <- [mip_H, mip_W, C]
            mip_cache[mip, :mip_height, :mip_width, :] = mip_texture

        mip_cache = mip_cache.to(self.device)

        return mip_cache

    def generate_superres_base(self) -> TensorType["num_mips", "H", "W", "num_channels"]:
        """Build baselines from a texture half the maximum feature-grid resolution."""
        base_cache = torch.zeros_like(self.mip_cache)
        texture_max_edge = max(self.texture_height, self.texture_width)
        ratio = texture_max_edge / float(self.superres_input_max_edge)
        source_mip_offset = max(0, int(round(math.log2(ratio)))) if ratio > 1.0 else 0
        actual_source_edge = texture_max_edge // (2 ** source_mip_offset)
        print(
            f"[SuperResolution] feature_grid_max={self.superres_input_max_edge * 2}, "
            f"input_texture_max={actual_source_edge}, source_mip_offset={source_mip_offset}"
        )
        for mip in range(self.num_mips):
            out_h = self.texture_height // (2 ** mip)
            out_w = self.texture_width // (2 ** mip)
            source_mip = min(mip + source_mip_offset, self.num_mips - 1)
            if source_mip == mip:
                base_cache[mip, :out_h, :out_w, :] = self.mip_cache[mip, :out_h, :out_w, :]
                continue
            low_h = self.texture_height // (2 ** source_mip)
            low_w = self.texture_width // (2 ** source_mip)
            low = self.mip_cache[source_mip, :low_h, :low_w, :].permute(2, 0, 1)[None].float()
            upsampled = torch.nn.functional.interpolate(
                low, size=(out_h, out_w), mode="bilinear", align_corners=False
            )
            base_cache[mip, :out_h, :out_w, :] = upsampled.squeeze(0).permute(1, 2, 0)
        return base_cache
    
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
        """Expand [B, num_channels] to canonical channels; fill missing texture positions with 0."""
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        out = torch.zeros((batch_size, self.canonical_num_channels), device=device, dtype=dtype)
        for tex_type in self.available_textures:
            ds_start, ds_end = self.channel_slices[tex_type]
            canon_start, canon_end = self.canonical_channel_slices[tex_type]
            out[:, canon_start:canon_end] = x[:, ds_start:ds_end]
        return out

    def expand_mip_to_canonical(self, mip_tensor: TensorType["num_mips", "H", "W", "num_channels"]) -> TensorType["num_mips", "H", "W", 11]:
        """Expand mip_cache [num_mips, H, W, num_channels] to [num_mips, H, W, canonical_channels]; fill missing with 0."""
        num_mips, h, w, c = mip_tensor.shape
        device = mip_tensor.device
        dtype = mip_tensor.dtype
        out = torch.zeros((num_mips, h, w, self.canonical_num_channels), device=device, dtype=dtype)
        for tex_type in self.available_textures:
            ds_start, ds_end = self.channel_slices[tex_type]
            canon_start, canon_end = self.canonical_channel_slices[tex_type]
            out[:, :, :, canon_start:canon_end] = mip_tensor[:, :, :, ds_start:ds_end]
        return out

    def get_canonical_loss_weights(self, config_weights: Optional[Dict[str, float]] = None) -> List[float]:
        """Return per-channel loss weights of canonical length; channels for missing textures are 0."""
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
        
        if "ROM" in filename:
            return "roughness"

        filename_lower = filename.lower()

        # Identify the texture type based on keywords in the filename
        for texture_type, keywords in self.texture_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return texture_type
