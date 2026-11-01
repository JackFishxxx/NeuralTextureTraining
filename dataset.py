from configs import Config
from typing import Dict, List, Optional, Tuple, Union
from torchtyping import TensorType

import torch
import json
import os
import math
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from Core.Feature_Grid.sampling import bilinear_texel_corners, resize_bilinear_repeat, sample_image
from Core.Material_Encoding.normal_encoding import (
    decode_normal,
    encode_normal,
    normal_encoding_channels,
    normal_encoding_vis_mode,
    normalize_normal_rgb,
    prepare_normal_input,
    normal_to_rgb,
)

ROMSD_TYPES = ('roughness', 'occlusion', 'metallic', 'specular', 'displacement')


def build_texture_packing(available_textures, romsd_format: str) -> Dict[str, List[str]]:
    """Pack actual maps in R/O/M/S/D priority, excluding BC1's binary alpha."""
    available = set(available_textures)
    groups = {name: [name] for name in ('diffuse', 'normal') if name in available}
    scalars = [name for name in ROMSD_TYPES if name in available]
    if scalars:
        capacity = 3 if romsd_format == 'BC1' else 4
        groups['romsd'] = scalars[:capacity]
    return groups


def material_size(model, height, width, *, resolution=None):
    """Use the configured material raster consistently in fitting and deployment."""
    edge = resolution if resolution is not None else getattr(model,'material_compare_resolution',None)
    if edge is not None and edge <= 0:
        raise ValueError('Material resolution must be positive')
    if edge is None:
        inference_size = getattr(model,'inference_size',())
        if inference_size:
            return tuple(inference_size)
    if edge is None:
        if hasattr(model,'feature_grid_specs'):
            edge = max(spec.max_resolution for spec in model.feature_grid_specs)
        else:
            edge = max(spec['max_resolution'] for spec in model.feature_grid_configs)
        edge = min(edge,max(height,width))
    scale = edge / max(height,width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def prefiltered_material(dataset, size):
    """Reuse the same material-aware mip for ASTC inputs and projection seeds."""
    for level in dataset.mip_cache:
        if tuple(level.shape[:2]) == tuple(size):
            return dataset.expand_to_canonical(level).permute(2, 0, 1)[None].float()
    source = dataset.mip_cache[0]
    result = source.new_zeros((*size, dataset.canonical_num_channels))
    for name in dataset.available_textures:
        start, end = dataset.channel_slices[name]
        a, b = dataset.canonical_channel_slices[name]
        result[..., a:b] = dataset._resize_texture_for_mip(
            name, source[..., start:end].permute(2, 0, 1), *size).permute(1, 2, 0)
    return result.permute(2, 0, 1)[None].float()


def clean_base_image(dataset, size):
    """Sample the actual low-resolution SR source directly onto the native raster."""
    ratio = max(dataset.texture_height, dataset.texture_width) / dataset.super_resolution_base_resolution
    level = min(max(0, round(math.log2(ratio))) if ratio > 1 else 0, dataset.num_mips - 1)
    image = dataset.expand_to_canonical(dataset.mip_cache[level]).permute(2, 0, 1)[None].float()
    return resize_bilinear_repeat(image, size) if tuple(image.shape[-2:]) != tuple(size) else image


def native_query_uv(uv, size):
    """Prepare shared native corners with fixed or exact output capacity.

    Sparse batches use fixed-capacity compaction to avoid synchronizing a
    dynamic unique count with the CPU. Repeated corners still share predictions
    and accumulate their gradients before the decoder's FP16 backward.
    Dense queries compact to an exact size to bound decoder work and memory.
    """
    height, width = size
    return _native_query_uv(uv, size, uv.new_tensor((width, height)),
                            uv.new_tensor(((0, 0), (1, 0), (0, 1), (1, 1))))


def _native_query_uv(uv, size, resolution, offsets):
    height, width = size
    corners, weights = bilinear_texel_corners(uv, resolution, offsets=offsets)
    ids = corners[..., 1] * width + corners[..., 0]
    if ids.numel() and ids.numel() <= height * width // 2:
        ordered, permutation = ids.flatten().sort()
        starts = torch.cat((ordered.new_ones(1), (ordered[1:] != ordered[:-1]).long()))
        groups = starts.cumsum(0) - 1
        # Equal IDs write the same value. Unused capacity contains texel zero;
        # inverse never refers to it, so its decoder gradient remains zero.
        unique = torch.zeros_like(ordered).scatter_(0, groups, ordered)
        inverse = torch.empty_like(groups).scatter_(0, permutation, groups)
    else:
        unique, inverse = ids.flatten().unique(return_inverse=True)
    centers = torch.stack(((unique % width + .5) / width, (unique // width + .5) / height), 1)
    return centers, inverse, weights


class NativeQueryPlan:
    """Replay fixed-capacity query preparation without per-operator CPU launches."""

    def __init__(self, uv, size):
        if not uv.is_cuda or uv.requires_grad or len(uv)*4 > size[0]*size[1]//2 or not len(uv):
            raise ValueError('A query plan requires a sparse, non-trainable CUDA coordinate batch')
        self.size = size
        with torch.cuda.device(uv.device):
            self._input = uv.detach().contiguous().clone()
            resolution = uv.new_tensor((size[1], size[0]))
            offsets = uv.new_tensor(((0, 0), (1, 0), (0, 1), (1, 1)))
            self._constants = resolution, offsets
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _native_query_uv(self._input, size, resolution, offsets)  # Initialize sort/scan workspaces.
            torch.cuda.current_stream().wait_stream(stream)
            self._graph = torch.cuda.CUDAGraph()
            from Common.cuda_graph import graph_capture_lifetime
            with graph_capture_lifetime(), torch.cuda.graph(self._graph):
                self._outputs = _native_query_uv(self._input, size, resolution, offsets)

    def __call__(self, uv):
        if (uv.requires_grad or uv.shape != self._input.shape or uv.dtype != self._input.dtype
                or uv.device != self._input.device):
            return native_query_uv(uv, self.size)
        self._input.copy_(uv)
        self._graph.replay()
        # Calls may precede a shared backward. Consumers must not retain views
        # into the graph buffers that the next query replay will overwrite.
        return tuple(value.clone() for value in self._outputs)


def sample_material(model, uv, size, base_image=None, *, base_sampler=None, query_plan=None):
    """Differentiate through U(clamp(base_native + decoder_native)) at arbitrary UVs.

    Only native raster centers reach the decoder. The four output values,
    rather than latent features, interpolate. Dense batches share predictions.
    """
    centers, inverse, weights = query_plan(uv) if query_plan is not None else native_query_uv(uv, size)
    inputs = torch.cat((centers, centers.new_zeros((len(centers), 1))), 1)
    base = (base_sampler(centers) if base_sampler is not None
            else sample_image(base_image, centers) if base_image is not None else None)
    if base is not None:
        inputs = torch.cat((inputs, base), 1)
    native_forward = getattr(model, 'forward_native_centers', None)
    values = (native_forward(inputs, size) if native_forward is not None else model(inputs)).float()
    if base is not None:
        values = values + base
    bounded = values.clamp(0, 1)
    if model.training:
        # Match the deployed raster in forward while letting reconstruction
        # gradients recover channels initialized outside [0, 1]. Hard clamp
        # alone permanently strands negative non-SR outputs at black.
        values = bounded.detach() + (values - values.detach())
    else:
        values = bounded
    return (values[inverse].reshape(len(uv), 4, -1) * weights[..., None]).sum(1)


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
        self.material_reconstruction_size = material_size(config, self.texture_height, self.texture_width)
        
        self.num_mips = int(min(math.log2(self.texture_height), math.log2(self.texture_width))) + 1

        self.mip_cache = self.generate_mip()
        self.superres_base_cache = self.generate_superres_base() if self.super_resolution_enable else None
    
    @torch.no_grad()
    def sample_continuous(self, sample_xy: torch.Tensor, mips: Union[torch.Tensor, int],
                          cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Bilinear repeat sampling; an integer mip avoids dynamic GPU masking."""
        source = self.mip_cache if cache is None else cache
        fixed_mip = isinstance(mips, int)
        if fixed_mip:
            level = source[mips]
            scale = float(2 ** mips)
            h, w = level.shape[:2]
        else:
            mip_i = mips.to(torch.long).reshape(-1)
            scale = torch.pow(torch.tensor(2.0, device=sample_xy.device), mip_i).float()
            h = torch.tensor([level.shape[0] for level in source], device=sample_xy.device)[mip_i]
            w = torch.tensor([level.shape[1] for level in source], device=sample_xy.device)[mip_i]
        x, y = sample_xy[:, 1].float() / scale, sample_xy[:, 0].float() / scale
        fx, fy = torch.floor(x), torch.floor(y)
        x0, y0 = torch.remainder(fx.to(torch.long), w), torch.remainder(fy.to(torch.long), h)
        x1, y1 = torch.remainder(x0 + 1, w), torch.remainder(y0 + 1, h)
        dtype = source[0].dtype
        tx, ty = (x - fx).to(dtype), (y - fy).to(dtype)
        if fixed_mip:
            c00, c10 = level[y0, x0], level[y0, x1]
            c01, c11 = level[y1, x0], level[y1, x1]
        else:
            c00 = torch.empty((len(mip_i), source[0].shape[-1]), device=source[0].device, dtype=dtype)
            c10, c01, c11 = c00.clone(), c00.clone(), c00.clone()
            for mip, level in enumerate(source):
                mask = mip_i == mip
                c00[mask], c10[mask] = level[y0[mask], x0[mask]], level[y0[mask], x1[mask]]
                c01[mask], c11[mask] = level[y1[mask], x0[mask]], level[y1[mask], x1[mask]]
        return ((1-tx[:,None])*(1-ty[:,None])*c00 + tx[:,None]*(1-ty[:,None])*c10
                + (1-tx[:,None])*ty[:,None]*c01 + tx[:,None]*ty[:,None]*c11)

    @torch.no_grad()
    def get_superres_base_continuous(self, sample_xy, mips):
        if self.superres_base_cache is None:
            raise RuntimeError("Super-resolution is not enabled")
        return self.sample_continuous(sample_xy, mips, cache=self.superres_base_cache)

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
        is_metallic_smoothness = any(
            token in name for token in ("metallicsmoothness", "metallic_smoothness", "metallic-smoothness")
        )

        if is_metallic_smoothness:
            # Unity-style packed map observed in data_batch: R=metallic and
            # A/G=smoothness. Convert smoothness to the roughness convention
            # used by the renderer and training target.
            rough_idx = 3 if tensor.shape[0] >= 4 else min(1, tensor.shape[0] - 1)
            channel_map = [("metallic", 0), ("roughness", rough_idx)]
        elif 'roms' in name:
            channel_map = [('roughness', 0), ('occlusion', 1), ('metallic', 2), ('specular', 3)]
        else:
            # Generic packed ROMD convention: R=roughness, G=occlusion, B=metallic, A=displacement.
            channel_map = [("roughness", 0), ("occlusion", 1), ("metallic", 2), ("displacement", 3)]

        packed: Dict[str, torch.Tensor] = {}
        for tex_type, channel_idx in channel_map:
            if channel_idx < tensor.shape[0]:
                channel = tensor[channel_idx:channel_idx + 1]
                if tex_type == "roughness" and is_metallic_smoothness:
                    channel = 1.0 - channel
                packed[tex_type] = channel
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
            if texture_type == "diffuse":
                tensor = torch.pow(tensor, 2.2)
            elif texture_type == "normal":
                tensor = prepare_normal_input(tensor, self.normal_encoding)
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
                if self._is_single_channel_content(tensor) and 'rom' not in filename.lower():
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

        romsd_format = self.config.material_compression_formats['romsd']
        self.texture_packing = build_texture_packing(textures, romsd_format)
        active = {name for names in self.texture_packing.values() for name in names}
        for name in list(textures):
            if name not in active:
                del textures[name]
                print(f'[Texture Packing] dropping {name}: does not fit {romsd_format} ROMSD packing')
        print(f'[Texture Packing] {self.texture_packing}')
        
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
                    if texture_type == "normal":
                        normal_vec = normalize_normal_rgb(
                            decode_normal(textures[texture_type], self.normal_encoding)
                        )
                        normal_vec = TF.resize(
                            normal_vec, [target_h, target_w],
                            interpolation=TF.InterpolationMode.BILINEAR,
                            antialias=True,
                        )
                        textures[texture_type] = encode_normal(
                            normal_to_rgb(normal_vec), self.normal_encoding
                        )
                    else:
                        textures[texture_type] = TF.resize(
                            textures[texture_type], [target_h, target_w],
                            interpolation=TF.InterpolationMode.BILINEAR,
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

    def save_texture_packing(self, path):
        """Export the executed mapping without coupling it to network parameters."""
        with open(path, 'w', encoding='utf-8') as output:
            json.dump({'texture_packing': self.texture_packing,
                       'canonical_channel_slices': self.canonical_channel_slices,
                       'normal_encoding': self.normal_encoding,
                       'normal_output_convention': 'opengl',
                       'padding_value': 1.0}, output, indent=2)
    
    def _resize_texture_for_mip(self, texture_type: str, tex_chw: torch.Tensor, mip_height: int, mip_width: int) -> torch.Tensor:
        if texture_type == "normal":
            normal_vec = normalize_normal_rgb(decode_normal(tex_chw, self.normal_encoding))
            mip_vec = TF.resize(
                normal_vec, [mip_height, mip_width],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            return encode_normal(normal_to_rgb(mip_vec), self.normal_encoding)

        mip_texture = TF.resize(
            tex_chw, [mip_height, mip_width],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return torch.clamp(mip_texture, min=0., max=1.)

    def generate_mip(self) -> TensorType["num_mips", "mip_height", "mip_width", "num_channels"]:

        mip_cache = []
        # here is a bug in pytorch while using a tensor on cuda to interpolate
        # from a large size to a small size, e.g. [1024, 1024] -> [8, 8]
        # the bug was not fixed since 2023
        # work on cpu seems not to have this problem
        textures = self.textures.cpu()

        # Build a conventional mip chain: each level is filtered from the
        # immediately preceding level, rather than repeatedly from Mip0.
        previous_parts = {}
        for mip in range(self.num_mips):
            mip_height = self.texture_height // (2 ** mip)
            mip_width = self.texture_width // (2 ** mip)
            mip_parts = []
            for texture_type in self.available_textures:
                ds_start, ds_end = self.channel_slices[texture_type]
                if mip == 0:
                    tex_chw = textures[:, :, ds_start:ds_end].permute(2, 0, 1)
                else:
                    tex_chw = previous_parts[texture_type]
                mip_parts.append(self._resize_texture_for_mip(texture_type, tex_chw, mip_height, mip_width))
                previous_parts[texture_type] = mip_parts[-1]
            mip_texture = torch.cat(mip_parts, dim=0).permute(1, 2, 0)
            # [mip, H, W, C] <- [mip_H, mip_W, C]
            mip_cache.append(mip_texture)

        mip_cache = [m.to(self.device) for m in mip_cache]

        return mip_cache

    def generate_superres_base(self) -> TensorType["num_mips", "H", "W", "num_channels"]:
        """Build baselines from a texture half the maximum feature-grid resolution."""
        base_cache = []
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
                base_cache.append(self.mip_cache[mip].clone())
                continue
            low = self.mip_cache[source_mip].permute(2, 0, 1)[None].float()
            upsampled = resize_bilinear_repeat(low, (out_h, out_w))
            base_cache.append(upsampled.squeeze(0).permute(1, 2, 0))
        return base_cache
    

    def expand_to_canonical(self, x: torch.Tensor) -> torch.Tensor:
        """Expand the last dimension to canonical channels, filling absent textures with 0."""
        out = x.new_zeros((*x.shape[:-1], self.canonical_num_channels))
        for tex_type in self.available_textures:
            ds_start, ds_end = self.channel_slices[tex_type]
            canon_start, canon_end = self.canonical_channel_slices[tex_type]
            out[..., canon_start:canon_end] = x[..., ds_start:ds_end]
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
        
        if "rom" in filename.lower():
            return "roughness"

        filename_lower = filename.lower()

        # Identify the texture type based on keywords in the filename
        for texture_type, keywords in self.texture_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return texture_type
