"""FNTC grid compression boundary: deployment, sampling and shared surface plans."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn

from Common.Block_Compression.astc import ASTCCodec
from Common.Block_Compression.bc import dds_header
from Common.Block_Compression.ktx2 import header_bytes as ktx2_header_bytes, pack_mipmaps
from .layout import surface_filename
from Core.Feature_Grid.feature_grid import (
    feature_to_unorm,
    unorm_to_feature,
    quantize_feature_tensor,
    quantize_feature_ste,
)


class SourceGrid(nn.Module):
    """Explicit marker for an ordinary, non-aware feature source."""

    proxy_kind = "source"


class MipProxyGrid(nn.Module):
    """One independently optimized codec representation per learned level."""

    proxy_kind = 'mip_pyramid'

    def __init__(self, levels):
        super().__init__()
        self.levels = nn.ModuleList(levels)  # smallest to largest, like the source tensor

    def clamp_parameters(self):
        for level in self.levels:
            level.clamp_parameters()

    def freeze(self):
        for level in self.levels:
            if hasattr(level, 'freeze'):
                level.freeze()
            else:
                level.requires_grad_(False)

    def thaw(self):
        for level in self.levels:
            if hasattr(level, 'thaw'):
                level.thaw()
            else:
                level.requires_grad_(True)


def proxy_levels(proxy):
    return proxy.levels if isinstance(proxy, MipProxyGrid) else (proxy,)


def level_slice(spec, level):
    return spec.level_offset(level), spec.level_feature_count(level), spec.level_resolution(level)


def plans_for(model, index):
    from .layout import surface_plan

    return surface_plan(
        model.feature_grid_specs[index],
        model.grid_compressions[index],
        getattr(model, "direct_diffuse_enabled", False) and index == 0,
    )


def latent_payload_bytes(model):
    return sum(
        p.payload_bytes(spec.level_resolution(level), spec.level_resolution(level))
        for i, spec in enumerate(model.feature_grid_specs)
        for level in range(spec.n_levels)
        for p in plans_for(model, i)
    )


def latent_file_bytes(model):
    return latent_payload_bytes(model) + sum(
        len(plans_for(model, i)) * (ktx2_header_bytes(spec.n_levels) if recipe.format == 'astc' else 128)
        for i, (spec, recipe) in enumerate(zip(model.feature_grid_specs, model.grid_compressions)))


def grid_image(model, index, level=None):
    spec = model.feature_grid_specs[index]
    offset, count, res = level_slice(spec, spec.n_levels - 1 if level is None else level)
    source = model.feature_grids[index].params.detach()[
        offset : offset + count
    ]
    return feature_to_unorm(quantize_feature_tensor(source, spec), spec).reshape(
        res, res, spec.n_features_per_level
    )


def astc_codec_for(model, index):
    recipe = model.grid_compressions[index]
    return ASTCCodec(
        model.astcenc_path,
        model.astcenc_quality,
        recipe.astc_block,
    )


def initialize_grid_proxy(model, index, *, training_seed=True):
    spec = model.feature_grid_specs[index]
    if spec.n_levels > 1:
        return MipProxyGrid(proxy_from_image(model, index, grid_image(model, index, level),
                                             training_seed=training_seed)
                            for level in range(spec.n_levels))
    return proxy_from_image(model, index, grid_image(model, index), training_seed=training_seed)


def proxy_from_image(model, index, image, *, training_seed=True):
    """Construct a codec representation using this grid's format and packing."""
    recipe = model.grid_compressions[index]
    if recipe.format == "bc1":
        from Core.BCn_Aware.alpha_proxy import BC1AlphaProxyGrid
        return BC1AlphaProxyGrid.from_image(image, binary_seed=training_seed)
    codec = astc_codec_for(model, index)
    plans = plans_for(model, index)
    if (
        recipe.astc_block == "6x6"
        and image.shape[-1] == 4
        and len(plans) == 1
        and getattr(model, "astc_codec_backend", "") == "astc_differentiable_proxy"
    ):
        from Core.ASTC_Aware.differentiable_proxy import ASTCProxyGrid

        return ASTCProxyGrid.from_astc(
            codec.encode_rgba((image * 255).round().byte().cpu().numpy()), model.device
        )
    from Core.ASTC_Aware.roundtrip_proxy import ASTCRoundtripGrid, PackedASTCRoundtripGrid

    if len(plans) > 1:
        return PackedASTCRoundtripGrid(image, codec, [p.channels for p in plans])
    return ASTCRoundtripGrid(image, codec)


@dataclass(frozen=True)
class EncodedGrid:
    streams: tuple[bytes, ...]
    image: torch.Tensor


def proxy_streams(proxy):
    if proxy.proxy_kind == 'bc1_rgba':
        return tuple(proxy.export_blocks())
    if hasattr(proxy, 'export_surfaces'):
        return proxy.export_surfaces()
    return (proxy.export_astc(),)


def encode_grid(model, index, level=None):
    spec = model.feature_grid_specs[index]
    level = spec.n_levels - 1 if level is None else level
    recipe = model.grid_compressions[index]
    proxies = getattr(model, "grid_proxies", ())
    proxy = proxies[index] if len(proxies) > index else SourceGrid()
    if isinstance(proxy, MipProxyGrid):
        proxy = proxy.levels[level]
    if proxy.proxy_kind != "source":
        return EncodedGrid(proxy_streams(proxy), proxy.decode_cpu_image())
    if recipe.format == "bc1":
        proxy = proxy_from_image(model, index, grid_image(model, index, level), training_seed=False)
        return EncodedGrid(tuple(proxy.export_blocks()), proxy.decode_cpu_image())
    image = grid_image(model, index, level)
    codec = astc_codec_for(model, index)
    streams = []
    decoded = []
    for plan in plans_for(model, index):
        rgba = np.full((*image.shape[:2], 4), 255, dtype=np.uint8)
        rgba[..., : len(plan.channels)] = (
            (image[..., list(plan.channels)].cpu().numpy() * 255).round().astype(np.uint8)
        )
        data = codec.encode_rgba(rgba)
        streams.append(data)
        restored = codec.decode_bytes(data)[..., : len(plan.channels)].copy()
        decoded.append(torch.from_numpy(restored).float() / 255)
    return EncodedGrid(tuple(streams), torch.cat(decoded, -1))


@torch.no_grad()
def _decoded_grid_image(model, index, level=None):
    """Decode existing native proxies without exporting a bitstream for evaluation."""
    proxies = getattr(model, 'grid_proxies', ())
    if len(proxies) > index:
        proxy = proxies[index]
        if isinstance(proxy, MipProxyGrid):
            proxy = proxy.levels[model.feature_grid_specs[index].n_levels - 1 if level is None else level]
        if proxy.proxy_kind in ('bc1_rgba', 'astc_native') and proxy.endpoints.is_cuda:
            return proxy.decode_image()
    return encode_grid(model, index, level).image.to(model.feature_grids[index].params.device)


@torch.no_grad()
def apply_latent_compression(model):
    for i, (grid, spec) in enumerate(zip(model.feature_grids, model.feature_grid_specs)):
        for level in range(spec.n_levels):
            image = _decoded_grid_image(model, i, level)
            offset, count, _ = level_slice(spec, level)
            grid.params[offset : offset + count].copy_(
                quantize_feature_tensor(unorm_to_feature(image.reshape(-1), spec), spec))
    model._pending_grid_sync.clear()


@torch.no_grad()
def sync_grid_sources(model, indices):
    """Refresh source mirrors from authoritative proxies at a consumer boundary."""
    indices = tuple(indices)
    for index in indices:
        proxy = model.grid_proxies[index]
        if proxy.proxy_kind == 'source':
            continue
        spec = model.feature_grid_specs[index]
        for level, surface in enumerate(proxy_levels(proxy)):
            offset, count, _ = level_slice(spec, level)
            model.feature_grids[index].params[offset:offset+count].copy_(
                quantize_feature_tensor(unorm_to_feature(surface.decode_image().flatten(), spec), spec))
    model._pending_grid_sync.difference_update(indices)
    model._qat_param_cache.clear()


@torch.no_grad()
def decoded_feature_grids(model):
    """Build frozen codec inputs without copying the decoder or its optimizer."""
    fields = []
    for i, (grid, spec) in enumerate(zip(model.feature_grids, model.feature_grid_specs)):
        field = quantize_feature_tensor(grid.params.detach(), spec)
        for level in range(spec.n_levels):
            offset, count, _ = level_slice(spec, level)
            field[offset:offset+count] = unorm_to_feature(_decoded_grid_image(model, i, level).flatten(), spec)
        fields.append(field)
    return fields


def export_latents(model, directory):
    for i, recipe in enumerate(model.grid_compressions):
        spec, plans = model.feature_grid_specs[i], plans_for(model, i)
        chains = [[] for _ in plans]
        for level in reversed(range(spec.n_levels)):
            encoded = encode_grid(model, i, level)
            if len(encoded.streams) != len(plans):
                raise RuntimeError('Codec output does not match surface plan')
            for chain, stream, plan in zip(chains, encoded.streams, plans):
                payload = len(stream) - (16 if recipe.format == 'astc' else 0)
                res = spec.level_resolution(level)
                if payload != plan.payload_bytes(res, res):
                    raise RuntimeError('Encoded payload does not match surface plan')
                chain.append(stream)
        for surface, chain in enumerate(chains):
            data = (pack_mipmaps(chain) if recipe.format == 'astc' else
                    dds_header(spec.max_resolution, spec.max_resolution, mip_count=spec.n_levels) + b''.join(chain))
            Path(directory, surface_filename(i, surface, recipe, len(plans))).write_bytes(data)


def clamp_proxies(model):
    for proxy in model.grid_proxies:
        if proxy.proxy_kind != "source" and any(p.requires_grad for p in proxy.parameters()):
            proxy.clamp_parameters()


def freeze_proxies(model):
    # Prepare all encoded surfaces before committing any freeze flags or modules.
    proxies = list(model.grid_proxies) or [SourceGrid() for _ in model.grid_compressions]
    for i, proxy in enumerate(proxies):
        if proxy.proxy_kind == "source":
            proxies[i] = proxy = initialize_grid_proxy(model, i, training_seed=False)
        for level in proxy_levels(proxy):
            if level.proxy_kind == 'astc_roundtrip':
                proxy_streams(level)
    for proxy in proxies:
        if hasattr(proxy, 'freeze'):
            proxy.freeze()
        else:
            proxy.requires_grad_(False)
    model.grid_proxies = torch.nn.ModuleList(proxies)
    model._codec_active_grids = tuple(range(len(proxies)))


def sample_proxy_grid(model, grid_idx, feature_grid):
    proxy = model.grid_proxies[grid_idx]
    spec = model.feature_grid_specs[grid_idx]
    active = getattr(model, '_codec_active_grids', None)
    if proxy.proxy_kind == "source" or (active is not None and grid_idx not in active):
        return model._source_qat_grid_params(grid_idx, feature_grid)
    version = tuple(p._version for p in proxy.parameters())
    key = (grid_idx, "codec")
    cached = model._qat_param_cache.get(key)
    if cached is not None and cached[0] == version:
        return cached[1]
    decoded = quantize_feature_ste(
        torch.cat([level.decode_image().reshape(-1) for level in proxy_levels(proxy)])
        * ((spec.quant_step_count - 1) / spec.quant_step_count)
        + spec.quant_min,
        spec,
    )
    active = torch.cat((decoded, feature_grid.params.detach()[decoded.numel():]))
    model._qat_param_cache[key] = (version, active)
    return active
