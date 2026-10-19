"""Texel-centered repeat sampling primitives shared by feature and material queries."""

from typing import Tuple

import torch
import torch.nn.functional as F


def bilinear_texel_corners(uv, resolution):
    resolution = torch.as_tensor(resolution, device=uv.device, dtype=uv.dtype)
    if resolution.ndim < 2:
        resolution = resolution.reshape(1, -1)
    position = torch.remainder(uv, 1.0) * resolution - 0.5
    lower = torch.floor(position)
    fraction = position - lower
    offsets = uv.new_tensor(((0, 0), (1, 0), (0, 1), (1, 1)))
    corners = torch.remainder(lower[:, None, :] + offsets, resolution[:, None, :]).long()
    tx, ty = fraction.unbind(1)
    weights = torch.stack(((1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty), 1)
    return corners, weights


def texel_aligned_grid_uv(uvs: torch.Tensor, selected_level, base_res: int) -> torch.Tensor:
    """Map texel-center UVs to tiny-cuda-nn dense-grid point coordinates."""
    if not torch.is_tensor(selected_level):
        selected_level = torch.full((uvs.shape[0], 1), float(selected_level), device=uvs.device, dtype=uvs.dtype)
    level_int = torch.round(selected_level).to(torch.int64)
    res = int(base_res) * torch.pow(
        torch.full_like(selected_level, 2.0),
        level_int.to(dtype=selected_level.dtype),
    )
    if bool(torch.all(res <= 1)):
        return torch.zeros_like(uvs)
    scale = torch.clamp(res - 1.0, min=1.0)
    # tiny-cuda-nn evaluates grid position = input * (resolution-1) + .5.
    # Cancel its .5 offset so a texture center samples its stored texel.
    # The first texel intentionally uses a small negative encoding input;
    # its resulting grid position is zero, not a wrapped negative index.
    return ((uvs * res - 0.5).clamp(min=0.0).minimum(res - 1.0) - 0.5) / scale


def nearest_grid_uv(uvs: torch.Tensor, selected_level, base_res: int) -> torch.Tensor:
    """Nearest texel center in tiny-cuda-nn grid coordinates."""
    if not torch.is_tensor(selected_level):
        selected_level = torch.full((uvs.shape[0], 1), float(selected_level), device=uvs.device, dtype=uvs.dtype)
    level_int = torch.round(selected_level).to(torch.int64)
    res = int(base_res) * torch.pow(
        torch.full_like(selected_level, 2.0),
        level_int.to(dtype=selected_level.dtype),
    )
    if bool(torch.all(res <= 1)):
        return torch.zeros_like(uvs)
    texel = torch.floor(uvs * res).clamp(min=0.0)
    texel = torch.minimum(texel, torch.clamp(res - 1.0, min=0.0))
    return (texel - 0.5) / torch.clamp(res - 1.0, min=1.0)


def bilinear_repeat(image: torch.Tensor, uv_grid: torch.Tensor) -> torch.Tensor:
    """Texel-centered bilinear repeat lookup of an NCHW image at normalized UVs."""
    h, w = image.shape[-2:]
    # Modulo alone is insufficient: interpolation across the seam also needs
    # the opposite edge texels. A circular one-texel halo supplies both corners.
    padded = F.pad(image.float(), (1, 1, 1, 1), mode="circular")
    uv = torch.remainder(uv_grid.float(), 1.0)
    scale = uv.new_tensor((w, h))
    grid = 2 * (uv * scale + 1) / (scale + 2) - 1
    return F.grid_sample(padded, grid, mode="bilinear", align_corners=False)


def resize_bilinear_repeat(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Rasterize repeat samples at destination texel centers."""
    h, w = size
    y = (torch.arange(h, device=image.device, dtype=torch.float32) + .5) / h
    x = (torch.arange(w, device=image.device, dtype=torch.float32) + .5) / w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    uv = torch.stack((xx, yy), -1)[None].expand(image.shape[0], -1, -1, -1)
    return bilinear_repeat(image, uv)


def sample_image(image, uv):
    return bilinear_repeat(image, uv[None, :, None]).squeeze(0).squeeze(-1).T
