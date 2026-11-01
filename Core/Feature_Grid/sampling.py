"""Texel-centered repeat sampling primitives shared by feature and material queries."""

from typing import Tuple

import torch
import torch.nn.functional as F


def tile_regions(height, width, tile_size):
    """Yield (y, x, height, width), including partial tiles on the right and bottom."""
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size < 1:
        raise ValueError('Tile size must be a positive integer')
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            yield y, x, min(tile_size, height-y), min(tile_size, width-x)


def bilinear_texel_corners(uv, resolution, *, offsets=None):
    resolution = torch.as_tensor(resolution, device=uv.device, dtype=uv.dtype)
    if resolution.ndim < 2:
        resolution = resolution.reshape(1, -1)
    position = torch.remainder(uv, 1.0) * resolution - 0.5
    lower = torch.floor(position)
    fraction = position - lower
    if offsets is None:
        offsets = uv.new_tensor(((0, 0), (1, 0), (0, 1), (1, 1)))
    corners = torch.remainder(lower[:, None, :] + offsets, resolution[:, None, :]).long()
    tx, ty = fraction.unbind(1)
    weights = torch.stack(((1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty), 1)
    return corners, weights


def bilinear_repeat(image: torch.Tensor, uv_grid: torch.Tensor) -> torch.Tensor:
    """Texel-centered bilinear repeat lookup of an NCHW image at normalized UVs."""
    h, w = image.shape[-2:]
    # Modulo alone is insufficient: interpolation across the seam also needs
    # the opposite edge texels. A circular one-texel halo supplies both corners.
    padded = F.pad(image.float(), (1, 1, 1, 1), mode="circular")
    scale = padded.new_tensor((w, h))
    return _sample_padded_repeat(padded, scale, uv_grid)


def _sample_padded_repeat(padded, scale, uv_grid):
    uv = torch.remainder(uv_grid.float(), 1.0)
    grid = 2 * (uv * scale + 1) / (scale + 2) - 1
    return F.grid_sample(padded, grid, mode="bilinear", align_corners=False)


class RepeatImageSampler:
    """Reuse the circular halo of a fixed NCHW source; refresh on in-place edits.

    Trainable images use bilinear_repeat directly so their graphs stay fresh.
    This sampler is for the immutable clean/compressed SR training bases.
    """

    def __init__(self, image):
        if image.requires_grad:
            raise ValueError('A cached repeat sampler requires a non-trainable image')
        self.image = image
        self._version = None

    def __call__(self, uv):
        if self._version != self.image._version:
            h, w = self.image.shape[-2:]
            self._padded = F.pad(self.image.float(), (1, 1, 1, 1), mode='circular')
            self._scale = self._padded.new_tensor((w, h))
            self._version = self.image._version
        return _sample_padded_repeat(self._padded, self._scale, uv[None, :, None]).squeeze(0).squeeze(-1).T


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
