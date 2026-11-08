"""Finite-difference feature inputs: center values followed by neighbor-minus-center."""

import torch

from .sampling import bilinear_texel_corners


def gradient_directions(count):
    if not isinstance(count, int) or isinstance(count, bool) or count not in (0, 2, 4, 8, 24):
        raise ValueError("feature_gradient_count must be 0, 2, 4, 8 or 24")
    if count == 24:
        return tuple((x, y) for y in range(-2, 3) for x in range(-2, 3) if (x, y) != (0, 0))
    return ((1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (-1, 1), (-1, -1), (1, -1))[:count]


def sample_feature_inputs(params, spec, uv, selected_level):
    """Return center followed by neighbor-minus-center differences.

    All samples use bilinear repeat filtering of the same parameter tensor, including decoded ASTC and frozen
    adaptation tensors. Eight directions add the four diagonal neighbors to the
    axial directions, completing the selected mip's 3x3 texel neighborhood.
    Twenty-four directions use the complete 5x5 neighborhood in row-major order.
    """
    directions = gradient_directions(spec.feature_gradient_count)
    selected = selected_level.reshape(-1).clamp(0, spec.n_levels - 1)
    lower = selected.floor().long()
    shifts = uv.new_tensor(((0, 0), *directions))
    samples = _sample_level_inputs(params, spec, uv, lower, shifts)
    if spec.n_levels > 1:
        upper = (lower + 1).clamp(max=spec.n_levels - 1)
        blend = (selected - lower).reshape(-1, 1)
        samples = torch.lerp(samples, _sample_level_inputs(params, spec, uv, upper, shifts), blend)
    return samples


def _sample_level_inputs(params, spec, uv, levels, shifts):
    resolution = (spec.base_resolution * 2 ** levels).reshape(-1, 1)
    # Dense levels grow by 4; avoid a GPU table transfer per sampling call.
    offsets = (spec.base_resolution ** 2 * (4 ** levels - 1) // 3) * spec.n_features_per_level
    queries = uv[:, None, :] + shifts[None, :, :] / resolution[:, None, :]
    query_res = resolution[:, None, :].expand(-1, len(shifts), -1).reshape(-1, 1)
    corners, weights = bilinear_texel_corners(queries.reshape(-1, 2), query_res)
    query_offsets = offsets[:, None].expand(-1, len(shifts)).reshape(-1, 1)
    indices = query_offsets + (corners[..., 1] * query_res + corners[..., 0]) * spec.n_features_per_level
    channels = torch.arange(spec.n_features_per_level, device=params.device)
    values = params[indices[..., None] + channels].float()
    samples = (values * weights[..., None]).sum(1).reshape(len(uv), len(shifts), -1)
    center = samples[:, :1]
    return torch.cat((center, samples[:, 1:] - center), 1).flatten(1)


def sample_feature_centers(params, spec, uv, *, shifts=None):
    """Gather known highest-level texel centers and their integer neighbors.

    The caller guarantees a matching power-of-two native raster and one grid
    level. General/fractional UVs continue to use sample_feature_inputs.
    """
    resolution = spec.max_resolution
    if shifts is None:
        shifts = torch.tensor(((0, 0), *gradient_directions(spec.feature_gradient_count)),
                              device=uv.device, dtype=torch.long)
    pixels = (uv * resolution).floor().long()
    xy = (pixels[:, None] + shifts).remainder(resolution)
    indices = xy[..., 1] * resolution + xy[..., 0]
    samples = params.reshape(-1, spec.n_features_per_level)[indices].float()
    return torch.cat((samples[:, :1], samples[:, 1:] - samples[:, :1]), 1).flatten(1)
