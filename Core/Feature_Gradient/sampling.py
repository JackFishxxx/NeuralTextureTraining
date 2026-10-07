"""Texel-centered repeat sampling and finite-difference decoder features."""

import torch


def gradient_directions(count):
    if count not in (0, 2, 4, 8, 24):
        raise ValueError("feature_gradient_count must be 0, 2, 4, 8 or 24")
    if count == 24:
        return tuple((x, y) for y in range(-2, 3) for x in range(-2, 3) if (x, y) != (0, 0))
    return ((1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (-1, 1), (-1, -1), (1, -1))[:count]


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


def sample_feature_inputs(params, spec, uv, selected_level, gradient_count=0):
    """Return center followed by neighbor-minus-center differences.

    All samples use the same parameter tensor, including decoded ASTC and frozen
    adaptation tensors. Eight directions add the four diagonal neighbors to the
    axial directions, completing the selected mip's 3x3 texel neighborhood.
    Twenty-four directions use the complete 5x5 neighborhood in row-major order.
    """
    directions = gradient_directions(gradient_count)
    levels = selected_level.round().long().reshape(-1).clamp(0, spec.n_levels - 1)
    resolution = (spec.base_resolution * 2 ** levels).reshape(-1, 1)
    # Dense levels grow by 4; avoid a GPU table transfer per sampling call.
    offsets = (spec.base_resolution ** 2 * (4 ** levels - 1) // 3) * spec.n_features_per_level
    shifts = uv.new_tensor(((0, 0), *directions))
    queries = uv[:, None, :] + shifts[None, :, :] / resolution[:, None, :]
    query_res = resolution[:, None, :].expand(-1, len(shifts), -1).reshape(-1, 1)
    corners, weights = bilinear_texel_corners(queries.reshape(-1, 2), query_res)
    if not gradient_count and spec.interpolation == "Nearest":
        corners = torch.remainder(torch.floor(torch.remainder(queries.reshape(-1, 2), 1.) * query_res),
                                  query_res).long()[:, None, :]
        weights = torch.ones_like(weights[:, :1])
    elif not gradient_count and spec.interpolation == "Smoothstep":
        fraction = torch.remainder(torch.remainder(queries.reshape(-1, 2), 1.) * query_res - .5, 1.)
        tx, ty = (fraction.square() * (3 - 2 * fraction)).unbind(1)
        weights = torch.stack(((1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty), 1)
    elif spec.interpolation not in ("Linear", "Nearest", "Smoothstep"):
        raise ValueError(f"Unsupported feature interpolation: {spec.interpolation}")
    query_offsets = offsets[:, None].expand(-1, len(shifts)).reshape(-1, 1)
    indices = query_offsets + (corners[..., 1] * query_res + corners[..., 0]) * spec.n_features_per_level
    channels = torch.arange(spec.n_features_per_level, device=params.device)
    values = params[indices[..., None] + channels].float()
    samples = (values * weights[..., None]).sum(1).reshape(len(uv), len(shifts), -1)
    center = samples[:, :1]
    return torch.cat((center, samples[:, 1:] - center), 1).flatten(1)
