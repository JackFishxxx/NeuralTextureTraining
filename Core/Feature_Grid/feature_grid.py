"""Utilities for learned feature grids used by FNTC.

This module centralizes repeated resolution, level-offset, and quantization math
used by training, export, and ASTC comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class FeatureGridSpec:
    max_resolution: int
    n_levels: int
    n_features_per_level: int
    quantize_bits: int
    save_bits: int
    learning_rate: float
    interpolation: str = "Linear"
    base_resolution: int = 1

    @classmethod
    def from_config(cls, cfg: Dict, default_quantize_bits: int, default_save_bits: int, default_lr: float) -> "FeatureGridSpec":
        unknown = set(cfg) - {'max_resolution', 'n_levels', 'quantize_bits', 'save_bits',
                              'learning_rate', 'interpolation'}
        if unknown:
            raise ValueError(f'Unknown feature_grid_configs keys: {sorted(unknown)}')
        for name in ('max_resolution', 'n_levels', 'quantize_bits', 'save_bits'):
            if name in cfg and (not isinstance(cfg[name], int) or isinstance(cfg[name], bool)):
                raise ValueError(f'feature_grid_configs.{name} must be an integer')
        if cfg.get('interpolation', 'Linear') not in ('Linear', 'Nearest', 'Smoothstep'):
            raise ValueError('Unsupported feature_grid_configs.interpolation')
        lr = cfg.get('learning_rate', default_lr)
        if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
            raise ValueError('feature_grid_configs.learning_rate must be finite and positive')
        max_res = int(cfg.get("max_resolution", 1024))
        if max_res <= 0 or (max_res & (max_res - 1)) != 0:
            raise ValueError("max_resolution must be a positive power of two")
        default_levels = max(1, max_res.bit_length() - 2)
        n_levels = int(cfg.get("n_levels", default_levels))
        qbits = int(cfg.get("quantize_bits", default_quantize_bits))
        sbits = int(cfg.get("save_bits", default_save_bits))
        if n_levels <= 0 or n_levels > max_res.bit_length():
            raise ValueError(
                f"n_levels must be in [1, {max_res.bit_length()}] for max_resolution={max_res}"
            )
        if qbits not in (2, 4, 8):
            raise ValueError("quantize_bits must be one of 2, 4, 8 for RGBA8 DDS export")
        if sbits not in (8, 16, 32) or sbits < qbits:
            raise ValueError("save_bits must be one of 8, 16, 32 and >= quantize_bits")
        if sbits % qbits != 0:
            raise ValueError("save_bits must be divisible by quantize_bits")
        if sbits // qbits > 4:
            raise ValueError("feature-grid export supports at most four channels per level")
        base_res = max(1, int(max_res >> (n_levels - 1)))
        return cls(
            max_resolution=max_res,
            n_levels=n_levels,
            n_features_per_level=sbits // qbits,
            quantize_bits=qbits,
            save_bits=sbits,
            learning_rate=float(cfg.get("learning_rate", default_lr)),
            interpolation=str(cfg.get("interpolation", "Linear")),
            base_resolution=base_res,
        )

    def level_resolution(self, level: int) -> int:
        return int(self.base_resolution) * (2 ** int(level))

    def level_feature_count(self, level: int) -> int:
        res = self.level_resolution(level)
        return res * res * int(self.n_features_per_level)

    def level_offset(self, level: int) -> int:
        return sum(self.level_feature_count(i) for i in range(int(level)))

    def highest_level_slice(self) -> tuple[int, int, int]:
        level = int(self.n_levels) - 1
        offset = self.level_offset(level)
        count = self.level_feature_count(level)
        return offset, count, self.level_resolution(level)

    @property
    def quant_min(self) -> float:
        n = 2 ** int(self.quantize_bits)
        return -float(n - 1) / (2.0 * float(n))

    @property
    def quant_max(self) -> float:
        return 0.5

    @property
    def quant_step_count(self) -> int:
        return 2 ** int(self.quantize_bits)


def compact_mip0_features(params: torch.Tensor, spec: FeatureGridSpec) -> torch.Tensor:
    """Read the highest level of a legacy dense pyramid into a single-level grid."""
    count = spec.max_resolution ** 2 * spec.n_features_per_level
    if params.numel() == count:
        return params
    for levels in range(2, spec.max_resolution.bit_length() + 1):
        base = spec.max_resolution >> (levels - 1)
        prefix = sum((base << level) ** 2 * spec.n_features_per_level for level in range(levels - 1))
        # TCNN pads each allocation to eight texels; legacy sampling/export
        # uses FeatureGridSpec's logical offsets, which must remain unchanged.
        padded = sum(((base << level) ** 2 + 7) // 8 * 8 * spec.n_features_per_level
                     for level in range(levels))
        if params.numel() in (prefix + count, padded):
            result = params.new_zeros(((spec.max_resolution ** 2 + 7) // 8) * 8 * spec.n_features_per_level)
            result[:count] = params[prefix:prefix + count]
            return result
    raise ValueError('Checkpoint feature count does not match a dense pyramid at the configured resolution')


def quantize_feature_tensor(params: torch.Tensor, spec: FeatureGridSpec) -> torch.Tensor:
    """Quantize feature values to the configured uniform grid and return floats."""
    n = spec.quant_step_count
    ints = torch.round((params - spec.quant_min) * n)
    ints = torch.clamp(ints, min=0.0, max=float(n - 1))
    return ints / float(n) + spec.quant_min


def quantize_feature_ste(params: torch.Tensor, spec: FeatureGridSpec) -> torch.Tensor:
    """Use export's quantization grid in forward and identity gradients in backward."""
    quantized = quantize_feature_tensor(params, spec)
    return params + (quantized - params).detach()


def feature_tensor_to_int(params: torch.Tensor, spec: FeatureGridSpec) -> torch.Tensor:
    """Map feature floats to integer quantization codes [0, 2^qbits - 1]."""
    n = spec.quant_step_count
    ints = torch.round((params - spec.quant_min) * n)
    return torch.clamp(ints, min=0.0, max=float(n - 1)).to(torch.int64)


def int_to_feature_tensor(values: torch.Tensor, spec: FeatureGridSpec) -> torch.Tensor:
    """Map integer quantization codes back to feature-space floats."""
    return torch.clamp(values.float() / float(spec.quant_step_count) + spec.quant_min, min=spec.quant_min, max=spec.quant_max)


def feature_to_unorm(x: torch.Tensor, spec: FeatureGridSpec) -> torch.Tensor:
    n = float(spec.quant_step_count)
    return ((x - spec.quant_min) * n / (n - 1.0)).clamp(0, 1)


def unorm_to_feature(x: torch.Tensor, spec: FeatureGridSpec) -> torch.Tensor:
    n = float(spec.quant_step_count)
    return torch.round(x.clamp(0, 1) * (n - 1.0)) / n + spec.quant_min


def enforce_wrap_boundary_constraint_inplace(
    flat_params: torch.Tensor,
    base_res: int,
    n_levels: int,
    n_features_per_level: int,
    strength: float = 1.0,
    highest_k_levels: int = 1,
) -> None:
    """Tie opposite borders of each dense grid level for seamless repeat sampling.

    This makes left/right and top/bottom borders consistent, which removes
    visible seams when feature textures are sampled with wrap mode at runtime.
    """
    strength = float(max(0.0, min(1.0, strength)))
    if strength <= 0.0:
        return

    highest_k_levels = max(1, int(highest_k_levels))
    start_level = max(0, n_levels - highest_k_levels)

    offset = 0
    one_minus = 1.0 - strength

    for level in range(n_levels):
        res = base_res * (2 ** level)
        level_count = res * res * n_features_per_level
        if level < start_level:
            offset += level_count
            continue

        level_flat = flat_params[offset: offset + level_count]
        level_tex = level_flat.view(res, res, n_features_per_level)

        # Match left/right edges
        left = level_tex[:, 0, :].clone()
        right = level_tex[:, -1, :].clone()
        lr_avg = 0.5 * (left + right)
        level_tex[:, 0, :] = left * one_minus + lr_avg * strength
        level_tex[:, -1, :] = right * one_minus + lr_avg * strength

        # Match top/bottom edges
        top = level_tex[0, :, :].clone()
        bottom = level_tex[-1, :, :].clone()
        tb_avg = 0.5 * (top + bottom)
        level_tex[0, :, :] = top * one_minus + tb_avg * strength
        level_tex[-1, :, :] = bottom * one_minus + tb_avg * strength

        # Keep four corners fully consistent to avoid corner pinching.
        c00 = level_tex[0, 0, :].clone()
        c01 = level_tex[0, -1, :].clone()
        c10 = level_tex[-1, 0, :].clone()
        c11 = level_tex[-1, -1, :].clone()
        corner_avg = 0.25 * (c00 + c01 + c10 + c11)
        level_tex[0, 0, :] = c00 * one_minus + corner_avg * strength
        level_tex[0, -1, :] = c01 * one_minus + corner_avg * strength
        level_tex[-1, 0, :] = c10 * one_minus + corner_avg * strength
        level_tex[-1, -1, :] = c11 * one_minus + corner_avg * strength

        offset += level_count


def qat_noise_multiplier(schedule, current_iter, max_iter, warmup_frac, mult_start, mult_end):
    """Scale fake-quant noise with the configured warmup and cosine decay."""
    if schedule == "none":
        return 1.0
    T = max(1, max_iter - 1)
    t = min(max(int(current_iter), 0), T)
    w = int(warmup_frac * T)
    if t <= w:
        return mult_start
    p = (t - w) / max(1, T - w)
    c = 0.5 * (1.0 + math.cos(math.pi * p))
    return mult_end + (mult_start - mult_end) * c
