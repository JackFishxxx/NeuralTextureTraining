"""Utilities for learned feature grids used by FNTC.

This module centralizes repeated resolution, level-offset, and quantization math
used by training, export, and ASTC comparison.
"""

from __future__ import annotations

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
        max_res = int(cfg.get("max_resolution", 1024))
        n_levels = int(cfg.get("n_levels", 1))
        qbits = int(cfg.get("quantize_bits", default_quantize_bits))
        sbits = int(cfg.get("save_bits", default_save_bits))
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


def quantize_feature_tensor(params: torch.Tensor, spec: FeatureGridSpec) -> torch.Tensor:
    """Quantize feature values to the configured uniform grid and return floats."""
    n = spec.quant_step_count
    ints = torch.round((params - spec.quant_min) * n)
    ints = torch.clamp(ints, min=0.0, max=float(n - 1))
    return ints / float(n) + spec.quant_min


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
