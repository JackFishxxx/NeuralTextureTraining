"""IEC 61966-2-1 sRGB transfer functions for RGB values (never alpha or latents)."""

import torch


SRGB_TRANSFER = 'srgb_iec61966_2_1'


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Decode nonnegative sRGB RGB values before filtering or shading."""
    rgb = rgb.clamp_min(0)
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    """Encode linear RGB for display, with finite gradients at black."""
    rgb = rgb.clamp_min(0)
    # Both branches of torch.where are evaluated. Keep the unused power branch
    # away from zero so backward cannot produce 0 * inf at black.
    nonlinear = 1.055 * rgb.clamp_min(0.0031308).pow(1.0 / 2.4) - 0.055
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, nonlinear)
