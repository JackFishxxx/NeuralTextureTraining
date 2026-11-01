"""Shared BC1 quantization primitives and RGB endpoint initialization."""
import math

import torch


class _ExactSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, exact):
        return exact

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def ste_round(x):
    return _ExactSTE.apply(x, x.round())


def pack565(codes):
    codes = codes.long()
    return (codes[..., 0] << 11) | (codes[..., 1] << 5) | codes[..., 2]


def unpack565(words):
    return torch.stack(((words >> 11) & 31, (words >> 5) & 63, words & 31), -1)


def expanded(codes):
    """BC1 bit replication, in integer UNORM8 units."""
    c = codes.long()
    return torch.stack(
        (
            (c[..., 0] << 3) | (c[..., 0] >> 2),
            (c[..., 1] << 2) | (c[..., 1] >> 4),
            (c[..., 2] << 3) | (c[..., 2] >> 2),
        ),
        -1,
    ).float()


@torch.no_grad()
def seed_rgb_endpoints(image):
    """Fit RGB565 endpoints for the RGB component of a binary-alpha surface."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('BC1 endpoint initialization expects HWC RGB')
    h, w, _ = image.shape
    bh, bw = math.ceil(h/4), math.ceil(w/4)
    y = torch.arange(bh*4, device=image.device).clamp(max=h-1)
    x = torch.arange(bw*4, device=image.device).clamp(max=w-1)
    padded = image[y[:, None], x[None, :]].clamp(0, 1)
    pixels = padded.reshape(bh, 4, bw, 4, 3).permute(0, 2, 1, 3, 4).reshape(-1, 16, 3)
    # PCA seed avoids choosing unrelated per-component min/max endpoints.
    centered = pixels - pixels.mean(1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered
    direction = torch.ones_like(pixels[:, 0])
    for _ in range(8):
        direction = (covariance @ direction[..., None]).squeeze(-1)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scores = (centered * direction[:, None]).sum(-1)
    row = torch.arange(len(pixels), device=image.device)
    ends = torch.stack((pixels[row, scores.argmax(1)], pixels[row, scores.argmin(1)]), 1)
    scale = image.new_tensor([31, 63, 31])
    codes = (ends * scale).round()
    packed = pack565(codes)
    hi, lo = packed.max(1).values, packed.min(1).values
    hi = torch.where(hi == lo, (hi + 1).clamp(max=65535), hi)
    lo = torch.where(hi == lo, (lo - 1).clamp(min=0), lo)
    codes = unpack565(torch.stack((hi, lo), 1)).float()
    return codes
