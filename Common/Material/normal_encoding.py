import warnings
import torch
from torch.nn import functional as F


NORMAL_XY_MIN_Z = 0.01


def filter_normal(encoded, size, normal_encoding='xyz'):
    """Decode, area-filter and re-encode a CHW/NCHW normal map."""
    value = decode_normal(encoded.float(), normal_encoding)
    batched = value.ndim == 4
    value = F.adaptive_avg_pool2d(value if batched else value[None], size)
    return encode_normal(value if batched else value[0], normal_encoding)


def _enc(normal_encoding):
    enc = str(normal_encoding).lower()
    if enc not in ('xyz', 'xy', 'hemi_oct'):
        raise ValueError(f'Unsupported normal encoding: {normal_encoding}')
    return enc


def _channel_dim(x: torch.Tensor) -> int:
    return 1 if x.ndim in (2, 4) else 0


def _take(x: torch.Tensor, start: int, length: int) -> torch.Tensor:
    return x.narrow(_channel_dim(x), start, length)


def _normalize(n: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return n / torch.sqrt(torch.clamp((n ** 2).sum(dim=_channel_dim(n), keepdim=True), min=eps))


def normalize_normal_rgb(rgb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return _normalize(rgb * 2.0 - 1.0, eps)


def normal_to_rgb(n: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.clamp(_normalize(n, eps) * 0.5 + 0.5, 0.0, 1.0)


def encode_normal(normal_rgb: torch.Tensor, normal_encoding: str = 'xyz') -> torch.Tensor:
    enc = _enc(normal_encoding)
    n = normalize_normal_rgb(normal_rgb)
    cd = _channel_dim(n)
    if enc == 'xyz':
        return normal_to_rgb(n)
    if enc == 'xy':
        return torch.clamp(_take(n, 0, 2) * 0.5 + 0.5, 0.0, 1.0)
    x = _take(n, 0, 1)
    y = _take(n, 1, 1)
    z = torch.clamp(_take(n, 2, 1), min=0.0)
    denom = torch.clamp(x.abs() + y.abs() + z, min=1e-8)
    px, py = x / denom, y / denom
    return torch.clamp(torch.cat([px + py, px - py], dim=cd) * 0.5 + 0.5, 0.0, 1.0)


def decode_normal(encoded: torch.Tensor, normal_encoding: str = 'xyz') -> torch.Tensor:
    enc = _enc(normal_encoding)
    cd = _channel_dim(encoded)
    if enc == 'xyz':
        return normal_to_rgb(normalize_normal_rgb(encoded))
    xy = _take(encoded, 0, 2) * 2.0 - 1.0
    if enc == 'xy':
        # Project out-of-disk predictions slightly inside the hemisphere so
        # sqrt() never receives a zero argument during training backprop.
        xy_sq = (xy ** 2).sum(dim=cd, keepdim=True)
        max_radius_sq = 1.0 - NORMAL_XY_MIN_Z ** 2
        scale = torch.sqrt(torch.clamp(
            max_radius_sq / torch.clamp(xy_sq, min=1e-8), max=1.0
        ))
        xy = xy * scale
        z = torch.sqrt(torch.clamp(1.0 - (xy ** 2).sum(dim=cd, keepdim=True), min=NORMAL_XY_MIN_Z ** 2))
        return normal_to_rgb(torch.cat([xy, z], dim=cd))
    x, y = xy.split(1, dim=cd)
    px = (x + y) * 0.5
    py = (x - y) * 0.5
    z = 1.0 - px.abs() - py.abs()
    return normal_to_rgb(torch.cat([px, py, z], dim=cd))


def normal_vectors(encoded: torch.Tensor, normal_encoding: str = 'xyz') -> torch.Tensor:
    return normalize_normal_rgb(decode_normal(encoded.float().clamp(0, 1), normal_encoding))


def prepare_normal_input(rgb, normal_encoding='xyz'):
    """Encode OpenGL source normals, reporting information lost by hemisphere encodings."""
    rgb = rgb.float()
    if not torch.isfinite(rgb).all():
        raise ValueError('Normal input contains nonfinite values')
    # Normalization cannot change Z's sign; avoid normalizing the full image twice.
    negative = _take(rgb, 2, 1) < .5
    if _enc(normal_encoding) in ('xy', 'hemi_oct') and bool(negative.any()):
        count = int(negative.sum())
        message = (f'{count} normal texels have negative Z, which {normal_encoding} cannot represent; '
                   'use normal_encoding: xyz to preserve the source hemisphere')
        warnings.warn(message, UserWarning, stacklevel=2)
    return encode_normal(rgb, normal_encoding)


def normal_encoding_channels(normal_encoding: str) -> int:
    return 2 if _enc(normal_encoding) in {'xy', 'hemi_oct'} else 3


def normal_encoding_vis_mode(normal_encoding: str) -> str:
    return 'normal_encoded' if normal_encoding_channels(normal_encoding) == 2 else 'normal'
