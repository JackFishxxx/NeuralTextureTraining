import warnings
import torch


NORMAL_XY_MIN_Z = 0.01


def _enc(normal_encoding):
    enc = str(normal_encoding).lower()
    if enc not in ('xyz', 'xy', 'hemi_oct'):
        raise ValueError(f'Unsupported normal encoding: {normal_encoding}')
    return enc


def _channel_dim(x: torch.Tensor) -> int:
    return 1 if x.ndim in (2, 4) else 0


def _take(x: torch.Tensor, start: int, length: int) -> torch.Tensor:
    return x.narrow(_channel_dim(x), start, length)


def normalize_normal_rgb(rgb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = rgb * 2.0 - 1.0
    cd = _channel_dim(n)
    return n / torch.sqrt(torch.clamp((n ** 2).sum(dim=cd, keepdim=True), min=eps))


def normal_to_rgb(n: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cd = _channel_dim(n)
    n = n / torch.sqrt(torch.clamp((n ** 2).sum(dim=cd, keepdim=True), min=eps))
    return torch.clamp(n * 0.5 + 0.5, 0.0, 1.0)


def encode_normal(normal_rgb: torch.Tensor, normal_encoding: str = 'xyz') -> torch.Tensor:
    enc = _enc(normal_encoding)
    n = normalize_normal_rgb(normal_rgb)
    cd = _channel_dim(n)
    if enc == 'xyz':
        return normal_to_rgb(n)
    if enc == 'xy':
        return torch.clamp(_take(n, 0, 2) * 0.5 + 0.5, 0.0, 1.0)
    if enc == 'hemi_oct':
        x = _take(n, 0, 1)
        y = _take(n, 1, 1)
        z = torch.clamp(_take(n, 2, 1), min=0.0)
        denom = torch.clamp(x.abs() + y.abs() + z, min=1e-8)
        px, py = x / denom, y / denom
        return torch.clamp(torch.cat([px + py, px - py], dim=cd) * 0.5 + 0.5, 0.0, 1.0)
    raise ValueError(f'Unsupported normal_encoding: {normal_encoding}')


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
        x = xy.narrow(cd, 0, 1)
        y = xy.narrow(cd, 1, 1)
        z = torch.sqrt(torch.clamp(1.0 - (xy ** 2).sum(dim=cd, keepdim=True), min=NORMAL_XY_MIN_Z ** 2))
        return normal_to_rgb(torch.cat([x, y, z], dim=cd))
    if enc == 'hemi_oct':
        x = xy.narrow(cd, 0, 1)
        y = xy.narrow(cd, 1, 1)
        px = (x + y) * 0.5
        py = (x - y) * 0.5
        z = 1.0 - px.abs() - py.abs()
        return normal_to_rgb(torch.cat([px, py, z], dim=cd))
    raise ValueError(f'Unsupported normal_encoding: {normal_encoding}')


def normal_vectors(encoded: torch.Tensor, normal_encoding: str = 'xyz') -> torch.Tensor:
    return normalize_normal_rgb(decode_normal(encoded.float().clamp(0, 1), normal_encoding))


def normal_angular_loss(pred_encoded: torch.Tensor, gt_encoded: torch.Tensor, normal_encoding: str = 'xyz',
                        reduction: str = 'mean') -> torch.Tensor:
    # Use the deployed UNORM range in forward, but allow out-of-range decoder
    # channels to recover. A second hard clamp here would strand them forever.
    pred = pred_encoded.float()
    bounded = pred + (pred.clamp(0, 1) - pred).detach()
    pred_n = normalize_normal_rgb(decode_normal(bounded, normal_encoding))
    gt_n = normal_vectors(gt_encoded, normal_encoding)
    dot = (pred_n * gt_n).sum(dim=_channel_dim(pred_n)).clamp(-1.0, 1.0)
    loss = (1.0 - dot) * 0.5
    if reduction == 'none':
        return loss
    if reduction != 'mean':
        raise ValueError('Normal angular loss reduction must be mean or none')
    return loss.mean()


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


def normal_angular_error_degrees(pred_encoded: torch.Tensor, gt_encoded: torch.Tensor, normal_encoding: str = 'xyz') -> torch.Tensor:
    pred_n = normal_vectors(pred_encoded, normal_encoding)
    gt_n = normal_vectors(gt_encoded, normal_encoding)
    dot = (pred_n * gt_n).sum(dim=_channel_dim(pred_n)).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def normal_angular_psnr(
    pred_encoded: torch.Tensor,
    gt_encoded: torch.Tensor,
    normal_encoding: str = 'xyz',
    max_angle_degrees: float = 180.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    deg = normal_angular_error_degrees(pred_encoded, gt_encoded, normal_encoding)
    mse = torch.clamp(torch.mean(deg ** 2), min=eps)
    max_angle = torch.tensor(float(max_angle_degrees), device=mse.device, dtype=mse.dtype)
    return 20.0 * torch.log10(max_angle) - 10.0 * torch.log10(mse)


def normal_encoding_channels(normal_encoding: str) -> int:
    return 2 if _enc(normal_encoding) in {'xy', 'hemi_oct'} else 3


def normal_encoding_vis_mode(normal_encoding: str) -> str:
    return 'normal_encoded' if normal_encoding_channels(normal_encoding) == 2 else 'normal'
