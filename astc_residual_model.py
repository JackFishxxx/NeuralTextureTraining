import torch
import torch.nn as nn


class ASTCResidualNoiseModel(nn.Module):
    """Train-time pseudo ASTC residual simulator.
    It perturbs UV/LOD sampling coordinates with block-correlated offsets
    to emulate ASTC 6x6-style deployment mismatch without changing inference graph.
    """

    def __init__(
        self,
        block_size: int = 6,
        noise_std: float = 0.02,
        channel_corr: float = 0.5,
        noise_prob: float = 0.0,
        warmup_ratio: float = 0.2,
        peak_ratio: float = 0.7,
        max_iter: int = 400000,
    ):
        super().__init__()
        self.block_size = max(1, int(block_size))
        self.noise_std = float(noise_std)
        self.channel_corr = float(channel_corr)
        self.noise_prob = float(noise_prob)
        self.warmup_ratio = float(warmup_ratio)
        self.peak_ratio = float(peak_ratio)
        self.max_iter = int(max_iter)

    def _curriculum_scale(self, curr_iter: int, enable: bool = True) -> float:
        if not enable:
            return 1.0
        p = float(curr_iter) / max(1.0, float(self.max_iter - 1))
        if p < self.warmup_ratio:
            return 0.0
        if p >= self.peak_ratio:
            return 1.0
        t = (p - self.warmup_ratio) / max(1e-6, self.peak_ratio - self.warmup_ratio)
        return float(max(0.0, min(1.0, t)))

    @torch.no_grad()
    def perturb_uvlod_input(self, batch_input: torch.Tensor, curr_iter: int, enable: bool = True):
        scale = self._curriculum_scale(curr_iter, enable=enable)
        if scale <= 0.0:
            return batch_input, False
        if torch.rand(1, device=batch_input.device).item() > self.noise_prob:
            return batch_input, False

        out = batch_input.clone()
        b = out.shape[0]

        block_ids = torch.arange(b, device=out.device) // self.block_size
        n_blocks = int(block_ids.max().item()) + 1
        block_noise = torch.randn(n_blocks, 2, device=out.device) * (self.noise_std * scale)
        uv_noise = block_noise[block_ids]

        corr = torch.randn(n_blocks, 1, device=out.device) * (self.noise_std * self.channel_corr * scale)
        uv_noise = uv_noise + corr[block_ids]

        out[:, 0:2] = torch.remainder(out[:, 0:2] + uv_noise, 1.0)
        out[:, 2:3] = torch.clamp(out[:, 2:3] + 0.25 * corr[block_ids], 0.0, 1.0)
        return out, True

    def build_sensitive_mask(self, gt_texture: torch.Tensor, percentile: float = 0.85, detach_mask: bool = True):
        # gt_texture: [B, C], use first 3 channels as luma proxy when possible
        if gt_texture.shape[1] >= 3:
            luma = 0.2126 * gt_texture[:, 0] + 0.7152 * gt_texture[:, 1] + 0.0722 * gt_texture[:, 2]
        else:
            luma = gt_texture.mean(dim=1)

        grad = torch.zeros_like(luma)
        grad[1:] = (luma[1:] - luma[:-1]).abs()
        q = torch.quantile(grad, torch.tensor(percentile, device=grad.device))
        m = (grad >= q).float().unsqueeze(1)
        m = m.expand_as(gt_texture)
        if detach_mask:
            m = m.detach()
        return 1.0 + m
