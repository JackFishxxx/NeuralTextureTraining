import math

import torch


class TiledPositionalEncoding(torch.nn.Module):
    """NeRF-style tiled sin/cos positional encoding with a configurable tile size.

    Frequencies are integer harmonics of the fundamental 2*pi / tile_uv, i.e.
    omega_k = 2*pi * 2^k / tile_uv. Every component therefore completes an integer
    number of cycles per tile, so the whole encoding is exactly periodic with the
    tile and tiles seamlessly (C^inf across tile borders).

    Output dims = 2 (u/v) * 2 (sin/cos) * n_frequencies = 4 * n_frequencies.
    n_frequencies = 0 -> zero output dims (encoding disabled).
    """

    def __init__(self, n_frequencies: int, tile_uv: float, device: str = "cuda"):
        super().__init__()
        self.n_frequencies = int(n_frequencies)
        self.tile_uv = float(max(tile_uv, 1e-9))
        n = int(n_frequencies)
        if n > 0:
            freqs = (2.0 * math.pi / self.tile_uv) * (
                2.0 ** torch.arange(n, dtype=torch.float32, device=device)
            )
        else:
            freqs = torch.zeros(0, dtype=torch.float32, device=device)
        # Non-persistent: deterministic from config, so old checkpoints still load.
        self.register_buffer("freqs", freqs, persistent=False)
        self.n_output_dims = 4 * self.n_frequencies

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        if self.n_frequencies == 0:
            return uv.new_zeros(uv.shape[0], 0)
        f = self.freqs  # [N]
        su = torch.sin(uv[:, 0:1] * f)  # [B, 1] x [N] -> [B, N]
        cu = torch.cos(uv[:, 0:1] * f)
        sv = torch.sin(uv[:, 1:2] * f)
        cv = torch.cos(uv[:, 1:2] * f)
        return torch.cat([su, cu, sv, cv], dim=1)  # [B, 4N]
