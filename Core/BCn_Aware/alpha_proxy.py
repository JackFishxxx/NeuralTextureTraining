"""BC1 RGBA representation: learned block modes and joint color/binary selectors.

Each surface carries RGB + one binary alpha at texel centers. Transparent alpha
is selector 3 in the three-color mode and always implies RGB=0. Opaque blocks
can instead spend that selector on a fourth interpolated RGB value. Both choices
are learned with categorical straight-through gradients and exact hard forward.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from Common.Block_Compression.bc import decode_blocks
from .differentiable_proxy import seed_rgb_endpoints, _ExactSTE, expanded, pack565, ste_round, unpack565


class BC1AlphaProxyGrid(nn.Module):
    proxy_kind = 'bc1_rgba'

    def __init__(self, endpoints, weights, height, width, channels, modes=None):
        if channels not in (4, 8):
            raise ValueError("BC1 RGBA requires 4 or 8 features per grid")
        super().__init__()
        self.endpoints = nn.Parameter(endpoints.float())
        self.weights = nn.Parameter(weights.float())
        self.register_buffer('shape', torch.tensor([height, width, channels], device=endpoints.device))
        self._image_shape, self.channels = (height, width), channels
        blocks = math.ceil(height / 4) * math.ceil(width / 4) * (channels // 4)
        if tuple(weights.shape) != (blocks, 16, 4) or tuple(endpoints.shape) != (blocks, 2, 3):
            raise ValueError("BC1 RGBA parameter shapes do not match image dimensions")
        # Mode 0: three RGB entries + transparent black. Mode 1: four opaque RGB entries.
        if modes is None:
            modes = endpoints.new_tensor([0.1, 0.0]).expand(blocks, 2).clone()
        self.modes = nn.Parameter(modes.float())
        self.register_buffer("bc1_alpha_version", torch.tensor(1, device=endpoints.device))
        self.register_buffer("temperature", torch.tensor(1.0, device=endpoints.device))

    @classmethod
    @torch.no_grad()
    def from_image(cls, image, *, binary_seed=True):
        if image.ndim != 3 or image.shape[-1] not in (4, 8):
            raise ValueError("BC1 RGBA seed requires HWC with 4 or 8 channels")
        h, w, c = image.shape
        if not binary_seed:
            # Offline proposals encode the candidate alpha values faithfully;
            # the sparse exploration mask belongs only to training startup.
            from Common.Block_Compression.bc import BCCodec

            codec = BCCodec()
            blocks = []
            for start in range(0, c, 4):
                rgba = (image[..., start:start+4].detach().clamp(0, 1)*255).round().byte().cpu().numpy()
                blocks.append(codec.encode_blocks(rgba))
            packed = np.frombuffer(b''.join(blocks), dtype=np.dtype([
                ('a', '<u2'), ('b', '<u2'), ('bits', '<u4')]))
            words = torch.tensor(np.stack((packed['a'], packed['b']), 1).astype(np.int64), device=image.device)
            bits = torch.tensor(packed['bits'].astype(np.int64), device=image.device)
            selectors = (bits[:, None] >> (2*torch.arange(16, device=image.device))) & 3
            modes = F.one_hot((words[:, 0] > words[:, 1]).long(), 2).float()*.25
            ends = unpack565(words.sort(1).values).float()
            return cls(ends, F.one_hot(selectors, 4).float()*.25, h, w, c, modes)
        bh, bw = math.ceil(h / 4), math.ceil(w / 4)
        surfaces = image.reshape(h, w, c // 4, 4).permute(2, 0, 1, 3).clamp(0, 1)
        endpoints, pixels = [], []
        y = torch.arange(bh * 4, device=image.device).clamp(max=h - 1)
        x = torch.arange(bw * 4, device=image.device).clamp(max=w - 1)
        for surface in surfaces:
            words = pack565(seed_rgb_endpoints(surface[..., :3])).sort(1).values
            endpoints.append(unpack565(words).float())
            padded = surface[y[:, None], x[None, :]]
            pixels.append(padded.reshape(bh, 4, bw, 4, 4).permute(0, 2, 1, 3, 4).reshape(-1, 16, 4))
        ends, source = torch.cat(endpoints), torch.cat(pixels)
        a, b = expanded(ends).unbind(1)
        palette = torch.stack((a, b, torch.floor((a + b) / 2)), 1) / 255
        selected = (source[:, :, None, :3] - palette[:, None]).square().sum(-1).argmin(-1)
        # A mildly sparse initial mask exercises both binary values from the start.
        # The initial grid alpha is only a spatial seed, never an independent bitstream.
        # Dense-grid initialization can quantize all seed alpha values to the
        # same code. Rank with a deterministic spatial tie-break so both binary
        # values exist even in a constant seed; each block starts with 4 zeros.
        tie_order = (
            (
                torch.arange(16, device=image.device)[None] * 5
                + torch.arange(len(source), device=image.device)[:, None] * 7
            )
            % 16
        ).argsort(dim=1)
        rank = source[..., 3].gather(1, tie_order).argsort(dim=1, stable=True)
        order = tie_order.gather(1, rank)
        transparent = torch.zeros_like(source[..., 3], dtype=torch.bool)
        transparent.scatter_(1, order[:, :4], True)
        selected = torch.where(transparent, 3, selected)
        logits = F.one_hot(selected, 4).float() * 0.25
        return cls(ends, logits, h, w, c)

    def _ascending_endpoints(self):
        limits = self.endpoints.new_tensor([31, 63, 31])
        codes = ste_round(self.endpoints.clamp(min=0).minimum(limits))
        order = pack565(codes.detach()).argsort(dim=1, stable=True)
        return codes.gather(1, order[..., None].expand(-1, -1, 3))

    def _legal_endpoints(self):
        low_high = self._ascending_endpoints()
        words = pack565(low_high.detach())
        # Four-color mode requires strict > even for a constant block.
        hi = torch.maximum(words[:, 1], words[:, 0] + 1).clamp(max=65535)
        lo = torch.minimum(words[:, 0], hi - 1).clamp(min=0)
        opaque = unpack565(torch.stack((hi, lo), 1)).to(low_high)
        opaque = _ExactSTE.apply(low_high.flip(1), opaque)
        mode = self.modes.argmax(-1).bool()
        return torch.where(mode[:, None, None], opaque, low_high)

    def _palettes(self):
        low_high = self._ascending_endpoints()
        linear = low_high * low_high.new_tensor([255 / 31, 255 / 63, 255 / 31])
        rgb = _ExactSTE.apply(linear, expanded(low_high.detach()))
        lo, hi = rgb.unbind(1)
        midpoint = _ExactSTE.apply((lo + hi) / 2, torch.floor((lo + hi).detach() / 2))
        alpha_rgb = torch.stack((lo, hi, midpoint, torch.zeros_like(lo)), 1) / 255
        alpha = alpha_rgb.new_tensor([1, 1, 1, 0]).expand(len(lo), -1)
        punchthrough = torch.cat((alpha_rgb, alpha[..., None]), -1)
        # Use the same endpoint quantization/crossing contract as the exporter.
        words = pack565(low_high.detach())
        high_word = torch.maximum(words[:, 1], words[:, 0] + 1).clamp(max=65535)
        low_word = torch.minimum(words[:, 0], high_word - 1).clamp(min=0)
        exact = expanded(unpack565(torch.stack((high_word, low_word), 1)))
        opaque_rgb = _ExactSTE.apply(rgb.flip(1), exact)
        a, b = opaque_rgb.unbind(1)
        mid2 = _ExactSTE.apply((2 * a + b) / 3, torch.floor((2 * a + b).detach() / 3))
        mid3 = _ExactSTE.apply((a + 2 * b) / 3, torch.floor((a + 2 * b).detach() / 3))
        opaque = torch.stack((a, b, mid2, mid3), 1) / 255
        opaque = torch.cat((opaque, torch.ones_like(opaque[..., :1])), -1)
        return torch.stack((punchthrough, opaque), 1)

    def probabilities(self):
        # Float32 keeps the surrogate stable even when the network uses fp16.
        temp = self.temperature.clamp_min(0.1)
        return (self.weights / temp).softmax(-1), (self.modes / temp).softmax(-1)

    def decode_image(self):
        palettes = self._palettes()
        rows = torch.arange(len(palettes), device=palettes.device)
        hard_palette = palettes[rows, self.modes.argmax(-1)]
        hard_pixels = hard_palette.gather(1, self.weights.argmax(-1)[..., None].expand(-1, -1, 4))
        # Endpoint gradients follow the selected legal colors. Categorical
        # gradients use the JOINT soft expectation: separate hard STE factors
        # can trap an opaque block with selector!=3 because neither factor on
        # its own exposes alpha. The joint surrogate lets it discover both.
        pixels = hard_pixels
        if torch.is_grad_enabled():
            selectors, modes = self.probabilities()
            soft_palette = (palettes.detach() * modes[:, :, None, None]).sum(1)
            soft_pixels = (soft_palette[:, None] * selectors[..., None]).sum(2)
            pixels = _ExactSTE.apply(hard_pixels + soft_pixels, hard_pixels.detach())
        h, w = self._image_shape
        bh, bw = math.ceil(h / 4), math.ceil(w / 4)
        return (
            pixels.reshape(-1, bh, bw, 4, 4, 4)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, bh * 4, bw * 4, 4)[:, :h, :w]
            .permute(1, 2, 0, 3)
            .reshape(h, w, self.channels)
        )

    def binary_usage_loss(self):
        """Weak anti-collapse + confidence prior; no independent binary texture.

        Penalize each surface if its hard transparent fraction leaves [.1,.9].
        Also sharpen the joint mode/selector decision as temperature anneals.
        """
        selectors, modes = self.probabilities()
        zero_prob = modes[:, :1] * selectors[..., 3]
        surfaces = zero_prob.reshape(self.channels // 4, -1)
        hard_zero = ((self.modes.argmax(-1) == 0)[:, None] & (self.weights.argmax(-1) == 3)).to(
            zero_prob
        )
        # Evaluate occupancy on the actual exported decisions; retain gradients
        # through joint selector/mode probabilities, including collapsed masks.
        occupancy = _ExactSTE.apply(zero_prob, hard_zero).reshape(self.channels // 4, -1).mean(-1)
        balance = F.relu(0.1 - occupancy).square() + F.relu(occupancy - 0.9).square()
        confidence = (surfaces * (1 - surfaces)).mean(-1)
        return (balance + 0.01 * confidence).mean()

    @torch.no_grad()
    def usage_stats(self):
        h, w = self._image_shape
        bh, bw = math.ceil(h / 4), math.ceil(w / 4)
        punchthrough = self.modes.argmax(-1) == 0
        transparent = punchthrough[:, None] & (self.weights.argmax(-1) == 3)
        transparent = (transparent.reshape(-1, bh, bw, 4, 4).permute(0, 1, 3, 2, 4)
                       .reshape(-1, bh * 4, bw * 4)[:, :h, :w])
        return {
            "alpha_zero_fraction": transparent.float().mean(),
            "punchthrough_block_fraction": punchthrough.float().mean(),
        }

    @torch.no_grad()
    def clamp_parameters(self):
        limits = self.endpoints.new_tensor([31, 63, 31])
        self.endpoints.copy_(self.endpoints.clamp(min=0).minimum(limits))
        # Center categorical logits without changing their decisions/probabilities.
        # Keeping continuous residuals is essential to categorical learning.
        for logits in (self.weights, self.modes):
            logits.sub_(logits.mean(-1, keepdim=True))
            logits.clamp_(-12, 12)

    @torch.no_grad()
    def export_blocks(self):
        ends = pack565(self._legal_endpoints()).cpu().numpy().astype("<u2")
        selectors = self.weights.argmax(-1).long()
        bits = (
            (selectors << (2 * torch.arange(16, device=selectors.device)))
            .sum(-1)
            .cpu()
            .numpy()
            .astype("<u4")
        )
        packed = np.empty(len(ends), dtype=np.dtype([("a", "<u2"), ("b", "<u2"), ("bits", "<u4")]))
        packed["a"], packed["b"], packed["bits"] = ends[:, 0], ends[:, 1], bits
        data = packed.tobytes()
        h, w = self._image_shape
        stride = math.ceil(h / 4) * math.ceil(w / 4) * 8
        return [data[i : i + stride] for i in range(0, len(data), stride)]

    @torch.no_grad()
    def decode_cpu_image(self):
        h, w = self._image_shape
        surfaces = [decode_blocks(data, w, h, rgba=True) for data in self.export_blocks()]
        return torch.from_numpy(np.concatenate(surfaces, -1).copy()).float() / 255

    @classmethod
    def from_state(cls, state, prefix, device):
        version = int(state[prefix + "bc1_alpha_version"])
        if version != 1:
            raise ValueError(f"Unsupported BC1 RGBA version: {version}")
        h, w, c = state[prefix + "shape"].tolist()
        return cls(
            state[prefix + "endpoints"].to(device),
            state[prefix + "weights"].to(device),
            h,
            w,
            c,
            state[prefix + "modes"].to(device),
        )
