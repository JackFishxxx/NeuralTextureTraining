"""Immutable compression recipes for FNTC feature grids."""

import re
from dataclasses import dataclass

from Common.Block_Compression.astc import normalize_astc_block


@dataclass(frozen=True)
class GridTraining:
    enabled: bool
    start_frac: float
    interval: int

    def active(self, iteration, steps):
        return self.enabled and iteration >= int(steps*self.start_frac)

    def due(self, iteration, steps):
        return self.active(iteration, steps) and (iteration-int(steps*self.start_frac)) % self.interval == 0


@dataclass(frozen=True)
class GridCompression:
    format: str = "astc"
    astc_block: str = "6x6"

    def __post_init__(self):
        if self.format not in ("astc", "bc1"):
            raise ValueError(f"Unsupported feature grid format: {self.format}")
        object.__setattr__(
            self,
            "astc_block",
            normalize_astc_block(self.astc_block) if self.format == "astc" else "6x6",
        )

    @classmethod
    def from_config(cls, cfg):
        if 'astc_block' in cfg or 'bc1_mode' in cfg:
            raise ValueError('Grid compression is configured only by format, e.g. astc6x6 or BC1')
        if not isinstance(cfg.get('format'), str):
            raise ValueError('Every feature grid requires format: astc6x6, astc4x4 or BC1')
        token = cfg['format'].strip().lower()
        match = re.fullmatch(r"astc(\d+[xX]\d+)", token)
        if match:
            block = normalize_astc_block(match.group(1))
            return cls("astc", block)
        if token == "bc1":
            return cls("bc1")
        raise ValueError(f"Unsupported feature grid format {token!r}; use astc6x6, astc4x4 or BC1")

    @property
    def tag(self):
        return f"astc_{self.astc_block}" if self.format == "astc" else "bc1"

    @property
    def block(self):
        return tuple(map(int, self.astc_block.split("x"))) if self.format == "astc" else (4, 4)

    def as_config(self):
        if self.format == "astc":
            return {"format": "astc" + self.astc_block}
        return {"format": "BC1"}
