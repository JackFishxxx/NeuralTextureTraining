"""Formats shared by conventional material compression and byte accounting."""
from dataclasses import dataclass
import math
from .astc import normalize_astc_block


@dataclass(frozen=True)
class TextureFormat:
    name: str

    def __post_init__(self):
        token = self.name.strip().upper()
        if token.startswith('ASTC'):
            token = 'ASTC' + normalize_astc_block(token[4:])
        elif token not in ('BC1', 'BC3', 'BC4', 'BC5', 'BC7'):
            raise ValueError(f'Unsupported material format {self.name!r}; use BC1/3/4/5/7 or ASTC6x6, etc.')
        object.__setattr__(self, 'name', token)

    @property
    def block(self):
        return tuple(map(int, self.name[4:].split('x'))) if self.name.startswith('ASTC') else (4, 4)

    @property
    def block_bytes(self):
        return 8 if self.name in ('BC1', 'BC4') else 16

    @property
    def channels(self):
        """Continuous material components; BC1's binary alpha is excluded."""
        return {'BC1': 3, 'BC4': 1, 'BC5': 2}.get(self.name, 4)

    def payload_bytes(self, height, width):
        bw, bh = self.block
        return math.ceil(width / bw) * math.ceil(height / bh) * self.block_bytes
