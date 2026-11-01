"""A single surface packing contract for encoding, deployment and byte accounting."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SurfacePlan:
    channels: tuple[int, ...]
    block: tuple[int, int]
    block_bytes: int

    def payload_bytes(self, height, width):
        bw, bh = self.block
        return math.ceil(width / bw) * math.ceil(height / bh) * self.block_bytes


def surface_plan(spec, recipe, direct_diffuse=False):
    channels = spec.n_features_per_level
    if direct_diffuse:
        groups = [tuple(range(min(3, channels)))]
        groups.extend(tuple(range(i, min(i + 4, channels))) for i in range(3, channels, 4))
    else:
        capacity = 4
        groups = [
            tuple(range(i, min(i + capacity, channels))) for i in range(0, channels, capacity)
        ]
    return tuple(
        SurfacePlan(group, recipe.block, 16 if recipe.format == "astc" else 8) for group in groups
    )


def surface_filename(index, surface, recipe, count):
    if recipe.format == "bc1":
        return f"feature_grid_{index}_bc1_{surface}.dds"
    suffix = "" if count == 1 else f"_{surface}"
    return f"feature_grid_{index}{suffix}.astc"
