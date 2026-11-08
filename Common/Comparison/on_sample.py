"""Portable affine Mobile decoder for the DX12 on-sample quality renderer.

The textures contain the actual deployment-decoded ASTC/BC latent values.
Filtering happens in HLSL before decoding; no material mip raster is sampled.
This is a quality reference, not an ASTC hardware/performance benchmark on PC.
"""
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import torch


class NeuralField(IntEnum):
    ENABLED = 0
    GRID_COUNT = 1
    FEATURE_COUNT = 2
    OUTPUT_COUNT = 3
    MIP_COUNT = 4
    ACTIVATION = 5
    DIRECT_DIFFUSE = 6
    REFERENCE_WIDTH = 7
    REFERENCE_HEIGHT = 8
    QUANT_SCALE = 9
    QUANT_MIN = 10
    MIP0_ONLY = 11
    VERSION = 12
    OUTPUT_LAYOUT = 16  # (start, count) pairs in material channel order
    GRID_CHANNELS = 32
    GRID_LOD_OFFSETS = 40


PROTOCOL_VERSION = 2
HEADER_WORDS = 64
MAX_GRIDS = 6
MAX_GRID_CHANNELS = 4
MAX_OUTPUTS = 12


def protocol_defines():
    """Single source for the Python header and native C++/HLSL build constants."""
    fields = {field.name: int(field) for field in NeuralField}
    return dict(fields, PROTOCOL_VERSION=PROTOCOL_VERSION, HEADER_WORDS=HEADER_WORDS,
                MAX_GRIDS=MAX_GRIDS, MAX_GRID_CHANNELS=MAX_GRID_CHANNELS, MAX_OUTPUTS=MAX_OUTPUTS)


def neural_header():
    header = np.zeros(HEADER_WORDS, np.float32)
    header[NeuralField.VERSION] = PROTOCOL_VERSION
    return header


@dataclass
class OnSampleMaterial:
    levels: tuple
    parameters: np.ndarray
    normal_encoding: str


@torch.no_grad()
def prepare_on_sample(model):
    """Read an already deployment-decoded model; never re-encode its latent."""
    levels = []
    for grid, spec in zip(model.feature_grids, model.feature_grid_specs):
        chain = []
        for level in reversed(range(spec.n_levels)):
            start, count = spec.level_offset(level), spec.level_feature_count(level)
            edge = spec.level_resolution(level)
            value = grid.params[start:start + count].reshape(edge, edge, spec.n_features_per_level)
            chain.append(value.permute(2, 0, 1).float())
        levels.append(tuple(chain))
    inputs = model.network.n_input_dims
    # tiny-cuda-nn fills padded input lanes with one, creating an implicit
    # constant even with use_bias disabled. CutlassMLP stores the affine matrix
    # output-major with both dimensions rounded to 16. Fold padded ones into
    # the bias; reject a different layout instead of guessing a fallback.
    padded = ((inputs + 15) // 16) * 16
    params = model.network.params.detach().half().float()
    if params.numel() != padded * ((model.num_channels + 15) // 16 * 16):
        raise ValueError('Unexpected CutlassMLP affine matrix layout')
    raw = params.reshape(-1, padded)[:model.num_channels]
    matrix = raw[:, :inputs].contiguous().cpu().numpy()
    origin = raw[:, inputs:].sum(1)
    header = neural_header()
    fields = {
        NeuralField.ENABLED: 1, NeuralField.GRID_COUNT: len(levels),
        NeuralField.FEATURE_COUNT: inputs - 1, NeuralField.OUTPUT_COUNT: model.num_channels,
        NeuralField.MIP_COUNT: model.num_mips,
        NeuralField.ACTIVATION: {'hard_swish': 0, 'hard_gelu': 1, 'leaky_relu': 2}[model.output_activation],
        NeuralField.DIRECT_DIFFUSE: {'disable': 0, 'rgb': 1, 'ycocg': 2}[model.direct_diffuse_infer_mode],
        NeuralField.REFERENCE_WIDTH: model.material_lod_reference_size[1],
        NeuralField.REFERENCE_HEIGHT: model.material_lod_reference_size[0],
        NeuralField.QUANT_SCALE: model.feature_grid_specs[0].quant_step_count / (model.feature_grid_specs[0].quant_step_count - 1),
        NeuralField.QUANT_MIN: model.feature_grid_specs[0].quant_min,
        NeuralField.MIP0_ONLY: int(model.mip0_only),
    }
    for field, value in fields.items():
        header[field] = value
    for offset, name in enumerate(('diffuse', 'normal', 'roughness', 'occlusion', 'metallic', 'specular')):
        start, end = model.channel_slices.get(name, (-1, -1))
        position = NeuralField.OUTPUT_LAYOUT + offset * 2
        header[position:position + 2] = [start, end - start]
    for index, channels in enumerate(model.feature_input_channels):
        header[NeuralField.GRID_CHANNELS + index] = channels
        header[NeuralField.GRID_LOD_OFFSETS + index] = model.feature_grid_lod_offsets[index]
    bias = (model.decoder_output_bias.detach().half().float() + origin).cpu().numpy()
    parameters = np.concatenate((header, matrix.reshape(-1), bias)).astype(np.float32)
    if not np.isfinite(parameters).all():
        raise ValueError('Nonfinite on-sample decoder parameters')
    return OnSampleMaterial(tuple(levels), parameters, model.normal_encoding)
