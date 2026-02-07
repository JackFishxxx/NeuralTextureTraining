from torchtyping import TensorType

import math
import torch
import numpy as np

def pack_features(
    features: TensorType["num_features"],
    quantize_bits: int=4,
    save_bits: int=8,
) -> TensorType["num_variables"]:
    
    if save_bits == 8:
        save_dtype = torch.uint8
    elif save_bits == 16:
        save_dtype = torch.uint16
    elif save_bits == 32:
        save_dtype = torch.uint32
    elif save_bits == 64:
        save_dtype = torch.uint64
    else:
        raise ValueError("Save bits should be 8, 16, or 32.")

    # float -> int, [- (N_k-1) / 2 * Q_k, 1 / 2] -> [0, 2 ** save_model.quantize_bits - 1]
    N_k = 2 ** quantize_bits
    Q_k = 1 / N_k
    min_quantize_range = - (N_k - 1) / 2 * Q_k
    features = (features + (-min_quantize_range)) * (2 ** quantize_bits)

    device = features.device
    num_features = features.shape[0]
    feat_per_var = save_bits // quantize_bits
    num_variables = math.ceil(num_features / feat_per_var)

    # since torch.uint16, torch.uint32 only has limited support
    # so we use torch.int64 to init tensor
    # after computation, we convert this tensor to torch.uint8
    feature_tensor = torch.zeros([num_variables * feat_per_var], dtype=torch.int64, device=device)
    save_tensor = torch.zeros([num_variables], dtype=torch.int64, device=device)
    
    feature_tensor[:num_features] = features
    for i in range(feat_per_var):
        shift_bits = (feat_per_var - 1 - i) * quantize_bits
        save_tensor[:] += feature_tensor[i::feat_per_var] << shift_bits
    
    save_tensor = save_tensor.to(save_dtype)

    return save_tensor


def unpack_features(
    save_tensor: TensorType["num_variables"],
    quantize_bits: int=4,
    save_bits: int=8,
) -> TensorType["num_features"]:

    device = save_tensor.device
    num_variables = save_tensor.shape[0]
    feat_per_var = save_bits // quantize_bits
    num_features = num_variables * feat_per_var

    # since torch.uint16, torch.uint32 only has limited support
    # so we use torch.int64 to init tensor
    # after computation, we convert this tensor to torch.uint8
    save_tensor = save_tensor.to(torch.int64)
    feature_tensor = torch.zeros([num_features], dtype=torch.int64, device=device)
    
    for i in range(feat_per_var):
        shift_bits = (feat_per_var - 1 - i) * quantize_bits
        bit_mask = 15  # 0000 0000 0000 1111
        feature_tensor[i::feat_per_var] = (save_tensor & (bit_mask << shift_bits)) >> shift_bits
    
    # int -> float, [0, 2 ** save_model.quantize_bits - 1] -> [- (N_k-1) / 2 * Q_k, 1 / 2]
    N_k = 2 ** quantize_bits
    Q_k = 1 / N_k
    min_quantize_range = - (N_k - 1) / 2 * Q_k
    features = feature_tensor / (2 ** quantize_bits) + min_quantize_range

    return features

import struct

def write_dds_r8g8b8a8(filepath: str, width: int, height: int, data: np.ndarray):
    """
    Write a DDS file with R8G8B8A8 format.

    Args:
        filepath: Output DDS file path
        width: Texture width
        height: Texture height
        data: RGBA data as numpy array with shape [height, width, 4] and dtype uint8
    """
    # DDS header constants
    DDS_MAGIC = 0x20534444  # "DDS "
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_PITCH = 0x8
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000
    DDSD_LINEARSIZE = 0x80000
    DDSD_DEPTH = 0x800000

    DDPF_ALPHAPIXELS = 0x1
    DDPF_FOURCC = 0x4
    DDPF_RGB = 0x40

    DDSCAPS_TEXTURE = 0x1000

    # Calculate pitch (bytes per row)
    pitch = width * 4  # 4 bytes per pixel for RGBA8

    # DDS_PIXELFORMAT structure
    pf_size = 32
    pf_flags = DDPF_RGB | DDPF_ALPHAPIXELS
    pf_fourcc = 0
    pf_rgb_bit_count = 32
    pf_r_bit_mask = 0x000000FF
    pf_g_bit_mask = 0x0000FF00
    pf_b_bit_mask = 0x00FF0000
    pf_a_bit_mask = 0xFF000000

    with open(filepath, 'wb') as f:
        # Write magic number
        f.write(struct.pack('<I', DDS_MAGIC))

        # Write DDS_HEADER
        f.write(struct.pack('<I', 124))  # dwSize
        f.write(struct.pack('<I', DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PITCH | DDSD_PIXELFORMAT))  # dwFlags
        f.write(struct.pack('<I', height))  # dwHeight
        f.write(struct.pack('<I', width))  # dwWidth
        f.write(struct.pack('<I', pitch))  # dwPitchOrLinearSize
        f.write(struct.pack('<I', 0))  # dwDepth
        f.write(struct.pack('<I', 1))  # dwMipMapCount
        f.write(struct.pack('<I', 0) * 11)  # dwReserved1[11]

        # Write DDS_PIXELFORMAT
        f.write(struct.pack('<I', pf_size))
        f.write(struct.pack('<I', pf_flags))
        f.write(struct.pack('<I', pf_fourcc))
        f.write(struct.pack('<I', pf_rgb_bit_count))
        f.write(struct.pack('<I', pf_r_bit_mask))
        f.write(struct.pack('<I', pf_g_bit_mask))
        f.write(struct.pack('<I', pf_b_bit_mask))
        f.write(struct.pack('<I', pf_a_bit_mask))

        # Write DDS_CAPS
        f.write(struct.pack('<I', DDSCAPS_TEXTURE))  # dwCaps
        f.write(struct.pack('<I', 0))  # dwCaps2
        f.write(struct.pack('<I', 0))  # dwCaps3
        f.write(struct.pack('<I', 0))  # dwCaps4
        f.write(struct.pack('<I', 0))  # dwReserved2

        # Write pixel data (row by row)
        f.write(data.tobytes())