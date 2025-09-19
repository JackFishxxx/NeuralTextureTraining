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
