"""Prepare the compiler environment shared by the CUDA codec extensions."""
import os

import torch

from .native_build import prepare_msvc


def prepare_cuda_build():
    """Locate MSVC and Ninja, and target the current GPU unless configured."""
    prepare_msvc()
    import ninja
    os.environ['PATH'] = str(ninja.BIN_DIR) + os.pathsep + os.environ['PATH']
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
