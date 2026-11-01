"""Build the optional fused BC1 decoder; cache binaries inside the project."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import warnings

import torch

from Common.cuda_build import prepare_cuda_build


@lru_cache(maxsize=1)
def native_extension():
    root = Path(__file__).resolve().parent / 'native'
    sources = [root / 'bc1.cpp', root / 'bc1.cu']
    try:
        prepare_cuda_build()
        from torch.utils.cpp_extension import load

        digest = hashlib.sha256(str(root).encode() + torch.__version__.encode()
                                + b''.join(p.read_bytes() for p in sources)).hexdigest()[:12]
        name = 'fntc_bc1_decode_' + digest
        cache = root.parents[2] / 'tools' / 'bc1_cuda_cache' / name
        cache.mkdir(parents=True, exist_ok=True)
        return load(name=name, sources=[str(p) for p in sources], build_directory=str(cache),
                    extra_cflags=['/O2'] if os.name == 'nt' else ['-O3'],
                    extra_cuda_cflags=['-O3', '--fmad=false'], verbose=False)
    except (OSError, RuntimeError, ImportError) as exc:
        detail = str(exc)
        if len(detail) > 1800:
            errors = [line for line in detail.splitlines() if 'error' in line.lower() or 'FAILED:' in line]
            detail = '\n'.join(errors[:12]) if errors else detail[-1800:]
        warnings.warn(f'Fused BC1 decoder unavailable; using PyTorch decoder: {detail}', RuntimeWarning)
        return None
