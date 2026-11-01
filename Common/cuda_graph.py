"""Keep cyclic garbage collection from destroying old CUDA graphs during capture."""
from contextlib import contextmanager
import gc
import torch


@contextmanager
def graph_capture_lifetime():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def make_graphed_callables(function, samples):
    with graph_capture_lifetime():
        return torch.cuda.make_graphed_callables(function, samples)
