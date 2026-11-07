"""Replay native feature preparation and material interpolation around tinycudann.

tinycudann owns its own graph lifecycle and stays outside these captures.
Latents are explicit graph inputs, preserving per-step QAT noise and codec
updates as well as gradient accumulation shared by primary/subpixel queries.
"""
import torch

from Common.cuda_graph import make_graphed_callables
from dataset import _native_query_uv


class MaterialQueryGraphs:
    def __init__(self, model, size, base_sampler):
        self.model, self.size, self.base_sampler = model, size, base_sampler
        self.base_version = base_sampler.image._version if base_sampler is not None else None
        self.graphs = {}
        self.pending = set()
        self.resolution = model.decoder_output_bias.new_tensor((size[1], size[0]))
        self.offsets = self.resolution.new_tensor(((0, 0), (1, 0), (0, 1), (1, 1)))

    def _prepare(self, uv, *params):
        model = self.model
        centers, inverse, weights = _native_query_uv(uv, self.size, self.resolution, self.offsets)
        inputs = centers if model.mip0_only else torch.cat((centers, centers.new_zeros((len(centers), 1))), 1)
        base = self.base_sampler(centers) if self.base_sampler is not None else centers.new_zeros((len(centers), 0))
        if self.base_sampler is not None:
            inputs = torch.cat((inputs, base), 1)
        features, _ = model._prepare_decoder_inputs(inputs, native_size=self.size, grid_params=params)
        return features, inverse, weights, base

    def _finish(self, output, bias, inverse, weights, base):
        values = output + bias.to(self.model.network.params.dtype)
        if not self.model.super_resolution_enable:
            values = self.model._apply_output_activation(values)
        values = values.float()
        if self.base_sampler is not None:
            values = values + base
        bounded = values.clamp(0, 1)
        values = bounded.detach() + (values - values.detach())
        return (values[inverse].reshape(len(weights), 4, -1) * weights[..., None]).sum(1)

    def _call(self, name, function, args):
        key = (name, tuple((tuple(x.shape), x.stride(), x.dtype, x.device, x.requires_grad) for x in args))
        if key in self.pending:
            return function(*args)
        graph = self.graphs.get(key)
        if graph is None:
            samples = tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in args)
            graph = make_graphed_callables(function, samples)
            self.graphs[key] = graph
        output = graph(*args)
        tensors = (output,) if isinstance(output, torch.Tensor) else output
        differentiable = [x for x in tensors if x.requires_grad]
        if differentiable:
            self.pending.add(key)
            differentiable[0].grad_fn.register_hook(lambda *unused: self.pending.discard(key))
        elif isinstance(output, tuple):
            # Frozen features still feed the decoder's weight backward. They
            # must survive another same-shape replay before that backward.
            return tuple(x.clone() for x in output)
        return output

    def __call__(self, uv, params):
        features, inverse, weights, base = self._call('prepare', self._prepare, (uv, *params))
        # make_graphed_callables attaches every floating output to its custom
        # autograd node. Query metadata and the fixed SR base have no gradient.
        inverse, weights, base = inverse.detach(), weights.detach(), base.detach()
        output = self.model.network(features)
        return self._call('finish', self._finish,
                          (output, self.model.decoder_output_bias, inverse, weights, base))
