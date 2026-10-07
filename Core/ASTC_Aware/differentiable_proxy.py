"""Train legal LDR ASTC endpoint symbols and weight grids, initialized by astcenc.

Forward uses the official quantization/decimation tables and UNORM8 integer
decoding. Backward uses surrogate gradients for quantization and packed endpoint
transforms. Block layouts remain fixed between validated CPU projections.
"""

from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
import torch

from .build import native_extension
from feature_grid import quantize_feature_ste


def encode_seed(codec, rgba):
    """Run the full CPU encoder once; preserve its actual 128-bit block choices."""
    if codec.astc_block != "6x6":
        raise ValueError("ASTC differentiable proxy currently supports 6x6 LDR blocks")
    with tempfile.TemporaryDirectory(prefix="fntc_astc_seed_") as directory:
        source, target = Path(directory) / "input.png", Path(directory) / "seed.astc"
        Image.fromarray(np.asarray(rgba, dtype=np.uint8)).save(source)
        subprocess.run(
            [
                codec.ensure_executable(),
                "-cl",
                str(source),
                str(target),
                "6x6",
                "-" + codec.astcenc_quality,
            ],
            check=True,
            capture_output=True,
        )
        return target.read_bytes()


class _DecodeASTC(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, endpoints, weights, metadata, partitions, indices, coefficients, colors, weight_lut
    ):
        ctx.save_for_backward(
            endpoints, weights, metadata, partitions, indices, coefficients, colors, weight_lut
        )
        return native_extension().decode_cuda(
            endpoints, weights, metadata, partitions, indices, coefficients, colors, weight_lut
        )

    @staticmethod
    def backward(ctx, gradient):
        ge, gw = native_extension().backward_cuda(gradient.contiguous(), *ctx.saved_tensors)
        return ge, gw, None, None, None, None, None, None


class ASTCProxyGrid(torch.nn.Module):
    def __init__(self, tensors, height, width, device):
        super().__init__()
        self.endpoints = torch.nn.Parameter(tensors[0].to(device))
        self.weights = torch.nn.Parameter(tensors[1].to(device))
        for name, value in zip(
            ("metadata", "partitions", "indices", "coefficients", "colors", "weight_lut"),
            tensors[2:],
        ):
            self.register_buffer(name, value.to(device))
        self.register_buffer("size", torch.tensor([height, width], device=device, dtype=torch.long))
        # Dimensions are fixed; reading the CUDA buffer each forward synchronizes.
        self._image_shape = (height, width)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._image_shape = tuple(self.size.tolist())

    @classmethod
    def from_astc(cls, data, device="cuda"):
        if data[:4] != bytes.fromhex("13aba15c") or data[4:7] != bytes([6, 6, 1]):
            raise ValueError("Expected a 2D 6x6 ASTC file")
        width, height, depth = [int.from_bytes(data[i : i + 3], "little") for i in (7, 10, 13)]
        if depth != 1 or width < 1 or height < 1:
            raise ValueError("Invalid ASTC dimensions")
        physical = torch.from_numpy(np.frombuffer(data[16:], dtype=np.uint8).copy()).reshape(-1, 16)
        if len(physical) != ((width + 5) // 6) * ((height + 5) // 6):
            raise ValueError("ASTC payload does not match dimensions")
        return cls(native_extension().import_blocks(physical), height, width, device)

    @classmethod
    def from_state(cls, state, prefix, device):
        names = (
            "endpoints",
            "weights",
            "metadata",
            "partitions",
            "indices",
            "coefficients",
            "colors",
            "weight_lut",
        )
        tensors = [state[prefix + name] for name in names]
        height, width = state[prefix + "size"].tolist()
        return cls(tensors, height, width, device)

    def decode_blocks(self, block_ids=None):
        def select(value):
            return value if block_ids is None else value.index_select(0, block_ids)

        values = [
            select(getattr(self, name)).contiguous()
            for name in (
                "endpoints",
                "weights",
                "metadata",
                "partitions",
                "indices",
                "coefficients",
            )
        ]
        return _DecodeASTC.apply(*values, self.colors, self.weight_lut)

    @torch.no_grad()
    def branch_raw_bounds_table(self):
        if not hasattr(self, "_branch_raw_bounds"):
            raw = torch.arange(256, device=self.colors.device)[None, :]
            bounds = []
            for flag in (0, 64, 128, 192):
                allowed = (self.colors & 192) == flag
                bounds.append(
                    torch.stack(
                        (
                            torch.where(allowed, raw, 256).amin(1),
                            torch.where(allowed, raw, -1).amax(1),
                        ),
                        dim=-1,
                    )
                )
            self._branch_raw_bounds = torch.stack(bounds, dim=1).to(torch.int32)
        return self._branch_raw_bounds

    @torch.no_grad()
    def endpoint_branch_bounds(self, block_ids=None):
        """Keep quantized packed flags inside the branch assumed by backward.

        Delta endpoint bytes multiplex continuous payload and discrete flags.
        Clamp the raw input to the connected quantizer preimage of those flags.
        Plain endpoints are unrestricted; modes/partitions remain fixed.
        """
        endpoints = self.endpoints if block_ids is None else self.endpoints[block_ids]
        metadata = self.metadata if block_ids is None else self.metadata[block_ids]
        codes = torch.round(endpoints * 255).clamp(0, 255).long()
        quant = (metadata[:, 3] - 4).clamp(0, 16).long()
        lookup = self.colors[quant]
        values = lookup.gather(1, codes)
        formats = metadata[:, 6:10].repeat_interleave(8, dim=1)
        positions = torch.arange(32, device=codes.device)[None, :] % 8
        active = metadata[:, 0:1] == 3
        masks = torch.zeros_like(codes)
        masks = torch.where((formats == 1) & (positions == 1), 192, masks)
        masks = torch.where((formats == 5) & (positions % 2 == 1), 192, masks)
        masks = torch.where(((formats == 9) | (formats == 13)) & (positions % 2 == 1), 192, masks)
        masks = torch.where(active, masks, 0)
        flags = values & masks
        # Only four flag regions exist. Build compact lookup once, instead of a
        # [blocks,32,256] temporary on every optimizer step (large for a full grid).
        bounds = self.branch_raw_bounds_table()[quant[:, None], flags // 64]
        lower = torch.where(masks != 0, bounds[:, :, 0], 0)
        upper = torch.where(masks != 0, bounds[:, :, 1], 255)
        return ((lower.float() - 0.49) / 255).clamp(0, 1), ((upper.float() + 0.49) / 255).clamp(
            0, 1
        )

    @torch.no_grad()
    def endpoint_blue_branches(self, endpoints, metadata):
        quant = (metadata[:, 3] - 4).clamp(0, 16).long()
        values = self.colors[
            quant[:, None], (endpoints * 255).round().clamp(0, 255).long()
        ].reshape(-1, 4, 8)
        delta = (values[:, :, [1, 3, 5]].long() >> 1) & 63
        delta = torch.where(delta >= 32, delta - 64, delta).sum(2)
        direct = values[:, :, [0, 2, 4]].sum(2) > values[:, :, [1, 3, 5]].sum(2)
        formats = metadata[:, 6:10]
        return (
            (metadata[:, 0:1] == 3)
            & (torch.where((formats == 9) | (formats == 13), delta < 0, direct))
            & ((formats == 8) | (formats == 9) | (formats == 12) | (formats == 13))
        )

    @torch.no_grad()
    def quantized_symbols(self, block_ids=None):
        """Return actual legal unpacked endpoint/weight values for diagnostics."""
        e = self.endpoints if block_ids is None else self.endpoints[block_ids]
        w = self.weights if block_ids is None else self.weights[block_ids]
        m = self.metadata if block_ids is None else self.metadata[block_ids]
        q = (m[:, 3] - 4).clamp(0, 16).long()[:, None]
        endpoint = self.colors[q, (e * 255).round().clamp(0, 255).long()]
        constant = m[:, 0] == 2
        endpoint[constant, :4] = (
            (e[constant, :4] * 65535).round().clamp(0, 65535).long().to(endpoint.dtype)
        )
        weight = self.weight_lut[m[:, 4].long()[:, None], (w * 64).round().clamp(0, 64).long()]
        return endpoint, weight

    def decode_image(self):
        height, width = self._image_shape
        by, bx = (height + 5) // 6, (width + 5) // 6
        return (
            self.decode_blocks()
            .reshape(by, bx, 6, 6, 4)
            .permute(0, 2, 1, 3, 4)
            .reshape(by * 6, bx * 6, 4)[:height, :width]
        )

    @torch.no_grad()
    def clamp_parameters(self):
        self.endpoints.clamp_(0, 1)
        self.weights.clamp_(0, 1)

    @torch.no_grad()
    def export_astc(self):
        height, width = self._image_shape
        physical = native_extension().export_blocks(
            self.endpoints.detach().cpu().contiguous(),
            self.weights.detach().cpu().contiguous(),
            self.metadata.cpu().contiguous(),
        )
        header = (
            bytes.fromhex("13aba15c")
            + bytes([6, 6, 1])
            + width.to_bytes(3, "little")
            + height.to_bytes(3, "little")
            + (1).to_bytes(3, "little")
        )
        return header + physical.numpy().tobytes()

    @torch.no_grad()
    def decode_cpu_image(self):
        data = self.export_astc()
        physical = torch.from_numpy(np.frombuffer(data[16:], dtype=np.uint8).copy()).reshape(-1, 16)
        blocks = native_extension().decode_cpu(physical)
        height, width = self.size.tolist()
        by, bx = (height + 5) // 6, (width + 5) // 6
        return (
            blocks.reshape(by, bx, 6, 6, 4)
            .permute(0, 2, 1, 3, 4)
            .reshape(by * 6, bx * 6, 4)[:height, :width]
        )


def initialize_astc_proxy(model, codec):
    """Create persistent block parameters. Never refit them on each training step."""
    from feature_grid import feature_tensor_to_int

    result = torch.nn.ModuleList()
    for i, spec in enumerate(model.feature_grid_specs):
        if spec.n_features_per_level != 4 or model.direct_diffuse_enabled:
            raise ValueError(
                "ASTC differentiable proxy requires four-channel grids and direct diffuse disabled"
            )
        offset, count, res = spec.highest_level_slice()
        codes = feature_tensor_to_int(
            model.feature_grids[i].params.detach()[offset : offset + count], spec
        )
        image = (
            torch.round(codes.float() * (255 / (spec.quant_step_count - 1)))
            .byte()
            .reshape(res, res, 4)
            .cpu()
            .numpy()
        )
        result.append(ASTCProxyGrid.from_astc(encode_seed(codec, image), model.device))
    return result


class ASTCBlockAdam:
    """Adam with per-block counters, updating only blocks sampled on this step.

    Standard dense Adam keeps applying old momentum to unsampled rows. Here
    each sampled block receives one optimizer step; untouched rows and moments
    remain unchanged. State is training-only, like the existing model optimizer.
    """

    def __init__(
        self,
        grids,
        endpoint_lr,
        weight_lr,
        quantization_aware=False,
        search_trials=3,
        fused=False,
        packed_pair_search=False,
    ):
        self.grids = grids
        self.lrs = (endpoint_lr, weight_lr)
        self.state = {}
        self.quantization_aware = quantization_aware
        self.search_trials = search_trials
        self.fused = fused
        self.packed_pair_search = packed_pair_search
        self._search_count = 0
        self.last_stats = {}

    def zero_grad(self):
        for grid in self.grids:
            grid.endpoints.grad = None
            grid.weights.grad = None

    @torch.no_grad()
    def reset_rows(self, index, ids):
        for parameter in (self.grids[index].endpoints, self.grids[index].weights):
            if parameter in self.state:
                for value in self.state[parameter]:
                    value[ids] = 0

    @torch.no_grad()
    def _state_for(self, parameter):
        if parameter not in self.state:
            self.state[parameter] = (
                torch.zeros_like(parameter),
                torch.zeros_like(parameter),
                torch.zeros(len(parameter), device=parameter.device, dtype=torch.long),
            )
        return self.state[parameter]

    @torch.no_grad()
    def step_full_grid(self):
        """Fused guarded Adam; skip rows without gradients, as in the full-grid path."""
        if not self.fused:
            raise ValueError("Fused optimizer is disabled")
        for grid in self.grids:
            if grid.endpoints.grad is None or grid.weights.grad is None:
                raise ValueError("Full-grid fused Adam requires endpoint and weight gradients")
            em, ev, ec = self._state_for(grid.endpoints)
            wm, wv, wc = self._state_for(grid.weights)
            native_extension().adam_full_grid_cuda(
                grid.endpoints,
                grid.weights,
                grid.endpoints.grad,
                grid.weights.grad,
                em,
                ev,
                ec,
                wm,
                wv,
                wc,
                grid.metadata,
                grid.colors,
                grid.branch_raw_bounds_table(),
                *self.lrs,
            )
            # Raw native writes do not increment PyTorch's tensor version counters.
            torch.autograd.graph.increment_version((grid.endpoints, grid.weights))
        self.last_stats = {}

    @torch.no_grad()
    def _adam_step(self, touched):
        for index, ids in touched.items():
            grid = self.grids[index]
            ids = ids.unique()
            valid = torch.ones(len(ids), device=ids.device, dtype=torch.bool)
            for parameter in (grid.endpoints, grid.weights):
                if parameter.grad is not None:
                    valid &= torch.isfinite(parameter.grad[ids]).all(1)
            ids = ids[valid]
            if not len(ids):
                continue
            lower, upper = grid.endpoint_branch_bounds(ids)
            for parameter, lr in zip((grid.endpoints, grid.weights), self.lrs):
                if parameter.grad is None:
                    continue
                moment, variance, count = self._state_for(parameter)
                gradient = parameter.grad[ids]
                m = moment[ids] * 0.9 + gradient * 0.1
                v = variance[ids] * 0.999 + gradient.square() * 0.001
                steps = count[ids] + 1
                mhat = m / (1 - torch.pow(0.9, steps.float()))[:, None]
                vhat = v / (1 - torch.pow(0.999, steps.float()))[:, None]
                change = -lr * mhat / (vhat.sqrt() + 1e-8)
                updated = parameter[ids] + change
                if parameter is grid.endpoints:
                    updated = torch.minimum(torch.maximum(updated, lower), upper)
                    previous = parameter[ids]
                    metadata = grid.metadata[ids]
                    changed = grid.endpoint_blue_branches(
                        previous, metadata
                    ) != grid.endpoint_blue_branches(updated, metadata)
                    # RGB sum selects a discontinuous swap/uncontract branch in
                    # addition to individual packed flags. Its crossing invalidates
                    # the continuous Jacobian; leave that partition for exact search.
                    updated = torch.where(changed.repeat_interleave(8, 1), previous, updated)
                else:
                    updated = updated.clamp(0, 1)
                parameter[ids] = updated
                moment[ids] = m
                variance[ids] = v
                count[ids] = steps

    @torch.no_grad()
    def _packed_pair_proposal(self, grid, ids):
        gradient = grid.endpoints.grad
        if gradient is None:
            result = torch.zeros((len(ids), 4), device=ids.device)
        else:
            result = native_extension().packed_pair_proposals_cuda(
                grid.endpoints,
                gradient,
                grid.metadata,
                grid.colors,
                grid.branch_raw_bounds_table(),
                ids.contiguous(),
            )
        return {
            "ids": ids,
            "columns": result[:, 0].long(),
            "values": result[:, 1],
            "second_values": result[:, 2],
            "gains": result[:, 3],
            "paired": True,
        }

    @torch.no_grad()
    def _neighbor_proposal(self, grid, ids, family=None):
        """One gradient-ranked legal neighboring symbol per sampled block."""
        metadata = grid.metadata[ids]
        if family != "weights":
            lower, upper = grid.endpoint_branch_bounds(ids)
        candidates = []
        for name, scale in (("endpoints", 255), ("weights", 64)):
            if family is not None and name != family:
                continue
            parameter = getattr(grid, name)
            current = parameter[ids]
            gradient = torch.zeros_like(current) if parameter.grad is None else parameter.grad[ids]
            if name == "endpoints":
                lookup = grid.colors[(metadata[:, 3] - 4).clamp(0, 16).long()]
                slot = torch.arange(32, device=ids.device)[None, :]
                part = slot // 8
                formats = metadata[:, 6:10].repeat_interleave(8, dim=1)
                used = (part < metadata[:, 1:2]) & ((slot % 8) < 2 * (formats // 4) + 2)
            else:
                lookup = grid.weight_lut[metadata[:, 4].long()]
                slot = torch.arange(64, device=ids.device)[None, :]
                used = (slot % 32) < metadata[:, 5:6]
                # Single-plane weight grids can contain up to 64 weights.
                used = torch.where(metadata[:, 2:3] < 0, slot < metadata[:, 5:6], used)
            raw = torch.arange(scale + 1, device=ids.device)[None, None, :]
            codes = torch.round(current * scale).clamp(0, scale).long()
            values = lookup.gather(1, codes)
            table = lookup[:, None, :]
            direction = gradient.sign()
            allowed = torch.where(
                direction[:, :, None] < 0, table > values[:, :, None], table < values[:, :, None]
            )
            allowed &= (
                (direction[:, :, None] != 0) & used[:, :, None] & (metadata[:, 0:1, None] == 3)
            )
            if name == "endpoints":
                allowed &= (raw / scale >= lower[:, :, None]) & (raw / scale <= upper[:, :, None])
            distance = (table - values[:, :, None]).abs()
            distance = torch.where(allowed, distance, 10000)
            best, position = distance.min(dim=2)
            selected = lookup.gather(1, position)
            same = (table == selected[:, :, None]) & allowed
            # Use the middle of the quantizer preimage, not a fragile boundary value.
            lo = torch.where(same, raw, scale + 1).amin(dim=2)
            hi = torch.where(same, raw, -1).amax(dim=2)
            value = (lo.float() + hi.float()) / (2 * scale)
            # Score actual symbol movement, independent of the shadow parameter
            # position inside its quantizer preimage. Acceptance uses real loss.
            gain = -gradient * (selected.float() - values.float()) / scale
            gain = torch.where((best < 10000) & torch.isfinite(gain), gain, torch.zeros_like(gain))
            candidates.append((value, gain))
        gains = torch.cat([candidate[1] for candidate in candidates], dim=1)
        gain, column = gains.max(dim=1)
        values = (
            torch.cat([candidate[0] for candidate in candidates], dim=1)
            .gather(1, column[:, None])
            .squeeze(1)
        )
        if family == "weights":
            column = column + 32
        return {"ids": ids, "columns": column, "values": values, "gains": gain}

    @torch.no_grad()
    def step(self, touched, objective=None, search=True):
        touched = {index: ids.unique() for index, ids in touched.items()}
        if not self.quantization_aware or not search:
            self._adam_step(touched)
            self.last_stats = {}
            return
        if objective is None:
            raise ValueError("Quantization-aware updates require an exact batch objective callback")
        before = {
            index: (
                self.grids[index].endpoints[ids].clone(),
                self.grids[index].weights[ids].clone(),
            )
            for index, ids in touched.items()
        }
        symbols_before = {
            index: self.grids[index].quantized_symbols(ids) for index, ids in touched.items()
        }

        def restore(saved):
            for index, ids in touched.items():
                self.grids[index].endpoints[ids] = saved[index][0]
                self.grids[index].weights[ids] = saved[index][1]

        def evaluate():
            loss = objective()
            loss = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
            return loss

        baseline = evaluate()
        if not np.isfinite(baseline):
            self.last_stats = {"nonfinite_objective": 1, "accepted_blocks": 0}
            return
        self._adam_step(touched)
        try:
            continuous = evaluate()
        except Exception:
            restore(before)
            raise
        continuous_accepted = np.isfinite(continuous) and continuous <= baseline
        if not continuous_accepted:
            restore(before)
            continuous = baseline
        anchors = {
            index: (
                self.grids[index].endpoints[ids].clone(),
                self.grids[index].weights[ids].clone(),
            )
            for index, ids in touched.items()
        }
        # A large weight quantization jump can dominate a tiny endpoint step's
        # linear score but overshoot. Search endpoint and weight families separately.
        families = {}
        family_counts = {}
        names = (
            ("packed_pairs", "endpoints", "weights")
            if self.packed_pair_search
            else ("endpoints", "weights")
        )
        shift = self._search_count % len(names)
        names = names[shift:] + names[:shift]
        self._search_count += 1
        family_trials = {name: 0 for name in names}
        family_accepted = {name: 0 for name in names}
        accepted = 0
        final = continuous
        trials = 0
        # Build the next family only if an earlier candidate was rejected.
        # Accepted paired moves need no large [blocks,slots,quantizer] tensors.
        for proposal_round in range(self.search_trials * len(names)):
            if trials >= self.search_trials:
                break
            restore(anchors)
            family_name = names[proposal_round % len(names)]
            if family_name not in families:
                family = {}
                for index, ids in touched.items():
                    grid = self.grids[index]
                    family[index] = (
                        self._packed_pair_proposal(grid, ids)
                        if family_name == "packed_pairs"
                        else self._neighbor_proposal(grid, ids, family=family_name)
                    )
                families[family_name] = family
                family_counts[family_name] = sum(
                    int((p["gains"] > 0).sum()) for p in family.values()
                )
            if family_counts[family_name] == 0:
                continue
            selected_count = 0
            family = families[family_name]
            for index, proposal in family.items():
                valid = torch.where(proposal["gains"] > 0)[0]
                if len(valid) == 0:
                    continue
                count = max(1, len(valid) // (2 ** (proposal_round // len(names))))
                selected = valid[torch.argsort(proposal["gains"][valid], descending=True)[:count]]
                rows = proposal["ids"][selected]
                columns = proposal["columns"][selected]
                values = proposal["values"][selected]
                grid = self.grids[index]
                if proposal.get("paired", False):
                    grid.endpoints[rows, columns] = values
                    grid.endpoints[rows, columns + 1] = proposal["second_values"][selected]
                else:
                    ep = columns < 32
                    grid.endpoints[rows[ep], columns[ep]] = values[ep]
                    grid.weights[rows[~ep], columns[~ep] - 32] = values[~ep]
                selected_count += count
            try:
                candidate = evaluate()
            except Exception:
                restore(anchors)
                raise
            trials += 1
            family_trials[family_name] += 1
            if np.isfinite(candidate) and candidate < continuous:
                accepted += selected_count
                final = candidate
                family_accepted[family_name] += selected_count
                break
        if not accepted:
            restore(anchors)
        symbol_changes = [0, 0]
        for index, ids in touched.items():
            for kind, (old, current) in enumerate(
                zip(symbols_before[index], self.grids[index].quantized_symbols(ids))
            ):
                symbol_changes[kind] += int((old != current).sum())
        self.last_stats = {
            "baseline_loss": baseline,
            "final_loss": final,
            "continuous_accepted": int(continuous_accepted),
            "proposed_blocks": sum(family_counts.values()),
            "accepted_blocks": accepted,
            "search_trials": trials,
            "endpoint_symbols_changed": symbol_changes[0],
            "weight_symbols_changed": symbol_changes[1],
        }
        for name in names:
            self.last_stats[name + "_trials"] = family_trials[name]
            self.last_stats[name + "_accepted_blocks"] = family_accepted[name]
        if self.packed_pair_search:
            self.last_stats["paired_proposed_blocks"] = family_counts.get("packed_pairs", 0)
            self.last_stats["paired_accepted_blocks"] = family_accepted["packed_pairs"]


def sample_proxy_grid(model, grid_idx, feature_grid):
    codec = model.astc_proxy[grid_idx]
    version = (codec.endpoints._version, codec.weights._version)
    key = (int(grid_idx), "full_astc")
    cached = model._qat_param_cache.get(key)
    if cached is not None and cached[0] == version:
        return cached[1]
    spec = model.feature_grid_specs[grid_idx]
    offset, count, _ = spec.highest_level_slice()
    decoded = codec.decode_image().reshape(-1)
    decoded = quantize_feature_ste(
        decoded * ((spec.quant_step_count - 1) / spec.quant_step_count) + spec.quant_min, spec
    )
    source = model._get_grid_params_tensor(feature_grid).detach()
    active = torch.cat((source[:offset], decoded, source[offset + count :]))
    model._qat_param_cache[key] = (version, active)
    return active
