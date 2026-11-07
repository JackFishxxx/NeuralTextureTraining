"""ASTC-only native parameter updates and decoder projection."""

import math
import torch
from Core.Feature_Grid.sampling import resize_bilinear_repeat
from dataset import prefiltered_material


class ASTCTrainingOperations:
    def _supports_astc_projection(self):
        model = self.model
        return (model.n_hidden_layers == 0 and model.n_frequencies == 0
                and not any(spec.feature_gradient_count for spec in model.feature_grid_specs)
                and self.super_resolution_enable
                and getattr(model, '_decoder_adaptation_features', None) is None)

    def _validate_astc_projection(self):
        if not self._supports_astc_projection():
            raise ValueError(
                "Material projection requires a linear SR decoder without PE/feature gradients"
            )

    @torch.no_grad()
    def _project_astc_decoder_target(self):
        """Conditionally solve each native grid while sampling all companion grids.

        Coordinate updates support different resolutions and mixed formats. Each
        proposal must improve the actual material objective before it is retained.
        """
        from Core.ASTC_Aware.differentiable_proxy import ASTCProxyGrid
        from Core.Feature_Grid.compression import astc_codec_for, sample_proxy_grid
        from Core.Feature_Grid.feature_grid import feature_to_unorm
        from Core.Feature_Grid.feature_gradients import sample_feature_inputs

        self._validate_astc_projection()
        model = self.model
        weights = torch.tensor(self.output_loss_weights, device=self.device)
        starts = [0]
        for channels in model.feature_input_channels:
            starts.append(starts[-1] + channels)
        dim = model.network.n_input_dims
        names = ("endpoints", "weights", "metadata", "partitions", "indices", "coefficients")
        results = []
        for index, grid in enumerate(model.grid_proxies):
            active = getattr(model, '_codec_active_grids', None)
            if active is not None and index not in active:
                continue
            if grid.proxy_kind != "astc_native" or not any(p.requires_grad for p in grid.parameters()):
                continue
            spec = model.feature_grid_specs[index]
            begin_channel = starts[index]
            basis = torch.zeros((5, dim), device=self.device)
            basis[1:, begin_channel:begin_channel+4] = torch.eye(4, device=self.device)
            basis[:, -1] = 1
            values = model.network(basis).float()
            matrix = (values[1:] - values[:1]).T
            gram = matrix.T @ (weights[:, None] * matrix)
            ridge = 1e-5 * gram.diag().mean().clamp_min(1e-6)
            inverse = torch.linalg.inv(gram + torch.eye(4, device=self.device) * ridge)
            transform = (weights[:, None] * matrix) @ inverse
            resolution = spec.max_resolution
            target = prefiltered_material(self.dataset, (resolution, resolution))[0]
            target = target.permute(1, 2, 0).reshape(-1, model.num_channels)
            base = self.sr_base_native
            if base.shape[-2:] != (resolution, resolution):
                base = resize_bilinear_repeat(base, (resolution, resolution))
            base = base[0].permute(1, 2, 0).reshape(-1, model.num_channels)
            model.clear_codec_cache()
            params = [sample_proxy_grid(model, i, source)
                      for i, source in enumerate(model.feature_grids)]
            projected = []
            for start in range(0, len(base), 65536):
                batch = base[start:start+65536]
                ids = torch.arange(start, start+len(batch), device=self.device)
                uv = torch.stack(((ids % resolution + .5)/resolution,
                                  (ids // resolution + .5)/resolution), 1)
                features = torch.cat([
                    model._decoder_features(sample_feature_inputs(p, other, uv,
                        uv.new_full((len(uv), 1), other.n_levels-1)), i)
                    for i, (p, other) in enumerate(zip(params, model.feature_grid_specs))], 1)
                features[:, begin_channel:begin_channel+4] = 0
                inputs = torch.cat((features, batch, uv.new_ones((len(uv), 1))), 1)
                residual = (target[start:start+len(batch)] - batch
                            - model.network(inputs).float() - model.decoder_output_bias.float())
                projected.append((residual @ transform).clamp(
                    spec.quant_min, spec.quant_min+(spec.quant_step_count-1)/spec.quant_step_count))
            latent = torch.cat(projected).reshape(resolution, resolution, 4)
            rgba = (feature_to_unorm(latent, spec)*255).round().byte().cpu().numpy()
            proposal = ASTCProxyGrid.from_astc(astc_codec_for(model, index).encode_rgba(rgba), self.device)
            baseline = float(self._codec_objective())
            saved = {name: getattr(grid, name).clone() for name in names}
            accepted = False
            try:
                for name in names:
                    getattr(grid, name).copy_(getattr(proposal, name))
                candidate = float(self._codec_objective())
                keep = math.isfinite(candidate) and candidate < baseline
                if keep:
                    self._grid_compression_optimizer.reset_rows(
                        index, torch.arange(len(grid.endpoints), device=self.device))
                accepted = keep
            finally:
                if not accepted:
                    for name in names:
                        getattr(grid, name).copy_(saved[name])
                model.clear_codec_cache()
            result = dict(grid=index, accepted=accepted, baseline_loss=baseline, candidate_loss=candidate)
            results.append(result)
            for name in ("accepted", "baseline_loss", "candidate_loss"):
                self.writer.add_scalar(f"ASTCProjection/grid{index}/{name}", result[name], model.current_iter)
        return results
