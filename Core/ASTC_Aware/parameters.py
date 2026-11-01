"""ASTC-only native parameter updates, codec patches and offline refinement."""

import math
import torch
from Core.Feature_Grid.sampling import resize_bilinear_repeat
from dataset import prefiltered_material


class ASTCTrainingOperations:
    def _supports_astc_projection(self):
        model = self.model
        return (model.n_hidden_layers == 0 and model.n_frequencies == 0
                and model.feature_gradient_count == 0 and self.super_resolution_enable
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
        from Core.ASTC_Aware.differentiable_proxy import ASTCProxyGrid, encode_seed
        from Core.Feature_Grid.compression import astc_codec_for, proxy_kind, sample_proxy_grid
        from Core.Feature_Grid.feature_grid import feature_to_unorm
        from Core.Feature_Grid.feature_gradients import sample_feature_inputs

        self._validate_astc_projection()
        model = self.model
        weights = torch.tensor(self.output_loss_weights, device=self.device)
        starts = [0]
        for spec in model.feature_grid_specs:
            starts.append(starts[-1] + spec.n_features_per_level)
        dim = starts[-1] + model.num_channels + 2
        names = ("endpoints", "weights", "metadata", "partitions", "indices", "coefficients")
        results = []
        for index, grid in enumerate(model.astc_proxy):
            active = getattr(model, '_codec_active_grids', None)
            if active is not None and index not in active:
                continue
            if proxy_kind(grid) != "astc_native" or not any(p.requires_grad for p in grid.parameters()):
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
            base = self.astc_superres_base_mip0.permute(2, 0, 1)[None].float()
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
                    sample_feature_inputs(p, other, uv,
                                          uv.new_full((len(uv), 1), other.n_levels-1))
                    for p, other in zip(params, model.feature_grid_specs)], 1)
                features[:, begin_channel:begin_channel+4] = 0
                inputs = torch.cat((features, uv.new_zeros((len(uv), 1)), batch,
                                    uv.new_ones((len(uv), 1))), 1)
                residual = (target[start:start+len(batch)] - batch
                            - model.network(inputs).float() - model.decoder_output_bias.float())
                projected.append((residual @ transform).clamp(
                    spec.quant_min, spec.quant_min+(spec.quant_step_count-1)/spec.quant_step_count))
            latent = torch.cat(projected).reshape(resolution, resolution, 4)
            rgba = (feature_to_unorm(latent, spec)*255).round().byte().cpu().numpy()
            proposal = ASTCProxyGrid.from_astc(encode_seed(astc_codec_for(model, index), rgba), self.device)
            baseline = float(self._astc_proxy_objective())
            saved = {name: getattr(grid, name).clone() for name in names}
            accepted = False
            try:
                for name in names:
                    getattr(grid, name).copy_(getattr(proposal, name))
                candidate = float(self._astc_proxy_objective())
                keep = math.isfinite(candidate) and candidate < baseline
                if keep:
                    self._astc_block_optimizer.reset_rows(
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

    def refine_gradient_codec(self, steps, learning_rate=0.001, blocks_per_batch=64):
        """Refine every native ASTC grid with all other latent inputs held fixed."""
        from Core.Feature_Grid.compression import proxy_kind

        model = self.model
        indices = [i for i, proxy in enumerate(model.astc_proxy)
                   if proxy_kind(proxy) == 'astc_native']
        if not indices:
            raise ValueError('Codec refinement requires native Mip0 ASTC proxies')
        saved = [(i, model.astc_proxy[i].endpoints.detach().clone(),
                  model.astc_proxy[i].weights.detach().clone(), model.feature_grids[i].params.detach().clone())
                 for i in indices]
        old_optimizer = self._astc_block_optimizer
        active = getattr(model, '_codec_active_grids', None)
        try:
            # Offline refinement uses every codec input, regardless of warmup
            # or the empty activity mask left by frozen decoder fitting.
            model._codec_active_grids = tuple(range(len(model.astc_proxy)))
            grids = [self._refine_gradient_grid(i, steps, learning_rate, blocks_per_batch)
                     for i in indices]
            return dict(steps=steps, grids=grids,
                        accepted_blocks=sum(g['accepted_blocks'] for g in grids),
                        continuous_accepted=sum(g['continuous_accepted'] for g in grids))
        except Exception:
            with torch.no_grad():
                for i, endpoints, weights, source in saved:
                    model.astc_proxy[i].endpoints.copy_(endpoints)
                    model.astc_proxy[i].weights.copy_(weights)
                    model.feature_grids[i].params.copy_(source)
            self._astc_block_optimizer = old_optimizer
            raise
        finally:
            model._codec_active_grids = active
            model._qat_param_cache.clear()

    def _refine_gradient_grid(self, grid_index, steps, learning_rate, blocks_per_batch):
        """Fit legal codec symbols with the decoder fixed over complete neighborhoods.

        This explicit refinement pass is separate from frozen decoder adaptation.
        The output patch covers each selected 6x6 block and the gradient/bilinear
        footprint around it (10x10 for 3x3 inputs, 12x12 for 5x5). Selected rows are
        updated, and proposals must improve the actual material/PBR objective.
        """
        from Core.ASTC_Aware.differentiable_proxy import ASTCBlockAdam
        from Core.Feature_Grid.feature_gradients import gradient_directions

        model = self.model
        if (
            not self.mip0_only
            or model._decoder_adaptation_features is not None
        ):
            raise ValueError(
                "Codec refinement requires a fitted Mip0 decoder outside frozen adaptation"
            )
        if (
            steps < 1
            or blocks_per_batch < 1
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("Refinement steps, block count and learning rate must be positive")
        grid = model.astc_proxy[grid_index]
        height, width = grid._image_shape
        if (
            self.texture_height % height
            or self.texture_width % width
        ):
            raise ValueError(
                "Codec refinement requires an integer GT/grid scale"
            )
        scale_y, scale_x = self.texture_height // height, self.texture_width // width
        if self.super_resolution_enable and self.astc_superres_base_mip0 is None:
            raise ValueError("Codec refinement requires the deployment ASTC base")
        blocks_x, blocks_y = (width + 5) // 6, (height + 5) // 6
        optimizer = ASTCBlockAdam(
            torch.nn.ModuleList([grid]),
            learning_rate,
            learning_rate,
            quantization_aware=True,
            packed_pair_search=True,
        )
        loss_weights = torch.tensor(self.output_loss_weights, device=self.device)
        halo = 1 + max(
            (max(abs(dx), abs(dy)) for dx, dy in gradient_directions(model.feature_gradient_count)),
            default=0,
        )
        if (scale_y, scale_x) != (1, 1):
            halo += 1
        ly, lx = torch.meshgrid(
            torch.arange(-halo * scale_y, (6 + halo) * scale_y, device=self.device),
            torch.arange(-halo * scale_x, (6 + halo) * scale_x, device=self.device),
            indexing="ij",
        )
        parameter_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
        was_training, old_iteration = model.training, model.current_iter
        old_branch = model.astc_codec_branch_enabled
        old_batches = getattr(self, "_astc_proxy_batches", None)
        old_weights = getattr(self, "_astc_proxy_loss_weights", None)
        stats = {
            "grid": grid_index,
            "steps": steps,
            "accepted_blocks": 0,
            "continuous_accepted": 0,
            "learning_rate": learning_rate,
            "blocks_per_batch": blocks_per_batch,
        }
        stats["loss_weights"] = list(self.output_loss_weights)
        try:
            model.train()
            for parameter, _ in parameter_flags:
                parameter.requires_grad_(False)
            grid.requires_grad_(True)
            for step in range(steps):
                bx = torch.randint(0, blocks_x, (blocks_per_batch,), device=self.device)
                by = torch.randint(0, blocks_y, (blocks_per_batch,), device=self.device)
                ids = (by * blocks_x + bx).unique()
                xs = (ids[:, None, None] % blocks_x * 6 * scale_x + lx) % self.texture_width
                ys = (ids[:, None, None] // blocks_x * 6 * scale_y + ly) % self.texture_height
                uv = torch.stack(
                    (
                        (xs.flatten() + 0.5) / self.texture_width,
                        (ys.flatten() + 0.5) / self.texture_height,
                    ),
                    1,
                )
                base = self._astc_base_at_uv(uv)
                target = self.dataset.expand_to_canonical(
                    self.dataset.mip_cache[0][ys.flatten(), xs.flatten()]
                ).to(torch.float16)
                inputs = torch.cat((uv, torch.zeros(len(uv), 1, device=self.device)), 1)
                if base is not None:
                    inputs = torch.cat((inputs, base), 1)
                model.current_iter = old_iteration + step + 1
                model._qat_param_cache.clear()
                optimizer.zero_grad()
                model.astc_codec_branch_enabled = True
                prediction = self._predict_material(inputs)
                loss = self._compute_astc_reconstruction_loss(
                    target, prediction, base, loss_weights
                )
                pbr = self._astc_pbr_loss(target, prediction, base)
                if pbr is not None:
                    loss = loss + self.pbr_loss_weight * pbr
                loss.backward()
                self._astc_proxy_batches = [(inputs, target, base, 1.0, True)]
                self._astc_proxy_loss_weights = loss_weights
                optimizer.step({0: ids}, objective=self._astc_proxy_objective)
                stats["accepted_blocks"] += optimizer.last_stats.get("accepted_blocks", 0)
                stats["continuous_accepted"] += optimizer.last_stats.get("continuous_accepted", 0)
            self._sync_astc_source((grid_index,))
            self._astc_block_optimizer = None
            return stats
        finally:
            optimizer.zero_grad()
            for parameter, flag in parameter_flags:
                parameter.requires_grad_(flag)
            model.train(was_training)
            model.current_iter = old_iteration
            model.astc_codec_branch_enabled = old_branch
            model._qat_param_cache.clear()
            self._astc_proxy_batches = old_batches
            self._astc_proxy_loss_weights = old_weights
