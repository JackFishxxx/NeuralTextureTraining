"""Feature-grid training logic used by the main Trainer.

ASTC native operations and BC1 endpoint updates remain format-specific.
"""

from functools import cached_property
import torch
from Common.Material.reconstruction import cached_sr_base
from Common.cuda_graph import make_graphed_callables
from Core.Feature_Grid.sampling import RepeatImageSampler
from dataset import material_size, sample_material, NativeQueryPlan


from Core.ASTC_Aware.parameters import ASTCTrainingOperations


def mip_probabilities(count):
    """Fixed fine-mip priority with a uniform floor for full-chain coverage."""
    weights = torch.pow(torch.tensor(.5, dtype=torch.float64), torch.arange(count))
    return .8 * weights / weights.sum() + .2 / count


def fixed_validation_uv(count, device):
    """Cover UV and subtexel phases without consuming the training RNG."""
    return torch.quasirandom.SobolEngine(2, scramble=True, seed=1729).draw(count).to(device)


def prepare_decoder_inputs(trainer):
    """Use the same immutable decoded latents and SR base for fresh and resumed fitting."""
    from .compression import decoded_feature_grids, freeze_proxies

    model = trainer.model
    proxies, active = model.grid_proxies, getattr(model, '_codec_active_grids', None)
    flags = [(p, p.requires_grad) for p in model.parameters()]
    frozen = [(m, m.codec_frozen.clone()) for m in proxies.modules() if hasattr(m, 'codec_frozen')]
    try:
        freeze_proxies(model)
        fields = decoded_feature_grids(model)
    except Exception:
        model.grid_proxies, model._codec_active_grids = proxies, active
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, value in frozen:
            module.codec_frozen.copy_(value)
        raise
    trainer._decoder_source_grad = [g.params.requires_grad for g in model.feature_grids]
    model._decoder_adaptation_features = fields
    for grid in model.feature_grids:
        grid.params.requires_grad_(False)


def release_decoder_inputs(trainer):
    trainer.model._decoder_adaptation_features = None
    for grid, flag in zip(trainer.model.feature_grids, trainer._decoder_source_grad):
        grid.params.requires_grad_(flag)
    del trainer._decoder_source_grad
    trainer.model._qat_param_cache.clear()


class FeatureGridTraining(ASTCTrainingOperations):
    @cached_property
    def _mip_sampling_probabilities(self):
        return mip_probabilities(1 if self.mip0_only else self.num_mips)

    def _initialize_grid_compression(self, dataset, configs):
        self.material_size = material_size(self.model, self.texture_height, self.texture_width)
        self.native_reconstruction = self.material_size != (self.texture_height, self.texture_width)
        self._native_query_plans = {}
        self.astc_codec_backend = configs.astc_codec_backend
        self._codec_phase_logged = False
        self._grid_compression_optimizer = None
        if (self.progress.stage in ('single', 'representation') and configs.codec_training_enable
                and configs.astc_aware_enable and configs.native_astc_grids):
            if not self.model.quantize:
                raise ValueError("Native ASTC optimization requires quantized feature grids")
            if configs.astc_decoder_projection_interval > 0 and not self._supports_astc_projection():
                print('[ASTC Projection] inactive: requires an affine SR decoder without PE/feature gradients')
            from Core.ASTC_Aware.differentiable_proxy import native_extension

            native_extension()
        print(
            f"[Latent Codec] grids={[r.tag for r in self.model.grid_compressions]}, "
            f"in_loop={configs.codec_training_enable}, interval={configs.codec_interval}, "
            f"start_frac={configs.codec_start_frac:.3f}"
        )
        self.sr_base_native = None
        self._sr_base_sampler = None
        if self.super_resolution_enable:
            self.sr_base_native = cached_sr_base(
                dataset,
                self.material_compression,
                configs.super_resolution_base_resolution,
                output_size=self.material_size,
            )
            self._sr_base_sampler = RepeatImageSampler(self.sr_base_native)

    def _predict_material(self, inputs):
        """Keep the residual API while supervising the deployed native-raster upsampler."""
        size = material_size(self.model, self.texture_height, self.texture_width)
        if (getattr(self.model, 'inference_mode', 'on_load') == 'on_sample'
                or size == (self.texture_height, self.texture_width)):
            return self.model(inputs)
        base_sampler = self._sr_base_sampler if self.super_resolution_enable else None
        uv = inputs[:, :2]
        if (self.model.training and torch.is_grad_enabled() and uv.is_cuda and uv.dtype == torch.float32
                and not uv.requires_grad and len(uv) and len(uv)*4 <= size[0]*size[1]//2
                and not torch.is_autocast_enabled()
                and not self.model.direct_diffuse_enabled
                and all(spec.n_levels == 1 and size == (spec.max_resolution, spec.max_resolution)
                        for spec in self.model.feature_grid_specs)):
            from .material_graph import MaterialQueryGraphs

            graphs = getattr(self, '_material_query_graphs', None)
            if (graphs is None or graphs.model is not self.model or graphs.size != size
                    or graphs.base_sampler is not base_sampler
                    or (base_sampler is not None and graphs.base_version != base_sampler.image._version)):
                graphs = self._material_query_graphs = MaterialQueryGraphs(self.model, size, base_sampler)
            params = tuple(self.model._get_qat_grid_params(i, grid)
                           if self.model.quantize or self.model._decoder_adaptation_features is not None
                           else grid.params for i, grid in enumerate(self.model.feature_grids))
            prediction = graphs(uv, params)
            return (prediction - inputs[:, 2 + (not self.mip0_only):].float()
                    if self.super_resolution_enable else prediction)
        query_plan = None
        if uv.is_cuda and not uv.requires_grad and len(uv) and len(uv)*4 <= size[0]*size[1]//2:
            key = (len(uv), uv.dtype, uv.device, size)
            if key not in self._native_query_plans:
                self._native_query_plans[key] = NativeQueryPlan(uv, size)
            query_plan = self._native_query_plans[key]
        prediction = sample_material(self.model, uv, size, base_sampler=base_sampler, query_plan=query_plan)
        return prediction - inputs[:, 2 + (not self.mip0_only):].float() if self.super_resolution_enable else prediction

    def _begin_codec_step(self, curr_iter):
        self.model.current_iter = curr_iter
        if self._grid_compression_optimizer is not None:
            self._grid_compression_optimizer.zero_grad()
        policies = self.configs.grid_training
        indices = tuple(i for i, policy in enumerate(policies)
                        if self.configs.codec_training_enable and self.progress.stage in ('single', 'representation')
                        and policy.active(curr_iter, self.configs.max_iter))
        self.model._codec_active_grids = indices
        self._codec_due_grids = tuple(i for i in indices if policies[i].due(curr_iter, self.configs.max_iter))
        active = bool(indices)
        if active and not self._codec_phase_logged:
            print(f"[Latent Codec] phase active at iter {curr_iter}", flush=True)
            self._codec_phase_logged = True
        return active

    def _training_batch_loss(self, inputs, target, base, weights, curr_iter, codec_due):
        if codec_due:
            total, primary = self._codec_batch_loss(inputs, target, weights, curr_iter)
            self._record_training_scalar("Loss/base", primary)
            return total
        # Warmup and frozen decoder fitting use the ordinary reconstruction path.
        prediction = self._predict_material(inputs)
        primary = self._compute_material_reconstruction_loss(
            target, prediction, base, weights, curr_iter
        )
        total = primary
        pbr_clean = self._pbr_batch_loss(target, prediction, base)
        subpixel = None
        if self.subpixel_sampling_enable and self.subpixel_sampling_ratio > 0:
            subinput, subtarget, subbase = self._sample_subpixel_batch()
            subpixel = self._compute_material_reconstruction_loss(
                subtarget, self._predict_material(subinput), subbase, weights
            )
        if subpixel is not None:
            total = total + self.subpixel_sampling_ratio * subpixel
        if pbr_clean is not None:
            total = total + self.pbr_loss_weight * pbr_clean
            self._record_training_scalar("Loss/pbr_render", pbr_clean)
        self._record_training_scalar("Loss/base", primary)
        return total

    def _pbr_batch_loss(self, target, residual, base):
        if not self.pbr_loss_enable or self.pbr_loss_weight <= 0:
            return None
        prediction = (base + residual).clamp(0, 1) if base is not None else residual.clamp(0, 1)
        target, prediction = target.float(), prediction.float()
        if (
            self.astc_codec_backend == "astc_differentiable_proxy"
            and target.is_cuda
            and len(target) == self.batch_size
        ):
            # One PBR batch per step. Replay its many small kernels together;
            # candidate checks reuse the same forward after training backward.
            if not hasattr(self, "_pbr_loss_graph"):
                with torch.enable_grad():
                    self._pbr_loss_graph = make_graphed_callables(
                        self._compute_pbr_render_loss,
                        (target.detach().clone(), prediction.detach().clone().requires_grad_(True)),
                    )
            return self._pbr_loss_graph(target, prediction)
        return self._compute_pbr_render_loss(target, prediction)

    @torch.no_grad()
    def _validate_sampled_mips(self, iteration, evaluation=None):
        from Common.Material.reconstruction import make_deployment_model
        model = getattr(evaluation, 'sample_model', None) or make_deployment_model(self.model)
        cases = getattr(self, '_sampled_validation_cases', None)
        if cases is None:
            uv = fixed_validation_uv(self.configs.mip_validation_samples, self.device)
            probabilities = self._mip_sampling_probabilities
            count = len(probabilities)
            cases = []
            for mip, probability in enumerate(probabilities.tolist()):
                # Keep model selection comparable when training LOD sampling changes.
                blend = .25 if mip + 1 < count else 0.
                for lod, weight in ((float(mip), probability * (1 - blend)), (mip + .5, probability * blend)):
                    if weight:
                        inputs, target, _ = self._batch_with_mip(uv, lod, self.dataset.sample_lod(uv, lod))
                        cases.append((lod, weight, inputs, target))
            self._sampled_validation_cases = cases
        weights = torch.tensor(self.eval_weights, device=self.device, dtype=torch.float32)
        score = 0.
        for lod, probability, inputs, target in cases:
            prediction = model(inputs).float().clamp(0, 1)
            mse = ((prediction - target.float()).square() * weights).sum(1).mean() / weights.sum().clamp_min(1e-8)
            psnr = float((-10 * mse.clamp_min(1e-12).log10()).item())
            self.writer.add_scalar(f'Validation/Mip{lod:g}/PSNR', psnr, iteration)
            score += probability * psnr
        self._sampled_validation = (iteration, score)
        self.writer.add_scalar('Validation/on_sample_mip_PSNR', score, iteration)
        return score

    def _sample_training_batch(self):
        on_sample = getattr(getattr(self, 'model', None), 'inference_mode', 'on_load') == 'on_sample'
        mip = (int(torch.multinomial(self._mip_sampling_probabilities, 1).item())
               if on_sample and not self.mip0_only else 0)
        lod = float(mip)
        if (on_sample and not self.mip0_only and mip < self.num_mips - 1
                and torch.rand(()).item() < self.configs.mip_fractional_sampling_ratio):
            lod += torch.rand(()).item()
        self._training_mip, self._training_lod = mip, lod
        # Output LOD may extend past the original GT's last available mip.
        level = self.dataset.mip_cache[min(mip, len(self.dataset.mip_cache) - 1)]
        height, width = level.shape[:2]
        ys = torch.randint(height, (self.batch_size,), device=self.device)
        xs = torch.randint(width, (self.batch_size,), device=self.device)
        uv = torch.stack(((xs.float() + .5) / width, (ys.float() + .5) / height), 1)
        target = (self.dataset.sample_lod(uv, lod) if on_sample else
                  self.dataset.expand_to_canonical(level[ys, xs]).to(
                      torch.float32 if self.native_reconstruction else torch.float16))
        return self._batch_with_mip(uv, lod, target)

    def _batch_with_mip(self, uv, mip, target):
        inputs = uv if self.mip0_only else torch.cat(
            (uv, uv.new_full((len(uv), 1), mip / max(1, self.num_mips - 1))), 1)
        base = self._sr_base_at_uv(uv) if self.super_resolution_enable else None
        if base is not None:
            inputs = torch.cat((inputs, base), 1)
        return inputs, target, base

    def _sample_subpixel_batch(self):
        count = max(1, int(round(self.batch_size * self.subpixel_sampling_ratio)))
        mip = min(getattr(self, '_training_mip', 0), len(self.dataset.mip_cache) - 1)
        height, width = self.dataset.mip_cache[mip].shape[:2]
        xy = torch.stack(
            (
                torch.randint(0, height, (count,), device=self.device),
                torch.randint(0, width, (count,), device=self.device),
            ),
            1,
        ).float()
        xy += (torch.rand_like(xy) * 2 - 1) * self.subpixel_jitter
        uv = torch.stack(((xy[:, 1] + .5) / width, (xy[:, 0] + .5) / height), 1)
        if getattr(getattr(self, 'model', None), 'inference_mode', 'on_load') == 'on_sample':
            lod = self._training_lod
            return self._batch_with_mip(uv, lod, self.dataset.sample_lod(uv, lod))
        target = self.dataset.expand_to_canonical(
            self.dataset.sample_continuous(xy * (2 ** mip), mip)
        ).to(torch.float32 if getattr(self, "native_reconstruction", False) else torch.float16)
        return self._batch_with_mip(uv, mip, target)

    def _codec_batch_loss(self, inputs, target, weights, curr_iter):
        self._ensure_grid_proxies()
        base = inputs[:, 2 + (not self.mip0_only):] if self.super_resolution_enable else None
        self._codec_batches = [(inputs, target, base, 1.0, True)]
        self._codec_loss_weights = weights
        self.model.codec_branch_enabled = True
        try:
            prediction = self._predict_material(inputs)
            primary = self._compute_material_reconstruction_loss(
                target, prediction, base, weights, curr_iter
            )
            total = primary
            pbr = self._pbr_batch_loss(target, prediction, base)
            if pbr is not None:
                total = total + self.pbr_loss_weight * pbr
            if self.subpixel_sampling_enable and self.subpixel_sampling_ratio > 0:
                subinput, subtarget, subbase = self._sample_subpixel_batch()
                self._codec_batches.append(
                    (subinput, subtarget, subbase, self.subpixel_sampling_ratio, False)
                )
                subloss = self._compute_material_reconstruction_loss(
                    subtarget, self._predict_material(subinput), subbase, weights
                )
                total = total + self.subpixel_sampling_ratio * subloss
            if any(r.format == "bc1" for r in self.model.grid_compressions):
                from Core.BCn_Aware.parameters import binary_regularization

                total = total + binary_regularization(self)
            self._record_training_scalar("Loss/astc_codec", primary)
            return total, primary
        finally:
            self.model.codec_branch_enabled = False

    @torch.no_grad()
    def _codec_objective(self):
        """Re-evaluate the same batches with fixed decoder for candidate acceptance."""
        self.model.clear_codec_cache()
        self.model.codec_branch_enabled = True
        try:
            total = 0
            for inputs, target, base, weight, use_pbr in self._codec_batches:
                prediction = self._predict_material(inputs)
                loss = self._compute_material_reconstruction_loss(
                    target, prediction, base, self._codec_loss_weights
                )
                total = total + weight * loss
                if use_pbr:
                    pbr = self._pbr_batch_loss(target, prediction, base)
                    if pbr is not None:
                        total = total + self.pbr_loss_weight * pbr
            return total
        finally:
            self.model.codec_branch_enabled = False

    def _finish_codec_step(self, codec_due):
        if codec_due:
            # Codec proxies are authoritative. Refresh source mirrors only when
            # a source forward or checkpoint actually needs their current values.
            self.model._pending_grid_sync.update(self._codec_due_grids)

    def _compute_material_reconstruction_loss(self, gt, prediction, base, loss_weights, curr_iter=None):
        material = prediction.float() if base is None else prediction.float() + base.float()
        return self._compute_reconstruction_loss(gt.float(), material, loss_weights, curr_iter)

    def _sr_base_at_uv(self, uv):
        if not self.super_resolution_enable:
            return None
        return self._sr_base_sampler(uv).to(torch.float16)

    def _sync_grid_sources(self, indices=None):
        from .compression import sync_grid_sources
        active = getattr(self.model, '_codec_active_grids', None)
        indices = range(len(self.model.grid_proxies)) if indices is None else indices
        sync_grid_sources(self.model, (i for i in indices
            if self.model._decoder_adaptation_features is not None or active is None or i in active))

    def _ensure_grid_proxies(self):
        from Core.Feature_Grid.optimization import GridCompressionOptimizer

        if self._grid_compression_optimizer is None:
            self._grid_compression_optimizer = GridCompressionOptimizer(self)
        else:
            self._grid_compression_optimizer.prepare()

    def _step_codec_parameters(self, codec_due):
        if codec_due:
            self._grid_compression_optimizer.step()
