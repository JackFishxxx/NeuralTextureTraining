"""Feature-grid training logic used by the main Trainer.

ASTC native operations and BC1 endpoint updates remain format-specific.
"""

import torch
from Common.Block_Compression.astc import ASTCCodec
from Common.Material.reconstruction import cached_astc_base
from Common.cuda_graph import make_graphed_callables
from Core.Feature_Grid.sampling import RepeatImageSampler
from dataset import material_size, clean_base_image, sample_material, NativeQueryPlan


from Core.ASTC_Aware.parameters import ASTCTrainingOperations


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
    if trainer.super_resolution_enable:
        trainer._decoder_source_base = trainer.dataset.superres_base_cache[0]
        base = trainer.astc_superres_base_mip0
        if base.shape[:2] == trainer._decoder_source_base.shape[:2]:
            slices = trainer.dataset.canonical_channel_slices
            trainer.dataset.superres_base_cache[0] = torch.cat(
                [base[..., slices[name][0]:slices[name][1]] for name in trainer.dataset.available_textures], -1)


def release_decoder_inputs(trainer):
    trainer.model._decoder_adaptation_features = None
    for grid, flag in zip(trainer.model.feature_grids, trainer._decoder_source_grad):
        grid.params.requires_grad_(flag)
    if trainer.super_resolution_enable:
        trainer.dataset.superres_base_cache[0] = trainer._decoder_source_base
        del trainer._decoder_source_base
    del trainer._decoder_source_grad
    trainer.model._qat_param_cache.clear()


class FeatureGridTraining(ASTCTrainingOperations):
    def _initialize_grid_compression(self, dataset, configs):
        self.material_size = material_size(self.model, self.texture_height, self.texture_width)
        self.native_reconstruction = self.material_size != (self.texture_height, self.texture_width)
        self._native_query_plans = {}
        if self.native_reconstruction and not self.mip0_only:
            raise ValueError(
                "Native material reconstruction below GT resolution currently requires mip0_only"
            )
        self.native_clean_base = (
            clean_base_image(dataset, self.material_size)
            if self.native_reconstruction and self.super_resolution_enable
            else None
        )
        self._clean_base_sampler = RepeatImageSampler(self.native_clean_base) if self.native_clean_base is not None else None
        self.astc_codec_backend = configs.astc_codec_backend
        self.astc_codec = ASTCCodec(self.astcenc_path, self.astcenc_quality, self.astc_block)
        self._codec_phase_logged = False
        self._grid_compression_optimizer = None
        if (self.progress.stage in ('single', 'representation') and configs.codec_training_enable
                and configs.astc_aware_enable and configs.native_astc_grids):
            if not self.mip0_only or not self.model.quantize:
                raise ValueError("Native ASTC optimization requires quantized Mip0")
            if configs.astc_decoder_projection_interval > 0 and not self._supports_astc_projection():
                print('[ASTC Projection] inactive: requires an affine SR decoder without PE/feature gradients')
            from Core.ASTC_Aware.differentiable_proxy import native_extension

            native_extension()
        print(
            f"[Latent Codec] grids={[r.tag for r in self.model.grid_compressions]}, "
            f"in_loop={configs.codec_training_enable}, interval={configs.codec_interval}, "
            f"start_frac={configs.codec_start_frac:.3f}"
        )
        if self.super_resolution_enable:
            self.astc_codec.ensure_executable()
        self.astc_superres_base_mip0 = None
        self._astc_base_sampler = None
        if self.super_resolution_enable:
            decoded = cached_astc_base(
                dataset,
                self.astc_codec,
                configs.super_resolution_base_resolution,
                output_size=self.material_size,
            )
            self.astc_superres_base_mip0 = decoded[0].permute(1, 2, 0).contiguous()
            self._astc_base_sampler = RepeatImageSampler(self.astc_superres_base_mip0.permute(2, 0, 1)[None])

    def _predict_material(self, inputs):
        """Keep the residual API while supervising the deployed native-raster upsampler."""
        size = material_size(self.model, self.texture_height, self.texture_width)
        if size == (self.texture_height, self.texture_width):
            return self.model(inputs)
        codec = (
            self.model.codec_branch_enabled
            or self.model._decoder_adaptation_features is not None
        )
        base_sampler = None
        if self.super_resolution_enable:
            base_sampler = self._astc_base_sampler if codec else self._clean_base_sampler
        uv = inputs[:, :2]
        query_plan = None
        if uv.is_cuda and not uv.requires_grad and len(uv) and len(uv)*4 <= size[0]*size[1]//2:
            key = (len(uv), uv.dtype, uv.device, size)
            if key not in self._native_query_plans:
                self._native_query_plans[key] = NativeQueryPlan(uv, size)
            query_plan = self._native_query_plans[key]
        prediction = sample_material(self.model, uv, size, base_sampler=base_sampler, query_plan=query_plan)
        return prediction - inputs[:, 3:].float() if self.super_resolution_enable else prediction

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
            subinput, subtarget, subbase = self._sample_astc_subpixel_batch(proxy=False)
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

    def _sample_astc_subpixel_batch(self, proxy):
        count = max(1, int(round(self.batch_size * self.subpixel_sampling_ratio)))
        xy = torch.stack(
            (
                torch.randint(0, self.texture_height, (count,), device=self.device),
                torch.randint(0, self.texture_width, (count,), device=self.device),
            ),
            1,
        ).float()
        xy += (torch.rand_like(xy) * 2 - 1) * self.subpixel_jitter
        target = self.dataset.expand_to_canonical(
            self.dataset.sample_continuous(xy, 0)
        ).to(torch.float32 if getattr(self, "native_reconstruction", False) else torch.float16)
        uv = torch.stack(
            ((xy[:, 1] + 0.5) / self.texture_width, (xy[:, 0] + 0.5) / self.texture_height), 1
        )
        inputs = torch.cat((uv, torch.zeros((count, 1), device=self.device)), 1)
        base = None
        if self.super_resolution_enable:
            base = (
                self._astc_base_at_uv(uv)
                if proxy
                else self.dataset.expand_to_canonical(
                    self.dataset.get_superres_base_continuous(xy, 0)
                ).to(torch.float16)
            )
            inputs = torch.cat((inputs, base), 1)
        return inputs, target, base

    def _codec_batch_loss(self, inputs, target, weights, curr_iter):
        self._ensure_grid_proxies()
        base = self._astc_base_at_uv(inputs[:, :2])
        inputs = torch.cat((inputs[:, :3], base), 1) if base is not None else inputs
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
                subinput, subtarget, subbase = self._sample_astc_subpixel_batch(proxy=True)
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
            # BC1 training reads proxies directly; refresh its source mirrors
            # only when a source forward or state export actually needs them.
            active = self.model._codec_active_grids
            self.model._pending_grid_sync.update(
                i for i in active if self.model.grid_compressions[i].format == 'bc1')
            self._sync_grid_sources(
                i for i in active if self.model.grid_compressions[i].format != 'bc1')

    def _compute_material_reconstruction_loss(self, gt, prediction, base, loss_weights, curr_iter=None):
        material = prediction.float() if base is None else prediction.float() + base.float()
        return self._compute_reconstruction_loss(gt.float(), material, loss_weights, curr_iter)

    def _astc_base_at_uv(self, uv):
        if not self.super_resolution_enable:
            return None
        return self._astc_base_sampler(uv).to(torch.float16)

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
