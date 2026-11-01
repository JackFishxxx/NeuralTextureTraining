"""Feature-grid training logic used by the main Trainer.

Historical _astc_* hook names are preserved for existing stage/experiment clients.
ASTC native operations and BC1 endpoint updates remain format-specific.
"""

import torch
from Core.Feature_Grid.compression import grid_recipes
from Common.Block_Compression.astc import ASTCCodec
from Common.comparison_astc import cached_astc_base
from Core.Feature_Grid.sampling import bilinear_repeat
from dataset import material_size, clean_base_image, sample_material


from Core.ASTC_Aware.parameters import ASTCTrainingOperations


def prepare_decoder_inputs(trainer):
    """Use the same immutable decoded latents and SR base for fresh and resumed fitting."""
    from .compression import decoded_feature_grids, freeze_proxies

    model = trainer.model
    proxies, active = model.astc_proxy, getattr(model, '_codec_active_grids', None)
    flags = [(p, p.requires_grad) for p in model.parameters()]
    frozen = [(m, m.codec_frozen.clone()) for m in proxies.modules() if hasattr(m, 'codec_frozen')]
    try:
        freeze_proxies(model)
        fields = decoded_feature_grids(model)
    except Exception:
        model.astc_proxy, model._codec_active_grids = proxies, active
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
    def _initialize_astc_aware(self, dataset, configs):
        self.material_size = material_size(self.model, self.texture_height, self.texture_width)
        self.native_reconstruction = self.material_size != (self.texture_height, self.texture_width)
        if self.native_reconstruction and not self.mip0_only:
            raise ValueError(
                "Native material reconstruction below GT resolution currently requires mip0_only"
            )
        self.native_clean_base = (
            clean_base_image(dataset, self.material_size)
            if self.native_reconstruction and self.super_resolution_enable
            else None
        )
        self.astc_codec_backend = configs.astc_codec_backend
        self.astc_codec = ASTCCodec(self.astcenc_path, self.astcenc_quality, self.astc_block)
        self._codec_phase_logged = False
        self._astc_block_optimizer = None
        self._astc_codec_pass_count = 0
        if (self.progress.stage in ('single', 'representation') and configs.codec_training_enable
                and configs.astc_aware_enable and configs.native_astc_grids):
            if not self.mip0_only or not self.model.quantize:
                raise ValueError("Native ASTC optimization requires quantized Mip0")
            if configs.astc_decoder_projection_interval > 0 and not self._supports_astc_projection():
                print('[ASTC Projection] inactive: requires an affine SR decoder without PE/feature gradients')
            from Core.ASTC_Aware.differentiable_proxy import native_extension

            native_extension()
        print(
            f"[Latent Codec] grids={[r.tag for r in grid_recipes(self.model)]}, "
            f"in_loop={configs.codec_training_enable}, interval={configs.codec_interval}, "
            f"start_frac={configs.codec_start_frac:.3f}"
        )
        if self.enable_astc_compare or self.enable_pbr_compare or self.super_resolution_enable:
            self.astc_codec.ensure_executable()
        self.astc_superres_base_mip0 = None
        if self.super_resolution_enable:
            decoded = cached_astc_base(
                dataset,
                self.astc_codec,
                configs.super_resolution_base_resolution,
                output_size=self.material_size,
            )
            self.astc_superres_base_mip0 = decoded[0].permute(1, 2, 0).contiguous()

    def _predict_material(self, inputs):
        """Keep the residual API while supervising the deployed native-raster upsampler."""
        size = material_size(self.model, self.texture_height, self.texture_width)
        if size == (self.texture_height, self.texture_width):
            return self.model(inputs)
        codec = (
            self.model.astc_codec_branch_enabled
            or self.model._decoder_adaptation_features is not None
        )
        base_image = None
        if self.super_resolution_enable:
            base_image = (
                self.astc_superres_base_mip0.permute(2, 0, 1)[None].float()
                if codec
                else self.native_clean_base
            )
        prediction = sample_material(self.model, inputs[:, :2], size, base_image)
        return prediction - inputs[:, 3:].float() if self.super_resolution_enable else prediction

    def _begin_astc_step(self, curr_iter):
        self.model.current_iter = curr_iter
        if self._astc_block_optimizer is not None:
            self._astc_block_optimizer.zero_grad()
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
            total, primary = self._astc_proxy_batch_loss(inputs, target, weights, curr_iter)
            self.writer.add_scalar("Loss/base", primary.item(), curr_iter)
            return total
        # Warmup and frozen decoder fitting use the ordinary reconstruction path.
        prediction = self._predict_material(inputs)
        primary = self._compute_astc_reconstruction_loss(
            target, prediction, base, weights, curr_iter
        )
        total = primary
        pbr_clean = self._astc_pbr_loss(target, prediction, base)
        subpixel = None
        if self.subpixel_sampling_enable and self.subpixel_sampling_ratio > 0:
            subinput, subtarget, subbase = self._sample_astc_subpixel_batch(proxy=False)
            subpixel = self._compute_astc_reconstruction_loss(
                subtarget, self._predict_material(subinput), subbase, weights
            )
        if subpixel is not None:
            total = total + self.subpixel_sampling_ratio * subpixel
        if pbr_clean is not None:
            total = total + self.pbr_loss_weight * pbr_clean
            self.writer.add_scalar("Loss/pbr_render", pbr_clean.item(), curr_iter)
        self.writer.add_scalar("Loss/base", primary.item(), curr_iter)
        return total

    def _astc_pbr_loss(self, target, residual, base):
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
                    self._pbr_loss_graph = torch.cuda.make_graphed_callables(
                        self._compute_pbr_render_loss,
                        (target.detach().clone(), prediction.detach().clone().requires_grad_(True)),
                    )
            return self._pbr_loss_graph(target, prediction)
        return self._compute_pbr_render_loss(target, prediction)

    def _sample_astc_subpixel_batch(self, proxy):
        count = max(1, int(round(self.batch_size * self.subpixel_sampling_ratio)))
        mips = (
            torch.zeros(count, device=self.device, dtype=torch.long)
            if proxy
            else self.sample_probabilities.multinomial(count, replacement=True)
        )
        xy = torch.stack(
            (
                torch.randint(0, self.texture_height, (count,), device=self.device),
                torch.randint(0, self.texture_width, (count,), device=self.device),
            ),
            1,
        ).float()
        xy += (torch.rand_like(xy) * 2 - 1) * self.subpixel_jitter
        sample_mips = 0 if proxy or self.mip0_only else mips
        target = self.dataset.expand_to_canonical(
            self.dataset.sample_continuous(xy, sample_mips)
        ).to(torch.float32 if getattr(self, "native_reconstruction", False) else torch.float16)
        uv = torch.stack(
            ((xy[:, 1] + 0.5) / self.texture_width, (xy[:, 0] + 0.5) / self.texture_height), 1
        )
        mip_input = (
            mips.float()[:, None] / (self.num_mips - 1)
            if self.num_mips > 1
            else torch.zeros((count, 1), device=self.device)
        )
        if not self.mip0_only:
            scale = torch.pow(2.0, mips.float())
            uv = torch.stack(
                (
                    (xy[:, 1] / scale + 0.5) / (self.texture_width / scale),
                    (xy[:, 0] / scale + 0.5) / (self.texture_height / scale),
                ),
                1,
            )
        inputs = torch.cat((uv, mip_input), 1)
        base = None
        if self.super_resolution_enable:
            base = (
                self._astc_base_at_uv(uv)
                if proxy
                else self.dataset.expand_to_canonical(
                    self.dataset.get_superres_base_continuous(xy, sample_mips)
                ).to(torch.float16)
            )
            inputs = torch.cat((inputs, base), 1)
        return inputs, target, base

    def _astc_proxy_batch_loss(self, inputs, target, weights, curr_iter):
        self._ensure_astc_proxy()
        base = self._astc_base_at_uv(inputs[:, :2])
        inputs = torch.cat((inputs[:, :3], base), 1) if base is not None else inputs
        self._astc_proxy_batches = [(inputs, target, base, 1.0, True)]
        self._astc_proxy_loss_weights = weights
        self.model.astc_codec_branch_enabled = True
        try:
            prediction = self._predict_material(inputs)
            primary = self._compute_astc_reconstruction_loss(
                target, prediction, base, weights, curr_iter
            )
            total = primary
            pbr = self._astc_pbr_loss(target, prediction, base)
            if pbr is not None:
                total = total + self.pbr_loss_weight * pbr
            if self.subpixel_sampling_enable and self.subpixel_sampling_ratio > 0:
                subinput, subtarget, subbase = self._sample_astc_subpixel_batch(proxy=True)
                self._astc_proxy_batches.append(
                    (subinput, subtarget, subbase, self.subpixel_sampling_ratio, False)
                )
                subloss = self._compute_astc_reconstruction_loss(
                    subtarget, self._predict_material(subinput), subbase, weights
                )
                total = total + self.subpixel_sampling_ratio * subloss
            if any(r.format == "bc1" for r in grid_recipes(self.model)):
                from Core.BCn_Aware.parameters import binary_regularization

                total = total + binary_regularization(self)
            self.writer.add_scalar("Loss/astc_codec", primary.item(), curr_iter)
            return total, primary
        finally:
            self.model.astc_codec_branch_enabled = False

    @torch.no_grad()
    def _astc_proxy_objective(self):
        """Re-evaluate the same batches with fixed decoder for candidate acceptance."""
        self.model.clear_codec_cache()
        self.model.astc_codec_branch_enabled = True
        try:
            total = 0
            for inputs, target, base, weight, use_pbr in self._astc_proxy_batches:
                prediction = self._predict_material(inputs)
                loss = self._compute_astc_reconstruction_loss(
                    target, prediction, base, self._astc_proxy_loss_weights
                )
                total = total + weight * loss
                if use_pbr:
                    pbr = self._astc_pbr_loss(target, prediction, base)
                    if pbr is not None:
                        total = total + self.pbr_loss_weight * pbr
            return total
        finally:
            self.model.astc_codec_branch_enabled = False

    def _finish_astc_step(self, codec_due):
        if codec_due:
            self._sync_astc_source()

    def _compute_astc_reconstruction_loss(self, gt, prediction, base, loss_weights, curr_iter=None):
        material = prediction.float() if base is None else prediction.float() + base.float()
        return self._compute_reconstruction_loss(gt.float(), material, loss_weights, curr_iter)

    def _astc_base_at_uv(self, uv):
        if not self.super_resolution_enable:
            return None
        return (
            bilinear_repeat(
                self.astc_superres_base_mip0.permute(2, 0, 1)[None].float(),
                uv.view(1, -1, 1, 2),
            )
            .squeeze(0)
            .squeeze(-1)
            .T.to(torch.float16)
        )

    @torch.no_grad()
    def _sync_astc_source(self, indices=None):
        # The decoded trainable bitstream is the feature texture in this mode.
        # Keep exports, ordinary eval and the subsequent frozen stage consistent.
        from Core.Feature_Grid.feature_grid import unorm_to_feature, quantize_feature_tensor

        active = getattr(self.model, '_codec_active_grids', None)
        for index, (grid, spec, codec) in enumerate(zip(
            self.model.feature_grids, self.model.feature_grid_specs, self.model.astc_proxy
        )):
            if indices is not None and index not in indices:
                continue
            if self.model._decoder_adaptation_features is None and active is not None and index not in active:
                continue
            if not hasattr(codec, "decode_image"):
                continue
            offset, count, _ = spec.highest_level_slice()
            grid.params[offset : offset + count].copy_(
                quantize_feature_tensor(
                    unorm_to_feature(codec.decode_image().reshape(-1), spec), spec
                )
            )
        self.model._qat_param_cache.clear()

    def _ensure_astc_proxy(self):
        from Core.Feature_Grid.optimization import GridCompressionOptimizer

        if self._astc_block_optimizer is None:
            self._astc_block_optimizer = GridCompressionOptimizer(self)
        else:
            self._astc_block_optimizer.prepare()

    def _step_astc_parameters(self, codec_due):
        if codec_due:
            self._astc_block_optimizer.step()
