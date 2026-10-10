"""ASTC-aware input selection, material losses and compressed-parameter updates."""

import math
import torch
import torch.nn.functional as F
from Comparison_ASTC import ASTCCodec, roundtrip_feature_blocks, _astc_roundtrip_superres_base_mip0
from dataset import bilinear_repeat, resize_bilinear_repeat


class ASTCAwareTrainer:
    def _initialize_astc_aware(self, dataset, configs):
        self.astc_codec_backend = configs.astc_codec_backend
        self.astc_codec_start_frac = configs.astc_codec_start_frac
        self.astc_codec = ASTCCodec(self.astcenc_path, self.astcenc_quality, self.astc_block)
        self._codec_phase_logged = False
        self._astc_block_optimizer = None
        self._astc_codec_pass_count = 0
        if configs.astc_codec_in_loop_enable and self.astc_codec_backend == "astc_differentiable_proxy":
            if (
                self.astc_block != "6x6"
                or not self.mip0_only
                or not self.model.quantize
                or self.model.direct_diffuse_enabled
            ):
                raise ValueError(
                    "ASTC differentiable proxy requires quantized Mip0, 6x6 and direct diffuse disabled"
                )
            if any(spec.n_features_per_level != 4 for spec in self.model.feature_grid_specs):
                raise ValueError("ASTC differentiable proxy requires four-channel feature grids")
            if configs.astc_decoder_projection_interval > 0:
                self._validate_astc_projection()
            from .differentiable_proxy import native_extension

            native_extension()
        print(
            f"[ASTC Codec] backend={self.astc_codec_backend}, "
            f"in_loop={configs.astc_codec_in_loop_enable}, "
            f"interval={configs.astc_codec_in_loop_interval}, start_frac={self.astc_codec_start_frac:.3f}"
        )
        if self.enable_astc_compare or self.enable_pbr_compare or configs.astc_codec_in_loop_enable:
            self.astc_codec.ensure_executable()
        self.astc_superres_base_mip0 = None
        if configs.astc_codec_in_loop_enable and self.super_resolution_enable:
            decoded = _astc_roundtrip_superres_base_mip0(
                dataset, self.astc_codec.roundtrip_rgba, configs.super_resolution_base_resolution
            )
            self.astc_superres_base_mip0 = decoded[0].permute(1, 2, 0).contiguous()

    def _begin_astc_step(self, curr_iter):
        if self._astc_block_optimizer is not None:
            self._astc_block_optimizer.zero_grad()
        start = int(self.max_iter * self.astc_codec_start_frac)
        active = self.configs.astc_codec_in_loop_enable and curr_iter >= start
        if active and not self._codec_phase_logged:
            print(f"[ASTC Codec] phase active at iter {curr_iter}", flush=True)
            self._codec_phase_logged = True
        return active and (curr_iter - start) % self.configs.astc_codec_in_loop_interval == 0

    def _ensure_astc_proxy(self):
        from .differentiable_proxy import ASTCBlockAdam, initialize_astc_proxy

        if not len(self.model.astc_proxy):
            print(
                "[ASTC Codec] initializing persistent endpoints/weights using astcenc", flush=True
            )
            self.model.astc_proxy = initialize_astc_proxy(self.model, self.astc_codec)
        if self._astc_block_optimizer is None:
            self._astc_block_optimizer = ASTCBlockAdam(
                self.model.astc_proxy,
                self.configs.astc_learning_rate,
                self.configs.astc_learning_rate,
                fused=True,
                quantization_aware=True,
                packed_pair_search=True,
            )

    def _training_batch_loss(self, inputs, target, base, weights, curr_iter, codec_due):
        if codec_due and self.astc_codec_backend == "astc_differentiable_proxy":
            total, primary = self._astc_proxy_batch_loss(inputs, target, weights, curr_iter)
            self.writer.add_scalar("Loss/base", primary.item(), curr_iter)
            return total
        # Warmup, ordinary CPU steps and frozen decoder adaptation share the
        # existing reconstruction path. CPU codec steps replace only primary
        # reconstruction/PBR, retaining the original clean subpixel supervision.
        prediction = self.model(inputs)
        primary = self._compute_astc_reconstruction_loss(
            target, prediction, base, weights, curr_iter
        )
        total = primary
        pbr_clean = self._astc_pbr_loss(target, prediction, base)
        subpixel = None
        if self.subpixel_sampling_enable and self.subpixel_sampling_ratio > 0:
            subinput, subtarget, subbase = self._sample_astc_subpixel_batch(proxy=False)
            subpixel = self._compute_astc_reconstruction_loss(
                subtarget, self.model(subinput), subbase, weights
            )
        pbr_codec = None
        if codec_due:
            uv, gt, _, codec_input, codec_base = self._sample_astc_codec_batch()
            self._prepare_astc_codec_patches(uv, curr_iter)
            self.model.astc_codec_branch_enabled = True
            try:
                decoded = self.model(codec_input)
            finally:
                self.model.astc_codec_branch_enabled = False
                self.model._astc_codec_patches.clear()
            total = self._compute_astc_reconstruction_loss(gt, decoded, codec_base, weights)
            pbr_codec = self._astc_pbr_loss(gt, decoded, codec_base)
            self.writer.add_scalar("Loss/astc_codec", total.item(), curr_iter)
            self.writer.add_scalar("ASTCInLoop/samples", len(uv), curr_iter)
        if subpixel is not None:
            total = total + self.subpixel_sampling_ratio * subpixel
        if pbr_clean is not None:
            pbr = pbr_codec if pbr_codec is not None else pbr_clean
            total = total + self.pbr_loss_weight * pbr
            self.writer.add_scalar("Loss/pbr_render", pbr.item(), curr_iter)
            self.writer.add_scalar("Loss/pbr_render_clean", pbr_clean.item(), curr_iter)
            if pbr_codec is not None:
                self.writer.add_scalar("Loss/pbr_render_astc", pbr_codec.item(), curr_iter)
            self.writer.add_scalar(
                "LossWeighted/pbr_render", self.pbr_loss_weight * pbr.item(), curr_iter
            )
        self.writer.add_scalar("Loss/base", primary.item(), curr_iter)
        return total

    def _astc_pbr_loss(self, target, residual, base):
        if not self.pbr_loss_enable or self.pbr_loss_weight <= 0:
            return None
        prediction = (base + residual).clamp(0, 1) if base is not None else residual.clamp(0, 1)
        target, prediction = target.float(), prediction.float()
        if (self.astc_codec_backend == "astc_differentiable_proxy"
                and target.is_cuda and len(target) == self.batch_size):
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
        target = self.dataset.expand_to_canonical(self.dataset.sample_continuous(xy, sample_mips)).to(
            torch.float16
        )
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
            prediction = self.model(inputs)
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
                    subtarget, self.model(subinput), subbase, weights
                )
                total = total + self.subpixel_sampling_ratio * subloss
            self.writer.add_scalar("Loss/astc_codec", primary.item(), curr_iter)
            return total, primary
        finally:
            self.model.astc_codec_branch_enabled = False

    @torch.no_grad()
    def _astc_proxy_objective(self):
        """Re-evaluate the same batches with fixed decoder for candidate acceptance."""
        self.model._qat_param_cache.clear()
        self.model.astc_codec_branch_enabled = True
        try:
            total = 0
            for inputs, target, base, weight, use_pbr in self._astc_proxy_batches:
                prediction = self.model(inputs)
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

    @torch.no_grad()
    def _prepare_astc_codec_patches(self, uv, curr_iter):
        from Core.Feature_Gradient.sampling import bilinear_texel_corners, gradient_directions

        block_size = uv.new_tensor(self.configs.astc_aware_block, dtype=torch.long)
        self.model._astc_codec_patches.clear()
        for index, spec in enumerate(self.model.feature_grid_specs):
            resolution = spec.highest_level_slice()[2]
            shifts = uv.new_tensor(((0, 0), *gradient_directions(self.model.feature_gradient_count)))
            queries = (uv[:, None, :] + shifts[None] / resolution).reshape(-1, 2)
            corners = bilinear_texel_corners(queries, resolution)[0].reshape(-1, 2)
            blocks = torch.unique(torch.div(corners, block_size, rounding_mode="floor"), dim=0)
            indices, decoded, rms = roundtrip_feature_blocks(
                self.model,
                index,
                blocks,
                self.astc_codec.roundtrip_rgba,
                self.configs.astc_aware_block,
            )
            self.model._astc_codec_patches[index] = (indices, decoded)
            self.writer.add_scalar(f"ASTCInLoop/grid{index}_rms_codes", rms, curr_iter)
            self.writer.add_scalar(f"ASTCInLoop/grid{index}_blocks", len(blocks), curr_iter)

    def _step_astc_parameters(self, codec_due):
        if not codec_due or self.astc_codec_backend != "astc_differentiable_proxy":
            return
        search = self._astc_codec_pass_count % 100 == 0
        self._astc_codec_pass_count += 1
        optimizer = self._astc_block_optimizer
        if search:
            touched = {
                index: torch.where(
                    (grid.endpoints.grad.abs().sum(1) + grid.weights.grad.abs().sum(1)) > 0
                )[0]
                for index, grid in enumerate(self.model.astc_proxy)
            }
            optimizer.step(touched, objective=self._astc_proxy_objective)
            self.writer.add_scalar(
                "ASTCInLoop/updated_blocks",
                sum(len(ids) for ids in touched.values()),
                self.model.current_iter,
            )
        else:
            optimizer.step_full_grid()
        for name, value in optimizer.last_stats.items():
            self.writer.add_scalar("ASTCQuantized/" + name, value, self.model.current_iter)
        interval = self.configs.astc_decoder_projection_interval
        if interval > 0 and self._astc_codec_pass_count % interval == 0:
            self._project_astc_decoder_target()

    def _finish_astc_step(self, codec_due):
        if codec_due and self.astc_codec_backend == "astc_differentiable_proxy":
            self._sync_astc_source()

    def _compute_astc_reconstruction_loss(self, gt, prediction, base, loss_weights, curr_iter=None):
        if base is None:
            return self._compute_reconstruction_loss(gt, prediction, loss_weights, curr_iter)
        return self._compute_superres_residual_loss(gt - base, prediction, loss_weights, curr_iter)

    @torch.no_grad()
    def _sample_astc_codec_batch(self):
        """Sample every texel of a few current feature-grid ASTC blocks."""
        resolution = self.model.feature_grid_specs[0].highest_level_slice()[2]
        block_w, block_h = self.configs.astc_aware_block
        blocks_x = (resolution + block_w - 1) // block_w
        blocks_y = (resolution + block_h - 1) // block_h
        count = min(32, blocks_x * blocks_y)
        block_ids = torch.randperm(blocks_x * blocks_y, device=self.device)[:count]
        block_x, block_y = block_ids % blocks_x, block_ids // blocks_x
        local_y, local_x = torch.meshgrid(
            torch.arange(block_h, device=self.device),
            torch.arange(block_w, device=self.device),
            indexing="ij",
        )
        xs = block_x[:, None, None] * block_w + local_x
        ys = block_y[:, None, None] * block_h + local_y
        valid = (xs < resolution) & (ys < resolution)
        uvs = torch.stack(((xs[valid] + 0.5) / resolution, (ys[valid] + 0.5) / resolution), dim=1)
        sample_xy = torch.stack(
            (uvs[:, 1] * self.texture_height - 0.5, uvs[:, 0] * self.texture_width - 0.5), dim=1
        )
        mips = torch.zeros(len(uvs), device=self.device, dtype=torch.long)
        gt = self.dataset.expand_to_canonical(self.dataset.sample_continuous(sample_xy, mips)).to(
            torch.float16
        )
        model_uv = torch.cat((uvs, torch.zeros((len(uvs), 1), device=self.device)), dim=1)
        if not self.super_resolution_enable:
            return uvs, gt, model_uv, model_uv, None

        clean_base = self.dataset.expand_to_canonical(
            self.dataset.get_superres_base_continuous(sample_xy, mips)
        ).to(torch.float16)
        codec_base = self._astc_base_at_uv(uvs)
        return (
            uvs,
            gt,
            torch.cat((model_uv, clean_base), dim=1),
            torch.cat((model_uv, codec_base), dim=1),
            codec_base,
        )

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
    def _sync_astc_source(self):
        # The decoded trainable bitstream is the feature texture in this mode.
        # Keep exports, ordinary eval and the subsequent frozen stage consistent.
        from feature_grid import unorm_to_feature, quantize_feature_tensor

        for grid, spec, codec in zip(
            self.model.feature_grids, self.model.feature_grid_specs, self.model.astc_proxy
        ):
            offset, count, _ = spec.highest_level_slice()
            grid.params[offset : offset + count].copy_(
                quantize_feature_tensor(
                    unorm_to_feature(codec.decode_image().reshape(-1), spec), spec
                )
            )
        self.model._qat_param_cache.clear()

    @torch.no_grad()
    def _project_astc_decoder_target(self):
        """Fit legal ASTC to a weighted linear-decoder target at grid texel centers.

        Unlike a normalized feature-gradient step, this proposal solves for the
        latent that minimizes material MSE at those centers before ASTC projection.
        Accept only the actual sampled material objective, keeping codec training
        on CUDA on all ordinary iterations.
        """
        from Core.ASTC_Aware.differentiable_proxy import ASTCProxyGrid, encode_seed
        from feature_grid import feature_to_unorm

        self._validate_astc_projection()
        model = self.model
        grid = model.astc_proxy[0]
        spec = model.feature_grid_specs[0]
        channels = model.num_channels
        dim = 4 + 1 + channels + 1  # feature basis, mip, SR base, explicit bias
        basis = torch.zeros((5, dim), device=self.device)
        basis[1:, :4] = torch.eye(4, device=self.device)
        basis[:, -1] = 1
        values = model.network(basis).float()
        matrix = (values[1:] - values[:1]).T  # [channels,4]
        weights = torch.tensor(self.output_loss_weights, device=self.device)
        gram = matrix.T @ (weights[:, None] * matrix)
        ridge = 1e-5 * gram.diag().mean().clamp_min(1e-6)
        inverse = torch.linalg.inv(gram + torch.eye(4, device=self.device) * ridge)
        transform = (weights[:, None] * matrix) @ inverse
        resolution = spec.highest_level_slice()[2]
        # Solve at feature texel centers. Bilinear resizing samples the same UVs
        # from GT and the deployment SR base when their dimensions differ from
        # the grid (e.g. 2K textures with the default 1K feature grid).
        target_image = self.dataset.mip_cache[0].permute(2, 0, 1)[None].float()
        base_image = self.astc_superres_base_mip0.permute(2, 0, 1)[None].float()
        if target_image.shape[-2:] != (resolution, resolution):
            target_image = F.interpolate(target_image, size=(resolution, resolution),
                                         mode="bilinear", align_corners=False)
            base_image = resize_bilinear_repeat(base_image, (resolution, resolution))
        target = self.dataset.expand_to_canonical(
            target_image[0].permute(1, 2, 0).reshape(-1, self.dataset.num_channels)
        ).float()
        base = base_image[0].permute(1, 2, 0).reshape(-1, channels)
        projected = []
        for begin in range(0, len(base), 65536):
            batch = base[begin : begin + 65536]
            inputs = torch.zeros((len(batch), dim), device=self.device)
            inputs[:, 5 : 5 + channels] = batch
            inputs[:, -1] = 1
            residual = (target[begin : begin + len(batch)] - batch
                        - model.network(inputs).float() - model.decoder_output_bias.float())
            projected.append(
                (residual @ transform).clamp(
                    spec.quant_min, spec.quant_min + (spec.quant_step_count - 1) / spec.quant_step_count
                )
            )
        latent = torch.cat(projected).reshape(resolution, resolution, 4)
        rgba = (feature_to_unorm(latent, spec) * 255).round().byte().cpu().numpy()
        proposal = ASTCProxyGrid.from_astc(encode_seed(self.astc_codec, rgba), self.device)
        baseline = float(self._astc_proxy_objective())
        names = ("endpoints", "weights", "metadata", "partitions", "indices", "coefficients")
        saved = {name: getattr(grid, name).clone() for name in names}
        for name in names:
            getattr(grid, name).copy_(getattr(proposal, name))
        try:
            candidate = float(self._astc_proxy_objective())
        except Exception:
            for name in names:
                getattr(grid, name).copy_(saved[name])
            raise
        accepted = math.isfinite(candidate) and candidate < baseline
        if accepted:
            self._astc_block_optimizer.reset_rows(
                0, torch.arange(len(grid.endpoints), device=self.device)
            )
        else:
            for name in names:
                getattr(grid, name).copy_(saved[name])
        self.writer.add_scalar("ASTCProjection/accepted", int(accepted), model.current_iter)
        self.writer.add_scalar("ASTCProjection/baseline_loss", baseline, model.current_iter)
        self.writer.add_scalar("ASTCProjection/candidate_loss", candidate, model.current_iter)

    def _validate_astc_projection(self):
        model = self.model
        if (model.n_hidden_layers != 0 or model.n_frequencies != 0
                or model.feature_gradient_count != 0
                or len(model.feature_grid_specs) != 1 or not self.super_resolution_enable):
            raise ValueError("Material projection requires one grid and a linear SR decoder without PE/feature gradients; "
                             "set astc_decoder_projection_interval: 0 for other configurations")

    def refine_gradient_codec(self, steps, learning_rate=0.001, blocks_per_batch=64):
        """Fit legal codec symbols with the decoder fixed over complete neighborhoods.

        This explicit refinement pass is separate from frozen decoder adaptation.
        The output patch covers each selected 6x6 block and the gradient/bilinear
        footprint around it (10x10 for 3x3 inputs, 12x12 for 5x5). Selected rows are
        updated, and proposals must improve the actual material/PBR objective.
        """
        from .differentiable_proxy import ASTCBlockAdam
        from Core.Feature_Gradient.sampling import gradient_directions

        model = self.model
        if (self.astc_codec_backend != "astc_differentiable_proxy" or not self.mip0_only
                or len(model.astc_proxy) != 1 or not model.feature_gradient_count
                or model._decoder_adaptation_features is not None):
            raise ValueError("Codec refinement requires a fitted gradient decoder and one Mip0 ASTC proxy")
        if steps < 1 or blocks_per_batch < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Refinement steps, block count and learning rate must be positive")
        grid = model.astc_proxy[0]
        height, width = grid._image_shape
        if (height, width) != (self.texture_height, self.texture_width):
            raise ValueError("Codec refinement currently requires feature and target resolutions to match")
        if self.super_resolution_enable and self.astc_superres_base_mip0 is None:
            raise ValueError("Codec refinement requires the deployment ASTC base")
        blocks_x, blocks_y = (width + 5) // 6, (height + 5) // 6
        optimizer = ASTCBlockAdam(model.astc_proxy, learning_rate, learning_rate,
                                  quantization_aware=True, packed_pair_search=True)
        loss_weights = torch.tensor(self.output_loss_weights, device=self.device)
        halo = 1 + max(max(abs(dx), abs(dy)) for dx, dy in gradient_directions(model.feature_gradient_count))
        ly, lx = torch.meshgrid(torch.arange(-halo, 6 + halo, device=self.device),
                               torch.arange(-halo, 6 + halo, device=self.device), indexing="ij")
        parameter_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
        was_training, old_iteration = model.training, model.current_iter
        old_branch = model.astc_codec_branch_enabled
        old_batches = getattr(self, "_astc_proxy_batches", None)
        old_weights = getattr(self, "_astc_proxy_loss_weights", None)
        saved = (grid.endpoints.detach().clone(), grid.weights.detach().clone())
        saved_source = model.feature_grids[0].params.detach().clone()
        stats = {"steps": steps, "accepted_blocks": 0, "continuous_accepted": 0,
                 "learning_rate": learning_rate, "blocks_per_batch": blocks_per_batch}
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
                xs = (ids[:, None, None] % blocks_x * 6 + lx) % width
                ys = (ids[:, None, None] // blocks_x * 6 + ly) % height
                uv = torch.stack(((xs.flatten() + .5) / width, (ys.flatten() + .5) / height), 1)
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
                prediction = model(inputs)
                loss = self._compute_astc_reconstruction_loss(target, prediction, base, loss_weights)
                pbr = self._astc_pbr_loss(target, prediction, base)
                if pbr is not None:
                    loss = loss + self.pbr_loss_weight * pbr
                loss.backward()
                self._astc_proxy_batches = [(inputs, target, base, 1., True)]
                self._astc_proxy_loss_weights = loss_weights
                optimizer.step({0: ids}, objective=self._astc_proxy_objective)
                stats["accepted_blocks"] += optimizer.last_stats.get("accepted_blocks", 0)
                stats["continuous_accepted"] += optimizer.last_stats.get("continuous_accepted", 0)
            self._sync_astc_source()
            self._astc_block_optimizer = None
            return stats
        except Exception:
            with torch.no_grad():
                grid.endpoints.copy_(saved[0])
                grid.weights.copy_(saved[1])
                model.feature_grids[0].params.copy_(saved_source)
            raise
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
