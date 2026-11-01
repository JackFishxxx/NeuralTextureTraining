"""Offline latent refitting and legal ASTC block selection; inference is unchanged."""
import math
import copy

import torch

from Core.Feature_Grid.feature_gradients import gradient_directions
from Core.Feature_Grid.feature_grid import feature_to_unorm, unorm_to_feature
from dataset import material_size
from Core.Feature_Grid.compression import proxy_streams


def _axis_colors(length, halo):
    # Include repeat seams and the final partial block in the conflict graph.
    regions = [{(6 * i + x) % length for x in range(-halo, 6 + halo)}
               for i in range((length + 5) // 6)]
    colors = []
    for i, region in enumerate(regions):
        used = {colors[j] for j in range(i) if region & regions[j]}
        colors.append(next(c for c in range(len(used) + 1) if c not in used))
    return colors


def _passes(before, after):
    return all(math.isfinite(x) and math.isfinite(y) and (y - x) * sign >= 0
               for x, y, sign in zip(before, after, (1, 1, -1)))


@torch.no_grad()
def _select(trainer, proposal, grid_index=0):
    model, grid = trainer.model, trainer.model.astc_proxy[grid_index]
    height, width = grid._image_shape
    halo = 1 + max((max(abs(x), abs(y)) for x, y in gradient_directions(model.feature_gradient_count)), default=0)
    scale_y, scale_x = trainer.texture_height // height, trainer.texture_width // width
    # Output interpolation expands each block's influence by another native texel.
    if (scale_y, scale_x) != (1, 1):
        halo += 1
    y, x = torch.meshgrid(torch.arange(-halo * scale_y, (6 + halo) * scale_y, device=model.device),
                          torch.arange(-halo * scale_x, (6 + halo) * scale_x, device=model.device), indexing='ij')
    local = torch.stack((y.flatten(), x.flatten()), 1).float()
    centers = len(local)
    subpixel = trainer.subpixel_sampling_enable and trainer.subpixel_sampling_ratio > 0
    if subpixel:
        jitter = local.new_tensor([[-.25, -.25], [.25, -.25], [-.25, .25], [.25, .25]])
        local = torch.cat((local, (local[:, None] + jitter).reshape(-1, 2)))
    cx, cy = _axis_colors(width, halo), _axis_colors(height, halo)
    blocks_x = len(cx)
    groups = {}
    for row in range(len(grid.endpoints)):
        groups.setdefault((cy[row // blocks_x], cx[row % blocks_x]), []).append(row)
    names = ('endpoints', 'weights', 'metadata', 'partitions', 'indices', 'coefficients')
    weights = torch.tensor(trainer.output_loss_weights, device=model.device)
    accepted = 0
    model.astc_codec_branch_enabled = True
    for group in groups.values():
        for start in range(0, len(group), 64):
            rows = torch.tensor(group[start:start + 64], device=model.device)
            origin = torch.stack((rows // blocks_x * 6 * scale_y, rows % blocks_x * 6 * scale_x), 1)
            xy = (origin[:, None] + local).remainder(local.new_tensor((trainer.texture_height, trainer.texture_width)))
            uv = torch.stack(((xy[..., 1] + .5) / trainer.texture_width,
                              (xy[..., 0] + .5) / trainer.texture_height), -1).reshape(-1, 2)
            target = trainer.dataset.expand_to_canonical(trainer.dataset.sample_continuous(xy.reshape(-1, 2), 0)).to(
                torch.float32 if (scale_y, scale_x) != (1, 1) else torch.float16)
            base = trainer._astc_base_at_uv(uv)
            inputs = torch.cat((uv, torch.zeros(len(uv), 1, device=model.device)), 1)
            if base is not None:
                inputs = torch.cat((inputs, base), 1)

            def losses():
                model._qat_param_cache.clear()
                residual = trainer._predict_material(inputs).float()
                prediction = residual if base is None else residual + base.float()
                errors = trainer._material_objective().per_sample(
                    target, prediction, weights).reshape(len(rows), -1)
                value = errors[:, :centers].mean(1)
                if subpixel:
                    value += trainer.subpixel_sampling_ratio * errors[:, centers:].mean(1)
                if trainer.pbr_loss_enable and trainer.pbr_loss_weight > 0:
                    gt = target.float().reshape(len(rows), len(local), -1)[:, :centers]
                    pred = prediction.clamp(0, 1).reshape(len(rows), len(local), -1)[:, :centers]
                    trainer._compute_pbr_render_loss(gt[0], pred[0])  # Initialize lighting before vmap.
                    value += trainer.pbr_loss_weight * torch.vmap(trainer._compute_pbr_render_loss)(gt, pred)
                return value

            before = losses()
            saved = {name: getattr(grid, name)[rows].clone() for name in names}
            for name in names:
                getattr(grid, name)[rows] = getattr(proposal, name)[rows]
            after = losses()
            keep = torch.isfinite(before) & torch.isfinite(after) & (after < before)
            for name in names:
                getattr(grid, name)[rows[~keep]] = saved[name][~keep]
            accepted += int(keep.sum())
    model._qat_param_cache.clear()
    return accepted


def refit_layout(trainer, steps):
    """Jointly refit latent fields and validate legal per-grid codec proposals.

    Native ASTC supports local block selection. Other footprints and BC1 use
    whole-grid proposals; all formats share the full-image acceptance gate.
    Rejection and exceptions restore every grid together.
    """
    from Core.Feature_Grid.compression import (
        freeze_proxies, proxy_from_image, proxy_kind,
    )

    model = trainer.model
    if steps < 0:
        raise ValueError('Layout refit steps must be nonnegative')
    if not steps:
        return {'applied': False, 'reason': 'disabled'}
    if (not trainer.mip0_only or not model.quantize or model.direct_diffuse_enabled
            or model._decoder_adaptation_features is not None):
        return {'applied': False, 'reason': 'requires quantized Mip0 outside direct diffuse/adaptation'}
    if trainer.super_resolution_enable and trainer.astc_superres_base_mip0 is None:
        raise ValueError('Layout refitting requires the deployment SR base')
    original_proxies = torch.nn.ModuleList(list(model.astc_proxy))
    saved_codecs = [{k: v.detach().clone() for k, v in proxy.state_dict().items()}
                    for proxy in original_proxies]
    sources = [grid.params for grid in model.feature_grids]
    saved_sources = [source.detach().clone() for source in sources]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    proxy_flags = [dict((name, p.requires_grad) for name, p in proxy.named_parameters())
                   for proxy in original_proxies]
    training, branch = model.training, model.astc_codec_branch_enabled
    active = getattr(model, '_codec_active_grids', None)
    previous_phase = getattr(trainer, '_metrics_phase', None)
    device = torch.device(model.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    weights = torch.tensor(trainer.output_loss_weights, device=model.device)
    stats = {'applied': True, 'steps': steps, 'accepted': False, 'best_step': 0,
             'grids': list(range(len(sources))), 'checks': []}
    completed = False
    try:
        freeze_proxies(model)
        trainer._metrics_phase = 'layout_refit'
        saved_streams = [proxy_streams(proxy) for proxy in model.astc_proxy]
        before = trainer.deployment_metrics()['average']
        before_pbr = trainer.run_pbr_comparison(curr_iter=model.current_iter, return_metrics=True, save_outputs=False)[1] if trainer.enable_pbr_compare else None
        best_material, best_pbr = before, before_pbr
        best_proxies = copy.deepcopy(model.astc_proxy)
        fields = [source.detach().clone() for source in sources]
        for field, spec, proxy in zip(fields, model.feature_grid_specs, model.astc_proxy):
            offset, count, _ = spec.highest_level_slice()
            field[offset:offset+count] = unorm_to_feature(proxy.decode_image().detach().flatten(), spec)
            field.requires_grad_(True)
        for p, _ in flags:
            p.requires_grad_(False)
        model.train()
        model.astc_codec_branch_enabled = False
        model._decoder_adaptation_features = fields
        optimizer = torch.optim.Adam(fields, lr=.0003)
        height, width = trainer.texture_height, trainer.texture_width

        def reconstruction(ids):
            uv = torch.stack(((ids % width + .5)/width, (ids // width + .5)/height), 1)
            target = trainer.dataset.expand_to_canonical(trainer.dataset.mip_cache[0][ids // width, ids % width]).to(
                torch.float32 if material_size(model, height, width) != (height, width) else torch.float16)
            base = trainer._astc_base_at_uv(uv)
            inputs = torch.cat((uv, torch.zeros(len(uv), 1, device=model.device)), 1)
            if base is not None:
                inputs = torch.cat((inputs, base), 1)
            prediction = trainer._predict_material(inputs)
            loss = trainer._compute_astc_reconstruction_loss(target, prediction, base, weights)
            if trainer.pbr_loss_enable and trainer.pbr_loss_weight > 0:
                material = prediction.float() if base is None else prediction.float()+base.float()
                loss += trainer.pbr_loss_weight * trainer._compute_pbr_render_loss(target.float(), material.clamp(0, 1))
            return loss

        def evaluate(step):
            nonlocal best_proxies, best_material, best_pbr
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                model._decoder_adaptation_features = None
                model.train()
                local_accepted = {}
                for index, (field, spec) in enumerate(zip(fields, model.feature_grid_specs)):
                    offset, count, resolution = spec.highest_level_slice()
                    image = feature_to_unorm(field[offset:offset+count].reshape(
                        resolution, resolution, spec.n_features_per_level), spec)
                    proposal = proxy_from_image(model, index, image, training_seed=False)
                    proposal.requires_grad_(False)
                    if (proxy_kind(proposal) == 'astc_native'
                            and not height % resolution and not width % resolution):
                        local_accepted[index] = _select(trainer, proposal, index)
                    else:
                        if proxy_kind(proposal) == 'astc_roundtrip':
                            proposal.freeze()
                        model.astc_proxy[index] = proposal
                trainer._sync_astc_source()
                iteration = model.current_iter + step
                after = trainer.deployment_metrics()['average']
                after_pbr = trainer.run_pbr_comparison(curr_iter=iteration, return_metrics=True,
                                                      save_outputs=False)[1] if before_pbr is not None else None
                changed = sum(_changed_blocks(old, proxy_streams(proxy), recipe.format)
                              for old, proxy, recipe in zip(saved_streams, model.astc_proxy, model.grid_compressions))
                accepted = (_passes(best_material, after)
                            and (best_pbr is None or _passes(best_pbr['fntc'], after_pbr['fntc'])))
                stats['checks'].append({'step': step, 'material': list(after), 'pbr': after_pbr,
                                        'local_accepted_blocks': local_accepted, 'accepted': accepted})
                if accepted and changed:
                    best_proxies = copy.deepcopy(model.astc_proxy)
                    best_material, best_pbr = after, after_pbr
                    stats.update(accepted=True, best_step=step, changed_blocks=changed)
                model.astc_proxy = copy.deepcopy(best_proxies)
                trainer._sync_astc_source()
                model.train()
                model.astc_codec_branch_enabled = False
                model._decoder_adaptation_features = fields
                model._qat_param_cache.clear()
            print(f"[Layout refit] step {step}/{steps}: PSNR={after[0]:.4f}, "
                  f"accepted={accepted}, best_step={stats['best_step']}", flush=True)

        with torch.random.fork_rng(devices=devices):
            evaluate(0)
            for step in range(steps):
                ids = torch.randint(height*width, (trainer.batch_size,), device=model.device)
                optimizer.zero_grad(set_to_none=True)
                loss = reconstruction(ids)
                if trainer.subpixel_sampling_enable and trainer.subpixel_sampling_ratio > 0:
                    inputs, target, base = trainer._sample_astc_subpixel_batch(proxy=True)
                    loss += trainer.subpixel_sampling_ratio * trainer._compute_astc_reconstruction_loss(
                        target, trainer._predict_material(inputs), base, weights)
                loss.backward()
                if not torch.isfinite(loss) or any(field.grad is None or not torch.isfinite(field.grad).all() for field in fields):
                    raise ValueError('Nonfinite latent refit loss or gradient')
                torch.nn.utils.clip_grad_norm_(fields, 1.)
                optimizer.step()
                with torch.no_grad():
                    for field, spec in zip(fields, model.feature_grid_specs):
                        field.clamp_(spec.quant_min, spec.quant_min+(spec.quant_step_count-1)/spec.quant_step_count)
                if (step+1) % 100 == 0 or step+1 == steps:
                    evaluate(step+1)
        model._decoder_adaptation_features = None
        model.astc_proxy = best_proxies
        trainer._sync_astc_source()
        stats.update(before=list(before), candidate=list(best_material),
                     before_pbr=before_pbr, candidate_pbr=best_pbr)
        completed = True
        return stats
    finally:
        if not completed or not stats['accepted']:
            model.astc_proxy = original_proxies
            for proxy, saved in zip(original_proxies, saved_codecs):
                proxy.load_state_dict(saved)
            with torch.no_grad():
                for source, saved in zip(sources, saved_sources):
                    source.copy_(saved)
        model._decoder_adaptation_features = None
        model.astc_codec_branch_enabled = branch
        model._codec_active_grids = active
        model.train(training)
        for p, flag in flags:
            p.requires_grad_(flag)
        for i, proxy in enumerate(model.astc_proxy):
            for name, p in proxy.named_parameters():
                p.requires_grad_(proxy_flags[i].get(name, False) if i < len(proxy_flags) else False)
        model._qat_param_cache.clear()
        trainer._metrics_phase = previous_phase
        if completed and stats['accepted']:
            trainer._astc_block_optimizer = None


def _changed_blocks(before, after, latent_format):
    changed = 0
    for old, new in zip(before, after):
        start, size = (16, 16) if latent_format == 'astc' else (0, 8)
        changed += sum(old[i:i+size] != new[i:i+size] for i in range(start, len(old), size))
    return changed
