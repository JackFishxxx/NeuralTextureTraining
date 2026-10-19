"""Offline latent refitting and legal ASTC block selection; inference is unchanged."""
import math

import torch

from Core.Feature_Grid.feature_gradients import gradient_directions
from Core.Feature_Grid.feature_grid import feature_to_unorm, unorm_to_feature
from Core.ASTC_Aware.differentiable_proxy import ASTCProxyGrid, encode_seed
from dataset import material_size


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
def _select(trainer, proposal):
    model, grid = trainer.model, trainer.model.astc_proxy[0]
    height, width = grid._image_shape
    halo = 1 + max(max(abs(x), abs(y)) for x, y in gradient_directions(model.feature_gradient_count))
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
            inputs = torch.cat((uv, torch.zeros(len(uv), 1, device=model.device), base), 1)

            def losses():
                model._qat_param_cache.clear()
                residual = trainer._predict_material(inputs).float()
                errors = ((residual - (target - base).float()).square() * weights).sum(1).reshape(len(rows), -1)
                value = errors[:, :centers].mean(1)
                if subpixel:
                    value += trainer.subpixel_sampling_ratio * errors[:, centers:].mean(1)
                if trainer.pbr_loss_enable and trainer.pbr_loss_weight > 0:
                    gt = target.float().reshape(len(rows), len(local), -1)[:, :centers]
                    pred = (base.float() + residual).clamp(0, 1).reshape(len(rows), len(local), -1)[:, :centers]
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
    """Refit a temporary field and retain the best formally validated codec every 100 steps.

    Decoder, optimizer and sampling RNG are preserved. Rejection or any exception
    restores codec/source state. This adds offline work only, with no readaptation.
    """
    model = trainer.model
    if steps < 0:
        raise ValueError('Layout refit steps must be nonnegative')
    if not steps:
        return {'applied': False, 'reason': 'disabled'}
    if (trainer.astc_codec_backend != 'astc_differentiable_proxy' or not trainer.mip0_only
            or not model.quantize or len(model.astc_proxy) != 1 or len(model.feature_grids) != 1
            or model.feature_grid_specs[0].n_features_per_level != 4
            or not model.feature_gradient_count or model._decoder_adaptation_features is not None
            or not trainer.super_resolution_enable or trainer.astc_superres_base_mip0 is None
            or model.astc_proxy[0]._image_shape != material_size(model, trainer.texture_height, trainer.texture_width)
            or trainer.texture_height % model.astc_proxy[0]._image_shape[0]
            or trainer.texture_width % model.astc_proxy[0]._image_shape[1]):
        return {'applied': False, 'reason': 'requires one gradient SR codec at native material resolution and integer GT scale'}
    grid, spec = model.astc_proxy[0], model.feature_grid_specs[0]
    source = model.feature_grids[0].params
    saved_source = source.detach().clone()
    saved_codec = {k: v.detach().clone() for k, v in grid.state_dict().items()}
    saved_stream = grid.export_astc()
    flags = [(p, p.requires_grad) for p in model.parameters()]
    training, branch = model.training, model.astc_codec_branch_enabled
    device = torch.device(model.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    weights = torch.tensor(trainer.output_loss_weights, device=model.device)
    method = 'fntc_astc_' + trainer.astc_block
    stats = {'applied': True, 'steps': steps, 'accepted': False, 'best_step': 0, 'checks': []}
    try:
        before = trainer.run_astc_comparison(curr_iter=model.current_iter)
        before_pbr = trainer.run_pbr_comparison(curr_iter=model.current_iter, return_metrics=True)[1] if trainer.enable_pbr_compare else None
        best_material, best_pbr = before[method]['average'], before_pbr
        best_codec = saved_codec
        field = source.detach().clone()
        offset, count, resolution = spec.highest_level_slice()
        with torch.no_grad():
            field[offset:offset + count] = unorm_to_feature(grid.decode_image().flatten(), spec)
        initial = field.clone()
        field.requires_grad_(True)
        for p, _ in flags:
            p.requires_grad_(False)
        model.train()
        model.astc_codec_branch_enabled = False
        model._decoder_adaptation_features = [field]
        optimizer = torch.optim.Adam([field], lr=.0003)
        height, width = trainer.texture_height, trainer.texture_width

        def reconstruction(ids):
            uv = torch.stack(((ids % width + .5) / width, (ids // width + .5) / height), 1)
            target = trainer.dataset.expand_to_canonical(trainer.dataset.mip_cache[0][ids // width, ids % width]).to(
                torch.float32 if material_size(model, height, width) != (height, width) else torch.float16)
            base = trainer._astc_base_at_uv(uv)
            residual = trainer._predict_material(torch.cat((uv, torch.zeros(len(uv), 1, device=model.device), base), 1))
            loss = trainer._compute_astc_reconstruction_loss(target, residual, base, weights)
            if trainer.pbr_loss_enable and trainer.pbr_loss_weight > 0:
                loss += trainer.pbr_loss_weight * trainer._compute_pbr_render_loss(target.float(), (base.float() + residual.float()).clamp(0, 1))
            return loss

        def evaluate(candidate, step):
            nonlocal best_codec, best_material, best_pbr
            # Formal evaluation and encoding must not perturb later training batches.
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                model._decoder_adaptation_features = None
                model.train()
                rgba = (feature_to_unorm(candidate[offset:offset + count].reshape(resolution, resolution, 4), spec) * 255).round().byte().cpu().numpy()
                proposal = ASTCProxyGrid.from_astc(encode_seed(trainer.astc_codec, rgba), model.device)
                local_accepted = _select(trainer, proposal)
                trainer._sync_astc_source()
                iteration = model.current_iter + step
                after = trainer.run_astc_comparison(curr_iter=iteration)[method]['average']
                after_pbr = trainer.run_pbr_comparison(curr_iter=iteration, return_metrics=True)[1] if before_pbr is not None else None
                stream = grid.export_astc()
                changed = sum(saved_stream[i:i+16] != stream[i:i+16] for i in range(16, len(saved_stream), 16))
                accepted = (_passes(best_material, after)
                            and (best_pbr is None or _passes(best_pbr['fntc'], after_pbr['fntc'])))
                stats['checks'].append({'step': step, 'material': list(after), 'pbr': after_pbr,
                                        'local_accepted_blocks': local_accepted, 'accepted': accepted})
                if accepted and changed:
                    best_codec = {k: v.detach().clone() for k, v in grid.state_dict().items()}
                    best_material, best_pbr = after, after_pbr
                    stats.update(accepted=True, best_step=step, changed_blocks=changed)
                else:
                    grid.load_state_dict(best_codec)
                # Resume the independent continuous trajectory even after rejection.
                model.train()
                model.astc_codec_branch_enabled = False
                model._decoder_adaptation_features = [field]
                model._qat_param_cache.clear()
            print(f"[Layout refit] step {step}/{steps}: accepted={accepted}, best_step={stats['best_step']}", flush=True)

        with torch.random.fork_rng(devices=devices):
            evaluate(initial, 0)
            for step in range(steps):
                ids = torch.randint(height * width, (trainer.batch_size,), device=model.device)
                optimizer.zero_grad(set_to_none=True)
                loss = reconstruction(ids)
                if trainer.subpixel_sampling_enable and trainer.subpixel_sampling_ratio > 0:
                    inputs, target, base = trainer._sample_astc_subpixel_batch(proxy=True)
                    loss += trainer.subpixel_sampling_ratio * trainer._compute_astc_reconstruction_loss(target, trainer._predict_material(inputs), base, weights)
                loss.backward()
                if not torch.isfinite(loss) or not torch.isfinite(field.grad).all():
                    raise ValueError('Nonfinite latent refit loss or gradient')
                torch.nn.utils.clip_grad_norm_([field], 1.)
                optimizer.step()
                with torch.no_grad():
                    field.clamp_(spec.quant_min, spec.quant_min + (spec.quant_step_count - 1) / spec.quant_step_count)
                if (step + 1) % 100 == 0 or step + 1 == steps:
                    evaluate(field.detach(), step + 1)
        model._decoder_adaptation_features = None
        grid.load_state_dict(best_codec)
        trainer._sync_astc_source()
        stats.update(before=list(before[method]['average']), candidate=list(best_material),
                     before_pbr=before_pbr, candidate_pbr=best_pbr)
        return stats
    finally:
        if not stats['accepted']:
            grid.load_state_dict(saved_codec)
            with torch.no_grad():
                source.copy_(saved_source)
        model._decoder_adaptation_features = None
        model.astc_codec_branch_enabled = branch
        model.train(training)
        for p, flag in flags:
            p.requires_grad_(flag)
        model._qat_param_cache.clear()
        if stats['accepted']:
            trainer._astc_block_optimizer = None
