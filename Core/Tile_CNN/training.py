"""Offline CNN fitting and transactional full-image deployment acceptance."""
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict

import torch
from torch.nn import functional as F

from Common.material_reconstruction import cached_astc_base, make_deployment_model, _render_model_mip0
from dataset import material_size
from .material_cnn import MaterialCNN
from checkpoint import atomic_torch_save


def validate_training_size(gt_size, inference_size):
    if (len(gt_size) != 2 or len(inference_size) != 2
            or any(gt < 1 or native < 1 or gt % native for gt, native in zip(gt_size, inference_size))):
        raise ValueError('Material CNN fitting requires an integer GT/inference size ratio; '
                         f'GT={tuple(gt_size)}, inference={tuple(inference_size)}')


@torch.no_grad()
def _full_quality(trainer, raw, gt, post, weights):
    native = raw
    if post is not None:
        training = post.training
        try:
            native = post.eval()(raw)
        finally:
            post.train(training)
    evaluation = trainer.material_evaluation(baseline=trainer.enable_pbr_compare, material_override=native)
    prediction = evaluation.images[trainer.primary_method]
    if not torch.isfinite(prediction).all():
        raise ValueError('Nonfinite material CNN full-image prediction')
    metrics = trainer._compute_material_metrics(prediction, gt)
    # Bound loss temporaries independently of the complete metric raster.
    pred_flat, gt_flat = prediction.flatten(2)[0], gt.flatten(2)[0]
    total = prediction.new_zeros(())
    for start in range(0, pred_flat.shape[1], 65536):
        total += trainer._material_objective().per_sample(
            gt_flat[:, start:start+65536].T, pred_flat[:, start:start+65536].T, weights).sum()
    quality = dict(loss=float(total / pred_flat.shape[1]), material=metrics, pbr=None)
    del native, pred_flat, gt_flat
    if trainer.enable_pbr_compare:
        metrics = trainer.run_pbr_comparison(
            return_metrics=True, save_outputs=False, include_detail_metrics=True,
            evaluation=evaluation)[1]
        quality['pbr'] = {view: values[trainer.primary_method] for view,values in metrics.items()}
    return quality


def accept_quality(before, after):
    """Require useful PSNR gain, no material-group/view regression, and finite scores."""
    if (not math.isfinite(before['loss']) or not math.isfinite(after['loss'])
            or after['loss'] > before['loss'] + 1e-8):
        return False, 'full-image material loss regressed or is nonfinite'
    for category in ('material', 'pbr'):
        reference = before[category] or {}
        candidate = after[category] or {}
        for group, scores in reference.items():
            if scores is None:
                continue
            values = candidate.get(group)
            if values is None or len(values) != 3:
                return False, f'missing {category}/{group} metrics'
            for name, sign, tolerance, old, new in zip(
                    ('PSNR', 'SSIM', 'LPIPS'), (1, 1, -1), (1e-4, 1e-6, 1e-6), scores, values):
                if not math.isfinite(old) or not math.isfinite(new) or (new-old)*sign < -tolerance:
                    return False, f'{category}/{group}/{name} regressed or is nonfinite'
    gain = after['material']['average'][0] - before['material']['average'][0]
    if gain < .01:
        return False, 'full-image weighted PSNR gain is below 0.01 dB'
    return True, 'full-image material and enabled PBR views passed'


def _signature(raw, gt, trainer, weights):
    digest = hashlib.sha256()
    for tensor in (raw, gt, weights):
        digest.update(memoryview(tensor.detach().cpu().contiguous().numpy()))
    recipe = dict(normal_encoding=trainer.normal_encoding, pbr_loss=trainer.pbr_loss_enable,
                  pbr_weight=trainer.pbr_loss_weight, shape=list(raw.shape), gt_shape=list(gt.shape),
                  texture_packing=trainer.dataset.texture_packing,
                  tile_size=trainer.configs.fntc_tile_size,
                  cnn_version=MaterialCNN.version)
    digest.update(json.dumps(recipe, sort_keys=True).encode())
    return digest.hexdigest()


def fit_material_cnn(trainer, steps=50000, validation_interval=1000, resume_state=None):
    if steps < 1 or validation_interval < 1:
        raise ValueError('Material CNN steps and validation interval must be positive')
    model = trainer.model
    if not trainer.mip0_only or model.material_cnn is not None:
        raise ValueError('Material CNN fitting requires an unfiltered Mip0 decoder')
    size = material_size(model,trainer.texture_height,trainer.texture_width,
                         resolution=trainer.material_compare_resolution)
    gt_size = (trainer.texture_height, trainer.texture_width)
    validate_training_size(gt_size, size)
    infer_h, infer_w = size
    sy, sx = (gt // native for gt, native in zip(gt_size, size))
    started = time.perf_counter()
    with torch.no_grad():
        decoded = make_deployment_model(model)
        base = (cached_astc_base(trainer.dataset, trainer.astc_codec,
                                trainer.configs.super_resolution_base_resolution, output_size=size)
                if trainer.super_resolution_enable else None)
        raw = _render_model_mip0(decoded, trainer.dataset, *size, trainer.num_mips, trainer.device,
                                 superres_base_override=base, inference_tile=trainer.configs.eval_inference_tile,
                                 inference_size=size)
        del decoded, base
    gt = trainer.dataset.expand_to_canonical(
        trainer.dataset.mip_cache[0]).permute(2, 0, 1)[None].float()
    tile_size = trainer.configs.fntc_tile_size
    post = MaterialCNN(model.num_channels, inference_size=size, tile_size=tile_size).to(trainer.device)
    optimizer = torch.optim.Adam(post.parameters(), lr=.001)
    weights = torch.tensor(trainer.output_loss_weights, device=trainer.device)
    signature = _signature(raw, gt, trainer, weights)
    def positions(batch):
        pixels = torch.stack((torch.randint(infer_h, (batch,), device=trainer.device),
                              torch.randint(infer_w, (batch,), device=trainer.device)), 1)
        # Sample tiles proportional to their area, including partial edge tiles.
        return pixels // tile_size * tile_size

    tile_h, tile_w = min(tile_size, infer_h), min(tile_size, infer_w)
    patch_y = torch.arange(-1-post.halo, tile_h+1+post.halo, device=trainer.device)
    patch_x = torch.arange(-1-post.halo, tile_w+1+post.halo, device=trainer.device)
    ty = torch.arange(tile_h*sy, device=trainer.device)
    tx = torch.arange(tile_w*sx, device=trainer.device)
    yy, xx = torch.meshgrid((ty+.5)/sy+1, (tx+.5)/sx+1, indexing='ij')
    sampling_grid = torch.stack((2*xx/(tile_w+2)-1, 2*yy/(tile_h+2)-1), -1)[None]
    raw_hwc, gt_hwc = raw[0].permute(1, 2, 0), gt[0].permute(1, 2, 0)

    def loss_at(origins):
        losses = []
        # Bound validation memory to the training batch size. Partial tiles use
        # the same repeat halo and a mask that preserves each tile's mean loss.
        for batch in origins.split(4):
            y, x = batch.unbind(1)
            iy = (y[:, None]+patch_y) % infer_h
            ix = (x[:, None]+patch_x) % infer_w
            patches = raw_hwc[iy[:, :, None], ix[:, None, :]].permute(0, 3, 1, 2)
            filtered = post.valid(patches)
            prediction = F.grid_sample(filtered, sampling_grid.expand(len(batch), -1, -1, -1),
                                       mode='bilinear', align_corners=False)
            gy = (y[:, None]*sy+ty) % gt_size[0]
            gx = (x[:, None]*sx+tx) % gt_size[1]
            target = gt_hwc[gy[:, :, None], gx[:, None, :]].reshape(-1, model.num_channels)
            prediction = prediction.permute(0,2,3,1).reshape(-1,model.num_channels)
            loss = trainer._material_objective().per_sample(target, prediction, weights)
            if trainer.pbr_loss_enable and trainer.pbr_loss_weight > 0:
                loss = loss + trainer.pbr_loss_weight * trainer._material_objective().pbr(
                    target, prediction, reduction='none')
            mask = ((ty[None, :, None] < (infer_h-y)[:, None, None]*sy)
                    & (tx[None, None, :] < (infer_w-x)[:, None, None]*sx))
            loss = loss.reshape(len(batch), len(ty), len(tx))
            losses.append((loss*mask).sum((1, 2)) / mask.sum((1, 2)))
        return torch.cat(losses).mean()

    progress_path = os.path.join(trainer.model_path, 'material_cnn_training.pth')
    device = torch.device(trainer.device)
    cuda_device = None
    if device.type == 'cuda':
        cuda_device = device.index if device.index is not None else torch.cuda.current_device()
    interval = min(validation_interval, steps)
    if resume_state is not None:
        saved = resume_state
        if (saved['version'] != 2 or saved['signature'] != signature or saved['steps'] != steps
                or saved['validation_interval'] != interval):
            raise ValueError('Material CNN resume data or training recipe does not match checkpoint')
        post.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        validation = saved['validation'].to(trainer.device)
        best, best_state, best_step = saved['best'], saved['best_state'], saved['best_step']
        history, completed = saved['history'], saved['completed']
        torch.set_rng_state(saved['cpu_rng'].cpu())
        if cuda_device is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), cuda_device)
        print(f'[Tile CNN] resuming step {completed}/{steps}', flush=True)
    else:
        validation = positions(16)
        with torch.no_grad():
            best = float(loss_at(validation))
        if not math.isfinite(best):
            raise RuntimeError('Nonfinite initial material CNN validation loss')
        best_state, best_step, completed = copy.deepcopy(post.state_dict()), 0, 0
        history = [dict(step=0, validation_loss=best)]

    # The frozen source belongs to the same atomic checkpoint as the optimizer.
    # Its tensors stay unchanged throughout this stage.
    source_state = model.state_dict()

    def checkpoint():
        atomic_torch_save(dict(version=2, signature=signature, steps=steps, validation_interval=interval,
                         completed=completed, source_iteration=model.current_iter, progress=asdict(trainer.progress),
                         source_model=source_state, model=post.state_dict(),
                         optimizer=optimizer.state_dict(), validation=validation,
                         best=best, best_state=best_state, best_step=best_step, history=history,
                         cpu_rng=torch.get_rng_state(),
                         cuda_rng=None if cuda_device is None else torch.cuda.get_rng_state(cuda_device)), progress_path)

    checkpoint()
    post.train()
    for step in range(completed+1, steps+1):
        ratio = step/steps
        lr = .001 if ratio <= .5 else (.0003 if ratio <= .8 else .0001)
        optimizer.param_groups[0]['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = loss_at(positions(4))
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise RuntimeError('Nonfinite material CNN training loss')
        loss.backward()
        optimizer.step()
        completed = step
        trainer.writer.add_scalar('MaterialCNN/train_loss', loss_value, step)
        if step % interval == 0 or step == steps:
            with torch.no_grad():
                value = float(loss_at(validation))
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite material CNN validation loss')
            if value < best:
                best, best_step, best_state = value, step, copy.deepcopy(post.state_dict())
            history.append(dict(step=step, validation_loss=value, best_step=best_step, lr=lr))
            trainer.writer.add_scalar('MaterialCNN/validation_loss', value, step)
            checkpoint()
            print(f'[Tile CNN] {step}/{steps}: validation={value:.8f}, best={best_step}', flush=True)
    post.load_state_dict(best_state)
    if not all(torch.isfinite(p).all() for p in post.parameters()):
        raise RuntimeError('Nonfinite material CNN parameters')
    before = after = None
    accepted, reason = False, 'no patch-validation improvement over identity'
    if best_step > 0:
        with torch.random.fork_rng(devices=[] if cuda_device is None else [cuda_device]):
            before = _full_quality(trainer, raw, gt, None, weights)
            after = _full_quality(trainer, raw, gt, post, weights)
        accepted, reason = accept_quality(before, after)
    if accepted:
        model.material_cnn = post.eval()
        from Core.Feature_Grid.serialization import refresh_inference_contract
        refresh_inference_contract(model)
    print(f'[Tile CNN] accepted={accepted}: {reason}', flush=True)
    return dict(steps=steps, best_step=best_step, best_validation_loss=best, accepted=accepted,
                reason=reason, baseline=before, candidate=after, inference_size=list(size),
                gt_size=list(gt_size), tile_size=tile_size,
                parameters=sum(p.numel() for p in post.parameters()) if accepted else 0,
                seconds=time.perf_counter()-started, history=history,
                progress_checkpoint=progress_path)
