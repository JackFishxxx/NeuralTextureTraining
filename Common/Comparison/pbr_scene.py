"""Deterministic DirectX 12 PBR comparison renders and image reports."""

import os
import math
import json
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .material import UNCOMPRESSED, baseline_method, comparison_label_color, load_overlay_font, metric_records, write_metrics, log_comparison_storage


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _triangle_tangent_frame(xyz, uv):
    """Orthonormal UV frame, preserving handedness for mirrored UVs."""
    edge1, edge2 = xyz[1] - xyz[0], xyz[2] - xyz[0]
    duv1, duv2 = uv[1] - uv[0], uv[2] - uv[0]
    determinant = duv1[0] * duv2[1] - duv1[1] * duv2[0]
    if abs(determinant) < 1e-8:
        raise ValueError('PBR tangent frame requires nondegenerate UVs')
    normal = _normalize(np.cross(edge1, edge2))
    dpdu = (edge1 * duv2[1] - edge2 * duv1[1]) / determinant
    dpdv = (edge2 * duv1[0] - edge1 * duv2[0]) / determinant
    tangent = _normalize(dpdu - normal * np.dot(normal, dpdu))
    bitangent = _normalize(np.cross(normal, tangent))
    if np.dot(bitangent, dpdv) < 0:
        bitangent = -bitangent
    return normal, tangent, bitangent


def _quad(a, b, c, d, uv_scale=(1.0, 1.0)):
    vertices = np.asarray([a, b, c, d], np.float32)
    u, v = uv_scale
    uvs = np.asarray([[0, 0], [u, 0], [u, v], [0, v]], np.float32)
    return vertices, uvs, ((0, 2, 1), (0, 3, 2))


def _box(center, size, rotation_y=0.0):
    """Return flat-shaded cube faces as independent textured quads."""
    cx, cy, cz = center
    sx, sy, sz = np.asarray(size, np.float32) * 0.5
    faces = [
        ((-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz)),
        ((sx, -sy, sz), (-sx, -sy, sz), (-sx, sy, sz), (sx, sy, sz)),
        ((-sx, -sy, sz), (-sx, -sy, -sz), (-sx, sy, -sz), (-sx, sy, sz)),
        ((sx, -sy, -sz), (sx, -sy, sz), (sx, sy, sz), (sx, sy, -sz)),
        ((-sx, sy, -sz), (sx, sy, -sz), (sx, sy, sz), (-sx, sy, sz)),
    ]
    angle = np.deg2rad(rotation_y)
    rot = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]], np.float32)
    result = []
    for face in faces:
        p = np.asarray(face, np.float32) @ rot.T + np.asarray([cx, cy, cz], np.float32)
        result.append(_quad(*p, uv_scale=(1.5, 1.5)))
    return result


def _cornell_meshes() -> Iterable[Tuple[np.ndarray, np.ndarray, tuple]]:
    # Open-front Cornell box. Every face, including the room, receives exactly
    # the same inferred texture set; only UV scale differs with surface size.
    yield _quad((-1.6, -1, -0.5), (1.6, -1, -0.5), (1.6, -1, 2.7), (-1.6, -1, 2.7), (3, 3))
    yield _quad((-1.6, 1, 2.7), (1.6, 1, 2.7), (1.6, 1, -0.5), (-1.6, 1, -0.5), (3, 3))
    yield _quad((-1.6, -1, 2.7), (1.6, -1, 2.7), (1.6, 1, 2.7), (-1.6, 1, 2.7), (3, 2))
    yield _quad((-1.6, -1, -0.5), (-1.6, -1, 2.7), (-1.6, 1, 2.7), (-1.6, 1, -0.5), (3, 2))
    yield _quad((1.6, -1, 2.7), (1.6, -1, -0.5), (1.6, 1, -0.5), (1.6, 1, 2.7), (3, 2))
    yield from _box((-0.62, -0.48, 1.35), (0.82, 1.04, 0.82), -18)
    yield from _box((0.58, -0.67, 0.72), (0.72, 0.66, 0.72), 24)


def _cornell_spheres():
    # Each sphere rests on its box's top face, below the room ceiling (y=1).
    return (((-0.62, 0.04 + 0.44, 1.35), 0.44),
            ((0.58, -0.34 + 0.40, 0.72), 0.40))


@dataclass
class CompressedTexture:
    """Actual full-image mip levels decoded from the exported compressed blocks."""
    levels: tuple
    normal_encoding: str | None = None


def _compressed_material_maps(compression, textures, dataset, size):
    maps = {}
    for group,levels in compression.prepare_textures(textures,dataset,size):
        offset = 0
        for name in dataset.texture_packing[group]:
            start,end = dataset.canonical_channel_slices[name]
            encoding = compression.normal_encoding(dataset) if name == 'normal' else None
            channels = end-start
            maps[name] = CompressedTexture(tuple(
                level[...,offset:offset+channels].transpose(2,0,1).astype(np.float32)/255
                for level in levels),encoding)
            offset += channels
    return maps


def render_pbr_scene(texture: torch.Tensor, slices: Dict[str, tuple], normal_encoding: str,
                     resolution: int = 512, available_textures=None,
                     mip_lod_bias: float = -1.0, lighting=None,
                     view_center=(0.5, 0.5), view_zoom: float = 1.0, texture_maps=None,
                     *, geometry_cache=None) -> np.ndarray:
    """Rasterize float sRGB; an off-axis viewport re-renders a magnified scene region."""
    return _scene(geometry_cache).render(texture, slices, normal_encoding, resolution,
                                        available_textures, mip_lod_bias, lighting, texture_maps,
                                        view_center, view_zoom)


def _scene(cache=None):
    from .pbr_dx12 import DX12Scene
    meshes, spheres = list(_cornell_meshes()), _cornell_spheres()
    key = (tuple((v.tobytes(), uv.tobytes(), tuple(tris)) for v,uv,tris in meshes), spheres)
    cache = {} if cache is None else cache
    if cache.get('key') != key:
        cache.clear()
        cache.update(key=key, scene=DX12Scene(meshes, spheres))
    return cache['scene']


def _scene_view(height):
    """Fit fixed scene geometry directly into a rectangular camera viewport."""
    from .pbr_dx12 import CAMERA_EYE, CAMERA_TARGET, CAMERA_FOV
    eye = np.asarray(CAMERA_EYE)
    forward = _normalize(np.asarray(CAMERA_TARGET) - eye)
    right = _normalize(np.cross([0., 1., 0.], forward))
    up = np.cross(forward, right)
    vertices = np.concatenate([v for v, _, _ in _cornell_meshes()])
    relative = vertices - eye
    depth = relative @ forward
    scale = 1 / np.tan(np.deg2rad(CAMERA_FOV) * .5)
    uv = np.stack([.5 + .5 * (relative @ right) * scale / depth,
                   .5 - .5 * (relative @ up) * scale / depth], axis=1)
    lower, upper = uv.min(axis=0), uv.max(axis=0)
    span = upper - lower
    width = max(2, round(height * span[0] / span[1]))
    return width, height, tuple((lower + upper) * .5), 1 / span[1]


def _render_metrics(pred: np.ndarray, gt: np.ndarray, device: torch.device,
                    ssim_metric, lpips_metric) -> Tuple[float, float, float]:
    """Compute full-frame metrics on the final sRGB PBR render."""
    pred_t = torch.from_numpy(pred).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)
    gt_t = torch.from_numpy(gt).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)
    mse = torch.mean((pred_t - gt_t) ** 2)
    psnr = float((-10.0 * torch.log10(torch.clamp(mse, min=1e-12))).item())
    ssim_metric.reset()
    ssim_result = ssim_metric(pred_t, gt_t)
    ssim = float((ssim_result[0] if isinstance(ssim_result, tuple) else ssim_result).item())
    ssim_metric.reset()
    lpips_metric.reset()
    lpips = float(lpips_metric(pred_t, gt_t).item())
    lpips_metric.reset()
    return tuple(0.0 if not math.isfinite(v) else v for v in (psnr, ssim, lpips))


def _annotated_render(image, name, metric=None):
    panel = Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)).convert('RGBA')
    overlay = Image.new('RGBA', panel.size)
    draw = ImageDraw.Draw(overlay)
    label = (f'{name}\nPSNR {metric[0]:.3f}  SSIM {metric[1]:.4f}  LPIPS {metric[2]:.4f}'
             if metric is not None else 'GT\nReference: PSNR inf  SSIM 1.0000  LPIPS 0.0000')
    font_size = max(18, min(28, int(round(panel.height * 22.0 / 512.0))))
    font = load_overlay_font(font_size, ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"))
    # Keep annotations inside small diagnostic renders as well as 1K outputs.
    while draw.multiline_textbbox((0, 0), label, font=font, spacing=3)[2] > panel.width - 44 and getattr(font, 'size', 10) > 10:
        font = ImageFont.truetype(font.path, font.size - 1)
    bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=3)
    draw.rounded_rectangle((10, 10, 34 + bbox[2], 30 + bbox[3]), radius=6,
                           fill=comparison_label_color(name))
    draw.multiline_text((23, 21), label, fill=(0, 0, 0, 220), font=font, spacing=3)
    draw.multiline_text((22, 20), label, fill=(255, 255, 255), font=font, spacing=3)
    return Image.alpha_composite(panel, overlay).convert('RGB')


def _save_render_panels(images, metrics, directory, path, prefix):
    panels = [_annotated_render(image, name, metrics.get(name)) for name,image in images.items()]
    canvas = Image.new('RGB', (sum(p.width for p in panels), panels[0].height))
    x = 0
    for (name,image),panel in zip(images.items(),panels):
        panel.save(os.path.join(directory, prefix+name+'.png'), compress_level=1)
        canvas.paste(panel,(x,0))
        x += panel.width
    canvas.save(path, compress_level=1)


@torch.no_grad()
def run_pbr_comparison_pipeline(trainer, evaluation, output_root, *, curr_iter=None,
                                report_context=None, save_outputs=True, include_detail_metrics=False):
    """One comparison per view: uncompressed FNTC, recompressed FNTC, baseline, GT."""
    config, dataset = trainer.configs, trainer.dataset
    compression = trainer.material_compression
    primary = trainer.primary_method
    reference = evaluation.reference
    if reference is None:
        raise ValueError('PBR comparisons require baseline and GT materials')
    cache = reference['pbr']
    key = (config.pbr_render_resolution, config.pbr_detail_resolution, config.pbr_mip_lod_bias,
           json.dumps(trainer.pbr_lighting,sort_keys=True),tuple(config.pbr_detail_center),config.pbr_detail_zoom,
           dataset.normal_encoding)
    if cache.get('key') != key:
        geometry = cache.get('geometry', {})
        cache.clear()
        cache.update(key=key, geometry=geometry)
    out_dir = os.path.join(output_root,'Compare_PBRScene')
    if curr_iter is not None:
        out_dir = os.path.join(out_dir,f'iter_{curr_iter:08d}')
    path = os.path.join(out_dir,'comparison_strip.png')
    if save_outputs:
        os.makedirs(out_dir,exist_ok=True)
    scores = {}
    sizes = {}
    on_sample = None
    if evaluation.sample_model is not None:
        from .on_sample import prepare_on_sample
        on_sample = prepare_on_sample(evaluation.sample_model)
    # Detail uses a fresh camera viewport, never a resized crop of the main view.
    views = [('Scene', _scene_view(config.pbr_render_resolution))]
    if save_outputs or include_detail_metrics:
        views.append(('Scene Detail', (config.pbr_detail_resolution, config.pbr_detail_resolution,
                                       config.pbr_detail_center, config.pbr_detail_zoom)))
    baseline_name = baseline_method(compression,dataset)
    fixed_names = (baseline_name,'GT')
    for name,textures in (evaluation.textures or {}).items():
        if name not in evaluation.compressed_maps:
            if name == baseline_name:
                if 'compressed_maps' not in reference:
                    reference['compressed_maps'] = _compressed_material_maps(compression,textures,dataset,evaluation.size)
                evaluation.compressed_maps[name] = reference['compressed_maps']
            else:
                evaluation.compressed_maps[name] = _compressed_material_maps(compression,textures,dataset,evaluation.size)
    for name in evaluation.native_images:
        if name not in (UNCOMPRESSED,'GT') and name not in evaluation.compressed_maps:
            raise ValueError(f'PBR requires compressed full-image mip chains for {name}')
    scene = _scene(cache.setdefault('geometry', {}))
    # Method first: prepare and upload once, then draw all missing viewports.
    rendered_views = {view: {} for view, _ in views}
    for name, image in evaluation.native_images.items():
        pending = []
        for view, viewport in views:
            fixed = cache.setdefault(view, {})
            if name in fixed_names and name in fixed:
                rendered_views[view][name] = fixed[name]
            else:
                pending.append((view, viewport))
        if pending:
            images = scene.render_views(image, dataset.canonical_channel_slices, dataset.normal_encoding,
                                        dataset.available_textures, config.pbr_mip_lod_bias, trainer.pbr_lighting,
                                        evaluation.compressed_maps.get(name), [viewport for _, viewport in pending],
                                        on_sample=on_sample if name == UNCOMPRESSED else None)
            for (view, _), rendered in zip(pending, images):
                rendered_views[view][name] = rendered
                if name in fixed_names:
                    cache[view][name] = rendered
    for view, _ in views:
        fixed, rendered = cache[view], rendered_views[view]
        gt = rendered['GT']
        sizes[view] = list(gt.shape[:2])
        fixed_metrics = fixed.setdefault('metrics',{})
        scores[view] = {}
        for name,image in rendered.items():
            if name == 'GT':
                continue
            if name == baseline_name and name in fixed_metrics:
                score = fixed_metrics[name]
            else:
                score = _render_metrics(image,gt,config.device,trainer.ssim,trainer.lpips)
                if name == baseline_name:
                    fixed_metrics[name] = score
            scores[view][name] = score
        if save_outputs:
            filename = 'comparison_strip.png' if view == 'Scene' else 'comparison_strip_detail.png'
            _save_render_panels(rendered,scores[view],out_dir,os.path.join(out_dir,filename),
                                'Scene_' if view == 'Scene' else 'Detail_')
            for name,metric in scores[view].items():
                print(f'[PBR Test] {view} Iter {curr_iter}: {name} '
                      f'PSNR={metric[0]:.4f} SSIM={metric[1]:.4f} LPIPS={metric[2]:.4f}')
    if save_outputs:
        storage = log_comparison_storage(trainer.model,dataset,compression,config,verbose=False)
        records = [row for view,methods in scores.items() for name,values in methods.items()
                   for row in metric_records({'image':values},name,view=view,storage_bytes=storage[name])]
        write_metrics(out_dir,records,evaluation='pbr',iteration=curr_iter,
                      metadata=dict(report_context or {},primary_method=primary,render_sizes=sizes,
                                    renderer_backend='directx12',
                                    renderer_adapter=scene.adapter,
                                    color_space='linear material inputs and HDR lighting; Reinhard tone map; sRGB output',
                                    shadows='depth maps; 3x3 PCF; directional and point lights',
                                    shadow_resolution=scene.shadow_size,
                                    compression_formats=compression.recipe(dataset),storage_bytes=storage,
                                    sr_base_codec=trainer.model.sr_base_codec if trainer.super_resolution_enable else None,
                                    sr_base_resolution=config.super_resolution_base_resolution if trainer.super_resolution_enable else None,
                                    fntc_tile_size=config.fntc_tile_size,
                                    gt_size=[dataset.texture_height,dataset.texture_width],
                                    inference_size=list(evaluation.size),
                                    material_sizes={name:list(image.shape[-2:]) for name,image in evaluation.native_images.items()},
                                    normal_encoding=dataset.normal_encoding,
                                    mip_sources={name: ('filtered deployment latent mips then pixel decoder' if on_sample is not None and name == UNCOMPRESSED
                                                       else 'compressed full-image mip chains' if name in evaluation.compressed_maps
                                                       else 'floating mip pyramid') for name in evaluation.native_images},
                                    texture_addressing=('FNTC: bilinear/trilinear latent filtering then pixel decoder; baseline: material mip sampling'
                                                        if on_sample is not None else
                                                        'repeat full material UVs; bilinear/trilinear sampling of assembled compressed mips'),
                                    inference_mode='on_sample' if on_sample is not None else 'on_load',
                                    lod_reference_size=(list(evaluation.sample_model.material_lod_reference_size)
                                                        if on_sample is not None else None),
                                    mip_lod_bias=config.pbr_mip_lod_bias,lighting=trainer.pbr_lighting,
                                    detail_view=dict(center=config.pbr_detail_center,zoom=config.pbr_detail_zoom,
                                                     resolution=config.pbr_detail_resolution,rendered_separately=True)))
    return path if save_outputs else None,scores
