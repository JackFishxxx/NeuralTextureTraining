"""
ASTC comparison pipeline: FNTC (quantized / grid-ASTC) vs traditional RGBA ASTC round-trip.

Uses official astcenc for encode+decode baselines. Visualization and metrics align on Mip0
full texture resolution; ref_astc_resolution sets the shared FNTC/ASTC material raster.
"""
from __future__ import annotations

import copy
import csv
import math
import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
from Core.Feature_Grid.compression import apply_latent_compression
from Core.Feature_Grid.sampling import resize_bilinear_repeat
from Core.Feature_Grid.compression import compressed_method, latent_payload_bytes, grid_recipes
from Core.Material_Encoding.normal_encoding import decode_normal, normal_angular_psnr
from dataset import prefiltered_material
from Common.material_reconstruction import (comparison_material_size, _render_gt_mip0, _render_model_mip0, _ref_astc_hw_from_side)


# ---------------------------------------------------------------------------
# Version / defaults
# ---------------------------------------------------------------------------

from Common.Block_Compression.astc import (
    ASTCCodec, ASTCENC_DEFAULT_VERSION, ASTCENC_DEFAULT_PATH, astc_roundtrip_rgba,
    resolve_astcenc_executable,
)
from Common.Block_Compression.astc import normalize_astc_block


def log_comparison_storage(model, dataset, block: str, ref_astc_resolution=None, verbose=True,
                           deployment_config=None):
    """Log Mip0 deployment payloads, excluding SR base and file containers."""
    block = normalize_astc_block(block)
    block_w, block_h = map(int, block.split('x'))

    def astc_bytes(height, width):
        return math.ceil(height / block_h) * math.ceil(width / block_w) * 16

    uncompressed_latent = sum(spec.max_resolution**2*4*math.ceil(spec.n_features_per_level/4)
                              for spec in model.feature_grid_specs)
    latent = latent_payload_bytes(model)
    decoder, cnn = model.decoder_storage_bytes(deployment_config)
    height, width = dataset.texture_height, dataset.texture_width
    ref_h, ref_w = comparison_material_size(model, height, width, ref_astc_resolution)
    maps = len(dataset.texture_packing)
    channels = sum(dataset.canonical_channel_slices[name][1] - dataset.canonical_channel_slices[name][0]
                   for name in dataset.available_textures)
    rows = (
        ('FNTC', latent + decoder + cnn,
         f'latent {getattr(model, "latent_format", "astc").upper()}={latent:,} B + decoder={decoder:,} B + material CNN={cnn:,} B; SR base excluded'),
        ('ASTC', maps * astc_bytes(ref_h, ref_w),
         f'{ref_w}x{ref_h}, {maps} packed maps, {block} blocks'),
        ('GT', height * width * channels,
         f'{width}x{height}, {channels} effective channels, UNORM8'),
    )
    if verbose:
        byte_width = max(len(f'{size:,}') for _, size, _ in rows)
        for name, size, details in rows:
            print(f'[Storage] {name:<4} total={size:>{byte_width},} B ({size / 1024:>10.2f} KiB) '
                  f'| {details}; Mip0 payload only')
    return {
        'fntc_quantized': uncompressed_latent + decoder + cnn,
        compressed_method(model): latent + decoder + cnn,
        f'ref_astc_{block}': maps * astc_bytes(ref_h, ref_w),
        'gt': height * width * channels,
    }


# ---------------------------------------------------------------------------
# Codec + subprocess round-trip
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tensor ↔ uint8 RGBA (traditional baseline)
# ---------------------------------------------------------------------------

def channels_to_rgba_u8(image_chw: torch.Tensor) -> np.ndarray:
    image_np = image_chw.detach().cpu().numpy()
    image_np = np.clip(np.round(image_np * 255.0), 0, 255).astype(np.uint8)
    c, h, w = image_np.shape
    if not 1 <= c <= 4:
        raise ValueError(f'An RGBA texture requires 1–4 channels, got {c}')
    # Use opaque alpha for partially populated RGBA groups. This avoids making
    # the ASTC endpoint fit depend on an artificial transparent channel.
    rgba = np.full((h, w, 4), 255, dtype=np.uint8)
    for idx in range(c):
        rgba[:, :, idx] = image_np[idx, :, :]
    return rgba


@torch.no_grad()
def build_traditional_astc_prediction(
    gt_image: torch.Tensor,
    canonical_slices: Dict[str, Tuple[int, int]],
    texture_packing: Dict[str, list],
    roundtrip_rgba: Callable[[np.ndarray], np.ndarray],
) -> torch.Tensor:
    """Execute dataset packing, one codec round-trip per declared texture."""
    astc_pred = torch.zeros_like(gt_image)
    for names in texture_packing.values():
        channels = [index for name in names for index in range(*canonical_slices[name])]
        tex = gt_image[0, channels]
        rgba_rt = roundtrip_rgba(channels_to_rgba_u8(tex))
        restored_np = np.ascontiguousarray(rgba_rt[:, :, :len(channels)]).copy()
        restored = torch.from_numpy(restored_np).permute(2, 0, 1).to(gt_image.device).float() / 255.0
        astc_pred[0, channels] = restored

    return torch.clamp(astc_pred, 0.0, 1.0)


@torch.no_grad()
def _traditional_baseline_resampled(
    gt_mip0: torch.Tensor,
    dataset,
    roundtrip_rgba: Callable[[np.ndarray], np.ndarray],
    ref_h: int,
    ref_w: int,
) -> torch.Tensor:
    """Run traditional ASTC at (ref_h, ref_w), then resize to gt_mip0 spatial size for comparison."""
    h0, w0 = gt_mip0.shape[-2], gt_mip0.shape[-1]
    src = prefiltered_material(dataset, (ref_h, ref_w))
    out = build_traditional_astc_prediction(
        gt_image=src,
        canonical_slices=dataset.canonical_channel_slices,
        texture_packing=dataset.texture_packing,
        roundtrip_rgba=roundtrip_rgba,
    )
    if (ref_h, ref_w) != (h0, w0):
        out = resize_bilinear_repeat(out, (h0, w0))
    return out


@torch.no_grad()
def _astc_roundtrip_superres_base_mip0(dataset, roundtrip_rgba, base_resolution: int, output_size=None) -> torch.Tensor:
    """Compress the reused low-resolution source mip before upsampling it."""
    h0, w0 = dataset.texture_height, dataset.texture_width
    ratio = max(h0, w0) / float(max(1, base_resolution))
    source_mip_offset = max(0, int(round(math.log2(ratio)))) if ratio > 1.0 else 0
    source_mip = min(source_mip_offset, dataset.num_mips - 1)
    h, w = h0 // (2 ** source_mip), w0 // (2 ** source_mip)
    source = dataset.mip_cache[source_mip]
    source = dataset.expand_to_canonical(source.reshape(-1, source.shape[-1]))
    source = source.reshape(h, w, -1).permute(2, 0, 1)[None, ...]
    compressed = build_traditional_astc_prediction(
        source,
        dataset.canonical_channel_slices,
        dataset.texture_packing,
        roundtrip_rgba,
    )
    output_size = getattr(dataset, 'material_reconstruction_size', (h0, w0)) if output_size is None else tuple(output_size)
    if compressed.shape[-2:] != output_size:
        compressed = resize_bilinear_repeat(compressed, output_size)
    return compressed.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Mip0 rendering
# ---------------------------------------------------------------------------

def _reference_cache(dataset, codec):
    """One cache per dataset/codec recipe; retain fixed images on CPU."""
    source = tuple((id(level), level._version, tuple(level.shape)) for level in dataset.mip_cache)
    key = (source, dataset.normal_encoding,
           tuple((name, tuple(bounds)) for name, bounds in dataset.canonical_channel_slices.items()),
           tuple((name, tuple(channels)) for name, channels in dataset.texture_packing.items()),
           getattr(codec, 'astcenc_path', None), getattr(codec, 'astcenc_quality', None), codec.astc_block)
    previous = getattr(dataset, '_comparison_reference_cache', None)
    if previous is None or previous[0] != key:
        dataset._comparison_reference_cache = (key, {})
    return dataset._comparison_reference_cache[1]


@torch.no_grad()
def comparison_reference(dataset, codec, ref_h, ref_w):
    cache = _reference_cache(dataset, codec)
    size = (ref_h, ref_w)
    if cache.get('material_size') != size:
        gt = _render_gt_mip0(dataset, dataset.texture_height, dataset.texture_width)
        astc = _traditional_baseline_resampled(gt, dataset, codec.roundtrip_rgba, ref_h, ref_w)
        cache['material_size'] = size
        cache['material'] = {'gt': gt.cpu(), 'astc': astc.cpu(), 'metrics': {}, 'pbr': {}}
    return cache['material']


@torch.no_grad()
def cached_astc_base(dataset, codec, base_resolution, output_size):
    cache = _reference_cache(dataset, codec)
    key = (base_resolution, tuple(output_size))
    if cache.get('base_size') != key:
        image = _astc_roundtrip_superres_base_mip0(dataset, codec.roundtrip_rgba, base_resolution, output_size)
        cache['base_size'], cache['base'] = key, image.cpu()
    return cache['base'].to(dataset.mip_cache[0].device)


# ---------------------------------------------------------------------------
# Metrics (per texture family + average)
# ---------------------------------------------------------------------------

METRIC_FIELDS = ('iteration', 'method', 'view', 'mip', 'group', 'psnr_type',
          'psnr', 'ssim', 'lpips', 'lpips_status', 'storage_bytes')


def method_label(method, block=None):
    if method == 'fntc_mixed':
        return 'FNTC_mixed'
    if method == 'fntc_bc1':
        return 'FNTC_BC1'
    if method == 'fntc_quantized':
        return 'FNTC_uncompressed'
    if method == 'gt':
        return 'GT'
    if method.startswith('fntc_astc_'):
        return f'FNTC_{method.removeprefix("fntc_astc_")}_compressed'
    if method.startswith('ref_astc_'):
        return f'ASTC_{method.removeprefix("ref_astc_")}_compressed'
    if method in ('fntc', 'astc') and block is not None:
        return f'{method.upper()}_{block}'
    return method


def metric_records(metrics, method, *, view='Material', mip=0, storage_bytes=None,
                   lpips_available=True):
    records = []
    for group, values in metrics.items():
        psnr, ssim, lpips = (float(v) if v is not None and math.isfinite(v) else None for v in values)
        records.append(dict(method=method, view=view, mip=mip, group=group,
                            psnr_type=('normal_angular' if group == 'normal' else
                                       'weighted_group_mean' if group == 'average' else 'image_mse'),
                            psnr=psnr, ssim=ssim, lpips=lpips if lpips_available else None,
                            lpips_status=('not_computed' if not lpips_available else
                                          'computed' if lpips is not None else 'nonfinite'),
                            storage_bytes=storage_bytes))
    return records


def write_metrics(directory, records, *, evaluation, iteration=None, metadata=None):
    """JSON retains full precision/context; CSV has the same rows and named columns."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = [dict(row, iteration=iteration) for row in records]
    report = dict(schema_version=1, evaluation=evaluation, iteration=iteration,
                  storage_scope=dict(unit='bytes', mip=0, sr_base_included=False, container_overhead_included=False),
                  metric_definitions={
                      'psnr': 'dB; psnr_type distinguishes image MSE, normal angular error and group mean; exact image matches report 120 dB',
                      'normal': 'PSNR: angular error in degrees (range 180); SSIM/LPIPS: decoded normal RGB',
                      'average': 'weighted mean of group scores, using the sum of effective channel weights',
                      'romsd_lpips': 'mean of per-scalar LPIPS, each scalar replicated to RGB',
                      'lpips_status': 'not_computed is null in JSON and empty in CSV; never a perfect score',
                      'storage': 'bytes; Mip0 deployment payload; SR base and file containers excluded',
                  }, metadata=metadata or {}, records=records)
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    (directory/'metrics.json').write_text(text, encoding='utf-8')
    with (directory/'metrics.csv').open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow({key: f'{value:.6f}' if isinstance(value, float) else value
                             for key, value in row.items()})
    return report


def _compute_ssim_safe(pred_ref: torch.Tensor, gt_ref: torch.Tensor, ssim_metric) -> float:
    ssim_metric.reset()
    pred_ref = torch.nan_to_num(pred_ref.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    gt_ref = torch.nan_to_num(gt_ref.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ssim_result = ssim_metric(pred_ref, gt_ref)
    ssim_t = ssim_result[0] if isinstance(ssim_result, tuple) else ssim_result
    v = float(ssim_t.item())
    ssim_metric.reset()
    if math.isfinite(v):
        return v
    if pred_ref.shape[1] > 1:
        pg = pred_ref.mean(dim=1, keepdim=True)
        gg = gt_ref.mean(dim=1, keepdim=True)
        ssim_result2 = ssim_metric(pg, gg)
        ssim_t2 = ssim_result2[0] if isinstance(ssim_result2, tuple) else ssim_result2
        v2 = float(ssim_t2.item())
        ssim_metric.reset()
        if math.isfinite(v2):
            return v2
    return 0.0


def _compute_metrics_from_refs(
    pred_ref: torch.Tensor, gt_ref: torch.Tensor, psnr_metric, ssim_metric, lpips_metric,
    is_normal: bool = False, normal_encoding: str = "xyz", scalar_channels: bool = False,
) -> Tuple[float, float, float]:
    pred_ref = torch.nan_to_num(pred_ref.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    gt_ref = torch.nan_to_num(gt_ref.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if is_normal:
        psnr_value = float(normal_angular_psnr(pred_ref, gt_ref, normal_encoding).item())
        pred_ref = decode_normal(pred_ref, normal_encoding)
        gt_ref = decode_normal(gt_ref, normal_encoding)
    else:
        psnr_metric.reset()
        psnr_value = float(psnr_metric(pred_ref, gt_ref).item())
        psnr_metric.reset()
        if psnr_value == math.inf:
            psnr_value = 120.0
    if not math.isfinite(psnr_value):
        psnr_value = 0.0
    ssim_value = _compute_ssim_safe(pred_ref, gt_ref, ssim_metric)

    if pred_ref.shape[-2] >= 128 and pred_ref.shape[-1] >= 128:
        c = pred_ref.shape[1]
        if c == 3 and not scalar_channels:
            lpips_pred, lpips_gt = pred_ref, gt_ref
        else:
            # Scalar maps have no RGB color relationship, regardless of count.
            values = []
            for channel_idx in range(c):
                p = pred_ref[:, channel_idx:channel_idx + 1].repeat(1, 3, 1, 1)
                g = gt_ref[:, channel_idx:channel_idx + 1].repeat(1, 3, 1, 1)
                lpips_metric.reset()
                values.append(float(lpips_metric(p.float(), g.float()).item()))
                lpips_metric.reset()
            lpips_value = float(np.mean(values)) if values else 0.0
            if not math.isfinite(lpips_value):
                lpips_value = 0.0
            return psnr_value, ssim_value, lpips_value
        lpips_metric.reset()
        lpips_value = float(lpips_metric(lpips_pred.float(), lpips_gt.float()).item())
        lpips_metric.reset()
        if not math.isfinite(lpips_value):
            lpips_value = 0.0
    else:
        lpips_value = 0.0
    return psnr_value, ssim_value, lpips_value


def _metric_channel_groups(dataset) -> Dict[str, list]:
    canon = dataset.canonical_channel_slices
    return {group: [index for name in names for index in range(*canon[name])]
            for group, names in dataset.texture_packing.items()}


def _group_metric_weights(dataset, eval_weights=None) -> Dict[str, float]:
    group_channels = _metric_channel_groups(dataset)
    if eval_weights is None:
        return {name: 1.0 for name in group_channels}

    w = torch.as_tensor(eval_weights, dtype=torch.float32)
    out: Dict[str, float] = {}
    for group_name, channels in group_channels.items():
        if not channels:
            continue
        idx = torch.tensor(channels, dtype=torch.long)
        out[group_name] = float(w.index_select(0, idx).sum().item())
    return out


@torch.no_grad()
def _compute_group_metrics(pred_image, gt_image, dataset, psnr_metric, ssim_metric, lpips_metric, eval_weights=None):
    group_channels = _metric_channel_groups(dataset)
    group_metrics: Dict[str, Tuple[float, float, float]] = {}
    for group_name, channels in group_channels.items():
        ch_idx = torch.tensor(channels, device=pred_image.device, dtype=torch.long)
        pred_ref = torch.index_select(pred_image, dim=1, index=ch_idx)
        gt_ref = torch.index_select(gt_image, dim=1, index=ch_idx)
        group_metrics[group_name] = _compute_metrics_from_refs(
            pred_ref, gt_ref, psnr_metric, ssim_metric, lpips_metric,
            is_normal=(group_name == "normal"),
            normal_encoding=str(getattr(dataset, "normal_encoding", "xyz")),
            scalar_channels=(group_name == "romsd"),
        )

    if group_metrics:
        order = [n for n in ("diffuse", "normal", "romsd") if n in group_metrics]
        weights = _group_metric_weights(dataset, eval_weights)
        vals = np.array([group_metrics[n] for n in order], dtype=np.float64)
        ws = np.array([max(0.0, weights.get(n, 1.0)) for n in order], dtype=np.float64)
        if ws.sum() <= 0:
            ws = np.ones_like(ws)
        ws = ws / ws.sum()
        avg_psnr, avg_ssim, avg_lpips = [float(v) for v in (vals * ws[:, None]).sum(axis=0)]
        if not math.isfinite(avg_ssim):
            avg_ssim = 0.0
        if not math.isfinite(avg_psnr):
            avg_psnr = 0.0
        if not math.isfinite(avg_lpips):
            avg_lpips = 0.0
        group_metrics["average"] = (avg_psnr, avg_ssim, avg_lpips)
    return group_metrics


# ---------------------------------------------------------------------------
# Comparison strip + annotated tiles
# ---------------------------------------------------------------------------

def _metric_labels(metric_group: Optional[str] = None) -> Tuple[str, str, str]:
    if metric_group == "normal":
        return "AngularPSNR", "SSIM", "LPIPS"
    return "PSNR", "SSIM", "LPIPS"


def _format_metric_triplet(
    metric: Optional[Tuple[float, float, float]],
    metric_group: Optional[str] = None,
) -> str:
    psnr_label, ssim_label, lpips_label = _metric_labels(metric_group)
    if metric is None:
        return f"{psnr_label}:{'-':>8}  {ssim_label}:{'-':>7}  {lpips_label}:{'-':>7}"
    lpips_text = f'{metric[2]:7.4f}' if metric[2] is not None else f'{"N/A":>7}'
    return f"{psnr_label}:{metric[0]:8.3f}  {ssim_label}:{metric[1]:7.4f}  {lpips_label}:{lpips_text}"


def _print_comparison_metrics(metrics, display_labels, curr_iter=None):
    label_width = max(len(label) for label in display_labels.values())
    for method, values in metrics.items():
        print(f"[ASTC Test] {display_labels[method]:<{label_width}} Iter {curr_iter}: Weighted {_format_metric_triplet(values.get('average'))} "
              f"(Diffuse {_format_metric_triplet(values.get('diffuse'))}, "
              f"NormalAngular {_format_metric_triplet(values.get('normal'))}, "
              f"ROMSD {_format_metric_triplet(values.get('romsd'))})")


def _resolve_metric_group_for_texture(texture_name: str) -> str:
    name = str(texture_name).lower()
    if any(k in name for k in ("diffuse", "albedo", "basecolor", "base_color", "color", "rgb")):
        return "diffuse"
    if "normal" in name:
        return "normal"
    if "specular" in name:
        return "romsd"
    if any(k in name for k in ("rough", "occlusion", "ao", "metal", "displace", "height")):
        return "romsd"
    return "average"


def _load_overlay_font(font_size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _concat_pil_h(images: list) -> Image.Image:
    if not images:
        raise ValueError("images must not be empty.")
    total_w = sum(img.size[0] for img in images)
    max_h = max(img.size[1] for img in images)
    out = Image.new("RGB", (total_w, max_h), color=(0, 0, 0))
    x = 0
    for img in images:
        out.paste(img, (x, 0))
        x += img.size[0]
    return out


def comparison_label_color(method_name: str):
    """Shared method colors for material and PBR comparison annotations."""
    name = method_name.lower()
    if name in ('fntc_quantized', 'fntc_uncompressed'):
        return (46, 196, 255, 140)
    if name.startswith('fntc_'):
        return (255, 181, 71, 140)
    if name.startswith(('ref_astc_', 'astc_')):
        return (255, 99, 146, 140)
    if name == 'gt':
        return (127, 255, 127, 140)
    return (120, 120, 255, 140)


def _make_annotated_tile(
    tile: torch.Tensor,
    method_name: str,
    metric: Optional[Dict[str, Tuple[float, float, float]]],
    metric_group: Optional[str] = None,
) -> Image.Image:
    tile_pil = TF.to_pil_image(tile).convert("RGBA")
    w, h = tile_pil.size

    if method_name == "fntc_quantized":
        display_name = "FNTC_uncompressed"
    elif method_name == "fntc_bc1":
        display_name = "FNTC_BC1"
    elif method_name.startswith("fntc_astc_"):
        display_name = "FNTC_" + method_name.removeprefix("fntc_astc_")
    elif method_name.startswith("ref_astc_"):
        display_name = "ASTC_" + method_name.removeprefix("ref_astc_")
    else:
        display_name = method_name.upper()

    if metric is None:
        text = f"{display_name}\nReference" if method_name == "gt" else f"{display_name}\n{_format_metric_triplet(None, metric_group)}"
    else:
        selected_group = metric_group if metric_group in metric else "average"
        selected_metric = metric.get(selected_group)
        group_name = "NORMAL ANGULAR" if selected_group == "normal" else selected_group.upper()
        text = f"{display_name} [{group_name}]\n{_format_metric_triplet(selected_metric, selected_group)}"

    font_size = max(18, min(36, h // 40, w // 28))
    font = _load_overlay_font(font_size)
    pad = max(8, w // 128)
    spacing = max(2, font_size // 8)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    box_w = text_bbox[2] - text_bbox[0] + pad * 2
    box_h = text_bbox[3] - text_bbox[1] + pad * 2
    box_x, box_y = pad, pad

    draw.rounded_rectangle(
        (box_x, box_y, min(w - pad, box_x + box_w), min(h - pad, box_y + box_h)),
        radius=max(4, font_size // 5),
        fill=comparison_label_color(method_name),
    )
    text_x, text_y = box_x + pad, box_y + pad
    draw.multiline_text((text_x + 1, text_y + 1), text, fill=(0, 0, 0, 220), font=font, spacing=spacing)
    draw.multiline_text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font, spacing=spacing)
    return Image.alpha_composite(tile_pil, overlay).convert("RGB")


@torch.no_grad()
def _save_comparison_strip(
    pred_fntc: torch.Tensor,
    pred_fntc_grid_astc: torch.Tensor,
    pred_traditional_astc: torch.Tensor,
    gt_image: torch.Tensor,
    compare_root: str,
    metrics: Dict[str, Dict[str, Tuple[float, float, float]]],
    fntc_astc_name: str,
    ref_astc_name: str,
    vis_configs,
    postprocess_for_vis: Callable[[torch.Tensor, str], torch.Tensor],
) -> None:
    os.makedirs(compare_root, exist_ok=True)

    fntc_chw = pred_fntc.squeeze(0)
    fntc_astc_chw = pred_fntc_grid_astc.squeeze(0)
    traditional_chw = pred_traditional_astc.squeeze(0)
    gt_chw = gt_image.squeeze(0)

    for vc in vis_configs:
        s, e = vc["canonical_channel_slice"]
        img_fntc = postprocess_for_vis(fntc_chw[s:e, ...], vc["vis_mode"])
        img_fntc_astc = postprocess_for_vis(fntc_astc_chw[s:e, ...], vc["vis_mode"])
        img_traditional = postprocess_for_vis(traditional_chw[s:e, ...], vc["vis_mode"])
        img_gt = postprocess_for_vis(gt_chw[s:e, ...], vc["vis_mode"])
        metric_group = _resolve_metric_group_for_texture(vc["display_name"])
        tiles_meta = [
            (img_fntc, "fntc_quantized", metrics.get("fntc_quantized"), metric_group),
            (img_fntc_astc, fntc_astc_name, metrics.get(fntc_astc_name), metric_group),
            (img_traditional, ref_astc_name, metrics.get(ref_astc_name), metric_group),
            (img_gt, "gt", None, metric_group),
        ]
        annotated = [_make_annotated_tile(*m) for m in tiles_meta]
        _concat_pil_h(annotated[1:]).save(os.path.join(compare_root, f"comparison_strip_{vc['display_name']}.png"))
        texture_type = vc.get('texture_type', vc['display_name'])
        texture_prefix = 'Diffuse' if texture_type in ('rgb', 'diffuse') else texture_type.capitalize()
        names = ("Uncompressed_FNTC", ("FNTC_mixed" if fntc_astc_name == "fntc_mixed" else "FNTC_BC1" if fntc_astc_name == "fntc_bc1" else f"FNTC_{fntc_astc_name.removeprefix('fntc_astc_')}"),
                 f"ASTC_{ref_astc_name.removeprefix('ref_astc_')}", "GT")
        for name, tile in zip(names, annotated):
            tile.save(os.path.join(compare_root, f"{texture_prefix}_{name}.png"))


# ---------------------------------------------------------------------------
# astcenc discovery / Windows auto-install
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public entry: full pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_astc_comparison_pipeline(
    model,
    dataset,
    astc_codec: ASTCCodec,
    psnr_metric,
    ssim_metric,
    lpips_metric,
    vis_configs,
    postprocess_for_vis: Callable[[torch.Tensor, str], torch.Tensor],
    output_root: str,
    texture_height: int,
    texture_width: int,
    num_mips: int,
    device: str,
    curr_iter: Optional[int] = None,
    ref_astc_resolution: Optional[int] = None,
    eval_weights=None,
    inference_tile: int = 512,
    verbose: bool = True,
    report_context=None,
    save_outputs=True,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    block_tag = astc_codec.astc_block.lower()
    storage = log_comparison_storage(model, dataset, astc_codec.astc_block, ref_astc_resolution, verbose=False)
    fntc_astc_name = compressed_method(model)
    ref_astc_name = f"ref_astc_{astc_codec.astc_block}"

    compare_root = os.path.join(output_root, f"Compare_ASTC{block_tag}")
    if curr_iter is not None:
        compare_root = os.path.join(compare_root, f"iter_{curr_iter:08d}")
    if save_outputs:
        os.makedirs(compare_root, exist_ok=True)

    h0, w0 = texture_height, texture_width
    feature_resolutions = []
    for spec in getattr(model, "feature_grid_specs", []):
        try:
            feature_resolutions.append(int(spec.highest_level_slice()[2]))
        except (AttributeError, TypeError, ValueError):
            pass
    ref_h, ref_w = comparison_material_size(model, h0, w0, ref_astc_resolution)
    reference = comparison_reference(dataset, astc_codec, ref_h, ref_w)
    gt_image = reference['gt'].to(device)
    native_h, native_w = ref_h, ref_w
    if save_outputs:
        with open(os.path.join(compare_root, 'reconstruction.json'), 'w', encoding='utf-8') as output:
            json.dump({'gt_size': [h0, w0], 'fntc_material_size': [native_h, native_w],
                   'astc_material_size': [ref_h, ref_w], 'feature_grid_resolutions': feature_resolutions,
                   'upsampler': 'bilinear_repeat' if (ref_h, ref_w) != (h0, w0) else None,
                   'texture_packing': dataset.texture_packing,
                   'texel_centers': True, 'decoder_queries': native_h * native_w}, output, indent=2)

    quant_model = copy.deepcopy(model)
    quant_model.simulate_quantize()
    quant_model.eval()
    astc_base = None
    if getattr(quant_model, "super_resolution_enable", False):
        astc_base = cached_astc_base(
            dataset, astc_codec,
            int(getattr(dataset, "super_resolution_base_resolution", dataset.superres_input_max_edge)),
            output_size=(ref_h, ref_w))
    pred_fntc = _render_model_mip0(quant_model, dataset, texture_height, texture_width, num_mips, device,
                                  superres_base_override=astc_base if quant_model._decoder_adaptation_features is not None else None,
                                  inference_tile=inference_tile, inference_size=(ref_h, ref_w))

    # The uncompressed raster is already materialized; reuse the isolated copy.
    astc_grid_model = quant_model
    apply_latent_compression(astc_grid_model)
    astc_grid_model.eval()
    pred_fntc_grid_astc = _render_model_mip0(
        astc_grid_model, dataset, texture_height, texture_width, num_mips, device,
        superres_base_override=astc_base,
        inference_tile=inference_tile,
        inference_size=(ref_h, ref_w),
    )

    pred_traditional_astc = reference['astc'].to(device)
    metric_key = (id(psnr_metric), id(ssim_metric), id(lpips_metric),
                  None if eval_weights is None else tuple(torch.as_tensor(eval_weights).flatten().tolist()))
    if metric_key not in reference['metrics']:
        reference['metrics'][metric_key] = _compute_group_metrics(
            pred_traditional_astc, gt_image, dataset, psnr_metric, ssim_metric, lpips_metric, eval_weights=eval_weights)

    metrics = {
        "fntc_quantized": _compute_group_metrics(
            pred_fntc, gt_image, dataset, psnr_metric, ssim_metric, lpips_metric, eval_weights=eval_weights
        ),
        fntc_astc_name: _compute_group_metrics(
            pred_fntc_grid_astc, gt_image, dataset, psnr_metric, ssim_metric, lpips_metric, eval_weights=eval_weights
        ),
        ref_astc_name: copy.deepcopy(reference['metrics'][metric_key]),
    }
    if not save_outputs:
        return metrics
    records = [row for method, groups in metrics.items()
               for row in metric_records(groups, method_label(method), storage_bytes=storage[method],
                                         lpips_available=min(h0, w0) >= 128)]
    write_metrics(compare_root, records, evaluation='astc', iteration=curr_iter,
                  metadata=dict(report_context or {}, feature_grid_compression=[r.as_config() for r in grid_recipes(model)], astc_block=astc_codec.astc_block,
                                gt_size=[h0, w0], inference_size=[ref_h, ref_w],
                                normal_encoding=dataset.normal_encoding,
                                group_weights=_group_metric_weights(dataset, eval_weights),
                                storage_bytes={method_label(name): value for name, value in storage.items()},
                                feature_grid_resolutions=feature_resolutions))
    display_labels = {method: f'{method_label(method)}[{storage[method]:,} B]' for method in metrics}
    display_metrics = {method: {group: (p, s, l if min(h0, w0) >= 128 else None)
                               for group, (p, s, l) in groups.items()}
                       for method, groups in metrics.items()}
    if verbose:
        _print_comparison_metrics(display_metrics, display_labels, curr_iter)

    _save_comparison_strip(
        pred_fntc=pred_fntc,
        pred_fntc_grid_astc=pred_fntc_grid_astc,
        pred_traditional_astc=pred_traditional_astc,
        gt_image=gt_image,
        compare_root=compare_root,
        metrics=display_metrics,
        fntc_astc_name=fntc_astc_name,
        ref_astc_name=ref_astc_name,
        vis_configs=vis_configs,
        postprocess_for_vis=postprocess_for_vis,
    )

    return metrics


def write_material_metrics(directory, records, storage, model, dataset, ref_astc_resolution,
                           channel_weights, *, report_context=None, iteration=None):
    size = comparison_material_size(model, dataset.texture_height, dataset.texture_width, ref_astc_resolution)
    return write_metrics(directory, records, evaluation='material', iteration=iteration,
                         metadata=dict(report_context or {},
                                       gt_size=[dataset.texture_height, dataset.texture_width],
                                       inference_size=list(size), normal_encoding=dataset.normal_encoding,
                                       channel_weights=channel_weights,
                                       storage_bytes={method_label(name): value for name, value in storage.items()},
                                       mip_aggregation='arithmetic_mean_of_mip_group_scores'))
