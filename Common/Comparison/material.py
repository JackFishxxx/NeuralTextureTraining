"""Material metrics, annotated outputs and mixed-format comparisons."""
import csv
import math
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
from Core.Feature_Grid.compression import latent_file_bytes
from Common.Material.reconstruction import (postprocess_material,
                                            make_deployment_model,render_material_mip0,cached_sr_base,
                                            sr_base_source_mip,export_sr_base)
from Core.Feature_Grid.sampling import resize_bilinear_repeat
from dataset import prefiltered_material, material_size

UNCOMPRESSED = 'FNTC_Uncompressed'


def _recipe_suffix(compression, dataset):
    return '_'.join(group.capitalize()+format for group,format in compression.recipe(dataset).items())


def baseline_method(compression, dataset):
    return 'Baseline_'+_recipe_suffix(compression,dataset)


def recompressed_method(compression, dataset):
    return 'FNTC_Recompressed_'+_recipe_suffix(compression,dataset)


def primary_method(compression, dataset, enabled):
    return recompressed_method(compression,dataset) if enabled else UNCOMPRESSED


def _at_gt_size(image, dataset):
    size = (dataset.texture_height,dataset.texture_width)
    return resize_bilinear_repeat(image,size) if tuple(image.shape[-2:]) != size else image


def comparison_reference(dataset, compression, size):
    key = (tuple((id(t), t._version) for t in dataset.mip_cache), tuple(size),compression.tile_size,
           tuple(compression.recipe(dataset).items()), dataset.normal_encoding,
           tuple((k, tuple(v)) for k, v in dataset.texture_packing.items()),
           tuple((k, getattr(c, 'astcenc_quality', None), getattr(c, 'astcenc_path', None))
                 for k,c in compression.codecs.items()))
    previous = getattr(dataset, '_material_reference_cache', None)
    if previous is None or previous[0] != key:
        gt = dataset.expand_to_canonical(dataset.mip_cache[0]).permute(2,0,1)[None].float()
        low = prefiltered_material(dataset, size)
        baseline, textures = compression.compress(low, dataset)
        low = _at_gt_size(low,dataset)
        dataset._material_reference_cache = (key, dict(gt=gt.cpu(), low_gt=low.cpu(), baseline=baseline.cpu(),
                                                      textures=textures,pbr={}))
    return dataset._material_reference_cache[1]


@dataclass
class MaterialEvaluation:
    """GT-sized images for image metrics/screenshots; native rasters for PBR sampling."""
    images: dict
    native_images: dict
    size: tuple
    reference: dict | None = None
    textures: dict = field(default_factory=dict)
    compressed_maps: dict = field(default_factory=dict)
    sample_model: object = None


@torch.no_grad()
def evaluate_materials(model, dataset, config, compression, *, baseline=True,
                       material_override=None, primary_only=False):
    size = material_size(model,dataset.texture_height,dataset.texture_width,
                         resolution=config.material_compare_resolution)
    native = material_override
    on_sample = getattr(model, 'inference_mode', 'on_load') == 'on_sample'
    if on_sample and material_override is not None:
        raise ValueError('on_sample evaluation requires the latent decoder, not a material override')
    if native is None:
        decoded = make_deployment_model(model)
        base = (cached_sr_base(dataset, compression, config.super_resolution_base_resolution, size)
                if model.super_resolution_enable else None)
        native = render_material_mip0(decoded, size, config.device, superres_base=base,
                                      inference_tile=config.eval_inference_tile)
    if tuple(native.shape[-2:]) != tuple(size):
        raise ValueError('Material override must use the configured inference tile raster')
    if not torch.isfinite(native).all():
        raise ValueError('Nonfinite FNTC material prediction')
    textures = {}
    native_images = {UNCOMPRESSED: native}
    if config.material_recompression_enable:
        name = recompressed_method(compression,dataset)
        decoded,textures[name] = compression.compress(native,dataset)
        native_images[name] = decoded
    names = [primary_method(compression,dataset,config.material_recompression_enable)] if primary_only else native_images
    images = {name: _at_gt_size(native_images[name],dataset) for name in names}
    evaluation = MaterialEvaluation(images=images, native_images=native_images, size=tuple(size), textures=textures)
    if on_sample:
        evaluation.sample_model = decoded
    return complete_material_evaluation(evaluation,dataset,compression) if baseline else evaluation


def complete_material_evaluation(evaluation, dataset, compression):
    """Add report images to this iteration's PSNR evaluation without rendering again."""
    if evaluation.reference is None:
        reference = comparison_reference(dataset,compression,evaluation.size)
        device = next(iter(evaluation.native_images.values())).device
        name = baseline_method(compression,dataset)
        evaluation.reference = reference
        evaluation.native_images[name] = reference['baseline'].to(device)
        evaluation.native_images['GT'] = reference['gt'].to(device)
        evaluation.textures[name] = reference['textures']
    for name,image in evaluation.native_images.items():
        if name not in evaluation.images:
            evaluation.images[name] = _at_gt_size(image,dataset)
    return evaluation


METRIC_FIELDS = ('iteration', 'method', 'view', 'mip', 'group', 'psnr_type',
          'psnr', 'ssim', 'lpips', 'lpips_status', 'storage_bytes')


def metric_records(metrics, method, *, view='Material', mip=0, storage_bytes=None,
                   lpips_available=True):
    records = []
    for group, values in metrics.items():
        psnr, ssim, lpips = (float(v) if v is not None and math.isfinite(v) else None for v in values)
        records.append(dict(method=method, view=view, mip=mip, group=group,
                            psnr_type=('weighted_group_mean' if group == 'average' else 'image_mse'),
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
    report = dict(schema_version=3, evaluation=evaluation, iteration=iteration,
                  storage_scope=dict(unit='bytes', material_mips='full_chain',
                                     latent_mips='all_exported_levels', decoder_metadata_included=True,
                                     sr_base_included=True, sr_base_mips='full_chain',
                                     container_overhead_included=True),
                  metric_definitions={
                      'psnr': 'dB; image MSE or weighted group mean; exact image matches report 120 dB',
                      'normal': 'PSNR/SSIM on encoded channels; LPIPS averages scalar channels replicated to RGB',
                      'average': 'weighted mean of group scores, using the sum of effective channel weights',
                      'romsd_lpips': 'mean of per-scalar LPIPS, each scalar replicated to RGB',
                      'lpips_status': 'not_computed is null in JSON and empty in CSV; never a perfect score',
                      'storage': 'deployment file bytes: all exported latent levels, decoder/CNN NPZ including metadata and containers, enabled full-mip-chain SR base; baseline/GT use full material mip chains regardless of latent level counts; training state and runtime output cache excluded',
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
    scalar_channels: bool = False,
    psnr_only: bool = False,
) -> Tuple[float, Optional[float], Optional[float]]:
    pred_ref = torch.nan_to_num(pred_ref.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    gt_ref = torch.nan_to_num(gt_ref.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    psnr_metric.reset()
    psnr_value = float(psnr_metric(pred_ref, gt_ref).item())
    psnr_metric.reset()
    if psnr_value == math.inf:
        psnr_value = 120.0
    if not math.isfinite(psnr_value):
        psnr_value = 0.0
    if psnr_only:
        return psnr_value, None, None
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
def _compute_group_metrics(pred_image, gt_image, dataset, psnr_metric, ssim_metric, lpips_metric,
                           eval_weights=None, *, psnr_only=False):
    group_channels = _metric_channel_groups(dataset)
    group_metrics: Dict[str, Tuple[float, Optional[float], Optional[float]]] = {}
    for group_name, channels in group_channels.items():
        ch_idx = torch.tensor(channels, device=pred_image.device, dtype=torch.long)
        pred_ref = torch.index_select(pred_image, dim=1, index=ch_idx)
        gt_ref = torch.index_select(gt_image, dim=1, index=ch_idx)
        group_metrics[group_name] = _compute_metrics_from_refs(
            pred_ref, gt_ref, psnr_metric, ssim_metric, lpips_metric,
            scalar_channels=(group_name != "diffuse"),
            psnr_only=psnr_only,
        )

    if group_metrics:
        order = [n for n in ("diffuse", "normal", "romsd") if n in group_metrics]
        weights = _group_metric_weights(dataset, eval_weights)
        vals = np.array([group_metrics[n][:1] if psnr_only else group_metrics[n] for n in order], dtype=np.float64)
        ws = np.array([max(0.0, weights.get(n, 1.0)) for n in order], dtype=np.float64)
        if ws.sum() <= 0:
            ws = np.ones_like(ws)
        ws = ws / ws.sum()
        averages = tuple(float(v) if math.isfinite(v) else 0.0 for v in (vals * ws[:, None]).sum(axis=0))
        group_metrics["average"] = averages + (None, None) if psnr_only else averages
    return group_metrics


# ---------------------------------------------------------------------------
# Comparison strip + annotated tiles
# ---------------------------------------------------------------------------

def _format_metric_triplet(
    metric: Optional[Tuple[float, float, float]],
) -> str:
    if metric is None:
        return f"PSNR:{'-':>8}  SSIM:{'-':>7}  LPIPS:{'-':>7}"
    lpips_text = f'{metric[2]:7.4f}' if metric[2] is not None else f'{"N/A":>7}'
    return f"PSNR:{metric[0]:8.3f}  SSIM:{metric[1]:7.4f}  LPIPS:{lpips_text}"


def _print_comparison_metrics(metrics, curr_iter=None):
    label_width = max(map(len,metrics))
    for method, values in metrics.items():
        print(f"[Material Test] {method:<{label_width}} Iter {curr_iter}: Weighted {_format_metric_triplet(values.get('average'))} "
              f"(Diffuse {_format_metric_triplet(values.get('diffuse'))}, "
              f"Normal {_format_metric_triplet(values.get('normal'))}, "
              f"ROMSD {_format_metric_triplet(values.get('romsd'))})")


def load_overlay_font(font_size: int, fonts=("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf")) -> ImageFont.ImageFont:
    for font_name in fonts:
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
    if name == 'fntc_uncompressed':
        return (46, 196, 255, 140)
    if name.startswith('fntc_'):
        return (255, 181, 71, 140)
    if name.startswith('baseline_') or name.startswith(('bc', 'astc')):
        return (255, 99, 146, 140)
    if name.startswith('gt'):
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

    display_name = method_name

    if metric is None:
        text = f"{display_name}\nReference" if method_name == "GT" else f"{display_name}\n{_format_metric_triplet(None)}"
    else:
        selected_group = metric_group if metric_group in metric else "average"
        selected_metric = metric.get(selected_group)
        text = f"{display_name}[{selected_group.upper()}]\n{_format_metric_triplet(selected_metric)}"

    font_size = max(18, min(36, h // 40, w // 28))
    font = load_overlay_font(font_size)
    draw = ImageDraw.Draw(tile_pil)
    while draw.multiline_textbbox((0, 0), text, font=font)[2] > w - 24 and font_size > 7:
        font_size -= 1
        font = load_overlay_font(font_size)
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


def log_comparison_storage(model, dataset, compression, config, *, verbose=True):
    size = material_size(model,dataset.texture_height,dataset.texture_width,
                         resolution=config.material_compare_resolution)
    latent = latent_file_bytes(model)
    network = model.network_data_bytes()
    tile = compression.file_bytes(dataset, size, full_chain=True)
    base = 0
    if model.super_resolution_enable:
        level = dataset.mip_cache[sr_base_source_mip(dataset,config.super_resolution_base_resolution)]
        base = compression.file_bytes(dataset,level.shape[:2],full_chain=True)
    channels = sum(b-a for name in dataset.available_textures
                   for a,b in [dataset.canonical_channel_slices[name]])
    storage = {UNCOMPRESSED: latent+network+base,
               baseline_method(compression,dataset): tile,
               'GT': sum(h*w*channels for h,w in compression.mip_sizes(
                   (dataset.texture_height, dataset.texture_width), full_chain=True))}
    if config.material_recompression_enable:
        storage[recompressed_method(compression,dataset)] = latent+network+base
    if verbose:
        rows = [('Method', 'Total (KiB)', 'Compression ratio', 'Latent (KiB)', 'Decoder/CNN (KiB)',
                 'SR base (Full Mips Chain) (KiB)')]
        for name in ('GT', baseline_method(compression,dataset),
                     primary_method(compression,dataset,config.material_recompression_enable)):
            count = storage[name]
            fntc = name.startswith('FNTC_')
            rows.append((name, f'{count/1024:,.2f}', f'{storage["GT"]/count:.2f}x' if count else '-',
                         f'{latent/1024:,.2f}' if fntc else '-', f'{network/1024:,.2f}' if fntc else '-',
                         f'{base/1024:,.2f}' if fntc else '-'))
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        print(f'[Storage] All latent levels; full mip chains: Baseline / SR base / GT (UNORM8 effective channels); '
              f'file headers included; compression ratio=GT/total; tile={compression.tile_size}')
        for row in rows:
            print('[Storage] ' + ' | '.join(value.ljust(width) if i == 0 else value.rjust(width)
                                           for i, (value, width) in enumerate(zip(row, widths))))
    return storage


def _display_metrics(metrics, size):
    return {name: {g: (p,s,l if min(size) >= 128 else None) for g,(p,s,l) in groups.items()}
            for name,groups in metrics.items()}


@torch.no_grad()
def save_material_images(evaluation, dataset, compression, metric_objects, directory, *, metrics=None):
    """Every screenshot uses GT dimensions; GT filenames retain source dimensions."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    height,width = dataset.texture_height,dataset.texture_width
    ih,iw = evaluation.size
    gt = evaluation.images['GT']
    low = evaluation.reference['low_gt'].to(gt.device)
    for vc in dataset.get_vis_configs():
        a,b = vc['canonical_channel_slice']
        group = next(group for group,names in dataset.texture_packing.items() if vc['texture_type'] in names)
        prefix = vc['texture_type'].capitalize()
        format_name = compression.formats[group].name
        items = [(f'GT[{width}x{height}]', gt)]
        if (ih,iw) != (height,width):
            items.insert(0,(f'GT[{iw}x{ih}]', low))
        baseline = baseline_method(compression,dataset)
        items += [(format_name, evaluation.images[baseline]),
                  (UNCOMPRESSED, evaluation.images[UNCOMPRESSED])]
        recompressed = recompressed_method(compression,dataset)
        if recompressed in evaluation.images:
            items.append(('FNTC_Recompressed', evaluation.images[recompressed]))
        panels = {}
        for filename,image in items:
            visual = postprocess_material(image[0,a:b], vc['vis_mode'], dataset.normal_encoding)
            # Reuse the group score when this image is exactly that group.
            method = {format_name: baseline, 'FNTC_Recompressed': recompressed}.get(filename,filename)
            if metrics is not None and method in metrics and dataset.texture_packing[group] == [vc['texture_type']]:
                values = metrics[method][group]
            else:
                values = _compute_metrics_from_refs(image[:,a:b],gt[:,a:b],*metric_objects,
                                                    scalar_channels=(vc['texture_type']!='diffuse'))
            scores = {group: (values[0],values[1],values[2] if min(height,width)>=128 else None)}
            label = f'{filename}_{format_name}' if filename == 'FNTC_Recompressed' else filename
            tile = _make_annotated_tile(visual, label, scores, group)
            # Lossless PNG at a low compression level keeps report generation
            # from blocking training on CPU compression of 2K/8K panels.
            tile.save(directory/f'{prefix}_{filename}.png', compress_level=1)
            panels[filename] = tile
        order = [UNCOMPRESSED]
        if 'FNTC_Recompressed' in panels:
            order.append('FNTC_Recompressed')
        order += [format_name,f'GT[{width}x{height}]']
        _concat_pil_h([panels[name] for name in order]).save(
            directory/f'comparison_strip_{prefix}.png', compress_level=1)


@torch.no_grad()
def run_material_comparison_pipeline(trainer, *, evaluation=None, curr_iter=None, output_root=None,
                                     verbose=True, save_outputs=True, report_context=None):
    evaluation = evaluation or evaluate_materials(trainer.model, trainer.dataset, trainer.configs,
                                                  trainer.material_compression)
    compression, dataset = trainer.material_compression, trainer.dataset
    gt = evaluation.images['GT']
    metrics = {name: trainer._compute_material_metrics(image,gt)
               for name,image in evaluation.images.items() if name != 'GT'}
    if not save_outputs:
        return metrics
    root = Path(output_root or trainer.media_path)/'Compare_Material'
    if curr_iter is not None:
        root = root/f'iter_{curr_iter:08d}'
    storage = log_comparison_storage(trainer.model,dataset,compression,trainer.configs,verbose=False)
    records = [r for name,groups in metrics.items()
               for r in metric_records(groups,name,storage_bytes=storage[name],
                                        lpips_available=min(gt.shape[-2:])>=128)]
    primary = primary_method(compression,dataset,trainer.configs.material_recompression_enable)
    sr_base = (export_sr_base(dataset,compression,trainer.configs.super_resolution_base_resolution,root/'SR_Base')
               if trainer.model.super_resolution_enable else None)
    write_metrics(root,records,evaluation='material',iteration=curr_iter,
                  metadata=dict(report_context or {},primary_method=primary,
                                inference_mode=getattr(trainer.model, 'inference_mode', 'on_load'),
                                material_sampling=('direct query at configured output texel centers; bilinear resize to original GT for image metrics; PBR uses pixel latent decoding'
                                                   if evaluation.sample_model is not None else 'native material raster upsampled to GT'),
                                compression_formats=compression.recipe(dataset),
                                sr_base=sr_base,
                                fntc_tile_size=trainer.configs.fntc_tile_size,
                                tile_layout=compression.tile_layout(dataset,evaluation.size),
                                material_mipmaps='full chain down to 1x1 in one DDS (BC) or KTX2 (ASTC) file per group; storage_bytes includes the container',
                                texture_packing=dataset.texture_packing,
                                canonical_channel_slices=dataset.canonical_channel_slices,
                                normal_encoding=dataset.normal_encoding,
                                latent_formats=[r.as_config() for r in trainer.model.grid_compressions],
                                gt_size=list(gt.shape[-2:]),inference_size=list(evaluation.size),
                                screenshot_size=list(gt.shape[-2:]),storage_bytes=storage,
                                storage_meaning={'FNTC':'all latent files + decoder/CNN NPZ metadata + enabled full-mip-chain SR base; file headers included; output cache excluded',
                                                 'Baseline':'complete material mip-chain files regardless of latent level counts'},
                                output_tile_cache_bytes=(compression.file_bytes(dataset,evaluation.size, full_chain=True)
                                                        if trainer.configs.material_recompression_enable else 0),
                                group_weights=_group_metric_weights(dataset,trainer.eval_weights)))
    display = _display_metrics(metrics,gt.shape[-2:])
    save_material_images(evaluation,dataset,compression,
                         (trainer.psnr,trainer.ssim,trainer.lpips),root,metrics=metrics)
    for method,textures in (evaluation.textures or {}).items():
        label = 'Baseline' if method == baseline_method(compression,dataset) else 'FNTC_Recompressed'
        compression.export_materials(textures,dataset,evaluation.size,root,method=label)
    if verbose:
        _print_comparison_metrics(display,curr_iter)
    return metrics
