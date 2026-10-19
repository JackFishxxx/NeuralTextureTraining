"""
ASTC comparison pipeline: FNTC (quantized / grid-ASTC) vs traditional RGBA ASTC round-trip.

Uses official astcenc for encode+decode baselines. Visualization and metrics align on Mip0
full texture resolution; optional ref_astc_resolution (int square side) resamples only the traditional baseline path.
"""
from __future__ import annotations

import copy
import math
import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
from Core.ASTC_Aware.differentiable_proxy import apply_astc_to_feature_grids
from Core.Feature_Grid.sampling import resize_bilinear_repeat, sample_image
from Core.Material_Encoding.normal_encoding import decode_normal, normal_angular_psnr
from dataset import material_size, prefiltered_material, clean_base_image


# ---------------------------------------------------------------------------
# Version / defaults
# ---------------------------------------------------------------------------

ASTCENC_DEFAULT_VERSION = "5.3.0"
ASTCENC_DEFAULT_PATH = "tools/astcenc"


def normalize_astc_block(block: str) -> str:
    block_str = str(block).strip().lower()
    if "x" not in block_str:
        raise ValueError(f"Invalid ASTC block format '{block}'. Expected like '6x6'.")
    w_str, h_str = block_str.split("x", 1)
    w, h = int(w_str), int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid ASTC block '{block}'. Block dimensions must be positive.")
    return f"{w}x{h}"


def _ref_astc_hw_from_side(side: Optional[int], base_h: int, base_w: int) -> Tuple[int, int]:
    """Traditional ASTC baseline: square side×side, or None → Mip0 (base_h, base_w)."""
    if side is None:
        return base_h, base_w
    if side <= 0:
        raise ValueError(f"ref_astc_resolution must be a positive int (square edge), got {side}")
    scale = float(side) / float(max(base_h, base_w))
    return max(1, int(round(base_h * scale))), max(1, int(round(base_w * scale)))


# ---------------------------------------------------------------------------
# Codec + subprocess round-trip
# ---------------------------------------------------------------------------

@dataclass
class ASTCCodec:
    astcenc_path: str
    astcenc_quality: str
    astc_block: str = "6x6"
    _astcenc_executable: Optional[str] = None

    def __post_init__(self) -> None:
        self.astc_block = normalize_astc_block(self.astc_block)

    def ensure_executable(self) -> str:
        if self._astcenc_executable is None:
            self._astcenc_executable = resolve_astcenc_executable(astcenc_path=self.astcenc_path)
        return self._astcenc_executable

    def roundtrip_rgba(self, rgba: np.ndarray) -> np.ndarray:
        return astc_roundtrip_rgba(
            rgba=rgba,
            astcenc_executable=self.ensure_executable(),
            astcenc_quality=self.astcenc_quality,
            astc_block=self.astc_block,
        )


def astc_roundtrip_rgba(
    rgba: np.ndarray,
    astcenc_executable: str,
    astcenc_quality: str,
    astc_block: str,
) -> np.ndarray:
    h, w, c = rgba.shape
    if c != 4:
        raise ValueError(f"Expected RGBA input with 4 channels, got {c}.")
    rgba_u8 = np.asarray(rgba, dtype=np.uint8)
    block = normalize_astc_block(astc_block)

    with tempfile.TemporaryDirectory(prefix=f"astc_{block}_") as tmp_dir:
        input_png = os.path.join(tmp_dir, "input_rgba.png")
        output_astc = os.path.join(tmp_dir, f"output_{block}.astc")
        output_png = os.path.join(tmp_dir, "output_rgba.png")

        Image.fromarray(rgba_u8).save(input_png)
        enc_cmd = [astcenc_executable, "-cl", input_png, output_astc, block, f"-{astcenc_quality}"]
        dec_cmd = [astcenc_executable, "-dl", output_astc, output_png]

        try:
            subprocess.run(enc_cmd, check=True, capture_output=True, text=True)
            subprocess.run(dec_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr_msg = (exc.stderr or "").strip()
            stdout_msg = (exc.stdout or "").strip()
            details = stderr_msg if stderr_msg else stdout_msg
            raise RuntimeError(
                f"astcenc roundtrip failed with command '{' '.join(exc.cmd)}'. Details: {details}"
            ) from exc

        decoded = np.array(Image.open(output_png).convert("RGBA"), dtype=np.uint8, copy=True)
        if decoded.shape[0] != h or decoded.shape[1] != w:
            decoded = decoded[:h, :w, :]
        return decoded


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

@torch.no_grad()
def _render_gt_mip0(dataset, texture_height: int, texture_width: int) -> torch.Tensor:
    mip = 0
    mip_height = texture_height // (2 ** mip)
    mip_width = texture_width // (2 ** mip)
    gt_slice = dataset.mip_cache[mip]
    gt_canonical = dataset.expand_to_canonical(gt_slice.reshape(-1, gt_slice.shape[-1]))
    gt_canonical = gt_canonical.reshape(mip_height, mip_width, -1)
    return gt_canonical.permute(2, 0, 1)[None, ...]


@torch.no_grad()
def _render_model_mip0(model, dataset, texture_height: int, texture_width: int, num_mips: int, device: str,
                       superres_base_override: Optional[torch.Tensor] = None,
                       inference_tile: int = 512) -> torch.Tensor:
    """Render Mip0 in tiles, bounding decoder and feature-gradient temporaries."""
    mip = 0
    H = texture_height >> mip
    W = texture_width >> mip
    target_size = (H, W)
    H, W = material_size(model, H, W)
    native_base = None
    if getattr(model, 'super_resolution_enable', False) and (H, W) != target_size:
        native_base = (clean_base_image(dataset, (H, W)) if superres_base_override is None
                       else superres_base_override)
        if tuple(native_base.shape[-2:]) != (H, W):
            raise ValueError('Render the compressed SR base at native material size before passing superres_base_override')
    mip_f = float(mip) / max(1, num_mips - 1)
    step = max(1, int(inference_tile) or max(H, W))
    hwc = None
    for h0 in range(0, H, step):
        h1 = min(h0 + step, H)
        for w0 in range(0, W, step):
            w1 = min(w0 + step, W)
            rr, cc = torch.meshgrid(torch.arange(h0, h1, device=device),
                                    torch.arange(w0, w1, device=device), indexing="ij")
            inp = torch.stack(((cc + 0.5) / W, (rr + 0.5) / H,
                               torch.full_like(rr, mip_f)), dim=-1).reshape(-1, 3)
            if getattr(model, "super_resolution_enable", False):
                if native_base is not None:
                    base = sample_image(native_base, inp[:, :2])
                elif superres_base_override is None:
                    base_data = dataset.superres_base_cache[mip][h0:h1, w0:w1]
                    base = dataset.expand_to_canonical(base_data.reshape(-1, dataset.num_channels)).float()
                else:
                    base_chw = superres_base_override[0] if superres_base_override.ndim == 4 else superres_base_override
                    base = base_chw[:, h0:h1, w0:w1].permute(1, 2, 0).reshape(-1, base_chw.shape[0]).float()
                y = (base + model(torch.cat([inp, base], dim=1)).float()).clamp(0.0, 1.0)
            else:
                y = model(inp).float()
            if hwc is None:
                hwc = torch.empty(H, W, y.shape[-1], device=device, dtype=torch.float32)
            hwc[h0:h1, w0:w1] = y.reshape(h1-h0, w1-w0, -1)
    if getattr(model, "direct_diffuse_enabled", False) and hasattr(model, "render_direct_diffuse_mip"):
        direct = model.render_direct_diffuse_mip(mip, H, W)
        hwc[:, :, :3] = direct.squeeze(0).permute(1, 2, 0).to(hwc.dtype)
    hwc = torch.nan_to_num(hwc, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    image = hwc.permute(2, 0, 1)[None, ...]
    return resize_bilinear_repeat(image, target_size) if (H, W) != target_size else image


# ---------------------------------------------------------------------------
# Metrics (per texture family + average)
# ---------------------------------------------------------------------------

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
        return "AngularPSNR", "AngularSSIM", "AngularLPIPS"
    return "PSNR", "SSIM", "LPIPS"


def _format_metric_triplet(
    metric: Optional[Tuple[float, float, float]],
    metric_group: Optional[str] = None,
) -> str:
    psnr_label, ssim_label, lpips_label = _metric_labels(metric_group)
    if metric is None:
        return f"{psnr_label}:{'-':>8}  {ssim_label}:{'-':>7}  {lpips_label}:{'-':>7}"
    return f"{psnr_label}:{metric[0]:8.3f}  {ssim_label}:{metric[1]:7.4f}  {lpips_label}:{metric[2]:7.4f}"


def _print_comparison_metrics(metrics, display_labels):
    label_width = max(len(label) for label in display_labels.values())
    for method, values in metrics.items():
        print(f"[ASTC Test] {display_labels[method]:<{label_width}} Weighted {_format_metric_triplet(values.get('average'))} "
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

    if method_name == "fntc_quantized":
        box_color = (46, 196, 255, 140)
    elif method_name.startswith("fntc_astc_"):
        box_color = (255, 181, 71, 140)
    elif method_name.startswith("ref_astc_"):
        box_color = (255, 99, 146, 140)
    elif method_name == "gt":
        box_color = (127, 255, 127, 140)
    else:
        box_color = (120, 120, 255, 140)

    draw.rounded_rectangle(
        (box_x, box_y, min(w - pad, box_x + box_w), min(h - pad, box_y + box_h)),
        radius=max(4, font_size // 5),
        fill=box_color,
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
        names = ("Uncompressed_FNTC", f"FNTC_{fntc_astc_name.removeprefix('fntc_astc_')}",
                 f"ASTC_{ref_astc_name.removeprefix('ref_astc_')}", "GT")
        for name, tile in zip(names, annotated):
            tile.save(os.path.join(compare_root, f"{texture_prefix}_{name}.png"))


# ---------------------------------------------------------------------------
# astcenc discovery / Windows auto-install
# ---------------------------------------------------------------------------

def resolve_astcenc_executable(astcenc_path: str) -> str:
    candidate = (astcenc_path or "").strip()
    if not candidate:
        raise RuntimeError("Empty astcenc path. Set --astcenc_path.")

    cmd_path = _resolve_candidate_command(candidate)
    if cmd_path:
        return cmd_path

    target_path = _to_abs_path(candidate)
    if os.path.isfile(target_path):
        return target_path

    if os.path.isdir(target_path):
        found = _find_astcenc_in_dir(target_path)
        if found:
            return found
        raise FileNotFoundError(f"astcenc directory exists but no executable found: '{target_path}'.")

    install_root = target_path
    if os.path.splitext(target_path)[1].lower() == ".exe":
        install_root = os.path.dirname(target_path)
    os.makedirs(install_root, exist_ok=True)
    installed = _auto_install_astcenc(ASTCENC_DEFAULT_VERSION, install_root)
    print(f"[ASTC] Using auto-installed astcenc: {installed}")
    return installed


def _resolve_candidate_command(candidate: str) -> Optional[str]:
    has_sep = (os.path.sep in candidate) or (os.path.altsep is not None and os.path.altsep in candidate)
    if os.path.isabs(candidate) or has_sep:
        abs_path = os.path.abspath(candidate)
        if os.path.isfile(abs_path):
            return abs_path
        return None
    resolved = shutil.which(candidate)
    if resolved and os.path.isfile(resolved):
        return resolved
    return None


def _to_abs_path(candidate: str) -> str:
    if os.path.isabs(candidate):
        return os.path.normpath(candidate)
    return os.path.abspath(os.path.normpath(candidate))


def _preferred_astcenc_names():
    if os.name == "nt":
        return ["astcenc-avx2.exe", "astcenc-sse4.1.exe", "astcenc-sse2.exe", "astcenc.exe"]
    return ["astcenc", "astcenc-avx2", "astcenc-sse4.1", "astcenc-sse2", "astcenc-neon"]


def _find_astcenc_in_dir(root_dir: str) -> Optional[str]:
    preferred = _preferred_astcenc_names()
    preferred_lower = [n.lower() for n in preferred]
    matches = []
    for walk_root, _, files in os.walk(root_dir):
        for fname in files:
            lower = fname.lower()
            full_path = os.path.join(walk_root, fname)
            if lower in preferred_lower:
                return full_path
            if lower.startswith("astcenc") and (os.name != "nt" or lower.endswith(".exe")):
                matches.append(full_path)
    return sorted(matches)[0] if matches else None


def _auto_install_astcenc(version: str, install_root: str) -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "windows" or machine not in {"amd64", "x86_64"}:
        raise RuntimeError(
            "Automatic astcenc install currently supports Windows x64 only. "
            "Please install astcenc manually or set --astcenc_path."
        )

    asset_name = f"astcenc-{version}-windows-x64.zip"
    url = f"https://github.com/ARM-software/astc-encoder/releases/download/{version}/{asset_name}"

    os.makedirs(install_root, exist_ok=True)
    zip_path = os.path.join(install_root, asset_name)
    extract_dir = os.path.join(install_root, f"astcenc-{version}-windows-x64")

    if not os.path.isfile(zip_path):
        print(f"[ASTC] Downloading astcenc {version} from {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to download astcenc from '{url}': {exc}") from exc

    os.makedirs(extract_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract '{zip_path}': {exc}") from exc

    found = _find_astcenc_in_dir(extract_dir)
    if found:
        return found
    raise RuntimeError(f"Auto-install succeeded but no astcenc executable found in '{extract_dir}'.")


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
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    block_tag = astc_codec.astc_block.lower()
    fntc_astc_name = f"fntc_astc_{astc_codec.astc_block}"
    ref_astc_name = f"ref_astc_{astc_codec.astc_block}"

    compare_root = os.path.join(output_root, f"Compare_ASTC{block_tag}")
    if curr_iter is not None:
        compare_root = os.path.join(compare_root, f"iter_{curr_iter:08d}")
    os.makedirs(compare_root, exist_ok=True)

    gt_image = _render_gt_mip0(dataset, texture_height, texture_width)
    h0, w0 = gt_image.shape[-2], gt_image.shape[-1]
    feature_resolutions = []
    for spec in getattr(model, "feature_grid_specs", []):
        try:
            feature_resolutions.append(int(spec.highest_level_slice()[2]))
        except (AttributeError, TypeError, ValueError):
            pass
    effective_ref_resolution = ref_astc_resolution or max(material_size(model, h0, w0))
    ref_h, ref_w = _ref_astc_hw_from_side(effective_ref_resolution, h0, w0)
    native_h, native_w = material_size(model, h0, w0)
    with open(os.path.join(compare_root, 'reconstruction.json'), 'w', encoding='utf-8') as output:
        json.dump({'gt_size': [h0, w0], 'fntc_material_size': [native_h, native_w],
                   'astc_material_size': [ref_h, ref_w], 'upsampler': 'bilinear_repeat',
                   'texture_packing': dataset.texture_packing,
                   'texel_centers': True, 'decoder_queries': native_h * native_w}, output, indent=2)
    if (ref_h, ref_w) != (h0, w0):
        print(f"[ASTC Test] {ref_astc_name} encode/decode at {ref_h}x{ref_w}, aligned to Mip0 {h0}x{w0} for metrics")

    quant_model = copy.deepcopy(model)
    quant_model.simulate_quantize()
    quant_model.eval()
    astc_base = None
    if getattr(quant_model, "super_resolution_enable", False):
        astc_base = _astc_roundtrip_superres_base_mip0(
            dataset, astc_codec.roundtrip_rgba,
            int(getattr(dataset, "super_resolution_base_resolution", dataset.superres_input_max_edge)),
            output_size=material_size(quant_model, texture_height, texture_width))
    pred_fntc = _render_model_mip0(quant_model, dataset, texture_height, texture_width, num_mips, device,
                                  superres_base_override=astc_base if quant_model._decoder_adaptation_features is not None else None,
                                  inference_tile=inference_tile)

    astc_grid_model = copy.deepcopy(quant_model)
    apply_astc_to_feature_grids(astc_grid_model, astc_codec.roundtrip_rgba)
    astc_grid_model.eval()
    pred_fntc_grid_astc = _render_model_mip0(
        astc_grid_model, dataset, texture_height, texture_width, num_mips, device,
        superres_base_override=astc_base,
        inference_tile=inference_tile,
    )

    pred_traditional_astc = _traditional_baseline_resampled(
        gt_mip0=gt_image,
        dataset=dataset,
        roundtrip_rgba=astc_codec.roundtrip_rgba,
        ref_h=ref_h,
        ref_w=ref_w,
    )

    metrics = {
        "fntc_quantized": _compute_group_metrics(
            pred_fntc, gt_image, dataset, psnr_metric, ssim_metric, lpips_metric, eval_weights=eval_weights
        ),
        fntc_astc_name: _compute_group_metrics(
            pred_fntc_grid_astc, gt_image, dataset, psnr_metric, ssim_metric, lpips_metric, eval_weights=eval_weights
        ),
        ref_astc_name: _compute_group_metrics(
            pred_traditional_astc, gt_image, dataset, psnr_metric, ssim_metric, lpips_metric, eval_weights=eval_weights
        ),
    }
    metrics_path = os.path.join(compare_root, f"metrics_astc_{astc_codec.astc_block}.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        group_order = ("diffuse", "normal", "romsd", "average")
        method_width = max(len(name) for name in metrics)
        for method_name, method_metrics in metrics.items():
            for group_name in group_order:
                if group_name not in method_metrics:
                    continue
                p, s, l = method_metrics[group_name]
                metric_name = "normal_angular" if group_name == "normal" else group_name
                labels = _metric_labels("normal" if group_name == "normal" else None)
                f.write(f"{method_name:<{method_width}} {metric_name:<14} {labels[0]:<12} {p:12.6f} "
                        f"{labels[1]:<12} {s:10.6f} {labels[2]:<12} {l:10.6f}\n")

    display_labels = {}
    for method_name in metrics:
        if method_name == "fntc_quantized":
            display_name = "FNTC_uncompressed"
        elif method_name == fntc_astc_name:
            display_name = f"FNTC_{astc_codec.astc_block}_compressed"
        elif method_name == ref_astc_name:
            display_name = f"ASTC_{astc_codec.astc_block}_compressed"
            source_text = f"{ref_w}x{ref_h}x3"
        else:
            display_name = method_name
            source_text = "unknown"
        if method_name != ref_astc_name:
            source_text = f"material={native_w}x{native_h},GT={w0}x{h0}"
        display_labels[method_name] = f"{display_name}[{source_text}]"
    _print_comparison_metrics(metrics, display_labels)

    _save_comparison_strip(
        pred_fntc=pred_fntc,
        pred_fntc_grid_astc=pred_fntc_grid_astc,
        pred_traditional_astc=pred_traditional_astc,
        gt_image=gt_image,
        compare_root=compare_root,
        metrics=metrics,
        fntc_astc_name=fntc_astc_name,
        ref_astc_name=ref_astc_name,
        vis_configs=vis_configs,
        postprocess_for_vis=postprocess_for_vis,
    )

    return metrics
