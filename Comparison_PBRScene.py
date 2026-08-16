"""Deterministic Cornell-box PBR comparison renderer.

The renderer is intentionally self-contained: it uses NumPy for software
rasterization and Pillow for output, so enabling comparisons does not add a
runtime dependency to the training environment.
"""

import os
import math
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from normal_encoding import decode_normal


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _load_overlay_font(panel_height: int):
    # Pillow's default bitmap font is roughly 10-11 px. Use a 22 px TrueType
    # font at normal output sizes to make comparison metrics about twice as large.
    font_size = max(18, min(28, int(round(panel_height * 22.0 / 512.0))))
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


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


def _directional_shadow_visibility(pos: np.ndarray, normal: np.ndarray,
                                   light_dir: np.ndarray) -> np.ndarray:
    """Trace one hard-shadow ray per shaded pixel against all scene triangles."""
    origins = pos + normal * 1e-3
    direction = _normalize(np.asarray(light_dir, np.float32))
    blocked = np.zeros(pos.shape[0], dtype=bool)
    eps = 1e-7
    for vertices, _, triangles in _cornell_meshes():
        for ids in triangles:
            v0, v1, v2 = vertices[np.asarray(ids)]
            edge1, edge2 = v1 - v0, v2 - v0
            h = np.cross(direction, edge2)
            det = float(np.dot(edge1, h))
            if abs(det) < eps:
                continue
            inv_det = 1.0 / det
            s = origins - v0
            u = inv_det * (s @ h)
            candidates = (~blocked) & (u >= 0.0) & (u <= 1.0)
            if not candidates.any():
                continue
            q = np.cross(s, edge1)
            v = inv_det * (q @ direction)
            t = inv_det * (q @ edge2)
            blocked |= candidates & (v >= 0.0) & ((u + v) <= 1.0) & (t > 1e-4)
    return (~blocked).astype(np.float32)[:, None]


def _sample_level(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Repeat-wrapped bilinear sampling at one mip level."""
    c, h, w = texture.shape
    xy = np.mod(uv, 1.0) * np.asarray([w, h], np.float32) - 0.5
    p0 = np.floor(xy).astype(np.int32)
    f = xy - p0
    x0, y0 = np.mod(p0[:, 0], w), np.mod(p0[:, 1], h)
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    a = texture[:, y0, x0].T * (1 - f[:, :1]) + texture[:, y0, x1].T * f[:, :1]
    b = texture[:, y1, x0].T * (1 - f[:, :1]) + texture[:, y1, x1].T * f[:, :1]
    return a * (1 - f[:, 1:2]) + b * f[:, 1:2]


def _mip_pyramid(texture: np.ndarray):
    levels = [texture]
    while max(levels[-1].shape[-2:]) > 1:
        src = levels[-1]
        h, w = src.shape[-2:]
        if h & 1:
            src = np.concatenate([src, src[:, -1:, :]], axis=1)
        if w & 1:
            src = np.concatenate([src, src[:, :, -1:]], axis=2)
        levels.append(
            0.25 * (src[:, 0::2, 0::2] + src[:, 1::2, 0::2]
                    + src[:, 0::2, 1::2] + src[:, 1::2, 1::2])
        )
    return levels


def _sample(texture: np.ndarray, uv: np.ndarray, uv_dx=None, uv_dy=None,
            mip_lod_bias: float = 0.0) -> np.ndarray:
    """Bilinear magnification and trilinear mip-filtered minification."""
    if uv_dx is None or uv_dy is None:
        return _sample_level(texture, uv)
    levels = _mip_pyramid(texture)
    h, w = texture.shape[-2:]
    scale = np.asarray([w, h], np.float32)
    rho = np.maximum(np.linalg.norm(uv_dx * scale, axis=1), np.linalg.norm(uv_dy * scale, axis=1))
    lod = np.clip(np.log2(np.maximum(rho, 1.0)) + float(mip_lod_bias), 0.0, len(levels) - 1.0)
    lo = np.floor(lod).astype(np.int32)
    hi = np.minimum(lo + 1, len(levels) - 1)
    blend = (lod - lo)[:, None]
    low_result = np.empty((uv.shape[0], texture.shape[0]), np.float32)
    high_result = np.empty_like(low_result)
    for level in np.unique(lo):
        mask = lo == level
        low_result[mask] = _sample_level(levels[int(level)], uv[mask])
    for level in np.unique(hi):
        mask = hi == level
        high_result[mask] = _sample_level(levels[int(level)], uv[mask])
    return low_result * (1.0 - blend) + high_result * blend


def _material_maps(texture: torch.Tensor, slices: Dict[str, tuple], normal_encoding: str,
                   available_textures=None) -> Dict[str, np.ndarray]:
    tex = torch.nan_to_num(texture[0].detach().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    maps = {}
    active = set(slices) if available_textures is None else set(available_textures)
    for name, (start, end) in slices.items():
        if name not in active:
            continue
        maps[name] = tex[start:end].cpu().numpy()
    if "normal" in maps:
        start, end = slices["normal"]
        maps["normal"] = decode_normal(tex[start:end][None], normal_encoding)[0].cpu().numpy()
    return maps


def render_pbr_scene(texture: torch.Tensor, slices: Dict[str, tuple], normal_encoding: str,
                     resolution: int = 512, available_textures=None,
                     mip_lod_bias: float = -1.0, lighting=None) -> np.ndarray:
    """Rasterize the Cornell scene and return an sRGB uint8 image."""
    maps = _material_maps(texture, slices, normal_encoding, available_textures)
    lighting = lighting or {}
    point_lights = lighting.get("point_lights", [
        {"position": [0.0, 0.82, 0.7], "intensity": 6.5, "color": [1.0, 1.0, 1.0]},
        {"position": [-0.8, 0.45, -0.15], "intensity": 1.2, "color": [1.0, 1.0, 1.0]},
    ])
    h = w = int(resolution)
    color = np.full((h, w, 3), 0.018, np.float32)
    depth = np.full((h, w), np.inf, np.float32)
    world = np.zeros((h, w, 3), np.float32)
    normal = np.zeros((h, w, 3), np.float32)
    tangent = np.zeros((h, w, 3), np.float32)
    bitangent = np.zeros((h, w, 3), np.float32)
    uv_buffer = np.zeros((h, w, 2), np.float32)
    covered = np.zeros((h, w), bool)

    eye = np.asarray([0.0, 0.08, -4.25], np.float32)
    target = np.asarray([0.0, -0.05, 1.15], np.float32)
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, np.asarray([0, 1, 0], np.float32)))
    up = _normalize(np.cross(right, forward))
    focal = 1.0 / np.tan(np.deg2rad(50.0) * 0.5)

    def project(p):
        rel = p - eye
        z = rel @ forward
        x = (rel @ right) * focal / z
        y = (rel @ up) * focal / z
        return np.stack([(x * 0.5 + 0.5) * (w - 1), (0.5 - y * 0.5) * (h - 1), z], axis=1)

    for vertices, uvs, triangles in _cornell_meshes():
        projected = project(vertices)
        for ids in triangles:
            ids = np.asarray(ids)
            p, xyz, tuv = projected[ids], vertices[ids], uvs[ids]
            xmin, xmax = max(0, int(np.floor(p[:, 0].min()))), min(w - 1, int(np.ceil(p[:, 0].max())))
            ymin, ymax = max(0, int(np.floor(p[:, 1].min()))), min(h - 1, int(np.ceil(p[:, 1].max())))
            if xmin > xmax or ymin > ymax or np.any(p[:, 2] <= 0):
                continue
            yy, xx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
            den = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
            if abs(den) < 1e-8:
                continue
            b0 = ((p[1, 1] - p[2, 1]) * (xx - p[2, 0]) + (p[2, 0] - p[1, 0]) * (yy - p[2, 1])) / den
            b1 = ((p[2, 1] - p[0, 1]) * (xx - p[2, 0]) + (p[0, 0] - p[2, 0]) * (yy - p[2, 1])) / den
            b2 = 1.0 - b0 - b1
            inside = (b0 >= -1e-5) & (b1 >= -1e-5) & (b2 >= -1e-5)
            invz = b0 / p[0, 2] + b1 / p[1, 2] + b2 / p[2, 2]
            z = 1.0 / np.maximum(invz, 1e-8)
            region = depth[ymin:ymax + 1, xmin:xmax + 1]
            mask = inside & (z < region)
            if not mask.any():
                continue
            weights = np.stack([b0 / p[0, 2], b1 / p[1, 2], b2 / p[2, 2]], -1) * z[..., None]
            face_n = _normalize(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]))
            face_t = _normalize(xyz[1] - xyz[0])
            face_b = _normalize(np.cross(face_n, face_t))
            region[mask] = z[mask]
            world[ymin:ymax + 1, xmin:xmax + 1][mask] = (weights @ xyz)[mask]
            uv_buffer[ymin:ymax + 1, xmin:xmax + 1][mask] = (weights @ tuv)[mask]
            normal[ymin:ymax + 1, xmin:xmax + 1][mask] = face_n
            tangent[ymin:ymax + 1, xmin:xmax + 1][mask] = face_t
            bitangent[ymin:ymax + 1, xmin:xmax + 1][mask] = face_b
            covered[ymin:ymax + 1, xmin:xmax + 1][mask] = True

    idx = np.where(covered)
    uv, pos = uv_buffer[idx], world[idx]
    uv_right = np.roll(uv_buffer, -1, axis=1)
    uv_down = np.roll(uv_buffer, -1, axis=0)
    uv_dx_full = np.mod(uv_right - uv_buffer + 0.5, 1.0) - 0.5
    uv_dy_full = np.mod(uv_down - uv_buffer + 0.5, 1.0) - 0.5
    # Do not derive LOD across geometry silhouettes or the image wrap edge.
    valid_x = covered & np.roll(covered, -1, axis=1)
    valid_y = covered & np.roll(covered, -1, axis=0)
    valid_x[:, -1] = False
    valid_y[-1, :] = False
    uv_dx_full[~valid_x] = 0.0
    uv_dy_full[~valid_y] = 0.0
    uv_dx, uv_dy = uv_dx_full[idx], uv_dy_full[idx]
    albedo = _sample(
        maps.get("diffuse", np.full((3, 1, 1), 0.65, np.float32)), uv, uv_dx, uv_dy, mip_lod_bias
    )
    if albedo.shape[1] == 1:
        albedo = np.repeat(albedo, 3, axis=1)
    albedo = np.clip(albedo, 0, 1) ** 2.2
    rough = np.clip(_sample(
        maps.get("roughness", np.full((1, 1, 1), 0.5, np.float32)), uv, uv_dx, uv_dy, mip_lod_bias
    ), 0.045, 1)
    ao = np.clip(_sample(
        maps.get("occlusion", np.ones((1, 1, 1), np.float32)), uv, uv_dx, uv_dy, mip_lod_bias
    ), 0, 1)
    metal = np.clip(_sample(
        maps.get("metallic", np.zeros((1, 1, 1), np.float32)), uv, uv_dx, uv_dy, mip_lod_bias
    ), 0, 1)
    spec = np.clip(_sample(
        maps.get("specular", np.full((1, 1, 1), 0.5, np.float32)), uv, uv_dx, uv_dy, mip_lod_bias
    ), 0, 1)
    n = normal[idx]
    if "normal" in maps:
        nts = _sample(maps["normal"], uv, uv_dx, uv_dy, mip_lod_bias) * 2.0 - 1.0
        n = _normalize(tangent[idx] * nts[:, 0:1] + bitangent[idx] * nts[:, 1:2] + n * nts[:, 2:3])
    v = _normalize(eye - pos)
    f0 = (0.02 + 0.14 * spec) * (1 - metal) + albedo * metal
    result = albedo * (0.025 + 0.12 * ao)
    for light in point_lights:
        intensity = float(light.get("intensity", 0.0))
        if intensity <= 0.0:
            continue
        delta = np.asarray(light["position"], np.float32) - pos
        light_color = np.asarray(light.get("color", [1.0, 1.0, 1.0]), np.float32)[None, :]
        dist2 = np.maximum((delta * delta).sum(1, keepdims=True), 0.08)
        l = _normalize(delta)
        hvec = _normalize(v + l)
        ndotl = np.clip((n * l).sum(1, keepdims=True), 0, 1)
        ndotv = np.clip((n * v).sum(1, keepdims=True), 1e-4, 1)
        ndoth = np.clip((n * hvec).sum(1, keepdims=True), 0, 1)
        vdoth = np.clip((v * hvec).sum(1, keepdims=True), 0, 1)
        alpha = rough * rough
        d = alpha * alpha / (np.pi * np.maximum((ndoth * ndoth * (alpha * alpha - 1) + 1) ** 2, 1e-6))
        k = (rough + 1) ** 2 / 8
        g = ndotv / (ndotv * (1 - k) + k) * ndotl / (ndotl * (1 - k) + k)
        fresnel = f0 + (1 - f0) * (1 - vdoth) ** 5
        specular = d * g * fresnel / np.maximum(4 * ndotv * ndotl, 1e-5)
        diffuse = (1 - fresnel) * (1 - metal) * albedo / np.pi
        result += (diffuse + specular) * ndotl * light_color * (intensity / dist2)

    # Sun-like directional light entering through the open front of the box.
    # Its visibility term is a real mesh occlusion query, so the objects cast
    # consistent shadows in every comparison panel.
    directional_intensity = float(lighting.get("directional_intensity", 2.4))
    if bool(lighting.get("directional_enable", True)) and directional_intensity > 0.0:
        directional_l = _normalize(np.asarray(
            lighting.get("directional_direction", [-0.45, 0.62, -0.72]), np.float32
        ))
        directional_color = np.asarray(
            lighting.get("directional_color", [1.0, 1.0, 1.0]), np.float32
        )[None, :]
        visibility = _directional_shadow_visibility(pos, n, directional_l)
        l = np.broadcast_to(directional_l, pos.shape)
        hvec = _normalize(v + l)
        ndotl = np.clip((n * l).sum(1, keepdims=True), 0, 1)
        ndotv = np.clip((n * v).sum(1, keepdims=True), 1e-4, 1)
        ndoth = np.clip((n * hvec).sum(1, keepdims=True), 0, 1)
        vdoth = np.clip((v * hvec).sum(1, keepdims=True), 0, 1)
        alpha = rough * rough
        d = alpha * alpha / (np.pi * np.maximum((ndoth * ndoth * (alpha * alpha - 1) + 1) ** 2, 1e-6))
        k = (rough + 1) ** 2 / 8
        g = ndotv / (ndotv * (1 - k) + k) * ndotl / (ndotl * (1 - k) + k)
        fresnel = f0 + (1 - f0) * (1 - vdoth) ** 5
        specular = d * g * fresnel / np.maximum(4 * ndotv * ndotl, 1e-5)
        diffuse = (1 - fresnel) * (1 - metal) * albedo / np.pi
        result += ((diffuse + specular) * ndotl * visibility
                   * directional_color * directional_intensity)
    color[idx] = result
    color = np.clip(color / (1 + color), 0, 1) ** (1 / 2.2)
    # Keep metrics in unquantized float sRGB. Conversion to 8-bit happens only
    # when the comparison image is written.
    return color.astype(np.float32)


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


def _crop_render_triplet(fntc: np.ndarray, astc: np.ndarray, gt: np.ndarray):
    """Remove the deterministic clear-color border using one shared GT crop."""
    clear = gt[0, 0]
    mask = np.max(np.abs(gt - clear[None, None, :]), axis=2) > (2.0 / 255.0)
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return fntc, astc, gt
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    return fntc[y0:y1, x0:x1], astc[y0:y1, x0:x1], gt[y0:y1, x0:x1]


def _resize_render_triplet_height(fntc: np.ndarray, astc: np.ndarray, gt: np.ndarray,
                                  target_height: int):
    """Resize a shared crop to the configured final height without changing aspect ratio."""
    source_h, source_w = gt.shape[:2]
    if source_h == target_height:
        return fntc, astc, gt
    target_w = max(1, int(round(source_w * target_height / float(source_h))))

    def resize(image):
        tensor = torch.from_numpy(image).permute(2, 0, 1)[None]
        resized = F.interpolate(tensor, size=(target_height, target_w), mode="area")
        return resized[0].permute(1, 2, 0).numpy()

    return resize(fntc), resize(astc), resize(gt)


@torch.no_grad()
def save_pbr_comparison(fntc_astc_texture: torch.Tensor, traditional_astc_texture: torch.Tensor,
                        gt_texture: torch.Tensor, dataset, output_root: str,
                        curr_iter: int = None, resolution: int = 512,
                        astc_block: str = "6x6", ssim_metric=None,
                        lpips_metric=None, mip_lod_bias: float = -1.0,
                        lighting=None
                        ) -> Tuple[str, Dict[str, Tuple[float, float, float]]]:
    """Render FNTC-ASTC, traditional ASTC, and GT texture sets in one scene."""
    available = getattr(dataset, "available_textures", None)
    # The Cornell box occupies about 56% of the square camera raster vertically.
    # Rasterize above the requested size so `resolution` describes the final,
    # cropped scene height rather than the temporary square framebuffer.
    raster_resolution = int(math.ceil(int(resolution) / 0.56))
    fntc_astc = render_pbr_scene(
        fntc_astc_texture, dataset.canonical_channel_slices, dataset.normal_encoding,
        raster_resolution, available, mip_lod_bias, lighting
    )
    traditional_astc = render_pbr_scene(
        traditional_astc_texture, dataset.canonical_channel_slices, dataset.normal_encoding,
        raster_resolution, available, mip_lod_bias, lighting
    )
    gt = render_pbr_scene(
        gt_texture, dataset.canonical_channel_slices, dataset.normal_encoding,
        raster_resolution, available, mip_lod_bias, lighting
    )
    fntc_astc, traditional_astc, gt = _crop_render_triplet(fntc_astc, traditional_astc, gt)
    fntc_astc, traditional_astc, gt = _resize_render_triplet_height(
        fntc_astc, traditional_astc, gt, int(resolution)
    )
    if ssim_metric is None or lpips_metric is None:
        raise ValueError("ssim_metric and lpips_metric are required for PBR comparison metrics")
    metrics = {
        "fntc": _render_metrics(fntc_astc, gt, gt_texture.device, ssim_metric, lpips_metric),
        "astc": _render_metrics(traditional_astc, gt, gt_texture.device, ssim_metric, lpips_metric),
    }
    panel_h, panel_w = gt.shape[:2]
    canvas = Image.new("RGB", (panel_w * 3, panel_h), (20, 20, 20))
    to_image = lambda image: Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8))
    canvas.paste(to_image(fntc_astc), (0, 0))
    canvas.paste(to_image(traditional_astc), (panel_w, 0))
    canvas.paste(to_image(gt), (panel_w * 2, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_overlay_font(panel_h)
    block = str(astc_block).upper()
    fp, fs, fl = metrics["fntc"]
    ap, ass, al = metrics["astc"]
    labels = (
        f"FNTC{block}\nPSNR {fp:.3f}  SSIM {fs:.4f}  LPIPS {fl:.4f}",
        f"ASTC{block}\nPSNR {ap:.3f}  SSIM {ass:.4f}  LPIPS {al:.4f}",
        "GT\nReference",
    )
    for panel_idx, label in enumerate(labels):
        x = panel_idx * panel_w
        bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=3)
        pad_x, pad_y = 12, 10
        box_w, box_h = bbox[2] - bbox[0] + pad_x * 2, bbox[3] - bbox[1] + pad_y * 2
        draw.rounded_rectangle((x + 10, 10, x + 10 + box_w, 10 + box_h), radius=6, fill=(0, 0, 0, 180))
        draw.multiline_text((x + 10 + pad_x, 10 + pad_y), label,
                            fill=(245, 245, 245), font=font, spacing=5)
    out_dir = os.path.join(output_root, "Compare_PBRScene")
    os.makedirs(out_dir, exist_ok=True)
    filename = "comparison.png" if curr_iter is None else f"{curr_iter}.png"
    path = os.path.join(out_dir, filename)
    canvas.save(path)
    return path, metrics
