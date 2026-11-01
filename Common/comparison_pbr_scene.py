"""Deterministic Cornell-box PBR comparison renderer.

The renderer is intentionally self-contained: it uses NumPy for software
rasterization and Pillow for output, so enabling comparisons does not add a
runtime dependency to the training environment.
"""

import os
import math
import json
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from Core.Feature_Grid.compression import compressed_method
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from Core.Material_Encoding.normal_encoding import decode_normal, encode_normal, normal_vectors
from Common.comparison_astc import (comparison_label_color, method_label, metric_records, write_metrics,
                             cached_astc_base, comparison_reference, log_comparison_storage)
from Common.material_reconstruction import comparison_material_size, make_deployment_model, _render_model_mip0


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


def _perspective_uv_derivatives(projected, uv, interpolated_uv, invz, denominator):
    """Differentiate this triangle's perspective UV mapping in screen pixels.

    Using the local triangle avoids sampling across occluders, UV seams or
    framebuffer edges. Unwrapped UVs retain the correct repeated-texture scale.
    """
    x, y = projected[:, 0], projected[:, 1]
    dbdx = np.asarray([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / denominator
    dbdy = np.asarray([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / denominator
    reciprocal_z = 1.0 / projected[:, 2]
    uv_over_z = uv * reciprocal_z[:, None]
    return tuple((db @ uv_over_z - interpolated_uv * (db @ reciprocal_z)) / invz[:, None]
                 for db in (dbdx, dbdy))


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


def _mip_pyramid(texture: np.ndarray, normal_encoding=None):
    levels = [texture]
    while max(levels[-1].shape[-2:]) > 1:
        src = levels[-1]
        if normal_encoding is not None:
            src = decode_normal(torch.from_numpy(np.ascontiguousarray(src)).float(), normal_encoding).numpy()
        h, w = src.shape[-2:]
        if h & 1:
            src = np.concatenate([src, src[:, -1:, :]], axis=1)
        if w & 1:
            src = np.concatenate([src, src[:, :, -1:]], axis=2)
        filtered = 0.25 * (src[:, 0::2, 0::2] + src[:, 1::2, 0::2]
                           + src[:, 0::2, 1::2] + src[:, 1::2, 1::2])
        if normal_encoding is not None:
            # Vector-prefilter each mip, then store the original encoding just
            # like TextureDataset. Bilinear/trilinear lookups filter that encoding.
            filtered = encode_normal(torch.from_numpy(filtered), normal_encoding).numpy()
        levels.append(filtered)
    return levels


def _sample(texture: np.ndarray, uv: np.ndarray, uv_dx=None, uv_dy=None,
            mip_lod_bias: float = 0.0, normal_encoding=None) -> np.ndarray:
    """Bilinear magnification and trilinear mip-filtered minification."""
    if uv_dx is None or uv_dy is None:
        return _sample_level(texture, uv)
    levels = _mip_pyramid(texture, normal_encoding)
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


def _sample_normal(texture, uv, uv_dx=None, uv_dy=None, mip_lod_bias=0., normal_encoding='xyz'):
    encoded = _sample(texture, uv, uv_dx, uv_dy, mip_lod_bias, normal_encoding)
    return normal_vectors(torch.from_numpy(encoded), normal_encoding).numpy()


def _material_maps(texture: torch.Tensor, slices: Dict[str, tuple], normal_encoding: str,
                   available_textures=None) -> Dict[str, np.ndarray]:
    tex = torch.nan_to_num(texture[0].detach().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    maps = {}
    active = set(slices) if available_textures is None else set(available_textures)
    for name, (start, end) in slices.items():
        if name not in active:
            continue
        maps[name] = tex[start:end].cpu().numpy()
    return maps


def render_pbr_scene(texture: torch.Tensor, slices: Dict[str, tuple], normal_encoding: str,
                     resolution: int = 512, available_textures=None,
                     mip_lod_bias: float = -1.0, lighting=None,
                     view_center=(0.5, 0.5), view_zoom: float = 1.0) -> np.ndarray:
    """Rasterize float sRGB; an off-axis viewport re-renders a magnified scene region."""
    if (not math.isfinite(view_zoom) or view_zoom < 1 or len(view_center) != 2
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in view_center)):
        raise ValueError('View zoom must be finite and >= 1; center must be two coordinates in [0,1]')
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
    uv_dx_buffer = np.zeros((h, w, 2), np.float32)
    uv_dy_buffer = np.zeros((h, w, 2), np.float32)
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
        if view_zoom == 1.0 and tuple(view_center) == (0.5, 0.5):
            return np.stack([(x * 0.5 + 0.5) * (w - 1), (0.5 - y * 0.5) * (h - 1), z], axis=1)
        screen_x = ((x * 0.5 + 0.5 - view_center[0]) * view_zoom + 0.5) * (w - 1)
        screen_y = ((0.5 - y * 0.5 - view_center[1]) * view_zoom + 0.5) * (h - 1)
        return np.stack([screen_x, screen_y, z], axis=1)

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
            face_n, face_t, face_b = _triangle_tangent_frame(xyz, tuv)
            local_uv = (weights @ tuv)[mask]
            uv_dx, uv_dy = _perspective_uv_derivatives(p, tuv, local_uv, invz[mask], den)
            region[mask] = z[mask]
            world[ymin:ymax + 1, xmin:xmax + 1][mask] = (weights @ xyz)[mask]
            uv_buffer[ymin:ymax + 1, xmin:xmax + 1][mask] = local_uv
            uv_dx_buffer[ymin:ymax + 1, xmin:xmax + 1][mask] = uv_dx
            uv_dy_buffer[ymin:ymax + 1, xmin:xmax + 1][mask] = uv_dy
            normal[ymin:ymax + 1, xmin:xmax + 1][mask] = face_n
            tangent[ymin:ymax + 1, xmin:xmax + 1][mask] = face_t
            bitangent[ymin:ymax + 1, xmin:xmax + 1][mask] = face_b
            covered[ymin:ymax + 1, xmin:xmax + 1][mask] = True

    idx = np.where(covered)
    uv, pos = uv_buffer[idx], world[idx]
    uv_dx, uv_dy = uv_dx_buffer[idx], uv_dy_buffer[idx]
    albedo = _sample(
        maps.get("diffuse", np.full((3, 1, 1), 0.65, np.float32)), uv, uv_dx, uv_dy, mip_lod_bias
    )
    if albedo.shape[1] == 1:
        albedo = np.repeat(albedo, 3, axis=1)
    albedo = np.clip(albedo, 0, 1)
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
        nts = _sample_normal(maps["normal"], uv, uv_dx, uv_dy, mip_lod_bias, normal_encoding)
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


def _render_crop_bounds(gt):
    clear = gt[0, 0]
    mask = np.max(np.abs(gt - clear[None, None, :]), axis=2) > (2.0 / 255.0)
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return 0, gt.shape[0], 0, gt.shape[1]
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    return y0, y1, x0, x1


def _crop_render_triplet(fntc: np.ndarray, astc: np.ndarray, gt: np.ndarray):
    """Remove the deterministic clear-color border using one shared GT crop."""
    y0, y1, x0, x1 = _render_crop_bounds(gt)
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


def _annotated_render(image, name, metric=None):
    panel = Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)).convert('RGBA')
    overlay = Image.new('RGBA', panel.size)
    draw = ImageDraw.Draw(overlay)
    label = (f'{name}\nPSNR {metric[0]:.3f}  SSIM {metric[1]:.4f}  LPIPS {metric[2]:.4f}'
             if metric is not None else 'GT\nReference: PSNR inf  SSIM 1.0000  LPIPS 0.0000')
    font = _load_overlay_font(panel.height)
    # Keep annotations inside small diagnostic renders as well as 1K outputs.
    while draw.multiline_textbbox((0, 0), label, font=font, spacing=3)[2] > panel.width - 44 and getattr(font, 'size', 10) > 10:
        font = ImageFont.truetype(font.path, font.size - 1)
    bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=3)
    draw.rounded_rectangle((10, 10, 34 + bbox[2], 30 + bbox[3]), radius=6,
                           fill=comparison_label_color(name))
    draw.multiline_text((23, 21), label, fill=(0, 0, 0, 220), font=font, spacing=3)
    draw.multiline_text((22, 20), label, fill=(255, 255, 255), font=font, spacing=3)
    return Image.alpha_composite(panel, overlay).convert('RGB')


def _save_render_panels(images, metrics, block, directory, comparison_path, prefix='', fntc_label=None):
    os.makedirs(directory, exist_ok=True)
    names = (fntc_label or f'FNTC_{block}', f'ASTC_{block}', 'GT')
    panels = [_annotated_render(image, name, metric) for image, name, metric in
              zip(images, names, (metrics['fntc'], metrics['astc'], None))]
    canvas = Image.new('RGB', (panels[0].width * 3, panels[0].height))
    for index, (panel, name) in enumerate(zip(panels, names)):
        panel.save(os.path.join(directory, prefix + name + '.png'))
        canvas.paste(panel, (index * panel.width, 0))
    canvas.save(comparison_path)


@torch.no_grad()
def save_pbr_comparison(fntc_astc_texture: torch.Tensor, traditional_astc_texture: torch.Tensor,
                        gt_texture: torch.Tensor, dataset, output_root: str,
                        curr_iter: int = None, resolution: int = 512,
                        astc_block: str = "6x6", ssim_metric=None,
                        lpips_metric=None, mip_lod_bias: float = -1.0,
                        lighting=None, detail_center=(0.5, 0.62), detail_zoom: float = 4.0,
                        reference_cache=None,
                        report_context=None,
                        detail_resolution: int = 2048,
                        save_outputs=True,
                        include_detail_metrics=False,
                        ) -> Tuple[str, Dict[str, Tuple[float, float, float]]]:
    """Render FNTC-ASTC, traditional ASTC, and GT texture sets in one scene."""
    fntc_label = (report_context or {}).get('fntc_label', f'FNTC_{astc_block}')
    available = getattr(dataset, "available_textures", None)
    if detail_resolution < 64:
        raise ValueError('PBR detail resolution must be at least 64')
    if (not math.isfinite(detail_zoom) or detail_zoom <= 1 or len(detail_center) != 2
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in detail_center)):
        raise ValueError('Detail zoom must be finite and > 1; center must be two coordinates in [0,1]')
    if ssim_metric is None or lpips_metric is None:
        raise ValueError("ssim_metric and lpips_metric are required for PBR comparison metrics")
    # The Cornell box occupies about 56% of the square camera raster vertically.
    # Rasterize above the requested size so `resolution` describes the final,
    # cropped scene height rather than the temporary square framebuffer.
    raster_resolution = int(math.ceil(int(resolution) / 0.56))
    cache = {} if reference_cache is None else reference_cache
    sources = (gt_texture, traditional_astc_texture)
    key = (tuple((id(t), t._version, tuple(t.shape)) for t in sources),
           tuple((name, tuple(bounds)) for name, bounds in dataset.canonical_channel_slices.items()),
           dataset.normal_encoding, None if available is None else tuple(available),
           resolution, detail_resolution, mip_lod_bias, json.dumps(lighting, sort_keys=True), tuple(detail_center), detail_zoom)
    if cache.get('key') != key:
        cache.clear()
        cache.update(key=key, sources=sources, metrics={})
    fntc_astc = render_pbr_scene(
        fntc_astc_texture, dataset.canonical_channel_slices, dataset.normal_encoding,
        raster_resolution, available, mip_lod_bias, lighting
    )
    if 'full' not in cache:
        traditional_astc, gt = [render_pbr_scene(
            texture, dataset.canonical_channel_slices, dataset.normal_encoding,
            raster_resolution, available, mip_lod_bias, lighting
        ) for texture in (traditional_astc_texture, gt_texture)]
        cache['crop'] = _render_crop_bounds(gt)
        _, traditional_astc, gt = _resize_render_triplet_height(
            *_crop_render_triplet(traditional_astc, traditional_astc, gt), int(resolution))
        cache['full'] = (traditional_astc, gt)
    traditional_astc, gt = cache['full']
    y0, y1, x0, x1 = cache['crop']
    # The baseline crop is fixed by GT, and only the current FNTC image is resized.
    crop = fntc_astc[y0:y1, x0:x1]
    if crop.shape[0] != int(resolution):
        h, w = crop.shape[:2]
        tensor = torch.from_numpy(crop).permute(2, 0, 1)[None]
        crop = F.interpolate(tensor, size=(int(resolution), max(1, round(w * resolution / h))),
                             mode='area')[0].permute(1, 2, 0).numpy()
    fntc_astc = crop
    metric_key = (id(ssim_metric), id(lpips_metric), str(fntc_astc_texture.device))
    fixed_metrics = cache['metrics'].setdefault(metric_key, {})
    if 'full' not in fixed_metrics:
        fixed_metrics['full'] = _render_metrics(traditional_astc, gt, fntc_astc_texture.device, ssim_metric, lpips_metric)
    metrics = {
        "fntc": _render_metrics(fntc_astc, gt, fntc_astc_texture.device, ssim_metric, lpips_metric),
        "astc": fixed_metrics['full'],
    }
    if not save_outputs and not include_detail_metrics:
        return None, metrics
    out_dir = os.path.join(output_root, "Compare_PBRScene")
    if curr_iter is not None:
        out_dir = os.path.join(out_dir, f'iter_{curr_iter:08d}')
    path = os.path.join(out_dir, 'comparison_strip.png')
    if save_outputs:
        os.makedirs(out_dir, exist_ok=True)
        _save_render_panels((fntc_astc, traditional_astc, gt), metrics, astc_block,
                            out_dir, path, prefix='Scene_', fntc_label=fntc_label)

    # A fresh narrow viewport changes rasterization and UV derivatives/LOD.
    # These are new samples and shading evaluations, never resized full-frame crops.
    detail_fntc = render_pbr_scene(fntc_astc_texture, dataset.canonical_channel_slices, dataset.normal_encoding,
                                 int(detail_resolution), available, mip_lod_bias, lighting,
                                 view_center=detail_center, view_zoom=detail_zoom)
    if 'detail' not in cache:
        cache['detail'] = [render_pbr_scene(texture, dataset.canonical_channel_slices, dataset.normal_encoding,
                                          int(detail_resolution), available, mip_lod_bias, lighting,
                                          view_center=detail_center, view_zoom=detail_zoom)
                           for texture in (traditional_astc_texture, gt_texture)]
    detail = [detail_fntc, *cache['detail']]
    if 'detail' not in fixed_metrics:
        fixed_metrics['detail'] = _render_metrics(detail[1], detail[2], fntc_astc_texture.device, ssim_metric, lpips_metric)
    detail_metrics = {
        'fntc': _render_metrics(detail[0], detail[2], fntc_astc_texture.device, ssim_metric, lpips_metric),
        'astc': fixed_metrics['detail'],
    }
    if not save_outputs:
        return None, dict(metrics, fntc_detail=detail_metrics['fntc'])
    _save_render_panels(detail, detail_metrics, astc_block, out_dir,
                        os.path.join(out_dir, 'comparison_strip_detail.png'), prefix='detail_', fntc_label=fntc_label)
    context = report_context or {}
    records = [row for view, scores in (('Scene', metrics), ('Scene Detail', detail_metrics))
               for method, values in scores.items()
               for row in metric_records({'image': values}, (fntc_label if method == 'fntc' else method_label(method, astc_block)), view=view,
                                         storage_bytes=context.get('storage_bytes', {}).get((fntc_label if method == 'fntc' else method_label(method, astc_block))))]
    write_metrics(out_dir, records, evaluation='pbr', iteration=curr_iter,
                  metadata=dict(context, astc_block=astc_block, normal_encoding=dataset.normal_encoding,
                                render_sizes={'Scene': list(gt.shape[:2]), 'Scene Detail': list(detail[2].shape[:2])},
                                mip_lod_bias=mip_lod_bias, lighting=lighting,
                                detail_view={'center': list(detail_center), 'zoom': detail_zoom,
                                             'resolution': int(detail_resolution), 'rendered_separately': True}))
    labels = (fntc_label, f'ASTC_{astc_block.lower()}')
    label_width = max(map(len, labels))
    for view, scores, image_path in (
            ('Scene', metrics, path),
            ('Scene Detail', detail_metrics, os.path.join(out_dir, 'comparison_strip_detail.png'))):
        for method, label in zip(('fntc', 'astc'), labels):
            psnr, ssim, lpips = scores[method]
            suffix = f'  output={image_path}' if method == 'astc' else ''
            print(f'[PBR Test] {view} Iter {curr_iter}: {label:<{label_width}} '
                  f'PSNR={psnr:8.4f}  SSIM={ssim:7.4f}  LPIPS={lpips:7.4f}{suffix}')
    return path, dict(metrics, fntc_detail=detail_metrics['fntc']) if include_detail_metrics else metrics


@torch.no_grad()
def run_pbr_comparison_pipeline(model, dataset, codec, config, output_root, *, curr_iter=None,
                                ssim_metric=None, lpips_metric=None, lighting=None,
                                report_context=None, save_outputs=True, include_detail_metrics=False,
                                material_override=None):
    """Reconstruct deployment materials and produce the scene comparison."""
    height, width = dataset.texture_height, dataset.texture_width
    ref_h, ref_w = comparison_material_size(model, height, width, config.ref_astc_resolution)
    reference = comparison_reference(dataset, codec, ref_h, ref_w)
    prediction = material_override
    if prediction is None:
        decoded = make_deployment_model(model)
        base = (cached_astc_base(dataset, codec, config.super_resolution_base_resolution, (ref_h, ref_w))
                if model.super_resolution_enable else None)
        prediction = _render_model_mip0(
            decoded, dataset, height, width, dataset.num_mips, config.device,
            superres_base_override=base, inference_tile=config.eval_inference_tile,
            inference_size=(ref_h, ref_w))
        del decoded
    elif prediction.shape != reference['gt'].shape:
        raise ValueError('PBR material override must match the canonical GT raster')
    storage = log_comparison_storage(model, dataset, codec.astc_block, config.ref_astc_resolution, verbose=False)
    fntc_label = method_label(compressed_method(model))
    context = dict(report_context or {}, fntc_label=fntc_label, gt_size=[height, width], inference_size=[ref_h, ref_w],
                   storage_bytes={fntc_label: storage[compressed_method(model)],
                                  method_label('astc', codec.astc_block): storage[f'ref_astc_{codec.astc_block}'],
                                  'GT': storage['gt']})
    return save_pbr_comparison(
        fntc_astc_texture=prediction, traditional_astc_texture=reference['astc'], gt_texture=reference['gt'],
        dataset=dataset, output_root=output_root, curr_iter=curr_iter,
        resolution=config.pbr_render_resolution, detail_resolution=config.pbr_detail_resolution,
        astc_block=codec.astc_block, ssim_metric=ssim_metric, lpips_metric=lpips_metric,
        mip_lod_bias=config.pbr_mip_lod_bias, lighting=lighting,
        detail_center=config.pbr_detail_center, detail_zoom=config.pbr_detail_zoom,
        reference_cache=reference['pbr'], report_context=context, save_outputs=save_outputs,
        include_detail_metrics=include_detail_metrics)
