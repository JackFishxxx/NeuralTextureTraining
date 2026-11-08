"""Headless Direct3D 12 rasterization with physical material mip chains."""
import ctypes as ct
import functools
import hashlib
import math
import os
from pathlib import Path
import subprocess
import threading

import numpy as np
import torch

from Common.native_build import prepare_msvc
from Common.Material.normal_encoding import filter_normal
from .on_sample import MAX_GRIDS, neural_header, protocol_defines

CAMERA_EYE = (0., .55, -4.25)
CAMERA_TARGET = (0., -.05, 1.15)
CAMERA_FOV = 50.


class _Light(ct.Structure):
    _fields_ = [('position', ct.c_float * 3), ('power', ct.c_float),
                ('color', ct.c_float * 3), ('padding', ct.c_float)]


class _View(ct.Structure):
    _fields_ = [('width', ct.c_uint), ('height', ct.c_uint),
                ('center', ct.c_float * 2), ('zoom', ct.c_float),
                ('mip_bias', ct.c_float), ('normal_mode', ct.c_uint),
                ('eye', ct.c_float * 3), ('target', ct.c_float * 3), ('fov', ct.c_float),
                ('sun', _Light)]


class _Mip(ct.Structure):
    _fields_ = [('data', ct.c_void_p), ('width', ct.c_uint), ('height', ct.c_uint)]


class _Texture(ct.Structure):
    _fields_ = [('levels', ct.POINTER(_Mip)), ('count', ct.c_uint), ('channels', ct.c_uint)]


class _Geometry(ct.Structure):
    _fields_ = [('data', ct.c_void_p), ('count', ct.c_uint)]


@functools.lru_cache(maxsize=1)
def _library():
    if os.name != 'nt':
        raise RuntimeError('PBR comparison uses DirectX 12 and requires Windows; disable enable_pbr_compare on other platforms')
    from filelock import FileLock
    native = Path(__file__).with_name('native')
    source = (native / 'pbr_dx12.cpp').read_bytes()
    # Compile both native languages against the same protocol used by Python.
    definitions = {f'NEURAL_{name}': value for name, value in protocol_defines().items()}
    prefix = ''.join(f'#define {name} {value}\n' for name, value in definitions.items()).encode()
    shader = prefix + (native / 'pbr.hlsl').read_bytes()
    digest = hashlib.sha256(source + shader).hexdigest()[:16]
    cache = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.cache'))) / 'NeuralTextureTraining' / 'dx12' / digest
    cache.mkdir(parents=True, exist_ok=True)
    dll = cache / 'pbr_dx12.dll'
    with FileLock(str(cache / 'build.lock')):
        if not dll.exists():
            prepare_msvc()
            result = subprocess.run(['cl', '/nologo', '/LD', '/O2', '/EHsc', '/std:c++17', '/DNOMINMAX',
                                     *(f'/D{name}={value}' for name, value in definitions.items()),
                                     str(native / 'pbr_dx12.cpp'), '/Fe:' + str(dll),
                                     '/Fo:' + str(cache / 'pbr_dx12.obj'), '/link',
                                     'd3d12.lib', 'dxgi.lib', 'd3dcompiler.lib'],
                                    cwd=cache, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('DirectX 12 renderer build failed:\n' + result.stdout + result.stderr)
    lib = ct.CDLL(str(dll))
    lib.pbr_create.argtypes = [ct.c_void_p, ct.c_uint, ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t]
    lib.pbr_create.restype = ct.c_void_p
    lib.pbr_destroy.argtypes = [ct.c_void_p]
    lib.pbr_render.argtypes = [ct.c_void_p, ct.POINTER(_Texture), ct.POINTER(_View),
                               ct.POINTER(_Light), ct.c_uint, ct.POINTER(_Geometry),
                               ct.c_void_p, ct.c_uint,
                               ct.c_void_p, ct.c_char_p, ct.c_size_t]
    lib.pbr_render.restype = ct.c_int
    lib.pbr_adapter.argtypes = [ct.c_void_p]
    lib.pbr_adapter.restype = ct.c_char_p
    lib.pbr_shadow_size.argtypes = []
    lib.pbr_shadow_size.restype = ct.c_uint
    return lib, shader


def _vertices(meshes, spheres, *, subdivide=False):
    """Position, smooth normal, UV tangent/handedness, UV; 48-byte vertices."""
    from .pbr_scene import _triangle_tangent_frame
    result = []
    for vertices, uvs, triangles in meshes:
        for ids in triangles:
            xyz, uv = vertices[list(ids)], uvs[list(ids)]
            n, t, b = _triangle_tangent_frame(xyz, uv)
            frame = np.r_[n, t, np.sign(np.dot(np.cross(n, t), b))]
            triangle = np.c_[xyz, np.broadcast_to(frame, (3, 7)), uv]
            if subdivide:
                # Refine coarse room/box faces; spheres already have a dense mesh.
                steps = 32  # Matching edge divisions allow adjacent faces to stay welded.
                y, x = np.indices((steps, steps))
                first = np.stack((x, y), -1)[x+y < steps]
                second = np.stack((x, y), -1)[x+y < steps-1]
                corners = np.concatenate((first[:, None] + [[0, 0], [1, 0], [0, 1]],
                                          second[:, None] + [[1, 0], [1, 1], [0, 1]])).reshape(-1, 2) / steps
                weights = np.c_[1-corners.sum(1), corners]
                triangle = weights @ triangle
            result.append(triangle)
    longitude, latitude = 128, 64
    u, v = np.meshgrid(np.linspace(0, 1, longitude + 1), np.linspace(0, 1, latitude + 1))
    theta, phi = (u - .5) * (2 * np.pi), v * np.pi
    normals = np.stack([np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)], -1)
    tangent = np.stack([-np.sin(theta), np.zeros_like(theta), np.cos(theta), np.ones_like(theta)], -1)
    a = np.arange(latitude)[:, None] * (longitude + 1) + np.arange(longitude)
    b, c, d = a + 1, a + longitude + 1, a + longitude + 2
    indices = np.stack([a, c, b, b, c, d], -1).reshape(latitude, longitude, 2, 3)
    # Omit the degenerate triangle at each duplicated pole.
    indices = np.concatenate([indices[0, :, 1], indices[1:-1].reshape(-1, 3), indices[-1, :, 0]])
    for center, radius in spheres:
        vertices = np.concatenate([normals * radius + center, normals, tangent, np.stack([u, v], -1)], -1)
        result.append(vertices.reshape(-1, 12)[indices.reshape(-1)])
    return np.ascontiguousarray(np.concatenate(result) if result else np.empty((0, 12)), dtype=np.float32)


def _displaced_vertices(vertices, height, scale):
    """Bilinear-repeat height displacement and its geometric surface gradient."""
    from .pbr_scene import _normalize
    result = vertices.copy()
    uv, normal = vertices[:, 10:12], vertices[:, 3:6]
    h, w = height.shape
    def sample(coords):
        xy = coords * [w, h] - .5
        base = np.floor(xy).astype(np.int64)
        t = xy - base
        x, y = base[:, 0] % w, base[:, 1] % h
        return ((height[y, x]*(1-t[:, 0]) + height[y, (x+1) % w]*t[:, 0])*(1-t[:, 1])
                + (height[(y+1) % h, x]*(1-t[:, 0]) + height[(y+1) % h, (x+1) % w]*t[:, 0])*t[:, 1])
    result[:, :3] += normal * ((sample(uv)-.5)*scale)[:, None]
    # UV seams/hard normals duplicate vertices. Average their displaced
    # positions so box corners, room edges and sphere seams remain closed.
    _, inverse, counts = np.unique(np.round(vertices[:, :3], 6), axis=0, return_inverse=True, return_counts=True)
    positions = np.zeros((len(counts), 3), np.float64)
    np.add.at(positions, inverse, result[:, :3])
    result[:, :3] = positions[inverse] / counts[inverse, None]
    triangle = vertices.reshape(-1, 3, 12)
    edges = triangle[:, 1:, :3] - triangle[:, :1, :3]
    duv = triangle[:, 1:, 10:12] - triangle[:, :1, 10:12]
    determinant = duv[:, 0, 0]*duv[:, 1, 1] - duv[:, 0, 1]*duv[:, 1, 0]
    dpdu = (edges[:, 0]*duv[:, 1, 1:2] - edges[:, 1]*duv[:, 0, 1:2]) / determinant[:, None]
    dpdv = (edges[:, 1]*duv[:, 0, :1] - edges[:, 0]*duv[:, 1, :1]) / determinant[:, None]
    dpdu, dpdv = np.repeat(dpdu, 3, axis=0), np.repeat(dpdv, 3, axis=0)
    dual_u, dual_v = np.cross(dpdv, normal), np.cross(normal, dpdu)
    denom = (dpdu*dual_u).sum(1, keepdims=True)
    valid = np.abs(denom) > 1e-10
    dual_u = np.divide(dual_u, denom, out=np.zeros_like(dual_u), where=valid)
    dual_v = np.divide(dual_v, denom, out=np.zeros_like(dual_v), where=valid)
    dhdu = (sample(uv + [1/w, 0]) - sample(uv - [1/w, 0])) * (w/2)
    dhdv = (sample(uv + [0, 1/h]) - sample(uv - [0, 1/h])) * (h/2)
    normal = _normalize(normal - scale*(dhdu[:, None]*dual_u + dhdv[:, None]*dual_v))
    result[:, 3:6] = normal
    tangent = vertices[:, 6:9]
    result[:, 6:9] = _normalize(tangent - normal*(normal*tangent).sum(1, keepdims=True))
    return result


def _mips(source, encoding=None):
    levels = [source]
    while max(levels[-1].shape[-2:]) > 1:
        h, w = levels[-1].shape[-2:]
        size = (max(1, h // 2), max(1, w // 2))
        levels.append(filter_normal(levels[-1], size, encoding) if encoding else
                      torch.nn.functional.adaptive_avg_pool2d(levels[-1], size))
    return levels


def _latent_material(material):
    """Bind decoded latent values with their independent mip levels intact."""
    arrays, mip_arrays, descriptions = [], [], []
    for slot in range(MAX_GRIDS):
        levels = material.levels[slot] if slot < len(material.levels) else (torch.zeros(1, 1, 1),)
        packed = torch.cat([level.detach().float().reshape(-1) for level in levels]).cpu().numpy()
        offset, mips = 0, []
        for level in levels:
            count = level.numel()
            value = packed[offset:offset + count].reshape(tuple(level.shape))
            array = np.ascontiguousarray(value.transpose(1, 2, 0))
            arrays.append(array)
            mips.append(_Mip(array.ctypes.data, array.shape[1], array.shape[0]))
            offset += count
        chain = (_Mip * len(mips))(*mips)
        mip_arrays.append(chain)
        descriptions.append(_Texture(chain, len(mips), levels[0].shape[0]))
    mode = {'xyz': 0, 'xy': 1, 'hemi_oct': 2}[material.normal_encoding]
    return (_Texture * MAX_GRIDS)(*descriptions), arrays, mip_arrays, mode, None


def _material(texture, slices, encoding, available_textures, texture_maps):
    names = ('diffuse', 'normal', 'roughness', 'occlusion', 'metallic', 'specular', 'displacement')
    defaults = ((.65, .65, .65), (.5, .5, 1.), (.5,), (1.,), (0.,), (.5,), (.5,))
    active = set(slices) if available_textures is None else set(available_textures)
    source = None if texture_maps is not None else torch.nan_to_num(texture[0].detach().float(), nan=0., posinf=1., neginf=0.).clamp(0, 1)
    arrays, mip_arrays, descriptions, chains = [], [], [], []
    device_levels = {}
    normal_mode = 'xyz'
    for name, default in zip(names, defaults):
        compressed = texture_maps.get(name) if texture_maps is not None else None
        if compressed is not None:
            levels = compressed.levels
            mode = compressed.normal_encoding if name == 'normal' else None
        elif source is not None and name in active and name in slices:
            start, end = slices[name]
            mode = encoding if name == 'normal' else None
            levels = [source[start:end]] if name == 'displacement' else _mips(source[start:end], mode)
        else:
            mode = 'xyz' if name == 'normal' else None
            levels = [np.asarray(default, np.float32)[:, None, None]]
        if name == 'normal':
            normal_mode = mode or encoding
        chain = []
        for level in (levels[:1] if name == 'displacement' else levels):
            # Dataset loading already linearizes albedo, including the values
            # compressed into exported mip chains. Never decode gamma again.
            if isinstance(level, torch.Tensor):
                if name == 'diffuse' and level.shape[0] == 1:
                    level = level.expand(3, -1, -1)
                if level.shape[0] == 3:
                    level = torch.cat((level, torch.ones_like(level[:1])), 0)
                level = level.permute(1, 2, 0).contiguous().float()
                if level.is_cuda:
                    # One transfer for every mip/channel on this device. Split
                    # host views retain the packed allocation through NumPy.
                    device_levels.setdefault(level.device, []).append((len(arrays), level))
                    array = None
                else:
                    array = level.numpy()
            else:
                value = np.asarray(level)
                if name == 'diffuse' and value.shape[0] == 1:
                    value = np.repeat(value, 3, axis=0)
                if value.shape[0] == 3:
                    value = np.concatenate([value, np.ones_like(value[:1])], 0)
                array = np.ascontiguousarray(value.transpose(1, 2, 0), dtype=np.float32)
            chain.append(len(arrays))
            arrays.append(array)
        chains.append(chain)
    for levels in device_levels.values():
        packed = torch.cat([level.reshape(-1) for _, level in levels]).cpu().numpy()
        offset = 0
        for index, level in levels:
            arrays[index] = packed[offset:offset + level.numel()].reshape(tuple(level.shape))
            offset += level.numel()
    for chain in chains:
        mip_descriptions = [_Mip(arrays[i].ctypes.data, arrays[i].shape[1], arrays[i].shape[0]) for i in chain]
        mips = (_Mip * len(mip_descriptions))(*mip_descriptions)
        mip_arrays.append(mips)
        descriptions.append(_Texture(mips, len(mips), arrays[chain[0]].shape[-1]))
    has_height = ('displacement' in texture_maps if texture_maps is not None
                  else 'displacement' in active and 'displacement' in slices)
    displacement = arrays[chains[-1][0]][..., 0] if has_height else None
    return ((_Texture * 6)(*descriptions[:6]), arrays, mip_arrays,
            ('xyz', 'xy', 'hemi_oct').index(normal_mode), displacement)


class DX12Scene:
    def __init__(self, meshes, spheres):
        self.lib, _ = _library()
        self.meshes, self.spheres = meshes, spheres
        self.handle = None
        self.lock = threading.Lock()
        self._refined_vertices = None
        self._open()

    def _open(self):
        vertices = _vertices(self.meshes, self.spheres)
        _, shader = _library()
        error = ct.create_string_buffer(4096)
        self.handle = self.lib.pbr_create(vertices.ctypes.data, len(vertices), shader, len(shader), error, len(error))
        if not self.handle:
            raise RuntimeError(error.value.decode('utf-8', errors='replace'))
        self.adapter = self.lib.pbr_adapter(self.handle).decode('utf-8')
        self.shadow_size = self.lib.pbr_shadow_size()
        self._height_key = None

    def _close(self):
        if self.handle:
            self.lib.pbr_destroy(self.handle)
            self.handle = None

    def __del__(self):
        if getattr(self, 'handle', None):
            self._close()

    @torch.no_grad()
    def render(self, texture, slices, normal_encoding, resolution, available_textures,
               mip_lod_bias, lighting, texture_maps, view_center, view_zoom):
        return self.render_views(texture, slices, normal_encoding, available_textures,
                                 mip_lod_bias, lighting, texture_maps,
                                 [(resolution, resolution, view_center, view_zoom)])[0]

    @torch.no_grad()
    def render_views(self, texture, slices, normal_encoding, available_textures,
                     mip_lod_bias, lighting, texture_maps, views, *, on_sample=None):
        """Prepare/upload one material, then reuse it for all requested cameras."""
        for width, height, center, zoom in views:
            if any(not math.isfinite(n) or int(n) != n or not 2 <= n <= 16384 for n in (width, height)):
                raise ValueError('PBR viewport dimensions must be integers in [2,16384]')
            if (len(center) != 2 or any(not math.isfinite(n) or not 0 <= n <= 1 for n in center)
                    or not math.isfinite(zoom) or zoom < 1 or not math.isfinite(mip_lod_bias)):
                raise ValueError('PBR viewport center, zoom and mip bias must be finite and valid')
        textures, arrays, mip_arrays, encoding, height = (_latent_material(on_sample) if on_sample is not None else
            _material(texture, slices, normal_encoding, available_textures, texture_maps))
        neural = on_sample.parameters if on_sample is not None else neural_header()
        lighting = lighting or {}
        displacement_scale = float(lighting.get('displacement_scale', .03))
        if on_sample is not None and displacement_scale != 0:
            raise ValueError('on_sample PBR currently requires pbr_displacement_scale: 0')
        if not math.isfinite(displacement_scale) or displacement_scale < 0:
            raise ValueError('PBR displacement scale must be finite and nonnegative')
        height_key = (height.shape, hashlib.sha256(height.tobytes()).digest(), displacement_scale) if (
            height is not None and displacement_scale > 0 and np.any(height != .5)) else None
        point_lights = lighting.get('point_lights', [
            dict(position=[0., .82, .7], intensity=6.5, color=[1., 1., 1.]),
            dict(position=[-.8, .45, -.15], intensity=1.2, color=[1., 1., 1.])])
        lights = [_Light(tuple(light['position']), float(light['intensity']),
                         tuple(light.get('color', [1., 1., 1.])), 0.)
                  for light in point_lights if float(light.get('intensity', 0)) > 0]
        lights = (_Light * len(lights))(*lights)
        power = float(lighting.get('directional_intensity', 2.4)) if lighting.get('directional_enable', False) else 0.
        sun = _Light(tuple(lighting.get('directional_direction', [-.45, .62, -.72])), power,
                     tuple(lighting.get('directional_color', [1., 1., 1.])), 0.)
        error = ct.create_string_buffer(4096)
        outputs = []
        with self.lock:
            if not self.handle:
                self._open()
            geometry = None
            if height_key != self._height_key:
                if height_key is None:
                    geometry = _Geometry(None, 0)  # Restore the original mesh.
                else:
                    if self._refined_vertices is None:
                        self._refined_vertices = _vertices(self.meshes, self.spheres, subdivide=True)
                    displaced = _displaced_vertices(self._refined_vertices, height, displacement_scale)
                    geometry = _Geometry(displaced.ctypes.data, len(displaced))
            for index, (width, height, center, zoom) in enumerate(views):
                width, height = int(width), int(height)
                settings = _View(width, height, tuple(center), zoom, mip_lod_bias, encoding,
                                 CAMERA_EYE, CAMERA_TARGET, CAMERA_FOV, sun)
                output = np.empty((height, width, 4), np.float32)
                ok = self.lib.pbr_render(self.handle, textures if index == 0 else None,
                                         ct.byref(settings), lights, len(lights),
                                         ct.byref(geometry) if index == 0 and geometry is not None else None,
                                         neural.ctypes.data if index == 0 else None, len(neural) if index == 0 else 0,
                                         output.ctypes.data, error, len(error))
                if ok != 1:
                    if ok < 0:  # Native recording/submission failed; discard its state.
                        self._close()
                    raise RuntimeError(error.value.decode('utf-8', errors='replace'))
                self._height_key = height_key
                outputs.append(output[:, :, :3].copy())
        return outputs
