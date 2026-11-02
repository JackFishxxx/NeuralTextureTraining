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


@functools.lru_cache(maxsize=1)
def _library():
    if os.name != 'nt':
        raise RuntimeError('PBR comparison uses DirectX 12 and requires Windows; disable enable_pbr_compare on other platforms')
    from filelock import FileLock
    native = Path(__file__).with_name('native')
    source = (native / 'pbr_dx12.cpp').read_bytes()
    shader = (native / 'pbr.hlsl').read_bytes()
    digest = hashlib.sha256(source + shader).hexdigest()[:16]
    cache = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.cache'))) / 'NeuralTextureTraining' / 'dx12' / digest
    cache.mkdir(parents=True, exist_ok=True)
    dll = cache / 'pbr_dx12.dll'
    with FileLock(str(cache / 'build.lock')):
        if not dll.exists():
            prepare_msvc()
            result = subprocess.run(['cl', '/nologo', '/LD', '/O2', '/EHsc', '/std:c++17', '/DNOMINMAX',
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
                               ct.POINTER(_Light), ct.c_uint, ct.c_void_p, ct.c_char_p, ct.c_size_t]
    lib.pbr_render.restype = ct.c_int
    lib.pbr_adapter.argtypes = [ct.c_void_p]
    lib.pbr_adapter.restype = ct.c_char_p
    lib.pbr_shadow_size.argtypes = []
    lib.pbr_shadow_size.restype = ct.c_uint
    return lib, shader


def _vertices(meshes, spheres):
    """Position, smooth normal, UV tangent/handedness, UV; 48-byte vertices."""
    from .pbr_scene import _triangle_tangent_frame
    result = []
    for vertices, uvs, triangles in meshes:
        for ids in triangles:
            xyz, uv = vertices[list(ids)], uvs[list(ids)]
            n, t, b = _triangle_tangent_frame(xyz, uv)
            frame = np.r_[n, t, np.sign(np.dot(np.cross(n, t), b))]
            result.append(np.c_[xyz, np.broadcast_to(frame, (3, 7)), uv])
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


def _mips(source, encoding=None):
    levels = [source]
    while max(levels[-1].shape[-2:]) > 1:
        h, w = levels[-1].shape[-2:]
        size = (max(1, h // 2), max(1, w // 2))
        levels.append(filter_normal(levels[-1], size, encoding) if encoding else
                      torch.nn.functional.adaptive_avg_pool2d(levels[-1], size))
    return levels


def _material(texture, slices, encoding, available_textures, texture_maps):
    names = ('diffuse', 'normal', 'roughness', 'occlusion', 'metallic', 'specular')
    defaults = ((.65, .65, .65), (.5, .5, 1.), (.5,), (1.,), (0.,), (.5,))
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
            levels = _mips(source[start:end], mode)
        else:
            mode = 'xyz' if name == 'normal' else None
            levels = [np.asarray(default, np.float32)[:, None, None]]
        if name == 'normal':
            normal_mode = mode or encoding
        chain = []
        for level in levels:
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
    return (_Texture * 6)(*descriptions), arrays, mip_arrays, ('xyz', 'xy', 'hemi_oct').index(normal_mode)


class DX12Scene:
    def __init__(self, meshes, spheres):
        self.lib, _ = _library()
        self.meshes, self.spheres = meshes, spheres
        self.handle = None
        self.lock = threading.Lock()
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
                     mip_lod_bias, lighting, texture_maps, views):
        """Prepare/upload one material, then reuse it for all requested cameras."""
        for width, height, center, zoom in views:
            if any(not math.isfinite(n) or int(n) != n or not 2 <= n <= 16384 for n in (width, height)):
                raise ValueError('PBR viewport dimensions must be integers in [2,16384]')
            if (len(center) != 2 or any(not math.isfinite(n) or not 0 <= n <= 1 for n in center)
                    or not math.isfinite(zoom) or zoom < 1 or not math.isfinite(mip_lod_bias)):
                raise ValueError('PBR viewport center, zoom and mip bias must be finite and valid')
        textures, arrays, mip_arrays, encoding = _material(texture, slices, normal_encoding, available_textures, texture_maps)
        lighting = lighting or {}
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
            for index, (width, height, center, zoom) in enumerate(views):
                width, height = int(width), int(height)
                settings = _View(width, height, tuple(center), zoom, mip_lod_bias, encoding,
                                 CAMERA_EYE, CAMERA_TARGET, CAMERA_FOV, sun)
                output = np.empty((height, width, 4), np.float32)
                ok = self.lib.pbr_render(self.handle, textures if index == 0 else None,
                                         ct.byref(settings), lights, len(lights), output.ctypes.data, error, len(error))
                if ok != 1:
                    if ok < 0:  # Native recording/submission failed; discard its state.
                        self._close()
                    raise RuntimeError(error.value.decode('utf-8', errors='replace'))
                outputs.append(output[:, :, :3].copy())
        return outputs
