"""One per-material compression recipe for baseline and FNTC tile output."""
import numpy as np
import torch
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from torch.nn import functional as F
from Common.Block_Compression.astc import ASTCCodec
from Common.Block_Compression.bc import BCCodec, dds_header
from Common.Block_Compression.formats import TextureFormat
from .normal_encoding import decode_normal, encode_normal, filter_normal
from Core.Feature_Grid.sampling import tile_regions

GROUP_NAMES = {'diffuse': 'Diffuse', 'normal': 'Normal', 'romsd': 'Romsd'}
DEFAULT_FORMATS = {'diffuse': 'BC7', 'normal': 'BC5', 'romsd': 'BC1'}


@lru_cache(maxsize=1)
def _physical_core_count():
    """Query physical cores without assuming that every CPU uses two-way SMT."""
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            query = ctypes.WinDLL('kernel32', use_last_error=True).GetLogicalProcessorInformationEx
            query.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))
            query.restype = wintypes.BOOL
            size = wintypes.DWORD()
            query(0, None, ctypes.byref(size))  # RelationProcessorCore
            buffer = ctypes.create_string_buffer(size.value)
            if not query(0, buffer, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            offset, cores = 0, 0
            while offset < size.value:
                relationship, length = np.frombuffer(buffer, np.uint32, count=2, offset=offset)
                if length < 8:
                    raise ValueError('Invalid processor topology record')
                cores += int(relationship == 0)
                offset += int(length)
            return max(1, cores)
        if sys.platform == 'darwin':
            import subprocess
            return max(1, int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu'])))
        cpus = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else range(os.cpu_count() or 1)
        topology = Path('/sys/devices/system/cpu')
        return max(1, len({((topology/f'cpu{cpu}/topology/physical_package_id').read_text(),
                            (topology/f'cpu{cpu}/topology/core_id').read_text()) for cpu in cpus}))
    except (OSError, ValueError, AttributeError):
        return 1


@lru_cache(maxsize=1)
def _tile_encoder_pool():
    # Shared across materials/runs; the executor starts only as many threads as
    # there are queued tiles and joins them at interpreter shutdown.
    return ThreadPoolExecutor(max_workers=_physical_core_count(), thread_name_prefix='material-codec')


def parse_material_formats(formats):
    if not isinstance(formats, dict) or set(formats) - GROUP_NAMES.keys():
        raise ValueError('material_compression_formats must map diffuse, normal, romsd to formats')
    if any(not isinstance(value,str) for value in formats.values()):
        raise ValueError('Material compression formats must be strings such as BC7 or ASTC6x6')
    return {group: TextureFormat(formats.get(group, default))
            for group, default in DEFAULT_FORMATS.items()}


class MaterialCompression:
    def __init__(self, formats, astcenc_path, astcenc_quality, *, tile_size=None):
        self.formats = parse_material_formats(formats)
        if tile_size is not None and (not isinstance(tile_size,int) or isinstance(tile_size,bool) or tile_size < 1):
            raise ValueError('Material compression tile size must be a positive integer')
        self.tile_size = tile_size
        self.codecs = {group: (ASTCCodec(astcenc_path, astcenc_quality, fmt.name[4:])
                              if fmt.name.startswith('ASTC') else BCCodec(fmt.name))
                       for group, fmt in self.formats.items()}

    def recipe(self, dataset):
        return {group: self.formats[group].name for group in GROUP_NAMES if group in dataset.texture_packing}

    def payload_bytes(self, dataset, size):
        return sum(self.formats[group].payload_bytes(*size) for group in dataset.texture_packing)

    def tile_layout(self, dataset, size):
        regions = list(tile_regions(*size,self.tile_size or max(size)))
        layout = []
        for group in dataset.texture_packing:
            fmt = self.formats[group]
            for y,x,h,w in regions:
                layout.append(dict(group=group,origin=[y,x],size=[h,w],format=fmt.name))
        return layout

    def validate_dataset(self, dataset):
        for group,names in dataset.texture_packing.items():
            channels = sum(b-a for name in names for a,b in [dataset.canonical_channel_slices[name]])
            fmt = self.formats[group]
            if channels > fmt.channels and not (group == 'normal' and fmt.name == 'BC5' and channels == 3):
                raise ValueError(f'{GROUP_NAMES[group]} has {channels} channels but {fmt.name} stores {fmt.channels} continuous channels')

    def _encode_tiled(self, rgba, group):
        """Assemble tile blocks; re-encode when tile boundaries cut codec blocks."""
        codec,fmt = self.codecs[group],self.formats[group]
        h,w = rgba.shape[:2]
        if self.tile_size is None or max(h,w) <= self.tile_size:
            return codec.encode_rgba(rgba)
        bw,bh = fmt.block
        aligned = ((h <= self.tile_size or self.tile_size % bh == 0)
                   and (w <= self.tile_size or self.tile_size % bw == 0))
        blocks = np.empty((math.ceil(h/bh),math.ceil(w/bw),fmt.block_bytes),np.uint8) if aligned else None
        decoded = np.empty_like(rgba) if not aligned else None
        header = dds_header(w,h,fmt.name) if isinstance(codec,BCCodec) else None
        offset = len(header) if header is not None else 16
        regions = list(tile_regions(h,w,self.tile_size))
        def encode(region):
            y,x,th,tw = region
            return codec.encode_rgba(rgba[y:y+th,x:x+tw])
        parallel = fmt.name in ('BC1', 'BC7') and len(regions) > 1 and _physical_core_count() > 1
        encoded = _tile_encoder_pool().map(encode, regions) if parallel else map(encode, regions)
        for (y,x,th,tw),data in zip(regions,encoded):
            if aligned:
                rows,cols = math.ceil(th/bh),math.ceil(tw/bw)
                blocks[y//bh:y//bh+rows,x//bw:x//bw+cols] = np.frombuffer(data[offset:],np.uint8).reshape(rows,cols,fmt.block_bytes)
            else:
                decoded[y:y+th,x:x+tw] = codec.decode_bytes(data)
        if not aligned:
            # A BC/ASTC block cannot represent two separately padded tile blocks.
            return codec.encode_rgba(decoded)
        if header is None:
            header = data[:7]+w.to_bytes(3,'little')+h.to_bytes(3,'little')+data[13:16]
        return header+blocks.tobytes()

    @torch.no_grad()
    def compress(self, image, dataset):
        self.validate_dataset(dataset)
        if not torch.isfinite(image).all():
            raise ValueError('Cannot compress nonfinite material values')
        result = image.clone()
        textures = {}
        for group, names in dataset.texture_packing.items():
            indices = [i for name in names for i in range(*dataset.canonical_channel_slices[name])]
            source = image[0, indices]
            fmt = self.formats[group]
            normal_rgb = group == 'normal' and fmt.name == 'BC5' and len(indices) == 3
            if normal_rgb:
                source = decode_normal(source, dataset.normal_encoding)[:2]
            h, w = source.shape[-2:]
            rgba = np.full((h, w, 4), 255, np.uint8)
            rgba[..., :len(source)] = (source.clamp(0, 1)*255).round().byte().permute(1, 2, 0).cpu().numpy()
            data = self._encode_tiled(rgba,group)
            extension = '.dds' if isinstance(self.codecs[group],BCCodec) else '.astc'
            textures[GROUP_NAMES[group]+extension] = data
            decoded = self.codecs[group].decode_bytes(data)
            restored = torch.from_numpy(decoded[..., :len(source)].copy()).permute(2, 0, 1).to(image).div_(255)
            if normal_rgb:
                xy = restored*2-1
                z = (1-xy.square().sum(0, keepdim=True)).clamp_min(0).sqrt()
                restored = encode_normal(torch.cat((restored, (z+1)/2), 0), dataset.normal_encoding)
            result[0, indices] = restored
        return result.clamp(0, 1), textures

    def roundtrip(self, image, dataset):
        return self.compress(image,dataset)[0]

    def export_materials(self, textures, dataset, size, directory, *, method):
        """Write the same compressed mip chains consumed by PBR evaluation."""
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        self.prepare_textures(textures,dataset,size)
        for filename,data in textures.items():
            source = Path(filename)
            group,separator,mip = source.stem.partition('_mip')
            name = f'{group}_{method}_{self.formats[group.lower()].name}'
            if separator:
                name += f'_mip{mip}'
            if source.suffix == '.dds':
                name = 'DDS_'+name
            (directory/(name+source.suffix)).write_bytes(data)

    def prepare_textures(self, textures, dataset, size):
        """Complete full-image chains once, shared by export and PBR sampling."""
        result = []
        for group in dataset.texture_packing:
            codec = self.codecs[group]
            filename = GROUP_NAMES[group]+('.dds' if isinstance(codec,BCCodec) else '.astc')
            data = textures[filename]
            encoding = None
            if group == 'normal':
                encoding = ('xy' if self.formats['normal'].name == 'BC5' and dataset.normal_encoding == 'xyz'
                            else dataset.normal_encoding)
            is_bc = isinstance(codec,BCCodec)
            levels = codec.decode_mipmaps(data) if is_bc else [codec.decode_bytes(data)]
            if tuple(levels[0].shape[:2]) != tuple(size):
                raise ValueError('Compressed material dimensions do not match inference size')
            if is_bc:
                if len(levels) == 1 and max(size) > 1:
                    data = codec.with_mipmaps(data,self._material_mipmaps(levels[0],encoding))
                    textures[filename] = data
                    levels = codec.decode_mipmaps(data)
            else:
                mip_files = [GROUP_NAMES[group]+f'_mip{mip}.astc'
                             for mip in range(1,max(size).bit_length())]
                if any(name not in textures for name in mip_files):
                    for name,rgba in zip(mip_files,self._material_mipmaps(levels[0],encoding)):
                        if name not in textures:
                            textures[name] = self._encode_tiled(rgba,group)
                levels.extend(codec.decode_bytes(textures[name]) for name in mip_files)
            if tuple(levels[-1].shape[:2]) != (1,1):
                raise ValueError('Compressed material requires a complete mip chain')
            result.append((group,levels))
        return result

    @staticmethod
    def _material_mipmaps(rgba, normal_encoding):
        level = torch.from_numpy(rgba.copy()).permute(2,0,1)[None].float()/255
        channels = 3 if normal_encoding == 'xyz' else 2
        while max(level.shape[-2:]) > 1:
            size = tuple(max(1,n//2) for n in level.shape[-2:])
            if normal_encoding is None:
                level = F.interpolate(level,size=size,mode='area')
            else:
                normal = filter_normal(level[:,:channels],size,normal_encoding)
                level = F.interpolate(level,size=size,mode='area')
                level[:,:channels] = normal
            yield (level[0].clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
