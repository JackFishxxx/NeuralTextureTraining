"""One per-material compression recipe for baseline and FNTC tile output."""
import numpy as np
import torch
import math
from pathlib import Path
from torch.nn import functional as F
from Common.Block_Compression.astc import ASTCCodec
from Common.Block_Compression.bc import BCCodec, dds_header
from Common.Block_Compression.formats import TextureFormat
from .normal_encoding import decode_normal, encode_normal
from Core.Feature_Grid.sampling import tile_regions

GROUP_NAMES = {'diffuse': 'Diffuse', 'normal': 'Normal', 'romsd': 'Romsd'}
DEFAULT_FORMATS = {'diffuse': 'BC7', 'normal': 'BC5', 'romsd': 'BC1'}


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
        bw,bh = fmt.block
        blocks = np.empty((math.ceil(h/bh),math.ceil(w/bw),fmt.block_bytes),np.uint8)
        aligned = (self.tile_size is None or
                   ((h <= self.tile_size or self.tile_size % bh == 0)
                    and (w <= self.tile_size or self.tile_size % bw == 0)))
        decoded = np.empty_like(rgba) if not aligned else None
        first = None
        for y,x,th,tw in tile_regions(h,w,self.tile_size or max(h,w)):
            data = codec.encode_rgba(rgba[y:y+th,x:x+tw])
            if first is None:
                first = data
            if aligned:
                offset = (128 if fmt.name == 'BC1' else 148) if isinstance(codec,BCCodec) else 16
                rows,cols = math.ceil(th/bh),math.ceil(tw/bw)
                blocks[y//bh:y//bh+rows,x//bw:x//bw+cols] = np.frombuffer(data[offset:],np.uint8).reshape(rows,cols,fmt.block_bytes)
            else:
                decoded[y:y+th,x:x+tw] = codec.decode_bytes(data)
        if not aligned:
            # A BC/ASTC block cannot represent two separately padded tile blocks.
            return codec.encode_rgba(decoded)
        if isinstance(codec,BCCodec):
            header = dds_header(w,h,fmt.name)
        else:
            header = first[:7]+w.to_bytes(3,'little')+h.to_bytes(3,'little')+first[13:16]
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
            if isinstance(codec,BCCodec):
                levels = codec.decode_mipmaps(data)
                if len(levels) == 1 and max(size) > 1:
                    data = codec.with_mipmaps(data,self._material_mipmaps(levels[0],encoding))
                    textures[filename] = data
                    levels = codec.decode_mipmaps(data)
            else:
                levels = [codec.decode_bytes(data)]
                for mip,rgba in enumerate(self._material_mipmaps(levels[0],encoding),1):
                    mip_file = GROUP_NAMES[group]+f'_mip{mip}.astc'
                    if mip_file not in textures:
                        textures[mip_file] = self._encode_tiled(rgba,group)
                    levels.append(codec.decode_bytes(textures[mip_file]))
            if tuple(levels[0].shape[:2]) != tuple(size):
                raise ValueError('Compressed material dimensions do not match inference size')
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
                normal = decode_normal(level[:,:channels],normal_encoding)
                normal = encode_normal(F.interpolate(normal,size=size,mode='area'),normal_encoding)
                level = F.interpolate(level,size=size,mode='area')
                level[:,:channels] = normal
            yield (level[0].clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
