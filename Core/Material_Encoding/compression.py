"""One per-material compression recipe for baseline and FNTC tile output."""
import numpy as np
import torch
from Common.Block_Compression.astc import ASTCCodec
from Common.Block_Compression.bc import BCCodec
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
        return sum(tile['payload_bytes'] for tile in self.tile_layout(dataset,size))

    def tile_layout(self, dataset, size):
        regions = list(tile_regions(*size,self.tile_size or max(size)))
        layout = []
        for group in dataset.texture_packing:
            fmt = self.formats[group]
            extension = '.astc' if fmt.name.startswith('ASTC') else '.dds'
            for y,x,h,w in regions:
                suffix = f'_tile_{y}_{x}' if len(regions) > 1 else ''
                layout.append(dict(group=group,file=GROUP_NAMES[group]+suffix+extension,
                                   origin=[y,x],size=[h,w],format=fmt.name,
                                   payload_bytes=fmt.payload_bytes(h,w)))
        return layout

    def validate_dataset(self, dataset):
        for group,names in dataset.texture_packing.items():
            channels = sum(b-a for name in names for a,b in [dataset.canonical_channel_slices[name]])
            fmt = self.formats[group]
            if channels > fmt.channels and not (group == 'normal' and fmt.name == 'BC5' and channels == 3):
                raise ValueError(f'{GROUP_NAMES[group]} has {channels} channels but {fmt.name} stores {fmt.channels} continuous channels')

    @torch.no_grad()
    def compress(self, image, dataset):
        self.validate_dataset(dataset)
        if not torch.isfinite(image).all():
            raise ValueError('Cannot compress nonfinite material values')
        result = image.clone()
        textures = {}
        layout = self.tile_layout(dataset,image.shape[-2:])
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
            decoded = np.empty_like(rgba)
            for tile in layout:
                if tile['group'] != group:
                    continue
                y,x = tile['origin']
                th,tw = tile['size']
                data = self.codecs[group].encode_rgba(rgba[y:y+th,x:x+tw])
                decoded[y:y+th,x:x+tw] = self.codecs[group].decode_bytes(data)
                textures[tile['file']] = data
            restored = torch.from_numpy(decoded[..., :len(source)].copy()).permute(2, 0, 1).to(image).div_(255)
            if normal_rgb:
                xy = restored*2-1
                z = (1-xy.square().sum(0, keepdim=True)).clamp_min(0).sqrt()
                restored = encode_normal(torch.cat((restored, (z+1)/2), 0), dataset.normal_encoding)
            result[0, indices] = restored
        return result.clamp(0, 1), textures

    def roundtrip(self, image, dataset):
        return self.compress(image,dataset)[0]
