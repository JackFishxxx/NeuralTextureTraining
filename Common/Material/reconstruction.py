"""Material raster reconstruction and shared SR sources for training and evaluation."""
import copy
import json
from pathlib import Path
import torch
from Core.Feature_Grid.sampling import resize_bilinear_repeat, tile_regions
from .normal_encoding import decode_normal, normal_to_rgb
from Core.Feature_Grid.compression import apply_latent_compression


@torch.no_grad()
def render_material_mip0(model, size, device, *, superres_base=None, inference_tile=512):
    """Render the native material raster; SR requires a canonical CHW/NCHW base."""
    H, W = size
    if H < 1 or W < 1:
        raise ValueError('Material raster dimensions must be positive')
    sr = getattr(model, 'super_resolution_enable', False)
    if sr:
        if superres_base is None:
            raise ValueError('SR reconstruction requires a base at native material size')
        if superres_base.ndim == 4 and superres_base.shape[0] == 1:
            superres_base = superres_base[0]
        if superres_base.ndim != 3 or tuple(superres_base.shape[-2:]) != (H, W):
            raise ValueError('SR base must be a CHW image at native material size')
    step = max(1, int(inference_tile) or max(H, W))
    hwc = None
    for h0, w0, h, w in tile_regions(H, W, step):
        rr, cc = torch.meshgrid(torch.arange(h0, h0+h, device=device),
                                torch.arange(w0, w0+w, device=device), indexing='ij')
        inp = torch.stack(((cc + 0.5) / W, (rr + 0.5) / H), dim=-1).reshape(-1, 2)
        if not model.mip0_only:
            inp = torch.cat((inp, inp.new_zeros((len(inp), 1))), 1)
        if sr:
            base = superres_base[:, h0:h0+h, w0:w0+w].permute(1, 2, 0).reshape(-1, superres_base.shape[0]).float()
            y = (base + model(torch.cat([inp, base], dim=1)).float()).clamp(0.0, 1.0)
        else:
            y = model(inp).float()
        if hwc is None:
            hwc = torch.empty(H, W, y.shape[-1], device=device, dtype=torch.float32)
        hwc[h0:h0+h, w0:w0+w] = y.reshape(h, w, -1)
    if getattr(model, "direct_diffuse_enabled", False) and hasattr(model, "render_direct_diffuse_mip"):
        direct = model.render_direct_diffuse_mip(0, H, W)
        hwc[:, :, :3] = direct.squeeze(0).permute(1, 2, 0).to(hwc.dtype)
    hwc = torch.nan_to_num(hwc, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    image = hwc.permute(2, 0, 1)[None, ...]
    if model.material_cnn is not None:
        image = model.material_cnn(image)
    return image


@torch.no_grad()
def make_deployment_model(model):
    decoded = copy.deepcopy(model).eval()
    fields = decoded._decoder_adaptation_features
    if fields is not None and all(not field.requires_grad for field in fields):
        # Decoder fitting already owns immutable, deployment-quantized fields.
        for grid, field in zip(decoded.feature_grids, fields):
            grid.params.copy_(field)
        decoded._pending_grid_sync.clear()
    else:
        decoded.simulate_quantize()
        apply_latent_compression(decoded)
    decoded._decoder_adaptation_features = None
    return decoded


def sr_base_source_mip(dataset, base_resolution):
    """Return the GT mip whose actual maximum edge matches the requested base."""
    if not isinstance(base_resolution, int) or isinstance(base_resolution, bool) or base_resolution < 1:
        raise ValueError('SR base resolution must be a positive integer')
    edges = [max(level.shape[:2]) for level in dataset.mip_cache]
    if base_resolution not in edges:
        raise ValueError(f'SR base resolution {base_resolution} must match a GT mip edge; available: {edges}')
    return edges.index(base_resolution)


@torch.no_grad()
def _sr_base(dataset, compression, base_resolution):
    """Cache the actual Baseline-codec bytes and their decoded native raster."""
    recipe = compression.recipe(dataset)
    astc = tuple((group, codec.astcenc_path, codec.astcenc_quality)
                 for group, codec in compression.codecs.items()
                 if group in recipe and hasattr(codec, 'astcenc_path'))
    key = (tuple((id(t),t._version) for t in dataset.mip_cache),
           tuple((g,tuple(names)) for g,names in dataset.texture_packing.items()),dataset.normal_encoding,
           tuple(recipe.items()),compression.tile_size,astc,base_resolution)
    previous = getattr(dataset,'_sr_base_cache',None)
    if previous is None or previous[0] != key:
        mip = sr_base_source_mip(dataset, base_resolution)
        source = canonical_gt_mip(dataset,mip).float()
        image, textures = compression.compress(source,dataset)
        dataset._sr_base_cache = (key,image,textures,{})
    return dataset._sr_base_cache


def cached_sr_base(dataset, compression, base_resolution, output_size):
    """Reuse the same compressed SR base for every training and deployment path."""
    _, image, _, resized = _sr_base(dataset, compression, base_resolution)
    size = tuple(output_size)
    if size == tuple(image.shape[-2:]):
        return image
    if size not in resized:
        resized[size] = resize_bilinear_repeat(image,size)
    return resized[size]


def export_sr_base(dataset, compression, base_resolution, directory):
    """Export the low-resolution SR base as a single-level texture."""
    _, image, textures, _ = _sr_base(dataset, compression, base_resolution)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in textures.items():
        (directory / name).write_bytes(data)
    metadata = dict(formats=compression.recipe(dataset), size=list(image.shape[-2:]),
                    source_gt_mip=sr_base_source_mip(dataset,base_resolution),
                    payload_bytes=compression.payload_bytes(dataset,image.shape[-2:]),
                    file_bytes=sum(map(len,textures.values())),
                    normal_encoding=compression.normal_encoding(dataset),
                    decoded_normal_encoding=dataset.normal_encoding,
                    sampling='bilinear_repeat_to_native_material_size', mip_count=1)
    (directory / 'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def postprocess_material(image: torch.Tensor, vis_mode: str, normal_encoding='xyz') -> torch.Tensor:
    """Apply visualization post-processing based on vis_mode.
    Args:
        image: [C, H, W] tensor in [0, 1]
        vis_mode: 'srgb' | 'normal' | 'normal_encoded' | 'linear'
    """
    image = torch.clamp(image, 0.0, 1.0)
    if vis_mode == 'srgb':
        image = torch.pow(image, 1.0 / 2.2)
    elif vis_mode == 'normal':
        image = normal_to_rgb(image * 2.0 - 1.0)
    elif vis_mode == 'normal_encoded':
        image = decode_normal(image, normal_encoding)
    # 'linear' -> no transform
    return image


def canonical_gt_mip(dataset, mip):
    level = dataset.mip_cache[mip]
    return dataset.expand_to_canonical(level).permute(2, 0, 1)[None]
