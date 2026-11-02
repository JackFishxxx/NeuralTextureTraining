"""Material raster reconstruction and shared SR sources for training and evaluation."""
import copy
import math
import torch
from Core.Feature_Grid.sampling import resize_bilinear_repeat, tile_regions
from .normal_encoding import decode_normal, normal_to_rgb
from Core.Feature_Grid.compression import apply_latent_compression
from .compression import MaterialCompression


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
        inp = torch.stack(((cc + 0.5) / W, (rr + 0.5) / H,
                           torch.zeros_like(rr)), dim=-1).reshape(-1, 3)
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


@torch.no_grad()
def cached_astc_base(dataset, codec, base_resolution, output_size):
    """Compress the shared SR source mip once, independently of material output formats."""
    formats = {group: 'ASTC'+codec.astc_block for group in dataset.texture_packing}
    key = (tuple((id(t),t._version) for t in dataset.mip_cache),tuple(output_size),
           tuple((g,tuple(names)) for g,names in dataset.texture_packing.items()),dataset.normal_encoding,
           codec.astcenc_path,codec.astcenc_quality,codec.astc_block,base_resolution)
    previous = getattr(dataset,'_sr_base_cache',None)
    if previous is None or previous[0] != key:
        ratio = max(dataset.texture_height,dataset.texture_width)/base_resolution
        mip = min(max(0,round(math.log2(ratio))),dataset.num_mips-1)
        source = canonical_gt_mip(dataset,mip).float()
        compression = MaterialCompression(formats,codec.astcenc_path,codec.astcenc_quality)
        image = compression.roundtrip(source,dataset)
        if tuple(image.shape[-2:]) != tuple(output_size):
            image = resize_bilinear_repeat(image,output_size)
        # Training and evaluation share this immutable GPU raster.
        dataset._sr_base_cache = (key,image)
    return dataset._sr_base_cache[1].to(dataset.mip_cache[0].device)


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
