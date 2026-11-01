"""Material raster reconstruction and shared SR sources for training and evaluation."""
import copy
import math
from typing import Optional, Tuple
import torch
from dataset import material_size, clean_base_image
from Core.Feature_Grid.sampling import resize_bilinear_repeat, sample_image
from Core.Material_Encoding.normal_encoding import decode_normal
from Core.Feature_Grid.compression import apply_latent_compression
from Core.Material_Encoding.compression import MaterialCompression


@torch.no_grad()
def _render_model_mip0(model, dataset, texture_height: int, texture_width: int, num_mips: int, device: str,
                       superres_base_override: Optional[torch.Tensor] = None,
                       inference_tile: int = 512,
                       inference_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Render Mip0 in tiles, bounding decoder and feature-gradient temporaries."""
    mip = 0
    H = texture_height >> mip
    W = texture_width >> mip
    target_size = (H, W)
    H, W = inference_size if inference_size is not None else material_size(model, H, W)
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
    if model.material_cnn is not None:
        image = model.material_cnn(image)
    return resize_bilinear_repeat(image, target_size) if (H, W) != target_size else image


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
        vis_mode: 'srgb' | 'normal' | 'linear'
    """
    image = torch.clamp(image, 0.0, 1.0)
    if vis_mode == 'srgb':
        image = torch.pow(image, 1.0 / 2.2)
    elif vis_mode == 'normal':
        n = image * 2.0 - 1.0
        norm = torch.sqrt(torch.clamp((n ** 2).sum(dim=0, keepdim=True), min=1e-8))
        n = n / norm
        image = torch.clamp((n + 1.0) * 0.5, 0.0, 1.0)
    elif vis_mode == 'normal_encoded':
        image = decode_normal(image, normal_encoding)
    # 'linear' -> no transform
    return image


def canonical_gt_mip(dataset, mip):
    level = dataset.mip_cache[mip]
    return dataset.expand_to_canonical(level).permute(2, 0, 1)[None]
