"""Conventional BC1/3/4/5/7 codecs and DDS containers; no training dependency."""
import math
import struct
import numpy as np
from .formats import TextureFormat

_DXGI_FORMATS = {'BC1': 71, 'BC3': 77, 'BC4': 80, 'BC5': 83, 'BC7': 98}


def dds_header(width, height, format='BC1'):
    fmt = TextureFormat(format)
    if fmt.name not in _DXGI_FORMATS or width < 1 or height < 1:
        raise ValueError('Expected a BC format and positive texture dimensions')
    fourcc = 0x31545844 if fmt.name == 'BC1' else 0x30315844
    header = (b'DDS ' + struct.pack('<7I',124,0x81007,height,width,fmt.payload_bytes(height,width),0,1)
              + bytes(44) + struct.pack('<8I',32,4,fourcc,0,0,0,0,0)
              + struct.pack('<5I',0x1000,0,0,0,0))
    if fmt.name != 'BC1':
        header += struct.pack('<5I',_DXGI_FORMATS[fmt.name],3,0,1,0)
    return header


def _bc1_palette(a, b):
    def rgb565(values):
        r,g,blue = values>>11,(values>>5)&63,values&31
        return np.stack(((r<<3)|(r>>2),(g<<2)|(g>>4),(blue<<3)|(blue>>2)),-1)
    x,y = rgb565(a),rgb565(b)
    palette = np.stack((x,y,(2*x+y)//3,(x+2*y)//3),1)
    punchthrough = a <= b
    palette[punchthrough,2] = ((x+y)//2)[punchthrough]
    palette[punchthrough,3] = 0
    return palette


def decode_blocks(data, width, height, rgba=False):
    bw, bh = math.ceil(width / 4), math.ceil(height / 4)
    if len(data) != bw * bh * 8:
        raise ValueError("BC1 payload length does not match image dimensions")
    words = np.frombuffer(data, dtype=np.dtype([("a", "<u2"), ("b", "<u2"), ("bits", "<u4")]))
    a, b = words["a"].astype(np.int32), words["b"].astype(np.int32)

    palette = _bc1_palette(a,b)
    punchthrough = a <= b
    if rgba:
        alpha = np.full((*palette.shape[:2], 1), 255, dtype=np.int32)
        alpha[punchthrough, 3] = 0
        palette = np.concatenate((palette, alpha), -1)
    selectors = (words["bits"][:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3
    pixels = palette[np.arange(len(words))[:, None], selectors]
    channels = 4 if rgba else 3
    image = (
        pixels.reshape(bh, bw, 4, 4, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(bh * 4, bw * 4, channels)
    )
    return image[:height, :width].astype(np.uint8)


def _encode_bc1_blocks(rgba):
    h, w, _ = rgba.shape
    bh, bw = math.ceil(h / 4), math.ceil(w / 4)
    padded = np.pad(rgba, ((0, bh * 4 - h), (0, bw * 4 - w), (0, 0)), mode="edge")
    pixels = padded.reshape(bh, 4, bw, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4)
    rgb = pixels[..., :3].astype(np.float32)
    opaque = pixels[..., 3] >= 128
    # Weighted PCA ignores transparent colors when fitting visible endpoints.
    count = opaque.sum(1, keepdims=True).clip(min=1)
    center = (rgb * opaque[..., None]).sum(1) / count
    delta = (rgb - center[:, None]) * opaque[..., None]
    covariance = delta.transpose(0, 2, 1) @ delta
    _, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, :, -1]
    score = (rgb * direction[:, None]).sum(-1)
    high = np.where(opaque, score, -np.inf).argmax(1)
    low = np.where(opaque, score, np.inf).argmin(1)
    rows = np.arange(len(pixels))
    colors = np.stack((rgb[rows, high], rgb[rows, low]), 1)
    codes = np.rint(colors * np.array([31, 63, 31]) / 255).astype(np.int32)
    words = (codes[..., 0] << 11) | (codes[..., 1] << 5) | codes[..., 2]
    lo, hi = words.min(1), words.max(1)
    punchthrough = ~opaque.all(1)
    hi = np.where((hi == lo) & ~punchthrough, np.minimum(hi + 1, 65535), hi)
    lo = np.where((hi == lo) & ~punchthrough, np.maximum(lo - 1, 0), lo)
    a = np.where(punchthrough, lo, hi)
    b = np.where(punchthrough, hi, lo)

    palette = _bc1_palette(a,b)
    distance = ((rgb[:, :, None] - palette[:, None]) ** 2).sum(-1)
    distance[punchthrough, :, 3] = np.inf
    selectors = distance.argmin(-1).astype(np.uint32)
    selectors[~opaque] = 3
    bits = (
        (selectors << (2 * np.arange(16, dtype=np.uint32)))
        .sum(1, dtype=np.uint64)
        .astype(np.uint32)
    )
    blocks = np.empty(
        len(pixels), dtype=np.dtype([("a", "<u2"), ("b", "<u2"), ("bits", "<u4")])
    )
    blocks["a"], blocks["b"], blocks["bits"] = a, b, bits
    return blocks.tobytes()


class BCCodec:
    """Encode/decode DDS, or raw blocks when a caller already owns the container."""

    def __init__(self, format='BC1'):
        self.format = TextureFormat(format)
        if self.format.name not in _DXGI_FORMATS:
            raise ValueError('BCCodec requires BC1/3/4/5/7')

    def encode_blocks(self, rgba):
        rgba = np.asarray(rgba, dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[-1] != 4 or min(rgba.shape[:2]) < 1:
            raise ValueError('Expected nonempty HWC RGBA image')
        if self.format.name == 'BC1':
            return _encode_bc1_blocks(rgba)
        try:
            import ispc_texcomp as encoder
        except ImportError as exc:
            raise RuntimeError('Install the BC encoder: pip install ispc_texcomp==1.0.1') from exc
        h, w = rgba.shape[:2]
        padded = np.pad(rgba, ((0, -h % 4), (0, -w % 4), (0, 0)), mode='edge')
        ph, pw = padded.shape[:2]
        # ISPC BC4/5 kernels consume packed R/RG, while BC3/7 consume RGBA.
        channels = self.format.channels if self.format.name in ('BC4', 'BC5') else 4
        surface = encoder.RGBASurface(np.ascontiguousarray(padded[..., :channels]).tobytes(),pw,ph,pw*channels)
        if self.format.name == 'BC7':
            return encoder.compress_blocks_bc7(surface,encoder.BC7EncSettings.from_profile('alpha_basic'))
        return getattr(encoder,'compress_blocks_'+self.format.name.lower())(surface)

    def decode_blocks(self, blocks, width, height):
        if width < 1 or height < 1 or len(blocks) != self.format.payload_bytes(height,width):
            raise ValueError('BC payload does not match dimensions')
        if self.format.name == 'BC1':
            return decode_blocks(blocks,width,height,rgba=True)
        try:
            import texture2ddecoder as decoder
        except ImportError as exc:
            raise RuntimeError('Install the BC decoder: pip install texture2ddecoder==1.0.6') from exc
        data = getattr(decoder,'decode_'+self.format.name.lower())(blocks,width,height)
        # The independent decoder returns BGRA; the codec API always returns RGBA.
        return np.frombuffer(data,np.uint8).reshape(height,width,4)[..., [2,1,0,3]].copy()

    def encode_rgba(self, rgba):
        rgba = np.asarray(rgba,dtype=np.uint8)
        blocks = self.encode_blocks(rgba)
        height,width = rgba.shape[:2]
        return dds_header(width,height,self.format.name)+blocks

    def decode_bytes(self, data):
        if len(data) < 128 or data[:4] != b'DDS ' or struct.unpack_from('<I',data,4)[0] != 124:
            raise ValueError('Expected a DDS texture')
        height,width = struct.unpack_from('<II',data,12)
        header = dds_header(width,height,self.format.name)
        if data[84:88] != header[84:88] or (self.format.name != 'BC1' and data[128:148] != header[128:148]):
            raise ValueError('DDS format does not match codec')
        return self.decode_blocks(data[len(header):],width,height)

    def roundtrip_rgba(self, rgba):
        return self.decode_bytes(self.encode_rgba(rgba))
