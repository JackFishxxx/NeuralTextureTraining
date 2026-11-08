"""KTX2 containers for linear 2D ASTC UNORM mip chains (no supercompression).

Only the container changes: ASTC blocks are copied verbatim. The level index
lists Mip0 first; image payloads are stored smallest first, aligned to 16 bytes.
See https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html.
"""
import math
import struct

from .astc import ASTC_2D_BLOCKS

IDENTIFIER = b'\xabKTX 20\xbb\r\n\x1a\n'
_HEADER = struct.Struct('<12s13I2Q')
_LEVEL = struct.Struct('<3Q')
_ASTC_MAGIC = b'\x13\xab\xa1\x5c'
# Vulkan assigns consecutive UNORM/SRGB pairs starting at ASTC_4x4_UNORM=157.
_VK_FORMATS = {157 + 2 * index: block for index, block in enumerate(sorted(ASTC_2D_BLOCKS))}
_WRITER = b'KTXwriter\0NeuralTextureTraining\0'
_KVD = struct.pack('<I', len(_WRITER)) + _WRITER + bytes(-len(_WRITER) % 4)


def _dfd(block):
    # Khronos basic DFD v2: ASTC model, BT709 primaries, linear transfer,
    # straight alpha, one 128-bit sample, one 16-byte plane, UNORM [0, 2^32-1].
    width, height = block
    return struct.pack('<11I', 44, 0, 2 | (40 << 16), 162 | (1 << 8) | (1 << 16),
                       (width - 1) | ((height - 1) << 8), 16, 0, 127 << 16, 0, 0, 0xffffffff)


def header_bytes(level_count):
    """Exact container overhead; every ASTC level payload is a multiple of 16."""
    if level_count < 1:
        raise ValueError('KTX2 requires at least one mip level')
    return (_HEADER.size + _LEVEL.size * level_count + 44 + len(_KVD) + 15) // 16 * 16


def _astc_level(data):
    if len(data) < 16 or data[:4] != _ASTC_MAGIC or data[6] != 1 or data[13:16] != b'\1\0\0':
        raise ValueError('Expected a 2D ASTC image')
    block = tuple(data[4:6])
    width, height = int.from_bytes(data[7:10], 'little'), int.from_bytes(data[10:13], 'little')
    if block not in ASTC_2D_BLOCKS or min(width, height) < 1:
        raise ValueError('Invalid ASTC dimensions or block footprint')
    expected = math.ceil(width / block[0]) * math.ceil(height / block[1]) * 16
    if len(data) != 16 + expected:
        raise ValueError('ASTC payload does not match its dimensions')
    return width, height, block, data[16:]


def pack_mipmaps(streams):
    """Wrap raw .astc images, ordered largest first, without re-encoding."""
    levels = [_astc_level(data) for data in streams]
    if not levels:
        raise ValueError('KTX2 requires at least one mip level')
    width, height, block, _ = levels[0]
    if len(levels) > max(width, height).bit_length():
        raise ValueError('ASTC mip chain extends past 1x1')
    for mip, (w, h, footprint, _) in enumerate(levels):
        if (w, h, footprint) != (max(1, width >> mip), max(1, height >> mip), block):
            raise ValueError('ASTC mip levels must share a footprint and follow floor-halving')
    vk_format = next(value for value, footprint in _VK_FORMATS.items() if footprint == block)
    dfd = _dfd(block)
    dfd_offset = _HEADER.size + _LEVEL.size * len(levels)
    kvd_offset = dfd_offset + len(dfd)
    offset = header_bytes(len(levels))
    index = [None] * len(levels)
    for mip in reversed(range(len(levels))):
        size = len(levels[mip][3])
        index[mip] = _LEVEL.pack(offset, size, size)
        offset += size
    header = _HEADER.pack(IDENTIFIER, vk_format, 1, width, height, 0, 0, 1, len(levels), 0,
                          dfd_offset, len(dfd), kvd_offset, len(_KVD), 0, 0)
    metadata = header + b''.join(index) + dfd + _KVD
    return metadata + bytes(-len(metadata) % 16) + b''.join(level[3] for level in reversed(levels))


def unpack_mipmaps(data):
    """Return raw .astc images in Mip0-first order for the supported KTX2 profile."""
    if len(data) < _HEADER.size:
        raise ValueError('Truncated KTX2 header')
    (identifier, vk_format, type_size, width, height, depth, layers, faces, count, compression,
     dfd_offset, dfd_size, kvd_offset, kvd_size, sgd_offset, sgd_size) = _HEADER.unpack_from(data)
    if (identifier != IDENTIFIER or vk_format not in _VK_FORMATS or type_size != 1
            or min(width, height) < 1 or max(width, height) >= 1 << 24
            or depth or layers or faces != 1 or compression or sgd_offset or sgd_size
            or not 1 <= count <= max(width, height).bit_length()):
        raise ValueError('Expected a non-supercompressed 2D ASTC UNORM KTX2 texture')
    block = _VK_FORMATS[vk_format]
    index_end = _HEADER.size + count * _LEVEL.size
    if (dfd_offset < index_end or dfd_size != 44 or data[dfd_offset:dfd_offset + dfd_size] != _dfd(block)
            or (kvd_size and (kvd_offset < dfd_offset + dfd_size or kvd_offset + kvd_size > len(data)))):
        raise ValueError('Invalid KTX2 format descriptor or metadata offsets')
    previous_end = max(index_end, dfd_offset + dfd_size, kvd_offset + kvd_size)
    levels = [None] * count
    for mip in reversed(range(count)):
        offset, size, uncompressed = _LEVEL.unpack_from(data, _HEADER.size + mip * _LEVEL.size)
        w, h = max(1, width >> mip), max(1, height >> mip)
        expected = math.ceil(w / block[0]) * math.ceil(h / block[1]) * 16
        if offset % 16 or offset < previous_end or size != expected or uncompressed != size or offset + size > len(data):
            raise ValueError('Invalid KTX2 mip index or payload size')
        header = (_ASTC_MAGIC + bytes((*block, 1)) + w.to_bytes(3, 'little')
                  + h.to_bytes(3, 'little') + b'\1\0\0')
        levels[mip] = header + data[offset:offset + size]
        previous_end = offset + size
    return levels
