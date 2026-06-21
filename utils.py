import struct

import numpy as np


def write_dds_r8g8b8a8(filepath: str, width: int, height: int, data: np.ndarray):
    """
    Write a DDS file with R8G8B8A8 format.

    Args:
        filepath: Output DDS file path
        width: Texture width
        height: Texture height
        data: RGBA data as numpy array with shape [height, width, 4] and dtype uint8
    """
    # DDS header constants
    DDS_MAGIC = 0x20534444  # "DDS "
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_PITCH = 0x8
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000
    DDSD_LINEARSIZE = 0x80000
    DDSD_DEPTH = 0x800000

    DDPF_ALPHAPIXELS = 0x1
    DDPF_FOURCC = 0x4
    DDPF_RGB = 0x40

    DDSCAPS_TEXTURE = 0x1000

    # Calculate pitch (bytes per row)
    pitch = width * 4  # 4 bytes per pixel for RGBA8

    # DX10 header constants
    DX10_FOURCC = b'DX10'
    DXGI_FORMAT_R8G8B8A8_UNORM = 28  # non-SRGB
    D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3
    MISC_FLAG = 0
    ARRAY_SIZE = 1
    DXGI_SAMPLE_DESC = (1, 0)

    with open(filepath, 'wb') as f:
        # Write magic number
        f.write(struct.pack('<I', DDS_MAGIC))

        # Write DDS_HEADER
        f.write(struct.pack('<I', 124))  # dwSize
        # dwFlags
        f.write(struct.pack('<I', DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PITCH | DDSD_PIXELFORMAT))
        f.write(struct.pack('<I', height))  # dwHeight
        f.write(struct.pack('<I', width))  # dwWidth
        f.write(struct.pack('<I', pitch))  # dwPitchOrLinearSize
        f.write(struct.pack('<I', 0))  # dwDepth
        f.write(struct.pack('<I', 1))  # dwMipMapCount
        f.write(struct.pack('<I', 0) * 11)  # dwReserved1[11]

        # Write DDS_PIXELFORMAT (with DX10 FourCC)
        pf_size = 32
        pf_flags = DDPF_FOURCC
        pf_fourcc = struct.unpack('<I', DX10_FOURCC)[0]
        pf_rgb_bit_count = 0
        pf_r_bit_mask = 0
        pf_g_bit_mask = 0
        pf_b_bit_mask = 0
        pf_a_bit_mask = 0
        f.write(struct.pack('<I', pf_size))
        f.write(struct.pack('<I', pf_flags))
        f.write(struct.pack('<I', pf_fourcc))
        f.write(struct.pack('<I', pf_rgb_bit_count))
        f.write(struct.pack('<I', pf_r_bit_mask))
        f.write(struct.pack('<I', pf_g_bit_mask))
        f.write(struct.pack('<I', pf_b_bit_mask))
        f.write(struct.pack('<I', pf_a_bit_mask))

        # Write DDS_CAPS
        f.write(struct.pack('<I', DDSCAPS_TEXTURE))  # dwCaps
        f.write(struct.pack('<I', 0))  # dwCaps2
        f.write(struct.pack('<I', 0))  # dwCaps3
        f.write(struct.pack('<I', 0))  # dwCaps4
        f.write(struct.pack('<I', 0))  # dwReserved2

        # Write DDS_HEADER_DXT10
        # DXGI_FORMAT, resourceDimension, miscFlag, arraySize, miscFlags2
        f.write(struct.pack('<I', DXGI_FORMAT_R8G8B8A8_UNORM))
        f.write(struct.pack('<I', D3D10_RESOURCE_DIMENSION_TEXTURE2D))
        f.write(struct.pack('<I', MISC_FLAG))
        f.write(struct.pack('<I', ARRAY_SIZE))
        f.write(struct.pack('<I', 0))  # miscFlags2

        # Write pixel data (row by row)
        f.write(data.tobytes())
