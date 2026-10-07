"""Build CUDA ASTC operators, reusing or installing the shared encoder source."""

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
import torch


# Core astcenc sources; paths are resolved against the shared Source directory.
ASTCENC_SOURCES = (
    "astcenc_averages_and_directions.cpp",
    "astcenc_block_sizes.cpp",
    "astcenc_color_quantize.cpp",
    "astcenc_color_unquantize.cpp",
    "astcenc_compress_symbolic.cpp",
    "astcenc_compute_variance.cpp",
    "astcenc_decompress_symbolic.cpp",
    "astcenc_diagnostic_trace.cpp",
    "astcenc_entry.cpp",
    "astcenc_find_best_partitioning.cpp",
    "astcenc_ideal_endpoints_and_weights.cpp",
    "astcenc_image.cpp",
    "astcenc_integer_sequence.cpp",
    "astcenc_mathlib.cpp",
    "astcenc_mathlib_softfloat.cpp",
    "astcenc_partition_tables.cpp",
    "astcenc_percentile_tables.cpp",
    "astcenc_pick_best_endpoint_format.cpp",
    "astcenc_quantization.cpp",
    "astcenc_symbolic_physical.cpp",
    "astcenc_weight_align.cpp",
    "astcenc_weight_quant_xfer_tables.cpp",
)


def _validate_encoder_source(source):
    required = (*ASTCENC_SOURCES, "astcenc.h", "astcenc_internal.h")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete astcenc source at {source}: {', '.join(missing)}")
    return source


def resolve_encoder_source(native_root=None):
    """Resolve the embedded source list against the shared encoder checkout.

    Experiment snapshots reuse a parent project's checkout. A clean project
    installs the same pinned release as the CPU executable into the established
    tools/astc_encoder_source directory, never into Core or a second clone.
    """
    from Comparison_ASTC import ASTCENC_DEFAULT_VERSION

    native_root = Path(native_root or Path(__file__).resolve().parent / "native").resolve()
    for parent in native_root.parents:
        source = parent / "tools/astc_encoder_source/Source"
        if source.is_dir():
            return _validate_encoder_source(source)

    # native -> ASTC_Aware -> Core -> project, independent of the working directory.
    project = next(
        (parent for parent in native_root.parents if (parent / "tools").is_dir()),
        native_root.parents[2],
    )
    install_root = project / "tools/astc_encoder_source"
    install_root.parent.mkdir(parents=True, exist_ok=True)
    version = ASTCENC_DEFAULT_VERSION
    url = f"https://codeload.github.com/ARM-software/astc-encoder/zip/refs/tags/{version}"
    with tempfile.TemporaryDirectory(prefix="astc_source_", dir=install_root.parent) as directory:
        staging = Path(directory) / "checkout"
        archive = Path(directory) / "source.zip"
        try:
            print(f"[ASTC] Downloading encoder source {version} to {install_root}", flush=True)
            urllib.request.urlretrieve(url, archive)
            prefix = f"astc-encoder-{version}/"
            with zipfile.ZipFile(archive) as package:
                for member in package.infolist():
                    if member.is_dir() or not member.filename.startswith(prefix):
                        continue
                    relative = Path(member.filename[len(prefix):])
                    # Only the flat Source directory and its upstream license are needed.
                    source_file = len(relative.parts) == 2 and relative.parts[0] == "Source"
                    if not source_file and relative != Path("LICENSE.txt"):
                        continue
                    if ".." in relative.parts or relative.is_absolute():
                        continue
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(member) as content, target.open("wb") as output:
                        shutil.copyfileobj(content, output)
            _validate_encoder_source(staging / "Source")
            if not (staging / "LICENSE.txt").is_file():
                raise FileNotFoundError("astcenc source archive has no upstream LICENSE.txt")
            # Publish only a complete checkout; preserve any existing user directory.
            if install_root.exists():
                return _validate_encoder_source(install_root / "Source")
            staging.rename(install_root)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to install astcenc {version} source from {url}: {exc}"
            ) from exc
    return _validate_encoder_source(install_root / "Source")


def _windows_toolchain():
    if os.name != "nt" or shutil.which("cl"):
        return
    installer = (
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    if not installer.exists():
        raise RuntimeError(
            "ASTC differentiable proxy requires MSVC: use an x64 Visual Studio developer shell"
        )
    installation = subprocess.check_output(
        [
            str(installer),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    script = Path(installation) / "VC/Auxiliary/Build/vcvars64.bat"
    if not script.exists():
        raise RuntimeError("MSVC x64 C++ toolchain not found")
    result = subprocess.check_output(f'call "{script}" >nul && set', shell=True, text=True)
    for line in result.splitlines():
        if "=" in line and not line.startswith("="):
            key, value = line.split("=", 1)
            os.environ[key] = value


@lru_cache(maxsize=1)
def native_extension():
    root = Path(__file__).resolve().parent / "native"
    upstream = resolve_encoder_source(root)
    _windows_toolchain()
    import ninja

    os.environ["PATH"] = str(ninja.BIN_DIR) + os.pathsep + os.environ["PATH"]
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    sources = [root / "astc_native.cpp", root / "astc_proxy.cu"]
    sources += [upstream / name for name in ASTCENC_SOURCES]
    defines = [
        "ASTCENC_SSE=0",
        "ASTCENC_AVX=0",
        "ASTCENC_NEON=0",
        "ASTCENC_SVE=0",
        "ASTCENC_POPCNT=0",
        "ASTCENC_F16C=0",
    ]
    flags = (
        (["/O2"] + ["/D" + d for d in defines])
        if os.name == "nt"
        else (["-O3"] + ["-D" + d for d in defines])
    )
    # Snapshot paths alter Ninja commands even when source bytes are identical.
    # On Windows another process may keep the old DLL loaded: isolate by path too.
    digest = hashlib.sha256(
        str(root.resolve()).encode()
        + b"\0"
        + b"".join(p.read_bytes() for p in sources)
        + b"".join(p.read_bytes() for p in sorted(upstream.glob("*.h")))
    ).hexdigest()[:10]
    return load(
        name="fntc_astc_proxy_" + digest,
        sources=[str(p) for p in sources],
        extra_include_paths=[str(upstream)],
        extra_cflags=flags,
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
