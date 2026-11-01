"""Traditional ASTC encoding/decoding and astcenc tool discovery."""

from __future__ import annotations
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import re

ASTC_2D_BLOCKS = {
    (4, 4),
    (5, 4),
    (5, 5),
    (6, 5),
    (6, 6),
    (8, 5),
    (8, 6),
    (8, 8),
    (10, 5),
    (10, 6),
    (10, 8),
    (10, 10),
    (12, 10),
    (12, 12),
}


def normalize_astc_block(value):
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", str(value))
    if not match or tuple(map(int, match.groups())) not in ASTC_2D_BLOCKS:
        raise ValueError(
            f"Invalid 2D ASTC block {value!r}; use a legal footprint such as 4x4, 6x6 or 8x8"
        )
    return "x".join(str(int(v)) for v in match.groups())


ASTCENC_DEFAULT_VERSION = "5.3.0"
ASTCENC_DEFAULT_PATH = "tools/astcenc"


@dataclass
class ASTCCodec:
    astcenc_path: str
    astcenc_quality: str
    astc_block: str = "6x6"
    _astcenc_executable: Optional[str] = None

    def __post_init__(self) -> None:
        self.astc_block = normalize_astc_block(self.astc_block)

    def ensure_executable(self) -> str:
        if self._astcenc_executable is None:
            self._astcenc_executable = resolve_astcenc_executable(astcenc_path=self.astcenc_path)
        return self._astcenc_executable

    def encode_rgba(self, rgba: np.ndarray) -> bytes:
        with tempfile.TemporaryDirectory(prefix="fntc_astc_") as directory:
            source, target = Path(directory) / "source.png", Path(directory) / "texture.astc"
            Image.fromarray(np.asarray(rgba, dtype=np.uint8)).save(source)
            subprocess.run(
                [
                    self.ensure_executable(),
                    "-cl",
                    str(source),
                    str(target),
                    self.astc_block,
                    "-" + self.astcenc_quality,
                ],
                check=True,
                capture_output=True,
            )
            return target.read_bytes()

    def decode_bytes(self, data: bytes) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="fntc_astc_") as directory:
            source, target = Path(directory) / "texture.astc", Path(directory) / "decoded.png"
            source.write_bytes(data)
            subprocess.run(
                [self.ensure_executable(), "-dl", str(source), str(target)],
                check=True,
                capture_output=True,
            )
            return np.array(Image.open(target).convert("RGBA"), dtype=np.uint8, copy=True)

    def roundtrip_rgba(self, rgba: np.ndarray) -> np.ndarray:
        if rgba.ndim != 3 or rgba.shape[-1] != 4:
            raise ValueError('Expected an HWC RGBA image')
        return self.decode_bytes(self.encode_rgba(rgba))


def resolve_astcenc_executable(astcenc_path: str) -> str:
    candidate = (astcenc_path or "").strip()
    if not candidate:
        raise RuntimeError("Empty astcenc path. Set --astcenc_path.")

    cmd_path = _resolve_candidate_command(candidate)
    if cmd_path:
        return cmd_path

    target_path = _to_abs_path(candidate)
    if os.path.isfile(target_path):
        return target_path

    if os.path.isdir(target_path):
        found = _find_astcenc_in_dir(target_path)
        if found:
            return found
        raise FileNotFoundError(
            f"astcenc directory exists but no executable found: '{target_path}'."
        )

    install_root = target_path
    if os.path.splitext(target_path)[1].lower() == ".exe":
        install_root = os.path.dirname(target_path)
    os.makedirs(install_root, exist_ok=True)
    installed = _auto_install_astcenc(ASTCENC_DEFAULT_VERSION, install_root)
    print(f"[ASTC] Using auto-installed astcenc: {installed}")
    return installed


def _resolve_candidate_command(candidate: str) -> Optional[str]:
    has_sep = (os.path.sep in candidate) or (
        os.path.altsep is not None and os.path.altsep in candidate
    )
    if os.path.isabs(candidate) or has_sep:
        abs_path = os.path.abspath(candidate)
        if os.path.isfile(abs_path):
            return abs_path
        return None
    resolved = shutil.which(candidate)
    if resolved and os.path.isfile(resolved):
        return resolved
    return None


def _to_abs_path(candidate: str) -> str:
    if os.path.isabs(candidate):
        return os.path.normpath(candidate)
    return os.path.abspath(os.path.normpath(candidate))


def _preferred_astcenc_names():
    if os.name == "nt":
        return ["astcenc-avx2.exe", "astcenc-sse4.1.exe", "astcenc-sse2.exe", "astcenc.exe"]
    return ["astcenc", "astcenc-avx2", "astcenc-sse4.1", "astcenc-sse2", "astcenc-neon"]


def _find_astcenc_in_dir(root_dir: str) -> Optional[str]:
    preferred = _preferred_astcenc_names()
    preferred_lower = [n.lower() for n in preferred]
    matches = []
    for walk_root, _, files in os.walk(root_dir):
        for fname in files:
            lower = fname.lower()
            full_path = os.path.join(walk_root, fname)
            if lower in preferred_lower:
                return full_path
            if lower.startswith("astcenc") and (os.name != "nt" or lower.endswith(".exe")):
                matches.append(full_path)
    return sorted(matches)[0] if matches else None


def _auto_install_astcenc(version: str, install_root: str) -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "windows" or machine not in {"amd64", "x86_64"}:
        raise RuntimeError(
            "Automatic astcenc install currently supports Windows x64 only. "
            "Please install astcenc manually or set --astcenc_path."
        )

    asset_name = f"astcenc-{version}-windows-x64.zip"
    url = f"https://github.com/ARM-software/astc-encoder/releases/download/{version}/{asset_name}"

    os.makedirs(install_root, exist_ok=True)
    zip_path = os.path.join(install_root, asset_name)
    extract_dir = os.path.join(install_root, f"astcenc-{version}-windows-x64")

    if not os.path.isfile(zip_path):
        print(f"[ASTC] Downloading astcenc {version} from {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to download astcenc from '{url}': {exc}") from exc

    os.makedirs(extract_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract '{zip_path}': {exc}") from exc

    found = _find_astcenc_in_dir(extract_dir)
    if found:
        return found
    raise RuntimeError(
        f"Auto-install succeeded but no astcenc executable found in '{extract_dir}'."
    )
