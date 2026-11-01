"""Prepare the compiler environment shared by the CUDA codec extensions."""
import os
from pathlib import Path
import shutil
import subprocess

import torch


def prepare_cuda_build():
    """Locate MSVC and Ninja, and target the current GPU unless configured."""
    if os.name == 'nt' and not shutil.which('cl'):
        installer = (Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))
                     / 'Microsoft Visual Studio/Installer/vswhere.exe')
        if not installer.exists():
            raise RuntimeError('CUDA extensions require MSVC: use an x64 Visual Studio developer shell')
        installation = subprocess.check_output(
            [str(installer), '-latest', '-products', '*', '-requires',
             'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'],
            text=True).strip()
        script = Path(installation) / 'VC/Auxiliary/Build/vcvars64.bat'
        if not script.exists():
            raise RuntimeError('MSVC x64 C++ toolchain not found')
        result = subprocess.check_output(f'call "{script}" >nul && set', shell=True, text=True)
        for line in result.splitlines():
            if '=' in line and not line.startswith('='):
                key, value = line.split('=', 1)
                os.environ[key] = value

    import ninja
    os.environ['PATH'] = str(ninja.BIN_DIR) + os.pathsep + os.environ['PATH']
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
