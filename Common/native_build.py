"""Locate the Windows C++ compiler shared by native GPU extensions."""
import os
from pathlib import Path
import shutil
import subprocess


def prepare_msvc():
    """Load the installed x64 MSVC/Windows SDK environment when needed."""
    if os.name == 'nt' and not shutil.which('cl'):
        installer = (Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))
                     / 'Microsoft Visual Studio/Installer/vswhere.exe')
        if not installer.exists():
            raise RuntimeError('Native GPU extensions require MSVC: use an x64 Visual Studio developer shell')
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
