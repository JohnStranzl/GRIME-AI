# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata

datas = [('sam2_transforms.pt', '.')]
datas += copy_metadata('torch')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['torch', 'torch._C', 'torch._utils', 'torch.fx', 'torch.fx.graph_module', 'torch.jit', 'torch.jit._recursive', 'torch.jit._script', 'torch.jit._script.RecursiveScriptModule', 'torch.jit._script.ScriptClass', 'torch.jit._script.ScriptFunction', 'torch.jit._script.ScriptModule', 'torch.jit._state', 'torch.package', 'torch._jit_internal', 'torchvision.datasets', 'torchvision.io', 'torchvision.ops', 'torchvision.transforms', 'torchvision.transforms._functional_pil', 'torchvision.transforms._functional_tensor', 'torchvision.transforms.functional', 'torchvision.transforms.functional_pil', 'torchvision.transforms.functional_tensor', 'torchvision.transforms.transforms', 'torchvision.transforms.v2', 'torchvision.transforms.v2._utils', 'torchvision.transforms.v2.functional', 'torchvision.utils', 'torch.nn.modules.activation', 'torch.nn.modules.loss', 'torch.nn.modules.upsampling', 'enum', 'inspect', 'types', 'imageio', 'imageio_ffmpeg', 'openpyxl', 'skimage.draw', 'skimage.io', 'skimage.transform', 'yaml'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory='.',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
