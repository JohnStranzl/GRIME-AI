# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('sam2_transforms.pt', '.'), ('resources/leaflet/leaflet.js', 'resources/leaflet'), ('resources/leaflet/leaflet.css', 'resources/leaflet'), ('resources/leaflet/images', 'resources/leaflet/images'), ('resources/splash_screens', 'resources/splash_screens'), ('resources/toolbar_icons', 'resources/toolbar_icons'), ('resources/shape_files', 'resources/shape_files'), ('dialogs/**/*.ui', 'dialogs')]
binaries = []
hiddenimports = ['torch', 'torch._C', 'torch._utils', 'torch.fx', 'torch.fx.graph_module', 'torch.jit', 'torch.jit._recursive', 'torch.jit._script', 'torch.jit._script.RecursiveScriptModule', 'torch.jit._script.ScriptClass', 'torch.jit._script.ScriptFunction', 'torch.jit._script.ScriptModule', 'torch.jit._state', 'torch.package', 'torch._jit_internal', 'torchvision.datasets', 'torchvision.io', 'torchvision.ops', 'torchvision.transforms', 'torchvision.transforms._functional_pil', 'torchvision.transforms._functional_tensor', 'torchvision.transforms.functional', 'torchvision.transforms.functional_pil', 'torchvision.transforms.functional_tensor', 'torchvision.transforms.transforms', 'torchvision.transforms.v2', 'torchvision.transforms.v2._utils', 'torchvision.transforms.v2.functional', 'torchvision.utils', 'torch.nn.modules.activation', 'torch.nn.modules.loss', 'torch.nn.modules.upsampling', 'enum', 'inspect', 'types', 'imageio', 'imageio_ffmpeg', 'openpyxl', 'skimage.draw', 'skimage.io', 'skimage.transform', 'neonutilities', 'neonutilities.aop_download', 'neonutilities.citation', 'neonutilities.files_by_uri', 'neonutilities.get_issue_log', 'neonutilities.helper_mods', 'neonutilities.read_table_neon', 'neonutilities.tabular_download', 'neonutilities.unzip_and_stack', 'yaml', 'transformers', 'transformers.utils', 'transformers.dependency_versions_check', 'tokenizers']
datas += copy_metadata('torch')
datas += copy_metadata('neonutilities')
tmp_ret = collect_all('neonutilities')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
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
    icon=['resources\\app_icons\\GRIME-AI Logo.ico'],
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
