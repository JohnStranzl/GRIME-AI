@echo on

REM # Author: John Edward Stranzl, Jr.
REM # Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
REM # Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
REM # Created: Mar 6, 2022
REM # License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

REM Build the executable with PyInstaller
  REM Hidden Imports by section:
    REM Torch Core
    REM TorchVision & Transforms
    REM Torch NN Modules
    REM Standard Library Helpers
    REM Utility & Scientific Libraries
pyinstaller --clean --noconfirm --onedir ^
  --copy-metadata torch ^
  --copy-metadata neonutilities ^
  --collect-all neonutilities ^
  --add-data="sam2_transforms.pt;." ^
  --add-data="resources/leaflet/leaflet.js;resources/leaflet" ^
  --add-data="resources/leaflet/leaflet.css;resources/leaflet" ^
  --add-data="resources/leaflet/images;resources/leaflet/images" ^
  --add-data="resources/splash_screens;resources/splash_screens" ^
  --add-data="resources/toolbar_icons;resources/toolbar_icons" ^
  --add-data="resources/shape_files;resources/shape_files" ^
  --add-data="dialogs/**/*.ui;dialogs" ^
  --icon="resources/app_icons/GRIME-AI Logo.ico" ^
  --hidden-import=torch ^
  --hidden-import=torch._C ^
  --hidden-import=torch._utils ^
  --hidden-import=torch.fx ^
  --hidden-import=torch.fx.graph_module ^
  --hidden-import=torch.jit ^
  --hidden-import=torch.jit._recursive ^
  --hidden-import=torch.jit._script ^
  --hidden-import=torch.jit._script.RecursiveScriptModule ^
  --hidden-import=torch.jit._script.ScriptClass ^
  --hidden-import=torch.jit._script.ScriptFunction ^
  --hidden-import=torch.jit._script.ScriptModule ^
  --hidden-import=torch.jit._state ^
  --hidden-import=torch.package ^
  --hidden-import=torch._jit_internal ^
  --hidden-import=torchvision.datasets ^
  --hidden-import=torchvision.io ^
  --hidden-import=torchvision.ops ^
  --hidden-import=torchvision.transforms ^
  --hidden-import=torchvision.transforms._functional_pil ^
  --hidden-import=torchvision.transforms._functional_tensor ^
  --hidden-import=torchvision.transforms.functional ^
  --hidden-import=torchvision.transforms.functional_pil ^
  --hidden-import=torchvision.transforms.functional_tensor ^
  --hidden-import=torchvision.transforms.transforms ^
  --hidden-import=torchvision.transforms.v2 ^
  --hidden-import=torchvision.transforms.v2._utils ^
  --hidden-import=torchvision.transforms.v2.functional ^
  --hidden-import=torchvision.utils ^
  --hidden-import=torch.nn.modules.activation ^
  --hidden-import=torch.nn.modules.loss ^
  --hidden-import=torch.nn.modules.upsampling ^
  --hidden-import=enum ^
  --hidden-import=inspect ^
  --hidden-import=types ^
  --hidden-import=imageio ^
  --hidden-import=imageio_ffmpeg ^
  --hidden-import=openpyxl ^
  --hidden-import=skimage.draw ^
  --hidden-import=skimage.io ^
  --hidden-import=skimage.transform ^
  --hidden-import=neonutilities ^
  --hidden-import=neonutilities.aop_download ^
  --hidden-import=neonutilities.citation ^
  --hidden-import=neonutilities.files_by_uri ^
  --hidden-import=neonutilities.get_issue_log ^
  --hidden-import=neonutilities.helper_mods ^
  --hidden-import=neonutilities.read_table_neon ^
  --hidden-import=neonutilities.tabular_download ^
  --hidden-import=neonutilities.unzip_and_stack ^
  --hidden-import=yaml ^
  --hidden-import=transformers ^
  --hidden-import=transformers.utils ^
  --hidden-import=transformers.dependency_versions_check ^
  --hidden-import=tokenizers ^
    --contents-directory "." main.py

REM Ensure the distribution folder exists
if not exist "dist\main" mkdir "dist\main"

REM Copy additional required directories and files
if not exist "dist\main\sam2" mkdir "dist\main\sam2"
xcopy "venv\Lib\site-packages\sam2" "dist\main\sam2" /s /e /Y || echo Error copying sam2 folder

REM Change the directory to the distribution folder and then rename the executable
cd dist\main
ren main.exe GRIME-AI.exe || echo Error renaming executable

REM Run the executable to validate its functionality
REM GRIME-AI || echo Error running the program

REM Return to the root directory
cd ..\..\
@echo off
