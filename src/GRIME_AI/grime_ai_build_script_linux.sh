#!/bin/bash
set -e  # exit on error
set -x  # echo commands

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# Clean old build
rm -rf dist build

# Build the executable with PyInstaller
pyinstaller --clean --noconfirm --onedir \
  --copy-metadata torch \
  --add-data "sam2_transforms.pt:." \
  --hidden-import=torch \
  --hidden-import=torch._C \
  --hidden-import=torch._utils \
  --hidden-import=torch.fx \
  --hidden-import=torch.fx.graph_module \
  --hidden-import=torch.jit \
  --hidden-import=torch.jit._recursive \
  --hidden-import=torch.jit._script \
  --hidden-import=torch.jit._script.RecursiveScriptModule \
  --hidden-import=torch.jit._script.ScriptClass \
  --hidden-import=torch.jit._script.ScriptFunction \
  --hidden-import=torch.jit._script.ScriptModule \
  --hidden-import=torch.jit._state \
  --hidden-import=torch.package \
  --hidden-import=torch._jit_internal \
  --hidden-import=torchvision.datasets \
  --hidden-import=torchvision.io \
  --hidden-import=torchvision.ops \
  --hidden-import=torchvision.transforms \
  --hidden-import=torchvision.transforms._functional_pil \
  --hidden-import=torchvision.transforms._functional_tensor \
  --hidden-import=torchvision.transforms.functional \
  --hidden-import=torchvision.transforms.functional_pil \
  --hidden-import=torchvision.transforms.functional_tensor \
  --hidden-import=torchvision.transforms.transforms \
  --hidden-import=torchvision.transforms.v2 \
  --hidden-import=torchvision.transforms.v2._utils \
  --hidden-import=torchvision.transforms.v2.functional \
  --hidden-import=torchvision.utils \
  --hidden-import=torch.nn.modules.activation \
  --hidden-import=torch.nn.modules.loss \
  --hidden-import=torch.nn.modules.upsampling \
  --hidden-import=enum \
  --hidden-import=inspect \
  --hidden-import=types \
  --hidden-import=imageio \
  --hidden-import=imageio_ffmpeg \
  --hidden-import=openpyxl \
  --hidden-import=skimage.draw \
  --hidden-import=skimage.io \
  --hidden-import=skimage.transform \
  --hidden-import=yaml \
  --contents-directory "." main.py

# Ensure the distribution folder exists
mkdir -p dist/main

# Copy splash screen graphics
cp "SplashScreen Images/Splash_007.jpg" dist/main/
cp "SplashScreen Images/GRIME-AI Logo.jpg" dist/main/

# Copy additional required directories and files
mkdir -p dist/main/sam2
cp -r venv/lib/python*/site-packages/sam2/* dist/main/sam2/ || echo "Error copying sam2 folder"

# Rename the executable
cd dist/main
mv main GRIME-AI || echo "Error renaming executable"

# Run the executable to validate functionality
./GRIME-AI || echo "Error running the program"

# Return to root directory
cd ../..
