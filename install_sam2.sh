#!/bin/bash
# Clone the SAM2 repository and install in editable mode

set -e  # exit on error

git clone https://github.com/facebookresearch/sam2.git
cd sam2

# Use the active Python environment's pip
python3 -m pip install -e .

echo "SAM2 has been installed in editable mode."
