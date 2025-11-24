@echo off
REM Clone the SAM2 repository and install in editable mode

git clone https://github.com/facebookresearch/sam2.git
cd sam2

REM Use the active Python environment's pip
python -m pip install -e .

echo.
echo SAM2 has been installed in editable mode.
pause
