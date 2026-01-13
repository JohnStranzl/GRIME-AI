# sam2_gui/utils/image_io.py
import numpy as np
from PIL import Image


def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)
