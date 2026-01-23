import os
import numpy as np

def load_binary_mask(path: str, target_shape=None) -> np.ndarray:
    """
    Load a binary mask from .tif/.tiff/.png/.jpg/.npy and return bool (H,W).
    target_shape: (H, W) to resize with nearest-neighbor if needed.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".tif", ".tiff"]:
        # PIL handles LZW-compressed tiffs well in most cases
        from PIL import Image
        m = np.array(Image.open(path))
    elif ext == ".npy":
        m = np.load(path)
    else:
        import cv2
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(path)
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    # Convert to boolean (your sample is 0/255)
    mask = m > 0

    # Resize to match image if needed
    if target_shape is not None and mask.shape != target_shape:
        import cv2
        mask = cv2.resize(
            mask.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    return mask
