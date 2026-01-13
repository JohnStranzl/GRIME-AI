# sam2_gui/utils/mask_ops.py
import numpy as np


def compute_mask_stats(mask: np.ndarray):
    mask_bool = mask.astype(bool)
    ys, xs = np.where(mask_bool)

    if len(xs) == 0:
        bbox = None
    else:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        bbox = (int(xmin), int(ymin), int(xmax), int(ymax))

    area = int(mask_bool.sum())

    return {
        "area_px": area,
        "bbox": bbox,
    }
