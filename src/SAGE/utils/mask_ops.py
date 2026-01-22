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

def sample_points_from_mask(mask_bool, n_fg=40, n_bg=40, ring_px=25, seed=0):
    """
    FG: random points inside mask
    BG: random points from ring around mask (dilate(mask) - mask)
    Returns: (fg_points, bg_points) as [(x,y), ...]
    """
    rng = np.random.default_rng(seed)

    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return [], []

    # FG points
    k_fg = min(n_fg, len(xs))
    idx_fg = rng.choice(len(xs), size=k_fg, replace=False)
    fg_points = [(int(xs[i]), int(ys[i])) for i in idx_fg]

    # BG ring
    import cv2
    k = max(3, ring_px // 2 * 2 + 1)  # odd kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = np.logical_and(dil, np.logical_not(mask_bool))

    rys, rxs = np.where(ring)
    if len(rxs) == 0:
        return fg_points, []

    k_bg = min(n_bg, len(rxs))
    idx_bg = rng.choice(len(rxs), size=k_bg, replace=False)
    bg_points = [(int(rxs[i]), int(rys[i])) for i in idx_bg]

    return fg_points, bg_points