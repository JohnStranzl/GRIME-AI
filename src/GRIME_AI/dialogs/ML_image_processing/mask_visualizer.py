import numpy as np
import cv2
import pycocotools.mask as maskutils


# ======================================================================================================================
# ======================================================================================================================
# =====                                           class MaskVisualizer                                             =====
# ======================================================================================================================
# ======================================================================================================================
class MaskVisualizer:
    """
    Stateless helper for rendering COCO masks and centroids onto images.

    Usage:
        viz = MaskVisualizer(category_color_fn=my_color_fn, marker_color=(255,255,255))
        out = viz.overlay_all_masks(img, annotations)
        out_single = viz.overlay_single_mask(img, annotation)
    """

    def __init__(self, category_color_fn=None, marker_color=(255, 255, 255)):
        self.category_color_fn = category_color_fn or (lambda cid: tuple(np.random.RandomState(int(cid)).randint(0,255,3).tolist()))
        self.marker_color = marker_color

    def category_color(self, cid):
        return self.category_color_fn(cid)

    def compute_mask_centroid(self, ann, h, w):
        seg = ann.get("segmentation", None)
        try:
            if isinstance(seg, list) and len(seg) > 0:
                total_area = 0.0
                cx_sum = 0.0
                cy_sum = 0.0
                polys = seg if isinstance(seg[0], (list, tuple)) else [seg[0]]
                for poly in polys:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    area = cv2.contourArea(pts)
                    if abs(area) < 1e-6:
                        mx = float(np.mean(pts[:, 0])); my = float(np.mean(pts[:, 1])); area = 1.0
                    else:
                        M = cv2.moments(pts)
                        if M["m00"] != 0:
                            mx = M["m10"] / M["m00"]; my = M["m01"] / M["m00"]
                        else:
                            mx = float(np.mean(pts[:, 0])); my = float(np.mean(pts[:, 1]))
                    cx_sum += mx * area; cy_sum += my * area; total_area += area
                if total_area == 0: return (None, None)
                return (cx_sum / total_area, cy_sum / total_area)

            elif isinstance(seg, dict):
                counts = seg.get("counts", None)
                seg_for_decode = seg if "size" in seg else {"counts": counts, "size": [h, w]}
                m = maskutils.decode(seg_for_decode)
                if m is None: return (None, None)
                if m.ndim == 3:
                    m_comb = np.any(m, axis=2).astype(np.uint8)
                else:
                    m_comb = (m > 0).astype(np.uint8)
                M = cv2.moments(m_comb)
                if M["m00"] == 0:
                    ys, xs = np.where(m_comb > 0)
                    if len(xs) == 0: return (None, None)
                    return (float(np.mean(xs)), float(np.mean(ys)))
                cx = M["m10"] / M["m00"]; cy = M["m01"] / M["m00"]
                return (cx, cy)
        except Exception:
            return (None, None)

    def draw_centroid(self, img, cx, cy, color=None, label=None):
        if cx is None or cy is None: return img
        color = color or self.marker_color
        ix = int(round(cx)); iy = int(round(cy))
        h, w = img.shape[:2]
        ix = max(0, min(w - 1, ix)); iy = max(0, min(h - 1, iy))

        try:
            cv2.drawMarker(img, (ix, iy), color, markerType=cv2.MARKER_CROSS,
                           markerSize=12, thickness=2, line_type=cv2.LINE_AA)
        except Exception:
            l = 8
            cv2.line(img, (ix - l, iy), (ix + l, iy), color, 2, lineType=cv2.LINE_AA)
            cv2.line(img, (ix, iy - l), (ix, iy + l), color, 2, lineType=cv2.LINE_AA)

        text = label if label is not None else f"({ix}, {iy})"
        font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.5; thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = ix + 10; ty = iy - 10
        if tx + tw > w - 1: tx = ix - 10 - tw
        if ty - th < 0: ty = iy + 10 + th
        tx = max(0, min(w - tw - 1, tx)); ty = max(th, min(h - 1, ty))
        cv2.rectangle(img, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, text, (tx, ty), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
        return img

    def overlay_all_masks(self, img, anns):
        overlay = img.copy()
        h, w = img.shape[:2]
        for ann in anns:
            seg = ann.get("segmentation", [])
            color = self.category_color(ann.get("category_id", 0))
            if isinstance(seg, list) and len(seg) > 0:
                try:
                    polys = seg if isinstance(seg[0], (list, tuple)) else [seg[0]]
                    for poly in polys:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(overlay, [pts], color)
                    cx, cy = self.compute_mask_centroid(ann, h, w)
                    if cx is not None: self.draw_centroid(overlay, cx, cy)
                except Exception:
                    continue
            elif isinstance(seg, dict):
                try:
                    counts = seg.get("counts", None)
                    if isinstance(counts, list):
                        rle = maskutils.frPyObjects(seg, h, w)
                        m = maskutils.decode(rle)
                    else:
                        if "size" not in seg:
                            seg = {"counts": counts, "size": [h, w]}
                        m = maskutils.decode(seg)
                    if m is not None:
                        overlay[m > 0] = color
                        cx, cy = self.compute_mask_centroid(ann, h, w)
                        if cx is not None: self.draw_centroid(overlay, cx, cy)
                except Exception:
                    continue
        return cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    def overlay_single_mask(self, img, ann):
        overlay = img.copy()
        h, w = img.shape[:2]
        seg = ann.get("segmentation", [])
        color = self.category_color(ann.get("category_id", 0))
        if isinstance(seg, list) and len(seg) > 0:
            try:
                polys = seg if isinstance(seg[0], (list, tuple)) else [seg[0]]
                for poly in polys:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(overlay, [pts], color)
                cx, cy = self.compute_mask_centroid(ann, h, w)
                if cx is not None: self.draw_centroid(overlay, cx, cy)
            except Exception:
                pass
        elif isinstance(seg, dict):
            try:
                counts = seg.get("counts", None)
                if isinstance(counts, list):
                    rle = maskutils.frPyObjects(seg, h, w)
                    m = maskutils.decode(rle)
                else:
                    if "size" not in seg:
                        seg = {"counts": counts, "size": [h, w]}
                    m = maskutils.decode(seg)
                if m is not None:
                    overlay[m > 0] = color
                    cx, cy = self.compute_mask_centroid(ann, h, w)
                    if cx is not None: self.draw_centroid(overlay, cx, cy)
            except Exception:
                pass
        return cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    def draw_mask_border(self, img, ann, color=(0, 255, 255), thickness=4, draw_inplace=True):
        """
        Draw polygon or RLE mask borders onto img and return the image.
        Works in-place by default; set draw_inplace=False to operate on a copy.
        """
        if not draw_inplace:
            img = img.copy()

        seg = ann.get("segmentation", [])
        # Polygon segmentation
        if isinstance(seg, list) and len(seg) > 0:
            try:
                polys = seg if isinstance(seg[0], (list, tuple)) else [seg[0]]
                for poly in polys:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(img, [pts], True, color, thickness, lineType=cv2.LINE_AA)
            except Exception:
                pass

        # RLE segmentation
        elif isinstance(seg, dict):
            try:
                h, w = img.shape[:2]
                counts = seg.get("counts", None)
                # Normalize seg for decode
                seg_for_decode = seg
                if "size" not in seg_for_decode:
                    seg_for_decode = {"counts": counts, "size": [h, w]}

                # Decode (frPyObjects if uncompressed)
                if isinstance(counts, list):
                    rle = maskutils.frPyObjects(seg_for_decode, h, w)
                    m = maskutils.decode(rle)
                else:
                    m = maskutils.decode(seg_for_decode)

                if m is None:
                    return img

                # maskutils.decode may return (H,W) or (H,W,N)
                if m.ndim == 3:
                    m_comb = np.any(m, axis=2).astype(np.uint8)
                else:
                    m_comb = (m > 0).astype(np.uint8)

                # findContours expects single-channel uint8
                contours, _ = cv2.findContours(m_comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(img, contours, -1, color, thickness, lineType=cv2.LINE_AA)
            except Exception:
                pass

        return img
