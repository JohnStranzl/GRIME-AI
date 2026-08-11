# sam2_gui/core/renderer.py
import numpy as np
import cv2
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QPolygonF
from PyQt5.QtCore import QPointF


def _mask_bbox(mask):
    """Tight (x0, y0, x1, y1) bounds of the True pixels, end-exclusive, or None
    if the mask is empty. Two axis reductions - far cheaper than the full-frame
    allocation and blend it lets us skip."""
    rows = np.any(mask, axis=1)
    if not rows.any():
        return None
    cols = np.any(mask, axis=0)
    ys = np.flatnonzero(rows)
    xs = np.flatnonzero(cols)
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _rgba_patch(sub_mask, color, alpha):
    """Build an RGBA buffer covering only the mask's bounding box.

    Alpha is filled by boolean assignment rather than `sub_mask * alpha`:
    multiplying a bool array by a Python int promotes the whole thing to int64,
    which on a full frame is an 8-bytes-per-pixel temporary allocated and thrown
    away for every mask on every repaint."""
    h, w = sub_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3][sub_mask] = alpha
    return rgba


def _brighten(color, amount=120):
    return (min(255, color[0] + amount),
            min(255, color[1] + amount),
            min(255, color[2] + amount))


class Renderer:
    # Contour simplification tolerance in pixels for drawn borders. Sub-pixel,
    # so outlines stay visually identical while shedding most of the staircase
    # vertices that segmentation masks produce. Set to 0 to disable.
    BORDER_SIMPLIFY_EPS = 0.5

    def __init__(self, base_image_np: np.ndarray):
        self.base_image_np = base_image_np
        self._base_pixmap = None

    def base_pixmap(self):
        """Cached. The source array does not change for the life of a Renderer
        (one is built per image), and this was re-converting the whole frame
        from numpy on every repaint."""
        if self._base_pixmap is None:
            h, w, _ = self.base_image_np.shape
            qimage = QImage(
                self.base_image_np.data,
                w,
                h,
                self.base_image_np.strides[0],
                QImage.Format_RGB888,
            )
            # copy() detaches from the numpy buffer so the pixmap stays valid
            # independently of the array's lifetime.
            self._base_pixmap = QPixmap.fromImage(qimage.copy())
        return self._base_pixmap

    def _draw_mask_fill(self, painter, mask, color, alpha, bbox):
        """Blend one mask over the painter, touching only its bounding box."""
        x0, y0, x1, y1 = bbox
        sub = mask[y0:y1, x0:x1]
        rgba = _rgba_patch(sub, color, alpha)
        qimage = QImage(
            rgba.data, rgba.shape[1], rgba.shape[0],
            rgba.strides[0], QImage.Format_RGBA8888,
        )
        painter.drawImage(x0, y0, qimage)
        # rgba must stay referenced until the draw completes: QImage wraps the
        # buffer without copying it.
        del qimage

    def _draw_mask_border(self, painter, mask, bbox):
        """Outline one mask. Contours are traced on the bounding-box crop and
        the vertices shifted back, instead of scanning the full frame."""
        x0, y0, x1, y1 = bbox
        sub = np.ascontiguousarray(mask[y0:y1, x0:x1]).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < 3:
                continue
            if self.BORDER_SIMPLIFY_EPS > 0:
                simplified = cv2.approxPolyDP(
                    contour, self.BORDER_SIMPLIFY_EPS, True)
                if len(simplified) >= 3:
                    contour = simplified
            pts = contour.reshape(-1, 2)
            polygon = QPolygonF(
                [QPointF(float(px + x0), float(py + y0)) for px, py in pts]
            )
            painter.drawPolygon(polygon)

    def overlay_masks(self, base_pixmap, masks, opacity=120, selected_mask_id=-1,
                      show_borders=True, flash_mask_id=-1):
        """
        masks: list of dicts:
          - 'mask': np.ndarray (H, W)
          - 'color': (r, g, b)
          - 'visible': bool
        selected_mask_id: draw this mask's border in yellow instead of black
        """
        result = QPixmap(base_pixmap)
        painter = QPainter(result)

        selected_pen = QPen(QColor(255, 255, 0), 2)
        normal_pen = QPen(QColor(255, 255, 0), 1)

        for m in masks:
            if not m.get("visible", True):
                continue
            mask = m["mask"]
            if mask.dtype != bool:
                mask = mask.astype(bool)

            bbox = _mask_bbox(mask)
            if bbox is None:
                continue          # empty mask: nothing to blend or outline

            color = m["color"]

            # Flash: brighten the fill and boost opacity for the flashed mask.
            is_flash = (m.get("id", -1) == flash_mask_id)
            if is_flash:
                color = _brighten(color)
                fill_opacity = min(255, opacity + 110)
            else:
                fill_opacity = opacity

            self._draw_mask_fill(painter, mask, color, fill_opacity, bbox)

            # Draw border - yellow, thicker for the selected mask
            is_selected = (m.get("id", -1) == selected_mask_id)
            if show_borders or is_selected:
                painter.setPen(selected_pen if is_selected else normal_pen)
                self._draw_mask_border(painter, mask, bbox)

        painter.end()
        return result

    def overlay_single_mask(self, base_pixmap, m, opacity=120):
        """Paint ONE mask, brightened, on top of an already-composited pixmap.
        Cheap path used by the selection flash - no full recomposite, no
        per-mask contour work. Matches the brighten used in overlay_masks."""
        result = QPixmap(base_pixmap)
        mask = m["mask"]
        if mask.dtype != bool:
            mask = mask.astype(bool)

        bbox = _mask_bbox(mask)
        if bbox is None:
            return result

        painter = QPainter(result)
        self._draw_mask_fill(
            painter, mask, _brighten(m["color"]),
            min(255, opacity + 110), bbox,
        )
        painter.end()
        return result

    def draw_points(self, pixmap, fg_points, bg_points):
        result = QPixmap(pixmap)
        painter = QPainter(result)
        fg_pen = QPen(QColor(0, 255, 0), 2)
        bg_pen = QPen(QColor(255, 0, 0), 2)

        for x, y in fg_points:
            painter.setPen(fg_pen)
            painter.drawPoint(int(x), int(y))

        for x, y in bg_points:
            painter.setPen(bg_pen)
            painter.drawPoint(int(x), int(y))

        painter.end()
        return result
