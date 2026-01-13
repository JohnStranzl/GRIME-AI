# sam2_gui/core/renderer.py
import numpy as np
import cv2
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QPolygonF
from PyQt5.QtCore import QPointF


class Renderer:
    def __init__(self, base_image_np: np.ndarray):
        self.base_image_np = base_image_np

    def base_pixmap(self):
        h, w, _ = self.base_image_np.shape
        qimage = QImage(
            self.base_image_np.data,
            w,
            h,
            self.base_image_np.strides[0],
            QImage.Format_RGB888,
        )
        return QPixmap.fromImage(qimage)

    def overlay_masks(self, base_pixmap, masks, opacity=120):
        """
        masks: list of dicts:
          - 'mask': np.ndarray (H, W)
          - 'color': (r, g, b)
          - 'visible': bool
        """
        result = QPixmap(base_pixmap)
        painter = QPainter(result)

        for m in masks:
            if not m.get("visible", True):
                continue
            mask = m["mask"].astype(bool)
            color = m["color"]

            # Draw filled mask
            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            rgba[..., 0] = color[0]
            rgba[..., 1] = color[1]
            rgba[..., 2] = color[2]
            rgba[..., 3] = mask * opacity

            qimage = QImage(
                rgba.data,
                rgba.shape[1],
                rgba.shape[0],
                rgba.strides[0],
                QImage.Format_RGBA8888,
            )
            painter.drawImage(0, 0, qimage)

            # Draw black border around mask
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            painter.setPen(QPen(QColor(0, 0, 0), 2))  # Black, 2 pixels wide

            for contour in contours:
                if len(contour) < 3:
                    continue

                # Convert contour to QPolygonF
                points = [QPointF(float(pt[0][0]), float(pt[0][1])) for pt in contour]
                polygon = QPolygonF(points)
                painter.drawPolygon(polygon)

        painter.end()
        return result

    def draw_points(self, pixmap, fg_points, bg_points):
        result = QPixmap(pixmap)
        painter = QPainter(result)
        fg_pen = QPen(QColor(0, 255, 0), 6)
        bg_pen = QPen(QColor(255, 0, 0), 6)

        for x, y in fg_points:
            painter.setPen(fg_pen)
            painter.drawPoint(int(x), int(y))

        for x, y in bg_points:
            painter.setPen(bg_pen)
            painter.drawPoint(int(x), int(y))

        painter.end()
        return result