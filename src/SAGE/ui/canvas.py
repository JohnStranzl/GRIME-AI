from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QPen, QPainterPath, QColor, QBrush

from SAGE.ui.mask_item import MaskItem


class Canvas(QGraphicsView):
    polygon_drawn = pyqtSignal(list)  # list of (x, y) tuples
    eraser_move = pyqtSignal(float, float)  # x,y in image coords

    def __init__(self, on_left_click, on_right_click, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._on_left_click = on_left_click
        self._on_right_click = on_right_click

        self._panning = False
        self._pan_start = None

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self._orig_width = None
        self._orig_height = None

        self._segmentation_mode = "points"
        self._drawing_polygon = False
        self._polygon_points = []
        self._polygon_path_item = None

        # Paint mode variables
        self._painting = False
        self._paint_points = []
        self._paint_path_item = None
        self._brush_size = 15  # pixels between paint points
        self._paint_is_negative = False  # track if current stroke is negative

        # Eraser mode variables
        self._eraser_enabled = False
        self._resize_anchor = None
        self._min_eraser_radius = 5
        self._max_eraser_radius = 150
        self._is_erasing = False
        self._eraser_radius = 18
        self._resizing_eraser = False
        self.setFocusPolicy(Qt.StrongFocus)

        # Eraser preview circle (cursor)
        self._eraser_preview = QGraphicsEllipseItem()
        self._eraser_preview.setZValue(10_000)  # always on top
        self._eraser_preview.setVisible(False)

        pen = QPen(QColor(0, 255, 255, 180))  # cyan-ish
        fill = QColor(0, 255, 255, 60)  # cyan, low alpha
        self._eraser_preview.setBrush(QBrush(fill))
        pen.setWidth(2)
        pen.setCosmetic(True)  # IMPORTANT: thickness stays same while zooming
        self._eraser_preview.setPen(pen)

        self._eraser_preview.setBrush(QBrush(Qt.NoBrush))
        self._scene.addItem(self._eraser_preview)

    def set_segmentation_mode(self, mode: str):
        if mode in ("points", "polygon", "paint", "manual_polygon"):
            self._segmentation_mode = mode

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap_item.setPixmap(pixmap)
        self._orig_width = pixmap.width()
        self._orig_height = pixmap.height()
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def set_eraser_enabled(self, enabled: bool):
        self._eraser_enabled = enabled
        if not enabled:
            self._is_erasing = False
        self._eraser_preview.setVisible(enabled)

    def _update_eraser_preview(self, x, y):
        r = float(self._eraser_radius)
        self._eraser_preview.setRect(x - r, y - r, 2 * r, 2 * r)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item.pixmap():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click events - MUST come before mousePressEvent logic"""
        if self._eraser_enabled:
            event.accept()
            return
        # ----------------------------------------------------
        # Right-double-click: Delete mask (same as Ctrl+Right-click)
        # ----------------------------------------------------
        if event.button() == Qt.RightButton:
            scene_pos = self.mapToScene(event.pos())
            item = self._scene.itemAt(scene_pos, self.transform())

            if isinstance(item, MaskItem):
                self._on_right_click(item.mask_id)
                event.accept()
                return

        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):

        # =========================================================
        # ERASER MODE OWNS THE MOUSE (blocks ALL other behaviors)
        # =========================================================
        if self._eraser_enabled:
            # If holding R (resizing mode), start anchor on left press
            if self._resizing_eraser and event.button() == Qt.LeftButton:
                self._is_erasing = False
                self._resize_anchor = event.pos()
                event.accept()
                return

            # Normal erase: left-click begins erasing
            if event.button() == Qt.LeftButton:
                self._is_erasing = True
                x, y = self._map_to_image_coords(event.pos())
                if self._is_in_bounds(x, y):
                    self._update_eraser_preview(x, y)
                    self.eraser_move.emit(x, y)

            # IMPORTANT: swallow everything (blocks right-click negative points, mask delete, etc.)
            event.accept()
            return

        # ----------------------------------------------------
        # Ctrl+Right-click: Delete mask
        # ----------------------------------------------------
        if event.button() == Qt.RightButton and event.modifiers() == Qt.ControlModifier:
            scene_pos = self.mapToScene(event.pos())
            item = self._scene.itemAt(scene_pos, self.transform())

            if isinstance(item, MaskItem):
                self._on_right_click(item.mask_id)
                event.accept()
                return

        # ----------------------------------------------------
        # Regular right-click: Add negative point OR paint negative brush
        # ----------------------------------------------------
        if event.button() == Qt.RightButton:
            scene_pos = self.mapToScene(event.pos())
            item = self._scene.itemAt(scene_pos, self.transform())

            # If clicking on a mask, wait to see if it's a double-click
            if isinstance(item, MaskItem):
                super().mousePressEvent(event)
                return

            # Paint mode: start negative brush stroke
            if self._segmentation_mode == "paint":
                pos = self.mapToScene(event.pos())
                x, y = pos.x(), pos.y()

                if (
                    self._orig_width is not None
                    and (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)
                ):
                    super().mousePressEvent(event)
                    return

                self._painting = True
                self._paint_is_negative = True  # Right-click = negative
                self._paint_points = [(x, y)]
                self._on_right_click(x, y)  # Add as background point
                self._init_paint_path_item()
                self._update_paint_path()
                return

            # Not paint mode: add single negative point
            pos = self.mapToScene(event.pos())
            x, y = pos.x(), pos.y()

            if (
                self._orig_width is not None
                and (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)
            ):
                super().mousePressEvent(event)
                return

            self._handle_click(QPointF(x, y), is_left=False)
            return

        # ----------------------------------------------------
        # Left-click: Eraser or Add positive point, draw polygon, or paint
        # ----------------------------------------------------
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            x = pos.x()
            y = pos.y()

            if (
                    self._orig_width is not None
                    and (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)
            ):
                super().mousePressEvent(event)
                return

            if self._segmentation_mode == "polygon" or self._segmentation_mode == "manual_polygon":
                self._drawing_polygon = True
                self._polygon_points = [(x, y)]
                self._init_polygon_path_item()
                self._update_polygon_path()

            elif self._segmentation_mode == "paint":
                self._painting = True
                self._paint_is_negative = False  # Left-click = positive
                self._paint_points = [(x, y)]
                self._on_left_click(x, y)  # Add as foreground point
                self._init_paint_path_item()
                self._update_paint_path()
            else:
                self._handle_click(QPointF(x, y), is_left=True)

            return

        # ----------------------------------------------------
        # Middle-click: Pan
        # ----------------------------------------------------
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def _is_in_bounds(self, x, y):
        """Check if coordinates are within image bounds"""
        if self._orig_width is None:
            return True
        return not (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)

    def _map_to_image_coords(self, view_pos):
        """
        Convert a QWidget mouse position (event.pos()) into *image pixel coords*.
        Uses the scene coords since your pixmap is drawn into the scene 1:1.
        """
        scene_pos = self.mapToScene(view_pos)
        return float(scene_pos.x()), float(scene_pos.y())

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )

        if self._drawing_polygon and (self._segmentation_mode == "polygon" or self._segmentation_mode == "manual_polygon"):
            pos = self.mapToScene(event.pos())
            x = pos.x()
            y = pos.y()

            if (
                    self._orig_width is not None
                    and (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)
            ):
                super().mouseMoveEvent(event)
                return

            self._polygon_points.append((x, y))
            self._update_polygon_path()

        # Paint mode - add points as mouse moves
        if self._painting and self._segmentation_mode == "paint":
            pos = self.mapToScene(event.pos())
            x = pos.x()
            y = pos.y()

            if (
                    self._orig_width is not None
                    and (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)
            ):
                super().mouseMoveEvent(event)
                return

            # Only add point if far enough from last point (brush spacing)
            if len(self._paint_points) > 0:
                last_x, last_y = self._paint_points[-1]
                dist = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
                if dist < self._brush_size:
                    super().mouseMoveEvent(event)
                    return

            self._paint_points.append((x, y))

            # Add point with correct type based on negative flag
            if self._paint_is_negative:
                self._on_right_click(x, y)  # Add as background point
            else:
                self._on_left_click(x, y)  # Add as foreground point

            self._update_paint_path()

        if self._eraser_enabled and self._resizing_eraser and self._resize_anchor:
            dy = event.pos().y() - self._resize_anchor.y()
            new_radius = self._eraser_radius + dy * 0.3

            self._eraser_radius = int(
                max(self._min_eraser_radius, min(self._max_eraser_radius, new_radius))
            )
            self._resize_anchor = event.pos()

            # update preview
            x, y = self._map_to_image_coords(event.pos())
            if self._is_in_bounds(x, y):
                self._update_eraser_preview(x, y)

            event.accept()
            return

        # Eraser enabled: always show preview; erase only if dragging
        if self._eraser_enabled:
            x, y = self._map_to_image_coords(event.pos())
            if self._is_in_bounds(x, y):
                self._update_eraser_preview(x, y)

                if self._is_erasing:  # <-- ONLY erase while mouse is held
                    self.eraser_move.emit(x, y)

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # Stop erasing on left release
        if self._eraser_enabled and event.button() == Qt.LeftButton:
            self._is_erasing = False
            event.accept()
            return

        # If resizing (R held), stop "drag anchor" on release too
        if self._eraser_enabled and self._resizing_eraser and event.button() == Qt.LeftButton:
            self._resize_anchor = None
            event.accept()
            return

        if event.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)

        if event.button() == Qt.LeftButton and self._drawing_polygon:
            self._drawing_polygon = False

            if len(self._polygon_points) >= 3:
                if self._polygon_points[0] != self._polygon_points[-1]:
                    self._polygon_points.append(self._polygon_points[0])
                self.polygon_drawn.emit(self._polygon_points.copy())

            if self._polygon_path_item is not None:
                self._scene.removeItem(self._polygon_path_item)
                self._polygon_path_item = None

            self._polygon_points = []

        # Paint mode release - allow multiple strokes before segmentation
        # Works for BOTH left and right button releases
        if (event.button() == Qt.LeftButton or event.button() == Qt.RightButton) and self._painting:
            self._painting = False

            # Clear the path item so we can start a new stroke
            if self._paint_path_item is not None:
                self._scene.removeItem(self._paint_path_item)
                self._paint_path_item = None

            # Keep paint_points for bounding box calculation
            # They'll be cleared when segmentation runs

        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        """Show eraser preview when mouse enters canvas"""
        if self._eraser_enabled and self._eraser_preview is not None:
            self._eraser_preview.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide eraser preview when mouse leaves canvas"""
        if self._eraser_preview is not None:
            self._eraser_preview.setVisible(False)
        super().leaveEvent(event)

    def _handle_click(self, scene_pos: QPointF, is_left: bool):
        x, y = scene_pos.x(), scene_pos.y()
        if x < 0 or y < 0:
            return
        if is_left:
            self._on_left_click(x, y)
        else:
            self._on_right_click(x, y)

    def _init_polygon_path_item(self):
        if self._polygon_path_item is None:
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            self._polygon_path_item = self._scene.addPath(QPainterPath(), pen)

    def _update_polygon_path(self):
        if not self._polygon_points or self._polygon_path_item is None:
            return

        path = QPainterPath()
        first_x, first_y = self._polygon_points[0]
        path.moveTo(first_x, first_y)
        for x, y in self._polygon_points[1:]:
            path.lineTo(x, y)

        self._polygon_path_item.setPath(path)

    def _init_paint_path_item(self):
        if self._paint_path_item is None:
            # Green for positive, Red for negative
            color = QColor(255, 0, 0) if self._paint_is_negative else QColor(0, 255, 0)
            pen = QPen(color)
            pen.setWidth(self._brush_size)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            self._paint_path_item = self._scene.addPath(QPainterPath(), pen)

    def _update_paint_path(self):
        if not self._paint_points or self._paint_path_item is None:
            return

        path = QPainterPath()
        first_x, first_y = self._paint_points[0]
        path.moveTo(first_x, first_y)
        for x, y in self._paint_points[1:]:
            path.lineTo(x, y)

        self._paint_path_item.setPath(path)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R and self._eraser_enabled:
            self._resizing_eraser = True
            self._resize_anchor = None
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_R:
            self._resizing_eraser = False
            self._resize_anchor = None
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _emit_erase(self, event):
        pos = self.mapToScene(event.pos())
        self.eraser_move.emit(int(pos.x()), int(pos.y()))
