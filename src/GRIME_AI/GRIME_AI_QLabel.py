from PyQt5 import Qt
from PyQt5.QtCore import QRect, QPoint, Qt
from PyQt5.QtGui import QPen, QBrush, QPainter, QPainterPath, QPolygon, QPolygonF
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QToolTip
from GRIME_AI.GRIME_AI_roiData import ROIShape

from enum import Enum

class DrawingMode(Enum):
    OFF                 = 0
    COLOR_SEGMENTATION  = 1
    MASK                = 2
    SLICE               = 3

class GRIME_AI_QLabel(QLabel):

    def __init__(self, parent=None):
        QLabel.__init__(self, parent=parent)

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.savedROIs = []  # persistent storage for drawn ROIs

        self.x0 = -1
        self.y0 = -1
        self.x1 = -1
        self.y1 = -1

        self.rect = QRect(self.x0, self.y0, self.x1, self.y1)

        self.flag = False

        self.shape = ROIShape.RECTANGLE

        self.drawingMode = DrawingMode.OFF
        self.enableFill = False

        self.path = QPainterPath()
        self.points = QPolygon()

        self.brushColor = Qt.green

        self.polygonList = []

        self.setWindowTitle("Slice Position")

        self.sliceCenter = int(self.size().width() / 2)
        self.sliceWidth = int(10)

        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover, True)
        self.tooltipGenerator = None  # Set this to a callable that returns the tooltip string.

        # Slice interaction state
        self._sliceHitMargin = 6          # px tolerance to hit a magenta edge
        self._minSliceWidth = 4           # px minimum
        self._draggingSlice = False
        self._resize = False              # False: move, True: resize
        self._lastMouseX = None

    def mousePressEvent(self, event):
        if self.drawingMode == DrawingMode.SLICE:
            x = event.x()
            left_edge = int(self.sliceCenter - self.sliceWidth // 2)
            right_edge = int(self.sliceCenter + self.sliceWidth // 2)

            hit_left = abs(x - left_edge) <= self._sliceHitMargin
            hit_right = abs(x - right_edge) <= self._sliceHitMargin

            if hit_left or hit_right:
                self._draggingSlice = True
                self._resize = (event.button() == Qt.RightButton)
                self._lastMouseX = x
                event.accept()
                return

        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

        if self.drawingMode == DrawingMode.MASK:
            self.points << event.pos()

        self.update()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.flag = False

    def mouseReleaseEvent(self, event):
        if self._draggingSlice:
            self._draggingSlice = False
            self._resize = False
            self._lastMouseX = None
            self.update()
            event.accept()
            return

        if self.flag:
            roi = self.getROI()
            if roi:
                self.savedROIs.append(roi)
        self.flag = False
        # Clear live coords so we don't redraw the just-saved ROI
        self.x0 = self.y0 = self.x1 = self.y1 = -1
        self.update()

    def mouseMoveEvent(self, event):
        if self.drawingMode == DrawingMode.SLICE and self._draggingSlice:
            x = event.x()
            w = self.size().width()
            dx = x - (self._lastMouseX if self._lastMouseX is not None else x)
            self._lastMouseX = x

            if self._resize:
                new_half = abs(x - self.sliceCenter)
                new_width = int(2 * new_half)
                # Clamp to bounds and min width
                new_width = max(self._minSliceWidth, new_width)
                left_edge = self.sliceCenter - new_width // 2
                right_edge = self.sliceCenter + new_width // 2
                if left_edge < 0:
                    new_width -= int(2 * (-left_edge))
                if right_edge > w:
                    new_width -= int(2 * (right_edge - w))
                self.sliceWidth = max(self._minSliceWidth, new_width)
            else:
                self.sliceCenter = max(0, min(w, self.sliceCenter + dx))

            self.update()
            event.accept()
            return

        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.drawingMode == DrawingMode.OFF:
            return

        painter = QPainter(self)

        # Always redraw any saved ROIs (persistent storage)
        if hasattr(self, "savedROIs"):
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            for roi in self.savedROIs:
                if self.getROIShape() == ROIShape.RECTANGLE:
                    painter.drawRect(roi)
                elif self.getROIShape() == ROIShape.ELLIPSE:
                    painter.drawEllipse(roi)

        # Draw the ROI currently being dragged
        if self.drawingMode == DrawingMode.COLOR_SEGMENTATION:
            self.drawColorSegmentationROI(painter)
        elif self.drawingMode == DrawingMode.MASK:
            self.drawPolygon(painter)
        elif self.drawingMode == DrawingMode.SLICE:
            self.drawCompositeSlice(painter, self.sliceCenter)

    def enterEvent(self, event):
        if self.tooltipGenerator and callable(self.tooltipGenerator):
            try:
                tooltip_text = self.tooltipGenerator()
            except Exception as e:
                tooltip_text = f"Error retrieving tooltip: {e}"
            self.setToolTip(tooltip_text)
            self.showToolTip(event.globalPos())  # Force tooltip display
        super(GRIME_AI_QLabel, self).enterEvent(event)

    def showToolTip(self, global_pos):
        QToolTip.showText(global_pos, self.toolTip(), self)

    def drawCompositeSlice(self, painter, sliceCenter):
        labelSize = self.size()

        # Keep internal state in sync
        self.sliceCenter = sliceCenter

        painter.setPen(QPen(Qt.red, 1, Qt.SolidLine))
        painter.drawLine(int(self.sliceCenter), 0, int(self.sliceCenter), labelSize.height())

        painter.setPen(QPen(Qt.magenta, 1, Qt.SolidLine))
        left_x = int(self.sliceCenter - self.sliceWidth // 2)
        right_x = int(self.sliceCenter + self.sliceWidth // 2)
        painter.drawLine(left_x, 0, left_x, labelSize.height())
        painter.drawLine(right_x, 0, right_x, labelSize.height())

    def drawContinuous(self):
        if self.flag:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawLine(self.x0, self.y0, self.x1, self.y1)
            self.x0 = self.x1
            self.y0 = self.y1

    def drawColorSegmentationROI(self, painter):
        if self.flag:
            # Always normalize coordinates
            x = min(self.x0, self.x1)
            y = min(self.y0, self.y1)
            w = abs(self.x1 - self.x0)
            h = abs(self.y1 - self.y0)
            rect = QRect(x, y, w, h)

            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            if self.getROIShape() == ROIShape.RECTANGLE:
                painter.drawRect(rect)
            elif self.getROIShape() == ROIShape.ELLIPSE:
                painter.drawEllipse(rect)

    def drawPolygon(self, qp):
        qp.setRenderHint(QPainter.Antialiasing)

        pen = QPen(Qt.red, 1)
        qp.setPen(pen)

        brush = QBrush(self.brushColor)
        qp.setBrush(brush)

        lp = QPoint()
        for myPoint in self.points:
            cp = myPoint
            qp.drawEllipse(cp, 2, 2)
            if not lp.isNull():
                qp.drawLine(lp, cp)
            lp = cp

        if self.enableFill:
            # Fill polygon
            polyPath = QPainterPath()
            polyPath.addPolygon(QPolygonF(self.points))

            # Draw polygon
            qp.drawPolygon(QPolygonF(self.points))
            qp.fillPath(polyPath, brush)

        for myPolygon in self.polygonList:
            lp = QPoint()
            for myPoints in myPolygon:
                cp = myPoints
                qp.drawEllipse(cp, 2, 2)
                if not lp.isNull():
                    qp.drawLine(lp, cp)
                lp = cp

                if self.enableFill:
                    # Fill polygon
                    polyPath = QPainterPath()
                    polyPath.addPolygon(QPolygonF(myPoints))

                    # Draw polygon
                    qp.drawPolygon(QPolygonF(myPoints))
                    qp.fillPath(polyPath, brush)

    def getPolygon(self):
        return self.polygonList

    def incrementPolygon(self):
        self.polygonList.append(self.points)
        self.points.clear()
        self.polygonListCount = len(self.polygonList)

    def getROI(self):
        # Prefer live ROI if dragging
        if self.x0 != -1 and self.x1 != -1:
            x = min(self.x0, self.x1)
            y = min(self.y0, self.y1)
            w = abs(self.x1 - self.x0)
            h = abs(self.y1 - self.y0)
            return QRect(x, y, w, h)

        # Fall back to last saved ROI
        if self.savedROIs:
            return self.savedROIs[-1]

        return None

    def setROIShape(self, shape):
        self.shape = shape

    def getROIShape(self):
        return self.shape

    def setBrushColor(self, brushColor):
        self.brushColor = brushColor

    def resetMask(self):
        if self.points.count() > 0:
            self.points = QPolygon()

    def setDrawingMode(self, mode):
        self.drawingMode = mode

    def enablePolygonFill(self, bFill):
        self.enableFill = bFill

    def setSliceCenter(self, sliceCenter):
        # Clamp and repaint automatically
        self.sliceCenter = max(0, min(self.width(), int(sliceCenter)))
        self.update()  # ensure immediate redraw

    def getSliceCenter(self):
        return self.sliceCenter

    def setSliceWidth(self, sliceWidth):
        # Clamp and repaint automatically (fix indentation bug)
        self.sliceWidth = max(self._minSliceWidth, min(self.width(), int(sliceWidth)))
        self.update()  # ensure immediate redraw

    def getSliceWidth(self):
        return self.sliceWidth

    def setROIs(self, roi_list):
        self.savedROIs = [roi.getDisplayROI() for roi in roi_list]
        self.update()

    def clearROIs(self):
        self.savedROIs.clear()
        self.x0 = self.y0 = self.x1 = self.y1 = -1
        self.flag = False
        self.update()
