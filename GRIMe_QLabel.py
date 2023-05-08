from PyQt5 import Qt
from PyQt5.QtCore import QRect, QPoint, Qt
from PyQt5.QtGui import QPen, QBrush, QPainter, QPainterPath, QPolygon, QPolygonF
from PyQt5.QtWidgets import QLabel, QVBoxLayout

from GRIMe_roiData import ROIShape

from enum import Enum

class DrawingMode(Enum):
    OFF                 = 0
    COLOR_SEGMENTATION  = 1
    MASK                = 2

# ======================================================================================================================
#
# ======================================================================================================================
class GRIMe_QLabel(QLabel):
    x0 = -1
    y0 = -1
    x1 = -1
    y1 = -1

    rect = QRect(x0, y0, x1, y1)

    flag = False

    shape = ROIShape.RECTANGLE

    drawingMode = DrawingMode.OFF
    enableFill = False

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None):
        QLabel.__init__(self, parent=parent)
        lay = QVBoxLayout(self)
        self.path = QPainterPath()
        self.points = QPolygon()

        self.brushColor = Qt.green

        self.polygonList = []

    # ------------------------------------------------------------------------------------------------------------------------
    # Mouse click event
    # ------------------------------------------------------------------------------------------------------------------------
    def mousePressEvent(self, event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

        if self.drawingMode == DrawingMode.MASK:
            self.points << event.pos()

        self.update()

        super().mousePressEvent(event)

        # DIAGNOSTICS
        #ss = 'Label Pos - X:' + str(self.x0).rjust(4) + ' Y:' + str(self.y0).rjust(4)
        #print(ss)

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def mouseDoubleClickEvent(self, event):
        self.flag = False

    # ------------------------------------------------------------------------------------------------------------------------
    # Mouse release event
    # ------------------------------------------------------------------------------------------------------------------------
    def mouseReleaseEvent(self, event):
        self.flag = False

    # ------------------------------------------------------------------------------------------------------------------------
    # Mouse movement events
    # ------------------------------------------------------------------------------------------------------------------------
    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    # ------------------------------------------------------------------------------------------------------------------------
    # Draw events
    # ------------------------------------------------------------------------------------------------------------------------
    def paintEvent(self, event):
        super().paintEvent(event)

        if self.drawingMode == DrawingMode.COLOR_SEGMENTATION:
            self.drawColorSegmentationROI()
        elif self.drawingMode == DrawingMode.MASK:
            self.drawPolygon()

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def drawContinuous(self):
        if self.flag:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawLine(self.x0, self.y0, self.x1, self.y1)
            self.x0 = self.x1
            self.y0 = self.y1

# ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def drawColorSegmentationROI(self):
        if self.flag:
            rect = QRect(self.x0, self.y0, (self.x1 - self.x0), (self.y1 - self.y0))
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            if self.getROIShape() == ROIShape.RECTANGLE:
                painter.drawRect(rect)
            elif self.getROIShape() == ROIShape.ELLIPSE:
                painter.drawEllipse(self.x0, self.y0, (self.x1 - self.x0), (self.y1 - self.y0))

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    def drawPolygon(self):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)

        pen = QPen(Qt.red, 1)
        qp.setPen(pen)

        brush = QBrush(self.brushColor)
        qp.setBrush(brush)

        # DRAW THE CURRENT POLYGON
        # ----------------------------------------------------------------------------------------------------
        lp = QPoint()
        for myPoint in self.points:
            cp = myPoint
            qp.drawEllipse(cp, 2, 2)
            if not lp.isNull():
                qp.drawLine(lp, cp)
            lp = cp

        # FILL THE POLYGON CURRENTLY BEING DRAWN
        # ----------------------------------------------------------------------------------------------------
        if self.enableFill:
            # Fill polygon
            polyPath = QPainterPath()
            polyPath.addPolygon(QPolygonF(self.points))

            # Draw polygon
            qp.drawPolygon(QPolygonF(self.points))
            qp.fillPath(polyPath, brush)

        # ----------------------------------------------------------------------------------------------------
        # DRAW ANY POLYGONS THAT HAVE BEEN PREVIOUSLY ADDED
        # ----------------------------------------------------------------------------------------------------
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
                polyPath.addPolygon(myPoints)

                # Draw polygon
                qp.drawPolygon(myPoints)
                qp.fillPath(polyPath, brush)

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def getPolygon(self):
        return(self.polygonList)

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def incrementPolygon(self):
        self.polygonList.append(self.points)
        del self.points
        self.points = QPolygon()

        self.polygonListCount = len(self.polygonList)

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def getROI(self):
        if self.x0 == -1:
            return None
        else:
            if self.x0 > self.x1:
                return QRect(self.x1, self.y1, self.x0-self.x1, self.y0-self.y1)
            else:
                return QRect(self.x0, self.y0, self.x1-self.x0, self.y1-self.y0)

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def setROIShape(self, shape):
        self.shape = shape

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def getROIShape(self):
        return self.shape

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def setBrushColor(self, brushColor):
        self.brushColor = brushColor

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def resetMask(self):
        if self.points.count() > 0:
            del self.points
            self.points = QPolygon()

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def setDrawingMode(self, mode):
        self.drawingMode = mode

    def enablePolygonFill(self, bFill):
        self.enableFill = bFill