from PyQt5 import Qt
from PyQt5.QtCore import QRect, QPoint, Qt
from PyQt5.QtGui import QPen, QBrush, QPainter, QPainterPath, QPolygon, QPolygonF
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QToolTip
from GRIME_AI_roiData import ROIShape

from enum import Enum

class DrawingMode(Enum):
    OFF                 = 0
    COLOR_SEGMENTATION  = 1
    MASK                = 2
    SLICE               = 3

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_QLabel(QLabel):
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
        layout = QVBoxLayout(self)
        self.setLayout(layout)

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
        elif self.drawingMode == DrawingMode.SLICE:
            self.drawCompositeSlice(self.sliceCenter)

    def enterEvent(self, event):
        """
        When the mouse enters the label's area, update its tooltip using the instance-specific tooltip generator.
        """
        if self.tooltipGenerator and callable(self.tooltipGenerator):
            try:
                tooltip_text = self.tooltipGenerator()
            except Exception as e:
                tooltip_text = f"Error retrieving tooltip: {e}"
            self.setToolTip(tooltip_text)
            self.showToolTip(event.globalPos())  # Force tooltip display
        super(GRIME_AI_QLabel, self).enterEvent(event)


    def showToolTip(self, global_pos):
        """Manually display the tooltip at the cursor location."""
        QToolTip.showText(global_pos, self.toolTip(), self)


    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def drawCompositeSlice(self, sliceCenter):
        painter = QPainter(self)

        labelSize = self.size()

        self.sliceCenter = sliceCenter

        painter.setPen(QPen(Qt.red, 1, Qt.SolidLine))
        painter.drawLine(int(self.sliceCenter), 0, int(self.sliceCenter), labelSize.height())

        painter.setPen(QPen(Qt.magenta, 1, Qt.SolidLine))
        painter.drawLine(int(self.sliceCenter - self.sliceWidth // 2), 0, int(self.sliceCenter - self.sliceWidth // 2), labelSize.height())
        painter.drawLine(int(self.sliceCenter + self.sliceWidth // 2), 0, int(self.sliceCenter + self.sliceWidth // 2), labelSize.height())

        self.update()


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
                    polyPath.addPolygon(QPolygonF(myPoints))

                    # Draw polygon
                    qp.drawPolygon(QPolygonF(myPoints))
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

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def setSliceCenter(self, sliceCenter):
        self.sliceCenter = sliceCenter

    def getSliceCenter(self):
        return(self.sliceCenter)

    def setSliceWidth(self, sliceWidth):
        self.sliceWidth = sliceWidth

    def getSliceWidth(self):
        return(self.sliceWidth)
