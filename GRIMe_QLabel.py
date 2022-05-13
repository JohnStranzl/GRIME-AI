from PyQt5 import Qt
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QPen, QPainter
from PyQt5.QtWidgets import QLabel, QVBoxLayout

from GRIMe_roiData import ROIShape


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

    def __init__(self, parent=None):
        QLabel.__init__(self, parent=parent)
        lay = QVBoxLayout(self)

    #Mouse click event
    def mousePressEvent(self, event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

        # DIAGNOSTICS
        #ss = 'Label Pos - X:' + str(self.x0).rjust(4) + ' Y:' + str(self.y0).rjust(4)
        #print(ss)

    def mouseDoubleClickEvent(self, event):
        self.flag = False

    #Mouse release event
    def mouseReleaseEvent(self, event):
        self.flag = False

    #Mouse movement events
    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    #Draw events
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.flag:
            rect = QRect(self.x0, self.y0, (self.x1-self.x0), (self.y1-self.y0))
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            if self.getROIShape() == ROIShape.RECTANGLE:
                painter.drawRect(rect)
            elif self.getROIShape() == ROIShape.ELLIPSE:
                painter.drawEllipse(self.x0, self.y0, (self.x1-self.x0), (self.y1-self.y0))

    def getROI(self):
        if self.x0 == -1:
            return None
        else:
            if self.x0 > self.x1:
                return QRect(self.x1, self.y1, self.x0-self.x1, self.y0-self.y1)
            else:
                return QRect(self.x0, self.y0, self.x1-self.x0, self.y1-self.y0)

    def setROIShape(self, shape):
        self.shape = shape

    def getROIShape(self):
        return self.shape
