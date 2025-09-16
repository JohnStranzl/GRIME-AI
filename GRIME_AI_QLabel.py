#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

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

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
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
        if self.flag:
            roi = self.getROI()
            if roi:
                self.savedROIs.append(roi)
        self.flag = False
        # Clear live coords so we don't redraw the just-saved ROI
        self.x0 = self.y0 = self.x1 = self.y1 = -1
        self.update()

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

    # ------------------------------------------------------------------------------------------------------------------------
    #
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
            # Always normalize coordinates
            x = min(self.x0, self.x1)
            y = min(self.y0, self.y1)
            w = abs(self.x1 - self.x0)
            h = abs(self.y1 - self.y0)
            rect = QRect(x, y, w, h)

            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            if self.getROIShape() == ROIShape.RECTANGLE:
                painter.drawRect(rect)
            elif self.getROIShape() == ROIShape.ELLIPSE:
                painter.drawEllipse(rect)

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
        self.points.clear()
        self.polygonListCount = len(self.polygonList)

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
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

    def setROIs(self, roi_list):
        def setROIs(self, roi_list):
            # Extract the display QRect from each ROI object
            self.savedROIs = [roi.getDisplayROI() for roi in roi_list]
            self.update()
            
    def clearROIs(self):
        self.savedROIs.clear()

        self.x0 = self.y0 = self.x1 = self.y1 = -1

        self.flag = False

        self.update()
