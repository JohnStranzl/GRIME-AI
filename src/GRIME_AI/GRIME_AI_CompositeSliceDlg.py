#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
import cv2
import os

class GRIME_AI_CompositeSliceDlg(QtWidgets.QDialog):
    compositeSliceCancelSignal = pyqtSignal()
    compositeSliceGenerateSignal = pyqtSignal()

    def __init__(self, parent=None):
        super(GRIME_AI_CompositeSliceDlg, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(Qt.NonModal)

        loadUi(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ui','QDialog_CompositeSlice.ui'), self)

        self.widthMultiplier = 0
        self.heightMultiplier = 0

        # Initialize from UI
        self.sliceCenter = int(self.horizontalSlider.value())
        self.sliceWidth = int(self.spinBox_Width.value())
        self.lineEdit_HorizontalPosition.setText(str(self.sliceCenter))

        # Push initial values into label (label setters trigger repaint)
        self.label_Image.setSliceCenter(self.sliceCenter)
        self.label_Image.setSliceWidth(self.sliceWidth)

        # Connect signals with correct signatures
        self.horizontalSlider.valueChanged.connect(self.valuechange)          # int -> slot(int)
        self.spinBox_Width.valueChanged.connect(self.spinBox_WidthChanged)    # int -> slot(int)

        self.pushButton_Generate.clicked.connect(self.pushButton_Generate_Clicked)
        self.pushButton_Generate.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

        self.pushButton_Cancel.clicked.connect(self.pushButton_Cancel_Clicked)

        # Remove monkey-patching of label mouse events.
        # The label class already implements slice dragging/resizing.
        # If you need dialog-side coordination, use signals from the label instead.

        self.dragStartPosition = None
        self.draggingCenter = False
        self.draggingLeftEdge = False
        self.draggingRightEdge = False

    def loadImage(self, filename):
        numpyImage_bgr = cv2.imread(filename)
        if numpyImage_bgr is None:
            return
        numpyImage_rgb = cv2.cvtColor(numpyImage_bgr, cv2.COLOR_BGR2RGB)
        myImage = QImage(numpyImage_rgb, numpyImage_rgb.shape[1], numpyImage_rgb.shape[0], QImage.Format_RGB888)
        myImage = QPixmap(myImage)

        # Compute multipliers relative to current label size
        self.widthMultiplier = myImage.width() / max(1, self.label_Image.width())
        self.heightMultiplier = myImage.height() / max(1, self.label_Image.height())

        # Show scaled image
        self.label_Image.setPixmap(myImage.scaled(self.label_Image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Update control ranges now that the label size is known
        max_center = max(0, self.label_Image.width())
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(max_center)
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.setPageStep(5)

        self.spinBox_Width.setMinimum(self.label_Image._minSliceWidth)
        self.spinBox_Width.setMaximum(max(1, self.label_Image.width()))
        self.spinBox_Width.setSingleStep(2)

        # Clamp current values to new ranges and push to label
        self.sliceCenter = max(0, min(max_center, self.sliceCenter))
        self.sliceWidth = max(self.label_Image._minSliceWidth, min(self.label_Image.width(), self.sliceWidth))
        self.horizontalSlider.setValue(self.sliceCenter)
        self.spinBox_Width.setValue(self.sliceWidth)
        self.label_Image.setSliceCenter(self.sliceCenter)
        self.label_Image.setSliceWidth(self.sliceWidth)

    def getMultipliers(self):
        return self.widthMultiplier, self.heightMultiplier, self.sliceCenter, self.sliceWidth

    def spinBox_WidthChanged(self, value: int):
        # Update width via label setter; repaint is automatic
        self.sliceWidth = int(value)
        self.label_Image.setSliceWidth(self.sliceWidth)

    def valuechange(self, value: int):
        # Update center via label setter; repaint is automatic
        self.sliceCenter = int(value)
        self.label_Image.setSliceCenter(self.sliceCenter)
        self.lineEdit_HorizontalPosition.setText(str(self.sliceCenter))

    def closeEvent(self, event):
        super(GRIME_AI_CompositeSliceDlg, self).closeEvent(event)

    def getSliceCenter(self):
        return self.sliceCenter

    def pushButton_Generate_Clicked(self):
        self.compositeSliceGenerateSignal.emit()

    def pushButton_Cancel_Clicked(self):
        self.compositeSliceCancelSignal.emit()
        self.close()

    def onCancel(self):
        self.compositeSliceCancelSignal.emit()
        self.close()

    # If you decide to keep dialog-driven dragging instead of label’s internal handling,
    # update the label only via setters (do NOT call drawCompositeSlice directly).
    # Leaving these here commented as reference.

    # def imageLabelMousePress(self, event):
    #     if event.buttons() == Qt.LeftButton:
    #         self.dragStartPosition = event.pos()
    #         self.draggingCenter = self.isWithinCenterLine(event.pos())
    #     elif event.buttons() == Qt.RightButton:
    #         self.dragStartPosition = event.pos()
   #         self.draggingLeftEdge = self.isWithinLeftEdgeLine(event.pos())
    #         self.draggingRightEdge = self.isWithinRightEdgeLine(event.pos())

    # def imageLabelMouseMove(self, event):
    #     if event.buttons() == Qt.LeftButton and self.draggingCenter:
    #         newCenter = self.sliceCenter + (event.pos().x() - self.dragStartPosition.x())
    #         self.horizontalSlider.setValue(int(newCenter))
    #         self.dragStartPosition = event.pos()
    #     elif event.buttons() == Qt.RightButton:
    #         if self.draggingLeftEdge:
    #             newWidth = (self.sliceCenter - event.pos().x()) * 2
    #             self.spinBox_Width.setValue(int(newWidth))
    #             self.dragStartPosition = event.pos()
    #         elif self.draggingRightEdge:
    #             newWidth = (event.pos().x() - self.sliceCenter) * 2
    #             self.spinBox_Width.setValue(int(newWidth))
    #             self.dragStartPosition = event.pos()

    # def imageLabelMouseRelease(self, event):
    #     self.draggingCenter = False
    #     self.draggingLeftEdge = False
    #     self.draggingRightEdge = False

    def isWithinCenterLine(self, pos):
        centerLineX = self.sliceCenter
        return abs(pos.x() - centerLineX) < 10  # 10 pixels tolerance

    def isWithinLeftEdgeLine(self, pos):
        leftEdgeX = self.sliceCenter - self.sliceWidth / 2
        return abs(pos.x() - leftEdgeX) < 10  # 10 pixels tolerance

    def isWithinRightEdgeLine(self, pos):
        rightEdgeX = self.sliceCenter + self.sliceWidth / 2
        return abs(pos.x() - rightEdgeX) < 10  # 10 pixels tolerance
