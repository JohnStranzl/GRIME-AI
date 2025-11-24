#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QDialog, QWidget
from PyQt5.uic import loadUi
from GRIME_AI.constants import edgeMethodsClass, featureMethodsClass
import os

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_EdgeDetectionDlg(QDialog):

    # SIGNALS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    edgeDetectionSignal = pyqtSignal(edgeMethodsClass)
    featureDetectionSignal = pyqtSignal(featureMethodsClass)

    returnEdgeData = edgeMethodsClass()
    returnFeatureData = featureMethodsClass()

    # -----------------------------------------------------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        loadUi(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ui','QDialog_EdgeDetection.ui'), self)

        # CONNECT THE SIGNALS TO THE FUNCTIONS IN THE PARENT ("CALLING") THREAD THAT WILL RECEIVE THE SIGNAL
        self.featureDetectionSignal.connect(parent.featureDetectionMethod)
        self.edgeDetectionSignal.connect(parent.edgeDetectionMethod)

        # CONNECT THE WIDGETS TO THE FUNCTIONS IN THIS CLASS THAT WILL GET INVOKED
        self.radioButtonCanny.clicked.connect(self.clicked_Canny)
        self.spinBoxCannyKernel.valueChanged.connect(self.spinBoxCannyKernelChanged)
        self.spinBoxCannyHighThreshold.valueChanged.connect(self.spinBoxCannyHighThresholdChanged)
        self.spinBoxCannyLowThreshold.valueChanged.connect(self.spinBoxCannyLowThresholdChanged)

        self.radioButtonSobelX.clicked.connect(self.clicked_SobelX)
        self.radioButtonSobelY.clicked.connect(self.clicked_SobelY)
        self.radioButtonSobelXY.clicked.connect(self.clicked_SobelXY)
        self.spinBoxSobelKernel.valueChanged.connect(self.spinBoxSobelKernelChanged)

        self.radioButtonLaplacian.clicked.connect(self.clicked_Laplacian)

        self.radioButtonSIFT.clicked.connect(self.clicked_SIFT)
        self.radioButtonORB.clicked.connect(self.clicked_ORB)
        self.spinBoxOrbMaxFeatures.valueChanged.connect(self.spinBoxOrbMaxFeaturesChanged)

        self.returnEdgeData.canny_threshold_high = self.spinBoxCannyHighThreshold.value()
        self.returnEdgeData.canny_threshold_low = self.spinBoxCannyLowThreshold.value()
        self.returnEdgeData.sobelKernel = self.spinBoxSobelKernel.value()

        #self.spinBoxCannyHighThreshold.setKeyboardTracking(False)
        #self.spinBoxCannyLowThreshold.setKeyboardTracking(False)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def clicked_Canny(self):
        self.returnFeatureData.method = featureMethodsClass.NONE

        self.returnEdgeData.method = edgeMethodsClass.CANNY
        self.returnEdgeData.selected = self.radioButtonCanny.isChecked()

        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def clicked_SobelX(self):
        self.returnFeatureData.method = featureMethodsClass.NONE

        self.returnEdgeData.method = edgeMethodsClass.SOBEL_X
        self.returnEdgeData.selected = self.radioButtonSobelX.isChecked()

        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def clicked_SobelY(self):
        self.returnFeatureData.method = featureMethodsClass.NONE

        self.returnEdgeData.method = edgeMethodsClass.SOBEL_Y
        self.returnEdgeData.selected = self.radioButtonSobelY.isChecked()

        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def clicked_SobelXY(self):
        self.returnFeatureData.method = featureMethodsClass.NONE

        self.returnEdgeData.method = edgeMethodsClass.SOBEL_XY
        self.returnEdgeData.selected = self.radioButtonSobelXY.isChecked()

        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxCannyHighThresholdChanged(self):
        self.returnEdgeData.canny_threshold_high = self.spinBoxCannyHighThreshold.value()
        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxCannyLowThresholdChanged(self):
        self.returnEdgeData.canny_threshold_low = self.spinBoxCannyLowThreshold.value()
        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxCannyKernelChanged(self):
        #imageNumber = self.spinBoxDailyImage.value()
        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxSobelKernelChanged(self):
        self.returnEdgeData.sobelKernel = self.spinBoxSobelKernel.value()
        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def clicked_Laplacian(self):
        self.returnFeatureData.method = featureMethodsClass.NONE

        self.returnEdgeData.method = edgeMethodsClass.LAPLACIAN
        self.returnEdgeData.selected = self.radioButtonLaplacian.isChecked()

        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def clicked_SIFT(self):
        self.returnEdgeData.method = edgeMethodsClass.NONE

        self.returnFeatureData.method = featureMethodsClass.SIFT
        self.returnFeatureData.selected = self.radioButtonSIFT.isChecked()

        self.featureDetectionSignal.emit(self.returnFeatureData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def clicked_ORB(self):
        self.returnEdgeData.method = edgeMethodsClass.NONE

        self.returnFeatureData.method = featureMethodsClass.ORB
        self.returnFeatureData.selected = self.radioButtonORB.isChecked()

        self.featureDetectionSignal.emit(self.returnFeatureData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxOrbMaxFeaturesChanged(self):
        self.returnFeatureData.orbMaxFeatures = self.spinBoxOrbMaxFeaturesChanged().value()
        self.featureDetectionSignal.emit(self.returnFeatureData)

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def onCancel(self):
        self.close()
