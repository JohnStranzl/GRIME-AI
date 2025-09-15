#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

# ====================================================================================================
#
# ====================================================================================================
class roiParameters:
    def __init__(self, parent=None):
        self.strROIName = ''
        self.numColorClusters = 4
        self.bDisplayROIs = True
        self.bDisplayROIColors = True

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_ColorSegmentationDlg(QDialog):

    # SIGNALS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    colorSegmentation_Signal = pyqtSignal(int)
    addROI_Signal = pyqtSignal(roiParameters)
    deleteAllROI_Signal = pyqtSignal()
    close_signal = pyqtSignal()
    buildFeatureFile_Signal = pyqtSignal()
    universalTestButton_Signal = pyqtSignal(int)
    greenness_index_signal = pyqtSignal()

    returnROIParameters = roiParameters()

    # -----------------------------------------------------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super(GRIME_AI_ColorSegmentationDlg, self).__init__(parent)
        # Removed: layout = QVBoxLayout(self)
        # SET BEHAVIOR OF DIALOG BOX
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        # Load the UI file (which now contains dynamic layouts)
        loadUi('QDialog_ColorSegmentation.ui', self)
        # Ensure the dialog starts with the intended size
        self.resize(400, 676)

        # CONNECT CONTROLS/WIDGETS TO FUNCTIONS THAT RESPOND TO CLICKS
        self.pushButtonAddROI.clicked.connect(self.addROI)
        self.pushButton_deleteAllROIs.clicked.connect(self.deleteAllROI)
        self.buttonBox_Close.clicked.connect(self.closeClicked)
        self.pushButton_Dlg_BuildFeatureFile.clicked.connect(self.buildFeatureFile)

        import getpass
        if getpass.getuser() == "johns" or getpass.getuser() == "tgilmore10":
            self.pushButton_Dlg_TEST.clicked.connect(self.universalTestButton)
        else:
            self.pushButton_Dlg_TEST.setEnabled(False)
            self.pushButton_Dlg_TEST.hide()

        # GREENNESS INDEX SELECTIONS
        self.checkBox_GCC.clicked.connect(self.GCC_Clicked)
        self.checkBox_GLI.clicked.connect(self.GLI_Clicked)
        self.checkBox_ExG.clicked.connect(self.ExG_Clicked)
        self.checkBox_RGI.clicked.connect(self.RGI_Clicked)
        self.checkBox_NDVI.clicked.connect(self.NDVI_Clicked)

        # SET CONTROL COLORS
        self.pushButton_Dlg_BuildFeatureFile.setStyleSheet(
            'QPushButton {background-color: steelblue; color: yellow;}'
        )

    # ----------------------------------------------------------------------------------------------------
    def buildFeatureFile(self):
        self.buildFeatureFile_Signal.emit()

    # ----------------------------------------------------------------------------------------------------
    def universalTestButton(self):
        self.universalTestButton_Signal.emit(1)

    # ----------------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        super(GRIME_AI_ColorSegmentationDlg, self).closeEvent(event)
        self.close_signal.emit()

    def closeClicked(self):
        self.close_signal.emit()

    # ----------------------------------------------------------------------------------------------------
    def colorSegmentationClicked(self):
        self.colorSegmentation_Signal.emit(1)

    # ----------------------------------------------------------------------------------------------------
    def addROI(self):
        self.returnROIParameters.strROIName = self.lineEdit_roiName.text()
        # Optionally, you could read the actual spinbox value like:
        # self.returnROIParameters.numColorClusters = self.spinBoxColorClusters.value()
        self.returnROIParameters.numColorClusters = 4
        self.returnROIParameters.bDisplayROIs = True
        self.returnROIParameters.bDisplayROIColors = True

        self.addROI_Signal.emit(self.returnROIParameters)

    # ----------------------------------------------------------------------------------------------------
    def deleteAllROI(self):
        self.deleteAllROI_Signal.emit()

    # ----------------------------------------------------------------------------------------------------
    def GCC_Clicked(self):
        self.greenness_index_signal.emit()

    def GLI_Clicked(self):
        self.greenness_index_signal.emit()

    def ExG_Clicked(self):
        self.greenness_index_signal.emit()

    def RGI_Clicked(self):
        self.greenness_index_signal.emit()

    def NDVI_Clicked(self):
        self.greenness_index_signal.emit()
