#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from GRIME_AI.utils.resource_utils import ui_path

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi


# ======================================================================================================================
class roiParameters:
    def __init__(self, parent=None):
        self.strROIName       = ''
        self.numColorClusters = 4
        self.bDisplayROIs     = True
        self.bDisplayROIColors = True


# ======================================================================================================================
class GRIME_AI_ColorSegmentationDlg(QDialog):

    # ------------------------------------------------------------------------------------------------------------------
    # SIGNALS
    # ------------------------------------------------------------------------------------------------------------------
    colorSegmentation_Signal   = pyqtSignal(int)
    addROI_Signal              = pyqtSignal(roiParameters)
    deleteAllROI_Signal        = pyqtSignal()
    close_signal               = pyqtSignal()
    buildFeatureFile_Signal    = pyqtSignal()
    universalTestButton_Signal = pyqtSignal(int)
    greenness_index_signal     = pyqtSignal()
    refresh_rois_signal        = pyqtSignal(roiParameters)

    returnROIParameters = roiParameters()

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super(GRIME_AI_ColorSegmentationDlg, self).__init__(parent)
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        loadUi(ui_path("color_segmentation/QDialog_ColorSegmentation.ui"), self)

        self.setMinimumSize(500, 320)
        self.resize(620, 380)

        # ------------------------------------------------------------------
        # ROI CONTROLS
        # ------------------------------------------------------------------
        self.pushButtonAddROI.clicked.connect(self.addROI)
        self.pushButton_deleteAllROIs.clicked.connect(self.deleteAllROI)
        self.buttonBox_Close.clicked.connect(self.closeClicked)
        self.pushButton_Dlg_BuildFeatureFile.clicked.connect(self.buildFeatureFile)
        self.spinBoxColorClusters.valueChanged[int].connect(self.colorClusterValueChanged)

        # ------------------------------------------------------------------
        # TEXTURE TOGGLE
        # ------------------------------------------------------------------
        self.checkBox_Texture.toggled.connect(self._on_texture_toggled)
        self.groupBox_Texture.setEnabled(self.checkBox_Texture.isChecked())

        # ------------------------------------------------------------------
        # GREENNESS INDEX
        # ------------------------------------------------------------------
        self.checkBox_GCC.clicked.connect(self.GCC_Clicked)
        self.checkBox_GLI.clicked.connect(self.GLI_Clicked)
        self.checkBox_ExG.clicked.connect(self.ExG_Clicked)
        self.checkBox_RGI.clicked.connect(self.RGI_Clicked)
        self.checkBox_NDVI.clicked.connect(self.NDVI_Clicked)

        # ------------------------------------------------------------------
        # DEVELOPER-ONLY TEST BUTTON
        # ------------------------------------------------------------------
        import getpass
        if getpass.getuser() in ("johns", "tgilmore10"):
            self.pushButton_Dlg_TEST.clicked.connect(self.universalTestButton)
        else:
            self.pushButton_Dlg_TEST.setEnabled(False)
            self.pushButton_Dlg_TEST.hide()

        # ------------------------------------------------------------------
        # STYLING
        # ------------------------------------------------------------------
        self.pushButton_Dlg_BuildFeatureFile.setStyleSheet(
            'QPushButton {background-color: steelblue; color: yellow;}'
        )

    # ------------------------------------------------------------------
    def _on_texture_toggled(self, checked):
        """Enable/disable texture sub-groupbox when Texture checkbox is toggled."""
        self.groupBox_Texture.setEnabled(checked)

    # ------------------------------------------------------------------
    def get_texture_options(self) -> dict:
        """
        Returns which texture methods are selected.
        Call this from buildFeatureFile before passing options to ExtractFeatures.
        """
        return {
            'enabled': self.checkBox_Texture.isChecked(),
            'glcm':    self.checkBox_Texture_GLCM.isChecked(),
            'gabor':   self.checkBox_Texture_Gabor.isChecked(),
            'lbp':     self.checkBox_Texture_LBP.isChecked(),
            'wavelet': self.checkBox_Texture_Wavelet.isChecked(),
            'fourier': self.checkBox_Texture_Fourier.isChecked(),
        }

    # ------------------------------------------------------------------
    def colorClusterValueChanged(self):
        self.returnROIParameters.numColorClusters = self.spinBoxColorClusters.value()
        self.refresh_rois_signal.emit(self.returnROIParameters)

    def buildFeatureFile(self):
        self.buildFeatureFile_Signal.emit()

    def universalTestButton(self):
        self.universalTestButton_Signal.emit(1)

    def closeEvent(self, event):
        super(GRIME_AI_ColorSegmentationDlg, self).closeEvent(event)
        self.close_signal.emit()

    def closeClicked(self):
        self.close_signal.emit()

    def colorSegmentationClicked(self):
        self.colorSegmentation_Signal.emit(1)

    def addROI(self):
        self.returnROIParameters.strROIName        = self.lineEdit_roiName.text()
        self.returnROIParameters.numColorClusters  = self.spinBoxColorClusters.value()
        self.returnROIParameters.bDisplayROIs      = True
        self.returnROIParameters.bDisplayROIColors = True
        self.addROI_Signal.emit(self.returnROIParameters)

    def deleteAllROI(self):
        self.deleteAllROI_Signal.emit()

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

    def disable_spinbox_color_clusters(self, disable_spinbox=True):
        self.spinBoxColorClusters.setDisabled(disable_spinbox)

    def get_num_color_clusters(self):
        return self.spinBoxColorClusters.value()
