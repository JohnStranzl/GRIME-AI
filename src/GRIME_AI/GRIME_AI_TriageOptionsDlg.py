#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from PyQt5.QtWidgets import QDialog

from PyQt5.uic import loadUi

import promptlib

from PyQt5 import QtGui

import os
# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_TriageOptionsDlg(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        loadUi(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ui','QDialog_TriageOptions.ui'), self)

        self.referenceImageFilename = ''

        self.pushButton_SelectReferenceImage.clicked.connect(self.selectReferenceImage)

        self.pushButton_SelectReferenceImage.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

    def selectReferenceImage(self):
        self.referenceImageFilename =  promptlib.Files().file()

    def getCreateReport(self):
        return self.checkBoxTriageImagesReport.isChecked()

    def getMoveImages(self):
        return self.checkBoxTriageImagesMove.isChecked()

    def getBlurThreshold(self):
        return self.doubleSpinBoxBlurThreshhold.value()

    def getShiftSize(self):
        return self.spinBoxShiftSize.value()

    def getBrightnessMin(self):
        return self.doubleSpinBoxBrightnessMin.value()

    def getBrightnessMax(self):
        return self.doubleSpinBoxBrightnessMax.value()

    def getSavePolylines(self):
        return self.checkBox_TriageImages_SavePolylines.isChecked()

    def getCorrectAlignment(self):
        return self.checkBox_TriageImages_CorrectAlignment.isChecked()

    def getReferenceImageFilename(self):
        return self.referenceImageFilename

    def getRotationThreshold(self):
        return self.doubleSpinBox_TriageOptions_RotationThreshold.value()
