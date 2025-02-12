from PyQt5.QtWidgets import QDialog

from PyQt5.uic import loadUi

import promptlib

from PyQt5 import QtGui

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_TriageOptionsDlg(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        loadUi('QDialog_TriageOptions.ui', self)

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
