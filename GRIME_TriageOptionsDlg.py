from PyQt5.QtWidgets import QDialog

from PyQt5.uic import loadUi

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_TriageOptionsDlg(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        loadUi('QDialog_TriageOptions.ui', self)

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

