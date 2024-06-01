from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.uic import loadUi
import promptlib

from PyQt5 import QtGui

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_buildModelDlg(QDialog):
    close_signal = pyqtSignal()

    # SIGNALS TO MAIN APP TO PERFORM MASK EDITOR RELATED FUNCTIONS
    saveModelMasks_Signal = pyqtSignal(bool)
    saveOriginalModelImage_Signal = pyqtSignal(bool)
    segment_Signal = pyqtSignal()

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        loadUi('QDialog_ModelBuilding.ui', self)

        #self.referenceImageFilename = ''
        #self.pushButton_SelectReferenceImage.clicked.connect(self.selectReferenceImage)

        #self.pushButton_SelectReferenceImage.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')
        self.pushButton_Segment.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')
        self.checkBox_saveModelMasks.clicked.connect(self.saveModelMasksClicked)
        self.checkBox_copyOriginalModelImage.clicked.connect(self.copyOriginalModelImageClicked)
        self.pushButton_Segment.clicked.connect(self.segmentClicked)

        # WHEN THE DIALOGBOX OPENS AND IS INITIALIZED, PUSH THE CURRENT CONTROL SETTINGS TO THE MAIN APP
        self.saveModelMasks_Signal.emit(self.checkBox_saveModelMasks.isChecked())
        self.saveOriginalModelImage_Signal.emit(self.checkBox_copyOriginalModelImage.isChecked())


    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def getSaveModelMasks(self):
        return self.checkBox_saveModelMasks.isChecked()

    def getMoveOriginalImage(self):
        return self.checkBox_copyOriginalModelImage.isChecked()

    def saveModelMasksClicked(self):
        self.saveModelMasks_Signal.emit(self.checkBox_saveModelMasks.isChecked())

    def copyOriginalModelImageClicked(self):
        self.saveOriginalModelImage_Signal.emit(self.checkBox_copyOriginalModelImage.isChecked())

    def segmentClicked(self):
        self.segment_Signal.emit()

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        super(GRIME_AI_buildModelDlg, self).closeEvent(event)
        self.close_signal.emit()