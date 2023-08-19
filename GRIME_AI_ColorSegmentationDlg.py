from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

# ====================================================================================================
#
# ====================================================================================================
class roiParameters():

    def __init__(self, parent=None):
        strROIName = ''
        numColorClusters = 4
        bDisplayROIs = True
        bDisplayROIColors = True

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_ColorSegmentationDlg(QDialog):

    # SIGNALS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    colorSegmentation_Signal = pyqtSignal(int)
    addROI_Signal = pyqtSignal(roiParameters)
    deleteAllROI_Signal = pyqtSignal()
    close_signal = pyqtSignal()

    returnROIParameters = roiParameters()

    # -----------------------------------------------------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        loadUi('QDialog_ColorSegmentation.ui', self)

        self.pushButtonAddROI.clicked.connect(self.addROI)
        self.pushButton_deleteAllROIs.clicked.connect(self.deleteAllROI)
        #self.checkBoxDisplayROIs.clicked.connect(self.displayROIs)
        self.pushButton_ColorSegmentation.clicked.connect(self.colorSegmentationClicked)
        self.buttonBox_Close.clicked.connect(self.closeClicked)

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        super(GRIME_ColorSegmentationDlg, self).closeEvent(event)
        self.close_signal.emit()

    def closeClicked(self):
        self.close_signal.emit()

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def colorSegmentationClicked(self):
        self.colorSegmentation_Signal.emit(1)

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def addROI(self):
        self.returnROIParameters.strROIName = self.lineEdit_roiName.text()
        self.returnROIParameters.numColorClusters = 4
        self.returnROIParameters.bDisplayROIs = True
        self.returnROIParameters.bDisplayROIColors = True

        self.addROI_Signal.emit(self.returnROIParameters)

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def deleteAllROI(self):
        self.deleteAllROI_Signal.emit()
