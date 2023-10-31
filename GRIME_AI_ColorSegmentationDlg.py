from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QVBoxLayout
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
    buildFeatureFile_Signal = pyqtSignal()

    returnROIParameters = roiParameters()

    # -----------------------------------------------------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        layout = QVBoxLayout(self)

        # SET BEHAVIOR OF DIALOG BOX
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        loadUi('QDialog_ColorSegmentation.ui', self)

        # CONNECT CONTROLS/WIDGETS TO FUNCTIONS THAT RESPOND TO CLICKS
        #self.checkBoxDisplayROIs.clicked.connect(self.displayROIs)
        self.pushButtonAddROI.clicked.connect(self.addROI)
        self.pushButton_deleteAllROIs.clicked.connect(self.deleteAllROI)
        self.buttonBox_Close.clicked.connect(self.closeClicked)
        self.pushButton_Dlg_BuildFeatureFile.clicked.connect(self.buildFeatureFile)

        # SET CONTROL COLORS
        self.pushButton_Dlg_BuildFeatureFile.setStyleSheet('QPushButton {background-color: steelblue; color: yellow;}')

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def buildFeatureFile(self):
        self.buildFeatureFile_Signal.emit()

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        super(GRIME_ColorSegmentationDlg, self).closeEvent(event)
        self.close_signal.emit()

    def closeClicked(self):
        self.close_signal.emit()

    # ----------------------------------------------------------------------------------------------------
    # self.pushButton_ColorSegmentation.clicked.connect(self.colorSegmentationClicked)
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
