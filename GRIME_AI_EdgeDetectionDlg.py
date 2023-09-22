from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QDialog, QWidget
from PyQt5.uic import loadUi
from constants import edgeMethodsClass, featureMethodsClass

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_EdgeDetectionDlg(QDialog):

    # SIGNALS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    edgeDetectionSignal = pyqtSignal(edgeMethodsClass)
    featureDetectionSignal = pyqtSignal(edgeMethodsClass)

    returnEdgeData    = edgeMethodsClass()
    returnFeatureData = featureMethodsClass()

    # -----------------------------------------------------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        loadUi('QDialog_EdgeDetection.ui', self)

        # CONNECT THE SIGNALS TO THE FUNCTIONS IN THE PARENT ("CALLING") THREAD THAT WILL RECEIVE THE SIGNAL
        self.featureDetectionSignal.connect(parent.featureDetectionMethod)
        self.edgeDetectionSignal.connect(parent.edgeDetectionMethod)

        # CONNECT THE WIDGETS TO THE FUNCTIONS IN THIS CLASS THAT WILL GET INVOKED
        self.radioButtonCanny.clicked.connect(self.updateCanny)
        self.spinBoxCannyKernel.valueChanged.connect(self.spinBoxCannyKernelChanged)
        self.spinBoxCannyHighThreshold.valueChanged.connect(self.spinBoxCannyHighThresholdChanged)
        self.spinBoxCannyLowThreshold.valueChanged.connect(self.spinBoxCannyLowThresholdChanged)

        self.radioButtonSobelX.clicked.connect(self.updateSobelX)
        self.radioButtonSobelY.clicked.connect(self.updateSobelY)
        self.radioButtonSobelXY.clicked.connect(self.updateSobelXY)
        self.spinBoxSobelKernel.valueChanged.connect(self.spinBoxSobelKernelChanged)

        self.radioButtonLaplacian.clicked.connect(self.updateLaplacian)

        self.radioButtonSIFT.clicked.connect(self.updateSIFT)
        self.radioButtonORB.clicked.connect(self.updateORB)

        #self.spinBoxOrbMaxFeatures.valueChanged.connect(self.spinBoxOrbMaxFeaturesChanged)


        #self.spinBoxCannyHighThreshold.setKeyboardTracking(False)
        #self.spinBoxCannyLowThreshold.setKeyboardTracking(False)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def updateCanny(self):
        self.returnEdgeData.method = edgeMethodsClass.CANNY
        self.returnEdgeData.selected = self.radioButtonCanny.isChecked()

        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def updateSobelX(self):
        self.returnEdgeData.method = edgeMethodsClass.SOBEL_X
        self.returnEdgeData.selected = self.radioButtonSobelX.isChecked()

        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def updateSobelY(self):
        self.returnEdgeData.method = edgeMethodsClass.SOBEL_Y
        self.returnEdgeData.selected = self.radioButtonSobelY.isChecked()

        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def updateSobelXY(self):
        self.returnEdgeData.method = edgeMethodsClass.SOBEL_XY
        self.returnEdgeData.selected = self.radioButtonSobelXY.isChecked()

        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxCannyHighThresholdChanged(self):
        self.edgeDetectionSignal.emit(1)
        #imageNumber = self.spinBoxDailyImage.value()
        #self.spinBoxChanged()
        #refreshImage(self)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxCannyLowThresholdChanged(self):
        self.edgeDetectionSignal.emit(1)
        #imageNumber = self.spinBoxDailyImage.value()
        #self.labelOriginalImage.clear()
        #self.spinBoxChanged()
        #refreshImage(self)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxCannyKernelChanged(self):
        #imageNumber = self.spinBoxDailyImage.value()
        self.edgeDetectionSignal.emit(1)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxSobelKernelChanged(self):
        #refreshImage(self)
        self.edgeDetectionSignal.emit(4)


    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def updateLaplacian(self):
        self.returnEdgeData.method = edgeMethodsClass.LAPLACIAN
        self.returnEdgeData.selected = self.radioButtonLaplacian.isChecked()

        self.edgeDetectionSignal.emit(self.returnEdgeData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def updateSIFT(self):
        self.returnFeatureData.method = featureMethodsClass.SIFT
        self.returnFeatureData.selected = self.radioButtonSIFT.isChecked()

        self.featureDetectionSignal.emit(self.returnFeatureData)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def updateORB(self):
        self.returnFeatureData.method = featureMethodsClass.ORB
        self.returnFeatureData.selected = self.radioButtonORB.isChecked()

        self.featureDetectionSignal.emit(self.returnFeatureData)
