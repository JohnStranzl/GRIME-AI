from PyQt5 import Qt, QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

userCancelled = "Cancelled"
userOk = "OK"

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_ImageNavigationDlg(QDialog):

    imageIndexSignal = pyqtSignal(int)

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        loadUi('QDialog_ImageNavigation.ui', self)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.imageIndexSignal.connect(parent.getImageIndex)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.imageCount = 0

        self.spinBoxImageIndex.valueChanged.connect(self.spinBoxCannyImageIndexChanged)
        self.pushButtonResetImageIndex.clicked.connect(self.pushButtonResetImageImageClicked)

        self.spinBoxImageIndex.setMinimum(1)
        self.spinBoxImageIndex.setMaximum(10000000)
        self.spinBoxImageIndex.setValue(1)

        self.spinBoxImageIndex.setKeyboardTracking(False)
        self.spinBoxImageIndex.editingFinished.connect(self.spinBoxChanged)
        self.spinBoxImageIndex.valueChanged.connect(self.spinBoxChanged)

        self.pushButtonPlay.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')
        self.pushButtonStop.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def spinBoxChanged(self):
        global gFrameCount

        if self.imageCount > 0:
            self.imageIndexSignal.emit(self.spinBoxImageIndex.value())

    def spinBoxCannyImageIndexChanged(self):
        self.imageIndexSignal.emit(self.spinBoxImageIndex.value())

    def pushButtonResetImageImageClicked(self):
        self.spinBoxImageIndex.setValue(1)

    def setImageIndex(self, imageIndex):
        self.spinBoxImageIndex.setValue(imageIndex)

    def setImageCount(self, imageCount):
        self.labelImageCountNumber.setText(str(imageCount))
        self.imageCount = imageCount
        self.spinBoxImageIndex.setMaximum(imageCount)

    def reset(self):
        self.spinBoxImageIndex.setValue(1)
        self.imageIndexSignal.emit(self.spinBoxImageIndex.value())

    def onCancel(self):
    #    self.result = userCancelled
        self.close()

    #def onOk(self):
    #    self.result = userOk
    #    self.close()

    #def closeEvent(self, evnt):
    #    evnt.ignore()

    #def initUI(self):
    #      # create our window
    #      # define window		xLoc,yLoc,xDim,yDim
    #      # self.setGeometry(250, 250, 400, 400)
    #      # self.setWindowTitle("Our Example Nonmodal Program Window")
    #      #self.setWindowFlags(Qt.WindowStaysOnTopHint)
    #      self.setMouseTracking(True)

    #      self.result = userCancelled
    #      self.show()

    #      return
