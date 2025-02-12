from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QDialogButtonBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi

import cv2

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_CompositeSliceDlg(QDialog):

    compositeSliceCancelSignal = pyqtSignal()
    compositeSliceGenerateSignal = pyqtSignal()

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        loadUi('QDialog_CompositeSlice.ui', self)

        self.widthMultiplier = 0
        self.heightMultiplier = 0

        self.sliceCenter = self.horizontalSlider.value()
        self.sliceWidth = self.spinBox_Width.value()
        self.lineEdit_HorizontalPosition.setText(self.sliceCenter.__str__())

        self.label_Image.setSliceCenter(self.sliceCenter)
        self.label_Image.setSliceWidth(self.sliceWidth)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #JES - WHY DID I SAVE THIS? self.imageIndexSignal.connect(parent.getImageIndex )
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.horizontalSlider.valueChanged.connect(self.valuechange)
        self.spinBox_Width.valueChanged.connect(self.spinBox_WidthChanged)

        self.pushButton_Generate.clicked.connect(self.pushButton_Generate_Clicked)
        self.pushButton_Generate.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

        self.pushButton_Cancel.clicked.connect(self.pushButton_Cancel_Clicked)

    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def loadImage(self, filename):
        # LOAD IMAGE FROM FILE
        numpyImage = cv2.imread(filename)

        # CONVERT numpy FORMATTED IMAGE TO A PIXEL MAP (i.e., QPixmap)
        myImage = QImage(numpyImage, numpyImage.shape[1], numpyImage.shape[0], QImage.Format_RGB888)
        myImage = QPixmap(myImage)

        self.widthMultiplier = myImage.width() / self.label_Image.width()
        self.heightMultiplier = myImage.height() / self.label_Image.height()

        # LOAD THE IMAGE INTO THE LABEL WIDGET FOR DISPLAY
        self.label_Image.setPixmap(
            myImage.scaled(self.label_Image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def getMultipliers(self):
        return self.widthMultiplier, self.heightMultiplier, self.sliceCenter, self.sliceWidth


    def spinBox_WidthChanged(self):
        self.sliceWidth = self.spinBox_Width.value()
        self.label_Image.setSliceWidth(self.sliceWidth)
        self.label_Image.drawCompositeSlice(self.sliceCenter)


    def valuechange(self):
        self.sliceCenter = self.horizontalSlider.value()
        self.label_Image.drawCompositeSlice(self.sliceCenter)
        self.lineEdit_HorizontalPosition.setText(self.sliceCenter.__str__())


    def closeEvent(self, event):
        super(GRIME_AI_CompositeSliceDlg, self).closeEvent(event)


    def getSliceCenter(self):
        return(self.sliceCenter)


    def pushButton_Generate_Clicked(self):
        self.compositeSliceGenerateSignal.emit()


    def pushButton_Cancel_Clicked(self):
        self.compositeSliceCancelSignal.emit()
        self.close()


    def onCancel(self):
        self.compositeSliceCancelSignal.emit()
        self.close()

