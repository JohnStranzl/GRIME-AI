from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
import cv2


class GRIME_AI_CompositeSliceDlg(QtWidgets.QDialog):
    compositeSliceCancelSignal = pyqtSignal()
    compositeSliceGenerateSignal = pyqtSignal()

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def __init__(self, parent=None):
        super(GRIME_AI_CompositeSliceDlg, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(Qt.NonModal)

        loadUi('QDialog_CompositeSlice.ui', self)

        self.widthMultiplier = 0
        self.heightMultiplier = 0

        self.sliceCenter = self.horizontalSlider.value()
        self.sliceWidth = self.spinBox_Width.value()
        self.lineEdit_HorizontalPosition.setText(str(self.sliceCenter))

        self.label_Image.setSliceCenter(self.sliceCenter)
        self.label_Image.setSliceWidth(self.sliceWidth)

        self.horizontalSlider.valueChanged.connect(self.valuechange)
        self.spinBox_Width.valueChanged.connect(self.spinBox_WidthChanged)

        self.pushButton_Generate.clicked.connect(self.pushButton_Generate_Clicked)
        self.pushButton_Generate.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

        self.pushButton_Cancel.clicked.connect(self.pushButton_Cancel_Clicked)

        self.label_Image.mousePressEvent = self.imageLabelMousePress
        self.label_Image.mouseMoveEvent = self.imageLabelMouseMove
        self.label_Image.mouseReleaseEvent = self.imageLabelMouseRelease

        self.dragStartPosition = None
        self.draggingCenter = False
        self.draggingLeftEdge = False
        self.draggingRightEdge = False


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def loadImage(self, filename):
        numpyImage_bgr = cv2.imread(filename)
        numpyImage_rgb = cv2.cvtColor(numpyImage_bgr, cv2.COLOR_BGR2RGB)
        myImage = QImage(numpyImage_rgb, numpyImage_rgb.shape[1], numpyImage_rgb.shape[0], QImage.Format_RGB888)
        myImage = QPixmap(myImage)
        self.widthMultiplier = myImage.width() / self.label_Image.width()
        self.heightMultiplier = myImage.height() / self.label_Image.height()
        self.label_Image.setPixmap(myImage.scaled(self.label_Image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def getMultipliers(self):
        return self.widthMultiplier, self.heightMultiplier, self.sliceCenter, self.sliceWidth


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def spinBox_WidthChanged(self):
        self.sliceWidth = self.spinBox_Width.value()
        self.label_Image.setSliceWidth(self.sliceWidth)
        self.label_Image.drawCompositeSlice(self.sliceCenter)


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def valuechange(self):
        self.sliceCenter = self.horizontalSlider.value()
        self.label_Image.drawCompositeSlice(self.sliceCenter)
        self.lineEdit_HorizontalPosition.setText(str(self.sliceCenter))


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def closeEvent(self, event):
        super(GRIME_AI_CompositeSliceDlg, self).closeEvent(event)


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def getSliceCenter(self):
        return self.sliceCenter


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def pushButton_Generate_Clicked(self):
        self.compositeSliceGenerateSignal.emit()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def pushButton_Cancel_Clicked(self):
        self.compositeSliceCancelSignal.emit()
        self.close()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def onCancel(self):
        self.compositeSliceCancelSignal.emit()
        self.close()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def imageLabelMousePress(self, event):
        if event.buttons() == Qt.LeftButton:
            self.dragStartPosition = event.pos()
            self.draggingCenter = self.isWithinCenterLine(event.pos())
        elif event.buttons() == Qt.RightButton:
            self.dragStartPosition = event.pos()
            self.draggingLeftEdge = self.isWithinLeftEdgeLine(event.pos())
            self.draggingRightEdge = self.isWithinRightEdgeLine(event.pos())


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def imageLabelMouseMove(self, event):
        if event.buttons() == Qt.LeftButton:
            if self.draggingCenter:
                newCenter = self.sliceCenter + (event.pos().x() - self.dragStartPosition.x())
                self.horizontalSlider.setValue(newCenter)
                self.dragStartPosition = event.pos()
        elif event.buttons() == Qt.RightButton:
            if self.draggingLeftEdge:
                newWidth = (self.sliceCenter - event.pos().x()) * 2
                self.spinBox_Width.setValue(newWidth)
                self.dragStartPosition = event.pos()
            elif self.draggingRightEdge:
                newWidth = (event.pos().x() - self.sliceCenter) * 2
                self.spinBox_Width.setValue(newWidth)
                self.dragStartPosition = event.pos()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def imageLabelMouseRelease(self, event):
        self.draggingCenter = False
        self.draggingLeftEdge = False
        self.draggingRightEdge = False


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def isWithinCenterLine(self, pos):
        centerLineX = self.sliceCenter
        return abs(pos.x() - centerLineX) < 10  # 10 pixels tolerance


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def isWithinLeftEdgeLine(self, pos):
        leftEdgeX = self.sliceCenter - self.sliceWidth / 2
        return abs(pos.x() - leftEdgeX) < 10  # 10 pixels tolerance


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def isWithinRightEdgeLine(self, pos):
        rightEdgeX = self.sliceCenter + self.sliceWidth / 2
        return abs(pos.x() - rightEdgeX) < 10  # 10 pixels tolerance
