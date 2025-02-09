import os
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.uic import loadUi
import promptlib

from GRIME_AI_QMessageBox import GRIME_AI_QMessageBox

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_ExportCOCOMasksDlg(QDialog):

    # SIGNALS
    # ------------------------------------------------------------------------------------------------------------------
    #fetchImageList_Signal = pyqtSignal(str, int)
    COCO_signal_ok = pyqtSignal()
    COCO_signal_cancel = pyqtSignal()

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        loadUi('QDialog_ExtractCOCOMasks.ui', self)

        self.annotationImagesFolder = ''
        self.annotationFile = ''

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #self.fetchImageList_Signal.connect(parent.fetchImageList)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.pushButton_BrowseAnnotationImages.clicked.connect(self.pushButton_BrowseAnnotationImagesClicked)
        self.pushButton_BrowseAnnotationFile.clicked.connect(self.pushButton_BrowseAnnotationFileClicked)

        self.lineEdit_AnnotationImagesFolder.textChanged.connect(self.annotation_images_folder_changed)
        self.lineEdit_COCOAnnotationFile.textChanged.connect(self.COCO_annotation_file_changed)

        self.buttonBox_ok_cancel.accepted.connect(self.accepted)
        self.buttonBox_ok_cancel.rejected.connect(self.rejected)

        self.pushButton_BrowseAnnotationImages.setStyleSheet('QPushButton {background-color: steelblue;}')
        self.pushButton_BrowseAnnotationFile.setStyleSheet('QPushButton {background-color: steelblue;}')


    def COCO_annotation_file_changed(self):
        self.annotationFile = self.lineEdit_COCOAnnotationFile.text()


    def annotation_images_folder_changed(self):
        self.annotationImagesFolder = self.lineEdit_AnnotationImagesFolder.text()


    def pushButton_BrowseAnnotationImagesClicked(self):
        prompter = promptlib.Files()

        folder = prompter.dir()

        if os.path.exists(folder):
            if os.listdir(folder):
                self.lineEdit_AnnotationImagesFolder.setText(folder)
                self.annotationImagesFolder = self.lineEdit_AnnotationImagesFolder.text()
            else:
                msgBox = GRIME_AI_QMessageBox('COCO Error', 'No files or no supported files types in folder.')
                response = msgBox.displayMsgBox()

    def pushButton_BrowseAnnotationFileClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "COCO Annotation File", "", "All Files (*.json);;COCO Annotation Files (*.json)", options=options)
        if fileName:
            self.lineEdit_COCOAnnotationFile.setText(fileName)
            self.annotationFile = self.lineEdit_COCOAnnotationFile.text()

            print(f"Selected file: {fileName}")


    def accepted(self):
        #msgBox = GRIME_AI_QMessageBox('COCO Error', 'No files or no supported files types in folder.')
        #msgBox.displayMsgBox()

        if self.annotationFile and self.annotationImagesFolder:
            self.COCO_signal_ok.emit()
            super().accept()
        else:
            if not self.annotationFile:
                msgBox = GRIME_AI_QMessageBox('COCO Mask Extraction', 'Mask extraction aborted! No COCO Annotation file was selected.')
                msgBox.displayMsgBox()
            elif not os.path.exists(self.annotationImagesFolder):
                msgBox = GRIME_AI_QMessageBox('Images Folder', 'Mask extraction aborted! No images folder specified or no images found.')
                msgBox.displayMsgBox()

            self.COCO_signal_cancel.emit()
            super().reject()


    def rejected(self):
        #super(GRIME_AI_ExportCOCOMasksDlg, self).closeEvent(event)
        self.COCO_signal_cancel.emit()
        super().reject()

    @pyqtSlot()
    def image_folder_changed(self):
        pass
        # LET THE MAIN APPLICATION KNOW THAT THE FOLDER PATH FOR THE IMAGES HAS CHANGED BY SENDING IT THE PATH
        #image_folder = self.lineEditImageFolder.text()
        #JsonEditor().update_json_entry("Local_Image_Folder", image_folder)

    def getAnnotationImagesFolder(self):
        return self.annotationImagesFolder

    def getAnnotationFile(self):
        return self.annotationFile

    # --------------------------------------------------
    # NEON/PHENOCAM IMAGE FOLDER
    # --------------------------------------------------
    def pushButtonBrowseSaveImagesOutputFolderClicked(self):
        prompter = promptlib.Files()
        folder = prompter.dir()

        if os.path.exists(folder):
            self.EditSaveImagesOutputFolder(prompter.dir())


    def setImageFolderPath(self, image_folder):
        self.lineEditImageFolder.setText(image_folder)



