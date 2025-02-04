import os
import promptlib

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal
from PyQt5.uic import loadUi

from GRIME_AI_Save_Utils import JsonEditor
from GRIME_AI_QMessageBox import GRIME_AI_QMessageBox


# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_buildModelDlg(QDialog):
    close_signal = pyqtSignal()

    # SIGNALS TO MAIN APP TO PERFORM MASK EDITOR RELATED FUNCTIONS
    save_model_masks_signal = pyqtSignal(bool)
    save_original_model_image_signal = pyqtSignal(bool)
    segment_image_signal = pyqtSignal()

    build_model_signal = pyqtSignal()
    tune_model_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        #JES self.setWindowFlags(Qt.WindowStaysOnTopHint)

        dirname = os.path.dirname(__file__)
        ui_file_absolute = os.path.join(dirname, 'QDialog_ModelBuilding.ui')
        loadUi(ui_file_absolute, self)

        self.accepted.connect(self.closeBuildModelDlg)
        self.rejected.connect(self.closeBuildModelDlg)

        self.segmentation_model_filename = None
        self.segmentation_images_folder = None
        self.training_images_folder = None
        self.annotation_filename = None
        self.model_tuning_annotation_file = None

        self.init_tab_widget()
        self.init_segmentation_tab()
        self.init_modeling_tab()

        # CLEAN-UP ALL THIS CODE. WHAT ARE WE DOING WITH THIS? VESTIGES SO DELETE? RE-IMPLEMENT? IMPROVE?
        #self.referenceImageFilename = ''
        #self.pushButton_SelectReferenceImage.clicked.connect(self.selectReferenceImage)
        #self.pushButton_SelectReferenceImage.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

        # WHEN THE DIALOGBOX OPENS AND IS INITIALIZED, PUSH THE CURRENT CONTROL SETTINGS TO THE MAIN APP
        self.save_model_masks_signal.emit(self.checkBox_saveModelMasks.isChecked())
        self.save_original_model_image_signal.emit(self.checkBox_copyOriginalModelImage.isChecked())

    # ----------------------------------------------------------------------------------------------------
    # INITIALIZE TAB WIDGET
    # ----------------------------------------------------------------------------------------------------
    def init_tab_widget(self):
        # Set stylesheet for the tabs to change color when a tab is selected.
        self.tabWidget_machine_learning.setStyleSheet("""
            QTabBar::tab {
                background-color: white;
                color: black;
            }
            QTabBar::tab:selected {
                background-color: steelblue;
                color: white;
            }
        """)

    # ----------------------------------------------------------------------------------------------------
    # INTIIALIZE IMAGE SEGMENTATION TAB
    # ----------------------------------------------------------------------------------------------------
    def init_segmentation_tab(self):
        self.pushButton_Segment.clicked.connect(self.segment_clicked)
        self.pushButton_Segment.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

        self.segmentation_model_filename = JsonEditor().getValue("Segmentation_Model_Filename")
        if self.segmentation_model_filename is not None:
            self.segmentation_model_filename = os.path.normpath(self.segmentation_model_filename)
            self.lineEdit_segmentation_model_file.setText(self.segmentation_model_filename)

        self.pushButton_Select_Model.clicked.connect(self.select_segmentation_model)
        self.pushButton_Select_Model.setStyleSheet('QPushButton {background-color: cyan; color: black;}')
        self.lineEdit_segmentation_model_file.textChanged.connect(self.select_segmentaiton_model_changed)


        self.segmentation_images_folder = JsonEditor().getValue("Segmentation_Images_Folder")
        if self.segmentation_images_folder is not None:
            self.segmentation_images_folder = os.path.normpath(self.segmentation_images_folder)
            self.lineEdit_segmentation_images_folder.setText(self.segmentation_images_folder)

        self.pushButton_Select_Images_Folder.clicked.connect(self.select_segmentation_images_folder)
        self.pushButton_Select_Images_Folder.setStyleSheet('QPushButton {background-color: cyan; color: black;}')
        self.lineEdit_segmentation_images_folder.textChanged.connect(self.select_segmentation_images_folder_changed)

        self.segmentation_option_save_masks = JsonEditor().getValue("Segmentation_Option_Save_Masks")
        if self.segmentation_option_save_masks is not None:
            self.checkBox_saveModelMasks.setChecked(self.segmentation_option_save_masks)
        else:
            self.checkBox_saveModelMasks.setChecked(True)
        self.checkBox_saveModelMasks.clicked.connect(self.saveModelMasksClicked)

        self.segmentation_option_copy_original = JsonEditor().getValue("Segmentation_Option_Copy_Original_Image")
        if self.segmentation_option_copy_original is not None:
            self.checkBox_copyOriginalModelImage.setChecked(self.segmentation_option_copy_original)
        else:
            self.checkBox_copyOriginalModelImage.setChecked(True)
        self.checkBox_copyOriginalModelImage.clicked.connect(self.copyOriginalModelImageClicked)


    # ----------------------------------------------------------------------------------------------------
    # INITIALIZE MODEL TUNING TAB
    # ----------------------------------------------------------------------------------------------------
    def init_modeling_tab(self):
        self.pushButton_Tune_Model.clicked.connect(self.tune_model)
        self.pushButton_Tune_Model.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

        self.training_images_folder = JsonEditor().getValue("Model_Tuning_Training_Images_Folder")
        if self.training_images_folder is not None:
            self.training_images_folder = os.path.normpath(self.training_images_folder)
            self.lineEdit_Training_Images_Folder.setText(self.training_images_folder)

        self.pushButton_Select_Training_Images_Folder.clicked.connect(self.select_training_images_folder)
        self.pushButton_Select_Training_Images_Folder.setStyleSheet('QPushButton {background-color: cyan; color: black;}')
        self.lineEdit_Training_Images_Folder.textChanged.connect(self.training_images_folder_changed)


        self.pushButton_Select_Annotation_File.clicked.connect(self.select_annotation_file)
        self.pushButton_Select_Annotation_File.setStyleSheet('QPushButton {background-color: cyan; color: black;}')
        self.model_tuning_annotation_file = JsonEditor().getValue("Model_Tuning_Annotation_File")

        if self.model_tuning_annotation_file is not None:
            self.model_tuning_annotation_file = os.path.normpath(JsonEditor().getValue("Model_Tuning_Annotation_File"))
            self.lineEdit_Annotation_File.setText(self.model_tuning_annotation_file)


    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def copyOriginalModelImageClicked(self):
        #self.save_original_model_image_signal.emit(self.checkBox_copyOriginalModelImage.isChecked())
        JsonEditor().update_json_entry("Segmentation_Option_Copy_Original_Image", self.checkBox_copyOriginalModelImage.isChecked())

    def getMoveOriginalImage(self):
        return self.checkBox_copyOriginalModelImage.isChecked()


    def saveModelMasksClicked(self):
        #self.save_model_masks_signal.emit(self.checkBox_saveModelMasks.isChecked())
        JsonEditor().update_json_entry("Segmentation_Option_Save_Masks", self.checkBox_saveModelMasks.isChecked())

    def getSaveModelMasks(self):
        return self.checkBox_saveModelMasks.isChecked()


    def segment_clicked(self):
        tmp_segmentation_mode_file = self.lineEdit_segmentation_model_file.text()
        tmp_segmentation_images_folder = self.lineEdit_segmentation_images_folder.text()

        if tmp_segmentation_mode_file is not None and tmp_segmentation_images_folder is not None and \
            os.path.exists(tmp_segmentation_mode_file) and os.path.exists(tmp_segmentation_images_folder):
            self.segment_image_signal.emit()
        else:
            strMessage = 'A model file and a folder of images is required for segmentation.'
            msgBox = GRIME_AI_QMessageBox('Missing Data Source(s)', strMessage, QMessageBox.Close)
            response = msgBox.displayMsgBox()


    # ----------------------------------------------------------------------------------------------------
    # SELECT SEGMENTATION MODEL
    # ----------------------------------------------------------------------------------------------------
    def select_segmentation_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Model Files", "", "Torch Files (*.torch);;PTH Files (*.pth);;All Files (*)", "Torch Files (*.torch)", options=options)
        if fileName:
            self.segmentation_model_filename = os.path.normpath(fileName)

            self.lineEdit_segmentation_model_file.setText(self.segmentation_model_filename)

            JsonEditor().update_json_entry("Segmentation_Model_Filename", fileName)


    def select_segmentaiton_model_changed(self):
        self.segmentation_model_filename = self.lineEdit_segmentation_model_file.text()

        JsonEditor().update_json_entry("Segmentation_Model_Filename", self.segmentation_model_filename)


    # ----------------------------------------------------------------------------------------------------
    # SELECT SEGMENTATION IMAGES FOLDER
    # ----------------------------------------------------------------------------------------------------
    def select_segmentation_images_folder(self):
        prompter = promptlib.Files()
        folder = prompter.dir()

        if os.path.exists(folder):
            self.lineEdit_segmentation_images_folder.setText(folder)

        if folder:
            self.segmentation_images_folder = os.path.normpath(folder)
            self.lineEdit_segmentation_images_folder.setText(self.segmentation_images_folder)

            JsonEditor().update_json_entry("Segmentation_Images_Folder", folder)


    def select_segmentation_images_folder_changed(self):
        self.segmentation_images_folder = self.lineEdit_segmentation_images_folder.text()

        JsonEditor().update_json_entry("Segmentation_Images_Folder", self.segmentation_images_folder)


    def get_selected_model_path_and_filename(self):
        return self.segmentation_model_filename


    def get_segmentation_images_folder(self):
        return os.path.normpath(self.segmentation_images_folder)

    # ----------------------------------------------------------------------------------------------------
    # SELECT TRAINING IMAGES FOLDER
    # ----------------------------------------------------------------------------------------------------
    def select_training_images_folder(self):
        prompter = promptlib.Files()
        folder = prompter.dir()

        if os.path.exists(folder):
            self.lineEdit_Training_Images_Folder.setText(folder)

        if folder:
            folder = os.path.normpath(folder)

            self.lineEdit_Training_Images_Folder.setText(folder)
            self.training_images_folder = self.lineEdit_Training_Images_Folder.text()

            JsonEditor().update_json_entry("Model_Tuning_Training_Images_Folder", folder)


    def training_images_folder_changed(self):
        self.training_images_folder = self.lineEdit_Training_Images_Folder.text()

        JsonEditor().update_json_entry("Model_Tuning_Training_Images_Folder", self.training_images_folder )


    def get_training_images_folder(self):
        return os.path.normpath(self.training_images_folder)


    # ----------------------------------------------------------------------------------------------------
    # SELECT ANNOTATION FILE
    # ----------------------------------------------------------------------------------------------------
    def select_annotation_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "COCA Annotation File", "", "All Files (*.json);; COCO Annotation Files (*.json)", options=options)
        if fileName:
            fileName = os.path.normpath(fileName)

            self.lineEdit_Annotation_File.setText(fileName)
            self.model_tuning_annotation_file = self.lineEdit_Annotation_File.text()

            JsonEditor().update_json_entry("Model_Tuning_Annotation_File", fileName)


    def annotation_file_changed(self):
        self.model_tuning_annotation_file = self.lineEdit_Annotation_File.text()

        JsonEditor().update_json_entry("Model_Tuning_Annotation_File", self.model_tuning_annotation_file )


    def get_annotation_filename(self):
        return os.path.normpath(self.annotation_filename)


    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def tune_model(self):
        self.tune_model_signal.emit()


    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def onCancel(self):
        self.close()


    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def closeBuildModelDlg(self):
        pass
