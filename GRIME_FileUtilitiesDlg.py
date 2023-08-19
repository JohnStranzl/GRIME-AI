import os
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
import promptlib
import GRIME_AI_Utils

class Datapaths():

    def __init__(self, parent=None):
        self.imageInputFolder = ""
        self.imageOutputFolder = ""
        self.videoInputFolder = ""
        self.videoOutputFilder = ""
        self.gifInputFilder = ""
        self.gifOutputFilder = ""


# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_FileUtilitiesDlg(QDialog):

    fetchImageList_Signal = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        loadUi('QDialog_FileUtilities.ui', self)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.fetchImageList_Signal.connect(parent.fetchImageList)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #self.lineEditHardDriveFolder.textChanged.connect(self.MSTVideoFolderChanged)
        self.pushButtonBrowseVideoOutputFolder.clicked.connect(self.pushButtonBrowseVideoOutputFolderClicked)
        self.pushButtonCreateVideo.clicked.connect(self.pushButtonCreateVideoClicked)
        self.pushButtonCreateGIF.clicked.connect(self.pushButtonCreateGIFClicked)
        self.pushButtonBrowseImageFolder.clicked.connect(self.pushButtonBrowseImageFolderClicked)
        #JES self.pushButtonBrowseSaveImagesOutputFolder.clicked.connect(self.pushButtonBrowseSaveImagesOutputFolderClicked)
        self.pushButtonBrowseEXIFOutputFolder.clicked.connect(self.pushButtonBrowseEXIFOutputFolderClicked)

        # VIDEO FUNCTIONS
        # JES self.pushButtonFetchVideoFilenames.clicked.connect(self.MSTFetchVideoFilenames)
        self.pushButtonExtractFrames.clicked.connect(self.MSTExtractFrames)
        self.MSTlineEditFrameFolder.textChanged.connect(self.MSTFrameCountUpdate)

        self.lineEditImageFolder.textChanged.connect(self.MSTVideoFolderChanged)

        #
        #JES self.checkBoxNEONSites.clicked.connect(self.checkboxNEONSitesClicked)
        self.pushButton_FetchImageList.clicked.connect(self.pushButtonFetchImageListClicked)

        #JES self.pushButtonBrowseImageFolderClicked.setStyleSheet('QPushButton {background-color: steelblue;}')
        #JES self.pushButtonBrowseEXIFOutputFolderClicked.setStyleSheet('QPushButton {background-color: steelblue;}')
        #JES self.pushButtonCreateVideo.setStyleSheet('QPushButton {background-color: steelblue;}')
        #JES self.pushButtonCreateGIF.setStyleSheet('QPushButton {background-color: steelblue;}')
        #JES self.pushButtonCreateEXIFFile.setStyleSheet('QPushButton {background-color: steelblue;}')

    def pushButtonFetchImageListClicked(self):
        if len(self.lineEditImageFolder.text()) > 0:
            self.fetchImageList_Signal.emit(self.lineEditImageFolder.text(), self.checkBox_FetchRecursive.isChecked())
        else:
            self.pushButtonBrowseImageFolderClicked()
            self.fetchImageList_Signal.emit(self.lineEditImageFolder.text(), self.checkBox_FetchRecursive.isChecked())

    def checkboxNEONSitesClicked(self):
        pass

    def checkboxNEONSitesClicked(self):
        pass

    def MSTFrameCountUpdate(self):
        pass

    def MSTExtractFrames(self):
        pass

    def pushButtonBrowseSaveImagesOutputFolderClicked(self):
        pass

    def pushButtonCreateVideoClicked(self):
        pass
        #JES createVideo(self)

    def pushButtonCreateGIFClicked(self):
        pass
        #JES createGIF(self, self.checkBox_FetchRecursive.isChecked())

    # --------------------------------------------------
    # IMAGE INPUT FOLDER
    # --------------------------------------------------
    def pushButtonBrowseImageFolderClicked(self):
        prompter = promptlib.Files()
        folder = prompter.dir()

        if os.path.exists(folder):
            self.lineEditImageFolder.setText(folder)
            #self.radioButtonHardDriveImages.setChecked(True)
            #self.checkBoxCreateEXIFFile.setEnabled(True)

            # RECURSE AND TRAVERSE FOLDERS FROM ROOT DOWNWARD TO GET A LIST OF
            #JES startDate, endDate = getLocalFileDates(folder, self.checkBox_FetchRecursive.isChecked())

            #JES updateProductTableDateRange(self.tableProducts, 0, startDate, endDate)

            #JES processLocalImage(self)

    # --------------------------------------------------
    # NEON/PHENOCAM IMAGE FOLDER
    # --------------------------------------------------
    def pushButtonBrowseSaveImagesOutputFolderClicked(self):
        prompter = promptlib.Files()
        folder = prompter.dir()

        if os.path.exists(folder):
            self.EditSaveImagesOutputFolder(prompter.dir())

    # --------------------------------------------------
    # OUTPUT FOLDERS
    # --------------------------------------------------
    def pushButtonBrowseVideoOutputFolderClicked(self):
        prompter = promptlib.Files()
        folder = prompter.dir()

        if os.path.exists(folder):
            self.EditVideoOutputFolder.setText(prompter.dir())

    def pushButtonBrowseSaveImagesOutputFolderClicked(self):
        prompter = promptlib.Files()
        folder = prompter.dir()

        if os.path.exists(folder):
            self.EditSaveImagesOutputFolder.setText(prompter.dir())

    def pushButtonBrowseEXIFOutputFolderClicked(self):
        prompter = promptlib.Files()
        folder = prompter.dir()

        if os.path.exists(folder):
            self.EditEXIFOutputFolder.setText(prompter.dir())

    # --------------------------------------------------
    # --------------------------------------------------
    def MSTVideoFolderChanged(self):
        #self.imageIndexSignal.emit(self.spinBoxImageIndex.value())
        pass
        '''
        global gFrameCount

        frameFolder = self.lineEditHardDriveFolder.text() + '/' + self.MSTlineEditFrameFolder.text()
        nFrameCount = MSTGetFrameCount(self, frameFolder)

        strFrameCount = 'Frame Count: ' + nFrameCount.__str__()
        self.MSTFrameCount.setText(strFrameCount)

        self.spinBoxDailyImage.setMinimum(1)
        self.spinBoxDailyImage.setMaximum(nFrameCount)
        self.spinBoxDailyImage.setValue(1)

        gFrameCount = nFrameCount
        '''

    def setImageFolder(self, str):
        self.lineEditImageFolder.setText(str)


# ======================================================================================================================
#
# ======================================================================================================================
def getLocalFileDates(filePath, bFetchRecursive):
    # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
    extensions = ('.jpg', '.jpeg', '.png')

    # INITIALIZE VARIABLES TO A KNOWN VALUE
    startDate = datetime.date(2500, 12, 31)
    endDate   = datetime.date(1970, 1, 1)

    files = GRIMe_Utils.getFileList(filePath, extensions, bFetchRecursive)

    for file in files:
        ext = os.path.splitext(file)[-1].lower()

        if ext in extensions:
            fileDate, fileTime = GRIMe_Utils.extractDateFromFilename(file)

            # use the date in the filenames to determine the start and end acquisition dates for the images
            if fileDate < startDate:
                startDate = fileDate

            if fileDate > endDate:
                endDate = fileDate

    return startDate, endDate

