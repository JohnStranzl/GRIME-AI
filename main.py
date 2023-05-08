# !/usr/bin/env python
# !/usr/bin/env python
# coding: utf-8
import webbrowser

import webdriver_manager.chrome
# 1. SCRAPE FIELD SITE TABLE (CSV) FILE FROM https://www.neonscience.org/field-sites/explore-field-sites
#
# 2. PERSIST A COPY OF THE CONTENTS OF THE FIELD SITE TABLE
#
# 2. POPULATE LISTBOX WITH SITES
#
# 3. SELECT A SITE
#
# 4. QUERY NEON SITE FOR INFO ON SELECTED SITE
#
# 5. DOWNLOAD DATA
#

# cv2.mahalanobis
# matplotlib.use('Qt5Agg')

# import pycurl

from PyQt5.QtCore import pyqtSignal, pyqtSlot

import csv
import datetime
import math
#import os
#import re
from enum import Enum
import shutil
import sys
import time
import pathlib
import urllib.request
from configparser import ConfigParser
from pathlib import Path
from urllib.request import urlopen

import imageio
import cv2
# import tensorflow as tf
import numpy as np
import promptlib
#import torch
# print(torch.__version__)
import requests

from PIL import Image, ImageQt

#from playsound import playsound

# ------------------------------------------------------------
#
# ------------------------------------------------------------
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QToolBar, QCheckBox, QDateTimeEdit, \
    QGraphicsScene, QMessageBox, QSplashScreen, QAction, QSpinBox, QDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from GRIME_TriageOptionsDlg import GRIME_TriageOptionsDlg
from GRIME_ReleaseNotesDlg import GRIME_ReleaseNotesDlg
from GRIME_EdgeDetectionDlg import GRIME_EdgeDetectionDlg
from GRIME_ImageNavigationDlg import GRIME_ImageNavigationDlg
from GRIME_ColorSegmentationDlg import GRIME_ColorSegmentationDlg
from GRIME_FileUtilitiesDlg import GRIME_FileUtilitiesDlg
from GRIME_MaskEditorDlg import GRIME_MaskEditorDlg

#import GRIME_KMeans

import sobelData
from GRIMe_Color import GRIMe_Color
from GRIME_ProcessImage import GRIMe_ProcessImage

# ------------------------------------------------------------
#
# ------------------------------------------------------------
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from chrome_driver import *

from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
#from webdriver_manager.core.utils import ChromeType

# ------------------------------------------------------------
#
# ------------------------------------------------------------
from GRIMe_Diagnostics import GRIMe_Diagnostics
from GRIMe_ImageData import imageData

# ------------------------------------------------------------
#
# ------------------------------------------------------------
from GRIMe_PhenoCam import GRIMe_PhenoCam, dailyList
from GRIMe_ProductTable import GRIMe_ProductTable
from GRIMe_QMessageBox import GRIMe_QMessageBox
from GRIMe_QProgressWheel import QProgressWheel
from GRIMe_Utils import GRIMe_Utils
from GRIMe_roiData import GRIMe_roiData, ROIShape
from GRIMe_QLabel import DrawingMode

# ------------------------------------------------------------
#
# ------------------------------------------------------------
from NEON_20033_NitrateData import NEON_20033_NitrateData
from NEON_FetchData import NEON_FetchData
from NEON_DownloadLatestImage import NEON_DownloadLatestImage
from NEON_DownloadFieldSiteTableFiles import NEON_DownloadFieldSiteTableFiles
from NEON_FetchSiteInfoFromNEON import NEON_FetchSiteInfoFromNEON
from NEON_FetchFieldSiteTableURL import NEON_FetchFieldSiteTableURL

# ------------------------------------------------------------
#
# ------------------------------------------------------------
from constants import edgeMethodsClass, featureMethodsClass

# ------------------------------------------------------------
#
# ------------------------------------------------------------
from exifData import EXIFData
from GRIMe_ImageStats import GRIMe_ImageStats
from USGS_HIVIS import USGS_HIVIS

global full
full = 1

if full == 1:
    from neonAIgui import Ui_MainWindow
elif full == 2:
    from GUIs.GRIMe_AIDownloadManager import Ui_MainWindow
elif full == 3:
    from GUIs.guiTesting import Ui_MainWindow
elif full == 4:
    from GRIME_AI_NSF import Ui_MainWindow

global bStartupComplete
bStartupComplete = False

# from tensorflow.python.client import device_lib as dev_lib

# ------------------------------------------------------------
# THESE HIDDEN IMPORTS ARE ONLY REQUIRED WHEN COMPILING WITH PYINSTALLER
# ------------------------------------------------------------
try:
    import sklearn.neighbors._typedefs
except:
    pass

try:
    import sklearn.neighbors._partition_nodes
except:
    pass

try:
    import sklearn.utils._weight_vector
except:
    pass

try:
    import sklearn.neighbors._quad_tree
except:
    pass

try:
    import skimage
except:
    pass

# ------------------------------------------------------------
# Get the base directory
# ------------------------------------------------------------
if 0:
    if getattr(sys, 'frozen', None):  # keyword 'frozen' is for setting basedir while in onefile mode in pyinstaller
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
        basedir = os.path.normpath(basedir)

    # Locate the SSL certificate for requests
    os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(basedir, 'requests', 'cacert.pem')

# ------------------------------------------------------------
# PIL LIBRARIES AND IMPORTS
# ------------------------------------------------------------

SITECODE = 'ARIK'
DOMAINCODE = 'D10'
SITENAME = ''
originalImg = []
dailyImagesList = dailyList([], [])
currentImage = []
currentImageIndex = -1
latestImage = []
siteList = []
nStop = 0
gWebImageCount = 0
gWebImagesAvailable = 0
gFrameCount = 0
gProcessClick = 0
currentImageFilename = ""
frame = 0
# Define the maximum number of gray levels
gray_level = 16
imageFileFolder = ""

# URLS
# url = "http://maps.googleapis.com/maps/api/geocode/json?address=googleplex&sensor=false"
url = 'https://www.neonscience.org/field-sites/explore-field-sites'
root_url = 'https://www.neonscience.org'
SERVER = 'http://data.neonscience.org/api/v0/'
MSTFolder = 'C:/Users/Astrid Haugen/Documents/000 - MeanStride'

class greennessIndex():
    GCC  = False
    GLI  = False
    ExG  = False
    RGI  = False
    NDVI = False

class features():
    shannonEntropy = True
    intensity      = True
    color_hsv      = True
    texture        = False

class regionSelect():
    wholeImage = False
    ROIShape   = True

class displayOptions():
    displayROIs = True

g_displayOptions   = displayOptions()
g_regionSelect     = regionSelect()
#g_featureDetection = featureDetection()
#g_edgeDetection    = edgeDetection()
#g_features         = features()
g_greennessIndex   = greennessIndex()

g_edgeMethodSettings = edgeMethodsClass()
g_featureMethodSettings = featureMethodsClass()

# ======================================================================================================================
#
# ======================================================================================================================
class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

# ======================================================================================================================
#
# ======================================================================================================================
class MainWindow(QMainWindow, Ui_MainWindow):
    xStart = 0
    yStart = 0
    roiList = []
    imageStatsList = []

    # INITIALIZE POP-UP DIALOG BOXES
    fileFolderDlg        = None
    edgeDetectionDlg     = None
    colorSegmentationDlg = None
    TriageDlg            = None
    maskEditorDlg        = None
    imageNavigationDlg   = None
    releaseNotesDlg      = None

    # def eventFilter(self, source, event):
    #     if (event.type() == QtCore.QEvent.MouseMove and source is self.label):
    #         pos = event.pos()
    #         print('mouse move: (%d, %d)' % (pos.x(), pos.y()))
    #
    #     if (event.type() == QtCore.QEvent.MouseButtonDblClick and source is self.label):
    #         print('Double click')
    #
    #     return QtGui.QWidget.eventFilter(self, source, event)

    # def _createMenuBar(self):
    # menuBar = QMenuBar(self)
    # self.setMenuBar(menuBar)

    def createToolBar(self):
        # ====================================================================================================
        # TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR
        # ====================================================================================================
        toolbar = QToolBar("GRIMe-AI Toolbar")
        self.addToolBar(toolbar)
        toolbar.setIconSize(QtCore.QSize(48, 48))

        #COLOR SEGMENTATION ----------------------------------------------------------------------------------
        icon_path = str(Path(__file__).parent / "icons/FileFolder_1.png")
        button_action = QAction(QIcon(icon_path), "Folder Operations", self)
        button_action.setStatusTip("Select input and output folder locations")
        button_action.triggered.connect(self.onMyToolBarFileFolder)
        toolbar.addAction(button_action)

        # IMAGE TRIAGE ---------------------------------------------------------------------------------------
        icon_path = str(Path(__file__).parent / "icons/Triage_2.png")
        button_action = QAction(QIcon(icon_path), "Image Triage", self)
        button_action.setStatusTip("Move images that are of poor quality")
        button_action.triggered.connect(toolbarButtonImageTriage)
        toolbar.addAction(button_action)

        #MASK EDITOR -------------------------------------------------------------------------------------
        icon_path = str(Path(__file__).parent / "icons/ImageNav_3.png")
        button_action = QAction(QIcon(icon_path), "Image Navigation", self)
        button_action.setStatusTip("Navigate (scroll) through images")
        button_action.triggered.connect(self.onMyToolBarImageNavigation)
        toolbar.addAction(button_action)

        #IMAGE NAVIGATION -------------------------------------------------------------------------------------
        icon_path = str(Path(__file__).parent / "icons/Mask.png")
        button_action = QAction(QIcon(icon_path), "Create Masks", self)
        button_action.setStatusTip("Draw polygons to create image masks")
        button_action.triggered.connect(self.onMyToolBarCreateMask)
        toolbar.addAction(button_action)

        #COLOR SEGMENTATION ----------------------------------------------------------------------------------
        icon_path = str(Path(__file__).parent / "icons/ColorWheel_4.png")
        button_action = QAction(QIcon(icon_path), "Color Segmentation", self)
        button_action.setStatusTip("Create ROIs to segment regions by color")
        button_action.triggered.connect(self.onMyToolBarColorSegmentation)
        toolbar.addAction(button_action)

        # EDGE FILTERS-----------------------------------------------------------------------------------------
        icon_path = str(Path(__file__).parent / "icons/EdgeFilters_2.png")
        button_action = QAction(QIcon(icon_path), "Edge and Feature Detection", self)
        button_action.setStatusTip("Edge Detection Filters")
        button_action.triggered.connect(self.toolbarButtonEdgeDetection)
        toolbar.addAction(button_action)

        # SETTINGS --------------------------------------------------------------------------------------------
        icon_path = str(Path(__file__).parent / "icons/Settings_1.png")
        button_action = QAction(QIcon(icon_path), "Settings", self)
        button_action.setStatusTip("Change options and settings")
        button_action.triggered.connect(self.onMyToolBarSettings)
        toolbar.addAction(button_action)

        # GRIME2 -----------------------------------------------------------------------------------------------
        icon_path = str(Path(__file__).parent / "icons/grime2_StopSign.png")
        button_action = QAction(QIcon(icon_path), "GRIME2", self)
        button_action.setStatusTip("GRIME2 - Water Level Measurement")
        button_action.triggered.connect(self.toolbarButtonGRIME2)
        toolbar.addAction(button_action)

        # HELP -----------------------------------------------------------------------------------------------
        icon_path = str(Path(__file__).parent / "icons/Help_2.png")
        button_action = QAction(QIcon(icon_path), "Help", self)
        button_action.setStatusTip("Help and Release Notes")
        button_action.triggered.connect(self.toolbarButtonReleaseNotes)
        toolbar.addAction(button_action)

        # ====================================================================================================
        #self.actionRelease_Notes.triggered.connect(self.ReleaseNotes)

        toolbar.addAction(button_action)

    # --------------------------------------------------
    # --------------------------------------------------
    def itemClicked(self, item):
        global SITECODE
        global gWebImagesAvailable
        global gProcessClick
        global latestImage
        global gWebImageCount

        if gProcessClick == 0:
            gProcessClick = 1

            SITECODE = updateNEONSiteInfo(self, item)

            updateSiteProducts(self, item)

            nErrorCode, latestImage, gWebImageCount = NEON_DownloadLatestImage(SITECODE, DOMAINCODE)
            if nErrorCode == 404:
                gWebImagesAvailable = 0
                self.labelLatestImage.setText("No Images Available")
            else:
                gWebImagesAvailable = 1

                displayLatestImage(self)

            gProcessClick = 0

    # --------------------------------------------------
    # --------------------------------------------------
    def productClicked(self, item):
        updateNEONProductTable(self, item)

    # --------------------------------------------------
    # --------------------------------------------------

    def pushButton_RetrieveNEONDataClicked(self, item):
        downloadProductDataFiles(self, item)


    # ======================================================================================================================
    # THIS FUNCTION WILL CALL THE FUNCTION THAT PROCESSES THE IMAGE BASED UPON THE SETTINGS SELECTED BY THE
    # END-USER AND THEN UPDATE THE GUI TO DISPLAY THE PROCESSED IMAGE.
    # ======================================================================================================================
    def refreshImage(self):
        global currentImage

        '''// PROCESS THE ORIGINAL IMAGE //'''
        pix = processImage(self, currentImage)

        '''// DISPLAY PROCESSED ORIGINAL IMAGE //'''
        if not pix == []:
            self.labelEdgeImage.setPixmap(
                pix.scaled(self.labelEdgeImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            img = GRIMe_Utils.convertQImageToMat(currentImage.toImage())

        # CALL PROCESSEVENTS IN ORDER TO UPDATE GUI
        QCoreApplication.processEvents()

    # --------------------------------------------------
    # --------------------------------------------------
    def spinBoxOrbMaxFeaturesChanged(self):
        self.refreshImage()

    # --------------------------------------------------
    # --------------------------------------------------
    def clearoutImagePanels(self):
        self.labelOriginalImage.clear()
        self.labelEdgeImage.clear()

    # --------------------------------------------------
    # --------------------------------------------------
    def getImageFolder(self):
        imageFolder = self.lineEditHardDriveFolder.text()
        if len(imageFolder) > 0:
            if os.path.exists(imageFolder):
                return imageFolder
            else:
                return ''
        else:
            return ''

    # --------------------------------------------------
    # --------------------------------------------------
    def updateImageSource(self):
        # WE NO LONGER OPERATE ON IMAGES LIVE FROM THE WEBSITE.
        # WE NO LONGER OPERATE ON IMAGES LIVE FROM THE WEBSITE.
        # WE NOW DOWNLOAD IMAGES IN THE TIME/DATE RANGE SELECTED BY THE USER BEFORE WORKING WITH THE IMAGE.
        '''
        global gFrameCount
        global dailyImagesList
        global SITENAME

        bRefreshImages = False
        if self.radioButtonWebSiteImages.isChecked():
            gFrameCount = len(dailyImagesList.getVisibleList())
            self.labelNEONSite.setText(SITENAME)
            bRefreshImages = True
        elif self.radioButtonHardDriveImages.isChecked():
            self.labelNEONSite.setText('')

            imageFolder = self.getImageFolder()
            if len(imageFolder):
                gFrameCount = MSTGetFrameCount(self, imageFolder)
                self.labelNEONSite.setText('')
                bRefreshImages = True
            else:
                msg = QMessageBox()
                msg.setWindowTitle('Folder Path Required')
                msg.setText('Please specify the folder path where the images are located.')
                x = msg.exec_()
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)

        if bRefreshImages:
            self.clearoutImagePanels()

            self.spinBoxDailyImage.setValue(1)
            self.spinBoxDailyImage.setMaximum(gFrameCount)
            self.labelImageCountNumber.setText(str(gFrameCount))

            refreshImage(self)
        '''

    # --------------------------------------------------
    # --------------------------------------------------
    def MSTFetchVideoFilenames(self):
        self.lineEditHardDriveFolder.setText(MSTFolder)
        videoFilePath = Path(MSTFolder)
        # JES videoFileList = [str(pp) for pp in videoFilePath.glob("**/*.mp4")]
        videoFileList = [str(pp) for pp in videoFilePath.glob("*.mp4")]

        self.MSTspinBoxSelectImage.setMaximum(len(videoFileList))
        self.MSTspinBoxSelectImage.setValue(1)

        for videoFilename in videoFileList:
            self.MSTlistWidgetVideoFiles.addItem(os.path.basename(videoFilename))

    # --------------------------------------------------
    # --------------------------------------------------
    def MSTExtractFrames(self):
        videoFrameFolder = MSTFolder + '/' + self.MSTlineEditFrameFolder.text()
        video = MSTFolder + '/' + self.MSTlistWidgetVideoFiles.selectedItems()[0].text()

        BVS = MSTExtractFrames(video, videoFrameFolder, 0, 'ALL')
        BVS.process_all_frames()

    # --------------------------------------------------
    # --------------------------------------------------
    def MSTFrameCountUpdate(self):
        frameFolder = self.lineEditHardDriveFolder.text() + '/' + self.MSTlineEditFrameFolder.text()
        nFrameCount = MSTGetFrameCount(self, frameFolder)

        strFrameCount = 'Frame Count: ' + nFrameCount.__str__()
        self.MSTFrameCount.setText(strFrameCount)

        self.MSTspinBoxSelectImage.setMaximum(nFrameCount)
        self.MSTspinBoxSelectImage.setValue(1)

        global gFrameCount
        gFrameCount = nFrameCount

        self.spinBoxDailyImage.setMinimum(1)
        self.spinBoxDailyImage.setMaximum(nFrameCount)
        self.spinBoxDailyImage.setValue(1)

    # --------------------------------------------------
    # --------------------------------------------------
    def MSTStop(self):
        global nStop
        nStop = 1


    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def getMaxNumColorClusters(self):
        maxColorClusters = 0

        for roiObj in self.roiList:
            if roiObj.getNumColorClusters() > maxColorClusters:
                maxColorClusters = roiObj.getNumColorClusters()

        return maxColorClusters


    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def buildImageScalarHeader(self, header):

        nClusters = self.getMaxNumColorClusters()

        if self.checkBox_GCC.isChecked():
            header = header + ", GCC"

        if self.checkBox_GLI.isChecked():
            header = header + ", GLI"

        if self.checkBox_NDVI.isChecked():
            header = header + ", NDVI"

        if self.checkBox_ExG.isChecked():
            header = header + ", ExG"

        if self.checkBox_RGI.isChecked():
            header = header + ", RGI"

        if self.checkBox_Intensity.isChecked():
            header = header + ", Intensity"

        if self.checkBox_ShannonEntropy.isChecked():
            header = header + ", Entropy"

        if self.checkBox_Texture.isChecked():
            header = header + ", Texture"

        if self.checkBoxColor_HSV.isChecked():
            template = ', ' + 'Image_' + '#'

            if nClusters == 1:
                header = (header + template).replace('#', 'H')
                header = (header + template).replace('#', 'S')
                header = (header + template).replace('#', 'V')
            else:
                for idx in range(nClusters):
                    header = (header + template).replace('#', 'H') + (': ' + idx.__str__())
                    header = (header + template).replace('#', 'S') + (': ' + idx.__str__())
                    header = (header + template).replace('#', 'V') + (': ' + idx.__str__())

        return header

    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def buildROI_ScalarHeader(self, header):

        newHeader = header

        # ADD HEADER TO CSV FILE FOR EACH ROI ASSUMING ROIs HAVE BEEN CREATED
        for roiObj in self.roiList:
            template = ', ' + (roiObj.getROIName() + ': ') + '#'

            if self.checkBox_GCC.isChecked():
                newHeader = (newHeader + template).replace('#', 'GCC')

            if self.checkBox_GLI.isChecked():
                newHeader = (newHeader + template).replace('#', 'GLI')

            if self.checkBox_NDVI.isChecked():
                newHeader = (newHeader + template).replace('#', 'NDVI')

            if self.checkBox_ExG.isChecked():
                newHeader = (newHeader + template).replace('#', 'ExG')

            if self.checkBox_RGI.isChecked():
                newHeader = (newHeader + template).replace('#', 'RGI')

            if self.checkBox_Intensity.isChecked():
                newHeader = (newHeader + template).replace('#', 'Intensity')

            if self.checkBox_ShannonEntropy.isChecked():
                newHeader = (newHeader + template).replace('#', 'Entropy')

            if self.checkBox_Texture.isChecked():
                newHeader = (newHeader + template).replace('#', 'Texture')

                    # IF THERE IS MORE THAN ONE (1) ROI, APPEND AN INDEX ONTO THE HEADER LABEL

            nClusters = roiObj.getNumColorClusters()

            if self.checkBoxColor_HSV.isChecked():
                template = ', ' + (roiObj.getROIName() + ': ') + '#'

                if nClusters == 1:
                    newHeader = (newHeader + template).replace('#', 'H')
                    newHeader = (newHeader + template).replace('#', 'S')
                    newHeader = (newHeader + template).replace('#', 'V')
                else:
                    for idx in range(nClusters):
                        newHeader = (newHeader + template).replace('#', 'H') + ('_' + idx.__str__())
                        newHeader = (newHeader + template).replace('#', 'S') + ('_' + idx.__str__())
                        newHeader = (newHeader + template).replace('#', 'V') + ('_' + idx.__str__())

        newHeader = newHeader + '\n'

        return newHeader


    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def buildScalarHeader(self, nClusters):

        # CREATE HEADER FOR THE ATTIRBUTE OF THE ENTIRE IMAGE
        header = 'Image, Date (ISO), Time (ISO)'

        if self.checkBoxScalarRegion_WholeImage.isChecked():
            header = self.buildImageScalarHeader(header)

        if self.checkBoxScalarRegion_ROI.isChecked():
            header = self.buildROI_ScalarHeader(header)

        return header


    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def GRIMe_NSF_Build(self):
        global dailyImagesList
        global nStop
        global gWebImageCount

        bCreateEXIFFile = False
        bCreateVideos = False

        myGRIMe_Color = GRIMe_Color()

        global imageFileFolder
        hardDriveImageFolder = imageFileFolder + '/'

        # GENERATE LIST OF IMAGE FILES IN FOLDER
        videoFileList = dailyImagesList.getVisibleList()
        nFrameCount = len(videoFileList)

        if nFrameCount > 0:

            # ----------------------------------------------------------------------------------------------------
            # CREATE AND OPEN NEW TRAINING DATA CSV FILE AND LABEL COLUMNS
            # ----------------------------------------------------------------------------------------------------
            csvFilename = 'TrainingData_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
            imageQualityFile = os.path.join(hardDriveImageFolder, csvFilename)
            csvFile = open(imageQualityFile, 'a', newline='')

            # ----------------------------------------------------------------------------------------------------
            # OUTPUT THE TRAINING FILE a) NAMES, b) TRAINED COLOR CLUSTERS, c) IMAGE INTENSITY, d) ETC.
            # ----------------------------------------------------------------------------------------------------
            nMaxNumColorClusters = self.getMaxNumColorClusters()

            for roiObj in self.roiList:
                strOutputString = 'Training Filename'

                strOutputString = strOutputString + ',Date (ISO),Time (ISO),Greenness,Intensity,Entropy'

                template = ', ' + (roiObj.getROIName() + ': ') + '#'

                if nMaxNumColorClusters == 1:
                    strOutputString = (strOutputString + template).replace('#', 'H')
                    strOutputString = (strOutputString + template).replace('#', 'S')
                    strOutputString = (strOutputString + template).replace('#', 'V')
                else:
                    for idx in range(nMaxNumColorClusters):
                        strOutputString = (strOutputString + template).replace('#', 'H') + ('_' + idx.__str__())
                        strOutputString = (strOutputString + template).replace('#', 'S') + ('_' + idx.__str__())
                        strOutputString = (strOutputString + template).replace('#', 'V') + ('_' + idx.__str__())
                strOutputString = strOutputString + "\n"

                csvFile.write(strOutputString)

                # LOAD THE IMAGE FILE
                img = myGRIMe_Color.loadColorImage(roiObj.getTrainingImageName())

                # BLUR THE IMAGE
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                height, width = gray.shape
                gray = cv2.resize(gray, (width, height))

                # IMAGE INTENSITY CALCULATIONS
                intensity = cv2.mean(gray)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway

                # COMPUTE ENTROPY FOR ENTIRE IMAGE
                entropyValue = self.calcEntropy(gray)[0]

                texture = -999

                red, green, blue = separateChannels(img)
                redSum, greenSum, blueSum = sumChannels(red, green, blue)

                # CREATE HYPERLINK TO FILE
                strOutputString = '=HYPERLINK(' + '"' + roiObj.getTrainingImageName() + '"' + ')'
                strOutputString = strOutputString + ','     # DATE
                strOutputString = strOutputString + ','     # TIME

                strOutputString = strOutputString + ',' + self.computeGreennessValue(redSum, greenSum, blueSum)

                strOutputString = strOutputString + ',' + '%3.4f' % intensity
                strOutputString = strOutputString + ',' + '%3.4f' % entropyValue

                hsvClusterCenters, hist = roiObj.getHSVClusterCenters()
                for i in range(nMaxNumColorClusters):
                    strOutputString = strOutputString + ',' + str(hsvClusterCenters[i][0]) + ',' + str(hsvClusterCenters[i][1]) + ',' + str(hsvClusterCenters[i][2])
                strOutputString = strOutputString + '\n'

                csvFile.write(strOutputString)

            # ONCE THE NUMBER OF COLOR CLUSTERS HAS BEEN SELECTED AND AN ROI TRAINED, LOCK THE NUMBER OF COLOR CLUSTERS
            # AND TRAIN ALL SUBSEQUENT ROIs FOR THE SAME NUMBER OF COLOR CLUSTERS
            nClusters = self.spinBoxColorClusters.value()

            header = '\n' + self.buildScalarHeader(nClusters)

            # WRITE THE HEADER TO THE CSV
            csvFile.write(header)

            # ----------------------------------------------------------------------------------------------------
            # CREATE PROGRESS WHEEL
            # ----------------------------------------------------------------------------------------------------
            progressBar = QProgressWheel(0, (len(self.roiList) * len(videoFileList)) + 1)
            progressBar.show()
            progressBarIndex = 1

            # ----------------------------------------------------------------------------------------------------
            # PROCESS IMAGES
            # ----------------------------------------------------------------------------------------------------
            #E:\\000 - University of Nebraska\\2022_USGS104b\\imagery\\PBT_MittelstetsMeanderCam\\20220709
            for fname in videoFileList:

                if os.path.isfile(fname.fullPathAndFilename):

                    # LOAD THE IMAGE FILE
                    img = myGRIMe_Color.loadColorImage(fname.fullPathAndFilename)

                    # CREATE HYPERLINK TO FILE
                    strHyperlink = '=HYPERLINK(' + '"' + fname.fullPathAndFilename + '"' + ')'

                    # EXTRACT DATE/TIME STAMP FROM IMAGE EXIF DATA
                    image_date, image_time = self.extractEXIFdata(fname.fullPathAndFilename)
                    strDate = image_date.isoformat().strip()
                    strTime = image_time.isoformat().strip()

                    strOutputString = '%s,%s,%s' % (strHyperlink, strDate, strTime)

                    # ------------------------------------------------------------------------------------------
                    # CALCULATE THE FEATURE SCALARS FOR THE ENTIRE IMAGE AND SAVE THEM TO THE CSV FILE
                    # ------------------------------------------------------------------------------------------
                    if self.checkBoxScalarRegion_WholeImage.isChecked():
                        # BLUR THE IMAGE
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # IMAGE INTENSITY CALCULATIONS
                        intensity = cv2.mean(gray)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway

                        # COMPUTE ENTROPY FOR ENTIRE IMAGE
                        entropyValue = self.calcEntropy(gray)[0]

                        # EXTRACT 'n' DOMINANT HSV COLORS
                        hist, clusterCenters = myGRIMe_Color.extractDominant_HSV(img, nClusters)

                        texture = -999

                        red, green, blue = separateChannels(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        redSum, greenSum, blueSum = sumChannels(red, green, blue)

                        strOutputString = strOutputString + ',' + self.computeGreennessValue(redSum, greenSum, blueSum)

                        if self.checkBox_Intensity.isChecked():
                            strOutputString = strOutputString + ', %3.4f' % intensity
                        if self.checkBox_ShannonEntropy.isChecked():
                            strOutputString = strOutputString + ', %3.4f' % entropyValue
                        if self.checkBox_Texture.isChecked():
                            strOutputString = strOutputString + ', %3.2f' % texture

                        if self.checkBoxColor_HSV.isChecked():
                            for idx in range(nClusters):
                                # CONVERT FROM OpenCV's HSV HUE DATA FORMAT 0 to 180 DEGREES TO THE HSV STANDARD FORMAT OF 0 to 360 DEGREES
                                h = float(clusterCenters[idx][0])
                                s = float(clusterCenters[idx][1])
                                v = float(clusterCenters[idx][2])
                                strOutputString = strOutputString + ', %3.2f, %3.2f, %3.2f' % (h, s, v)

                    # ------------------------------------------------------------------------------------------
                    #
                    # ------------------------------------------------------------------------------------------
                    progressBar.setValue(progressBarIndex)
                    progressBarIndex = progressBarIndex + 4

                    if self.checkBoxScalarRegion_ROI.isChecked():

                        strROI = self.calculateROIScalars(img)

                        strOutputString = strOutputString + strROI


                    # WRITE STRING TO CSV FILE
                    strOutputString = strOutputString  + '\n'
                    csvFile.write(strOutputString)

            csvFile.close()

            # CLOSE AND DELETE THE PROGRESSBAR
            progressBar.close()
            del progressBar


    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def trainROI(self, roiParameters):
        global currentImage

        if currentImage:
            myGRIMe_Color = GRIMe_Color()

            # CREATE AN ROI OBJECT
            roiObj = GRIMe_roiData()

            # POPULATE ROI OBJECT WITH ROI INFORMATION
            # --------------------------------------------------
            if len(roiParameters.strROIName) > 0:
                roiObj.setROIName(roiParameters.strROIName)
            else:
                msgBox = GRIMe_QMessageBox('ROI Error', 'A name for the ROI is required!')
                response = msgBox.displayMsgBox()
                return

            # --------------------------------------------------
            rectROI = self.labelOriginalImage.getROI()

            if rectROI != None:
                roiObj.setDisplayROI(rectROI)
            else:
                msgBox = GRIMe_QMessageBox('ROI Error', 'Please draw the ROI on the image!')
                response = msgBox.displayMsgBox()
                return

            # --------------------------------------------------
            try:
                roiObj.setImageSize(currentImage.size())
                scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                roiObj.setDisplaySize(scaledCurrentImage.size())
                roiObj.calcROI()

                if full == 1:
                    roiObj.setROIShape(ROIShape.RECTANGLE)
                    #if self.radioButton_ROIShapeRectangle.isChecked():
                    #roiObj.setROIShape(ROIShape.RECTANGLE)
                    #else:
                    #roiObj.setROIShape(ROIShape.ELLIPSE)
                else:
                    roiObj.setROIShape(ROIShape.RECTANGLE)
            except:
                msgBox = GRIMe_QMessageBox('ROI Error',
                                           'An unexpected error occurred calculating the ROI of the full resolution image!')
                response = msgBox.displayMsgBox()

                return

            # ----------------------------------------------------------------------------------------------------
            # CALCULATE COLOR CLUSTERS FOR THE ROI AND SAVE THEM TO THE ROI LIST
            # ----------------------------------------------------------------------------------------------------
            # EXTRACT THE ROI FROM THE ORIGINAL IMAGE
            img1 = GRIMe_Utils.convertQImageToMat(currentImage.toImage())
            rgb = extractROI(roiObj.getImageROI(), img1)

            roiObj.setNumColorClusters(roiParameters.numColorClusters)

            # EXTRACT DOMINANT RGB COLORS AND ADD THEM TO THE ROI OBJECT
            qImg, clusterCenters, hist = myGRIMe_Color.KMeans(rgb, roiObj.getNumColorClusters())
            roiObj.setClusterCenters(clusterCenters, hist)

            # EXTRACT DOMINANT HSV COLORS AND ADD THEM TO THE ROI OBJECT
            hist, colorClusters = myGRIMe_Color.extractDominant_HSV(rgb, roiObj.getNumColorClusters())
            roiObj.setHSVClusterCenters(colorClusters, hist)

            roiObj.setTrainingImageName(currentImageFilename)

            self.roiList.append(roiObj)

            # ----------------------------------------------------------------------------------------------------
            # DISPLAY IN FEATURE TABLE
            # ----------------------------------------------------------------------------------------------------
            # CREATE NEW ROW IN ROI TABLE
            nRow = self.tableWidget_ROIList.rowCount()
            self.tableWidget_ROIList.insertRow(nRow)

            # CREATE COLOR BAR TO DISPLAY CLUSTER COLORS
            colorBar = self.createColorBar(hist, colorClusters)

            # CONVERT colorBar TO A QImage FOR USE IN DISPLAYING IN QT GUI
            qImg = QImage(colorBar.data, colorBar.shape[1], colorBar.shape[0], QImage.Format_BGR888)

            # INSERT THE DOMINANT COLORS INTO A QLabel IN ORDER TO ADD IT TO THE FEATURE TABLE
            self.label = QtWidgets.QLabel()
            self.label.setPixmap(QPixmap(qImg.scaled(100, 50)))
            self.tableWidget_ROIList.setCellWidget(nRow, 1, self.label)

            # INSERT ROI NAME INTO TABLE
            nCol = 0
            self.tableWidget_ROIList.setItem(nRow, nCol, QTableWidgetItem(roiParameters.strROIName))

            self.tableWidget_ROIList.resizeColumnsToContents()

            #JES pix = QPixmap(QImage(colorBar.data, colorBar.shape[1], colorBar.shape[0], QImage.Format_BGR888))
            #JES self.labelDominantColors.setPixmap(pix.scaled(self.labelEdgeImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

            # self.tableWidget_ROIList.setCellWidget(nRow, nCol, QCheckBox())

            # ----------------------------------------------------------------------------------------------------
            #
            # ----------------------------------------------------------------------------------------------------
            global currentImageIndex

            processLocalImage(self, currentImageIndex)
            self.refreshImage()

            # ----------------------------------------------------------------------------------------------------
            # ONCE AN ROI IS DEFINED FOR A SPECIFIC NUMBER OF COLOR CLUSTERS, DISABLE THE CONTROL SO THAT THE USER
            # CANNOT CHANGE THE VALUE FOR SUBSEQUENT TRAINED ROIs.
            # ----------------------------------------------------------------------------------------------------
            #JES self.spinBoxColorClusters.setDisabled(True)


    # ------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------
    def computeGreennessValue(self, redSum, greenSum, blueSum):
        if self.checkBox_GCC.isChecked():
            return('%3.2f' % computeGreennessIndex(redSum, greenSum, blueSum))
        if self.checkBox_GLI.isChecked():
            return('%3.2f' % computeGreenLeafIndex(redSum, greenSum, blueSum))
        if self.checkBox_NDVI.isChecked():
            return('%3.2f' % computeNormalizedDifferenceVegetationIndex(redSum, greenSum))
        if self.checkBox_ExG.isChecked():
            return('%3.2f' % computeExcessGreenIndex(redSum, greenSum, blueSum))
        if self.checkBox_RGI.isChecked():
            return('%3.4f' % computeRedGreenIndex(redSum, greenSum))


    # ------------------------------------------------------------------------------------------
    # CALCULATE THE FEATURE SCALARS FOR THE VARIOUS ROIs AND SAVE THEM TO THE CSV FILE
    # ------------------------------------------------------------------------------------------
    def calculateROIScalars(self, img):

        strOutputString = ''

        for roiObj in self.roiList:
            texture = -999

            nClusters = roiObj.getNumColorClusters()

            rgb = extractROI(roiObj.getImageROI(), img)

            # CONVERT THE IMAGE FROM BGR TO RGB AND HSV
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            # IMAGE INTENSITY CALCULATIONS
            intensity = cv2.mean(gray)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway

            # COMPUTE ENTROPY FOR ROI
            entropyValue = self.calcEntropy(gray)

            # EXTRACT 'n' DOMINANT HSV COLORS
            myGRIMe_Color = GRIMe_Color()
            qhsvImg, hsvClusterCenters, hist = myGRIMe_Color.KMeans(hsv, nClusters)

            # KMeans QUANTIZES THE HUE VALUE TO 0..180 WHEN THE ACTUAL HSV COLOR SPACE HUE VALUE 0..360.
            # THEREFORE WE MULTIPLY THE KMeans HUE VALUE BY 2 TO STANDARDIZE ON THE ACTUAL COLOR SPACE HUE RANGE.
            hsvClusterCenters[:, 0] = hsvClusterCenters[:, 0] * 2.0

            red, green, blue = separateChannels(rgb)
            redSum, greenSum, blueSum = sumChannels(red, green, blue)

            strOutputString = strOutputString + ',' + self.computeGreennessValue(redSum, greenSum, blueSum)

            if self.checkBox_Intensity.isChecked():
                strOutputString = strOutputString + ', %3.4f' % intensity
            if self.checkBox_ShannonEntropy.isChecked():
                strOutputString = strOutputString + ', %3.4f' % entropyValue
            if self.checkBox_Texture.isChecked():
                strOutputString = strOutputString + ', %3.2f' % texture
            if self.checkBoxColor_HSV.isChecked():
                for idx in range(nClusters):
                    # CONVERT FROM OpenCV's HSV HUE DATA FORMAT 0 to 180 DEGREES TO THE HSV STANDARD FORMAT OF 0 to 360 DEGREES
                    h = float(hsvClusterCenters[idx][0])
                    s = float(hsvClusterCenters[idx][1])
                    v = float(hsvClusterCenters[idx][2])
                    strOutputString = strOutputString + ', %3.2f, %3.2f, %3.2f' % (h, s, v)

        return strOutputString


    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def calcEntropy(self, img):
        entropy = []

        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        total_pixel = img.shape[0] * img.shape[1]

        for item in hist:
            probability = item / total_pixel
            if probability == 0:
                en = 0
            else:
                en = -1 * probability * (np.log(probability) / np.log(2))
            entropy.append(en)

        try:
            sum_en = np.sum(entropy)
        except:
            sum_en = 0.0

        return sum_en


    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def extractEXIFdata(self, fullPathAndFilename):
        nYear = 1970
        nMonth = 1
        nDay = 1

        try:
            # extract EXIF info to determine what time the image was acquired. If EXIF info is not found,
            # throw an exception and see if the information is embedded in the filename. Currently, we are
            # working with images from NEON and PBT. The PBT images have EXIF data and the NEON/PhenoCam
            # do not appear to have EXIF data.
            myEXIFData = EXIFData()
            myEXIFData.extractEXIFData(fullPathAndFilename)

            strTemp = str(myEXIFData.getEXIF()[8])
            dateOriginal = re.search('\d{4}:\d{2}:\d{2}', strTemp).group(0)
            timeOriginal = re.search(' \d{2}:\d{2}:\d{2}', strTemp).group(0)

            nYear = int(str(dateOriginal[0:4]))
            nMonth = int(str(dateOriginal[5:7]))
            nDay = int(str(dateOriginal[8:10]))

            nHours = int(str(timeOriginal[1:3]))
            nMins = int(str(timeOriginal[4:6]))
            nSecs = int(str(timeOriginal[7:9]))

            bEXIFDataFound = True
        except:
            # assume the filename contains the timestamp for the image (assumes the image file is a PBT image)
            bEXIFDataFound = False

            try:
                nHours = int(str(strTime[0:2]))
                nMins = int(str(strTime[2:4]))
                nSecs = int(str(strTime[4:6]))
            except:
                nHours = 0
                nMins = 0
                nSecs = 0

        image_date = datetime.date(nYear, nMonth, nDay)
        image_time = datetime.time(nHours, nMins, nSecs)

        return(image_date, image_time)


    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def saveSettings(self):
        configFilePath = os.path.expanduser('~')
        configFilePath = os.path.join(configFilePath, 'Documents')
        configFilePath = os.path.join(configFilePath, 'GRIMe-AI')
        if not os.path.exists(configFilePath):
            os.mkdir(configFilePath)
        configFilePath = os.path.join(configFilePath, 'Settings')
        if not os.path.exists(configFilePath):
            os.mkdir(configFilePath)

        config = ConfigParser()

        configFile = os.path.join(configFilePath, 'GRIMe-AI.cfg')
        f = open(configFile, 'w+')

        config.read(configFile)
        config.add_section('ROI')
        config.set('ROI', 'Sky', 'TBD')
        config.set('ROI', 'Grass1', 'TBD')
        config.set('ROI', 'Grass2', 'TBD')
        config.set('ROI', 'Trees1', 'TBD')
        config.set('ROI', 'Trees2', 'TBD')

        config.write(f)
        f.close()

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def createColorBar(self, hist, centroids):
        # initialize the bar chart representing the relative frequency of each of the colors
        # bar = np.zeros((50, 300, 3), dtype="uint8")
        colorBar = np.zeros((50, 100, 3), dtype="uint8")

        startX = 0

        # if only some of the clusters are used, we must rescale their size based upon their percentages to fill the bitmap
        multiplier = 1.0 / hist.sum()

        centroids[:, 0] = centroids[:, 0] / 2.0
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * multiplier * 100)
            cv2.rectangle(colorBar, (int(startX), 0), (int(endX), 100), color.astype("uint8").tolist(), -1)

            startX = endX

        # CONVERT HSV TO RGB
        colorBar = cv2.cvtColor(colorBar, cv2.COLOR_HSV2RGB)

        # RETURN THE COLOR BAR
        return colorBar


    # --------------------------------------------------
    # --------------------------------------------------
    def displayROIs(self):
        processLocalImage(self)
        self.refreshImage()


    # --------------------------------------------------
    # --------------------------------------------------
    def initProductTable(self):
        headerList = ['<-Select', 'Site', "Image Count", ' min Date ', ' max Date ', 'Start Date', 'End Date', 'Start Time', 'End Time']
        stylesheet = "::section{Background-color:rgb(116,175,80);border-radius:14px;}"
        header = self.tableProducts.horizontalHeader()

        i = 0
        for item in headerList:
            headerItem = QTableWidgetItem(item)
            headerItem.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tableProducts.setHorizontalHeaderItem(i, headerItem)
            self.tableProducts.setStyleSheet(stylesheet)
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
            i = i + 1

        font = QFont()
        font.setBold(True)
        self.tableProducts.horizontalHeader().setFont(font)
        self.tableProducts.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.tableProducts.resizeColumnsToContents()

        try:
            self.tableWidget_ROIList.horizontalHeader().setVisible(True)
        except:
             pass

    # --------------------------------------------------
    # --------------------------------------------------
    def initUSGS_Table(self):
        headerList = ['Select', 'Site', 'Image Count', ' min Date ', ' max Date ', 'Start Date', 'End Date', 'Start Time', 'End Time']
        stylesheet = "::section{Background-color:rgb(116,175,80);border-radius:14px;}"
        header = self.table_USGS_Sites.horizontalHeader()

        i = 0
        for item in headerList:
            headerItem = QTableWidgetItem(item)
            headerItem.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table_USGS_Sites.setHorizontalHeaderItem(i, headerItem)
            self.table_USGS_Sites.setStyleSheet(stylesheet)
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
            i = i + 1

        font = QFont()
        font.setBold(True)
        self.table_USGS_Sites.horizontalHeader().setFont(font)
        self.table_USGS_Sites.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.table_USGS_Sites.resizeColumnsToContents()


    # --------------------------------------------------
    # --------------------------------------------------
    def initROITable(self):
        headerList = ['ROI Name', 'Ref. Image', 'Cur. Image', 'Greenness', 'Intensity', 'Entropy']
        stylesheet = "::section{Background-color:rgb(116,175,80);border-radius:14px;}"
        header = self.tableWidget_ROIList.horizontalHeader()
        self.tableWidget_ROIList.setColumnCount(len(headerList))
        i = 0
        for item in headerList:
            headerItem = QTableWidgetItem(item)
            headerItem.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tableWidget_ROIList.setHorizontalHeaderItem(i, headerItem)
            self.tableWidget_ROIList.setStyleSheet(stylesheet)
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
            i = i + 1

        font = QFont()
        font.setBold(True)
        self.tableWidget_ROIList.horizontalHeader().setFont(font)
        self.tableWidget_ROIList.horizontalHeader().setVisible(True)

    # --------------------------------------------------
    # --------------------------------------------------
    def radioButtonHardDriveImagesClicked(self):
        #self.checkBoxCreateEXIFFile.setEnabled(True)
        self.lineEditHardDriveFolder.setEnabled(True)
        if len(self.lineEditHardDriveFolder.text()) > 0:
            if os.path.exists(self.lineEditHardDriveFolder.text()):
                self.updateImageSource()

    def tabWidgetChanged(self, index):
        #JES if full == 1 or full == 4:
        #JES     if self.tabWidget_ToolPalette.currentIndex() == 0:
        #JES        self.tabWidget_ToolPalette.setCurrentIndex(1)
        pass

    def ROIShapeClicked(self):
        if self.radioButton_ROIShapeRectangle.isChecked() == True:
            self.labelOriginalImage.setROIShape(ROIShape.RECTANGLE)
        elif self.radioButton_ROIShapeEllipse.isChecked() == True:
            self.labelOriginalImage.setROIShape(ROIShape.ELLIPSE)

    def pushButtonColorSegmentationClicked(self):
        global currentImage

        myGRIMe_Color = GRIMe_Color()

        if len(self.roiList) == 0:
            strError = "You must train at least one ROI before segmenting the image."
            msgBox = GRIMe_QMessageBox('Color Segmentation Error', strError)
            response = msgBox.displayMsgBox()
            return

        # ----------------------------------------------------------------------------------------------------
        # DISPLAY IMAGE FROM NEON SITE
        # ----------------------------------------------------------------------------------------------------
        if currentImage:
            scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            width = scaledCurrentImage.width()
            height = scaledCurrentImage.height()

            # KMeans EXPECTS THE BYTE ORDER TO BE RGB
            img1 = GRIMe_Utils.convertQImageToMat(currentImage.toImage())

            rgb = cv2.blur(img1, ksize=(11, 11))

            # convert image to HSV
            hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)

            if len(self.roiList) > 0:
                # DIAGNOSTICS
                if self.checkBoxColorDiagnostics.checkState():
                    GRIMe_Diagnostics.RGB3DPlot(rgb)
                    GRIMe_Diagnostics.plotHSVChannelsGray(hsv)
                    GRIMe_Diagnostics.plotHSVChannelsColor(hsv)

                # segment colors
                rgb1 = myGRIMe_Color.segmentColors(rgb, hsv, self.roiList)

            # display the segemented image on the GUI
            qImg = QImage(rgb1.data, rgb1.shape[1], rgb1.shape[0], QImage.Format_BGR888)
            pix = QPixmap(qImg)
            self.labelColorSegmentation.setPixmap(pix.scaled(self.labelColorSegmentation.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def checkboxNEONSitesClicked(self):
        if self.checkBoxNEONSites.isChecked():
            # GET LIST OF ALL SITES ON NEON
            siteList = readFieldSiteTable()

    def pushButton_USGSDownloadClicked(self):
        fetchUSGSImages(self.table_USGS_Sites, self.EditSaveImages_USGS_DownloadFolder)

    def pushButtonTrainGoodTriggered(self):
        blur, intensity = computeBlurAndBrightness(self.spinBoxShiftSize.value())
        imageStats = GRIMe_ImageStats()
        imageStats.setBlurValue(blur)
        imageStats.setBrightnessValue(intensity)
        imageStats.setLabel('good')
        self.imageStatsList.append(imageStats)

    def pushButtonTrainBadTriggered(self):
        blur, intensity = computeBlurAndBrightness(self.spinBoxShiftSize.value())
        imageStats = GRIMe_ImageStats()
        imageStats.setBlurValue(blur)
        imageStats.setBrightnessValue(intensity)
        imageStats.setLabel('bad')
        self.imageStatsList.append(imageStats)

    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def toolbarButtonReleaseNotes(self):
        releaseNotesDlg = GRIME_ReleaseNotesDlg(frame)

        releaseNotesDlg.show()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def toolbarButtonGRIME2(self):
        strMessage = 'Potential future home for GRIME2 Water Level/Stage measurement functionality.'
        msgBox = GRIMe_QMessageBox('Water Level Measurement', strMessage, QMessageBox.Close)
        response = msgBox.displayMsgBox()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def toolbarButtonEdgeDetection(self):
        self.edgeDetectionDlg = GRIME_EdgeDetectionDlg(frame)

        self.edgeDetectionDlg.edgeDetectionSignal.connect(self.edgeDetectionMethod)
        self.edgeDetectionDlg.featureDetectionSignal.connect(self.featureDetectionMethod)

        self.edgeDetectionDlg.show()

    def edgeDetectionMethod(self, edgeMethod):
        global g_edgeMethodSettings
        g_edgeMethodSettings.method = edgeMethod.method
        g_edgeMethodSettings.selected = edgeMethod.selected

        processLocalImage(self)
        self.refreshImage()

    def featureDetectionMethod(self, featureMethod):
        global g_featureMethodSettings
        g_featureMethodSettings.method = featureMethod.method
        g_featureMethodSettings.selected = featureMethod.selected

        processLocalImage(self)
        self.refreshImage()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def onMyToolBarFileFolder(self):
        self.fileFolderDlg = GRIME_FileUtilitiesDlg(frame)
        self.fileFolderDlg.fetchImageList_Signal.connect(self.fetchImageList)

        global imageFileFolder
        if len(imageFileFolder) > 0 and os.path.exists(imageFileFolder):
            self.fileFolderDlg.setImageFolder(imageFileFolder)

        self.fileFolderDlg.show()

        try:
            global gFrameCount
            self.imageNavigationDlg.setImageCount(gFrameCount)
            self.imageNavigationDlg.reset()
        except:
            pass

    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def onMyToolBarImageNavigation(self):

        if self.imageNavigationDlg == None:
            global gFrameCount
            global currentImageCount

            if gFrameCount > 0:
                self.imageNavigationDlg = GRIME_ImageNavigationDlg(frame)
                self.imageNavigationDlg.imageIndexSignal.connect(self.getImageIndex)

                self.imageNavigationDlg.accepted.connect(self.closeNavigationDlg)
                self.imageNavigationDlg.rejected.connect(self.closeNavigationDlg)

                self.imageNavigationDlg.setImageIndex(currentImageIndex)
                self.imageNavigationDlg.setImageCount(gFrameCount)
                #self.imageNavigationDlg.reset()

                self.imageNavigationDlg.show()
            else:
                strMessage = 'You must first fetch images to navigate and/or operate on.'
                msgBox = GRIMe_QMessageBox('Image Navigation', strMessage, QMessageBox.Close)
                response = msgBox.displayMsgBox()

    def getImageIndex(self, imageIndex):
        global currentImageIndex

        currentImageIndex = imageIndex

        processLocalImage(self, imageIndex)
        self.refreshImage()

    def closeNavigationDlg(self):
        del self.imageNavigationDlg

        self.imageNavigationDlg = None


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    #@pyqtSlot()
    def onMyToolBarColorSegmentation(self):
        if self.maskEditorDlg == None:
            self.labelOriginalImage.setDrawingMode(DrawingMode.COLOR_SEGMENTATION)

            self.colorSegmentationDlg = GRIME_ColorSegmentationDlg()

            self.colorSegmentationDlg.colorSegmentation_Signal.connect(self.colorSegmentation)
            self.colorSegmentationDlg.addROI_Signal.connect(self.trainROI)
            self.colorSegmentationDlg.deleteAllROI_Signal.connect(self.deleteAllROI)
            self.colorSegmentationDlg.close_signal.connect(self.closeColorSegmentationDlg)

            self.colorSegmentationDlg.accepted.connect(self.closeColorSegmentationDlg)
            self.colorSegmentationDlg.rejected.connect(self.closeColorSegmentationDlg)

            self.colorSegmentationDlg.show()

    # ----------------------------------------------------------------------------------------------------------------------
    def closeColorSegmentationDlg(self):
        if self.colorSegmentationDlg != None:
            self.colorSegmentationDlg.close()
            del self.colorSegmentationDlg
            self.colorSegmentationDlg = None

        self.labelOriginalImage.setDrawingMode(DrawingMode.OFF)

    # ----------------------------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------------------------
    def colorSegmentation(self, int):
        global dailyImagesList
        videoFileList = dailyImagesList.getVisibleList()

        myGRIMe_Color = GRIMe_Color()

        nImageIndex = 1

        if len(videoFileList) > 0:
            if nImageIndex > gFrameCount:
                nImageIndex = gFrameCount

            inputFrame = videoFileList[nImageIndex - 1].fullPathAndFilename  # zero based index

            if os.path.isfile(inputFrame):
                global currentImageFilename
                currentImageFilename = inputFrame
                numpyImage = myGRIMe_Color.loadColorImage(inputFrame)

                hsv = cv2.cvtColor(numpyImage, cv2.COLOR_BGR2HSV)

                # Threshold of blue in HSV space
                lower_blue = np.array([60, 35, 140])
                upper_blue = np.array([180, 255, 255])

                # preparing the mask to overlay
                mask = cv2.inRange(hsv, lower_blue, upper_blue)

                # The black region in the mask has the value of 0,
                # so when multiplied with original image removes all non-blue regions
                result = cv2.bitwise_and(numpyImage, numpyImage, mask=mask)

                cv2.imshow('frame', numpyImage)
                cv2.imshow('mask', mask)
                cv2.imshow('result', result)

                tempCurrentImage = QImage(numpyImage, numpyImage.shape[1], numpyImage.shape[0], QImage.Format_RGB888)
                currentImage = QPixmap(tempCurrentImage)

    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def onMyToolBarCreateMask(self):

        if self.colorSegmentationDlg == None:
            self.labelOriginalImage.setDrawingMode(DrawingMode.MASK)

            self.maskEditorDlg = GRIME_MaskEditorDlg()

            self.maskEditorDlg.addMask_Signal.connect(self.addMask)
            self.maskEditorDlg.generateMask_Signal.connect(self.generateMask)
            self.maskEditorDlg.drawingColorChange_Signal.connect(self.changePolygonColor)
            self.maskEditorDlg.reset_Signal.connect(self.resetMask)
            self.maskEditorDlg.close_signal.connect(self.maskDialogClose)
            self.maskEditorDlg.polygonFill_Signal.connect(self.fillPolygonChanged)

            self.maskEditorDlg.show()

    def fillPolygonChanged(self, bFill):
        self.labelOriginalImage.enablePolygonFill(bFill)

    def maskDialogClose(self):
        if self.maskEditorDlg == None:
            self.maskEditorDlg.close()
            del self.maskEditorDlg
            self.maskEditorDlg = None

        self.labelOriginalImage.setDrawingMode(DrawingMode.OFF)

    def resetMask(self):
        self.labelOriginalImage.resetMask()
        self.labelOriginalImage.update()

    def addMask(self):
        self.labelOriginalImage.incrementPolygon()

    def generateMask(self):

        #global imageFileFolder
        global currentImageFilename

        scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio,
                                                 QtCore.Qt.SmoothTransformation)

        widthMultiplier = currentImage.size().width() / scaledCurrentImage.size().width()
        heightMultiplier = currentImage.size().height() / scaledCurrentImage.size().height()

        # CONVERT IMAGE TO A MAT FORMAT TO USE ITS PARAMETERS TO CREATE A MASK IMAGE TEMPLATE
        # ------------------------------------------------------------------------------------------------------------------------
        img1 = GRIMe_Utils.convertQImageToMat(currentImage.toImage())

        # CREATE A MASK IMAGE
        mask = np.zeros(img1.shape[:2], np.uint8)

        # ITERATE THROUGH EACH ONE OF THE POLYGONS
        # ------------------------------------------------------------------------------------------------------------------------
        polygonList = self.labelOriginalImage.getPolygon()

        for polygon in polygonList:
            myPoints = []
            for i in range(polygon.count()):
                myPoints.append([polygon.point(i).x() * widthMultiplier, polygon.point(i).y() * heightMultiplier])

            cv2.fillPoly(mask, np.int32([myPoints]), color=(255, 255, 255))

        masked = cv2.bitwise_and(img1, img1, mask=mask)

        # DISPLAY THE MASK IN THE GUI
        # ------------------------------------------------------------------------------------------------------------------------
        qImg = QImage(masked.data, masked.shape[1], masked.shape[0], QImage.Format_BGR888)
        pix = QPixmap(qImg)
        self.labelColorSegmentation.setPixmap(pix.scaled(self.labelColorSegmentation.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        # SAVE THE MASK RASTER AND POLYGON TO FILE
        if self.maskEditorDlg.getCheckBox_Save():
            # Extract image folder path to create a mask subfolder
            maskFolderPath = os.path.join(os.path.dirname(currentImageFilename), 'Masks')

            # Check for the existence of the filename path and create if it doesn't exist
            if not os.path.exists(maskFolderPath):
                os.mkdir(maskFolderPath)

            # Extract image filename to be used for creating the mask and polygon filenames
            filename = os.path.basename(currentImageFilename)
            filename_without_ext = filename[:filename.rindex('.')]
            extension = filename[filename.rindex('.'):]

            # Create filenames with fully qualified paths
            mask_filename = os.path.join(maskFolderPath, (filename_without_ext+'.mask.bmp'))
            poly_filename = os.path.join(maskFolderPath, (filename_without_ext+'.poly.csv'))

            bSave = True
            # Check for the existence of the files. If they exist, display overwrite option dialog box
            if os.path.isfile(mask_filename) or os.path.isfile(mask_filename):
                strMessage = 'The mask and/or polygon file exist. Overwrite files?'
                msgBox = GRIMe_QMessageBox('Save Mask Files', strMessage, QMessageBox.Yes | QMessageBox.No)
                response = msgBox.displayMsgBox()

                if response == QMessageBox.No:
                    bSave = False

            if bSave:
                # Write the mask to a file
                cv2.imwrite(mask_filename, mask)

                csvFile = open(poly_filename, 'w', newline='')

                # Write the polygon(s) vertices to a file
                for polygon in polygonList:
                    csvFile.write('mask\n')
                    csvFile.write('x, y\n')

                    for myPoints in polygon:
                        x = (int)(myPoints.x() * widthMultiplier)
                        y = (int)(myPoints.y() * heightMultiplier)
                        outputString = "{0}, {1}\n".format(x, y)
                        csvFile.write(outputString)

                # WRITE ALL POLYGONS BEFORE CLOSING FILE
                csvFile.close()

                # EXTRACT DOMINANT RGB COLORS
                myGRIMe_Color = GRIMe_Color()

                _, _, hist = myGRIMe_Color.KMeans(masked, 6)

                # EXTRACT DOMINANT HSV COLORS
                hist, colorClusters = myGRIMe_Color.extractDominant_HSV(masked, 6)

                # CREATE COLOR BAR TO DISPLAY CLUSTER COLORS
                colorBar = self.createColorBar(hist, colorClusters[0:5])

                # CONVERT colorBar TO A QImage FOR USE IN DISPLAYING IN QT GUI
                qImg = QImage(colorBar.data, colorBar.shape[1], colorBar.shape[0], QImage.Format_BGR888)

    def changePolygonColor(self, polygonColor):
        self.labelOriginalImage.setBrushColor(polygonColor)
        self.labelOriginalImage.drawPolygon()

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def deleteAllROI(self):
        del self.roiList[:]
        self.tableWidget_ROIList.clearContents()
        self.tableWidget_ROIList.setRowCount(0)

        processLocalImage(self)

        #JES self.spinBoxColorClusters.setDisabled(False)

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def onMyToolBarSettings(self):
        pass

    # --------------------------------------------------
    # --------------------------------------------------
    def __init__(self, parent=None, win=None, session=None):
        super(MainWindow, self).__init__(parent)
        self.mainwin = win
        self.session = session
        self.ui = Ui_MainWindow()

        # --------------------------------------------------------------------------------------------------------------
        # MainWindow stuff: create toolbars, setup UI, set window titles and so forth and so on
        # --------------------------------------------------------------------------------------------------------------
        self.setWindowTitle("GRIMe-AI: John E. Stranzl Jr.")

        self.setupUi(self)

        self.createToolBar()

        # --------------------------------------------------------------------------------------------------------------
        # create and/or initialize variables
        # --------------------------------------------------------------------------------------------------------------
        global dailyImagesList
        dailyImagesList = dailyList([], [])

        if full == 1:
            self.initUSGS_Table()

        self.initProductTable()
        if full == 1 or full == 4:
            self.initROITable()

        self.listboxNEONSites.itemDoubleClicked.connect(self.itemClicked)
        self.listboxNEONSites.itemClicked.connect(self.itemClicked)
        self.listboxNEONSites.currentItemChanged.connect(self.itemClicked)
        self.listboxSiteProducts.itemClicked.connect(self.productClicked)


        #JES self.pushButtonRetrieveData.clicked.connect(self.pushButton_RetrieveNEONDataClicked)

        self.labelLatestImage.mouseDoubleClickEvent = labelMouseDoubleClickEvent
        self.labelLatestImage.installEventFilter(self)

        #JES self.pushButtonBrowseSaveImagesOutputFolder.clicked.connect(self.pushButtonBrowseSaveImagesOutputFolderClicked)

        #        self.splitter.splitterMoved.connect(self.splitterMoved)
        # ****************************************************************************************************

        # SCALED BACK FUNCTIONALITY
        # ****************************************************************************************************
        if full == 1 or full == 4:
            #self.labelOriginalImage.mouseDoubleClickEvent = labelOriginalImageDoubleClickEvent

            self.labelEdgeImage.installEventFilter(self)
            self.labelOriginalImage.installEventFilter(self)

            #JES self.pushButton_BuildFeatureFile.clicked.connect(self.GRIMe_NSF_Build)

            self.pushButton_BuildFeatureFile.clicked.connect(self.GRIMe_NSF_Build)

            #
            #JES if full == 1:
            #JES     self.radioButton_ROIShapeRectangle.clicked.connect(self.ROIShapeClicked)
            #JES     self.radioButton_ROIShapeEllipse.clicked.connect(self.ROIShapeClicked)

            #
            #JES self.pushButton_ColorSegmentation.clicked.connect(self.pushButtonColorSegmentationClicked)

            #
            if full == 1:
                self.pushButton_USGSDownload.clicked.connect(self.pushButton_USGSDownloadClicked)

            # THE DESIGN NOW HAS A TAB FOR EACH SPECIFIC WEBSITE SO A DROP DOWN IS NO LONGER USED ON THE 'NEON' TAB
            #ImageWebSiteList = []
            #ImageWebSiteList.append("NEON")
            #self.listWidgetImageWebSites.addItems(ImageWebSiteList)

            # INITIALIZE WIDGETS
            maxRows = self.tableWidget_ROIList.rowCount()
            nCol = 0
            for i in range(0, maxRows):
                self.tableWidget_ROIList.removeRow(0)

            #JES self.pushButton_TrainGood.clicked.connect(self.pushButtonTrainGoodTriggered)
            #JES self.pushButton_TrainBad.clicked.connect(self.pushButtonTrainBadTriggered)

        # SAVE AND RECALL SETTINGS
        self.actionSave_Settings.triggered.connect(self.saveSettings)

        #
        self.tabWidget.currentChanged.connect(self.tabWidgetChanged)

        # GRAPH TAB(S)

        self.labelLatestImage.setScaledContents(True)
        # self.ui.labelLatestImage.setScaledContents(True)
        # self.ui.labelOriginalImage.setScaledContents(True)
        # self.ui.labelEdgeImage.setScaledContents(True)

    # --------------------------------------------------
    # --------------------------------------------------
    # THESE EVENT FILTERS WILL BE USED TO TRACK MOUSE MOVEMENT AND MOUSE BUTTON CLICKS FOR DISPLAYING ADDITIONAL
    # INFORMATION, VIEWS, POP-UP MENUS AND DRAWING REGIONS-OF-INTEREST (ROI) AROUND SPECIFIC AREAS OF AN IMAGE.
    # ------------------------------------------------------------------------------------------------------------------
    def eventFilter(self, source, event):

        if full == 1 or full == 4:
            if event.type() == QtCore.QEvent.MouseMove and source is self.labelEdgeImage:
                # print("A")
                pass

            if event.type() == QtCore.QEvent.MouseMove and source is self.labelOriginalImage:
                if 0:
                    x, y = pyautogui.position()
                    pixelColor = pyautogui.screenshot().getpixel((x, y))
                    ss = 'Screen Pos - X:' + str(x).rjust(4) + ' Y:' + str(y).rjust(4)
                    ss += ' RGB: (' + str(pixelColor[0]).rjust(3)
                    ss += ', ' + str(pixelColor[1]).rjust(3)
                    ss += ', ' + str(pixelColor[2]).rjust(3) + ')'
                    print(ss)
                    # print("B")
                pass

        #if full == 1 or full == 4:
        #    if event.type() == QtCore.QEvent.MouseButtonDblClick and source is self.labelOriginalImage:
        #        # labelEdgeImageDoubleClickEvent(self)
        #        # labelMouseDoubleClickEvent(self)
        #        labelOriginalImageDoubleClickEvent(self)

        return super(MainWindow, self).eventFilter(source, event)


    def closeEvent(self, event):
        # DESTROY ANY MODELESS DIALOG BOXES THAT ARE OPEN
        if self.edgeDetectionDlg != None:
            self.edgeDetectionDlg.close()

        if self.colorSegmentationDlg != None:
            self.colorSegmentationDlg.close()

        if self.TriageDlg != None:
            self.TriageDlg.close()

        if self.fileFolderDlg != None:
            self.fileFolderDlg.close()

        if self.maskEditorDlg != None:
            self.maskEditorDlg.close()

        if self.imageNavigationDlg != None:
            self.imageNavigationDlg.close()

        if self.releaseNotesDlg != None:
            self.releaseNotesDlg.close()

        webdriver.Chrome.quit()

        QMainWindow.closeEvent(self, event)

    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def fetchImageList(self, imageFolder, bRecursive):
        global imageFileFolder
        imageFileFolder = imageFolder
        fetchLocalImageList(self, imageFileFolder, bRecursive, False)  # start_date, end_date, start_time, end_time)

        try:
            global gFrameCount
            self.imageNavigationDlg.setImageCount(gFrameCount)
            self.imageNavigationDlg.reset()
        except:
            pass


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
def fetchLocalImageList(self, filePath, bFetchRecursive, bCreateEXIFFile, start_date=datetime.date(1970, 1, 1),
                        end_date=datetime.date(2099, 12, 31), start_time='000000', end_time='000000'):
    global gWebImageCount
    global dailyImagesList

    # CLEAR THE PREVIOUSLY DOWNLOADED IMAGE LIST, IF ANY
    dailyImagesList.clear()

    # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # INITIALIZE VARIABLES TO A KNOWN VALUE
    nStartDate = 99999999
    nEndDate = -1
    strStartDate = ""
    strEndDate = ""
    List = []

    # THE SOFTWARE IS NOW DESIGNED TO REQUIRE THE IMAGES TO BE DOWNLOADED FIRST FOR A VARIETY OF REASONS
    bSaveImages = True
    imageOutputFolder = self.EditSaveImagesOutputFolder.text()

    # if self.checkBoxSaveImages.isChecked():
    #    bSaveImages = True
    #    imageOutputFolder = self.EditSaveImagesOutputFolder.text()

    # create the csv writer
    if bCreateEXIFFile:
        # open the file in the write mode
        EXIFFolder = self.EditEXIFOutputFolder.text()
        csvFile = open(EXIFFolder + '/' + 'EXIFData.csv', 'w', newline='')

        writer = csv.writer(csvFile)

        bWriteHeader = True

    # count the number of images that will potentially be processed and possibly saved with the specified extension
    # to display an "hourglass" to give an indication as to how long the process will take. Furthermore, the number
    # of images will help determine whether or not there is enough disk space to accomodate storing the images.
    imageCount = GRIMe_Utils.getImageCount(filePath, extensions)

    progressBar = QProgressWheel()
    progressBar.setRange(0, imageCount + 1)
    progressBar.show()

    # RECURSE AND TRAVERSE FROM THE SPECIFIED FOLDER DOWN TO DETERMINE THE DATE RANGE FOR THE IMAGES FOUND
    imageIndex = 0

    files = GRIMe_Utils.getFileList(filePath, extensions, bFetchRecursive)

    # traverse all files in folder that meet the criteria for retrieval
    # 1. does the file have the specified file extension
    # 2. extract the date from the filename
    # 3. does the file meet the date criteria? if "yes," then continue; if "no" check next file in the list
    # 4. if the files meets the date criteria, extract the EXIF data from the file to ascertain the time at
    #    which the image was acquired
    # 5. if the file meets the time range criteria, add the file to a list of images to be used for the session
    #    and also add the file's EXIF data to a CSV EXIF log file if the option is selected by the user. Last but
    #    not least, if the user selects the option to copy the image to a separate folder, then copy the file to
    #    the folder specified by the user
    for file in files:
        progressBar.setWindowTitle(file)
        progressBar.setValue(imageIndex)
        progressBar.repaint()
        imageIndex += 1

        ext = os.path.splitext(file)[-1].lower()

        if ext in extensions:
            fileDate, fileTime = GRIMe_Utils.extractDateFromFilename(file)

            if fileDate >= start_date and fileDate <= end_date:
                fullPathAndFilename = file

                try:
                    # extract EXIF info to determine what time the image was acquired. If EXIF info is not found,
                    # throw an exception and see if the information is embedded in the filename. Currently, we are
                    # working with images from NEON and PBT. The PBT images have EXIF data and the NEON/PhenoCam
                    # do not appear to have EXIF data.
                    myEXIFData = EXIFData()
                    myEXIFData.extractEXIFData(fullPathAndFilename)

                    strTemp = str(myEXIFData.getEXIF()[8])
                    timeOriginal = re.search(' \d{2}:\d{2}:\d{2}', strTemp).group(0)

                    nHours = int(str(timeOriginal[1:3]))
                    nMins = int(str(timeOriginal[4:6]))
                    nSecs = int(str(timeOriginal[7:9]))

                    bEXIFDataFound = True
                except:
                    # assume the filename contains the timestamp for the image (assumes the image file is a PBT image)
                    bEXIFDataFound = False

                    try:
                        nHours = int(str(strTime[0:2]))
                        nMins = int(str(strTime[2:4]))
                        nSecs = int(str(strTime[4:6]))
                    except:
                        nHours = 0
                        nMins = 0
                        nSecs = 0

                image_time = datetime.time(nHours, nMins, nSecs)

                # if ((start_time == datetime.time(0, 0, 0)) and (end_time == datetime.time(0, 0, 0))) or \
                #        ((image_time >= start_time) and (image_time <= end_time)):

                # WRITE THE HEADER ONLY ONCE WHEN THE FIRST FILE IS PROCESSED
                if bCreateEXIFFile and bEXIFDataFound:
                    if bWriteHeader:
                        writer.writerow(myEXIFData.getHeader())
                        bWriteHeader = False
                    else:
                        writer.writerow(myEXIFData.getEXIF())

                List.append(imageData(fullPathAndFilename, 0, 0, 0))

                if bSaveImages:
                    shutil.copy(fullPathAndFilename, imageOutputFolder)

                # delete EXIFData object
                del myEXIFData

    dailyImagesList.setVisibleList(List)

    global gFrameCount
    gFrameCount = len(dailyImagesList.getVisibleList())
    gWebImageCount = len(dailyImagesList.getVisibleList())

    # INIT SPINBOX CONTROLS BASED UPON NUMBER OF IMAGES AVAILABLE
    # dailyURLvisible = []

    # clean-up before exiting function
    # 1. close and delete the progress bar
    # 2. close the EXIF log file, if opened
    progressBar.close()
    del progressBar

    if bCreateEXIFFile:
        csvFile.close()

    if len(files) > 0:
        processLocalImage(self)
        #refreshImage(self)


# ======================================================================================================================
#
# ======================================================================================================================
def processLocalImage(self, nImageIndex=0):
    global currentImage
    global imageFileFolder


    myGRIMe_Color = GRIMe_Color()

    if os.path.exists(imageFileFolder):
        # videoFilePath = Path(frameFolder)
        ## JES videoFileList = [str(pp) for pp in videoFilePath.glob("**/*.jpg")]
        # videoFileList = [str(pp) for pp in videoFilePath.glob("*.jpg")]

        global dailyImagesList
        videoFileList = dailyImagesList.getVisibleList()

        if len(videoFileList) > 0:
            if nImageIndex > gFrameCount:
                nImageIndex = gFrameCount

            inputFrame = videoFileList[nImageIndex - 1].fullPathAndFilename  # zero based index

            if os.path.isfile(inputFrame):
                global currentImageFilename
                currentImageFilename = inputFrame
                numpyImage = myGRIMe_Color.loadColorImage(inputFrame)

                tempCurrentImage = QImage(numpyImage, numpyImage.shape[1], numpyImage.shape[0], QImage.Format_RGB888)
                currentImage = QPixmap(tempCurrentImage)

    # ----------------------------------------------------------------------------------------------------
    # DISPLAY IMAGE FROM NEON SITE
    # ----------------------------------------------------------------------------------------------------
    if currentImage:

        # CALCULATE THE GREENNESS INDEX (PHENOCAM GCC) FOR THE ENTIRE IMAGE
        if currentImage != []:
            numpyImg = GRIMe_Utils.convertQImageToMat(currentImage.toImage())
            #JES self.label_GreennessIndex_Value.setText(str(computeGreennessIndex(numpyImg)))

        scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        width = scaledCurrentImage.width()
        height = scaledCurrentImage.height()

        currentImageRescaled = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        if g_displayOptions.displayROIs:
            painter = QPainter(currentImageRescaled)

        img = GRIMe_Utils.convertQImageToMat(currentImage.toImage())

        if g_regionSelect.wholeImage:
            # BLUR THE IMAGE
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            red, green, blue = separateChannels(img)
            redSum, greenSum, blueSum = sumChannels(red, green, blue)
            strGCC = '%3.3f' % (computeGreennessIndex(redSum, greenSum, blueSum))
            self.label_GreennessIndex_Value.setText(strGCC)

            # IMAGE INTENSITY CALCULATIONS
            intensity = cv2.mean(gray)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway
            strIntensity = '%3.3f' % (intensity)
            self.label_ImageIntensityValue.setText(strIntensity)

            # COMPUTE ENTROPY FOR ENTIRE IMAGE
            entropyValue = self.calcEntropy(gray)
            strEntropy = '%3.3f' % (entropyValue)
            self.label_ImageEntropyValue.setText(strEntropy)
        else:
            self.label_GreennessIndex_Value.setText("")
            self.label_ImageIntensityValue.setText("")
            self.label_ImageEntropyValue.setText("")

        progressBar = QProgressWheel()
        progressBar.setRange(0, len(self.roiList) + 1)
        #JES progressBar.show()

        nRow = 0
        for roiObj in self.roiList:
            progressBar.setValue(nRow+1)
            progressBar.repaint()

            try:
                #JES if self.checkBoxCalcColors.isChecked():
                if (1):
                    # EXTRACT ROI FOR WHICH COLOR CLUSTERING IS TO BE PERFORMED
                    rgb = extractROI(roiObj.getImageROI(), img)

                    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

                    # CALCULATE THE GREENNESS INDEX FOR THE ROI
                    red, green, blue = separateChannels(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                    redSum, greenSum, blueSum = sumChannels(red, green, blue)
                    strGreennessIndex = "{:.4f}".format(computeGreennessIndex(redSum, greenSum, blueSum))
                    self.greennessLabel = QtWidgets.QLabel()
                    self.greennessLabel.setText(strGreennessIndex)
                    self.tableWidget_ROIList.setCellWidget(nRow, 3, self.greennessLabel)

                    try:
                        # CALCULATE THE INTENSITY FOR THE ROI
                        strIntensity = "{:.4f}".format(cv2.mean(gray)[0])  # The range for a pixel's value in grayscale is (0-255), 127 lies midway
                        self.intensityLabel = QtWidgets.QLabel()
                        self.intensityLabel.setText(strIntensity)
                        self.tableWidget_ROIList.setCellWidget(nRow, 4, self.intensityLabel)
                    except:
                        pass

                    try:
                        # COMPUTE ENTROPY FOR ENTIRE IMAGE
                        strEntropyValue = "{:.4f}".format(self.calcEntropy(gray)[0])
                        self.entropyLabel = QtWidgets.QLabel()
                        self.entropyLabel.setText(strEntropyValue)
                        self.tableWidget_ROIList.setCellWidget(nRow, 5, self.entropyLabel)
                    except:
                        pass

                    # EXTRACT DOMINANT RGB COLORS
                    _, _, hist = myGRIMe_Color.KMeans(rgb, roiObj.getNumColorClusters())

                    # EXTRACT DOMINANT HSV COLORS
                    hist, colorClusters = myGRIMe_Color.extractDominant_HSV(rgb, roiObj.getNumColorClusters())

                    # CREATE COLOR BAR TO DISPLAY CLUSTER COLORS
                    colorBar = self.createColorBar(hist, colorClusters)

                    # CONVERT colorBar TO A QImage FOR USE IN DISPLAYING IN QT GUI
                    qImg = QImage(colorBar.data, colorBar.shape[1], colorBar.shape[0], QImage.Format_BGR888)

                    # INSERT THE DOMINANT COLORS INTO A QLabel IN ORDER TO ADD IT TO THE FEATURE TABLE
                    self.label = QtWidgets.QLabel()
                    self.label.setPixmap(QPixmap(qImg.scaled(100, 50)))
                    self.tableWidget_ROIList.setCellWidget(nRow, 2, self.label)

                    self.tableWidget_ROIList.resizeColumnsToContents()

                    # INSERT COLOR INTO THE CONTROL ON THE COLOR SEGMENTATION TAB
                    #JES self.labelDominantColors.setPixmap(QPixmap(qImg).scaled(self.labelEdgeImage.size()))

                nRow = nRow + 1
            except:
                nErrorCode = -1

            # OVERLAY ROI BOUNDARY ON IMAGE
            #JESif self.checkBoxDisplayROIs.isChecked():
            if (1):
                pen = QPen(QtCore.Qt.red, 1, QtCore.Qt.SolidLine)
                painter.setPen(pen)

                if roiObj.getROIShape() == ROIShape.RECTANGLE:
                    painter.drawRect(roiObj.getDisplayROI())
                elif roiObj.getROIShape() == ROIShape.ELLIPSE:
                    painter.drawEllipse(roiObj.getDisplayROI())

                font = painter.font()
                font.setPointSize(8)
                painter.setPen(QPen(QtCore.Qt.red, 1, QtCore.Qt.SolidLine))
                painter.setFont(font)
                painter.drawText(roiObj.getDisplayROI().x(), roiObj.getDisplayROI().y() - 16, 50, 16, QtCore.Qt.AlignLeft, roiObj.getROIName())

        if g_displayOptions.displayROIs:
            painter.end()

        # close and delete the progress bar
        progressBar.close()
        del progressBar

        self.labelOriginalImage.setPixmap(currentImageRescaled)

        pix = processImage(self, currentImage)
        gray = cv2.cvtColor(numpyImage, cv2.COLOR_BGR2GRAY)

        '''
        entropy_image = entropy(gray, disk(7))
        npa = np.asarray(entropy_image, dtype=np.float64) * 255
        npa = npa.astype(np.uint8)
        colorImg = cv2.applyColorMap(npa, cv2.COLORMAP_MAGMA)
        qImg = QImage(colorImg.data, colorImg.shape[1], colorImg.shape[0], QImage.Format_RGB888)
        pix = QPixmap(qImg)

        self.labelEdgeImage.setPixmap(pix.scaled(self.labelEdgeImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        '''

        del currentImageRescaled
        del scaledCurrentImage

    # CALL PROCESSEVENTS IN ORDER TO UPDATE GUI
    QCoreApplication.processEvents()


# ======================================================================================================================
# THIS FUNCTION WILL PROCESS THE CURRENT IMAGE BASED UPON THE SETTINGS SELECTED BY THE END-USER.
# THE IMAGE STORAGE TYPE IS QImage
# ======================================================================================================================
def processImage(self, myImage):
    global g_edgeMethodSettings
    global g_featureMethodSettings

    pix = []

    if not myImage == []:
        # CONVERT IMAGE FROM QImage FORMAT TO Mat FORMAT (BYTE ORDER IS R, G, B)
        img1 = GRIMe_Utils.convertQImageToMat(myImage.toImage())

        #JES if self.checkboxKMeans.isChecked():
        #    myGRIMe_Color = GRIMe_Color()
        #    qImg, clusterCenters, hist = myGRIMe_Color.KMeans(img1, self.spinBoxColorClusters.value())

        # CONVERT COLOR IMAGE TO GRAY SCALE
        gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        # REMOVE NOISE FROM THE IMAGE
        if len(gray) != 0:
            grayEQ = cv2.equalizeHist(gray)
            grayBlur = cv2.GaussianBlur(grayEQ, (15, 15), 0)
            cv2.erode(grayBlur, (7,7), gray)

        # EDGE DETECTION METHODS
        if len(gray) != 0:

            myProcessImage = GRIMe_ProcessImage()

            if g_edgeMethodSettings.method == edgeMethodsClass.CANNY:
                pix = myProcessImage.processCanny(img1, gray)

            elif g_edgeMethodSettings.method == edgeMethodsClass.LAPLACIAN:
                pix = myProcessImage.processLaplacian(img1)

            elif g_edgeMethodSettings.method == edgeMethodsClass.SOBEL_X or g_edgeMethodSettings.method == edgeMethodsClass.SOBEL_Y or g_edgeMethodSettings.method == edgeMethodsClass.SOBEL_XY:
                pix = myProcessImage.processSobel(img1, g_edgeMethodSettings.method)

            elif g_featureMethodSettings.method == featureMethodsClass.SIFT:
                pix = myProcessImage.processSIFT(img1, gray)

            elif g_featureMethodSettings.method == featureMethodsClass.ORB:
                pix = myProcessImage.processORB(img1, gray)

    return pix


# ======================================================================================================================
#
# ======================================================================================================================
def toolbarButtonImageTriage(checkBox_FetchRecursive):
    strMessage = 'You are about to perform Image Triage. Would you like to continue?'
    msgBox = GRIMe_QMessageBox('Download Image Files', strMessage, QMessageBox.Yes | QMessageBox.No)
    response = msgBox.displayMsgBox()

    if response == QMessageBox.Yes:
        prompter = promptlib.Files()
        folder = prompter.dir()

        if len(folder) == 0:
            strMessage = 'ERROR! Please specify an image folder containing images to triage.'
            msgBox = GRIMe_QMessageBox('Image Triage', strMessage)
            response = msgBox.displayMsgBox()
        else:
#            global TriageDlg
            TriageDlg = GRIME_TriageOptionsDlg()

            response = TriageDlg.exec_()

            if response == 1:
                cleanImages(folder, \
                            True, \
                            TriageDlg.getBlurThreshold(), TriageDlg.getShiftSize(), \
                            TriageDlg.getBrightnessMin(), TriageDlg.getBrightnessMax(), \
                            TriageDlg.getCreateReport(), TriageDlg.getMoveImages())

                strMessage = 'Image triage is complete.'
                msgBox = GRIMe_QMessageBox('Image Triage', strMessage)
                response = msgBox.displayMsgBox()
            else:
                strMessage = 'ABORT! You cancelled the triage operation.'
                msgBox = GRIMe_QMessageBox('Image Triage', strMessage)
                response = msgBox.displayMsgBox()
    else:
        strMessage = 'ABORT! You cancelled the triage operation.'
        msgBox = GRIMe_QMessageBox('Image Triage', strMessage)
        response = msgBox.displayMsgBox()

# ======================================================================================================================
#
# ======================================================================================================================
def resizeImage(image, scale_percent):
    # --------------------------------------------------------------------------------
    # reshape the image to be a list of pixels
    # --------------------------------------------------------------------------------
    if scale_percent == 100.0:
        return image
    else:
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        dim = (width, height)

        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        return resized


# ======================================================================================================================
#
# ======================================================================================================================
def computeBlurAndBrightness(shiftSize):
    global currentImage

    img1 = GRIMe_Utils.convertQImageToMat(currentImage.toImage())
    grayImage = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    # DECIMATE IMAGE
    grayImage = resizeImage(grayImage, 50.0)

    hist = cv2.calcHist([grayImage], [0], None, [256], [0, 256])

    ''' BLUR DETECTION CALCULATIONS'''
    # grab the dimensions of the image and use the dimensions to derive the center (x, y)-coordinates
    (h, w) = grayImage.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    fft = np.fft.fft2(grayImage)
    fftShift = np.fft.fftshift(fft)

    # compute the magnitude spectrum of the transform
    magnitude = 20 * np.log(np.abs(fftShift))

    # zero-out the center of the FFT shift (i.e., remove low frequencies),
    # apply the inverse shift such that the DC component once again becomes the top-left,
    # and then apply the inverse FFT
    fftShift[cY - shiftSize:cY + shiftSize, cX - shiftSize:cX + shiftSize] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # IMAGE INTENSITY CALCULATIONS
    # blur = cv2.blur(grayImage, (5, 5))  # With kernel size depending upon image size
    blur = cv2.GaussianBlur(grayImage, (0, 0), 1) if 0. < 1 else grayImage
    intensity = cv2.mean(blur)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway

    # FREE UP MEMORY FOR THE NEXT IMAGE TO BE PROCESSSED
    del fftShift
    del fft
    del recon
    del blur

    return mean, intensity


# ======================================================================================================================
#
# ======================================================================================================================
def cleanImages(folder, bFetchRecursive, blurThreshhold, shiftSize, brightnessMin, brightnessMAX, bCreateReport, bMoveImages):
    extensions = ('.jpg', '.jpeg', '.png')

    myGRIMe_Color = GRIMe_Color()

    if bCreateReport:
        csvFilename = 'ImageTriage_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
        imageQualityFile = os.path.join(folder, csvFilename)
        csvFile = open(imageQualityFile, 'a', newline='')
        csvFile.write('Focus Value, Focus Attrib, Intensity Value, Intensity Attrib., Filename\n')

    # count the number of images that will potentially be processed and possibly saved with the specified extension
    # to display an "hourglass" to give an indication as to how long the process will take. Furthermore, the number
    # of images will help determine whether or not there is enough disk space to accomodate storing the images.
    imageCount = GRIMe_Utils.getImageCount(folder, extensions)

    progressBar = QProgressWheel(0, imageCount + 1)
    progressBar.show()

    imageIndex = 0

    # process images to determine which ones are too dark/too light, blurry/clear, etc and move them into a subfolder
    # created so they are not processed with nominal images.
    files = GRIMe_Utils.getFileList(folder, extensions, bFetchRecursive)

    for file in files:
        progressBar.setWindowTitle(file)
        progressBar.setValue(imageIndex)
        progressBar.repaint()
        imageIndex += 1

        ext = os.path.splitext(file)[-1].lower()

        if ext in extensions:
            filename = os.path.join(folder, file)
            numpyImage = myGRIMe_Color.loadColorImage(filename)
            grayImage = cv2.cvtColor(numpyImage, cv2.COLOR_RGB2GRAY)

            # DECIMATE IMAGE TO SPEED UP PROCESSING
            grayImage = resizeImage(grayImage, 50.0)

            hist = cv2.calcHist([grayImage], [0], None, [256], [0, 256])

            ''' BLUR DETECTION CALCULATIONS'''
            # grab the dimensions of the image and use the dimensions to derive the center (x, y)-coordinates
            (h, w) = grayImage.shape
            (cX, cY) = (int(w / 2.0), int(h / 2.0))

            fft = np.fft.fft2(grayImage)
            fftShift = np.fft.fftshift(fft)

            # compute the magnitude spectrum of the transform
            magnitude = 20 * np.log(np.abs(fftShift))

            # zero-out the center of the FFT shift (i.e., remove low frequencies),
            # apply the inverse shift such that the DC component once again becomes the top-left,
            # and then apply the inverse FFT
            fftShift[cY - shiftSize:cY + shiftSize, cX - shiftSize:cX + shiftSize] = 0
            fftShift = np.fft.ifftshift(fftShift)
            recon = np.fft.ifft2(fftShift)

            # compute the magnitude spectrum of the reconstructed image,
            # then compute the mean of the magnitude values
            magnitude = 20 * np.log(np.abs(recon))
            mean = np.mean(magnitude)

            # IMAGE INTENSITY CALCULATIONS
            # blur = cv2.blur(grayImage, (5, 5))  # With kernel size depending upon image size
            blur = cv2.GaussianBlur(grayImage, (0, 0), 1) if 0. < 1 else grayImage
            intensity = cv2.mean(blur)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway

            # DECISION LOGIC
            bMove = False
            strFFTFocusMetric = 'Nominal'
            strFocusMetric = 'N/A'
            strIntensity = 'Nominal'

            # CHECK MEAN AGAINST THRESHHOLD TO DETERMINE IF THE IMAGE IS BLURRY/FOGGY/OUT-OF-FOCUS/ETC.
            if mean <= blurThreshhold:
                strFFTFocusMetric = "Blurry"
                bMove = True

            # CHECK TO SEE IF THE OVERALL IMAGE IS TOO DARK OR TOO BRIGHT
            if intensity < brightnessMin:
                strIntensity = "Too Dark"
                bMove = True
            elif intensity > brightnessMAX:
                strIntensity = "Too Light"
                bMove = True

            # MOVE THE IMAGE REJECTS TO A SUBFOLDER IF THE USER CHOOSE THIS OPTION
            if bMoveImages and bMove:
                # create a subfolder beneath the current root folder if the option to move less than nominal images is selected
                filename = os.path.basename(file)
                filepath = os.path.dirname(file)
                tempFolder = os.path.join(filepath, "MovedImages")
                if not os.path.exists(tempFolder):
                    os.mkdir(tempFolder)

                shutil.move(file, tempFolder)

                filename = os.path.join(tempFolder, filename)

            # CREATE A CSV FILE THAT CONTAINS THE FOCUS AND INTENSITY METRICS ALONG WITH HYPERLINKS TO THE IMAGES
            if bCreateReport:
                strHyperlink = '=HYPERLINK(' + '"' + filename + '"' + ')'
                strOutputString = '%3.2f,%s,%3.2f,%s,%s\n' % (mean, strFFTFocusMetric, intensity, strIntensity, strHyperlink)
                csvFile.write(strOutputString)

            # FREE UP MEMORY FOR THE NEXT IMAGE TO BE PROCESSSED
            del fftShift
            del fft
            del recon
            del blur

    # clean-up before exiting function
    # 1. close and delete the progress bar
    # 2. close the EXIF log file, if opened
    if bCreateReport:
        csvFile.close()
    progressBar.close()
    del progressBar


# ======================================================================================================================
#
# ======================================================================================================================
def updateProductTableDateRange(productTable, nRow, startDate, endDate):

    productTable.setItem(0, 1, QTableWidgetItem('Image Source: Hard Drive'))

    # CONFIGURE DATES FOR SPECIFIC PRODUCT
    for m in [2, 4]:
        date_widget = QtWidgets.QDateEdit(QtCore.QDate(startDate.year, startDate.month, startDate.day))
        productTable.setCellWidget(nRow, m, date_widget)

        productTable.resizeColumnToContents(m)
        if m == 2:
            date_widget.setDisabled(True)
        else:
            date_widget.setDisabled(False)

    for m in [3, 5]:
        date_widget = QtWidgets.QDateEdit(QtCore.QDate(endDate.year, endDate.month, endDate.day))
        if m == 3:
            date_widget.setDisabled(True)
        else:
            date_widget.setDisabled(False)

        productTable.setCellWidget(nRow, m, date_widget)

        productTable.resizeColumnToContents(m)

    productTable.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)


# ======================================================================================================================
#
# ======================================================================================================================
def MSTGetFrameCount(self, frameFolder):
    if os.path.exists(frameFolder):
        frameFolderPath = Path(frameFolder)
        # JES videoFileList = [str(pp) for pp in frameFolderPath.g("**/*.jpg")]
        videoFileList = [str(pp) for pp in frameFolderPath.glob("*.jpg")]
        nFrameCount = len(videoFileList)
    else:
        nFrameCount = 0

    return nFrameCount


# ======================================================================================================================
#
# ======================================================================================================================
def compute_average_image_color(img):
    width, height = img.size

    r_total = 0
    g_total = 0
    b_total = 0

    count = 0
    for x in range(0, width):
        for y in range(0, height):
            r, g, b = img.getpixel((x, y))
            r_total += r
            g_total += g
            b_total += b
            count += 1

    return (r_total / count, g_total / count, b_total / count)


# ======================================================================================================================
#
# ======================================================================================================================
def checkColorMatch(painter, clusterCenters, hist, roiList):
    global currentImage

    refCluster, refhist = roiList[0].getClusterCenters()
    refLab = rgb2lab(refCluster[0])

    if len(roiList) > 0:
        for referenceROI in roiList:
            lab0 = rgb2lab(clusterCenters[0])

            pen = QPen(QtCore.Qt.red, 1, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            font = painter.font()
            font.setPointSize(font.pointSize())
            painter.setFont(font)
            if lab0[0] > refLab[0]:
                label = referenceROI.getROIName()[2:len(referenceROI.getROIName())]
            else:
                label = referenceROI.getROIName()

            # delta_e_76 = delta_e_cie1976(refLab, lab0)

            # DRAW TEXT WITH L*a*b* VALUES FOR THE REFERENCE IMAGE
            #painter.drawText(QtCore.QRect(1, 1, 1200, 200), QtCore.Qt.AlignLeft, "Ref. Values")
            #painter.drawText(QtCore.QRect(1800, 300, 1200, 200), QtCore.Qt.AlignLeft, 'L: ' + refLab[0].__str__())
            #painter.drawText(QtCore.QRect(1800, 500, 1200, 200), QtCore.Qt.AlignLeft, 'a*: ' + refLab[1].__str__())
            #painter.drawText(QtCore.QRect(1800, 700, 1200, 200), QtCore.Qt.AlignLeft, 'b*: ' + refLab[2].__str__())

            # DRAW TEXT WITH L*a*b* VALUE FOR THE CURRENT IMAGE
            #painter.drawText(QtCore.QRect(referenceROI.getDisplayROI().x, referenceROI.getDisplayROI().y, 1200, 200), QtCore.Qt.AlignLeft, label)
            #painter.drawText(QtCore.QRect(referenceROI.getDisplayROI().x, referenceROI.getDisplayROI().y+200, 1200, 200), QtCore.Qt.AlignLeft, 'L: ' + lab0[0].__str__())
            #painter.drawText(QtCore.QRect(referenceROI.getDisplayROI().x, referenceROI.getDisplayROI().y+400, 1200, 200), QtCore.Qt.AlignLeft, 'a*: ' + lab0[1].__str__())
            #painter.drawText(QtCore.QRect(referenceROI.getDisplayROI().x, referenceROI.getDisplayROI().y+600, 1200, 200), QtCore.Qt.AlignLeft, 'b*: ' + lab0[2].__str__())

            # painter.end()
    else:
        lab0 = rgb2lab(clusterCenters[0])

        # painter = QPainter(currentImage)
        pen = QPen(QtCore.Qt.red, 2, QtCore.Qt.SolidLine)
        painter.setPen(pen)
        font = painter.font()
        font.setPointSize(font.pointSize() * 15)
        painter.setFont(font)
        painter.drawText(QtCore.QRect(100, 300, 1200, 200), QtCore.Qt.AlignLeft, 'L: ' + lab0[0].__str__())
        painter.drawText(QtCore.QRect(100, 600, 1200, 200), QtCore.Qt.AlignLeft, 'a*: ' + lab0[1].__str__())
        painter.drawText(QtCore.QRect(100, 600, 1200, 200), QtCore.Qt.AlignLeft, 'a*: ' + lab0[1].__str__())
        painter.drawText(QtCore.QRect(100, 900, 1200, 200), QtCore.Qt.AlignLeft, 'b*: ' + lab0[2].__str__())

        painter.end()


# ======================================================================================================================
#
# ======================================================================================================================
def featureMatch():

    # Read the query image as query_img
    # and train image This query image
    # is what you need to find in train image
    # Save it in the same directory
    # with the name image.jpg
    query_img = cv2.imread('F://000 - University of Nebraska//0000 - Hydrology Images//PBT//Oxbow//Temp//A.jpg')
    train_img = cv2.imread('F://000 - University of Nebraska//0000 - Hydrology Images//PBT//Oxbow//Temp//B.jpg')

    # Convert it to grayscale
    query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    #query_img_bw = cv2.GaussianBlur(query_img_bw, (21, 21), 0)
    #train_img_bw = cv2.GaussianBlur(train_img_bw, (21, 21), 0)

    #query_img_bw = cv2.Canny(query_img_bw, 100, 150)
    #train_img_bw = cv2.Canny(train_img_bw, 100, 150)

    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create()
    #orb.setEdgeThreshold(50)
    #orb.setNLevels(10)
    #orb.setPatchSize(30)
    #orb.setMaxFeatures(8000)

    #kp = orb.detect(train_img, None)
    #kp, des = orb.compute(train_img, kp)

    #edges = cv2.drawKeypoints(train_img, kp, None, color=(0, 255, 0), flags=0)

    # Now detect the keypoints and compute the descriptors for the query image and train image
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

    # Initialize the Matcher for matching the keypoints and then match the keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.match(queryDescriptors, trainDescriptors)
    #matches = sorted(matches, key=lambda x:x.distance)

    # draw the matches to the final image containing both the images the drawMatches()
    # function takes both images and keypoints and outputs the matched query image with its train image
    final_img = cv2.drawMatches(query_img_bw, queryKeypoints,
                                train_img_bw, trainKeypoints, matches[:20], None)

    final_img = cv2.resize(final_img, (1500, 1150))

    # Show the final image
    cv2.imshow("Matches", final_img)
    cv2.waitKey(0)


# ======================================================================================================================
#
# ======================================================================================================================
def extractROI(rect, image):
    return(image[rect.y():rect.y() + rect.height(), rect.x():rect.x() + rect.width()])


# ======================================================================================================================
#
# ======================================================================================================================
def separateChannels(image):
    # greennessIndex = green / (red + green + blue)
    red = image[:, :, 0]
    red = red.flatten()
    red = red.astype(float)

    green = image[:, :, 1]
    green = green.flatten()
    green - green.astype(float)

    blue = image[:, :, 2]
    blue = blue.flatten()
    blue = blue.astype(float)

    return red, green, blue


# ======================================================================================================================
#
# ======================================================================================================================
def sumChannels(red, green, blue):

    redSum   = np.sum(red)
    greenSum = np.sum(green)
    blueSum  = np.sum(blue)

    return redSum, greenSum, blueSum

# ======================================================================================================================
#
# ======================================================================================================================
def computeGreennessIndex(redSum, greenSum, blueSum):

    try:
        greennessIndex = greenSum / (redSum + greenSum + blueSum)
    except:
        greennessIndex = -999

    return(greennessIndex)


# ======================================================================================================================
#
# ======================================================================================================================
def computeExcessGreenIndex(redSum, greenSum, blueSum):

    try:
        greennessIndex = (2.0 * greenSum) - redSum - blueSum
    except:
        greennessIndex = -999

    return(greennessIndex)


# ======================================================================================================================
#
# ======================================================================================================================
def computeGreenLeafIndex(redSum, greenSum, blueSum):

    try:
        greennessIndex = ((2.0 * greenSum) - redSum - blueSum)/ ((2.0 * greenSum) + redSum + blueSum)
    except:
        greennessIndex = -999

    return(greennessIndex)


# ======================================================================================================================
#
# ======================================================================================================================
def computeNormalizedDifferenceVegetationIndex(redSum, greenSum):

    try:
        greennessIndex = (greenSum - redSum) / (redSum + greenSum)
    except:
        greennessIndex = -999

    return(greennessIndex)

# ======================================================================================================================
#
# ======================================================================================================================
def computeRedGreenIndex(redSum, greenSum):

    redGreenIndex = redSum / greenSum

    return(redGreenIndex)

# ======================================================================================================================
#
# ======================================================================================================================
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read it into OpenCV format
    resp = urlopen(url)

    image = np.asarray(bytearray(resp.read()), dtype="uint8")

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image


# ======================================================================================================================
#
# ======================================================================================================================
def isColor(r, g, b):
    return (r != g != b)


# ======================================================================================================================
#
# ======================================================================================================================
def rgb2lab(inputColor):
    num = 0
    RGB = [0, 0, 0]

    for value in inputColor:
        value = float(value) / 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        RGB[num] = value * 100
        num = num + 1

    XYZ = [0, 0, 0, ]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)

    XYZ[0] = float(XYZ[0]) / 95.047  # ref_X =  95.047   Observer= 2°, Illuminant= D65
    XYZ[1] = float(XYZ[1]) / 100.0  # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883  # ref_Z = 108.883

    num = 0
    for value in XYZ:

        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)

        XYZ[num] = value
        num = num + 1

    Lab = [0, 0, 0]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    Lab[0] = round(L, 4)
    Lab[1] = round(a, 4)
    Lab[2] = round(b, 4)

    return Lab


# ======================================================================================================================
#
# ======================================================================================================================
'''
def closest_colour(requested_colour):
    min_colours = {}

    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name

    return min_colours[min(min_colours.keys())]


# ======================================================================================================================
#
# ======================================================================================================================
def top_colors(image, n):
    # convert the image to rgb
    # image = image.convert('RGB')

    # resize the image to 300 x 300
    # image = image.resize((300, 300))

    detected_colors = []
    for x in range(image.width):
        for y in range(image.height):
            detected_colors.append(closest_colour(image.getpixel((x, y))))
    Series_Colors = pd.Series(detected_colors)
    output = Series_Colors.value_counts() / len(Series_Colors)
    return (output.head(n))
'''


# ======================================================================================================================
# THIS FUNCTION WILL FETCH AND READ THE NEON FIELD SITE TABLE THAT CONTAINS INFORMATION ABOUT ALL THE
# NEON SITES INCLUDING ALL THE IMAGES THAT ARE AVAILABLE FOR A SITE ON THE PHENOCAM WEBSITE.
# ======================================================================================================================
def readFieldSiteTable():
    siteList = []

    nErrorCode = GRIMe_Utils.check_url_validity(url)

    # IF AT LEAST ONE FIELD SITE TABLE IS FOUND ON THE NEON SITE...
    if nErrorCode == 0:
        csv_links = NEON_FetchFieldSiteTableURL(url)

        # download all CSV files
        filename_with_path = NEON_DownloadFieldSiteTableFiles(csv_links)

        siteList = GRIMe_Utils.parseCSV(filename_with_path)
    # ELSE IF NO FIELD SITE TABLES ARE FOUND, RETURN AN EMPTY LIST
    elif nErrorCode == -1:
        siteList = []

    return siteList


# ======================================================================================================================
# THIS FUNCTION UPDATES THE GUI WITH THE INFO FOR A NEON SITE SELECTED BY THE END-USER.
# ======================================================================================================================
def updateNEONSiteInfo(self, item):
    self.listboxNEONSiteInfo.clear()

    # EXTRACT THE SITE ID FOR THE SELECTED ITEM
    siteID = self.listboxNEONSites.currentItem().text()

    global SITECODE
    SITECODE = siteID.split(' - ')[0]

    self.labelNEONSiteDetails.setText(SITECODE)

    siteInfo = NEON_FetchSiteInfoFromNEON(SERVER, SITECODE)

    global DOMAINCODE
    DOMAINCODE = siteInfo['data']['domainCode']

    global SITENAME
    SITENAME = siteInfo['data']['siteName']
    if full == 1 or full == 4:
        self.labelNEONSite.setText(SITENAME)

    keys = siteInfo['data'].keys()

    for key in keys:
        tmp = key
        tmpSiteInfo = str(siteInfo['data'][key])
        self.listboxNEONSiteInfo.addItem(tmp + ": " + tmpSiteInfo)

    return (SITECODE)


# ======================================================================================================================
# THIS FUNCTION WILL DISPLAY THE LATEST IMAGE ON THE GUI.
# ======================================================================================================================
def displayLatestImage(self):
    global latestImage

    self.labelLatestImage.setPixmap(latestImage.scaled(self.labelLatestImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


# ======================================================================================================================
# THIS FUNCTION WILL UPDATE THE PRODUCT TABLE IN THE GUI WITH THE PRODUCTS THAT ARE AVAILABLE FOR A SPECIFIC SITE.
# ======================================================================================================================
def formatNEONProductTable(tableProducts):
    maxRows = 1

    # JES: MUST MAKE CODE DYNAMIC TO ONLY DELETE UNSELECTED ITEMS
    for i in range(tableProducts.rowCount()):
        tableProducts.removeRow(0)

    tableProducts.insertRow(0)

    for i in range(maxRows):
        m = 0
        myCheckBox = QCheckBox()
        myCheckBox.setChecked(True)
        tableProducts.setCellWidget(i, m, myCheckBox)

        m += 1
        tableProducts.setItem(i, m, QTableWidgetItem(''))

        # CONFIGURE DATES FOR SPECIFIC PRODUCT
        m += 1
        date_widget = QtWidgets.QDateEdit()
        date_widget.setDisabled(True)
        tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit()
        date_widget.setDisabled(True)
        tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(2021, 12, 1))
        tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(2022, 1, 1))
        tableProducts.setCellWidget(i, m, date_widget)

        # --------------------
        # --------------------
        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        tableProducts.setCellWidget(i, m, dateTime)

        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        tableProducts.setCellWidget(i, m, dateTime)

# ======================================================================================================================
# THIS FUNCTION WILL UPDATE THE PRODUCT TABLE IN THE GUI WITH THE PRODUCTS THAT ARE AVAILABLE FOR A SPECIFIC SITE.
# ======================================================================================================================
def updateNEONProductTable(self, item):
    products = self.listboxSiteProducts.selectedItems()

    # JES: FUTURE CONSIDERATION - MUST MAKE CODE DYNAMIC TO ONLY DELETE UNSELECTED ITEMS
    for i in range(self.tableProducts.rowCount()):
        self.tableProducts.removeRow(0)

    for i in range(len(products)):
        strText = self.listboxSiteProducts.selectedItems()[i].text()
        self.tableProducts.insertRow(i)

        productID = strText.split(':')[0]
        availableMonthList = findAvailableMonths(productID)

        length = len(availableMonthList['availableMonths'])
        strFirstMonth = availableMonthList['availableMonths'][0]
        strLastMonth = availableMonthList['availableMonths'][length - 1]

        monthFields = strFirstMonth.split('-')
        firstMonth = int(monthFields[1])
        firstYear = int(monthFields[0])

        monthFields = strLastMonth.split('-')
        lastMonth = int(monthFields[1])
        lastYear = int(monthFields[0])

        m = 0
        myCheckBox = QCheckBox()
        myCheckBox.setChecked(True)
        self.tableProducts.setCellWidget(i, m, myCheckBox)

        m += 1
        self.tableProducts.setItem(i, m, QTableWidgetItem(strText))

        # CONFIGURE DATES FOR SPECIFIC PRODUCT
        m += 2
        nYear, nMonth, nDay = GRIMe_PhenoCam.getStartDate()
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        date_widget.setDisabled(True)
        self.tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        nYear, nMonth, nDay = GRIMe_PhenoCam.getEndDate()
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        date_widget.setDisabled(True)
        self.tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        nYear, nMonth, nDay = GRIMe_PhenoCam.getStartDate()
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        # trigger event when the user changes the date
        date_widget.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.tableProducts))
        date_widget.setKeyboardTracking(False)
        self.tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        nYear, nMonth, nDay = GRIMe_PhenoCam.getEndDate()
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        # trigger event when the user changes the date
        date_widget.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.tableProducts))
        date_widget.setKeyboardTracking(False)
        self.tableProducts.setCellWidget(i, m, date_widget)

        # --------------------
        # --------------------
        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        # trigger event when the user changes the time
        dateTime.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.tableProducts))
        date_widget.setKeyboardTracking(False)
        self.tableProducts.setCellWidget(i, m, dateTime)

        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        # trigger event when the user changes the time
        dateTime.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.tableProducts))
        date_widget.setKeyboardTracking(False)
        self.tableProducts.setCellWidget(i, m, dateTime)

        self.tableProducts.resizeColumnsToContents()

# ======================================================================================================================
#
# ======================================================================================================================
def NEON_dateChangeMethod(date_widget, tableWidget):
    global SITECODE
    global DOMAINCODE
    global DOMAINCODE

    nRow = tableWidget.currentIndex().row()

    strProductIDCell = tableWidget.item(nRow, 1).text().upper()

    if strProductIDCell.find('LAND-WATER') >= 0:
        # FETCH DATES
        start_date, start_time, end_date, end_time = GRIMe_ProductTable().fetchTableDates(tableWidget, nRow)

        imageCount = GRIMe_PhenoCam.getPhenocamImageCount(SITECODE, DOMAINCODE, start_date, end_date, start_time, end_time)

        tableWidget.setItem(nRow, 2, QTableWidgetItem(str(imageCount)))

# ======================================================================================================================
#
# ======================================================================================================================
def populateUSGS_SiteTable(siteTable):
    num_cameras = len(USGS_HIVIS.fetchCameraList())

    # JES: FUTURE CONSIDERATION - MUST MAKE CODE DYNAMIC TO ONLY DELETE UNSELECTED ITEMS
    for i in range(siteTable.rowCount()):
        siteTable.removeRow(0)

    for index, camera in enumerate(USGS_HIVIS.fetchCameraList()):
        siteTable.insertRow(index)

        m = 0
        myCheckBox = QCheckBox()
        myCheckBox.setChecked(False)
        siteTable.setCellWidget(index, m, myCheckBox)

        m += 1
        siteTable.setItem(index, m, QTableWidgetItem(camera))

        imageList = USGS_HIVIS.getImageList(camera, USGS_HIVIS.getEndpoint())
        try:
            nImageCount = len(imageList)

            m += 1
            strCount = str(nImageCount) + " / " + str(len(imageList))
            siteTable.setItem(index, m, QTableWidgetItem(strCount))

            # CONFIGURE DATES FOR SPECIFIC PRODUCT
            fileStartDate, fileStartTime = USGS_HIVIS.extractDateFromFilename(imageList[0])
            fileEndDate, fileEndTime = USGS_HIVIS.extractDateFromFilename(imageList[len(imageList)-1])

            m += 1
            date_widget = QtWidgets.QDateEdit(QtCore.QDate(fileStartDate.year, fileStartDate.month, fileStartDate.day))
            date_widget.setDisabled(True)
            siteTable.setCellWidget(index, m, date_widget)
            m += 1
            date_widget = QtWidgets.QDateEdit(QtCore.QDate(fileEndDate.year, fileEndDate.month, fileEndDate.day))
            date_widget.setDisabled(True)
            siteTable.setCellWidget(index, m, date_widget)

            m += 1
            date_widget = QtWidgets.QDateEdit(calendarPopup=True)
            # JES - FOR EASE OF USE FOR NOW, LET'S SET THE MIN SELECTION DATE TO THE MAX VALUE SO WE DON'T HAVE TO SCROLL
            # THROUGH YEARS OF DATA TO GET TO THE DATE THAT WE WANT
            date_widget.setDate(QtCore.QDate(fileStartDate.year, fileStartDate.month, fileStartDate.day))
            date_widget.dateTimeChanged.connect(lambda: HIVIS_dateChangeMethod(date_widget, siteTable))
            siteTable.setCellWidget(index, m, date_widget)

            m += 1
            date_widget = QtWidgets.QDateEdit(calendarPopup=True)
            date_widget.setDate(QtCore.QDate(fileEndDate.year, fileEndDate.month, fileEndDate.day))
            date_widget.dateTimeChanged.connect(lambda: HIVIS_dateChangeMethod(date_widget, siteTable))
            siteTable.setCellWidget(index, m, date_widget)

            # --------------------
            # --------------------
            m += 1
            dateTime = QDateTimeEdit()
            dateTime.setDisplayFormat("hh:mm")
            dateTime.setFrame(False)
            dateTime.dateTimeChanged.connect(lambda: HIVIS_dateChangeMethod(date_widget, siteTable))
            siteTable.setCellWidget(index, m, dateTime)

            m += 1
            dateTime = QDateTimeEdit()
            dateTime.setDisplayFormat("hh:mm")
            dateTime.setFrame(False)
            dateTime.dateTimeChanged.connect(lambda: HIVIS_dateChangeMethod(date_widget, siteTable))
            siteTable.setCellWidget(index, m, dateTime)

            siteTable.resizeColumnsToContents()

            #time.sleep(0.100)
        except:
            nImageCount = 0

# ======================================================================================================================
#
# ======================================================================================================================
def HIVIS_dateChangeMethod(date_widget, tableWidget):
    # setting text to the label
    nRow = tableWidget.currentIndex().row()

    # FETCH DATES and TIMES
    start_date, start_time, end_date, end_time = GRIMe_ProductTable().fetchTableDates(tableWidget, nRow)

    camera = USGS_HIVIS.fetchCameraList()[nRow]

    # FILTER FOR IMAGES WITHIN DATE RANGE
    imageList = USGS_HIVIS.getImageList(camera, USGS_HIVIS.getEndpoint())

    newImageList = []
    midnight = datetime.time(0, 0, 0)
    for index, image in enumerate(imageList):
        fileDate, fileTime = USGS_HIVIS.extractDateFromFilename(image)
        if ((fileDate >= start_date) and (fileDate <= end_date)):
            if ((fileTime >= start_time) and (fileTime <= end_time)) or ((start_time == midnight) and (end_time == midnight)):
                newImageList.append(image)

    strCount = str(len(newImageList)) + " / " + str(len(imageList))

    tableWidget.setItem(nRow, 2, QTableWidgetItem(strCount))


# ======================================================================================================================
#
# ======================================================================================================================
def DP1_20002_fetchImageList(self, nRow, start_date, end_date, start_time, end_time):
    global SITECODE
    global DOMAINCODE
    global gWebImageCount
    global dailyImagesList

    if nRow > -1:
        delta = end_date - start_date

        # CREATE PROGRESS BAR
        progressBar = QProgressWheel()
        progressBar.setRange(0, delta.days + 1)
        progressBar.show()

        # CLEAR THE PREVIOUSLY DOWNLOADED IMAGE LIST, IF ANY
        dailyImagesList.clear()

        i = 1
        while start_date <= end_date:
            print(start_date)
            ymd = start_date.strftime("%Y-%d-%b")
            progressBar.setWindowTitle(ymd)
            progressBar.setValue(float(i) / float(delta.days + 1) * delta.days)
            progressBar.repaint()
            i += 1

            QCoreApplication.processEvents()

            # ----------
            #'https://phenocam.nau.edu/webcam/browse/NEON.D03.BARC.DP1.20002/2022/10/08'
            #dailyURLvisible = 'https://phenocam.nau.edu/data/latest/NEON.D10.ARIK.DP1.20002' + '/' + str(start_date.year) + '/' + str(start_date.month).zfill(2) + '/' + str(start_date.day).zfill(2)
            dailyURLvisible = 'https://phenocam.nau.edu/webcam/browse/NEON.D10.ARIK.DP1.20002' + '/' + str(start_date.year) + '/' + str(start_date.month).zfill(2) + '/' + str(start_date.day).zfill(2)

            # ----------
            dailyURLvisible = dailyURLvisible.replace('ARIK', SITECODE)
            dailyURLvisible = dailyURLvisible.replace('D10', DOMAINCODE)

            phenoCam = GRIMe_PhenoCam()
            tmpList = phenoCam.getVisibleImages(dailyURLvisible, start_time, end_time)

            dailyImagesList.setVisibleList(tmpList.getVisibleList())

            start_date += datetime.timedelta(days=1)

        gWebImageCount = len(dailyImagesList.getVisibleList())

        # INIT SPINBOX CONTROLS BASED UPON NUMBER OF IMAGES AVAILABLE
        if full == 1 or full == 4:
            self.labelImageCountNumber.setText(str(gWebImageCount))
            self.spinBoxDailyImage.setMinimum(1)
            self.spinBoxDailyImage.setMaximum(gWebImageCount)
            self.spinBoxDailyImage.setValue(1)
            self.labelImageCountNumber.setText(str(gWebImageCount))
    else:
        dailyURLvisible = []

    gWebImageCount = len(dailyImagesList.getVisibleList())

    # -------------------
    # -------------------
    # THE SOFTWARE IS NOW DESIGNED TO REQUIRE THE IMAGES TO BE DOWNLOADED FIRST FOR A VARIETY OF REASONS
    isChecked = True
    #if full == 1:
    #    isChecked = self.checkBoxSaveImages.isChecked()
    #else:
    #    isChecked = True

    if gWebImageCount > 0 and isChecked:
        # CREATE PROGRESS BAR
        progressBar = QProgressWheel()
        progressBar.setRange(0, gWebImageCount + 1)
        progressBar.show()

        i = 0
        for image in dailyImagesList.getVisibleList():
            progressBar.setWindowTitle('Download & Save Images...')
            progressBar.setValue(float(i) / float(gWebImageCount + 1) * gWebImageCount)
            i += 1

            filename = image.fullPathAndFilename.split('/')[-1]

            downloadsFilePath = os.path.join(self.EditSaveImagesOutputFolder.text(), 'Images')
            if not os.path.exists(downloadsFilePath):
                os.mkdir(downloadsFilePath)
            completeFilename = os.path.join(downloadsFilePath, filename)
            urllib.request.urlretrieve(image.fullPathAndFilename, completeFilename)

        # THE SOFTWARE IS NOW DESIGNED TO REQUIRE THE IMAGES TO BE DOWNLOADED FIRST FOR A VARIETY OF REASONS
        # DISABLE THE CHECK BOX SO THAT ONCE THE IMAGES ARE DOWNLOADED, IMAGES ARE NOT CONTINUED TO BE SAVED UNBEKNOWNST TO THE USER
        #if full == 1:
        #    self.checkBoxSaveImages.setChecked(False)

        # clean-up before exiting function
        # 1. close and delete the progress bar
        # 2. no other clean-up tasks
        progressBar.close()
        del progressBar

        #if full == 1:
        #    strMessage = 'Data download is complete!' + '\n\n' + 'If you selected "Save Images", it will be deselected to avoid unnecessary re-downloading of the same data.' + '\n\n' + 'If you want to re-download the same data or additional data, you will have to reselect "Save Images."'
        #    msgBox = GRIMe_QMessageBox('Data Download', strMessage)
        #    response = msgBox.displayMsgBox()
        strMessage = 'Data download is complete!'
        msgBox = GRIMe_QMessageBox('Data Download', strMessage)
        response = msgBox.displayMsgBox()


# ======================================================================================================================
#
# ======================================================================================================================
def downloadPBTImageFiles(self):
    GRIMe_ProductTableObj = GRIMe_ProductTable()
    GRIMe_ProductTableObj.fetchTableDates(self.tableProducts, nRow)

    start_date = GRIMe_ProductTableObj.getStartDate()
    strStartDate = GRIMe_ProductTableObj.getStartDate()

    end_date = GRIMe_ProductTableObj.getEndDate()
    strEndDate = GRIMe_ProductTableObj.getEndDate()

    startTime = GRIMe_ProductTableObj.getStartTime()
    endTime = GRIMe_ProductTableObj.getEndTime()

    delta = GRIMe_ProductTableObj.getDelta()

    start_date += datetime.timedelta(days=1)


# ======================================================================================================================
# DOWNLOAD THE PRODUCT FILES SELECTED IN THE GUI BY THE END-USER.
# ======================================================================================================================
def downloadProductDataFiles(self, item):
    global dailyImagesList
    global currentImageIndex

    nitrateList = []

    # ----------------------------------------------------------------------------------------------------
    # SAVE DOWNLOADED DATA TO THE USER GRIMe-AI FOLDER THAT IS AUTOMATICALLY CREATED, IF IT DOES NOT EXIST,
    # CREATE IT IN THE USER'S DOCUMENT FOLDER
    # ----------------------------------------------------------------------------------------------------
    downloadsFilePath = self.EditSaveImagesOutputFolder.text()

    if len(downloadsFilePath) == 0:
        strMessage = 'A download folder has not been specified. Would you like to use the default GRIME-AI download folder?'
        msgBox = GRIMe_QMessageBox('Download Image Files', strMessage, QMessageBox.Yes | QMessageBox.No)
        response = msgBox.displayMsgBox()

        if response == QMessageBox.Yes:
            downloadsFilePath = os.path.expanduser('~')
            downloadsFilePath = os.path.join(downloadsFilePath, 'Documents')
            downloadsFilePath = os.path.join(downloadsFilePath, 'GRIMe-AI')
            if not os.path.exists(downloadsFilePath):
                os.mkdir(downloadsFilePath)
            downloadsFilePath = os.path.join(downloadsFilePath, 'Downloads')
            if not os.path.exists(downloadsFilePath):
                os.mkdir(downloadsFilePath)

            self.EditSaveImagesOutputFolder.setText(downloadsFilePath)

    # --------------------------------------------------------------------------------
    # FIND IMAGE PRODUCT (20002) ROW TO GET DATE RANGE
    # --------------------------------------------------------------------------------
    if self.checkBoxNEONSites.isChecked():
            rowRange = range(self.tableProducts.rowCount())
    else:
        rowRange = range(1)

    for nRow in rowRange:
        # --------------------------------------------------------------------------------
        # FETCH THE DATE RANGE THE USER SELECTED FOR THE PARTICULAR DATA
        # --------------------------------------------------------------------------------
        GRIMe_ProductTableObj = GRIMe_ProductTable()
        start_date, start_time, end_date, end_time = GRIMe_ProductTableObj.fetchTableDates(self.tableProducts, nRow)

        #if self.checkBoxNEONSites.isChecked() and self.radioButtonWebSiteImages.isChecked():
        #if self.checkBoxNEONSites.isChecked():
        # EXTRACT THE PRODUCT ID
        strProductIDCell = self.tableProducts.item(nRow, 1).text()
        nProductID = int(strProductIDCell.split('.')[1])
        #else:
        #    nProductID = -999

        if nProductID < 0 and nRow == 0:
            if self.radioButtonHardDriveImages.isChecked():
                global imageFileFolder
                #JES fetchLocalImageList(self, imageFileFolder, self.checkBox_FetchRecursive.isChecked(), self.checkBoxCreateEXIFFile.isChecked(), start_date, end_date, start_time, end_time)
                fetchLocalImageList(self, imageFileFolder, True, self.checkBoxCreateEXIFFile.isChecked(), start_date, end_date, start_time, end_time)
        elif nProductID > 0:
            # ----------------------------------------------------------------------------------------------------
            # FETCH THE DATA FOR THE PARTICULAR PRODUCT FOR THE SPECIFIED DATE RANGE AND PROCESS THE DATA ACCORDINGLY
            # ----------------------------------------------------------------------------------------------------
            if nProductID == 20002:
                DP1_20002_fetchImageList(self, nRow, start_date, end_date, start_time, end_time)

                if full == 1 or full == 4:
                    processLocalImage(self)
                    #refreshImage(self)
            elif nProductID == 20033:
                # CREATE PROGRESS BAR
                nitrateDataObj = NEON_20033_NitrateData()

                strStartYearMonth = str(start_date.year) + '-' + str(start_date.month).zfill(2)
                strEndYearMonth = str(end_date.year) + '-' + str(end_date.month).zfill(2)
                nitrateList = nitrateDataObj.DP1_20033_FetchNitrateData(SITECODE, strProductIDCell, strStartYearMonth, strEndYearMonth, downloadsFilePath)

                if full == 1 or full == 4:
                    if len(nitrateList) > 0:
                        # JES - USE NITRATE DATA FOR DEVELOPING GENERIC CSV READING AND DATA GRAPHING CAPABILITIES
                        scene = QGraphicsScene()
                        self.scene = scene
                        nWidth = self.graphicsView.width()
                        nHeight = self.graphicsView.height()
                        nX = self.graphicsView.x()
                        nY = self.graphicsView.y()
                        self.scene.setSceneRect(0, 0, nWidth, nHeight)
                        # self.graphicsView.setWindowTitle('Nitrate Data')
                        self.graphicsView.setScene(self.scene)
                        figure = Figure()
                        axes = figure.gca()
                        axes.set_title("Nitrate Data")

                        i = 0
                        for i, nitrateData in enumerate(nitrateList):
                            y = float(nitrateData.getNitrateMean())
                            axes.plot(i, y, '.', markersize=2)

                        canvas = FigureCanvas(figure)
                        canvas.resize(nWidth, nHeight)
                        self.scene.addWidget(canvas)
                        self.graphicsView.show()
            else:
                fetchData = NEON_FetchData()

                strStartYearMonth = str(start_date.year) + '-' + str(start_date.month).zfill(2)
                strEndYearMonth = str(end_date.year) + '-' + str(end_date.month).zfill(2)
                fetchData.NEON_FetchData(SITECODE, strProductIDCell, strStartYearMonth, strEndYearMonth, downloadsFilePath)

# ======================================================================================================================
#
# ======================================================================================================================
def labelEdgeImageDoubleClickEvent(self):
    global currentImage

    # CONVERT IMAGE FROM QImage FORMAT TO Mat FORMAT
    img = GRIMe_Utils.convertQImageToMat(currentImage.toImage())
    edges = []

    if self.radioButtonCanny.isChecked():
        # EDGE DETECTION: CANNY
        edges = cv2.Canny(img, 100, 200)
    elif self.radioButtonSIFT.isChecked():
        # CONVERT COLOR IMAGE TO GRAY SCALE
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # REMOVE NOISE FROM THE IMAGE
        if len(gray) != 0:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        edges = cv2.drawKeypoints(gray, kp, img)
    elif self.radioButtonORB.isChecked():
        value = self.spinBoxOrbMaxFeatures.value()
        edges = calcOrb(img, value)
    elif self.radioButtonSobelX.isChecked() or self.radioButtonSobelY or self.radioButtonSobelXY:
        mySobel = sobelData()
        sobelKernelSize = self.spinBoxSobelKernel.value()
        mySobel.setSobelX(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobelKernelSize))
        mySobel.setSobelY(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobelKernelSize))
        mySobel.setSobelXY(cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=sobelKernelSize))
        if self.radioButtonSobelX.isChecked():
            edges = (mySobel.getSobelX() * 255 / mySobel.getSobelX().max()).astype(np.uint8)
        elif self.radioButtonSobelY.isChecked():
            edges = (mySobel.getSobelY() * 255 / mySobel.getSobelY().max()).astype(np.uint8)
        elif self.radioButtonSobelXY.isChecked():
            edges = (mySobel.getSobelXY() * 255 / mySobel.getSobelXY().max()).astype(np.uint8)

    if not edges == []:
        cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
        imS = cv2.resize(edges, (960, 540))
        cv2.imshow('Output', imS)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# ======================================================================================================================
#
# ======================================================================================================================
def labelOriginalImageDoubleClickEvent(self):
    global currentImage

    img = GRIMe_Utils.convertQImageToMat(currentImage.toImage())

    self.setMouseTracking(False)

    cv2.imshow('Original', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    self.setMouseTracking(True)


# ======================================================================================================================
#
# ======================================================================================================================
def labelMouseDoubleClickEvent(self):
    global latestImage

    img = GRIMe_Utils.convertQImageToMat(latestImage.toImage())
    cv2.imshow('Original', img)

    # ----------
    if 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        cv2.imshow('Mask', mask)

        res = cv2.bitwise_and(img, img, mask=mask)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        cv2.imshow('laplacian', laplacian)

        mySobel = sobelData()
        mySobel.setSobelX(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5))
        mySobel.setSobelY(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5))

        cv2.imshow('Sobel: x-axis', mySobel.sobelX)
        cv2.imshow('Sobel: y-axis', mySobel.sobelY)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ----------
    if 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('Original', img)
        edges = cv2.Canny(img, 100, 200)
        cv2.imshow('Canny Edges', edges)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ----------
    if 0:
        cv2.imshow("Latest Image (Color)", img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Latest Image (Gray)", gray)

        cv2.waitKey(0)

        cv2.destroyAllWindows()


# ======================================================================================================================
#
# ======================================================================================================================
def updateSiteProducts(self, item):
    site_json = NEON_FetchSiteInfoFromNEON(SERVER, SITECODE)

    self.listboxSiteProducts.clear()

    for product in site_json['data']['dataProducts']:
        strText = product['dataProductCode'] + ": " + product['dataProductTitle']
        assert isinstance(strText, object)
        self.listboxSiteProducts.addItem(strText)

    self.listboxSiteProducts.show()

    # JES - TEMPORARILY SET NITRATE DATA ('should only be one nitrate product') AS THE DEFAULT SELECTION
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    itemNitrate = self.listboxSiteProducts.findItems('Nitrate', QtCore.Qt.MatchContains)
    nIndex = 0
    if len(itemNitrate) > 0:
        for item in itemNitrate:
            nIndex = self.listboxSiteProducts.row(item)
            self.listboxSiteProducts.setCurrentRow(nIndex)

        updateNEONProductTable(self, nIndex)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # JES

    # JES - TEMPORARILY SET NITRATE DATA ('should only be one nitrate product') AS THE DEFAULT SELECTION
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    item20002 = self.listboxSiteProducts.findItems('20002', QtCore.Qt.MatchContains)
    nIndex = 0
    if len(item20002) > 0:
        for item in item20002:
            nIndex = self.listboxSiteProducts.row(item)
            self.listboxSiteProducts.setCurrentRow(nIndex)

        updateNEONProductTable(self, nIndex)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # JES


# ======================================================================================================================
#
# ======================================================================================================================
def retranslateUi(self, MainWindow):
    _translate = QtCore.QCoreApplication.translate
    MainWindow.setWindowTitle(_translate("GRIMe-AI: John E. Stranzl Jr.", "GRIMe-AI: John E. Stranzl Jr."))

# ======================================================================================================================
# FIND THE MONTHS THAT DATA IS AVAILABLE FOR A PARTICULAR PRODUCT FOR A PARTICULAR SITE
# ======================================================================================================================
def findAvailableMonths(item):
    global SERVER
    global SITECODE

    PRODUCTCODE = item

    # RETRIEVE INFORMATION FROM THE NEON WEBSITE FOR THE PARTICULAR SITE
    site_json = NEON_FetchSiteInfoFromNEON(SERVER, SITECODE)

    # EXTRACT THE AVAILABLE MONTH AND THE URL FOR THE DATA FOR EACH AVAILABLE MONTH
    monthList = {}
    for product in site_json['data']['dataProducts']:
        if (product['dataProductCode'] == PRODUCTCODE):
            monthList['availableMonths'] = product['availableMonths']
            monthList['availableDataUrls'] = product['availableDataUrls']
            break

    return (monthList)


# ======================================================================================================================
#
# ======================================================================================================================
def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape

    print("The height and width of the image are: height,width", height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
                print("max_gray_level:", max_gray_level)

    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()

    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)

    # If the number of gray levels is greater than gray_level, the gray level of the image is reduced to gray_level, reduce the size of the gray-level co-occurrence matrix
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def feature_computer(p):
    # con:Contrast reflects the sharpness of the image and the depth of the grooves of the texture. The sharper the texture, the greater the contrast, the greater the contrast.
    # eng:Entropy,ENT) measures the randomness of the amount of information contained in the image and expresses the complexity of the image. When all the values in the co-occurrence matrix are equal or the pixel values show the greatest randomness, the entropy is the largest.
    # agm:Angle second-order moment (energy), a measure of the uniformity of image gray distribution and texture thickness. When the image texture is uniform and regular, the energy value is large; on the contrary, the element values of the gray-level co-occurrence matrix are similar, and the energy value is small.
    # idm:The inverse difference matrix is also called the inverse variance, which reflects the clarity and regularity of the texture. The texture is clear, regular, easy to describe, and has a larger value.
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0

    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])

    return Asm, Con, -Eng, Idm


def test(img):
    img_shape = img.shape

    # If you use &#39;/&#39;Will report TypeError: integer argument expected, got float
    # In fact, the main error is because of cv2.The parameter in resize is required to be an integer
    img = cv2.resize(img, (img_shape[1] // 2, img_shape[0] // 2), interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    glcm_0 = getGlcm(img_gray, 1, 0)
    # glcm_1=getGlcm(src_gray,0,1)
    # glcm_2=getGlcm(src_gray,1,1)
    # glcm_3=getGlcm(src_gray,-11)print(glcm_0)

    asm, con, eng, idm = feature_computer(glcm_0)
    return [asm, con, eng, idm]


# ======================================================================================================================
#
# ======================================================================================================================
################################################################################
class MSTExtractFrames():
    def __init__(self, inputVideo, outFolder, start, end, *args, **kwargs):
        super(MSTExtractFrames, self).__init__(*args, **kwargs)
        self.inputVideo = inputVideo
        self.outFolder = outFolder
        self.vidcap = cv2.VideoCapture(inputVideo)
        self.frameCount = self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        self.frameFailures = 0

        self.startFrame = int(start)
        if end == 'ALL':
            self.endFrame = int(self.frameCount - 1)
        else:
            self.endFrame = int(end)

        # Now, go do it!
        self.makeOutFolder()

    def makeOutFolder(self):
        if not os.path.exists(self.outFolder):
            os.makedirs(self.outFolder)

    def process_frame(self, frameIndex):
        print("Get Frame at frameIndex", frameIndex)

        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        hasFrames, image1 = self.vidcap.read()
        num = str(frameIndex)
        znum = num.zfill(4)
        fn = "image" + znum + ".jpg"
        path = self.outFolder + "/" + fn
        cv2.imwrite(path, image1)

    def process_all_frames(self):
        for frameIndex in range(self.startFrame, self.endFrame):
            self.process_frame(frameIndex)

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def googleMap(self):
        # Enter your api key here
        api_key = "_your_api_key_"

        # url variable store url
        url = "https://maps.googleapis.com/maps/api/staticmap?"

        # center defines the center of the map,
        # equidistant from all edges of the map.
        center = "Dehradun"

        # zoom defines the zoom
        # zoom defines the zoom
        # level of the map
        zoom = 10

        # get method of requests module
        # return response object
        r = requests.get(
            url + "center =" + center + "&zoom =" + str(zoom) + "&size = 400x400&key =" + api_key + "sensor = false")

        # wb mode is stand for write binary mode
        f = open('address of the file location ', 'wb')

        # r.content gives content,
        # in this case gives image
        f.write(r.content)

        # close method of file object
        # save and close the file
        f.close()

# ======================================================================================================================
#
# ======================================================================================================================
def fetchUSGSImages(table_USGS_Sites, output_folder_base):

    num_cameras = len(USGS_HIVIS.fetchCameraList())

    for nIndex in range(table_USGS_Sites.rowCount()):

        if table_USGS_Sites.cellWidget(nIndex, 0).isChecked():
            start_date, start_time, end_date, end_time = GRIMe_ProductTable().fetchTableDates(table_USGS_Sites, nIndex)

            # FETCH CAMERA FOR SELECTED LOCATION
            camera = USGS_HIVIS.fetchCameraList()[nIndex]

            # FETCH IMAGE LIST FOR SELECTED LOCATION
            imageList = USGS_HIVIS.getImageList(camera, USGS_HIVIS.getEndpoint())

            # NUMBER OF IMAGES FOR SELECTED LOCATION
            nImageCount = len(imageList)

            # INITIALIZE PROGRESS "BAR" FOR DISPLAYING PROGRESS OF CULLING IMAGES BY DATE/TIME
            progressBar = QProgressWheel(0, nImageCount + 1)
            progressBar.setWindowTitle('Culling Images (Date/Time)...')
            progressBar.show()

            # CULL IMAGES BASED UPON SELECTED DATE/TIME STAMPLES
            newImageList = []
            midnight = datetime.time(0, 0, 0)
            for index, image in enumerate(imageList):
                progressBar.setValue(index)
                fileDate, fileTime = USGS_HIVIS.extractDateFromFilename(image)
                if ((fileDate >= start_date) and (fileDate <= end_date)):
                    if ((fileTime >= start_time) and (fileTime <= end_time)) or ((start_time == midnight) and (end_time == midnight)):
                        newImageList.append(image)

            progressBar.close()
            del progressBar

            # NOTE FROM FRANK ENGEL:
            # The NATWEB server sits in the USGS AWS govCloud. AWS imposes http request limits. You can probably pull
            # up to 100 requests in 10 mins or so before NATWEB closes the connection.
            output_folder = output_folder_base + os.sep + 'USGS'+ os.sep + camera

            download_USGS_HIVIS_Images(newImageList, camera_name=camera, save_directory=output_folder, endpoint=USGS_HIVIS.getEndpoint())


# ======================================================================================================================
#
# ======================================================================================================================
def loadChromeDriver():

    # ----------------------------------------------------------------------------------------------------
    # GET THE VERSION OF CHROME INSTALLED ON THE COMPUTER AND THE VERSION OF THE CHROME DRIVER INSTALLED WITH GRIME-AI
    # ----------------------------------------------------------------------------------------------------
    strChromeVersion = get_chrome_version()
    print(strChromeVersion)

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')  # Last I checked this was necessary.

    # strChromeExe = os.path.join('C:/Program Files (x86)/GRIME-AI/chromedriver/', strChromeVersion, 'chromedriver.exe')
    # Old driver = webdriver.Chrome(strChromeExe, options=options)  # Optional argument, if not specified will search path.

    options = webdriver.ChromeOptions()

    #webdriver.Chrome.close()

    try:
        try:
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        except:
            strChromeExe = os.path.join('C:/Program Files (x86)/GRIME-AI/chromedriver/chromedriver.exe')
            print(strChromeExe)
            # driver = webdriver.Chrome(strChromeExe, options=options)  # Optional argument, if not specified will search path.
            driver = webdriver.Chrome(strChromeExe)  # Optional argument, if not specified will search path.

        strChromeDriverVersion = driver.capabilities['browserVersion']
    except:
        pass

    """
    if strChromeVersion != strChromeDriverVersion:
        msgBox = GRIMe_QMessageBox('Chrome Driver Error!', 'Chrome Version: ' + strChromeVersion + '\nChrome Driver Version: ' + strChromeDriverVersion + '\n\nYou can use the software for analyzing data but you cannot download images from the Internet until the Chrome Driver and Chrome Browser versions match.')
        response = msgBox.displayMsgBox()
    else:
        msgBox = GRIMe_QMessageBox('Chrome Version', 'Chrome Version: ' + strChromeVersion + '\nChrome Driver Version: ' + strChromeDriverVersion)
        response = msgBox.displayMsgBox()
    """

# ======================================================================================================================
#
# ======================================================================================================================
if __name__ == '__main__':

    # CREATE MAIN APP WINDOW
    app = QApplication(sys.argv)
    frame = MainWindow()

    frame.move(app.desktop().screen().rect().center() - frame.rect().center())

    loadChromeDriver()

    # ----------------------------------------------------------------------------------------------------
    # DISPLAY SPLASH SCREEN
    # ----------------------------------------------------------------------------------------------------
    if full == 1:
        pixmap = QPixmap('Splash_007.jpg')
    elif full == 4:
        pixmap = QPixmap('Splash_007FE.jpg')
    else:
        pixmap = QPixmap('Splash_007DM.jpg')
    splash = QSplashScreen(pixmap)
    splash.show()
    time.sleep(5)
    splash.finish(frame)

    # IF THE FILE IS NOT IN THE EXECUTABLE DIRECTORY, MAKE SURE THE SOFTWARE STILL STARTS UP.
    #try:
    #    playsound('shall-we-play-a-game.mp3')
    #except:
    #    pass

    # PROCESS ANY EVENTS THAT WERE DELAYED BECAUSE OF THE SPLASH SCREEN
    app.processEvents()

    # CREATE REQUIRED FOLDERS IN THE USER'S DOCUMENTS FOLDER
    GRIMe_Utils.createGRIMeFolders(frame, full)

    #if torch.cuda.is_available():
        #print("Torch CUDA!")

    #fetchUSGSImages()
    #exit()

    #featureMatch()
    #exit()

    # ----------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------
    if 0:
        try:
            cainfo = certifi.where()
        except AttributeError:
            cainfo = None

        if cainfo:
            pycurl.Curl().setopt(pycurl.CAINFO, cainfo)
            print(cainfo)

    # config = tf.compat.v1.ConfigProto()
    # tf.config.list_physical_devices('GPU')

    # print(dev_lib.list_local_devices())

    # INITIALIZE GUI CONTROLS
    # frame.listboxNEONSites.setCurrentRow(1)

    frame.checkBoxNEONSites.setChecked(True)

    # GET LIST OF ALL SITES ON NEON
    if frame.checkBoxNEONSites.isChecked():
        siteList = readFieldSiteTable()
    else:
        formatNEONProductTable(frame.tableProducts)

    if len(siteList) == 0:
        pass
        #frame.radioButtonHardDriveImages.setChecked(True)
        #frame.radioButtonHardDriveImages.setDisabled(False)
    # IF THERE ARE FIELD SITE TABLES AVAILABLE, ENABLE GUI WIDGETS PERTAINING TO WEB SITE DATA/IMAGES
    else:
        myList = []

        for site in siteList:
            strSiteName = site.siteID + ' - ' + site.siteName
            myList.append(strSiteName)

        frame.listboxNEONSites.addItems(myList)

        # JES - TEMPORARILY SET BARCO LAKE AS THE DEFAULT SELECTION
        frame.listboxNEONSites.setCurrentRow(2)
        frame.listboxNEONSites.show()

    #JES if full == 1:
    #JES    frame.radioButton_ROIShapeRectangle.setChecked(True)

        # ----------------------------------------------------------------------
        # CHANGE THE COLOR OF OPTIONS THAT HAVE SPECIAL CONDITIONS AND USAGES
        #frame.checkBoxSaveImages.setStyleSheet('QCheckBox {color: red;}')

        #JES frame.pushButtonAddROI.setStyleSheet('QPushButton {background-color: steelblue;}')

    '''
    if full == 1 or full == 4:
        frame.spinBoxCannyHighThreshold.setValue(255)
        frame.spinBoxCannyLowThreshold.setValue(254)
    '''

    if full == 1:
        frame.graphicsView.setVisible(False)

        print("--- Fetching NEON Sites Info ---")
        start_time = time.time()
        formatNEONProductTable(frame.table_USGS_Sites)
        print("--- NEON Sites Info = %s seconds ---" % (time.time() - start_time))

        print("--- DISABLED! USGS Site is SLOW!   Fetching USGS Sites Info ---")
        start_time = time.time()
        #JES populateUSGS_SiteTable(frame.table_USGS_Sites)
        print("--- DISABLED!  USGS Sites Info = %s seconds ---" % (time.time() - start_time))

    #JES frame.pushButton_RetrieveNEONData.setStyleSheet('QPushButton {background-color: steelblue;}')
    #JES frame.pushButton_FetchImageList.setStyleSheet('QPushButton {background-color: steelblue;}')

    #frame.pushButtonBrowseImageFolder.setStyleSheet('QPushButton {background-color: steelblue;}')
    #frame.pushButtonBrowseVideoOutputFolder.setStyleSheet('QPushButton {background-color: steelblue;}')
    #frame.pushButtonBrowseSaveImagesOutputFolder.setStyleSheet('QPushButton {background-color: steelblue;}')
    #frame.pushButtonBrowseEXIFOutputFolder.setStyleSheet('QPushButton {background-color: steelblue;}')

    '''
    # Enter your api key here
    api_key = "_your_api_key_"

    # url variable store url
    url = "https://maps.googleapis.com/maps/api/staticmap?"

    # center defines the center of the map,
    # equidistant from all edges of the map.
    center = "Dehradun"

    # zoom defines the zoom
    # level of the map
    zoom = 10

    # get method of requests module
    # return response object
    r = requests.get(url + "center =" + center + "&zoom =" +
                     str(zoom) + "&size = 400x400&key =" +
                     api_key + "sensor = false")

    # wb mode is stand for write binary mode
    f = open('address of the file location ', 'wb')

    # r.content gives content,
    # in this case gives image
    f.write(r.content)

    # close method of file object
    # save and close the file
    f.close()
    '''

    # product_json = NEON_QueryProductInfo('DP1.00004.001')

    # data_request = requests.get('http://data.neonscience.org/api/v0/' + 'data/' + 'DP1.00004.001' + '/' + 'ARIK' + '/' + '2018-06')
    # data_json = data_request.json()

    # NEON_API_Example('DP1.00004.001')

    bStartupComplete = True

    # SHOW MAIN WINDOW
    frame.show()

    # Run the program
    sys.exit(app.exec())

def initUSGS():
    # check connection to HIVIS
    # load list of sites
    # get image count (per site)
    #
    x = 1




# NOTES
# field_domain_id
# field_site_id
# field_site_name
# field_site_type
# field_site_subtype
# field_colocated_site
# field_site_host
# field_site_url
# field_nonneon_research_allowed
# field_access_details
# field_neon_field_operations_office
# field_latitude
# field_longitude
# field_geodetic_datum
# field_utm_northing
# field_utm_easting
# field_utm_zone
# field_site_county
# field_site_state
# field_site_country
# field_mean_elevation_m
# field_minimum_elevation_m
# field_maximum_elevation_m
# field_mean_annual_temperature_C
# field_mean_annual_precipitation_mm
# field_dominant_wind_direction
# field_mean_canopy_height_m
# field_dominant_nlcd_classes
# field_domint_plant_species
# field_usgs_huc
# field_watershed_name
# field_watershed_size_km2
# field_lake_depth_mean_m
# field_lake_depth_max_m
# field_tower_height_m
# field_usgs_geology_unit
# field_megapit_soil_family
# field_soil_subgroup
# field_avg_number_of_green_days
# field_avg_grean_increase_doy
# field_avg_green_max_doy
# field_avg_green_decrease_doy
# field_avg_green_min_doy
# field_phenocams
# field_number_tower_levels

# EXTRACT AND PARSE XMP DATA FROM IMAGE
if 0:
    with open(videoFileList[i], "rb") as fin:
        img = fin.read()
        imgAsString = str(img)
        xmp_start = imgAsString.find('<x:xmpmeta')
        xmp_end = imgAsString.find('</x:xmpmeta')
        if xmp_start != xmp_end:
            xmpString = imgAsString[xmp_start:xmp_end + 12]

        xmpAsXML = BeautifulSoup(xmpString)
        print(xmpAsXML.prettify())

if 0:
    PIL_Image = Image.open(videoFileList[i])

    exifdata = PIL_Image._getexif()

    # ITERATE THROUGH ALL THE TAGS AND EXTRACT EXIF DATA
    strHeader = []
    strHeader.append("Filename")

    strEXIF = []
    strFilename = os.path.basename(videoFileList[i])
    strEXIF.append(strFilename)

    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes
        if isinstance(data, bytes):
            data = data.decode()
        print(f"{tag:25}: {data}")
        if i == 0:
            strHeader.append(f"{tag:25}")

        strEXIF.append(f"{data}")

    # write a row to the csv file
    if i == 0:
        writer.writerow(strHeader)
    writer.writerow(strEXIF)


