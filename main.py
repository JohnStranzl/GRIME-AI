# !/usr/bin/env python
# coding: utf-8

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

import colorsys
import csv
import datetime
import math
import os
import re
import shutil
import ssl
import sys
import time
import pathlib
import urllib.request
from configparser import ConfigParser
from pathlib import Path
from urllib.request import urlopen

import PIL
import imageio
import cv2
import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import promptlib
#import torch
# print(torch.__version__)
import requests
import scipy.misc
from PIL import Image, ImageQt
# ------------------------------------------------------------
#
# ------------------------------------------------------------
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QToolBar, QCheckBox, QDateTimeEdit, \
    QGraphicsScene, QMessageBox, QSplashScreen
from bs4 import BeautifulSoup
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.cluster import KMeans

import sobelData
from GRIMe_Color import GRIMe_Color
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
from NEON_20033_NitrateData import NEON_20033_NitrateData
from NEON_FetchData import NEON_FetchData
from exifData import EXIFData
from GRIMe_ImageStats import GRIMe_ImageStats
from USGS_HIVIS import USGS_HIVIS

if 1:
    from neonAIgui import Ui_MainWindow

    full = 1
else:
    from GRIMe_AIDownloadManager import Ui_MainWindow

    full = 0

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

global SITECODE
SITECODE = 'ARIK'

global DOMAINCODE
DOMAINCODE = 'D10'

global SITENAME
SITENAME = ''

global originalImg
originalImg = []

global dailyImagesList
dailyImagesList = dailyList([], [])

global currentImage
currentImage = []

global currentImageIndex
currentImageIndex = -1

global latestImage
latestImage = []

siteList = []

global nStop
nStop = 0

global gWebImageCount
gWebImageCount = 0

global gWebImagesAvailable
gWebImagesAvailable = 0

global gFrameCount
gFrameCount = 0

global gProcessClick
gProcessClick = 0

# Define the maximum number of gray levels
gray_level = 16

# URLS
# url = "http://maps.googleapis.com/maps/api/geocode/json?address=googleplex&sensor=false"
url = 'https://www.neonscience.org/field-sites/explore-field-sites'
root_url = 'https://www.neonscience.org'
SERVER = 'http://data.neonscience.org/api/v0/'
MSTFolder = 'C:/Users/Astrid Haugen/Documents/000 - MeanStride'


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

    # --------------------------------------------------
    # --------------------------------------------------


    # --------------------------------------------------
    # --------------------------------------------------
    def itemClicked(self, item):
        global SITECODE
        global gWebImagesAvailable
        global gProcessClick

        if gProcessClick == 0:
            gProcessClick = 1

            SITECODE = updateNEONSiteInfo(self, item)

            updateSiteProducts(self, item)

            nErrorCode = downloadLatestImage(SITECODE, DOMAINCODE)
            if nErrorCode == 404:
                gWebImagesAvailable = 0
                self.labelLatestImage.setText("No Images Available")

                self.radioButtonWebSiteImages.setDisabled(True)
                self.radioButtonHardDriveImages.setChecked(True)
            else:
                gWebImagesAvailable = 1

                if full == 1:
                    self.radioButtonWebSiteImages.setDisabled(False)
                    self.radioButtonWebSiteImages.setChecked(True)

                    self.radioButtonHardDriveImages.setChecked(False)
                    self.radioButtonHardDriveImages.show()

                displayLatestImage(self)

            gProcessClick = 0

    # --------------------------------------------------
    # --------------------------------------------------
    def productClicked(self, item):
        updateSelectedProductsTable(self, item)

    # --------------------------------------------------
    # --------------------------------------------------

    def pushButtonRetrieveDataClicked(self, item):
        downloadProductDataFiles(self, item)

    # --------------------------------------------------
    # --------------------------------------------------
    def spinBoxChanged(self):
        global currentImageIndex
        global gFrameCount

        #if self.radioButtonWebSiteImages.isChecked():
        #    currentImageIndex = self.spinBoxDailyImage.value()
        #    processSpinBoxImage(self, currentImageIndex)
        #    refreshImage(self)
        #elif self.radioButtonHardDriveImages.isChecked():
        if currentImageIndex != self.spinBoxDailyImage.value():
            currentImageIndex = self.spinBoxDailyImage.value()
            imageFolder = self.lineEditHardDriveFolder.text()
            if len(imageFolder) > 0:
                if gFrameCount > 0:
                    processLocalImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def updateCanny(self):
        imageNumber = self.spinBoxDailyImage.value()
        #self.spinBoxChanged()
        refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def spinBoxCannyHighThresholdChanged(self):
        imageNumber = self.spinBoxDailyImage.value()
        #self.spinBoxChanged()
        refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def spinBoxCannyLowThresholdChanged(self):
        imageNumber = self.spinBoxDailyImage.value()
        self.labelOriginalImage.clear()
        #self.spinBoxChanged()
        refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def spinBoxCannyKernelChanged(self):
        imageNumber = self.spinBoxDailyImage.value()
        #self.spinBoxChanged()
        refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def updateSobelX(self):
        refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def updateSobelY(self):
        refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def updateSobelXY(self):
        refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def spinBoxSobelKernelChanged(self):
        refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def updateSIFT(self):
        refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def updateORB(self):
        refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def updateLaplacian(self):
        refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def initTab3(self):
        if processSpinBoxImage(self, self.spinBoxDailyImage.value()) == 0:
            refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def spinBoxOrbMaxFeaturesChanged(self):
        refreshImage(self)

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
    def MSTVideoFolderChanged(self):
        global gFrameCount

        frameFolder = self.lineEditHardDriveFolder.text() + '/' + self.MSTlineEditFrameFolder.text()
        nFrameCount = MSTGetFrameCount(self, frameFolder)

        strFrameCount = 'Frame Count: ' + nFrameCount.__str__()
        self.MSTFrameCount.setText(strFrameCount)

        self.spinBoxDailyImage.setMinimum(1)
        self.spinBoxDailyImage.setMaximum(nFrameCount)
        self.spinBoxDailyImage.setValue(1)

        gFrameCount = nFrameCount

    # --------------------------------------------------
    # --------------------------------------------------
    def MSTStop(self):
        global nStop
        nStop = 1

    def pushButtonCreateVideoClicked(self):
        createVideo(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def GRIMePlay(self):
        global dailyImagesList
        global nStop
        global gWebImageCount

        bCreateEXIFFile = False
        bCreateVideos = False

        hardDriveImageFolder = self.lineEditHardDriveFolder.text() + '/'

        # ----------------------------------------------------------------------------------------------------
        # GENERATE LIST OF IMAGE FILES IN FOLDER
        # ----------------------------------------------------------------------------------------------------
        videoFileList = dailyImagesList.getVisibleList()
        nFrameCount = len(videoFileList)

        if len(videoFileList) > 0:
            nImageIndex = self.spinBoxDailyImage.value()
            if nImageIndex > gFrameCount:
                nImageIndex = gFrameCount

            inputFrame = videoFileList[nImageIndex - 1].fullPathAndFilename  # zero based index

            nStop = 0

            # ----------------------------------------------------------------------------------------------------
            # READ THE FIRST IMAGE IN THE LIST IN ORDER TO GET ITS HEIGHT, WIDTH AND LAYER INFORMATION
            # THE ASSUMPTION IS THAT ALL THE IMAGES IN THE FOLDER ARE OF THE SAME SIZE
            # ----------------------------------------------------------------------------------------------------
            filename = videoFileList[0].fullPathAndFilename

            nFrameCount = gWebImageCount

            img = GRIMe_Color.loadColorImage(self.radioButtonWebSiteImages, filename)
            height, width, layers = img.shape
            size = (width, height)

            # ----------------------------------------------------------------------------------------------------
            # EXTRACT EXIF DATA
            # ----------------------------------------------------------------------------------------------------
            if self.radioButtonHardDriveImages.isChecked() and self.checkBoxCreateEXIFFile.isChecked():
                # open the file in the write mode
                EXIFFolder = self.EditEXIFOutputFolder.text()
                csvFile = open(EXIFFolder + '/' + 'EXIFData.csv', 'w', newline='')

                # create the csv writer
                if bCreateEXIFFile:
                    writer = csv.writer(csvFile)

                myEXIFData = EXIFData()

                for i in range(0, nFrameCount - 1):

                    myEXIFData.extractEXIFData(videoFileList[i])

                    # WRITE THE HEADER ONLY ONCE WHEN THE FIRST FILE IS PROCESSED
                    if bCreateEXIFFile:
                        if i == 0:
                            writer.writerow(myEXIFData.getHeader())
                        writer.writerow(myEXIFData.getEXIF())

                # CLOSE THE EXIF FILE
                if bCreateEXIFFile:
                    csvFile.close()

            # ----------------------------------------------------------------------------------------------------
            # PROCESS THE ORIGINAL IMAGE
            # ----------------------------------------------------------------------------------------------------
            if bCreateVideos:
                videoFolder = self.EditVideoOutputFolder.text()
                out = cv2.VideoWriter(videoFolder + '/' + 'Original.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
                out2 = cv2.VideoWriter(videoFolder + '/' + 'Edges.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

            i = self.spinBoxDailyImage.value()

            while i < nFrameCount and nStop == 0:
                self.spinBoxDailyImage.setValue(i)

                filename = videoFileList[i].fullPathAndFilename

                origImg = GRIMe_Color.loadColorImage(self.radioButtonWebSiteImages, filename)

                # WRITE THE ORIGINAL IMAGE TO A VIDEO FILE
                if bCreateVideos:
                    ii = 0
                    while ii < 10:
                        out.write(origImg)
                        ii = ii + 1

                # NO NEED TO DISPLAY IMAGE IF THE IMAGE TAB IS NOT SELECTED
                if self.tabWidget.currentIndex() == 2:
                    # DISPLAY THE ORIGINAL IMAGE
                    qImg = QImage(origImg.data, origImg.shape[1], origImg.shape[0], QImage.Format_RGB888)
                    pix = QPixmap(qImg)
                    self.labelOriginalImage.setPixmap(
                        pix.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio,
                                   QtCore.Qt.SmoothTransformation))

                # PROCESS THE ORIGINAL IMAGE
                edges, imageFormat = processImageMat(self, origImg)

                # WRITE THE PROCESSED IMAGE TO A VIDEO FILE
                if bCreateVideos:
                    out2.write(edges)

                # NO NEED TO DISPLAY IMAGE IF THE IMAGE TAB IS NOT SELECTED
                if self.tabWidget.currentIndex() == 2:
                    # DISPLAY THE PROCESSED IMAGE
                    qImg = QImage(edges.data, edges.shape[1], edges.shape[0], imageFormat)
                    pix = QPixmap(qImg)
                    # pix = QPixmap.fromImage(QImage(edges.data, edges.cols, edges.rows, QImage.Format_BGR888))
                    self.labelEdgeImage.setPixmap(pix.scaled(self.labelEdgeImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

                # INCREMENT INDEX TO NEXT IMAGE FILE
                i = i + 1

                # CALL PROCESSEVENTS IN ORDER TO UPDATE GUI
                QCoreApplication.processEvents()

            # ----------------------------------------------------------------------------------------------------
            # HOUSEKEEPING (CLOSE FILES, RESET VARIABLES, ETC.)
            # ----------------------------------------------------------------------------------------------------
            nStop = 0

            if bCreateVideos:
                out.release()
                out2.release()

    # --------------------------------------------------
    # --------------------------------------------------
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

    # --------------------------------------------------
    # --------------------------------------------------
    def trainROI(self):
        global currentImage

        if currentImage:
            # CREATE AN ROI OBJECT
            roiObj = GRIMe_roiData()

            # POPULATE ROI OBJECT WITH ROI INFORMATION
            # --------------------------------------------------
            roiName = self.lineEdit_roiName.text()
            if len(roiName) > 0:
                roiObj.setROIName(roiName)
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
                roiObj.setDisplaySize(self.labelOriginalImage.size())
                roiObj.calcROI()
                if self.radioButton_ROIShapeRectangle.isChecked():
                    roiObj.setROIShape(ROIShape.RECTANGLE)
                else:
                    roiObj.setROIShape(ROIShape.ELLIPSE)
            except:
                msgBox = GRIMe_QMessageBox('ROI Error',
                                           'An unexpected error occurred calculating the ROI of the full resolution image!')
                response = msgBox.displayMsgBox()

                return

            # EXTRACT THE ROI FROM THE ORIGINAL IMAGE
            img1 = GRIMe_Utils.convertQImageToMat(currentImage.toImage())
            img1 = cv2.blur(img1, ksize=(5, 5))
            rect = roiObj.getImageROI()
            rgb = img1[rect.y():rect.y() + rect.height(), rect.x():rect.x() + rect.width()]

            # EXTRACT DOMINANT RGB COLORS
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # KMeans EXPECTS THE BYTE ORDER TO BE RGB
            qImg, clusterCenters, hist = myKMeans(self, rgb)
            roiObj.setClusterCenters(clusterCenters, hist)

            # EXTRACT DOMINANT HSV COLORS
            hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
            qhsvImg, hsvClusterCenters, hist = myKMeans(self, hsv)
            roiObj.setHSVClusterCenters(hsvClusterCenters, hist)

            minHSV = hsvClusterCenters[:][0]
            lower_hsv = np.array([min(hsvClusterCenters[:,0]), 0, 20])
            upper_hsv = np.array([max(hsvClusterCenters[:,0]), 150, 255])

            hsvImage = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            colorMask = cv2.inRange(hsvImage, lower_hsv, upper_hsv)

            chickImg = cv2.bitwise_and(img1, img1, mask=colorMask)

            cv2.imshow("Mask", chickImg)

            # SAVE FRAME USING FRAME INDEX TO GENERATE FILENAME
            sum = np.sum(colorMask)


            # APPEND THE ROI TO THE LIST
            self.roiList.append(roiObj)

            # CREATE NEW ROW IN ROI TABLE
            nRow = self.tableWidget_ROIList.rowCount()
            self.tableWidget_ROIList.insertRow(nRow)

            # INSERT ROI NAME INTO TABLE
            nCol = 0
            # self.tableWidget_ROIList.setCellWidget(nRow, nCol, QCheckBox())
            # nCol += 1
            self.tableWidget_ROIList.setItem(nRow, nCol, QTableWidgetItem(self.lineEdit_roiName.text()))

            # INSERT THE DOMINANT COLORS INTO THE TABLE
            self.label = QtWidgets.QLabel()
            pix = QPixmap(qImg)
            self.label.setPixmap(pix)
            nWidth = qImg.size().width()
            nCol += 1
            self.tableWidget_ROIList.setCellWidget(nRow, nCol, self.label)

            processLocalImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def deleteAllROI(self):
        del self.roiList[:]
        self.tableWidget_ROIList.clearContents()
        self.tableWidget_ROIList.setRowCount(0)
        #processSpinBoxImage(self, self.spinBoxDailyImage.value())
        processLocalImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def displayROIs(self):
        processLocalImage(self)
        #refreshImage(self)

    # --------------------------------------------------
    # --------------------------------------------------
    def initProductTable(self):
        headerList = ['Select', 'Product ID', ' min Date ', ' max Date ', 'Start Date', 'End Date', 'Start Time',
                      'End Time']
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
        headerList = ['Site', 'Image Count', ' min Date ', ' max Date ', 'Start Date', 'End Date', 'Start Time', 'End Time']
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
        headerList = ['ROI Name', 'Ref. Image', 'Cur. Image']
        stylesheet = "::section{Background-color:rgb(116,175,80);border-radius:14px;}"
        header = self.tableWidget_ROIList.horizontalHeader()
        self.tableWidget_ROIList.setColumnCount(3)
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
    # IMAGE INPUT FOLDER
    # --------------------------------------------------
    def pushButtonBrowseImageFolderClicked(self):
        prompter = promptlib.Files()
        folder = prompter.dir()

        if os.path.exists(folder):
            self.lineEditHardDriveFolder.setText(folder)
            self.radioButtonHardDriveImages.setChecked(True)
            #self.checkBoxCreateEXIFFile.setEnabled(True)

            # RECURSE AND TRAVERSE FOLDERS FROM ROOT DOWNWARD TO GET A LIST OF
            strStartDate, strEndDate = getLocalFileDates(folder)

            updateProductTableDateRange(self.tableProducts, 0, strStartDate, strEndDate)

            processLocalImage(self)

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
    def radioButtonHardDriveImagesClicked(self):
        #self.checkBoxCreateEXIFFile.setEnabled(True)
        self.lineEditHardDriveFolder.setEnabled(True)
        if len(self.lineEditHardDriveFolder.text()) > 0:
            if os.path.exists(self.lineEditHardDriveFolder.text()):
                self.updateImageSource()

    # --------------------------------------------------
    # --------------------------------------------------
    def radioButtonWebSiteImagesClicked(self):
        #self.checkBoxCreateEXIFFile.setEnabled(False)
        self.lineEditHardDriveFolder.setEnabled(False)

    # --------------------------------------------------
    # --------------------------------------------------
    def pushButtonResetImageIndexClicked(self):
        self.spinBoxDailyImage.setValue(1)
        refreshImage(self)

    def tabWidgetChanged(self, index):
        if full == 1:
            if self.tabWidget_ToolPalette.currentIndex() == 0:
                self.tabWidget_ToolPalette.setCurrentIndex(1)

    def ROIShapeClicked(self):
        if self.radioButton_ROIShapeRectangle.isChecked() == True:
            self.labelOriginalImage.setROIShape(ROIShape.RECTANGLE)
        elif self.radioButton_ROIShapeEllipse.isChecked() == True:
            self.labelOriginalImage.setROIShape(ROIShape.ELLIPSE)

    def pushButtonColorSegmentationClicked(self):
        global currentImage

        if len(self.roiList) == 0:
            strError = "You must train at least one ROI before segmenting the image."
            msgBox = GRIMe_QMessageBox('Color Segmentation Error', strError)
            response = msgBox.displayMsgBox()
            return

        # ----------------------------------------------------------------------------------------------------
        # DISPLAY IMAGE FROM NEON SITE
        # ----------------------------------------------------------------------------------------------------
        if currentImage:
            scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio,
                                                     QtCore.Qt.SmoothTransformation)
            width = scaledCurrentImage.width()
            height = scaledCurrentImage.height()

            # KMeans EXPECTS THE BYTE ORDER TO BE RGB
            img1 = GRIMe_Utils.convertQImageToMat(currentImage.toImage())

            img1 = cv2.blur(img1, ksize=(11, 11))

            rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            # convert image to HSV
            hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

            if len(self.roiList) > 0:
                # DIAGNOSTICS
                if self.checkBoxColorDiagnostics.checkState():
                    GRIMe_Diagnostics.RGB3DPlot(rgb)
                    GRIMe_Diagnostics.plotHSVChannelsGray(hsv)
                    GRIMe_Diagnostics.plotHSVChannelsColor(hsv)

                # segment colors
                rgb1 = GRIMe_Color.segmentColors(rgb, hsv, self.roiList)

            # display the segemented image on the GUI
            qImg = QImage(rgb1.data, rgb1.shape[1], rgb1.shape[0], QImage.Format_BGR888)
            pix = QPixmap(qImg)
            self.labelColorSegmentation.setPixmap(
                pix.scaled(self.labelColorSegmentation.size(), QtCore.Qt.KeepAspectRatio,
                           QtCore.Qt.SmoothTransformation))

    def checkboxNEONSitesClicked(self):
        if self.checkBoxNEONSites.isChecked():
            # GET LIST OF ALL SITES ON NEON
            siteList = readFieldSiteTable()

            '''
            if len(siteList) > 0:
                self.radioButtonHardDriveImages.setChecked(True)
                self.radioButtonHardDriveImages.setDisabled(False)

                # DISABLE AND UNCHECK THE RADIO BUTTON FOR WEB SITE IMAGES IF INFORMATION CANNOT BE ACCESSED FROM THE NEON SITE
                self.radioButtonWebSiteImages.setChecked(False)
                self.radioButtonWebSiteImages.setDisabled(True)
            '''

    def pushButton_USGSDownloadClicked(self):
        fetchUSGSImages()

    def cleanImagesButton(self):
        folder = self.lineEditHardDriveFolder.text()


        cleanImages(folder,\
                    self.doubleSpinBoxBlurThreshhold.value(),\
                    self.spinBoxShiftSize.value(),\
                    self.doubleSpinBoxBrightnessMin.value(),\
                    self.doubleSpinBoxBrightnessMax.value(),\
                    self.checkBoxTriageImagesReport.isChecked(),\
                    self.checkBoxTriageImagesMove.isChecked(), \
                    self.radioButtonWebSiteImages)

    def pushButtonFetchImageListClicked(self):
        fetchLocalImageList(self, self.lineEditHardDriveFolder.text(), False)
                            #start_date, end_date, start_time, end_time)

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

    # --------------------------------------------------
    # --------------------------------------------------
    def __init__(self, parent=None, win=None, session=None):
        super(MainWindow, self).__init__(parent)
        self.mainwin = win
        self.session = session
        self.ui = Ui_MainWindow()

        global dailyImagesList
        dailyImagesList = dailyList([], [])

        self.setWindowTitle("GRIMe-AI: John E. Stranzl Jr.")
        toolbar = QToolBar("GRIMe-AI Toolbar")
        self.addToolBar(toolbar)

        self.setupUi(self)

        self.initUSGS_Table()

        self.initProductTable()
        if full == 1:
            self.initROITable()

        self.listboxNEONSites.itemDoubleClicked.connect(self.itemClicked)
        self.listboxNEONSites.itemClicked.connect(self.itemClicked)
        self.listboxNEONSites.currentItemChanged.connect(self.itemClicked)
        self.listboxSiteProducts.itemClicked.connect(self.productClicked)

        self.pushButtonRetrieveData.clicked.connect(self.pushButtonRetrieveDataClicked)

        #        self.labelLatestImage = GRIMe_QLabel()
        self.labelLatestImage.mouseDoubleClickEvent = labelMouseDoubleClickEvent
        self.labelLatestImage.installEventFilter(self)

        #        self.splitter.splitterMoved.connect(self.splitterMoved)
        # ****************************************************************************************************
        # SCALED BACK FUNCTIONALITY
        # ****************************************************************************************************
        if full == 1:
            self.spinBoxDailyImage.setKeyboardTracking(False)
            self.spinBoxDailyImage.editingFinished.connect(self.spinBoxChanged)
            self.spinBoxDailyImage.valueChanged.connect(self.spinBoxChanged)

            # self.labelOriginalImage.mouseDoubleClickEvent = labelOriginalImageDoubleClickEvent

            self.labelEdgeImage.installEventFilter(self)
            self.labelOriginalImage.installEventFilter(self)

            # EDGE DETECTION METHODS
            self.radioButtonCanny.clicked.connect(self.updateCanny)
            self.spinBoxCannyHighThreshold.valueChanged.connect(self.spinBoxCannyHighThresholdChanged)
            self.spinBoxCannyLowThreshold.valueChanged.connect(self.spinBoxCannyLowThresholdChanged)
            self.spinBoxCannyKernel.valueChanged.connect(self.spinBoxCannyKernelChanged)

            self.radioButtonSobelX.clicked.connect(self.updateSobelX)
            self.radioButtonSobelY.clicked.connect(self.updateSobelY)
            self.radioButtonSobelXY.clicked.connect(self.updateSobelXY)
            self.spinBoxSobelKernel.valueChanged.connect(self.spinBoxSobelKernelChanged)

            self.radioButtonSIFT.clicked.connect(self.updateSIFT)

            self.radioButtonORB.clicked.connect(self.updateORB)
            self.spinBoxOrbMaxFeatures.valueChanged.connect(self.spinBoxOrbMaxFeaturesChanged)

            self.radioButtonLaplacian.clicked.connect(self.updateLaplacian)

            self.spinBoxCannyHighThreshold.setKeyboardTracking(False)
            self.spinBoxCannyLowThreshold.setKeyboardTracking(False)

            # EDGE TAB
            # self.tabWidget.currentChanged.connect(self.initTab3())

            # TRAIN TAB
            self.pushButtonAddROI.clicked.connect(self.trainROI)
            self.pushButton_deleteAllROIs.clicked.connect(self.deleteAllROI)
            self.checkBoxDisplayROIs.clicked.connect(self.displayROIs)

            # VIDEO FUNCTIONS
            self.pushButtonFetchVideoFilenames.clicked.connect(self.MSTFetchVideoFilenames)
            self.pushButtonExtractFrames.clicked.connect(self.MSTExtractFrames)
            self.MSTlineEditFrameFolder.textChanged.connect(self.MSTFrameCountUpdate)
            self.pushButtonPlay.clicked.connect(self.GRIMePlay)
            self.pushButtonStop.clicked.connect(self.MSTStop)
            self.lineEditHardDriveFolder.textChanged.connect(self.MSTVideoFolderChanged)

            #
            self.radioButtonHardDriveImages.clicked.connect(self.updateImageSource)
            self.radioButtonWebSiteImages.clicked.connect(self.updateImageSource)

            #
            self.pushButtonBrowseVideoOutputFolder.clicked.connect(self.pushButtonBrowseVideoOutputFolderClicked)
            self.pushButtonBrowseImageFolder.clicked.connect(self.pushButtonBrowseImageFolderClicked)
            self.pushButtonBrowseSaveImagesOutputFolder.clicked.connect(
                self.pushButtonBrowseSaveImagesOutputFolderClicked)
            self.pushButtonBrowseEXIFOutputFolder.clicked.connect(self.pushButtonBrowseEXIFOutputFolderClicked)

            #
            self.radioButtonHardDriveImages.clicked.connect(self.radioButtonHardDriveImagesClicked)
            self.radioButtonWebSiteImages.clicked.connect(self.radioButtonWebSiteImagesClicked)

            #
            self.pushButtonResetImageIndex.clicked.connect(self.pushButtonResetImageIndexClicked)

            #
            self.radioButton_ROIShapeRectangle.clicked.connect(self.ROIShapeClicked)
            self.radioButton_ROIShapeEllipse.clicked.connect(self.ROIShapeClicked)

            #
            self.pushButton_ColorSegmentation.clicked.connect(self.pushButtonColorSegmentationClicked)

            #
            self.pushButtonCreateVideo.clicked.connect(self.pushButtonCreateVideoClicked)

            #
            self.pushButton_USGSDownload.clicked.connect(self.pushButton_USGSDownloadClicked)

            #
            self.checkBoxNEONSites.clicked.connect(self.checkboxNEONSitesClicked)
            self.pushButtonFetchImageList.clicked.connect(self.pushButtonFetchImageListClicked)
            # INITIALIZE WIDGETS
            ImageWebSiteList = []
            ImageWebSiteList.append("NEON")
            self.listWidgetImageWebSites.addItems(ImageWebSiteList)

            maxRows = self.tableWidget_ROIList.rowCount()
            nCol = 0
            for i in range(0, maxRows):
                self.tableWidget_ROIList.removeRow(0)

            self.pushButton_TrainGood.clicked.connect(self.pushButtonTrainGoodTriggered)
            self.pushButton_TrainBad.clicked.connect(self.pushButtonTrainBadTriggered)

        # SAVE AND RECALL SETTINGS
        self.actionSave_Settings.triggered.connect(self.saveSettings)

        #
        self.tabWidget.currentChanged.connect(self.tabWidgetChanged)

        #
        self.actionClean.triggered.connect(self.cleanImagesButton)
        self.pushButtonTriageImages.clicked.connect(self.cleanImagesButton)

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
        if full == 1:
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

        if full == 1:
            if event.type() == QtCore.QEvent.MouseButtonDblClick and source is self.labelOriginalImage:
                # labelEdgeImageDoubleClickEvent(self)
                # labelMouseDoubleClickEvent(self)
                labelOriginalImageDoubleClickEvent(self)

        return super(MainWindow, self).eventFilter(source, event)


# ======================================================================================================================
#
# ======================================================================================================================
def computeBlurAndBrightness(shiftSize):
    global currentImage

    img1 = GRIMe_Utils.convertQImageToMat(currentImage.toImage())
    grayImage = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

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
def createVideo(self):

    out = None

    rootFolder = self.EditVideoOutputFolder.text()

    filePath = os.path.join(rootFolder, "Videos")
    if not os.path.exists(filePath):
        os.mkdir(filePath)

    # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
    extensions = ('.jpg', '.jpeg', '.png')

    imageCount = GRIMe_Utils.getImageCount(rootFolder, extensions)

    progressBar = QProgressWheel(0, imageCount + 1)
    progressBar.show()

    filenames = []
    imageIndex = 1

    # RECURSE AND TRAVERSE FROM THE SPECIFIED FOLDER DOWN TO DETERMINE THE DATE RANGE FOR THE IMAGES FOUND
    for root, dirs, files in os.walk(rootFolder):
        path = root.split(os.sep)

        for file in files:
            ext = os.path.splitext(file)[-1].lower()

            if ext in extensions:
                progressBar.setValue(imageIndex)
                imageIndex += 1

                imgFile = os.path.join(root, file)

                filenames.append(imgFile)

                img = GRIMe_Color.loadColorImage(self.radioButtonWebSiteImages, imgFile)
                height, width, layers = img.shape

                # WE CAN'T OPEN THE VIDEO STREAM UNTIL WE KNOW THE SIZE OF ONE OF THE IMAGES WHICH ALSO ASSUMES THAT
                # ALL IMAGES ARE OF THE SAME SIZE.
                if out == None:
                    videoFile = 'Original_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.mp4'
                    out = cv2.VideoWriter(filePath + '/' + videoFile, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

                    # processedVideoFile = 'Processed_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.mp4'
                    # out2 = cv2.VideoWriter(filePath + '/' + processedVideoFile, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

                out.write(img)

                # PROCESS THE ORIGINAL IMAGE
                #edges, imageFormat = processImageMat(self, img)

                # WRITE THE PROCESSED IMAGE TO A VIDEO FILE
                #out2.write(edges)

    images = []
    gifFile = 'Original_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.gif'
    for filename in filenames:
        images.append(imageio.imread(filename))

    gifFile = os.path.join(filePath, gifFile)
    imageio.mimsave(gifFile, images, duration=0.25)

    out.release()
    #out2.release()

    del images
    del progressBar

# ======================================================================================================================
#
# ======================================================================================================================
def cleanImages(folder, blurThreshhold, shiftSize, brightnessMin, brightnessMAX, bCreateReport, bMoveImages, bWebSiteImages):
    extensions = ('.jpg', '.jpeg', '.png')

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
    for root, dirs, files in os.walk(folder):
        path = root.split(os.sep)

        # create a subfolder beneath the current root folder if the option to move less than nominal images is selected
        if bMoveImages:
            tempFolder = os.path.join(root, "temp")
            if not os.path.exists(tempFolder):
                os.mkdir(tempFolder)

        for file in files:
            progressBar.setWindowTitle(file)
            progressBar.setValue(imageIndex)
            progressBar.repaint()
            imageIndex += 1

            ext = os.path.splitext(file)[-1].lower()

            if ext in extensions:
                filename = os.path.join(folder, file)
                numpyImage = GRIMe_Color.loadColorImage(bWebSiteImages, filename)
                grayImage = cv2.cvtColor(numpyImage, cv2.COLOR_RGB2GRAY)
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
                    shutil.move(filename, tempFolder)
                    filename = os.path.join(folder, 'MovedImages', file)

                # CREATE A CSV FILE THAT CONTAINS THE FOCUS AND INTENSITY METRICS ALONG WITH HYPERLINKS TO THE IMAGES
                if bCreateReport:
                    strHyperlink = '=HYPERLINK(' + '"' + filename + '"' + ')'
                    strOutputString = '%3.2f,%s,%3.2f,%s,%s\n' % (
                    mean, strFFTFocusMetric, intensity, strIntensity, strHyperlink)
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
def laplace_of_gaussian(gray_img, sigma=1., kappa=0.75, pad=False):
    """
    Applies Laplacian of Gaussians to grayscale image.

    :param gray_img: image to apply LoG to
    :param sigma:    Gauss sigma of Gaussian applied to image, <= 0. for none
    :param kappa:    difference threshold as factor to mean of image values, <= 0 for none
    :param pad:      flag to pad output w/ zero border, keeping input image size
    """
    assert len(gray_img.shape) == 2
    img = cv2.GaussianBlur(gray_img, (0, 0), sigma) if 0. < sigma else gray_img
    img = cv2.Laplacian(img, cv2.CV_64F)
    rows, cols = img.shape[:2]
    # min/max of 3x3-neighbourhoods
    min_map = np.minimum.reduce(list(img[r:rows - 2 + r, c:cols - 2 + c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows - 2 + r, c:cols - 2 + c]
                                     for r in range(3) for c in range(3)))
    # bool matrix for image value positiv (w/out border pixels)
    pos_img = 0 < img[1:rows - 1, 1:cols - 1]
    # bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0
    # bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0
    # sign change at pixel?
    zero_cross = neg_min + pos_max
    # values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.
    # optional thresholding
    if 0. <= kappa:
        thresh = float(np.absolute(img).mean()) * kappa
        values[values < thresh] = 0.
    log_img = values.astype(np.uint8)
    if pad:
        log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)

    return log_img


# ======================================================================================================================
#
# ======================================================================================================================
def getLocalFileDates(filePath):
    # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
    extensions = ('.jpg', '.jpeg', '.png')

    # INITIALIZE VARIABLES TO A KNOWN VALUE
    nStartDate = 99999999
    nEndDate = -1
    strStartDate = ""
    strEndDate = ""

    # RECURSE AND TRAVERSE FROM THE SPECIFIED FOLDER DOWN TO DETERMINE THE DATE RANGE FOR THE IMAGES FOUND
    for root, dirs, files in os.walk(filePath):
        path = root.split(os.sep)
        # print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            ext = os.path.splitext(file)[-1].lower()

            if ext in extensions:

                fileDate, nYear, nMonth, nDay, nTime = GRIMe_Utils.extractDateFromFilename(file)

                # use the date in the filenames to determine the start and end acquisition dates for the images
                if int(str(fileDate)) < nStartDate:
                    nStartDate = int(fileDate)
                    strStartDate = fileDate
                if int(fileDate) > nEndDate:
                    nEndDate = int(fileDate)
                    strEndDate = str(fileDate)

    return strStartDate, strEndDate


# ======================================================================================================================
#
# ======================================================================================================================
def updateProductTableDateRange(productTable, nRow, strStartDate, strEndDate):
    productTable.setItem(0, 1, QTableWidgetItem('Image Source: Hard Drive'))

    # CONFIGURE DATES FOR SPECIFIC PRODUCT
    for m in [2, 4]:
        firstYear = int(strStartDate[0:4])
        firstMonth = int(strStartDate[4:6])
        firstDay = int(strStartDate[6:8])

        date_widget = QtWidgets.QDateEdit(QtCore.QDate(firstYear, firstMonth, firstDay))
        productTable.setCellWidget(nRow, m, date_widget)

        productTable.resizeColumnToContents(m)
        if m == 2:
            date_widget.setDisabled(True)
        else:
            date_widget.setDisabled(False)

    for m in [3, 5]:
        lastYear = int(strEndDate[0:4])
        lastMonth = int(strEndDate[4:6])
        lastDay = int(strEndDate[6:8])

        date_widget = QtWidgets.QDateEdit(QtCore.QDate(lastYear, lastMonth, lastDay))
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
# ======================================================================================================================
# THIS FUNCTION WILL RETRIEVE A SPECIFIC VISIBLE IMAGE AND IT CORRESPONDING INFRARED (IR) IMAGE FROM
# THE PHENOCAM WEBSITE WHENEVER THE INDEX IN THE DAILY IMAGE SPIN BOX CHANGES.
# ======================================================================================================================
def processSpinBoxImage(self, item):
    global dailyImagesList
    global currentImage

    return

    if len(dailyImagesList.getVisibleList()):
        # ----------------------------------------------------------------------------------------------------
        visibleList = dailyImagesList.getVisibleList()
        filename = visibleList[item - 1].fullPathAndFilename
        # imageFile = 'https://phenocam.sr.unh.edu' + filename
        imageFile = filename

        # ----------------------------------------------------------------------------------------------------
        # DOWNLOAD IMAGE FORM NEON SITE
        # ----------------------------------------------------------------------------------------------------
        print("---\n---")
        data = urlopen(imageFile).read()
        currentImage = QPixmap()
        currentImage.loadFromData(data)

        # READ AS NUMPY INTO OPENCV
        # currentImageBGR = url_to_image(imageFile)
        # cv2.imshow("Image", currentImageBGR)

        # ----------------------------------------------------------------------------------------------------
        # DISPLAY IMAGE FROM NEON SITE
        # ----------------------------------------------------------------------------------------------------
        if currentImage:
            scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio,
                                                     QtCore.Qt.SmoothTransformation)
            width = scaledCurrentImage.width()
            height = scaledCurrentImage.height()

            # KMeans EXPECTS THE BYTE ORDER TO BE RGB
            img1 = GRIMe_Utils.convertQImageToMat(currentImage.toImage())
            rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            if self.checkBoxDisplayROIs.isChecked():
                painter = QPainter(currentImage)
                pen = QPen(QtCore.Qt.red, 2, QtCore.Qt.SolidLine)
                painter.setPen(pen)

                nRow = 0
                for obj in self.roiList:
                    if self.checkBoxCalcColors.isChecked():
                        img1 = GRIMe_Utils.convertQImageToMat(currentImage.toImage())
                        rect = obj.getImageROI()
                        rgb1 = img1[rect.y():rect.y() + rect.height(), rect.x():rect.x() + rect.width()]

                        qImg, clusterCenters, hist = myKMeans(self, rgb1)

                        checkColorMatch(painter, clusterCenters, hist, self.roiList)

                        # INSERT COLOR(S) INTO ROI TABLE
                        self.label = QtWidgets.QLabel()
                        pix = QPixmap(qImg)
                        self.label.setPixmap(pix)
                        self.tableWidget_ROIList.setCellWidget(nRow, 2, self.label)

                    # OVERLAY ROI BOUNDARY ON IMAGE
                    if self.checkBoxDisplayROIs.isChecked():
                        if obj.getROIShape() == ROIShape.RECTANGLE:
                            painter.drawRect(obj.getImageROI())
                        elif obj.getROIShape() == ROIShape.ELLIPSE:
                            painter.drawEllipse(obj.getImageROI())

                    # LABEL ROIs
                    if self.checkBoxDisplayROIs.isChecked():
                        font = painter.font()
                        font.setPointSize(font.pointSize() * 2)
                        painter.setFont(font)
                        painter.drawText(obj.getImageROI(), QtCore.Qt.AlignLeft, obj.getROIName())

                    nRow = nRow + 1

            if self.checkBoxDisplayROIs.isChecked():
                painter.end()

            if 0:
                # convert from RGB color-space to YCrCb
                ycrcb_img = cv2.cvtColor(currentImage, cv2.COLOR_RGB2YCrCb)
                # equalize the histogram of the Y channel
                ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
                # convert back to RGB color-space from YCrCb
                equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)
                self.labelOriginalImage.setPixmap(
                    equalized_img.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio,
                                         QtCore.Qt.SmoothTransformation))

            self.labelOriginalImage.setPixmap(
                currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio,
                                    QtCore.Qt.SmoothTransformation))

            pix = processImage(self, currentImage)
            self.labelEdgeImage.setPixmap(pix.scaled(self.labelEdgeImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


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
def processLocalImage(self):
    global currentImage

    frameFolder = self.lineEditHardDriveFolder.text() + '/' + self.MSTlineEditFrameFolder.text()

    if os.path.exists(frameFolder):
        # videoFilePath = Path(frameFolder)
        ## JES videoFileList = [str(pp) for pp in videoFilePath.glob("**/*.jpg")]
        # videoFileList = [str(pp) for pp in videoFilePath.glob("*.jpg")]

        global dailyImagesList
        videoFileList = dailyImagesList.getVisibleList()

        if len(videoFileList) > 0:
            nImageIndex = self.spinBoxDailyImage.value()
            if nImageIndex > gFrameCount:
                nImageIndex = gFrameCount

            inputFrame = videoFileList[nImageIndex - 1].fullPathAndFilename  # zero based index

            if os.path.isfile(inputFrame):
                imageOnWebsite = False
                numpyImage = GRIMe_Color.loadColorImage(imageOnWebsite, inputFrame)

                tempCurrentImage = QImage(numpyImage, numpyImage.shape[1], numpyImage.shape[0], QImage.Format_RGB888)
                currentImage = QPixmap(tempCurrentImage)

    # ----------------------------------------------------------------------------------------------------
    # DISPLAY IMAGE FROM NEON SITE
    # ----------------------------------------------------------------------------------------------------
    if currentImage:
        scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        width = scaledCurrentImage.width()
        height = scaledCurrentImage.height()

        currentImageRescaled = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        if self.checkBoxDisplayROIs.isChecked():
            painter = QPainter(currentImageRescaled)

        progressBar = QProgressWheel()
        progressBar.setRange(0, len(self.roiList) + 1)
        progressBar.show()

        nRow = 0
        for obj in self.roiList:
            progressBar.setValue(nRow+1)
            progressBar.repaint()

            try:
                if self.checkBoxCalcColors.isChecked():
                    # EXTRACT ROI FOR WHICH COLOR CLUSTERING IS TO BE PERFORMED
                    img1 = GRIMe_Utils.convertQImageToMat(currentImage.toImage())
                    rect = obj.getImageROI()
                    rgb1 = img1[rect.y():rect.y() + rect.height(), rect.x():rect.x() + rect.width()]

                    # KMeans EXPECTS THE BYTE ORDER TO BE RGB
                    rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)

                    # PERFORM COLOR CLUSTERING
                    qImg, clusterCenters, hist = myKMeans(self, rgb1)

                    checkColorMatch(painter, clusterCenters, hist, self.roiList)

                    # INSERT COLOR(S) INTO ROI TABLE
                    self.label = QtWidgets.QLabel()
                    pix = QPixmap(qImg)
                    self.label.setPixmap(pix)
                    self.tableWidget_ROIList.setCellWidget(nRow, 2, self.label)

                    # OVERLAY ROI BOUNDARY ON IMAGE
                    if self.checkBoxDisplayROIs.isChecked():
                        pen = QPen(QtCore.Qt.red, 1, QtCore.Qt.SolidLine)
                        painter.setPen(pen)

                        if obj.getROIShape() == ROIShape.RECTANGLE:
                            painter.drawRect(obj.getDisplayROI())
                        elif obj.getROIShape() == ROIShape.ELLIPSE:
                            painter.drawEllipse(obj.getDisplayROI())

                        font = painter.font()
                        font.setPointSize(8)
                        painter.setPen(QPen(QtCore.Qt.red, 1, QtCore.Qt.SolidLine))
                        painter.setFont(font)
                        painter.drawText(obj.getDisplayROI().x(), obj.getDisplayROI().y()-16, 50, 16, QtCore.Qt.AlignLeft, obj.getROIName())

                    nRow = nRow + 1
            except:
                nErrorCode = -1

        if self.checkBoxDisplayROIs.isChecked():
            painter.end()

        # close and delete the progress bar
        progressBar.close()
        del progressBar

        self.labelOriginalImage.setPixmap(currentImageRescaled)

        pix = processImage(self, currentImage)
        self.labelEdgeImage.setPixmap(pix.scaled(self.labelEdgeImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        del currentImageRescaled
        del scaledCurrentImage

    # CALL PROCESSEVENTS IN ORDER TO UPDATE GUI
    QCoreApplication.processEvents()


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
# THIS FUNCTION WILL CALL THE FUNCTION THAT PROCESSES THE IMAGE BASED UPON THE SETTINGS SELECTED BY THE
# END-USER AND THEN UPDATE THE GUI TO DISPLAY THE PROCESSED IMAGE.
# ======================================================================================================================
def refreshImage(self):
    global currentImage

    '''// PROCESS THE ORIGINAL IMAGE //'''
    pix = processImage(self, currentImage)

    '''// DISPLAY PROCESSED ORIGINAL IMAGE //'''
    if not pix == []:
        self.labelEdgeImage.setPixmap(pix.scaled(self.labelEdgeImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        img = GRIMe_Utils.convertQImageToMat(currentImage.toImage())

    # CALL PROCESSEVENTS IN ORDER TO UPDATE GUI
    QCoreApplication.processEvents()


# ======================================================================================================================
# THIS FUNCTION WILL PROCESS THE CURRENT IMAGE BASED UPON THE SETTINGS SELECTED BY THE END-USER.
# THE IMAGE STORAGE TYPE IS QImage
# ======================================================================================================================
def processImage(self, myImage):
    pix = []

    if not myImage == []:
        # CONVERT IMAGE FROM QImage FORMAT TO Mat FORMAT
        # BYTE ORDER IS R, G, B
        img1 = GRIMe_Utils.convertQImageToMat(myImage.toImage())

        if self.checkboxKMeans.isChecked():
            qImg, clusterCenters, hist = myKMeans(self, img1)

        # CONVERT COLOR IMAGE TO GRAY SCALE
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # REMOVE NOISE FROM THE IMAGE
        if len(gray) != 0:
            grayEQ = cv2.equalizeHist(gray)
            grayBlur = cv2.GaussianBlur(grayEQ, (15, 15), 0)
            cv2.erode(grayBlur, (7,7), gray)

        # EDGE DETECTION METHODS
        if len(gray) != 0:
            if self.radioButtonCanny.isChecked():
                highThreshold = self.spinBoxCannyHighThreshold.value()
                lowThreshold = self.spinBoxCannyLowThreshold.value()
                kernelSize = self.spinBoxCannyKernel.value()

                temp = gray
                #temp = gray[0:myImage.size().height(), 0:myImage.size().width()]

                # Find Canny edges
                edges = cv2.Canny(temp, highThreshold, lowThreshold, kernelSize)

                # Find Contours
                # Use a copy of the image e.g. edged.copy() since findContours alters the image
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # Draw all contours
                # -1 = draw all contours
                cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)
                #cv2.drawContours(temp, contours, -1, 255, 6)
                # JES ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^S


                # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
                # for line in lines:
                # x1, y1, x2, y2 = line[0]
                # cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.imwrite('C:/Users/Astrid Haugen/Documents/houghlines5.jpg', img1)
                #q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8)
                #q_img = QImage(temp.data, temp.shape[1], temp.shape[0], QImage.Format_Grayscale8)
                q_img = QImage(img1.data, img1.shape[1], img1.shape[0], QImage.Format_RGB888)

                '''
                del temp
                del img1
                del gray
                del grayEQ
                del grayBlur
                del contours
                del hierarchy
                '''
            elif self.radioButtonSIFT.isChecked():
                edges = calcSIFT(img1, gray)
                q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_RGB888)
            elif self.radioButtonORB.isChecked():
                value = self.spinBoxOrbMaxFeatures.value()

                # convert from RGB color-space to YCrCb
                ycrcb_img = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)

                # equalize the histogram of the Y channel
                ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

                gray = cv2.cvtColor(ycrcb_img, cv2.COLOR_RGB2GRAY)

                lapImg = laplace_of_gaussian(gray)
                gray = cv2.equalizeHist(lapImg)
                cv2.dilate(gray, (5, 5), gray)
                cv2.erode(gray, (5, 5), gray)

                #gray = Image.fromarray(gray)

                gray = cv2.cvtColor(ycrcb_img, cv2.COLOR_RGB2GRAY)
                edges = calcOrb(gray, value)
                q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_RGB888)

                del gray
                del grayEQ
                del grayBlur
                #del lapImg
                del edges
            elif self.radioButtonLaplacian.isChecked():
                # convert from RGB color-space to YCrCb
                ycrcb_img = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)

                # equalize the histogram of the Y channel
                ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

                gray = cv2.cvtColor(ycrcb_img, cv2.COLOR_RGB2GRAY)

                #gray = cv2.equalizeHist(gray)
                #gray = cv2.GaussianBlur(gray, (7, 7), 0)

                # convert back to RGB color-space from YCrCb
                #equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)
                lapImg = laplace_of_gaussian(gray)
                gray = cv2.equalizeHist(lapImg)
                cv2.dilate(gray, (5, 5), gray)
                cv2.erode(gray, (5, 5), gray)

                gray = Image.fromarray(gray)
                q_img = ImageQt.toqimage(gray)
            elif self.radioButtonSobelX.isChecked() or self.radioButtonSobelY.isChecked() or self.radioButtonSobelXY.isChecked():
                mySobel = sobelData.sobelData()
                sobelKernelSize = self.spinBoxSobelKernel.value()
                mySobel.setSobelX(cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=sobelKernelSize))
                mySobel.setSobelY(cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=sobelKernelSize))
                mySobel.setSobelXY(cv2.Sobel(img1, cv2.CV_64F, 1, 1, ksize=sobelKernelSize))
                if self.radioButtonSobelX.isChecked():
                    edges = (mySobel.getSobelX() * 255 / mySobel.getSobelX().max()).astype(np.uint8)
                    q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8)
                elif self.radioButtonSobelY.isChecked():
                    edges = (mySobel.getSobelY() * 255 / mySobel.getSobelY().max()).astype(np.uint8)
                    edges = (mySobel.getSobelY() * 255 / mySobel.getSobelY().max()).astype(np.uint8)
                    q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8)
                elif self.radioButtonSobelXY.isChecked():
                    edges = (mySobel.getSobelXY() * 255 / mySobel.getSobelXY().max()).astype(np.uint8)
                    q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8)

            if self.radioButtonLaplacian.isChecked():
                pix = QPixmap.fromImage(q_img)
            else:
                pix = QPixmap(q_img)

    return pix


# ======================================================================================================================
# THIS FUNCTION WILL PROCESS THE CURRENT IMAGE BASED UPON THE SETTINGS SELECTED BY THE END-USER.
# THE IMAGE STORAGE TYPE IS mat
# ======================================================================================================================
def processImageMat(self, myImage):
    edges = []
    img2 = []
    imageFormat = 0

    if not myImage == []:
        img1 = myImage

        if self.checkboxKMeans.isChecked():
            qImg, clusterCenters, hist = myKMeans(self, img1)

            checkColorMatch(clusterCenters, hist, self.roiList)

        # CONVERT COLOR IMAGE TO GRAY SCALE
        gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        # REMOVE NOISE FROM THE IMAGE
        if len(gray) != 0:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # EDGE DETECTION METHODS
        if len(gray) != 0:
            if self.radioButtonCanny.isChecked():
                highThreshold = self.spinBoxCannyHighThreshold.value()
                lowThreshold = self.spinBoxCannyLowThreshold.value()
                kernelSize = self.spinBoxCannyKernel.value()
                edges = cv2.Canny(img1, highThreshold, lowThreshold, kernelSize)
                # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
                # for line in lines:
                # x1, y1, x2, y2 = line[0]
                # cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.imwrite('C:/Users/Astrid Haugen/Documents/houghlines5.jpg', img1)

            elif self.radioButtonSIFT.isChecked():
                edges = calcSIFT(img1, gray)
            elif self.radioButtonORB.isChecked():
                value = self.spinBoxOrbMaxFeatures.value()
                edges = calcOrb(img1, value)
                # edges = cv2.cvtColor(edges, cv2.COLOR_RGBA2RGB)
            elif self.radioButtonLaplacian.isChecked():
                edges = cv2.Laplacian(img1, cv2.CV_64F)
            elif self.radioButtonSobelX.isChecked() or self.radioButtonSobelY.isChecked() or self.radioButtonSobelXY.isChecked():
                mySobel = sobelData()
                sobelKernelSize = self.spinBoxSobelKernel.value()
                mySobel.setSobelX(cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=sobelKernelSize))
                mySobel.setSobelY(cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=sobelKernelSize))
                mySobel.setSobelXY(cv2.Sobel(img1, cv2.CV_64F, 1, 1, ksize=sobelKernelSize))
                if self.radioButtonSobelX.isChecked():
                    edges = (mySobel.getSobelX() * 255 / mySobel.getSobelX().max()).astype(np.uint8)
                elif self.radioButtonSobelY.isChecked():
                    edges = (mySobel.getSobelY() * 255 / mySobel.getSobelY().max()).astype(np.uint8)
                elif self.radioButtonSobelXY.isChecked():
                    edges = (mySobel.getSobelXY() * 255 / mySobel.getSobelXY().max()).astype(np.uint8)

            if self.radioButtonSIFT.isChecked():
                imageFormat = QImage.Format_RGB888
            elif self.radioButtonORB.isChecked():
                imageFormat = QImage.Format_RGB888
            else:
                imageFormat = QImage.Format_Grayscale8

    return edges, imageFormat


# ======================================================================================================================
#
# ======================================================================================================================
def myKMeans(self, img1):
    # --------------------------------------------------------------------------------
    # reshape the image to be a list of pixels
    # --------------------------------------------------------------------------------
    scale_percent = 100  # percent of original size
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)

    # --------------------------------------------------------------------------------
    # resize image
    # --------------------------------------------------------------------------------
    resized = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)

    # --------------------------------------------------------------------------------
    # CHOOSE BETWEEN RESIZED (DOWNSAMPLED) IMAGE OR THE FULL SCALE IMAGE
    # --------------------------------------------------------------------------------
    if 0:
        image = resized.reshape((resized.shape[0] * resized.shape[1], 3))
    else:
        image = img1.reshape((img1.shape[0] * img1.shape[1], 3))

    # --------------------------------------------------------------------------------
    # COLOR CLUSTERING
    # --------------------------------------------------------------------------------
    nClusters = self.spinBoxColorClusters.value()
    # kmeans = KMeans(n_clusters=nClusters, n_jobs=-1)
    kmeans = KMeans(n_clusters=nClusters)
    s = kmeans.fit(image)

    # labels = kmeans.labels_
    # print(labels)
    # labels = list(labels)

    # build a histogram of clusters and then create a figure representing the number of pixels labeled to each color
    hist = centroid_histogram(kmeans)

    for (percent, color) in zip(hist, kmeans.cluster_centers_):
        mask = cv2.inRange(resized, color, color)
        result = cv2.bitwise_and(resized, resized, mask=mask)

    bar = plot_colors(hist, kmeans.cluster_centers_)

    qImg = QImage(bar.data, bar.shape[1], bar.shape[0], QImage.Format_RGB888)
    pix = QPixmap(qImg)

    # pix = QPixmap.fromImage(QImage(edges.data, edges.cols, edges.rows, QImage.Format_BGR888))
    self.labelDominantColors.setPixmap(pix.scaled(self.labelEdgeImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    del s
    del pix
    del mask
    del bar

    return qImg, kmeans.cluster_centers_, hist


# ======================================================================================================================
#
# ======================================================================================================================
def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency of each of the colors
    # bar = np.zeros((50, 300, 3), dtype="uint8")
    bar = np.zeros((50, 100, 3), dtype="uint8")

    startX = 0

    if 0:
        # loop over the percentage of each cluster and the color of each cluster
        colorFile = 'R:/HSV.csv'
        if not os.path.isfile(colorFile):
            csvFile = open(colorFile, 'a', newline='')
            csvFile.write('Precent, Red, Green, Blue, "", Hue, Saturation, Value\n')
        else:
            csvFile = open(colorFile, 'a', newline='')

        csvFile.write('Image:\n')

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 100)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)

        if 0:
            h, s, v = colorsys.rgb_to_hsv(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

            txt = '{:.3f}, {:.3f}, {:.3f}, {:.3f}, "" ,{:.3f}, {:.3f}, {:.3f}'.format(percent, color[0], color[1],
                                                                                      color[2],
                                                                                      h * 360, s * 100, v * 100) + '\n'

            csvFile.write(txt)

        startX = endX

        if 0:
            csvFile.close()

    # return the bar chart
    return bar


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
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


# ======================================================================================================================
# THIS FUNCTION WILL USE THE SIFT FEATURE DETECTION ALGORITHM TO FIND FEATURES IN THE IMAGE THAT IS PASSED TO THIS FUNCTION.
# ======================================================================================================================
def calcSIFT(image, gray):
    # REMOVE NOISE FROM THE IMAGE
    img1 = cv2.GaussianBlur(gray, (3, 3), 0)

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img1, None)
    edges = cv2.drawKeypoints(gray, kp, img1)

    return edges


# ======================================================================================================================
# ThIS FUNCTION WILL USE THE ORB FEATURE DETECTION ALGORITHM TO FIND FEATURES IN THE IMAGE THAT IS
# PASSED TO THIS FUNCTION.
# ======================================================================================================================
def calcOrb(image, nMaxFeatures):

    if len(image.shape) == 3:
       imageBW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        imageBW = image

    orb = cv2.ORB_create()

    orb.setEdgeThreshold(50)
    orb.setNLevels(10)
    orb.setPatchSize(30)
    orb.setMaxFeatures(nMaxFeatures)

    kp = orb.detect(imageBW, None)
    kp, des = orb.compute(imageBW, kp)

    edges = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)

    return edges


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

    nErrorCode = check_url_validity(url)

    # IF AT LEAST ONE FIELD SITE TABLE IS FOUND ON THE NEON SITE...
    if nErrorCode == 0:
        csv_links = fetchFieldSiteTableURL(url)

        # download all CSV files
        filename_with_path = downloadFieldSiteTableFiles(csv_links)

        siteList = GRIMe_Utils.parseCSV(filename_with_path)
    # ELSE IF NO FIELD SITE TABLES ARE FOUND, RETURN AN EMPTY LIST
    elif nErrorCode == -1:
        siteList = []

    return siteList


# ======================================================================================================================
#
# ======================================================================================================================
def check_url_validity(my_url):
    nErrorCode = -1
    nRetryCount = 3

    responses = {
        100: ('Continue', 'Request received, please continue'),
        101: ('Switching Protocols',
              'Switching to new protocol; obey Upgrade header'),

        200: ('OK', 'Request fulfilled, document follows'),
        201: ('Created', 'Document created, URL follows'),
        202: ('Accepted',
              'Request accepted, processing continues off-line'),
        203: ('Non-Authoritative Information', 'Request fulfilled from cache'),
        204: ('No Content', 'Request fulfilled, nothing follows'),
        205: ('Reset Content', 'Clear input form for further input.'),
        206: ('Partial Content', 'Partial content follows.'),

        300: ('Multiple Choices',
              'Object has several resources -- see URI list'),
        301: ('Moved Permanently', 'Object moved permanently -- see URI list'),
        302: ('Found', 'Object moved temporarily -- see URI list'),
        303: ('See Other', 'Object moved -- see Method and URL list'),
        304: ('Not Modified',
              'Document has not changed since given time'),
        305: ('Use Proxy',
              'You must use proxy specified in Location to access this '
              'resource.'),
        307: ('Temporary Redirect',
              'Object moved temporarily -- see URI list'),

        400: ('Bad Request',
              'Bad request syntax or unsupported method'),
        401: ('Unauthorized',
              'No permission -- see authorization schemes'),
        402: ('Payment Required',
              'No payment -- see charging schemes'),
        403: ('Forbidden',
              'Request forbidden -- authorization will not help'),
        404: ('Not Found', 'Nothing matches the given URI'),
        405: ('Method Not Allowed',
              'Specified method is invalid for this server.'),
        406: ('Not Acceptable', 'URI not available in preferred format.'),
        407: ('Proxy Authentication Required', 'You must authenticate with '
                                               'this proxy before proceeding.'),
        408: ('Request Timeout', 'Request timed out; try again later.'),
        409: ('Conflict', 'Request conflict.'),
        410: ('Gone',
              'URI no longer exists and has been permanently removed.'),
        411: ('Length Required', 'Client must specify Content-Length.'),
        412: ('Precondition Failed', 'Precondition in headers is false.'),
        413: ('Request Entity Too Large', 'Entity is too large.'),
        414: ('Request-URI Too Long', 'URI is too long.'),
        415: ('Unsupported Media Type', 'Entity body in unsupported format.'),
        416: ('Requested Range Not Satisfiable',
              'Cannot satisfy request range.'),
        417: ('Expectation Failed',
              'Expect condition could not be satisfied.'),

        500: ('Internal Server Error', 'Server got itself in trouble'),
        501: ('Not Implemented',
              'Server does not support this operation'),
        502: ('Bad Gateway', 'Invalid responses from another server/proxy.'),
        503: ('Service Unavailable',
              'The server cannot process the request due to a high load'),
        504: ('Gateway Timeout',
              'The gateway server did not receive a timely response'),
        505: ('HTTP Version Not Supported', 'Cannot fulfill request.'),
    }

    while nErrorCode == -1 and nRetryCount > 0:
        req = urllib.request.Request(my_url)

        try:
            response = urlopen(req)
            nErrorCode = 0
        except urllib.error.HTTPError as e:
            strError = 'The server couldn\'t fulfill the request.\n' + 'Error code: ' + e.code
            nErrorCode = -1
            nRetryCount = nRetryCount - 1
        except urllib.error.URLError as e:
            if nRetryCount == 1:
                strError = 'We failed to reach a server.\n' + 'Reason: [' + str(e.reason.args[0]) + '] ' + \
                           e.reason.args[1]
                msgBox = GRIMe_QMessageBox('NEON SITE Info URL Error', strError)
                response = msgBox.displayMsgBox()
            nErrorCode = -1
            nRetryCount = nRetryCount - 1

    return nErrorCode


# ======================================================================================================================
# THIS FUNCTION FETCHES THE FIELD SITE TABLE FROM THE NEON SITE AND PARSES ITS INFORMATION.
# ======================================================================================================================
def fetchFieldSiteTableURL(my_url):
    csv_links = []

    # r = requests.get(my_url)
    #ssl._create_default_https_context = ssl._create_unverified_context
    r = urlopen(my_url)
    # context = ssl._create_unverified_context()
    # r = urlopen(my_url, context=context)

    if 1:
        # if r.status_code == 200:
        # create beautiful-soup object
        # soup = BeautifulSoup(r.content, 'html5lib')
        soup = BeautifulSoup(r, 'html5lib')

        # FIND ALL CSV LINKS ON THE WEB-PAGE. CURRENTLY THERE IS ONLY ONE. HOWEVER, THERE COULD BE MULTIPLES IN THE FUTURE
        links = soup.findAll("a", href=lambda href: href and "csv" in href)

        # CREATE COMPLETE URL FOR LINK TO CSV FILE. ASSUME THERE IS ONLY ONE FOR NOW BUT LOOP FOR FUTURE USE-CASES
        for link in links:
            #csvLink = root_url + link['href']
            csvLink = link['href']

        csv_links = csvLink

    return csv_links


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

    siteInfo = fetchSiteInfoFromNEON()

    global DOMAINCODE
    DOMAINCODE = siteInfo['data']['domainCode']

    global SITENAME
    SITENAME = siteInfo['data']['siteName']
    if full == 1:
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

    self.labelLatestImage.setPixmap(
        latestImage.scaled(self.labelLatestImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


# ======================================================================================================================
# THIS FUNCTION WILL UPDATE THE PRODUCT TABLE IN THE GUI WITH THE PRODUCTS THAT ARE AVAILABLE FOR A SPECIFIC SITE.
# ======================================================================================================================
def formatProductTableCells(self):
    maxRows = 1

    # JES: MUST MAKE CODE DYNAMIC TO ONLY DELETE UNSELECTED ITEMS
    for i in range(maxRows):
        self.tableProducts.removeRow(0)

    for i in range(maxRows):
        m = 0
        myCheckBox = QCheckBox()
        myCheckBox.setChecked(True)
        self.tableProducts.setCellWidget(i, m, myCheckBox)

        m += 1
        self.tableProducts.setItem(i, m, QTableWidgetItem(''))

        # CONFIGURE DATES FOR SPECIFIC PRODUCT
        m += 1
        date_widget = QtWidgets.QDateEdit()
        date_widget.setDisabled(True)
        self.tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit()
        date_widget.setDisabled(True)
        self.tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(2021, 12, 1))
        self.tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(2022, 1, 1))
        self.tableProducts.setCellWidget(i, m, date_widget)

        # --------------------
        # --------------------
        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        self.tableProducts.setCellWidget(i, m, dateTime)

        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        self.tableProducts.setCellWidget(i, m, dateTime)


# ======================================================================================================================
# THIS FUNCTION WILL UPDATE THE PRODUCT TABLE IN THE GUI WITH THE PRODUCTS THAT ARE AVAILABLE FOR A SPECIFIC SITE.
# ======================================================================================================================
def updateSelectedProductsTable(self, item):
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
        m += 1
        date_widget = QtWidgets.QDateEdit(QtCore.QDate(firstYear, firstMonth, 1))
        date_widget.setDisabled(True)
        self.tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit(QtCore.QDate(lastYear, lastMonth, 1))
        date_widget.setDisabled(True)
        self.tableProducts.setCellWidget(i, m, date_widget)

        # CONFIGURE USER SELECTABLE DATE RANGES FOR A SPECIFIC PRODUCT
        minDate = QtCore.QDate(firstYear, firstMonth, 1)
        maxDate = QtCore.QDate(lastYear, lastMonth, 1)

        # --------------------
        # --------------------
        # JES - OVERRIDE MAX DATE BECAUSE THE NEON CSV FILE IS NOT KEPT UP TO DATE
        today = datetime.datetime.today()
        maxDate = QtCore.QDate(today.year, today.month, 1)

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        # JES - FOR EASE OF USE FOR NOW, LET'S SET THE MIN SELECTION DATE TO THE MAX VALUE SO WE DON'T HAVE TO SCROLL
        # THROUGH YEARS OF DATA TO GET TO THE DATE THAT WE WANT
        date_widget.setDate(QtCore.QDate(lastYear, lastMonth, 1))
        # date_widget.setDate(QtCore.QDate(firstYear, firstMonth, 1))
        date_widget.setDateRange(minDate, maxDate)
        self.tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(lastYear, lastMonth, 1))
        date_widget.setDateRange(minDate, maxDate)
        self.tableProducts.setCellWidget(i, m, date_widget)

        # --------------------
        # --------------------
        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        self.tableProducts.setCellWidget(i, m, dateTime)

        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        self.tableProducts.setCellWidget(i, m, dateTime)

        self.tableProducts.resizeColumnsToContents()


# ======================================================================================================================
#
# ======================================================================================================================
def fetchLocalImageList(self, filePath, bCreateEXIFFile, start_date=datetime.date(1970, 1, 1), end_date=datetime.date(2099, 12, 31), start_time='000000', end_time='000000'):
    global gWebImageCount
    global dailyImagesList

    # CLEAR THE PREVIOUSLY DOWNLOADED IMAGE LIST, IF ANY
    dailyImagesList.clear()

    # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
    extensions = ('.jpg', '.jpeg', '.png')

    # INITIALIZE VARIABLES TO A KNOWN VALUE
    nStartDate = 99999999
    nEndDate = -1
    strStartDate = ""
    strEndDate = ""
    List = []
    bSaveImages = False

    if self.checkBoxSaveImages.isChecked():
        bSaveImages = True
        imageOutputFolder = self.EditSaveImagesOutputFolder.text()

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
    for root, dirs, files in os.walk(filePath):

        # folder path
        path = root.split(os.sep)

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
                fileDate, nYear, nMonth, nDay, strTime = GRIMe_Utils.extractDateFromFilename(file)

                fileDateTime = datetime.date(nYear, nMonth, nDay)
                if fileDateTime >= start_date and fileDateTime <= end_date:
                    fullPathAndFilename = root + '\\' + file

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

                    #if ((start_time == datetime.time(0, 0, 0)) and (end_time == datetime.time(0, 0, 0))) or \
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
    if full == 1:
        self.labelImageCountNumber.setText(str(gWebImageCount))
        self.spinBoxDailyImage.setMinimum(1)
        self.spinBoxDailyImage.setMaximum(gWebImageCount)
        self.spinBoxDailyImage.setValue(1)
        self.labelImageCountNumber.setText(str(gWebImageCount))
    else:
        dailyURLvisible = []

    # clean-up before exiting function
    # 1. close and delete the progress bar
    # 2. close the EXIF log file, if opened
    progressBar.close()
    del progressBar

    if bCreateEXIFFile:
        csvFile.close()

    processLocalImage(self)


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

            strYear = str(start_date.year)
            strMonth = str(start_date.month).zfill(2)
            strDay = str(start_date.day).zfill(2)

            # ----------
            dailyURLvisible = 'https://phenocam.sr.unh.edu/webcam/browse/NEON.D10.ARIK.DP1.20002' + '/' + strYear + '/' + strMonth.zfill(
                2) + '/' + strDay.zfill(2)

            # ----------
            dailyURLvisible = dailyURLvisible.replace('ARIK', SITECODE)
            dailyURLvisible = dailyURLvisible.replace('D10', DOMAINCODE)

            phenoCam = GRIMe_PhenoCam()
            tmpList = phenoCam.getVisibleImages(dailyURLvisible, start_time, end_time)

            dailyImagesList.setVisibleList(tmpList.getVisibleList())

            start_date += datetime.timedelta(days=1)

        gWebImageCount = len(dailyImagesList.getVisibleList())

        # INIT SPINBOX CONTROLS BASED UPON NUMBER OF IMAGES AVAILABLE
        if full == 1:
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
    if full == 1:
        isChecked = self.checkBoxSaveImages.isChecked()
    else:
        isChecked = True

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

            if full == 1:
                isChecked = self.checkBoxSaveImages.isChecked()
            else:
                isChecked = True

            if isChecked == True:
                if full == 1:
                    downloadsFilePath = self.EditSaveImagesOutputFolder.text()
                else:
                    downloadsFilePath = os.path.join(fetchDownloadsFolderPath(), 'Images')
                completeFilename = os.path.join(downloadsFilePath, filename)
                urllib.request.urlretrieve(image.fullPathAndFilename, completeFilename)

        # DISABLE THE CHECK BOX SO THAT ONCE THE IMAGES ARE DOWNLOADED, IMAGES ARE NOT CONTINUED TO BE SAVED UNBEKNOWNST
        # TO THE USER
        if full == 1:
            self.checkBoxSaveImages.setChecked(False)

        # clean-up before exiting function
        # 1. close and delete the progress bar
        # 2. no other clean-up tasks
        progressBar.close()
        del progressBar

        if full == 1:
            strMessage = 'Data download is complete!' + '\n\n' + 'If you selected "Save Images", it will be deselected to avoid unnecessary re-downloading of the same data.' + '\n\n' + 'If you want to re-download the same data or additional data, you will have to reselect "Save Images."'
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
    downloadsFilePath = os.path.expanduser('~')
    downloadsFilePath = os.path.join(downloadsFilePath, 'Documents')
    downloadsFilePath = os.path.join(downloadsFilePath, 'GRIMe-AI')
    if not os.path.exists(downloadsFilePath):
        os.mkdir(downloadsFilePath)
    downloadsFilePath = os.path.join(downloadsFilePath, 'Downloads')
    if not os.path.exists(downloadsFilePath):
        os.mkdir(downloadsFilePath)

    # --------------------------------------------------------------------------------
    # FIND IMAGE PRODUCT (20002) ROW TO GET DATE RANGE
    # --------------------------------------------------------------------------------
    #if self.checkBoxNEONSites.isChecked() and self.radioButtonWebSiteImages.isChecked():
    if self.checkBoxNEONSites.isChecked():
            rowRange = range(self.tableProducts.rowCount())
    else:
        rowRange = range(1)

    for nRow in rowRange:
        # --------------------------------------------------------------------------------
        # FETCH THE DATE RANGE THE USER SELECTED FOR THE PARTICULAR DATA
        # --------------------------------------------------------------------------------
        GRIMe_ProductTableObj = GRIMe_ProductTable()
        GRIMe_ProductTableObj.fetchTableDates(self.tableProducts, nRow)

        start_date = GRIMe_ProductTableObj.getStartDate()
        strStartDate = GRIMe_ProductTableObj.getStartDate()

        end_date = GRIMe_ProductTableObj.getEndDate()
        strEndDate = GRIMe_ProductTableObj.getEndDate()

        start_time = GRIMe_ProductTableObj.getStartTime()
        end_time = GRIMe_ProductTableObj.getEndTime()

        delta = GRIMe_ProductTableObj.getDelta()

        #if self.checkBoxNEONSites.isChecked() and self.radioButtonWebSiteImages.isChecked():
        if self.checkBoxNEONSites.isChecked():
            # EXTRACT THE PRODUCT ID
            ProductIDCell = self.tableProducts.item(nRow, 1).text()
            nProductID = int(ProductIDCell.split('.')[1])
        #else:
        #    nProductID = -999

        if nProductID < 0 and nRow == 0:
            if self.radioButtonHardDriveImages.isChecked():
                fetchLocalImageList(self, self.lineEditHardDriveFolder.text(), self.checkBoxCreateEXIFFile.isChecked(),
                                    start_date, end_date, start_time, end_time)
        elif nProductID > 0:
            # ----------------------------------------------------------------------------------------------------
            # FETCH THE DATA FOR THE PARTICULAR PRODUCT FOR THE SPECIFIED DATE RANGE AND PROCESS THE DATA ACCORDINGLY
            # ----------------------------------------------------------------------------------------------------
            if nProductID == 20002:

                # IF THE USER HAS NOW SELECTED THE 'Save Images' CHECKBOX, VERIFY WHETHER OR NOT THE USER
                # WANTS TO SAVE THE IMAGES LOCALLY
                if full == 1:
                    if self.checkBoxSaveImages.isChecked() == False:
                        strMessage = 'Do you also want to save the images locally to your computer?'
                        msgBox = GRIMe_QMessageBox('Download Image Files', strMessage, QMessageBox.Yes | QMessageBox.No)
                        response = msgBox.displayMsgBox()

                        if response == QMessageBox.Yes:
                            self.checkBoxSaveImages.setChecked(True)

                #if self.radioButtonWebSiteImages.isChecked():
                DP1_20002_fetchImageList(self, nRow, start_date, end_date, start_time, end_time)
                #else:
                    #fetchLocalImageList(self, self.lineEditHardDriveFolder.text(), nRow, start_date, end_date, start_time, end_time)

                if full == 1:
                    #if self.radioButtonWebSiteImages.isChecked():
                        #processSpinBoxImage(self, currentImageIndex)
                    #else:
                    processLocalImage(self)
                    refreshImage(self)
            elif nProductID == 20033:
                # CREATE PROGRESS BAR
                nitrateDataObj = NEON_20033_NitrateData()

                strStartYearMonth = str(strStartDate.year) + '-' + str(strStartDate.month).zfill(2)
                strEndYearMonth = str(strEndDate.year) + '-' + str(strEndDate.month).zfill(2)
                nitrateList = nitrateDataObj.DP1_20033_FetchNitrateData(SITECODE, nProductID, strStartYearMonth,
                                                                        strEndYearMonth, downloadsFilePath)

                if full == 1:
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

                strStartYearMonth = str(strStartDate.year) + '-' + str(strStartDate.month).zfill(2)
                strEndYearMonth = str(strEndDate.year) + '-' + str(strEndDate.month).zfill(2)
                fetchData.NEON_FetchData(SITECODE, nProductID, strStartYearMonth, strEndYearMonth, downloadsFilePath)


# ======================================================================================================================
#
# ======================================================================================================================
def fetchSiteInfoFromNEON():
    # Make request, using the sites/ endpoint
    global SITECODE
    site_request = requests.get(SERVER + 'sites/' + SITECODE)

    # Convert to Python JSON object
    site_json = site_request.json()

    return (site_json)


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
    site_json = fetchSiteInfoFromNEON()

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

        updateSelectedProductsTable(self, nIndex)
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

        updateSelectedProductsTable(self, nIndex)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # JES


# ======================================================================================================================
#
# ======================================================================================================================
def retranslateUi(self, MainWindow):
    _translate = QtCore.QCoreApplication.translate
    MainWindow.setWindowTitle(_translate("GRIMe-AI: John E. Stranzl Jr.", "GRIMe-AI: John E. Stranzl Jr."))


# ======================================================================================================================
#
# ======================================================================================================================
def downloadFieldSiteTableFiles(csv_links):
    link = csv_links

    # obtain filename by splitting url and getting last string
    file_name = link.split('/')[-1]

    # create response object
    r = requests.get(link, stream=True)

    configFilePath = os.path.expanduser('~')
    configFilePath = os.path.join(configFilePath, 'Documents')
    configFilePath = os.path.join(configFilePath, 'GRIMe-AI')
    if not os.path.exists(configFilePath):
        os.mkdir(configFilePath)
    filename_with_path = os.path.join(configFilePath, file_name)

    # download started
    with open(filename_with_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return filename_with_path


# ======================================================================================================================
# THIS FUNCTION DOWNLOADS THE LATEST IMAGE FOR THE SITE SELECTED BY THE END-USER AND DISPLAYS IT IN THE GUI SO THE
# END-USER CAN CAN SEE WHAT THE PARTICULAR SITE LOOKS LIKE.
# ======================================================================================================================
def downloadLatestImage(siteCode, domainCode):
    global latestImage
    global gWebImageCount

    nErrorCode = -1
    nRetryCount = 3

    latestImageURL = 'https://phenocam.sr.unh.edu/data/latest/NEON.D10.ARIK.DP1.20002.jpg'
    tmp = latestImageURL.replace('ARIK', siteCode)
    latestImageURL = tmp.replace('D10', domainCode)

    while nErrorCode == -1 and nRetryCount > 0:
        r = requests.get(latestImageURL, stream=True)

        if r.status_code != 404:
            gWebImageCount = 1
            data = urlopen(latestImageURL).read()
            latestImage = QPixmap()
            latestImage.loadFromData(data)
            nErrorCode = 0
        else:
            gWebImageCount = 0
            print("404: Download Latest Image")
            nErrorCode = -1
            nRetryCount = nRetryCount - 1

    return r.status_code


# ======================================================================================================================
# FIND THE MONTHS THAT DATA IS AVAILABLE FOR A PARTICULAR PRODUCT FOR A PARTICULAR SITE
# ======================================================================================================================
def findAvailableMonths(item):
    PRODUCTCODE = item

    # RETRIEVE INFORMATION FROM THE NEON WEBSITE FOR THE PARTICULAR SITE
    site_json = fetchSiteInfoFromNEON()

    # EXTRACT THE AVAILABLE MONTH AND THE URL FOR THE DATA FOR EACH AVAILABLE MONTH
    monthList = {}
    for product in site_json['data']['dataProducts']:
        if (product['dataProductCode'] == PRODUCTCODE):
            monthList['availableMonths'] = product['availableMonths']
            monthList['availableDataUrls'] = product['availableDataUrls']
            break

    return (monthList)


# ======================================================================================================================
# The purpose of this function is to query information above a specific product. The information
# contains the product description, sites for which the product is available among other information.
# ======================================================================================================================
def queryProductInfo(productCode):
    product_request = requests.get(SERVER + 'products/' + productCode)
    product_json = product_request.json()

    return product_json


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

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
def createGRIMeFolders(self):
    rootFolder = os.path.expanduser('~')
    rootFolder = os.path.join(rootFolder, 'Documents')
    rootFolder = os.path.join(rootFolder, 'GRIMe-AI')
    if not os.path.exists(rootFolder):
        os.mkdir(rootFolder)

    # ----------------------------------------------------------------------------------------------------
    # SAVE DOWNLOADED DATA TO THE USER GRIMe-AI FOLDER THAT IS AUTOMATICALLY CREATED, IF IT DOES NOT EXIST,
    # IN THE USER'S DOCUMENT FOLDER
    # ----------------------------------------------------------------------------------------------------
    if full == 1:
        videoFolder = os.path.join(rootFolder, 'Videos')
        if not os.path.exists(videoFolder):
            os.mkdir(videoFolder)

        self.EditVideoOutputFolder.setText(videoFolder)

    # --------------------------------------------------
    # CREATE SETTINGS FOLDERS IN USER'S DOCUMENTS FOLDER
    configFilePath = os.path.join(rootFolder, 'Settings')
    if not os.path.exists(configFilePath):
        os.mkdir(configFilePath)

    # --------------------------------------------------
    # CREATE DOWNLOAD FOLDER IN USER'S DOCUMENTS FOLDER
    downloadsFolder = os.path.join(rootFolder, 'Downloads')
    if not os.path.exists(downloadsFolder):
        os.mkdir(downloadsFolder)

    # --------------------------------------------------
    # CREATE FOLDER IN USER'S DOCUMENTS FOLDER FOR DOWNLOADED IMAGES
    downloadsFolder = os.path.join(downloadsFolder, 'Images')
    if not os.path.exists(downloadsFolder):
        os.mkdir(downloadsFolder)

    if full == 1:
        self.EditSaveImagesOutputFolder.setText(downloadsFolder)

    # --------------------------------------------------
    # CREATE DOWNLOAD FOLDER IN USER'S DOCUMENTS FOLDER
    EXIFFolder = os.path.join(rootFolder, 'EXIF')
    if not os.path.exists(EXIFFolder):
        os.mkdir(EXIFFolder)

    if full == 1:
        self.EditEXIFOutputFolder.setText(EXIFFolder)

# ======================================================================================================================
#
# ======================================================================================================================
def populateUSGS_SiteTable(self):
    num_cameras = len(USGS_HIVIS.fetchCameraList())

    for index, camera in enumerate(USGS_HIVIS.fetchCameraList()):
        self.table_USGS_Sites.setItem(index, 0, QTableWidgetItem(camera))
        nImageCount = USGS_HIVIS.getHIVIS_ImageCount(camera, USGS_HIVIS.getEndpoint())

        self.table_USGS_Sites.setItem(index, 1, QTableWidgetItem(str(nImageCount)))

        time.sleep(0.100)

# ======================================================================================================================
#
# ======================================================================================================================
def fetchUSGSImages():
    num_cameras = len(USGS_HIVIS.fetchCameraList())

    for camera in USGS_HIVIS.fetchCameraList():
        # Get a list of ALL images available on NATWEB for the current camera
        images = get_image_list_from_HIVIS(camera, USGS_HIVIS.getEndpoint())

        # Save queried images to output_folder, which is named by camera
        # Makes the directory if needed
        #
        # NOTE: I am slicing images to just return the LAST/Most recent image
        # The NATWEB server sits in the USGS AWS govCloud. They impose http request
        # limits. You can probably pull up to 100 requests in 10 mins or so before
        # NATWEB closes your connection.
        output_folder_base = 'C:\\Users\\Astrid Haugen\\Pictures\\USGS'  # change for your purposes
        output_folder = output_folder_base + os.sep + camera
        download_images(images[-1:], camera_name=camera, save_directory=output_folder, endpoint=USGS_HIVIS.getEndpoint())

# ======================================================================================================================
#
# ======================================================================================================================
def get_image_list_from_HIVIS(camera_name, endpoint, command='/php/listFiles.php?path='):
    """Query HIVIS api to retrieve all images for a specific camera name

    Parameters
    ----------
    camera_name
    endpoint
    command

    Returns
    -------
    list_of_images

    """
    uri = endpoint + command + camera_name
    response = urllib.request.urlopen(uri)
    data = response.read()  # a `bytes` object
    raw_text = data.decode('utf-8')

    # HIVIS returns simple ls of file server. Split text and pop the 1st 2 entries
    t = raw_text.split(sep=',')[2:]

    # Remove the quotes
    t = [elem.replace('"', '') for elem in t]

    # Replace any spaces with URL safe %20
    t = [elem.replace(' ', '%20') for elem in t]

    # Finally, just keep strings that correspond to images.
    # These will all have '___' in them
    subs = '___'
    list_of_images = list(filter(lambda x: subs in x, t))

    return list_of_images

# ======================================================================================================================
#
# ======================================================================================================================
def download_images(list_of_images, camera_name, save_directory, endpoint, subpage='/media/cameras/'):
    """Make urlrequest and download result into specified file.

    Parameters
    ----------
    list_of_images
    camera_name
    save_directory
    endpoint
    subpage

    Returns
    -------

    """
    number_images = len(list_of_images)
    printEnd = ''  # needed to get progress bar to show in PyCharm IDE, '\r' otherwise
    # printProgressBar(0, number_images, prefix='Progress:', suffix='Complete', length=50, printEnd=printEnd)
    i = 0
    for image in list_of_images:
        uri = endpoint + subpage + camera_name + '/' + image

        # Create a folder named by the camera if it doesn't exist,
        # then create the file name full path within that folder
        save_directory = pathlib.Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        out_file = str(save_directory) + os.sep + image  # file name is the same as original

        # Download the image
        with urllib.request.urlopen(uri) as response, open(out_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        i += 1


# ======================================================================================================================
#
# ======================================================================================================================
def fetchDownloadsFolderPath():
    downloadsFilePath = os.path.expanduser('~')
    downloadsFilePath = os.path.join(downloadsFilePath, 'Documents')
    downloadsFilePath = os.path.join(downloadsFilePath, 'GRIMe-AI')
    if not os.path.exists(downloadsFilePath):
        os.mkdir(downloadsFilePath)
    downloadsFilePath = os.path.join(downloadsFilePath, 'Downloads')
    if not os.path.exists(downloadsFilePath):
        os.mkdir(downloadsFilePath)

    return downloadsFilePath


# ======================================================================================================================
#
# ======================================================================================================================
if __name__ == '__main__':
    # CREATE MAIN APP WINDOW
    app = QApplication(sys.argv)
    frame = MainWindow()

    frame.move(app.desktop().screen().rect().center() - frame.rect().center())

    # DISPLAY SPLASH SCREEN
    if full == 1:
        pixmap = QPixmap('Splash_007.jpg')
    else:
        pixmap = QPixmap('Splash_007DM.jpg')
    splash = QSplashScreen(pixmap)
    splash.show()
    time.sleep(5)
    splash.finish(frame)

    # PROCESS ANY EVENTS THAT WERE DELAYED BECAUSE OF THE SPLASH SCREEN
    app.processEvents()

    # CREATE REQUIRED FOLDERS IN THE USER'S DOCUMENTS FOLDER
    createGRIMeFolders(frame)

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
        formatProductTableCells(frame)

    if len(siteList) == 0:
        frame.radioButtonHardDriveImages.setChecked(True)
        frame.radioButtonHardDriveImages.setDisabled(False)

        # DISABLE AND UNCHECK THE RADIO BUTTON FOR WEB SITE IMAGES IF INFORMATION CANNOT BE ACCESSED FROM THE NEON SITE
        frame.radioButtonWebSiteImages.setChecked(False)
        frame.radioButtonWebSiteImages.setDisabled(True)
    # IF THERE ARE FIELD SITE TABLES AVAILABLE, ENABLE GUI WIDGETS PERTAINING TO WEB SITE DATA/IMAGES
    else:
        myList = []

        for site in siteList:
            strSiteName = site.siteID + ' - ' + site.siteName
            myList.append(strSiteName)

        frame.listboxNEONSites.addItems(myList)
        if full == 1:
            frame.spinBoxDailyImage.setValue(1)

        # JES - TEMPORARILY SET BARCO LAKE AS THE DEFAULT SELECTION
        frame.listboxNEONSites.setCurrentRow(2)
        frame.listboxNEONSites.show()

        # SET DEFAULT TO IMAGES STORED ON A HARD DRIVE OR ON A MAPPED DRIVE SO THAT THERE ARE NO UNNECESSARY ATTEMPTS TO ACCESS THE INTERNET/WEB
        if full == 1:
            frame.radioButtonHardDriveImages.setChecked(False)
            frame.radioButtonHardDriveImages.setDisabled(False)

            frame.radioButtonWebSiteImages.setChecked(True)
            frame.radioButtonWebSiteImages.setDisabled(False)

    '''
    if full == 1:
        if frame.radioButtonWebSiteImages.isChecked():
            frame.checkBoxCreateEXIFFile.setEnabled(False)
            processSpinBoxImage(frame, 1)
            refreshImage(frame)
        elif frame.radioButtonHardDriveImages.isChecked():
            processLocalImage(frame)
            frame.checkBoxCreateEXIFFile.setEnabled(True)
    '''

    if full == 1:
        frame.radioButton_ROIShapeRectangle.setChecked(True)

        # ----------------------------------------------------------------------
        # CHANGE THE COLOR OF OPTIONS THAT HAVE SPECIAL CONDITIONS AND USAGES
        frame.checkBoxSaveImages.setStyleSheet('QCheckBox {color: red;}')

        frame.pushButtonRetrieveData.setStyleSheet('QPushButton {background-color: steelblue;}')
        frame.pushButtonFetchImageList.setStyleSheet('QPushButton {background-color: steelblue;}')

        frame.pushButtonBrowseImageFolder.setStyleSheet('QPushButton {background-color: steelblue;}')
        frame.pushButtonBrowseVideoOutputFolder.setStyleSheet('QPushButton {background-color: steelblue;}')
        frame.pushButtonBrowseSaveImagesOutputFolder.setStyleSheet('QPushButton {background-color: steelblue;}')
        frame.pushButtonBrowseEXIFOutputFolder.setStyleSheet('QPushButton {background-color: steelblue;}')

        frame.pushButtonPlay.setStyleSheet('QPushButton {background-color: steelblue;}')
        frame.pushButtonStop.setStyleSheet('QPushButton {background-color: steelblue;}')
        frame.pushButtonAddROI.setStyleSheet('QPushButton {background-color: steelblue;}')

    frame.spinBoxCannyHighThreshold.setValue(255)
    frame.spinBoxCannyLowThreshold.setValue(254)

    frame.graphicsView.setVisible(False)

    populateUSGS_SiteTable(frame)

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

    # product_json = queryProductInfo('DP1.00004.001')

    # data_request = requests.get('http://data.neonscience.org/api/v0/' + 'data/' + 'DP1.00004.001' + '/' + 'ARIK' + '/' + '2018-06')
    # data_json = data_request.json()

    # NEON_API_Example('DP1.00004.001')

    # SHOW MAIN WINDOW
    frame.show()

    # Run the program
    sys.exit(app.exec())

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

'''
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
'''
