# !/usr/bin/env python
# !/usr/bin/env python
# coding: utf-8

# THIS TO INVESTIGATE
# https://github.com/smhassanerfani/atlantis/tree/master/aquanet
# https://github.com/smhassanerfani/atlantis/tree/master



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

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import os
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.4.1'
#os.system[str('R_HOME')] = str("C:\\Program Files\\R\\R-4.4.1")
import sys
import shutil
import re
import json
import random
from pathlib import Path
import iopath

import numpy as np


# ------------------------------------------------------------
# PIL LIBRARIES AND IMPORTS
# ------------------------------------------------------------
from PIL import Image, ImageDraw


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import datetime
from datetime import date
#from datetime import datetime
import time
from time import sleep

import pandas as pd

import math
import csv

import requests
import urllib.request
from configparser import ConfigParser
from urllib.request import urlopen
#import chromedriver_autoinstaller

import promptlib

import cv2


# ------------------------------------------------------------
# WHERE THE BITS MEET THE DIGITAL ROAD
# ------------------------------------------------------------
'''
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
'''

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QToolBar, QCheckBox, QDateTimeEdit, \
    QGraphicsScene, QMessageBox, QAction, QHeaderView

from GRIME_AI_SplashScreen import GRIME_AI_SplashScreen


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import sobelData


# ----------------------------------------------------------------------------------------------------------------------
# POP-UP/MODELESS DIALOG BOXES
# ----------------------------------------------------------------------------------------------------------------------
from GRIME_AI_ColorSegmentationDlg import GRIME_ColorSegmentationDlg
from GRIME_AI_EdgeDetectionDlg import GRIME_AI_EdgeDetectionDlg
from GRIME_AI_ImageNavigationDlg import GRIME_ImageNavigationDlg
from GRIME_AI_FileUtilitiesDlg import GRIME_AI_FileUtilitiesDlg
from GRIME_AI_MaskEditorDlg import GRIME_AI_MaskEditorDlg
from GRIME_AI_CompositeSliceDlg import GRIME_CompositeSliceDlg
from GRIME_ProcessImage import GRIME_ProcessImage
from GRIME_AI_ReleaseNotesDlg import GRIME_ReleaseNotesDlg
from GRIME_TriageOptionsDlg import GRIME_TriageOptionsDlg
from GRIME_AI_buildModelDlg import GRIME_AI_buildModelDlg
from GRIME_AI_Color import GRIME_AI_Color
from GRIME_AI_CompositeSlices import GRIME_AI_CompositeSlices
from GRIME_AI_Vegetation_Indices import GRIME_AI_Vegetation_Indices, greennessIndex
from GRIME_AI_ExportCOCOMasksDlg import GRIME_AI_ExportCOCOMasksDlg

from GRIME_AI_Save_Utils import JsonEditor


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from GRIME_AI_Feature_Export import GRIME_AI_Feature_Export


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from GRIME_AI_Diagnostics import GRIMe_Diagnostics
from GRIME_AI_ImageData import imageData
from GRIMe_ImageStats import GRIMe_ImageStats


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from GRIME_AI_PhenoCam import GRIME_PhenoCam, dailyList
from GRIME_ProductTable import GRIMe_ProductTable
from GRIME_QLabel import DrawingMode
from GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_QProgressWheel import QProgressWheel
from GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI_roiData import GRIME_roiData, ROIShape

from GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI_Resize_Controls import GRIME_AI_Resize_Controls

from GRIME_AI_TimeStamp_Utils import GRIME_AI_TimeStamp_Utils
from GRIME_AI_ImageTriage import GRIME_AI_ImageTriage

from GRIME_AI_DeepLearning import GRIME_AI_DeepLearning

from colorSegmentationParams import colorSegmentationParamsClass


# ----------------------------------------------------------------------------------------------------------------------
# HYDRA (for SAM2)
# ----------------------------------------------------------------------------------------------------------------------
import hydra
from hydra import initialize, compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from NEON_API import NEON_API


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from USGS_NIMS import USGS_NIMS


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from chrome_driver import *


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from constants import edgeMethodsClass, featureMethodsClass, modelSettingsClass


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from exifData import EXIFData

from playsound import playsound
# import GRIME_KMeans
# from webdriver_manager.core.utils import ChromeType


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# import tensorflow as tf
try:
    import torch
    print(torch.__version__)

    import torchvision.transforms as T
    from torch.cuda.amp import GradScaler, autocast
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.nn.functional import binary_cross_entropy_with_logits

    print("GRIME AI Deep Learning: PyTorch imported successfully.")
except ImportError as e:
    print("GRIME AI Deep Learning: Error importing PyTorch:", e)
    # Remove the faulty package from sys.modules to prevent further issues
    if 'torch' in sys.modules:
        del sys.modules['torch']

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2'))
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if 0:
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    if not GlobalHydra.instance().is_initialized():
        initialize_config_module("sam2", version_base="1.2")


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools import mask as coco_mask


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

try:
    import imageio
except:
    pass

try:
    import imageio_ffmpeg
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


SITECODE = 'ARIK'
DOMAINCODE = 'D10'
SITENAME = ''
originalImg = []
dailyImagesList = dailyList([], [])
currentImage = []
currentImageIndex = -1
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

# URLS
# url = "http://maps.googleapis.com/maps/api/geocode/json?address=googleplex&sensor=false"
url = 'https://www.neonscience.org/field-sites/explore-field-sites'
root_url = 'https://www.neonscience.org'
SERVER = 'http://data.neonscience.org/api/v0/'

SW_VERSION = "Ver.: 0.0.5.11d"

class displayOptions():
    displayROIs = True

g_displayOptions   = displayOptions()
#g_featureDetection = featureDetection()
#g_edgeDetection    = edgeDetection()
#g_features         = features()
g_greennessIndex   = greennessIndex()

g_edgeMethodSettings = edgeMethodsClass()
g_featureMethodSettings = featureMethodsClass()

g_modelSettings = modelSettingsClass()

# ======================================================================================================================
# 2. DEEP LEARNING: DEFINE A CONVOLUTIONAL NETWORK
# ======================================================================================================================
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''


# ==================================================================================================================
#
# ==================================================================================================================
class SAM2FullModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.image_encoder = model.forward_image
        self._prepare_backbone_features = model._prepare_backbone_features
        self.directly_add_no_mem_embed = model.directly_add_no_mem_embed
        self.no_mem_embed = model.no_mem_embed
        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

    def forward(self, image, point_coords, point_labels):
        backbone_out = self.image_encoder(image)
        _, vision_feats, _, _ = self._prepare_backbone_features(backbone_out)
        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.no_mem_embed
        feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size) for feat, feat_size in
                 zip(vision_feats[::-1], self._bb_feat_sizes[::-1])][::-1]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in features["high_res_feats"]]
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=(point_coords, point_labels), boxes=None,
                                                                  masks=None)
        low_res_masks, iou_predictions, _, _ = self.mask_decoder(
            image_embeddings=features["image_embed"][-1].unsqueeze(0),
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=point_coords.shape[0] > 1,
            high_res_features=high_res_features,
        )
        out = {"low_res_masks": low_res_masks, "iou_predictions": iou_predictions}
        return out


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

    #os.environ[str('R_HOME')] = str('C:/Program Files/R/R-4.4.1')
    print(os.environ.get('R_HOME'))

    # INITIALIZE POP-UP DIALOG BOXES
    fileFolderDlg        = None
    edgeDetectionDlg     = None
    colorSegmentationDlg = None
    TriageDlg            = None
    maskEditorDlg        = None
    compositeSliceDlg    = None
    imageNavigationDlg   = None
    releaseNotesDlg      = None
    buildModelDlg        = None


    imageFileFolder = None

    global dailyImagesList
    dailyImagesList = dailyList([], [])

    NEON_latestImage = []
    USGS_latestImage = []

    # def eventFilter(self, source, event):
    #     if (event.type() == QtCore.QEvent.MouseMove and source is self.label):
    #         pos = event.pos()
    #         print('mouse move: (%d, %d)' % (pos.x(), pos.y()))
    #
    #     if (event.type() == QtCore.QEvent.MouseButtonDblClick and source is self.label):
    #         print('Double click')
    #
    #     return QtGui.QWidget.eventFilter(self, source, event)


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def resizeEvent(self, event):

        currentTabIndex = self.tabWidget.currentIndex()

        # PARENT CLASS WHICH CONTAINS ALL FUNCTIONS TO RESIZE ALL CONTROLS ON THE GUI, AS NEEDED
        resizeControls = GRIME_AI_Resize_Controls()

        # TAB 0 - NEON SITES
        resizeControls.resizeTab_0(self, event)
        self.NEON_DisplayLatestImage()

        # TAB 1 - NEON DOWNLOAD MANAGER
        resizeControls.resizeTab_1(self, event)

        # TAB 2 - USGS SITES
        resizeControls.resizeTab_2(self, event)
        self.USGS_DisplayLatestImage()

        # TAB 3 - USGS DOWNLOAD MANAGER
        resizeControls.resizeTab_3(self, event)

        # TAB 4 - IMAGE ANALYSIS
        resizeControls.resizeTab_4(self, event)

        # TAB 5 - SENSOR DATA GRAPHS

        #QtWidgets.resizeEvent(self, event)

    # ------------------------------------------------------------------------------------------------------------------
    # CLASS INITIALIZATION
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None, win=None, session=None):
        super(MainWindow, self).__init__(parent)
        self.mainwin = win
        self.session = session
        self.ui = Ui_MainWindow()

        #os.environ[str('R_HOME')] = str("C:\\Program Files\\R\\R-4.4.1")
        #JES - THIS DOESN'T WORK! - os.system[str('R_HOME')] = str("C:\\Program Files\\R\\R-4.4.1")

        self.setWindowTitle("GRIMe-AI: John E. Stranzl Jr.")
        #self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setupUi(self)

        # Set stylesheet for the tabs to change color when a tab is selected.
        self.tabWidget.setStyleSheet("""
            QTabBar::tab {
                background-color: white;
                color: black;
            }
            QTabBar::tab:selected {
                background-color: steelblue;
                color: white;
            }
        """)


        # ------------------------------------------------------------------------------------------------------------------
        # DISPLAY SPLASH SCREEN
        # ------------------------------------------------------------------------------------------------------------------
        splash = GRIME_AI_SplashScreen(QPixmap('Splash_007.jpg'), strVersion=SW_VERSION)
        splash.show(self)
        splash = GRIME_AI_SplashScreen(QPixmap('GRIME-AI Logo.jpg'), delay=5)
        splash.show(self)


        # ------------------------------------------------------------------------------------------------------------------
        # CREATE REQUIRED FOLDERS IN THE USER'S DOCUMENTS FOLDER
        # ------------------------------------------------------------------------------------------------------------------
        utils = GRIME_AI_Utils()
        utils.create_GRIME_folders(full)

        self.populate_controls()

        self.loss_values = []
        self.val_loss_values = []
        self.epoch_list = []
        self.scaler = GradScaler()

        # ----------------------------------------------------------------------------------------------------
        #
        # ----------------------------------------------------------------------------------------------------
        #JES file_utils = GRIME_AI_Save_Utils()
        #JES file_utils.read_config_file()

        global imageFileFolder
        imageFileFolder = JsonEditor().getValue("Local_Image_Folder")

        #JES folderPath = GRIME_AI_Save_Utils().NEON_getSaveFolderPath()
        #JES self.edit_NEONSaveFilePath.1setText(folderPath)

        #JES folderPath = GRIME_AI_Save_Utils().USGS_getSaveFolderPath()
        #JES self.edit_USGSSaveFilePath.setText(folderPath)


        # ----------------------------------------------------------------------------------------------------
        #
        # ----------------------------------------------------------------------------------------------------
        self.colorSegmentationParams = colorSegmentationParamsClass()


        # ----------------------------------------------------------------------------------------------------
        # GET DATA, POPULATE WIDGETS, ETC.
        # ----------------------------------------------------------------------------------------------------
        self.USGS_InitProductTable()

        self.USGS_FormatProductTable(self.table_USGS_Sites)

        self.NEON_FormatProductTableHeader()

        self.initROITable()


        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # JES - REVISIT DOUBLE CLICKING ON IMAGES
        self.NEON_labelLatestImage.mouseDoubleClickEvent = NEON_labelMouseDoubleClickEvent

        self.NEON_labelLatestImage.installEventFilter(self)
        self.labelEdgeImage.installEventFilter(self)
        self.labelOriginalImage.installEventFilter(self)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        self.pushButton_RetrieveNEONData.clicked.connect(self.pushbutton_NEONDownloadClicked)

        self.pushButton_USGS_BrowseImageFolder.clicked.connect(self.pushButton_USGS_BrowseImageFolder_Clicked)
        self.pushButton_NEON_BrowseImageFolder.clicked.connect(self.pushButton_NEON_BrowseImageFolder_Clicked)

        #JES self.radioButton_ROIShapeRectangle.clicked.connect(self.ROIShapeClicked)
        #JES self.radioButton_ROIShapeEllipse.clicked.connect(self.ROIShapeClicked)
        #JES self.pushButton_TrainGood.clicked.connect(self.pushButtonTrainGoodTriggered)
        #JES self.pushButton_TrainBad.clicked.connect(self.pushButtonTrainBadTriggered)

        # INITIALIZE WIDGETS
        maxRows = self.tableWidget_ROIList.rowCount()
        nCol = 0
        for i in range(0, maxRows):
            self.tableWidget_ROIList.removeRow(0)

        # SAVE AND RECALL SETTINGS
        self.action_SaveSettings.triggered.connect(self.menubarSaveSettings)
        self.action_ReleaseNotes.triggered.connect(self.toolbarButtonReleaseNotes)
        self.action_CompositeSlices.triggered.connect(self.menubarCompositeSlices)
        self.action_ExtractCOCOMasks.triggered.connect(self.menubarExtractCOCOMasks)
        self.action_TriageImages.triggered.connect(toolbarButtonImageTriage)

        self.action_RefreshNEON.triggered.connect(self.menubar_RefreshNEON)

        # GRAPH TAB(S)

        self.NEON_labelLatestImage.setScaledContents(True)
        # self.ui.labelLatestImage.setScaledContents(True)
        # self.ui.labelOriginalImage.setScaledContents(True)
        # self.ui.labelEdgeImage.setScaledContents(True)

        # ------------------------------------------------------------------------------------------------------------------
        # NEON
        # ------------------------------------------------------------------------------------------------------------------
        self.NEON_listboxSites.itemClicked.connect(self.NEON_SiteClicked)
        self.NEON_listboxSiteProducts.itemClicked.connect(self.NEON_ProductClicked)

        # ------------------------------------------------------------------------------------------------------------------
        # USGS
        # ------------------------------------------------------------------------------------------------------------------
        self.USGS_listboxSites.itemClicked.connect(self.USGS_SiteClicked)
        self.pushButton_USGSDownload.clicked.connect(self.pushButton_USGSDownloadClicked)

        # ------------------------------------------------------------------------------------------------------------------
        # NIMS
        # ------------------------------------------------------------------------------------------------------------------
        try:
            self.myNIMS = USGS_NIMS()

            cameraDictionary = self.myNIMS.getCameraDictionary()
            cameraList = self.myNIMS.getCameraList()
            self.USGS_listboxSites.clear()
            self.USGS_listboxSites.addItems(cameraList)
            self.USGS_listboxSites.show()

            cameraIndex = 1
            self.USGS_listboxSites.setCurrentRow(cameraIndex)

            strCamID = self.USGS_listboxSites.currentItem().text()

            cameraInfo = self.myNIMS.getCameraInfo(strCamID)
            self.listboxUSGSSiteInfo.addItems(cameraInfo)

            self.USGS_updateSiteInfo(1)

        except:
            msgBox = GRIME_AI_QMessageBox('USGS NIMS Error', 'Unable to access USGS NIMS Database!')
            response = msgBox.displayMsgBox()

        #self.edit_USGSSaveFilePath.setText("C:\\Users\\Astrid Haugen\\Documents\\GRIMe-AI\\Downloads\\USGS_Test")

        # ------------------------------------------------------------------------------------------------------------------
        # USGS
        # ------------------------------------------------------------------------------------------------------------------
        #exif = EXIFData().extractEXIFdata('F:/000 - Hydrology Images/Reconyx/RCNX0009.jpg')

        #x = 1

        print("Create toolbar...")
        self.createToolBar()
        print("Toolbar create...")

        # ------------------------------------------------------------------------------------------------------------------
        # MENU
        # ------------------------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------------------
        # SET THE BACKGROUND COLORS OF SPECIFIC BUTTONS
        # ------------------------------------------------------------------------------------------------------------------
        self.pushButton_RetrieveNEONData.setStyleSheet('QPushButton {background-color: steelblue;}')
        self.pushButton_NEON_BrowseImageFolder.setStyleSheet('QPushButton {background-color: steelblue;}')

        self.pushButton_USGSDownload.setStyleSheet('QPushButton {background-color: steelblue;}')
        self.pushButton_USGS_BrowseImageFolder.setStyleSheet('QPushButton {background-color: steelblue;}')

        # INITIALIZE GUI CONTROLS
        # frame.NEON_listboxSites.setCurrentRow(1)

        # GET LIST OF ALL SITES ON NEON
        # if frame.checkBoxNEONSites.isChecked():
        print("Download NEON Field Site Table from NEON website...")
        myNEON_API = NEON_API()
        siteList = myNEON_API.readFieldSiteTable()

        if len(siteList) == 0:
            print("NEON Field Site Table from NEON website FAILED...")
            pass
            # frame.radioButtonHardDriveImages.setChecked(True)
            # frame.radioButtonHardDriveImages.setDisabled(False)
        # IF THERE ARE FIELD SITE TABLES AVAILABLE, ENABLE GUI WIDGETS PERTAINING TO WEB SITE DATA/IMAGES
        else:
            print("Populate NEON Products tab on GUI...")
            myList = []

            for site in siteList:
                strSiteName = site.siteID + ' - ' + site.siteName
                myList.append(strSiteName)

            self.NEON_listboxSites.addItems(myList)

            # JES - TEMPORARILY SET BARCO LAKE AS THE DEFAULT SELECTION
            try:
                self.NEON_listboxSites.setCurrentRow(2)
                self.NEON_listboxSites.show()
                self.NEON_SiteClicked(2)
            except:
                pass

        self.show()


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def populate_controls(self):

        # NEON CONTROLS
        NEON_download_file_path = JsonEditor().getValue("NEON_Root_Folder")
        self.edit_NEONSaveFilePath.setText(NEON_download_file_path)

        # USGS CONTROLS
        USGS_download_file_path = JsonEditor().getValue("USGS_Root_Folder")
        self.edit_USGSSaveFilePath.setText(USGS_download_file_path)


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def buildFeatureFile(self):
        global dailyImagesList
        myFeatureExport = GRIME_AI_Feature_Export()
        imagesList = dailyImagesList.getVisibleList()

        if self.colorSegmentationDlg != None:
            self.getColorSegmentationParams()

        global imageFileFolder
        myFeatureExport.ExtractFeatures(imagesList, imageFileFolder, self.roiList, self.colorSegmentationParams)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def universalTestButton(self, testFunction):
        if testFunction == 1:
            print('This is Test Function 1.')

            # KMeans EXPECTS THE BYTE ORDER TO BE RGB
            #img1 = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())
            img1 = GRIME_AI_Utils().convertQImageToMat(myImage.toImage())

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

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def getColorSegmentationParams(self):
        self.colorSegmentationParams.GCC            = self.colorSegmentationDlg.checkBox_GCC.isChecked()
        self.colorSegmentationParams.GLI            = self.colorSegmentationDlg.checkBox_GLI.isChecked()
        self.colorSegmentationParams.NDVI           = self.colorSegmentationDlg.checkBox_NDVI.isChecked()
        self.colorSegmentationParams.ExG            = self.colorSegmentationDlg.checkBox_ExG.isChecked()
        self.colorSegmentationParams.RGI            = self.colorSegmentationDlg.checkBox_RGI.isChecked()

        self.colorSegmentationParams.Intensity      = self.colorSegmentationDlg.checkBox_Intensity.isChecked()
        self.colorSegmentationParams.ShannonEntropy = self.colorSegmentationDlg.checkBox_ShannonEntropy.isChecked()
        self.colorSegmentationParams.Texture        = self.colorSegmentationDlg.checkBox_Texture.isChecked()

        self.colorSegmentationParams.wholeImage     = self.colorSegmentationDlg.checkBoxScalarRegion_WholeImage.isChecked()

    # ------------------------------------------------------------------------------------------------------------------
    # TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR
    # ------------------------------------------------------------------------------------------------------------------
    def createToolBar(self):
        #--- CREATE EMPTY TOOLBAR
        toolbar = QToolBar("GRIMe-AI Toolbar")
        self.addToolBar(toolbar)
        toolbar.setIconSize(QtCore.QSize(48, 48))

        parent_path = Path(__file__).parent
        print("Toolbar Initialization: Parent path of executable: ", parent_path)

        #--- COLOR SEGMENTATION
        icon_path = os.path.normpath(str(parent_path / "icons/FileFolder_1.png"))
        button_action = QAction(QIcon(icon_path), "Folder Operations", self)
        button_action.setStatusTip("Select input and output folder locations")
        button_action.triggered.connect(self.onMyToolBarFileFolder)
        toolbar.addAction(button_action)
        print("Toolbar Initialization: File folder icon path: ", icon_path)

        #--- IMAGE TRIAGE
        icon_path = os.path.normpath(str(parent_path / "icons/Triage_2.png"))
        button_action = QAction(QIcon(icon_path), "Image Triage", self)
        button_action.setStatusTip("Move images that are of poor quality")
        button_action.triggered.connect(toolbarButtonImageTriage)
        toolbar.addAction(button_action)
        print("Toolbar Initialization: Triage icon path: ", icon_path)

        #--- MASK EDITOR
        icon_path = os.path.normpath(str(parent_path / "icons/ImageNav_3.png"))
        button_action = QAction(QIcon(icon_path), "Image Navigation", self)
        button_action.setStatusTip("Navigate (scroll) through images")
        button_action.triggered.connect(self.onMyToolBarImageNavigation)
        toolbar.addAction(button_action)
        print("Toolbar Initialization: Image Navigation icon path: ", icon_path)

        #--- IMAGE NAVIGATION
        icon_path = os.path.normpath(str(parent_path / "icons/Mask.png"))
        button_action = QAction(QIcon(icon_path), "Create Masks", self)
        button_action.setStatusTip("Draw polygons to create image masks")
        button_action.triggered.connect(self.onMyToolBarCreateMask)
        toolbar.addAction(button_action)
        print("Toolbar Initialization: Create Masks icon path: ", icon_path)

        #--- COLOR SEGMENTATION
        icon_path = os.path.normpath(str(parent_path / "icons/ColorWheel_4.png"))
        button_action = QAction(QIcon(icon_path), "Color Segmentation", self)
        button_action.setStatusTip("Create ROIs to segment regions by color")
        button_action.triggered.connect(self.onMyToolBarColorSegmentation)
        toolbar.addAction(button_action)
        print("Toolbar Initialization: Color Segmentation icon path: ", icon_path)

        #--- EDGE FILTERS
        icon_path = os.path.normpath(os.path.normpath(str(parent_path / "icons/EdgeFilters_2.png")))
        button_action = QAction(QIcon(icon_path), "Edge and Feature Detection", self)
        button_action.setStatusTip("Edge Detection Filters")
        button_action.triggered.connect(self.toolbarButtonEdgeDetection)
        toolbar.addAction(button_action)
        print("Toolbar Initialization: Edge and Feature Detection icon path: ", icon_path)

        #--- SETTINGS
        icon_path = os.path.normpath(str(parent_path / "icons/Settings_1.png"))
        icon_path = os.path.normpath(icon_path)
        button_action = QAction(QIcon(icon_path), "Settings", self)
        button_action.setStatusTip("Change options and settings")
        button_action.triggered.connect(self.onMyToolBarSettings)
        toolbar.addAction(button_action)
        print("Toolbar Initialization: Settings icon path: ", icon_path)

        #--- DEEP LEARNING
        icon_path = os.path.normpath(str(parent_path / "icons/Green Brain Icon.png"))
        button_action = QAction(QIcon(icon_path), "Deep Learning", self)
        button_action.setStatusTip("Deep Learning - EXPERIMENTAL")
        #button_action.triggered.connect(self.toolbarButtonDeepLearning)
        button_action.triggered.connect(self.onMyToolBarBuildModel)
        toolbar.addAction(button_action)
        print("Toolbar Initialization: Deep Learning (brain) icon path: ", icon_path)

        #--- GRIME2
        icon_path = os.path.normpath(str(parent_path / "icons/grime2_StopSign.png"))
        button_action = QAction(QIcon(icon_path), "GRIME2", self)
        button_action.setStatusTip("GRIME2 - Water Level Measurement")
        button_action.triggered.connect(self.toolbarButtonGRIME2)
        toolbar.addAction(button_action)
        print("Toolbar Initialization: GRIME2 icon path: ", icon_path)

        #--- HELP
        icon_path = os.path.normpath(str(parent_path / "icons/Help_2.png"))
        button_action = QAction(QIcon(icon_path), "Help", self)
        button_action.setStatusTip("Help and Release Notes")
        button_action.triggered.connect(self.toolbarButtonReleaseNotes)
        toolbar.addAction(button_action)
        print("Toolbar Initialization: Help icon path: ", icon_path)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PROCESS NEON SITE CHANGE
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def NEON_SiteClicked(self, item):
        global SITECODE
        global gWebImagesAvailable
        global gProcessClick
        global gWebImageCount

        print("NEON Site selected...")
        try:
            # gProcessClick is checked to see if another process is already handling a click event (gProcessClick == 0).
            # If not, it sets gProcessClick to 1 to prevent concurrent clicks.

            if gProcessClick == 0:
                gProcessClick = 1

                # --------------------------------------------------------------------------------
                # --------------------------------------------------------------------------------
                print("Updating site info...")

                start_time = time.time()
                SITECODE = NEON_updateSiteInfo(self, item)
                end_time = time.time()
                print ("NEON Site Info Elapsed Time: ", end_time - start_time)

                # --------------------------------------------------------------------------------
                # --------------------------------------------------------------------------------
                print("Updating site products...")
                time.sleep(2.0)

                start_time = time.time()
                self.NEON_updateSiteProducts(item)
                end_time = time.time()
                print ("NEON Site Products Elapsed Time: ", end_time - start_time)

                # --------------------------------------------------------------------------------
                # --------------------------------------------------------------------------------
                print("Download latest image...")
                time.sleep(2.0)

                start_time = time.time()
                nErrorCode, self.NEON_latestImage, gWebImageCount = NEON_API().DownloadLatestImage(SITECODE, DOMAINCODE)
                end_time = time.time()
                print ("NEON Latest Image Elapsed Time: ", end_time - start_time)

                # --------------------------------------------------------------------------------
                # --------------------------------------------------------------------------------
                if nErrorCode == 404:
                    gWebImagesAvailable = 0
                    self.NEON_labelLatestImage.setText("No Images Available")
                else:
                    gWebImagesAvailable = 1

                    start_time = time.time()
                    self.NEON_DisplayLatestImage()
                    end_time = time.time()
                    print ("NEON Display Latest Image Elapsed Time: ", end_time - start_time)

                gProcessClick = 0
        except:
            gProcessClick = 0

    # ------------------------------------------------------------------------------------------------------------------
    # UPDATE NEON SITE PRODUCT INFORMATION
    # ------------------------------------------------------------------------------------------------------------------
    def NEON_ProductClicked(self, item):
        NEON_updateProductTable(self, item)

    # ------------------------------------------------------------------------------------------------------------------
    # DOWNLOAD NEON PRODUCT FILES
    # ------------------------------------------------------------------------------------------------------------------
    def pushbutton_NEONDownloadClicked(self, item):
        downloadProductDataFiles(self, item)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PROCESS USGS DOWNLOAD MANAGER ACTIONS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ======================================================================================================================
    #
    # ======================================================================================================================
    def USGS_InitProductTable(self):
        # HEADER TITLES
        headerList = ['Site', 'Image Count', ' min Date ', ' max Date ', 'Start Date', 'End Date', 'Start Time', 'End Time']

        # DEFINE HEADER STYLE
        stylesheet = "::section{Background-color:rgb(116,175,80);border-radius:14px;}"

        # POINTER TO HEADER
        header = self.table_USGS_Sites.horizontalHeader()

        # SET DEFAULT HEADER SETTINGS
        header.setMinimumSectionSize(120)
        header.setDefaultSectionSize(140)
        header.setHighlightSections(False)
        header.setStretchLastSection(False)

        # INSERT TITLES INTO HEADER AND FORMAT HEADER
        # MAKE COLUMNS 1 THRU 'n' SIZE TO CONTENTS
        # MAKE COLUMN 0 STRETCH TO FILL UP REMAINING EMPTY SPACE IN THE TABLE
        for i, item in enumerate(headerList):
            headerItem = QTableWidgetItem(item)
            headerItem.setTextAlignment(QtCore.Qt.AlignCenter)

            self.table_USGS_Sites.setHorizontalHeaderItem(i, headerItem)
            self.table_USGS_Sites.setStyleSheet(stylesheet)

            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        header.setSectionResizeMode(0, QHeaderView.Stretch)

        # SET THE HEADER FONT
        font = QFont()
        font.setBold(True)
        self.table_USGS_Sites.horizontalHeader().setFont(font)
        #self.table_USGS_Sites.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        #self.table_USGS_Sites.resizeColumnsToContents()

        date_widget = QtWidgets.QDateEdit(QtCore.QDate(date.today().year, date.today().month, date.today().day))
        self.table_USGS_Sites.setCellWidget(0, 4, date_widget)
        self.table_USGS_Sites.setCellWidget(0, 5, date_widget)

        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
        date_widget.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
        date_widget.setKeyboardTracking(False)
        self.table_USGS_Sites.setCellWidget(0, 6, date_widget)

        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
        date_widget.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
        date_widget.setKeyboardTracking(False)
        self.table_USGS_Sites.setCellWidget(0, 7, date_widget)

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def USGS_dateChangeMethod(self, date_widget, tableWidget):
        imageCount = self.USGS_getImageCount()

        tableWidget.setItem(0, 1, QTableWidgetItem(imageCount.__str__()))

    # ======================================================================================================================
    # THIS FUNCTION WILL UPDATE THE PRODUCT TABLE IN THE GUI WITH THE PRODUCTS THAT ARE AVAILABLE FOR A SPECIFIC SITE.
    # ======================================================================================================================
    def USGS_FormatProductTable(self, tableProducts):
        maxRows = 1

        # JES: MUST MAKE CODE DYNAMIC TO ONLY DELETE UNSELECTED ITEMS
        for i in range(tableProducts.rowCount()):
            tableProducts.removeRow(0)

        tableProducts.insertRow(0)

        for i in range(maxRows):
            m = 0
            tableProducts.setItem(i, m, QTableWidgetItem(''))

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

            # date_widget = QtWidgets.QDateEdit(calendarPopup=True)
            # date_widget.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
            # tableProducts.setCellWidget(i, m, date_widget)

            # date_widget = QtWidgets.QDateEdit(calendarPopup=True)
            # date_widget.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
            # tableProducts.setCellWidget(i, m, date_widget)

            m += 1
            date_widget = QtWidgets.QDateEdit(calendarPopup=True)
            date_widget.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
            date_widget.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
            date_widget.setKeyboardTracking(False)
            self.table_USGS_Sites.setCellWidget(i, m, date_widget)

            m += 1
            date_widget = QtWidgets.QDateEdit(calendarPopup=True)
            date_widget.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
            date_widget.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
            date_widget.setKeyboardTracking(False)
            self.table_USGS_Sites.setCellWidget(i, m, date_widget)

            m += 1
            dateTime = QDateTimeEdit()
            dateTime.setDisplayFormat("hh:mm")
            #dateTime.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
            dateTime.setKeyboardTracking(False)
            dateTime.setFrame(False)
            tableProducts.setCellWidget(i, m, dateTime)

            m += 1
            dateTime = QDateTimeEdit()
            dateTime.setDisplayFormat("hh:mm")
            #dateTime.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
            dateTime.setKeyboardTracking(False)
            dateTime.setFrame(False)
            tableProducts.setCellWidget(i, m, dateTime)

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def USGS_updateSiteInfo(self, item):

        cameraIndex = self.USGS_listboxSites.currentRow()
        if (cameraIndex >= 0):
            strCamID    = self.USGS_listboxSites.currentItem().text()

            currentRow = self.USGS_listboxSites.currentRow()

            try:
                #cameraDictionary = self.myNIMS.getCameraDictionary()
                cameraList = self.myNIMS.getCameraList()
                self.USGS_listboxSites.clear()
                self.USGS_listboxSites.addItems(cameraList)
                self.listboxUSGSSiteInfo.clear()
                self.listboxUSGSSiteInfo.addItems(self.myNIMS.getCameraInfo(strCamID))

                self.USGS_listboxSites.setCurrentRow(currentRow)

                siteName = self.myNIMS.get_camId()

                nErrorCode, self.USGS_latestImage = self.myNIMS.getLatestImage(siteName)

                if nErrorCode == 404:
                    self.USGS_labelLatestImage.setText("No Images Available")
                else:
                    self.USGS_labelLatestImage.setPixmap(self.USGS_latestImage.scaled(self.USGS_labelLatestImage.size(),
                                                                                     QtCore.Qt.KeepAspectRatio,
                                                                                     QtCore.Qt.SmoothTransformation))
                self.table_USGS_Sites.setItem(0, 0, QTableWidgetItem(strCamID))
            except:
                pass

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def USGS_DisplayLatestImage(self):
        if self.USGS_latestImage != []:
            self.USGS_labelLatestImage.setPixmap(self.USGS_latestImage.scaled(self.USGS_labelLatestImage.size(),
                                                                         QtCore.Qt.KeepAspectRatio,
                                                                         QtCore.Qt.SmoothTransformation))

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def USGS_SiteClicked(self, item):
        USGSSiteIndex = self.USGS_updateSiteInfo(item)

        imageCount = self.USGS_getImageCount()

        self.table_USGS_Sites.setItem(0, 1, QTableWidgetItem(imageCount.__str__()))


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def pushButton_NEON_BrowseImageFolder_Clicked(self, item):
        # PROMPT USER FOR FOLDER INTO WHICH TO DOWNLOAD THE IMAGES/FILES
        folder =  promptlib.Files().dir()

        if os.path.exists(folder):
            self.edit_NEONSaveFilePath.setText(folder)
        else:
            os.makedirs(folder)

        JsonEditor().update_json_entry("NEON_Image_Folder", folder)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def pushButton_USGS_BrowseImageFolder_Clicked(self, item):
        # PROMPT USER FOR FOLDER INTO WHICH TO DOWNLOAD THE IMAGES/FILES
        folder =  promptlib.Files().dir()

        if os.path.exists(folder):
            self.edit_USGSSaveFilePath.setText(folder)
        else:
            os.makedirs(folder)

        JsonEditor().update_json_entry("USGS_Root_Folder", folder)


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def spinBoxOrbMaxFeaturesChanged(self):
        self.refreshImage()


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def clearoutImagePanels(self):
        self.labelOriginalImage.clear()
        self.labelEdgeImage.clear()


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def getImageFolder(self):
        imageFolder = self.lineEditHardDriveFolder.text()
        if len(imageFolder) > 0:
            if os.path.exists(imageFolder):
                return imageFolder
            else:
                return ''
        else:
            return ''

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def trainROI(self, roiParameters):
        global currentImage

        if currentImage:
            myGRIME_Color = GRIME_AI_Color()

            # CREATE AN ROI OBJECT
            roiObj = GRIME_roiData()

            # POPULATE ROI OBJECT WITH ROI INFORMATION
            if len(roiParameters.strROIName) > 0:
                roiObj.setROIName(roiParameters.strROIName)
            else:
                msgBox = GRIME_AI_QMessageBox('ROI Error', 'A name for the ROI is required!', buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()
                return

            # --------------------------------------------------
            rectROI = self.labelOriginalImage.getROI()

            if rectROI != None:
                roiObj.setDisplayROI(rectROI)
            else:
                msgBox = GRIME_AI_QMessageBox('ROI Error', 'Please draw the ROI on the image!', buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()
                return

            # --------------------------------------------------
            try:
                roiObj.setImageSize(currentImage.size())
                scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                roiObj.setDisplaySize(scaledCurrentImage.size())
                roiObj.calcROI()

                roiObj.setROIShape(ROIShape.RECTANGLE)
                #if self.radioButton_ROIShapeRectangle.isChecked():
                #roiObj.setROIShape(ROIShape.RECTANGLE)
                #else:
                #roiObj.setROIShape(ROIShape.ELLIPSE)
            except:
                msgBox = GRIME_AI_QMessageBox('ROI Error',
                                           'An unexpected error occurred calculating the ROI of the full resolution image!', buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()

                return

            # ----------------------------------------------------------------------------------------------------------
            # CALCULATE COLOR CLUSTERS FOR THE ROI AND SAVE THEM TO THE ROI LIST
            # ----------------------------------------------------------------------------------------------------------
            # EXTRACT THE ROI FROM THE ORIGINAL IMAGE
            img1 = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())
            rgb = extractROI(roiObj.getImageROI(), img1)

            roiObj.setNumColorClusters(roiParameters.numColorClusters)

            # EXTRACT DOMINANT RGB COLORS AND ADD THEM TO THE ROI OBJECT
            qImg, clusterCenters, hist = myGRIME_Color.KMeans(rgb, roiObj.getNumColorClusters())
            roiObj.setClusterCenters(clusterCenters, hist)

            # EXTRACT DOMINANT HSV COLORS AND ADD THEM TO THE ROI OBJECT
            hist, colorClusters = myGRIME_Color.extractDominant_HSV(rgb, roiObj.getNumColorClusters())
            roiObj.setHSVClusterCenters(colorClusters, hist)

            roiObj.setTrainingImageName(currentImageFilename)

            self.roiList.append(roiObj)

            # ----------------------------------------------------------------------------------------------------------
            # DISPLAY IN FEATURE TABLE
            # ----------------------------------------------------------------------------------------------------------
            # CREATE NEW ROW IN ROI TABLE
            nRow = self.tableWidget_ROIList.rowCount()
            self.tableWidget_ROIList.insertRow(nRow)

            #if (nRow == 0):
            #    numToAdd = 3
            #else:
            #    numToAdd = 1

            #for i in range(numToAdd):
            #    self.tableWidget_ROIList.insertRow(nRow)
            #    nRow += 1

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

            # ----------------------------------------------------------------------------------------------------------
            #
            # ----------------------------------------------------------------------------------------------------------
            global currentImageIndex

            processLocalImage(self, currentImageIndex)
            self.refreshImage()

            # ----------------------------------------------------------------------------------------------------------
            # ONCE AN ROI IS DEFINED FOR A SPECIFIC NUMBER OF COLOR CLUSTERS, DISABLE THE CONTROL SO THAT THE USER
            # CANNOT CHANGE THE VALUE FOR SUBSEQUENT TRAINED ROIs.
            # ----------------------------------------------------------------------------------------------------------
            #JES self.spinBoxColorClusters.setDisabled(True)

    # ==================================================================================================================
    #
    # ==================================================================================================================
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
            sum_en = sum(entropy)
        except:
            sum_en = 0.0

        #from scipy.stats import entropy
        #base = 2
        #H = entropy(hist, base=base)

        return sum_en[0]


    # ==================================================================================================================
    #
    # ==================================================================================================================
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


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def displayROIs(self):
        processLocalImage(self)
        self.refreshImage()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def NEON_FormatProductTableHeader(self):

        # HEADER TITLES
        headerList = ['Site', "Image Count", ' min Date ', ' max Date ', 'Start Date', 'End Date', 'Start Time', 'End Time']

        # DEFINE HEADER STYLE
        stylesheet = "::section{Background-color:rgb(116,175,80);border-radius:14px;}"

        # POINTER TO HEADER
        header = self.NEON_tableProducts.horizontalHeader()

        # DEFAULT HEADER SETTINGS
        header.setMinimumSectionSize(120)
        header.setDefaultSectionSize(140)
        header.setHighlightSections(False)
        header.setStretchLastSection(False)

        # INSERT TITLES INTO HEADER AND FORMAT HEADER
        # MAKE COLUMNS 1 THRU 'n' SIZE TO CONTENTS
        # MAKE COLUMN 0 STRETCH TO FILL UP REMAINING EMPTY SPACE IN THE TABLE
        for i, item in enumerate(headerList):
            headerItem = QTableWidgetItem(item)
            headerItem.setTextAlignment(QtCore.Qt.AlignCenter)
            self.NEON_tableProducts.setHorizontalHeaderItem(i, headerItem)
            self.NEON_tableProducts.setStyleSheet(stylesheet)

            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        header.setSectionResizeMode(0, QHeaderView.Stretch)

        # SET THE HEADER FONT
        font = QFont()
        font.setBold(True)
        self.NEON_tableProducts.horizontalHeader().setFont(font)
        self.NEON_tableProducts.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.NEON_tableProducts.resizeColumnsToContents()

        date_widget = QtWidgets.QDateEdit(QtCore.QDate(date.today().year, date.today().month, date.today().day))
        self.NEON_tableProducts.setCellWidget(1, 4, date_widget)
        self.NEON_tableProducts.setCellWidget(1, 5, date_widget)
        self.NEON_tableProducts.setCellWidget(1, 6, date_widget)
        self.NEON_tableProducts.setCellWidget(1, 7, date_widget)

        try:
            self.tableWidget_ROIList.horizontalHeader().setVisible(True)
        except:
             pass

    # ======================================================================================================================
    # THIS FUNCTION WILL UPDATE THE PRODUCT TABLE IN THE GUI WITH THE PRODUCTS THAT ARE AVAILABLE FOR A SPECIFIC SITE.
    # ======================================================================================================================
    def NEON_FormatProductTable(self, tableProducts):
        maxRows = 1

        # JES: MUST MAKE CODE DYNAMIC TO ONLY DELETE UNSELECTED ITEMS
        for i in range(tableProducts.rowCount()):
            tableProducts.removeRow(0)

        tableProducts.insertRow(0)

        for i in range(maxRows):
            m = 0
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

    # ==================================================================================================================
    #
    # ==================================================================================================================
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

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def ROIShapeClicked(self):
        if self.radioButton_ROIShapeRectangle.isChecked() == True:
            self.labelOriginalImage.setROIShape(ROIShape.RECTANGLE)
        elif self.radioButton_ROIShapeEllipse.isChecked() == True:
            self.labelOriginalImage.setROIShape(ROIShape.ELLIPSE)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def USGS_getImageCount(self):
        currentRow = self.USGS_listboxSites.currentRow()
        if (currentRow > -1):
            site = self.USGS_listboxSites.currentItem().text()

        try:
            startDateCol = 4
            nYear, nMonth, nDay = self.separateDate(self.table_USGS_Sites.cellWidget(0, startDateCol).date())
            startDate = datetime.date(nYear, nMonth, nDay)

            endDateCol = 5
            nYear, nMonth, nDay = self.separateDate(self.table_USGS_Sites.cellWidget(0, endDateCol).date())
            endDate = datetime.date(nYear, nMonth, nDay)

            startTimeCol = 6
            nHour, nMinute, nSecond = self.separateTime(self.table_USGS_Sites.cellWidget(0, startTimeCol).dateTime().time())
            startTime = datetime.time(nHour, nMinute, nSecond)

            endTimeCol = 7
            nHour, nMinute, nSecond = self.separateTime(self.table_USGS_Sites.cellWidget(0, endTimeCol).dateTime().time())
            endTime = datetime.time(nHour, nMinute, nSecond)

            nwisID = self.myNIMS.get_nwisID()

            imageCount = self.myNIMS.getImageCount(siteName=site, nwisID=nwisID, startDate=startDate, endDate=endDate, startTime=startTime, endTime=endTime)
        except:
            imageCount = 0

        return imageCount

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def pushButton_USGSDownloadClicked(self):

        # VERIFY THAT THE FOLDER HAS BEEN SPECIFIED
        USGS_download_file_path = self.edit_USGSSaveFilePath.text()
        JsonEditor().update_json_entry("USGS_Root_Folder", USGS_download_file_path)

        if len(USGS_download_file_path) == 0:
            strMessage = 'A download folder has not been specified. Would you like to use the last GRIME-AI USGS download folder used?'
            msgBox = GRIME_AI_QMessageBox('USGS Root Download Folder', strMessage, QMessageBox.Yes | QMessageBox.No)
            response = msgBox.displayMsgBox()

            if response == QMessageBox.Yes:
                #USGS_download_file_path = os.path.expanduser('~')
                #USGS_download_file_path = os.path.join(USGS_download_file_path, 'Documents')
                #USGS_download_file_path = os.path.join(USGS_download_file_path, 'GRIMe-AI')

                USGS_download_file_path = JsonEditor().getValue("USGS_Root_Folder")

                if not os.path.exists(USGS_download_file_path):
                    os.makedirs(USGS_download_file_path)
                #NEON_download_file_path = os.path.join(USGS_download_file_path, 'Downloads')
                #if not os.path.exists(USGS_download_file_path):
                #    os.makedirs(USGS_download_file_path)

                self.edit_USGSSaveFilePath.setText(USGS_download_file_path)
                JsonEditor().update_json_entry("USGS_Root_Folder", USGS_download_file_path)
        else:
            # MAKE SURE THE PATH EXISTS. IF IT DOES NOT, THEN CREATE IT.
            if not os.path.exists(USGS_download_file_path):
                os.makedirs(USGS_download_file_path)

        currentRow = self.USGS_listboxSites.currentRow()

        if currentRow >= 0:
            site = self.USGS_listboxSites.currentItem().text()

            startDateCol = 4
            nYear, nMonth, nDay = self.separateDate(self.table_USGS_Sites.cellWidget(0, startDateCol).date())
            startDate = datetime.date(nYear, nMonth, nDay)

            endDateCol = 5
            nYear, nMonth, nDay = self.separateDate(self.table_USGS_Sites.cellWidget(0, endDateCol).date())
            endDate = datetime.date(nYear, nMonth, nDay)

            startTimeCol = 6
            nHour, nMinute, nSecond = self.separateTime(self.table_USGS_Sites.cellWidget(0, startTimeCol).dateTime().time())
            startTime = datetime.time(nHour, nMinute, nSecond)

            endTimeCol = 7
            nHour, nMinute, nSecond = self.separateTime(self.table_USGS_Sites.cellWidget(0, endTimeCol).dateTime().time())
            endTime = datetime.time(nHour, nMinute, nSecond)

            nwisID = self.myNIMS.get_nwisID()

            #downloadsFilePath = os.path.join(self.edit_USGSSaveFilePath.text(), 'Images')
            downloadsFilePath = self.edit_USGSSaveFilePath.text()
            if not os.path.exists(downloadsFilePath):
                os.makedirs(downloadsFilePath)

            saveFolder = os.path.join(downloadsFilePath, "Images")
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            self.myNIMS.downloadImages(siteName=site, nwisID=nwisID, saveFolder=saveFolder, startDate=startDate, endDate=endDate, startTime=startTime, endTime=endTime)

            saveFolder = os.path.join(downloadsFilePath, "Data")
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            self.myNIMS.fetchStageAndDischarge(nwisID, site, startDate, endDate, startTime, endTime, saveFolder)

            #fetchUSGSImages(self.table_USGS_Sites, self.edit_USGSSaveFilePath)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def separateDate(self, date):
        nYear     = date.year()
        nMonth    = date.month()
        nDay      = date.day()

        return nYear, nMonth, nDay

    def separateTime(self, time):
        nHour   = time.hour()
        nMinute = time.minute()
        nSecond = time.second()

        return nHour, nMinute, nSecond

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def pushButtonTrainGoodTriggered(self):
        blur, intensity = GRIME_AI_ImageTriage.computeBlurAndBrightness(self.spinBoxShiftSize.value())
        imageStats = GRIMe_ImageStats()
        imageStats.setBlurValue(blur)
        imageStats.setBrightnessValue(intensity)
        imageStats.setLabel('good')
        self.imageStatsList.append(imageStats)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def pushButtonTrainBadTriggered(self):
        blur, intensity = GRIME_AI_ImageTriage.computeBlurAndBrightness(self.spinBoxShiftSize.value())
        imageStats = GRIMe_ImageStats()
        imageStats.setBlurValue(blur)
        imageStats.setBrightnessValue(intensity)
        imageStats.setLabel('bad')
        self.imageStatsList.append(imageStats)

    # ======================1===============================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def menubar_RefreshNEON(self):
        # INITIALIZE GUI CONTROLS
        # frame.NEON_listboxSites.setCurrentRow(1)

        # GET LIST OF ALL SITES ON NEON
        # if frame.checkBoxNEONSites.isChecked():
        myNEON_API = NEON_API()
        siteList = myNEON_API.readFieldSiteTable()
        # else:
        # NEON_FormatProductTable(frame.tableProducts)

        if len(siteList) == 0:
            pass
            # frame.radioButtonHardDriveImages.setChecked(True)
            # frame.radioButtonHardDriveImages.setDisabled(False)
        # IF THERE ARE FIELD SITE TABLES AVAILABLE, ENABLE GUI WIDGETS PERTAINING TO WEB SITE DATA/IMAGES
        else:
            myList = []

            for site in siteList:
                strSiteName = site.siteID + ' - ' + site.siteName
                myList.append(strSiteName)

            self.NEON_listboxSites.addItems(myList)

            # JES - TEMPORARILY SET BARCO LAKE AS THE DEFAULT SELECTION
            try:
                self.NEON_listboxSites.setCurrentRow(2)
                self.NEON_listboxSites.show()
                self.NEON_SiteClicked(2)

            except:
                pass

        print("Initialize USGS product table...")
        self.USGS_InitProductTable()
        self.USGS_FormatProductTable(self.table_USGS_Sites)
        self.NEON_FormatProductTableHeader()

        self.show()


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubarCompositeSlices(self):
        global dailyImagesList
        global imageFileFolder

        if len(dailyImagesList.getVisibleList()) == 0:
            strMessage = 'You must first create a list of images to operate on. Use the FETCH files feature of GRIME AI.'
            msgBox = GRIME_AI_QMessageBox('Composite Slice Error', strMessage, QMessageBox.Close)
            response = msgBox.displayMsgBox()
        else:
            imageFilename = dailyImagesList.getVisibleList()[0]
            self.compositeSliceDlg = GRIME_CompositeSliceDlg()

            self.compositeSliceDlg.compositeSliceGenerateSignal.connect(self.generateCompositeSlices)
            self.compositeSliceDlg.compositeSliceCancelSignal.connect(self.closeCompositeSlices)

            imageFilename = dailyImagesList.getVisibleList()[0].fullPathAndFilename
            self.compositeSliceDlg.loadImage(imageFilename)

            self.compositeSliceDlg.label_Image.setDrawingMode(DrawingMode.SLICE)

            self.compositeSliceDlg.show()


    def generateCompositeSlices(self):
        print("Generating composite slices image(s)...")

        global imageFileFolder
        if not os.path.exists(imageFileFolder+'\compositeSlices'):
            os.makedirs(imageFileFolder+'\compositeSlices')

        widthMultiplier, heightMultiplier, sliceCenter, sliceWidth = self.compositeSliceDlg.getMultipliers()

        actualSliceCenter = self.compositeSliceDlg.getSliceCenter() * widthMultiplier

        compositeSlices = GRIME_AI_CompositeSlices(actualSliceCenter, sliceWidth)
        compositeSlices.create_composite_image(dailyImagesList.visibleList, imageFileFolder+'\compositeSlices')

    def closeCompositeSlices(self):
        if self.compositeSliceDlg != None:
            self.compositeSliceDlg.close()
            self.compositeSliceDlg    = None


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def create_masks(self, coco_annotation_file, image_dir, output_dir):
        print("Extract masks from COCO file...")

        # Load COCO annotations
        coco = COCO(coco_annotation_file)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get all image ids
        img_ids = coco.getImgIds()

        for img_id in img_ids:
            # Load image info
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(image_dir, img_info['file_name'])

            # Load image
            image = cv2.imread(img_path)
            height, width, _ = image.shape

            # Create an empty mask
            mask = np.zeros((height, width), dtype=np.uint8)

            # Get annotation ids for the image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                # Get binary mask for the annotation
                rle = coco.annToRLE(ann)
                binary_mask = maskUtils.decode(rle)

                # Combine binary mask with the main mask
                mask = np.maximum(mask, binary_mask * 255)

            # Save the mask
            mask_path = os.path.join(output_dir, f"{img_info['file_name'].split('.')[0]}_mask.png")
            cv2.imwrite(mask_path, mask)


    def menubarExtractCOCOMasks(self):
        self.COCOdlg = GRIME_AI_ExportCOCOMasksDlg(self)

        self.COCOdlg.COCO_signal_ok.connect(self.accepted_COCODlg)
        self.COCOdlg.COCO_signal_cancel.connect(self.rejected_COCODlg)

        self.COCOdlg.show()


    def accepted_COCODlg(self):
        coco_annotation_file = self.COCOdlg.getAnnotationFile()
        image_dir = self.COCOdlg.getAnnotationImagesFolder()
        output_dir = os.path.join(image_dir, "masks")

        self.create_masks(coco_annotation_file, image_dir, output_dir)

    def rejected_COCODlg(self):
        pass

    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubarSaveSettings(self):
        utils = GRIME_AI_Save_Utils()
        utils.saveSettings()


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
        msgBox = GRIME_AI_QMessageBox('Water Level Measurement', strMessage, QMessageBox.Close)
        response = msgBox.displayMsgBox()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def toolbarButtonEdgeDetection(self):
        self.edgeDetectionDlg = GRIME_AI_EdgeDetectionDlg(frame)

        self.edgeDetectionDlg.edgeDetectionSignal.connect(self.edgeDetectionMethod)
        self.edgeDetectionDlg.featureDetectionSignal.connect(self.featureDetectionMethod)

        self.edgeDetectionDlg.show()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    # Function to get prompts for each image
    def get_prompts(self, image_name):
        # Customize this function to return prompts based on the image name or content
        # For example:
        if 'cat' in image_name:
            return {'texts': ['cat']}
        elif 'dog' in image_name:
            return {'texts': ['dog']}
        # Add more conditions as needed
        return {'texts': ['default object description']}  # Default prompt


    # ======================================================================================================================
    #
    # ======================================================================================================================
    '''
    def tuneSAM(self):
        import torch
        from segment_anything import SamPredictor, sam_model_registry, SamTrainer
        from pycocotools.coco import COCO

        # Load your labeled images in COCO format
        coco = COCO('instances_default.json')

        #sam = sam_model_registry"<model_type>"
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h"
        sam = sam_model_registry[MODEL_TYPE]
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)

        # Prepare your dataset
        # Make sure to organize your dataset in the format expected by the SAM model
        # Typically, this involves having a directory with images and another with annotations

        # Define the training parameters
        train_params = {
            "batch_size": 4,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True
        }

        # Initialize the trainer
        trainer = SamTrainer(sam, train_params)

        # Train the model
        trainer.train(coco, epochs=100)

        # Save the trained model
        torch.save(sam.state_dict(), 'my_sam_vit_h_4b8939.pth')
    '''

    '''
        # CHECK TO SEE IF THE COMPUTER HAS A GPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. LOAD AND NORMALIZE CIFAR10
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 4

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        net = Net()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. DEEP LEARNING: DEFINE A LOSS FUNCTION AND OPTIMIZER
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4a. DEEP LEARNING: TRAIN THE NETWORK
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4b. DEEP LEARNING: SAVE THE TRAINED MODEL
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        PATH = './cifar_net.pth'
        torch.save(net.state_dict(), PATH)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5a. DEEP LEARNING: TEST THE NETWORK ON THE TEST DATA
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        dataiter = iter(testloader)
        images, labels = next(dataiter)

        # print images
        self.imshow( torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5b. DEEP LEARNING: RELOAD THE TEST DATA
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        net = Net()
        net.load_state_dict(torch.load(PATH))

        # LET'S SEE WHAT THE NEURAL NET CLASSIFIES THE IMAGES AS
        outputs = net(images)

        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                      for j in range(4)))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # CHECK TO SEE HOW THE NEURAL NET PERFORMS ON THE WHOLE DATA SET
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # WHICH CLASSES PERFORMED WELL AND WHICH CLASSES PERFORMED POORLY
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    '''

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def edgeDetectionMethod(self, edgeMethod):
        global g_edgeMethodSettings
        g_edgeMethodSettings = edgeMethod

        processLocalImage(self)

        self.refreshImage()

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def featureDetectionMethod(self, featureMethod):
        global g_edgeMethodSettings
        global g_featureMethodSettings

        g_edgeMethodSettings.method = edgeMethodsClass.NONE

        g_featureMethodSettings = featureMethod

        processLocalImage(self)

        self.refreshImage()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def onMyToolBarFileFolder(self):
        self.fileFolderDlg = GRIME_AI_FileUtilitiesDlg(frame)

        self.fileFolderDlg.create_composite_slice_signal.connect(self.menubarCompositeSlices)
        self.fileFolderDlg.triage_images_signal.connect(toolbarButtonImageTriage_1)

        self.fileFolderDlg.accepted.connect(self.closeFilefolderDlg)
        self.fileFolderDlg.rejected.connect(self.closeFilefolderDlg)

        self.fileFolderDlg.show()

        try:
            global gFrameCount
            self.imageNavigationDlg.setImageCount(gFrameCount)
            self.imageNavigationDlg.reset()
        except:
            pass


    # ------------------------------------------------------------------------------------------
    def closeFilefolderDlg(self):
            pass


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
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
                msgBox = GRIME_AI_QMessageBox('Image Navigation', strMessage, QMessageBox.Close)
                response = msgBox.displayMsgBox()

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def getImageIndex(self, imageIndex):
        global currentImageIndex

        currentImageIndex = imageIndex

        processLocalImage(self, imageIndex)
        self.refreshImage()

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def closeNavigationDlg(self):
        del self.imageNavigationDlg

        self.imageNavigationDlg = None


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    #@pyqtSlot()
    def onMyToolBarColorSegmentation(self):
        if self.colorSegmentationDlg == None:
            if self.maskEditorDlg == None:
                self.labelOriginalImage.setDrawingMode(DrawingMode.COLOR_SEGMENTATION)

                self.colorSegmentationDlg = GRIME_ColorSegmentationDlg()

                self.colorSegmentationDlg.colorSegmentation_Signal.connect(self.colorSegmentation)
                self.colorSegmentationDlg.addROI_Signal.connect(self.trainROI)
                self.colorSegmentationDlg.deleteAllROI_Signal.connect(self.deleteAllROI)
                self.colorSegmentationDlg.buildFeatureFile_Signal.connect(self.buildFeatureFile)
                self.colorSegmentationDlg.universalTestButton_Signal.connect(self.universalTestButton)

                self.colorSegmentationDlg.close_signal.connect(self.closeColorSegmentationDlg)
                self.colorSegmentationDlg.accepted.connect(self.closeColorSegmentationDlg)
                self.colorSegmentationDlg.rejected.connect(self.closeColorSegmentationDlg)

                self.colorSegmentationDlg.show()
            else:
                strMessage = 'Please close the Mask Editor toolbox if you want to use the Mask Editor toolbox.\nThis will be resolved in a future design change.'
                msgBox = GRIME_AI_QMessageBox('Tool Conflict', strMessage, QMessageBox.Yes | QMessageBox.No)
                response = msgBox.displayMsgBox()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def closeColorSegmentationDlg(self):
        if self.colorSegmentationDlg != None:
            self.getColorSegmentationParams()

            self.colorSegmentationDlg.close()
            del self.colorSegmentationDlg
            self.colorSegmentationDlg = None

        self.labelOriginalImage.setDrawingMode(DrawingMode.OFF)


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def colorSegmentation(self, int):
        global dailyImagesList
        videoFileList = dailyImagesList.getVisibleList()

        myGRIMe_Color = GRIME_AI_Color()

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


    # ==================================================================================================================
    # ==================================================================================================================
    # IMAGE MASK FUNCTIONALITY
    # ==================================================================================================================
    # ==================================================================================================================
    def onMyToolBarBuildModel(self):

        if self.buildModelDlg == None:
            self.buildModelDlg = GRIME_AI_buildModelDlg()

            self.buildModelDlg.rejected.connect(self.buildModelDialogClose)

            self.buildModelDlg.save_model_masks_signal.connect(self.saveModelMasksChanged)
            self.buildModelDlg.save_original_model_image_signal.connect(self.saveOriginalModelImageChanged)
            self.buildModelDlg.segment_image_signal.connect(self.segment_image_clicked)
            self.buildModelDlg.tune_model_signal.connect(self.tune_model_clicked)

            self.buildModelDlg.show()

            global g_modelSettings
            g_modelSettings.saveModelMasks = self.buildModelDlg.getSaveModelMasks()
            g_modelSettings.saveOriginalModelImage = self.buildModelDlg.getMoveOriginalImage()


    # ------------------------------------------------------------------------------------------------------------------
    def buildModelDialogClose(self):
        global g_modelSettings
        g_modelSettings.saveModelMasks = self.buildModelDlg.getSaveModelMasks()
        g_modelSettings.saveOriginalModelImage = self.buildModelDlg.getMoveOriginalImage()

        # self.buildModelDlg.close()

        del self.buildModelDlg
        self.buildModelDlg = None


    # ------------------------------------------------------------------------------------------------------------------
    def saveModelMasksChanged(self, bSave):
        global g_modelSettings
        g_modelSettings.saveModelMasks = bSave


    # ------------------------------------------------------------------------------------------------------------------
    def saveOriginalModelImageChanged(self, bSave):
        global g_modelSettings
        g_modelSettings.saveOriginalModelImage = bSave


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def tune_model_clicked(self):

        global g_modelSettings
        g_modelSettings.saveModelMasks = self.buildModelDlg.getSaveModelMasks()
        g_modelSettings.saveOriginalModelImage = self.buildModelDlg.getMoveOriginalImage()
        g_modelSettings.model_file = self.buildModelDlg.get_selected_model_path_and_filename()

        if self.buildModelDlg == None:
            self.buildModelDlg.close()
            del self.buildModelDlg
            self.buildModelDlg = None

        if g_modelSettings.model_file:
            self.myDeepLearning(g_modelSettings)

        self.GRIME_AI_tune_sam2_model()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def segment_image_clicked(self):

        global progress_bar_closed

        def on_progress_bar_closed(obj):
            global progress_bar_closed
            progress_bar_closed = True

        progressBar = QProgressWheel()
        progressBar.destroyed.connect(on_progress_bar_closed)
        progress_bar_closed = False
        progressBar.show()

        if 0:
            # Clear the global Hydra instance if it's already initialized
            if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
                hydra.core.global_hydra.GlobalHydra.instance().clear()

            config_dir = "./sam2/sam2/configs/sam2.1"   # RELATIVE PATH
            # CONVERT TO ABSOLUTE PATH, IF NEEDED
            if 0:
                dirname = os.path.dirname(__file__)
                config_dir = os.path.join(dirname, config_dir)
                config_dir = os.path.normpath(config_dir)

            with initialize(config_path=config_dir):
                cfg = compose(config_name="sam2.1_hiera_l.yaml")
                print(OmegaConf.to_yaml(cfg))

        #JES - INVESTIGATE - WHY DOES IT ONLY WORK WITH AN ABSOLUTE FILE PATH???
        dirname = os.path.dirname(__file__)
        model_cfg = os.path.join(dirname, "sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_l.yaml")
        model_cfg = os.path.normpath(model_cfg)
        print(model_cfg)

        sam2_checkpoint = os.path.join(dirname, "sam2\\checkpoints\\sam2.1_hiera_large.pt")
        sam2_checkpoint = os.path.normpath(sam2_checkpoint)
        print(sam2_checkpoint)

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(DEVICE)

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE, mode='eval')

        predictor = SAM2ImagePredictor(sam2_model)

        # from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        # CHECKPOINT_PATH = '..\\..\\models\\sam_vit_h_4b8939.pth'
        # MODEL_TYPE = "vit_h"
        # sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        # sam.to(device=DEVICE)
        # predictor = SAM2ImagePredictor(sam)

        input_point = np.array([[619, 396]])
        input_label = np.array([1])

        model_filename = self.buildModelDlg.get_selected_model_path_and_filename()
        predictor.model.load_state_dict(torch.load(model_filename, map_location=torch.device(DEVICE)))

        if 0:
            predictor.model.load_state_dict(torch.load(".\\models\\model_20 (DK Grand Island)-002.torch"))

        if 0:
            folder = self.buildModelDlg.get_images_folder()
            images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            for image in images:
                image = Image.open(f"{folder}\\{image}").convert("RGB")
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                show_points(input_point, input_label, plt.gca())
                plt.axis('on')
                plt.show()

        folder = self.buildModelDlg.get_segmentation_images_folder()
        images = [f for f in os.listdir(folder) if f.endswith('.jpg')]

        progressBar.setRange(0, len(images) + 1)

        for image_index, image in enumerate(images):
            if progress_bar_closed is False:
                progressBar.setWindowTitle(image)
                progressBar.setValue(image_index)

                predictor.set_image(np.array(Image.open(f"{folder}\\{image}").convert("RGB")))

                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]

                img = cv2.imread(f"{folder}\\{image}", cv2.COLOR_BGR2RGB)
                masked_images, masks_only = self.show_masks(img, masks, scores, borders=True)

                for mask_index, composite_mask_and_image in enumerate(masked_images):
                    filename_only = Path(f"{folder}\\{image}").stem

                    output_folder = f"{folder}\\mask_overlays"
                    os.makedirs(output_folder, exist_ok=True)

                    filename = f"{filename_only}_mask_overlay_{mask_index}.jpg"
                    output_file = os.path.join(output_folder, filename)
                    cv2.imwrite(output_file, np.array(cv2.cvtColor(composite_mask_and_image, cv2.COLOR_RGB2BGR)))

                    # SAVE EACH MASK AS A SEPARATE FILE IF SELECTED
                    if self.buildModelDlg.getSaveModelMasks():
                        mask_filename = f"{filename_only}_mask_only_{mask_index}.jpg"
                        output_file = os.path.join(output_folder, mask_filename)
                        cv2.imwrite(output_file, np.array(cv2.cvtColor(masks_only[mask_index], cv2.COLOR_RGB2BGR)))

            else:
                strMessage = 'You have cancelled the image segmentation currently in-progress. Not all images have been segmented.'
                msgBox = GRIME_AI_QMessageBox('Image Segmentation Terminated', strMessage, QMessageBox.Close)
                response = msgBox.displayMsgBox()
                break

        # close the progressBar only if the user did not close it (i.e., terminated image segmentation)
        if progress_bar_closed is False:
            progressBar.close()
        del progressBar


    def show_masks(self, image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        """
        Display masks on an image with optional points and boxes.

        Args:
            image (ndarray): The image on which to display the masks.
            masks (list of ndarray): A list of masks to display.
            scores (list of float): A list of scores corresponding to each mask.
            point_coords (list of tuple, optional): Coordinates of points to display. Defaults to None.
            box_coords (list of tuple, optional): Coordinates of boxes to display. Defaults to None.
            input_labels (list of int, optional): Labels for the points. Required if point_coords is provided. Defaults to None.
            borders (bool, optional): Whether to display borders around masks. Defaults to True.

        Raises:
            AssertionError: If point_coords is provided without input_labels.

        """
        masked_images = []
        masks_only = []

        for i, (mask, score) in enumerate(zip(masks, scores)):
            #JES plt.figure(figsize=(10, 10))
            #JES plt.imshow(image)
            #JES mask_image = self.show_mask(mask, plt.gca(), borders=borders)
            mask_image = self.show_mask(mask, borders=borders)

            if point_coords is not None:
                assert input_labels is not None
                show_points(point_coords, input_labels, plt.gca())

            if box_coords is not None:
                # boxes
                show_box(box_coords, plt.gca())

            #JES if len(scores) > 1:
            #JES     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            #JES plt.axis('off')
            #JES plt.show()

            # Overlay the mask on the original image
            # Ensure the mask has an alpha channel
            import cv2
            # Resize the mask to match the size of the original image
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            color = np.array([30, 144, 255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_np = np.array(mask_image)
            mask_np = mask_np.astype(int)
            mask_cv2 = mask_np
            mask_cv2 = mask_cv2.astype(np.float32)
            masks_only.append(mask_cv2)

            image_np = np.array(image)
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGRA)
            image_cv2 = image_cv2.astype(np.float32)

            combined = cv2.addWeighted(image_cv2, 1, mask_cv2, 0.5, 0)

            masked_images.append(combined)

        return masked_images, masks_only


    """
    Module for displaying masks on images.

    This module provides a function to display a mask on an image with optional random coloring
    and border drawing.

    Functions:
        show_mask(mask, ax, random_color=False, borders=True): Displays a mask on an image.

    Example:
        fig, ax = plt.subplots()
        mask = np.array([[0, 1], [1, 0]])
        show_mask(mask, ax, random_color=True, borders=True)

    Author:
        Your Name

    Date:
        YYYY-MM-DD
    """

    #JES def show_mask(self, mask, ax, random_color=False, borders=True):
    def show_mask(self, mask, random_color=False, borders=True):

        """
        Displays a mask on an image.

        Args:
            mask (np.ndarray): The mask to be displayed.
            ax (matplotlib.axes.Axes): The axes on which to display the mask.
            random_color (bool, optional): If True, use a random color for the mask. Defaults to False.
            borders (bool, optional): If True, draw borders around the mask. Defaults to True.

        Returns:
            None

        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

        return mask_image


    """
    Module for displaying points on a plot.

    This module provides a function to display positive and negative points on a plot
    with different colors and markers.

    Functions:
        show_points(coords, labels, ax, marker_size=375): Displays positive and negative points on a plot.

    Example:
        fig, ax = plt.subplots()
        coords = np.array([[1, 2], [3, 4], [5, 6]])
        labels = np.array([1, 0, 1])
        show_points(coords, labels, ax)

    Author:
        Your Name

    Date:
        YYYY-MM-DD
    """

    def show_points(self, coords, labels, ax, marker_size=375):
        """
        Displays positive and negative points on a plot.

        Args:
            coords (np.ndarray): Array of coordinates for the points.
            labels (np.ndarray): Array of labels for the points (1 for positive, 0 for negative).
            ax (matplotlib.axes.Axes): The axes on which to display the points.
            marker_size (int, optional): Size of the markers. Defaults to 375.

        Returns:
            None

        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)


    # ==================================================================================================================
    # https: // pytorch.org / tutorials / beginner / blitz / cifar10_tutorial.html
    # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    # https://www.cs.toronto.edu/~kriz/cifar.html
    # ==================================================================================================================
    #JES def myDeepLearning(self, modelSettings):

        '''
        # THIS IS THE OLD CODE THAT HAS BEEN ENCAPULATED INTO A CLASS. THE NEW CODE NEEDS TO BE ENAPSULATED INTO THIS
        # CLASS A.S.A.P!
        #self.tuneSAM()
        DL = GRIME_AI_DeepLearning()
        DL.SAM_001(modelSettings, dailyImagesList)
        #self.SAM_002()
        '''


    def GRIME_AI_tune_sam2_model(self):
        training_images_folder = [os.path.normpath(self.buildModelDlg.get_training_images_folder())]

        annotation_file = [self.buildModelDlg.get_annotation_filename()]

        all_images, all_annotations = self.load_images_and_annotations(training_images_folder, annotation_file)

        train_images, val_images, test_images, annotations = self.split_dataset(all_images, all_annotations)

        input_point = np.array([[1254, 934]])
        input_label = np.array([1])

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(DEVICE)

        sam2_checkpoint = "CheckPoints\\sam2.1_hiera_large.pt"
        model_cfg = "configs\\sam2.1\\sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
        predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder

        model = SAM2FullModel(predictor.model)

        model.to(DEVICE)

        optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=0.0001, weight_decay=0.001)

        now = datetime.now()
        print(now)
        self.GRIME_AI_train_sam(sam2_model, predictor, train_images, annotations, optimizer, val_images, annotations, input_point, input_label,
                  epochs=30, modelSettings=g_modelSettings)
        now = datetime.now()
        print(now)

        now = datetime.now()
        formatted_time = now.strftime('%d%m_%H%M')
        plt.plot(self.epoch_list, self.loss_values, marker='*')
        plt.title('Epoch vs loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.savefig("EpochVsLoss_{}.png".format(formatted_time))

        predictor.model.load_state_dict(torch.load("model_20.torch"))

        num_classes = 2
        all_true_labels = []
        all_predicted_labels = []

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        categories = [
            {"id": 1, "name": "object"}
        ]
        coco_data["categories"].extend(categories)

        image_id = 0
        annotation_id = 0

        output_dir = Path("output_annotations")
        output_dir.mkdir(exist_ok=True)

        for idx, image_file in enumerate(test_images):

            image = np.array(Image.open(image_file).convert("RGB"))

            height, width = image.shape[:2]
            image_info = {
                "file_name": os.path.basename(image_file),
                "height": height,
                "width": width,
                "id": image_id
            }
            coco_data["images"].append(image_info)
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(multimask_output=False)

            if masks.size > 0:
                mask = masks[np.argmax(scores)]
                mask_tensor = torch.tensor(mask, dtype=torch.uint8)

                true_mask = load_true_mask(image_file, annotations)

                true_labels = true_mask.flatten()
                predicted_labels = mask_tensor.flatten()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_labels)

                pos = np.where(mask)
                xmin = int(np.min(pos[1]))
                xmax = int(np.max(pos[1]))
                ymin = int(np.min(pos[0]))
                ymax = int(np.max(pos[0]))
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                segmentation = mask.ravel().tolist()

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": int(np.sum(mask)),
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)

                annotation_id += 1

            image_id += 1

        output_file = output_dir / "annotations_coco_format8.json"
        with open(output_file, "w") as f:
            json.dump(coco_data, f, indent=4)

        print(f"COCO annotations saved to {output_file}")

    def GRIME_AI_train_sam(self, sam2_model, predictor, train_images, annotations, optimizer, val_images, val_annotations,
                  input_point, input_label, epochs=20, modelSettings=g_modelSettings):
        """
        Train the SAM model using the provided training images and annotations.

        Args:
            predictor (SAM2ImagePredictor): The predictor object for the SAM model.
            train_images (list of str): List of file paths to the training images.
            annotations (dict): A dictionary containing image and annotation data.
            optimizer (torch.optim.Optimizer): The optimizer for training the model.
            val_images (list of str, optional): List of file paths to the validation images. Defaults to None.
            val_annotations (dict, optional): A dictionary containing validation image and annotation data. Defaults to None.
            input_point (list of tuple): List of input points for mask prediction.
            input_label (list of int): List of labels corresponding to the input points.
            epochs (int, optional): Number of training epochs. Defaults to 20.

        Returns:
            None

        Raises:
            ValueError: If an image file is not found in the annotations.

        """
        ECHO = 0
        epochs = 50

        sam2_model.train()
        predictor = SAM2ImagePredictor(sam2_model)

        for epoch in range(epochs):
            self.epoch_list.append(epoch + 1)
            epoch_loss = 0.0
            if ECHO:
                print(f"Epoch {epoch + 1}/{epochs}")

            np.random.shuffle(train_images)
            loss_fn = nn.BCEWithLogitsLoss()

            for idx, image_file in enumerate(train_images):
                image = np.array(Image.open(image_file).convert("RGB"))
                true_mask = self.load_true_mask(image_file, annotations)

                if true_mask is None:
                    print(f"No annotation found for image {image_file}, skipping.")
                    continue

                predictor.set_image(image)

                # Prepare prompts for mask prediction
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label,
                                                                                        box=None, mask_logits=None,
                                                                                        normalize_coords=True)
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels),
                    boxes=None,
                    masks=None
                )

                # Mask decoder prediction
                batched_mode = unnorm_coords.shape[0] > 1  # multi-object prediction
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in
                                     predictor._features["high_res_feats"]]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks,
                                                                    predictor._orig_hw[-1])  # Upscale masks

                # Loss calculations
                gt_mask = torch.tensor(true_mask.astype(np.float32)).cuda().unsqueeze(0)
                prd_mask = torch.sigmoid(prd_masks[:, 0]).unsqueeze(0)

                try:
                    # Segmentation Loss
                    seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log(
                        (1 - prd_mask) + 0.00001)).mean()

                    # Score Loss (IOU)
                    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                    iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                    # Combine losses
                    loss = seg_loss + score_loss * 0.05

                    # Backpropagation with mixed precision
                    optimizer.zero_grad()
                    with autocast():
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()

                    epoch_loss += loss.detach().item()
                    if ECHO:
                        print(f"Image {idx + 1}/{len(train_images)} processed. Loss: {loss.item()}")
                except:
                    print("Something is wrong with the tensors!")

            # Average epoch loss
            avg_epoch_loss = epoch_loss / len(train_images)
            self.loss_values.append(avg_epoch_loss)

            if ECHO:
                print(f"Epoch {epoch + 1} Training Loss: {avg_epoch_loss}")

            # Validation step
            if val_images is not None and val_annotations is not None:
                val_loss = 0.0
                with torch.no_grad():
                    for val_idx, val_image_file in enumerate(val_images):
                        val_image = np.array(Image.open(val_image_file).convert("RGB"))
                        val_true_mask = self.load_true_mask(val_image_file, val_annotations)

                        if val_true_mask is None:
                            if ECHO:
                                print(f"No annotation found for validation image {val_image_file}, skipping.")
                            continue

                        predictor.set_image(val_image)
                        masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label,
                                                             multimask_output=False)

                        if masks.size > 0:
                            best_mask = masks[np.argmax(scores)]
                            # best_mask_tensor = torch.tensor(best_mask, dtype=torch.float32, attn_implementation="flash_attention_2").unsqueeze(0).to("cuda")
                            # val_true_mask_tensor = torch.tensor(val_true_mask, dtype=torch.float32, attn_implementation="flash_attention_2").unsqueeze(0).to("cuda")Salt Lake City, Utah
                            best_mask_tensor = torch.tensor(best_mask, dtype=torch.float32).unsqueeze(0).to("cpu")
                            val_true_mask_tensor = torch.tensor(val_true_mask, dtype=torch.float32).unsqueeze(0).to(
                                "cpu")

                            val_loss += loss_fn(best_mask_tensor, val_true_mask_tensor).item()

                avg_val_loss = val_loss / len(val_images)
                if ECHO:
                    print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss}")
        torch.save(predictor.model.state_dict(), "model_20.torch")


    """
    Module for loading images and annotations.

    This module provides a function to load images and their corresponding annotations
    from specified folders and annotation files.

    Functions:
        load_images_and_annotations(folders, annotation_files): Loads images and annotations from the given folders and annotation files.

    Example:
        folders = ['path/to/folder1', 'path/to/folder2']
        annotation_files = ['path/to/annotations1.json', 'path/to/annotations2.json']
        images, annotations = load_images_and_annotations(folders, annotation_files)

    Author:
        Your Name

    Date:
        YYYY-MM-DD
    """
    def load_images_and_annotations(self, folders, annotation_files):
        """
        Loads images and annotations from the given folders and annotation files.

        Args:
            folders (list): List of folder paths containing images.
            annotation_files (list): List of paths to annotation files.

        Returns:
            tuple: A tuple containing:
                - all_images (list): List of paths to all images.
                - all_annotations (dict): Dictionary containing 'images' and 'annotations' lists.

        """

        if 0:
            all_images = []
            all_annotations = {'images': [], 'annotations': []}

            for folder, annotation_file in zip(folders, annotation_files):
                images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
                all_images.extend([(os.path.join(folder, img)) for img in images])

                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)

                all_annotations['images'].extend(annotations['images'])
                all_annotations['annotations'].extend(annotations['annotations'])
        else:
            all_images = []
            all_annotations = {'images': [], 'annotations': []}

            for folder, annotation_file in zip(folders, annotation_files):
                water_category_id = None
                images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
                all_images.extend([(os.path.join(folder, img)) for img in images])

                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)

                # Retrieve 'water' category ID if not already found
                if water_category_id is None:
                    for category in annotations.get('categories', []):
                        if category['name'] == 'water':
                            water_category_id = category['id']
                            break

                if water_category_id is None:
                    raise ValueError("The 'water' category is not found in the categories list.")

                # Filter annotations for 'water' category
                water_annotations = [
                    ann for ann in annotations['annotations']
                    if ann['category_id'] == water_category_id
                ]

                # Add filtered annotations and images
                all_annotations['images'].extend(annotations['images'])
                all_annotations['annotations'].extend(water_annotations)

                return all_images, all_annotations

        return all_images, all_annotations

    """
    Module: dataset_splitter

    This module provides a function to split a dataset of images and annotations into
    training, validation, and test sets.

    Functions:
        split_dataset(all_images, annotations, train_split=0.7, val_split=0.15, test_split=0.15)
            Splits the dataset into training, validation, and test sets based on the given split ratios.

    Example:
        all_images = ['image1.jpg', 'image2.jpg', 'image3.jpg', ...]
        annotations = {'images': [...], 'annotations': [...]}
        train_images, val_images, test_images, annotations = split_dataset(all_images, annotations)

    Author: Your Name
    Date: YYYY-MM-DD
    """

    def split_dataset(self, all_images, annotations, train_split=0.7, val_split=0.15, test_split=0.15):
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            all_images (list): List of all image paths.
            annotations (dict): Dictionary containing image and annotation data.
            train_split (float): Proportion of the dataset to include in the training set.
            val_split (float): Proportion of the dataset to include in the validation set.
            test_split (float): Proportion of the dataset to include in the test set.

        Returns:
            tuple: A tuple containing:
                - train_images (list): List of training image paths.
                - val_images (list): List of validation image paths.
                - test_images (list): List of test image paths.
                - annotations (dict): Dictionary containing image and annotation data.

        """
        random.shuffle(all_images)

        num_images = len(all_images)
        train_size = int(train_split * num_images)
        val_size = int(val_split * num_images)

        train_images = all_images[:train_size]
        val_images = all_images[train_size:train_size + val_size]
        test_images = all_images[train_size + val_size:]

        print(
            f"Train: {len(train_images)} images, Validation: {len(val_images)} images, Test: {len(test_images)} images")
        return train_images, val_images, test_images, annotations


    def load_true_mask(self, image_file, annotations):
        """
        Load the true mask for a given image file from annotations.

        Args:
            image_file (str): The path to the image file.
            annotations (dict): A dictionary containing image and annotation data.

        Returns:
            ndarray: The mask for the image as a float32 numpy array. Returns None if no annotation is found.

        Raises:
            ValueError: If the image file is not found in the annotations.

        """
        # Find corresponding annotation for the image
        file_name_only = os.path.basename(image_file)
        image_info = next((img for img in annotations['images'] if img['file_name'] == file_name_only), None)
        print(image_info)

        if image_info is None:
            raise ValueError(f"Image file {image_file} not found in annotations.")

        image_id = image_info['id']

        annotation = next((ann for ann in annotations['annotations'] if ann['image_id'] == image_id), None)
        if annotation is None:
            return None

        height = image_info['height']
        width = image_info['width']

        segmentation = annotation['segmentation']
        rle = coco_mask.frPyObjects(segmentation, height, width)
        mask = coco_mask.decode(rle)
        return mask.astype(np.float32)


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def toolbarButtonDeepLearning(self):
        self.myDeepLearning(g_modelSettings)

    # ==================================================================================================================
    # ==================================================================================================================
    # IMAGE MASK FUNCTIONALITY
    # ==================================================================================================================
    # ==================================================================================================================
    def onMyToolBarCreateMask(self):

        if self.maskEditorDlg == None:
            if self.colorSegmentationDlg == None:
                self.labelOriginalImage.setDrawingMode(DrawingMode.MASK)

                self.maskEditorDlg = GRIME_AI_MaskEditorDlg()

                self.maskEditorDlg.addMask_Signal.connect(self.addMask)
                self.maskEditorDlg.generateMask_Signal.connect(self.generateMask)
                self.maskEditorDlg.drawingColorChange_Signal.connect(self.changePolygonColor)
                self.maskEditorDlg.reset_Signal.connect(self.resetMask)
                self.maskEditorDlg.polygonFill_Signal.connect(self.fillPolygonChanged)

                self.maskEditorDlg.close_signal.connect(self.maskDialogClose)
                self.maskEditorDlg.close_signal.connect(self.maskDialogClose)
                self.maskEditorDlg.accepted.connect(self.maskDialogClose)
                self.maskEditorDlg.rejected.connect(self.maskDialogClose)

                self.maskEditorDlg.show()
            else:
                strMessage = 'Please close the Color Segmentatoin toolbox if you want to use the Mask Editor toolbox.\nThis will be resolved in a future design change.'
                msgBox = GRIME_AI_QMessageBox('Tool Conflict', strMessage, QMessageBox.Yes | QMessageBox.No)
                response = msgBox.displayMsgBox(on_top=True)


    # ------------------------------------------------------------------------------------------------------------------
    def maskDialogClose(self):
        if self.maskEditorDlg != None:
            self.maskEditorDlg.close()
            del self.maskEditorDlg
            self.maskEditorDlg = None

        self.labelOriginalImage.setDrawingMode(DrawingMode.OFF)

    # ------------------------------------------------------------------------------------------------------------------
    def fillPolygonChanged(self, bFill):
        self.labelOriginalImage.enablePolygonFill(bFill)

    # ------------------------------------------------------------------------------------------------------------------
    def resetMask(self):
        self.labelOriginalImage.resetMask()
        self.labelOriginalImage.update()

    # ------------------------------------------------------------------------------------------------------------------
    def addMask(self):
        self.labelOriginalImage.incrementPolygon()

    # ------------------------------------------------------------------------------------------------------------------
    def generateMask(self):
        global currentImageFilename

        scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio,
                                                 QtCore.Qt.SmoothTransformation)

        widthMultiplier = currentImage.size().width() / scaledCurrentImage.size().width()
        heightMultiplier = currentImage.size().height() / scaledCurrentImage.size().height()

        # CONVERT IMAGE TO A MAT FORMAT TO USE ITS PARAMETERS TO CREATE A MASK IMAGE TEMPLATE
        # --------------------------------------------------------------------------------------------------------------
        img1 = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

        # CREATE A MASK IMAGE
        mask = np.zeros(img1.shape[:2], np.uint8)

        # ITERATE THROUGH EACH ONE OF THE POLYGONS
        # --------------------------------------------------------------------------------------------------------------
        polygonList = self.labelOriginalImage.getPolygon()

        for polygon in polygonList:
            myPoints = []
            for i in range(polygon.count()):
                myPoints.append([polygon.point(i).x() * widthMultiplier, polygon.point(i).y() * heightMultiplier])

            if len(myPoints) > 0:
                cv2.fillPoly(mask, np.int32([myPoints]), color=(255, 255, 255))

        masked = cv2.bitwise_and(img1, img1, mask=mask)

        # DISPLAY THE MASK IN THE GUI
        # --------------------------------------------------------------------------------------------------------------
        qImg = QImage(masked.data, masked.shape[1], masked.shape[0], QImage.Format_BGR888)
        pix = QPixmap(qImg)
        self.labelColorSegmentation.setPixmap(pix.scaled(self.labelColorSegmentation.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        # SAVE THE MASK RASTER AND POLYGON TO FILE
        if self.maskEditorDlg.getCheckBox_Save():
            # Extract image folder path to create a mask subfolder
            maskFolderPath = os.path.join(os.path.dirname(currentImageFilename), 'Masks')

            # Check for the existence of the filename path and create if it doesn't exist
            if not os.path.exists(maskFolderPath):
                os.makedirs(maskFolderPath)

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
                msgBox = GRIME_AI_QMessageBox('Save Mask Files', strMessage, QMessageBox.Yes | QMessageBox.No)
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
                myGRIMe_Color = GRIME_AI_Color()

                _, _, hist = myGRIMe_Color.KMeans(masked, 6)

                # EXTRACT DOMINANT HSV COLORS
                hist, colorClusters = myGRIMe_Color.extractDominant_HSV(masked, 6)

                # CREATE COLOR BAR TO DISPLAY CLUSTER COLORS
                colorBar = self.createColorBar(hist, colorClusters[0:5])

                # CONVERT colorBar TO A QImage FOR USE IN DISPLAYING IN QT GUI
                qImg = QImage(colorBar.data, colorBar.shape[1], colorBar.shape[0], QImage.Format_BGR888)

    # ------------------------------------------------------------------------------------------------------------------
    def changePolygonColor(self, polygonColor):
        self.labelOriginalImage.setBrushColor(polygonColor)
        self.labelOriginalImage.drawPolygon()

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def deleteAllROI(self):
        del self.roiList[:]
        self.tableWidget_ROIList.clearContents()
        self.tableWidget_ROIList.setRowCount(0)

        processLocalImage(self)

        #JES self.spinBoxColorClusters.setDisabled(False)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def onMyToolBarSettings(self):
        pass

    # ==================================================================================================================
    # THESE EVENT FILTERS WILL BE USED TO TRACK MOUSE MOVEMENT AND MOUSE BUTTON CLICKS FOR DISPLAYING ADDITIONAL
    # INFORMATION, VIEWS, POP-UP MENUS AND DRAWING REGIONS-OF-INTEREST (ROI) AROUND SPECIFIC AREAS OF AN IMAGE.
    # ==================================================================================================================
    def eventFilter(self, source, event):

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

        if event.type() == QtCore.QEvent.MouseButtonDblClick and source is self.labelOriginalImage:
            # labelEdgeImageDoubleClickEvent(self)
            # labelMouseDoubleClickEvent(self)
            NEON_labelOriginalImageDoubleClickEvent(self)

        return super(MainWindow, self).eventFilter(source, event)

    # ==================================================================================================================
    #
    # ==================================================================================================================
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

        #webdriver.Chrome.quit()

        QMainWindow.closeEvent(self, event)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def fetchImageList(self, imageFolder, bRecursive):
        global imageFileFolder
        imageFileFolder = imageFolder
        fetchLocalImageList(self, imageFileFolder, bRecursive, False)  # start_date, end_date, start_time, end_time)

        try:
            global gFrameCount
            self.onMyToolBarImageNavigation()
            self.imageNavigationDlg.setImageCount(gFrameCount)
            #JES self.imageNavigationDlg.reset()
        except:
            pass

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
            img = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

        # CALL PROCESSEVENTS IN ORDER TO UPDATE GUI
        QCoreApplication.processEvents()

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def NEON_updateSiteProducts(self, item):
        site_json = NEON_API().FetchSiteInfoFromNEON(SERVER, SITECODE)

        self.NEON_listboxSiteProducts.clear()

        for product in site_json['data']['dataProducts']:
            strText = product['dataProductCode'] + ": " + product['dataProductTitle']
            assert isinstance(strText, object)
            self.NEON_listboxSiteProducts.addItem(strText)

        self.NEON_listboxSiteProducts.show()

        # JES - TEMPORARILY SET NITRATE DATA ('should only be one nitrate product') AS THE DEFAULT SELECTION
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        itemNitrate = self.NEON_listboxSiteProducts.findItems('Nitrate', QtCore.Qt.MatchContains)
        nIndex = 0
        if len(itemNitrate) > 0:
            for item in itemNitrate:
                nIndex = self.NEON_listboxSiteProducts.row(item)
                self.NEON_listboxSiteProducts.setCurrentRow(nIndex)

            NEON_updateProductTable(self, nIndex)

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # JES
        # JES - TEMPORARILY SET NITRATE DATA ('should only be one nitrate product') AS THE DEFAULT SELECTION
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        item20002 = self.NEON_listboxSiteProducts.findItems('20002', QtCore.Qt.MatchContains)
        nIndex = 0
        if len(item20002) > 0:
            for item in item20002:
                nIndex = self.NEON_listboxSiteProducts.row(item)
                self.NEON_listboxSiteProducts.setCurrentRow(nIndex)

            NEON_updateProductTable(self, nIndex)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # JES

        self.NEON_listboxSiteProducts.item(0).setToolTip("Hello?")

    # ======================================================================================================================
    # THIS FUNCTION WILL DISPLAY THE LATEST IMAGE ON THE GUI.
    # ======================================================================================================================
    def NEON_DisplayLatestImage(self):

        if self.NEON_latestImage == []:
            self.NEON_labelLatestImage.setText("No Images Available")
        else:
            self.NEON_labelLatestImage.setPixmap(self.NEON_latestImage.scaled(self.NEON_labelLatestImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def WholeImage_ExtractFeatures(self, img, bWholeImageCalc):
        if bWholeImageCalc:
            # BLUR THE IMAGE
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            red, green, blue = GRIME_AI_Utils().separateChannels(img)
            redSum, greenSum, blueSum = GRIME_AI_Utils().sumChannels(red, green, blue)
            strGCC = '%3.3f' % (GRIME_AI_Vegetation_Indices().computeGreennessIndex(redSum, greenSum, blueSum))

            # IMAGE INTENSITY CALCULATIONS
            intensity = cv2.mean(gray)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway
            strIntensity = '%3.3f' % (intensity)

            # COMPUTE ENTROPY FOR ENTIRE IMAGE
            entropyValue = self.calcEntropy(gray)
            strEntropy = '%3.3f' % (entropyValue)
        else:
            strGCC = '---'
            strIntensity = '---'
            strEntropy = '---'

        return strGCC, strIntensity, strEntropy


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
    # bSaveImages = True
    bSaveImages = False     #JES - Is this flag needed any longer?
    imageOutputFolder = self.fileFolderDlg.lineEdit_images_folder.text()

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
    imageCount = GRIME_AI_Utils().getImageCount(filePath, extensions)

    progressBar = QProgressWheel()
    progressBar.setRange(0, imageCount + 1)
    progressBar.show()

    # RECURSE AND TRAVERSE FROM THE SPECIFIED FOLDER DOWN TO DETERMINE THE DATE RANGE FOR THE IMAGES FOUND
    files = GRIME_AI_Utils().getFileList(filePath, extensions, bFetchRecursive)

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
    for image_index, file in enumerate(files):
        progressBar.setWindowTitle(file)
        progressBar.setValue(image_index)
        progressBar.repaint()

        ext = os.path.splitext(file)[-1].lower()

        if ext in extensions:
            fileDate, fileTime = GRIME_AI_TimeStamp_Utils().extractDateFromFilename(file)

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
def processLocalImage(self, nImageIndex=0, imageFileFolder=''):
    global currentImage


    myGRIMe_Color = GRIME_AI_Color()

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

    # ==================================================================================================================
    # DISPLAY IMAGE FROM NEON SITE
    # ==================================================================================================================
    if currentImage:

        # CALCULATE THE GREENNESS INDEX (PHENOCAM GCC) FOR THE ENTIRE IMAGE
        if currentImage != []:
            numpyImg = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())
            #JES self.label_GreennessIndex_Value.setText(str(GRIME_Vegetation_Indices.computeGreennessIndex(numpyImg)))

        scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        width = scaledCurrentImage.width()
        height = scaledCurrentImage.height()

        currentImageRescaled = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        if g_displayOptions.displayROIs:
            painter = QPainter(currentImageRescaled)

        img = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

        # EXTRACT FEATURES FOR WHOLE IMAGE
        strGCC, strIntensity, strEntropy = self.WholeImage_ExtractFeatures(img, self.colorSegmentationParams.wholeImage)

        nRow = self.tableWidget_ROIList.rowCount()
        if nRow == 0:
            self.tableWidget_ROIList.insertRow(nRow)

        wholeImageLabel = QtWidgets.QLabel()
        wholeImageLabel.setText("Whole Image")
        self.tableWidget_ROIList.setCellWidget(0, 0, wholeImageLabel)

        greennessLabel = QtWidgets.QLabel()
        greennessLabel.setText(strGCC)
        self.tableWidget_ROIList.setCellWidget(0, 3, greennessLabel)

        intensityLabel = QtWidgets.QLabel()
        intensityLabel.setText(strIntensity)
        self.tableWidget_ROIList.setCellWidget(0, 4, intensityLabel)

        entropyLabel = QtWidgets.QLabel()
        entropyLabel.setText(strEntropy)
        self.tableWidget_ROIList.setCellWidget(0, 5,entropyLabel)

        # DISPLAY THE PROGRESS WHEEL
        progressBar = QProgressWheel()
        progressBar.setRange(0, len(self.roiList) + 1)
        #JES progressBar.show()

        # EXTRACT THE FEATURES FOR EACH ROI
        nRow = 1
        for roiObj in self.roiList:
            progressBar.setValue(nRow+1)
            progressBar.repaint()

            try:
                if (1):
                    # EXTRACT ROI FOR WHICH COLOR CLUSTERING IS TO BE PERFORMED
                    rgb = extractROI(roiObj.getImageROI(), img)

                    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

                    # ------------------------------------------------------------------------------------------
                    # COLOR SEGMENTATION
                    # ------------------------------------------------------------------------------------------
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

                    # ------------------------------------------------------------------------------------------
                    # CALCULATE THE GREENNESS INDEX FOR THE ROI
                    # ------------------------------------------------------------------------------------------
                    try:
                        # CALCULATE THE ROI'S GREENNES INDEX
                        red, green, blue = GRIME_AI_Utils().separateChannels(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                        redSum, greenSum, blueSum = GRIME_AI_Utils().sumChannels(red, green, blue)
                        fGreennessIndex = GRIME_AI_Vegetation_Indices().computeGreennessIndex(redSum, greenSum, blueSum)

                        # DISPLAY ROI'S GREENNESS INDEX ON THE GUI
                        strGreennessIndex = "{:.4f}".format(fGreennessIndex)
                        self.greennessLabel = QtWidgets.QLabel()
                        self.greennessLabel.setText(strGreennessIndex)
                        self.tableWidget_ROIList.setCellWidget(nRow, 3, self.greennessLabel)
                    except:
                        print('Something went wrong with the ROI Greenness Index calculation.')

                    # ------------------------------------------------------------------------------------------
                    # CALCULATE THE INTENSITY FOR THE ROI
                    # ------------------------------------------------------------------------------------------
                    try:
                        # CALCULATE THE ROI'S INTENSITY
                        strIntensity = "{:.4f}".format(cv2.mean(gray)[0])  # The range for a pixel's value in grayscale is (0-255), 127 lies midway

                        # DISPALY THE ROI'S INTENSITY ON THE GUI
                        self.intensityLabel = QtWidgets.QLabel()
                        self.intensityLabel.setText(strIntensity)
                        self.tableWidget_ROIList.setCellWidget(nRow, 4, self.intensityLabel)
                    except:
                        print('Something went wrong with the ROI Intensity calculation.')

                    # ------------------------------------------------------------------------------------------
                    # COMPUTE ENTROPY FOR ENTIRE IMAGE
                    # ------------------------------------------------------------------------------------------
                    try:
                        # CALCULATE THE ROI'S ENTROPY
                        strEntropyValue = "{:.4f}".format(self.calcEntropy(gray))

                        # DISPLAY THE ROI'S ENTROPY ON THE GUI
                        self.entropyLabel = QtWidgets.QLabel()
                        self.entropyLabel.setText(strEntropyValue)
                        self.tableWidget_ROIList.setCellWidget(nRow, 5, self.entropyLabel)
                    except:
                        pass

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
        img1 = GRIME_AI_Utils().convertQImageToMat(myImage.toImage())

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

            myProcessImage = GRIME_ProcessImage()

            if g_edgeMethodSettings.method == edgeMethodsClass.CANNY:
                pix = myProcessImage.processCanny(img1, gray, g_edgeMethodSettings)

            elif g_edgeMethodSettings.method == edgeMethodsClass.LAPLACIAN:
                pix = myProcessImage.processLaplacian(img1)

            elif g_edgeMethodSettings.method == edgeMethodsClass.SOBEL_X or g_edgeMethodSettings.method == edgeMethodsClass.SOBEL_Y or g_edgeMethodSettings.method == edgeMethodsClass.SOBEL_XY:
                pix = myProcessImage.processSobel(gray, g_edgeMethodSettings.getSobelKernel(), g_edgeMethodSettings.method)

            elif g_featureMethodSettings.method == featureMethodsClass.SIFT:
                pix = myProcessImage.processSIFT(img1, gray)

            elif g_featureMethodSettings.method == featureMethodsClass.ORB:
                pix = myProcessImage.processORB(img1, gray, g_featureMethodSettings)

    return pix


# ======================================================================================================================
#
# ======================================================================================================================
def toolbarButtonImageTriage_1(folder_path):
    toolbarButtonImageTriage(folder_path, False)

def toolbarButtonImageTriage(folder_path=[], checkBox_FetchRecursive=False):
    strMessage = 'You are about to perform Image Triage. Would you like to continue?'
    msgBox = GRIME_AI_QMessageBox('Download Image Files', strMessage, QMessageBox.Yes | QMessageBox.No)
    response = msgBox.displayMsgBox()

    if response == QMessageBox.Yes:
        if folder_path == []:
            prompter = promptlib.Files()
            folder = prompter.dir()
        else:
            folder = folder_path

        if len(folder) == 0:
            strMessage = 'ERROR! Please specify an image folder containing images to triage.'
            msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
            response = msgBox.displayMsgBox()
        else:
            TriageDlg = GRIME_TriageOptionsDlg()

            response = TriageDlg.exec_()

            if response == 1:

                if len(TriageDlg.getReferenceImageFilename()) == 0 and TriageDlg.getCorrectAlignment() == True:
                    strMessage = 'Please select reference image if you want to correct image alignment.'
                    msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                    response = msgBox.displayMsgBox()
                else:
                    myTriage = GRIME_AI_ImageTriage()
                    myTriage.cleanImages(folder, \
                                False, \
                                TriageDlg.getBlurThreshold(), TriageDlg.getShiftSize(), \
                                TriageDlg.getBrightnessMin(), TriageDlg.getBrightnessMax(), \
                                TriageDlg.getCreateReport(), TriageDlg.getMoveImages(), \
                                TriageDlg.getCorrectAlignment(), TriageDlg.getSavePolylines(),
                                TriageDlg.getReferenceImageFilename(), TriageDlg.getRotationThreshold())

                    strMessage = 'Image triage is complete!'
                    msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                    response = msgBox.displayMsgBox()
            else:
                strMessage = 'ABORT! You cancelled the triage operation.'
                msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()
    else:
        strMessage = 'ABORT! You cancelled the triage operation.'
        msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
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
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read it into OpenCV format
    ssl._create_default_https_context = ssl._create_unverified_context
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
# THIS FUNCTION UPDATES THE GUI WITH THE INFO FOR A NEON SITE SELECTED BY THE END-USER.
# ======================================================================================================================
def NEON_updateSiteInfo(self, item):
    self.NEON_listboxSiteInfo.clear()

    # EXTRACT THE SITE ID FOR THE SELECTED ITEM
    siteID = self.NEON_listboxSites.currentItem().text()

    global SITECODE
    SITECODE = siteID.split(' - ')[0]

    self.labelNEONSiteDetails.setText(SITECODE)

    siteInfo = NEON_API().FetchSiteInfoFromNEON(SERVER, SITECODE)

    global DOMAINCODE
    DOMAINCODE = siteInfo['data']['domainCode']

    global SITENAME
    SITENAME = siteInfo['data']['siteName']
    #JES self.labelNEONSite.setText(SITENAME)

    keys = siteInfo['data'].keys()

    for key in keys:
        tmp = key
        tmpSiteInfo = str(siteInfo['data'][key])
        self.NEON_listboxSiteInfo.addItem(tmp + ": " + tmpSiteInfo)

    return (SITECODE)

# ======================================================================================================================
# THIS FUNCTION WILL UPDATE THE PRODUCT TABLE IN THE GUI WITH THE PRODUCTS THAT ARE AVAILABLE FOR A SPECIFIC SITE.
# ======================================================================================================================
def NEON_updateProductTable(self, item):
    products = self.NEON_listboxSiteProducts.selectedItems()

    # JES: FUTURE CONSIDERATION - MUST MAKE CODE DYNAMIC TO ONLY DELETE UNSELECTED ITEMS
    for i in range(self.NEON_tableProducts.rowCount()):
        self.NEON_tableProducts.removeRow(0)

    for i in range(len(products)):
        strText = self.NEON_listboxSiteProducts.selectedItems()[i].text()
        self.NEON_tableProducts.insertRow(i)

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
        self.NEON_tableProducts.setItem(i, m, QTableWidgetItem(strText))

        # CONFIGURE DATES FOR SPECIFIC PRODUCT
        m += 2
        nYear = 1970
        nMonth = 1
        nDay = 1
        #nYear, nMonth, nDay = GRIME_PhenoCam().getStartDate()
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        date_widget.setDisabled(True)
        self.NEON_tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        nYear = 1970
        nMonth = 1
        nDay = 1
        #nYear, nMonth, nDay = GRIME_PhenoCam().getEndDate()
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        date_widget.setDisabled(True)
        self.NEON_tableProducts.setCellWidget(i, m, date_widget)

        # --------------------
        # SET THE CALENDAR START AND END DATE THE SAME USING THE DATE FOR THE LAST DAY FOR WHICH DATA IS AVAILABLE
        # --------------------
        nYear, nMonth, nDay = GRIME_PhenoCam().getEndDate()

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        #JES - SET FOR TODAY'S DATE. USER'S MAY NOT WANT TO GO BACK MANY YEARS.
        #nYear, nMonth, nDay = GRIME_PhenoCam.getStartDate()
        #date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        #date_widget.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
        # trigger event when the user changes the date
        date_widget.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.NEON_tableProducts, self.checkBox_UniqueDates.isChecked()))
        date_widget.setKeyboardTracking(False)
        self.NEON_tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        # trigger event when the user changes the date
        date_widget.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.NEON_tableProducts, self.checkBox_UniqueDates.isChecked()))
        date_widget.setKeyboardTracking(False)
        self.NEON_tableProducts.setCellWidget(i, m, date_widget)

        # --------------------
        # --------------------
        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        # trigger event when the user changes the time
        dateTime.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.NEON_tableProducts, self.checkBox_UniqueDates.isChecked()))
        date_widget.setKeyboardTracking(False)
        self.NEON_tableProducts.setCellWidget(i, m, dateTime)

        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        # trigger event when the user changes the time
        dateTime.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.NEON_tableProducts, self.checkBox_UniqueDates.isChecked()))
        date_widget.setKeyboardTracking(False)
        self.NEON_tableProducts.setCellWidget(i, m, dateTime)

        self.NEON_tableProducts.resizeColumnsToContents()

# ======================================================================================================================
#
# ======================================================================================================================
def NEON_dateChangeMethod(date_widget, tableWidget, bUniqueDates):
    global SITECODE
    global DOMAINCODE

    nRow = tableWidget.currentIndex().row()

    strProductIDCell = tableWidget.item(nRow, 0).text().upper()

    # FETCH DATE THAT CHANGED FOR THE SPECIFIC ROW
    start_date, start_time, end_date, end_time = GRIMe_ProductTable().fetchTableDates(tableWidget, nRow)

    if bUniqueDates == False:
        for i in range(tableWidget.rowCount()):
            tableWidget.cellWidget(i, 4).setDate(start_date)
            tableWidget.cellWidget(i, 5).setDate(end_date)
    else:
        tableWidget.cellWidget(nRow, 4).setDate(start_date)
        tableWidget.cellWidget(nRow, 5).setDate(end_date)

    #imageCount = GRIME_PhenoCam.getPhenocamImageCount(SITECODE, DOMAINCODE, start_date, end_date, start_time, end_time)

    #tableWidget.setItem(nRow, 2, QTableWidgetItem(str(imageCount)))


# ======================================================================================================================
#
# ======================================================================================================================
def DP1_20002_fetchImageList(self, nRow, start_date, end_date, start_time, end_time, downloadsFilePath):
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

            phenoCam = GRIME_PhenoCam()
            tmpList = phenoCam.getVisibleImages(dailyURLvisible, start_time, end_time)

            dailyImagesList.setVisibleList(tmpList.getVisibleList())

            start_date += datetime.timedelta(days=1)

        gWebImageCount = len(dailyImagesList.getVisibleList())

    else:
        dailyURLvisible = []

    gWebImageCount = len(dailyImagesList.getVisibleList())

    if gWebImageCount > 0:
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

            if not os.path.exists(downloadsFilePath):
                os.makedirs(downloadsFilePath)
            completeFilename = os.path.join(downloadsFilePath, filename)

            if os.path.isfile(completeFilename) == False:
                urllib.request.urlretrieve(image.fullPathAndFilename, completeFilename)

        # clean-up before exiting function
        # 1. close and delete the progress bar
        # 2. no other clean-up tasks
        progressBar.close()
        del progressBar

        #jes LET THE CALLING FUNCTION BE RESPONSIBLE FOR REPORTING DOWNLOAD COMPLETION.
        #jes MODIFY THIS IN A FUTURE RELEASE TO RETURN A PASS/FAIL MESSAGE TO THE FUNCTION THAT INVOKED THIS FUNCTION.
        #jes strMessage = 'Data download is complete!'
        #jes msgBox = GRIME_AI_QMessageBox('Data Download', strMessage)
        #jes response = msgBox.displayMsgBox()


# ======================================================================================================================
#
# ======================================================================================================================
def downloadPBTImageFiles(self):
    GRIMe_ProductTableObj = GRIMe_ProductTable()
    GRIMe_ProductTableObj.fetchTableDates(self.NEON_tableProducts, nRow)

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

    missing_data_message = ""
    nitrateList = []
    nError = 0;

    myNEON_API = NEON_API()

    # ----------------------------------------------------------------------------------------------------
    # SAVE DOWNLOADED DATA TO THE USER GRIMe-AI FOLDER THAT IS AUTOMATICALLY CREATED, IF IT DOES NOT EXIST,
    # CREATE IT IN THE USER'S DOCUMENT FOLDER
    # ----------------------------------------------------------------------------------------------------
    NEON_download_file_path = self.edit_NEONSaveFilePath.text()
    JsonEditor().update_json_entry("NEON_Root_Folder", NEON_download_file_path)

    if len(NEON_download_file_path) == 0:
        strMessage = 'A download folder has not been specified. Would you like to use the last GRIME-AI NEON download folder?'
        msgBox = GRIME_AI_QMessageBox('NEON Root Download Folder', strMessage, QMessageBox.Yes | QMessageBox.No)
        response = msgBox.displayMsgBox()

        if response == QMessageBox.Yes:
            #NEON_download_file_path = os.path.expanduser('~')
            #NEON_download_file_path = os.path.join(NEON_download_file_path, 'Documents')
            #NEON_download_file_path = os.path.join(NEON_download_file_path, 'GRIMe-AI')

            NEON_download_file_path = JsonEditor().getValue("NEON_Root_Folder")

            if not os.path.exists(NEON_download_file_path):
                os.makedirs(NEON_download_file_path)
            self.edit_NEONSaveFilePath.setText(NEON_download_file_path)
            JsonEditor().update_json_entry("NEON_Root_Folder", NEON_download_file_path)
    else:
        # MAKE SURE THE PATH EXISTS. IF IT DOES NOT, THEN CREATE IT.
        if not os.path.exists(NEON_download_file_path):
            os.makedirs(NEON_download_file_path)


    # --------------------------------------------------------------------------------
    # FIND IMAGE PRODUCT (20002) ROW TO GET DATE RANGE
    # --------------------------------------------------------------------------------
    rowRange = range(self.NEON_tableProducts.rowCount())

    for nRow in rowRange:
        GRIMe_ProductTableObj = GRIMe_ProductTable()
        start_date, start_time, end_date, end_time = GRIMe_ProductTableObj.fetchTableDates(self.NEON_tableProducts, nRow)

        # EXTRACT THE PRODUCT ID
        prodIDCol = 0
        strProductIDCell = self.NEON_tableProducts.item(nRow, prodIDCol).text()
        nProductID = int(strProductIDCell.split('.')[1])
        #else:
        #    nProductID = -999

        if nProductID > 0:
            PRODUCTCODE = strProductIDCell.split(':')[0]

            # PHENOCAM IMAGES
            # ----------------------------------------------------------------------------------------------------------
            if nProductID == 20002:
                downloadsFilePath = os.path.join(self.edit_NEONSaveFilePath.text(), 'Images')
                if not os.path.exists(downloadsFilePath):
                    os.makedirs(downloadsFilePath)

                DP1_20002_fetchImageList(self, nRow, start_date, end_date, start_time, end_time, downloadsFilePath)

                processLocalImage(self, imageFileFolder=downloadsFilePath)

            # ALL OTHER NEON DATA
            # ----------------------------------------------------------------------------------------------------------
            if nProductID != 20002:
                strStartYearMonth = str(start_date.year) + '-' + str(start_date.month).zfill(2)
                strEndYearMonth = str(end_date.year) + '-' + str(end_date.month).zfill(2)

                PRODUCTCODE = strProductIDCell.split(':')[0]

                # GET THE RANGE OF MONTHS FROM THE START DATE TO THE END DATE
                dateRange = GRIME_AI_Utils().getRangeOfDates(strStartYearMonth, strEndYearMonth)

                # GET THE AVAILABLE MONTHS FOR THE SELECTED DATA SET
                availableMonths = NEON_API().getAvailableMonths(SITECODE, PRODUCTCODE)

                monthCount = 0
                missingMonths = []
                for month in dateRange:
                    if month in availableMonths:
                        monthCount += 1
                    else:
                        missingMonths.append(month)

                if monthCount == 0:
                    missing_data_message = missing_data_message + 'NEON Error!  ' + strProductIDCell + 'Data is not available for some or all of the dates selected!\n'
                elif (monthCount < len(dateRange)):
                    strMsg = '%d of %d months unavailable: %s' % (len(missingMonths), len(dateRange), missingMonths)
                    missing_data_message = missing_data_message + 'Partial Download!\n   ' + strProductIDCell + strMsg + '\n'

                if monthCount > 0:
                    downloadsFilePath = os.path.join(NEON_download_file_path, 'Data')
                    if not os.path.exists(downloadsFilePath):
                        os.makedirs(downloadsFilePath)

                    nError = myNEON_API.FetchData(SITECODE, strProductIDCell, strStartYearMonth, strEndYearMonth, downloadsFilePath)
        else:
            missing_data_message = missing_data_message + 'NEON Error!\n  ' + strProductIDCell + 'Product not available!' + '\n'

    if missing_data_message != "":
        msgBox = GRIME_AI_QMessageBox('Download Error!', missing_data_message, buttons=QMessageBox.Close)
    else:
        msgBox = GRIME_AI_QMessageBox('Download Complete!', 'Download Complete!', buttons=QMessageBox.Close)
    response = msgBox.displayMsgBox()

        # ----------------------------------------------------------------------------------------------------------
        # NITRATE DATA
        # ----------------------------------------------------------------------------------------------------------
        # if nProductID == 20033:
        #     nitrateList = myNEON_API.parseNitrateCSV()
        #
        #     if len(nitrateList) > 0:
        #         # JES - USE NITRATE DATA FOR DEVELOPING GENERIC CSV READING AND DATA GRAPHING CAPABILITIES
        #         scene = QGraphicsScene()
        #         self.scene = scene
        #         nWidth = self.graphicsView.width()
        #         nHeight = self.graphicsView.height()
        #         nX = self.graphicsView.x()
        #         nY = self.graphicsView.y()
        #         self.scene.setSceneRect(0, 0, nWidth, nHeight)
        #         # self.graphicsView.setWindowTitle('Nitrate Data')
        #         self.graphicsView.setScene(self.scene)
        #         figure = Figure()
        #         axes = figure.gca()
        #         axes.set_title("Nitrate Data")
        #
        #         i = 0
        #         for i, nitrateData in enumerate(nitrateList):
        #             y = float(nitrateData.getNitrateMean())
        #             axes.plot(i, y, '.', markersize=2)
        #
        #         canvas = FigureCanvas(figure)
        #         canvas.resize(nWidth, nHeight)
        #         self.scene.addWidget(canvas)
        #         self.graphicsView.show()


# ======================================================================================================================
#
# ======================================================================================================================
def labelEdgeImageDoubleClickEvent(self):
    global currentImage

    # CONVERT IMAGE FROM QImage FORMAT TO Mat FORMAT
    img = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())
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
def NEON_labelOriginalImageDoubleClickEvent(self):
    global currentImage

    if currentImage != []:
        img = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

        self.setMouseTracking(False)

        cv2.imshow('Original', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.setMouseTracking(True)


# ======================================================================================================================
#
# ======================================================================================================================
def NEON_labelMouseDoubleClickEvent(self, event):
    img = GRIME_AI_Utils().convertQImageToMat(self.NEON_labelLatestImage.toImage())
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
    monthList = {}

    # RETRIEVE INFORMATION FROM THE NEON WEBSITE FOR THE PARTICULAR SITE
    site_json = NEON_API().FetchSiteInfoFromNEON(SERVER, SITECODE)

    if site_json is not []:
        # EXTRACT THE AVAILABLE MONTH AND THE URL FOR THE DATA FOR EACH AVAILABLE MONTH
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


# ======================================================================================================================
#
# ======================================================================================================================
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


# ======================================================================================================================
#
# ======================================================================================================================
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


# ======================================================================================================================
#
# ======================================================================================================================
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
#jes @hydra.main(config_path=None, config_name=None, version_base=None)
#jes def main(cfg: DictConfig):
if __name__ == '__main__':
    if 0:
        if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.core.global_hydra.GlobalHydra.instance().clear()
        else:
            hydra.initialize(config_path=None)


    #os.environ[str('R_HOME')] = str("C:\\Program Files\\R\\R-4.4.1")
    #JES - THIS DOESN'T WORK! - os.system[str('R_HOME')] = str("C:\\Program Files\\R\\R-4.4.1")
    print(os.environ.get('R_HOME'))

    # CREATE MAIN APP WINDOW
    app = QApplication(sys.argv)
    frame = MainWindow()

    frame.move(app.desktop().screen().rect().center() - frame.rect().center())

    # ------------------------------------------------------------------------------------------------------------------
    # PROCESS ANY EVENTS THAT WERE DELAYED BECAUSE OF THE SPLASH SCREEN
    # ------------------------------------------------------------------------------------------------------------------
    app.processEvents()

    frame.graphicsView.setVisible(True)

    # ------------------------------------------------------------------------------------------------------------------
    # http://localhost:8888/notebooks/intro-seg.ipynb
    # ------------------------------------------------------------------------------------------------------------------
    bStartupComplete = True

    # SHOW MAIN WINDOW
    #frame.show()

    # Run the program
    sys.exit(app.exec())


'''
    from torchvision import models
    fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    from PIL import Image
    import matplotlib.pyplot as plt
    import torch

    !wget - nv
    https: // static.independent.co.uk / s3fs - public / thumbnails / image / 2018 / 04 / 10 / 19 / pinyon - jay - bird.jpg - O
    bird.png
    img = Image.open('./bird.png')
    plt.imshow(img);
    plt.show()

    # ----------------------------------------------------------------------------------------------------
    # Apply the transformations needed
    # ----------------------------------------------------------------------------------------------------
    import torchvision.transforms as T
    trf = T.Compose([T.Resize(256),
                     T.CenterCrop(224),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)

    # ----------------------------------------------------------------------------------------------------
    # Pass the input through the net
    # ----------------------------------------------------------------------------------------------------
    out = fcn(inp)['out']
    print(out.shape)

    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    import numpy as np
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    print(om.shape)
    print(np.unique(om))

    # ----------------------------------------------------------------------------------------------------
    # Define the helper function
    # ----------------------------------------------------------------------------------------------------
    def decode_segmap(image, nc=21):
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    rgb = decode_segmap(om)
    plt.imshow(rgb);
    plt.show()

    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def segment(net, path, show_orig=True, dev='cuda'):
        img = Image.open(path)
        if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
        # Comment the Resize and CenterCrop for better inference results
        trf = T.Compose([T.Resize(640),
                         # T.CenterCrop(224),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        inp = trf(img).unsqueeze(0).to(dev)
        out = net.to(dev)(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        rgb = decode_segmap(om)
        plt.imshow(rgb);
        plt.axis('off');
        plt.show()

        !wget - nv
        https: // www.learnopencv.com / wp - content / uploads / 2021 / 01 / horse - segmentation.jpeg - O
        horse.png
        segment(fcn, './horse.png')

    # ----------------------------------------------------------------------------------------------------
    # DeepLabv4
    # ----------------------------------------------------------------------------------------------------
    dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

    segment(dlab, './horse.png')

    !wget - nv
    "https://www.learnopencv.com/wp-content/uploads/2021/01/person-segmentation.jpeg" - O
    person.png
    img = Image.open('./person.png')
    plt.imshow(img);
    plt.show()

    print('Segmenatation Image on FCN')
    segment(fcn, path='./person.png', show_orig=False)

    print('Segmenatation Image on DeepLabv3')
    segment(dlab, path='./person.png', show_orig=False)


    # ----------------------------------------------------------------------------------------------------
    # INFERENCE TIME
    # ----------------------------------------------------------------------------------------------------
    import time

    def infer_time(net, path='./horse.png', dev='cuda'):
        img = Image.open(path)
        trf = T.Compose([T.Resize(256),
                         T.CenterCrop(224),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        inp = trf(img).unsqueeze(0).to(dev)

        st = time.time()
        out1 = net.to(dev)(inp)
        et = time.time()

        return et - st

    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def onCPU()
        avg_over = 100

        fcn_infer_time_list_cpu = [infer_time(fcn, dev='cpu') for _ in range(avg_over)]
        fcn_infer_time_avg_cpu = sum(fcn_infer_time_list_cpu) / avg_over

        dlab_infer_time_list_cpu = [infer_time(dlab, dev='cpu') for _ in range(avg_over)]
        dlab_infer_time_avg_cpu = sum(dlab_infer_time_list_cpu) / avg_over
        print('The Average Inference time on FCN is:     {:.2f}s'.format(fcn_infer_time_avg_cpu))
        print('The Average Inference time on DeepLab is: {:.2f}s'.format(dlab_infer_time_avg_cpu))

    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def onGPU()
        avg_over = 100

        fcn_infer_time_list_gpu = [infer_time(fcn) for _ in range(avg_over)]
        fcn_infer_time_avg_gpu = sum(fcn_infer_time_list_gpu) / avg_over

        dlab_infer_time_list_gpu = [infer_time(dlab) for _ in range(avg_over)]
        dlab_infer_time_avg_gpu = sum(dlab_infer_time_list_gpu) / avg_over

        print('The Average Inference time on FCN is:     {:.3f}s'.format(fcn_infer_time_avg_gpu))
        print('The Average Inference time on DeepLab is: {:.3f}s'.format(dlab_infer_time_avg_gpu))

    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def modelSize()
        import os

        resnet101_size = os.path.getsize('/root/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth')
        fcn_size = os.path.getsize('/root/.cache/torch/hub/checkpoints/fcn_resnet101_coco-7ecb50ca.pth')
        dlab_size = os.path.getsize('/root/.cache/torch/hub/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth')

        fcn_total = fcn_size + resnet101_size
        dlab_total = dlab_size + resnet101_size

        print('Size of the FCN model with Resnet101 backbone is:       {:.2f} MB'.format(fcn_total / (1024 * 1024)))
        print('Size of the DeepLabv3 model with Resnet101 backbone is: {:.2f} MB'.format(dlab_total / (1024 * 1024)))

        plt.bar([0, 1], [fcn_total / (1024 * 1024), dlab_total / (1024 * 1024)])
        plt.ylabel('Size of the model in MegaBytes')
        plt.xticks([0, 1], ['FCN', 'DeepLabv3'])
        plt.title('Comparison of the model size of FCN and DeepLabv3')
        plt.show()



    # ==================================================================================================================
    # DIAGNOSTIC FUNCTION???
    # JES self.pushButton_ColorSegmentation.clicked.connect(self.pushButtonColorSegmentationClicked)
    # ==================================================================================================================
    def pushButtonColorSegmentationClicked(self):
        global currentImage

        myGRIMe_Color = GRIME_AI_Color()

        if len(self.roiList) == 0:
            strError = "You must train at least one ROI before segmenting the image."
            msgBox = GRIME_AI_QMessageBox('Color Segmentation Error', strError)
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
            img1 = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

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
'''


'''
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection  

# Define transformations for data augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load COCO dataset
train_dataset = CocoDetection(root='path/to/coco/train', annFile='annotations/train.json', transform=transform)
val_dataset = CocoDetection(root='path/to/coco/val', annFile='annotations/val.json', transform=transform)

# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

# Define the model
model = MaskRCNN(num_classes=91)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Move model to the device
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # Update the learning rate
    lr_scheduler.step()

    # Evaluation on the validation dataset
    model.eval()
    for images, targets in val_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            val_loss_dict = model(images, targets)
# Save the trained model
    
    # Print training and validation losses
    print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {losses.item()}, Validation Loss: {sum(val_loss_dict.values()).item()}')

torch.save(model.state_dict(), 'trained_model.pth')
'''