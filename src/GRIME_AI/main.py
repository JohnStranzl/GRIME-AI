#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

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
# main.py – at the very top

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
try:
    # TRY TO IMPORT THE VERSION CONSTANT FROM VERSION.PY
    from GRIME_AI.version import SW_VERSION
except ImportError:
    # FALLBACK IF VERSION.PY DOES NOT EXIST. 0.0.0.0 IS AN INVALID VERSION NUMBER
    SW_VERSION = "Ver. 0.0.0.0"

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import logging

# Optional: set up logging so you can track initialization
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    import torch
    logging.debug("Successfully imported torch.")

    # If you need to disable profiling to avoid TorchScript type inference issues,
    # do it explicitly here. (Remove these lines if you want profiling enabled.)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    logging.debug("Disabled Torch profiling mode and executor.")

    # Force torch.jit initialization.
    jit_doc = torch.jit.__doc__
    logging.debug("Torch JIT initialized successfully, first 100 chars of doc: %.100s", jit_doc)
except Exception as e:
    logging.error("Error during Torch initialization", exc_info=True)

try:
    import torch.jit
except Exception as e:
    logging.error("Error importing torch.jit", exc_info=True)

try:
    import torch.jit._script    # This can be important if your models are scripted.
except Exception as e:
    logging.error("Error importing torch.jit._script", exc_info=True)

try:
    import sklearn.neighbors._typedefs
except Exception:
    pass

try:
    import sklearn.neighbors._partition_nodes
except Exception:
    pass

try:
    import sklearn.utils._weight_vector
except Exception:
    pass

try:
    import sklearn.neighbors._quad_tree
except Exception:
    pass

try:
    import skimage
except Exception:
    pass

try:
    import imageio
except Exception:
    pass

try:
    import imageio_ffmpeg
except Exception:
    pass


import os
# Must be set before Hydra starts to get the full stack trace if an error occurs.
# This works programmtically in case the user does not have privileges to modify their operating system environment
# because it only modifies the environment of the current Python process and its children.
os.environ["HYDRA_FULL_ERROR"] = "1"

import sys
import argparse
import shutil
import getpass

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def excepthook(exc_type, exc_value, exc_tb):
    print("Uncaught exception:")
    traceback.print_exception(exc_type, exc_value, exc_tb)

sys.excepthook = excepthook

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import datetime
from datetime import date
import time

import csv

import urllib.request

import promptlib

import traceback

from dataclasses import dataclass

from GRIME_AI.utils.resource_utils import icon_path

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QToolBar, QDateTimeEdit, \
    QMessageBox, QAction, QHeaderView, QDialog, QFileDialog
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtWidgets import QTreeWidgetItem

from GRIME_AI.GRIME_AI_SplashScreen import GRIME_AI_SplashScreen

import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import GRIME_AI.sobelData

from GRIME_AI.usgs.client import USGSClient

# ----------------------------------------------------------------------------------------------------------------------
# POP-UP/MODELESS DIALOG BOXES
# ----------------------------------------------------------------------------------------------------------------------
from GRIME_AI.dialogs.color_segmentation.GRIME_AI_ColorSegmentationDlg import GRIME_AI_ColorSegmentationDlg
from GRIME_AI.dialogs.edge_detection.GRIME_AI_EdgeDetectionDlg import GRIME_AI_EdgeDetectionDlg
from GRIME_AI.dialogs.image_navigation.GRIME_AI_ImageNavigationDlg import GRIME_AI_ImageNavigationDlg
from GRIME_AI.dialogs.file_utilities.GRIME_AI_FileUtilitiesDlg import GRIME_AI_FileUtilitiesDlg
from GRIME_AI.dialogs.mask_editor.GRIME_AI_MaskEditorDlg import GRIME_AI_MaskEditorDlg
from GRIME_AI.dialogs.composite_slice.GRIME_AI_CompositeSliceDlg import GRIME_AI_CompositeSliceDlg
from GRIME_AI_ProcessImage import GRIME_AI_ProcessImage
from GRIME_AI.dialogs.release_notes.GRIME_AI_ReleaseNotesDlg import GRIME_AI_ReleaseNotesDlg
from GRIME_AI.dialogs.triage.GRIME_AI_TriageOptionsDlg import GRIME_AI_TriageOptionsDlg
from GRIME_AI.GRIME_AI_Color import GRIME_AI_Color
from GRIME_AI.GRIME_AI_CompositeSlices import GRIME_AI_CompositeSlices
from GRIME_AI.GRIME_AI_Vegetation_Indices import GRIME_AI_Vegetation_Indices, GreennessIndex
from GRIME_AI.dialogs.extract_coco_masks.GRIME_AI_ExportCOCOMasksDlg import GRIME_AI_ExportCOCOMasksDlg
from GRIME_AI.dialogs.ML_image_processing.GRIME_AI_ML_ImageProcessingDlg import GRIME_AI_ML_ImageProcessingDlg
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.GRIME_AI_Feature_Export import GRIME_AI_Feature_Export
from GRIME_AI.GRIME_AI_Diagnostics import GRIME_AI_Diagnostics
from GRIME_AI.GRIME_AI_ImageData import imageData
from GRIME_AI.dialogs.image_organizer.GRIME_AI_ImageOrganizerDlg import GRIME_AI_ImageOrganizerDlg
from GRIME_AI.phenocam.GRIME_AI_PhenoCam import GRIME_AI_PhenoCam, dailyList
from GRIME_AI.GRIME_AI_ProductTable import GRIME_AI_ProductTable
from GRIME_AI.GRIME_AI_QLabel import DrawingMode
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI.GRIME_AI_roiData import GRIME_AI_roiData, ROIShape
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.GRIME_AI_Resize_Controls import GRIME_AI_Resize_Controls
from GRIME_AI.GRIME_AI_TimeStamp_Utils import GRIME_AI_TimeStamp_Utils
from GRIME_AI.GRIME_AI_ImageTriage import GRIME_AI_ImageTriage
from GRIME_AI.GRIME_AI_GreenImageGenerator import GreenImageGenerator
from GRIME_AI.GRIME_AI_COCO_Utils import GRIME_AI_COCO_Utils
from GRIME_AI.phenocam.GRIME_AI_Phenocam_API import GRIME_AI_Phenocam_API

from GRIME_AI.colorSegmentationParams import colorSegmentationParamsClass

#from GRIME_AI_DeepLearning import GRIME_AI_DeepLearning

from GRIME_AI.geomaps.google_maps_viewer import GoogleMapWidget
from GRIME_AI.geomaps.openstreetmap_viewer import OpenStreetMapWidget

# ----------------------------------------------------------------------------------------------------------------------
# HYDRA (for SAM2)
# ----------------------------------------------------------------------------------------------------------------------
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from neon.NEON_API import NEON_API

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from constants import edgeMethodsClass, featureMethodsClass, modelSettingsClass

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from exifData import EXIFData

# import GRIME_AI_KMeans

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from GRIME_AI.ml_core.ml_model_training import MLModelTraining
from GRIME_AI.ml_core.ml_image_segmentation import MLImageSegmentation

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from neonAIgui import Ui_MainWindow

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
global bStartupComplete
bStartupComplete = False

global bShow_GUI
bShow_GUI = False

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import os
import cv2
import numpy as np

# ======================================================================================================================
#
# ======================================================================================================================
SITECODE = 'ARIK'
DOMAINCODE = 'D10'
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
frame = []
# Define the maximum number of gray levels
gray_level = 16

# URLS
# url = "http://maps.googleapis.com/maps/api/geocode/json?address=googleplex&sensor=false"
url = 'https://www.neonscience.org/field-sites/explore-field-sites'
root_url = 'https://www.neonscience.org'
SERVER = 'http://data.neonscience.org/api/v0/'

class displayOptions():
    displayROIs = True

g_displayOptions   = displayOptions()
g_edgeMethodSettings = edgeMethodsClass()
g_featureMethodSettings = featureMethodsClass()

g_modelSettings = modelSettingsClass()

hyperparameterDlg = None

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
    compositeSliceDlg    = None
    imageNavigationDlg   = None
    releaseNotesDlg      = None
    buildModelDlg        = None


    imageFileFolder = None

    global dailyImagesList
    dailyImagesList = dailyList([], [])

    NEON_siteList = []
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

        self.setupUi(self)
        self.setWindowTitle("GRIME AI" + " " + SW_VERSION + " - John E. Stranzl Jr.")
        #self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowStaysOnTopHint)

        # Initialize a variable to hold the current NEON site information
        self.current_site_info = ["No site info available."]
        # Set the tooltip generator on your GRIME_AI_QLabel widget(s). For example, if your widget is named NEON_labelLatestImage:
        if 0:
            if hasattr(self.NEON_labelLatestImage, "tooltipGenerator"):
                self.NEON_labelLatestImage.tooltipGenerator = self.siteInfoTooltip
            else:
                print("Warning: NEON_labelLatestImage is not an instance of GRIME_AI_QLabel.")

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
        base_path = Path(__file__).resolve().parent
        splash_dir = base_path / "resources" / "splash_screens"

        splash = GRIME_AI_SplashScreen(
            QPixmap(os.path.join(splash_dir, "Splash_007.jpg")),
            strVersion=SW_VERSION
        )
        splash.show(self)

        splash = GRIME_AI_SplashScreen(
            QPixmap(os.path.join(splash_dir, "GRIME-AI Logo.jpg")),
            delay=5
        )
        splash.show(self)

        # ------------------------------------------------------------------------------------------------------------------
        # CREATE REQUIRED FOLDERS IN THE USER'S DOCUMENTS FOLDER
        # ------------------------------------------------------------------------------------------------------------------
        utils = GRIME_AI_Utils()
        utils.create_GRIME_folders()

        self.populate_controls()

        # ----------------------------------------------------------------------------------------------------
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
        # ----------------------------------------------------------------------------------------------------
        self.greenness_index_list = []
        self.colorSegmentationParams = colorSegmentationParamsClass()
        self.getColorSegmentationParams()

        # ----------------------------------------------------------------------------------------------------
        # GET DATA, POPULATE WIDGETS, ETC.
        # ----------------------------------------------------------------------------------------------------
        self.USGS_InitProductTable()

        self.USGS_FormatProductTable(self.table_USGS_Sites)

        self.NEON_FormatProductTableHeader()

        self.initROITable(self.greenness_index_list)

        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #JES - REVISIT DOUBLE CLICKING ON IMAGES
        self.NEON_labelLatestImage.mouseDoubleClickEvent = NEON_labelMouseDoubleClickEvent

        self.NEON_labelLatestImage.installEventFilter(self)
        self.labelEdgeImage.installEventFilter(self)
        self.labelOriginalImage.installEventFilter(self)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        self.pushButton_RetrieveNEONData.clicked.connect(self.pushbutton_NEONDownloadClicked)

        self.pushButton_USGS_BrowseImageFolder.clicked.connect(self.pushButton_USGS_BrowseImageFolder_Clicked)
        self.pushButton_NEON_BrowseImageFolder.clicked.connect(self.pushButton_NEON_BrowseImageFolder_Clicked)

        # INITIALIZE WIDGETS
        maxRows = self.tableWidget_ROIList.rowCount()
        nCol = 0
        for i in range(0, maxRows):
            self.tableWidget_ROIList.removeRow(0)

        # SAVE AND RECALL SETTINGS
        self.action_SaveSettings.triggered.connect(self.menubarSaveSettings)
        self.action_ReleaseNotes.triggered.connect(self.toolbarButtonReleaseNotes)
        self.action_CompositeSlices.triggered.connect(self.menubarCompositeSlices)
        self.action_TriageImages.triggered.connect(self.toolbarButtonImageTriage_2)
        self.action_Generate_Greenness_Test_Images.triggered.connect(self.menubar_Generate_Greenness_Test_Images)

        self.action_RefreshNEON.triggered.connect(self.menubar_RefreshNEON)

        self.action_CreateJSON.triggered.connect(self.menubar_CreateJSON)
        self.action_ExtractCOCOMasks.triggered.connect(self.menubarExtractCOCOMasks)
        self.action_Sync_JSON_Annotations.triggered.connect(self.menubar_sync_json_annotations)
        self.action_Inspect_Annotations.triggered.connect(self.menubar_inspect_annotations)

        self.action_ImageOrganizer.triggered.connect(self.menubar_ImageOrganizer)

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
        self.usgs = USGSClient()
        self.usgs.initialize()
        self.USGS_listboxSites.itemClicked.connect(self.USGS_SiteClicked)
        self.pushButton_USGSDownload.clicked.connect(self.pushButton_USGSDownloadClicked)

        # ------------------------------------------------------------------------------------------------------------------
        # NIMS
        # ------------------------------------------------------------------------------------------------------------------
        try:
            from usgs.shim import USGS_NIMS_Shim

            self.myNIMS = USGS_NIMS_Shim()
            self.cameraDictionary = self.myNIMS.get_camera_dictionary()
            self.cameraList = self.myNIMS.get_camera_list()
            self.USGS_listboxSites.clear()
            self.USGS_listboxSites.addItems(self.cameraList)

            self.USGS_listboxSites.show()

            cameraIndex = 1
            self.USGS_listboxSites.setCurrentRow(cameraIndex)

            strCamID = self.USGS_listboxSites.currentItem().text()

            cameraInfo = self.myNIMS.get_camera_info(strCamID)
            self.listboxUSGSSiteInfo.addItems(cameraInfo)

            self.USGS_updateSiteInfo(1)

        except Exception:
            msgBox = GRIME_AI_QMessageBox('USGS NIMS Error', 'Unable to access USGS NIMS Database!')
            response = msgBox.displayMsgBox()

        #self.edit_USGSSaveFilePath.setText("C:\\Users\\Astrid Haugen\\Documents\\GRIME-AI\\Downloads\\USGS_Test")

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
        _, self.NEON_siteList = myNEON_API.readFieldSiteTable()

        if len(self.NEON_siteList) == 0:
            print("NEON Field Site Table from NEON website FAILED...")
        # IF THERE ARE FIELD SITE TABLES AVAILABLE, ENABLE GUI WIDGETS PERTAINING TO WEB SITE DATA/IMAGES
        else:
            print("Populate NEON Products tab on GUI...")
            myList = []

            for site in self.NEON_siteList:
                strSiteName = site.siteID + ' - ' + site.siteName
                myList.append(strSiteName)

            self.NEON_listboxSites.addItems(myList)

            #JES - TEMPORARILY SET BARCO LAKE AS THE DEFAULT SELECTION
            try:
                self.NEON_listboxSites.setCurrentRow(2)
                self.NEON_listboxSites.show()
                self.NEON_SiteClicked(2)
            except Exception:
                pass

        self.show()

        try:
            self.phenocam_site_dictionary = myNEON_API.scrape_phenocam_table()
        except:
            print('ERROR: Unable to access Phenocam site information on Phenocam server!')

        ###JES - USING OPENSTREENMAP INSTEAD OF GOOLE TO AVOID PAYING A FEE TO GOOGLE.
        #self.init_google_maps(self.cameraDictionary)

        #geojson_str = self.shapefile_to_geojson_string(shapefile_path)
        geojson_str = None
        self.init_openstreetmap(self.cameraDictionary, redList=self.NEON_siteList, geojson_str=geojson_str)

        shapefile_path = self.osm_widget.load_shapefile("Ogallala", "NE_Aggregated_polygons.shp")

        if self.cameraDictionary:
            for name, coords in self.cameraDictionary.items():
                self.osm_widget.add_pin(coords["lat"], coords["lng"], color="usgs_green", label=name)

        if self.phenocam_site_dictionary:
            for site_id, info in self.phenocam_site_dictionary.items():
                self.osm_widget.add_pin(info["lat"], info["lon"], color="yellow", label=site_id)

        if self.NEON_siteList:
            for coords in self.NEON_siteList:
                self.osm_widget.add_pin(coords.latitude, coords.longitude, color="gold", label=coords.siteName)

        # IF THE PHENCOCAM SITE DICTIONARY IS NOT EMPTY, THEN POPULATE THE TREE WIDGET ON THE MAIN GRIME AI CANVAS.
        # OTHERWISE, THE ASSUMPTION IS THAT THE PHENOCAM SERVER IS INACCESSIBLE/OFFLINE.
        if self.phenocam_site_dictionary:
            self.populate_phenocam_tree()

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def init_openstreetmap(self, cameraDictionary, redList=None, geojson_str=None):
        osm_tab = self.findChild(QWidget, "tab_GoogleMaps")

        self.osm_widget = OpenStreetMapWidget(parent=osm_tab)

        layout = QVBoxLayout(osm_tab)
        layout.addWidget(self.osm_widget)

        # POPULATE THE MAP WITH PINS FOR USGS HIVIS CAMERA SITES
        for name, coords in self.cameraDictionary.items():
            self.osm_widget.add_pin(coords["lat"], coords["lng"], color="usgs_green", label=name)

        # POPULATE THE MAP WITH PINS FOR PHENOCAM CAMERA SITES REFERENCED BY NEON
        for site_id, info in self.phenocam_site_dictionary.items():
            self.osm_widget.add_pin(info["lat"], info["lon"], color="gold", label=site_id)

        # POPULATE THE MAP WITH PINS FOR PHENOCAM CAMERA SITES REFERENCED BY NEON
        for coords in self.NEON_siteList:
            self.osm_widget.add_pin(coords.latitude, coords.longitude, color="yellow")

        # ADD A PIN FOR FLAGSTAFF, AZ WHERE NORTHERN ARIZONA UNIVERSITY IS LOCATED.
        self.osm_widget.add_pin(35.1878, -111.6528, color="blue", label="Northern Arizona University")

        # CENTER MAP ON LINCOLN, NE WHERE THE UNIVERSITY OF NEBRASKA-LINCOLN IS LOCATED.
        self.osm_widget.set_center(40.8136, -96.7026, zoom=12, add_marker=True, label="Lincoln, NE", color="red")

    def populate_phenocam_tree(self):
        """
        Populate the Phenocam tree widget with sites (roots) and camera-like children
        based on available columns in the current DataFrame schema.
        """
        api = GRIME_AI_Phenocam_API()
        cameras_df = api.get_cameras(all_records=True)

        # Preserve your schema but also create lowercase aliases for robust access
        cols_lower = {c.lower(): c for c in cameras_df.columns}

        def has(col):
            return col in cameras_df.columns

        def get_col(*candidates):
            for c in candidates:
                if c in cameras_df.columns:
                    return c
            return None

        # Pick site column (prefer your capitalized 'Sitename' if present)
        site_col = get_col("Sitename", "sitename")
        if cameras_df.empty or site_col is None:
            print("Camera data missing expected site column.")
            return

        # Choose a child label column from what's available
        # Priority: camera_description, camera_orientation, active, date_first/date_last
        child_label_col = get_col("sitemetadata.camera_description",
                                  "sitemetadata.camera_orientation")
        # Optional extra detail to append
        extra_detail_cols = [c for c in [
            get_col("active", "infrared"),
            get_col("date_first"),
            get_col("date_last")
        ] if c is not None]

        # Clear existing items
        self.treeWidget_Phenocam.clear()

        # Group by site and create nodes
        for site_name, group in cameras_df.groupby(site_col):
            site_item = QTreeWidgetItem([str(site_name)])

            # If we have a descriptive child column, create one child per row; else, add a single child
            if child_label_col is not None:
                for _, row in group.iterrows():
                    label = str(row.get(child_label_col, site_name))
                    # Append extra details in a readable way
                    details = []
                    for col in extra_detail_cols:
                        val = row.get(col)
                        if pd.notna(val):
                            details.append(f"{col}:{val}")
                    child_text = label if not details else f"{label}  ({', '.join(details)})"
                    camera_item = QTreeWidgetItem([child_text])
                    site_item.addChild(camera_item)
            else:
                # No explicit child label available—add a single placeholder child per site
                camera_item = QTreeWidgetItem([str(site_name)])
                site_item.addChild(camera_item)

            self.treeWidget_Phenocam.addTopLevelItem(site_item)

        self.treeWidget_Phenocam.expandAll()

        ###JES - THIS INTERACTION WITH THE PHENOCAM SERVERS TAKES ON THE ORDER OF 4 MINUTES; SO I
        if 0:
            print("START: Phenocam Meta Data Export...")
            api.export_all_to_excel(output_dir=GRIME_AI_Save_Utils().get_phenocam_folder())
            print("COMPLETED: Phenocam Meta Data Export.")

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _usgs_progress(self, idx: int, total: int, label: str | None) -> None:
        """Progress callback for USGSClient operations."""
        if not hasattr(self, "_usgsProgress"):
            self._usgsProgress = QProgressWheel(self)
            self._usgsProgress.setRange(0, total)
            self._usgsProgress.setWindowTitle("USGS Operation")
            self._usgsProgress.show()

        self._usgsProgress.setMaximum(total)
        self._usgsProgress.setValue(idx)

        if label:
            self._usgsProgress.setWindowTitle(str(label))

        # Clean up when finished
        if idx >= total:
            self._usgsProgress.close()
            del self._usgsProgress

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def init_google_maps(self, cameraDictionary):
        google_tab = self.findChild(QWidget, "tab_GoogleMaps")
        self.google_maps_widget = GoogleMapWidget(cameraDictionary, parent=google_tab)

        layout = QVBoxLayout(google_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.google_maps_widget)
        google_tab.setLayout(layout)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def shapefile_to_geojson_string(self, shapefile_path):
        import geopandas as gpd

        gdf = gpd.read_file(shapefile_path)
        gdf = gdf.to_crs(epsg=4326)  # ensure WGS84
        geojson_str = gdf.to_json()  # returns a JSON string
        return geojson_str

    def siteInfoTooltip(self):
        """
        This function returns the dynamic tooltip string for GRIME_AI_QLabel.
        It uses self.current_site_info (a list of strings) to create the tooltip.
        """
        if self.current_site_info:
            return "NEON Site Info:\n" + "\n".join(self.current_site_info)
        else:
            return "No site info available."

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
        myFeatureExport.ExtractFeatures(imagesList, imageFileFolder, self.roiList, self.colorSegmentationParams, self.greenness_index_list)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def universalTestButton(self, testFunction):
        if testFunction == 1:
            print('This is Test Function 1.')

            # KMeans EXPECTS THE BYTE ORDER TO BE RGB
            img1 = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())
            #img1 = GRIME_AI_Utils().convertQImageToMat(myImage.toImage())

            rgb = cv2.blur(img1, ksize=(11, 11))

            # convert image to HSV
            hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)

            if len(self.roiList) > 0:
                # DIAGNOSTICS
                #if self.checkBoxColorDiagnostics.checkState():
                GRIME_AI_Diagnostics.RGB3DPlot(rgb)
                GRIME_AI_Diagnostics.plotHSVChannelsGray(hsv)
                GRIME_AI_Diagnostics.plotHSVChannelsColor(hsv)

                # segment colors
                rgb1 = myGRIMe_Color.segmentColors(rgb, hsv, self.roiList)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def getColorSegmentationParams(self):
        if self.colorSegmentationDlg is not None:
            self.colorSegmentationParams.GCC            = self.colorSegmentationDlg.checkBox_GCC.isChecked()
            self.colorSegmentationParams.GLI            = self.colorSegmentationDlg.checkBox_GLI.isChecked()
            self.colorSegmentationParams.NDVI           = self.colorSegmentationDlg.checkBox_NDVI.isChecked()
            self.colorSegmentationParams.ExG            = self.colorSegmentationDlg.checkBox_ExG.isChecked()
            self.colorSegmentationParams.RGI            = self.colorSegmentationDlg.checkBox_RGI.isChecked()

            self.colorSegmentationParams.Intensity      = self.colorSegmentationDlg.checkBox_Intensity.isChecked()
            self.colorSegmentationParams.ShannonEntropy = self.colorSegmentationDlg.checkBox_ShannonEntropy.isChecked()
            self.colorSegmentationParams.Texture        = self.colorSegmentationDlg.checkBox_Texture.isChecked()

            self.colorSegmentationParams.wholeImage     = self.colorSegmentationDlg.checkBoxScalarRegion_WholeImage.isChecked()

            self.colorSegmentationParams.numColorClusters = self.colorSegmentationDlg.get_num_color_clusters()

        self.greenness_index_list.clear()

        if self.colorSegmentationParams.GCC:
            self.greenness_index_list.append(GreennessIndex(GRIME_AI_Vegetation_Indices.GCC))
        if self.colorSegmentationParams.GLI:
            self.greenness_index_list.append(GreennessIndex(GRIME_AI_Vegetation_Indices.GLI))
        if self.colorSegmentationParams.NDVI:
            self.greenness_index_list.append(GreennessIndex(GRIME_AI_Vegetation_Indices.NDVI))
        if self.colorSegmentationParams.ExG:
            self.greenness_index_list.append(GreennessIndex(GRIME_AI_Vegetation_Indices.ExG))
        if self.colorSegmentationParams.RGI:
            self.greenness_index_list.append(GreennessIndex(GRIME_AI_Vegetation_Indices.RGI))

        self.initROITable(self.greenness_index_list)

    # ------------------------------------------------------------------------------------------------------------------
    # TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR
    # ------------------------------------------------------------------------------------------------------------------
    def createToolBar(self):
        toolbar = QToolBar("GRIME-AI Toolbar")
        self.addToolBar(toolbar)
        toolbar.setIconSize(QtCore.QSize(48, 48))

        parent_path = os.path.abspath(os.path.dirname(__file__))
        icons_dir = os.path.join(parent_path, "resources", "toolbar_icons")
        print("Toolbar Initialization: Icons directory:", icons_dir)

        # Define all toolbar buttons in one place
        buttons = [
            ("FileFolder_1.png", "Data Exploration",
             "Select input and output folder locations", self.onMyToolBarFileFolder),
            ("Triage_2.png", "Image Triage",
             "Move images that are of poor quality", self.toolbarButtonImageTriage),
            ("ImageNav_3.png", "Image Navigation",
             "Navigate (scroll) through images", self.onMyToolBarImageNavigation),
            ("Mask.png", "Create Masks",
             "Draw polygons to create image masks", self.onMyToolBarCreateMask),
            ("ColorWheel_4.png", "Color Segmentation",
             "Create ROIs to segment regions by color", self.onMyToolBarColorSegmentation),
            ("EdgeFilters_2.png", "Edge and Feature Detection",
             "Edge Detection Filters", self.toolbarButtonEdgeDetection),
            ("Settings_1.png", "Settings",
             "Change options and settings", self.onMyToolBarSettings),
            ("Green Brain Icon.png", "Deep Learning",
             "Deep Learning - EXPERIMENTAL", self.menubar_CreateJSON),
            ("grime2_StopSign.png", "GRIME2",
             "GRIME2 - Water Level Measurement", self.toolbarButtonGRIME2),
            ("Help_2.png", "Help",
             "Help and Release Notes", self.toolbarButtonReleaseNotes),
        ]

        # Generic creation loop
        for filename, text, tip, slot in buttons:
            icon_file = icon_path("toolbar_icons", filename)
            action = QAction(QIcon(icon_file), text, self)
            action.setStatusTip(tip)
            action.triggered.connect(slot)
            toolbar.addAction(action)
            print(f"Toolbar Initialization: {text} icon path: {icon_file}")

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
                SITECODE = NEON_updateSiteInfo(self)
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
        except Exception:
            gProcessClick = 0

        print("NEON site selection complete.")

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
    # ==================================================================================================================
    #
    # ==================================================================================================================
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
        imageCount = self.USGS_get_image_count()

        tableWidget.setItem(0, 1, QTableWidgetItem(imageCount.__str__()))

    # ======================================================================================================================
    # THIS FUNCTION WILL UPDATE THE PRODUCT TABLE IN THE GUI WITH THE PRODUCTS THAT ARE AVAILABLE FOR A SPECIFIC SITE.
    # ======================================================================================================================
    def USGS_FormatProductTable(self, tableProducts):
        maxRows = 1

        #JES: MUST MAKE CODE DYNAMIC TO ONLY DELETE UNSELECTED ITEMS
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
        currentRow = self.USGS_listboxSites.currentRow()
        if currentRow < 0:
            return

        strCamID = self.USGS_listboxSites.currentItem().text()

        try:
            # Update site info listbox
            self.listboxUSGSSiteInfo.clear()
            self.listboxUSGSSiteInfo.addItems(
                self.usgs.get_camera_info_lines(strCamID)
            )
            self.USGS_listboxSites.setCurrentRow(currentRow)

            # Latest image
            code, pix = self.usgs.get_latest_pixmap(strCamID)
            if code == 404 or pix is None:
                self.USGS_labelLatestImage.setText("No Images Available")
            else:
                self.USGS_labelLatestImage.setPixmap(
                    pix.scaled(self.USGS_labelLatestImage.size(),
                               QtCore.Qt.KeepAspectRatio,
                               QtCore.Qt.SmoothTransformation))

            # Update table
            self.table_USGS_Sites.setItem(0, 0, QTableWidgetItem(strCamID))

        except Exception as e:
            print("Error in USGS_updateSiteInfo:", e)

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

        imageCount = self.USGS_get_image_count()

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


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def trainROI(self, roiParameters):
        global currentImage

        if currentImage:
            myGRIME_Color = GRIME_AI_Color()

            # CREATE AN ROI OBJECT
            roiObj = GRIME_AI_roiData()

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
            except Exception:
                msgBox = GRIME_AI_QMessageBox('ROI Error',
                                           'An unexpected error occurred calculating the ROI of the full resolution image!', buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()

                return

            # ----------------------------------------------------------------------------------------------------------
            # CALCULATE COLOR CLUSTERS FOR THE ROI AND SAVE THEM TO THE ROI LIST
            # ----------------------------------------------------------------------------------------------------------
            # EXTRACT THE ROI FROM THE ORIGINAL IMAGE
            roiObj.setNumColorClusters(roiParameters.numColorClusters)

            # EXTRACT DOMINANT RGB COLORS AND ADD THEM TO THE ROI OBJECT
            #JES - PROVISIONAL - RGB CLUSTERS ARE NOT CURRENTLY USED.
            #JES qImg, clusterCenters, hist = myGRIME_Color.KMeans(rgb, roiObj.getNumColorClusters())
            #JES roiObj.setClusterCenters(clusterCenters, hist)

            # EXTRACT DOMINANT HSV COLORS AND ADD THEM TO THE ROI OBJECT
            img1 = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())
            rgb = extractROI(roiObj.getImageROI(), img1)
            hist, colorClusters = myGRIME_Color.extractDominant_HSV(rgb, roiObj.getNumColorClusters())
            roiObj.setHSVClusterCenters(colorClusters, hist)

            roiObj.setTrainingImageName(currentImageFilename)

            self.roiList.append(roiObj)

            # ----------------------------------------------------------------------------------------------------------
            # DISPLAY IN FEATURE TABLE
            # ----------------------------------------------------------------------------------------------------------
            #if (nRow == 0):
            #    numToAdd = 3
            #else:
            #    numToAdd = 1

            #for i in range(numToAdd):
            #    self.tableWidget_ROIList.insertRow(nRow)
            #    nRow += 1

            # CREATE COLOR BAR TO DISPLAY CLUSTER COLORS
            colorBar = GRIME_AI_Color.create_color_bar(hist, colorClusters)

            # CONVERT colorBar TO A QImage FOR USE IN DISPLAYING IN QT GUI
            qImg = QImage(colorBar.data, colorBar.shape[1], colorBar.shape[0], QImage.Format_BGR888)

            # INSERT THE DOMINANT COLORS INTO A QLabel IN ORDER TO ADD IT TO THE FEATURE TABLE
            nRow = self.tableWidget_ROIList.rowCount()
            self.tableWidget_ROIList.insertRow(nRow)

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

            self.labelOriginalImage.clearROIs()
            self.labelOriginalImage.setROIs(self.roiList)

            processLocalImage(self, currentImageIndex)
            self.refreshImage()

            # ----------------------------------------------------------------------------------------------------------
            # ONCE AN ROI IS DEFINED FOR A SPECIFIC NUMBER OF COLOR CLUSTERS, DISABLE THE CONTROL SO THAT THE USER
            # CANNOT CHANGE THE VALUE FOR SUBSEQUENT TRAINED ROIs.
            # ----------------------------------------------------------------------------------------------------------
            if self.colorSegmentationDlg != None:
                if len(self.roiList) > 0:
                    self.colorSegmentationDlg.disable_spinbox_color_clusters(True)
                else:
                    self.colorSegmentationDlg.disable_spinbox_color_clusters(False)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def calcEntropy(self, img):
        entropy = []

        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        total_pixel = img.shape[0] * img.shape[1]

        for item in hist.flatten():
            probability = item / total_pixel
            if probability == 0:
                en = 0
            else:
                en = -1 * probability * (np.log(probability) / np.log(2))
            entropy.append(en)

        try:
            sum_en = sum(entropy)
        except Exception:
            sum_en = 0.0

        #from scipy.stats import entropy
        #base = 2
        #H = entropy(hist, base=base)

        return sum_en

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def displayROIs(self, roiParameters):
        if self.colorSegmentationDlg != None:
            self.getColorSegmentationParams()

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
        except Exception:
             pass


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def initROITable(self, greenness_list=None):

        if greenness_list == []:
            headerList = ['ROI Name', 'Ref. Image', 'Cur. Image', 'Intensity', 'Entropy']
        else:
            headerList = ['ROI Name', 'Ref. Image', 'Cur. Image']

            for greenness_name in greenness_list:
                headerList.append(greenness_name.get_name())

            headerList.append('Intensity')
            headerList.append('Entropy')


        stylesheet = "::section{Background-color:rgb(116,175,80);border-radius:14px;}"
        header = self.tableWidget_ROIList.horizontalHeader()
        font = QFont()
        font.setBold(True)
        self.tableWidget_ROIList.horizontalHeader().setFont(font)
        self.tableWidget_ROIList.horizontalHeader().setVisible(True)
        self.tableWidget_ROIList.setStyleSheet(stylesheet)
        self.tableWidget_ROIList.setColumnCount(len(headerList))

        for i, item in enumerate(headerList):
            headerItem = QTableWidgetItem(item)
            headerItem.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tableWidget_ROIList.setHorizontalHeaderItem(i, headerItem)

            # Automatically resize columns when text is inserted
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)

        # Set the stretch mode to adapt to any remaining space
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def USGS_get_image_count(self):
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

            imageCount = self.myNIMS.get_image_count(siteName=site, nwisID=nwisID, startDate=startDate, endDate=endDate, startTime=startTime, endTime=endTime)
        except Exception:
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
                #USGS_download_file_path = os.path.join(USGS_download_file_path, 'GRIME-AI')

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

            ###JES self.myNIMS.download_images(siteName=site, nwisID=nwisID, saveFolder=saveFolder, startDate=startDate, endDate=endDate, startTime=startTime, endTime=endTime)
            downloaded, missing = self.usgs.download_images(
                site,
                startDate,
                endDate,
                startTime,
                endTime,
                saveFolder,
                progress=self._usgs_progress  # <-- new: hook for your QProgressWheel
            )

            saveFolder = os.path.join(downloadsFilePath, "../../data")
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
    # ==================================================================================================================
    # ==================================================================================================================
    def menubar_CreateJSON(self):
        global hyperparameterDlg

        # If it’s already alive, just raise & activate it
        '''
        if hyperparameterDlg is not None and hyperparameterDlg.isVisible():
            hyperparameterDlg.raise_()
            hyperparameterDlg.activateWindow()
            return None
        '''

        settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
        CONFIG_FILENAME = "site_config.json"
        site_configuration_file = os.path.normpath(os.path.join(settings_folder, CONFIG_FILENAME))

        hyperparameterDlg = GRIME_AI_ML_ImageProcessingDlg(frame)

        hyperparameterDlg.ml_train_signal.connect(train_main)
        hyperparameterDlg.ml_segment_signal.connect(load_model_main)

        #hyperparameterDlg.accepted.connect(closehyperparameterDlg)
        #hyperparameterDlg.rejected.connect(closehyperparameterDlg)

        hyperparameterDlg.finished.connect(closehyperparameterDlg)

        #config = hyperparameterDlg.load_config_from_json(site_configuration_file)
        #hyperparameterDlg.initialize_dialog_from_config(config)

        # Show the dialog and capture user response.
        if hyperparameterDlg.exec_() == QDialog.Accepted:
            hyperparameters = hyperparameterDlg.get_values()
            #config = dialog.getValues()
            print("Configuration Options:")
            for key, value in hyperparameters.items():
                print(f"  {key}: {value}")


    def menubar_RefreshNEON(self):
        # INITIALIZE GUI CONTROLS
        # frame.NEON_listboxSites.setCurrentRow(1)

        # GET LIST OF ALL SITES ON NEON
        # if frame.checkBoxNEONSites.isChecked():
        myNEON_API = NEON_API()
        _, siteList = myNEON_API.readFieldSiteTable()
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

            #JES - TEMPORARILY SET BARCO LAKE AS THE DEFAULT SELECTION
            try:
                self.NEON_listboxSites.setCurrentRow(2)
                self.NEON_listboxSites.show()
                self.NEON_SiteClicked(2)

            except Exception:
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
            self.compositeSliceDlg = GRIME_AI_CompositeSliceDlg()

            self.compositeSliceDlg.compositeSliceGenerateSignal.connect(self.generateCompositeSlices)
            self.compositeSliceDlg.compositeSliceCancelSignal.connect(self.closeCompositeSlices)

            imageFilename = dailyImagesList.getVisibleList()[0].fullPathAndFilename
            self.compositeSliceDlg.loadImage(imageFilename)

            self.compositeSliceDlg.label_Image.setDrawingMode(DrawingMode.SLICE)

            self.compositeSliceDlg.show()

    def generateCompositeSlices(self):
        print("Generating composite slices image(s)...")

        global imageFileFolder
        composite_slices_folder = GRIME_AI_Save_Utils().create_composite_slices_folder(imageFileFolder)

        # Get original image dimensions
        orig_h, orig_w, _ = self.compositeSliceDlg._originalImage.shape

        # Use the label’s aspect-ratio–aware mapping
        rect = self.compositeSliceDlg.label_Image.getSliceRectInOriginal(orig_w, orig_h)

        # Compute center and width directly in original image coordinates
        actualSliceCenter = rect.left() + rect.width() // 2
        actualSliceWidth = rect.width()

        compositeSlices = GRIME_AI_CompositeSlices(actualSliceCenter, actualSliceWidth)
        compositeSlices.create_composite_image(dailyImagesList.visibleList, composite_slices_folder)

    def _drawn_pixmap_geometry(self):
        pm = self.pixmap()
        if not pm or pm.isNull():
            return 0, 0, 0, 0  # draw_w, draw_h, x_off, y_off

        label_w, label_h = self.width(), self.height()
        pm_w, pm_h = pm.width(), pm.height()

        # The pixmap is already scaled to fit the label preserving AR,
        # so pm_w/pm_h are the drawn dimensions. Compute insets.
        x_off = (label_w - pm_w) // 2
        y_off = (label_h - pm_h) // 2
        return pm_w, pm_h, x_off, y_off

    def closeCompositeSlices(self):
        if self.compositeSliceDlg != None:
            self.compositeSliceDlg.close()
            self.compositeSliceDlg    = None

    def menubar_Generate_Greenness_Test_Images(self):
        # initialize with default settings

        rootFolder = os.path.expanduser('~')
        rootFolder = os.path.join(rootFolder, 'Documents', 'GRIME-AI', 'Test Images')
        gen = GreenImageGenerator(out_dir=rootFolder)

        # generate all images (solids, splotches, masks)
        gen.generate_all()

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def toolbarButtonImageTriage_1(self, folder_path=None):
        self.toolbarButtonImageTriage()

    def toolbarButtonImageTriage_2(self):
        self.toolbarButtonImageTriage()

    def toolbarButtonImageTriage(self, folder_path=None, checkBox_FetchRecursive=False):
        strMessage = 'You are about to perform Image Triage. Would you like to continue?'
        msgBox = GRIME_AI_QMessageBox('Download Image Files', strMessage, QMessageBox.Yes | QMessageBox.No)
        response = msgBox.displayMsgBox()

        if response == QMessageBox.Yes:
            prompter = promptlib.Files()
            folder = prompter.dir()

            if len(folder) == 0:
                strMessage = 'ERROR! Please specify an image folder containing images to triage.'
                msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()
            else:
                TriageDlg = GRIME_AI_TriageOptionsDlg()

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


    def menubarExtractCOCOMasks(self):
        self.COCOdlg = GRIME_AI_ExportCOCOMasksDlg(self)

        self.COCOdlg.COCO_signal_ok.connect(self.accepted_COCODlg)
        self.COCOdlg.COCO_signal_cancel.connect(self.rejected_COCODlg)

        self.COCOdlg.show()


    def accepted_COCODlg(self):
        image_dir = self.COCOdlg.getAnnotationImagesFolder()
        output_dir = os.path.join(image_dir, "training_masks")

        #JES CLEANUP TASK: THE ANNOTATION FILE IS NOT REQUIRED TO BE IN THE TRAINING IMAGES FOLDER.
        utils = GRIME_AI_COCO_Utils(image_dir)
        utils.extract_masks(image_dir, output_dir)

    def rejected_COCODlg(self):
        pass


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubar_sync_json_annotations(self):
        selected_dir = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select a Folder",
            directory="C:/",  # initial directory
            options=QFileDialog.ShowDirsOnly
        )

        if selected_dir:
            util = GRIME_AI_COCO_Utils(selected_dir)
            print("Selected folder:", selected_dir)

            """Execute full validation and cleaning pipeline."""
            util.find_json_file()
            util.load_json()

            present, missing = util.check_images()
            if not missing:
                print("All images in JSON are present.")
                return
            else:
                # Create the message box
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Confirmation")
                msg_box.setText("Do you want to sync the JSON annotations with the available images?")

                # Use Question icon and add buttons
                msg_box.setIcon(QMessageBox.Question)
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

                if msg_box.exec_() == QMessageBox.Yes:
                    util.backup_original()
                    cleaned = util.clean_data(present)
                    util.write_json(cleaned)

                    msg_box.setWindowTitle("Completion")
                    msg_box.setText("JSON annotations sync'ed with the available images.")

                    # Use Question icon and add buttons
                    msg_box.setIcon(QMessageBox.Question)
                    msg_box.setStandardButtons(QMessageBox.Ok)

                    # Execute and capture response
                    if msg_box.exec_() == QMessageBox.Ok:
                        print("New JSON created for the images available in the folder.")


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubar_inspect_annotations(self):
        selected_dir = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select a Folder",
            options=QFileDialog.ShowDirsOnly
        )
        #directory = "C:/",  # initial directory

        if selected_dir:
            utils = GRIME_AI_COCO_Utils(selected_dir)
            print("Selected folder:", selected_dir)
            utils.load_coco()

            now = datetime.datetime.now()
            self.formatted_time = now.strftime('%Y%m%d_%H%M%S')
            inspection_file = f"{self.formatted_time}_Inspect_Annotations.xlsx"
            inspection_file = os.path.join(selected_dir, inspection_file)

            utils.write_image_label_counts_to_xlsx(inspection_file)
            print(f"Annotations Inspection: {inspection_file}")


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubarSaveSettings(self):
        utils = GRIME_AI_Save_Utils()
        utils.saveSettings()


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubar_ImageOrganizer(self):

        # --------------------------------------------------------------------------------------------------------------
        # JES - The Image Organizer functionality is currently in development and not intended for public release.
        # JES - Access is restricted to the development team. While it may be technically possible to circumvent
        # JES - these restrictions, GRIME Lab and its developers accept no liability or responsibility for any
        # JES - consequences arising from such actions.
        #
        # JES - Licensed under the Apache License, Version 2.0 (the "License");
        # JES - you may not use this file except in compliance with the License.
        # JES - You may obtain a copy of the License at:
        # JES -     http://www.apache.org/licenses/LICENSE-2.0
        # --------------------------------------------------------------------------------------------------------------
        if getpass.getuser() == "johns" or getpass.getuser() == "tgilmore10" or getpass.getuser() == "rissa3":
            pass
        else:
            strMessage = 'The Image Organizer functionality is not ready for general distribution.'
            msgBox = GRIME_AI_QMessageBox('Image Organizer Info', strMessage, QMessageBox.Close)
            response = msgBox.displayMsgBox()
            return

        # --------------------------------------------------------------------------------------------------------------
        # BEGIN PROCESSING
        # --------------------------------------------------------------------------------------------------------------
        print("[DEBUG] menubar_ImageOrganizer triggered")

        # Close any existing instance first
        try:
            if hasattr(self, "imageOrganizerDlg") and self.imageOrganizerDlg is not None:
                print("[DEBUG] Closing existing Image Organizer dialog")
                self.imageOrganizerDlg.close()
        except Exception as e:
            print(f"[WARN] Could not close existing dialog: {e}")

        # Create and show new dialog
        try:
            print("[DEBUG] Creating new Image Organizer dialog")
            self.imageOrganizerDlg = GRIME_AI_ImageOrganizerDlg(self)
            print("[DEBUG] Showing Image Organizer dialog")
            self.imageOrganizerDlg.show()
            print("[INFO] Image Organizer dialog launched successfully")
        except Exception as e:
            print(f"[ERROR] Failed to launch Image Organizer dialog: {e}")
            traceback.print_exc()  # full traceback to terminal


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def toolbarButtonReleaseNotes(self):
        global frame
        releaseNotesDlg = GRIME_AI_ReleaseNotesDlg(frame)

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
        global frame
        self.edgeDetectionDlg = GRIME_AI_EdgeDetectionDlg(frame)

        self.edgeDetectionDlg.edgeDetectionSignal.connect(self.edgeDetectionMethod)
        self.edgeDetectionDlg.featureDetectionSignal.connect(self.featureDetectionMethod)

        self.edgeDetectionDlg.show()

    # ==================================================================================================================
    #
    # ==================================================================================================================
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
        global frame
        self.fileFolderDlg = GRIME_AI_FileUtilitiesDlg(frame)

        self.fileFolderDlg.create_composite_slice_signal.connect(self.menubarCompositeSlices)
        self.fileFolderDlg.triage_images_signal.connect(self.toolbarButtonImageTriage_1)

        self.fileFolderDlg.accepted.connect(self.closeFilefolderDlg)
        self.fileFolderDlg.rejected.connect(self.closeFilefolderDlg)

        self.fileFolderDlg.show()

        try:
            global gFrameCount
            self.imageNavigationDlg.setImageCount(gFrameCount)
            self.imageNavigationDlg.reset()
        except Exception:
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
                self.imageNavigationDlg = GRIME_AI_ImageNavigationDlg(frame)
                self.imageNavigationDlg.imageIndexSignal.connect(self.getImageIndex)

                self.imageNavigationDlg.accepted.connect(self.closeNavigationDlg)
                self.imageNavigationDlg.rejected.connect(self.closeNavigationDlg)

                self.imageNavigationDlg.setImageList(dailyImagesList.getVisibleList())
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

                self.colorSegmentationDlg = GRIME_AI_ColorSegmentationDlg()

                self.colorSegmentationDlg.colorSegmentation_Signal.connect(self.colorSegmentation)
                self.colorSegmentationDlg.addROI_Signal.connect(self.trainROI)
                self.colorSegmentationDlg.deleteAllROI_Signal.connect(self.deleteAllROI)
                self.colorSegmentationDlg.buildFeatureFile_Signal.connect(self.buildFeatureFile)
                self.colorSegmentationDlg.universalTestButton_Signal.connect(self.universalTestButton)
                self.colorSegmentationDlg.refresh_rois_signal.connect(self.displayROIs)

                self.colorSegmentationDlg.greenness_index_signal.connect(self.greenness_index_changed)

                self.colorSegmentationDlg.close_signal.connect(self.closeColorSegmentationDlg)
                self.colorSegmentationDlg.accepted.connect(self.closeColorSegmentationDlg)
                self.colorSegmentationDlg.rejected.connect(self.closeColorSegmentationDlg)

                self.getColorSegmentationParams()

                self.colorSegmentationDlg.show()
            else:
                strMessage = 'Please close the Mask Editor toolbox if you want to use the Mask Editor toolbox.\nThis will be resolved in a future design change.'
                msgBox = GRIME_AI_QMessageBox('Tool Conflict', strMessage, QMessageBox.Yes | QMessageBox.No)
                response = msgBox.displayMsgBox()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def greenness_index_changed(self):
        if self.colorSegmentationDlg != None:
            self.getColorSegmentationParams()

            # UPDATE THE FEATURE TABLE
            self.initROITable(self.greenness_index_list)

            processLocalImage(self, currentImageIndex)
            #JES - TROUBLESHOOT DISPLAY ISSUE
            # self.refreshImage()


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
    def onMyToolBarCreateMask(self):

        # JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES
        # JES - PROVISIONAL - MASK CREATION IS NOT AVAILABLE FOR THE USGS SOFTWARE RELEASE
        strMessage = 'Mask Creation is not available in this software release.\nThis functionality may be consumed into other pre-existing functionality at some later date.'
        msgBox = GRIME_AI_QMessageBox('Tool Conflict', strMessage, QMessageBox.Yes | QMessageBox.No)
        response = msgBox.displayMsgBox(on_top=True)
        return
        # ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^

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
                colorBar = GRIME_AI_Color.create_color_bar(hist, colorClusters[0:5])

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

        self.colorSegmentationDlg.disable_spinbox_color_clusters(False)

        processLocalImage(self)

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

        ###JES if event.type() == QtCore.QEvent.MouseButtonDblClick and source is self.labelOriginalImage:
            # labelEdgeImageDoubleClickEvent(self)
            # labelMouseDoubleClickEvent(self)
            ###JES NEON_labelOriginalImageDoubleClickEvent(self)

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

        global hyperparameterDlg
        if hyperparameterDlg != None:
            hyperparameterDlg.close()

        #webdriver.Chrome.quit()

        QMainWindow.closeEvent(self, event)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def fetchImageList(self, imageFolder, bRecursive=False):
        global imageFileFolder
        imageFileFolder = imageFolder
        fetchLocalImageList(self, imageFileFolder, bRecursive, False) #, start_date, end_date, start_time, end_time)

        try:
            global gFrameCount
            global dailyImagesList

            self.onMyToolBarImageNavigation()
            self.imageNavigationDlg.setImageCount(gFrameCount)
            self.imageNavigationDlg.setImageList(dailyImagesList.getVisibleList())
            self.imageNavigationDlg.reset()
        except Exception:
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

        #JES - TEMPORARILY SET NITRATE DATA ('should only be one nitrate product') AS THE DEFAULT SELECTION
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        itemNitrate = self.NEON_listboxSiteProducts.findItems('Nitrate', QtCore.Qt.MatchContains)
        nIndex = 0
        if len(itemNitrate) > 0:
            for item in itemNitrate:
                nIndex = self.NEON_listboxSiteProducts.row(item)
                self.NEON_listboxSiteProducts.setCurrentRow(nIndex)

            NEON_updateProductTable(self, nIndex)

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #JES
        #JES - TEMPORARILY SET NITRATE DATA ('should only be one nitrate product') AS THE DEFAULT SELECTION
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        item20002 = self.NEON_listboxSiteProducts.findItems('20002', QtCore.Qt.MatchContains)
        nIndex = 0
        if len(item20002) > 0:
            for item in item20002:
                nIndex = self.NEON_listboxSiteProducts.row(item)
                self.NEON_listboxSiteProducts.setCurrentRow(nIndex)

            NEON_updateProductTable(self, nIndex)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #JES

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

            # IMAGE INTENSITY CALCULATIONS
            intensity = cv2.mean(gray)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway
            strIntensity = '%3.3f' % (intensity)

            # COMPUTE ENTROPY FOR ENTIRE IMAGE
            entropyValue = self.calcEntropy(gray)
            strEntropy = '%3.3f' % (entropyValue)
        else:
            strIntensity = '---'
            strEntropy = '---'

        return strIntensity, strEntropy

    # ==================================================================================================================
    #  WHOLE IMAGE - EXTRACT FEATURES (GREENNESS INDEX, INTENSITY, ENTROPY, ETC.)
    # ==================================================================================================================0
    def whole_image_feature_extraction(self, img):
        features = self.compute_whole_image_features(img)
        self.display_whole_image_features(features)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_whole_image_features(self, img):
        """Return computed features for the whole image without touching the UI."""
        strIntensity, strEntropy = self.WholeImage_ExtractFeatures(
            img, self.colorSegmentationParams.wholeImage
        )

        # Extract dominant HSV colors
        hist, colorClusters = GRIME_AI_Color.extractDominant_HSV(
            img, self.colorSegmentationParams.numColorClusters
        )
        colorBar = GRIME_AI_Color.create_color_bar(hist, colorClusters)

        # Compute greenness values
        greenness_values = []
        for index, greenness in enumerate(self.greenness_index_list):
            greenness_updated = GRIME_AI_Vegetation_Indices().get_greenness(greenness, img)
            self.greenness_index_list[index] = greenness_updated
            greenness_values.append(greenness_updated.get_value())

        return {
            "intensity": strIntensity,
            "entropy": strEntropy,
            "colorBar": colorBar,
            "greenness": greenness_values,
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def display_whole_image_features(self, features):
        """Update the Qt table with the computed features."""
        nRow = self.tableWidget_ROIList.rowCount()
        if nRow == 0:
            self.tableWidget_ROIList.insertRow(nRow)

        # Column 0: Label
        wholeImageLabel = QtWidgets.QLabel("Whole Image")
        self.tableWidget_ROIList.setCellWidget(0, 0, wholeImageLabel)

        # Column 1: Color bar
        qImg = QImage(
            features["colorBar"].data,
            features["colorBar"].shape[1],
            features["colorBar"].shape[0],
            QImage.Format_BGR888,
        )
        pixmap = QPixmap.fromImage(qImg)

        # Scale to fit cell
        cell_width = self.tableWidget_ROIList.columnWidth(0)
        cell_height = self.tableWidget_ROIList.rowHeight(0)
        pixmap_scaled = pixmap.scaledToHeight(cell_height, QtCore.Qt.SmoothTransformation)
        if pixmap_scaled.width() != cell_width:
            h_scale = cell_width / pixmap_scaled.width()
            transform = QtGui.QTransform()
            transform.scale(h_scale, 1)
            pixmap_scaled = pixmap_scaled.transformed(transform, QtCore.Qt.SmoothTransformation)

        label = QtWidgets.QLabel()
        label.setPixmap(pixmap_scaled)
        self.tableWidget_ROIList.setCellWidget(0, 1, label)

        # Column 2: n/a
        na_Label = QtWidgets.QLabel("n/a")
        self.tableWidget_ROIList.setCellWidget(0, 2, na_Label)

        # Columns 3+: Greenness values
        col = 3
        for g in features["greenness"]:
            greennessLabel = QtWidgets.QLabel("{:.3f}".format(g))
            self.tableWidget_ROIList.setCellWidget(0, col, greennessLabel)
            col += 1

        # Intensity
        intensityLabel = QtWidgets.QLabel(features["intensity"])
        self.tableWidget_ROIList.setCellWidget(0, col, intensityLabel)
        col += 1

        # Entropy
        entropyLabel = QtWidgets.QLabel(features["entropy"])
        self.tableWidget_ROIList.setCellWidget(0, col, entropyLabel)

    # ==================================================================================================================
    #  ROI (region-of-interest) - EXTRACT FEATURES (GREENNESS INDEX, INTENSITY, ENTROPY, ETC.)
    # ==================================================================================================================
    def roi_feature_extraction(self, painter, img, roiList):
        progressBar = QProgressWheel()
        progressBar.setRange(0, len(roiList))

        for row, roiObj in enumerate(roiList, start=1):
            progressBar.setValue(row)

            try:
                features = self.compute_roi_features(roiObj, img)
                self.display_roi_features(row, features)
                self.draw_roi_overlay(painter, roiObj)
            except Exception as e:
                print(f"Error processing ROI {roiObj.getROIName()}: {e}")

        progressBar.close()
        del progressBar

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @dataclass
    class ROIFeatures:
        roi_name: str
        color_bar: np.ndarray
        greenness: list[float]
        intensity: float
        entropy: float

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_roi_features(self, roiObj, img) -> ROIFeatures:
        rgb = extractROI(roiObj.getImageROI(), img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Dominant HSV colors
        hist, colorClusters = GRIME_AI_Color.extractDominant_HSV(rgb, roiObj.getNumColorClusters())
        colorBar = GRIME_AI_Color.create_color_bar(hist, colorClusters)

        # Greenness indices
        greenness_values = []
        for index, greenness in enumerate(self.greenness_index_list):
            greenness_updated = GRIME_AI_Vegetation_Indices().get_greenness(greenness, rgb)
            self.greenness_index_list[index] = greenness_updated
            greenness_values.append(greenness_updated.get_value())

        # Intensity
        intensity = cv2.mean(gray)[0]

        # Entropy
        entropy = self.calcEntropy(gray)

        return ROIFeatures(
            roi_name=roiObj.getROIName(),
            color_bar=colorBar,
            greenness=greenness_values,
            intensity=intensity,
            entropy=entropy,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def display_roi_features(self, row: int, features: ROIFeatures):
        # Column 0: ROI name
        self.tableWidget_ROIList.setCellWidget(row, 0, QtWidgets.QLabel(features.roi_name))

        # Column 2: Color bar
        qImg = QImage(features.color_bar.data,
                      features.color_bar.shape[1],
                      features.color_bar.shape[0],
                      QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)

        cell_width = self.tableWidget_ROIList.columnWidth(2)
        cell_height = self.tableWidget_ROIList.rowHeight(row)
        pixmap_scaled = pixmap.scaled(cell_width, cell_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        label = QtWidgets.QLabel()
        label.setPixmap(pixmap_scaled)
        self.tableWidget_ROIList.setCellWidget(row, 2, label)

        # Columns 3+: Greenness
        col = 3
        for g in features.greenness:
            self.tableWidget_ROIList.setCellWidget(row, col, QtWidgets.QLabel(f"{g:.3f}"))
            col += 1

        # Intensity
        self.tableWidget_ROIList.setCellWidget(row, col, QtWidgets.QLabel(f"{features.intensity:.4f}"))
        col += 1

        # Entropy
        self.tableWidget_ROIList.setCellWidget(row, col, QtWidgets.QLabel(f"{features.entropy:.4f}"))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def draw_roi_overlay(self, painter, roiObj):
        pen = QPen(QtCore.Qt.red, 1, QtCore.Qt.SolidLine)
        painter.setPen(pen)

        if roiObj.getROIShape() == ROIShape.RECTANGLE:
            painter.drawRect(roiObj.getDisplayROI())
        elif roiObj.getROIShape() == ROIShape.ELLIPSE:
            painter.drawEllipse(roiObj.getDisplayROI())

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(
            roiObj.getDisplayROI().x(),
            roiObj.getDisplayROI().y() - 16,
            50, 16,
            QtCore.Qt.AlignLeft,
            roiObj.getROIName()
        )


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
    imageCount = GRIME_AI_Utils().get_image_count(filePath, extensions)

    # RECURSE AND TRAVERSE FROM THE SPECIFIED FOLDER DOWN TO DETERMINE THE DATE RANGE FOR THE IMAGES FOUND
    file_count, files = GRIME_AI_Utils().getFileList(filePath, extensions, bFetchRecursive)

    if bShow_GUI:
        progressBar = QProgressWheel()
        progressBar.setRange(0, file_count + 1)
        progressBar.show()

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
    print("File Count: ", file_count)

    image_index = 0
    while image_index < file_count:
        file = files[image_index]
        if bShow_GUI:
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
                except Exception:
                    # assume the filename contains the timestamp for the image (assumes the image file is a PBT image)
                    bEXIFDataFound = False

                    try:
                        nHours = int(str(strTime[0:2]))
                        nMins = int(str(strTime[2:4]))
                        nSecs = int(str(strTime[4:6]))
                    except Exception:
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

        image_index += 1

    dailyImagesList.setVisibleList(List)

    global gFrameCount
    gFrameCount = len(dailyImagesList.getVisibleList())
    gWebImageCount = len(dailyImagesList.getVisibleList())

    # INIT SPINBOX CONTROLS BASED UPON NUMBER OF IMAGES AVAILABLE
    # dailyURLvisible = []

    # clean-up before exiting function
    # 1. close and delete the progress bar
    # 2. close the EXIF log file, if opened
    if bShow_GUI:
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
    ##JES videoFileList = [str(pp) for pp in videoFilePath.glob("**/*.jpg")]
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

    # ------------------------------------------------------------------------------------------------------------------
    # DISPLAY IMAGE FROM NEON SITE
    # ------------------------------------------------------------------------------------------------------------------
    if currentImage:
        numpyImg = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

        scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        width = scaledCurrentImage.width()
        height = scaledCurrentImage.height()

        currentImageRescaled = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        img = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

        self.whole_image_feature_extraction(img)

        if g_displayOptions.displayROIs:
            painter = QPainter(currentImageRescaled)

        self.roi_feature_extraction(painter, img, self.roiList)

        if g_displayOptions.displayROIs:
            painter.end()

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

            myProcessImage = GRIME_AI_ProcessImage()

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
def closehyperparameterDlg():

    global hyperparameterDlg
    del hyperparameterDlg
    hyperparameterDlg = None


# ======================================================================================================================
#
# ======================================================================================================================
def extractROI(rect, image):
    return(image[rect.y():rect.y() + rect.height(), rect.x():rect.x() + rect.width()])


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
def NEON_updateSiteInfo(self):

    # EXTRACT THE SITE ID FOR THE SELECTED ITEM
    siteID = self.NEON_listboxSites.currentItem().text()

    global SITECODE
    SITECODE = siteID.split(' - ')[0]
    print (f'Fetching site info from NEON server for {SITECODE}...')
    siteInfo = NEON_API().FetchSiteInfoFromNEON(SERVER, SITECODE)

    global DOMAINCODE
    DOMAINCODE = siteInfo['data']['domainCode']

    keys = siteInfo['data'].keys()
    items = [f"{key}: {str(siteInfo['data'][key])}" for key in keys]

    if 0:
        self.current_site_info = items[0:9]

    self.NEON_listboxSiteInfo.clear()
    self.NEON_listboxSiteInfo.addItems(items)

    self.labelNEONSiteDetails.setText(SITECODE)

    return (SITECODE)


# ======================================================================================================================
# THIS FUNCTION WILL UPDATE THE PRODUCT TABLE IN THE GUI WITH THE PRODUCTS THAT ARE AVAILABLE FOR A SPECIFIC SITE.
# ======================================================================================================================
def NEON_updateProductTable(self, item):
    products = self.NEON_listboxSiteProducts.selectedItems()

    #JES: FUTURE CONSIDERATION - MUST MAKE CODE DYNAMIC TO ONLY DELETE UNSELECTED ITEMS
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
        #nYear, nMonth, nDay = GRIME_AI_PhenoCam().getStartDate()
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        date_widget.setDisabled(True)
        self.NEON_tableProducts.setCellWidget(i, m, date_widget)

        m += 1
        nYear = 1970
        nMonth = 1
        nDay = 1
        #nYear, nMonth, nDay = GRIME_AI_PhenoCam().getEndDate()
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        date_widget.setDisabled(True)
        self.NEON_tableProducts.setCellWidget(i, m, date_widget)

        # --------------------
        # SET THE CALENDAR START AND END DATE THE SAME USING THE DATE FOR THE LAST DAY FOR WHICH DATA IS AVAILABLE
        # --------------------
        nYear, nMonth, nDay = GRIME_AI_PhenoCam().getEndDate()

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        #JES - SET FOR TODAY'S DATE. USER'S MAY NOT WANT TO GO BACK MANY YEARS.
        #nYear, nMonth, nDay = GRIME_AI_PhenoCam.getStartDate()
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
    start_date, start_time, end_date, end_time = GRIME_AI_ProductTable().fetchTableDates(tableWidget, nRow)

    if bUniqueDates == False:
        for i in range(tableWidget.rowCount()):
            tableWidget.cellWidget(i, 4).setDate(start_date)
            tableWidget.cellWidget(i, 5).setDate(end_date)
    else:
        tableWidget.cellWidget(nRow, 4).setDate(start_date)
        tableWidget.cellWidget(nRow, 5).setDate(end_date)

    #imageCount = GRIME_AI_PhenoCam.getPhenocamImageCount(SITECODE, DOMAINCODE, start_date, end_date, start_time, end_time)

    #tableWidget.setItem(nRow, 2, QTableWidgetItem(str(imageCount)))


# ======================================================================================================================
#
# ======================================================================================================================
def DP1_20002_fetchImageList(self, nRow, start_date, end_date, start_time, end_time, downloadsFilePath):

    imageList = DP1_20002_buildImageList(self, nRow, start_date, end_date, start_time, end_time)

    DP1_20002_downloadImages(self, imageList, downloadsFilePath)

# ======================================================================================================================
#
# ======================================================================================================================
def DP1_20002_buildImageList(self, nRow, start_date, end_date, start_time, end_time):
    global SITECODE, DOMAINCODE, dailyImagesList, gWebImageCount

    if nRow <= -1:
        return []

    delta = end_date - start_date

    # CREATE PROGRESS BAR
    progressBarDates = QProgressWheel()
    progressBarDates.setRange(0, delta.days + 1)
    progressBarDates.setWindowTitle('Build image list...')
    progressBarDates.show()

    # CLEAR PREVIOUS LIST
    dailyImagesList.clear()

    i = 1
    while start_date <= end_date:
        ymd = start_date.strftime("%Y-%d-%b")
        progressBarDates.setWindowTitle(ymd)
        progressBarDates.setValue(float(i) / float(delta.days + 1) * delta.days)
        progressBarDates.repaint()
        i += 1

        QCoreApplication.processEvents()

        # Build URL
        dailyURLvisible = (
            f"https://phenocam.nau.edu/webcam/browse/NEON.{DOMAINCODE}.{SITECODE}.DP1.20002/"
            f"{start_date.year}/{str(start_date.month).zfill(2)}/{str(start_date.day).zfill(2)}"
        )

        phenoCam = GRIME_AI_PhenoCam()
        tmpList = phenoCam.getVisibleImages(dailyURLvisible, start_time, end_time)

        dailyImagesList.setVisibleList(tmpList.getVisibleList())

        start_date += datetime.timedelta(days=1)

    progressBarDates.close()

    gWebImageCount = len(dailyImagesList.getVisibleList())
    return dailyImagesList.getVisibleList()

def DP1_20002_downloadImages(self, imageList, downloadsFilePath):
    if not imageList:
        return

    progressBarDownloads = QProgressWheel()
    progressBarDownloads.setRange(0, len(imageList) + 1)
    progressBarDownloads.setWindowTitle('Download & Save Images...')
    progressBarDownloads.show()

    for i, image in enumerate(imageList):
        progressBarDownloads.setValue(float(i) / float(len(imageList) + 1) * len(imageList))

        filename = os.path.basename(image.fullPathAndFilename)
        if not os.path.exists(downloadsFilePath):
            os.makedirs(downloadsFilePath)
        completeFilename = os.path.join(downloadsFilePath, filename)

        if not os.path.isfile(completeFilename):
            urllib.request.urlretrieve(image.fullPathAndFilename, completeFilename)

    progressBarDownloads.close()

#jes LET THE CALLING FUNCTION BE RESPONSIBLE FOR REPORTING DOWNLOAD COMPLETION.
#jes MODIFY THIS IN A FUTURE RELEASE TO RETURN A PASS/FAIL MESSAGE TO THE FUNCTION THAT INVOKED THIS FUNCTION.
#jes strMessage = 'Data download is complete!'
#jes msgBox = GRIME_AI_QMessageBox('Data Download', strMessage)
#jes response = msgBox.displayMsgBox()


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
    # SAVE DOWNLOADED DATA TO THE USER GRIME-AI FOLDER THAT IS AUTOMATICALLY CREATED, IF IT DOES NOT EXIST,
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
            #NEON_download_file_path = os.path.join(NEON_download_file_path, 'GRIME-AI')

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
        GRIME_AI_ProductTableObj = GRIME_AI_ProductTable()
        start_date, start_time, end_date, end_time = GRIME_AI_ProductTableObj.fetchTableDates(self.NEON_tableProducts, nRow)

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
                    downloadsFilePath = os.path.join(NEON_download_file_path, '../../data')
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
        #         #JES - USE NITRATE DATA FOR DEVELOPING GENERIC CSV READING AND DATA GRAPHING CAPABILITIES
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

    szWindowsTitle = "GRIME AI" + " " + SW_VERSION + " - John E. Stranzl Jr."

    _translate = QtCore.QCoreApplication.translate
    MainWindow.setWindowTitle(_translate(szWindowsTitle, szWindowsTitle))

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


def run_gui():
    global frame

    # If Hydra is already initialized, clear it
    from GRIME_AI_Save_Utils import GRIME_AI_Save_Utils

    settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
    print(settings_folder)
    hydra_working_folder = os.path.normpath(os.path.join(settings_folder, "MyHydraOutputs"))
    sys.argv.append(f"hydra.run.dir={hydra_working_folder}")

    if 0:
        if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.core.global_hydra.GlobalHydra.instance().clear()
        else:
            hydra.initialize(config_path=None)

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

    # Run the program
    sys.exit(app.exec())


def my_main():
    global bShow_GUI

    # Main parser
    parser = argparse.ArgumentParser(description='CLI for GRIME AI')

    # Subparsers
    subparsers = parser.add_subparsers(dest='command')

    # Triage parser
    triage_parser = subparsers.add_parser('triage', help='Perform Image Triage')
    triage_parser.add_argument("-m", "--min", type=float, required=False, default=65.0,
                               help="Minimum brightness (default: 60.0).")
    triage_parser.add_argument("-x", "--max", type=float, required=False, default=180.0,
                               help="Maximum brightness (default: 180.0).")
    triage_parser.add_argument("-r", "--report", type=bool, required=False, default=True, help="Generate a report.")
    triage_parser.add_argument("-v", "--move", type=bool, required=False, default=True,
                               help="Move images to subfolder.")
    triage_parser.add_argument("-p", "--poly", type=bool, required=False, default=False, help="Save polyline images.")
    triage_parser.add_argument("-a", "--alignment", type=bool, required=False, default=False,
                               help="Correct rotated images.")
    triage_parser.add_argument("-d", "--delta", type=float, required=False, default=0.25,
                               help="Rotation tolerance for considering an image over- or under-rotated.")
    triage_parser.add_argument("-i", "--image", type=float, required=False, default="",
                               help="Image to use as ground truth for rotation angle.")
    triage_parser.add_argument("-b", "--blurthreshold", type=float, required=False, default=17.50,
                               help="Blur threshold for FFT.")
    triage_parser.add_argument("-s", "--shift", type=int, required=False, default=60,
                               help="FFT Shift size for blur estimation.")
    triage_parser.add_argument("-f", "--folder", type=str, required=True, help="A folder must be specified.")

    # Slice parser
    slice_parser = subparsers.add_parser('slice', help='Perform Image Slicing')
    slice_parser.add_argument("-c", "--center", type=int, required=True, default=0, help="Center of slice (in pixels).")
    slice_parser.add_argument("-w", "--width", type=int, required=True, default=20,
                              help="Width of the slice (in pixels).")
    slice_parser.add_argument("-f", "--folder", type=str, required=True, help="A folder must be specified.")

    # COCO parser
    coco_parser = subparsers.add_parser('coco', help='Generate COCO annotation file from images and masks')
    coco_parser.add_argument("--folder", required=True,
                             help="Folder containing image files (and optional masks).")
    coco_parser.add_argument("--shared-mask", required=False,
                             help="If provided, uses this single mask file for all images in the folder.")
    coco_parser.add_argument("--output", required=False,
                             help="(Optional) Override output JSON file path.")

    # Segment parser
    segment_parser = subparsers.add_parser('segment', help='Segment images using a Hydra-configured model')
    segment_parser.add_argument('--model_file',
                                 help='Path to the model checkpoint file (e.g., ckpt.pth)')
    segment_parser.add_argument('--images_folder', help='Directory containing images to segment')
    segment_parser.add_argument('--overrides', nargs='*', help='Optional Hydra overrides (e.g. training.lr=1e-4)')

    # Custom help handling
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Global Help: CLI for GRIME AI")
        parser.print_help()  # General help for the main parser
        print("\nHelp for 'triage' command:")
        triage_parser.print_help()  # Help for triage subparser
        print("\nHelp for 'slice' command:")
        slice_parser.print_help()  # Help for slice subparser
        print("\nHelp for 'segment' command:")
        segment_parser.print_help()  # Help for segment subparser
        sys.exit(0)  # Exit after displaying help

    args = parser.parse_args()

    # Check if no arguments were provided
    if len(sys.argv) == 1:
        bShow_GUI = True
        run_gui()
    else:
        bShow_GUI = False
        run_cli(args)

    # SHOW MAIN WINDOW
    #frame.show()

def run_cli(args):
    if args.command == 'triage':
        create_report = True
        move_images = True
        fetch_recursive = False
        blur_threshold = 17.50
        shift_size = 60
        #min_val = 65.0
        #max_val = 180.0
        correct_alignment = False
        save_poly_lines = False
        reference_image_filename = ""
        rotation_threshold = 0.15

        print(args.command)

        print("These are the Triage parameters:", args.folder, args.min, args.max)
        myTriage = GRIME_AI_ImageTriage(False)
        myTriage.cleanImages(args.folder, \
                             fetch_recursive, \
                             blur_threshold, shift_size, \
                             args.min, args.max, \
                             create_report, move_images, \
                             correct_alignment, save_poly_lines,
                             reference_image_filename, rotation_threshold)

        print('Image triage is complete!')
    elif args.command == 'slice':
        filenames = cli_fetchLocalImageList(args.folder)

        compositeSlices = GRIME_AI_CompositeSlices(args.center, args.width, False)
        compositeSlices.create_composite_image(filenames, args.folder+'\compositeSlices')

        print("Composite slice complete!")
    elif args.command == "coco":
        # Import your CocoGenerator class from coco_generator.py
        from coco_generator import CocoGenerator
        print("[INFO] Running COCO generation command...")
        folder = Path(args.folder)
        output_path = Path(args.output) if args.output else folder / "instances_default.json"
        if args.shared_mask:
            shared_mask = Path(args.shared_mask)
            if not shared_mask.exists():
                print(f"[ERROR] Shared mask file not found: {shared_mask}")
                sys.exit(1)
            # Instantiate in shared mask mode
            generator = CocoGenerator(folder=folder, shared_mask=shared_mask, output_path=output_path)
        else:
            # Instantiate for one-to-one mode
            generator = CocoGenerator(folder=folder, output_path=output_path)
        generator.generate_annotations()
    elif args.command == 'segment':
        # 1. Point Hydra at your config directory
        initialize(config_path=myconfig_path, job_name="segment_image_cli")

        # 2. Gather any user-provided overrides
        overrides = args.overrides or []

        # 3. Inject the model checkpoint path and the folder of images
        overrides.extend([
            f"model_file={args.model_file}",
            f"images_folder={args.images_folder}"
        ])

        # 4. Build the DictConfig from base + overrides
        cfg: DictConfig = compose(
            config_name=myconfig_name,
            overrides=overrides
        )

        # 5. Invoke the Hydra-decorated entry point
        load_model_main(cfg)
    else:
        parser.print_help()
        sys.exit(1)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
def cli_fetchLocalImageList(filePath, bFetchRecursive=False):

    # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    List = []

    # RECURSE AND TRAVERSE FROM THE SPECIFIED FOLDER DOWN TO DETERMINE THE DATE RANGE FOR THE IMAGES FOUND
    file_count, files = GRIME_AI_Utils().getFileList(filePath, extensions, bFetchRecursive)

    for image_index, file in enumerate(files):
        ext = os.path.splitext(file)[-1].lower()

        if ext in extensions:
            List.append(imageData(file, 0, 0, 0))

    return List


# ======================================================================================================================
#
# ======================================================================================================================
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()

'''
dirname = os.path.dirname(__file__)
myconfig_path=os.path.normpath(os.path.join(dirname, "sam2\\sam2\\configs\\sam2.1"))
print(myconfig_path)
myconfig_name=os.path.normpath(os.path.join(dirname, "sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_l.yaml"))
print(myconfig_name)
'''

# Resolve config path relative to this script
dirname = os.path.dirname(os.path.abspath(__file__))
myconfig_path = os.path.join(dirname, "sam2", "sam2", "configs", "sam2.1")
print("Config path:", myconfig_path)
myconfig_name = "sam2.1_hiera_l"
###JES myconfig_name = os.path.join(dirname, "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l")
print("Config name:", myconfig_name)

# ======================================================================================================================
#
# ======================================================================================================================
@hydra.main(config_path=myconfig_path, config_name=myconfig_name, version_base="1.3")
def train_main(cfg: DictConfig) -> None:
    # If Hydra is already initialized, clear it
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    print("Hydra SAM2 training config:")
    print(OmegaConf.to_yaml(cfg))

    global hyperparameterDlg
    if hyperparameterDlg != None:
        copy_original_image = hyperparameterDlg.get_copy_original_image()
        save_masks = hyperparameterDlg.get_saved_masks()
        selected_label_categories = hyperparameterDlg.get_selected_label_categories()
        selected_training_model = hyperparameterDlg.get_selected_training_model()
        selected_segment_model = hyperparameterDlg.get_selected_segment_model()

        hyperparameterDlg.close()
        del hyperparameterDlg
        hyperparameterDlg = None

        # Instantiate your SAM2 training object (assumed to accept cfg)
        print("Instantiate MLModelTraining class...")
        myML_Dispatcher = MLModelTraining(cfg)
        print("Execute ML training...")

        myML_Dispatcher.Model_Training_Dispatcher(cfg=cfg, mode=selected_training_model)

# ======================================================================================================================
#
# ======================================================================================================================
@hydra.main(config_path=myconfig_path, config_name=myconfig_name, version_base="1.3")
def load_model_main(cfg: DictConfig) -> None:
    # If Hydra is already initialized, clear it
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    print("Hydra SAM2 training config:")
    print(OmegaConf.to_yaml(cfg))

    global hyperparameterDlg
    if hyperparameterDlg != None:
        copy_original_image = hyperparameterDlg.get_copy_original_image()
        save_masks = hyperparameterDlg.get_saved_masks()
        selected_label_categories = hyperparameterDlg.get_selected_label_categories()
        selected_training_model = hyperparameterDlg.get_selected_training_model()
        selected_segment_model = hyperparameterDlg.get_selected_segment_model()

        hyperparameterDlg.close()
        del hyperparameterDlg
        hyperparameterDlg = None

        mySegmentation = MLImageSegmentation(cfg)
        mySegmentation.ML_Segmentation_Dispatcher(copy_original_image, save_masks, selected_label_categories, mode=selected_segment_model)

# ======================================================================================================================
#
# ======================================================================================================================
if __name__ == '__main__':

    my_main()


