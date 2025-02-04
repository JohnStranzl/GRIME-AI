import os
import json
import re

import csv
import cv2
import numpy as np

import pandas as pd
from datetime import datetime

import urllib.request
from urllib.request import urlopen
import ssl

from siteData import siteData

from pathlib import Path

from GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI_Save_Utils import JsonEditor

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_Utils:

    def __init__(self, parent=None):
        self.className = "GRIME_AI_Utils"
        self.instance = 1

    # ======================================================================================================================
    # Converts a QImage into an opencv MAT format
    # ======================================================================================================================
    def convertQImageToMat(self, incomingImage):
        incomingImage = incomingImage.convertToFormat(4)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())

        # COPY THE DATA
        arr = np.array(ptr).reshape(height, width, 4)

        # CONVERT FROM RGBA TO RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

        return arr


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)


    # ======================================================================================================================
    # THIS FUNCTION PARSES THE FIELD SITE TABLE THAT IS FETCHED FROM THE NEON SITE.
    # ======================================================================================================================
    def parseCSV(self, filename_with_path):
        # FULLY QUALIFIED PATH OF THE CSV DOWNLOADED ONTO THE LOCAL COMPUTER
        filename = filename_with_path

        fields = []
        rows = []
        siteList = []

        with open(filename, 'r') as data:
            dict_Reader = csv.DictReader(data)
            ordered_dict_from_csv = list(dict_Reader)[0]
            dict_from_csv = dict(ordered_dict_from_csv)
            keys = dict_from_csv.keys()

        # READ CSV FILE
        with open(filename, 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)

            # EXTRACT FIELD NAMES FROM THE FIRST ROW OF THE CSV
            fields = next(csvreader)

            # READ ONE ROW AT A TIME AND APPEND INTO A LIST
            for row in csvreader:
                rows.append(row)

        fieldSiteIDIndex = fields.index("field_site_id")
        fieldSiteNameIndex = fields.index("field_site_name")
        fieldPhenocamIndex = fields.index("field_phenocams")
        fieldLatitudeIndex = fields.index("field_latitude")
        fieldLongitudeIndex = fields.index("field_longitude")

        for row in rows:
            siteList.append(siteData(row[fieldSiteIDIndex], row[fieldSiteNameIndex], row[fieldPhenocamIndex],
                                     row[fieldLatitudeIndex], row[fieldLongitudeIndex]))

        return siteList

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getImageCount(self, folder, extensions):
        imageCount = 0

        for filename in os.listdir(folder):
            ext = os.path.splitext(filename)[-1].lower()
            if ext in extensions:
                imageCount += 1

        return imageCount


    def getImageCount_walk(self, folder, extensions):
        imageCount = 0

        for root, dirs, files in os.walk(folder):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in extensions:
                    imageCount += 1

        return imageCount

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getFileList(self, folder='', extensions='jpg', bFetchRecursive=False):

        filenames = []

        if bFetchRecursive:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    ext = os.path.splitext(file)[-1].lower()
                    if ext in extensions:
                        filenames.append(os.path.join(root, file))
        else:
            for imageIndex, file in enumerate(os.listdir(folder)):
                ext = os.path.splitext(file)[-1].lower()
                if ext in extensions:
                    filenames.append(os.path.join(folder, file))

        return filenames


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def drawGridOnImage(self, img):
        GRID_SIZE = 100

        height, width, channels = img.shape
        for x in range(0, width - 1, GRID_SIZE):
            cv2.line(img, (x, 0), (x, height), (255, 0, 0), 1, 1)

        for y in range(0, width - 1, GRID_SIZE):
            cv2.line(img, (0, y), (height, y), (255, 0, 0), 1, 1)

        # cv2.imshow('Hehe', numpyImage)
        # key = cv2.waitKey(0)


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def create_GRIME_folders(self, full):
        # --------------------------------------------------------------------------------------------------------------
        # CREATE A GRIME-AI FOLDER IN THE USER'S DOCUMENTS FOLDER
        # <user>/Documents/GRIME-AI
        # --------------------------------------------------------------------------------------------------------------
        rootFolder = os.path.expanduser('~')
        rootFolder = os.path.join(rootFolder, 'Documents', 'GRIMe-AI')
        if not os.path.exists(rootFolder):
            os.mkdir(rootFolder)

        # --------------------------------------------------------------------------------------------------------------
        # CREATE A SETTINGS FOLDERS IN THE USER'S GRIME-AI FOLDER IN WHICH TO STORE THE USER'S PROGRAM SETTINGS
        # <user>/Documents/GRIME-AI/Settings
        # --------------------------------------------------------------------------------------------------------------
        configFilePath = os.path.join(rootFolder, 'Settings')
        if not os.path.exists(configFilePath):
            os.mkdir(configFilePath)

        # CHECK TO SEE IF THE GRIME-AI CONFIGURATION FILE EXISTS. IF IT DOES NOT, THEN CREATE IT USING touch
        configFile = os.path.join(configFilePath, 'GRIMe-AI.json')
        if not os.path.isfile(configFile):
            configFileWithPath = Path(configFile)
            configFileWithPath.touch(exist_ok=True)

        # --------------------------------------------------------------------------------------------------------------
        # CREATE A SCRATCHPAD FOLDER IN THE USER'S GRIME-AI FOLDER AS A WORKAROUND TO THE NEON API "PATH TOO LONG" ISSUE
        # <user>/Documents/GRIME-AI/Settings
        # --------------------------------------------------------------------------------------------------------------
        scratchpadFilePath = os.path.join(rootFolder, 'Scratchpad')
        if not os.path.exists(scratchpadFilePath):
            os.mkdir(scratchpadFilePath)

        JsonEditor().update_json_entry('Scratchpad_folder', os.path.normpath(scratchpadFilePath))

        # --------------------------------------------------------------------------------------------------------------
        # CREATE DEFAULT FOLDERS INTO WHICH  DOWNLOADED DATA WILL BE SAVED FOR SUPPORTED PRODUCTS
        # e.g., NEON, USGS, PBT (and create an OTHER folder into which a user can download data from other
        # sources
        # --------------------------------------------------------------------------------------------------------------
        default_folders = ['Downloads/NEON',  'Downloads/NEON/Images',  'Downloads/NEON/Data',  'Downloads/NEON/Videos',  'Downloads/NEON/EXIF', 'Downloads/NEON/MetaData', \
                           'Downloads/USGS',  'Downloads/USGS/Images',  'Downloads/USGS/Data',  'Downloads/USGS/Videos',  'Downloads/USGS/EXIF', \
                           'Downloads/PBT',   'Downloads/PBT/Images',   'Downloads/PBT/Data',   'Downloads/PBT/Videos',   'Downloads/PBT/EXIF', \
                           'Downloads/OTHER', 'Downloads/OTHER/Images', 'Downloads/OTHER/Data', 'Downloads/OTHER/Videos', 'Downloads/OTHER/EXIF', \
                           'Downloads/KOLA',  'Downloads/KOLA/Images',  'Downloads/KOLA/Data',  'Downloads/KOLA/Videos',  'Downloads/KOLA/EXIF']

        for folder in default_folders:
            make_these_folders = os.path.join(rootFolder, folder)
            if not os.path.exists(make_these_folders):
                os.makedirs(make_these_folders)

            # SAVE THESE DEFAULT IMAGE FOLDER PATHS TO THE GRIME-AI CONFIGURATION FILE (GRIME-AI.json) ONLY IF THEY DON'T EXIST
            # The regular expression pattern to find the word 'images' and extract the word between slashes
            pattern = r'\/([^\/]+)\/Images\b'

            # Using re.search to find the match
            match = re.search(pattern, make_these_folders, re.IGNORECASE)

            if match:
                word_between_slashes = match.group(1)

                entry_key = word_between_slashes + "_Image_Folder"

                # FIRST TRY TO READ THE ENTRY KEY VALUE. IF IT IS EMPTY, SET IT TO THE DEFAULT IMAGE FOLDER PATH
                if not JsonEditor().getValue(entry_key):
                    JsonEditor().update_json_entry(entry_key, os.path.normpath(make_these_folders))

                entry_key = word_between_slashes + "_Root_Folder"
                if not JsonEditor().getValue(entry_key):
                    parent_dir = os.path.dirname(make_these_folders)
                    JsonEditor().update_json_entry(entry_key, os.path.normpath(parent_dir))


    # ****************************************************************************************
    #
    # ****************************************************************************************
    def saveSettings(self, settings_folder):
        # Collect texts from all QLineEdit widgets
        settings = {
            "camera_1_image_folder": self.ui.lineEdit_camera_1.text(),
            "camera_2_image_folder": self.ui.lineEdit_camera_2.text(),
            "camera_3_image_folder": self.ui.lineEdit_camera_3.text(),
            "camera_4_image_folder": self.ui.lineEdit_camera_4.text(),
        }
        # Save to a JSON file
        with open(settings_folder, 'w') as file:
            json.dump(settings, file)


    def loadSettings(self, settings_folder):
        try:
            with open(settings_folder, 'r') as file:
                settings = json.load(file)

                # Populate the QLineEdit widgets with saved values
                self.ui.lineEdit_camera_1.setText(settings.get("camera_1_image_folder", ""))
                self.image_folder_1 = settings.get("camera_1_image_folder", "")

        except FileNotFoundError:
            pass  # It's okay if the file doesn't exist yet


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def check_url_validity(self, my_url):
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
                ssl._create_default_https_context = ssl._create_unverified_context
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
                    msgBox = GRIME_AI_QMessageBox('NEON SITE Info URL Error', strError)
                    response = msgBox.displayMsgBox()
                nErrorCode = -1
                nRetryCount = nRetryCount - 1

        return nErrorCode


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def fetchDownloadsFolderPath(self):
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
    def getRangeOfDates(self, strStartYearMonth, strEndYearMonth):
        # GET A LIST OF THE MONTHS FOR THE YEARS BETWEEN THE START DATA AND END DATE
        start_date = datetime.strptime(strStartYearMonth, "%Y-%m")
        end_date = datetime.strptime(strEndYearMonth, "%Y-%m")

        # Difference between each date. M means one month
        date_list = pd.date_range(start_date, end_date, freq='MS')

        # if you want dates in string format then convert it into string
        date_list = date_list.strftime("%Y-%m")

        return date_list.tolist()


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def separateChannels(self, image):
        # greennessIndex = green / (red + green + blue)
        red = image[:, :, 0]
        red = red.flatten()
        red = red.astype(float)

        green = image[:, :, 1]
        green = green.flatten()
        green = green.astype(float)

        blue = image[:, :, 2]
        blue = blue.flatten()
        blue = blue.astype(float)

        return red, green, blue


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def sumChannels(self, red, green, blue):

        redSum = np.sum(red)
        greenSum = np.sum(green)
        blueSum = np.sum(blue)

        return redSum, greenSum, blueSum


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def getMaxNumColorClusters(self, roiList):
        maxColorClusters = 0

        for roiObj in roiList:
            if roiObj.getNumColorClusters() > maxColorClusters:
                maxColorClusters = roiObj.getNumColorClusters()

        return maxColorClusters

