import os
from datetime import datetime

import csv
import cv2
import numpy as np

import pandas as pd

import urllib.request
from urllib.request import urlopen
import ssl

from siteData import siteData

from pathlib import Path

from GRIME_QMessageBox import GRIMe_QMessageBox

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_Utils:

    def __init__(self):
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

        for root, dirs, files in os.walk(folder):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in extensions:
                    imageCount += 1

        return imageCount


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getFileList(self, folder, extensions, bFetchRecursive):

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


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def createGRIMeFolders(self, full):
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

            #JES self.EditVideoOutputFolder.setText(videoFolder)

        # --------------------------------------------------
        # CREATE SETTINGS FOLDERS IN USER'S DOCUMENTS FOLDER
        configFilePath = os.path.join(rootFolder, 'Settings')
        if not os.path.exists(configFilePath):
            os.mkdir(configFilePath)

        # CHECK TO SEE IF THE CONFIGURATION FILE EXISTS. IF IT DOES NOT, THEN CREATE IT USING touch
        configFile = os.path.join(configFilePath, 'GRIMe-AI.cfg')
        if not os.path.isfile(configFile):
            configFileWithPath = Path(configFile)
            configFileWithPath.touch(exist_ok=True)

        # --------------------------------------------------
        # CREATE DOWNLOAD FOLDER IN USER'S DOCUMENTS FOLDER
        downloadsFolder = os.path.join(rootFolder, 'Downloads')
        if not os.path.exists(downloadsFolder):
            os.mkdir(downloadsFolder)

        #self.EditSaveImagesOutputFolder.setText(downloadsFolder)

        # --------------------------------------------------
        # CREATE FOLDER IN USER'S DOCUMENTS FOLDER FOR DOWNLOADED IMAGES
        downloadsFolder = os.path.join(downloadsFolder, 'Images')
        if not os.path.exists(downloadsFolder):
            os.mkdir(downloadsFolder)

        # --------------------------------------------------
        # CREATE FOLDER IN USER'S DOCUMENTS FOLDER FOR DOWNLOADED IMAGES FROM THE USGS
        downloadsFolder = os.path.join(downloadsFolder, 'USGS')
        if not os.path.exists(downloadsFolder):
            os.mkdir(downloadsFolder)

        # --------------------------------------------------
        # CREATE DOWNLOAD FOLDER IN USER'S DOCUMENTS FOLDER
        EXIFFolder = os.path.join(rootFolder, 'EXIF')
        if not os.path.exists(EXIFFolder):
            os.mkdir(EXIFFolder)

        #JES if full == 1:
        #JES    self.EditEXIFOutputFolder.setText(EXIFFolder)


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
                    msgBox = GRIMe_QMessageBox('NEON SITE Info URL Error', strError)
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
        start_date = datetime.datetime.strptime(strStartYearMonth, "%Y-%m")
        end_date = datetime.datetime.strptime(strEndYearMonth, "%Y-%m")

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

