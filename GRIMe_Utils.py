import os

import re

import csv
import cv2
import numpy as np

from siteData import siteData


# ======================================================================================================================
#
# ======================================================================================================================
class GRIMe_Utils:
    def __init__(self):
        self.className = "GRIMe_Utils"

    # ======================================================================================================================
    # Converts a QImage into an opencv MAT format
    # ======================================================================================================================
    def convertQImageToMat(incomingImage):
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
    def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
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
    def parseCSV(filename_with_path):
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
    def extractDateFromFilename(filename):

        nYear = 1970
        nMonth = 1
        nDay = 1
        strTime = ''

        # first, try PBT file naming convention:
        fileDate = re.search('(\d{4})\d{2}\d{2}', filename)
        if fileDate != None:
            fileDate = fileDate.group(0)

            nYear = int(fileDate[0:4])
            nMonth = int(fileDate[4:6])
            nDay = int(fileDate[6:8])

        if nYear > 2100 or nYear < 1970 or nMonth > 12 or nDay > 31:
            nYear = 1970
            nMonth = 1
            nDay = 1

        # if the date could not be found in the PBT filename, it might be a NEON filename.
        if fileDate == None:
            fileDateTime = re.search('\d{4}_\d{2}_\d{2}_\d{6}', filename)
            try:
                fileDateTime = str(fileDateTime[0]).replace('_', '')

                fileDate = fileDateTime[0:8]
                nYear = int(fileDateTime[0:4])
                nMonth = int(fileDateTime[4:6])
                nDay = int(fileDateTime[6:8])

                strTime = fileDateTime[8:14]
            except:
                fileDate = "19700101"
                nYear = 1970
                nMonth = 1
                nDay = 1
                strTime = "000000"

        return fileDate, nYear, nMonth, nDay, strTime


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getImageCount(folder, extensions):
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
    def drawGridOnImage(img):
        GRID_SIZE = 100

        height, width, channels = img.shape
        for x in range(0, width - 1, GRID_SIZE):
            cv2.line(img, (x, 0), (x, height), (255, 0, 0), 1, 1)

        for y in range(0, width - 1, GRID_SIZE):
            cv2.line(img, (0, y), (height, y), (255, 0, 0), 1, 1)

        # cv2.imshow('Hehe', numpyImage)
        # key = cv2.waitKey(0)

