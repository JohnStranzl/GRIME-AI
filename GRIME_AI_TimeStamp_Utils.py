import os
from datetime import datetime
import datetime as dt

import re

from exifData import EXIFData

FROM_FILENAME = 1
FROM_EXIF = 2

class dateTimeFormat:
    def __init__(self, date_format, time_format):
        self.date_format = date_format
        self.time_format = time_format

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_TimeStamp_Utils:

    def __init__(self):
        self.className = "GRIME_AI_TimeStamp_Utils"
        self.instance = 1

        self.datePattern = None
        self.timePattern = None
        self.dateTimeSource = None


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def detectDateTime(self, filename):

        # --------------------------------------------------------------------------------------------------------------
        # FIRST WE WILL CHECK FOR THE DATE/TIME STAMP IN THE FILENAME
        # --------------------------------------------------------------------------------------------------------------
        searchPatternList = []

        searchPatternList.append(dateTimeFormat(r'(\d{4}-\d{2}-\d{2}T)', r'(\d{2}-\d{2}-\d{2}Z)'))
        searchPatternList.append(dateTimeFormat(r'(\d{4}_\d{2}_\d{2})', r'(\d{6})'))

        bFound = False
        for dateTimeObj in searchPatternList:
            if not bFound:
                date_pattern = re.compile(dateTimeObj.date_format)
                time_pattern = re.compile(dateTimeObj.time_format)

                date = date_pattern.search(filename)
                time = time_pattern.search(filename)

                if date is not None and time is not None:
                    bFound = True
                    self.datePattern = date_pattern
                    self.timePattern = time_pattern
                    self.dateTimeSource = FROM_FILENAME

        # --------------------------------------------------------------------------------------------------------------
        # IF THE DATE/TIME STAMP IS NOT IN THE FILENAME, EXTRACT IT FROM THE EXIF DATA, IF IT HAS EXIF DATA, AND IF THE
        # EXIF DATA CONTAINS THE DATE/TIME
        # --------------------------------------------------------------------------------------------------------------
        if not bFound:
            self.dateTimeSource = FROM_EXIF

    # ======================================================================================================================
    # THIS IS AN ATTEMPT TO IMPLEMENT A METHOD THAT EXTRACTS THE DATE AND TIME STAMP THAT AN IMAGE HAS BEEN TAKEN
    # WITHOUT KNOWING WHETHER THE DATE/TIME STAMP IS IN THE FILENAME, OR EXIF DATA, OR BOTH, OF THE IMAGE FILE. THIS
    # IS TO GET AROUND THE FACT THAT NO ONE FOLLOWS A STANDARD DATE/TIME LABELING METHOD.
    # ======================================================================================================================
    def extractDateTime(self, filename):

        if self.dateTimeSource == FROM_FILENAME:
            if self.datePattern is not None and self.timePattern is not None:
                date = self.datePattern.search(filename)
                time = self.timePattern.search(filename)

                pattern = r'[0-9_-]+'
                image_date = re.findall(pattern, date[0])[0]
                image_time = re.findall(pattern, time[0])[0]

                image_time = self.convert_to_iso_time(image_time)
        elif self.dateTimeSource == FROM_EXIF:
            myEXIF_Data = EXIFData()
            image_date, image_time = myEXIF_Data.extractEXIFDataDateTime(filename)

        return image_date, image_time

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def convert_to_iso_time(self, time_string):
        try:
            # Parse the input time string
            if len(time_string) == 6:
                iso_time = datetime.strptime(time_string, '%H%M%S')
            elif len(time_string) == 8:
                if '_' in time_string:
                    iso_time = datetime.strptime(time_string, '%H_%M_%S')
                elif '-' in time_string:
                    iso_time = datetime.strptime(time_string, '%H-%M-%S')
                else:
                    iso_time = None
            else:
                iso_time = None

            iso_time = iso_time.strftime("%H:%M:%S")

            return iso_time

        except ValueError:
            return "Invalid time format. Please provide a 6-character string in the format 'HHMMSS'."

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def extractDateFromFilename(self, filename):

        nYear = 1970
        nMonth = 1
        nDay = 1
        arrDate = []
        arrTime = []

        # ----------------------------------------------------------------------------------------------------
        # TRY PBT DATE/TIME FORMAT: 'MicksSlide_20220610_Forsberg_814.jpg'
        # ----------------------------------------------------------------------------------------------------
        try:
            arrDate.append(re.search('\d{4}\d{2}\d{2}', filename)[0][0:4])
            arrDate.append(re.search('\d{4}\d{2}\d{2}', filename)[0][4:6])
            arrDate.append(re.search('\d{4}\d{2}\d{2}', filename)[0][6:8])
            arrTime.append('0')
            arrTime.append('0')
            arrTime.append('0')
            delimiter = ''
        except:
            arrDate = None
            arrTime = None

        # ----------------------------------------------------------------------------------------------------
        # TRY NEON DATE/TIME FORMAT: 'NEON.D03.BARC.DP1.20002_2022_06_01_073006.jpg'
        # ----------------------------------------------------------------------------------------------------
        if arrDate == None:
            arrDate = re.search('\d{4}_\d{2}_\d{2}', filename)
            arrTime = re.search('\d{2}\d{2}\d{2}', filename)
            delimiter = '_'

        # ----------------------------------------------------------------------------------------------------
        # TRY USGS DATE/TIME FORMAT:
        # FORMAT #1: 'VA_Pinewood_Virginia_Beach___2022-06-04_11-00-01-8586-05-00_overlay.jpg'
        # FORMAT #2: 'NE_Platte_River_near_Grand_Island___2023-04-0600-02-51Z.jpg'
        # ----------------------------------------------------------------------------------------------------
        try:
            if arrDate == None:
                arrDate = re.search('\d{4}-\d{2}-\d{2}', filename)
                arrTime = re.search('T\d{2}-\d{2}-\d{2}', re.search('T\d{2}-\d{2}-\d{2}', filename)[0])
                delimiter = '-'
        except:
            arrDate = None
            arrTime = None

        # PARSE DATE
        try:
            if len(delimiter) > 0:
                nYear  = int(arrDate.group().split(delimiter)[0])
                nMonth = int(arrDate.group().split(delimiter)[1])
                nDay   = int(arrDate.group().split(delimiter)[2])
            else:
                nYear  = int(arrDate[0])
                nMonth = int(arrDate[1])
                nDay   = int(arrDate[2])
        except:
            nYear  = 1970
            nMonth = 1
            nDay   = 1

        # PARSE TIME
        try:
            if len(delimiter) > 0:
                nHours   = int(arrTime.group()[1:11].split(delimiter)[0])
                nMinutes = int(arrTime.group()[1:11].split(delimiter)[1])
                nSeconds = int(arrTime.group()[1:11].split(delimiter)[2])
            else:
                nHours   = int(arrTime[0])
                nMinutes = int(arrTime[1])
                nSeconds = int(arrTime[2])
        except:
            nHours   = 0
            nMinutes = 0
            nSeconds = 0

        # PERFORM RANGE CHECK ON DATE
        if nYear > 2100 or nYear < 1970 or nMonth > 12 or nDay > 31:
            nYear  = 1970
            nMonth = 1
            nDay   = 1

        # CREATE DATE AND TIME OBJECTS
        fileDate = dt.date(nYear, nMonth, nDay)
        fileTime = dt.time(nHours, nMinutes, nSeconds)

        return fileDate, fileTime
