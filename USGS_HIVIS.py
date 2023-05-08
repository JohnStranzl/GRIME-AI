import time
import urllib
import re
import datetime

# USGS Cameras
camera_list = ['0204300267_VA_Beggars_Cr_nr_Dawley_Corners',
               '0204295505_VA_Pinewood_Virginia_Beach',
               '05536890_IL_Chicago_Canal_nr_Lemont',
               '05404147_Redstone_Outlet',
               '04085068_Ashwaubenon_LRapids',
               '04085078_Dutchman_Hansen',
               '05404140_Redstone_Clark',
               '05404150_Redstone_LaValleRd',
               '05427718_Yahara_Windsor',
               '04087120_Meno70th',
               '01387998_NJ_Ramapo_Pompton_Dam',
               '01437500_NY_Neversink_Goddefroy',
               '05366800_Chip_River_Eau_Claire',
               '05335450_Burnett_Malone_RD',
               '05335329_Burnett_Soderberg_RD',
               '08073000_TX_Addicks_Res_Outflow_Culverts',
               '01480870_PA_Brandywine_Downingtown',
               '04085108_east_river_east_river',
               '08051100_TX_Ray_Roberts_Lake',
               '451021089064901_DF_PM_AO1_Field',
               '0158175720_MD_Lake_Serene_Edgewood',
               '05434500_Martintown',
               ]

class USGS_HIVIS:
    def __init__(self):
        self.instance = 1
        endpoint = 'https://apps.usgs.gov/hivis'

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getImageList(camera_name, endpoint, command='/php/listFiles.php?path=') -> object:

        try:
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

            # Just keep JPGs (JPGs have '___' in their filename)
            subs = '___'
            list_of_images = list(filter(lambda x: subs in x, t))
        except:
            list_of_images = []

        return list_of_images

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getImageCount(camera_name, endpoint, command='/php/listFiles.php?path='):
        try:
            return USGS_HIVIS.getImageList(camera_name, endpoint, command)
        except:
            return []

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def fetchCameraList(self):
        return camera_list

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getEndpoint(self):
        return 'https://apps.usgs.gov/hivis'

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getCameraCount(self):
        return len(camera_list)

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getStartEndDate(camera_name, endpoint, command='/php/listFiles.php?path='):

        imageList = USGS_HIVIS.getImageList(camera_name, endpoint, command)

        try:
            strStartDate = USGS_HIVIS.extractDateFromFilename(imageList[0])
            strEndDate   = USGS_HIVIS.extractDateFromFilename(imageList[-1:])
        except:
            strStartDate = ""
            strEndDate   = ""

        return strStartDate, strEndDate

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def extractDateFromFilename(filename):
        fileDateTime = re.search('\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', filename)

        try:
            arrDate = fileDateTime.group().split('_')[0].split('-')
            nYear = int(arrDate[0])
            nMonth = int(arrDate[1])
            nDay = int(arrDate[2])

            arrTime = fileDateTime.group().split('_')[1].split('-')
            nHours = int(arrTime[0])
            nMinutes = int(arrTime[1])
            nSeconds = int(arrTime[2])
        except:
            nYear = 1970
            nMonth = 1
            nDay = 1

            nHours = 0
            nMinutes = 0
            nSeconds = 0

        fileDate = datetime.date(nYear, nMonth, nDay)
        fileTime = datetime.time(nHours, nMinutes, nSeconds)

        return fileDate, fileTime
