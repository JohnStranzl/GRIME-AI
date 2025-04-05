import os
import json
import urllib
import requests
from urllib.request import urlopen
import ssl

import pandas as pd

from datetime import timedelta

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox

from GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI_QProgressWheel import QProgressWheel

endpoint = "https://jj5utwupk5.execute-api.us-east-1.amazonaws.com"
imageEnpoint = "https://usgs-nims-images.s3.amazonaws.com/overlay"

class USGS_NIMS:
    def __init__(self):
        self.instance = 1

        self.nwisId = None
        self.camName = None
        self.camId = None

        self.cameraDictionary = self.initCameraDictionary()

        self.siteCount = 0;

        __dfs = []

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def initCameraDictionary(self):
        self.cameraDictionary = []

        try:
            # QUERY CAMERA LIST
            uri = "https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/cameras?enabled=true"

            #ssl._create_default_https_context = ssl._create_unverified_context
            response = urllib.request.urlopen(uri)
            data = response.read()  # a `bytes` object
            cameraData = json.loads(data.decode('utf-8'))

            self.cameraDictionary = {}
            sites_with_hideCam = []

            for element in cameraData:
                if element.get('locus') == 'aws':
                    if element.get('hideCam', True):
                        cam_id = element.get('camId')
                        if cam_id:
                            sites_with_hideCam.append(cam_id)
                        else:
                            print(f"Site with hideCam=True has no camId provided.")
                    else:
                        self.cameraDictionary[element['camId']] = element

            # Sort the list of sites with hideCam=True
            sites_with_hideCam.sort()

            # Print each site on a separate line
            print("Site with hideCam=True:")
            for site in sites_with_hideCam:
                print(site)
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"Error fetching or parsing data: {e}")
            self.cameraDictionary = {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.cameraDictionary = {}

        return self.cameraDictionary

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def getCameraDictionary(self):
        return(self.cameraDictionary)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def getCameraList(self):
        myList = []

        for element in self.cameraDictionary.values():
            myList.append(element['camId'])

        #myList = sorted(myList, key=lambda x: x[0].split('_')[0])
        myList = sorted(myList)

        self.siteCount = len(myList)

        return(myList)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def getCameraInfo(self, strCameraID):
        myList = []

        try:
            for camId, camData in self.cameraDictionary.items():
                # Check if the current camera matches strCameraID
                if camData.get('camId') == strCameraID:
                    for key, keyData in camData.items():
                        if keyData is not None and not isinstance(keyData, (list, dict, int)):
                            # Assign values based on specific keys
                            if key == 'nwisId':
                                self.nwisId = keyData
                            elif key == 'camName':
                                self.camName = keyData
                            elif key == 'camId':
                                self.camId = keyData

                            # Format keyData and append to myList
                            formattedData = f"{key}: {keyData}"
                            myList.append(formattedData)

        except Exception:
            myList.append("No information available for this site.")

        if len(myList) == 0:
            myList.append("No information available for this site.")

        return(myList)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def get_nwisID(self):
        return self.nwisId

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def get_camName(self):
        return self.camName

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def get_camId(self):
        return self.camId

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def getLatestImage(self, siteName):
        nErrorCode = -1
        nRetryCount = 3
        nWebImageCount = 0

        latestImageURL = imageEnpoint + '/' + siteName + '/' + siteName + '_newest.jpg'

        while nErrorCode == -1 and nRetryCount > 0:
            r = requests.get(latestImageURL, stream=True)

            if r.status_code != 404:
                nWebImageCount = 1
                #ssl._create_default_https_context = ssl._create_unverified_context
                data = urlopen(latestImageURL).read()
                latestImage = QPixmap()
                latestImage.loadFromData(data)
                nErrorCode = 0
            else:
                nWebImageCount = 0
                latestImage = []
                nErrorCode = -1
                nRetryCount = nRetryCount - 1
                if nRetryCount == 0:
                    print("404: NIMS - Download Latest Image Fail")

        if nErrorCode == -1:
            nErrorCode = 404

        return nErrorCode, latestImage


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def getImageCount(self, siteName, nwisID, startDate, endDate, startTime, endTime):

        listOfImages = ""
        numberOfDays = (endDate - startDate).days + 1

        if numberOfDays > 1:
            progressBar = QProgressWheel()
            progressBar.setRange(0, numberOfDays + 1)
            progressBar.setWindowTitle("Calculating Images in Date Range")
            progressBar.show()

        # FETCH LIST OF IMAGES
        for i in range(numberOfDays):
            if numberOfDays > 1:
                progressBar.setValue(i)

            after, before = self.buildImageDateTimeFilter(i, startDate, endDate, startTime, endTime)

            listOfImages_text = self.fetchListOfImages(siteName, after, before)

            if listOfImages_text == '[]':
                listOfImages_text = ''

            if len(listOfImages) == 0:
                listOfImages = listOfImages_text
            else:
                listOfImages += "," + listOfImages_text

        # SPLIT LIST INTO AN ARRAY OF INDIVIDUAL IMAGE NAMES
        listOfImages = listOfImages.split(',')

        # CLOSE AND DELETE THE PROGRESSBAR
        if numberOfDays > 1:
            progressBar.close()
            del progressBar

        if len(listOfImages) == 0:
            strMessage = 'No images available for the site or for the time/date range specified.'
            msgBox = GRIME_AI_QMessageBox('Images unavailable', strMessage, QMessageBox.Close)
            response = msgBox.displayMsgBox()

        return len(listOfImages)


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def downloadImages(self, siteName, nwisID, startDate, endDate, startTime, endTime, saveFolder):

        listOfImages = ""
        numberOfDays = (endDate - startDate).days + 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # FETCH IMAGES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i in range(numberOfDays):
            after, before = self.buildImageDateTimeFilter(i, startDate, endDate, startTime, endTime)

            listOfImages_text = self.fetchListOfImages(siteName, after, before)

            if len(listOfImages) == 0:
                listOfImages = listOfImages_text
            else:
                listOfImages += "," + listOfImages_text

        # SPLIT LIST INTO AN ARRAY OF INDIVIDUAL IMAGE NAMES
        listOfImages = listOfImages.split(',')

        if any(listOfImages):
            progressBar = QProgressWheel()
            progressBar.setRange(0, len(listOfImages) + 1)
            progressBar.show()

            # DOWNLOAD AND SAVE IMAGES
            missingImageCount = 0
            for imageIndex, image in enumerate(listOfImages):
                progressBar.setWindowTitle(image)
                progressBar.setValue(imageIndex)
                #progressBar.repaint()

                if image != '[]':
                    try:
                        fullURL = imageEnpoint + '/' + siteName + '/' + image
                        fullFilename = os.path.join(saveFolder, image)
                        if os.path.isfile(fullFilename) == False:
                            urllib.request.urlretrieve(fullURL, fullFilename)
                    except Exception:
                        if missingImageCount == 0:
                            strMessage = 'One or more images reported as available by NIMS are not available.'
                            msgBox = GRIME_AI_QMessageBox('Images unavailable', strMessage, QMessageBox.Close)
                            response = msgBox.displayMsgBox()
                        missingImageCount += 1

            # CLOSE AND DELETE THE PROGRESSBAR
            progressBar.close()
            del progressBar

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # FETCH STAGE AND DISCHARGE
            # https://waterservices.usgs.gov/test-tools/
            # https://help.waterdata.usgs.gov/codes-and-parameters/parameters
            #
            # OLD NWIS SITE
            # https://waterservices.usgs.gov/rest/IV-Test-Tool.html
            #
            # NEW https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=06800500&startDT=2023-12-01&endDT=2024-02-01&siteStatus=all&siteType=ST&outputDataTypeCd=iv,dv,gw,qw,id.
            # OLD https://waterservices.usgs.gov/nwis/iv/?format=rdb,1.0&sites=
            #
            # If start and end time
            # https://waterservices.usgs.gov/nwis/iv/?format=rdb,1.0&sites=06800500&startDT=2024-02-01&endDT=2024-02-01&siteStatus=all
            # https://waterservices.usgs.gov/nwis/iv/?format=rdb,1.0&sites=06800500&startDT=2024-02-01&endDT=2024-02-01&siteStatus=all
            #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # FETCH STAGE AND DISCHARGE
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #     water_services_endpoint = "https://waterservices.usgs.gov/nwis/iv/?format=json&sites="
            #     fullURL = water_services_endpoint + nwisID + startDT + endDT + '&parameterCd=00060,00065&siteStatus=all'


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def fetchStageAndDischarge(self, nwisID, siteName, startDate, endDate, startTime, endTime, saveFolder):
        water_services_endpoint = "https://waterservices.usgs.gov/nwis/iv/?format=rdb,1.0&sites="
        #fullURL = water_services_endpoint + nwisID + startDate + endDate + '&parameterCd=00060,00065&siteStatus=all'
        fullURL = water_services_endpoint + nwisID + '&startDT=' + startDate.strftime("%Y-%m-%d") + '&endDT=' + endDate.strftime("%Y-%m-%d") + '&siteStatus=all'

        timeStamp = startDate.strftime("%Y-%m-%d") + "T" + startTime.strftime("%H%M") + " - " + endDate.strftime("%Y-%m-%d") + "T" + endTime.strftime("%H%M")
        fullFilename_txt = os.path.join(saveFolder, siteName + " - " + nwisID + " - " + timeStamp + ".txt")

        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            with urllib.request.urlopen(fullURL) as response:
                response.read()

            # RETRIEVE DISCHARGE REPORT
            urllib.request.urlretrieve(fullURL, fullFilename_txt)

            fullFilename_csv = os.path.join(saveFolder, siteName + " - " + nwisID + " - " + timeStamp + ".csv")
            self.reformat_file(fullFilename_txt, fullFilename_csv)
        except Exception:
            strMessage = 'Unable to retrieve data from the USGS site.'
            msgBox = GRIME_AI_QMessageBox('USGS - Retrieval Error', strMessage, QMessageBox.Close)
            msgBox.displayMsgBox()


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def buildImageDateTimeFilter(self, index, startDate, endDate, startTime, endTime):
        startDay = startDate + timedelta(days=index)

        # IF START AND END TIME ARE 0, GET ALL IMAGES FOR THE ENTIRE DAY
        if startTime.hour == 0 and startTime.minute == 0 and endTime.hour == 0 and endTime.minute == 0:
            after  = "&after="  + startDay.strftime("%Y-%m-%d") + ":00:00:00"
            before = "&before=" + startDay.strftime("%Y-%m-%d") + ":23:59:59"
        # OTHERWISE, ONLY GET IMAGES BETWEEN THE SPECIFIED START AND END TIMES FOR EACH DAY
        else:
            after  = "&after="  + startDay.strftime("%Y-%m-%d") + ":" + startTime.strftime("%H:%M:%S")
            before = "&before=" + startDay.strftime("%Y-%m-%d") + ":" + endTime.strftime("%H:%M:%S")

        return after, before


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def fetchListOfImages(self, siteName, after, before):
        imagesToGet = endpoint + "/prod/cameras?enabled=true/listFiles?camId=" + siteName + after + before
        listOfImages_response = requests.get(imagesToGet)
        listOfImages_text = listOfImages_response.text

        if listOfImages_text != '[]':
            # FORMAT LIST
            listOfImages_text = listOfImages_text.replace("[","")
            listOfImages_text = listOfImages_text.replace("]","")
            listOfImages_text = listOfImages_text.replace('"','')

        return listOfImages_text


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def reformat_file(self, input_folder_path, output_file):
        """
        :param input_folder_path:
        :param output_file:
        :return:
        """
        # Initialize an empty list to store DataFrames
        dfs = []

        #import file and remove commented-out rows at the top of the original file
        df_temp = pd.read_csv(input_folder_path, delimiter='\t', comment='#')
        dfs.append(df_temp)

        # Concatenate all DataFrames into a single DataFrame
        USGS_stage_df = pd.concat(dfs, ignore_index=True)
        USGS_stage_df = USGS_stage_df[~USGS_stage_df['agency_cd'].astype(str).str.contains("5s")]

        USGS_stage_df.to_csv(output_file, index=False)



# THIS "JUNK" IS FOR REFERENCE IN DIAGNOSING ISSUES WITH THE USGS API
    # ------------------------------------------------------------------------------------------------------------------
    # ASCII Format
    #//waterservices.usgs.gov/nwis/iv/?format=rdb,1.0
    # &sites=01646500
    # &startDT=2023-07-27T10:00-0400
    # &endDT=2023-07-30T14:00%2b0000
    # &parameterCd=00060,00065
    # &siteStatus = all
    #
    # JSON Format
    #//waterservices.usgs.gov/nwis/iv/?format=json
    # &sites=01646500
    # &startDT = 2023-08-15T10:00-0400
    # &endDT=2023-08-17T16:00-0400
    # &parameterCd=00060,00065
    # &siteStatus=all
    # ------------------------------------------------------------------------------------------------------------------
    #def fetchStageAndFlowData(self):

# API COMMAND FORMAT - https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles?camId=NE_Platte_River_near_Grand_Island&after=2023-07-16%2015:00:00.000   before = DATESTRING & after = DATESTRING
# endpoint = "https://jj5utwupk5.execute-api.us-east-1.amazonaws.com"
# imageEnpoint = "https://usgs-nims-images.s3.amazonaws.com/overlay"
# latest image = https://usgs-nims-images.s3.amazonaws.com/overlay/WI_Green_Bay_Oil_Depot/WI_Green_Bay_Oil_Depot_newest.jpg

#https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles?camId=NE_Platte_River_near_Grand_Island&after=2023-07-16%2015:00:00.000
#before=DATESTRING&after=DATESTRING

#https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles?camId=NE_Platte_River_near_Grand_Island&after=2023-06-01

#https://usgs-nims-images.s3.amazonaws.com/overlay/NE_Platte_River_near_Grand_Island/NE_Platte_River_near_Grand_Island___2023-05-16T17-00-54Z.jpg

#imagesToGet = "https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles?camId=NE_Platte_River_near_Grand_Island&after=2023-07-16%2015:00:00.000"

#https://waterservices.usgs.gov/rest/IV-Test-Tool.html

#//waterservices.usgs.gov/nwis/iv/?format=json&sites=01646500&parameterCd=00060,00065&siteStatus=all
#//waterservices.usgs.gov/nwis/iv/?format=json&sites=06178500&startDT=2023-07-17&endDT=2023-07-19&parameterCd=00060,00065&siteStatus=all

#//waterservices.usgs.gov/nwis/iv/?format=rdb,1.0&sites=01646500&startDT=2023-07-20T04:00-0400&endDT=2023-07-20T09:00-0400&parameterCd=00060,00065&siteStatus=all
