#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import json
import urllib
import requests
from urllib.request import urlopen

import pandas as pd

from datetime import datetime, timedelta, time

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox

from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel


import re
import datetime

endpoint = "https://jj5utwupk5.execute-api.us-east-1.amazonaws.com"
imageEnpoint = "https://usgs-nims-images.s3.amazonaws.com/overlay"

class USGS_NIMS:
    def __init__(self):
        self.instance = 1

        overlayDir   = ""
        thumbDir     = ""
        tlDir        = ""
        smallDir     = ""
        locus        = ""
        FRP          = ""
        nwisId       = ""
        camId        = ""
        camName      = ""
        camDesc      = ""
        stateAbrv    = ""
        lat          = ""
        long         = ""
        createdDate  = ""
        modifiedDate = ""
        tz           = ""

        self.camera_dictionary = self.init_camera_dictionary()

        self.siteCount = 0;

        __dfs = []

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def init_camera_dictionary(self):
        self.camera_dictionary = {}

        try:
            # QUERY CAMERA LIST
            uri = "https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/cameras?enabled=true"

            response = urllib.request.urlopen(uri)
            data = response.read()  # a `bytes` object
            cameraData = json.loads(data.decode('utf-8'))

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
                        self.camera_dictionary[element['camId']] = element

            # Sort the list of sites with hideCam=True
            sites_with_hideCam.sort()

            # Print each site on a separate line
            print("Site with hideCam=True:")
            for site in sites_with_hideCam:
                print(site)
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"Error fetching or parsing data: {e}")
            self.camera_dictionary = {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.camera_dictionary = {}

        return self.camera_dictionary

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def get_camera_dictionary(self):
        return(self.camera_dictionary)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def get_camera_list(self):
        myList = []

        for element in self.camera_dictionary.values():
            myList.append(element['camId'])

        myList = sorted(myList)

        self.siteCount = len(myList)

        return(myList)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def get_camera_info(self, camera_id: str) -> list[str]:
        """
        Retrieve formatted camera information lines for the given camera_id.
        Updates instance attributes nwisId, camName, and camId if present.
        Returns a list of strings or a single-item list indicating no data.
        """
        # Direct dictionary lookup; assume cameraDictionary maps IDs to info dicts
        camera = self.camera_dictionary.get(camera_id)
        if camera is None:
            return ["No information available for this site."]

        info_lines: list[str] = []
        for key, value in camera.items():
            # Skip None, lists, dicts, and ints
            if value is None or isinstance(value, (list, dict, int)):
                continue

            # Update instance attributes when relevant
            if key in ("nwisId", "camName", "camId"):
                setattr(self, key, value)

            # Format and collect the info line
            info_lines.append(f"{key}: {value}")

        # Fallback if no valid info was collected
        return info_lines or ["No information available for this site."]

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
    def get_latest_image(self, siteName):
        nErrorCode = -1
        nRetryCount = 3
        nWebImageCount = 0

        latestImageURL = imageEnpoint + '/' + siteName + '/' + siteName + '_newest.jpg'

        while nErrorCode == -1 and nRetryCount > 0:
            r = requests.get(latestImageURL, stream=True)

            if r.status_code != 404:
                nWebImageCount = 1
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
    def get_image_count(self, siteName, nwisID, startDate, endDate, startTime, endTime):

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
    def download_images(self, siteName, nwisID, startDate, endDate, startTime, endTime, saveFolder):

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
    import datetime

    def buildImageDateTimeFilter(self, index, startDate, endDate, startTime, endTime):
        # startDate is expected to be of type datetime.date,
        # startTime and endTime are expected to be of type datetime.time

        # Compute the current day.
        startDay = startDate + datetime.timedelta(days=index)

        # Check if both start and end times are 00:00, meaning whole day request.
        if startTime.hour == 0 and startTime.minute == 0 and endTime.hour == 0 and endTime.minute == 0:
            day_start = datetime.datetime.combine(startDay, datetime.time(0, 0, 0))
            day_end = datetime.datetime.combine(startDay, datetime.time(23, 59, 59))
        else:
            day_start = datetime.datetime.combine(startDay, startTime)
            day_end = datetime.datetime.combine(startDay, endTime)

        # Adjust the boundaries: subtract 30 seconds from the "after" datetime,
        # add 30 seconds to the "before" datetime.
        after_dt = day_start - datetime.timedelta(seconds=30)
        before_dt = day_end + datetime.timedelta(seconds=30)

        # Format the adjusted datetimes as strings in the "YYYY-MM-DD:HH:MM:SS" format.
        after = "&after=" + after_dt.strftime("%Y-%m-%d:%H:%M:%S")
        before = "&before=" + before_dt.strftime("%Y-%m-%d:%H:%M:%S")

        return after, before


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def fetchListOfImages(self, siteName, after, before):
        imagesToGet = endpoint + "/prod/listFiles?camId=" + siteName + after + before
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

#https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles?camId=NE_Platte_River_near_Grand_Island&after=2023-07-16%2015:00:00.000
#before=DATESTRING&after=DATESTRING

#https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles?camId=NE_Platte_River_near_Grand_Island&after=2023-06-01

#https://usgs-nims-images.s3.amazonaws.com/overlay/NE_Platte_River_near_Grand_Island/NE_Platte_River_near_Grand_Island___2023-05-16T17-00-54Z.jpg

#imagesToGet = "https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles?camId=NE_Platte_River_near_Grand_Island&after=2023-07-16%2015:00:00.000"

#https://waterservices.usgs.gov/rest/IV-Test-Tool.html

#//waterservices.usgs.gov/nwis/iv/?format=json&sites=01646500&parameterCd=00060,00065&siteStatus=all
#//waterservices.usgs.gov/nwis/iv/?format=json&sites=06178500&startDT=2023-07-17&endDT=2023-07-19&parameterCd=00060,00065&siteStatus=all

#//waterservices.usgs.gov/nwis/iv/?format=rdb,1.0&sites=01646500&startDT=2023-07-20T04:00-0400&endDT=2023-07-20T09:00-0400&parameterCd=00060,00065&siteStatus=all
