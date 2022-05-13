import time
import datetime
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver

import requests
from GRIMe_ImageData import imageData

class GRIMe_PhenoCam():

    def __init__(self):
        self.phenoCamURL = 'https://phenocam.sr.unh.edu'

    # ======================================================================================================================
    # THIS FUNCTION GETS THE LIST OF ALL AVAILABLE VISIBLE IMAGES FOR A GIVEN DATE.
    # ======================================================================================================================
    def getVisibleImages(self, dailyURLvisible, start_time, end_time):

        dailyImagesList = dailyList([], [])

        # ----------
        # DETERMINE NUMBER OF IMAGES AVAILABLE FOR THE SELECTED DATE AND BUILT A LIST OF FILENAMES
        r = requests.get(dailyURLvisible)

        if r.status_code != 404:
            List = []

            # ----------
            # GET THE FILENAMES OF EACH VISIBLE LIGHT IMAGE FOR THE SELECTED DATE
            if 1:
                driver = webdriver.Chrome('C:/Program Files (x86)/GRIME-AI/chromedriver/chromedriver.exe')  # Optional argument, if not specified will search path.
                driver.get(dailyURLvisible)

                username = driver.find_element_by_name('username')
                username.send_keys('JohnStranzl')
                password = driver.find_element_by_name('password')
                password.send_keys('V3ntur3s63*')
                password.submit()
                #time.sleep(5)  # Let the user actually see something!

                driver.get(dailyURLvisible)
                webContent = driver.page_source
                driver.quit()

                # PARSE THE HTML TO EXTRACT OUT THE JPG LINKS
                soup = BeautifulSoup(webContent, 'html5lib')
                links = soup.findAll("a", href=lambda href: href and "jpg" in href)
            else:
                # ----------
                # GET THE FILENAMES OF EACH VISIBLE LIGHT IMAGE FOR THE SELECTED DATE
                response = requests.get(dailyURLvisible)
                soup = BeautifulSoup(response.text, 'html.parser')
                links = soup.find_all('jpg')

            listlength = len(links)

            for link in links:
                fullPathAndFilename = link['href']
                filename = fullPathAndFilename.split('/')[-1]
                timestamp = filename.split('_')[-1]
                hour = int(timestamp[0:2])
                minute = int(timestamp[2:4])
                second = int(timestamp[4:6])
                photo_time = datetime.time(int(hour), int(minute), int(second))

                bDownload = False
                if (photo_time >= start_time) and (photo_time <= end_time):
                    bDownload = True
                elif (start_time == datetime.time(0, 0, 0)) and (end_time == datetime.time(0, 0, 0)):
                    bDownload = True

                if bDownload:
                    fullPathAndFilename = self.phenoCamURL + fullPathAndFilename
                    List.append(imageData(fullPathAndFilename, timestamp[0:2], timestamp[2:4], timestamp[4:6]))

            dailyImagesList.setVisibleList(List)
        else:
            print("404: Update Daily Images")

        return dailyImagesList

# ======================================================================================================================
# THIS CLASS WILL HOLD THE NAMES OF ALL THE VISIBLE AND INFRARED (IR) IMAGES FOR A SPECIFIED DATE RANGE
# ======================================================================================================================
class dailyList:
    def __init__(self, visibleList, IRList):
        self.visibleList = visibleList
        self.IRList = IRList

    def clear(self):
        self.visibleList = []
        self.IRList= []

    def setVisibleList(self, visibleList):
        self.visibleList.extend(visibleList)

    def setIRList(self, IRList):
        self.IRList = IRList

    def getVisibleList(self):
        return self.visibleList

    def getIRList(self):
        return self.IRList

