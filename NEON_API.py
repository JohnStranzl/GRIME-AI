import os
import csv
import multiprocessing
import requests
from PyQt5.QtGui import QPixmap
import shutil
from urllib.request import urlopen
from bs4 import BeautifulSoup
from GRIME_AI_Utils import GRIME_AI_Utils

# THIRD PARTY MODULES
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr

# GRIMe-AI MODULES
from GRIME_QProgressWheel import QProgressWheel

from nitrateData import nitrateData

SERVER = 'http://data.neonscience.org/api/v0/'

# https://www.neonscience.org/sites/default/files/NEON_Field_Site_Metadata_20230309.csv

class  NEON_API:
    def __init__(self, parent=None):
        self.instance = 1
        self.className = "NEON API"
        self.dest = ""


    # ======================================================================================================================
    # The purpose of this function is to query information above a specific product. The information
    # contains the product description, sites for which the product is available among other information.
    # ======================================================================================================================
    def QueryProductInfo(self, productCode):
        product_request = requests.get(SERVER + 'products/' + productCode)
        product_json = product_request.json()

        return product_json


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def FetchSiteInfoFromNEON(self, server, siteCode):

        # Make request, using the sites endpoint
        site_request = requests.get(server + 'sites/' + siteCode)

        # Convert to Python JSON object
        site_json = site_request.json()

        return (site_json)


    # ======================================================================================================================
    # THIS FUNCTION FETCHES THE FIELD SITE TABLE FROM THE NEON SITE AND PARSES ITS INFORMATION.
    # ======================================================================================================================
    def FetchFieldSiteTableURL(self, my_url):
        csv_links = []

        # r = requests.get(my_url)
        #ssl._create_default_https_context = ssl._create_unverified_context
        r = urlopen(my_url)
        # context = ssl._create_unverified_context()
        # r = urlopen(my_url, context=context)

        if 1:
            # if r.status_code == 200:
            # create beautiful-soup object
            # soup = BeautifulSoup(r.content, 'html5lib')
            soup = BeautifulSoup(r, 'html5lib')

            # FIND ALL CSV LINKS ON THE WEB-PAGE. CURRENTLY THERE IS ONLY ONE. HOWEVER, THERE COULD BE MULTIPLES IN THE FUTURE
            links = soup.findAll("a", href=lambda href: href and "csv" in href)

            # CREATE COMPLETE URL FOR LINK TO CSV FILE. ASSUME THERE IS ONLY ONE FOR NOW BUT LOOP FOR FUTURE USE-CASES
            for link in links:
                #csvLink = root_url + link['href']
                csvLink = link['href']

            csv_links = csvLink

        return csv_links


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def DownloadFieldSiteTableFiles(self, csv_links):
        link = csv_links

        # obtain filename by splitting url and getting last string
        file_name = link.split('/')[-1]

        # create response object
        r = requests.get(link, stream=True)

        configFilePath = os.path.expanduser('~')
        configFilePath = os.path.join(configFilePath, 'Documents')
        configFilePath = os.path.join(configFilePath, 'GRIMe-AI')
        if not os.path.exists(configFilePath):
            os.mkdir(configFilePath)
        filename_with_path = os.path.join(configFilePath, file_name)

        # download started
        with open(filename_with_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        return filename_with_path

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def FetchData(self, SiteCode, strProduct, strStartDate, strEndDate, downloadsFilePath):

        nError = 0

        if os.path.exists(downloadsFilePath) == False:
            os.makedirs(downloadsFilePath)

        foldername = strProduct.split('.')[1].zfill(5) + ' -' + strProduct.split('.')[2].split(':')[1]
        strProduct = 'DP1.' + strProduct.split('.')[1] + '.001'

        progressBar = QProgressWheel()
        progressBar.setRange(0, 100)
        progressBar.show()

        neonUtilities = rpackages.importr('neonUtilities')
        utils = importr('utils')

        progressBar.setValue(20)

        try:
            utils.capture_output(neonUtilities.zipsByProduct(dpID       = strProduct,
                                        site       = SiteCode,
                                        savepath   = downloadsFilePath,
                                        startdate  = strStartDate,
                                        enddate    = strEndDate,
                                        package    = 'basic',
                                        check_size = 'FALSE'))

            progressBar.setValue(40)

            # PATH WHERE THE zipsByProduct PLACED THE DOWNLOADED ZIP FILES
            myFolderPath = downloadsFilePath + '\\filesToStack' + strProduct.split('.')[1].zfill(5)

            # PATH WHERE WE WANT TO PLACE THE STACKED FILES (i.e., CONCATENATED MONTHLY DATA FILES) WILL BE STORED
            mySavePath = downloadsFilePath + '\\' + foldername
            if os.path.exists(mySavePath):
                shutil.rmtree(mySavePath)

            # USE AS MANY CORES ARE AVAILABLE FOR STACKING THE DATA (i.e., UNZIP ALL THE INDIVIDUAL MONTHS DOWNLOADED ZIP
            # FILES AND CONCATENATE INTO ONE CSV FILE)
            nMyCores = multiprocessing.cpu_count()

            progressBar.setValue(60)

            neonUtilities.stackByTable(filepath=myFolderPath, savepath=mySavePath, nCores=nMyCores)

            # IF ALL ZIPPED FILES WERE STACKED PROPERLY, AND THE FUNCTION stackByTable REMOVES THE ZIP FILES ONCE ALL
            # FILES ARE CONCATENATED, WE CAN DELETE WHAT NOW SHOULD BE THE EMPTY ZIP FILE DOWNLOAD FOLDER.
            if os.path.exists(myFolderPath):
                shutil.rmtree(myFolderPath)

            # LET'S REMOVE ONE LEVEL OF INDIRECTION AND MOVE THE FILES STACKED BY stackByTable TO THE ROOT PRODUCT FOLDER IN THE
            # DOWNLOAD DIRECTORY
            src = downloadsFilePath + '\\' + foldername + '\\stackedFiles'
            self.dest = downloadsFilePath + '\\' + foldername
            if os.path.exists(src) and os.path.exists(self.dest):
                filenames = os.listdir(src)
                for filename in filenames:
                    shutil.move(os.path.join(src, filename), self.dest)

                # NOW WE REMOVE THE EMPTY STACK FOLDER SINCE THE FILES HAVE BEEN MOVED
                shutil.rmtree(src)

            progressBar.setValue(100)

        except:
            nError = -1

        progressBar.close()
        del progressBar

        return nError


    # ======================================================================================================================
    # THIS FUNCTION DOWNLOADS THE LATEST IMAGE FOR THE SITE SELECTED BY THE END-USER AND DISPLAYS IT IN THE GUI SO THE
    # END-USER CAN CAN SEE WHAT THE PARTICULAR SITE LOOKS LIKE.
    # ======================================================================================================================
    def DownloadLatestImage(self, siteCode, domainCode):
        nErrorCode = -1
        nRetryCount = 3
        nWebImageCount = 0

        latestImageURL = 'https://phenocam.nau.edu/data/latest/NEON.D10.ARIK.DP1.20002.jpg'
        tmp = latestImageURL.replace('ARIK', siteCode)
        latestImageURL = tmp.replace('D10', domainCode)

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
                    print("404: PhenoCam - Download Latest Image Fail")


        return r.status_code, latestImage, nWebImageCount


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getAvailableMonths(self, SITE, PRODUCTCODE):
        availableMonths = []

        url = SERVER + 'sites/' + SITE

        site_json = requests.get(url).json()

        # Get available months of Ecosystem structure data products for TEAK site
        # Loop through the 'dataProducts' list items (each one is a dictionary) at the site
        for product in site_json['data']['dataProducts']:
            # if a list item's 'dataProductCode' dict element equals the product code string
            if (product['dataProductCode'] == PRODUCTCODE):
                availableMonths = product['availableMonths']

        return availableMonths


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getProductAbstract(self, SITE, PRODUCTCODE):
        productAbstract = []

        url = SERVER + 'products/' + PRODUCTCODE

        product_json = requests.get(url).json()

        try:
            productAbstract = product_json['data']['productAbstract']
        except:
            pass

        return productAbstract


    # ======================================================================================================================
    # THIS FUNCTION WILL FETCH AND READ THE NEON FIELD SITE TABLE THAT CONTAINS INFORMATION ABOUT ALL THE
    # NEON SITES INCLUDING ALL THE IMAGES THAT ARE AVAILABLE FOR A SITE ON THE PHENOCAM WEBSITE.
    # ======================================================================================================================
    def readFieldSiteTable(self):
        siteList = []
        url = 'https://www.neonscience.org/field-sites/explore-field-sites'

        nErrorCode = GRIME_AI_Utils().check_url_validity(url)

        # IF AT LEAST ONE FIELD SITE TABLE IS FOUND ON THE NEON SITE...
        if nErrorCode == 0:
            csv_links = self.FetchFieldSiteTableURL(url)

            # download all CSV files
            filename_with_path = NEON_API().DownloadFieldSiteTableFiles(csv_links)

            siteList = GRIME_AI_Utils().parseCSV(filename_with_path)
        # ELSE IF NO FIELD SITE TABLES ARE FOUND, RETURN AN EMPTY LIST
        elif nErrorCode == -1:
            siteList = []

        return siteList


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def parseNitrateCSV(self):
        # FULLY QUALIFIED PATH OF THE CSV DOWNLOADED ONTO THE LOCAL COMPUTER
        filename = self.dest + '\\NSW_15_minute.csv'

        nitrateList = []
        fields = []
        rows = []

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

        for row in rows:
            # VALIDATE DATA: SOME OF THE FIELDS IN EACH DATA RECORD ARE NULL SO WE HAVE TO HANDLE THESE CASES. 'NOT SURE
            # WHY RECORDS WITH MISSING DATA ARE NOT EXCLUDED FROM THE DATA FILES PROVIDED BY NEON
            for i in range(6, 13):
                if len(row[i]) == 0:
                    row[i] = '0.0'

            nitrateList.append(nitrateData(row[0],
                                           row[1],
                                           row[2],
                                           row[3],
                                           row[4],
                                           row[5],
                                           float(row[6]),
                                           float(row[7]),
                                           float(row[8]),
                                           float(row[9]),
                                           float(row[10]),
                                           float(row[11]),
                                           float(row[12]),
                                           float(row[13]),
                                           row[14],
                                           row[15]))

        return nitrateList
