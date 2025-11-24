#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os

#os.environ['R_HOME'] = 'C:/Program Files/R/R-4.4.1'

from io import StringIO

import csv
import re
import multiprocessing
import requests
from PyQt5.QtGui import QPixmap
import shutil
from urllib.request import urlopen
import ssl

import time

from bs4 import BeautifulSoup

# THIRD PARTY MODULES

import rpy2.robjects as robjects
#import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr

# GRIME-AI MODULES
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI.GRIME_AI_Save_Utils import JsonEditor

from GRIME_AI.nitrateData import nitrateData

from GRIME_AI.siteData import siteData


SERVER = 'http://data.neonscience.org/api/v0/'

# https://www.neonscience.org/sites/default/files/NEON_Field_Site_Metadata_20240423.csv

# ======================================================================================================================
# ======================================================================================================================
# =====    =====     =====     =====     =====       HELPER FUNCTIONS      =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================

def parse_NEON_field_site_metadata(filename_with_path):
    """
    Module: parse_NEON_field_site_metadata
    Author: John Edward Stranzl, Jr.
    Created: 2021-04-xx
    License: Apache License 2.0

        This file is licensed under the Apache License, Version 2.0.
        You may not use this file except in compliance with the License.
        You may obtain a copy of the License at:
            http://www.apache.org/licenses/LICENSE-2.0

    Description:
        Utilities for parsing phenocam site CSV files. The core function, parse_NEON_field_site_metadata,
        reads a CSV listing sites and extracts the following fields:
          - site_id
          - site_name
          - phenocams
          - latitude
          - longitude
        It automatically handles headers that may be prefixed with 'field_'.

    Dependencies:
        - csv (stdlib)
        - is_valid_csv (custom validator for CSV integrity)
        - find_field_index (helper to locate columns with or without 'field_' prefix)
        - siteData (data class for site attributes)

    Functions:
        parse_NEON_field_site_metadata(self, filename_with_path) -> List[siteData]
            Parse the CSV at the given path and return a list of siteData objects.
            Raises ValueError if the CSV fails validation. Returns an empty list
            on any parsing error.

    Usage Example:
        from phenocam_csv_parser import PhenocamParser, siteData

        parser = PhenocamParser()
        try:
            sites = parser.parse_NEON_field_site_metadata("/data/site_list.csv")
        except ValueError as e:
            print(f"Validation error: {e}")
    """
    # Validate before parsing
    if not is_valid_csv(filename_with_path):
        raise ValueError(f"Invalid CSV content detected in '{filename_with_path}'.")

    rows = []
    siteList = []

    try:
        # Read all rows into memory (you could stream this if files get huge)
        with open(filename_with_path, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            # Extract header
            fields = next(csvreader)
            # Buffer the rest of the rows
            for row in csvreader:
                rows.append(row)

        # Locate the needed columns, with or without 'field_' prefix
        site_id_idx = find_field_index(fields, "site_id")
        site_name_idx = find_field_index(fields, "site_name")
        phenocam_idx = find_field_index(fields, "phenocams")
        lat_idx = find_field_index(fields, "latitude")
        lon_idx = find_field_index(fields, "longitude")

        # Build your siteData objects
        for row in rows:
            siteList.append(
                siteData(
                    row[site_id_idx],
                    row[site_name_idx],
                    row[phenocam_idx],
                    row[lat_idx],
                    row[lon_idx]
                )
            )

    except Exception:
        # Swallow any parsing errors and return empty list
        siteList = []

    return siteList


def is_valid_csv(source, sniff_lines=5, max_rows=10, encoding="utf-8-sig"):
    """
    Function: is_valid_csv
    Author: John Edward Stranzl, Jr.
    Created: 2025-09-04
    License: Apache License 2.0

        This file is licensed under the Apache License, Version 2.0.
        You may not use this file except in compliance with the License.
        You may obtain a copy of the License at:
            http://www.apache.org/licenses/LICENSE-2.0

    Description:
        Determine whether the provided source (a filesystem path or raw CSV text)
        constitutes a well-formed CSV. Checks include:
          - Quick HTML rejection
          - Sniffing a sample of lines for dialect
          - Consistent column counts across up to max_rows
          - At least a header plus one data row

    Dependencies:
        - os (stdlib)
        - csv (stdlib)
        - io.StringIO (stdlib)

    Parameters:
        source       Path to a CSV file or a raw CSV string.
        sniff_lines  Number of initial lines to sample for dialect sniffing (default: 5).
        max_rows     Maximum number of rows to read for validation (default: 10).
        encoding     Encoding used when reading a file source (default: "utf-8-sig").

    Returns:
        bool         True if source passes all CSV validity checks; False otherwise.
    """

    # Load the file’s contents if 'source' is a valid path
    if isinstance(source, str) and os.path.exists(source):
        with open(source, 'r', encoding=encoding) as f:
            text = f.read()
    else:
        text = source

    # Strip BOM if present
    if text.startswith('\ufeff'):
        text = text.lstrip('\ufeff')

    # Quick HTML rejection
    head = text[:2048].lower()
    for sig in ['<!doctype html', '<html', '<head', '<body', '<script']:
        if sig in head:
            return False

    # Build a sample of complete lines for sniffing
    lines = text.splitlines(keepends=True)
    sample = ''.join(lines[:sniff_lines])
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        # Fallback to basic CSV with commas
        dialect = csv.excel
        dialect.delimiter = ','
        dialect.quotechar = '"'
        dialect.doublequote = True

    # Read up to max_rows and collect them
    reader = csv.reader(StringIO(text), dialect)
    rows = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        rows.append(row)

    # Need at least header + one data row
    if len(rows) < 2:
        return False

    expected_cols = len(rows[0])
    if expected_cols < 1:
        return False

    # Ensure every sampled row matches the header’s column count
    return all(len(r) == expected_cols for r in rows)


def find_field_index(fields, base_name):
    """
    Return the index of either 'field_<base_name>' or '<base_name>' in the header list.
    Raises ValueError if neither is present.
    """
    for prefix in ("field_", ""):
        key = prefix + base_name
        if key in fields:
            return fields.index(key)
    raise ValueError(f"Could not find column '{base_name}' (with or without 'field_' prefix) in CSV header.")


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

        try:
            # Make request, using the sites endpoint
            site_request = requests.get(server + 'sites/' + siteCode)

            # Convert to Python JSON object
            site_json = site_request.json()
        except Exception:
            site_json = []

        return (site_json)


    # ======================================================================================================================
    # THIS FUNCTION FETCHES THE FIELD SITE TABLE FROM THE NEON SITE AND PARSES ITS INFORMATION.
    # ======================================================================================================================
    def FetchFieldSiteTableURL(self, my_url):
        csv_links = []

        # r = requests.get(my_url)
        #ssl._create_default_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = ssl._create_unverified_context
        r = urlopen(my_url)
        # context = ssl._create_unverified_context()
        # r = urlopen(my_url, context=context)

        if 1:
            # if r.status_code == 200:
            # create beautiful-soup object
            # soup = BeautifulSoup(r.content, 'html5lib')
            soup = BeautifulSoup(r, 'html5lib')

            # FIND ALL CSV LINKS ON THE WEB-PAGE. CURRENTLY THERE IS ONLY ONE. HOWEVER, THERE COULD BE MULTIPLES IN THE FUTURE
            links = soup.findAll("a", href=lambda href: href and "Metadata" in href)

            # CREATE COMPLETE URL FOR LINK TO CSV FILE. ASSUME THERE IS ONLY ONE FOR NOW BUT LOOP FOR FUTURE USE-CASES
            for link in links:
                #csvLink = root_url + link['href']
                csvLink = link['href']

            csv_links = csvLink

        return csv_links


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def Download_Field_Site_Metadata(self, csv_links):
        link = 'https://www.neonscience.org' + csv_links + '.csv'

        # obtain filename by splitting url and getting last string
        file_name = link.split('/')[-1]

        # create response object
        r = requests.get(link, stream=True)

        configFilePath = os.path.expanduser('~')
        configFilePath = os.path.join(configFilePath, 'Documents', 'GRIME-AI', 'Downloads', 'NEON', 'Metadata')
        if not os.path.exists(configFilePath):
            os.makedirs(configFilePath)
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

        self.dest = downloadsFilePath + '\\' + foldername
        if not os.path.exists(self.dest):
            os.makedirs(self.dest)

        #JES OVERRIDE DOWNLOAD PATH DUE TO WINDOWS FILE PATH LENGTH LIMITATION.

        downloadsFilePath = JsonEditor().getValue("Scratchpad_folder")


        strProduct = 'DP1.' + strProduct.split('.')[1] + '.001'

        progressBar = QProgressWheel()
        progressBar.setRange(0, 100)
        progressBar.show()

        progressBar.setValue(20)

        neonUtilities = importr('neonUtilities')
        utils = importr('utils')
        base = importr('base')


        try:
            downloadsFilePath = re.sub(r'\\', '/', downloadsFilePath)
            base.options(timeout=300)
            '''
            neon_zips = neonUtilities.zipsByProduct(dpID = strProduct, \
                                                             site = base.c(SiteCode), \
                                                             savepath = base.c(downloadsFilePath), \
                                                             startdate = base.c(strStartDate), \
                                                             enddate = base.c(strEndDate), \
                                                             package = base.c('basic'), \
                                                             include_provisional = robjects.vectors.BoolVector([True]), \
                                                             load = robjects.vectors.BoolVector([True]), \
                                                             check_size = robjects.vectors.BoolVector([False]));
            '''

            neon_zips = neonUtilities.zipsByProduct(dpID = strProduct,
                                                    site = base.c(SiteCode),
                                                    savepath = downloadsFilePath,
                                                    startdate = strStartDate,
                                                    enddate = strEndDate,
                                                    package = 'basic',
                                                    include_provisional = robjects.vectors.BoolVector([True]),
                                                    check_size = 'FALSE');
                                                    #load = 'TRUE',
                                                    #check_size = 'FALSE');
            '''
            neon_zips = neonUtilities.zipsByProduct(dpID='DP1.10003.001',
                                        site=base.c('HARV', 'BART'),
                                        savepath='C:/Users/johns/Documents/GRIME-AI/Downloads/NEON/Data/filesToStack10003',
                                        package='basic',
                                        check_size='FALSE',
                                        include_provisional = robjects.vectors.BoolVector([True]));
            '''

            utils.capture_output(neon_zips)

            progressBar.setValue(40)

            # PATH WHERE THE zipsByProduct PLACED THE DOWNLOADED ZIP FILES
            myFolderPath = os.path.join(downloadsFilePath, ('filesToStack' + strProduct.split('.')[1].zfill(5)))
            myFolderPath = os.path.normpath(myFolderPath)

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
            #JES self.dest = downloadsFilePath + '\\' + foldername
            if os.path.exists(src) and os.path.exists(self.dest):
                filenames = os.listdir(src)
                for filename in filenames:
                    shutil.move(os.path.join(src, filename), self.dest)

                # NOW WE REMOVE THE EMPTY TEMPORARY DOWNLOAD FOLDER SINCE THE FILES HAVE BEEN MOVED
                shutil.rmtree(os.path.join(downloadsFilePath, foldername))

            progressBar.setValue(100)

        except Exception:
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

        start_time = time.time()

        while nErrorCode == -1 and nRetryCount > 0:
            r = requests.get(latestImageURL, stream=True)

            if r.status_code != 404:
                nWebImageCount = 1
                ssl._create_default_https_context = ssl._create_unverified_context
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

        end_time = time.time()
        print ("Download Latest Image Elapsed Time: ", end_time - start_time)

        return r.status_code, latestImage, nWebImageCount


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def getAvailableMonths(self, SITE, PRODUCTCODE):
        availableMonths = []

        url = SERVER + 'sites/' + SITE

        nRetry = 3
        bSuccess = False
        while (nRetry > 0) and (bSuccess == False):
            try:
                site_json = requests.get(url).json()

                # Get available months of Ecosystem structure data products for TEAK site
                # Loop through the 'dataProducts' list items (each one is a dictionary) at the site
                for product in site_json['data']['dataProducts']:
                    # if a list item's 'dataProductCode' dict element equals the product code string
                    if (product['dataProductCode'] == PRODUCTCODE):
                        availableMonths = product['availableMonths']
                bSuccess = True
            except Exception:
                print("Retry NEON Site Access...")
                nRetry = nRetry - 1

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
        except Exception:
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
            filename_with_path = NEON_API().Download_Field_Site_Metadata(csv_links)

            try:
                siteList = parse_NEON_field_site_metadata(filename_with_path)
            except Exception as e:
                nErrorCode = -2
        # ELSE IF NO FIELD SITE TABLES ARE FOUND, RETURN AN EMPTY LIST
        elif nErrorCode == -1:
            siteList = []

        return nErrorCode, siteList


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

