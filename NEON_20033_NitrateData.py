
# ======================================================================================================================
# IMPORT MODULES
# ======================================================================================================================
# PYTHON MODULES
import csv
import multiprocessing
import os
import shutil

# THIRD PARTY MODULES
import rpy2.robjects.packages as rpackages

from GRIME_QProgressWheel import QProgressWheel
# GRIMe-AI MODULES
from nitrateData import nitrateData


# ======================================================================================================================
#
# ======================================================================================================================
def parseNitrateCSV(filename_with_path):
    # FULLY QUALIFIED PATH OF THE CSV DOWNLOADED ONTO THE LOCAL COMPUTER
    filename = filename_with_path

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


class NEON_20033_NitrateData:
    def __init__(self):
        self.instance = 1

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def DP1_20033_FetchNitrateData(self, SiteCode, strProduct, strStartDate, strEndDate, downloadsFilePath):
        progressBar = QProgressWheel()
        progressBar.show()

        neonUtilities = rpackages.importr('neonUtilities')

        progressBar.setValue(20)

        foldername = strProduct.split('.')[1].zfill(5) + ' -' + strProduct.split('.')[2].split(':')[1]

        strProduct = 'DP1.' + strProduct.split('.')[1].zfill(5) + '.001'
        neonUtilities.zipsByProduct(dpID=strProduct,
                                    site=SiteCode,
                                    savepath=downloadsFilePath,
                                    startdate=strStartDate,
                                    enddate=strEndDate,
                                    package='basic',
                                    check_size='FALSE')

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
        dest = downloadsFilePath + '\\' + foldername
        if os.path.exists(src) and os.path.exists(dest):
            filenames = os.listdir(src)
            for filename in filenames:
                shutil.move(os.path.join(src, filename), dest)

            # NOW WE REMOVE THE EMPTY STACK FOLDER SINCE THE FILES HAVE BEEN MOVED
            shutil.rmtree(src)

        progressBar.setValue(80)

        csvFile = dest + '\\NSW_15_minute.csv'
        nitrateList = parseNitrateCSV(csvFile)

        progressBar.setValue(100)

        progressBar.close()
        del progressBar

        return nitrateList

    # ======================================================================================================================
    # THIS FUNCTION PARSES THE FIELD SITE TABLE THAT IS FETCHED FROM THE NEON SITE.
    # ======================================================================================================================


