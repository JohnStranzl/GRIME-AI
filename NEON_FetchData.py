# ======================================================================================================================
# IMPORT MODULES
# ======================================================================================================================
# PYTHON MODULES
import multiprocessing
import os
import shutil

# THIRD PARTY MODULES
import rpy2.robjects.packages as rpackages

# GRIMe-AI MODULES
from GRIME_QProgressWheel import QProgressWheel


# ======================================================================================================================
#
# ======================================================================================================================
class NEON_FetchData:
    def __init__(self):
        self.instance = 1

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def NEON_FetchData(self, SiteCode, strProduct, strStartDate, strEndDate, downloadsFilePath):
        foldername = strProduct.split('.')[1].zfill(5) + ' -' + strProduct.split('.')[2].split(':')[1]
        strProduct = 'DP1.' + strProduct.split('.')[1] + '.001'

        progressBar = QProgressWheel()
        progressBar.setRange(0, 100)
        progressBar.show()

        neonUtilities = rpackages.importr('neonUtilities')

        progressBar.setValue(20)

        try:
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

            progressBar.setValue(100)

        except:
            nError = -1

        progressBar.close()
        del progressBar
