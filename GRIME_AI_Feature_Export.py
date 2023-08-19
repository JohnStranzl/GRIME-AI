import os
import datetime

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_Feature_Export:
    def __init__(self):
        csvFilename = ''

    def createTrainingDataFilename(hardDriveImageFolder):
        csvFilename = 'TrainingData_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
        imageQualityFile = os.path.join(hardDriveImageFolder, csvFilename)
        csvFilename = open(imageQualityFile, 'a', newline='')

        return csvFilename

