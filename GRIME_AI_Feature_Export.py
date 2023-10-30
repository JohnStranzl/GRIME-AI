import os
import re
import datetime
import cv2
from GRIME_QProgressWheel import QProgressWheel
from GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI_Color import GRIME_AI_Color
from GRIME_AI_Vegetation_Indices import GRIME_AI_Vegetation_Indices
from GRIME_roiData import GRIME_roiData

from exifData import EXIFData

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_Feature_Export:
    def __init__(self):
        csvFilename = ''
        self.instance = 1

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def createTrainingDataFilename(self, hardDriveImageFolder):
        csvFilename = 'TrainingData_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
        imageQualityFile = os.path.join(hardDriveImageFolder, csvFilename)
        csvFilename = open(imageQualityFile, 'a', newline='')

        return csvFilename

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def ExtractFeatures(self, imagesList, imageFileFolder, roiList, colorSegmentationParams):

        bCreateEXIFFile = False
        bCreateVideos = False

        myGRIMe_Color = GRIME_AI_Color()

        hardDriveImageFolder = imageFileFolder + '/'

        # GENERATE LIST OF IMAGE FILES IN FOLDER
        videoFileList = imagesList
        nFrameCount = len(videoFileList)

        if nFrameCount > 0:

            # ----------------------------------------------------------------------------------------------------------
            # CREATE AND OPEN NEW TRAINING DATA CSV FILE AND LABEL COLUMNS
            # ----------------------------------------------------------------------------------------------------------
            csvFile = self.createTrainingDataFilename(hardDriveImageFolder);

            # ----------------------------------------------------------------------------------------------------------
            # OUTPUT THE TRAINING FILE a) NAMES, b) TRAINED COLOR CLUSTERS, c) IMAGE INTENSITY, d) ETC.
            # ----------------------------------------------------------------------------------------------------------
            nMaxNumColorClusters = GRIME_AI_Utils().getMaxNumColorClusters(roiList)

            for roiObj in roiList:
                strOutputString = 'Training Filename'

                strOutputString = strOutputString + ',Date (ISO),Time (ISO),Greenness,Intensity,Entropy'

                template = ', ' + (roiObj.getROIName() + ': ') + '#'

                if nMaxNumColorClusters == 1:
                    strOutputString = (strOutputString + template).replace('#', 'H')
                    strOutputString = (strOutputString + template).replace('#', 'S')
                    strOutputString = (strOutputString + template).replace('#', 'V')
                else:
                    for idx in range(nMaxNumColorClusters):
                        strOutputString = (strOutputString + template).replace('#', 'H') + ('_' + idx.__str__())
                        strOutputString = (strOutputString + template).replace('#', 'S') + ('_' + idx.__str__())
                        strOutputString = (strOutputString + template).replace('#', 'V') + ('_' + idx.__str__())
                strOutputString = strOutputString + "\n"

                csvFile.write(strOutputString)

                # LOAD THE IMAGE FILE
                img = myGRIMe_Color.loadColorImage(roiObj.getTrainingImageName())

                # BLUR THE IMAGE
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                height, width = gray.shape
                gray = cv2.resize(gray, (width, height))

                # IMAGE INTENSITY CALCULATIONS
                intensity = cv2.mean(gray)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway

                # COMPUTE ENTROPY FOR ENTIRE IMAGE
                entropyValue = 0.0
                #JES - THIS IS CRASHING. THE SUM FUNCTION WITHIN calcEntropy IS TRYING TO SUM A STRUCTURE.
                #JES entropyValue = self.calcEntropy(gray)[0]

                texture = -999

                red, green, blue = GRIME_AI_Utils().separateChannels(img)
                redSum, greenSum, blueSum = GRIME_AI_Utils().sumChannels(red, green, blue)

                # CREATE HYPERLINK TO FILE
                strOutputString = '=HYPERLINK(' + '"' + roiObj.getTrainingImageName() + '"' + ')'
                strOutputString = strOutputString + ','     # DATE
                strOutputString = strOutputString + ','     # TIME

                strOutputString = strOutputString + ',' + GRIME_AI_Vegetation_Indices().computeGreennessValue(colorSegmentationParams, redSum, greenSum, blueSum)

                strOutputString = strOutputString + ',' + '%3.4f' % intensity
                strOutputString = strOutputString + ',' + '%3.4f' % entropyValue

                hsvClusterCenters, hist = roiObj.getHSVClusterCenters()
                for i in range(nMaxNumColorClusters):
                    strOutputString = strOutputString + ',' + str(hsvClusterCenters[i][0]) + ',' + str(hsvClusterCenters[i][1]) + ',' + str(hsvClusterCenters[i][2])
                strOutputString = strOutputString + '\n'

                csvFile.write(strOutputString)

            # ONCE THE NUMBER OF COLOR CLUSTERS HAS BEEN SELECTED AND AN ROI TRAINED, LOCK THE NUMBER OF COLOR CLUSTERS
            # AND TRAIN ALL SUBSEQUENT ROIs FOR THE SAME NUMBER OF COLOR CLUSTERS
            nClusters = colorSegmentationParams.numColorClusters

            header = '\n' + self.buildScalarHeader(nClusters, roiList, colorSegmentationParams)

            # WRITE THE HEADER TO THE CSV
            csvFile.write(header)

            # ----------------------------------------------------------------------------------------------------------
            # CREATE PROGRESS WHEEL
            # ----------------------------------------------------------------------------------------------------------
            progressBar = QProgressWheel(0, (len(roiList) * len(videoFileList)) + 1)
            progressBar.show()
            progressBarIndex = 1

            # ----------------------------------------------------------------------------------------------------------
            # PROCESS IMAGES
            # ----------------------------------------------------------------------------------------------------------
            #E:\\000 - University of Nebraska\\2022_USGS104b\\imagery\\PBT_MittelstetsMeanderCam\\20220709
            for fname in videoFileList:

                if os.path.isfile(fname.fullPathAndFilename):

                    # LOAD THE IMAGE FILE
                    img = myGRIMe_Color.loadColorImage(fname.fullPathAndFilename)

                    # CREATE HYPERLINK TO FILE
                    strHyperlink = '=HYPERLINK(' + '"' + fname.fullPathAndFilename + '"' + ')'

                    # EXTRACT DATE/TIME STAMP FROM IMAGE EXIF DATA
                    image_date, image_time = EXIFData().extractEXIFdata(fname.fullPathAndFilename)
                    strDate = image_date.isoformat().strip()
                    strTime = image_time.isoformat().strip()

                    strOutputString = '%s,%s,%s' % (strHyperlink, strDate, strTime)

                    # --------------------------------------------------------------------------------------------------
                    # CALCULATE THE FEATURE SCALARS FOR THE ENTIRE IMAGE AND SAVE THEM TO THE CSV FILE
                    # --------------------------------------------------------------------------------------------------
                    if colorSegmentationParams.wholeImage:
                        # BLUR THE IMAGE
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # IMAGE INTENSITY CALCULATIONS
                        intensity = cv2.mean(gray)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway

                        # COMPUTE ENTROPY FOR ENTIRE IMAGE
                        entropyValue = self.calcEntropy(gray)[0]

                        # EXTRACT 'n' DOMINANT HSV COLORS
                        hist, clusterCenters = myGRIMe_Color.extractDominant_HSV(img, nClusters)

                        texture = -999

                        red, green, blue = GRIME_AI_Utils().separateChannels(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        redSum, greenSum, blueSum = GRIME_AI_Utils().sumChannels(red, green, blue)

                        strOutputString = strOutputString + ',' + GRIME_AI_Vegetation_Indices().computeGreennessValue(colorSegmentationParams, redSum, greenSum, blueSum)

                        if colorSegmentationParams.Intensity:
                            strOutputString = strOutputString + ', %3.4f' % intensity
                        if colorSegmentationParams.ShannonEntropy:
                            strOutputString = strOutputString + ', %3.4f' % entropyValue
                        if colorSegmentationParams.Texture:
                            strOutputString = strOutputString + ', %3.2f' % texture

                        if colorSegmentationParams.HSV:
                            for idx in range(nClusters):
                                # CONVERT FROM OpenCV's HSV HUE DATA FORMAT 0 to 180 DEGREES TO THE HSV STANDARD FORMAT OF 0 to 360 DEGREES
                                h = float(clusterCenters[idx][0])
                                s = float(clusterCenters[idx][1])
                                v = float(clusterCenters[idx][2])
                                strOutputString = strOutputString + ', %3.2f, %3.2f, %3.2f' % (h, s, v)

                    # --------------------------------------------------------------------------------------------------
                    #
                    # --------------------------------------------------------------------------------------------------
                    progressBar.setValue(progressBarIndex)
                    progressBarIndex = progressBarIndex + 4

                    if colorSegmentationParams.ROI:

                        strROI = self.calculateROIScalars(img, roiList, colorSegmentationParams)

                        strOutputString = strOutputString + strROI


                    # WRITE STRING TO CSV FILE
                    strOutputString = strOutputString  + '\n'
                    csvFile.write(strOutputString)

            csvFile.close()

            # CLOSE AND DELETE THE PROGRESSBAR
            progressBar.close()
            del progressBar

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def buildScalarHeader(self, nClusters, roiList, colorSegmentationParams):
        # CREATE HEADER FOR THE ATTIRBUTE OF THE ENTIRE IMAGE
        header = 'Image, Date (ISO), Time (ISO)'

        if colorSegmentationParams.wholeImage:
            header = self.buildImageScalarHeader(header)

        if colorSegmentationParams.ROI:
            header = self.buildROI_ScalarHeader(header, roiList, colorSegmentationParams)

        return header

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def buildROI_ScalarHeader(self, header, roiList, colorSegmentationParams):

        newHeader = header

        # ADD HEADER TO CSV FILE FOR EACH ROI ASSUMING ROIs HAVE BEEN CREATED
        for roiObj in roiList:
            template = ', ' + (roiObj.getROIName() + ': ') + '#'

            if colorSegmentationParams.GCC:
                newHeader = (newHeader + template).replace('#', 'GCC')

            if colorSegmentationParams.GLI:
                newHeader = (newHeader + template).replace('#', 'GLI')

            if colorSegmentationParams.NDVI:
                newHeader = (newHeader + template).replace('#', 'NDVI')

            if colorSegmentationParams.ExG:
                newHeader = (newHeader + template).replace('#', 'ExG')

            if colorSegmentationParams.RGI:
                newHeader = (newHeader + template).replace('#', 'RGI')

            if colorSegmentationParams.Intensity:
                newHeader = (newHeader + template).replace('#', 'Intensity')

            if colorSegmentationParams.ShannonEntropy:
                newHeader = (newHeader + template).replace('#', 'Entropy')

            if colorSegmentationParams.Texture:
                newHeader = (newHeader + template).replace('#', 'Texture')

            # IF THERE IS MORE THAN ONE (1) ROI, APPEND AN INDEX ONTO THE HEADER LABEL

            nClusters = roiObj.getNumColorClusters()

            if colorSegmentationParams.HSV:
                template = ', ' + (roiObj.getROIName() + ': ') + '#'

                if nClusters == 1:
                    newHeader = (newHeader + template).replace('#', 'H')
                    newHeader = (newHeader + template).replace('#', 'S')
                    newHeader = (newHeader + template).replace('#', 'V')
                else:
                    for idx in range(nClusters):
                        newHeader = (newHeader + template).replace('#', 'H') + ('_' + idx.__str__())
                        newHeader = (newHeader + template).replace('#', 'S') + ('_' + idx.__str__())
                        newHeader = (newHeader + template).replace('#', 'V') + ('_' + idx.__str__())

        newHeader = newHeader + '\n'

        return newHeader

    # ==================================================================================================================
    # CALCULATE THE FEATURE SCALARS FOR THE VARIOUS ROIs AND SAVE THEM TO THE CSV FILE
    # ==================================================================================================================
    def calculateROIScalars(self, img, roiList, colorSegmentationParams):

        strOutputString = ''

        for roiObj in roiList:
            texture = -999

            nClusters = roiObj.getNumColorClusters()

            rgb = GRIME_roiData().extractROI(roiObj.getImageROI(), img)

            # CONVERT THE IMAGE FROM BGR TO RGB AND HSV
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            # IMAGE INTENSITY CALCULATIONS
            intensity = cv2.mean(gray)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway

            # COMPUTE ENTROPY FOR ROI
            entropyValue = GRIME_roiData().calcEntropy(gray)

            # EXTRACT 'n' DOMINANT HSV COLORS
            myGRIMe_Color = GRIME_AI_Color()
            qhsvImg, hsvClusterCenters, hist = myGRIMe_Color.KMeans(hsv, nClusters)

            # KMeans QUANTIZES THE HUE VALUE TO 0..180 WHEN THE ACTUAL HSV COLOR SPACE HUE VALUE 0..360.
            # THEREFORE WE MULTIPLY THE KMeans HUE VALUE BY 2 TO STANDARDIZE ON THE ACTUAL COLOR SPACE HUE RANGE.
            hsvClusterCenters[:, 0] = hsvClusterCenters[:, 0] * 2.0

            red, green, blue = GRIME_AI_Utils().separateChannels(rgb)
            redSum, greenSum, blueSum = GRIME_AI_Utils().sumChannels(red, green, blue)

            strOutputString = strOutputString + ',' + GRIME_AI_Vegetation_Indices().computeGreennessValue(colorSegmentationParams, redSum, greenSum, blueSum)

            if colorSegmentationParams.Intensity:
                strOutputString = strOutputString + ', %3.4f' % intensity
            if colorSegmentationParams.ShannonEntropy:
                strOutputString = strOutputString + ', %3.4f' % entropyValue
            if colorSegmentationParams.Texture:
                strOutputString = strOutputString + ', %3.2f' % texture
            if colorSegmentationParams.HSV:
                for idx in range(nClusters):
                    # CONVERT FROM OpenCV's HSV HUE DATA FORMAT 0 to 180 DEGREES TO THE HSV STANDARD FORMAT OF 0 to 360 DEGREES
                    h = float(hsvClusterCenters[idx][0])
                    s = float(hsvClusterCenters[idx][1])
                    v = float(hsvClusterCenters[idx][2])
                    strOutputString = strOutputString + ', %3.2f, %3.2f, %3.2f' % (h, s, v)

        return strOutputString

