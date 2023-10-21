import os
import datetime
import cv2
from GRIME_QProgressWheel import QProgressWheel
from GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI_Color import GRIME_AI_Color
from GRIME_AI_Vegetation_Indices import GRIME_AI_Vegetation_Indices

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
    def ExtractFeatures(self, imagesList, imageFileFolder, roiList, dlg):

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

                strOutputString = strOutputString + ',' + GRIME_AI_Vegetation_Indices().computeGreennessValue(dlg, redSum, greenSum, blueSum)

                strOutputString = strOutputString + ',' + '%3.4f' % intensity
                strOutputString = strOutputString + ',' + '%3.4f' % entropyValue

                hsvClusterCenters, hist = roiObj.getHSVClusterCenters()
                for i in range(nMaxNumColorClusters):
                    strOutputString = strOutputString + ',' + str(hsvClusterCenters[i][0]) + ',' + str(hsvClusterCenters[i][1]) + ',' + str(hsvClusterCenters[i][2])
                strOutputString = strOutputString + '\n'

                csvFile.write(strOutputString)

            # ONCE THE NUMBER OF COLOR CLUSTERS HAS BEEN SELECTED AND AN ROI TRAINED, LOCK THE NUMBER OF COLOR CLUSTERS
            # AND TRAIN ALL SUBSEQUENT ROIs FOR THE SAME NUMBER OF COLOR CLUSTERS
            nClusters = dlg.spinBoxColorClusters.value()

            header = '\n' + self.buildScalarHeader(nClusters)

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
                    image_date, image_time = self.extractEXIFdata(fname.fullPathAndFilename)
                    strDate = image_date.isoformat().strip()
                    strTime = image_time.isoformat().strip()

                    strOutputString = '%s,%s,%s' % (strHyperlink, strDate, strTime)

                    # --------------------------------------------------------------------------------------------------
                    # CALCULATE THE FEATURE SCALARS FOR THE ENTIRE IMAGE AND SAVE THEM TO THE CSV FILE
                    # --------------------------------------------------------------------------------------------------
                    if dlg.checkBoxScalarRegion_WholeImage.isChecked():
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

                        strOutputString = strOutputString + ',' + GRIME_AI_Vegetation_Indices().computeGreennessValue(dlg, redSum, greenSum, blueSum)

                        if dlg.checkBox_Intensity.isChecked():
                            strOutputString = strOutputString + ', %3.4f' % intensity
                        if dlg.checkBox_ShannonEntropy.isChecked():
                            strOutputString = strOutputString + ', %3.4f' % entropyValue
                        if dlg.checkBox_Texture.isChecked():
                            strOutputString = strOutputString + ', %3.2f' % texture

                        if dlg.checkBoxColor_HSV.isChecked():
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

                    if dlg.checkBoxScalarRegion_ROI.isChecked():

                        strROI = self.calculateROIScalars(img)

                        strOutputString = strOutputString + strROI


                    # WRITE STRING TO CSV FILE
                    strOutputString = strOutputString  + '\n'
                    csvFile.write(strOutputString)

            csvFile.close()

            # CLOSE AND DELETE THE PROGRESSBAR
            progressBar.close()
            del progressBar
