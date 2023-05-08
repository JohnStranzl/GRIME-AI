from enum import Enum

from PyQt5.QtCore import QRect, QSize


# ======================================================================================================================
#
# ======================================================================================================================
class ROIShape(Enum):
    RECTANGLE = 0
    ELLIPSE = 1

# ======================================================================================================================
#
# ======================================================================================================================
class GRIMe_roiData:
    def __init__(self):
        self.trainingImageName = ""
        self.roiName = ""
        self.displayROI = QRect()
        self.imageROI = QRect()
        self.imageSize = QSize()
        self.displaySize = QSize()
        self.clusterCenters = []
        self.hist = []
        self.hsvClusterCenters = []
        self.hsvHist = []
        self.intensity = []
        self.shannonEntropy = []
        self.nNumColorClusters = []

    # --------------------------------------------------------------------------------
    def setTrainingImageName(self, trainingImageName):
        self.trainingImageName = trainingImageName
    def getTrainingImageName(self):
        return(self.trainingImageName)

    # --------------------------------------------------------------------------------
    def setNumColorClusters(self, nNumColorClusters):
        self.nNumColorClusters = nNumColorClusters
    def getNumColorClusters(self):
        return self.nNumColorClusters

    # --------------------------------------------------------------------------------
    def setROIName(self, roiName):
        self.roiName = roiName
    def getROIName(self):
        return self.roiName

    # --------------------------------------------------------------------------------
    def setDisplayROI(self, rect):
        self.displayROI = rect
    def getDisplayROI(self):
        return self.displayROI

    # --------------------------------------------------------------------------------
    def setImageROI(self, imageROI):
        self.imageROI = imageROI
    def getImageROI(self):
        temp = self.imageROI
        #temp.setY(temp.y() - 25)
        return temp
        #return self.imageROI

    # --------------------  ------------------------------------------------------------
    def setImageSize(self, imageSize):
        self.imageSize = imageSize

    # --------------------------------------------------------------------------------
    def setDisplaySize(self, displaySize):
        self.displaySize = displaySize

    # --------------------------------------------------------------------------------
    def calcROI(self):
        # GET ROI, LABEL and ORIGINAL IMAGE SIZE
        widthMultiplier = self.imageSize.width() / self.displaySize.width()
        heightMultiplier = self.imageSize.height() / self.displaySize.height()

        # CALCULATE THE ROI OF THE ACTUAL IMAGE SIZE
        x = int(self.displayROI.x() * heightMultiplier)
        y = int(self.displayROI.y() * widthMultiplier)
        roiWidth = int(self.displayROI.width() * widthMultiplier)
        roiHeight = int(self.displayROI.height() * heightMultiplier)

        rect = QRect(x, y, roiWidth, roiHeight)
        self.setImageROI(rect)

    # --------------------------------------------------------------------------------
    def setROIShape(self, shape):
        self.shape = shape
    def getROIShape(self):
        return self.shape

    # --------------------------------------------------------------------------------
    def setClusterCenters(self, clusterCenters, hist):
        self.clusterCenters = clusterCenters
        self.hist = hist
    def getClusterCenters(self):
        return self.clusterCenters, self.hist

    def setHSVClusterCenters(self, clusterCenters, hist):
        self.hsvClusterCenters = clusterCenters
        self.hsvHist = hist
    def getHSVClusterCenters(self):
        return self.hsvClusterCenters, self.hsvHist