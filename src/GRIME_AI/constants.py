# ====================================================================================================
#
# ====================================================================================================
class edgeMethodsClass():

    #CONSTANTS
    NONE      = 0
    CANNY     = 1
    SOBEL_X   = 2
    SOBEL_Y   = 3
    SOBEL_XY  = 4
    LAPLACIAN = 5

    def __init__(self, parent=None):
        self.selected = False

        self.method = self.CANNY
        self.canny_threshold_high = 100
        self.canny_threshold_low  =  70
        self.cannyKernel          =   3

        self.sobelKernel = 10

    def getSelected(self):
        return self.selected

    def getEdgeMethod(self):
        return self.method

    def getCannyThresholdHigh(self):
        return self.canny_threshold_high

    def getCannyThresholdLow(self):
        return self.canny_threshold_low

    def getCannyKernel(self):
        return self.cannyKernel

    def getSobelKernel(self):
        return self.sobelKernel

# ====================================================================================================
#
# ====================================================================================================
class featureMethodsClass():

    # CONSTANTS
    NONE = 0
    SIFT = 20
    ORB = 21

    def __init__(self, parent=None):
        self.selected = False
        self.method = self.ORB
        self.orbMaxFeatures = 10000

    def getSelected(self):
        return self.selected

    def getEdgeMethod(self):
        return self.method

    def getOrbMaxFeatures(self):
        return self.orbMaxFeatures

# ====================================================================================================
#
# ====================================================================================================
class modelSettingsClass():

    def __init__(self, parent=None):
        self.saveOriginalModelImage = False
        self.saveModelMasks = False

    def getSaveModelMasks(self):
        return self.saveModelMasks

    def getSaveOriginalModelMask(self):
        return self.saveOriginalModelImage