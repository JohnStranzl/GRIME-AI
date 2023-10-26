class colorSegmentationParamsClass():

    def __init__(self, parent=None):
        self.className = "colorSegmentationParamsClass"
        self.instance = 1

        self.GCC            = True
        self.GLI            = False
        self.NDVI           = False
        self.ExG            = False
        self.RGI            = False

        self.Intensity      = True

        self.ShannonEntropy = False

        self.Texture        = False

        self.HSV            = True

        self.ROI            = True

        self.wholeImage     = False

        self.numColorClusters = 4
