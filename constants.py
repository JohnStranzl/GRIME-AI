# ====================================================================================================
#
# ====================================================================================================
class edgeMethodsClass():

    #CONSTANTS
    CANNY = 1
    SOBEL_X = 2
    SOBEL_Y = 3
    SOBEL_XY = 4
    LAPLACIAN = 5

    def __init__(self, parent=None):
        self.selected = False
        self.method = self.CANNY

    def getSelected(self):
        return self.selected

    def getEdgeMethod(self):
        return self.method

# ====================================================================================================
#
# ====================================================================================================
class featureMethodsClass():

    # CONSTANTS
    SIFT = 20
    ORB = 21

    def __init__(self, parent=None):
        self.selected = False
        self.method = self.ORB

    def getSelected(self):
        return self.selected

    def getEdgeMethod(self):
        return self.method
