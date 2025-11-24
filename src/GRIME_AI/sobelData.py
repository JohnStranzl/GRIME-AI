import numpy as np

# ======================================================================================================================
# THIS CLASS WILL HOLD THE SOBEL EDGE DETECTION INFORMATION FOR THE X-AXIS, THE Y-AXIS AND THE COMBINED
# X- AND Y-AXIS FOR AN IMAGE
# ======================================================================================================================
class sobelData:
    def __init__(self):
        self.className = "sobelData"

    def setSobelX(self, sobelX):
        sobel_uint8 = np.uint8(sobelX)
        self.sobelX = sobel_uint8

    def setSobelY(self, sobelY):
        sobel_uint8 = np.uint8(sobelY)
        self.sobelY = sobel_uint8

    def setSobelXY(self, sobelXY):
        self.sobelXY = sobelXY

    def getSobelX(self):
        return (self.sobelX)

    def getSobelY(self):
        return (self.sobelY)

    def getSobelXY(self):
        return (self.sobelXY)


