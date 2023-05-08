# ======================================================================================================================
# THIS CLASS WILL HOLD THE SOBEL EDGE DETECTION INFORMATION FOR THE X-AXIS, THE Y-AXIS AND THE COMBINED
# X- AND Y-AXIS FOR AN IMAGE
# ======================================================================================================================
class sobelData:
    def __init__(self):
        self.className = "sobelData"

    def setSobelX(self, sobelX):
        self.sobelX = sobelX

    def setSobelY(self, sobelY):
        self.sobelY = sobelY

    def setSobelXY(self, sobelXY):
        self.sobelXY = sobelXY

    def getSobelX(self):
        return (self.sobelX)

    def getSobelY(self):
        return (self.sobelY)

    def getSobelXY(self):
        return (self.sobelXY)


