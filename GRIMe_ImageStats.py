# ======================================================================================================================
#
# ======================================================================================================================
class GRIMe_ImageStats:
    def __init__(self, blurValue=0.0, brightnessValue=0.0, label=''):
        self.blurValue = blurValue
        self.brightnessValue = brightnessValue
        self.label = label

    def setBlurValue(self, blurValue):
        self.blurValue = blurValue

    def getBlurValue(self):
        return self.blurValue

    def setBrightnessValue(self, brightnessValue):
        self.brightnessValue = brightnessValue

    def getBrightnessValue(self):
        return self.brightnessValue

    def setLabel(self, label):
        self.label = label

    def getLabel(self):
        return self.label