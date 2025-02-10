# ======================================================================================================================
# THIS CLASS WILL HOLD THE FULLY QUALIFIED URL PATH AND THE TIME STAMP FOR A PARTICULAR IMAGE SELECTED
# BY THE END-USER.
# ======================================================================================================================
class imageData():

    def __init__(self, fullPathAndFilename, hours, minutes, seconds):
        self.fullPathAndFilename = fullPathAndFilename
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds


