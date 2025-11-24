class nitrateData:
    def __init__(self, domainID, siteID, horizontalPosition,verticalPosition, startDateTime, endDateTime, surfWaterNitrateMean, surfWaterNitrateMinimum, surfWaterNitrateMaximum, surfWaterNitrateVariance, surfWaterNitrateNumPts, surfWaterNitrateExpUncert, surfWaterNitrateStdErMean, finalQF, publicationDate, release):
        self.domainID = domainID
        self.siteID = siteID
        self.horizontalPosition = horizontalPosition
        self.verticalPosition = verticalPosition
        self.startDateTime = startDateTime
        self.endDateTime = endDateTime
        self.surfWaterNitrateMean = surfWaterNitrateMean
        self.surfWaterNitrateMinimum = surfWaterNitrateMinimum
        self.surfWaterNitrateMaximum = surfWaterNitrateMaximum
        self.surfWaterNitrateVariance = surfWaterNitrateVariance
        self.surfWaterNitrateNumPts = surfWaterNitrateNumPts
        self.surfWaterNitrateExpUncert = surfWaterNitrateExpUncert
        self.surfWaterNitrateStdErMean = surfWaterNitrateStdErMean
        self.finalQF = finalQF
        self.publicationDate = publicationDate
        self.release = release

    def getNitrateMean(self):
        return self.surfWaterNitrateMean

