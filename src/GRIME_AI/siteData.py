# ======================================================================================================================
# THIS CLASS WILL HOLD SITE INFORMATION DATA FOR A PARTICULAR SITE SELECTED BY THE END-USER.
# ======================================================================================================================
class siteData:
    def __init__(self, siteID, siteName, phenocamSite, latitude, longitude):
        self.siteID = siteID
        self.siteName = siteName
        self.phenocamSite = phenocamSite
        self.latitude = latitude
        self.longitude = longitude