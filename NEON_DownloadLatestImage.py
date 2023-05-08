import requests
from urllib.request import urlopen
from PyQt5.QtGui import QPixmap

# ======================================================================================================================
# THIS FUNCTION DOWNLOADS THE LATEST IMAGE FOR THE SITE SELECTED BY THE END-USER AND DISPLAYS IT IN THE GUI SO THE
# END-USER CAN CAN SEE WHAT THE PARTICULAR SITE LOOKS LIKE.
# ======================================================================================================================
def NEON_DownloadLatestImage(siteCode, domainCode):
    nErrorCode = -1
    nRetryCount = 3
    nWebImageCount = 0

    latestImageURL = 'https://phenocam.nau.edu/data/latest/NEON.D10.ARIK.DP1.20002.jpg'
    tmp = latestImageURL.replace('ARIK', siteCode)
    latestImageURL = tmp.replace('D10', domainCode)

    while nErrorCode == -1 and nRetryCount > 0:
        r = requests.get(latestImageURL, stream=True)

        if r.status_code != 404:
            nWebImageCount = 1
            data = urlopen(latestImageURL).read()
            latestImage = QPixmap()
            latestImage.loadFromData(data)
            nErrorCode = 0
        else:
            nWebImageCount = 0
            latestImage = []
            print("404: Download Latest Image")
            nErrorCode = -1
            nRetryCount = nRetryCount - 1

    return r.status_code, latestImage, nWebImageCount
