import os
import re
import datetime

import exifreader

from PIL import Image, ImageQt
from PIL.ExifTags import TAGS

# ======================================================================================================================
#
# ======================================================================================================================
class EXIFData:

    def __init__(self, fullPathAndFilename=None):

        if fullPathAndFilename != None:
            self.fullPathAndFilename = fullPathAndFilename

        self.header = []
        self.header.append("Filename")

        self.EXIF = []

    def getHeader(self):
        return self.header

    def getEXIF(self):
        return self.EXIF

    def setFullPathAndFilename(self, fullPathAndFilename):
        self.fullPathAndFilename = fullPathAndFilename

    def extractEXIFData(self, fullPathAndFilename=None):
        # ITERATE THROUGH ALL THE TAGS AND EXTRACT EXIF DATA

        if fullPathAndFilename == None:
            fullPathAndFilename = self.fullPathAndFilename

        strFilename = os.path.basename(fullPathAndFilename)
        self.EXIF.append(strFilename)

        with open(fullPathAndFilename, 'rb') as f:
            exif = []

            self.PIL_get_exif(f)

            exif = exifreader.process_file(f)

            for k in sorted(exif.keys()):
                if k not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
                    self.header.append(k)

                    self.EXIF.append(exif[k])

                    # TEST POINT
                    # geo = {m: exif[m] for m in exif.keys() if m.startswith('GPS')}

        f.close()

        #return self.header, self.EXIF
        return exif

    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------
    def PIL_get_exif(self, fn):
        ret = {}
        i = Image.open(fn)
        info = i._getexif()
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            ret[decoded] = value
        return ret

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def extractEXIFdata(self, fullPathAndFilename):
        nYear = 1970
        nMonth = 1
        nDay = 1

        try:
            # extract EXIF info to determine what time the image was acquired. If EXIF info is not found,
            # throw an exception and see if the information is embedded in the filename. Currently, we are
            # working with images from NEON and PBT. The PBT images have EXIF data and the NEON/PhenoCam
            # do not appear to have EXIF data.
            myEXIFData = EXIFData()
            header, data = myEXIFData.extractEXIFData(fullPathAndFilename)

            strTemp = str(myEXIFData.getEXIF()[8])
            dateOriginal = re.search('\d{4}:\d{2}:\d{2}', strTemp).group(0)
            timeOriginal = re.search(' \d{2}:\d{2}:\d{2}', strTemp).group(0)

            nYear = int(str(dateOriginal[0:4]))
            nMonth = int(str(dateOriginal[5:7]))
            nDay = int(str(dateOriginal[8:10]))

            nHours = int(str(timeOriginal[1:3]))
            nMins = int(str(timeOriginal[4:6]))
            nSecs = int(str(timeOriginal[7:9]))

            bEXIFDataFound = True
        except:
            # assume the filename contains the timestamp for the image (assumes the image file is a PBT image)
            bEXIFDataFound = False

            try:
                nHours = int(str(strTime[0:2]))
                nMins = int(str(strTime[2:4]))
                nSecs = int(str(strTime[4:6]))
            except:
                nHours = 0
                nMins = 0
                nSecs = 0

        image_date = datetime.date(nYear, nMonth, nDay)
        image_time = datetime.time(nHours, nMins, nSecs)

        return(image_date, image_time)


