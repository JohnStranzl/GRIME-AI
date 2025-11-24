import os
from datetime import datetime, date, time

import exifreader

from PIL import Image, ImageQt, ExifTags
from PIL.ExifTags import TAGS
import piexif


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
    def extractEXIFDataDateTime(self, fullPathAndFilename):
        nYear = 1970
        nMonth = 1
        nDay = 1

        try:
            # extract EXIF info to determine what time the image was acquired. If EXIF info is not found,
            # throw an exception and see if the information is embedded in the filename. Currently, we are
            # working with images from NEON and PBT. The PBT images have EXIF data and the NEON/PhenoCam
            # do not appear to have EXIF data.
            #myEXIFData = EXIFData()
            #header, data = myEXIFData.extractEXIFData(fullPathAndFilename)

            # FETCH THE EXIF DATA FROM THE IMAGE FILE
            image_exif = Image.open(fullPathAndFilename)._getexif()

            exif = {ExifTags.TAGS[k]: v for k, v in image_exif.items() if k in ExifTags.TAGS and type(v) is not bytes}

            # FETCH THE DATE
            date_obj = datetime.strptime(exif['DateTimeOriginal'], '%Y:%m:%d %H:%M:%S')

            nYear = date_obj.date().year
            nMonth = date_obj.date().month
            nDay = date_obj.date().day

            nHours = date_obj.time().hour
            nMins = date_obj.time().minute
            nSecs = date_obj.time().second

            bEXIFDataFound = True
        except Exception:
            # assume the filename contains the timestamp for the image (assumes the image file is a PBT image)
            bEXIFDataFound = False

            try:
                nHours = int(str(strTime[0:2]))
                nMins = int(str(strTime[2:4]))
                nSecs = int(str(strTime[4:6]))
            except Exception:
                nHours = 0
                nMins = 0
                nSecs = 0

        image_date = date(nYear, nMonth, nDay)
        image_time = time(nHours, nMins, nSecs)

        return(image_date, image_time)



    def get_original_datetime(self, image_path):
        # Open the image file
        img = Image.open(image_path)

        # Extract the EXIF data
        exif_data = piexif.load(img.info['exif'])

        # Get the original date/time
        original_datetime = exif_data['Exif'][piexif.ExifIFD.DateTimeOriginal]

        # Convert bytes to string
        original_datetime = original_datetime.decode('utf-8')

        return original_datetime
