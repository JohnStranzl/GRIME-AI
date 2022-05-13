import os

import exifreader


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

            exif = exifreader.process_file(f)

            for k in sorted(exif.keys()):
                if k not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
                    self.header.append(k)

                    self.EXIF.append(exif[k])

                    # TEST POINT
                    # geo = {m: exif[m] for m in exif.keys() if m.startswith('GPS')}

        f.close()

        return self.header, self.EXIF

