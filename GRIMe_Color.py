from urllib.request import urlopen

import cv2
import numpy as np
from PyQt5.QtGui import QPixmap

from GRIMe_Utils import GRIMe_Utils


class GRIMe_Color:
    def __init__(self):
        self.className = "GRIMe_Color"

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def loadColorImage(radioButton, filename):
        #if radioButton.isChecked():
        #    data = urlopen(filename).read()
        #    origImg = QPixmap()
        #    origImg.loadFromData(data)
        #    img = GRIMe_Utils.convertQImageToMat(origImg.toImage())
        #else:
        img = cv2.imread(filename)

        try:
            origImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            origImg = []

        return origImg

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def segmentColors(rgb, hsv, roiList):
        """
        segmentColors takes an image converted from an RGB color space to an HSV color space and extracts the colors defined
        by the colors clusters extracted from one or more regions-of-interest trained throughout the image.
        :param p1: a color image (HSV color space)
        :param p2: a color image (HSV color space)
        :param p3: a list of the ROIs and their color clusters that are to be extracted from the input image
        :return: a color image (HSV) of the colors extracted from the input image
        """

        # initialize return image to null
        rgb1 = []

        # Each ROI contains one or more colors extracted via kMeans clustering.
        # These color clusters are used as masks to extract colors that match the color clusters.
        i = 0

        for roi in roiList:
            hsv0 = roi.getHSVClusterCenters()

            for hsv1 in hsv0[0]:
                # refer to hue channel (in the colorbar)
                lower_mask = hsv[:, :, 0] > hsv1[0] - 8
                # refer to hue channel (in the colorbar)
                upper_mask = hsv[:, :, 0] < hsv1[0] + 15
                # refer to transparency channel (in the colorbar)
                saturation_mask = hsv[:, :, 1] > hsv1[1] - 3

                mask = upper_mask * lower_mask * saturation_mask
                red = rgb[:, :, 0] * mask
                green = rgb[:, :, 1] * mask
                blue = rgb[:, :, 2] * mask
                bags_masked = np.dstack((red, green, blue))
                if i > 0:
                    final_bags_masked = cv2.add(bags_masked, final_bags_masked)
                else:
                    final_bags_masked = bags_masked
                i = i + 1

        rgb1 = cv2.cvtColor(final_bags_masked, cv2.COLOR_RGB2BGR)

        return(rgb1)
