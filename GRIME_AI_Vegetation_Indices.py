from urllib.request import urlopen

import cv2
import numpy as np

from GRIME_AI_Utils import GRIME_AI_Utils

class greennessIndex():
    GCC  = False
    GLI  = False
    ExG  = False
    RGI  = False
    NDVI = False

class GRIME_Vegetation_Indices:
    def __init__(self):
        self.className = "GRIME_AI_Vegetation_Indices"

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def computeGreennessIndex(redSum, greenSum, blueSum):

        try:
            greennessIndex = greenSum / (redSum + greenSum + blueSum)
        except:
            greennessIndex = -999

        return(greennessIndex)


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def computeExcessGreenIndex(redSum, greenSum, blueSum):

        try:
            greennessIndex = (2.0 * greenSum) - redSum - blueSum
        except:
            greennessIndex = -999

        return(greennessIndex)


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def computeGreenLeafIndex(redSum, greenSum, blueSum):

        try:
            greennessIndex = ((2.0 * greenSum) - redSum - blueSum)/ ((2.0 * greenSum) + redSum + blueSum)
        except:
            greennessIndex = -999

        return(greennessIndex)


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def computeNormalizedDifferenceVegetationIndex(redSum, greenSum):

        try:
            greennessIndex = (greenSum - redSum) / (redSum + greenSum)
        except:
            greennessIndex = -999

        return(greennessIndex)

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def computeRedGreenIndex(redSum, greenSum):

        redGreenIndex = redSum / greenSum

        return(redGreenIndex)

