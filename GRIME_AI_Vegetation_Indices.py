from urllib.request import urlopen

import cv2
import numpy as np

class greennessIndex():
    GCC  = False
    GLI  = False
    ExG  = False
    RGI  = False
    NDVI = False

class GRIME_AI_Vegetation_Indices:
    def __init__(self):
        self.className = "GRIME_AI_Vegetation_Indices"

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def computeGreennessIndex(self, redSum, greenSum, blueSum):

        try:
            denominator = redSum + greenSum + blueSum
            if denominator > 0.0:
                greennessIndex = greenSum / (redSum + greenSum + blueSum)
            else:
                greennessIndex = 0.0
        except:
            greennessIndex = -999

        return(greennessIndex)


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def computeExcessGreenIndex(self, redSum, greenSum, blueSum):

        try:
            greennessIndex = (2.0 * greenSum) - redSum - blueSum
        except:
            greennessIndex = -999

        return(greennessIndex)


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def computeGreenLeafIndex(self, redSum, greenSum, blueSum):

        try:
            greennessIndex = ((2.0 * greenSum) - redSum - blueSum)/ ((2.0 * greenSum) + redSum + blueSum)
        except:
            greennessIndex = -999

        return(greennessIndex)


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def computeNormalizedDifferenceVegetationIndex(self, redSum, greenSum):

        try:
            greennessIndex = (greenSum - redSum) / (redSum + greenSum)
        except:
            greennessIndex = -999

        return(greennessIndex)

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def computeRedGreenIndex(self, redSum, greenSum):

        redGreenIndex = redSum / greenSum

        return(redGreenIndex)


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def computeGreennessValue(self, colorSegmentationParams, redSum, greenSum, blueSum):

        if colorSegmentationParams.GCC:
            return ('%3.2f' % GRIME_AI_Vegetation_Indices().computeGreennessIndex(redSum, greenSum, blueSum))
        if colorSegmentationParams.GLI:
            return('%3.2f' % GRIME_AI_Vegetation_Indices().computeGreenLeafIndex(redSum, greenSum, blueSum))
        if colorSegmentationParams.NDVI:
            return('%3.2f' % GRIME_AI_Vegetation_Indices().computeNormalizedDifferenceVegetationIndex(redSum, greenSum))
        if colorSegmentationParams.ExG:
            return('%3.2f' % GRIME_AI_Vegetation_Indices().computeExcessGreenIndex(redSum, greenSum, blueSum))
        if colorSegmentationParams.RGI:
            return('%3.4f' % GRIME_AI_Vegetation_Indices().computeRedGreenIndex(redSum, greenSum))
