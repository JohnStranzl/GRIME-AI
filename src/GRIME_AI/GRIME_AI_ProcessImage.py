#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import cv2
import GRIME_AI.sobelData
import numpy as np

from PIL import Image, ImageQt
from PyQt5.QtGui import QPixmap, QImage
from GRIME_AI.constants import edgeMethodsClass, featureMethodsClass

from GRIME_AI.GRIME_AI_Image_Processing import GRIME_AI_Image_Processing
from GRIME_AI.GRIME_AI_Image_Conversion import GRIME_AI_Image_Conversion

from GRIME_AI.GRIME_AI_Utils import GRIME_AI_Utils


class GRIME_AI_ProcessImage:
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.className = "GRIME_AI_ProcessImage"

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def processCanny(self, img1, gray, edgeMethodSettings):

        #JESkernelSize = self.spinBoxCannyKernel.value()
        highThreshold = edgeMethodSettings.getCannyThresholdHigh()
        lowThreshold  = edgeMethodSettings.getCannyThresholdLow()
        kernelSize    = 3

        # BLUR IMAGE TO REMOVE NOISE
        img_blur = cv2.GaussianBlur(gray, (3, 3), 0)

        #otsu_thresh, _     = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)
        #triangle_thresh, _ = cv2.threshold(img_blur, 0, 255, cv2.THRESH_TRIANGLE)
        otsu_thresh, _     = cv2.threshold(img_blur, lowThreshold, highThreshold, cv2.THRESH_OTSU)
        triangle_thresh, _ = cv2.threshold(img_blur, lowThreshold, highThreshold, cv2.THRESH_TRIANGLE)

        otsu_thresh     = self.getThresholdRange(otsu_thresh)
        triangle_thresh = self.getThresholdRange(triangle_thresh)

        # Find Canny edges
        #edges = cv2.Canny(img_blur, highThreshold, lowThreshold, kernelSize)
        #edges = cv2.Canny(img_blur, *otsu_thresh, kernelSize)
        #edges = cv2.Canny(img_blur, *triangle_thresh)
        edges = cv2.Canny(img_blur, highThreshold, lowThreshold, kernelSize)

        # Find Contours
        # Use a copy of the image e.g. edged.copy() since findContours alters the image
        #contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours (-1 = draw all contours)
        cv2.drawContours(img1, contours, -1, (0, 255, 0), 2)

        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        # for line in lines:
        # x1, y1, x2, y2 = line[0]
        # cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imwrite('C:/Users/Astrid Haugen/Documents/houghlines5.jpg', img1)
        # q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8)
        # q_img = QImage(temp.data, temp.shape[1], temp.shape[0], QImage.Format_Grayscale8)

        # QImage BYTE ORDER IS B, G, R
        #q_img = QImage(img1.data, img1.shape[1], img1.shape[0], QImage.Format_BGR888)
        q_img = QImage(img1.data, img1.shape[1], img1.shape[0], QImage.Format_BGR888)

        return(QPixmap(q_img))

    def getThresholdRange(self, threshold, sigma=0.33):
        return (1 - sigma) * threshold, (1 + sigma) * threshold

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def processSobel(self, gray, sobelKernelSize, method):

        myImageProcssing = GRIME_AI_Image_Conversion()
        #img1 = myImageProcssing.qimage_to_opencv(img1)

        #gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        mySobel = sobelData.sobelData()

        mySobel.setSobelX(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobelKernelSize))
        mySobel.setSobelY(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobelKernelSize))

        if method == edgeMethodsClass.SOBEL_X:
            #edges = mySobel.getSobelX().astype(np.uint8)
            edges = mySobel.getSobelX()
            q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8)
            pix = QPixmap(q_img)

        elif method == edgeMethodsClass.SOBEL_Y:
            #edges = mySobel.getSobelY().astype(np.uint8)
            edges = mySobel.getSobelY()
            q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8)
            pix = QPixmap(q_img)

        elif method == edgeMethodsClass.SOBEL_XY:
            myImageProcessing = GRIME_AI_Image_Processing()
            edges = myImageProcessing.apply_sobel_filter(gray, sobelKernelSize)
            q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8)
            pix = QPixmap(q_img)

        return(pix)


    def getGradientMagnitude(self, im):
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(im, ddepth, 1, 0)
        dy = cv2.Sobel(im, ddepth, 0, 1)
        magnitude = cv2.magnitude(dx, dy)
        return magnitude

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def processLaplacian(self, img1):
        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)

        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

        gray = cv2.cvtColor(ycrcb_img, cv2.COLOR_RGB2GRAY)

        # convert back to RGB color-space from YCrCb
        lapImg = laplace_of_gaussian(gray)
        gray = cv2.equalizeHist(lapImg)
        cv2.dilate(gray, (5, 5), gray)
        cv2.erode(gray, (5, 5), gray)

        gray = Image.fromarray(gray)
        q_img = ImageQt.toqimage(gray)

        return(QPixmap.fromImage(q_img))


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def processSIFT(self, img1, gray):
        edges = calcSIFT(img1, gray)
        q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_RGB888)
        pix = QPixmap(q_img)


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def processORB(self, img1, gray, featureMethodSettings):

        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)

        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

        gray = cv2.cvtColor(ycrcb_img, cv2.COLOR_RGB2GRAY)

        lapImg = laplace_of_gaussian(gray)
        gray = cv2.equalizeHist(lapImg)
        cv2.dilate(gray, (5, 5), gray)
        cv2.erode(gray, (5, 5), gray)

        # gray = Image.fromarray(gray)

        gray = cv2.cvtColor(ycrcb_img, cv2.COLOR_RGB2GRAY)
        edges = calcOrb(gray, featureMethodSettings.orbMaxFeatures)
        q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_RGB888)

        del gray
        del edges

        return(QPixmap(q_img))

# ======================================================================================================================
# THIS FUNCTION WILL PROCESS THE CURRENT IMAGE BASED UPON THE SETTINGS SELECTED BY THE END-USER.
# THE IMAGE STORAGE TYPE IS mat
# ======================================================================================================================
def processImageMat(self, myImage):
    edges = []
    img2 = []
    imageFormat = 0

    if not myImage == []:
        img1 = myImage

        if self.checkboxKMeans.isChecked():
            myGRIMe_Color = GRIMe_Color()
            qImg, clusterCenters, hist = myGRIMe_Color.KMeans(self, img1, self.spinBoxColorClusters.value())

            checkColorMatch(clusterCenters, hist, self.roiList)

        # CONVERT COLOR IMAGE TO GRAY SCALE
        gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        # REMOVE NOISE FROM THE IMAGE
        if len(gray) != 0:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # EDGE DETECTION METHODS
        if len(gray) != 0:
            if self.radioButtonCanny.isChecked():
                highThreshold = self.spinBoxCannyHighThreshold.value()
                lowThreshold = self.spinBoxCannyLowThreshold.value()
                kernelSize = self.spinBoxCannyKernel.value()
                edges = cv2.Canny(img1, highThreshold, lowThreshold, kernelSize)
                # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
                # for line in lines:
                # x1, y1, x2, y2 = line[0]
                # cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.imwrite('C:/Users/Astrid Haugen/Documents/houghlines5.jpg', img1)

            elif self.radioButtonSIFT.isChecked():
                edges = calcSIFT(img1, gray)
            elif self.radioButtonORB.isChecked():
                value = self.spinBoxOrbMaxFeatures.value()
                edges = calcOrb(img1, value)
                # edges = cv2.cvtColor(edges, cv2.COLOR_RGBA2RGB)
            elif self.radioButtonLaplacian.isChecked():
                edges = cv2.Laplacian(img1, cv2.CV_64F)
            elif self.radioButtonSobelX.isChecked() or self.radioButtonSobelY.isChecked() or self.radioButtonSobelXY.isChecked():
                mySobel = sobelData()
                sobelKernelSize = self.spinBoxSobelKernel.value()
                mySobel.setSobelX(cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=sobelKernelSize))
                mySobel.setSobelY(cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=sobelKernelSize))
                mySobel.setSobelXY(cv2.Sobel(img1, cv2.CV_64F, 1, 1, ksize=sobelKernelSize))
                if self.radioButtonSobelX.isChecked():
                    edges = (mySobel.getSobelX() * 255 / mySobel.getSobelX().max()).astype(np.uint8)
                elif self.radioButtonSobelY.isChecked():
                    edges = (mySobel.getSobelY() * 255 / mySobel.getSobelY().max()).astype(np.uint8)
                elif self.radioButtonSobelXY.isChecked():
                    edges = (mySobel.getSobelXY() * 255 / mySobel.getSobelXY().max()).astype(np.uint8)

            if self.radioButtonSIFT.isChecked():
                imageFormat = QImage.Format_RGB888
            elif self.radioButtonORB.isChecked():
                imageFormat = QImage.Format_RGB888
            else:
                imageFormat = QImage.Format_Grayscale8

    return edges, imageFormat


# ======================================================================================================================
#
# ======================================================================================================================
def laplace_of_gaussian(gray_img, sigma=1., kappa=0.75, pad=False):
    """
    Applies Laplacian of Gaussians to grayscale image.

    :param gray_img: image to apply LoG to
    :param sigma:    Gauss sigma of Gaussian applied to image, <= 0. for none
    :param kappa:    difference threshold as factor to mean of image values, <= 0 for none
    :param pad:      flag to pad output w/ zero border, keeping input image size
    """
    assert len(gray_img.shape) == 2
    img = cv2.GaussianBlur(gray_img, (0, 0), sigma) if 0. < sigma else gray_img
    img = cv2.Laplacian(img, cv2.CV_64F)
    rows, cols = img.shape[:2]
    # min/max of 3x3-neighbourhoods
    min_map = np.minimum.reduce(list(img[r:rows - 2 + r, c:cols - 2 + c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows - 2 + r, c:cols - 2 + c]
                                     for r in range(3) for c in range(3)))
    # bool matrix for image value positiv (w/out border pixels)
    pos_img = 0 < img[1:rows - 1, 1:cols - 1]
    # bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0
    # bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0
    # sign change at pixel?
    zero_cross = neg_min + pos_max
    # values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.
    # optional thresholding
    if 0. <= kappa:
        thresh = float(np.absolute(img).mean()) * kappa
        values[values < thresh] = 0.
    log_img = values.astype(np.uint8)
    if pad:
        log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)

    return log_img


# ======================================================================================================================
# THIS FUNCTION WILL USE THE SIFT FEATURE DETECTION ALGORITHM TO FIND FEATURES IN THE IMAGE THAT IS PASSED TO THIS FUNCTION.
# ======================================================================================================================
def calcSIFT(image, gray):
    # REMOVE NOISE FROM THE IMAGE
    img1 = cv2.GaussianBlur(gray, (3, 3), 0)

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img1, None)
    edges = cv2.drawKeypoints(gray, kp, img1)

    return edges


# ======================================================================================================================
# THIS FUNCTION WILL USE THE ORB FEATURE DETECTION ALGORITHM TO FIND FEATURES IN THE IMAGE THAT IS
# PASSED TO THIS FUNCTION.
# ======================================================================================================================
def calcOrb(image, nMaxFeatures):

    if len(image.shape) == 3:
       imageBW = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        imageBW = image

    orb = cv2.ORB_create()

    orb.setEdgeThreshold(50)
    orb.setNLevels(10)
    orb.setPatchSize(30)
    orb.setMaxFeatures(nMaxFeatures)

    kp = orb.detect(imageBW, None)
    kp, des = orb.compute(imageBW, kp)

    edges = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)

    return edges


# ======================================================================================================================
# ALIGN IMAGES USING A REFERENCE IMAGE
# ======================================================================================================================
def imageAlignment(referenceImageFilename, imageFilename):

    # Open the image files.
    #img1_color = cv2.imread("NM_Pecos_River_near_Acme___2023-08-17T00-00-46Z.jpg")  # Image to be aligned.
    #img2_color = cv2.imread("NM_Pecos_River_near_Acme___2023-12-19T23-30-07Z.jpg")  # Reference image.
    img1_color = cv2.imread(imageFilename)  # Image to be aligned.
    img2_color = cv2.imread(referenceImageFilename)  # Reference image.

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    # matches.sort(key = lambda x: x.distance)
    matches = sorted(matches, key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

        # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color, homography, (width, height))

    # Save the output.
    cv2.imwrite('output.jpg', transformed_img)

    return (transformed_img)
