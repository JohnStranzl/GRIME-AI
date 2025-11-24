#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import shutil

import cv2
import numpy as np

from GRIME_AI.GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI.GRIME_AI_Color import GRIME_AI_Color
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from datetime import datetime

class GRIME_AI_ImageTriage:

    def __init__(self, show_gui=True):
        self.className = "GRIME_AI_ImageTriage"
        self.show_gui = show_gui

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def computeBlurAndBrightness(self, shiftSize):
        global currentImage

        img1 = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())
        grayImage = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        # DECIMATE IMAGE
        grayImage = self.resizeImage(grayImage, 50.0)

        hist = cv2.calcHist([grayImage], [0], None, [256], [0, 256])

        ''' BLUR DETECTION CALCULATIONS'''
        # grab the dimensions of the image and use the dimensions to derive the center (x, y)-coordinates
        (h, w) = grayImage.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))

        fft = np.fft.fft2(grayImage)
        fftShift = np.fft.fftshift(fft)

        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))

        # zero-out the center of the FFT shift (i.e., remove low frequencies),
        # apply the inverse shift such that the DC component once again becomes the top-left,
        # and then apply the inverse FFT
        fftShift[cY - shiftSize:cY + shiftSize, cX - shiftSize:cX + shiftSize] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)

        # compute the magnitude spectrum of the reconstructed image,
        # then compute the mean of the magnitude values
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)

        # IMAGE INTENSITY CALCULATIONS
        # blur = cv2.blur(grayImage, (5, 5))  # With kernel size depending upon image size
        blur = cv2.GaussianBlur(grayImage, (0, 0), 1) if 0. < 1 else grayImage
        intensity = cv2.mean(blur)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway

        # FREE UP MEMORY FOR THE NEXT IMAGE TO BE PROCESSSED
        del fftShift
        del fft
        del recon
        del blur

        return mean, intensity

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def cleanImages(self, folder, bFetchRecursive, blurThreshhold, shiftSize, brightnessMin, brightnessMAX, bCreateReport,
                    bMoveImages, bCorrectAlignment, bSavePolylines, strReferenceImageFilename, rotationThreshold):

        badImageCount = 0
        rotationAngle = 0.0
        horizontal_shift = 0.0
        vertical_shift = 0.0

        extensions = ('.jpg', '.jpeg', '.png')

        myGRIMe_Color = GRIME_AI_Color()

        if bCreateReport:
            csvFilename = 'ImageTriage_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
            imageQualityFile = os.path.join(folder, csvFilename)
            csvFile = open(imageQualityFile, 'a', newline='')
            csvFile.write('Focus Value, Focus Attrib, Intensity Value, Intensity Attrib., Rotation, H. Shift, V. Shift, Filename\n')

        # count the number of images that will potentially be processed and possibly saved with the specified extension
        # to display an "hourglass" to give an indication as to how long the process will take. Furthermore, the number
        # of images will help determine whether or not there is enough disk space to accommodate storing the images.
        imageCount = GRIME_AI_Utils().get_image_count(folder, extensions)

        if self.show_gui:
            progressBar = QProgressWheel(0, imageCount + 1)
            progressBar.show()

        imageIndex = 0

        # --------------------------------------------------------------------------------------------------------------
        #
        # --------------------------------------------------------------------------------------------------------------
        if bSavePolylines:
            polyFolder = folder + "\\poly\\"
            if not os.path.exists(polyFolder):
                os.mkdir(polyFolder)

        if bCorrectAlignment:
            warpFolder = folder + "\\warp\\"
            if not os.path.exists(warpFolder):
                os.mkdir(warpFolder)

        # --------------------------------------------------------------------------------------------------------------
        #
        # --------------------------------------------------------------------------------------------------------------
        # process images to determine which ones are too dark/too light, blurry/clear, etc and move them into a subfolder
        # created so they are not processed with nominal images.
        file_count, files = GRIME_AI_Utils().getFileList(folder, extensions, bFetchRecursive)

        if bCorrectAlignment:
            refImage = myGRIMe_Color.loadColorImage(strReferenceImageFilename)

        for file in files:
            if self.show_gui:
                progressBar.setWindowTitle(file)
                progressBar.setValue(imageIndex)
                progressBar.repaint()
            imageIndex += 1

            ext = os.path.splitext(file)[-1].lower()

            if ext in extensions:
                filename = os.path.join(folder, file)
                numpyImage = myGRIMe_Color.loadColorImage(filename)
                grayImage = cv2.cvtColor(numpyImage, cv2.COLOR_RGB2GRAY)

                hist = cv2.calcHist([grayImage], [0], None, [256], [0, 256])

                ''' BLUR DETECTION CALCULATIONS'''
                # grab the dimensions of the image and use the dimensions to derive the center (x, y)-coordinates
                (h, w) = grayImage.shape
                (cX, cY) = (int(w / 2.0), int(h / 2.0))

                fft = np.fft.fft2(grayImage)
                fftShift = np.fft.fftshift(fft)

                # compute the magnitude spectrum of the transform
                magnitude = 20 * np.log(np.abs(fftShift))

                # zero-out the center of the FFT shift (i.e., remove low frequencies),
                # apply the inverse shift such that the DC component once again becomes the top-left,
                # and then apply the inverse FFT
                fftShift[cY - shiftSize:cY + shiftSize, cX - shiftSize:cX + shiftSize] = 0
                fftShift = np.fft.ifftshift(fftShift)
                recon = np.fft.ifft2(fftShift)

                # compute the magnitude spectrum of the reconstructed image, then compute the mean of the magnitude values
                magnitude = 20 * np.log(np.abs(recon))
                mean = np.mean(magnitude)

                # ----------------------------------------------------------------------------------------------------
                # IMAGE INTENSITY CALCULATIONS
                # ----------------------------------------------------------------------------------------------------
                # blur = cv2.blur(grayImage, (5, 5))  # With kernel size depending upon image size
                blur = cv2.GaussianBlur(grayImage, (0, 0), 1) if 0. < 1 else grayImage
                intensity = cv2.mean(blur)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway

                # DECISION LOGIC
                bMove = False
                strFFTFocusMetric = 'Nominal'
                strFocusMetric = 'N/A'
                strIntensity = 'Nominal'

                # ----------------------------------------------------------------------------------------------------
                # CHECK MEAN AGAINST THRESHOLD TO DETERMINE IF THE IMAGE IS BLURRY/FOGGY/OUT-OF-FOCUS/ETC.
                # ----------------------------------------------------------------------------------------------------
                if mean <= blurThreshhold:
                    strFFTFocusMetric = "Blurry"
                    bMove = True

                # ----------------------------------------------------------------------------------------------------
                # CHECK TO SEE IF THE OVERALL IMAGE IS TOO DARK OR TOO BRIGHT
                # ----------------------------------------------------------------------------------------------------
                if float(intensity) < float(brightnessMin):
                    strIntensity = "Too Dark"
                    bMove = True
                elif float(intensity) > float(brightnessMAX):
                    strIntensity = "Too Light"
                    bMove = True

                if len(strReferenceImageFilename) > 0 and bCorrectAlignment is True:
                    # ----------------------------------------------------------------------------------------------------
                    # GET THE ROTATION ANGLE OF THE IMAGE
                    # ----------------------------------------------------------------------------------------------------
                    rotationAngle, poly_img, warp_img = self.checkImageAlignment(refImage, numpyImage)

                    baseFilename = os.path.basename(file)

                    if bSavePolylines:
                        polyFilename = baseFilename + "_poly.jpg"
                        polyFilename_with_path = os.path.join(polyFolder, polyFilename)
                        cv2.imwrite(polyFilename_with_path, poly_img)

                    if (bCorrectAlignment and (rotationAngle > rotationThreshold)):
                        warpFilename = baseFilename + "_align.jpg"
                        warpFilename_with_path = os.path.join(warpFolder, warpFilename)
                        cv2.imwrite(warpFilename_with_path, warp_img)

                    # ----------------------------------------------------------------------------------------------------
                    # CHECK FOR IMAGE TRANSLATION
                    # ----------------------------------------------------------------------------------------------------
                    horizontal_shift, vertical_shift = self.checkImageShift(refImage, grayImage)

                # MOVE THE IMAGE REJECTS TO A SUBFOLDER IF THE USER CHOOSE THIS OPTION
                if bMoveImages and bMove:
                    # create a subfolder beneath the current root folder if the option to move less than nominal images is selected
                    filename = os.path.basename(file)
                    filepath = os.path.dirname(file)
                    tempFolder = os.path.join(filepath, "MovedImages")
                    if not os.path.exists(tempFolder):
                        os.makedirs(tempFolder)

                    shutil.move(file, tempFolder)

                    filename = os.path.join(tempFolder, filename)

                if bMove:
                    badImageCount = badImageCount + 1

                # ----------------------------------------------------------------------------------------------------
                # CREATE A CSV FILE THAT CONTAINS THE FOCUS AND INTENSITY METRICS ALONG WITH HYPERLINKS TO THE IMAGES
                # ----------------------------------------------------------------------------------------------------
                if bCreateReport:
                    # Build the hyperlink formula
                    formula = f'=HYPERLINK("{filename}", "{os.path.basename(filename)}")'
                    #strHyperlink = strHyperlink.replace("\\", "\\\\")  # escape backslashes

                    # If the formula contains commas, enclose the entire field in extra quotes and escape inner quotes.
                    if ',' in formula:
                        safe_formula = formula.replace('"', '""')
                        formula = f'"{safe_formula}"'

                    # Wrap the formula in double quotes so Excel sees it as a formula
                    quotedHyperlink = f'"{formula}"'

                    # Format the CSV line
                    strOutputString = '%3.2f,%s,%3.2f,%s,%3.2f,%3.2f,%3.2f,%s\n' % (
                        mean, strFFTFocusMetric, intensity, strIntensity,
                        rotationAngle, horizontal_shift, vertical_shift, formula
                    )
                    csvFile.write(strOutputString)

                # ----------------------------------------------------------------------------------------------------
                # FREE UP MEMORY FOR THE NEXT IMAGE TO BE PROCESSSED
                # ----------------------------------------------------------------------------------------------------
                del fftShift
                del fft
                del recon
                del blur

        if badImageCount == 0:
            strMessage = 'No bad images found.'
            print(strMessage)
            if self.show_gui:
                msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage)
                response = msgBox.displayMsgBox()

        # ----------------------------------------------------------------------------------------------------
        # clean-up before exiting function
        # 1. close and delete the progress bar
        # 2. close the EXIF log file, if opened
        # ----------------------------------------------------------------------------------------------------
        if bCreateReport:
            csvFile.close()

        if self.show_gui:
            progressBar.close()
            del progressBar


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def checkImageShift(self, refImage, image):
        # Convert images to grayscale
        refImageGray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
        imageGray = image

        # Compute the keypoints and descriptors
        orb = cv2.ORB_create()

        # Find keypoints and descriptors.
        # The first arg is the image, second arg is the mask (which is not required in this case).
        keypoints1, descriptors1 = orb.detectAndCompute(refImageGray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(imageGray, None)

        # Match descriptors between the two images
        # We create a Brute Force matcher with Hamming distance as measurement mode.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the two sets of descriptors.
        matches = matcher.match(descriptors1, descriptors2)

        # Sort matches on the basis of their Hamming distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography matrix
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Compute horizontal and vertical shifts
        horizontal_shift = M[0, 2]
        vertical_shift = M[1, 2]

        return horizontal_shift, vertical_shift


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def checkImageAlignment(self, refImage, image):
        # Convert to grayscale.
        refImageGray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
        imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Create ORB detector with 5000 features.
        orb_detector = cv2.ORB_create(5000)

        # Find keypoints and descriptors.
        # The first arg is the image, second arg is the mask (which is not required in this case).
        keypoints1, descriptors1 = orb_detector.detectAndCompute(imageGray, None)
        keypoints2, descriptors2 = orb_detector.detectAndCompute(refImageGray, None)

        # Match features between the two images.
        # We create a Brute Force matcher with Hamming distance as measurement mode.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the two sets of descriptors.
        matches = matcher.match(descriptors1, descriptors2)

        # Sort matches on the basis of their Hamming distance.
        matches = sorted(matches, key=lambda x: x.distance)





        ## Take the top 90 % matches forward.
        matches = matches[:int(len(matches) * 0.9)]
        no_of_matches = len(matches)

        ## Define empty matrices of shape no_of_matches * 2.
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = keypoints1[matches[i].queryIdx].pt
            p2[i, :] = keypoints2[matches[i].trainIdx].pt

        ## Find the homography matrix.
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

        # EXTRACT MATCHED KEYPOINTS
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # COMPUTE ROTATION ANGLE
        rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

        if 1:
            # Use this matrix to transform the colored image wrt the reference image.
            width = image.shape[1]
            height = image.shape[0]
            swapImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            warp_img = cv2.warpPerspective(swapImage, homography, (width, height))

        if 1:
            h, w = refImage.shape[:2]

            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # draw found regions
            img2 = cv2.polylines(image, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)

            # draw match lines
            poly_img = cv2.drawMatches(refImage, keypoints1, image, keypoints2, matches[:20], None, flags=2)

        return (rotation_angle, poly_img, warp_img)

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def resizeImage(self, image, scale_percent):
        # --------------------------------------------------------------------------------
        # reshape the image to be a list of pixels
        # --------------------------------------------------------------------------------
        if scale_percent == 100.0:
            return image
        else:
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)

            dim = (width, height)

            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            return resized


