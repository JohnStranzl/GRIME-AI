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
from GRIME_AI.GRIME_AI_ImageQualityCheck import ImageQualityAnalyzer
from datetime import datetime


class GRIME_AI_ImageTriage:

    def __init__(self, show_gui=True):
        self.className = "GRIME_AI_ImageTriage"
        self.show_gui  = show_gui

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def cleanImages(self, folder, bFetchRecursive, blurThreshhold, shiftSize, brightnessMin, brightnessMAX,
                    bCreateReport, bMoveImages, bCorrectAlignment, bSavePolylines,
                    strReferenceImageFilename, rotationThreshold,
                    laplacian_threshold=150.0, blur_logic="OR"):
        """
        Triage images in folder, flagging those that are blurry, too dark, or too light.

        Image quality metrics are fully delegated to ImageQualityAnalyzer.
        This method handles file I/O, progress reporting, alignment, CSV writing, and moving.

        Parameters
        ----------
        blurThreshhold      : float  FFT blur threshold (passed to ImageQualityAnalyzer as fft_blur_threshold)
        shiftSize           : int    FFT DC-mask radius (passed as fft_shift_radius)
        brightnessMin       : float  Minimum acceptable brightness
        brightnessMAX       : float  Maximum acceptable brightness
        laplacian_threshold : float  Laplacian variance threshold (default 150.0 — calibrated)
        blur_logic          : str    "OR" or "AND" — how FFT and Laplacian are combined
        """

        badImageCount    = 0
        rotationAngle    = 0.0
        horizontal_shift = 0.0
        vertical_shift   = 0.0

        extensions   = ('.jpg', '.jpeg', '.png')
        myGRIMe_Color = GRIME_AI_Color()

        # ── Image quality analyser ─────────────────────────────────────────────────────
        analyzer = ImageQualityAnalyzer(
            fft_shift_radius    = shiftSize,
            fft_blur_threshold  = blurThreshhold,
            laplacian_threshold = laplacian_threshold,
            blur_logic          = blur_logic,
            brightness_min      = brightnessMin,
            brightness_max      = brightnessMAX,
            use_contrast        = False,        # not used in triage
            use_exposure_clipping = False,      # not used in triage
            resize_percent      = 50.0,
        )

        # ── CSV report ────────────────────────────────────────────────────────────────
        if bCreateReport:
            csvFilename      = 'ImageTriage_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
            imageQualityFile = os.path.join(folder, csvFilename)
            csvFile          = open(imageQualityFile, 'a', newline='')
            csvFile.write('Focus Value, Focus Attrib, Laplacian Value, Intensity Value, Intensity Attrib., '
                          'Rotation, H. Shift, V. Shift, Filename, Moved\n')

        # ── Progress bar ──────────────────────────────────────────────────────────────
        imageCount = GRIME_AI_Utils().get_image_count(folder, extensions)
        if self.show_gui:
            progressBar = QProgressWheel(0, imageCount + 1)
            progressBar.show()

        imageIndex = 0

        # ── Optional output subfolders ────────────────────────────────────────────────
        if bSavePolylines:
            polyFolder = os.path.join(folder, "poly")
            os.makedirs(polyFolder, exist_ok=True)

        if bCorrectAlignment:
            warpFolder = os.path.join(folder, "warp")
            os.makedirs(warpFolder, exist_ok=True)

        # ── Reference image for alignment ─────────────────────────────────────────────
        refImage = None
        if bCorrectAlignment and len(strReferenceImageFilename) > 0:
            refImage = myGRIMe_Color.loadColorImage(strReferenceImageFilename)

        # ── Main loop ─────────────────────────────────────────────────────────────────
        file_count, files = GRIME_AI_Utils().getFileList(folder, extensions, bFetchRecursive)

        for file in files:
            # Check if user closed the progress bar — stop triage
            if self.show_gui and progressBar._is_closed:
                break

            if self.show_gui:
                progressBar.setWindowTitle(file)
                progressBar.setValue(imageIndex)
                progressBar.repaint()
            imageIndex += 1

            ext = os.path.splitext(file)[-1].lower()
            if ext not in extensions:
                continue

            filename    = os.path.join(folder, file)
            numpyImage  = myGRIMe_Color.loadColorImage(filename)

            # ── Quality analysis ──────────────────────────────────────────────────────
            quality = analyzer.analyze(numpyImage)

            blur_fft        = quality.get("blur_fft",       0.0)
            blur_laplacian  = quality.get("blur_laplacian",  0.0)
            brightness      = quality.get("brightness",      0.0)
            is_blurry       = quality.get("is_blurry",      False)
            is_too_dark     = quality.get("is_too_dark",    False)
            is_too_light    = quality.get("is_too_light",   False)

            # ── Decision ──────────────────────────────────────────────────────────────
            bMove           = False
            strFocusMetric  = "Nominal"
            strIntensity    = "Nominal"

            if is_blurry:
                strFocusMetric = "Blurry"
                bMove          = True

            if is_too_dark:
                strIntensity = "Too Dark"
                bMove        = True
            elif is_too_light:
                strIntensity = "Too Light"
                bMove        = True

            # ── Alignment correction ──────────────────────────────────────────────────
            if refImage is not None and bCorrectAlignment:
                rotationAngle, poly_img, warp_img = self.checkImageAlignment(refImage, numpyImage)

                baseFilename = os.path.basename(file)

                if bSavePolylines:
                    cv2.imwrite(os.path.join(polyFolder, baseFilename + "_poly.jpg"), poly_img)

                if bCorrectAlignment and rotationAngle > rotationThreshold:
                    cv2.imwrite(os.path.join(warpFolder, baseFilename + "_align.jpg"), warp_img)

                grayImage        = cv2.cvtColor(numpyImage, cv2.COLOR_RGB2GRAY)
                horizontal_shift, vertical_shift = self.checkImageShift(refImage, grayImage)

            # ── Move rejects ──────────────────────────────────────────────────────────
            if bMoveImages and bMove:
                filepath   = os.path.dirname(file)
                tempFolder = os.path.join(filepath, "MovedImages")
                os.makedirs(tempFolder, exist_ok=True)
                shutil.move(file, tempFolder)
                filename   = os.path.join(tempFolder, os.path.basename(file))

            if bMove:
                badImageCount += 1

            # ── CSV row ───────────────────────────────────────────────────────────────
            if bCreateReport:
                formula     = f'=HYPERLINK("{filename}", "{os.path.basename(filename)}")'
                if ',' in formula:
                    safe_formula = formula.replace('"', '""')
                    formula      = f'"{safe_formula}"'
                quotedHyperlink = f'"{formula}"'

                strOutputString = '%3.2f,%s,%3.2f,%3.2f,%s,%3.2f,%3.2f,%3.2f,%s,%s\n' % (
                    blur_fft, strFocusMetric,
                    blur_laplacian,
                    brightness, strIntensity,
                    rotationAngle, horizontal_shift, vertical_shift,
                    formula,
                    'Y' if bMove else 'N'
                )
                csvFile.write(strOutputString)

        # ── Summary ───────────────────────────────────────────────────────────────────
        if badImageCount == 0:
            strMessage = 'No bad images found.'
            print(strMessage)
            if self.show_gui:
                msgBox   = GRIME_AI_QMessageBox('Image Triage', strMessage)
                response = msgBox.displayMsgBox()

        # ── Cleanup ───────────────────────────────────────────────────────────────────
        if bCreateReport:
            csvFile.close()

        if self.show_gui:
            progressBar.close()
            del progressBar

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def checkImageShift(self, refImage, image):
        refImageGray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
        imageGray    = image

        orb                         = cv2.ORB_create()
        keypoints1, descriptors1    = orb.detectAndCompute(refImageGray, None)
        keypoints2, descriptors2    = orb.detectAndCompute(imageGray,    None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(descriptors1, descriptors2), key=lambda x: x.distance)

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, _             = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        horizontal_shift = M[0, 2]
        vertical_shift   = M[1, 2]

        return horizontal_shift, vertical_shift

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def checkImageAlignment(self, refImage, image):
        refImageGray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
        imageGray    = cv2.cvtColor(image,    cv2.COLOR_RGB2GRAY)

        orb_detector                = cv2.ORB_create(5000)
        keypoints1, descriptors1    = orb_detector.detectAndCompute(imageGray,    None)
        keypoints2, descriptors2    = orb_detector.detectAndCompute(refImageGray, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(descriptors1, descriptors2), key=lambda x: x.distance)
        matches = matches[:int(len(matches) * 0.9)]
        no_of_matches = len(matches)

        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))
        for i, m in enumerate(matches):
            p1[i, :] = keypoints1[m.queryIdx].pt
            p2[i, :] = keypoints2[m.trainIdx].pt

        homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC)

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _    = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

        width, height = image.shape[1], image.shape[0]
        swapImage     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        warp_img      = cv2.warpPerspective(swapImage, homography, (width, height))

        h, w  = refImage.shape[:2]
        pts   = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst   = cv2.perspectiveTransform(pts, M)
        img2  = cv2.polylines(image, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)
        poly_img = cv2.drawMatches(refImage, keypoints1, image, keypoints2, matches[:20], None, flags=2)

        return rotation_angle, poly_img, warp_img
