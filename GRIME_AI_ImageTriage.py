import os
import shutil

import cv2
import numpy as np

from GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI_Color import GRIME_AI_Color
from GRIME_QProgressWheel import QProgressWheel
from datetime import datetime

class GRIME_AI_ImageTriage():

    def __init__(self):
        self.className = "GRIME_AI_ImageTriage"

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
                    bMoveImages):
        extensions = ('.jpg', '.jpeg', '.png')

        myGRIMe_Color = GRIME_AI_Color()

        if bCreateReport:
            csvFilename = 'ImageTriage_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
            imageQualityFile = os.path.join(folder, csvFilename)
            csvFile = open(imageQualityFile, 'a', newline='')
            csvFile.write('Focus Value, Focus Attrib, Intensity Value, Intensity Attrib., Rotation, H. Shift, V. Shift, Filename\n')

        # count the number of images that will potentially be processed and possibly saved with the specified extension
        # to display an "hourglass" to give an indication as to how long the process will take. Furthermore, the number
        # of images will help determine whether or not there is enough disk space to accomodate storing the images.
        imageCount = GRIME_AI_Utils().getImageCount(folder, extensions)

        progressBar = QProgressWheel(0, imageCount + 1)
        progressBar.show()

        imageIndex = 0

        # process images to determine which ones are too dark/too light, blurry/clear, etc and move them into a subfolder
        # created so they are not processed with nominal images.
        files = GRIME_AI_Utils().getFileList(folder, extensions, bFetchRecursive)

        refImage = myGRIMe_Color.loadColorImage(files[0])

        for file in files:
            progressBar.setWindowTitle(file)
            progressBar.setValue(imageIndex)
            progressBar.repaint()
            imageIndex += 1

            ext = os.path.splitext(file)[-1].lower()

            if ext in extensions:
                filename = os.path.join(folder, file)
                numpyImage = myGRIMe_Color.loadColorImage(filename)
                grayImage = cv2.cvtColor(numpyImage, cv2.COLOR_RGB2GRAY)

                # DECIMATE IMAGE TO SPEED UP PROCESSING
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

                # DECISION LOGIC
                bMove = False
                strFFTFocusMetric = 'Nominal'
                strFocusMetric = 'N/A'
                strIntensity = 'Nominal'
                #os.makedirs(tempFolder)

                # CHECK MEAN AGAINST THRESHHOLD TO DETERMINE IF THE IMAGE IS BLURRY/FOGGY/OUT-OF-FOCUS/ETC.
                if mean <= blurThreshhold:
                    strFFTFocusMetric = "Blurry"
                    bMove = True

                # CHECK TO SEE IF THE OVERALL IMAGE IS TOO DARK OR TOO BRIGHT
                if intensity < brightnessMin:
                    strIntensity = "Too Dark"
                    bMove = True
                elif intensity > brightnessMAX:
                    strIntensity = "Too Light"
                    bMove = True

                # GET THE ROTATION ANGLE OF THE IMAGE
                rotationAngle, res = self.checkImageAlignment(refImage, grayImage)
                cv2.imwrite(filename+".tmp.jpg", res)

                horizontal_shift, vertical_shift = self.checkImageShift(refImage, grayImage)

                # MOVE THE IMAGE REJECTS TO A SUBFOLDER IF THE USER CHOOSE THIS OPTION
                if bMoveImages and bMove:
                    # create a subfolder beneath the current root folder if the option to move less than nominal images is selected
                    filename = os.path.basename(file)
                    filepath = os.path.dirname(file)
                    tempFolder = os.path.join(filepath, "MovedImages")
                    if not os.path.exists(tempFolder):
                        pass

                    shutil.move(file, tempFolder)

                    filename = os.path.join(tempFolder, filename)

                # CREATE A CSV FILE THAT CONTAINS THE FOCUS AND INTENSITY METRICS ALONG WITH HYPERLINKS TO THE IMAGES
                if bCreateReport:
                    strHyperlink = '=HYPERLINK(' + '"' + filename + '"' + ')'
                    strOutputString = '%3.2f,%s,%3.2f,%s,%3.2f,%3.2f,%3.2f,%s\n' % (
                        mean, strFFTFocusMetric, intensity, strIntensity, rotationAngle, horizontal_shift, vertical_shift, strHyperlink)
                    csvFile.write(strOutputString)

                # FREE UP MEMORY FOR THE NEXT IMAGE TO BE PROCESSSED
                del fftShift
                del fft
                del recon
                del blur

        # clean-up before exiting function
        # 1. close and delete the progress bar
        # 2. close the EXIF log file, if opened
        if bCreateReport:
            csvFile.close()
        progressBar.close()
        del progressBar

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def checkImageShift(self, refImage, image):
            # Convert images to grayscale
            gray1 = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
            gray2 = image

            # Compute the keypoints and descriptors
            orb = cv2.ORB_create()
            keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
            keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

            # Match descriptors between the two images
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(descriptors1, descriptors2)

            # Sort matches by distance
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
        #imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageGray = image
        height, width = imageGray.shape

        #cv2.imshow("", refImageGray)
        #cv2.imshow("", imageGray)

        # Create ORB detector with 5000 features.
        orb_detector = cv2.ORB_create(5000)

        # Find keypoints and descriptors.
        # The first arg is the image, second arg is the mask
        #  (which is not required in this case).
        kp1, d1 = orb_detector.detectAndCompute(refImageGray, None)
        kp2, d2 = orb_detector.detectAndCompute(imageGray, None)

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

        # Compute rotation angle based on matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

        if 0:
            # Use this matrix to transform the colored image wrt the reference image.
            transformed_img = cv2.warpPerspective(imageGray, homography, (width, height))

            # Save the output.
            cv2.imwrite('output.jpg', transformed_img)

        if 1:
            h, w = refImage.shape[:2]

            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            ## draw found regions
            img2 = cv2.polylines(image, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)
            #cv2.imshow("found", image)

            ## draw match lines
            res = cv2.drawMatches(refImage, kp1, image, kp2, matches[:20], None, flags=2)

            #cv2.imshow("orb_match", res);
            #cv2.waitKey();
            #cv2.destroyAllWindows()

        return (rotation_angle, res)

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


