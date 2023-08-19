from urllib.request import urlopen

import cv2
import numpy as np

from GRIME_AI_Utils import GRIME_AI_Utils

class GRIME_AI_Color:
    def __init__(self):
        self.className = "GRIME_AI_Color"

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def loadColorImage(self, filename):
        img = cv2.imread(filename)

        try:
            origImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            origImg = []

        return origImg


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def segmentColors(self, rgb, hsv, roiList):
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


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def KMeans(self, img1, nClusters):
        # --------------------------------------------------------------------------------
        # reshape the image to be a list of pixels
        # --------------------------------------------------------------------------------
        width = int(img1.shape[1])
        height = int(img1.shape[0])
        channels = int(img1.shape[2])

        dim = (width, height)

        pixel_values = img1.reshape((-1, channels))

        pixel_values = np.float32(pixel_values)

        # --------------------------------------------------------------------------------
        # COLOR CLUSTERING
        # --------------------------------------------------------------------------------
        # kmeans = KMeans(n_clusters=nClusters, n_jobs=-1)

        # define criteria, number of clusters(K) and apply kmeans()
        # We are going to cluster with k = 2, because the image will have just two colours ,a white background and the colour of the patch
        attempts = 10
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_PP_CENTERS
        _, labels, (centers) = cv2.kmeans(pixel_values, nClusters, None, criteria, 10, flags)

        # convert back to 8 bit values
        centers = np.uint8(centers)

        # flatten the labels array
        labels = labels.flatten()
        hist = self.centroid_histogram(labels)

        # convert all pixels to the color of the centroids
        segmented_image = centers[labels.flatten()]

        # reshape back to the original image dimension
        segmented_image = segmented_image.reshape(img1.shape)

        bar = self.plot_colors(hist, centers)

        return bar, centers, hist


    # ======================================================================================================================
    # EXTRACT DOMINANT HSV COLORS
    # ======================================================================================================================
    def extractDominant_HSV(self, rgb, nNumClusters):

        if 0:
            rgb = cv2.blur(rgb, ksize=(5, 5))

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        _, hsvClusterCenters, hist = self.KMeans(hsv, nNumClusters)

        # KMeans QUANTIZES THE HUE VALUE TO 0..180 WHEN THE ACTUAL HSV COLOR SPACE HUE VALUE 0..360.
        # THEREFORE WE MULTIPLY THE KMeans HUE VALUE BY 2 TO STANDARDIZE ON THE ACTUAL COLOR SPACE HUE RANGE.
        hsvClusterCenters[:,0] = hsvClusterCenters[:,0] * 2.0

        if nNumClusters > 1 or len(hist) > nNumClusters:
            inds = hist.argsort()[::-1]
            sortedClusters = hsvClusterCenters[inds]
            sortedHist = hist[inds]

            sortedHist = sortedHist[0:nNumClusters]
            sortedClusters = sortedClusters[0:nNumClusters]
        else:
            sortedHist = hist
            sortedClusters = hsvClusterCenters

        return sortedHist, sortedClusters


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def plot_colors(self, hist, centroids):
        # initialize the bar chart representing the relative frequency of each of the colors
        # bar = np.zeros((50, 300, 3), dtype="uint8")
        bar = np.zeros((50, 400, 3), dtype="uint8")

        startX = 0

        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 100)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)

            startX = endX

        # return the bar chart
        return bar


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def centroid_histogram(self, labels_):
        # grab the number of different clusters and create a histogram based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(labels_)) + 1)
        (hist, _) = np.histogram(labels_, bins=numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist


