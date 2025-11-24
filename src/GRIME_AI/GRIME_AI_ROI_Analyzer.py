#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import sys
from io import BytesIO

import cv2
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PyQt5.QtGui import QImage, QPixmap

# ======================================================================================================================
# ======================================================================================================================
#  =====     =====     =====     =====     class GRIME_AI_ROI_Analyzer     =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class GRIME_AI_ROI_Analyzer:
    def __init__(self, image_filename, mask_filename, clusters=3):
        self.image_filename = image_filename
        self.mask_filename = mask_filename
        self.clusters = clusters

        self.image = None
        self.mask = None
        self.mask_bin = None
        self.composite = None

        self.roi_intensity = None
        self.roi_entropy = None
        self.roi_texture = None
        self.mean_gli = None
        self.mean_gcc = None
        self.ROI_total_pixels = None
        self.ROI_total_area = None

        self.dominant_hsv_list = []
        self.dominant_rgb_list = []
        self.percentages_list = []


    def generate_file_pairs(self, folder):
        """
        Scan the folder for paired files. Each pair consists of an original image (JPG)
        and its corresponding mask (PNG) with the same base filename, where the mask filename
        ends with '_mask' before the extension. Files with '_overlay' in their name are ignored.

        Parameters:
            folder (str): The folder in which to look for files.

        Returns:
            pairs (list of tuples): A list of tuples (original_file_path, mask_file_path).
        """
        # Get a list of all files in the folder.
        files = os.listdir(folder)
        # Filter out files that contain '_overlay' (case-insensitive).
        files = [f for f in files if '_overlay' not in f.lower()]

        originals = {}  # key: base filename (without extension), value: filename of original image
        masks = {}  # key: base filename corresponding to the original, value: filename of mask image

        for f in files:
            base, ext = os.path.splitext(f)
            ext = ext.lower()
            if ext == ".jpg":
                # Ensure this file does NOT already end with '_mask'.
                if not base.lower().endswith("_mask"):
                    originals[base] = f
            elif ext == ".png":
                # Consider only files that end with '_mask'.
                if base.lower().endswith("_mask"):
                    # Remove the '_mask' suffix to obtain the corresponding original's base name.
                    base_original = base[:-5]  # removes exactly "_mask"
                    masks[base_original] = f

        # Construct the list of paired file paths.
        pairs = []
        for base, orig_file in originals.items():
            if base in masks:
                orig_full = os.path.join(folder, orig_file)
                mask_full = os.path.join(folder, masks[base])
                pairs.append((orig_full, mask_full))

        return pairs


    def load_data(self):
        if not os.path.exists(self.image_filename):
            print(f"Error: Image file not found: {self.image_filename}")
            sys.exit(1)
        if not os.path.exists(self.mask_filename):
            print(f"Error: Mask file not found: {self.mask_filename}")
            sys.exit(1)
        self.image = cv2.imread(self.image_filename)
        if self.image is None:
            print(f"Error: Could not load image from: {self.image_filename}")
            sys.exit(1)
        self.mask = cv2.imread(self.mask_filename, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            print(f"Error: Could not load mask from: {self.mask_filename}")
            sys.exit(1)

        # Resize mask if necessary.
        if self.image.shape[:2] != self.mask.shape:
            self.mask = cv2.resize(self.mask, (self.image.shape[1], self.image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Create binary mask.
        ret, self.mask_bin = cv2.threshold(self.mask, 1, 255, cv2.THRESH_BINARY)


    def compute_composite(self):
        # Convert image to BGRA.
        image_bgra = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
        # Set alpha channel using the mask: 255 inside ROI; 0 outside.
        image_bgra[:, :, 3] = self.mask_bin

        # Create a white background.
        white_bg = np.ones_like(self.image, dtype=np.uint8) * 255
        white_bg = cv2.cvtColor(white_bg, cv2.COLOR_BGR2BGRA)

        # Composite the image over the white background using the alpha mask.
        alpha = self.mask_bin.astype(np.float32) / 255.0
        alpha_4 = np.dstack([alpha, alpha, alpha, alpha])
        self.composite = (image_bgra.astype(np.float32) * alpha_4 +
                          white_bg.astype(np.float32) * (1 - alpha_4)).astype(np.uint8)


    @staticmethod
    def calculate_shannon_entropy(gray_roi):
        hist, _ = np.histogram(gray_roi, bins=256, range=(0, 256))
        hist_sum = np.sum(hist)
        if hist_sum == 0:
            return 0.0
        probs = hist.astype(np.float32) / hist_sum
        entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
        return entropy


    def analyze_roi(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        roi_pixels = gray_image[self.mask_bin > 0]
        self.roi_intensity = np.mean(roi_pixels)
        self.roi_entropy = self.calculate_shannon_entropy(roi_pixels)

        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        roi_grad = grad_magnitude[self.mask_bin > 0]
        self.roi_texture = np.mean(roi_grad)


    def compute_mask_area(self):
        """
        Calculate the total number of pixels in the masked region and
        store both the pixel count and area in class variables.

        Also stores:
            self.image_width, self.image_height  - dimensions of the image
            self.image_total_pixels              - total number of pixels in the image
            self.ROI_percentage                  - ROI pixel count as % of total image pixels
        """
        if self.mask_bin is None:
            raise ValueError("Mask binary (self.mask_bin) is not initialized. Run load_data() first.")

        # Get image dimensions from the mask
        self.image_height, self.image_width = self.mask_bin.shape[:2]
        self.image_total_pixels = self.image_width * self.image_height

        # Count nonzero pixels in the binary mask
        self.ROI_total_pixels = int(np.count_nonzero(self.mask_bin))

        # Area in pixels² (same as pixel count unless scaling is applied)
        self.ROI_total_area = float(self.ROI_total_pixels)

        # Percentage of ROI pixels relative to total image pixels
        self.ROI_percentage = (self.ROI_total_pixels / self.image_total_pixels) * 100.0


    def compute_greenness(self):
        roi_pixels_color = self.image[self.mask_bin > 0].astype(np.float32)
        B = roi_pixels_color[:, 0]
        G = roi_pixels_color[:, 1]
        R = roi_pixels_color[:, 2]
        eps = 1e-6
        gli_values = (2 * G - R - B) / (2 * G + R + B + eps)
        self.mean_gli = np.mean(gli_values)
        gcc_values = G / (R + G + B + eps)
        self.mean_gcc = np.mean(gcc_values)


    def extract_dominant_colors(self):
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        if len(self.mask.shape) != 2:
            mask_gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = self.mask.copy()
        _, mask_bin_local = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        masked_pixels = hsv_image[mask_bin_local > 0]
        if masked_pixels.size == 0:
            print("Error: No pixels found under the provided mask.")
            sys.exit(1)
        clusters = self.clusters if masked_pixels.shape[0] >= self.clusters else 1
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        kmeans.fit(masked_pixels)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        counts = np.bincount(labels)
        percentages = counts / float(np.sum(counts)) * 100
        sorted_indices = np.argsort(percentages)[::-1]

        self.dominant_hsv_list = []
        self.dominant_rgb_list = []
        self.percentages_list = []
        for idx in sorted_indices:
            center = cluster_centers[idx]
            center_uint8 = np.clip(center, 0, 255).astype(np.uint8)
            center_hsv = np.array([[center_uint8]])
            center_rgb = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2RGB)[0][0]
            self.dominant_hsv_list.append(tuple(int(x) for x in center_uint8))
            self.dominant_rgb_list.append(tuple(int(x) for x in center_rgb))
            self.percentages_list.append(percentages[idx])


    def get_results_pixmap(self):
        # 1) Prepare images
        orig_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        composite_rgba = cv2.cvtColor(self.composite, cv2.COLOR_BGRA2RGBA)

        # 2) Try to load an overlay image if it exists
        base, _ = os.path.splitext(self.image_filename)
        overlay_rgb = None
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = f"{base}_overlay{ext}"
            if os.path.exists(candidate):
                ov = cv2.imread(candidate)
                overlay_rgb = cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)
                break

        # 3) Swatch sizing
        swatch_count = len(self.dominant_rgb_list)
        sw = int(100 * 0.75)  # 75×75 px

        # 4) Decide number of columns: 3 if overlay, else 2, but at least swatch_count
        ncols = 3 if overlay_rgb is not None else 2
        ncols = max(ncols, swatch_count)

        # 5) Build figure & GridSpec
        #    – smaller canvas: width = 3″ per col, height = 4″ total
        #    – row0 = 2.5× row1; wspace = 0.3
        fig = plt.figure(figsize=(3 * ncols, 4))
        gs = fig.add_gridspec(
            2, ncols,
            height_ratios=[2.5, 1],
            wspace=0.3
        )

        # --- Row 0: Original, ROI composite, (optional) overlay ---
        axes = []

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(orig_rgb)
        ax0.set_title("Original")
        axes.append(ax0)

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(composite_rgba)
        ax1.set_title(
            f"ROI\n"
            f"I:{self.roi_intensity:.1f} "
            f"E:{self.roi_entropy:.1f}\n"
            f"T:{self.roi_texture:.1f} "
            f"GLI:{self.mean_gli:.2f} "
            f"GCC:{self.mean_gcc:.2f}"
        )
        axes.append(ax1)

        if overlay_rgb is not None:
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.imshow(overlay_rgb)
            ax2.set_title("Overlay")
            axes.append(ax2)

        # 6) Draw square frames around each top‐row image
        for ax in axes:
            ax.axis('off')  # hide ticks/labels
            rect = Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                fill=False,
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(rect)

        # --- Row 1: Dominant-color swatches ---
        for idx, rgb in enumerate(self.dominant_rgb_list):
            ax = fig.add_subplot(gs[1, idx])
            swatch = np.zeros((sw, sw, 3), dtype=np.uint8)
            swatch[:] = rgb
            ax.imshow(swatch)
            ax.set_title(f"{rgb}\n{self.percentages_list[idx]:.1f}%")
            ax.axis('off')

        # 7) Tighten vertical spacing manually (no tight_layout)
        fig.subplots_adjust(
            top=0.98,  # pull top edge in
            bottom=0.02,  # pull bottom edge in
            hspace=0.02,  # only 2% of axes height between rows
            wspace=0.3  # keep horizontal gap
        )

        # 8) Render to QPixmap and return
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)

        qimg = QImage()
        qimg.loadFromData(buf.getvalue(), 'PNG')
        return QPixmap.fromImage(qimg)


    def run_analysis(self):
        self.load_data()
        self.compute_composite()
        self.analyze_roi()
        self.compute_greenness()
        self.compute_mask_area()
        self.extract_dominant_colors()
        # You can choose to use the traditional display_results() method here,
        # but for integration with the UI we'll use get_results_pixmap().
