#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import cv2
import numpy as np

class ImageQualityAnalyzer:
    def __init__(
        self,
        use_fft_blur=True,
        use_laplacian_blur=True,
        use_contrast=True,
        use_brightness=True,
        use_exposure_clipping=True,
        fft_shift_radius=30,
        fft_blur_threshold=10.0,
        laplacian_threshold=100.0,
        contrast_threshold=30.0,
        brightness_threshold=40.0,
        clip_percent=0.25,
        dark_clip=10,
        bright_clip=245,
        resize_percent=50.0
    ):
        self.use_fft_blur = use_fft_blur
        self.use_laplacian_blur = use_laplacian_blur
        self.use_contrast = use_contrast
        self.use_brightness = use_brightness
        self.use_exposure_clipping = use_exposure_clipping

        self.fft_shift_radius = fft_shift_radius
        self.fft_blur_threshold = fft_blur_threshold
        self.laplacian_threshold = laplacian_threshold
        self.contrast_threshold = contrast_threshold
        self.brightness_threshold = brightness_threshold
        self.clip_percent = clip_percent
        self.dark_clip = dark_clip
        self.bright_clip = bright_clip
        self.resize_percent = resize_percent

    def analyze(self, image) -> dict:
        gray = self._preprocess(image)
        result = {}

        if self.use_fft_blur:
            blur_fft = self._compute_blur_fft(gray)
            result["blur_fft"] = blur_fft
            result["is_blurry_fft"] = blur_fft < self.fft_blur_threshold

        if self.use_laplacian_blur:
            laplacian_var = self._compute_laplacian_var(gray)
            result["blur_laplacian"] = laplacian_var
            result["is_blurry_laplacian"] = laplacian_var < self.laplacian_threshold

        if self.use_contrast:
            contrast = self._compute_contrast(gray)
            result["contrast"] = contrast
            result["is_low_contrast"] = contrast < self.contrast_threshold

        if self.use_brightness:
            brightness = self._compute_brightness(gray)
            result["brightness"] = brightness
            result["is_too_dark"] = brightness < self.brightness_threshold

        if self.use_exposure_clipping:
            clip_fraction = self._compute_clip_fraction(gray)
            result["clip_fraction"] = clip_fraction
            result["is_clipped"] = clip_fraction > self.clip_percent

        return result

    def _preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.resize_percent != 100.0:
            scale = self.resize_percent / 100.0
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray

    def _compute_blur_fft(self, gray):
        h, w = gray.shape
        cX, cY = w // 2, h // 2
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        r = self.fft_shift_radius
        fft_shift[cY - r:cY + r, cX - r:cX + r] = 0
        recon = np.fft.ifft2(np.fft.ifftshift(fft_shift))
        magnitude = 20 * np.log(np.abs(recon) + 1e-8)
        return np.mean(magnitude)

    def _compute_laplacian_var(self, gray):
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _compute_contrast(self, gray):
        return float(np.max(gray) - np.min(gray))

    def _compute_brightness(self, gray):
        smoothed = cv2.GaussianBlur(gray, (0, 0), 1)
        return float(np.mean(smoothed))

    def _compute_clip_fraction(self, gray):
        clipped = np.sum((gray <= self.dark_clip) | (gray >= self.bright_clip))
        return clipped / gray.size
