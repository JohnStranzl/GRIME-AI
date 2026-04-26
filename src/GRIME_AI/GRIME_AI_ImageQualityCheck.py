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
    """
    Centralised image quality analysis for GRIME AI triage and related workflows.

    All blur, brightness, contrast, and exposure metrics are computed here so that
    GRIME_AI_ImageTriage and any other callers do not contain duplicated algorithm code.

    Default thresholds calibrated against 966-image labelled dataset (S/F/M triplets)
    via grid-search optimisation:
        fft_shift_radius    = 40    (radius of DC-component mask)
        fft_blur_threshold  = 21.0  (FFT mean below this → blurry)
        laplacian_threshold = 150.0 (Laplacian variance below this → blurry)
    Combined OR logic: F1=0.921, focus-blur recall=99.7%, motion-blur recall=91.3%.
    """

    def __init__(
        self,
        # ── blur ──────────────────────────────────────────────────────────────
        use_fft_blur=True,
        use_laplacian_blur=True,
        fft_shift_radius=40,
        fft_blur_threshold=21.0,
        laplacian_threshold=150.0,
        blur_logic="AND",           # "AND" → blurry if both metrics fail
                                    # "OR"  → blurry if either metric fails
        # ── brightness ────────────────────────────────────────────────────────
        use_brightness=True,
        brightness_min=40.0,        # below this → Too Dark
        brightness_max=215.0,       # above this → Too Light
        # ── contrast ──────────────────────────────────────────────────────────
        use_contrast=True,
        contrast_threshold=30.0,
        # ── exposure clipping ─────────────────────────────────────────────────
        use_exposure_clipping=True,
        clip_percent=0.25,
        dark_clip=10,
        bright_clip=245,
        # ── preprocessing ─────────────────────────────────────────────────────
        resize_percent=50.0,
    ):
        self.use_fft_blur          = use_fft_blur
        self.use_laplacian_blur    = use_laplacian_blur
        self.fft_shift_radius      = fft_shift_radius
        self.fft_blur_threshold    = fft_blur_threshold
        self.laplacian_threshold   = laplacian_threshold
        self.blur_logic            = blur_logic.upper()

        self.use_brightness        = use_brightness
        self.brightness_min        = brightness_min
        self.brightness_max        = brightness_max

        self.use_contrast          = use_contrast
        self.contrast_threshold    = contrast_threshold

        self.use_exposure_clipping = use_exposure_clipping
        self.clip_percent          = clip_percent
        self.dark_clip             = dark_clip
        self.bright_clip           = bright_clip

        self.resize_percent        = resize_percent

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self, image) -> dict:
        """
        Analyse a single BGR/RGB numpy image.

        Returns a dict with raw metrics, boolean flags, and top-level summary:
            is_bad  (bool) – True if any enabled check fails
            reason  (str)  – comma-separated failing checks, or 'Nominal'

        Metric keys (present when corresponding check is enabled):
            blur_fft, is_blurry_fft
            blur_laplacian, is_blurry_laplacian
            is_blurry          (combined, respects blur_logic)
            brightness, is_too_dark, is_too_light
            contrast, is_low_contrast
            clip_fraction, is_clipped
        """
        gray    = self._preprocess(image)
        result  = {}
        reasons = []

        # ── blur ──────────────────────────────────────────────────────────────
        fft_blurry = False
        lap_blurry = False

        if self.use_fft_blur:
            blur_fft               = self._compute_blur_fft(gray)
            result["blur_fft"]     = blur_fft
            fft_blurry             = blur_fft < self.fft_blur_threshold
            result["is_blurry_fft"] = fft_blurry

        if self.use_laplacian_blur:
            lap_var                        = self._compute_laplacian_var(gray)
            result["blur_laplacian"]       = lap_var
            lap_blurry                     = lap_var < self.laplacian_threshold
            result["is_blurry_laplacian"]  = lap_blurry

        if self.use_fft_blur or self.use_laplacian_blur:
            is_blurry = (fft_blurry and lap_blurry) if self.blur_logic == "AND" else (fft_blurry or lap_blurry)
            result["is_blurry"] = is_blurry
            if is_blurry:
                reasons.append("Blurry")

        # ── brightness ────────────────────────────────────────────────────────
        if self.use_brightness:
            brightness              = self._compute_brightness(gray)
            result["brightness"]    = brightness
            too_dark                = brightness < self.brightness_min
            too_light               = brightness > self.brightness_max
            result["is_too_dark"]   = too_dark
            result["is_too_light"]  = too_light
            if too_dark:
                reasons.append("Too Dark")
            if too_light:
                reasons.append("Too Light")

        # ── contrast ──────────────────────────────────────────────────────────
        if self.use_contrast:
            contrast                    = self._compute_contrast(gray)
            result["contrast"]          = contrast
            low_contrast                = contrast < self.contrast_threshold
            result["is_low_contrast"]   = low_contrast
            if low_contrast:
                reasons.append("Low Contrast")

        # ── exposure clipping ─────────────────────────────────────────────────
        if self.use_exposure_clipping:
            clip_fraction           = self._compute_clip_fraction(gray)
            result["clip_fraction"] = clip_fraction
            clipped                 = clip_fraction > self.clip_percent
            result["is_clipped"]    = clipped
            if clipped:
                reasons.append("Exposure Clipped")

        result["is_bad"] = bool(reasons)
        result["reason"] = ", ".join(reasons) if reasons else "Nominal"
        return result

    def is_bad_image(self, image) -> tuple:
        """
        Convenience wrapper. Returns (is_bad: bool, reason: str).
        """
        r = self.analyze(image)
        return r["is_bad"], r["reason"]

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess(self, image) -> np.ndarray:
        """Convert to grayscale (supports BGR and RGB inputs) and optionally resize."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if self.resize_percent != 100.0:
            scale = self.resize_percent / 100.0
            gray  = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray

    def _compute_blur_fft(self, gray: np.ndarray) -> float:
        """
        FFT-based blur metric. Zero-out DC component, reconstruct, return mean
        log-magnitude of high-frequency residual. Low score → blurry.
        """
        h, w      = gray.shape
        cX, cY    = w // 2, h // 2
        fft_shift = np.fft.fftshift(np.fft.fft2(gray))
        r         = self.fft_shift_radius
        fft_shift[cY - r:cY + r, cX - r:cX + r] = 0
        recon     = np.fft.ifft2(np.fft.ifftshift(fft_shift))
        magnitude = 20 * np.log(np.abs(recon) + 1e-8)
        return float(np.mean(magnitude))

    def _compute_laplacian_var(self, gray: np.ndarray) -> float:
        """Laplacian variance. Low value → blurry."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _compute_brightness(self, gray: np.ndarray) -> float:
        """
        Mean pixel intensity after light Gaussian smoothing.
        Matches original GRIME_AI_ImageTriage brightness calculation.
        """
        smoothed = cv2.GaussianBlur(gray, (0, 0), 1)
        return float(cv2.mean(smoothed)[0])

    def _compute_contrast(self, gray: np.ndarray) -> float:
        """Pixel value range (max − min)."""
        return float(int(gray.max()) - int(gray.min()))

    def _compute_clip_fraction(self, gray: np.ndarray) -> float:
        """Fraction of pixels at or near black/white clipping limits."""
        clipped = np.sum((gray <= self.dark_clip) | (gray >= self.bright_clip))
        return float(clipped) / gray.size
