#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: April 2026
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# GRIME_AI_TriageCalibrator.py
#
# Pure calibration logic — no Qt dependency.
# Loads images from three labelled folders (good, blurry, exposure),
# computes ImageQualityAnalyzer metrics, and grid-searches for the
# optimal FFT, Laplacian, and brightness thresholds.
#
# Can be called from the GUI dialog or from a CLI script.

import os
import itertools
import numpy as np
import cv2
from pathlib import Path

from GRIME_AI.GRIME_AI_ImageQualityCheck import ImageQualityAnalyzer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass (plain dict-compatible)
# ──────────────────────────────────────────────────────────────────────────────

class CalibrationResult:
    """Holds the optimal threshold parameters found by calibration."""

    def __init__(self):
        self.fft_blur_threshold   = 21.0
        self.laplacian_threshold  = 150.0
        self.fft_shift_radius     = 40
        self.brightness_min       = 40.0
        self.brightness_max       = 215.0
        self.blur_f1              = 0.0
        self.brightness_accuracy  = 0.0
        self.n_good               = 0
        self.n_blurry             = 0
        self.n_exposure           = 0
        self.success              = False
        self.error_message        = ""

    def to_dict(self) -> dict:
        return {
            "fft_blur_threshold":  self.fft_blur_threshold,
            "laplacian_threshold": self.laplacian_threshold,
            "fft_shift_radius":    self.fft_shift_radius,
            "brightness_min":      self.brightness_min,
            "brightness_max":      self.brightness_max,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main calibrator class
# ──────────────────────────────────────────────────────────────────────────────

class GRIME_AI_TriageCalibrator:

    def __init__(self, resize_percent: float = 50.0):
        self.className      = "GRIME_AI_TriageCalibrator"
        self.resize_percent = resize_percent

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def calibrate(self,
                  good_folder:     str,
                  blurry_folder:   str,
                  exposure_folder: str,
                  progress_callback=None) -> CalibrationResult:
        """
        Run calibration against three labelled folders.

        progress_callback(message: str, percent: int) — optional callable for
        progress reporting back to a GUI or CLI.

        Returns a CalibrationResult instance.
        """
        result = CalibrationResult()

        # ── Load images ───────────────────────────────────────────────────────
        self._report(progress_callback, "Loading images...", 0)

        good_images     = self._load_images(good_folder)
        blurry_images   = self._load_images(blurry_folder)
        exposure_images = self._load_images(exposure_folder)

        result.n_good     = len(good_images)
        result.n_blurry   = len(blurry_images)
        result.n_exposure = len(exposure_images)

        if result.n_good == 0:
            result.error_message = "No images found in the Good folder."
            return result
        if result.n_blurry == 0 and result.n_exposure == 0:
            result.error_message = "No images found in either the Blurry or Exposure folders."
            return result

        # ── Compute metrics once per image ────────────────────────────────────
        self._report(progress_callback, "Computing image metrics...", 5)

        fft_radii = [20, 30, 40]

        good_metrics     = self._compute_all_metrics(good_images,     fft_radii, progress_callback, 5,  35)
        blurry_metrics   = self._compute_all_metrics(blurry_images,   fft_radii, progress_callback, 35, 60)
        exposure_metrics = self._compute_all_metrics(exposure_images, fft_radii, progress_callback, 60, 75)

        # ── Grid search — blur thresholds ─────────────────────────────────────
        self._report(progress_callback, "Searching blur thresholds...", 75)

        if result.n_blurry > 0:
            best_fft_thr, best_lap_thr, best_radius, best_f1 = self._search_blur_thresholds(
                good_metrics, blurry_metrics, fft_radii
            )
            result.fft_blur_threshold  = best_fft_thr
            result.laplacian_threshold = best_lap_thr
            result.fft_shift_radius    = best_radius
            result.blur_f1             = best_f1

        # ── Grid search — brightness thresholds ───────────────────────────────
        self._report(progress_callback, "Searching brightness thresholds...", 88)

        if result.n_exposure > 0:
            bmin, bmax, b_acc = self._search_brightness_thresholds(
                good_metrics, exposure_metrics
            )
            result.brightness_min      = bmin
            result.brightness_max      = bmax
            result.brightness_accuracy = b_acc

        result.success = True
        self._report(progress_callback, "Calibration complete.", 100)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Image loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_images(self, folder: str) -> list:
        """Return list of (path, numpy_image) for all images in folder."""
        images = []
        if not folder or not os.path.isdir(folder):
            return images
        for fname in sorted(os.listdir(folder)):
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                path = os.path.join(folder, fname)
                img  = cv2.imread(path)
                if img is not None:
                    images.append((path, img))
        return images

    # ──────────────────────────────────────────────────────────────────────────
    # Metric computation
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_all_metrics(self, images: list, fft_radii: list,
                              progress_callback, pct_start: int, pct_end: int) -> list:
        """
        For each image compute FFT (all radii) and Laplacian + brightness once.
        Returns list of dicts: {path, brightness, laplacian, fft: {radius: value}}
        Arrays are discarded immediately after metric extraction.
        """
        results = []
        n = len(images)
        if n == 0:
            return results

        analyzer = ImageQualityAnalyzer(
            use_fft_blur=False,
            use_laplacian_blur=False,
            use_brightness=False,
            use_contrast=False,
            use_exposure_clipping=False,
            resize_percent=self.resize_percent,
        )

        for i, (path, img) in enumerate(images):
            pct = pct_start + int((i / n) * (pct_end - pct_start))
            self._report(progress_callback, f"  {os.path.basename(path)}", pct)

            gray = analyzer._preprocess(img)

            # Brightness
            brightness = analyzer._compute_brightness(gray)

            # Laplacian
            laplacian = analyzer._compute_laplacian_var(gray)

            # FFT — compute shift once, apply all radii
            fft_shift = np.fft.fftshift(np.fft.fft2(gray))
            fft_scores = {}
            h, w = gray.shape
            cX, cY = w // 2, h // 2
            for r in fft_radii:
                fs = fft_shift.copy()
                fs[cY - r:cY + r, cX - r:cX + r] = 0
                recon = np.fft.ifft2(np.fft.ifftshift(fs))
                fft_scores[r] = float(np.mean(20 * np.log(np.abs(recon) + 1e-8)))

            del gray, fft_shift

            results.append({
                "path":       path,
                "brightness": brightness,
                "laplacian":  laplacian,
                "fft":        fft_scores,
            })

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Blur threshold grid search
    # ──────────────────────────────────────────────────────────────────────────

    def _search_blur_thresholds(self, good_metrics: list, blurry_metrics: list,
                                 fft_radii: list) -> tuple:
        """
        Grid search over FFT threshold, Laplacian threshold, and FFT radius.
        Uses OR logic: blurry if either FFT or Laplacian fails.
        Optimises for F1 score.
        Returns (best_fft_thr, best_lap_thr, best_radius, best_f1).
        """
        fft_thresholds = np.arange(5.0, 30.0, 1.0).tolist()
        lap_thresholds = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300]

        best_f1      = -1.0
        best_fft_thr = 21.0
        best_lap_thr = 150.0
        best_radius  = 40

        for radius, fft_thr, lap_thr in itertools.product(fft_radii, fft_thresholds, lap_thresholds):
            tp = fp = fn = tn = 0

            for m in good_metrics:
                predicted = (m["fft"][radius] < fft_thr) or (m["laplacian"] < lap_thr)
                if predicted: fp += 1
                else:         tn += 1

            for m in blurry_metrics:
                predicted = (m["fft"][radius] < fft_thr) or (m["laplacian"] < lap_thr)
                if predicted: tp += 1
                else:         fn += 1

            precision = tp / (tp + fp) if (tp + fp) else 0
            recall    = tp / (tp + fn) if (tp + fn) else 0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

            if f1 > best_f1:
                best_f1      = f1
                best_fft_thr = round(fft_thr, 2)
                best_lap_thr = float(lap_thr)
                best_radius  = radius

        return best_fft_thr, best_lap_thr, best_radius, best_f1

    # ──────────────────────────────────────────────────────────────────────────
    # Brightness threshold grid search
    # ──────────────────────────────────────────────────────────────────────────

    def _search_brightness_thresholds(self, good_metrics: list,
                                       exposure_metrics: list) -> tuple:
        """
        Find brightness_min and brightness_max that best separate good from
        exposure-bad images. Optimises for overall accuracy.
        Returns (brightness_min, brightness_max, accuracy).
        """
        bmin_candidates = list(range(10, 100, 5))
        bmax_candidates = list(range(150, 250, 5))

        best_acc  = -1.0
        best_bmin = 40.0
        best_bmax = 215.0

        good_brightness     = [m["brightness"] for m in good_metrics]
        exposure_brightness = [m["brightness"] for m in exposure_metrics]

        for bmin, bmax in itertools.product(bmin_candidates, bmax_candidates):
            if bmin >= bmax:
                continue

            correct = 0
            total   = len(good_brightness) + len(exposure_brightness)

            for b in good_brightness:
                if bmin <= b <= bmax:
                    correct += 1

            for b in exposure_brightness:
                if b < bmin or b > bmax:
                    correct += 1

            acc = correct / total if total else 0

            if acc > best_acc:
                best_acc  = acc
                best_bmin = float(bmin)
                best_bmax = float(bmax)

        return best_bmin, best_bmax, best_acc

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _report(callback, message: str, percent: int):
        if callback is not None:
            try:
                callback(message, percent)
            except Exception:
                pass
        else:
            print(f"[{percent:3d}%] {message}")
