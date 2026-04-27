#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: April 2026
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# GRIME_AI_TriageCalibrateDlg.py
#
# Dialog for calibrating triage parameters from labelled image folders.
# Delegates all calibration logic to GRIME_AI_TriageCalibrator (no Qt there).
# Saves results to GRIME-AI.json and emits calibrated params to caller.

import os
import json

from PyQt5.QtWidgets import (QDialog, QWidget, QFileDialog, QApplication,
                              QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.uic import loadUi

from GRIME_AI.dialogs.triage.GRIME_AI_TriageCalibrator import GRIME_AI_TriageCalibrator, CalibrationResult
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils

BUTTON_CSS_STEEL_BLUE = 'QPushButton {background-color: steelblue; color: white;}'
GRIME_AI_CONFIG_FILENAME = "GRIME-AI.json"


# ──────────────────────────────────────────────────────────────────────────────
# Background worker thread — keeps UI responsive during calibration
# ──────────────────────────────────────────────────────────────────────────────

class CalibrationWorker(QThread):
    progress    = pyqtSignal(str, int)   # message, percent
    finished    = pyqtSignal(object)     # CalibrationResult

    def __init__(self, calibrator, good_folder, blurry_folder, exposure_folder):
        super().__init__()
        self.calibrator      = calibrator
        self.good_folder     = good_folder
        self.blurry_folder   = blurry_folder
        self.exposure_folder = exposure_folder

    def run(self):
        result = self.calibrator.calibrate(
            good_folder     = self.good_folder,
            blurry_folder   = self.blurry_folder,
            exposure_folder = self.exposure_folder,
            progress_callback = lambda msg, pct: self.progress.emit(msg, pct)
        )
        self.finished.emit(result)


# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────

class GRIME_AI_TriageCalibrateDlg(QDialog):
    """
    Calibration dialog.

    After successful calibration and user approval, emits calibration_applied
    with a CalibrationResult so the parent triage options dialog can update
    its spinboxes.
    """

    calibration_applied = pyqtSignal(object)   # CalibrationResult

    def __init__(self, parent=None):
        super().__init__(parent)

        ui_dir_name      = os.path.dirname(__file__)
        ui_file_absolute = os.path.join(ui_dir_name, 'QDialog_TriageCalibrate.ui')
        loadUi(ui_file_absolute, self)

        normal_font = QFont("Arial", 9)
        normal_font.setStyleHint(QFont.SansSerif)
        self.setFont(normal_font)
        for widget in self.findChildren(QWidget):
            widget.setFont(normal_font)

        self.adjustSize()

        self._calibration_result = None
        self._worker             = None

        self._setup_connections()
        self._apply_styles()

    # ──────────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_connections(self):
        self.pushButton_BrowseGood.clicked.connect(
            lambda: self._browse_folder(self.lineEdit_GoodFolder))
        self.pushButton_BrowseBlurry.clicked.connect(
            lambda: self._browse_folder(self.lineEdit_BlurryFolder))
        self.pushButton_BrowseExposure.clicked.connect(
            lambda: self._browse_folder(self.lineEdit_ExposureFolder))

        self.lineEdit_GoodFolder.textChanged.connect(self._update_run_button_state)
        self.lineEdit_BlurryFolder.textChanged.connect(self._update_run_button_state)
        self.lineEdit_ExposureFolder.textChanged.connect(self._update_run_button_state)

        self.pushButton_RunCalibration.clicked.connect(self._run_calibration)
        self.pushButton_ApplyAndClose.clicked.connect(self._apply_and_close)
        self.pushButton_Cancel.clicked.connect(self.reject)

    def _apply_styles(self):
        self.pushButton_BrowseGood.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_BrowseBlurry.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_BrowseExposure.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_RunCalibration.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_ApplyAndClose.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

    # ──────────────────────────────────────────────────────────────────────────
    # Folder browsing
    # ──────────────────────────────────────────────────────────────────────────

    def _browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            line_edit.setText(folder)

    # ──────────────────────────────────────────────────────────────────────────
    # Button state
    # ──────────────────────────────────────────────────────────────────────────

    def _update_run_button_state(self):
        good_set    = bool(self.lineEdit_GoodFolder.text().strip())
        blurry_set  = bool(self.lineEdit_BlurryFolder.text().strip())
        exposure_set= bool(self.lineEdit_ExposureFolder.text().strip())
        enabled = good_set and (blurry_set or exposure_set)
        self.pushButton_RunCalibration.setEnabled(enabled)

    # ──────────────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────────────

    def _run_calibration(self):
        self.pushButton_RunCalibration.setEnabled(False)
        self.pushButton_ApplyAndClose.setEnabled(False)
        self.progressBar.setValue(0)
        self.labelStatus.setText("Starting calibration...")
        self._clear_results()

        calibrator = GRIME_AI_TriageCalibrator(resize_percent=50.0)

        self._worker = CalibrationWorker(
            calibrator      = calibrator,
            good_folder     = self.lineEdit_GoodFolder.text().strip(),
            blurry_folder   = self.lineEdit_BlurryFolder.text().strip(),
            exposure_folder = self.lineEdit_ExposureFolder.text().strip(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_calibration_finished)
        self._worker.start()

    def _on_progress(self, message: str, percent: int):
        self.progressBar.setValue(percent)
        self.labelStatus.setText(message)
        QApplication.processEvents()

    def _on_calibration_finished(self, result: CalibrationResult):
        self._calibration_result = result
        self.pushButton_RunCalibration.setEnabled(True)

        if not result.success:
            self.labelStatus.setText(f"Error: {result.error_message}")
            QMessageBox.critical(self, "Calibration Failed", result.error_message)
            return

        # Populate result fields
        self.lineEdit_ResultFftThreshold.setText(f"{result.fft_blur_threshold:.2f}")
        self.lineEdit_ResultLaplacianThreshold.setText(f"{result.laplacian_threshold:.1f}")
        self.lineEdit_ResultFftRadius.setText(str(result.fft_shift_radius))
        self.lineEdit_ResultBrightnessMin.setText(f"{result.brightness_min:.1f}")
        self.lineEdit_ResultBrightnessMax.setText(f"{result.brightness_max:.1f}")

        summary = (f"Done. Images: {result.n_good} good, {result.n_blurry} blurry, "
                   f"{result.n_exposure} exposure.")
        if result.n_blurry > 0:
            summary += f"  Blur F1: {result.blur_f1:.3f}."
        self.labelStatus.setText(summary)

        self.pushButton_ApplyAndClose.setEnabled(True)

        # Save to GRIME-AI.json
        self._save_to_config(result)

    # ──────────────────────────────────────────────────────────────────────────
    # Apply and close
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_and_close(self):
        if self._calibration_result and self._calibration_result.success:
            self.calibration_applied.emit(self._calibration_result)
        self.accept()

    # ──────────────────────────────────────────────────────────────────────────
    # Config persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _save_to_config(self, result: CalibrationResult):
        try:
            settings_folder = os.path.normpath(GRIME_AI_Save_Utils().get_settings_folder())
            config_path     = os.path.join(settings_folder, GRIME_AI_CONFIG_FILENAME)

            config = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)

            config.setdefault("triage", {})
            config["triage"].update(result.to_dict())

            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

            print(f"[TriageCalibrateDlg] Saved calibration to {config_path}")

        except Exception as e:
            print(f"[TriageCalibrateDlg] Could not save calibration: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _clear_results(self):
        for widget in [self.lineEdit_ResultFftThreshold,
                       self.lineEdit_ResultLaplacianThreshold,
                       self.lineEdit_ResultFftRadius,
                       self.lineEdit_ResultBrightnessMin,
                       self.lineEdit_ResultBrightnessMax]:
            widget.clear()
