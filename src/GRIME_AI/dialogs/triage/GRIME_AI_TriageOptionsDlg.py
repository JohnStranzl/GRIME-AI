#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import json

from PyQt5.QtWidgets import QDialog, QWidget, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIntValidator, QDoubleValidator
from PyQt5.uic import loadUi

from GRIME_AI.dialogs.triage.GRIME_AI_TriageCalibrateDlg import GRIME_AI_TriageCalibrateDlg

BUTTON_CSS_STEEL_BLUE = 'QPushButton {background-color: steelblue; color: white;}'
GRIME_AI_CONFIG_FILENAME = "GRIME-AI.json"

# Slider ranges — direct 1:1 mapping to actual values
SLIDER_RANGES = {
    "lap":  (0,   500, 150),   # min, max, default
    "fft":  (0,    30,  21),
    "uaf":  (1,   100,  40),
    "bmin": (0,   255,  40),
    "bmax": (0,   255, 215),
}


class GRIME_AI_TriageOptionsDlg(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        ui_dir_name = os.path.dirname(__file__)
        ui_file_absolute = os.path.join(ui_dir_name, 'QDialog_TriageOptions.ui')
        loadUi(ui_file_absolute, self)

        normal_font = QFont("Arial", 9)
        normal_font.setStyleHint(QFont.SansSerif)
        self.setFont(normal_font)
        for widget in self.findChildren(QWidget):
            widget.setFont(normal_font)

        self.adjustSize()
        self.referenceImageFilename = ''

        self._setup_validators()
        self._setup_connections()
        self._apply_styles()
        self._sync_all_displays()
        self._load_calibration_from_config()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _setup_validators(self):
        """Restrict editable value fields to their valid numeric ranges."""
        self.lineEdit_lap_value.setValidator(QIntValidator(0, 500))
        self.lineEdit_fft_value.setValidator(QDoubleValidator(0.0, 30.0, 1))
        self.lineEdit_uaf_value.setValidator(QIntValidator(1, 100))
        self.lineEdit_bmin_value.setValidator(QIntValidator(0, 255))
        self.lineEdit_bmax_value.setValidator(QIntValidator(0, 255))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _setup_connections(self):
        self.pushButton_SelectReferenceImage.clicked.connect(self.selectReferenceImage)
        self.pushButton_Calibrate.clicked.connect(self._open_calibration_dialog)

        # Checkbox enable/disable
        self.checkBox_UseFftBlur.toggled.connect(self._on_fft_toggled)
        self.checkBox_UseLaplacian.toggled.connect(self._on_laplacian_toggled)

        # Slider → value display
        self.sliderLaplacian.valueChanged.connect(self._on_lap_slider_changed)
        self.sliderFftBlur.valueChanged.connect(self._on_fft_slider_changed)
        self.sliderUniformArea.valueChanged.connect(self._on_uaf_slider_changed)
        self.sliderBrightnessMin.valueChanged.connect(self._on_bmin_slider_changed)
        self.sliderBrightnessMax.valueChanged.connect(self._on_bmax_slider_changed)

        # Value field → slider (user types a value)
        self.lineEdit_lap_value.editingFinished.connect(self._on_lap_edit_finished)
        self.lineEdit_fft_value.editingFinished.connect(self._on_fft_edit_finished)
        self.lineEdit_uaf_value.editingFinished.connect(self._on_uaf_edit_finished)
        self.lineEdit_bmin_value.editingFinished.connect(self._on_bmin_edit_finished)
        self.lineEdit_bmax_value.editingFinished.connect(self._on_bmax_edit_finished)

        # Set initial enable states
        self._on_fft_toggled(self.checkBox_UseFftBlur.isChecked())
        self._on_laplacian_toggled(self.checkBox_UseLaplacian.isChecked())

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _apply_styles(self):
        self.pushButton_SelectReferenceImage.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_Calibrate.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _sync_all_displays(self):
        """Sync all value displays with current slider positions."""
        self._on_lap_slider_changed(self.sliderLaplacian.value())
        self._on_fft_slider_changed(self.sliderFftBlur.value())
        self._on_uaf_slider_changed(self.sliderUniformArea.value())
        self._on_bmin_slider_changed(self.sliderBrightnessMin.value())
        self._on_bmax_slider_changed(self.sliderBrightnessMax.value())

    # ------------------------------------------------------------------------------------------------------------------
    # Enable/disable
    # ------------------------------------------------------------------------------------------------------------------
    def _on_fft_toggled(self, checked):
        for w in [self.sliderFftBlur, self.sliderUniformArea,
                  self.labelFftLeft, self.labelFftMiddle, self.labelFftRight,
                  self.labelUniformLeft, self.labelUniformMiddle, self.labelUniformRight,
                  self.lineEdit_fft_value, self.lineEdit_uaf_value,
                  self.doubleSpinBoxBlurThreshhold, self.spinBoxShiftSize]:
            w.setEnabled(checked)

    def _on_laplacian_toggled(self, checked):
        for w in [self.sliderLaplacian,
                  self.labelLapLeft, self.labelLapMiddle, self.labelLapRight,
                  self.lineEdit_lap_value,
                  self.doubleSpinBoxLaplacianThreshold]:
            w.setEnabled(checked)

    # ------------------------------------------------------------------------------------------------------------------
    # Slider → display (slider drives the value field and hidden spinbox)
    # ------------------------------------------------------------------------------------------------------------------
    def _on_lap_slider_changed(self, val):
        self.lineEdit_lap_value.setText(str(val))
        self.doubleSpinBoxLaplacianThreshold.setValue(float(val))

    def _on_fft_slider_changed(self, val):
        self.lineEdit_fft_value.setText(str(val))
        self.doubleSpinBoxBlurThreshhold.setValue(float(val))

    def _on_uaf_slider_changed(self, val):
        self.lineEdit_uaf_value.setText(str(val))
        self.spinBoxShiftSize.setValue(val)

    def _on_bmin_slider_changed(self, val):
        self.lineEdit_bmin_value.setText(str(val))
        self.doubleSpinBoxBrightnessMin.setValue(float(val))

    def _on_bmax_slider_changed(self, val):
        self.lineEdit_bmax_value.setText(str(val))
        self.doubleSpinBoxBrightnessMax.setValue(float(val))

    # ------------------------------------------------------------------------------------------------------------------
    # Value field → slider (user types a value, slider follows)
    # ------------------------------------------------------------------------------------------------------------------
    def _clamp(self, val, min_val, max_val):
        return max(min_val, min(max_val, val))

    def _on_lap_edit_finished(self):
        try:
            val = self._clamp(int(self.lineEdit_lap_value.text()), 0, 500)
            self.sliderLaplacian.setValue(val)
        except ValueError:
            self._on_lap_slider_changed(self.sliderLaplacian.value())

    def _on_fft_edit_finished(self):
        try:
            val = self._clamp(int(round(float(self.lineEdit_fft_value.text()))), 0, 30)
            self.sliderFftBlur.setValue(val)
        except ValueError:
            self._on_fft_slider_changed(self.sliderFftBlur.value())

    def _on_uaf_edit_finished(self):
        try:
            val = self._clamp(int(self.lineEdit_uaf_value.text()), 1, 100)
            self.sliderUniformArea.setValue(val)
        except ValueError:
            self._on_uaf_slider_changed(self.sliderUniformArea.value())

    def _on_bmin_edit_finished(self):
        try:
            val = self._clamp(int(self.lineEdit_bmin_value.text()), 0, 255)
            self.sliderBrightnessMin.setValue(val)
        except ValueError:
            self._on_bmin_slider_changed(self.sliderBrightnessMin.value())

    def _on_bmax_edit_finished(self):
        try:
            val = self._clamp(int(self.lineEdit_bmax_value.text()), 0, 255)
            self.sliderBrightnessMax.setValue(val)
        except ValueError:
            self._on_bmax_slider_changed(self.sliderBrightnessMax.value())

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def selectReferenceImage(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff)"
        )
        if filename:
            self.referenceImageFilename = filename
            self.lineEdit_ReferenceImagePath.setText(filename)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _open_calibration_dialog(self):
        dlg = GRIME_AI_TriageCalibrateDlg(parent=self)
        dlg.calibration_applied.connect(self._apply_calibration)
        dlg.exec_()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _apply_calibration(self, result):
        """Receive CalibrationResult and update sliders directly."""
        self.sliderFftBlur.setValue(int(round(result.fft_blur_threshold)))
        self.sliderUniformArea.setValue(int(round(result.fft_shift_radius)))
        self.sliderLaplacian.setValue(int(round(result.laplacian_threshold)))
        self.sliderBrightnessMin.setValue(int(round(result.brightness_min)))
        self.sliderBrightnessMax.setValue(int(round(result.brightness_max)))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _load_calibration_from_config(self):
        """Load previously saved calibration from GRIME-AI.json if present."""
        try:
            from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
            settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
            config_path     = os.path.join(settings_folder, GRIME_AI_CONFIG_FILENAME)
            if not os.path.exists(config_path):
                return
            with open(config_path, "r") as f:
                config = json.load(f)
            triage = config.get("triage", {})
            if not triage:
                return
            if "fft_blur_threshold"  in triage:
                self.sliderFftBlur.setValue(int(round(triage["fft_blur_threshold"])))
            if "fft_shift_radius"    in triage:
                self.sliderUniformArea.setValue(int(round(triage["fft_shift_radius"])))
            if "laplacian_threshold" in triage:
                self.sliderLaplacian.setValue(int(round(triage["laplacian_threshold"])))
            if "brightness_min"      in triage:
                self.sliderBrightnessMin.setValue(int(round(triage["brightness_min"])))
            if "brightness_max"      in triage:
                self.sliderBrightnessMax.setValue(int(round(triage["brightness_max"])))
        except Exception as e:
            print(f"[TriageOptionsDlg] Could not load calibration: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------
    def getCreateReport(self):
        return self.checkBoxTriageImagesReport.isChecked()

    def getMoveImages(self):
        return self.checkBoxTriageImagesMove.isChecked()

    def getUseFftBlur(self):
        return self.checkBox_UseFftBlur.isChecked()

    def getUseLaplacian(self):
        return self.checkBox_UseLaplacian.isChecked()

    def getBlurThreshold(self):
        return self.doubleSpinBoxBlurThreshhold.value()

    def getLaplacianThreshold(self):
        return self.doubleSpinBoxLaplacianThreshold.value()

    def getShiftSize(self):
        return self.spinBoxShiftSize.value()

    def getBrightnessMin(self):
        return self.doubleSpinBoxBrightnessMin.value()

    def getBrightnessMax(self):
        return self.doubleSpinBoxBrightnessMax.value()

    def getSavePolylines(self):
        return self.checkBox_TriageImages_SavePolylines.isChecked()

    def getCorrectAlignment(self):
        return self.checkBox_TriageImages_CorrectAlignment.isChecked()

    def getReferenceImageFilename(self):
        return self.referenceImageFilename

    def getRotationThreshold(self):
        return self.doubleSpinBox_TriageOptions_RotationThreshold.value()
