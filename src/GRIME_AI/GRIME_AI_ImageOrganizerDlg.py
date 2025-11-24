#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author(s):  Razin Bin Issa | Troy E. Gilmore | John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: rissa3@huskers.unl.edu, gilmore@unl.edut, jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Sep. 15, 2025
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
from datetime import datetime
from pathlib import Path
import traceback
import os

from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QTextOption
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox

from GRIME_AI.image_organizer import (
    organize_images, example_filename,
    scan_for_datetime_presence, move_files_to_subfolder
)

from GRIME_AI.GRIME_AI_CSS_Styles import BUTTON_CSS_STEEL_BLUE

# ----------------------------------------------------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------------------------------------------------
UI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),'ui',"QDialog_ImageOrganizer.ui")

# ======================================================================================================================
# ======================================================================================================================
#  =====     =====     =====     =====  class GRIME_AI_ImageOrganizerDlg   =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class GRIME_AI_ImageOrganizerDlg(QDialog):
    """
    Dynamic-UI Image Organizer dialog.

    - Folder/Site Name/Copyright heights match the 'Browse...' button.
    - Site Information wraps long words and the dialog auto-expands (never shrinks).
    - Pre-scan: Abort / Move Missing / Proceed.
    - Uses finalized core (seconds handling, tags merge, EXIF writes, CSV).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            uic.loadUi(str(UI_FILE), self)
        except Exception as e:
            print("[ERROR] uic.loadUi failed:", e)
            traceback.print_exc()
            raise

        # Global button style
        self.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        # Line-wrap behavior for Site Information
        self.site_info_edit.setLineWrapMode(self.site_info_edit.WidgetWidth)
        self.site_info_edit.setWordWrapMode(QTextOption.WrapAnywhere)

        # Initial state
        self.run_btn.setEnabled(False)
        self.example_lbl.setText("Example filename: (select a folder)")

        # Signals
        self.src_btn.clicked.connect(self.pick_src)
        self.src_edit.textChanged.connect(self.update_example)
        self.site_edit.textChanged.connect(self.update_example)
        self.sec_chk.toggled.connect(self.update_example)
        self.run_btn.clicked.connect(self.run_organizer)

        # Sync heights and capture initial dialog size after layout settles
        self._initial_size = None
        QTimer.singleShot(0, self._sync_text_field_heights)
        QTimer.singleShot(0, self._capture_initial_size)

        # Grow dialog when Site Information grows (but never shrink)
        self.site_info_edit.textChanged.connect(self._grow_for_site_info)

        # Track pre-scan decision
        self._prescan_decision = None  # "abort" | "move" | "proceed"

    # ---------- sizing helpers ----------
    def _capture_initial_size(self):
        self._initial_size = self.size()
        self.setMinimumSize(self._initial_size)

    def _sync_text_field_heights(self):
        h = self.src_btn.sizeHint().height()
        for le in (self.src_edit, self.site_edit, self.copyright_edit):
            le.setMinimumHeight(h)
            le.setMaximumHeight(h)

    def _grow_for_site_info(self):
        """
        Grow the QTextEdit to fit content and enlarge the dialog if needed.
        Never shrink below the current or initial size.
        """
        doc = self.site_info_edit.document()
        doc.adjustSize()

        frame = int(self.site_info_edit.frameWidth() * 2)
        content_h = int(doc.size().height()) + frame + 12
        min_h = max(140, content_h)
        self.site_info_edit.setMinimumHeight(min_h)

        hint = self.sizeHint()
        target_w = max(self.width(), hint.width(),
                       self._initial_size.width() if self._initial_size else 0)
        target_h = max(self.height(), hint.height(),
                       self._initial_size.height() if self._initial_size else 0)
        if target_w > self.width() or target_h > self.height():
            self.resize(target_w, target_h)

    # ---------- misc helpers ----------
    def _steel_blue_msgbox(self, title: str, text: str, icon=QMessageBox.Information):
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(icon)
        box.setText(text)
        box.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        return box

    def _compute_example(self) -> str:
        return example_filename(
            self.site_edit.text().strip(),
            self.sec_chk.isChecked(),
            example_dt=datetime.now(),
            ext=".jpg"
        )

    @pyqtSlot()
    def update_example(self):
        if not self.src_edit.text().strip():
            self.example_lbl.setText("Example filename: (select a folder)")
        else:
            self.example_lbl.setText(f"Example filename: {self._compute_example()}")

    # ---------- pre-scan on folder selection ----------
    @pyqtSlot()
    def pick_src(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder", str(Path.home()))
        if not d:
            return

        self.src_edit.setText(d)
        print("[DEBUG] Selected folder:", d)

        try:
            has_dt, missing_dt = scan_for_datetime_presence(Path(d), self.recurse_chk.isChecked())
            n_has, n_missing = len(has_dt), len(missing_dt)
            print(f"[INFO] Pre-scan: with date={n_has}, missing date={n_missing}")

            if n_has == 0 and n_missing > 0:
                title = "No 'Date Taken' Found"
                text = ("The program did not find any 'Date Taken' (creation date) "
                        "in the images' metadata.\n\nWhat would you like to do?")
                icon = QMessageBox.Warning
            else:
                title = "Date Taken Scan Results"
                text = (f"{n_has} image(s) HAVE 'Date Taken'.\n"
                        f"{n_missing} image(s) DO NOT.\n\n"
                        "How would you like to proceed?")
                icon = QMessageBox.Information

            box = self._steel_blue_msgbox(title, text, icon)
            abort_btn = box.addButton("Abort", QMessageBox.RejectRole)
            move_btn = box.addButton("Move Missing to Subfolder", QMessageBox.ActionRole)
            proceed_btn = box.addButton("Proceed Anyway", QMessageBox.AcceptRole)
            box.exec_()

            clicked = box.clickedButton()
            if clicked == abort_btn:
                self._prescan_decision = "abort"
                print("[INFO] User chose Abort. Closing dialog.")
                self.close()
                return
            elif clicked == move_btn:
                self._prescan_decision = "move"
                subfolder = move_files_to_subfolder(missing_dt, base_dir=Path(d))
                print(f"[INFO] Moved {n_missing} no-date file(s) to: {subfolder}")
                if self.recurse_chk.isChecked():
                    self.recurse_chk.setChecked(False)
                    print("[DEBUG] Recursion disabled after move.")
                info_box = self._steel_blue_msgbox(
                    "Moved Files",
                    f"Moved {n_missing} image(s) without 'Date Taken' to:\n{subfolder}\n\n"
                    "Now choose options and click Run."
                )
                info_box.exec_()
            else:
                self._prescan_decision = "proceed"
                print("[INFO] User chose Proceed.")

            self.run_btn.setEnabled(True)
            self.update_example()

        except Exception as e:
            print("[ERROR] Pre-scan failed:", e)
            err_box = self._steel_blue_msgbox("Pre-scan Error", str(e), QMessageBox.Critical)
            err_box.exec_()
            self.run_btn.setEnabled(False)

    # ---------- main action ----------
    @pyqtSlot()
    def run_organizer(self):
        folder = self.src_edit.text().strip()
        if not folder:
            warn = self._steel_blue_msgbox("Missing Folder", "Please choose a folder of images.", QMessageBox.Warning)
            warn.exec_()
            return

        if self._prescan_decision == "abort":
            print("[WARN] Run pressed after Abort; ignoring.")
            return

        try:
            log_csv, rows, tags, errors = organize_images(
                src_folder=Path(folder),
                recursive=self.recurse_chk.isChecked(),
                site=self.site_edit.text().strip(),
                include_seconds=self.sec_chk.isChecked(),
                copyright_holder=self.copyright_edit.text().strip(),
                site_info=self.site_info_edit.toPlainText().strip()
            )

            msg = f"Image Organizer completed.\nLog file:\n{log_csv}"
            if errors:
                msg += f"\n\nCompleted with {len(errors)} warning(s)/error(s) (see terminal)."
            else:
                msg += "\n\nAll images processed successfully."

            done = self._steel_blue_msgbox("Done", msg, QMessageBox.Information)
            done.exec_()
            print("[INFO] Completed. Log:", log_csv)

        except Exception as e:
            print("[ERROR] Image Organizer failed:", e)
            err = self._steel_blue_msgbox("Error", str(e), QMessageBox.Critical)
            err.exec_()
