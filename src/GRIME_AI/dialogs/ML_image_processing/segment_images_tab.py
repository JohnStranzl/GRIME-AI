# segment_images_tab.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Ported: direct port of Segment Images tab from GRIME_AI_ML_ImageProcessingDlg.py
# License: Apache License, Version 2.0

import os

from pathlib import Path
from typing import Optional
from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QSizePolicy
from PyQt5.uic import loadUi

from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.dialogs.ML_image_processing.model_config_manager import ModelConfigManager
from GRIME_AI.utils.resource_utils import ui_path
from GRIME_AI.GRIME_AI_CSS_Styles import BUTTON_CSS_STEEL_BLUE

# import torch if using torch metadata extraction
try:
    import torch
except Exception:
    torch = None


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===            HELPER FUNCTIONS            ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
def _normalize_labels(raw):
    if raw is None:
        return None
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return [str(c.get("name") or c.get("label") or c.get("class") or repr(c)) for c in raw]
    if isinstance(raw, dict):
        try:
            items = sorted(raw.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])
            return [str(v) for _, v in items]
        except Exception:
            return [str(v) for v in raw.values()]
    if isinstance(raw, str):
        return [raw]
    try:
        return [str(x) for x in list(raw)]
    except Exception:
        return None

# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===         class SegmentImagesTab         ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class SegmentImagesTab(QWidget):
    """
    Direct port of the Segment Images tab UI and logic.
    This preserves original comments, method names, and expected attributes.
    """

    # Signal to notify parent/dialog that segmentation should start.
    ml_segment_signal = pyqtSignal()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        # Load the dedicated Segment Images UI
        loadUi(ui_path("ML_image_processing/segment_images_tab.ui"), self)

        self.setup_ui_properties()

        # Preserve dialog-level state expected by methods copied from the original dialog
        self.transferred_items = set()
        self.original_folders = []
        self.selected_label_categories = []
        self.categories_available = False

        # Default selection
        self.selected_segment_model = "sam2"

        layout = self.horizontalLayoutSegmentImages
        layout.setStretch(0, 4)  # left content area
        layout.setStretch(1, 1)  # right 'Labels' group box

        settings_folder = Path(GRIME_AI_Save_Utils().get_settings_folder()).resolve()
        config_file = (settings_folder / "site_config.json").resolve()

        # Backup existing config if present
        if not config_file.exists():
            # Start with template
            self.site_config = self.create_site_config_template()
        else:
            self.site_config = JsonEditor().load_json_file(config_file)

        # WIRE SIGNALS TO WIDGETS (BUTTONS, RADIOBUTTONS, ETC.)
        self.setup_connections()

        self.populate_segment_images_tab()

        self.setup_from_config_file()

        # UPDATE BUTTON STATE AFTER INITIALIZATION
        try:
            self.updateSegmentButtonState()
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_from_config_file(self):
        """
        Initialize dialog controls from a configuration dictionary.
        """
        load_model_conf = self.site_config.get("load_model", {})
        model_type = load_model_conf.get("MODEL", "").lower()

        # Select the correct model file key based on MODEL type
        if model_type == "sam2":
            model_file = load_model_conf.get("SAM2_MODEL", "")
        elif model_type == "segformer":
            model_file = load_model_conf.get("SEGFORMER_MODEL", "")
        elif model_type == "maskrcnn":
            model_file = load_model_conf.get("MASKRCNN_MODEL", "")
        else:
            model_file = ""

        if model_file:
            self.lineEdit_segmentation_model_file.setText(model_file)
            self.populate_model_labels(model_file)
        else:
            self.lineEdit_segmentation_model_file.clear()

        # Images folder path
        input_dir = load_model_conf.get("segmentation_images_path", "")
        if input_dir:
            self.lineEdit_segmentation_images_folder.setText(input_dir)
        else:
            self.lineEdit_segmentation_images_folder.clear()

        # Checkboxes (default to True if missing)
        self.checkBox_save_predicted_masks.setChecked(load_model_conf.get("save_model_masks", True))
        self.checkBox_copyOriginalModelImage.setChecked(load_model_conf.get("copy_original_model_image", True))
        self.checkBox_save_probability_maps.setChecked(load_model_conf.get("save_probability_maps", True))

        # Set the appropriate radio button based on MODEL
        if model_type == "sam2":
            self.radioButton_segment_model_sam2.setChecked(True)
        elif model_type == "segformer":
            self.radioButton_segment_model_segformer.setChecked(True)
        elif model_type == "maskrcnn":
            self.radioButton_segment_model_mask_rcnn.setChecked(True)
        else:
            # If no model type is stored, clear all selections
            self.radioButton_segment_model_sam2.setChecked(False)
            self.radioButton_segment_model_segformer.setChecked(False)
            self.radioButton_segment_model_mask_rcnn.setChecked(False)

        # Select items in the listbox based on SEGMENTATION_CATEGORIES
        stored_categories = load_model_conf.get("SEGMENTATION_CATEGORIES", [])
        if stored_categories:
            for idx in range(self.listWidget_labels.count()):
                item = self.listWidget_labels.item(idx)
                if item and item.text().strip() in stored_categories:
                    item.setSelected(True)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_values(self, site_config: dict) -> dict:
        """
        Collect values from dialog controls and update the given site_config dictionary.
        Returns the updated site_config.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # _____ GRIME AI ML parameters _______________________________________
        site_config["save_model_masks"] = self.checkBox_save_predicted_masks.isChecked()
        site_config["copy_original_model_image"] = self.checkBox_copyOriginalModelImage.isChecked()
        site_config["save_probability_maps"] = self.checkBox_save_probability_maps.isChecked()

        # _____ Load model section ___________________________________________
        site_config.setdefault("load_model", {})

        segmentation_images_path = self.lineEdit_segmentation_images_folder.text().strip()
        if segmentation_images_path:
            site_config["load_model"]["segmentation_images_path"] = segmentation_images_path

            predictions_output_path = os.path.normpath(os.path.join(segmentation_images_path, f"{timestamp}_predictions"))
            site_config["load_model"]["predictions_output_path"] = predictions_output_path

        # Selected segmentation categories from listbox
        selected_labels = [
            self.listWidget_labels.item(idx).text().strip()
            for idx in range(self.listWidget_labels.count())
            if self.listWidget_labels.item(idx) and self.listWidget_labels.item(idx).isSelected()
        ]
        site_config["load_model"]["SEGMENTATION_CATEGORIES"] = selected_labels

        # Selected model type from radio buttons
        if self.radioButton_segment_model_sam2.isChecked():
            site_config["load_model"]["MODEL"] = "sam2"
            site_config["load_model"]["SAM2_MODEL"] = self.lineEdit_segmentation_model_file.text().strip()
        elif self.radioButton_segment_model_segformer.isChecked():
            site_config["load_model"]["MODEL"] = "segformer"
            site_config["load_model"]["SEGFORMER_MODEL"] = self.lineEdit_segmentation_model_file.text().strip()
        elif self.radioButton_segment_model_mask_rcnn.isChecked():
            site_config["load_model"]["MODEL"] = "maskrcnn"
            site_config["load_model"]["MASKRCNN_MODEL"] = self.lineEdit_segmentation_model_file.text().strip()
        else:
            site_config["load_model"]["MODEL"] = ""  # fallback if none selected
            site_config["load_model"]["SAM2_MODEL"] = ""
            site_config["load_model"]["SEGFORMER_MODEL"] = ""
            site_config["load_model"]["MASKRCNN_MODEL"] = ""

        return site_config

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def populate_segment_images_tab(self):
        """
        Initialize segmentation tab UI state that the original dialog expected.
        Mirrors original dialog setup: ensures attributes exist and sets sensible defaults.
        """
        # Ensure attributes expected by other methods exist
        if not hasattr(self, "transferred_items"):
            self.transferred_items = set()
        if not hasattr(self, "original_folders"):
            self.original_folders = []
        if not hasattr(self, "selected_label_categories"):
            self.selected_label_categories = []
        if not hasattr(self, "categories_available"):
            self.categories_available = False

        # Clear labels and model path by default (keeps behavior predictable)
        try:
            self.listWidget_labels.clear()
        except Exception:
            pass

        try:
            self.lineEdit_segmentation_model_file.setText("")
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_ui_properties(self):
        """Set size policies and layout stretch factors."""
        self.pushButton_Segment.setStyleSheet(
            'QPushButton {background-color: steelblue; color: white; }'
            'QPushButton:disabled {background-color: gray; color: black; }'
        )
        self.pushButton_Segment.setMinimumSize(150, 40)
        self.pushButton_Segment.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Keep local widget policies similar to original dialog

        try:
            self.listWidget_labels.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.listWidget_labels.setMinimumHeight(200)
        except Exception:
            pass

        try:
            self.pushButton_Segment.setMinimumSize(150, 40)
            self.pushButton_Segment.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_connections(self):
        """Connect signals with their slot methods (mirrors original dialog)."""

        # Segment model radio buttons
        self.radioButton_segment_model_sam2.toggled.connect(lambda checked: self.set_segment_model("sam2", checked))
        self.radioButton_segment_model_sam2.toggled.connect(self.update_model_config)

        self.radioButton_segment_model_segformer.toggled.connect(lambda checked: self.set_segment_model("segformer", checked))
        self.radioButton_segment_model_segformer.toggled.connect(self.update_model_config)

        self.radioButton_segment_model_mask_rcnn.toggled.connect(lambda checked: self.set_segment_model("maskrcnn", checked))
        self.radioButton_segment_model_mask_rcnn.toggled.connect(self.update_model_config)

        # Buttons
        self.pushButton_Select_Model.clicked.connect(self.select_segmentation_model)
        self.pushButton_Select_Model.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_Select_Images_Folder.clicked.connect(self.select_segmentation_images_folder)
        self.pushButton_Select_Images_Folder.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_Segment.clicked.connect(self.segment_images)
        self.pushButton_Segment.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        # Line edits and checkboxes → flush config on change
        self.lineEdit_segmentation_model_file.textChanged.connect(self.onModelPathChanged)
        self.lineEdit_segmentation_model_file.textChanged.connect(self.update_model_config)

        self.lineEdit_segmentation_images_folder.textChanged.connect(self.updateSegmentButtonState)
        self.lineEdit_segmentation_images_folder.textChanged.connect(self.update_model_config)

        self.checkBox_save_predicted_masks.toggled.connect(self.on_save_predicted_masks_toggled)
        self.checkBox_save_predicted_masks.toggled.connect(self.update_model_config)

        self.checkBox_save_probability_maps.toggled.connect(self.on_save_probability_maps_toggled)
        self.checkBox_save_probability_maps.toggled.connect(self.update_model_config)

        self.checkBox_copyOriginalModelImage.toggled.connect(self.on_copy_original_toggled)
        self.checkBox_copyOriginalModelImage.toggled.connect(self.update_model_config)

        # Labels list affects segment button state
        try:
            self.listWidget_labels.itemSelectionChanged.connect(self.updateSegmentButtonState)
            self.listWidget_labels.itemSelectionChanged.connect(self.update_model_config)
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def set_segment_model(self, model_name: str, checked: bool):
        """Update selected_segment_model when a radio button is toggled on."""
        if checked:  # only update when the button is checked, not unchecked
            self.selected_segment_model = model_name
            print(f"Selected segment model: {self.selected_segment_model}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def select_segmentation_model(self):
        """
        Open a file dialog to select a segmentation model file (only .torch files).
        Clears the label list and populates it from the model metadata.
        """
        model_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Segmentation Model",
            "",
            "Torch Model Files (*.torch)"
        )

        if model_file:
            self.lineEdit_segmentation_model_file.setText(model_file)

            print("Segmentation model selected:", model_file)

            # Clear the label listbox before repopulating
            self.listWidget_labels.clear()

            # Try to populate labels from model metadata if available
            try:
                self.populate_model_labels(model_file)
            except Exception as e:
                print(f"populate_model_labels failed: {e}")

            self.updateSegmentButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def select_segmentation_images_folder(self):
        """Open directory dialog to choose images folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if folder:
            self.lineEdit_segmentation_images_folder.setText(folder)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def onModelPathChanged(self, path: str):
        """React to model path changes (basic validation)."""
        if path:
            p = Path(path)
            if not p.exists():
                print(f"[SegmentImagesTab] Warning: model path does not exist: {path}")
        self.updateSegmentButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def updateSegmentButtonState(self):
        """
        Enable the Segment button only when a model path and images folder are provided.
        Optionally require at least one label selected if labels exist.
        """
        model_path = self.lineEdit_segmentation_model_file.text().strip()
        images_folder = self.lineEdit_segmentation_images_folder.text().strip()

        labels_exist = (self.listWidget_labels.count() > 0)
        labels_selected = True
        if labels_exist:
            labels_selected = any(self.listWidget_labels.item(i).isSelected() for i in range(self.listWidget_labels.count()))

        enabled = bool(model_path and images_folder and labels_selected)
        try:
            self.pushButton_Segment.setEnabled(enabled)
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def segment_images(self):
        """
        Called when the Segment button is clicked.
        Validates label selections, updates configuration, runs segmentation,
        then post-processes outputs according to the two checkboxes.
        """
        if self.categories_available:
            # Ensure that at least one label is selected
            selected_labels = []
            for idx in range(self.listWidget_labels.count()):
                item = self.listWidget_labels.item(idx)
                if item and item.isSelected():
                    selected_labels.append(item.text())

            if not selected_labels:
                QMessageBox.warning(self, "Segmentation Error",
                                    "Please select at least one label before segmenting images.")
                return

            # Populate selected_label_categories from selection
            self.selected_label_categories = []
            if self.categories_available == True:
                for idx in range(self.listWidget_labels.count()):
                    item_text = self.listWidget_labels.item(idx).text().strip()
                    if item_text and item_text in selected_labels:
                        self.selected_label_categories.append({
                            "id": idx + 1,
                            "name": item_text
                        })

            print("Selected categories:", self.selected_label_categories)
        else:   # IF IT IS AN OLDER MODEL NOT CONTAINING LABELS, DEFAULT TO ID: 2, NAME: Water
            self.selected_label_categories = [{"id": 2, "name": "water"}]

        # update model config file (JSON) for downstream pipeline
        self.update_model_config()

        # Kick off the actual segmentation
        self.ml_segment_signal.emit()

        # Close dialog as “Accepted”
        #QtCore.QMetaObject.invokeMethod(
        #    self, 'done', Qt.QueuedConnection,
        #    QtCore.Q_ARG(int, QDialog.Accepted)
        #)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def update_model_config(self):
        """
        Gather all dialog values and update the JSON configuration file.
        Preserves existing settings from other classes by merging instead of resetting.
        """
        settings_folder = Path(GRIME_AI_Save_Utils().get_settings_folder()).resolve()
        config_file = (settings_folder / "site_config.json").resolve()

        # Use ModelConfigManager to handle backup + load
        manager = ModelConfigManager(filepath=config_file)
        site_config = manager.backup_config()  # backs up existing file and loads config (or {} if none)

        # Merge values from controls directly into site_config (includes segmentation categories now)
        site_config = self.get_values(site_config)

        # Save updated config using ModelConfigManager
        manager.config = site_config
        try:
            manager.save_config(config_file)
            print("Custom JSON file 'site_config.json' updated successfully.")
        except Exception as e:
            print(f"Failed to save config: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def on_save_predicted_masks_toggled(self, checked: bool):
        print(f"Save Masks checkbox toggled: {checked}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def on_save_probability_maps_toggled(self, checked: bool):
        print(f"Save Probability Maps checkbox toggled: {checked}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def on_copy_original_toggled(self, checked: bool):
        print(f"Copy Original Image checkbox toggled: {checked}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def populate_model_labels(self, model_path):
        """
        Load label categories from a torch checkpoint that contains metadata.
        Expected checkpoint layout (example you provided):
            ckpt = {
                "model_state_dict": ...,
                "categories": self.categories,
                "creation_UTC": ...,
                ...
            }
        If categories are not found, the UI is updated exactly as before.
        """
        # Try safe allowlist first, then fall back to full load only if necessary
        ckpt = None
        labels = None

        try:
            # Minimal allowlist for the numpy scalar global reported by PyTorch
            allowlist = [np.core.multiarray.scalar]

            try:
                with torch.serialization.safe_globals(allowlist):
                    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            except Exception as e_safe:
                # If the error message names additional globals, try to extract numpy-like symbols and add them
                msg = str(e_safe)
                found = re.findall(r"([A-Za-z0-9_\.]+numpy[^\s\]\)]+)", msg)
                for sym in found:
                    try:
                        parts = sym.split(".")
                        mod = __import__(".".join(parts[:-1]), fromlist=[parts[-1]])
                        obj = getattr(mod, parts[-1])
                        if obj not in allowlist:
                            allowlist.append(obj)
                    except Exception:
                        pass
                # Retry with expanded allowlist
                with torch.serialization.safe_globals(allowlist):
                    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)

        except Exception as e_outer:
            # Safe allowlist failed; fallback to full unpickle only if you trust the file source
            print("Safe allowlist load failed:", e_outer)
            try:
                print("Falling back to full load with weights_only=False (trusted source only).")
                ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            except Exception as e_full:
                print("Full load failed:", e_full)
                ckpt = None

        # Extract categories if checkpoint loaded
        if isinstance(ckpt, dict):
            labels = ckpt.get("categories") or ckpt.get("classes") or ckpt.get("labels") \
                     or (ckpt.get("meta") or {}).get("classes") or ckpt.get("target_category_name")

        labels = _normalize_labels(labels)

        # Clear previous entries
        self.listWidget_labels.clear()

        # No labels found — same UI behavior as before
        if not labels:
            QMessageBox.warning(
                self,
                "Load Model Categories",
                "No category list found in:\n" + model_path
            )
            self.categories_available = False
            self.listWidget_labels.addItem("<Older model format>")
            self.listWidget_labels.addItem("<Labels unavailable>")
            self.listWidget_labels.addItem("<ID 2 : Water assumed>")
            self.listWidget_labels.setDisabled(True)
            return

        # Populate listbox with only the label name (string)
        self.categories_available = True
        self.listWidget_labels.setDisabled(False)
        for entry in labels:
            if isinstance(entry, dict):
                # prefer common name keys; skip entries with no readable name
                name = entry.get("name") or entry.get("label") or entry.get("class") or entry.get("title")
                if name is None:
                    # if you prefer to skip entries without a name, continue; otherwise fallback to stringifying
                    continue
            else:
                name = str(entry)
            self.listWidget_labels.addItem(str(name))


# WILL NEED THIS IN ORDER TO UPDATE THE JSON SETTINGS FILE
'''
        # Normalize segmentation images folder
        seg_images_folder = self.lineEdit_segmentation_images_folder.text().strip()
        seg_images_folder = os.path.abspath(seg_images_folder).replace("\\", "/") if seg_images_folder else ""

        # Handle segmentation model path
        seg_model_path = self.lineEdit_segmentation_model_file.text().strip()
        if seg_model_path and seg_model_path.lower().endswith('.torch') and os.path.isfile(seg_model_path):
            abs_model_path = os.path.normpath(seg_model_path.strip())
            if self.selected_segment_model == "sam2":
                site_config["load_model"]["SAM2_MODEL"] = abs_model_path
            elif self.selected_segment_model == "segformer":
                site_config["load_model"]["SEGFORMER_MODEL"] = abs_model_path
            elif self.selected_segment_model == "maskrcnn":
                site_config["load_model"]["MASKRCNN_MODEL"] = abs_model_path
            print("Updated MODEL path to:", abs_model_path)
        else:
            print("No valid segmentation model file selected; using default MODEL path.")
'''