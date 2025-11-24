#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# ML_Load_Model.py

from GRIME_AI.ML_Dependencies import *
import os
import sys
import json
import warnings
import shutil
from pathlib import Path

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, Normalize

from skimage.measure import find_contours

from omegaconf import OmegaConf, DictConfig


# Append SAM2 folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2'))
from sam2.sam2_image_predictor import SAM2ImagePredictor
# We no longer use the legacy build_sam2 method.
from sam2.modeling import sam2_base
print(sam2_base.__file__)

from PyQt5.QtWidgets import QMessageBox

from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Suppress specific warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

np.random.seed(3)


class ML_Load_Model:
    def __init__(self, cfg: DictConfig = None):
        self.className = "ML_Load_Model"

        # If no config is provided—or if there is no "load_model" entry, then fall back to reading a local configuration file.

        if cfg is None or "load_model" not in cfg:
            settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
            CONFIG_FILENAME = "site_config.json"
            config_file = os.path.join(settings_folder, CONFIG_FILENAME)
            with open(config_file, 'r') as file:
                # Expecting a "load_model" key in the JSON file
                self.config = json.load(file).get("load_model", {})
        else:
            # Convert Hydra DictConfig to a standard dict.a
            self.config = OmegaConf.to_container(cfg.load_model, resolve=True)

        # Set default values with the possibility to override via configuration.
        dirname = os.path.dirname(__file__)

        self.SAM2_CHECKPOINT = self.config.get("SAM2_CHECKPOINT", "")
        self.SAM2_CHECKPOINT = os.path.join(dirname, self.SAM2_CHECKPOINT)

        self.MODEL_CFG = self.config.get("MODEL_CFG", "")
        self.MODEL_CFG = os.path.normpath(self.MODEL_CFG)

        self.INPUT_DIR = self.config.get("INPUT_DIR", "")
        self.INPUT_DIR = os.path.normpath(self.INPUT_DIR)

        self.OUTPUT_DIR = self.config.get("OUTPUT_DIR", "")
        self.OUTPUT_DIR = os.path.normpath(self.OUTPUT_DIR)

        self.MODEL = self.config.get("MODEL", "")
        self.MODEL = os.path.normpath(self.MODEL)

        if self.SAM2_CHECKPOINT == "" or self.MODEL_CFG == "" or self.INPUT_DIR == "" or self.OUTPUT_DIR == "" or self.MODEL == "":
            print ("ERROR: Configuration file missing items.")

        self._check_for_required_files()


    def _check_for_required_files(self):
        nError = 0

        # Collect all the critical paths you need to verify
        paths_to_check = [
            ("Input directory", self.INPUT_DIR),
            ("Trained model file", self.MODEL),
        ]
        #("Model config file", self.MODEL_CFG),
        #("SAM2 checkpoint", self.SAM2_CHECKPOINT),
        #("Output directory", self.OUTPUT_DIR),

        self.missing_items = []
        for name, path in paths_to_check:
            if not os.path.exists(path):
                self.missing_items.append((name, path))

        if self.missing_items:
            nError = -1
            self._show_missing_files_dialog(self.missing_items)

        return nError


    def _show_missing_files_dialog(self, missing_items):
        """
        Show a critical QMessageBox listing all missing files/dirs,
        then terminate the application cleanly.
        """
        # Build a human-readable message
        lines = [
            f"{name}: {path}"
            for name, path in missing_items
        ]
        full_msg = (
            "The following files or directories are missing or have been moved:\n\n"
            + "\n".join(lines) + "\n"
        )

        # Display the error box
        msgBox = GRIME_AI_QMessageBox('Model Configuration Error', full_msg, QMessageBox.Close, icon=QMessageBox.Critical)
        msgBox.displayMsgBox()


    def get_color_for_category(self, category_id):
        color_map = {
            0: np.array([1.0, 0.0, 0.0, 0.6]),  # Red
            1: np.array([0.0, 1.0, 0.0, 0.6]),  # Green
            2: np.array([0.0, 0.0, 1.0, 0.6]),  # Blue
            3: np.array([1.0, 1.0, 0.0, 0.6]),  # Yellow
            4: np.array([1.0, 0.0, 1.0, 0.6]),  # Magenta
            5: np.array([0.0, 1.0, 1.0, 0.6]),  # Cyan
            6: np.array([0.5, 0.0, 0.0, 0.6]),  # Dark Red
            7: np.array([0.0, 0.5, 0.0, 0.6]),  # Dark Green
            8: np.array([0.0, 0.0, 0.5, 0.6]),  # Navy
            9: np.array([0.5, 0.5, 0.0, 0.6]),  # Olive
            10: np.array([0.5, 0.0, 0.5, 0.6]),  # Purple
            11: np.array([0.0, 0.5, 0.5, 0.6]),  # Teal
            12: np.array([1.0, 0.5, 0.0, 0.6]),  # Orange
            13: np.array([0.5, 1.0, 0.0, 0.6]),  # Lime
            14: np.array([0.0, 1.0, 0.5, 0.6]),  # Spring Green
            15: np.array([0.0, 0.5, 1.0, 0.6]),  # Azure
            16: np.array([0.5, 0.0, 1.0, 0.6]),  # Violet
            17: np.array([1.0, 0.0, 0.5, 0.6]),  # Rose
            18: np.array([0.25, 0.25, 0.25, 0.6]),  # Charcoal Gray
            19: np.array([0.7, 0.7, 0.7, 0.6]),  # Light Gray
            20: np.array([1.0, 0.85, 0.8, 0.6]),  # Pink
            21: np.array([0.3, 0.3, 1.0, 0.6]),  # Periwinkle
            22: np.array([1.0, 0.9, 0.2, 0.6]),  # Gold
            23: np.array([0.2, 0.8, 0.2, 0.6]),  # Fern
            24: np.array([0.2, 0.4, 0.8, 0.6]),  # Denim
            25: np.array([0.8, 0.2, 0.4, 0.6]),  # Crimson
            26: np.array([0.4, 0.8, 0.8, 0.6]),  # Ice Blue
            27: np.array([0.6, 0.3, 0.0, 0.6]),  # Rust
            28: np.array([0.9, 0.6, 0.4, 0.6]),  # Apricot
            29: np.array([0.3, 0.6, 0.1, 0.6]),  # Moss
            30: np.array([0.6, 0.1, 0.6, 0.6]),  # Orchid
            31: np.array([0.3, 0.7, 0.6, 0.6]),  # Seafoam
        }

        return color_map.get(category_id, np.array([0.5, 0.5, 0.5, 0.6]))  # Default: gray


    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1],
                   color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1],
                   color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)


    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h,
                                   edgecolor='green',
                                   facecolor=(0, 0, 0, 0),
                                   lw=2))


    def show_masks(self, output_file_with_path, image, mask, scores,
                   point_coords=None, box_coords=None, input_labels=None,
                   borders=True, category_id=None):
        fig, ax = plt.subplots()
        ax.imshow(image)
        # hide axes, ticks, and spines
        ax.axis('off')

        self.show_mask(mask, ax, category_id=category_id, borders=borders)
        fig.savefig(output_file_with_path)
        plt.close(fig)


    def show_mask(self, mask, ax, category_id=None, borders=True):
        color = self.get_color_for_category(category_id)
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = np.zeros((h, w, 4), dtype=np.float32)
        rgba_color = color.reshape((1, 1, -1))  # RGBA shape

        mask_image += mask.reshape(h, w, 1) * rgba_color

        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()

                if contour.ndim != 2 or contour.shape[1] != 2:
                    continue  # Skip invalid contours

                ax.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color="white")

        ax.imshow(mask_image)


    def mask_to_polygon(self, mask, min_contour_area=50):
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        if hierarchy is None:
            return segmentation
        for i, contour in enumerate(contours):
            contour = contour.flatten().tolist()
            if len(contour) < 6 or cv2.contourArea(contours[i]) < min_contour_area:
                continue
            if hierarchy[0][i][3] == -1:
                segmentation.append(contour)
            else:
                segmentation.append(contour[::-1])
        return segmentation


    def ML_Load_Model_Main(self, copy_original_image, save_masks, selected_label_categories):
        if self.missing_items:
            return

        global progress_bar_closed

        # INITIALIZE COLORWHEEL PROGRESS BAR
        def on_progress_bar_closed(obj):
            global progress_bar_closed
            progress_bar_closed = True

        progressBar = QProgressWheel()
        progressBar.destroyed.connect(on_progress_bar_closed)
        progress_bar_closed = False
        progressBar.setRange(0, 1)
        progressBar.setValue(1)
        progressBar.show()

        # DETERMINE WHETHER TO USE THE CPU FOR IF CUDA CORES ARE AVAILABLE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # LOAD THE SAM2 CHECKPOINT
        config_dir = os.path.join("sam2", "sam2", "configs", "sam2.1")
        model_cfg_path = os.path.join(config_dir, os.path.basename(self.MODEL_CFG))
        print("Model config path:", model_cfg_path)
        print("Checkpoint path:", self.SAM2_CHECKPOINT)

        from hydra import initialize, compose
        from hydra.utils import instantiate

        with initialize(config_path=config_dir, version_base=None):
            cfg_intern = compose(config_name=os.path.basename(model_cfg_path))
            raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)
            offending_keys = ["no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs", "device"]
            for key in offending_keys:
                raw_model_cfg.pop(key, None)
            new_cfg = OmegaConf.create(raw_model_cfg)
            model = instantiate(new_cfg, _recursive_=True)

        sam2_model = model.to(device)
        predictor = SAM2ImagePredictor(sam2_model)

        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, self.MODEL)
        checkpoint = torch.load(model_path, map_location=device)

        # --------------------------------------------------------------------------------------------------------------
        # PRINT CHECKPOINT CONTENTS TO CONSOLE
        # --------------------------------------------------------------------------------------------------------------
        print("Checkpoint contents:")
        for key, value in checkpoint.items():
            # show type for metadata, dict size for nested dicts
            if isinstance(value, dict):
                print(f"  {key}: dict[{len(value)}]")
            else:
                print(f"  {key}: {type(value).__name__}")

        # --------------------------------------------------------------------------------------------------------------
        # PROVISIONAL
        # --------------------------------------------------------------------------------------------------------------
        if 0:
            # if there's a state dict, show each tensor’s shape
            if "model_state_dict" in checkpoint:
                print("model_state_dict parameter shapes:")
                for name, tensor in checkpoint["model_state_dict"].items():
                    print(f"  {name}: {tuple(tensor.shape)}")

        print()

        # --------------------------------------------------------------------------------------------------------------
        #
        # --------------------------------------------------------------------------------------------------------------
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [],
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {
                "contributor": "", "date_created": "",
                "description": "", "url": "",
                "version": "", "year": ""
            },
        }

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            predictor.model.load_state_dict(checkpoint["model_state_dict"])
            coco_data["categories"].extend(selected_label_categories)
        else:
            # Grab the model’s expected keys
            model_keys = set(predictor.model.state_dict().keys())

            # Filter out keys that do not match
            clean_ckpt = {
                k: v for k, v in checkpoint.items()
                if k in model_keys
            }

            # Load only matching keys
            predictor.model.load_state_dict(clean_ckpt)

            fallback_category = {"id": 2, "name": "water"}
            coco_data["categories"].extend([fallback_category])

        image_id = 1
        annotation_id = 1

        # CREATE THE OUTPUT FOLDER WHERE THE PREDICTIONS AND OTHER OUTPUTS WILL BE SAVED
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # BUILD A LIST OF THE IMAGES TO BE SEGMENTED
        VALID_EXTS = ('.jpg', '.jpeg')
        images_list = [f for f in os.listdir(self.INPUT_DIR) if f.lower().endswith(VALID_EXTS)]

        # SET THE RANGE OF THE PROGRESS BAR BASED UPON THE NUMBER OF IMAGES TO SEGMENT
        progressBar.setRange(0, len(images_list) + 1)

        # --------------------------------------------------------------------------------------------------------------
        # BEGIN SEGMENTING IMAGES
        # --------------------------------------------------------------------------------------------------------------
        for img_index, image in enumerate(images_list):
            if progress_bar_closed is False:
                progressBar.setValue(img_index)

                # LOAD THE IMAGE FROM FILE AND FEED IT TO THE PREDICTOR
                image_path = os.path.join(self.INPUT_DIR, image)
                pil_image = Image.open(image_path).convert("RGB")
                image_array = np.array(pil_image)
                predictor.set_image(image_array)
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    multimask_output=True,
                )

                # Warn if more than one mask candidate is returned, which might require manual review
                if len(masks) > 1:
                    print(f"Multiple masks detected for image: {image} (Total: {len(masks)})")

                # Choose the mask with the highest confidence score as the final output
                sorted_ind = np.argmax(scores)
                mask = masks[sorted_ind]
                score = scores[sorted_ind]

                # === Select category_id
                category_id = selected_label_categories[0]["id"] if selected_label_categories else 2

                overlay_filename = os.path.splitext(image)[0] + "_overlay.png"
                overlay_output_with_path = os.path.join(self.OUTPUT_DIR, overlay_filename)
                self.show_masks(
                    overlay_output_with_path,
                    pil_image,
                    mask,
                    score,
                    borders=True,
                    category_id=category_id
                )

                if save_masks:
                    mask_filename = os.path.splitext(image)[0] + "_mask.png"
                    mask_output_path = os.path.join(self.OUTPUT_DIR, mask_filename)
                    mask_to_save = (mask.astype(np.uint8)) * 255
                    cv2.imwrite(mask_output_path, mask_to_save)
                    print(f"Mask saved to {mask_output_path}")

                if copy_original_image:
                    copied_image_path = os.path.join(self.OUTPUT_DIR, os.path.basename(image_path))
                    try:
                        shutil.copy(image_path, copied_image_path)
                        print(f"Copied original image to: {copied_image_path}")
                    except Exception as e:
                        print(f"Failed to copy image '{image_path}': {e}")

                height, width = image_array.shape[:2]
                coco_data["images"].append({
                    "file_name": os.path.basename(image),
                    "height": height,
                    "width": width,
                    "id": image_id,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0
                })

                segmentation = self.mask_to_polygon(mask)
                if not segmentation:
                    continue

                pos = np.where(mask)
                xmin, xmax = int(np.min(pos[1])), int(np.max(pos[1]))
                ymin, ymax = int(np.min(pos[0])), int(np.max(pos[0]))
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": int(np.sum(mask.astype(np.uint8))),
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
                image_id += 1
            else:
                strMessage = 'Image segmentation was cancelled before completion.'
                msgBox = GRIME_AI_QMessageBox('Image Segmentation Terminated', strMessage, QMessageBox.Close)
                msgBox.displayMsgBox()
                break

        if progress_bar_closed is False:
            progressBar.close()
        del progressBar

        output_file = Path(self.OUTPUT_DIR) / "predictions.json"
        with open(output_file, "w") as f:
            json.dump(coco_data, f, indent=4)

        print(f"COCO annotations saved to {output_file}")
