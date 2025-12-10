#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# MLImageSegmentation.py

import os
import sys
import json
import shutil

import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import QMessageBox

from omegaconf import OmegaConf, DictConfig

# Project imports
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils

# Engines
from GRIME_AI.ml_core.sam2_inference_engine import SAM2InferenceEngine
from GRIME_AI.ml_core.segformer_inference_engine import SegFormerInferenceEngine
from GRIME_AI.ml_core.ml_helpers import add_coco_entries


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===       class MLImageSegmentation        ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class MLImageSegmentation:
    def __init__(self, cfg: DictConfig = None):
        self.className = "MLImageSegmentation"
        self.progress_bar_closed = False

        # Load config (same as before)
        if cfg is None or "load_model" not in cfg:
            settings_folder = os.path.normpath(GRIME_AI_Save_Utils().get_settings_folder())
            CONFIG_FILENAME = "site_config.json"
            config_file = os.path.normpath(os.path.join(settings_folder, CONFIG_FILENAME))
            with open(config_file, 'r') as file:
                self.config = json.load(file).get("load_model", {})
        else:
            self.config = OmegaConf.to_container(cfg.load_model, resolve=True)

        main_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.SAM2_CHECKPOINT = os.path.normpath(os.path.join(main_dir, self.config.get("SAM2_CHECKPOINT", "")))
        self.MODEL_CFG = os.path.normpath(self.config.get("MODEL_CFG", ""))

        self.segmentation_images_path = os.path.normpath(self.config.get("segmentation_images_path", ""))
        self.predictions_output_path = os.path.normpath(self.config.get("predictions_output_path", ""))

        self.SAM2_MODEL = os.path.normpath(self.config.get("SAM2_MODEL", ""))
        self.SAM3_MODEL = os.path.normpath(self.config.get("SAM3_MODEL", ""))
        self.SEGFORMER_MODEL = os.path.normpath(self.config.get("SEGFORMER_MODEL", ""))

        if self.SAM2_CHECKPOINT == "" or self.MODEL_CFG == "" or self.segmentation_images_path == "" or self.predictions_output_path == "" or (self.SAM2_MODEL == "" and self.SEGFORMER_MODEL == ""):
            print("ERROR: Configuration file missing items.")

        self._check_for_required_files()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _check_for_required_files(self):
        nError = 0
        paths_to_check = [("Input directory", self.segmentation_images_path), ("Trained model file", self.SAM2_MODEL)]
        self.missing_items = [(name, path) for name, path in paths_to_check if not os.path.exists(path)]
        if self.missing_items:
            nError = -1
            self.show_missing_files_dialog(self.missing_items)
        return nError

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_missing_files_dialog(self, missing_items):
        lines = [f"{name}: {path}" for name, path in missing_items]
        full_msg = "The following files or directories are missing or have been moved:\n\n" + "\n".join(lines) + "\n"
        msgBox = GRIME_AI_QMessageBox('Model Configuration Error', full_msg, QMessageBox.Close, icon=QMessageBox.Critical)
        msgBox.displayMsgBox()

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def ML_Segmentation_Dispatcher(self, copy_original_image, save_masks, selected_label_categories, mode="segformer"):
        if self.missing_items:
            return

        self.progress_bar_closed = False
        progressBar = QProgressWheel(
            title="Segmenting images...", total=1,
            on_close=lambda: setattr(self, "progress_bar_closed", True)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes = len(selected_label_categories) + 1  # +1 for background

        if mode.lower() == "sam2":
            # Pass INPUT_DIR and OUTPUT_DIR
            engine = SAM2InferenceEngine(
                device=torch.device("cuda"),
                model_cfg="sam2.1_hiera_l.yaml",
                trained_checkpoint_path=self.SAM2_MODEL,
                input_dir=self.segmentation_images_path,
                output_dir=self.predictions_output_path
            )
            '''
            engine = SAM2InferenceEngine(
                device,
                self.MODEL_CFG,
                self.SAM2_CHECKPOINT,
                self.SAM2_MODEL,
                self.segmentation_images_path,
                self.predictions_output_path
            )
            '''

            predictor = engine.run_sam2_inference(
                copy_original_image, save_masks, selected_label_categories, progressBar
            )

        elif mode.lower() == "segformer":
            # Pass INPUT_DIR and OUTPUT_DIR
            engine = SegFormerInferenceEngine(
                device,
                self.SEGFORMER_MODEL,
                self.segmentation_images_path,
                self.predictions_output_path
            )
            predictor = engine.run_segformer_inference(
                copy_original_image, save_masks, selected_label_categories, progressBar
            )

        elif mode.lower() == "maskrcnn":
            predictor = self.load_maskrcnn_model(device, selected_label_categories)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return predictor

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def process_image(self, image_path, predictor, mode, device, save_masks, copy_original_image, coco_data, image_id,
                      annotation_id, category_id):
        pil_image = Image.open(image_path).convert("RGB")
        image_array = np.array(pil_image)

        if mode.lower() == "sam2":
            mask, score = predictor.predict_sam2(predictor, image_array)
        elif mode.lower() == "segformer":
            from torchvision import transforms as T
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            to_tensor = T.ToTensor()
            x = to_tensor(pil_image.resize((512, 512)))
            x = normalize(x).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = predictor(pixel_values=x).logits
            probs = torch.softmax(logits, dim=1)
            water_prob = probs[0, 1]
            mask = (water_prob > 0.2).cpu().numpy().astype(np.uint8)
            score = float(water_prob.mean().item())
        elif mode.lower() == "maskrcnn":
            mask, score = self.predict_maskrcnn(predictor, pil_image, device)
        else:
            pass

        if mask is None:
            return None, None

        if mode.lower() == "segformer":
            if copy_original_image:
                shutil.copy(image_path, os.path.join(self.predictions_output_path, os.path.basename(image_path)))
            add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id)
        else:
            predictor.save_outputs(image_path, pil_image, mask, score, save_masks, copy_original_image, category_id)
            add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id)

        return image_id + 1, annotation_id + 1
