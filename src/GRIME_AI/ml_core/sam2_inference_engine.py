#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# sam2_inference_engine.py

import os
import sys
import cv2
import shutil
import torch
import numpy as np

from PIL import Image

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, prevents GUI windows
import matplotlib.pyplot as plt

# SAM2 imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../sam2'))
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from GRIME_AI.ml_core.ml_helpers import (get_color_for_category, init_coco_structure, add_coco_entries, save_coco_json)


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===        class SAM2InferenceEngine       ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class SAM2InferenceEngine:
    def __init__(self, device, model_cfg, trained_checkpoint_path, input_dir, output_dir):
        """
        Args:
            device: torch device (cuda/cpu)
            model_cfg: path to SAM2 model config yaml (e.g., "sam2.1_hiera_l.yaml")
            trained_checkpoint_path: path to YOUR TRAINED checkpoint (.torch file)
            input_dir: directory with input images
            output_dir: directory for output predictions
        """
        self.device = device
        self.MODEL_CFG = model_cfg
        self.TRAINED_CHECKPOINT = trained_checkpoint_path  # ← Your trained model
        self.segmentation_images_path = input_dir
        self.predictions_output_path = output_dir

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_sam2_model(self):
        """Load SAM2 model architecture and trained weights from .torch checkpoint."""

        # 1. Setup paths
        main_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        config_file = os.path.join(main_dir, "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")

        print(f"Model config: {config_file}")
        print(f"Trained checkpoint: {self.TRAINED_CHECKPOINT}")

        # 2. Load model architecture from config
        cfg_intern = OmegaConf.load(config_file)
        raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)

        for key in ["no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs", "device"]:
            raw_model_cfg.pop(key, None)

        new_cfg = OmegaConf.create(raw_model_cfg)
        model = instantiate(new_cfg, _recursive_=True)

        # 3. Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam2_model = model.to(device).eval()
        predictor = SAM2ImagePredictor(sam2_model)

        # 4. Load trained checkpoint with metadata
        # Safe loading with proper error handling
        try:
            checkpoint = torch.load(
                self.TRAINED_CHECKPOINT,
                map_location=device,
                weights_only=False  # Required for checkpoints with metadata
            )
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise

        # 5. Display checkpoint info
        print("\n=== Checkpoint Information ===")
        print(f"Site: {checkpoint.get('site_name', 'N/A')}")
        print(f"Created: {checkpoint.get('creation_UTC', 'N/A')}")
        print(f"Epochs trained: {checkpoint.get('epochs', 'N/A')}")
        print(f"Learning rate: {checkpoint.get('learning_rate', 'N/A')}")
        print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"Validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
        print(f"Mean IoU: {checkpoint.get('miou', 'N/A')}")
        print(f"Categories: {len(checkpoint.get('categories', []))} classes")

        # 6. Load model weights
        if "model_state_dict" in checkpoint:
            predictor.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("✓ Loaded trained model weights successfully")
        else:
            raise ValueError("Checkpoint missing 'model_state_dict' key!")

        print("=== Model ready for inference ===\n")
        return predictor

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def predict_sam2(self, predictor, image_array, multimask_output=False):
        """
        Run SAM2 prediction on an image.
        If multimask_output=True, return the highest-scoring mask among multiple candidates.
        If multimask_output=False, return the single mask directly.
        """
        predictor.set_image(image_array)

        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=multimask_output
        )

        if len(scores) == 0:
            return None, None, None

        if multimask_output:
            # Multiple masks returned, pick the best one
            best_idx = int(np.argmax(scores))
            return masks[best_idx], scores[best_idx], logits[best_idx]
        else:
            # Single mask returned, take it directly
            return masks[0], scores[0], logits[0]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_masks(self, output_file_with_path, image, mask, borders=True, category_id=None):
        fig, ax = plt.subplots()
        try:
            ax.imshow(image)
            ax.axis('off')
            self.show_mask(mask, ax, category_id=category_id, borders=borders)
            fig.savefig(output_file_with_path, bbox_inches='tight', pad_inches=0)
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_mask(self, mask, ax, category_id=None, borders=True):
        color = get_color_for_category(category_id)
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = np.zeros((h, w, 4), dtype=np.float32)
        rgba_color = color.reshape((1, 1, -1))
        mask_image += mask.reshape(h, w, 1) * rgba_color

        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()
                if contour.ndim != 2 or contour.shape[1] != 2:
                    continue
                ax.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color="white")

        ax.imshow(mask_image)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1],
                   color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1],
                   color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h,
                                   edgecolor='green',
                                   facecolor=(0, 0, 0, 0),
                                   lw=2))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_outputs(self, image_path, pil_image, mask, prob_map, score, save_masks, copy_original_image, category_id):
        base = os.path.splitext(os.path.basename(image_path))[0]
        overlay_path = os.path.join(self.predictions_output_path, f"{base}_overlay.png")
        self.show_masks(overlay_path, pil_image, mask, borders=False, category_id=category_id)

        if save_masks:
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            mask_path = os.path.join(self.predictions_output_path, f"{base}_mask.png")
            cv2.imwrite(mask_path, (mask.astype(np.uint8)) * 255)

        if copy_original_image:
            shutil.copy(image_path, os.path.join(self.predictions_output_path, os.path.basename(image_path)))

        # IF PROB_MAP IS NONE, CREATE A FLAT ZERO MAP FOR PANEL TO AVOID CRASHES
        if prob_map is None:
            prob_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)

        self._save_panel(np.array(pil_image), mask, prob_map, self.predictions_output_path, base)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def run_sam2_inference(self, copy_original_image, save_masks, selected_label_categories, progressBar):
        predictor = self.load_sam2_model()
        coco_data = init_coco_structure(selected_label_categories)

        os.makedirs(self.predictions_output_path, exist_ok=True)
        VALID_EXTS = ('.jpg', '.jpeg')
        images_list = [f for f in os.listdir(self.segmentation_images_path) if f.lower().endswith(VALID_EXTS)]
        if progressBar is not None:
            progressBar.setRange(0, len(images_list) + 1)

        image_id = 1
        annotation_id = 1

        for img_index, image in enumerate(images_list):
            if progressBar is not None and progressBar.isVisible():
                progressBar.setValue(img_index)

            image_path = os.path.join(self.segmentation_images_path, image)

            try:
                pil_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                continue

            image_array = np.array(pil_image)

            multimask_output = False
            mask, score, logits = self.predict_sam2(predictor, image_array, multimask_output=multimask_output)
            if mask is None:
                continue

            target_h, target_w = image_array.shape[:2]
            # ENSURE THE MASK MATCHES THE IMAGE SIZE0
            if mask.shape != (target_h, target_w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST)


            # Resize logits to image size for heatmap
            if logits is not None:
                prob_map = cv2.resize(logits.astype(np.float32), (target_w, target_h),
                                      interpolation=cv2.INTER_LINEAR)
                base = os.path.splitext(os.path.basename(image_path))[0]
                self._save_heatmap(prob_map, self.predictions_output_path, base)

            category_id = selected_label_categories[0]["id"] if selected_label_categories else 2

            # Save overlay/mask outputs
            self.save_outputs(image_path, pil_image, mask, prob_map, score, save_masks, copy_original_image,
                              category_id)

            # COCO bookkeeping
            add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id)

            image_id += 1
            annotation_id += 1

        if progressBar is not None and progressBar.isVisible():
            progressBar.close()

        save_coco_json(coco_data, self.predictions_output_path)

        return predictor

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_heatmap(self, prob_map, out_dir, base):
        # ENSURE FLOAT32
        prob_map = prob_map.astype(np.float32)

        # NORMALIZE PROBABILITY HEATMAP [0-1] TO [0–255]
        min_val, max_val = prob_map.min(), prob_map.max()
        if max_val > min_val:
            norm_map = ((prob_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            norm_map = (prob_map * 255).astype(np.uint8)  # fallback if flat

        heatmap_dir = os.path.normpath(os.path.join(out_dir, "heatmaps"))
        os.makedirs(heatmap_dir, exist_ok=True)

        gray_path = os.path.join(heatmap_dir, f"{base}_heatmap_gray.png")
        cv2.imwrite(gray_path, norm_map)

        jet_path = os.path.join(heatmap_dir, f"{base}_heatmap_jet.png")
        cv2.imwrite(jet_path, cv2.applyColorMap(norm_map, cv2.COLORMAP_JET))

        print(f"Saved heatmaps: {gray_path}, {jet_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_panel(self, img, pred, prob_map, out_dir, base):
        """
        Create a 2x2 composite panel:
          [0,0] Original image
          [0,1] Overlay (mask on original)
          [1,0] Binary mask
          [1,1] Probability heatmap
        """
        try:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

            # Original
            axs[0, 0].imshow(img)
            axs[0, 0].set_title("Original")
            axs[0, 0].axis("off")

            # Overlay
            overlay = img.copy()
            overlay[pred == 1] = (0, 150, 255)
            blended = (0.6 * img + 0.4 * overlay).astype(np.uint8)
            axs[0, 1].imshow(blended)
            axs[0, 1].set_title("Overlay")
            axs[0, 1].axis("off")

            # Binary mask
            axs[1, 0].imshow(pred, cmap="gray")
            axs[1, 0].set_title("Binary Mask")
            axs[1, 0].axis("off")

            # Heatmap
            axs[1, 1].imshow(prob_map, cmap="jet")
            axs[1, 1].set_title("Probability Heatmap")
            axs[1, 1].axis("off")

            panel_path = os.path.normpath(os.path.join(out_dir, "panels"))
            os.makedirs(panel_path, exist_ok=True)
            output_file = os.path.join(panel_path, f"{base}_panel.png")

            plt.tight_layout()
            plt.savefig(output_file)

            #print(f"Saved side-by-side panel: {output_file}")
        finally:
            plt.close()
