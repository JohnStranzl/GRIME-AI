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
from PyQt5.QtWidgets import QMessageBox


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===        class SAM2InferenceEngine       ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class SAM2InferenceEngine:

    def __init__(self, device, model_cfg, trained_checkpoint_path, input_dir, output_dir):
        # =========================================================================
        # TEST MODE CONFIGURATION
        # Set self.TEST_MODE = True to run inference on hardcoded water + sky with colors
        # Set self.TEST_MODE = False to use normal configuration
        # =========================================================================
        self.TEST_MODE = False  # CHANGE THIS to True to enable test mode
        self.TEST_CATEGORIES = [
        {"id": 1, "name": "water", "color": (255, 255, 0)},  # Yellow
        {"id": 2, "name": "sky", "color": (255, 165, 0)}     # Orange
        ]
        # =========================================================================

        # =========================================================================
        # LABEL DISPLAY CONFIGURATION
        # Set self.DISPLAY_LABELS = True to show category names on overlay images
        # Set self.DISPLAY_LABELS = False to hide labels
        # =========================================================================
        self.DISPLAY_LABELS = True  # CHANGE THIS to False to hide labels
        # =========================================================================

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
        self.predictions_output_path = output_dir + " (sam2)"

        self.category_centroids = {}
        self.target_category_name = None

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

        # LOAD CATEGORY CENTROIDS METADATA
        self.category_centroids = checkpoint.get("category_centroids", {})
        print(f"Loaded {sum(len(v) for v in self.category_centroids.values())} centroids across "
              f"{len(self.category_centroids)} categories")

        # Store target category name from checkpoint
        self.target_category_name = checkpoint.get("target_category_name", None)
        if self.target_category_name:
            print(f"Model trained on category: {self.target_category_name}")

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
    # RUN PREDICTION WITH CENTROID PROMPTS
    def predict_with_centroids(self, predictor, image_array, category_id, multimask_output=False):
        """
        Run prediction using centroids as prompts.
        Uses POSITIVE prompts from target category and NEGATIVE prompts from all other categories.
        This matches the training approach for better generalization.
        """
        predictor.set_image(image_array)

        h, w = image_array.shape[:2]
        
        # ============================================================
        # POSITIVE PROMPTS (target category)
        # ============================================================
        positive_centroids = self.category_centroids.get(int(category_id), [])
        if not positive_centroids:
            # No centroids for this category - model wasn't trained on it
            print(f"Warning: No centroids found for category ID {category_id}. Model may not be trained for this category.")
            return None, None, None

        # Denormalize positive centroids to pixel coords
        positive_coords = []
        for entry in positive_centroids:
            if isinstance(entry, dict):
                cx_norm, cy_norm = entry["centroid_norm"]
            else:
                cx_norm, cy_norm = entry
            cx_px = int(round(cx_norm * (w - 1)))
            cy_px = int(round(cy_norm * (h - 1)))
            positive_coords.append([cx_px, cy_px])
        
        # ============================================================
        # NEGATIVE PROMPTS (all other categories)
        # ============================================================
        negative_coords = []

        for cat_id, centroids in self.category_centroids.items():
            # Skip the target category
            if int(cat_id) == int(category_id):
                continue

            # Denormalize negative centroids
            for entry in centroids:
                if isinstance(entry, dict):
                    cx_norm, cy_norm = entry["centroid_norm"]
                else:
                    cx_norm, cy_norm = entry
                cx_px = int(round(cx_norm * (w - 1)))
                cy_px = int(round(cy_norm * (h - 1)))
                negative_coords.append([cx_px, cy_px])

        # Balance negatives (same as training)
        if len(negative_coords) > len(positive_coords) * 3:
            import random
            negative_coords = random.sample(negative_coords, len(positive_coords) * 3)
        
        # ============================================================
        # COMBINE POSITIVE AND NEGATIVE PROMPTS
        # ============================================================
        all_coords = positive_coords + negative_coords
        all_labels = [1] * len(positive_coords) + [0] * len(negative_coords)

        point_coords = torch.tensor(all_coords, device=self.device, dtype=torch.float32)
        point_labels = torch.tensor(all_labels, device=self.device, dtype=torch.int64)

        # Run prediction with both positive and negative prompts
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )

        if len(scores) == 0:
            return None, None, None
        if multimask_output:
            best_idx = int(np.argmax(scores))
            return masks[best_idx], scores[best_idx], logits[best_idx]
        return masks[0], scores[0], logits[0]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_masks(self, output_file_with_path, image, mask, borders=True, category_id=None, category_name=None, display_label=True):
        fig, ax = plt.subplots()
        try:
            ax.imshow(image)
            ax.axis('off')
            self.show_mask(mask, ax, category_id=category_id, borders=borders, category_name=category_name, display_label=display_label)
            fig.savefig(output_file_with_path, bbox_inches='tight', pad_inches=0)
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_mask(self, mask, ax, category_id=None, borders=True, category_name=None, display_label=True):
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

        # Add category label at center of mask (if enabled)
        if display_label and category_name:
            # Find the largest connected component
            num_labels, labels = cv2.connectedComponents(mask)
            if num_labels > 1:  # 0 is background
                # Find largest component (excluding background)
                largest_component = 1
                largest_size = 0
                for label_id in range(1, num_labels):
                    size = np.sum(labels == label_id)
                    if size > largest_size:
                        largest_size = size
                        largest_component = label_id
                
                # Calculate centroid of largest component
                ys, xs = np.where(labels == largest_component)
                if len(xs) > 0 and len(ys) > 0:
                    center_x = xs.mean()
                    center_y = ys.mean()
                    
                    # Draw text at centroid
                    ax.text(center_x, center_y, category_name.upper(),
                           fontsize=7, fontweight='bold',
                           color='white', ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.25',
                                   facecolor='black', alpha=0.25, edgecolor='none'))


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
    def save_outputs(self, image_path, pil_image, mask, prob_map, score, save_masks, copy_original_image, category_id, category_name=None, display_label=True):
        base = os.path.splitext(os.path.basename(image_path))[0]
        overlay_path = os.path.join(self.predictions_output_path, f"{base}_overlay.png")
        self.show_masks(overlay_path, pil_image, mask, borders=False, category_id=category_id, category_name=category_name, display_label=display_label)

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
        
        # Validate that selected categories exist in model
        if not self.TEST_MODE and selected_label_categories:
            requested_category = selected_label_categories[0]
            requested_id = requested_category.get("id")
            requested_name = requested_category.get("name", "unknown")
            
            # Check if this category has centroids in the model
            if int(requested_id) not in self.category_centroids:
                # Close progress bar before showing dialog
                if progressBar is not None and progressBar.isVisible():
                    progressBar.close()
                
                # Build available categories message
                if self.target_category_name:
                    available_msg = f"This model was trained on: '{self.target_category_name}'"
                else:
                    available_msg = f"Available category IDs in model: {list(self.category_centroids.keys())}"
                
                QMessageBox.critical(
                    None,
                    "Category Not Found in Model",
                    f"The selected category '{requested_name}' (ID: {requested_id}) was not found in the trained model.\n\n"
                    f"{available_msg}\n\n"
                    f"Please select a different model or category."
                )
                return None
            
            # Check if centroids list is empty
            if not self.category_centroids.get(int(requested_id)):
                # Close progress bar before showing dialog
                if progressBar is not None and progressBar.isVisible():
                    progressBar.close()
                
                QMessageBox.critical(
                    None,
                    "No Training Data for Category",
                    f"The selected category '{requested_name}' exists in the model but has no centroid data.\n\n"
                    f"This indicates the model was not properly trained on this category.\n\n"
                    f"Please use a different model."
                )
                return None
        
        if self.TEST_MODE:
            # === TEST MODE: Multi-category inference with colored output ===
            print("\n" + "=" * 70)
            print("TEST MODE ENABLED: Running inference on", [c["name"] for c in self.TEST_CATEGORIES])
            print("=" * 70 + "\n")
            
            return self._run_test_mode_inference(predictor, copy_original_image, save_masks, progressBar)
        else:
            # === NORMAL MODE: Standard single-category inference ===
            return self._run_normal_inference(predictor, copy_original_image, save_masks, 
                                             selected_label_categories, progressBar)
    
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _run_normal_inference(self, predictor, copy_original_image, save_masks, 
                             selected_label_categories, progressBar):
        """Normal inference mode - processes single category."""
        coco_data = init_coco_structure(selected_label_categories)

        os.makedirs(self.predictions_output_path, exist_ok=True)
        VALID_EXTS = ('.jpg', '.jpeg')
        images_list = [f for f in os.listdir(self.segmentation_images_path) if f.lower().endswith(VALID_EXTS)]
        if progressBar is not None:
            progressBar.setRange(0, len(images_list) + 1)

        image_id = 1
        annotation_id = 1

        # Track success/failure counts
        images_found = 0
        images_not_found = 0
        images_processed = 0
        total_images_in_folder = len(images_list)
        target_category_name = selected_label_categories[0].get("name", "unknown") if selected_label_categories else "unknown"
        cancelled = False

        for img_index, image in enumerate(images_list):
            # Update progress at start of iteration
            if progressBar is not None and progressBar.isVisible():
                progressBar.setValue(img_index)
            
            # Check for cancellation
            if progressBar is not None:
                if hasattr(progressBar, 'wasCanceled') and progressBar.wasCanceled():
                    print("Inference cancelled by user")
                    cancelled = True
                    break
                # If progress bar is closed/hidden, treat as cancellation
                if not progressBar.isVisible():
                    print("Inference stopped - progress bar closed")
                    cancelled = True
                    break

            image_path = os.path.join(self.segmentation_images_path, image)

            try:
                pil_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                images_not_found += 1
                images_processed += 1
                continue

            image_array = np.array(pil_image)

            multimask_output = False
            category_id = selected_label_categories[0]["id"] if selected_label_categories else 2
            mask, score, logits = self.predict_with_centroids(
                predictor, image_array, category_id, multimask_output=multimask_output
            )
            if mask is None:
                images_not_found += 1
                images_processed += 1
                continue

            # ENSURE THE MASK MATCHES THE IMAGE SIZE
            target_h, target_w = image_array.shape[:2]
            if mask.shape != (target_h, target_w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST)

            # ============================================================
            # FILTER BLOBS NOT NEAR CENTROID PROMPTS (matches training)
            # ============================================================
            # Get positive centroids for this category
            positive_centroids = self.category_centroids.get(int(category_id), [])
            
            if positive_centroids:
                # Denormalize centroids
                centroid_coords = []
                for entry in positive_centroids:
                    if isinstance(entry, dict):
                        cx_norm, cy_norm = entry["centroid_norm"]
                    else:
                        cx_norm, cy_norm = entry
                    cx_px = int(round(cx_norm * (target_w - 1)))
                    cy_px = int(round(cy_norm * (target_h - 1)))
                    centroid_coords.append([cx_px, cy_px])
                
                # Filter blobs using connected components
                labels_np = mask.astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(labels_np)
                valid_mask = np.zeros_like(labels, dtype=np.uint8)
                
                # Adaptive radius threshold (same as training)
                img_diagonal = np.sqrt(target_h * target_h + target_w * target_w)
                radius_threshold = max(10, int(0.025 * img_diagonal))
                
                for lbl in range(1, num_labels):  # skip background
                    ys, xs = np.nonzero(labels == lbl)
                    if xs.size == 0:
                        continue
                    cx_blob, cy_blob = xs.mean(), ys.mean()
                    
                    # Keep blob if near any centroid prompt
                    for cx, cy in centroid_coords:
                        if np.linalg.norm([cx - cx_blob, cy - cy_blob]) < radius_threshold:
                            valid_mask[labels == lbl] = 1
                            break
                
                # Replace mask with filtered version
                mask = valid_mask

            # Check if mask has any foreground pixels
            if np.sum(mask) == 0:
                # Mask is empty - category not found in this image
                images_not_found += 1
                images_processed += 1
                continue

            # Resize logits to image size for heatmap
            if logits is not None:
                prob_map = cv2.resize(logits.astype(np.float32), (target_w, target_h),
                                      interpolation=cv2.INTER_LINEAR)
                base = os.path.splitext(os.path.basename(image_path))[0]
                self._save_heatmap(prob_map, self.predictions_output_path, base)

            category_id = selected_label_categories[0]["id"] if selected_label_categories else 2
            category_name = selected_label_categories[0].get("name", "unknown") if selected_label_categories else "unknown"

            # Save overlay/mask outputs
            self.save_outputs(image_path, pil_image, mask, prob_map, score, save_masks, copy_original_image,
                              category_id, category_name=category_name, display_label=self.DISPLAY_LABELS)

            # COCO bookkeeping
            add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id)

            image_id += 1
            annotation_id += 1
            images_found += 1  # Successfully segmented this image
            images_processed += 1

            # Clear GPU cache periodically to prevent memory accumulation
            if torch.cuda.is_available() and img_index % 10 == 0:
                torch.cuda.empty_cache()

        if progressBar is not None and progressBar.isVisible():
            progressBar.close()
        
        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_coco_json(coco_data, self.predictions_output_path)

        # Show summary dialog
        if cancelled:
            # Inference was cancelled
            QMessageBox.warning(
                None,
                "Inference Cancelled",
                f"Inference was cancelled by user.\n\n"
                f"Results for '{target_category_name}':\n"
                f"- Total images in folder: {total_images_in_folder}\n"
                f"- Images processed: {images_processed}\n"
                f"- Category found in: {images_found} images\n"
                f"- Category not found in: {images_not_found} images"
            )
        elif images_not_found == 0 and images_found == images_processed:
            # All processed images successfully segmented
            QMessageBox.information(
                None,
                "Inference Complete",
                f"Category '{target_category_name}' was found in all {images_processed} images."
            )
        elif images_found == 0:
            # No images found
            QMessageBox.warning(
                None,
                "No Segmentations Found",
                f"The category '{target_category_name}' was not found in any of the {images_processed} images processed.\n\n"
                f"This may indicate the model was not trained on this category."
            )
        else:
            # Partial success
            QMessageBox.information(
                None,
                "Inference Complete",
                f"Segmentation results for '{target_category_name}':\n\n"
                f"- Category found in: {images_found} images\n"
                f"- Category not found in: {images_not_found} images\n"
                f"- Total processed: {images_processed} images"
            )

        return predictor

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _run_test_mode_inference(self, predictor, copy_original_image, save_masks, progressBar):
        """Test mode inference - processes multiple categories with colored output."""
        coco_data = init_coco_structure(self.TEST_CATEGORIES)

        os.makedirs(self.predictions_output_path, exist_ok=True)
        VALID_EXTS = ('.jpg', '.jpeg')
        images_list = [f for f in os.listdir(self.segmentation_images_path) if f.lower().endswith(VALID_EXTS)]
        if progressBar is not None:
            progressBar.setRange(0, len(images_list) + 1)

        image_id = 1
        annotation_id = 1

        for img_index, image in enumerate(images_list):
            # Check for cancellation
            if progressBar is not None:
                if hasattr(progressBar, 'wasCanceled') and progressBar.wasCanceled():
                    print("Inference cancelled by user")
                    break
                # If progress bar is closed/hidden, treat as cancellation
                if not progressBar.isVisible():
                    print("Inference stopped - progress bar closed")
                    break
                    progressBar.setValue(img_index)

            image_path = os.path.join(self.segmentation_images_path, image)

            try:
                pil_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                continue

            image_array = np.array(pil_image)
            
            # Create composite image for both categories
            composite_overlay = image_array.copy()
            
            # Process each category
            for category in self.TEST_CATEGORIES:
                category_id = category["id"]
                category_name = category["name"]
                category_color = category["color"]
                
                print(f"  Processing {category_name} for {os.path.basename(image_path)}")
                
                multimask_output = False
                mask, score, logits = self.predict_with_centroids(
                    predictor, image_array, category_id, multimask_output=multimask_output
                )
                
                if mask is None:
                    print(f"    No mask for {category_name}")
                    continue

                # Ensure mask matches image size
                target_h, target_w = image_array.shape[:2]
                if mask.shape != (target_h, target_w):
                    mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                
                # Apply category-specific color to overlay
                composite_overlay[mask == 1] = category_color
                
                # Add label text at center of largest blob
                labels_np = mask.astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(labels_np)
                if num_labels > 1:
                    # Find largest blob
                    largest_blob = 0
                    largest_size = 0
                    for lbl in range(1, num_labels):
                        size = np.sum(labels == lbl)
                        if size > largest_size:
                            largest_size = size
                            largest_blob = lbl
                    
                    if largest_blob > 0:
                        # Get center of largest blob
                        ys, xs = np.nonzero(labels == largest_blob)
                        center_x = int(xs.mean())
                        center_y = int(ys.mean())
                        
                        # Add text label
                        cv2.putText(
                            composite_overlay,
                            category_name.upper(),
                            (center_x - 50, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 255, 255),  # White text
                            3,
                            cv2.LINE_AA
                        )
            
            # Save the composite overlay image
            base = os.path.splitext(os.path.basename(image_path))[0]
            composite_path = os.path.join(self.predictions_output_path, f"{base}_composite.png")
            cv2.imwrite(composite_path, cv2.cvtColor(composite_overlay, cv2.COLOR_RGB2BGR))
            
            if copy_original_image:
                shutil.copy(image_path, os.path.join(self.predictions_output_path, os.path.basename(image_path)))

            image_id += 1
            annotation_id += 1
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and img_index % 10 == 0:
                torch.cuda.empty_cache()

        if progressBar is not None and progressBar.isVisible():
            progressBar.close()
        
        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_coco_json(coco_data, self.predictions_output_path)

        return predictor


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
