#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Nov 18, 2025
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# segformer_inference_engine.py

import os
import cv2
import torch
import numpy as np
import random
import shutil
from pathlib import Path

from PIL import Image

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, prevents GUI windows
import matplotlib.pyplot as plt

import torchvision.transforms as T
from transformers import SegformerForSemanticSegmentation
from peft import LoraConfig, get_peft_model
from GRIME_AI.ml_core.ml_helpers import (init_coco_structure, add_coco_entries, save_coco_json)


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===     class SegFormerInferenceEngine     ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class SegFormerInferenceEngine:
    def __init__(self, device, segformer_model, input_dir, output_dir,
                 image_size: int = 512, threshold: float = 0.2):
        # Original attributes
        self.device = device
        self.SEGFORMER_MODEL = segformer_model
        self.segmentation_images_path = input_dir
        self.predictions_output_path = output_dir

        # Parameters
        self.image_size = image_size
        self.threshold = threshold

        # Transforms
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.to_tensor = T.ToTensor()

        # Load model
        self.model = self._load_model()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _load_model(self):
        if not self.SEGFORMER_MODEL or not os.path.exists(self.SEGFORMER_MODEL):
            raise FileNotFoundError(f"SegFormer model checkpoint not found: {self.SEGFORMER_MODEL}")

        base = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            ignore_mismatched_sizes=True
        )
        base.config.num_labels = 2
        base.decode_head.classifier = torch.nn.Conv2d(
            base.decode_head.classifier.in_channels, 2, kernel_size=1
        )

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["query", "key", "value", "proj"],
            modules_to_save=["decode_head.classifier"],
        )
        model = get_peft_model(base, lora_cfg)

        ckpt = torch.load(self.SEGFORMER_MODEL, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(device).eval()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def segment_image(self, image_path: str, out_path: str, gt_mask_path: str = None):
        # --- Load original image ---
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(image_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # --- Resize for model input ---
        img_resized = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        x = self.to_tensor(img_resized)
        x = self.normalize(x).unsqueeze(0).to(str(self.device))

        # --- Forward pass ---
        logits = self.model(pixel_values=x).logits
        logits = torch.nn.functional.interpolate(
            logits, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )

        probs = torch.softmax(logits, dim=1)
        water_prob_resized = probs[0, 1].cpu().numpy()  # model-size probability map
        pred_resized = (water_prob_resized > self.threshold).astype(np.uint8)

        # --- Map predictions back to original image size ---
        pred = cv2.resize(pred_resized, (w, h), interpolation=cv2.INTER_NEAREST)
        water_prob = cv2.resize(water_prob_resized, (w, h), interpolation=cv2.INTER_LINEAR)

        # --- Prepare output paths ---
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        # --- Save outputs aligned to original image ---
        self._save_overlay(img, pred, out_path)
        self._save_mask(pred, out_dir, base)
        self._save_heatmaps(torch.tensor(water_prob), out_dir, base)
        self._save_panel(img, pred, torch.tensor(water_prob), out_dir, base)
        if gt_mask_path:
            self._save_error_map(img, pred, gt_mask_path, out_dir, base)
        self._save_components(img, pred, out_dir, base)

        # Return for downstream COCO bookkeeping
        return pred, water_prob

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def run_segformer_inference(self, copy_original_image, save_masks, selected_label_categories, progressBar):
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

            # Engine overlays/masks
            out_overlay_path = os.path.join(self.predictions_output_path, f"{Path(image).stem}_overlay.png")
            pred, water_prob = self.segment_image(image_path, out_overlay_path)

            # Optional copy of original image
            if copy_original_image:
                shutil.copy(image_path, os.path.join(self.predictions_output_path, os.path.basename(image_path)))

            # COCO entries
            pil_image = Image.open(image_path).convert("RGB")
            image_array = np.array(pil_image)

            # Threshold using the class's threshold to keep behavior consistent
            mask = (water_prob > self.threshold).astype(np.uint8)
            # Score kept for parity with Block B's computation
            score = float(np.mean(water_prob))  # not used by add_coco_entries, retained for completeness

            add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id)
            image_id += 1
            annotation_id += 1

        if progressBar is not None and progressBar.isVisible():
            progressBar.close()

        save_coco_json(coco_data, self.predictions_output_path)
        return self  # matches Block B's return of the engine instance

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_overlay(self, orig_img, pred, out_path):
        h, w = orig_img.shape[:2]
        pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = orig_img.copy()
        overlay[pred_resized == 1] = (0, 150, 255)

        blended = (0.6 * orig_img + 0.4 * overlay).astype(np.uint8)
        out_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        cv2.imwrite(out_path, out_bgr)
        print(f"Saved overlay: {out_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_mask(self, pred, out_dir, base):
        mask_path = os.path.join(out_dir, f"{base}_mask.png")
        cv2.imwrite(mask_path, pred * 255)
        print(f"Saved mask: {mask_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_heatmaps(self, water_prob, out_dir, base):
        prob_map = (water_prob.cpu().numpy() * 255).astype(np.uint8)

        prob_path = os.path.normpath(os.path.join(out_dir, "probability maps"))
        os.makedirs(prob_path, exist_ok=True)

        gray_path = os.path.join(prob_path, f"{base}_prob_gray.png")
        cv2.imwrite(gray_path, prob_map)
        print(f"Saved probability heatmap (gray): {gray_path}")

        jet_path = os.path.join(prob_path, f"{base}_prob_jet.png")
        cv2.imwrite(jet_path, cv2.applyColorMap(prob_map, cv2.COLORMAP_JET))
        print(f"Saved probability heatmap (jet): {jet_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_panel(self, img, pred, prob_map, out_dir, base):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0,0].imshow(img); axs[0,0].set_title("Original"); axs[0,0].axis("off")
        overlay = img.copy(); overlay[pred == 1] = (0, 150, 255)

        blended = (0.6 * img + 0.4 * overlay).astype(np.uint8)

        axs[0,1].imshow(blended); axs[0,1].set_title("Overlay"); axs[0,1].axis("off")
        axs[1,0].imshow(pred, cmap="gray"); axs[1,0].set_title("Binary Mask"); axs[1,0].axis("off")
        axs[1,1].imshow(prob_map.cpu().numpy(), cmap="jet"); axs[1,1].set_title("Probability Heatmap"); axs[1,1].axis("off")

        panel_path = os.path.normpath(os.path.join(out_dir, "panels"))
        os.makedirs(panel_path, exist_ok=True)

        output_file = os.path.join(panel_path, f"{base}_panel.png")

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        print(f"Saved side-by-side panel: {panel_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_error_map(self, img, pred, gt_path, out_dir, base):
        if not os.path.exists(gt_path):
            return
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        gt = (gt > 127).astype(np.uint8)
        error_map = np.zeros_like(img)
        error_map[(gt == 1) & (pred == 1)] = (0, 255, 0)
        error_map[(gt == 1) & (pred == 0)] = (255, 0, 0)
        error_map[(gt == 0) & (pred == 1)] = (255, 0, 255)
        error_path = os.path.join(out_dir, f"{base}_error.png")
        cv2.imwrite(error_path, cv2.cvtColor(error_map, cv2.COLOR_RGB2BGR))
        print(f"Saved error map: {error_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_components(self, img, pred, out_dir, base):
        num_labels, labels = cv2.connectedComponents(pred)
        cc_vis = np.zeros_like(img)
        for lbl in range(1, num_labels):
            mask = labels == lbl
            color = [random.randint(0,255) for _ in range(3)]
            cc_vis[mask] = color

        component_path = os.path.normpath(os.path.join(out_dir, "mask components"))
        os.makedirs(component_path, exist_ok=True)

        cc_path = os.path.join(component_path, f"{base}_components.png")

        cv2.imwrite(cc_path, cv2.cvtColor(cc_vis, cv2.COLOR_RGB2BGR))
        print(f"Saved connected components visualization: {cc_path}")
