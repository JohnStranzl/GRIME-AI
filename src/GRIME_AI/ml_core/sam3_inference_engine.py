#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sam3_inference_engine.py

import os
import cv2
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# ===== Local infra =====
from GRIME_AI.ml_core.ml_helpers import (
    get_color_for_category, init_coco_structure, add_coco_entries, save_coco_json
)

# ===== Import adapter/predictor =====
from sam3_adapter import SAM3Adapter, SAM3ImagePredictor


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===        class SAM3InferenceEngine       ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class SAM3InferenceEngine:
    def __init__(self, device, model_cfg, checkpoint, model_path, input_dir, output_dir):
        self.device = device
        self.MODEL_CFG = model_cfg
        self.SAM3_CHECKPOINT = checkpoint
        self.SAM2_MODEL = model_path
        self.segmentation_images_path = input_dir
        self.predictions_output_path = output_dir

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_sam3_model(self):
        config_dir = os.path.dirname(self.MODEL_CFG)
        model_cfg_name = os.path.basename(self.MODEL_CFG)
        print("Model config path:", os.path.join(config_dir, model_cfg_name))
        print("Checkpoint path:", self.SAM3_CHECKPOINT)

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize(config_path=config_dir, version_base=None):
            cfg_intern = compose(config_name=model_cfg_name)
            raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)
            for key in ["no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs", "device"]:
                raw_model_cfg.pop(key, None)
            new_cfg = OmegaConf.create(raw_model_cfg)

        dev = torch.device(self.device if isinstance(self.device, str) else self.device)
        sam3_adapter = SAM3Adapter(model_cfg=new_cfg, checkpoint_path=self.SAM3_CHECKPOINT, device=dev)
        sam3_model = sam3_adapter.to(dev).eval()
        predictor = SAM3ImagePredictor(sam3_model)

        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, self.SAM2_MODEL)
        checkpoint = torch.load(
            model_path,
            map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            predictor.model.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model_keys = set(predictor.model.model.state_dict().keys())
            clean_ckpt = {k: v for k, v in checkpoint.items() if k in model_keys}
            predictor.model.model.load_state_dict(clean_ckpt, strict=False)

        return predictor

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def predict_sam3(self, predictor, image_array, multimask_output=False):
        predictor.set_image(image_array)
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=multimask_output
        )
        if masks is None or len(scores) == 0:
            return None, None, None
        if multimask_output:
            best_idx = int(np.argmax(scores))
            return masks[best_idx], scores[best_idx], logits[best_idx]
        else:
            return masks[0], scores[0], logits[0]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_masks(self, output_file_with_path, image, mask, borders=True, category_id=None):
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')
        self.show_mask(mask, ax, category_id=category_id, borders=borders)
        fig.savefig(output_file_with_path, bbox_inches='tight', pad_inches=0)
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

        self._save_panel(np.array(pil_image), mask, prob_map, self.predictions_output_path, base)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def run_sam3_inference(self, copy_original_image, save_masks, selected_label_categories, progressBar):
        predictor = self.load_sam3_model()
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
            pil_image = Image.open(image_path).convert("RGB")
            image_array = np.array(pil_image)

            multimask_output = False
            mask, score, logits = self.predict_sam3(predictor, image_array, multimask_output=multimask_output)
            if mask is None:
                continue

            if mask.shape != image_array.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image_array.shape[1], image_array.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            prob_map = None
            if logits is not None:
                prob_map = cv2.resize(
                    logits.astype(np.float32),
                    (image_array.shape[1], image_array.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                base = os.path.splitext(os.path.basename(image_path))[0]
                self._save_heatmap(prob_map, self.predictions_output_path, base)

            category_id = selected_label_categories[0]["id"] if selected_label_categories else 2
            self.save_outputs(image_path, pil_image, mask, prob_map, score, save_masks, copy_original_image, category_id)

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
        # Ensure float32
        prob_map = prob_map.astype(np.float32)

        # Normalize to [0, 255]
        min_val, max_val = prob_map.min(), prob_map.max()
        if max_val > min_val:
            norm_map = ((prob_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            norm_map = (prob_map * 255).astype(np.uint8)

        heatmap_dir = os.path.normpath(os.path.join(out_dir, "heatmaps"))
        os.makedirs(heatmap_dir, exist_ok=True)

        gray_path = os.path.join(heatmap_dir, f"{base}_heatmap_gray.png")
        cv2.imwrite(gray_path, norm_map)

        jet_path = os.path.join(heatmap_dir, f"{base}_heatmap_jet.png")
        cv2.imwrite(jet_path, cv2.applyColorMap(norm_map, cv2.COLORMAP_JET))

        print(f"Saved heatmaps: {gray_path}, {jet_path}")

    def _save_panel(self, img, pred, prob_map, out_dir, base, category_id=None):
        """
        Create a 2x2 composite panel:
          [0,0] Original image
          [0,1] Overlay (mask on original, category‑specific color)
          [1,0] Binary mask
          [1,1] Probability heatmap
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Original
        axs[0, 0].imshow(img)
        axs[0, 0].set_title("Original")
        axs[0, 0].axis("off")

        # Overlay with category‑specific color
        overlay = img.copy()
        if category_id is not None:
            color = get_color_for_category(category_id)[:3]  # take RGB from RGBA
        else:
            color = np.array([0, 150, 255])  # fallback
        overlay[pred == 1] = color
        blended = (0.6 * img + 0.4 * overlay).astype(np.uint8)
        axs[0, 1].imshow(blended)
        axs[0, 1].set_title("Overlay")
        axs[0, 1].axis("off")

        # Binary mask
        axs[1, 0].imshow(pred, cmap="gray")
        axs[1, 0].set_title("Binary Mask")
        axs[1, 0].axis("off")

        # Heatmap
        if prob_map is not None:
            axs[1, 1].imshow(prob_map, cmap="jet")
            axs[1, 1].set_title("Probability Heatmap")
        else:
            axs[1, 1].imshow(np.zeros_like(img[..., 0]), cmap="gray")
            axs[1, 1].set_title("Probability Heatmap (n/a)")
        axs[1, 1].axis("off")

        panel_path = os.path.normpath(os.path.join(out_dir, "panels"))
        os.makedirs(panel_path, exist_ok=True)
        output_file = os.path.join(panel_path, f"{base}_panel.png")

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        print(f"Saved side-by-side panel: {output_file}")
