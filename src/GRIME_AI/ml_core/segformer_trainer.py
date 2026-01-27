#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, prevents GUI windows
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from GRIME_AI.ml_core.coco_segmentation_datasets import MultiCocoTargetDataset
from GRIME_AI.ml_core.lora_segmentation_losses import BinaryDiceLoss, MultiClassDiceLoss
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
import torchvision.transforms.functional as TF

# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====      AUGMENTATION      =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
def apply_augmentation(image, mask, cfg):
    """
    Apply data augmentation to image and mask together.
    Applies same geometric transforms to both image and mask.
    
    Args:
        image: PIL Image or Tensor [C,H,W]
        mask: Tensor [H,W]
        cfg: SegFormerConfig with augmentation parameters
    """
    # Convert to PIL if needed
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image)
    
    # Random horizontal flip
    if random.random() < cfg.aug_horizontal_flip:
        image = TF.hflip(image)
        mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
    
    # Random vertical flip
    if random.random() < cfg.aug_vertical_flip:
        image = TF.vflip(image)
        mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
    
    # Random rotation
    if cfg.aug_rotation > 0:
        angle = random.uniform(-cfg.aug_rotation, cfg.aug_rotation)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask.unsqueeze(0), angle).squeeze(0)
    
    # Color jitter (image only, not mask)
    if cfg.aug_brightness > 0 or cfg.aug_contrast > 0:
        brightness_factor = 1.0 + random.uniform(-cfg.aug_brightness, cfg.aug_brightness)
        contrast_factor = 1.0 + random.uniform(-cfg.aug_contrast, cfg.aug_contrast)
        image = TF.adjust_brightness(image, brightness_factor)
        image = TF.adjust_contrast(image, contrast_factor)
    
    # Convert back to tensor
    image = TF.to_tensor(image)
    
    return image, mask

# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====      HELPER FUNCTIONS      =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
GLOBAL_SEED = 42

def _worker_init_fn(worker_id):
    np.random.seed(GLOBAL_SEED + worker_id)
    random.seed(GLOBAL_SEED + worker_id)
    torch.manual_seed(GLOBAL_SEED + worker_id)

# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====   class SegFormerConfig    =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
@dataclass
class SegFormerConfig:
    images_dir: str = ""
    ann_path: str = ""
    categories: Optional[List[str]] = field(default_factory=list)
    target_category_name: str = ""
    image_size: int = 512
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    val_every: int = 1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs_segformer"
    amp: bool = True
    grad_clip_norm: float = 1.0

    # Early stopping
    early_stopping: bool = False
    patience: int = 10

    # Checkpoint management
    max_best_checkpoints: int = 3

    # Validation overlays
    save_val_overlays: bool = True
    num_val_overlays: int = 5  # Save overlays for first N validation images

    # Inference threshold (should match inference engine)
    inference_threshold: float = 0.2  # Probability threshold for positive prediction
    
    # Multi-class support
    target_class_id: int = 1  # Which class to evaluate/visualize (for multi-class datasets)
    
    # Data augmentation (for training only)
    use_augmentation: bool = True
    aug_horizontal_flip: float = 0.5  # Probability of horizontal flip
    aug_vertical_flip: float = 0.5    # Probability of vertical flip
    aug_rotation: float = 15.0        # Max rotation degrees (+/-)
    aug_brightness: float = 0.2       # Brightness adjustment range
    aug_contrast: float = 0.2         # Contrast adjustment range


# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====   class SegFormerTrainer   =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class SegFormerTrainer:
    """
    Pure SegFormer trainer: builds model, data loaders, runs training/eval, saves checkpoints/curves.
    No LoRA logic inside. If you want LoRA, wrap the model externally before calling train().
    """
    def __init__(self, cfg: SegFormerConfig):
        self.cfg = cfg
        self.best_iou = 0.0
        self.progressBar = None
        self._last_checkpoint_path = None
        self._progress_total = 0

        # Early stopping tracking
        self.patience_counter = 0
        self.best_val_iou = 0.0

        # Checkpoint management
        self.best_checkpoints = []  # List of (val_iou, filepath) tuples

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def build_model(self, num_labels: int) -> nn.Module:
        """
        Build base SegFormer (no LoRA). If you want LoRA, wrap the returned model externally.
        """
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            ignore_mismatched_sizes=True
        )
        model.config.num_labels = num_labels
        model.config.id2label = {i: f"class_{i}" for i in range(num_labels)}
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}
        model.decode_head.classifier = nn.Conv2d(
            model.decode_head.classifier.in_channels, num_labels, kernel_size=1
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(device).train()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        inter = ((pred == 1) & (target == 1)).sum().item()
        union = ((pred == 1) | (target == 1)).sum().item()
        return (inter / union) if union > 0 else 1.0

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        tp = ((pred == 1) & (target == 1)).sum().item()
        fp = ((pred == 1) & (target == 0)).sum().item()
        fn = ((pred == 0) & (target == 1)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, model, val_loader):
        model.eval()
        ious, precisions, recalls, f1s = [], [], [], []
        for imgs, masks in val_loader:
            imgs = imgs.to(self.cfg.device, non_blocking=True)
            masks = masks.to(self.cfg.device, non_blocking=True)

            outputs = model(pixel_values=imgs)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            
            # Use threshold-based prediction (works for both binary and multi-class)
            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, self.cfg.target_class_id] > self.cfg.inference_threshold).long()

            for b in range(preds.size(0)):
                pb = preds[b].cpu()
                mb = (masks[b] == self.cfg.target_class_id).long().cpu()  # Convert mask to binary for target class
                iou = self.compute_iou(pb, mb)
                prec, rec, f1 = self.compute_metrics(pb, mb)
                ious.append(iou); precisions.append(prec); recalls.append(rec); f1s.append(f1)

            self._tick_progress()

        torch.use_deterministic_algorithms(True)

        return {
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "f1": float(np.mean(f1s)) if f1s else 0.0,
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_validation_overlay(self, epoch, images, pred_masks, true_masks, output_dir):
        """Save validation overlays showing prediction vs ground truth (SAM2 style)"""
        overlay_dir = os.path.join(output_dir, "validation_overlays")
        os.makedirs(overlay_dir, exist_ok=True)

        num_samples = min(self.cfg.num_val_overlays, len(images))

        for i in range(num_samples):
            img = images[i].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            pred = pred_masks[i].cpu().numpy()
            true = true_masks[i].cpu().numpy()

            # Denormalize image (assuming ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            # Create overlay with color coding:
            # Green = True Positive (correct prediction)
            # Red = False Positive (predicted but not GT)  
            # Cyan/Blue = False Negative (GT but not predicted)
            overlay = img.copy()

            # True Positives (both pred and GT) - Green/Yellow
            true_positive = (pred == 1) & (true == 1)
            overlay[true_positive] = [0, 1, 0]  # Green

            # False Positives (pred but not GT) - Red
            false_positive = (pred == 1) & (true == 0)
            overlay[false_positive] = [1, 0, 0]  # Red
            
            # False Negatives (GT but not pred) - Cyan
            false_negative = (pred == 0) & (true == 1)
            overlay[false_negative] = [0, 1, 1]  # Cyan
            
            # Blend with original image
            blended = (0.6 * img + 0.4 * overlay).astype(np.float32)
            
            # Save as single image
            plt.figure(figsize=(10, 10))
            plt.imshow(blended)
            plt.title(f"Validation Overlay - Epoch {epoch} (Green=TP, Red=FP, Cyan=FN)", fontsize=12)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(overlay_dir, f"ep{epoch:03d}_sample{i:02d}.png"), 
                       bbox_inches='tight', pad_inches=0.1)
            plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_training_curves(self, train_losses, val_ious, val_precisions, val_recalls, val_f1s):
        torch.use_deterministic_algorithms(True)

        metrics_dict = {
            "train_loss": train_losses,
            "val_iou": val_ious,
            "val_precision": val_precisions,
            "val_recall": val_recalls,
            "val_f1": val_f1s,
        }
        for name, values in metrics_dict.items():
            plt.figure()
            plt.plot(values)
            plt.xlabel("Epoch"); plt.ylabel(name); plt.title(name)
            save_path = os.path.join(self.cfg.output_dir, f"{name}.png")
            plt.savefig(save_path); plt.close()
            print(f"Saved {name} curve to {save_path}")

        torch.use_deterministic_algorithms(False)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_checkpoint(self, model, optimizer, scaler, categories, site_name,
                        learnrate, epochs, output_dir,
                        suffix=None, val_loss=None, val_accuracy=None, miou=None, target_category_name=None,
                        val_iou=None, epoch_num=None):
        """Save checkpoint with top-N management based on val_iou"""
        timestamp = str(np.datetime64('now', 's')).replace('-', '').replace(':', '').replace('T', '_')

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "categories": categories,
            "creation_UTC": timestamp,
            "site_name": site_name,
            "learning_rate": learnrate,
            "epochs": epochs,
            "num_classes": getattr(model.config, "num_labels", None),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "miou": miou,
            "target_category_name": target_category_name,
            "base_model": "segformer"
        }

        # If this is a final save or no val_iou provided, use simple naming
        if val_iou is None or suffix == "final":
            if suffix == "final":
                torch_filename = f"{timestamp}_{site_name}_{suffix}_lr{learnrate}_epoch{epochs}.torch"
            else:
                torch_filename = f"{timestamp}_{site_name}_epoch{epochs}_lr{learnrate}.torch"
            save_path = os.path.join(output_dir, torch_filename)
            torch.save(ckpt, save_path)
            print(f"Model checkpoint saved to {save_path}")
            self._last_checkpoint_path = save_path
            return save_path

        # Top-N checkpoint management based on val_iou
        temp_filename = f"temp_{timestamp}_{site_name}_{target_category_name}_ep{epoch_num:03d}_lr{learnrate}.torch"
        temp_path = os.path.join(output_dir, temp_filename)
        torch.save(ckpt, temp_path)

        # Add to tracking list
        self.best_checkpoints.append((val_iou, temp_path))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)  # Sort by IoU descending

        # Remove worst checkpoints if exceeding max
        while len(self.best_checkpoints) > self.cfg.max_best_checkpoints:
            _, worst_path = self.best_checkpoints.pop()
            if os.path.exists(worst_path):
                try:
                    os.remove(worst_path)
                    print(f"Removed lower-ranked checkpoint: {os.path.basename(worst_path)}")
                except Exception as e:
                    print(f"Failed to remove {worst_path}: {e}")

        # Rename all checkpoints with their current rank
        for rank, (iou, old_path) in enumerate(self.best_checkpoints, 1):
            rank_suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
            new_filename = f"best_{rank_suffix}_{target_category_name}_epoch{epoch_num:03d}_iou{iou:.4f}_lr{learnrate}.torch"
            new_path = os.path.join(output_dir, new_filename)

            if old_path != new_path:
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    self.best_checkpoints[rank - 1] = (iou, new_path)
                    print(f"Ranked #{rank}: {os.path.basename(new_path)}")

        self._last_checkpoint_path = self.best_checkpoints[0][1] if self.best_checkpoints else temp_path
        return self._last_checkpoint_path

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _init_progress(self, train_loader, val_loader):
        self._progress_total = self.cfg.num_epochs * (len(train_loader) + len(val_loader))
        self.progressBar = QProgressWheel(
            title="SegFormer Training in-progress...",
            total=self._progress_total,
            on_close=lambda: setattr(self.cfg, "progress_bar_closed", True)
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _tick_progress(self, inc: int = 1):
        if self.progressBar and not getattr(self.cfg, "progress_bar_closed", False):
            self.progressBar.setValue(self.progressBar.getValue() + inc)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _close_progress(self):
        if self.progressBar and not getattr(self.cfg, "progress_bar_closed", False):
            self.progressBar.close()
        self.progressBar = None

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train(self, image_dirs, ann_paths, model: Optional[nn.Module] = None,
              optimizer: Optional[torch.optim.Optimizer] = None,
              categories: Optional[List[str]] = None,
              site_name: str = "segformer"):
        """
        Train loop that accepts an externally provided model and optimizer.
        - If model is None, builds a plain SegFormer.
        - If optimizer is None, uses AdamW on all model params (no LoRA assumption).
        This enables composition with LoRA without mixing concerns.
        """
        self.set_seed()
        self.ensure_dir(self.cfg.output_dir)

        train_ds = MultiCocoTargetDataset(image_dirs, ann_paths, self.cfg.target_category_name, self.cfg.image_size,
                                          split="train")
        val_ds = MultiCocoTargetDataset(image_dirs, ann_paths, self.cfg.target_category_name, self.cfg.image_size,
                                        split="val")

        g = torch.Generator().manual_seed(self.cfg.seed)
        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True,
            generator=g, worker_init_fn=_worker_init_fn,
            persistent_workers=(self.cfg.num_workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=_worker_init_fn, persistent_workers=(self.cfg.num_workers > 0),
        )

        self._init_progress(train_loader, val_loader)

        # BINARY SEGMENTATION
        # num_labels = max(2, len(categories or [self.cfg.target_category_name]))
        # MULTI-CLASS SEGMENTATION
        # num_labels = (categories or [self.cfg.target_category_name])
        if categories:
            num_labels = len(categories)
        else:
            num_labels = 2  # background + one target

        model = model or self.build_model(num_labels=num_labels)
        optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

        ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        # Use appropriate Dice loss based on number of classes
        if num_labels == 2:
            dice_loss = BinaryDiceLoss()
        else:
            dice_loss = MultiClassDiceLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)

        metrics_log_path = os.path.join(self.cfg.output_dir, "metrics.json")
        train_losses, val_ious, val_precisions, val_recalls, val_f1s = [], [], [], [], []

        last_completed_epoch = 0

        try:
            for epoch in range(1, self.cfg.num_epochs + 1):
                model.train()
                total_loss = 0.0

                for imgs, masks in train_loader:
                    # Apply augmentation to training data
                    if self.cfg.use_augmentation:
                        augmented_imgs = []
                        augmented_masks = []
                        for i in range(imgs.size(0)):
                            aug_img, aug_mask = apply_augmentation(imgs[i], masks[i], self.cfg)
                            augmented_imgs.append(aug_img)
                            augmented_masks.append(aug_mask)
                        imgs = torch.stack(augmented_imgs)
                        masks = torch.stack(augmented_masks)
                    
                    imgs = imgs.to(self.cfg.device, non_blocking=True)
                    masks = masks.to(self.cfg.device, non_blocking=True)
                    masks = masks.round().long()
                    masks[(masks < 0) | (masks > num_labels - 1)] = 255

                    optimizer.zero_grad(set_to_none=True)

                    from contextlib import contextmanager

                    @contextmanager
                    def allow_nondeterminism():
                        torch.use_deterministic_algorithms(False)
                        try:
                            yield
                        finally:
                            torch.use_deterministic_algorithms(True, warn_only=True)

                    with allow_nondeterminism():
                        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                            outputs = model(pixel_values=imgs)
                            logits = outputs.logits
                            logits = torch.nn.functional.interpolate(
                                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                            )
                            # CE takes logits, Dice takes probabilities
                            ce = ce_loss(logits, masks)

                            # Dice loss: handle binary vs multi-class
                            probs = torch.softmax(logits, dim=1)
                            if num_labels == 2:
                                # Binary: slice foreground channel
                                probs_dice = probs[:, 1:2]  # [B,1,H,W]
                                masks_bin = (masks == 1).float().unsqueeze(1)  # [B,1,H,W]
                                dice = dice_loss(probs_dice, masks_bin)
                            else:
                                # Multi-class: use all channels
                                dice = dice_loss(probs, masks)

                            loss = ce + dice

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()

                    total_loss += float(loss.detach().cpu())
                    self._tick_progress()

                avg_loss = total_loss / max(1, len(train_loader))
                train_losses.append(avg_loss)
                last_completed_epoch = epoch
                print(f"Epoch {epoch}/{self.cfg.num_epochs} | Train loss: {avg_loss:.4f}")

                # GPU memory cleanup after each epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if epoch % self.cfg.val_every == 0:
                    metrics = self.evaluate(model, val_loader)
                    val_ious.append(metrics['mean_iou'])
                    val_precisions.append(metrics['precision'])
                    val_recalls.append(metrics['recall'])
                    val_f1s.append(metrics['f1'])

                    print(
                        f"Epoch {epoch} | "
                        f"IoU: {metrics['mean_iou']:.4f} | "
                        f"Precision: {metrics['precision']:.4f} | "
                        f"Recall: {metrics['recall']:.4f} | "
                        f"F1: {metrics['f1']:.4f}"
                    )

                    # Save validation overlays
                    if self.cfg.save_val_overlays:
                        # Get a batch from validation loader for overlay
                        val_iter = iter(val_loader)
                        sample_imgs, sample_masks = next(val_iter)
                        sample_imgs = sample_imgs.to(self.cfg.device)
                        sample_masks = sample_masks.to(self.cfg.device)

                        with torch.no_grad():
                            sample_outputs = model(pixel_values=sample_imgs)
                            sample_logits = sample_outputs.logits
                            sample_logits = torch.nn.functional.interpolate(
                                sample_logits, size=sample_masks.shape[-2:],
                                mode="bilinear", align_corners=False
                            )
                            # Use threshold-based prediction (works for both binary and multi-class)
                            sample_probs = torch.softmax(sample_logits, dim=1)
                            sample_preds = (sample_probs[:, self.cfg.target_class_id] > self.cfg.inference_threshold).long()
                            # Convert masks to binary for target class
                            sample_masks_binary = (sample_masks == self.cfg.target_class_id).long()

                        self.save_validation_overlay(
                            epoch, sample_imgs, sample_preds, sample_masks_binary, self.cfg.output_dir
                        )

                    # Early stopping and checkpoint management
                    current_val_iou = metrics['mean_iou']

                    if current_val_iou > self.best_val_iou:
                        self.best_val_iou = current_val_iou
                        self.patience_counter = 0

                        # Save checkpoint with top-N management
                        self.save_checkpoint(
                            model, optimizer, scaler,
                            categories=self.cfg.categories,
                            site_name=getattr(self.cfg, 'site_name', 'segformer'),
                            learnrate=self.cfg.lr,
                            epochs=epoch,
                            output_dir=self.cfg.output_dir,
                            miou=current_val_iou,
                            target_category_name=self.cfg.target_category_name,
                            val_iou=current_val_iou,
                            epoch_num=epoch
                        )
                    else:
                        self.patience_counter += 1
                        print(f"No improvement for {self.patience_counter} epochs")

                        if self.cfg.early_stopping and self.patience_counter >= self.cfg.patience:
                            print(f"Early stopping triggered at epoch {epoch}")
                            break

                    with open(metrics_log_path, "a") as f:
                        f.write(json.dumps({"epoch": epoch, "train_loss": avg_loss, **metrics}) + "\n")

        finally:
            if last_completed_epoch > 0:
                print(f"Saving final checkpoint for epoch {last_completed_epoch}")

            self.save_training_curves(train_losses, val_ious, val_precisions, val_recalls, val_f1s)
            self._close_progress()

        return model
