#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, prevents GUI windows
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from GRIME_AI.ml_core.coco_segmentation_datasets import MultiCocoTargetDataset
from GRIME_AI.ml_core.lora_segmentation_losses import DiceLoss
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel

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
    def build_model(self, num_labels: int = 2) -> nn.Module:
        """
        Build base SegFormer (no LoRA). If you want LoRA, wrap the returned model externally.
        """
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            ignore_mismatched_sizes=True
        )
        model.config.num_labels = num_labels
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
        torch.use_deterministic_algorithms(True)

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
            preds = torch.argmax(logits, dim=1)

            for b in range(preds.size(0)):
                pb = preds[b].cpu()
                mb = masks[b].cpu()
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
                        suffix="final", val_loss=None, val_accuracy=None, miou=None, target_category_name=None):
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

        if suffix == "final":
            torch_filename = f"{timestamp}_{site_name}_{suffix}_lr{learnrate}_epoch{epochs}.torch"
        else:
            torch_filename = f"{timestamp}_{site_name}_{suffix}_{learnrate}.torch"

        save_path = os.path.join(output_dir, torch_filename)

        torch.save(ckpt, save_path)
        print(f"Model checkpoint saved to {save_path}")

        # Delete the previous checkpoint if it exists
        if self._last_checkpoint_path and os.path.exists(self._last_checkpoint_path):
            try:
                os.remove(self._last_checkpoint_path)
                print(f"Deleted previous checkpoint: {self._last_checkpoint_path}")
            except Exception as e:
                print(f"Warning: could not delete previous checkpoint {self._last_checkpoint_path}: {e}")

        # Update the tracker to the current checkpoint
        self._last_checkpoint_path = save_path

        return save_path

        return save_path

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

        model = model or self.build_model(num_labels=2)
        optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

        ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        dice_loss = DiceLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)

        metrics_log_path = os.path.join(self.cfg.output_dir, "metrics.jsonl")
        train_losses, val_ious, val_precisions, val_recalls, val_f1s = [], [], [], [], []

        last_completed_epoch = 0

        try:
            for epoch in range(1, self.cfg.num_epochs + 1):
                model.train()
                total_loss = 0.0

                for imgs, masks in train_loader:
                    imgs = imgs.to(self.cfg.device, non_blocking=True)
                    masks = masks.to(self.cfg.device, non_blocking=True)

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
                            loss = ce_loss(logits, masks) + dice_loss(logits, masks)

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)
                        scaler.step(optimizer);
                        scaler.update()

                    total_loss += float(loss.detach().cpu())
                    self._tick_progress()

                avg_loss = total_loss / max(1, len(train_loader))
                train_losses.append(avg_loss)
                last_completed_epoch = epoch
                print(f"Epoch {epoch}/{self.cfg.num_epochs} | Train loss: {avg_loss:.4f}")

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

                    with open(metrics_log_path, "a") as f:
                        f.write(json.dumps({"epoch": epoch, "train_loss": avg_loss, **metrics}) + "\n")

                    current_iou = metrics['mean_iou']
                    suffix = f"valbest_ep{epoch:03d}" if current_iou > self.best_iou else f"ep{epoch:03d}"
                    if current_iou > self.best_iou:
                        self.best_iou = current_iou

                    self.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        categories=categories or [self.cfg.target_category_name],
                        site_name=site_name,
                        learnrate=self.cfg.lr,
                        epochs=epoch,
                        output_dir=self.cfg.output_dir,
                        suffix=suffix,
                        val_loss=avg_loss,
                        val_accuracy=None,
                        miou=current_iou,
                        target_category_name=self.cfg.target_category_name,
                    )
        finally:
            if last_completed_epoch > 0:
                print(f"Saving final checkpoint for epoch {last_completed_epoch}")
                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    categories=categories or [self.cfg.target_category_name],
                    site_name=site_name,
                    learnrate=self.cfg.lr,
                    epochs=last_completed_epoch,
                    output_dir=self.cfg.output_dir,
                    suffix="final",
                    val_loss=train_losses[-1],
                    val_accuracy=None,
                    miou=val_ious[-1] if val_ious else None,
                    target_category_name=self.cfg.target_category_name,
                )

            self.save_training_curves(train_losses, val_ious, val_precisions, val_recalls, val_f1s)
            self._close_progress()

        return model
