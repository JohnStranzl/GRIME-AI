#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# !/usr/bin/env python3
# model_training_visualization.py
# OPTIMIZED VERSION - All fixes and improvements applied

import os
import numpy as np

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, no GUI overhead
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)


# ======================================================================================================================
# ======================================================================================================================
#  =====     =====     =====     =====     =====     =====     =====     =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class ModelTrainingVisualization:
    def __init__(self, models_folder: str, formatted_time: str, categories: list[dict]):
        """
        models_folder: path to save figures
        formatted_time: timestamp string, e.g. '2025-08-07_13-10-00'
        categories:   list of {"id": cid, "name": cname}
        """
        self.models_folder = models_folder
        self.formatted_time = formatted_time

        # sort categories once, build label IDs + names
        sorted_cats = sorted(categories, key=lambda c: c["id"])
        self._label_ids = [c["id"] for c in sorted_cats]
        self._class_names = [c["name"] for c in sorted_cats]
        self.categories = categories

        # placeholders – populate these before plotting
        self.epoch_list = []
        self.loss_values = []
        self.train_accuracy_values = []
        self.val_accuracy_values = []
        self.val_loss_values = []

        # ✅ OPTIMIZATION: Configurable parameters
        self.max_samples = 100_000  # maximum number of points to plot
        self.plot_dpi = 150  # DPI for saved figures (consistent across all plots)
        self.grid_alpha = 0.3  # Transparency for grid lines

        # apply Seaborn global styling
        sns.set_theme(style="whitegrid", palette="deep")

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ NEW: Helper method for consistent title generation
    # ------------------------------------------------------------------------------------------------------------------
    def _make_title(self, base_title: str, site_name: str = None, lr: float = None) -> str:
        """
        Generate consistent plot titles with optional site name and learning rate.

        Args:
            base_title: Base title for the plot
            site_name: Optional site name to prepend
            lr: Optional learning rate to append

        Returns:
            Formatted title string
        """
        title = base_title
        if site_name:
            title = f"{site_name} {title}"
        if lr is not None:
            title += f" (lr={lr:.5f})"
        return title

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Added grid, standardized DPI, used title helper
    # ------------------------------------------------------------------------------------------------------------------
    def plot_accuracy(
            self,
            train_epochs: list[int],
            train_acc: list[float],
            val_epochs: list[int],
            val_acc: list[float],
            site_name: str,
            lr: float
    ):
        if not train_epochs or not train_acc or not val_epochs or not val_acc:
            return

        # vectorize inputs
        tr_ep = np.asarray(train_epochs)
        tr_acc = np.asarray(train_acc)
        vl_ep = np.asarray(val_epochs)
        vl_acc = np.asarray(val_acc)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tr_ep, tr_acc, label="Train Accuracy", lw=1, color="tab:blue")
        ax.plot(vl_ep, vl_acc, label="Val Accuracy", lw=1, color="tab:orange")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(self._make_title("Accuracy", site_name, lr))  # ✅ Use helper
        ax.legend(loc="best")
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Added grid

        out_file = os.path.join(
            self.models_folder,
            f"{self.formatted_time}_{site_name}_AccuracyCurves_lr{lr:.5f}.png"
        )
        fig.tight_layout()
        fig.savefig(out_file, dpi=self.plot_dpi, bbox_inches="tight")  # ✅ Consistent DPI
        plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Added grid, used title helper
    # ------------------------------------------------------------------------------------------------------------------
    def plot_loss_curves(
            self,
            train_epochs: list[int],
            train_loss: list[float],
            val_epochs: list[int],
            val_loss: list[float],
            site_name: str,
            lr: float
    ):
        if not train_epochs or not train_loss or not val_epochs or not val_loss:
            return

        # vectorize + subsample if too many points
        tr_ep = np.asarray(train_epochs)
        tr = np.asarray(train_loss)
        vl_ep = np.asarray(val_epochs)
        vl = np.asarray(val_loss)

        if tr_ep.size > self.max_samples:
            idxs = np.linspace(0, tr_ep.size - 1, self.max_samples, dtype=int)
            tr_ep = tr_ep[idxs]
            tr = tr[idxs]
            print(f"[plot_loss_curves] downsampled train to {tr_ep.size} points")

        if vl_ep.size > self.max_samples:
            idxs = np.linspace(0, vl_ep.size - 1, self.max_samples, dtype=int)
            vl_ep = vl_ep[idxs]
            vl = vl[idxs]
            print(f"[plot_loss_curves] downsampled val to {vl_ep.size} points")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tr_ep, tr, label="Train Loss", color="tab:blue", lw=1)
        ax.plot(vl_ep, vl, label="Val Loss", color="tab:orange", lw=1)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(self._make_title("Loss", site_name, lr))  # ✅ Use helper
        ax.legend(loc="best")
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Added grid

        out_file = os.path.join(
            self.models_folder,
            f"{self.formatted_time}_{site_name}_LossCurves_lr{lr:.5f}.png"
        )
        fig.tight_layout()
        fig.savefig(out_file, dpi=self.plot_dpi, bbox_inches="tight")  # ✅ Already consistent
        plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ CRITICAL FIX: Memory leak fixed - capture fig reference and close it properly
    # ✅ FIXED: Standardized DPI
    # ------------------------------------------------------------------------------------------------------------------
    def plot_confusion_matrix(
            self,
            y_true: list[int],
            y_pred: list[int],
            site_name: str = "",
            lr: float = 0.0,
            normalize: bool = False,
            file_prefix: str = ""
    ):
        cats_sorted = sorted(self.categories, key=lambda c: c["id"])
        labels = [c["id"] for c in cats_sorted]
        class_names = [c["name"] for c in cats_sorted]

        # force inclusion of every category by its actual id
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if normalize:
            # compute row sums (shape: [n_classes, 1])
            row_sums = cm.sum(axis=1, keepdims=True).astype(float)

            # safe divide: wherever row_sums != 0, do cm/row_sums, else leave zeros
            cm = np.divide(
                cm.astype(float),
                row_sums,
                out=np.zeros_like(cm, dtype=float),
                where=(row_sums != 0)
            )
            fmt = ".2f"
            title = "Normalized Confusion Matrix"
        else:
            fmt = "d"
            title = "Confusion Matrix"

        fig = plt.figure(figsize=(6, 5))  # ✅ CRITICAL: Capture figure reference
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(title)

        prefix_tag = f"{file_prefix}_" if file_prefix else ""
        norm_tag = "_norm" if normalize else ""
        filename = (
            f"{self.formatted_time}_{site_name}_"
            f"{prefix_tag}ConfusionMatrix_{lr}{norm_tag}.png"
        )
        png_path = os.path.join(self.models_folder, filename)

        plt.tight_layout()
        plt.savefig(png_path, dpi=self.plot_dpi)  # ✅ Consistent DPI
        plt.close(fig)  # ✅ CRITICAL: Close the specific figure (was plt.close())

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Added grid, used title helper, simplified downsampling
    # ------------------------------------------------------------------------------------------------------------------
    def plot_precision_recall(
            self,
            y_true,
            y_scores,
            site_name=None,
            lr=None,
            file_prefix=None
    ):
        # vectorize & cast
        y_true_arr = np.asarray(y_true)
        y_scores_arr = np.asarray(y_scores)  # ✅ Let numpy decide dtype

        if y_true_arr.size == 0 or y_scores_arr.size == 0:
            print("Cannot plot precision-recall: empty inputs.")
            return

        # enforce binary
        labels = np.unique(y_true_arr)
        if not np.all(np.isin(labels, [0, 1])):
            y_true_arr = (y_true_arr == labels.max()).astype(int)

        # subsample raw inputs
        if y_scores_arr.size > self.max_samples:
            idxs = np.linspace(0, y_scores_arr.size - 1, self.max_samples, dtype=int)
            y_true_arr = y_true_arr[idxs]
            y_scores_arr = y_scores_arr[idxs]
            print(f"[plot_precision_recall] downsampled to {y_true_arr.size} points")

        precision, recall, _ = precision_recall_curve(y_true_arr, y_scores_arr)

        # subsample PR curve
        if precision.size > self.max_samples:
            idxs = np.linspace(0, precision.size - 1, self.max_samples, dtype=int)
            precision = precision[idxs]
            recall = recall[idxs]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recall, precision, lw=1, color="tab:blue")
        ax.set_title(self._make_title("Precision-Recall Curve", site_name, lr))  # ✅ Use helper
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Added grid
        fig.tight_layout()

        if file_prefix:
            filename = f"{file_prefix}_precision_recall_curve.png"
            fig.savefig(os.path.join(self.models_folder, filename), dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"PR plot saved to {filename}")
        else:
            plt.close(fig)  # ✅ Ensure cleanup even if not saving

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Removed redundant try-catch, added grid, used title helper
    # ------------------------------------------------------------------------------------------------------------------
    def plot_roc_curve(
            self,
            y_true,
            y_scores,
            site_name=None,
            lr=None,
            file_prefix=None
    ):
        # vectorize & cast
        y_true_arr = np.asarray(y_true)
        y_scores_arr = np.asarray(y_scores)  # ✅ Let numpy decide dtype

        if y_true_arr.size == 0 or y_scores_arr.size == 0:
            print("Cannot plot ROC: empty inputs.")
            return

        labels = np.unique(y_true_arr)
        if not np.all(np.isin(labels, [0, 1])):
            y_true_arr = (y_true_arr == labels.max()).astype(int)

        # Downsample before computing ROC (prevents memory issues)
        if y_scores_arr.size > self.max_samples:
            idxs = np.linspace(0, y_scores_arr.size - 1, self.max_samples, dtype=int)
            y_true_arr = y_true_arr[idxs]
            y_scores_arr = y_scores_arr[idxs]
            print(f"[plot_roc_curve] downsampled to {y_true_arr.size} points")

        # ✅ REMOVED: Redundant try-catch (already downsampled above)
        fpr, tpr, _ = roc_curve(y_true_arr, y_scores_arr)
        roc_auc = auc(fpr, tpr)

        # subsample ROC curve points if still too large
        if fpr.size > self.max_samples:
            idxs = np.linspace(0, fpr.size - 1, self.max_samples, dtype=int)
            fpr, tpr = fpr[idxs], tpr[idxs]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color="tab:blue", lw=1, label=f"AUC={roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")

        ax.set_title(self._make_title("ROC Curve", site_name, lr))  # ✅ Use helper
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Added grid
        fig.tight_layout()

        if file_prefix:
            filename = f"{file_prefix}_roc_curve.png"
            fig.savefig(os.path.join(self.models_folder, filename), dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"ROC plot saved to {filename}")
        else:
            plt.close(fig)  # ✅ Ensure cleanup

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Added grid, used title helper
    # ------------------------------------------------------------------------------------------------------------------
    def plot_f1_score(
            self,
            y_true,
            y_scores,
            site_name: str = None,
            lr: float = None,
            file_prefix: str = None
    ):
        y_true_arr = np.asarray(y_true)
        y_scores_arr = np.asarray(y_scores)  # ✅ Let numpy decide dtype

        if y_true_arr.size == 0 or y_scores_arr.size == 0:
            print("Cannot plot F1: empty inputs.")
            return

        labels = np.unique(y_true_arr)
        if not np.all(np.isin(labels, [0, 1])):
            y_true_arr = (y_true_arr == labels.max()).astype(int)

        precision, recall, thresholds = precision_recall_curve(y_true_arr, y_scores_arr)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        f1 = f1_scores[:-1]
        thr = thresholds

        best_idx = np.argmax(f1)
        best_thr = thr[best_idx]
        best_f1 = f1[best_idx]

        if f1.size > self.max_samples:
            idxs = np.linspace(0, f1.size - 1, self.max_samples, dtype=int)
            f1 = f1[idxs]
            thr = thr[idxs]
            print(f"[plot_f1_score] downsampled to {f1.size} points")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thr, f1, color="tab:green", lw=1, label="F1 Score")
        ax.scatter(
            best_thr,
            best_f1,
            color="red",
            s=50,
            zorder=5,
            label=f"Max F1={best_f1:.2f} @thr={best_thr:.2f}"
        )

        ax.set_title(self._make_title("F1 Score vs Threshold", site_name, lr))  # ✅ Use helper
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 Score")
        ax.legend(loc="best")
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Added grid
        fig.tight_layout()

        if file_prefix:
            filename = f"{file_prefix}_f1_curve.png"
            fig.savefig(os.path.join(self.models_folder, filename), dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"F1 plot saved to {filename}")
        else:
            plt.close(fig)  # ✅ Ensure cleanup

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Used title helper (grid already present)
    # ------------------------------------------------------------------------------------------------------------------
    def plot_miou_curve(
            self,
            epochs: list[int],
            miou_values: list[float],
            site_name: str = None,
            lr: float = None,
            file_prefix: str = "miou"
    ):
        """
        Plots Mean IoU vs. epoch. Ensures x and y have the same length by
        truncating to the smaller size if they differ, then applies optional
        downsampling to max_samples.
        """
        # Early exit on empty data
        if not epochs or not miou_values:
            print("Cannot plot Mean IoU curve: empty inputs.")
            return

        # Convert to numpy arrays
        ep = np.asarray(epochs)
        mi = np.asarray(miou_values)

        # Dimension check: truncate to the smaller of two lengths
        if ep.size != mi.size:
            print(
                f"[plot_miou_curve] dimension mismatch: "
                f"epochs ({ep.size}) != miou ({mi.size}); "
                "truncating to the smaller length."
            )
            min_len = min(ep.size, mi.size)
            ep = ep[:min_len]
            mi = mi[:min_len]

        # Optional downsampling if still more points than max_samples
        if ep.size > self.max_samples:
            idxs = np.linspace(0, ep.size - 1, self.max_samples, dtype=int)
            ep = ep[idxs]
            mi = mi[idxs]
            print(f"[plot_miou_curve] downsampled to {ep.size} points")

        # Plotting
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ep, mi, color="tab:blue", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean IoU")
        ax.set_title(self._make_title("Mean IoU over Epochs", site_name, lr))  # ✅ Use helper
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Already present, now uses consistent alpha
        fig.tight_layout()

        # Save or show
        if file_prefix:
            filename = f"{file_prefix}_miou_curve.png"
            fig.savefig(os.path.join(self.models_folder, filename), dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"Mean IoU plot saved to {filename}")
        else:
            plt.close(fig)  # ✅ Ensure cleanup

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Used title helper (grid already present)
    # ------------------------------------------------------------------------------------------------------------------
    def plot_dice_curve(
            self,
            epochs: list[int],
            dice_values: list[float],
            site_name: str = None,
            lr: float = None,
            file_prefix: str = "dice"
    ):
        if not epochs or not dice_values:
            print("Cannot plot Dice curve: empty inputs.")
            return

        ep = np.asarray(epochs)
        dv = np.asarray(dice_values)

        if ep.size != dv.size:
            print(
                f"[plot_dice_curve] dimension mismatch: "
                f"epochs ({ep.size}) != dice ({dv.size}); "
                "truncating to the smaller length."
            )
            min_len = min(ep.size, dv.size)
            ep = ep[:min_len]
            dv = dv[:min_len]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ep, dv, color="tab:purple", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice")
        ax.set_title(self._make_title("Validation Dice over Epochs", site_name, lr))  # ✅ Use helper
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Already present, now uses consistent alpha
        fig.tight_layout()

        if file_prefix:
            filename = f"{file_prefix}_dice_curve.png"
            fig.savefig(os.path.join(self.models_folder, filename), dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"Dice plot saved to {filename}")
        else:
            plt.close(fig)  # ✅ Ensure cleanup

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Used title helper (grid already present)
    # ------------------------------------------------------------------------------------------------------------------
    def plot_iou_curve(
            self,
            epochs: list[int],
            iou_values: list[float],
            site_name: str = None,
            lr: float = None,
            file_prefix: str = "iou"
    ):
        if not epochs or not iou_values:
            print("Cannot plot IoU curve: empty inputs.")
            return

        ep = np.asarray(epochs)
        iv = np.asarray(iou_values)

        if ep.size != iv.size:
            print(
                f"[plot_iou_curve] dimension mismatch: "
                f"epochs ({ep.size}) != iou ({iv.size}); "
                "truncating to the smaller length."
            )
            min_len = min(ep.size, iv.size)
            ep = ep[:min_len]
            iv = iv[:min_len]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ep, iv, color="tab:green", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("IoU")
        ax.set_title(self._make_title("Validation IoU over Epochs", site_name, lr))  # ✅ Use helper
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Already present, now uses consistent alpha
        fig.tight_layout()

        if file_prefix:
            filename = f"{file_prefix}_iou_curve.png"
            fig.savefig(os.path.join(self.models_folder, filename), dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"IoU plot saved to {filename}")
        else:
            plt.close(fig)  # ✅ Ensure cleanup