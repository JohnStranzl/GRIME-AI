import os

import numpy as np

import matplotlib
matplotlib.use("Agg")  # PATCH: non-interactive backend, no GUI overhead

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc
)


# ======================================================================================================================
# ======================================================================================================================
#  =====     =====     =====     =====     =====     =====     =====     =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class GRIME_AI_Model_Training_Visualization:
    def __init__(self, models_folder: str, formatted_time: str, categories: list[dict]):
        """
        models_folder: path to save figures
        formatted_time: timestamp string, e.g. '2025-08-07_13-10-00'
        categories:   list of {"id": cid, "name": cname}
        """
        self.models_folder = models_folder
        self.formatted_time = formatted_time
        self.categories = categories

        # placeholders – populate these before plotting
        self.epoch_list = []
        self.loss_values = []
        self.train_accuracy_values = []
        self.val_accuracy_values = []
        self.val_loss_values = []

        # maximum number of (y_true, y_score) points to plot
        self.max_samples = 100_000


    def plot_accuracy(self,
                      epochs: list[int],
                      train_acc: list[float],
                      val_acc: list[float],
                      site_name: str,
                      lr: float):
        if not epochs or not train_acc or not val_acc:
            return

        # Create a single Axes instead of a 6×4 grid
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot curves
        #ax.plot(epochs, train_acc, label="Train Accuracy", marker="o")
        #ax.plot(epochs, val_acc, label="Val Accuracy", marker="s")
        ax.plot(epochs, train_acc, label="Train Accuracy")
        ax.plot(epochs, val_acc, label="Val Accuracy")

        # Labeling
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{site_name} Accuracy (lr={lr:.5f})")
        ax.legend(loc="best")

        # Save figure
        fig.tight_layout()
        out_file = os.path.join(
            self.models_folder,
            f"{self.formatted_time}_{site_name}_AccuracyCurves_lr{lr:.5f}.png"
        )
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_loss_curves(
            self,
            epochs: list,
            train_loss: list,
            val_loss: list,
            site_name: str,
            lr: float
    ):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # ── subsample if too many points ────────────────────────────────────────────
        n = len(epochs)
        if n > self.max_samples:
            idxs = np.linspace(0, n - 1, self.max_samples, dtype=int)
            epochs = [epochs[i] for i in idxs]
            train_loss = [train_loss[i] for i in idxs]
            val_loss = [val_loss[i] for i in idxs]

        # ── plotting ────────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 4))

        # PATCH: thin plain lines, no markers for speed
        ax.plot(epochs, train_loss, label="Train Loss", color="tab:blue", lw=1)
        ax.plot(epochs, val_loss, label="Val Loss", color="tab:orange", lw=1)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train vs. Validation Loss")
        ax.legend(loc="best")

        # ── save with lower DPI ──────────────────────────────────────────────────────
        png_file = os.path.join(
            self.models_folder,
            f"{self.formatted_time}_{site_name}_LossCurves_{lr}.png"
        )
        fig.savefig(png_file, dpi=150)  # PATCH: faster write, removed compress_level
        plt.close(fig)


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

        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

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

        plt.figure(figsize=(6, 5))
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
        plt.savefig(png_path)
        plt.close()

    def plot_precision_recall(self, y_true, y_scores, site_name=None, lr=None, file_prefix=None):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        # to numpy
        y_true   = np.array(y_true)
        y_scores = np.array(y_scores)

        # defensive checks
        if y_true.size == 0 or y_scores.size == 0:
            print("Cannot plot precision-recall: y_true or y_scores is empty.")
            return

        # binary labels
        unique_labels = np.unique(y_true)
        if not np.all(np.isin(unique_labels, [0, 1])):
            print(f"[plot_precision_recall] y_true contains unexpected labels {unique_labels}, converting to binary.")
            y_true = (y_true == 1).astype(int)

        # compute curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=1)

        # ── subsample PR points if too many ───────────────────────────────
        n_pr = precision.shape[0]
        if n_pr > self.max_samples:
            idxs = np.linspace(0, n_pr - 1, self.max_samples, dtype=int)
            precision = precision[idxs]
            recall    = recall[idxs]

        # plot
        plt.figure()
        plt.plot(recall, precision, marker='.')
        title = "Precision-Recall Curve"
        if site_name is not None:
            title = f"{site_name} {title}"
        if lr is not None:
            title += f" (lr={lr})"
        plt.title(title)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()

        # save or show
        if file_prefix is not None:
            filename = f"{file_prefix}_precision_recall_curve.png"
            png_path = os.path.join(self.models_folder, filename)
            plt.savefig(png_path)
            plt.close()
            print(f"Plot saved to {filename}")
        else:
            plt.show()

    def plot_roc_curve(self, y_true, y_scores, site_name=None, lr=None, file_prefix=None):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        # to numpy
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # defensive checks
        if y_true.size == 0 or y_scores.size == 0:
            print("Cannot plot ROC curve: y_true or y_scores is empty.")
            return

        # binary labels
        unique_labels = np.unique(y_true)
        if not np.all(np.isin(unique_labels, [0, 1])):
            print(f"[plot_roc_curve] y_true contains unexpected labels {unique_labels}, converting to binary.")
            y_true = (y_true == 1).astype(int)

        # compute curve
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # ── subsample if too many points ─────────────────────────────
        n_roc = fpr.size
        if n_roc > self.max_samples:
            idxs = np.linspace(0, n_roc - 1, self.max_samples, dtype=int)
            fpr = fpr[idxs]
            tpr = tpr[idxs]

        # ── plotting ─────────────────────────────────────────────────
        fig, ax = plt.subplots()  # PATCH: use subplots for faster management
        ax.plot(
            fpr,
            tpr,
            color='blue',
            lw=1,  # PATCH: thinner line, no markers
            label=f'ROC curve (AUC = {roc_auc:.2f})'
        )
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

        title = "ROC Curve"
        if site_name:
            title = f"{site_name} {title}"
        if lr is not None:
            title += f" (lr={lr})"
        ax.set_title(title)

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        # plt.tight_layout()  # optional; can be left out for a tiny speed gain

        # ── save or show ─────────────────────────────────────────────
        if file_prefix:
            filename = f"{file_prefix}_roc_curve.png"
            png_path = os.path.join(self.models_folder, filename)
            fig.savefig(png_path, dpi=150)  # PATCH: removed compress_level
            plt.close(fig)
            print(f"ROC plot saved to {filename}")
        else:
            plt.show()


    def plot_f1_score(self,
                      y_true,
                      y_scores,
                      site_name: str = None,
                      lr: float = None,
                      file_prefix: str = None):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        # to numpy
        y_true   = np.array(y_true)
        y_scores = np.array(y_scores)

        # defensive checks
        if y_true.size == 0 or y_scores.size == 0:
            print("Cannot plot F1 curve: y_true or y_scores is empty.")
            return

        # binary labels
        unique_labels = np.unique(y_true)
        if not np.all(np.isin(unique_labels, [0, 1])):
            print(f"[plot_f1_score] converting labels {unique_labels} to binary.")
            y_true = (y_true == unique_labels.max()).astype(int)

        # compute precision/recall and thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores, pos_label=1)

        # compute F1 = 2·(P·R)/(P+R)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        # drop the last point without threshold
        f1  = f1_scores[:-1]
        thr = thresholds

        # find best F1 on full array
        best_idx = np.argmax(f1)
        best_thr = thr[best_idx]
        best_f1  = f1[best_idx]

        # ── subsample F1-threshold curve if too many ───────────────────────
        n_f1 = f1.shape[0]
        if n_f1 > self.max_samples:
            idxs = np.linspace(0, n_f1 - 1, self.max_samples, dtype=int)
            f1  = f1[idxs]
            thr = thr[idxs]

        # plot
        plt.figure()
        plt.plot(thr, f1, color="green", lw=2, label="F1 Score")
        plt.scatter(best_thr, best_f1, color="red", label=f"Max F1={best_f1:.2f} at thr={best_thr:.2f}")
        title = "F1 Score vs Threshold"
        if site_name:
            title = f"{site_name} {title}"
        if lr is not None:
            title += f" (lr={lr})"
        plt.title(title)
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.legend(loc="best")
        plt.tight_layout()

        # save or show
        if file_prefix:
            filename = f"{file_prefix}_f1_curve.png"
            png_path = os.path.join(self.models_folder, filename)
            plt.savefig(png_path)
            plt.close()
            print(f"Saved F1 curve to {png_path}")
        else:
            plt.show()


    def plot_miou_curve(self,
                        epochs: list[int],
                        miou_values: list[float],
                        site_name: str   = None,
                        lr: float        = None,
                        file_prefix: str = "miou"):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # safety checks
        if not epochs or not miou_values:
            print("Cannot plot Mean IoU curve: epochs or miou_values is empty.")
            return

        # build the plot
        fig, ax = plt.subplots()
        #ax.plot(epochs, miou_values, marker="o", color="blue", lw=2)
        ax.plot(epochs, miou_values, color="blue", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean IoU")
        title = "Mean IoU over Epochs"
        if site_name:
            title = f"{site_name} {title}"
        if lr is not None:
            title += f" (lr={lr:.5f})"
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()

        # save or show
        if file_prefix:
            filename = f"{file_prefix}_miou_curve.png"
            png_path = os.path.join(self.models_folder, filename)
            fig.savefig(png_path)
            plt.close(fig)
            print(f"Saved Mean IoU curve to {png_path}")
        else:
            plt.show()
