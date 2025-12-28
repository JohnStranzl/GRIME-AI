#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import math
import json
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

###JES from hydra.experimental import initialize, compose
###JES from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

import torch
from torch import nn
from torch.amp import autocast
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.append(os.path.join(os.path.dirname(__file__), '../sam2'))
from sam2.modeling import sam2_base
from sam2.sam2_image_predictor import SAM2ImagePredictor

from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.ml_core.model_training_visualization import ModelTrainingVisualization
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.dialogs.ML_image_processing.model_config_manager import ModelConfigManager

from GRIME_AI.utils.datasetutils import DatasetUtils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(sam2_base.__file__)


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
def _normalize_centroid(cx: float, cy: float, w: int, h: int):
    """
    Normalize centroid coordinates to [0,1] range.
    Args:
        cx, cy: pixel coordinates of centroid
        w, h: image width and height
    Returns:
        (cx_norm, cy_norm) as floats in [0,1]
    """
    if w <= 1 or h <= 1:
        raise ValueError(f"Invalid image dimensions for normalization: w={w}, h={h}")
    cx_norm = cx / float(w - 1)
    cy_norm = cy / float(h - 1)
    return (cx_norm, cy_norm)


# =======================================================================
# =======================================================================
# = = =                     HELPER FUNCTIONS                        = = =
# =======================================================================
# =======================================================================
def compute_mean_iou(y_true: list[int], y_pred: list[int]) -> float:
    arr_t = np.array(y_true, dtype=bool)
    arr_p = np.array(y_pred, dtype=bool)
    inter = np.logical_and(arr_t, arr_p).sum()
    union = np.logical_or(arr_t, arr_p).sum()
    return float(inter) / float(union) if union > 0 else 1.0

# --- Add below compute_mean_iou ---
class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # probs, targets: [B,1,H,W] in [0,1]
        probs_f = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)
        inter = (probs_f * targets_f).sum(dim=1)
        union = probs_f.sum(dim=1) + targets_f.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def dice_coeff_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item()
    return float((2.0 * inter + eps) / (union + eps))


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def iou_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - inter
    return float((inter + eps) / (union + eps))


# ============================================================================
# ============================================================================
# = = =                       class SAM2Trainer                          = = =
# ============================================================================
# ============================================================================
class SAM2Trainer:


    def __init__(self, cfg: DictConfig = None):
        # =========================================================================
        # TEST MODE CONFIGURATION
        # Set self.TEST_MODE = True to train on hardcoded water + sky for testing
        # Set self.TEST_MODE = False to use normal configuration from site_config.json
        # =========================================================================
        self.TEST_MODE = False  # CHANGE THIS to True to enable test mode
        self.TEST_CATEGORIES = ["water", "sky"]  # Categories to use in test mode
        # =========================================================================

        self.sam2_model = None
        self._last_checkpoint_path = None
        self.selected_backend = None
        self.progress_bar_closed = False

        # Track top N best validation checkpoints
        self.best_checkpoints = []  # List of (val_loss, path) tuples
        self.max_best_checkpoints = 3  # Keep top 3 best checkpoints

        # ALL FILES SAVED WILL BE TAGGED WITH THE DATE AND TIME THAT TRAINING STARTED.
        self.now = datetime.now()
        self.formatted_time = self.now.strftime('%Y%m%d_%H%M%S')

        self.epoch_list = []
        self.loss_values = []
        self.train_accuracy_values = []
        self.val_loss_values = []
        self.val_accuracy_values = []
        self.val_true_list = []
        self.val_pred_list = []
        self.val_score_list = []

        # track mean–IoU per epoch
        self.miou_values = []

        self.train_dice_values = []
        self.train_iou_values = []

        self.category_centroids = {}

        self.dataset_util = DatasetUtils()

        # load site_config from Hydra or from saved JSON
        if cfg is None or "site_config" not in cfg:
            settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
            CONFIG_FILENAME = "site_config.json"
            site_configuration_file = os.path.normpath(os.path.join(settings_folder, CONFIG_FILENAME))
            print(site_configuration_file)

            # Use ModelConfigManager instead of JsonEditor
            mgr = ModelConfigManager(site_configuration_file)
            self.site_config = mgr.load_config(return_type="dict")
        else:
            # Convert the Hydra DictConfig to a standard dict using OmegaConf.to_container.
            self.site_config = OmegaConf.to_container(cfg.site_config, resolve=True)

        self.site_name = self.site_config['siteName']
        self.learning_rates = self.site_config['learningRates']
        self.optimizer_type = self.site_config['optimizer']
        self.loss_function = self.site_config['loss_function']
        self.weight_decay = self.site_config['weight_decay']
        self.num_epochs = self.site_config['number_of_epochs']
        self.save_model_frequency = self.site_config['save_model_frequency']
        self.early_stopping = self.site_config['early_stopping']
        self.patience = self.site_config['patience']

        # Validation overlay settings (default: 5 samples per epoch)
        self.validation_overlay_samples = self.site_config.get('validation_overlay_samples', 5)

        self.dataset = {}

        self.val_dice_values = []
        self.val_iou_values = []

        self.folders = None
        self.annotation_files = None
        self.all_folders = []
        self.all_annotations = []
        self.categories = []
        self.annotation_index = []

        # create output folder
        try:
            self.model_output_folder = os.path.join(
                GRIME_AI_Save_Utils().get_models_folder(), 'sam2',
                f"{self.formatted_time}_{self.site_name}"
            )
            os.makedirs(self.model_output_folder, exist_ok=True)
        except OSError as e:
            self.model_output_folder = None
            print(f"Error creating folders: {e}")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def run_training_pipeline(self):
        # Collect folders and annotations (flatten lists)
        self.all_folders = []
        self.all_annotations = []

        paths = self.site_config.get('Path', [])
        for path in paths:
            directory_path = path['directoryPaths']
            folders = directory_path.get('folders', [])
            annotations = directory_path.get('annotations', [])
            # extend directly with the lists of strings
            self.all_folders.extend(folders)
            self.all_annotations.extend(annotations)

        # Build categories first (needed for both modes)
        self.categories = self.build_unique_categories(self.all_annotations)
        
        if self.TEST_MODE:
            # === TEST MODE: Train on multiple hardcoded categories ===
            print("\n" + "=" * 70)
            print("TEST MODE ENABLED: Training on", self.TEST_CATEGORIES)
            print("=" * 70 + "\n")
            
            # Don't combine datasets here - we'll reload per-category during training
            # This avoids the bug where second category's annotations get discarded
            # Just load first category to initialize the dataset structure
            first_category = self.TEST_CATEGORIES[0]
            print(f"Loading initial dataset structure using: {first_category}")
            self.dataset = self.dataset_util.load_images_and_annotations(
                self.all_folders, self.all_annotations, first_category
                )
            self.annotation_index = self.dataset_util.build_annotation_index(self.dataset)
            
        else:
            # === NORMAL MODE: Use TRAINING_CATEGORIES from train_model config ===
            training_cats = self.site_config.get("train_model", {}).get("TRAINING_CATEGORIES", [])
            configured_categories = [cat["label_name"] for cat in training_cats if "label_name" in cat]
            
            # Handle empty config
            if not configured_categories:
                print("\n" + "!" * 70)
                print("ERROR: No categories configured in TRAINING_CATEGORIES!")
                print("Please select categories in the UI")
                print("!" * 70 + "\n")
                raise ValueError("No training categories configured")
            
            # Load dataset once with first category (but keeps all annotations)
            first_category = configured_categories[0]
            print(f"Loading dataset with all annotations...")
            self.dataset = self.dataset_util.load_images_and_annotations(
                self.all_folders, self.all_annotations, first_category
            )
            self.annotation_index = self.dataset_util.build_annotation_index(self.dataset)


        # Split dataset
        train_images, val_images = self.dataset_util.split_dataset(self.dataset)
        split_dataset_filename = os.path.join(
            self.model_output_folder,
            f"{self.formatted_time}_{self.site_name}training_and_validation_sets.json"
        )
        self.dataset_util.save_split_dataset(train_images, val_images, split_dataset_filename)

        stats_train = self.summarize_mask_imbalance(train_images)
        stats_val = self.summarize_mask_imbalance(val_images)
        print(f"Foreground ratio (train): {stats_train}")
        print(f"Foreground ratio (val):   {stats_val}")

        if len(train_images) == 0:
            print("No training images found!")
            return

        main_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        model_cfg = os.path.normpath(os.path.join(main_dir, "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml"))
        sam2_checkpoint = os.path.normpath(os.path.join(main_dir, "sam2", "checkpoints", "sam2.1_hiera_large.pt"))
        config_dir = os.path.normpath(os.path.join(main_dir, "sam2", "sam2", "configs", "sam2.1"))
        print("Model config path: ", model_cfg)
        print("Checkpoint path: ", sam2_checkpoint)
        print("config_dir path: ", config_dir)

        ###JES -
        '''
        # Clear Hydra if already initialized
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Re‑initialize Hydra
        #with initialize(config_path=config_dir, version_base=None):
        with initialize(config_module="GRIME_AI.sam2.sam2.configs.sam2.1", version_base=None):
            cfg_intern = compose(config_name="sam2.1_hiera_l.yaml")
            raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)

            # Strip unsupported keys
            for key in ["no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs", "device"]:
                raw_model_cfg.pop(key, None)

            new_cfg = OmegaConf.create(raw_model_cfg)
            model = instantiate(new_cfg, _recursive_=True)

            checkpoint = torch.load(
                sam2_checkpoint,
                map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            )
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        '''

        # --- REPLACEMENT FOR HYDRA THAT IS A BEAR TO USE AND MAINTAIN
        cfg_file = os.path.join(main_dir, "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")
        checkpoint = os.path.join(main_dir, "sam2", "checkpoints", "sam2.1_hiera_large.pt")

        cfg_intern = OmegaConf.load(cfg_file)
        raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)

        for k in ["no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs", "device"]:
            raw_model_cfg.pop(k, None)

        new_cfg = OmegaConf.create(raw_model_cfg)
        model = instantiate(new_cfg, _recursive_=True)

        ckpt = torch.load(checkpoint,
                          map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
        # --- END HYDRA PATCH

        self.sam2_model = model.to(device).train()

        if self.TEST_MODE:
            # === TEST MODE: Train on each test category separately ===
            for category_name in self.TEST_CATEGORIES:
                print("\n" + "=" * 70)
                print(f"TRAINING ON: {category_name.upper()}")
                print("=" * 70 + "\n")
                
                # RELOAD dataset for THIS specific category
                # This ensures we have the correct annotations for this category
                print(f"Loading dataset for {category_name}...")
                self.dataset = self.dataset_util.load_images_and_annotations(
                    self.all_folders, self.all_annotations, category_name
                )
                self.annotation_index = self.dataset_util.build_annotation_index(self.dataset)
                
                # DON'T re-split! Use the SAME train/val split from initial split
                # This is crucial - random.shuffle() would give different split each time
                print(f"  Using shared train/val split:")
                print(f"  Training images: {len(train_images)}")
                print(f"  Validation images: {len(val_images)}")
                
                # Get category ID
                target_id = next((c["id"] for c in self.categories if c["name"] == category_name), None)
                if target_id is None:
                    print(f"Warning: Category '{category_name}' not found in categories, skipping")
                    continue
                
                # Check if we have any annotations for this category
                if not train_images:
                    print(f"Warning: No training images for '{category_name}', skipping")
                    continue
                
                # Train on this category
                for lr in self.learning_rates:
                    print(f"\nTraining {category_name} with learning rate: {lr}")
                    self.train_sam(lr, self.weight_decay, train_images, val_images, 
                                 epochs=self.num_epochs, target_label=category_name)
                    self._plot_training_graphs(lr)
        else:
            # === NORMAL MODE: Train on ALL categories from TRAINING_CATEGORIES ===
            training_cats = self.site_config.get("train_model", {}).get("TRAINING_CATEGORIES", [])
            configured_categories = [cat["label_name"] for cat in training_cats if "label_name" in cat]
            
            print(f"\nWill train on {len(configured_categories)} category(ies): {configured_categories}")
            
            # Train on each configured category
            for category_name in configured_categories:
                print("\n" + "=" * 70)
                print(f"TRAINING ON: {category_name.upper()}")
                print("=" * 70 + "\n")
                
                # Verify category exists
                target_id = next((c["id"] for c in self.categories if c["name"] == category_name), None)
                if target_id is None:
                    print(f"Warning: Category '{category_name}' not found, skipping")
                    print(f"Available categories: {[c['name'] for c in self.categories]}")
                    continue
                
                # Train on this category (dataset already has all annotations)
                for lr in self.learning_rates:
                    print(f"\nTraining {category_name} with learning rate: {lr}")
                    self.train_sam(lr, self.weight_decay, train_images, val_images,
                                     epochs=self.num_epochs, target_label=category_name)
                    self._plot_training_graphs(lr)



        config_file = os.path.join(
            self.model_output_folder,
            f"{self.formatted_time}_{self.site_name}_configuration.txt"
        )
        self.save_config_to_text(config_file)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def train_sam(self, learnrate, weight_decay, train_images, val_images, epochs=20, target_label=None):

        self.reset_metrics()

        total_iterations = epochs * (len(train_images) + (len(val_images) if val_images else 0))

        progressBar = QProgressWheel(
            title="Training in-progress...", total=total_iterations,
            on_close=lambda: setattr(self, "progress_bar_closed", True)
        )

        # Set seeds for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Select best SDPA backend once (not per-sample)
        self.selected_backend = self._select_best_sdpa_backend()

        self.sam2_model.train()

        predictor = SAM2ImagePredictor(self.sam2_model)

        optimizer = torch.optim.AdamW(self.sam2_model.parameters(), lr=learnrate, weight_decay=weight_decay)
        # Note: loss_fn defined here but not passed to validation anymore
        use_amp = True
        if use_amp and device.type == "cuda":
            # MAKE PORTABLE ACROSS GPU AND CPU MACHINES
            try:
                from torch.amp import GradScaler
                scaler = GradScaler('cuda')
            except ImportError:
                from torch.cuda.amp import GradScaler  # FALLBACK FOR OLDER PYTORCH
                scaler = GradScaler()
        else:
            scaler = None

        best_val_loss = float("inf")
        patience_counter = 0
        divergence_threshold = 1e3
        last_completed_epoch = 0

        # Get target_label - use parameter if provided (TEST_MODE), otherwise get from config
        if target_label is None:
            # Normal mode: get from config
            target_label = self.site_config["load_model"]["SEGMENTATION_CATEGORIES"][0]
        # else: use the provided target_label (TEST_MODE)

        try:
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")

                if use_amp:
                    avg_epoch_loss, train_accuracy, train_dice, train_iou = self._train_one_epoch(
                        epoch=epoch,
                        train_images=train_images,
                        predictor=predictor,
                        optimizer=optimizer,
                        progressBar=progressBar,
                        use_amp=use_amp,
                        scaler=scaler,
                        target_label=target_label
                    )
                else:
                    avg_epoch_loss, train_accuracy, train_dice, train_iou = self._train_one_epoch(
                        epoch=epoch,
                        train_images=train_images,
                        predictor=predictor,
                        optimizer=optimizer,
                        progressBar=progressBar,
                        use_amp=use_amp,
                        target_label = target_label
                    )

                if avg_epoch_loss is None:
                    return

                last_completed_epoch = epoch + 1

                # ALWAYS track training metrics for ALL epochs
                self.loss_values.append(avg_epoch_loss)
                self.train_accuracy_values.append(train_accuracy)
                self.train_dice_values.append(train_dice)
                self.train_iou_values.append(train_iou)
                print(f"Epoch {epoch + 1} Training "
                      f"Loss: {avg_epoch_loss:.4f} "
                      f"Acc: {train_accuracy:.4f} "
                      f"Dice: {train_dice:.4f} "
                      f"IoU: {train_iou:.4f}")

                # Clear GPU cache after training to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if math.isnan(avg_epoch_loss) or avg_epoch_loss > divergence_threshold:
                    print("Training diverged. Aborting early.")
                    break

                # Deduplicate centroids after first epoch to prevent memory leak
                if epoch == 0:
                    print("\n=== Finalizing Centroid Collection ===")
                    self._deduplicate_centroids()
                    total_centroids = sum(len(v) for v in self.category_centroids.values())
                    print(f"Total unique centroids collected: {total_centroids}")
                    print("Centroid collection frozen - will not accumulate in future epochs\n")


                # Skip validation during warmup period (first 10% of epochs)
                validation_warmup_epochs = max(1, int(epochs * 0.1))

                if val_images and (epoch + 1) > validation_warmup_epochs:
                    avg_val_loss, val_accuracy, miou, avg_val_dice, avg_val_iou = self._validate_one_epoch(
                        val_images=val_images,
                        predictor=predictor,
                        progressBar=progressBar,
                        target_label=target_label
                    )

                    if avg_val_loss is None:
                        return

                    # Add to epoch list when validation runs
                    self.epoch_list.append(epoch + 1)

                    self.val_loss_values.append(avg_val_loss)
                    self.val_accuracy_values.append(val_accuracy)
                    self.miou_values.append(miou)
                    self.val_dice_values.append(avg_val_dice)
                    self.val_iou_values.append(avg_val_iou)

                    print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f} "
                          f"Acc: {val_accuracy:.4f} Dice: {avg_val_dice:.4f} "
                          f"IoU: {avg_val_iou:.4f} mIoU: {miou:.4f}")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        self._save_model_checkpoint(
                            predictor, learnrate, epoch + 1,
                            suffix=f"valbest_ep{epoch + 1:03d}",
                            val_loss=avg_val_loss, val_accuracy=val_accuracy, miou=miou,
                            target_category_name=target_label
                        )
                    else:
                        patience_counter += 1
                        if self.early_stopping and patience_counter >= self.patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}.")
                            break

        finally:
            if not self.progress_bar_closed and 'progressBar' in locals():
                progressBar.close()
            if 'progressBar' in locals():
                del progressBar

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def reset_metrics(self):
        self.epoch_list.clear()
        self.loss_values.clear()
        self.train_accuracy_values.clear()
        self.val_loss_values.clear()
        self.val_accuracy_values.clear()
        self.val_true_list.clear()
        self.val_pred_list.clear()
        self.val_score_list.clear()
        self.miou_values.clear()
        self.train_dice_values.clear()
        self.train_iou_values.clear()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _select_best_sdpa_backend(self):
        """
        Selects the best available SDPA backend once, to be reused throughout training.
        Returns the selected backend or None if all fail.
        """
        desired = ("FLASH_ATTENTION", "XFORMERS", "EFFICIENT_ATTENTION", "MATH")
        backends = [getattr(SDPBackend, name) for name in desired if hasattr(SDPBackend, name)]

        print("\n=== Selecting SDPA Backend ===")
        for backend in backends:
            try:
                # Quick test with dummy tensors
                dummy_q = torch.randn(1, 8, 16, 64, device=device)
                dummy_k = torch.randn(1, 8, 16, 64, device=device)
                dummy_v = torch.randn(1, 8, 16, 64, device=device)

                with sdpa_kernel(backend):
                    _ = torch.nn.functional.scaled_dot_product_attention(dummy_q, dummy_k, dummy_v)

                print(f"✓ Selected SDPA backend: {backend.name}")
                return backend
            except Exception as e:
                print(f"✗ Backend {backend.name} failed: {e}")
                continue

        print("All SDPA backends failed; will use default attention")
        return None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _train_one_epoch(
            self,
            epoch,
            train_images,
            predictor,
            optimizer,
            progressBar,
            use_amp=False,
            scaler=None,
            target_label=None
    ):
        """
        Runs one training epoch. Returns (avg_epoch_loss, train_accuracy, avg_dice, avg_iou).

        CORRECTIONS APPLIED:
        - Uses BCEWithLogitsLoss properly on logits
        - Only applies sigmoid when computing Dice loss and metrics
        - Maintains numerical stability
        - Direct progress_bar_closed attribute access (Issue #8)
        """
        import torch.nn.functional as F  # ✅ Moved import to top of method

        self.sam2_model.train()

        epoch_loss = 0.0
        train_correct, train_total = 0, 0
        processed_count = 0
        dice_sum, iou_sum = 0.0, 0.0

        np.random.shuffle(train_images)

        # Define loss function once (not per-sample)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        dice_loss_fn = DiceLoss()

        # --- FILTER MASK BY TARGET_LABEL ---
        if target_label is None:
            raise ValueError("target_label is required but was None.")

        for idx, image_file in enumerate(train_images):
            if self.progress_bar_closed:
                self._terminate_training(progressBar)
                return None, None, None, None

            # ----------------------------------------
            # LOAD GROUND TRUTH MASK
            # ----------------------------------------
            # RESOLVE TARGET_ID
            target_id = next((c["id"] for c in self.categories if c["name"] == target_label), None)
            if target_id is None:
                available = ", ".join(f'{c["name"]}:{c["id"]}' for c in self.categories)
                raise ValueError(f"Unknown target_label '{target_label}'. Available: {available}")

            # LOAD GROUND TRUTH MASK
            true_mask = self.dataset_util.load_true_mask(image_file, self.annotation_index, mode="binary", target_id=target_id)

            # SKIP IF NO USABLE MASK
            if true_mask is None or true_mask.sum() == 0:
                print(f"No usable mask for target_id={target_id} in {image_file}, skipping.")
                continue

            if true_mask.ndim == 3:
                true_mask = true_mask[..., 0]
            true_mask = true_mask.astype(np.uint8)

            image = np.array(Image.open(image_file).convert("RGB"))
            predictor.set_image(image)

            # ============================================================
            # COMPUTE CENTROIDS FOR POSITIVE PROMPTS (selected category)
            # ============================================================
            h, w = true_mask.shape[:2]
            num_labels, labels = cv2.connectedComponents(true_mask.astype(np.uint8))
            positive_coords = []
            
            for lbl in range(1, num_labels):  # skip background
                ys, xs = np.nonzero(labels == lbl)
                if xs.size > 0:
                    cx = float(xs.mean())
                    cy = float(ys.mean())
                    positive_coords.append([cx, cy])

                    # Only collect centroids during first epoch to prevent memory leak
                    if epoch == 0:
                        # Store normalized centroid for this category
                        centroid_px = (int(round(cx)), int(round(cy)))
                        centroid_norm = _normalize_centroid(cx, cy, w, h)

                        self.category_centroids.setdefault(int(target_id), []).append({
                            "centroid_px": centroid_px,
                            "centroid_norm": centroid_norm
                        })

            if not positive_coords:
                print(f"Skipping {image_file} - no usable centroid for label {target_label}.")
                continue

            # ============================================================
            # COMPUTE CENTROIDS FOR NEGATIVE PROMPTS (all other categories)
            # ============================================================
            negative_coords = []

            # Get all categories except the target (dynamic - not hardcoded!)
            negative_categories = [
                c for c in self.categories 
                if c["name"] != target_label
            ]
            
            for neg_category in negative_categories:
                neg_id = neg_category["id"]
                
                # Load mask for this negative category
                neg_mask = self.dataset_util.load_true_mask(
                    image_file, self.annotation_index, 
                    mode="binary", target_id=neg_id
                )
                
                # Skip if no annotation for this category in this image
                if neg_mask is None or neg_mask.sum() == 0:
                    continue
                
                if neg_mask.ndim == 3:
                    neg_mask = neg_mask[..., 0]
                
                # Compute centroids for negative regions
                num_labels_neg, labels_neg = cv2.connectedComponents(neg_mask.astype(np.uint8))
                for lbl in range(1, num_labels_neg):
                    ys, xs = np.nonzero(labels_neg == lbl)
                    if xs.size > 0:
                        cx, cy = float(xs.mean()), float(ys.mean())
                        negative_coords.append([cx, cy])
            
            # Balance negatives to avoid overwhelming positive prompts
            # (SAM2 works best with balanced prompt sets)
            if len(negative_coords) > len(positive_coords) * 3:
                negative_coords = random.sample(negative_coords, len(positive_coords) * 3)
            
            # ============================================================
            # COMBINE POSITIVE AND NEGATIVE PROMPTS
            # ============================================================
            all_coords = positive_coords + negative_coords
            all_labels = [1] * len(positive_coords) + [0] * len(negative_coords)
            
            # Log prompt counts for monitoring
            if idx % 10 == 0:  # Log every 10th image
                print(f"  Image {os.path.basename(image_file)}: "
                      f"{len(positive_coords)} positive, {len(negative_coords)} negative prompts")

            # --- Build tensors for all centroids (positive + negative) ---
            point_coords = torch.tensor(all_coords, device=device, dtype=torch.float32)
            point_labels = torch.tensor(all_labels, device=device, dtype=torch.int64)

            # Prompt embeddings using multiple points (positive and negative)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(point_coords.unsqueeze(0), point_labels.unsqueeze(0)),
                boxes=None,
                masks=None,
            )

            batched_mode = True
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

            optimizer.zero_grad()

            # Pick dtype depending on device
            if use_amp and device.type == "cuda":
                # MAKE PORTABLE ACROSS GPU AND CPU MACHINES
                try:
                    from torch.amp import autocast
                except ImportError:
                    from torch.cuda.amp import autocast  # FALLBACK FOR OLDER PYTORCH

            if device.type == "cuda":
                autocast_dtype = torch.float16  # or torch.bfloat16 if you prefer
            else:  # CPU
                autocast_dtype = torch.bfloat16  # CPU autocast only supports bfloat16

            with autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                # Use pre-selected SDPA backend (selected once in train_sam)
                if hasattr(self, 'selected_backend') and self.selected_backend is not None:
                    with sdpa_kernel(self.selected_backend):
                        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            repeat_image=batched_mode,
                            high_res_features=high_res_features,
                        )
                else:
                    # Fallback to default if backend selection failed
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        repeat_image=batched_mode,
                        high_res_features=high_res_features,
                    )

                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                gt_mask = torch.tensor(true_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
                if gt_mask.sum() == 0:
                    print(f"Skipping {image_file} - ground-truth mask is empty for label {target_label}.")
                    continue

                # Keep predictions as LOGITS (don't apply sigmoid yet)
                prd_mask_logits = prd_masks[:, 0].unsqueeze(1)  # [1,1,H,W] - raw logits

                # Resize logits if mismatch
                if prd_mask_logits.shape[-2:] != gt_mask.shape[-2:]:
                    prd_mask_logits = F.interpolate(
                        prd_mask_logits,
                        size=gt_mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                if prd_mask_logits.shape != gt_mask.shape:
                    raise ValueError(f"Shape mismatch {prd_mask_logits.shape} vs {gt_mask.shape} for {image_file}")

                # USE BCEWithLogitsLoss on logits (more stable than manual BCE on probabilities)
                seg_loss = bce_loss_fn(prd_mask_logits, gt_mask)

                # FOR DICE LOSS, WE NEED PROBABILITIES, SO APPLY SIGMOID
                prd_mask_probs = torch.sigmoid(prd_mask_logits)

                dice_loss = dice_loss_fn(prd_mask_probs, gt_mask)

                # COMPUTE IOU FOR SCORE PREDICTION LOSS
                prd_mask_binary = (prd_mask_probs > 0.5).float()

                # FILTER OUT BLOBS NOT OVERLAPPING CENTROID PROMPTS
                labels_np = prd_mask_binary.squeeze().cpu().numpy().astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(labels_np)
                valid_mask = np.zeros_like(labels, dtype=np.uint8)

                # ADAPTIVE RADIUS THRESHOLD BASED ON IMAGE SIZE (2.5% OF DIAGONAL).
                # THIS WILL SCALE WITH IMAGE RESOLUTION
                # BLOB FILTERING = 0.025
                img_diagonal = math.sqrt(h * h + w * w)
                radius_threshold = max(10, int(0.025 * img_diagonal))  # min 10px

                for lbl in range(1, num_labels):  # skip background
                    ys, xs = np.nonzero(labels == lbl)
                    if xs.size == 0:
                        continue
                    cx_blob, cy_blob = xs.mean(), ys.mean()

                    # KEEP ONLY IF BLOB CENTROID IS CLOSE TO ONE OF THE PROMPT CENTROIDS
                    for cx, cy in positive_coords:
                        if np.linalg.norm([cx - cx_blob, cy - cy_blob]) < radius_threshold:
                            valid_mask[labels == lbl] = 1
                            break

                prd_mask_binary = torch.tensor(valid_mask, device=device).unsqueeze(0).unsqueeze(1).float()

                inter = (gt_mask * prd_mask_binary).sum((1, 2, 3))
                union = gt_mask.sum((1, 2, 3)) + prd_mask_binary.sum((1, 2, 3)) - inter
                iou = inter / (union + 1e-6)
                iou[union == 0] = 1.0

                # Align decoder score with IoU
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                # Combined loss
                loss = 0.5 * seg_loss + 0.5 * dice_loss + 0.05 * score_loss

            # Backward
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.sam2_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sam2_model.parameters(), max_norm=1.0)
                optimizer.step()

            # Metrics (use probabilities for evaluation)
            d_metric = dice_coeff_from_probs(prd_mask_probs, gt_mask)
            j_metric = iou_from_probs(prd_mask_probs, gt_mask)

            dice_sum += d_metric
            iou_sum += j_metric
            epoch_loss += loss.item()
            processed_count += 1

            pred_binary = (prd_mask_probs > 0.5).cpu().numpy()
            true_binary = gt_mask.cpu().numpy()
            train_correct += np.sum(pred_binary == true_binary)
            train_total += np.prod(true_binary.shape)

            if not self.progress_bar_closed:  # ✅ ISSUE #8 FIX: Direct attribute access
                progressBar.setValue(progressBar.getValue() + 1)

        avg_epoch_loss = epoch_loss / processed_count if processed_count > 0 else 0.0
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        avg_dice = dice_sum / processed_count if processed_count > 0 else 0.0
        avg_iou = iou_sum / processed_count if processed_count > 0 else 0.0

        return avg_epoch_loss, train_accuracy, avg_dice, avg_iou

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _validate_one_epoch(self, val_images, predictor, progressBar, target_label):
        """
        Runs one validation epoch. Returns (avg_val_loss, val_accuracy, miou, avg_val_dice, avg_val_iou).

        FIXED: Now uses logits with BCEWithLogitsLoss (matches training approach).
        FIXED: Uses n_items for avg_val_loss calculation (Issue #9).
        FIXED: Added error handling around predict() (Issue #11).
        FIXED: Removed unused loss_fn parameter.
        FIXED: Added pixel sampling for memory efficiency.
        FIXED: Removed duplicate variable references (best_mask_tensor, prob_full).
        """
        import torch.nn.functional as F

        val_loss = 0.0
        dice_sum, iou_sum = 0.0, 0.0
        val_correct, val_total = 0, 0
        n_items = 0

        self.sam2_model.eval()
        progressBar.setWindowTitle("Validation in-progress")

        # Define loss functions (same as training)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        dice_loss_fn = DiceLoss()

        # target_label is now passed as parameter (works for both normal and TEST_MODE)
        target_id = next((c["id"] for c in self.categories if c["name"] == target_label), None)
        with torch.no_grad():
            for val_idx, val_image_file in enumerate(val_images):
                if self.progress_bar_closed:
                    self._terminate_validation(progressBar)
                    return None, None, None, None, None

                val_image = np.array(Image.open(val_image_file).convert("RGB"))
                val_true_mask = self.dataset_util.load_true_mask(val_image_file, self.annotation_index,
                                                                 mode="binary", target_id=target_id)

                if val_true_mask is None:
                    print(f"No annotation found for validation image {val_image_file}, skipping.")
                    continue

                predictor.set_image(val_image)

                # ===== ERROR HANDLING =====
                try:
                    masks, scores, low_res_logits = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        multimask_output=False
                    )

                    if masks.size == 0 or scores.size == 0:
                        print(f"Warning: No masks predicted for {val_image_file}")
                        continue

                except Exception as e:
                    print(f"Error during prediction for {val_image_file}: {e}")
                    continue

                best_idx = int(np.argmax(scores))

                # ===== LOGITS APPROACH =====
                # Get logits before sigmoid (low resolution)
                best_logits = low_res_logits[best_idx]  # [256, 256]

                # Convert to tensor
                logit_tensor = torch.tensor(
                    best_logits,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256]

                # Prepare ground truth mask
                if len(val_true_mask.shape) > 2:
                    val_true_mask = val_true_mask[:, :, 0]

                val_true_mask_tensor = torch.tensor(
                    val_true_mask,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

                # Upsample logits to match ground truth size
                H, W = val_true_mask_tensor.shape[2:]
                logit_upsampled = F.interpolate(
                    logit_tensor,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )  # [1, 1, H, W]

                # Compute BCE loss on LOGITS (same as training)
                bce_val = bce_loss_fn(logit_upsampled, val_true_mask_tensor).item()

                # Compute Dice loss on PROBABILITIES
                prob_upsampled = torch.sigmoid(logit_upsampled)
                dice_val = dice_loss_fn(prob_upsampled, val_true_mask_tensor).item()

                # Compute score loss (IoU prediction vs actual IoU)
                pred_binary = (prob_upsampled > 0.5).float()
                inter = (val_true_mask_tensor * pred_binary).sum()
                union = val_true_mask_tensor.sum() + pred_binary.sum() - inter
                iou_actual = inter / (union + 1e-6)
                if union == 0:
                    iou_actual = torch.tensor(1.0, device=device)

                score_val = torch.abs(scores[best_idx] - iou_actual).item()

                # Combined loss (same weights as training)
                val_loss += 0.5 * bce_val + 0.5 * dice_val + 0.05 * score_val

                # Metrics for tracking
                d_metric = dice_coeff_from_probs(prob_upsampled, val_true_mask_tensor)
                j_metric = iou_from_probs(prob_upsampled, val_true_mask_tensor)
                dice_sum += d_metric
                iou_sum += j_metric
                n_items += 1

                # ===== ACCURACY & PIXEL SAMPLING =====
                pred_binary_np = pred_binary.cpu().numpy()
                true_binary_np = val_true_mask_tensor.cpu().numpy()
                val_correct += np.sum(pred_binary_np == true_binary_np)
                val_total += np.prod(true_binary_np.shape)

                # Flatten for sampling
                true_flat = true_binary_np.flatten()
                pred_flat = pred_binary_np.flatten()

                # Get configuration for sampling
                max_samples = self.site_config.get('max_pixel_samples', 10000)
                total_pixels = len(true_flat)

                # Get scores (use prob_upsampled, not undefined prob_full)
                score_flat = prob_upsampled.squeeze().cpu().numpy().flatten()

                # Sample pixels for memory efficiency
                if total_pixels > max_samples:
                    sample_indices = np.random.choice(total_pixels, max_samples, replace=False)
                    true_sampled = true_flat[sample_indices]
                    pred_sampled = pred_flat[sample_indices]
                    score_sampled = score_flat[sample_indices]
                else:
                    true_sampled = true_flat
                    pred_sampled = pred_flat
                    score_sampled = score_flat

                # Add SAMPLED pixels to lists
                self.val_true_list.extend(int(x) for x in true_sampled)
                self.val_pred_list.extend(int(x) for x in pred_sampled)
                self.val_score_list.extend(score_sampled.tolist())

                # ===== SAVE OVERLAY IMAGES =====
                if val_idx < self.validation_overlay_samples:
                    img_vis = val_image.copy()
                    overlay = (prob_upsampled.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    overlay_color = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
                    alpha = 0.4
                    blended = cv2.addWeighted(img_vis, 1.0, overlay_color, alpha, 0.0)
                    out_path = os.path.join(
                        self.model_output_folder,
                        f"{self.formatted_time}_{self.site_name}_val_overlay_e{len(self.epoch_list)}_{val_idx}.png"
                    )
                    cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

                progressBar.setValue(progressBar.getValue() + 1)

        # Compute averages
        avg_val_loss = val_loss / n_items if n_items > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        avg_val_dice = dice_sum / n_items if n_items > 0 else 0.0
        avg_val_iou = iou_sum / n_items if n_items > 0 else 0.0
        miou = compute_mean_iou(self.val_true_list, self.val_pred_list)

        # Clear GPU cache after validation to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_val_loss, val_accuracy, miou, avg_val_dice, avg_val_iou

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _save_model_checkpoint(self, predictor, learnrate, epochs, suffix="final", val_loss=None, val_accuracy=None,
                               miou=None, target_category_name=None):
        """
        Save the current model state_dict and metadata to a .torch file.
        Only saves if this is a top-N best checkpoint by validation loss.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Try to infer num_classes from the predictor if available
        num_classes = None
        try:
            if hasattr(predictor.model, "roi_heads"):
                num_classes = predictor.model.roi_heads.box_predictor.cls_score.out_features
        except Exception:
            pass

        ckpt = {
            "model_state_dict": predictor.model.state_dict(),
            "categories": self.categories,
            "creation_UTC": timestamp,
            "site_name": self.site_name,
            "learning_rate": learnrate,
            "epochs": epochs,
            "num_classes": num_classes,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "miou": miou,
            "target_category_name": target_category_name,
            "base_model": "sam2",
            "category_centroids": self.category_centroids
        }

        # Only save best validation checkpoints (not "final")
        if suffix.startswith("valbest") and val_loss is not None:
            # Check if this checkpoint would be in top N
            temp_list = self.best_checkpoints + [(val_loss, None)]
            temp_list.sort(key=lambda x: x[0])

            # Only proceed if this checkpoint makes the top N
            if len(temp_list) <= self.max_best_checkpoints or val_loss <= temp_list[self.max_best_checkpoints - 1][0]:

                # Create temporary checkpoint name (will be renamed based on rank)
                temp_filename = f"temp_{timestamp}_{self.site_name}_{target_category_name}_ep{epochs:03d}_lr{learnrate}.torch"
                temp_path = os.path.join(self.model_output_folder, temp_filename)

                # Save the checkpoint
                torch.save(ckpt, temp_path)

                # Add to tracking list with temp path
                self.best_checkpoints.append((val_loss, temp_path))

                # Sort by validation loss (ascending - best first)
                self.best_checkpoints.sort(key=lambda x: x[0])

                # Remove worst checkpoint if exceeding max
                if len(self.best_checkpoints) > self.max_best_checkpoints:
                    _, worst_path = self.best_checkpoints.pop()
                    if os.path.exists(worst_path):
                        try:
                            os.remove(worst_path)
                            print(f"Removed checkpoint: {os.path.basename(worst_path)}")
                        except Exception as e:
                            print(f"Warning: could not delete checkpoint {worst_path}: {e}")

                # Rename all checkpoints with meaningful rank names
                rank_names = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]

                for rank, (loss, old_path) in enumerate(self.best_checkpoints, start=1):
                    rank_suffix = rank_names[rank - 1] if rank <= len(rank_names) else f"{rank}th"

                    # Extract epoch number from the checkpoint metadata or filename
                    epoch_num = epochs if old_path == temp_path else self._extract_epoch_from_path(old_path)

                    # Create new meaningful name with category
                    new_filename = f"best_{rank_suffix}_{target_category_name}_epoch{epoch_num:03d}_valloss{loss:.4f}_lr{learnrate}.torch"
                    new_path = os.path.join(self.model_output_folder, new_filename)

                    # Rename if needed
                    if old_path != new_path and os.path.exists(old_path):
                        try:
                            os.rename(old_path, new_path)
                            # Update the path in the list
                            self.best_checkpoints[rank - 1] = (loss, new_path)
                        except Exception as e:
                            print(f"Warning: could not rename {old_path} to {new_path}: {e}")

                # Print current best checkpoints
                print(f"\nTop {len(self.best_checkpoints)} checkpoints:")
                for rank, (loss, path) in enumerate(self.best_checkpoints, start=1):
                    print(f"  {rank}. {os.path.basename(path)} (val_loss={loss:.4f})")

                return temp_path if temp_path in [p for _, p in self.best_checkpoints] else self.best_checkpoints[0][1]
            else:
                print(
                    f"Checkpoint at epoch {epochs} (val_loss={val_loss:.4f}) not in top {self.max_best_checkpoints}, not saved")
                return None

        return None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _extract_epoch_from_path(self, path):
        """Extract epoch number from checkpoint filename."""
        import re
        match = re.search(r'epoch(\d+)', os.path.basename(path))
        if match:
            return int(match.group(1))
        return 0

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_best_checkpoint_path(self):
        """Returns the path to the best checkpoint (lowest validation loss)."""
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]
        return None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def build_unique_categories(self, annotation_files):
        merged = []
        id_to_name = {}
        name_to_id = {}

        for p in annotation_files:
            try:
                data = json.load(open(p, 'r'))
                cats = data.get('categories', [])
            except Exception as e:
                print(f"Failed loading '{p}': {e}")
                continue

            for cat in cats:
                cid = cat.get('id')
                cname = cat.get('name')
                if cid is None or cname is None:
                    print(f"Warning: bad category entry in '{p}': {cat}")
                    continue

                # Check ID conflict BEFORE assignment
                if cid in id_to_name and id_to_name[cid] != cname:
                    print(f"⚠️ ID conflict: {cid} is '{id_to_name[cid]}' and '{cname}'")
                    continue

                # Check name conflict BEFORE assignment
                if cname in name_to_id and name_to_id[cname] != cid:
                    print(f"⚠️ Name conflict: '{cname}' → {name_to_id[cname]} vs {cid}")
                    continue

                # Only assign if no conflicts
                id_to_name[cid] = cname
                name_to_id[cname] = cid
                merged.append({"id": cid, "name": cname})

        # dedupe
        unique = {(c['id'], c['name']): c for c in merged}
        return list(unique.values())

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def save_config_to_text(self, output_text_file):
        with open(output_text_file, 'w') as text_file:
            # Write the details to the text file
            text_file.write(f"Site: {self.site_name}\n")
            text_file.write(f"Learning Rates: {self.learning_rates}\n")
            text_file.write(f"Optimizer: {self.optimizer_type}\n")
            text_file.write(f"Loss Function: {self.loss_function}\n")
            text_file.write(f"Weight Decay: {self.weight_decay}\n")
            text_file.write(f"Number of Epochs: {self.num_epochs}\n")
            text_file.write(f"Save Model Frequency: {self.save_model_frequency}\n")
            text_file.write(f"Early Stopping: {self.early_stopping}\n")
            text_file.write(f"Patience: {self.patience}\n")
            text_file.write(f"Device: {device}\n")
            text_file.write(f"Folders: {self.folders}\n")
            text_file.write(f"Annotations: {self.annotation_files}\n")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _plot_training_graphs(self, lr: float):
        """
        Generate and save all training/validation plots for a given learning rate.
        """
        progressBar = QProgressWheel(
            title="Generating graphs...", total=9,
            on_close=lambda: setattr(self, "progress_bar_closed", True)
        )

        viz = ModelTrainingVisualization(
            self.model_output_folder, self.formatted_time, self.categories
        )

        # Create epoch lists for training (all epochs) and validation (only validated epochs)
        train_epochs = list(range(1, len(self.loss_values) + 1))
        val_epochs = self.epoch_list  # Only epochs where validation ran

        # 1) Loss curves - use DIFFERENT epoch lists for train vs val
        viz.plot_loss_curves(
            train_epochs=train_epochs,
            train_loss=self.loss_values,
            val_epochs=val_epochs,
            val_loss=self.val_loss_values,
            site_name=self.site_name,
            lr=lr
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 2) Accuracy curves - same issue
        viz.plot_accuracy(
            train_epochs=train_epochs,
            train_acc=self.train_accuracy_values,
            val_epochs=val_epochs,
            val_acc=self.val_accuracy_values,
            site_name=self.site_name,
            lr=lr
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 3) Confusion matrix
        viz.plot_confusion_matrix(
            y_true=self.val_true_list,
            y_pred=self.val_pred_list,
            site_name=self.site_name,
            lr=lr,
            normalize=True,
            file_prefix="Normalized"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 4) ROC Curve + AUC
        viz.plot_roc_curve(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 5) Precision–Recall
        viz.plot_precision_recall(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 6) F1 vs. Threshold
        viz.plot_f1_score(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 7) Mean IoU curve
        viz.plot_miou_curve(
            epochs=self.epoch_list,
            miou_values=self.miou_values,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        viz.plot_dice_curve(
            epochs=self.epoch_list,
            dice_values=self.val_dice_values,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        viz.plot_iou_curve(
            epochs=self.epoch_list,
            iou_values=self.val_iou_values,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        progressBar.close()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def summarize_mask_imbalance(self, images: list[str]) -> dict:
        """
        Computes mean/median/std of foreground ratio (water pixels / total) across given images.
        Uses dataset_util.load_true_mask and your annotation_index.
        """
        ratios = []
        
        # Get target label - handle TEST_MODE
        if self.TEST_MODE:
            # In TEST_MODE, use first test category
            target_label = self.TEST_CATEGORIES[0]
        else:
            # In normal mode, use config
            target_label = self.site_config["train_model"]["TRAINING_CATEGORIES"][0]
        
        target_id = next((c["id"] for c in self.categories if c["name"] == target_label), None)
        for img in images:
            m = self.dataset_util.load_true_mask(img, self.annotation_index, mode="binary", target_id=target_id)
            if m is None:
                continue
            if m.ndim == 3:
                m = m[..., 0]
            total = m.size
            fg = (m > 0).sum()
            ratios.append(float(fg) / float(total) if total > 0 else 0.0)

        if not ratios:
            return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0}

        arr = np.array(ratios, dtype=np.float64)

        return {"count": len(ratios),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std())}

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _terminate_training(self, progressBar):
        msg = "You have cancelled the model training currently in-progress. A model has not been generated."
        msgBox = GRIME_AI_QMessageBox('Model Training Terminated', msg, GRIME_AI_QMessageBox.Close)
        msgBox.displayMsgBox()

        if progressBar and progressBar.isVisible():
            progressBar.close()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _terminate_validation(self, progressBar):
        msg = "You have cancelled the validation currently in-progress."
        msgBox = GRIME_AI_QMessageBox('Validation Terminated', msg, GRIME_AI_QMessageBox.Close)
        msgBox.displayMsgBox()

        if progressBar and progressBar.isVisible():
            progressBar.close()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _deduplicate_centroids(self, tolerance_px=5):
        """
        Deduplicate centroids within each category to prevent memory leak.
        Keeps only unique centroids within tolerance_px distance.
        Call after first epoch to establish representative centroid set.
        """
        for cat_id, centroids in self.category_centroids.items():
            if not centroids:
                continue

            unique_centroids = []
            for new_centroid in centroids:
                cx_new, cy_new = new_centroid["centroid_px"]
                is_unique = True

                for existing in unique_centroids:
                    cx_exist, cy_exist = existing["centroid_px"]
                    dist = ((cx_new - cx_exist) ** 2 + (cy_new - cy_exist) ** 2) ** 0.5
                    if dist < tolerance_px:
                        is_unique = False
                        break

                if is_unique:
                    unique_centroids.append(new_centroid)

            before_count = len(centroids)
            after_count = len(unique_centroids)
            self.category_centroids[cat_id] = unique_centroids

            if before_count > after_count:
                print(f"  Category {cat_id}: Deduplicated {before_count} → {after_count} centroids")
