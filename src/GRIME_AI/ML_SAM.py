#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# ML_SAM.py

from GRIME_AI.ML_Dependencies import *  # JES - Boy, do I have issues with this. :(

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchvision.transforms import InterpolationMode
_ = InterpolationMode.BILINEAR  # Ensures inclusion during PyInstaller freeze

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
from datetime import datetime
import json
import random

from GRIME_AI.utils.datasetutils import DatasetUtils

from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.GRIME_AI_Model_Training_Visualization import GRIME_AI_Model_Training_Visualization

if True:
    import logging
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.disable(logging.INFO)


# ----------------------------------------------------------------------------------------------------------------------
# HYDRA (for SAM2)
# ----------------------------------------------------------- -----------------------------------------------------------
from omegaconf import OmegaConf, DictConfig

from torch.cuda.amp import autocast, GradScaler


import hydra
from hydra.core.global_hydra import GlobalHydra

sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2'))
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling import sam2_base
print(sam2_base.__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DEBUG = False  # Set to True if you want print statements


# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====      INLINE FUNCTIONS      =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
def compute_mean_iou(y_true: list[int], y_pred: list[int]) -> float:
    arr_t = np.array(y_true, dtype=bool)
    arr_p = np.array(y_pred, dtype=bool)
    inter = np.logical_and(arr_t, arr_p).sum()
    union = np.logical_or(arr_t, arr_p).sum()
    return float(inter) / float(union) if union > 0 else 1.0


class ML_SAM:

    def __init__(self, cfg: DictConfig = None):
        self.className = "ML_SAM"

        self.model_output_folder = None

        # ALL FILES SAVED WITH BE TAGGED WITH THE DATE AND TIME THAT TRAINING STARTED.
        self.now = datetime.now()
        self.formatted_time = self.now.strftime('%Y%m%d_%H%M%S')

        # load site_config from Hydra or from saved JSON
        if cfg is None or "site_config" not in cfg:
            settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
            CONFIG_FILENAME = "site_config.json"
            site_configuration_file = os.path.normpath(
                os.path.join(settings_folder, CONFIG_FILENAME)
            )
            print(site_configuration_file)

            with open(site_configuration_file, 'r') as file:
                self.site_config = json.load(file)
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
        self.device = self.site_config.get('device', str(device))

        self.dataset = {}

        self.loss_values = []
        self.val_loss_values = []
        self.epoch_list = []
        self.train_accuracy_values = []
        self.val_accuracy_values = []
        self.val_true_list = []
        self.val_pred_list = []
        self.val_score_list = []
        # track mean–IoU per epoch
        self.miou_values = []

        self.sam2_model = None
        self.folders = None
        self.annotation_files = None
        self.all_folders = []
        self.all_annotations = []
        self.categories = []

        self.image_shape_cache = {}

        # objects for other classes
        self.dataset_util = DatasetUtils()

    def debug_print(self, msg):
        if DEBUG:
            print(msg)

    def find_best_water_points(self, image_path):
        """
        Finds a water point by computing the centroid of the annotated water mask (true mask).
        If no water region is found (or an error occurs), defaults to returning the center of the image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            np.ndarray: A numpy array of shape (1, 2) containing the coordinate [x, y] of the water point.
        """
        try:
            true_mask = self.dataset_util.load_true_mask(image_path, self.annotation_index)
        except Exception as e:
            print(f"Error loading true mask for {image_path}: {e}")
            true_mask = None

        if true_mask is not None and true_mask.sum() > 0:
            # Compute the centroid of the water region using nonzero indices
            indices = np.argwhere(true_mask > 0)
            centroid = indices.mean(axis=0)  # [row, col]
            # Return in (x, y) order
            return np.array([[int(centroid[1]), int(centroid[0])]])
        else:
            # Fallback: return the center of the image if no valid water region is found,
            # using the cached dimensions if available.
            if image_path in self.image_shape_cache:
                h, w = self.image_shape_cache[image_path]
            else:
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"Image file {image_path} not found.")
                h, w = image.shape[:2]
                self.image_shape_cache[image_path] = (h, w)
            return np.array([[w // 2, h // 2]])


    def train_sam(self, learnrate, weight_decay, predictor, train_images, val_images, epochs=20):
        progress_bar_closed = False

        # reset all metrics
        self.epoch_list.clear()
        self.loss_values.clear()
        self.train_accuracy_values.clear()
        self.val_loss_values.clear()
        self.val_accuracy_values.clear()
        self.val_true_list.clear()
        self.val_pred_list.clear()
        self.val_score_list.clear()

        total_iterations = epochs * (len(train_images) + (len(val_images) if val_images else 0))
        global_iteration = 0

        progressBar = self._make_progress_bar("Training in-progress...", total_iterations)

        # Seed for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Force deterministic kernels
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Build model, optimizer, predictor
        self.sam2_model.train()
        predictor = SAM2ImagePredictor(self.sam2_model)
        optimizer = torch.optim.AdamW(self.sam2_model.parameters(), lr=learnrate, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        # Initialize the GradScaler just once per epoch if using CUDA.
        scaler = GradScaler() if device.type == "cuda" else None

        for epoch in range(epochs):

            # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # TRAIN ON EACH EPOCH
            # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            self.sam2_model.train()

            self.epoch_list.append(epoch + 1)
            epoch_loss, train_correct, train_total = 0.0, 0, 0
            print(f"\nEpoch {epoch + 1}/{epochs}")

            np.random.shuffle(train_images)

            # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # TRAIN ON ALL IMAGES FOR EACH EPOCH
            # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            for idx, image_file in enumerate(train_images):
                if progress_bar_closed:
                    self._terminate_training(progressBar)
                    return

                image = np.array(Image.open(image_file).convert("RGB"))
                true_mask = self.dataset_util.load_true_mask(image_file, self.annotation_index)

                if true_mask is None:
                    print(f"No annotation found for image {image_file}, skipping.")
                    continue

                predictor.set_image(image)

                # Prepare prompts for mask prediction
                # mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None
                )

                # Mask decoder prediction
                batched_mode = True  # unnorm_coords.shape[0] > 1  # multi-object prediction
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in
                                     predictor._features["high_res_feats"]]

                optimizer.zero_grad()
                use_amp = False
                with autocast(enabled=use_amp):
                    # ─── Multi-backend scaled-dot-product-attention ───
                    desired = ("FLASH_ATTENTION", "XFORMERS", "DEFAULT", "MATMUL")
                    backends = []
                    for name in desired:
                        if hasattr(SDPBackend, name):
                            backends.append(getattr(SDPBackend, name))
                        else:
                            warnings.warn(f"SDPBackend has no attribute {name!r}; skipping.")

                    # Attempt each available backend, fall back to direct call if all fail
                    last_exc = None
                    for backend in backends:
                        try:
                            with sdpa_kernel(backend):
                                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                                    image_embeddings         = predictor._features["image_embed"][-1].unsqueeze(0),
                                    image_pe                 = predictor.model.sam_prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings = sparse_embeddings,
                                    dense_prompt_embeddings  = dense_embeddings,
                                    multimask_output         = False,
                                    repeat_image             = batched_mode,
                                    high_res_features        = high_res_features,
                                )
                            print(f"Using SDPA backend: {backend.name}")
                            break
                        except Exception as e:
                            warnings.warn(f"SDPA backend {backend.name!r} failed: {e}")
                            last_exc = e
                    else:
                        warnings.warn(
                            f"All SDPA kernels failed "
                            f"({', '.join(b.name for b in backends)}): {last_exc}. "
                            "Falling back to default kernel."
                        )
                        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                            image_embeddings         = predictor._features["image_embed"][-1].unsqueeze(0),
                            image_pe                 = predictor.model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings = sparse_embeddings,
                            dense_prompt_embeddings  = dense_embeddings,
                            multimask_output         = False,
                            repeat_image             = batched_mode,
                            high_res_features        = high_res_features,
                        )
                    # ───────────────────────────────────────────────────

                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                    #if prd_masks.shape != true_mask.shape:
                    #    print(f"prd_mask shape {prd_masks.shape} and true mask shapes {true_mask.shape} are different.")

                    # If the ground-truth mask is 3D, keep just one channel

                    if len(true_mask.shape) == 3:
                        true_mask = true_mask[..., 0]  # shape -> [H, W]

                    # Convert to a float32 tensor on GPU, and add batch & channel dimensions
                    gt_mask = torch.tensor(true_mask, dtype=torch.float32, device=device)  # [H, W]
                    gt_mask = gt_mask.unsqueeze(0).unsqueeze(1)  # [1,1,H,W]

                    # If there are no positive pixels, optionally skip
                    if gt_mask.sum() == 0:
                        print(f"Skipping {image_file} - ground-truth mask is empty.")
                        continue

                    # prd_masks is [1,3,H,W] if multimask_output=True. Pick the first or best mask:
                    prd_mask = prd_masks[:, 0]  # [1,H,W]
                    prd_mask = torch.sigmoid(prd_mask).unsqueeze(1)  # [1,1,H,W]

                    # Now check that they match
                    if prd_mask.shape != gt_mask.shape:
                        raise ValueError(
                            f"Mismatched shapes for {image_file}: {prd_mask.shape} vs {gt_mask.shape}"
                        )

                    # Segmentation Loss using binary cross entropy formula
                    seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

                    # Score Loss (IOU)
                    inter = (gt_mask * (prd_mask > 0.5)).sum((1, 2, 3))  # If shape is [B,1,H,W]
                    union = gt_mask.sum((1, 2, 3)) + (prd_mask > 0.5).sum((1, 2, 3)) - inter

                    # union might be zero. Let's create a boolean mask:
                    zero_union_mask = (union == 0)

                    # Option A: set IoU=1 if union == 0 and intersection == 0
                    iou = inter / (union + 1e-6)  # add epsilon to avoid dividing by zero
                    iou[zero_union_mask] = 1.0

                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    loss = seg_loss + 0.05 * score_loss

                optimizer.zero_grad()
                if use_amp and scaler is not None:
                    # Wrap the backward pass in autocast to leverage mixed-precision training
                    # Note: In many cases it is preferable to include the forward pass in the autocast context,
                    # but if your forward pass was already done outside, using the scaler here enables scaling.
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.detach().item()
                now_start = datetime.now()
                print(f"{now_start.strftime('%H:%M:%S')}: Image {idx + 1}/{len(train_images)} processed. Loss: {loss.item()}")

                # Accuracy calculation
                pred_binary = (prd_mask > 0.5).cpu().numpy()
                true_binary = gt_mask.cpu().numpy()
                train_correct += np.sum(pred_binary == true_binary)
                train_total += np.prod(true_binary.shape)

                # Inside the training loop
                if true_mask is None:
                    print(f"[DEBUG] Skipping {image_file}: no annotation")
                    continue

                if not progress_bar_closed:
                    global_iteration += 1
                    progressBar.setValue(global_iteration)

            # Average epoch loss
            avg_epoch_loss = epoch_loss / len(train_images)
            self.loss_values.append(avg_epoch_loss)

            train_accuracy = train_correct / train_total
            self.train_accuracy_values.append(train_accuracy)
            print(f"Loss Values: {self.loss_values}")

            print(f"Epoch {epoch + 1} Training Loss: {avg_epoch_loss}")


            # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # VALIDATION
            # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            if val_images:
                self.sam2_model.eval()

                progressBar.setWindowTitle("Validation in-progress")

                val_loss, val_correct, val_total = 0.0, 0, 0

                # VARIABLES FOR CALCULATING THE CONFUSION MATRIX
                with torch.no_grad():
                    for val_idx, val_image_file in enumerate(val_images):
                        if progress_bar_closed:
                            self._terminate_validation(progressBar)
                            return

                        val_image = np.array(Image.open(val_image_file).convert("RGB"))
                        val_true_mask = self.dataset_util.load_true_mask(val_image_file, self.annotation_index)

                        if val_true_mask is None:
                            print(f"No annotation found for validation image {val_image_file}, skipping.")
                            continue

                        predictor.set_image(val_image)
                        masks, scores, low_res_logits  = predictor.predict(point_coords=None, point_labels=None, multimask_output=False)

                        if masks.size > 0:
                            best_mask = masks[np.argmax(scores)]

                            # Remove the extra dimension from val_true_mask_tensor
                            if len(val_true_mask.shape) > 2:
                                val_true_mask = val_true_mask[:, :, 0]

                            best_mask_tensor = torch.tensor(best_mask, dtype=torch.float32).unsqueeze(0).to(
                                device.type)  # Shape: [1, 1080, 1920]
                            val_true_mask_tensor = torch.tensor(val_true_mask, dtype=torch.float32).unsqueeze(0).to(
                                device.type)  # Shape: [1, 1080, 1920, 1]

                            val_loss += loss_fn(best_mask_tensor, val_true_mask_tensor).item()
                            pred_binary = (best_mask_tensor > 0.5).cpu().numpy()
                            true_binary = val_true_mask_tensor.cpu().numpy()
                            val_correct += np.sum(pred_binary == true_binary)
                            val_total += np.prod(true_binary.shape)

                            # EXTEND THE LISTS FOR THE CONFUSION MATRIX
                            # flatten your arrays (no .astype here; conversion happens in the comprehension)
                            true_flat = true_binary.flatten()
                            pred_flat = pred_binary.flatten()
                            self.val_true_list.extend(int(x) for x in true_flat)
                            self.val_pred_list.extend(int(x) for x in pred_flat)

                            # extend with native Python ints/floats (satisfies Iterable[int/float])
                            # scoring with raw logits → full [0,1] distribution
                            best_idx = int(np.argmax(scores))
                            best_logit = low_res_logits [best_idx]
                            best_logit_tensor = torch.tensor(best_logit, dtype=torch.float32).unsqueeze(0).to(device)

                            import torch.nn.functional as F

                            # pick the same best‐mask index
                            best_idx = int(np.argmax(scores))

                            # build a tensor from the low‐res logit
                            logit_lr = torch.tensor(
                                low_res_logits[best_idx],
                                dtype=torch.float32,
                                device=device
                            ).unsqueeze(0).unsqueeze(0)  # shape [1,1,H_lr,W_lr]

                            # convert to probabilities
                            prob_lr = torch.sigmoid(logit_lr)

                            # upsample to match full resolution
                            H_full, W_full = val_true_mask_tensor.shape[1], val_true_mask_tensor.shape[2]
                            prob_full = F.interpolate(
                                prob_lr,
                                size=(H_full, W_full),
                                mode='bilinear',
                                align_corners=False
                            )

                            # flatten and extend scores list
                            score_flat = prob_full.squeeze(0).squeeze(0).cpu().numpy().flatten()
                            self.val_score_list.extend(score_flat.tolist())

                            global_iteration += 1
                            progressBar.setValue(global_iteration)

                avg_val_loss = val_loss / len(val_images)
                self.val_loss_values.append(avg_val_loss)
                val_accuracy = val_correct / val_total
                self.val_accuracy_values.append(val_accuracy)
                # compute Mean IoU for this epoch

                miou = compute_mean_iou(self.val_true_list, self.val_pred_list)
                self.miou_values.append(miou)

                print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss}")

        # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        # SAVE MODEL
        # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        timestamp = self.now.strftime("%Y%m%d_%H%M%S")
        ckpt = {
            "model_state_dict": predictor.model.state_dict(),
            "categories": self.categories,
            "creation_UTC": timestamp,
            "site_name": self.site_name,
            "learning_rate": learnrate,
            "epochs": epochs
        }
        torch_filename = f"{self.formatted_time}_{self.site_name}_final_{learnrate}.torch"
        torch.save(ckpt, os.path.join(self.model_output_folder, torch_filename))

        # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        # CLEAN-UP
        # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        if not progress_bar_closed:
            progressBar.close()
        del progressBar


    def _terminate_training(self, progressBar):
        msg = "You have cancelled the model training currently in-progress. A model has not been generated."
        msgBox = GRIME_AI_QMessageBox('Model Training Terminated', msg, GRIME_AI_QMessageBox.Close)
        msgBox.displayMsgBox()
        if progressBar:
            progressBar.close()
        del progressBar


    def _terminate_validation(self, progressBar):
        msg = "You have cancelled validation. The current validation pass was not completed."
        msgBox = GRIME_AI_QMessageBox("Validation Cancelled", msg, GRIME_AI_QMessageBox.Close)
        msgBox.displayMsgBox()
        if progressBar:
            progressBar.close()
        del progressBar


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

    def ML_SAM_Main(self, cfg=None):
        now_start = datetime.now()
        print(f"Execution Started: {now_start.strftime('%y%m%d %H:%M:%S')}")

        # all files created during model training will have a time/date prefix corresponding to the start of model training.
        now_start = datetime.now()
        print(f"Execution Started: {now_start.strftime('%y%m%d %H:%M:%S')}")

        # create output folder in user's Documents/GRIME-AI/Models folder for the models
        try:
            self.model_output_folder = os.path.join(GRIME_AI_Save_Utils().get_models_folder(), f"{self.formatted_time}_{self.site_name}")
            os.makedirs(self.model_output_folder, exist_ok=True)
        except OSError as e:
            print(f"Error creating folders: {e}")

        paths = self.site_config.get('Path', [])
        for path in paths:
            # If site_name is mentioned as "all_sites" in the site_config.json, then all the sites are considered i.e., all the folders and annotatons listed under the Path in the json
            if self.site_name == "all_sites":
                directory_path = path['directoryPaths']
                self.folders = directory_path.get('folders', [])
                self.annotation_files = directory_path.get('annotations', [])
                self.all_folders.extend(self.folders)
                self.all_annotations.extend(self.annotation_files)
            # If site_name is specific site then only the folders and annotatons of that particular site is considered for training. 
            #JES WHY DOES IT HAVE TO EQUAL A SITE NAME???? elif path['siteName'] == self.site_name:
            else:
                directory_path = path['directoryPaths']
                self.folders = directory_path.get('folders', [])
                self.annotation_files = directory_path.get('annotations', [])
                self.all_folders.extend(self.folders)
                self.all_annotations.extend(self.annotation_files)

        self.dataset = self.dataset_util.load_images_and_annotations(self.all_folders, self.all_annotations)
        self.annotation_index = self.dataset_util.build_annotation_index(self.dataset)

        # 3. Build categories if available
        self.categories = self.build_unique_categories(self.all_annotations)

        # Split dataset into train and validation sets
        train_images, val_images = self.dataset_util.split_dataset(self.dataset)
        split_dataset_filename = os.path.join(self.model_output_folder, f"{self.formatted_time}_{self.site_name}training_and_validation_sets.json")
        self.dataset_util.save_split_dataset(train_images, val_images, split_dataset_filename)

        print(f"[DEBUG] train_images count: {len(train_images)}")
        if len(train_images) == 0:
            print("No training images found!")
            #JES raise ValueError("No train images found!")
            return

        dirname = os.path.dirname(__file__)
        # Build absolute path for the SAM2 YAML config file
        model_cfg = os.path.join(dirname, "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")
        model_cfg = os.path.normpath(model_cfg)
        print("Model config path:", model_cfg)

        # (Optionally, you might use sam2_checkpoint later to load weights.)
        sam2_checkpoint = os.path.join(dirname, "sam2", "checkpoints", "sam2.1_hiera_large.pt")
        sam2_checkpoint = os.path.normpath(sam2_checkpoint)
        print("Checkpoint path:", sam2_checkpoint)

        ### NEW: Instead of calling build_sam2(), we load and instantiate SAM2 model ourselves.
        # Hydra requires that the config_path be relative.
        config_dir = os.path.join("sam2", "sam2", "configs", "sam2.1")
        print(f"Initializing Hydra with config_path: {config_dir}")

        from hydra import initialize, compose
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # GlobalHydra.instance().clear()
        # Initialize Hydra explicitly with the relative path. (version_base can be None to suppress version warnings.)
        with initialize(config_path=config_dir, version_base=None):
            # The config_name should be the base name of the YAML file.
            cfg_intern = compose(config_name=os.path.basename(model_cfg))
            # Convert to a plain Python dictionary.
            raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)
            # Filter out keys that SAM2Base.__init__ does not expect.
            offending_keys = [
                "no_obj_embed_spatial",
                      "use_signed_tpos_enc_to_obj_ptrs",
                "device"  # avoid passing "device" into SAM2Base
            ]
            for key in offending_keys:
                raw_model_cfg.pop(key, None)
            # Recreate a DictConfig from the filtered dictionary.
            new_cfg = OmegaConf.create(raw_model_cfg)
            # Instantiate the model without passing the extra keyword.
            model = instantiate(new_cfg, _recursive_=True)
            checkpoint = torch.load(sam2_checkpoint, map_location=device)  # or 'cuda' if you prefer

            # if model key is in checkpoint
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=False)
            else:
                print("[INFO] Found raw state_dict checkpoint.")
                model.load_state_dict(checkpoint, strict=False)

        # Move the newly instantiated model to the proper device.
        self.sam2_model = model.to(device)

        # Now create the predictor.
        predictor = SAM2ImagePredictor(self.sam2_model)

        predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
        predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder

        # Iterate over learning rates and run training.
        # instantiate the visualizer
        for lr in self.learning_rates:
            print(f"Training with learning rate: {lr}")

            # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # TRAIN AND VALIDATE MODEL
            # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            self.train_sam(lr, self.weight_decay, predictor, train_images, val_images, epochs=self.num_epochs)

            # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # PLOT GRAPHS
            # *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            progressBar = self._make_progress_bar("Generating graphs...", 7)
            progressBar.show()

            # instantiate the viz once for this LR
            viz = GRIME_AI_Model_Training_Visualization(self.model_output_folder, self.formatted_time, self.categories)

            plot_index = 0
            # 1) Loss curves
            viz.plot_loss_curves(
                epochs=self.epoch_list,
                train_loss=self.loss_values,
                val_loss=self.val_loss_values,
                site_name=self.site_name,
                lr=lr
            )
            progressBar.setValue(plot_index := plot_index + 1)
            progressBar.show()

            # 2) Accuracy curves
            viz.plot_accuracy(
                epochs=self.epoch_list,
                train_acc=self.train_accuracy_values,
                val_acc=self.val_accuracy_values,
                site_name=self.site_name,
                lr=lr
            )
            progressBar.setValue(plot_index := plot_index + 1)
            progressBar.show()

            # 3) Confusion matrix
            viz.plot_confusion_matrix(
                y_true      = self.val_true_list,
                y_pred      = self.val_pred_list,
                site_name   = self.site_name,
                lr          = lr,
                normalize   = True,
                file_prefix = "Normalized"
            )
            progressBar.setValue(plot_index := plot_index + 1)
            progressBar.show()

            # 4) ROC Curve + AUC
            viz.plot_roc_curve(
                y_true      = self.val_true_list,
                y_scores    = self.val_score_list,
                site_name   = self.site_name,
                lr          = lr,
                file_prefix = f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
            )
            progressBar.setValue(plot_index := plot_index + 1)
            progressBar.show()

            # 5) Precision–Recall
            viz.plot_precision_recall(
                y_true=self.val_true_list,
                y_scores=self.val_score_list,
                site_name=self.site_name,
                lr=lr,
                file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
            )
            progressBar.setValue(plot_index := plot_index + 1)
            progressBar.show()

            # 6) F1 vs. Threshold
            viz.plot_f1_score(
                y_true=self.val_true_list,
                y_scores=self.val_score_list,
                site_name=self.site_name,
                lr=lr,
                file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
            )
            progressBar.setValue(plot_index := plot_index + 1)
            progressBar.show()

            # 7) Mean IoU curve
            viz.plot_miou_curve(
                epochs=self.epoch_list,
                miou_values=self.miou_values,
                site_name=self.site_name,
                lr=lr,
                file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
            )
            progressBar.setValue(plot_index := plot_index + 1)
            progressBar.show()

            progressBar.close()
            del progressBar

        config_file = os.path.join(self.model_output_folder, f"{self.formatted_time}_{self.site_name}_configuration.txt")
        self.save_config_to_text(config_file)

        #--------------------------------------------------------------------------------------------------------------
        #8. Save final config & print timing
        #--------------------------------------------------------------------------------------------------------------
        now_end = datetime.now()
        print(f"Execution Started: {now_start.strftime('%y%m%d %HH:%MM:%SS')}")
        print(f"Execution Ended: {now_end.strftime('%y%m%d %HH:%MM:%SS')}")


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
                cid = cat.get('id'); cname = cat.get('name')
                if cid is None or cname is None:
                    print(f"Warning: bad category entry in '{p}': {cat}")
                    continue

                # check ID↔name consistency
                if cid in id_to_name and id_to_name[cid] != cname:
                    print(f"⚠️ ID conflict: {cid} is '{id_to_name[cid]}' and '{cname}'")
                    continue
                id_to_name[cid] = cname

                if cname in name_to_id and name_to_id[cname] != cid:
                    print(f"⚠️ Name conflict: '{cname}' → {name_to_id[cname]} vs {cid}")
                    continue
                name_to_id[cname] = cid

                merged.append({"id": cid, "name": cname})

        # dedupe
        unique = {(c['id'], c['name']): c for c in merged}
        return list(unique.values())


    def _make_progress_bar(self, title: str, total: int):
        """
        Create, configure and show a QProgressWheel.
        Sets self.progress_bar_closed=True when the window is destroyed.
        """
        self.progress_bar_closed = False

        pb = QProgressWheel()
        pb.setWindowTitle(title)
        pb.destroyed.connect(lambda _: setattr(self, "progress_bar_closed", True))
        pb.setRange(0, total)
        pb.setValue(1)
        pb.show()

        return pb
