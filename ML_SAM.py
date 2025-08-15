# ML_SAM.py

from ML_Dependencies import *  # JES - Boy, do I have issues with this. :(
from torchvision.transforms import InterpolationMode
_ = InterpolationMode.BILINEAR  # Ensures inclusion during PyInstaller freeze
import os
import sys
from datetime import datetime
import json
from typing import Callable, Optional

from PyQt5.QtWidgets import QWidget

from utils.datasetutils import DatasetUtils

from GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI_Model_Training_Visualization import GRIME_AI_Model_Training_Visualization


if True:
    import logging
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.disable(logging.INFO)

# ----------------------------------------------------------------------------------------------------------------------
# HYDRA (for SAM2)
# ----------------------------------------------------------------------------------------------------------------------
from omegaconf import OmegaConf, DictConfig

from torch.cuda.amp import GradScaler


import hydra
from hydra.core.global_hydra import GlobalHydra

sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2'))
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling import sam2_base
print(sam2_base.__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DEBUG = False  # Set to True if you want print statements


class ML_SAM:

    def __init__(self, cfg: DictConfig = None):
        self.className = "ML_SAM"

        self.model_output_folder = None

        # ALL FILES SAVED WITH BE TAGGED WITH THE DATE AND TIME THAT TRAINING STARTED.
        now = datetime.now()
        self.formatted_time = now.strftime('%Y%m%d_%H%M%S')

        # load site_config from Hydra or from saved JSON
        if cfg is None or "site_config" not in cfg:
            settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
            site_configuration_file = os.path.normpath(
                os.path.join(settings_folder, "site_config.json")
            )
            print(site_configuration_file)

            with open(site_configuration_file, 'r') as file:
                self.site_config = json.load(file)
        else:
            # Convert the Hydra DictConfig to a standard dict using OmegaConf.to_container.
            self.site_config = OmegaConf.to_container(cfg.site_config, resolve=True)

        #dirname = os.path.dirname(__file__)
        #site_configuration_file = os.path.normpath(os.path.join(dirname, "site_config.json"))
        #with open(site_configuration_file, 'r') as file:
        #    self.site_config = json.load(file)

        self.site_name           = self.site_config['siteName']
        self.learning_rates      = self.site_config['learningRates']
        self.optimizer_type      = self.site_config['optimizer']
        self.loss_function       = self.site_config['loss_function']
        self.weight_decay        = self.site_config['weight_decay']
        self.num_epochs          = self.site_config['number_of_epochs']
        self.save_model_frequency= self.site_config['save_model_frequency']
        self.early_stopping      = self.site_config['early_stopping']
        self.patience            = self.site_config['patience']
        self.device              = self.site_config.get('device', str(device))

        self.dataset             = {}
        self.loss_values         = []
        self.val_loss_values     = []
        self.epoch_list          = []
        self.train_accuracy_values = []
        self.val_accuracy_values   = []

        self.sam2_model          = None
        self.folders             = None
        self.annotation_files    = None
        self.all_folders         = []
        self.all_annotations     = []
        self.categories          = []

        self.image_shape_cache   = {}

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

        def on_progress_bar_closed(obj):
            nonlocal progress_bar_closed
            progress_bar_closed = True

        total_iterations = epochs * (len(train_images) + (len(val_images) if val_images else 0))
        global_iteration = 0

        progressBar = QProgressWheel()
        progressBar.setWindowTitle("Training in-progress...")
        progressBar.destroyed.connect(on_progress_bar_closed)
        progressBar.setRange(0, total_iterations)
        progressBar.setValue(1)
        progressBar.show()

        optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=learnrate, weight_decay=weight_decay)
        self.sam2_model.train()
        predictor = SAM2ImagePredictor(self.sam2_model)

        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            self.epoch_list.append(epoch + 1)
            epoch_loss, train_correct, train_total = 0.0, 0, 0
            print(f"\nEpoch {epoch + 1}/{epochs}")

            np.random.shuffle(train_images)

            # Initialize the GradScaler just once per epoch if using CUDA.
            if device.type == "cuda":
                scaler = GradScaler()
            else:
                scaler = None

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
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )

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

                #print(f"Final shapes -> prd_mask: {prd_mask.shape}, gt_mask: {gt_mask.shape}")

                # Ensure both tensors have the same shape
                #print(f"Modified gt_mask shape: {gt_mask.shape}")
                #print(f"Modified prd_mask shape: {prd_mask.shape}")

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

                # If you prefer iou=0 for empty-empties, do:
                # iou[zero_union_mask] = 0.0

                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                loss = seg_loss + 0.05 * score_loss

                if 0:
                    print(f"Segmentation loss is : {seg_loss}")
                    print(f"iou is : {iou}")
                    print(f"score loss is : {score_loss}")
                    print(f"total loss is : {loss}")
            
                optimizer.zero_grad()
                if scaler is not None:
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
            train_accuracy = train_correct / train_total
            self.train_accuracy_values.append(train_accuracy)
            self.loss_values.append(avg_epoch_loss)
            print(f"Loss Values: {self.loss_values}")

            print(f"Epoch {epoch + 1} Training Loss: {avg_epoch_loss}")

            # Validation step
            if val_images is not None:
                progressBar.setWindowTitle("Validation in-progress")
                val_loss, val_correct, val_total = 0.0, 0, 0
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
                        masks, scores, _ = predictor.predict(point_coords=None, point_labels=None,multimask_output=False)

                        if masks.size > 0:
                            best_mask = masks[np.argmax(scores)]
                            if False:
                                print(f"best_mask:{best_mask.shape}")
                                print(f"value_mask:{val_true_mask.shape}")

                            # Remove the extra dimension from val_true_mask_tensor
                            if len(val_true_mask.shape) > 2:
                                val_true_mask = val_true_mask[:, :, 0]

                            best_mask_tensor = torch.tensor(best_mask, dtype=torch.float32).unsqueeze(0).to(
                                device.type)  # Shape: [1, 1080, 1920]
                            val_true_mask_tensor = torch.tensor(val_true_mask, dtype=torch.float32).unsqueeze(0).to(
                                device.type)  # Shape: [1, 1080, 1920, 1]

                            if False:
                                print(f"best_mask after change:{best_mask_tensor.shape}")
                                print(f"value_mask after change:{val_true_mask_tensor.shape}")

                            '''
                            ## changing

                            gt_mask = torch.tensor(val_true_mask.astype(np.float32)).cuda()#.unsqueeze(0  # Add batch dimension


                            # Reshape gt_mask to remove the last dimension if present
                            if gt_mask.shape[-1] == 1:  # Check if the last dimension is singleton
                                gt_mask = gt_mask.squeeze(-1)  # Remove the last dimension

                            #Add channel dimension to prd_mask
                            prd_mask = torch.sigmoid(prd_masks[:, 0]).unsqueeze(1)  # Add channel dimension

                            # Ensure both tensors have the same shape
                            print(f"Modified gt_mask shape: {gt_mask.shape}")
                            print(f"Modified prd_mask shape: {prd_mask.shape}")

                            prd_mask = torch.sigmoid(prd_masks[:, 0])#.unsqueeze(1)
                            # Segmentation Loss
                            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

                            # Score Loss (IOU)
                            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                            # Combine losses
                            loss = seg_loss + score_loss * 0.05
                            '''

                            val_loss += loss_fn(best_mask_tensor, val_true_mask_tensor).item()
                            pred_binary = (best_mask_tensor > 0.5).cpu().numpy()
                            true_binary = val_true_mask_tensor.cpu().numpy()
                            val_correct += np.sum(pred_binary == true_binary)
                            val_total += np.prod(true_binary.shape)

                            global_iteration += 1
                            progressBar.setValue(global_iteration)

                avg_val_loss = val_loss / len(val_images)
                val_accuracy = val_correct / val_total
                self.val_accuracy_values.append(val_accuracy)
                self.val_loss_values.append(avg_val_loss)
                print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss}")

        ckpt = {
            "model_state_dict": predictor.model.state_dict(),
            "categories": self.categories
        }
        torch_filename = f"{self.formatted_time}_{self.site_name}_final_{learnrate}.torch"
        torch.save(ckpt, os.path.join(self.model_output_folder, torch_filename))

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
        self.dataset_util.save_split_dataset(train_images, val_images)

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
            self.train_sam(lr, self.weight_decay, predictor, train_images, val_images, epochs=self.num_epochs)

            #JES FUTURE
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            #viz = GRIME_AI_Model_Training_Visualization(self.model_output_folder, self.formatted_time, self.categories)
            #viz.epoch_list = self.epoch_list
            #viz.train_accuracy_values = self.train_accuracy_values
            #viz.val_accuracy_values = self.val_accuracy_values

            # 1. Epoch vs. Loss Plot
            #viz.plot_loss(self.site_name, learnrate)

            # 2. Training vs. Validation Accuracy
            #viz.plot_accuracy(self.site_name, learnrate)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            # plot loss
            plt.plot(self.epoch_list, self.loss_values, marker='*')
            plt.title('Epoch vs loss')
            plt.xlabel('Epoch');
            plt.ylabel('Loss')
            plot_output_file = os.path.join(self.model_output_folder, f"{self.formatted_time}_{self.site_name}_EpochVsLoss_{lr}.png")
            plt.savefig(plot_output_file)
            plt.close()

            # plot accuracy
            plt.plot(self.epoch_list, self.train_accuracy_values, marker='o', label="Train Acc")
            plt.plot(self.epoch_list, self.val_accuracy_values, marker='s', label="Val Acc")
            plt.xlabel("Epoch");
            plt.ylabel("Accuracy")
            plt.title("Training and Validation Accuracy")
            plt.legend()
            plot_output_file = os.path.join(self.model_output_folder, f"{self.formatted_time}_{self.site_name}_Accuracy_{lr}.png")
            plt.savefig(plot_output_file)
            plt.close()

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