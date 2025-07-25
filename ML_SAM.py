# ML_SAM.py

from ML_Dependencies import *  # JES - Boy, do I have issues with this. :(
from torchvision.transforms import InterpolationMode
_ = InterpolationMode.BILINEAR  # Ensures inclusion during PyInstaller freeze
import os
import sys
from datetime import datetime
import json  # Ensure json is imported since we use it later

from utils.datasetutils import DatasetUtils

from GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI_QMessageBox import GRIME_AI_QMessageBox

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

        now = datetime.now()
        self.formatted_time = now.strftime('%d%m_%H_%M_%S')

        if cfg is None or "site_config" not in cfg:
            settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
            site_configuration_file = os.path.normpath(os.path.join(settings_folder, "site_config.json"))
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

        predictor.model.sam_mask_decoder.train(True)
        predictor.model.sam_prompt_encoder.train(True)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            self.epoch_list.append(epoch + 1)
            epoch_loss, train_correct, train_total = 0.0, 0, 0
            print(f"\nEpoch {epoch + 1}/{epochs}")

            np.random.shuffle(train_images)
            scaler = GradScaler() if device.type == "cuda" else None
            optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=learnrate, weight_decay=weight_decay)

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
                if predictor._features is None or "high_res_feats" not in predictor._features:
                    print(f"[ERROR] Predictor features not initialized for {image_file}. Skipping.")
                    continue

                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=None, boxes=None, masks=None
                )

                high_res_features = [
                    feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]
                ]

                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=True,
                    high_res_features=high_res_features,
                )

                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                if len(true_mask.shape) == 3:
                    true_mask = true_mask[..., 0]

                gt_mask = torch.tensor(true_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
                if gt_mask.sum() == 0:
                    print(f"Skipping {image_file} - ground-truth mask is empty.")
                    continue

                prd_mask = prd_masks[:, 0]
                prd_mask = torch.sigmoid(prd_mask).unsqueeze(1)

                if prd_mask.shape != gt_mask.shape:
                    raise ValueError(f"Mismatched shapes for {image_file}: {prd_mask.shape} vs {gt_mask.shape}")

                seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log(
                    1 - prd_mask + 1e-5)).mean()
                inter = (gt_mask * (prd_mask > 0.5)).sum((1, 2, 3))
                union = gt_mask.sum((1, 2, 3)) + (prd_mask > 0.5).sum((1, 2, 3)) - inter
                iou = inter / (union + 1e-6)
                iou[union == 0] = 1.0
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                loss = seg_loss + 0.05 * score_loss

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.detach().item()
                print(
                    f"{datetime.now().strftime('%H:%M:%S')}: Image {idx + 1}/{len(train_images)} processed. Loss: {loss.item()}")

                pred_binary = (prd_mask > 0.5).cpu().numpy()
                true_binary = gt_mask.cpu().numpy()
                train_correct += np.sum(pred_binary == true_binary)
                train_total += np.prod(true_binary.shape)

                if not progress_bar_closed:
                    global_iteration += 1
                    progressBar.setValue(global_iteration)

            if train_total > 0:
                avg_epoch_loss = epoch_loss / max(1, len(train_images))
                train_accuracy = train_correct / train_total
                self.train_accuracy_values.append(train_accuracy)
                self.loss_values.append(avg_epoch_loss)
                print(f"Epoch {epoch + 1} Training Loss: {avg_epoch_loss}")
            else:
                print(f"Epoch {epoch + 1}: No training samples processed.")
                continue

            if val_images:
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
                            print(f"No annotation for {val_image_file}, skipping.")
                            continue

                        predictor.set_image(val_image)
                        masks, scores, _ = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            multimask_output=False
                        )

                        if masks.size > 0:
                            best_mask = masks[np.argmax(scores)]
                            if len(val_true_mask.shape) > 2:
                                val_true_mask = val_true_mask[:, :, 0]

                            pred = torch.tensor(best_mask, dtype=torch.float32).unsqueeze(0).to(device)
                            gt = torch.tensor(val_true_mask, dtype=torch.float32).unsqueeze(0).to(device)

                            val_loss += loss_fn(pred, gt).item()
                            pred_binary = (pred > 0.5).cpu().numpy()
                            true_binary = gt.cpu().numpy()
                            val_correct += np.sum(pred_binary == true_binary)
                            val_total += np.prod(true_binary.shape)

                            global_iteration += 1
                            progressBar.setValue(global_iteration)

                if val_total > 0:
                    self.val_accuracy_values.append(val_correct / val_total)
                    self.val_loss_values.append(val_loss / len(val_images))
                    print(f"Epoch {epoch + 1} Validation Loss: {val_loss / len(val_images)}")

            if (epoch + 1) % self.save_model_frequency == 0:
                models_folder = GRIME_AI_Save_Utils().get_models_folder()
                torch_file = os.path.join(models_folder,
                                          f"{self.site_name}_{epoch}_{learnrate}_{self.formatted_time}.torch")

                # Wrap into a checkpoint dictionary
                checkpoint = {
                    "model_state_dict": predictor.model.state_dict(),  # replace with your model
                    "categories": self.categories
                }

                torch.save(checkpoint, torch_file)
                print(f"[INFO] Model checkpoint saved: {torch_file}")

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


    def _terminate_training(self, progressBar):
        msg = "You have cancelled the model training currently in-progress. A model has not been generated."
        msgBox = GRIME_AI_QMessageBox('Model Training Terminated', msg, GRIME_AI_QMessageBox.Close)
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
        all_folders = []
        all_annotations = []

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
            elif path['siteName'] == self.site_name:
                directory_path = path['directoryPaths']
                self.folders = directory_path.get('folders', [])
                self.annotation_files = directory_path.get('annotations', [])
                self.all_folders.extend(self.folders)
                self.all_annotations.extend(self.annotation_files)

        self.dataset = self.dataset_util.load_images_and_annotations(self.all_folders, self.all_annotations)
        self.annotation_index = self.dataset_util.build_annotation_index(self.dataset)

        self.categories = self.build_unique_categories(self.all_annotations)

        # Split dataset into train and validation sets
        train_images, val_images = self.dataset_util.split_dataset(self.dataset)
        self.dataset_util.save_split_dataset(train_images, val_images)

        print(f"[DEBUG] train_images count: {len(train_images)}")
        if len(train_images) == 0:
            raise ValueError("No train images found!")

        dirname = os.path.dirname(__file__)
        # Build absolute path for the SAM2 YAML config file
        model_cfg = os.path.join(dirname, "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")
        model_cfg = os.path.normpath(model_cfg)
        print("Model config path:", model_cfg)

        # (Optionally, you might use sam2_checkpoint later to load weights.)
        sam2_checkpoint = os.path.join(dirname, "sam2", "checkpoints", "sam2.1_hiera_large.pt")
        sam2_checkpoint = os.path.normpath(sam2_checkpoint)
        assert os.path.isfile(model_cfg), f"Config file '{model_cfg}' does not exist."
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
            # Use the config name without the ".yaml" extension.
            cfg_intern = compose(config_name=os.path.splitext(os.path.basename(model_cfg))[0], overrides=[])
            raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)
            # Filter out keys that SAM2Base.__init__ does not expect.
            offending_keys = [
                "no_obj_embed_spatial",
                "use_signed_tpos_enc_to_obj_ptrs",
                "device"
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
        for learnrate in self.learning_rates:
            print(f"Training with learning rate: {learnrate}")
            self.train_sam(learnrate, self.weight_decay, predictor, train_images, val_images, epochs=self.num_epochs)

            if len(self.epoch_list) != len(self.loss_values):
                print(f"[WARNING] Mismatch detected! Epoch list contains {len(self.epoch_list)} entries, "
                      f"but loss values only contain {len(self.loss_values)} entries.")
                print(f"Epoch list: {self.epoch_list}")
                print(f"Loss values: {self.loss_values}")
                self.epoch_list = self.epoch_list[:len(self.loss_values)]

            plt.plot(self.epoch_list, self.loss_values, marker='*')
            plt.title('Epoch vs loss')
            plt.xlabel('Epoch')
            plt.ylabel('loss')
            plt.savefig(f"{self.site_name}_EpochVsLoss_{learnrate}_{self.formatted_time}.png")
            plt.close()


            plt.plot(self.epoch_list, self.train_accuracy_values, label="Training Accuracy", marker='o')
            plt.plot(self.epoch_list, self.val_accuracy_values, label="Validation Accuracy", marker='s')
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training and Validation Accuracy Over Epochs")
            plt.legend()
            plt.savefig(f"{self.site_name}_Accuracy_{learnrate}_{self.formatted_time}.png")
            plt.close()
        self.save_config_to_text(f"{self.site_name}_configuration_{self.formatted_time}.txt")

        now_end = datetime.now()
        print(f"Execution Started: {now_start.strftime('%y%m%d %H:%M:%S')}")
        print(f"Execution Ended: {now_end.strftime('%y%m%d %H:%M:%S')}")


    def build_unique_categories(self, annotation_files):
        merged_categories = []
        id_to_name = {}
        name_to_id = {}

        for path in annotation_files:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    categories = data.get("categories", [])
            except Exception as e:
                print(f"Failed to load '{path}': {e}")
                continue

            for cat in categories:
                cat_id = cat.get("id")
                cat_name = cat.get("name")

                if cat_id is None or cat_name is None:
                    print(f"Warning: Category in '{path}' missing 'id' or 'name': {cat}")
                    continue

                # Check for ID–name consistency
                if cat_id in id_to_name:
                    if id_to_name[cat_id] != cat_name:
                        print(f"⚠️ ID conflict in '{path}': ID {cat_id} is '{id_to_name[cat_id]}' but also used for '{cat_name}'")
                        continue  # Skip conflicting entry
                else:
                    id_to_name[cat_id] = cat_name

                # Check for name–ID consistency
                if cat_name in name_to_id:
                    if name_to_id[cat_name] != cat_id:
                        print(f"⚠️ Name conflict in '{path}': Name '{cat_name}' has ID {name_to_id[cat_name]} and also ID {cat_id}")
                        continue  # Skip conflicting entry
                else:
                    name_to_id[cat_name] = cat_id

                # Safe to add
                merged_categories.append({
                    "id": cat_id,
                    "name": cat_name
                })

        # Remove duplicates by ID–name pair
        unique_categories = { (cat["id"], cat["name"]): cat for cat in merged_categories }
        return list(unique_categories.values())