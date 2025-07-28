# ML_SAM.py

from ML_Dependencies import *  # JES - Boy, do I have issues with this. :(
from torchvision.transforms import InterpolationMode
_ = InterpolationMode.BILINEAR  # Ensures inclusion during PyInstaller freeze
import os
import sys
from datetime import datetime
import json

from utils.datasetutils import DatasetUtils

from GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI_QMessageBox import GRIME_AI_QMessageBox

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DEBUG = False  # Set to True if you want print statements


class ML_SAM:

    def __init__(self, cfg: DictConfig = None):
        self.className = "ML_SAM"
        now = datetime.now()
        self.formatted_time = now.strftime('%d%m_%H_%M_%S')

        # load site_config from Hydra or from saved JSON
        if cfg is None or "site_config" not in cfg:
            settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
            site_configuration_file = os.path.normpath(
                os.path.join(settings_folder, "site_config.json")
            )
            print(site_configuration_file)
            with open(site_configuration_file, 'r') as f:
                self.site_config = json.load(f)
        else:
            self.site_config = OmegaConf.to_container(cfg.site_config, resolve=True)

        # unpack common settings
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

        # placeholders for dataset, metrics, model, etc.
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

        # defer DatasetUtils until we know the folders & annotation_files
        self.dataset_util        = None

    def debug_print(self, msg):
        if DEBUG:
            print(msg)

    def find_best_water_points(self, image_path):
        """
        Compute the centroid of the water mask, or fallback to image center.
        """
        try:
            true_mask = self.dataset_util.load_true_mask(image_path)
        except Exception as e:
            print(f"Error loading true mask for {image_path}: {e}")
            true_mask = None

        if true_mask is not None and true_mask.sum() > 0:
            coords = np.argwhere(true_mask > 0)
            centroid = coords.mean(axis=0)  # [row, col]
            return np.array([[int(centroid[1]), int(centroid[0])]])
        else:
            if image_path in self.image_shape_cache:
                h, w = self.image_shape_cache[image_path]
            else:
                img = cv2.imread(image_path)
                if img is None:
                    raise FileNotFoundError(f"Image file {image_path} not found.")
                h, w = img.shape[:2]
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
                true_mask = self.dataset_util.load_true_mask(image_file)
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
                        val_true_mask = self.dataset_util.load_true_mask(val_image_file)
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

        # ------------------------------------------------------------
        # 1. Collect all image‐folders and annotation JSONs from site_config
        # ------------------------------------------------------------
        all_folders = []
        all_annotations = []
        paths = self.site_config.get('Path', [])
        for path in paths:
            directory_path = path['directoryPaths']
            folders = directory_path.get('folders', [])
            annotation_files = directory_path.get('annotations', [])

            if self.site_name == "all_sites" or path.get('siteName') == self.site_name or self.site_name == "custom":
                all_folders.extend(folders)
                all_annotations.extend(annotation_files)

        # ------------------------------------------------------------
        # 2. Instantiate DatasetUtils (loads & indexes in __init__)
        # ------------------------------------------------------------
        self.dataset_util = DatasetUtils(
            image_dirs=all_folders,
            annotation_files=all_annotations
        )

        # expose dataset & index on self if you still rely on them elsewhere
        self.dataset = self.dataset_util.dataset
        self.annotation_index = self.dataset_util.annotation_index

        # ------------------------------------------------------------
        # 3. Build your categories if needed
        # ------------------------------------------------------------
        self.categories = self.build_unique_categories(all_annotations)

        # ------------------------------------------------------------
        # 4. Flatten image paths, split into train / val, and save them
        # ------------------------------------------------------------
        all_images = []
        for entry in self.dataset.values():
            all_images.extend(entry['images'])

        train_images, val_images = self.dataset_util.split_dataset(
            image_list=all_images,
            train_split=0.9,
            seed=42
        )

        self.dataset_util.save_split_dataset(
            train_images=train_images,
            val_images=val_images
        )

        print(f"[DEBUG] train_images count: {len(train_images)}")
        if not train_images:
            raise ValueError("No train images found!")

        # ------------------------------------------------------------
        # 5. Prepare SAM2 config & checkpoint
        # ------------------------------------------------------------
        dirname = os.path.dirname(__file__)
        model_cfg = os.path.normpath(os.path.join(dirname, "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml"))
        sam2_checkpoint = os.path.normpath(os.path.join(dirname, "sam2", "checkpoints", "sam2.1_hiera_large.pt"))

        assert os.path.isfile(model_cfg), f"Config file '{model_cfg}' does not exist."
        assert os.path.isfile(sam2_checkpoint), f"Checkpoint '{sam2_checkpoint}' does not exist."
        print("Model config path:", model_cfg)
        print("Checkpoint path:", sam2_checkpoint)

        # ------------------------------------------------------------
        # 6. Initialize Hydra + instantiate SAM2 model + load weights
        # ------------------------------------------------------------
        config_dir = os.path.join("sam2", "sam2", "configs", "sam2.1")
        print(f"Initializing Hydra with config_path: {config_dir}")

        from hydra import initialize, compose
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        with initialize(config_path=config_dir, version_base=None):
            cfg_intern = compose(
                config_name=os.path.splitext(os.path.basename(model_cfg))[0],
                overrides=[]
            )
            raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)
            for k in ("no_obj_embed_spatial",
                      "use_signed_tpos_enc_to_obj_ptrs",
                      "device"):
                raw_model_cfg.pop(k, None)
            new_cfg = OmegaConf.create(raw_model_cfg)
            model = instantiate(new_cfg, _recursive_=True)
            checkpoint = torch.load(sam2_checkpoint, map_location=device)

            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=False)
            else:
                print("[INFO] Found raw state_dict checkpoint.")
                model.load_state_dict(checkpoint, strict=False)

        self.sam2_model = model.to(device)
        predictor = SAM2ImagePredictor(self.sam2_model)
        predictor.model.sam_mask_decoder.train(True)
        predictor.model.sam_prompt_encoder.train(True)

        # ------------------------------------------------------------
        # 7. Training loop over learning rates
        # ------------------------------------------------------------
        for lr in self.learning_rates:
            print(f"Training with learning rate: {lr}")
            self.train_sam(
                lr, self.weight_decay,
                predictor,
                train_images,
                val_images,
                epochs=self.num_epochs
            )

            # synchronize epoch & loss lengths
            if len(self.epoch_list) != len(self.loss_values):
                print(f"[WARNING] Epoch list has {len(self.epoch_list)} entries, "
                      f"loss list has {len(self.loss_values)} entries.")
                self.epoch_list = self.epoch_list[:len(self.loss_values)]

            # plot loss
            plt.plot(self.epoch_list, self.loss_values, marker='*')
            plt.title('Epoch vs loss')
            plt.xlabel('Epoch');
            plt.ylabel('Loss')
            plt.savefig(f"{self.site_name}_EpochVsLoss_{lr}_{self.formatted_time}.png")
            plt.close()

            # plot accuracy
            plt.plot(self.epoch_list, self.train_accuracy_values, marker='o', label="Train Acc")
            plt.plot(self.epoch_list, self.val_accuracy_values, marker='s', label="Val Acc")
            plt.xlabel("Epoch");
            plt.ylabel("Accuracy")
            plt.title("Training and Validation Accuracy")
            plt.legend()
            plt.savefig(f"{self.site_name}_Accuracy_{lr}_{self.formatted_time}.png")
            plt.close()

        # ------------------------------------------------------------
        # 8. Save final config & print timing
        # ------------------------------------------------------------
        self.save_config_to_text(f"{self.site_name}_configuration_{self.formatted_time}.txt")
        now_end = datetime.now()
        print(f"Execution Ended:   {now_end.strftime('%y%m%d %H:%M:%S')}")


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
