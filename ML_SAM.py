# SAM.py

from ML_Dependencies import *
import os
import sys
import shutil
from datetime import datetime

if True:
    import logging
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.disable(logging.INFO)

# ----------------------------------------------------------------------------------------------------------------------
# HYDRA (for SAM2)
# ----------------------------------------------------------------------------------------------------------------------
from omegaconf import OmegaConf, DictConfig

from torch.cuda.amp import GradScaler

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
        self.formatted_time = now.strftime('%d%m_%H%M')

        if cfg is None or "site_config" not in cfg:
            dirname = os.path.dirname(__file__)
            site_configuration_file = os.path.normpath(os.path.join(dirname, "site_config.json"))
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

        self.site_name = self.site_config['siteName']
        self.learning_rates = self.site_config['learningRates']
        self.optimizer_type = self.site_config['Optimizer']
        self.loss_function = self.site_config['loss_function']
        self.weight_decay = self.site_config['weight_decay']
        self.num_epochs = self.site_config['number_of_epochs']
        self.save_model_frequency = self.site_config['save_model_frequency']
        self.early_stopping = self.site_config['early_stopping']
        self.patience = self.site_config['patience']
        self.device = self.site_config.get('device', str(device))

        self.site_short_name = self.site_name.split('_')[-1]

        self.dataset = {}

        self.loss_values = []
        self.val_loss_values = []
        self.epoch_list = []
        self.train_accuracy_values = []
        self.val_accuracy_values = []

        self.sam2_model = None
        self.folders = None
        self.annotation_files = None

        self.image_shape_cache = {}


    def debug_print(self, msg):
        if DEBUG:
            print(msg)

    def load_images_and_annotations(self, folders, annotation_files):
        dataset = {}

        for folder, annotation_file in zip(folders, annotation_files):
            water_category_id = None
            images = [f for f in os.listdir(folder) if f.endswith('.jpg')]

            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            if water_category_id is None:
                for category in annotations.get('categories', []):
                    if category['name'] == 'water':
                        water_category_id = category['id']
                        break

            if water_category_id is None:
                raise ValueError(f"The 'water' category is not found in {annotation_file}.")

            water_annotations = [
                ann for ann in annotations['annotations']
                if ann['category_id'] == water_category_id
            ]

            dataset[folder] = {
                "images": [os.path.join(folder, img) for img in images],
                "annotations": {
                    "images": annotations["images"],
                    "annotations": water_annotations
                }
            }

        return dataset


    def build_annotation_index(self):
        """
        Build and return a mapping from image file basenames to their corresponding annotation data.
        """
        annotation_index = {}
        for folder, data in self.dataset.items():
            for image_path in data["images"]:
                # Using the basename of the image as the key
                base_name = os.path.basename(image_path)
                annotation_index[base_name] = data["annotations"]
        return annotation_index


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
            true_mask = self.load_true_mask(image_path, self.dataset)
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


    def split_dataset(self, dataset_dict, train_split=0.9, val_split=0.1):
        all_images = []
        for data in dataset_dict.values():
            all_images.extend(data["images"])
        random.shuffle(all_images)

        num_images = len(all_images)
        train_size = int(train_split * num_images)

        train_images = all_images[:train_size]
        val_images = all_images[train_size:]
        print(f"Train: {len(train_images)} images, Validation: {len(val_images)} images")
        return train_images, val_images

    np.random.seed(3)


    def save_split_dataset(self, train_images, val_images):
        # Set the output directory to the current execution directory
        output_dir = os.getcwd()

        # Create train and validation directories within the current directory
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "validation")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        # Copy training images
        for image_path in train_images:
            file_name = os.path.basename(image_path)
            dest_path = os.path.join(train_dir, file_name)
            shutil.copy2(image_path, dest_path)  # copy2 preserves metadata
            print(f"Copied train image: {image_path} -> {dest_path}")

        # Copy validation images
        for image_path in val_images:
            file_name = os.path.basename(image_path)
            dest_path = os.path.join(val_dir, file_name)
            shutil.copy2(image_path, dest_path)
            print(f"Copied validation image: {image_path} -> {dest_path}")

        if len(train_images) == 0 or len(val_images) == 0:
            print("Empty split — cannot train/validate properly.")
            return


    def show_mask(self, mask, ax, random_color=False, borders = True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            import cv2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):    
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def show_masks(self, image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                #boxes
            	self.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()


    def load_image_and_annotation(self, folder, image_file, annotations):
        image_path = os.path.join(folder, image_file)
        image = np.array(Image.open(image_path))

        image_info = next((img for img in annotations['images'] if img['file_name'] == image_file), None)

        if image_info is None:
            raise ValueError(f"Image file {image_file} not found in annotations.")

        image_id = image_info['id']
        annotation = next((ann for ann in annotations['annotations'] if ann['image_id'] == image_id), None)

        if annotation is None:
            raise ValueError(f"Annotation for image ID {image_id} not found.")

        return image, annotation


    def load_true_mask(self, image_file, dataset):
        """
        Efficiently loads the true mask for an image by using the precomputed annotation_index.
        """
        base_name = os.path.basename(image_file)
        if not hasattr(self, 'annotation_index') or base_name not in self.annotation_index:
            raise ValueError(f"Image file {image_file} not found in the annotation index.")

        annotation_data = self.annotation_index[base_name]

        # Find image metadata
        image_info = next((img for img in annotation_data['images'] if img['file_name'] == base_name), None)
        if image_info is None:
            raise ValueError(f"Image file {image_file} not found in annotations.")

        image_id, height, width = image_info['id'], image_info['height'], image_info['width']

        # Get all annotations for the image
        annotations_for_image = [ann for ann in annotation_data['annotations'] if ann['image_id'] == image_id]
        if not annotations_for_image:
            return np.zeros((height, width), dtype=np.uint8)

        # Initialize an empty mask
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        # Iterate over each annotation and decode RLE mask
        for ann in annotations_for_image:
            rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
            mask = coco_mask.decode(rle)

            # Merge multiple segmentation parts if necessary
            if len(mask.shape) == 3:
                mask = np.any(mask, axis=2)

            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)

        return combined_mask.astype(np.float32)


    def train_sam(self, learnrate, weight_decay, predictor, train_images, val_images, input_point_op, input_label_op,
                  epochs=20):
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
                input_point = self.find_best_water_points(image_file)
                input_label = np.ones(len(input_point), dtype=int)
                image = np.array(Image.open(image_file).convert("RGB"))
                true_mask = self.load_true_mask(image_file, self.dataset)

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

            # Average epoch loss
            avg_epoch_loss = epoch_loss / len(train_images)
            train_accuracy = train_correct / train_total
            self.train_accuracy_values.append(train_accuracy)
            self.loss_values.append(avg_epoch_loss)
            print(f"Loss Values: {self.loss_values}")

            print(f"Epoch {epoch + 1} Training Loss: {avg_epoch_loss}")

            # Validation step
            if val_images is not None:
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for val_idx, val_image_file in enumerate(val_images):
                        val_image = np.array(Image.open(val_image_file).convert("RGB"))
                        val_true_mask = self.load_true_mask(val_image_file, self.dataset)

                        if val_true_mask is None:
                            print(f"No annotation found for validation image {val_image_file}, skipping.")
                            continue

                        predictor.set_image(val_image)
                        masks, scores, _ = predictor.predict(point_coords=None, point_labels=input_label,multimask_output=False)

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

                avg_val_loss = val_loss / len(val_images)
                val_accuracy = val_correct / val_total
                self.val_accuracy_values.append(val_accuracy)
                self.val_loss_values.append(avg_val_loss)
                print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss}")
            if (epoch + 1) % self.save_model_frequency == 0:
                torch.save(predictor.model.state_dict(),
                           f"{self.site_short_name}_{epoch}_{learnrate}_{self.formatted_time}.torch")


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

        paths = self.site_config.get('Path', [])
        for path in paths:
            if path['siteName'] == self.site_name:
                directory_path = path['directoryPaths']
                self.folders = directory_path.get('folders', [])
                self.annotation_files = directory_path.get('annotations', [])

        self.dataset = self.load_images_and_annotations(self.folders, self.annotation_files)
        self.annotation_index = self.build_annotation_index()

        # Split dataset into train and validation sets
        train_images, val_images = self.split_dataset(self.dataset)
        self.save_split_dataset(train_images, val_images)

        print(f"[DEBUG] train_images count: {len(train_images)}")
        if len(train_images) == 0:
            raise ValueError("No train images found!")

        input_point = np.array([[427, 917]])
        input_label = np.array([1])

        import os
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
        import os

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

        # Move the newly instantiated model to the proper device.
        self.sam2_model = model.to(device)

        # Now create the predictor.
        predictor = SAM2ImagePredictor(self.sam2_model)

        predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
        predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder

        # Iterate over learning rates and run training.
        for learnrate in self.learning_rates:
            print(f"Training with learning rate: {learnrate}")
            self.train_sam(learnrate, self.weight_decay, predictor, train_images, val_images,
                           input_point, input_label, epochs=20)


            plt.plot(self.epoch_list, self.loss_values, marker='*')
            plt.title('Epoch vs loss')
            plt.xlabel('Epoch')
            plt.ylabel('loss')
            plt.savefig(f"{self.site_short_name}_EpochVsLoss_{learnrate}_{self.formatted_time}.png")
            plt.close()


            plt.plot(self.epoch_list, self.train_accuracy_values, label="Training Accuracy", marker='o')
            plt.plot(self.epoch_list, self.val_accuracy_values, label="Validation Accuracy", marker='s')
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training and Validation Accuracy Over Epochs")
            plt.legend()
            plt.savefig(f"{self.site_short_name}_Accuracy_{learnrate}_{self.formatted_time}.png")
            plt.close()
        self.save_config_to_text(f"{self.site_short_name}_configuration_{self.formatted_time}.txt")

        now_end = datetime.now()
        print(f"Execution Started: {now_start.strftime('%y%m%d %HH:%MM:%SS')}")
        print(f"Execution Ended: {now_end.strftime('%y%m%d %HH:%MM:%SS')}")
