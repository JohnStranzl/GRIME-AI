# ML_Load_Model.py

from ML_Dependencies import *
import os
import sys
import json
import warnings
import shutil

import cv2
from PIL import Image

from omegaconf import OmegaConf, DictConfig

import matplotlib.pyplot as plt
from pathlib import Path

from GRIME_AI_Save_Utils import GRIME_AI_Save_Utils

# Append SAM2 folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2'))
from sam2.sam2_image_predictor import SAM2ImagePredictor
# We no longer use the legacy build_sam2 method.
from sam2.modeling import sam2_base

from PyQt5.QtWidgets import QMessageBox

from GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI_QMessageBox import GRIME_AI_QMessageBox

print(sam2_base.__file__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Suppress specific warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

np.random.seed(3)


class ML_Load_Model:
    def __init__(self, cfg: DictConfig = None):
        self.className = "ML_Load_Model"

        # If no config is provided—or if there is no "load_model" entry, then fall back to reading a local configuration file.

        if cfg is None or "load_model" not in cfg:
            settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
            config_file = os.path.join(settings_folder, "site_config.json")
            with open(config_file, 'r') as file:
                # Expecting a "load_model" key in the JSON file
                self.config = json.load(file).get("load_model", {})
        else:
            # Convert Hydra DictConfig to a standard dict.a
            self.config = OmegaConf.to_container(cfg.load_model, resolve=True)

        # Set default values with the possibility to override via configuration.
        dirname = os.path.dirname(__file__)

        self.SAM2_CHECKPOINT = self.config.get("SAM2_CHECKPOINT", "")
        self.SAM2_CHECKPOINT = os.path.join(dirname, self.SAM2_CHECKPOINT)

        self.MODEL_CFG = self.config.get("MODEL_CFG", "")
        self.MODEL_CFG = os.path.normpath(self.MODEL_CFG)

        self.INPUT_DIR = self.config.get("INPUT_DIR", "")
        self.INPUT_DIR = os.path.normpath(self.INPUT_DIR)

        self.OUTPUT_DIR = self.config.get("OUTPUT_DIR", "")
        self.OUTPUT_DIR = os.path.normpath(self.OUTPUT_DIR)

        self.MODEL = self.config.get("MODEL", "")
        self.MODEL = os.path.normpath(self.MODEL)

        if self.SAM2_CHECKPOINT == "" or self.MODEL_CFG == "" or self.INPUT_DIR == "" or self.OUTPUT_DIR == "" or self.MODEL == "":
            print ("ERROR: Configuration file missing items.")

        self._check_for_required_files()


    def _check_for_required_files(self):
        nError = 0

        # Collect all the critical paths you need to verify
        paths_to_check = [
            ("Input directory", self.INPUT_DIR),
            ("Trained model file", self.MODEL),
        ]
        #("Model config file", self.MODEL_CFG),
        #("SAM2 checkpoint", self.SAM2_CHECKPOINT),
        #("Output directory", self.OUTPUT_DIR),

        self.missing_items = []
        for name, path in paths_to_check:
            if not os.path.exists(path):
                self.missing_items.append((name, path))

        if self.missing_items:
            nError = -1
            self._show_missing_files_dialog(self.missing_items)

        return nError


    def _show_missing_files_dialog(self, missing_items):
        """
        Show a critical QMessageBox listing all missing files/dirs,
        then terminate the application cleanly.
        """
        # Build a human-readable message
        lines = [
            f"{name}: {path}"
            for name, path in missing_items
        ]
        full_msg = (
            "The following files or directories are missing or have been moved:\n\n"
            + "\n".join(lines) + "\n"
        )

        # Display the error box
        msgBox = GRIME_AI_QMessageBox('Model Configuration Error', full_msg, QMessageBox.Close, icon=QMessageBox.Critical)
        msgBox.displayMsgBox()


    def show_mask(self, mask, ax, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1],
                   color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1],
                   color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h,
                                   edgecolor='green',
                                   facecolor=(0, 0, 0, 0),
                                   lw=2))


    def show_masks(self, output_file_with_path, image, mask, scores, point_coords=None, box_coords=None, input_labels=None,
                   borders=True):
        plt.figure(figsize=(10, 10))
        plt.ioff()
        plt.imshow(image)

        self.show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            self.show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            self.show_box(box_coords, plt.gca())

        plt.axis('off')
        plt.savefig(output_file_with_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        if 0:
            plt.axis('off')
            plt.show()


    def mask_to_polygon(self, mask, min_contour_area=50):
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        if hierarchy is None:
            return segmentation
        for i, contour in enumerate(contours):
            contour = contour.flatten().tolist()
            if len(contour) < 6 or cv2.contourArea(contours[i]) < min_contour_area:
                continue
            if hierarchy[0][i][3] == -1:
                segmentation.append(contour)
            else:
                segmentation.append(contour[::-1])
        return segmentation


    def ML_Load_Model_Main(self, copy_original_image, save_masks, selected_label_categories):

        if self.missing_items:
            return

        global progress_bar_closed
        def on_progress_bar_closed(obj):
            global progress_bar_closed
            progress_bar_closed = True

        progressBar = QProgressWheel()
        progressBar.destroyed.connect(on_progress_bar_closed)
        progress_bar_closed = False
        progressBar.setRange(0, 1)
        progressBar.setValue(1)
        progressBar.show()

        # Reconfirm device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Use a relative path for Hydra initialization.
        config_dir = os.path.join("sam2", "sam2", "configs", "sam2.1")
        # Build the relative model config path.
        model_cfg_path = os.path.join(config_dir, os.path.basename(self.MODEL_CFG))
        print("Model config path:", model_cfg_path)
        print("Checkpoint path:", self.SAM2_CHECKPOINT)

        from hydra import initialize, compose
        from hydra.utils import instantiate

        # Initialize Hydra with a relative config_path.
        with initialize(config_path=config_dir, version_base=None):
            cfg_intern = compose(config_name=os.path.basename(model_cfg_path))
            raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)
            # Remove keys that SAM2Base.__init__ does not expect.
            offending_keys = ["no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs", "device"]
            for key in offending_keys:
                raw_model_cfg.pop(key, None)
            new_cfg = OmegaConf.create(raw_model_cfg)
            model = instantiate(new_cfg, _recursive_=True)

        # Move the instantiated model to the proper device.
        sam2_model = model.to(device)
        predictor = SAM2ImagePredictor(sam2_model)

        # Load the weights (using an absolute path for the model file).
        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, self.MODEL)

        checkpoint = torch.load(model_path, map_location=device)

        # Prepare a COCO-formatted dictionary to save prediction annotations.
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [],
            "licenses": [{
                "name": "",
                "id": 0,
                "url": ""
            }],
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": ""
            },
        }

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            predictor.model.load_state_dict(checkpoint["model_state_dict"])

            coco_data["categories"].extend(selected_label_categories)
        else:
            predictor.model.load_state_dict(checkpoint)

            categories = [{"id": 2, "name": "water"}]
            coco_data["categories"].extend(categories)

        image_id = 1
        annotation_id = 1

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # List all JPEG images from the configured input directory.
        VALID_EXTS = ('.jpg', '.jpeg')
        images_list = [
            f for f in os.listdir(self.INPUT_DIR)
            if f.lower().endswith(VALID_EXTS)
        ]

        progressBar.setRange(0, len(images_list) + 1)

        for img_index, image in enumerate(images_list):
            if progress_bar_closed is False:
                progressBar.setValue(img_index)

                image_path = os.path.join(self.INPUT_DIR, image)
                pil_image = Image.open(image_path).convert("RGB")
                image_array = np.array(pil_image)
                predictor.set_image(image_array)
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    multimask_output=True,
                )
                sorted_ind = np.argmax(scores)
                mask = masks[sorted_ind]
                score = scores[sorted_ind]

                # Display and save the predicted mask overlay.
                os.makedirs(f"{self.OUTPUT_DIR}", exist_ok=True)
                # ### SAVE OVERLAY
                overlay_filename = os.path.splitext(image)[0] + "_overlay.png"
                overlay_output_with_path = os.path.join(self.OUTPUT_DIR, overlay_filename)
                self.show_masks(overlay_output_with_path, pil_image, mask, score, borders=True)

                # ### SAVE MASK
                if save_masks:
                    # Create a new filename by appending '_mask' before the file extension.
                    mask_filename = os.path.splitext(image)[0] + "_mask.png"
                    mask_output_path = os.path.join(self.OUTPUT_DIR, mask_filename)

                    # Convert the binary mask (values 0 or 1) to an 8-bit image (values 0 or 255)
                    mask_to_save = (mask.astype(np.uint8)) * 255

                    cv2.imwrite(mask_output_path, mask_to_save)
                    print(f"Mask saved to {mask_output_path}")

                # === COPY ORIGINAL IMAGE TO OUTPUT_DIR IF ENABLED ===
                if copy_original_image:
                    copied_image_path = os.path.join(self.OUTPUT_DIR, os.path.basename(image_path))
                    try:
                        shutil.copy(image_path, copied_image_path)
                        print(f"Copied original image to: {copied_image_path}")
                    except Exception as e:
                        print(f"Failed to copy image '{image_path}': {e}")

                height, width = image_array.shape[:2]
                image_info = {
                    "file_name": os.path.basename(image),
                    "height": height,
                    "width": width,
                    "id": image_id,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0
                }
                coco_data["images"].append(image_info)

                segmentation = self.mask_to_polygon(mask)
                if not segmentation:
                    continue

                pos = np.where(mask)
                xmin = int(np.min(pos[1]))
                xmax = int(np.max(pos[1]))
                ymin = int(np.min(pos[0]))
                ymax = int(np.max(pos[0]))
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                mask_uint8 = mask.astype(np.uint8)
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 2,
                    "segmentation": segmentation,
                    "area": int(np.sum(mask_uint8)),
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
                image_id += 1
            else:
                strMessage = 'You have cancelled the image segmentation currently in-progress. Not all images have been segmented.'
                msgBox = GRIME_AI_QMessageBox('Image Segmentation Terminated', strMessage, QMessageBox.Close)
                response = msgBox.displayMsgBox()
                break

        # close the progressBar only if the user did not close it (i.e., terminated image segmentation)
        if progress_bar_closed is False:
            progressBar.close()
        del progressBar

        output_file = Path(self.OUTPUT_DIR) / "instances_default.json"
        with open(output_file, "w") as f:
            json.dump(coco_data, f, indent=4)

        print(f"COCO annotations saved to {output_file}")
