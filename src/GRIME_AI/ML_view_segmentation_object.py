#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# view_segmentation_object.py

from GRIME_AI.ML_Dependencies import *
from datetime import datetime

from omegaconf import DictConfig

# Configure warnings (using the warnings module imported in dependencies.py)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

now = datetime.now()
formatted_time = now.strftime('%d%m_%H%M')

class ML_view_segmentation_object:

    def __init__(self, cfg: DictConfig = None):
        self.className = "ML_view_segmentation_object"

    def load_images_and_annotations(self, folders, annotation_files):
        dataset = {}
        for folder, annotation_file in zip(folders, annotation_files):
            water_category_id = None
            VALID_EXTS = ('.jpg', '.jpeg')
            images = [
                f for f in os.listdir(folder)
                if f.lower().endswith(VALID_EXTS)
            ]

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

    np.random.seed(3)

    def show_mask(self, mask, ax, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])

        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        # Multiply only the RGB channels for OpenCV compatibility
        mask_image = mask.reshape(h, w, 1) * color[:3].reshape(1, 1, -1)

        if borders:
            mask_binary = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image_cv = (mask_image * 255).astype(np.uint8)
            if mask_image_cv.shape[-1] == 3:
                mask_image_cv = cv2.cvtColor(mask_image_cv, cv2.COLOR_RGB2BGR)
            cv2.drawContours(mask_image_cv, contours, -1, (255, 255, 255), thickness=2)
            mask_image = mask_image_cv.astype(np.float32) / 255.0
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
                   marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
                   marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                                   facecolor=(0, 0, 0, 0), lw=2))

    def show_masks(self, image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                self.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

    def load_true_mask(self, image_file, dataset):
        annotation_data = None
        for folder, data in dataset.items():
            if image_file in data["images"]:
                annotation_data = data["annotations"]
                break
        if annotation_data is None:
            raise ValueError(f"Image file {image_file} not found in the dataset.")
        file_name_only = os.path.basename(image_file)
        image_info = next((img for img in annotation_data['images'] if img['file_name'] == file_name_only), None)
        if image_info is None:
            raise ValueError(f"Image file {image_file} not found in annotations.")
        image_id, height, width = image_info['id'], image_info['height'], image_info['width']
        annotations_for_image = [ann for ann in annotation_data['annotations'] if ann['image_id'] == image_id]
        if not annotations_for_image:
            return np.zeros((height, width), dtype=np.uint8)
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for ann in annotations_for_image:
            rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
            mask = coco_mask.decode(rle)
            if len(mask.shape) == 3:
                mask = np.any(mask, axis=2)
            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
        return combined_mask.astype(np.float32)

    def ML_view_segmentation_object_main(self):
        # Define file-specific folders and annotation files
        folders = [
            "c:/MD_Lake_Serene_Edgewood/COCO Products/CT_MD_Lake_Serene_Edgewood_COCO/images/default",
            "c:/MD_Lake_Serene_Edgewood/COCO Products/DK_MD_Lake_Serene_Edgewood_COCO/images/default"
        ]
        annotation_files = [
            "c:/MD_Lake_Serene_Edgewood/COCO Products/CT_MD_Lake_Serene_Edgewood_COCO/annotations/instances_default.json",
            "c:/MD_Lake_Serene_Edgewood/COCO Products/DK_MD_Lake_Serene_Edgewood_COCO/annotations/instances_default.json"
        ]

        tmp_folders = [os.path.normpath(folder) for folder in folders]
        folders = tmp_folders

        tmp_file = [os.path.normpath(ann_file) for ann_file in annotation_files]
        annotation_files = tmp_file

        # Specify the image path (adjust as necessary)
        image_path = "c:/MD_Lake_Serene_Edgewood/COCO Products/DK_MD_Lake_Serene_Edgewood_COCO/images/default/MD_Lake_Serene_at_Edgewood___2024-01-12T14-15-02Z.jpg"
        image_path = os.path.normpath(image_path)

        dataset = self.load_images_and_annotations(folders, annotation_files)
        image = Image.open(image_path)
        image_np = np.array(image)
        true_mask = self.load_true_mask(image_path, dataset)

        if true_mask is not None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image_np)
            self.show_mask(true_mask, ax, borders=True)
            plt.axis("off")
            plt.show()
        else:
            print(f"No mask found for {image_path}")
