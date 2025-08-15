
import os
import json
import cv2
import shutil
import random
import numpy as np
from pycocotools import mask as coco_mask

class DatasetUtils:
    def __init__(self):
        self.image_shape_cache = {}

    def load_images_and_annotations(self, folders, annotation_files):
        dataset = {}

        for folder, annotation_file in zip(folders, annotation_files):
            water_category_id = None
            images = [f for f in os.listdir(folder) if f.endswith('.jpg')]

            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            if water_category_id is None:
                for category in annotations.get('categories', []):
                    if category['name'] == 'water' or category['name'] == 'Vegetation':
                        water_category_id = category['id']
                        break

            if water_category_id is None:
                raise ValueError(f"The 'water' or 'Vegetation' category were not found in {annotation_file}.")

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

    def build_annotation_index(self, dataset):
        """
        Build and return a mapping from image file basenames to their corresponding annotation data.
        """
        annotation_index = {}
        for folder, data in dataset.items():
            for image_path in data["images"]:
                base_name = os.path.basename(image_path)
                annotation_index[base_name] = data["annotations"]
        return annotation_index

    def load_true_mask(self, image_file,annotation_index):
        """
        Efficiently loads the true mask for an image by using the precomputed annotation_index.
        """
        base_name = os.path.basename(image_file)
        if base_name not in annotation_index:
            raise ValueError(f"Image file {image_file} not found in the annotation index.")

        annotation_data = annotation_index[base_name]

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

    def save_split_dataset(self, train_images, val_images):
        output_dir = os.getcwd()
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "validation")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for image_path in train_images:
            file_name = os.path.basename(image_path)
            dest_path = os.path.join(train_dir, file_name)
            shutil.copy2(image_path, dest_path)
            print(f"Copied train image: {image_path} -> {dest_path}")

        for image_path in val_images:
            file_name = os.path.basename(image_path)
            dest_path = os.path.join(val_dir, file_name)
            shutil.copy2(image_path, dest_path)
            print(f"Copied validation image: {image_path} -> {dest_path}")

        if len(train_images) == 0 or len(val_images) == 0:
            print("Empty split — cannot train/validate properly.")
