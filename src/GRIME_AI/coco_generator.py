#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Define the COCO categories and mapping
CATEGORIES = [
    {"id": 1, "name": "sky", "supercategory": ""},
    {"id": 2, "name": "water", "supercategory": ""},
    {"id": 3, "name": "snow", "supercategory": ""},
    {"id": 4, "name": "unknown", "supercategory": ""}
]
VALID_IDS = {c["id"] for c in CATEGORIES}
LABEL_MAP = {
    255: 2,  # white = water
    0: 4     # black = unknown (optional; remove if not desired)
}


class CocoGenerator:
    """
    A class that generates COCO-style annotation JSON files from image/mask pairs.
    Supports:
     - 1:1 mode using individual mask files (filename ending with '_mask.jpg').
     - Shared mask mode by supplying a single mask applied to all images.
    """

    def __init__(self, folder, shared_mask=None, output_path=None):
        """
        Initialize the generator.

        Args:
            folder (str or Path): Folder containing images (and masks).
            shared_mask (str or Path, optional): Path to a single mask file to use for all images.
            output_path (str or Path, optional): Output JSON file path. Defaults to <folder>/instances_default.json.
        """
        self.folder = Path(folder)
        self.shared_mask = Path(shared_mask) if shared_mask else None
        self.output_path = Path(output_path) if output_path else self.folder / "instances_default.json"

    @staticmethod
    def find_image_mask_pairs(folder):
        """
        Finds image/mask pairs using filenames that end with '_mask.jpg' for the mask.

        Args:
            folder (Path): The folder to search.

        Returns:
            list of tuple: (image_path, mask_path) for each valid pair.
        """
        pairs = []
        for file in os.listdir(folder):
            if file.lower().endswith('_mask.jpg'):
                mask_path = Path(folder) / file
                base_name = file.rsplit('_mask.jpg', 1)[0]
                image_path = Path(folder) / f"{base_name}.jpg"
                if image_path.exists():
                    pairs.append((image_path, mask_path))
                else:
                    print(f"[!] No image found for {file}")
        return pairs

    @staticmethod
    def extract_annotations(mask_array, image_id, start_ann_id):
        """
        Extract COCO-formatted annotation(s) from a mask image array by finding contours.

        Args:
            mask_array (ndarray): The mask image as a numpy array.
            image_id (int or None): The image ID.
            start_ann_id (int): Starting annotation ID.

        Returns:
            tuple: (list of annotation dictionaries, new annotation id)
        """
        annotations = []
        ann_id = start_ann_id

        for pixel_val in np.unique(mask_array):
            if pixel_val not in LABEL_MAP:
                continue
            category_id = LABEL_MAP[pixel_val]
            binary_mask = np.uint8(mask_array == pixel_val)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 3:
                    continue
                segmentation = cnt.flatten().tolist()
                if len(segmentation) < 6:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "bbox": [x, y, w, h],
                    "area": float(area),
                    "iscrowd": 0
                })
                ann_id += 1
        return annotations, ann_id

    def generate_instances_json(self):
        """
        Generates the COCO annotations JSON file. If a shared mask is provided,
        that mask is applied to all images in the folder (excluding the mask file itself);
        otherwise, the generator expects paired image and mask files.
        The output JSON file is saved in the same folder as the input folder by default.
        """
        images = []
        annotations = []
        ann_id = 1
        image_id = 1

        if self.shared_mask:
            # Load the shared mask once.
            mask_array = np.array(Image.open(self.shared_mask).convert("L"))
            shared_anns, _ = CocoGenerator.extract_annotations(mask_array, None, ann_id)
            images_list = [p for p in self.folder.glob("*.jpg") if p.name != self.shared_mask.name]
            if not images_list:
                print("No images found in folder. COCO file will not be created.")
                return
            for img_path in tqdm(images_list, desc="Annotating images with shared mask"):
                img = Image.open(img_path)
                width, height = img.size
                images.append({
                    "id": image_id,
                    "file_name": img_path.name,
                    "height": height,
                    "width": width
                })
                for ann in shared_anns:
                    new_ann = ann.copy()
                    new_ann["image_id"] = image_id
                    new_ann["id"] = ann_id
                    annotations.append(new_ann)
                    ann_id += 1
                image_id += 1
        else:
            pairs = CocoGenerator.find_image_mask_pairs(self.folder)
            if not pairs:
                print("No valid image-mask pairs found. COCO file will not be created.")
                return
            for img_path, mask_path in tqdm(pairs, desc="Processing image-mask pairs"):
                img = Image.open(img_path)
                width, height = img.size
                images.append({
                    "id": image_id,
                    "file_name": img_path.name,
                    "height": height,
                    "width": width
                })
                mask_array = np.array(Image.open(mask_path).convert("L"))
                anns, ann_id = CocoGenerator.extract_annotations(mask_array, image_id, ann_id)
                annotations.extend(anns)
                image_id += 1

        # Only write the output file if images list is not empty.
        if not images:
            print("No images were processed. COCO file will not be created.")
            return

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": CATEGORIES
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w") as f:
            json.dump(coco, f, indent=2)
        print(f"COCO file written to {self.output_path.resolve()}")


    def generate_annotations(self):
        """
        Public method to generate and write the COCO annotation JSON.
        """
        self.generate_instances_json()
