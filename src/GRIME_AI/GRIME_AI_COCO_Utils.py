#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

#!/usr/bin/env python3
"""
coco_utils.py

Utility class for validating, cleaning, and analyzing COCO 1.0 JSON.
"""

import json
import os
import sys
import random

import openpyxl
from openpyxl.utils import get_column_letter

from typing import Any, Dict, List, Tuple
import cv2

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


class GRIME_AI_COCO_Utils:
    """Utility class for validating, cleaning, and summarizing a COCO-format JSON file."""

    def __init__(self, folder_path: str) -> None:
        self.folder_path: str = folder_path
        self.json_filename: str = ""
        self.json_path: str = ""
        self.data: Dict[str, Any] = {}


    def find_json_file(self) -> None:
        try:
            entries = os.listdir(self.folder_path)
        except FileNotFoundError:
            print(f"Error: folder not found -> {self.folder_path}")
            sys.exit(1)

        json_files = [f for f in entries if f.lower().endswith(".json")]
        if not json_files:
            print(f"No JSON file found in {self.folder_path}")
            sys.exit(1)

        if len(json_files) > 1:
            print(f"Multiple JSON files found: {json_files}")
            sys.exit(1)

        self.json_filename = json_files[0]
        self.json_path = os.path.join(self.folder_path, self.json_filename)


    def load_json(self) -> None:
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if "images" not in self.data:
            print("Invalid COCO JSON: missing 'images' section.")
            sys.exit(1)


    def load_coco(self) -> None:
        """
        Convenience method to find and load the COCO JSON in one call.
        """
        self.find_json_file()
        self.load_json()


    def check_images(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        present_images: List[Dict[str, Any]] = []
        missing_images: List[Dict[str, Any]] = []

        for img in self.data["images"]:
            fname = img.get("file_name", "")
            path = os.path.join(self.folder_path, fname)
            if fname and os.path.isfile(path):
                present_images.append(img)
            else:
                missing_images.append(img)

        return present_images, missing_images


    def backup_original(self) -> None:
        orig = os.path.join(self.folder_path, self.json_filename)
        backup = orig + ".ORIGINAL"
        os.rename(orig, backup)


    def clean_data(
        self, present_images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        new_data = dict(self.data)
        new_data["images"] = present_images

        if "annotations" in self.data:
            valid_ids = {img["id"] for img in present_images}
            new_data["annotations"] = [
                ann
                for ann in self.data["annotations"]
                if ann.get("image_id") in valid_ids
            ]

        return new_data


    def write_json(self, new_data: Dict[str, Any]) -> None:
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4)


    def process(self) -> None:
        self.find_json_file()
        self.load_json()

        present, missing = self.check_images()
        if not missing:
            print("All images in JSON are present.")
            return

        self.backup_original()
        cleaned = self.clean_data(present)
        self.write_json(cleaned)
        print("New JSON created for the images available in the folder.")


    def get_image_label_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Returns a mapping from image file names to a dict of category names
        and their respective annotation counts (mask counts).
        """
        # Build lookups
        id_to_name = {c["id"]: c["name"] for c in self.data.get("categories", [])}
        id_to_file = {img["id"]: img["file_name"] for img in self.data.get("images", [])}

        # Initialize counts dict
        counts: Dict[str, Dict[str, int]] = {}
        for fname in id_to_file.values():
            counts[fname] = {name: 0 for name in id_to_name.values()}

        # Tally annotations
        for ann in self.data.get("annotations", []):
            img_id = ann.get("image_id")
            cat_id = ann.get("category_id")
            if img_id in id_to_file and cat_id in id_to_name:
                fname = id_to_file[img_id]
                cat_name = id_to_name[cat_id]
                counts[fname][cat_name] += 1

        return counts


    def write_image_label_counts_to_xlsx(self, output_file: str) -> None:
        """
        Writes the image-wise label counts to an Excel file.

        The sheet will have one row per image, with columns:
        [Image Name, <Label1>, <Label2>, …].

        :param output_filepath: Path to write the .xlsx file.
        """
        # 1) Ensure data is loaded
        if not self.data:
            raise RuntimeError("COCO data not loaded. Call load_coco() first.")

        # 2) Fetch counts dict: { image_name: { label: count, … }, … }
        counts = self.get_image_label_counts()

        # 3) Create workbook & sheet
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Image Label Counts"

        # 4) Build header row
        #    - first col is "Image"
        #    - rest are all label names, sorted alphabetically
        all_labels = list(next(iter(counts.values())).keys())
        header = ["Image"] + sorted(all_labels)
        ws.append(header)

        # 5) Populate rows
        for img_name, label_map in counts.items():
            row = [img_name] + [label_map.get(lbl, 0) for lbl in header[1:]]
            ws.append(row)

        # 6) Auto-adjust column widths
        for idx, col_title in enumerate(header, start=1):
            max_length = max(
                len(str(cell.value)) for cell in ws[get_column_letter(idx)]
            )
            # add a little padding
            ws.column_dimensions[get_column_letter(idx)].width = max_length + 2

        # 7) Save to disk
        output_file = os.path.join(self.folder_path, output_file)
        wb.save(output_file)


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def extract_masks(self, image_dir, output_dir):
        print("Extract masks from COCO file...")

        # Load COCO annotations
        coco_annotation_file = os.path.join(image_dir, 'instances_default.json')
        coco = COCO(coco_annotation_file)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Prepare list to collect image → labels mapping
        all_results = []

        # Seed for reproducible random colors
        random.seed(0)

        # Dynamic map: category name → random BGR color
        color_map = {}

        # Iterate all images in the COCO
        img_ids = coco.getImgIds()
        for img_id in img_ids:
            # Load image info and file
            img_info = coco.loadImgs(img_id)[0]
            filename = img_info['file_name']
            base, _ = os.path.splitext(filename)
            img_path = os.path.join(image_dir, filename)

            # Read image to get dimensions
            image = cv2.imread(img_path)
            height, width, _ = image.shape

            # Prepare masks
            gray_mask = np.zeros((height, width), dtype=np.uint8)
            color_mask = np.zeros((height, width, 3), dtype=np.uint8)

            # Gather annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            labels = []
            for ann in anns:
                # Decode RLE to a 0/1 mask
                rle = coco.annToRLE(ann)
                binary_mask = maskUtils.decode(rle).astype(bool)

                # Merge into your single gray mask
                gray_mask = np.maximum(gray_mask, binary_mask.astype(np.uint8) * 255)

                # Get category name
                cat = coco.loadCats(ann['category_id'])[0]['name']
                labels.append(cat)

                # If this label is new, assign it a random color
                if cat not in color_map:
                    # ensure up to 32 distinct colors
                    b = random.randint(0, 255)
                    g = random.randint(0, 255)
                    r = random.randint(0, 255)
                    color_map[cat] = (b, g, r)

                # Paint that category’s region
                color_mask[binary_mask] = color_map[cat]

            # Save the grayscale mask
            gray_path = os.path.join(output_dir, f"{base}_mask.png")
            cv2.imwrite(gray_path, gray_mask)

            # Save the colored mask
            colored_path = os.path.join(output_dir, f"{base}_color_mask.png")
            cv2.imwrite(colored_path, color_mask)

            # Record unique labels for Excel
            unique_labels = sorted(set(labels))
            all_results.append({
                "image_file": filename,
                "labels": ", ".join(unique_labels)
            })

        # Write out Excel summary
        df = pd.DataFrame(all_results, columns=["image_file", "labels"])
        excel_path = os.path.join(output_dir, "mask_labels.xlsx")
        df.to_excel(excel_path, index=False)

        print(f"Saved gray masks, color masks, and wrote summary Excel to {output_dir}")
