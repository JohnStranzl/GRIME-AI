# coco_utils.py

#!/usr/bin/env python3
"""
Author: John Edward Stranzl, Jr.
Date: 2025-08-05
Company: Blade Vision Systems, LLC
GitHub: https://github.com/JohnStranzl
LinkedIn: https://www.linkedin.com/in/johnestranzl/

COCO Utilities

A class-based tool to validate and clean a COCO 1.0 JSON annotation
file against the images present in a directory.
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple


class GRIME_AI_COCO_Utils:
    """Utility class for validating and cleaning a COCO-format JSON file."""

    def __init__(self, folder_path: str) -> None:
        """Initialize with path to a directory containing images and one JSON."""
        self.folder_path: str = folder_path
        self.json_filename: str = ""
        self.json_path: str = ""
        self.data: Dict[str, Any] = {}

    def find_json_file(self) -> None:
        """Locate exactly one .json file in the folder and set paths."""
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
            print("Please ensure exactly one JSON file in the directory.")
            sys.exit(1)

        self.json_filename = json_files[0]
        self.json_path = os.path.join(self.folder_path, self.json_filename)

    def load_json(self) -> None:
        """Load JSON data from disk and validate presence of 'images' key."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if "images" not in self.data:
            print("Invalid COCO JSON: missing 'images' section.")
            sys.exit(1)

    def check_images(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Check which images in JSON exist on disk.

        Returns:
            present_images: images found in folder
            missing_images: images not found in folder
        """
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
        """Rename the original JSON file by appending '.ORIGINAL'."""
        orig = os.path.join(self.folder_path, self.json_filename)
        backup = orig + ".ORIGINAL"
        os.rename(orig, backup)

    def clean_data(
        self, present_images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Construct a cleaned data dictionary.

        Keeps only present_images and filters annotations accordingly.
        """
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
        """Write new_data back out to the original JSON filename."""
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4)

    def process(self) -> None:
        """Execute full validation and cleaning pipeline."""
        self.find_json_file()
        self.load_json()

        present, missing = self.check_images()
        if not missing:
            print("All images in JSON are present.")
            return

        self.backup_original()
        cleaned = self.clean_data(present)
        self.write_json(cleaned)
        print(
            "New JSON created for the images available "
            "in the folder."
        )
