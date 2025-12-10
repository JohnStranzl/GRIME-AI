#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
COLOR_MAP = {
    0: np.array([1.0, 0.0, 0.0, 0.6]),
    1: np.array([0.0, 1.0, 0.0, 0.6]),
    2: np.array([0.0, 0.0, 1.0, 0.6]),
    3: np.array([1.0, 1.0, 0.0, 0.6]),
    4: np.array([1.0, 0.0, 1.0, 0.6]),
    5: np.array([0.0, 1.0, 1.0, 0.6]),
    6: np.array([0.5, 0.0, 0.0, 0.6]),
    7: np.array([0.0, 0.5, 0.0, 0.6]),
    8: np.array([0.0, 0.0, 0.5, 0.6]),
    9: np.array([0.5, 0.5, 0.0, 0.6]),
    10: np.array([0.5, 0.0, 0.5, 0.6]),
    11: np.array([0.0, 0.5, 0.5, 0.6]),
    12: np.array([1.0, 0.5, 0.0, 0.6]),
    13: np.array([0.5, 1.0, 0.0, 0.6]),
    14: np.array([0.0, 1.0, 0.5, 0.6]),
    15: np.array([0.0, 0.5, 1.0, 0.6]),
    16: np.array([0.5, 0.0, 1.0, 0.6]),
    17: np.array([1.0, 0.0, 0.5, 0.6]),
    18: np.array([0.25, 0.25, 0.25, 0.6]),
    19: np.array([0.7, 0.7, 0.7, 0.6]),
    20: np.array([1.0, 0.85, 0.8, 0.6]),
    21: np.array([0.3, 0.3, 1.0, 0.6]),
    22: np.array([1.0, 0.9, 0.2, 0.6]),
    23: np.array([0.2, 0.8, 0.2, 0.6]),
    24: np.array([0.2, 0.4, 0.8, 0.6]),
    25: np.array([0.8, 0.2, 0.4, 0.6]),
    26: np.array([0.4, 0.8, 0.8, 0.6]),
    27: np.array([0.6, 0.3, 0.0, 0.6]),
    28: np.array([0.9, 0.6, 0.4, 0.6]),
    29: np.array([0.3, 0.6, 0.1, 0.6]),
    30: np.array([0.6, 0.1, 0.6, 0.6]),
    31: np.array([0.3, 0.7, 0.6, 0.6]),
}


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def get_color_for_category(category_id):
    """Return RGBA color for a given category id."""
    return COLOR_MAP.get(category_id, np.array([0.5, 0.5, 0.5, 0.6]))


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def init_coco_structure(selected_label_categories):
    """Initialize COCO JSON structure."""
    return {
        "images": [],
        "annotations": [],
        "categories": selected_label_categories or [{"id": 2, "name": "Vegetation"}],
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "",
                 "description": "", "url": "",
                 "version": "", "year": ""}
    }


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def mask_to_polygon(mask, min_contour_area=50):
    """Convert binary mask to polygon segmentation."""
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


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id):
    """Add image and annotation entries to COCO JSON."""
    height, width = image_array.shape[:2]
    coco_data["images"].append({
        "file_name": os.path.basename(image_path),
        "height": height,
        "width": width,
        "id": image_id,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0
    })

    segmentation = mask_to_polygon(mask)
    if not segmentation:
        return

    pos = np.where(mask)
    xmin, xmax = int(np.min(pos[1])), int(np.max(pos[1]))
    ymin, ymax = int(np.min(pos[0])), int(np.max(pos[0]))
    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

    coco_data["annotations"].append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 2,
        "segmentation": segmentation,
        "area": int(np.sum(mask.astype(np.uint8))),
        "bbox": bbox,
        "iscrowd": 0
    })


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def save_coco_json(coco_data, output_dir):
    """Save COCO annotations to predictions.json."""
    output_file = Path(output_dir) / "predictions.json"
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=4)
    print(f"COCO annotations saved to {output_file}")
