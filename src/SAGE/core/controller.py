# sam2_gui/core/controller.py
import itertools
import numpy as np
from utils.mask_ops import compute_mask_stats
from utils.colors import get_color_for_index


class SegmentationController:
    def __init__(self, model_manager, image_np: np.ndarray):
        self.model_manager = model_manager
        self.image_np = image_np
        self.model_manager.initialize_backend(image_np)

        self.fg_points = []  # list of (x, y)
        self.bg_points = []  # list of (x, y)

        self.masks = []  # list of dicts: id, label, mask, color, visible, stats
        self._mask_id_counter = itertools.count(1)

        self.opacity = 120

    # ---- point management ----
    def add_point(self, x, y, is_fg=True):
        if is_fg:
            self.fg_points.append((x, y))
        else:
            self.bg_points.append((x, y))

    def clear_points(self):
        self.fg_points.clear()
        self.bg_points.clear()

    def remove_last_point(self):
        if self.fg_points:
            self.fg_points.pop()
        elif self.bg_points:
            self.bg_points.pop()

    # ---- segmentation ----
    def run_segmentation(self, label=None):
        if not self.fg_points and not self.bg_points:
            return None

        # >>> Exclude ONLY visible (checked) masks <<<
        exclude_mask = None
        for m in self.masks:
            if m["visible"]:  # <-- CHECKED = EXCLUDED
                if exclude_mask is None:
                    exclude_mask = m["mask"].copy()
                else:
                    exclude_mask |= m["mask"]

        mask = self.model_manager.segment_from_points(
            fg_points=self.fg_points,
            bg_points=self.bg_points,
            exclude_mask=exclude_mask,
        )
        if mask is None:
            return None

        mask_id = next(self._mask_id_counter)
        color = get_color_for_index(len(self.masks))
        stats = compute_mask_stats(mask)

        mask_entry = {
            "id": mask_id,
            "label": label or f"Region {mask_id}",
            "mask": mask,
            "color": color,
            "visible": True,
            "stats": stats,
        }

        self.masks.append(mask_entry)
        self.clear_points()
        return mask_entry

    def set_mask_label(self, mask_id, new_label):
        for m in self.masks:
            if m["id"] == mask_id:
                m["label"] = new_label
                break

    def set_mask_visibility(self, mask_id, visible):
        for m in self.masks:
            if m["id"] == mask_id:
                m["visible"] = visible
                break

    def set_opacity(self, value: int):
        self.opacity = int(value)

    def get_visible_masks(self):
        return [m for m in self.masks if m["visible"]]

    def save_coco(self, filepath):
        """
        Save all masks to a COCO 1.0 annotation file.
        Each mask becomes one annotation with polygon segmentation.
        """

        height, width = self.image_np.shape[:2]

        coco = {
            "images": [
                {
                    "id": 1,
                    "file_name": "image.png",
                    "width": width,
                    "height": height,
                }
            ],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "region"}
            ],
        }

        ann_id = 1

        for m in self.masks:
            mask = m["mask"].astype(np.uint8)

            # Extract polygon(s)
            import cv2
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            segmentation = []
            area = 0
            bbox = None

            for cnt in contours:
                if len(cnt) < 3:
                    continue

                poly = cnt.reshape(-1, 2).tolist()
                segmentation.append([coord for point in poly for coord in point])

                area += cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                if bbox is None:
                    bbox = [x, y, w, h]

            if not segmentation:
                continue

            annotation = {
                "id": ann_id,
                "image_id": 1,
                "category_id": 1,
                "segmentation": segmentation,
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0,
                "label": m["label"],
            }

            coco["annotations"].append(annotation)
            ann_id += 1

        import json
        with open(filepath, "w") as f:
            json.dump(coco, f, indent=2)

    def delete_mask(self, mask_id):
        """Remove a mask by its ID."""
        self.masks = [m for m in self.masks if m["id"] != mask_id]