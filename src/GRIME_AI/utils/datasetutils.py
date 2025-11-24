
import os
import random
from pycocotools import mask as coco_mask
import cv2
import numpy as np
import json


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
            # normalize here so every path in all_images is clean
            for img in data["images"]:
                if isinstance(img, dict) and "path" in img:
                    img = {**img, "path": os.path.normpath(img["path"])}
                else:
                    img = os.path.normpath(str(img))
                all_images.append(img)

        random.shuffle(all_images)

        num_images = len(all_images)
        train_size = int(train_split * num_images)

        train_images = all_images[:train_size]
        val_images = all_images[train_size:]
        print(f"Train: {len(train_images)} images, Validation: {len(val_images)} images")
        return train_images, val_images

    '''
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
    '''

    def save_split_dataset(self, train_images, val_images, output_file):
        """
        Save metadata-only split: no image copying.
        Writes train/val image paths to splits.json in out_dir.
        """
        import json, os
        from pathlib import Path

        out_dir = os.path.dirname(output_file)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def extract_paths(images):
            return [img['path'] if isinstance(img, dict) else str(img) for img in images]

        split_data = {
            "train": extract_paths(train_images),
            "val": extract_paths(val_images)
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)

        print(f"[split] Saved metadata-only split to {output_file}")


    def get_image_size(self, image_path: str) -> tuple[int, int]:
        """
        Returns (height, width) of the image.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return img.shape[:2]

    def get_annotations_for_image(
        self,
        image_path: str,
        annotation_index: dict
    ) -> list[dict]:
        """
        Returns the list of COCO‐style annotation dicts for this image.
        Assumes annotation_index maps image_path (or stem) to a list of anns.
        """
        # If your index keys are full paths, use image_path directly;
        # if they are stems/IDs you may need Path(image_path).stem
        return annotation_index.get(image_path, [])

    def rasterize_polygon(
        self,
        segmentation: list,
        image_shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Given a COCO‐style polygon (list of [x0,y0,x1,y1,...]), rasterize into
        a H×W binary mask.
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        # segmentation may be a list of lists
        if isinstance(segmentation, list):
            for poly in segmentation:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], color=1)
        else:
            # if it's RLE or other, extend here
            raise ValueError("Unsupported segmentation format")
        return mask

    def load_all_true_masks(
        self,
        image_path: str,
        annotation_index: dict
    ) -> dict[int, np.ndarray]:
        """
        Returns a dict mapping category_id -> H×W binary mask for every annotation
        in this image. Categories missing from the image (e.g. snow) simply won't
        appear in the dict.
        """
        # 1) get all annotations for this image
        anns = self.get_annotations_for_image(image_path, annotation_index)

        # 2) get image size
        h, w = self.get_image_size(image_path)

        all_masks: dict[int, np.ndarray] = {}
        for ann in anns:
            cid = ann.get("category_id")
            seg = ann.get("segmentation")
            if cid is None or seg is None:
                continue

            mask = self.rasterize_polygon(seg, (h, w))
            if mask.sum() > 0:
                all_masks[cid] = mask

        return all_masks