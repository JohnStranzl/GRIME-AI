import os
import random
import json
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask
from torch.nn.functional import binary_cross_entropy_with_logits
from torch import nn
import cv2
import numpy as np
from datetime import datetime


from torch import nn
from torch.cuda.amp import GradScaler, autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

now=datetime.now()
formatted_time=now.strftime('%d%m_%H%M')

def load_images_and_annotations(folders, annotation_files):
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
    


np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])

    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)  # Ensure mask is uint8
    mask_image = mask.reshape(h, w, 1) * color[:3].reshape(1, 1, -1)  # Remove alpha for OpenCV

    if borders:
        # Convert mask to binary (0, 255) for contour detection
        mask_binary = (mask * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]

        # Convert mask_image to OpenCV-compatible format
        mask_image_cv = (mask_image * 255).astype(np.uint8)

        # Ensure 3-channel RGB image for OpenCV
        if mask_image_cv.shape[-1] == 3:
            mask_image_cv = cv2.cvtColor(mask_image_cv, cv2.COLOR_RGB2BGR)

        # Draw contours
        cv2.drawContours(mask_image_cv, contours, -1, (255, 255, 255), thickness=2)  # White border

        # Convert back to matplotlib format
        mask_image = mask_image_cv.astype(np.float32) / 255.0

    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def load_true_mask(image_file, dataset):

    # Find the corresponding annotation set
    annotation_data = None
    for folder, data in dataset.items():
        if image_file in data["images"]:
            annotation_data = data["annotations"]
            break
    
    if annotation_data is None:
        raise ValueError(f"Image file {image_file} not found in the dataset.")

    # Extract the image filename
    file_name_only = os.path.basename(image_file)

    # Find image metadata
    image_info = next((img for img in annotation_data['images'] if img['file_name'] == file_name_only), None)
    if image_info is None:
        raise ValueError(f"Image file {image_file} not found in annotations.")

    image_id, height, width = image_info['id'], image_info['height'], image_info['width']

    # Get all annotations for the image
    annotations_for_image = [ann for ann in annotation_data['annotations'] if ann['image_id'] == image_id]
    if not annotations_for_image:
        return np.zeros((height, width), dtype=np.uint8)  # Return an empty mask if no annotations exist

    # Initialize an empty mask
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each annotation and decode RLE mask
    for ann in annotations_for_image:
        rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
        mask = coco_mask.decode(rle)

        # Handle multiple segmentation parts
        if len(mask.shape) == 3:
            mask = np.any(mask, axis=2)  # Merge multiple masks into one

        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)

    return combined_mask.astype(np.float32)  # Convert back to float for visualization


# change the folders based on site
folders = [ 
           "MD_Lake_Serene_Edgewood/COCO Products/CT_MD_Lake_Serene_Edgewood_COCO/images/default", 
           "MD_Lake_Serene_Edgewood/COCO Products/DK_MD_Lake_Serene_Edgewood_COCO/images/default"

    
           ]
annotation_files = [
    "MD_Lake_Serene_Edgewood/COCO Products/CT_MD_Lake_Serene_Edgewood_COCO/annotations/instances_default.json",
    "MD_Lake_Serene_Edgewood/COCO Products/DK_MD_Lake_Serene_Edgewood_COCO/annotations/instances_default.json"
    
]


#Change the path of the image
image_path="MD_Lake_Serene_Edgewood/COCO Products/DK_MD_Lake_Serene_Edgewood_COCO/images/default/MD_Lake_Serene_at_Edgewood___2024-01-12T14-15-02Z.jpg" 

dataset = load_images_and_annotations(folders, annotation_files)

image = Image.open(image_path)
image_np = np.array(image)

# Load corresponding mask
true_mask = load_true_mask(image_path, dataset)


if true_mask is not None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)  # Show the image

    # Show the mask using your function
    show_mask(true_mask, ax, borders=True)

    plt.axis("off")
    plt.show()
else:
    print(f"No mask found for {image_path}")
