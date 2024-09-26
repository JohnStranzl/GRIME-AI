import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

def create_masks(coco_annotation_file, image_dir, output_dir):
    # Load COCO annotations
    coco = COCO(coco_annotation_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image ids
    img_ids = coco.getImgIds()
    
    for img_id in img_ids:
        # Load image info
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        
        # Load image
        image = cv2.imread(img_path)
        height, width, _ = image.shape
        
        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get annotation ids for the image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            # Get binary mask for the annotation
            rle = coco.annToRLE(ann)
            binary_mask = maskUtils.decode(rle)
            
            # Combine binary mask with the main mask
            mask = np.maximum(mask, binary_mask * 255)
        
        # Save the mask
        mask_path = os.path.join(output_dir, f"{img_info['file_name'].split('.')[0]}_mask.png")
        cv2.imwrite(mask_path, mask)

if __name__ == "__main__":
    coco_annotation_file = r'C:\Users\johns\Documents\000 - UNL\USGS\DOI\Student_Labeling_Images\NE_Platte_River_near_Grand_Island\Chris - instances_default.json'
    image_dir = r'C:\Users\johns\Documents\000 - UNL\USGS\DOI\Student_Labeling_Images\NE_Platte_River_near_Grand_Island\All'
    output_dir = r'C:\Users\johns\Documents\000 - UNL\USGS\DOI\Student_Labeling_Images\NE_Platte_River_near_Grand_Island\All\Chris - masks'
    
    create_masks(coco_annotation_file, image_dir, output_dir)
