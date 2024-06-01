from pycocotools.coco import COCO
import numpy as np
from PIL import Image

def convert_coco_annotations(dataDir, dataType, annFile):
    """
    Convert COCO annotations to mask images.

    Parameters:
    - dataDir: str, directory where the COCO images are stored.
    - dataType: str, type of the dataset (e.g., 'train2014', 'val2014').
    - annFile: str, path to the COCO annotation file.

    Returns:
    - masks: dict, a dictionary where keys are image ids and values are mask arrays.
    """
    # Initialize COCO api for instance annotations
    coco = COCO(annFile)
    
    # Load the categories and get all images
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds(catIds=catIds)
    
    masks = {}
    for imgId in imgIds:
        img_info = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        # Load the image
        img = Image.open(f'{dataDir}/{dataType}/{img_info["file_name"]}')
        img_array = np.array(img)
        
        # Create a mask for each annotation
        mask = np.zeros((img_info['height'], img_info['width']))
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann) * ann['category_id'])
        
        masks[img_info['id']] = mask
    
    return masks

# Example usage:
# dataDir = 'path_to_COCO_dataset'
# dataType = 'train2014'
# annFile = f'{dataDir}/annotations/instances_{dataType}.json'
# masks = convert_coco_annotations(dataDir, dataType, annFile)
