#pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
import os
#from transformers import SamModel, SamFeatureExtractor
from transformers import SamModel, SamProcessor
import numpy as np
from PIL import Image


# Custom dataset class for COCO
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # For training, you'd need to create masks from annotations
        # Here we just return the annotations
        return img, coco_annotation

    def __len__(self):
        return len(self.ids)

def trainSAM():
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor(),])

    # Load your dataset
    train_dataset = CocoDataset(
        image_dir='C:\\Users\\johns\\Documents\\UNL\\USGS\\DOI\\Student_Labeling_Images\\NM_Pecos_River_near_Acme\\Originals',
        annotation_file='instances_default.json',
        transform=transform
    )

    # ==============================================================================================================
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # ==============================================================================================================
    # Initialize SAM model and feature extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    # ==============================================================================================================
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 1
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, annotations in train_loader:
            # Prepare images and annotations
            inputs = processor(images=images, return_tensors="pt")

            # convert annotations to  target format
            #targets = convert_coco_annotations(annotations)
            annFile = 'instances_default.json'
            dataType = ''
            dataDir = 'C:\\Users\\johns\\Documents\\UNL\\USGS\\DOI\\Student_Labeling_Images\\NM_Pecos_River_near_Acme\\Originals'
            targets = convert_coco_annotations(dataDir, dataType, annFile)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            loss = loss_fn(outputs, targets)

            # BACKWARD PASS AND OPTIMIZATION
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # ==============================================================================================================
    # Save the trained model
    torch.save(model.state_dict(), 'sam_model.pth')

# **************************************************************************************************************
#
# **************************************************************************************************************
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
        #img = Image.open(f'{dataDir}/{dataType}/{img_info["file_name"]}')
        img = Image.open(f'{dataDir}/{img_info["file_name"]}')
        img_array = np.array(img)

        # Create a mask for each annotation
        mask = np.zeros((img_info['height'], img_info['width']))
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann) * ann['category_id'])

        masks[img_info['id']] = mask

    return masks


# **************************************************************************************************************
#
# **************************************************************************************************************
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

def coco_to_masks(coco_file):
    # INITIALIZE COCO API FOR INSTANCE ANNOTATIONS
    coco = COCO(coco_file)

    # GET IDs OF ALL IMAGES IN THE COCO FILE
    img_ids = coco.getImgIds()

    masks = []
    for img_id in img_ids:
        # GET ANNOTATION IDs FOR THE CURRENT IMAGE
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # GET IMAGE INFORMATION
        img_info = coco.loadImgs(img_id)[0]
        img_height = img_info['height']
        img_width = img_info['width']

        # CREATE AN EMPTY MASK
        mask = np.zeros((img_height, img_width))

        for ann in anns:
            # FETCH THE MASK FOR THE CURRENT ANNOTATION
            rle = maskUtils.frPyObjects(ann['segmentation'], img_height, img_width)
            m = maskUtils.decode(rle)
            m = m.reshape(img_height, img_width) * 255

            # ADD TO THE LIST OF MASKS
            mask = np.maximum(mask, m)

        masks.append(mask)

    return masks


# Example usage:
# dataDir = 'path_to_COCO_dataset'
# dataType = 'train2014'
# annFile = f'{dataDir}/annotations/instances_{dataType}.json'
# masks = convert_coco_annotations(dataDir, dataType, annFile)

# ======================================================================================================================
#
# ======================================================================================================================
if __name__ == '__main__':
    annFile = 'instances_default.json'
    dataType = ''
    dataDir = 'C:\\Users\\johns\\Documents\\UNL\\USGS\\DOI\\Student_Labeling_Images\\NM_Pecos_River_near_Acme\\Originals'
    #convert_coco_annotations(dataDir, dataType, annFile)
    #trainSAM()

    masks = coco_to_masks(annFile)

    # TEST POINT
    # from matplotlib import pyplot as plt
    # plt.imshow(masks[0], interpolation='nearest')
    # plt.show()

    print("Finished!")


