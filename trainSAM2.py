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

def trainSAM2():
    sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters())
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        image_embedding = sam_model.image_encoder(input_image)

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

    low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)

    from torch.nn.functional import threshold, normalize

    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)

    loss = loss_fn(binary_mask, gt_binary_mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), PATH)


def trainSAM3():
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.datasets import CocoDetection
    import torchvision.transforms as T

    # Define the model
    class SegmentAnythingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=91)

        def forward(self, images, targets=None):
            return self.model(images, targets)

    # Define the dataset
    class CocoSegmentationDataset(CocoDetection):
        def __init__(self, root, annFile, transform=None):
            super().__init__(root, annFile)
            self.transform = transform

        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            image_id = self.ids[index]
            target = {"image_id": image_id, "annotations": target}
            if self.transform is not None:
                img, target = self.transform(img, target)
            return img, target

    # Define the transforms
    def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    # Initialize the dataset
    dataset = CocoSegmentationDataset("path/to/your/images", "path/to/your/coco.json", get_transform(train=True))

    # Initialize the dataloader
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    # Initialize the model
    model = SegmentAnythingModel()

    # Move model to the right device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # And a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # Train for one epoch, printing every 10 iterations
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        # Update the learning rate
        lr_scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), "path/to/save/model.pt")

# ======================================================================================================================
#
# ======================================================================================================================
'''
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.datasets import CocoDetection
import utils
from engine import train_one_epoch, evaluate

def get_model_instance_segmentation(num_classes):
    # Load a pre-trained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask classifier with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def trainSAM4():
    # Define the classes of the COCO dataset
    num_classes = 91  # 90 COCO classes + background

    # Load the dataset
    dataset = CocoDetection('path/to/coco/train', 'path/to/coco/annotations')
    dataset_test = CocoDetection('path/to/coco/val', 'path/to/coco/annotations')

    # Define the data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                                              collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                                                   collate_fn=utils.collate_fn)

    # Get the model
    model = get_model_instance_segmentation(num_classes)

    # Move model to the right device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train for 10 epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        evaluate(model, data_loader_test, device=device)

    # Save the trained model
    torch.save(model.state_dict(), 'segmentation_model.pth')
'''

# ======================================================================================================================
#
# ======================================================================================================================
def trainSAM5(root, annFile):
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.datasets import CocoDetection
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # Load the COCO dataset
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
    ])

    #JES transform = transforms.Compose([transforms.ToTensor()])

    dataset = CocoDetection(root, annFile, transform=transform)

    # Define the dataloader
    #data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    data_loader = DataLoader(dataset, batch_size=1)

    # Load a pre-trained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    num_classes = 3
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask classifier with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    # Move model to the right device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            #targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            new_targets = []
            for t in targets:
                temp_dict = {}
                for k, v in t.items():
                    temp_dict[k] = v.to(device)
                new_targets.append(temp_dict)
            targets = new_targets

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), 'segment_anything_model_water.pth')


# ======================================================================================================================
#
# ======================================================================================================================
if __name__ == '__main__':
    annFile = 'instances_default.json'
    dataType = ''
    annFile = 'C:\\Users\\johns\\Documents\\UNL\\USGS\\DOI\\Student_Labeling_Images\\NM_Pecos_River_near_Acme\\Originals\\instances_default.json'
    dataDir = 'C:\\Users\\johns\\Documents\\UNL\\USGS\\DOI\\Student_Labeling_Images\\NM_Pecos_River_near_Acme\\Originals'

    #annFile = 'C:\\Users\\johns\\Documents\\UNL\\USGS\\DOI\\Student_Labeling_Images\\NM_Pecos_River_near_Acme\\Originals'
    #dataDir = 'C:\\Users\\johns\\Documents\\UNL\\USGS\\DOI\\Student_Labeling_Images\\NM_Pecos_River_near_Acme\\Originals'

    #convert_coco_annotations(dataDir, dataType, annFile)
    #trainSAM()

    if 0:
        masks = coco_to_masks(annFile)

    if 1:
        trainSAM5(dataDir, annFile)

    # TEST POINT
    # from matplotlib import pyplot as plt
    # plt.imshow(masks[0], interpolation='nearest')
    # plt.show()

    print("Finished!")


