import cv2
import matplotlib.pyplot as plt

from segment_anything import SamPredictor
import torch

CHECKPOINT_PATH = 'sam_model.pth'
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

model = SamPredictor('sam_model.pth')
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================================================================================
#
# ====================================================================================================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

mask_predictor = SamPredictor(sam)




image = cv2.imread('C:\\Users\\johns\\pycharmprojects\\neonAI\\jupyterlab\\CT_Labels\\ne_kearney_outdoor_learning_area _ 5_16_24 - 5_28_24_dataset_2024_07_12_15_15_51_coco 1.0\\images\\default\\NE_Kearney_Outdoor_Learning_Area___2024-05-16T15-30-00Z.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = model.predict(image_rgb)

plt.imshow(image_rgb)
for mask in masks:
    plt.imshow(mask, alpha=0.5)
plt.show()
