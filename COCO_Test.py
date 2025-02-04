import sys

try:
    import torch

    import torchvision
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import CocoDetection
    print("COCO Test: PyTorch imported successfully.")
except ImportError as e:
    print("COCO Test: Error importing PyTorch:", e)
    # Remove the faulty package from sys.modules to prevent further issues
    if 'torch' in sys.modules:
        del sys.modules['torch']


# Define transformations for data augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load COCO dataset
train_dataset = CocoDetection(root='path/to/coco/train', annFile='annotations/train.json', transform=transform)
val_dataset = CocoDetection(root='path/to/coco/val', annFile='annotations/val.json', transform=transform)

# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

# Define the model
model = MaskRCNN(num_classes=91)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Move model to the device
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # Update the learning rate
    lr_scheduler.step()

    # Evaluation on the validation dataset
    model.eval()
    for images, targets in val_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            val_loss_dict = model(images, targets)
    
    # Print training and validation losses
    print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {losses.item()}, Validation Loss: {sum(val_loss_dict.values()).item()}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')