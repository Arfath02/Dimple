import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from transformers import SegformerForSemanticSegmentation
from dataset import CustomDataset

# Create 'models' folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
train_dataset = CustomDataset('data/train/Images', 'data/train/Masks', image_size=(256, 256))
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# DeepLabV3+ Model
deeplab_model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
deeplab_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # Binary output
deeplab_model = deeplab_model.to(device)

# SegFormer Model (semantic segmentation)
segformer_model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-ade-512-512',
    num_labels=1,
    ignore_mismatched_sizes=True
)
segformer_model = segformer_model.to(device)

# Loss and Optimizers
criterion = nn.BCEWithLogitsLoss()
optimizer_deeplab = optim.Adam(deeplab_model.parameters(), lr=1e-5)
optimizer_segformer = optim.Adam(segformer_model.parameters(), lr=1e-5)

# Training
num_epochs = 5
best_loss_deeplab = float('inf')
best_loss_segformer = float('inf')

for epoch in range(num_epochs):
    deeplab_model.train()
    segformer_model.train()
    running_loss_deeplab = 0.0
    running_loss_segformer = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Train DeepLabV3+
        optimizer_deeplab.zero_grad()
        deeplab_output = deeplab_model(images)['out']
        loss_deeplab = criterion(deeplab_output, masks)
        loss_deeplab.backward()
        optimizer_deeplab.step()
        running_loss_deeplab += loss_deeplab.item()

        # Train SegFormer
        optimizer_segformer.zero_grad()
        segformer_output = segformer_model(pixel_values=images).logits
        segformer_output = torch.nn.functional.interpolate(
            segformer_output,
            size=masks.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        loss_segformer = criterion(segformer_output, masks)
        loss_segformer.backward()
        optimizer_segformer.step()
        running_loss_segformer += loss_segformer.item()

    avg_loss_deeplab = running_loss_deeplab / len(train_loader)
    avg_loss_segformer = running_loss_segformer / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  DeepLabV3+ Loss: {avg_loss_deeplab:.4f}")
    print(f"  SegFormer   Loss: {avg_loss_segformer:.4f}")

    # Save best models
    if avg_loss_deeplab < best_loss_deeplab:
        best_loss_deeplab = avg_loss_deeplab
        torch.save(deeplab_model.state_dict(), 'models/deeplab_best_model.pth')

    if avg_loss_segformer < best_loss_segformer:
        best_loss_segformer = avg_loss_segformer
        torch.save(segformer_model.state_dict(), 'models/segformer_best_model.pth')

print("✅ Training Complete.")
