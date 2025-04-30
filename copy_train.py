# train_segformer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from dataset import CustomDataset

# Create model directory
os.makedirs('models', exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Load dataset
print("📁 Loading dataset...")
train_dataset = CustomDataset('data/train/Images', 'data/train/Masks', image_size=(256, 256))

# Setup DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
print(f"✅ Dataset loaded. Total samples: {len(train_dataset)}")

# Load SegFormer model
print("📦 Loading SegFormer model...")
segformer_model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-ade-512-512',
    num_labels=1,
    ignore_mismatched_sizes=True
)
segformer_model = segformer_model.to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(segformer_model.parameters(), lr=1e-5)

# Training settings
num_epochs = 5
best_loss = float('inf')

print("🚀 Starting training...")
for epoch in range(num_epochs):
    segformer_model.train()
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        # Ensure masks have correct shape
        if masks.ndim == 3:
            masks = masks.unsqueeze(1).float()
        else:
            masks = masks.float()

        optimizer.zero_grad()
        outputs = segformer_model(pixel_values=images).logits
        outputs = torch.nn.functional.interpolate(
            outputs, size=masks.shape[2:], mode="bilinear", align_corners=False
        )
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"\n📊 Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(segformer_model.state_dict(), 'models/segformer_best_model.pth')
        print(f"💾 Best model saved at epoch {epoch+1}.")

print("✅ SegFormer Training Complete.")
