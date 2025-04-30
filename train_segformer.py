# train_segformer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from dataset import CustomDataset

os.makedirs('models', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = CustomDataset('data/train/Images', 'data/train/Masks', image_size=(256, 256))
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

segformer_model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-ade-512-512',
    num_labels=1,
    ignore_mismatched_sizes=True
)
segformer_model = segformer_model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(segformer_model.parameters(), lr=1e-5)

num_epochs = 5
best_loss = float('inf')

for epoch in range(num_epochs):
    segformer_model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = segformer_model(pixel_values=images).logits
        outputs = torch.nn.functional.interpolate(
            outputs, size=masks.shape[2:], mode="bilinear", align_corners=False
        )
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], SegFormer Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(segformer_model.state_dict(), 'models/segformer_best_model.pth')

print("✅ SegFormer Training Complete.")
