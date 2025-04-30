import torch
import matplotlib.pyplot as plt
from PIL import Image
from dataset import CustomDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from train import deeplab_model, segformer_model  # Import trained models

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms for images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load test data
test_dataset = CustomDataset('data/test/Images', 'data/test/Masks', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Function to evaluate model
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            predictions = torch.sigmoid(outputs) > 0.5
            correct += (predictions == masks).sum().item()
            total += masks.numel()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# Load the best models
deeplab_model.load_state_dict(torch.load('models/deeplab_best_model.pth'))
segformer_model.load_state_dict(torch.load('models/segformer_best_model.pth'))

# Evaluate both models
print("Evaluating DeepLabV3...")
evaluate(deeplab_model, test_loader, device)

print("Evaluating SegFormer...")
evaluate(segformer_model, test_loader, device)
