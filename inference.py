import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Inference function for both models
def inference_deeplab(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    image = transform(image).unsqueeze(0).to(device)

    output = deeplab_model(image)['out']
    pred = torch.sigmoid(output) > 0.5  # Binary mask
    return pred.squeeze().cpu().numpy()

def inference_segformer(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    image = transform(image).unsqueeze(0).to(device)

    output = segformer_model(image)
    pred = torch.sigmoid(output) > 0.5  # Binary mask
    return pred.squeeze().cpu().numpy()

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DeepLabV3+ Model
deeplab_model = models.segmentation.deeplabv3_resnet101(pretrained=False)
deeplab_model.load_state_dict(torch.load('models/deeplab_best_model.pth'))
deeplab_model = deeplab_model.to(device)
deeplab_model.eval()

# SegFormer Model (replace with your actual SegFormer model)
segformer_model = SegFormerModel()  # Define your SegFormer model here
segformer_model.load_state_dict(torch.load('models/segformer_best_model.pth'))
segformer_model = segformer_model.to(device)
segformer_model.eval()

# Choose which model to use
image_path = 'data/test_images/test_image.png'
model_choice = 'deeplab'  # Change to 'segformer' for SegFormer inference

if model_choice == 'deeplab':
    predicted_mask = inference_deeplab(image_path)
elif model_choice == 'segformer':
    predicted_mask = inference_segformer(image_path)

# Visualize the prediction
plt.imshow(predicted_mask, cmap='gray')
plt.title(f'Predicted Mask using {model_choice}')
plt.show()
