import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Load DeepLabV3+ model
deeplab_model = models.segmentation.deeplabv3_resnet101(pretrained=False)
deeplab_model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))  # for binary class
deeplab_model.load_state_dict(torch.load('models/deeplab_best_model.pth'))
deeplab_model = deeplab_model.to(device)
deeplab_model.eval()

# -------------------------
# Load SegFormer model
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

segformer_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=1,
    ignore_mismatched_sizes=True
)
segformer_model.load_state_dict(torch.load('models/segformer_best_model.pth'))
segformer_model = segformer_model.to(device)
segformer_model.eval()

feature_extractor = SegformerFeatureExtractor(do_resize=True, size=256, do_normalize=True)

# -------------------------
# Inference function for DeepLabV3+
def inference_deeplab(image):
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    output = deeplab_model(img_tensor)['out']
    pred = torch.sigmoid(output) > 0.5
    return pred.squeeze().cpu().numpy()

# -------------------------
# Inference function for SegFormer
def inference_segformer(image):
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segformer_model(**inputs)
        logits = outputs.logits  # (1, 1, H, W)
        pred = torch.sigmoid(logits) > 0.5
    return pred.squeeze().cpu().numpy()

# -------------------------
# Run inference on both models
image_path = 'data/test_images/test_image.png'
image = Image.open(image_path).convert("RGB")

deeplab_mask = inference_deeplab(image)
segformer_mask = inference_segformer(image)

# -------------------------
# Visualize both masks
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title("Input Image")

axes[1].imshow(deeplab_mask, cmap='gray')
axes[1].set_title("DeepLabV3+ Prediction")

axes[2].imshow(segformer_mask, cmap='gray')
axes[2].set_title("SegFormer Prediction")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
