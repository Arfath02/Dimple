import os
import random
import shutil

# Paths to your data
images_path = 'data/Images'
masks_path = 'data/Masks'

# Paths for training and testing
train_images_path = 'data/train/Images'
train_masks_path = 'data/train/Masks'
test_images_path = 'data/test/Images'
test_masks_path = 'data/test/Masks'

# Create directories if they don't exist
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_masks_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_masks_path, exist_ok=True)

# List all image files
images = os.listdir(images_path)

# Shuffle and split the data (80% for training, 20% for testing)
random.shuffle(images)
split_idx = int(0.8 * len(images))  # 80% for training
train_images = images[:split_idx]
test_images = images[split_idx:]

# Move the files to respective directories
for img in train_images:
    shutil.move(os.path.join(images_path, img), os.path.join(train_images_path, img))
    shutil.move(os.path.join(masks_path, img.replace('image', 'mask')), os.path.join(train_masks_path, img.replace('image', 'mask')))

for img in test_images:
    shutil.move(os.path.join(images_path, img), os.path.join(test_images_path, img))
    shutil.move(os.path.join(masks_path, img.replace('image', 'mask')), os.path.join(test_masks_path, img.replace('image', 'mask')))
