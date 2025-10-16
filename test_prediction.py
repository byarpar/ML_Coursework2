"""
Test Prediction Script

This script creates sample test images from the CIFAR-10 dataset and tests the prediction functionality.
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image

print("\n" + "=" * 60)
print("TEST PREDICTION SCRIPT")
print("=" * 60)

# Check if model exists
if not os.path.exists('results/best_model.pth'):
    print("\n❌ Error: Model not found!")
    print("\nYou need to train the model first:")
    print("  python3 quick_start.py")
    print("\nOr run full training:")
    print("  python3 main.py")
    sys.exit(1)

print("\n✓ Model found!")

# Create sample images from CIFAR-10 test dataset
print("\nCreating sample test images from CIFAR-10 dataset...")

os.makedirs('test_images', exist_ok=True)

# Load CIFAR-10 test batch
test_batch_path = 'data/cifar-10-batches-py/test_batch'

if not os.path.exists(test_batch_path):
    print("\n❌ Error: CIFAR-10 dataset not found!")
    print("\nPlease run training first to download the dataset:")
    print("  python3 quick_start.py")
    sys.exit(1)

# Load test data
with open(test_batch_path, 'rb') as f:
    test_dict = pickle.load(f, encoding='bytes')
    test_data = test_dict[b'data']
    test_labels = test_dict[b'labels']

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Extract sample images for specific classes
samples_to_extract = {
    'airplane': 0,
    'cat': 3,
    'dog': 5,
    'ship': 8,
    'automobile': 1
}

print("\nExtracting sample images:")
for class_name, class_idx in samples_to_extract.items():
    # Find first occurrence of this class in test data
    for i, label in enumerate(test_labels):
        if label == class_idx:
            # Reshape image from flat array to 32x32x3
            img_data = test_data[i]
            img_data = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
            
            # Create PIL Image and save
            img = Image.fromarray(img_data)
            filepath = f'test_images/{class_name}.jpg'
            img.save(filepath)
            print(f"  ✓ {class_name}.jpg saved")
            break

print("\n" + "=" * 60)
print("Test images created in 'test_images/' folder")
print("=" * 60)

print("\nNow you can test predictions:")
print("  python3 predict.py test_images/airplane.jpg")
print("  python3 predict.py test_images/cat.jpg")
print("  python3 predict.py test_images/dog.jpg")
print("  python3 predict.py test_images/ship.jpg")
print("  python3 predict.py test_images/automobile.jpg")

print("\nOr run interactive mode:")
print("  python3 predict.py")

print("\n" + "=" * 60)

# Optionally run prediction on first image
if os.path.exists('test_images/airplane.jpg'):
    print("\n🚀 Running sample prediction on airplane.jpg...")
    print("=" * 60 + "\n")
    os.system('python3 predict.py test_images/airplane.jpg')
