"""
Simple Test - Use CIFAR-10 Test Images

This script uses images from the CIFAR-10 dataset that's already downloaded.
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import load_model, predict_image

print("\n" + "=" * 60)
print("SIMPLE PREDICTION TEST")
print("=" * 60)

# Check if model exists
if not os.path.exists('results/best_model.pth'):
    print("\n❌ Error: Model not found!")
    print("\nTrain the model first:")
    print("  python3 quick_start.py")
    sys.exit(1)

print("\n✓ Model file found!")

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading model on {device}...")
model = load_model(device=device)

# Load CIFAR-10 test dataset
print("\nLoading CIFAR-10 test images...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                       (0.2470, 0.2435, 0.2616))
])

test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Test on a few random images
print("\n" + "=" * 60)
print("Testing on 5 random CIFAR-10 test images:")
print("=" * 60)

num_tests = 5
correct = 0

for i in range(num_tests):
    # Get random test image
    idx = np.random.randint(len(test_dataset))
    image, true_label = test_dataset[idx]
    
    # Make prediction
    image_batch = image.unsqueeze(0).to(device)
    predicted_idx, probabilities = predict_image(model, image_batch, device)
    
    true_class = classes[true_label]
    pred_class = classes[predicted_idx]
    confidence = probabilities[predicted_idx] * 100
    
    is_correct = (predicted_idx == true_label)
    if is_correct:
        correct += 1
    
    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    
    print(f"\nTest {i+1}:")
    print(f"  True label:      {true_class}")
    print(f"  Predicted:       {pred_class} ({confidence:.1f}% confidence)")
    print(f"  Result:          {status}")

print("\n" + "=" * 60)
print(f"Accuracy: {correct}/{num_tests} = {correct/num_tests*100:.1f}%")
print("=" * 60)

print("\n✅ Test completed!")
print("\nTo test with your own images:")
print("  1. Run: python3 test_prediction.py (downloads sample images)")
print("  2. Or: python3 predict.py /path/to/your/image.jpg")
print("  3. Or: python3 predict.py (interactive mode)")
print("\n")
