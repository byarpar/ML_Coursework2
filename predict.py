"""
Custom Image Prediction Script

This script allows users to test the trained model on their own images.
It loads the trained model and makes predictions on custom images.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


class ImageClassificationNN(nn.Module):
    """Same model architecture as used in training"""
    
    def __init__(self, num_classes=10):
        super(ImageClassificationNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


def load_model(model_path='results/best_model.pth', device='cpu'):
    """
    Load the trained model from file.
    
    Args:
        model_path: Path to the saved model weights
        device: Device to load the model on
    
    Returns:
        Loaded model in evaluation mode
    """
    print(f"\nLoading model from '{model_path}'...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please train the model first by running 'python main.py'"
        )
    
    model = ImageClassificationNN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!")
    return model


def preprocess_image(image_path):
    """
    Load and preprocess an image for prediction.
    
    Args:
        image_path: Path to the input image
    
    Returns:
        Preprocessed image tensor and original image
    """
    print(f"\nLoading image from '{image_path}'...")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at '{image_path}'")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    # Define preprocessing transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to CIFAR-10 size
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616))
    ])
    
    # Preprocess
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    print("✓ Image loaded and preprocessed successfully!")
    return image_tensor, original_image


def predict_image(model, image_tensor, device='cpu'):
    """
    Make prediction on preprocessed image.
    
    Args:
        model: Trained neural network model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
    
    Returns:
        Predicted class index and probabilities for all classes
    """
    print("\nMaking prediction...")
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
    
    return predicted_idx, probabilities.cpu().numpy()[0]


def visualize_prediction(original_image, predicted_class, probabilities, 
                        classes, save_path='results/custom_prediction.png'):
    """
    Visualize the prediction results.
    
    Args:
        original_image: Original input image
        predicted_class: Predicted class name
        probabilities: Probabilities for all classes
        classes: List of class names
        save_path: Path to save the visualization
    """
    print("\nGenerating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Display original image
    ax1.imshow(original_image)
    ax1.set_title(f'Input Image\nPredicted: {predicted_class}', 
                 fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display probability distribution
    colors = ['green' if classes[i] == predicted_class else 'lightblue' 
             for i in range(len(classes))]
    
    bars = ax2.barh(range(len(classes)), probabilities * 100, color=colors)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(classes, fontsize=10)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to '{save_path}'")
    plt.show()


def print_prediction_details(predicted_class, probabilities, classes):
    """
    Print detailed prediction information.
    
    Args:
        predicted_class: Predicted class name
        probabilities: Probabilities for all classes
        classes: List of class names
    """
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Predicted Class: {predicted_class.upper()}")
    print(f"   Confidence: {probabilities[classes.index(predicted_class)] * 100:.2f}%")
    
    print(f"\n📊 Top 3 Predictions:")
    print("-" * 60)
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {classes[idx]:<12} - {probabilities[idx] * 100:>6.2f}%")
    
    print("\n📋 All Class Probabilities:")
    print("-" * 60)
    for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
        bar = '█' * int(prob * 50)
        print(f"  {class_name:<12} {prob * 100:>6.2f}% {bar}")
    
    print("=" * 60 + "\n")


def interactive_mode():
    """
    Interactive mode to continuously predict images.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION MODE")
    print("=" * 60)
    print("\nEnter image paths to classify them.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model once
    try:
        model = load_model(device=device)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    while True:
        # Get image path from user
        image_path = input("\nEnter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode. Goodbye!")
            break
        
        if not image_path:
            print("⚠️  Please enter a valid image path.")
            continue
        
        try:
            # Preprocess image
            image_tensor, original_image = preprocess_image(image_path)
            
            # Make prediction
            predicted_idx, probabilities = predict_image(model, image_tensor, device)
            predicted_class = classes[predicted_idx]
            
            # Display results
            print_prediction_details(predicted_class, probabilities, classes)
            
            # Visualize
            visualize_prediction(original_image, predicted_class, 
                               probabilities, classes)
            
        except Exception as e:
            print(f"\n❌ Error processing image: {str(e)}")
            print("Please try another image.\n")


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("IMAGE CLASSIFICATION - CUSTOM PREDICTION")
    print("=" * 60)
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Check if image path provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        try:
            # Load model
            model = load_model(device=device)
            
            # Preprocess image
            image_tensor, original_image = preprocess_image(image_path)
            
            # Make prediction
            predicted_idx, probabilities = predict_image(model, image_tensor, device)
            predicted_class = classes[predicted_idx]
            
            # Display results
            print_prediction_details(predicted_class, probabilities, classes)
            
            # Visualize
            visualize_prediction(original_image, predicted_class, 
                               probabilities, classes)
            
            print("\n✅ Prediction completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            sys.exit(1)
    else:
        # No argument provided, enter interactive mode
        print("\nNo image path provided. Entering interactive mode...")
        interactive_mode()


if __name__ == "__main__":
    main()
