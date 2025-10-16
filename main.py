"""
Machine Learning Coursework 2: Image Classification with Neural Networks
Author: [Your Name]
Date: November 2024

This script implements a neural network for image classification using the CIFAR-10 dataset.
The dataset contains 60,000 32x32 color images in 10 classes.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
import pickle
from datetime import datetime


# ================== 1. DATA LOADING AND PREPROCESSING ==================

def load_and_prepare_data():
    """
    Load CIFAR-10 dataset and prepare it for training.
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        classes: List of class names
    """
    print("=" * 60)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("=" * 60)
    
    # Define transformations for the data
    # Normalize with CIFAR-10 mean and std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Download and load training data
    print("\nDownloading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
    
    # Download and load testing data
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2)
    
    # Class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"\nDataset Information:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Testing samples: {len(test_dataset)}")
    print(f"  Number of classes: {len(classes)}")
    print(f"  Classes: {classes}")
    print(f"  Image shape: 3x32x32 (RGB)")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, test_loader, classes


def visualize_sample_data(train_loader, classes, num_samples=12):
    """
    Visualize sample images from the dataset.
    
    Args:
        train_loader: DataLoader for training data
        classes: List of class names
        num_samples: Number of samples to display
    """
    print("\nVisualizing sample images...")
    
    # Get a batch of training data
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Denormalize images for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle('Sample Images from CIFAR-10 Dataset', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            # Denormalize and convert to numpy
            img = images[idx] * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(f'Class: {classes[labels[idx]]}', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
    print("  ✓ Sample images saved to 'results/sample_images.png'")
    plt.close()


# ================== 2. NEURAL NETWORK DESIGN ==================

class ImageClassificationNN(nn.Module):
    """
    Convolutional Neural Network for Image Classification
    
    Architecture:
    - Input Layer: 3x32x32 (RGB image)
    - Conv Layer 1: 32 filters, 3x3 kernel, ReLU activation
    - Conv Layer 2: 64 filters, 3x3 kernel, ReLU activation
    - Max Pooling: 2x2
    - Conv Layer 3: 128 filters, 3x3 kernel, ReLU activation
    - Max Pooling: 2x2
    - Flatten
    - Fully Connected Layer 1: 512 neurons, ReLU, Dropout(0.5)
    - Fully Connected Layer 2: 256 neurons, ReLU, Dropout(0.3)
    - Output Layer: 10 neurons (10 classes)
    """
    
    def __init__(self, num_classes=10):
        super(ImageClassificationNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        )
        
        # Fully connected layers
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
        """Forward pass through the network"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def print_model_architecture(model):
    """
    Print detailed information about the model architecture.
    
    Args:
        model: Neural network model
    """
    print("\n" + "=" * 60)
    print("STEP 2: NEURAL NETWORK ARCHITECTURE")
    print("=" * 60)
    
    print("\nModel Architecture:")
    print(model)
    
    print(f"\nTotal Trainable Parameters: {model.count_parameters():,}")
    
    # Calculate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    print(f"Model Size: {size_mb:.2f} MB")


# ================== 3. TRAINING PROCESS ==================

def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU or GPU)
    
    Returns:
        Average training loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {running_loss / (batch_idx + 1):.4f} '
                  f'Acc: {100. * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device (CPU or GPU)
    
    Returns:
        Average test loss and accuracy
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def train_and_evaluate(model, train_loader, test_loader, num_epochs=30, 
                      learning_rate=0.001, device='cpu'):
    """
    Complete training and evaluation process.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device (CPU or GPU)
    
    Returns:
        Training history (losses and accuracies)
    """
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING THE NEURAL NETWORK")
    print("=" * 60)
    
    print(f"\nTraining Configuration:")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: Adam")
    print(f"  Loss function: CrossEntropyLoss")
    print(f"  Device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=3)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_test_acc = 0.0
    
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        
        # Train
        train_loss, train_acc = train_model(model, train_loader, criterion, 
                                           optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'results/best_model.pth')
            print(f'  ✓ New best model saved! (Accuracy: {best_test_acc:.2f}%)')
    
    print("\n" + "-" * 60)
    print(f"Training completed!")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    
    return history, model


# ================== 4. EVALUATION AND VISUALIZATION ==================

def plot_training_history(history):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary containing training history
    """
    print("\nGenerating training history plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-s', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-s', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
    print("  ✓ Training history saved to 'results/training_history.png'")
    plt.close()


def generate_confusion_matrix(model, test_loader, classes, device):
    """
    Generate and plot confusion matrix.
    
    Args:
        model: Trained neural network model
        test_loader: DataLoader for test data
        classes: List of class names
        device: Device (CPU or GPU)
    """
    print("\nGenerating confusion matrix...")
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("  ✓ Confusion matrix saved to 'results/confusion_matrix.png'")
    plt.close()
    
    return cm, all_targets, all_preds


def generate_classification_report(all_targets, all_preds, classes):
    """
    Generate and save classification report.
    
    Args:
        all_targets: True labels
        all_preds: Predicted labels
        classes: List of class names
    """
    print("\n" + "=" * 60)
    print("STEP 4: EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(all_targets, all_preds, 
                                  target_names=classes, digits=4)
    print(report)
    
    # Save report to file
    with open('results/classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print("  ✓ Classification report saved to 'results/classification_report.txt'")
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_targets, all_preds)
    print(f"\n{'='*60}")
    print(f"OVERALL TEST ACCURACY: {accuracy * 100:.2f}%")
    print(f"{'='*60}")


def visualize_predictions(model, test_loader, classes, device, num_samples=12):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained neural network model
        test_loader: DataLoader for test data
        classes: List of class names
        device: Device (CPU or GPU)
        num_samples: Number of samples to display
    """
    print("\nGenerating prediction visualizations...")
    
    model.eval()
    
    # Get a batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    images_device = images.to(device)
    outputs = model(images_device)
    _, predicted = outputs.max(1)
    
    # Denormalize images for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    fig.suptitle('Model Predictions on Test Images', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            # Denormalize and convert to numpy
            img = images[idx] * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            
            true_label = classes[labels[idx]]
            pred_label = classes[predicted[idx].cpu()]
            
            # Color code: green if correct, red if incorrect
            color = 'green' if labels[idx] == predicted[idx].cpu() else 'red'
            title = f'True: {true_label}\nPred: {pred_label}'
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/predictions.png', dpi=150, bbox_inches='tight')
    print("  ✓ Predictions saved to 'results/predictions.png'")
    plt.close()


# ================== 5. MAIN EXECUTION ==================

def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 60)
    print("MACHINE LEARNING COURSEWORK 2")
    print("IMAGE CLASSIFICATION WITH NEURAL NETWORKS")
    print("=" * 60)
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Step 1: Load and prepare data
    train_loader, test_loader, classes = load_and_prepare_data()
    visualize_sample_data(train_loader, classes)
    
    # Step 2: Create model
    model = ImageClassificationNN(num_classes=len(classes))
    print_model_architecture(model)
    
    # Step 3: Train and evaluate
    history, trained_model = train_and_evaluate(
        model, train_loader, test_loader,
        num_epochs=30,
        learning_rate=0.001,
        device=device
    )
    
    # Step 4: Visualize results
    plot_training_history(history)
    
    # Step 5: Generate evaluation metrics
    cm, all_targets, all_preds = generate_confusion_matrix(
        trained_model, test_loader, classes, device
    )
    generate_classification_report(all_targets, all_preds, classes)
    
    # Step 6: Visualize predictions
    visualize_predictions(trained_model, test_loader, classes, device)
    
    # Save training history
    with open('results/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("\n  ✓ Training history saved to 'results/training_history.pkl'")
    
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  1. results/best_model.pth - Trained model weights")
    print("  2. results/sample_images.png - Sample dataset images")
    print("  3. results/training_history.png - Training curves")
    print("  4. results/confusion_matrix.png - Confusion matrix")
    print("  5. results/classification_report.txt - Detailed metrics")
    print("  6. results/predictions.png - Sample predictions")
    print("  7. results/training_history.pkl - Training data")
    print("\nNext Steps:")
    print("  - Run 'python visualize_model.py' to see model architecture diagram")
    print("  - Run 'python predict.py <image_path>' to test on custom images")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
