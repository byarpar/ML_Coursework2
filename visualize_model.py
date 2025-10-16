"""
Model Architecture Visualization Script

This script creates a visual diagram of the neural network architecture
and displays detailed information about each layer.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


class ImageClassificationNN(nn.Module):
    """Same model as in main.py for visualization"""
    
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


def print_model_summary():
    """Print detailed model summary"""
    print("\n" + "=" * 70)
    print("NEURAL NETWORK ARCHITECTURE SUMMARY")
    print("=" * 70)
    
    model = ImageClassificationNN(num_classes=10)
    
    print("\n📊 MODEL OVERVIEW")
    print("-" * 70)
    print(f"Model Name: ImageClassificationNN")
    print(f"Task: Image Classification (CIFAR-10)")
    print(f"Input: 3×32×32 RGB images")
    print(f"Output: 10 classes")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📈 PARAMETERS")
    print("-" * 70)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    print(f"Model Size: {size_mb:.2f} MB")
    
    print(f"\n🔧 LAYER DETAILS")
    print("-" * 70)
    
    layer_info = [
        ("Input", "3×32×32", "RGB Image", "-"),
        ("Conv1-1", "32×32×32", "Conv2D(3→32, 3×3)", "9,408"),
        ("Conv1-2", "32×32×32", "Conv2D(32→32, 3×3)", "9,216"),
        ("Pool1", "32×16×16", "MaxPool2D(2×2)", "0"),
        ("Conv2-1", "64×16×16", "Conv2D(32→64, 3×3)", "18,432"),
        ("Conv2-2", "64×16×16", "Conv2D(64→64, 3×3)", "36,864"),
        ("Pool2", "64×8×8", "MaxPool2D(2×2)", "0"),
        ("Conv3-1", "128×8×8", "Conv2D(64→128, 3×3)", "73,728"),
        ("Conv3-2", "128×8×8", "Conv2D(128→128, 3×3)", "147,456"),
        ("Pool3", "128×4×4", "MaxPool2D(2×2)", "0"),
        ("Flatten", "2048", "Flatten", "0"),
        ("FC1", "512", "Linear(2048→512)", "1,048,576"),
        ("Dropout1", "512", "Dropout(0.5)", "0"),
        ("FC2", "256", "Linear(512→256)", "131,072"),
        ("Dropout2", "256", "Dropout(0.3)", "0"),
        ("Output", "10", "Linear(256→10)", "2,570"),
    ]
    
    print(f"{'Layer':<12} {'Output Shape':<15} {'Layer Type':<25} {'Parameters':<15}")
    print("-" * 70)
    for layer, shape, layer_type, params in layer_info:
        print(f"{layer:<12} {shape:<15} {layer_type:<25} {params:<15}")
    
    print("-" * 70)
    print(f"\n✅ Total: {total_params:,} parameters")
    print("=" * 70 + "\n")


def visualize_architecture():
    """Create visual diagram of the architecture"""
    print("Generating architecture visualization...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'Convolutional Neural Network Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    ax.text(8, 9.0, 'Image Classification on CIFAR-10 Dataset', 
            fontsize=14, ha='center', style='italic')
    
    # Define layer positions and properties
    layers = [
        # (x, y, width, height, label, details, color)
        (0.5, 6.5, 1.0, 2.0, "Input\nLayer", "3×32×32\nRGB Image", "#E8F4F8"),
        (2.0, 6.5, 1.5, 2.0, "Conv\nBlock 1", "32 filters\n3×3 kernel\n16×16", "#B3E5FC"),
        (4.0, 6.5, 1.5, 2.0, "Conv\nBlock 2", "64 filters\n3×3 kernel\n8×8", "#81D4FA"),
        (6.0, 6.5, 1.5, 2.0, "Conv\nBlock 3", "128 filters\n3×3 kernel\n4×4", "#4FC3F7"),
        (8.0, 6.5, 1.2, 2.0, "Flatten", "2048\nfeatures", "#29B6F6"),
        (9.7, 6.5, 1.3, 2.0, "Dense 1", "512 units\nReLU\nDropout", "#03A9F4"),
        (11.5, 6.5, 1.3, 2.0, "Dense 2", "256 units\nReLU\nDropout", "#0288D1"),
        (13.3, 6.5, 1.2, 2.0, "Output", "10 classes\nSoftmax", "#01579B"),
    ]
    
    # Draw layers
    for x, y, w, h, label, details, color in layers:
        # Box
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        
        # Label
        ax.text(x + w/2, y + h - 0.3, label, fontsize=11, fontweight='bold',
               ha='center', va='top')
        
        # Details
        ax.text(x + w/2, y + h/2 - 0.2, details, fontsize=8,
               ha='center', va='center', style='italic')
    
    # Draw arrows
    arrow_positions = [
        (1.5, 7.5, 2.0, 7.5),
        (3.5, 7.5, 4.0, 7.5),
        (5.5, 7.5, 6.0, 7.5),
        (7.5, 7.5, 8.0, 7.5),
        (9.2, 7.5, 9.7, 7.5),
        (11.0, 7.5, 11.5, 7.5),
        (12.8, 7.5, 13.3, 7.5),
    ]
    
    for x1, y1, x2, y2 in arrow_positions:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               color='black', linewidth=2)
        ax.add_patch(arrow)
    
    # Add feature extraction and classification labels
    ax.text(4.5, 5.8, 'Feature Extraction', fontsize=13, fontweight='bold',
           ha='center', bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))
    ax.text(11.5, 5.8, 'Classification', fontsize=13, fontweight='bold',
           ha='center', bbox=dict(boxstyle='round', facecolor='#FFE082', alpha=0.8))
    
    # Add detailed annotations below
    y_pos = 4.5
    annotations = [
        ("Convolutional Blocks:", "Extract spatial features using learnable filters"),
        ("Batch Normalization:", "Stabilizes training and improves convergence"),
        ("ReLU Activation:", "Introduces non-linearity (f(x) = max(0, x))"),
        ("Max Pooling:", "Reduces spatial dimensions while preserving features"),
        ("Dropout:", "Prevents overfitting by randomly deactivating neurons"),
        ("Dense Layers:", "Learn complex decision boundaries for classification"),
    ]
    
    ax.text(8, y_pos + 0.3, "Key Components:", fontsize=13, fontweight='bold', ha='center')
    
    for i, (component, description) in enumerate(annotations):
        y = y_pos - (i * 0.5)
        ax.text(2, y, f"• {component}", fontsize=9, fontweight='bold', ha='left')
        ax.text(5.5, y, description, fontsize=9, ha='left', style='italic')
    
    # Add training info
    info_box_text = (
        "Training Details:\n"
        "• Optimizer: Adam\n"
        "• Learning Rate: 0.001\n"
        "• Batch Size: 128\n"
        "• Epochs: 30\n"
        "• Loss: CrossEntropyLoss"
    )
    
    ax.text(8, 0.8, info_box_text, fontsize=9, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('results/model_architecture.png', dpi=300, bbox_inches='tight')
    print("  ✓ Architecture diagram saved to 'results/model_architecture.png'")
    plt.close()


def visualize_feature_maps_concept():
    """Visualize the concept of convolutional layers"""
    print("Generating feature extraction visualization...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Convolutional Neural Network: Feature Learning Process', 
                fontsize=16, fontweight='bold')
    
    # Simulate feature maps at different layers
    np.random.seed(42)
    
    # Input image
    input_img = np.random.rand(32, 32, 3)
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Input Image\n3×32×32', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Conv1 features (low-level: edges)
    conv1 = np.random.rand(16, 16)
    axes[0, 1].imshow(conv1, cmap='viridis')
    axes[0, 1].set_title('Conv Block 1\nEdges & Textures\n32×16×16', 
                        fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Conv2 features (mid-level: shapes)
    conv2 = np.random.rand(8, 8)
    axes[0, 2].imshow(conv2, cmap='viridis')
    axes[0, 2].set_title('Conv Block 2\nShapes & Patterns\n64×8×8', 
                        fontsize=10, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Conv3 features (high-level: objects)
    conv3 = np.random.rand(4, 4)
    axes[0, 3].imshow(conv3, cmap='viridis')
    axes[0, 3].set_title('Conv Block 3\nObject Parts\n128×4×4', 
                        fontsize=10, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Show example filters
    filter_examples = [
        ("3×3 Filter", "Detects\nEdges"),
        ("Pooling", "Reduces\nSize"),
        ("Activation", "Non-linear\nTransform"),
        ("Output", "Class\nProbabilities")
    ]
    
    for idx, (title, desc) in enumerate(filter_examples):
        if idx == 0:
            # Show a sample 3x3 filter
            filter_vis = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            axes[1, idx].imshow(filter_vis, cmap='RdBu', vmin=-1, vmax=1)
        elif idx == 1:
            # Show pooling effect
            axes[1, idx].text(0.5, 0.5, '2×2\nMax\nPooling', 
                            ha='center', va='center', fontsize=14,
                            transform=axes[1, idx].transAxes)
        elif idx == 2:
            # Show ReLU activation
            x = np.linspace(-2, 2, 100)
            y = np.maximum(0, x)
            axes[1, idx].plot(x, y, 'b-', linewidth=3)
            axes[1, idx].set_title('ReLU Activation', fontsize=10, fontweight='bold')
            axes[1, idx].grid(True, alpha=0.3)
        else:
            # Show class probabilities
            classes = ['plane', 'car', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
            probs = np.random.rand(10)
            probs = probs / probs.sum()
            top_idx = np.argmax(probs)
            
            colors = ['red' if i == top_idx else 'lightblue' for i in range(10)]
            axes[1, idx].barh(range(10), probs, color=colors)
            axes[1, idx].set_yticks(range(10))
            axes[1, idx].set_yticklabels(classes, fontsize=7)
            axes[1, idx].set_xlabel('Probability', fontsize=8)
        
        axes[1, idx].set_title(f'{title}\n{desc}', fontsize=10, fontweight='bold')
        if idx < 2:
            axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/feature_learning_process.png', dpi=300, bbox_inches='tight')
    print("  ✓ Feature learning visualization saved to 'results/feature_learning_process.png'")
    plt.close()


def create_training_flowchart():
    """Create a flowchart of the training process"""
    print("Generating training process flowchart...")
    
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(5, 13.5, 'Training Process Flowchart', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Define flowchart boxes
    boxes = [
        (5, 12.5, 3, 0.6, "START", "#4CAF50", "white"),
        (5, 11.5, 3, 0.6, "Load CIFAR-10 Dataset", "#2196F3", "white"),
        (5, 10.5, 3, 0.6, "Preprocess & Normalize", "#2196F3", "white"),
        (5, 9.5, 3, 0.6, "Initialize CNN Model", "#2196F3", "white"),
        (5, 8.5, 3, 0.8, "For Each Epoch (1-30)", "#FF9800", "white"),
        (5, 7.3, 3, 0.8, "For Each Batch (128 images)", "#FF9800", "white"),
        (5, 6.2, 3, 0.6, "Forward Pass", "#9C27B0", "white"),
        (5, 5.2, 3, 0.6, "Calculate Loss", "#9C27B0", "white"),
        (5, 4.2, 3, 0.6, "Backward Pass (Gradients)", "#9C27B0", "white"),
        (5, 3.2, 3, 0.6, "Update Weights (Adam)", "#9C27B0", "white"),
        (5, 2.2, 3, 0.6, "Evaluate on Test Set", "#2196F3", "white"),
        (5, 1.2, 3, 0.6, "Save Best Model", "#2196F3", "white"),
        (5, 0.3, 3, 0.6, "END", "#4CAF50", "white"),
    ]
    
    # Draw boxes
    for x, y, w, h, text, color, text_color in boxes:
        if "For Each" in text or "START" in text or "END" in text:
            box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=color, 
                                linewidth=2.5)
        else:
            box = mpatches.Rectangle((x - w/2, y - h/2), w, h,
                                    edgecolor='black', facecolor=color,
                                    linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=10, fontweight='bold',
               ha='center', va='center', color=text_color)
    
    # Draw arrows
    arrow_y_positions = [12.2, 11.2, 10.2, 9.2, 8.1, 6.9, 5.9, 4.9, 3.9, 2.9, 1.9, 0.9]
    for y in arrow_y_positions:
        arrow = FancyArrowPatch((5, y), (5, y - 0.5),
                               arrowstyle='->', mutation_scale=20,
                               color='black', linewidth=2)
        ax.add_patch(arrow)
    
    # Add side annotations
    annotations = [
        (8.5, 6.7, "→ Input → CNN\n   → Output", 9),
        (8.5, 5.2, "→ CrossEntropyLoss", 9),
        (8.5, 4.2, "→ Compute ∂L/∂W", 9),
        (8.5, 3.2, "→ W = W - α·∇W", 9),
        (8.5, 2.2, "→ Accuracy check", 9),
    ]
    
    for x, y, text, fontsize in annotations:
        ax.text(x, y, text, fontsize=fontsize, ha='left', 
               style='italic', color='#555')
    
    # Add loop indicators
    ax.annotate('', xy=(7.2, 7.7), xytext=(7.2, 2.8),
               arrowprops=dict(arrowstyle='<-', color='#FF5722', 
                             linewidth=2, linestyle='dashed'))
    ax.text(7.7, 5.2, 'Repeat\nfor all\nbatches', fontsize=9, 
           ha='left', color='#FF5722', fontweight='bold')
    
    ax.annotate('', xy=(2.3, 8.9), xytext=(2.3, 1.8),
               arrowprops=dict(arrowstyle='<-', color='#FF9800', 
                             linewidth=2.5, linestyle='dashed'))
    ax.text(1.0, 5.5, 'Repeat for\n30 epochs', fontsize=10, 
           ha='center', color='#FF9800', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFF9C4'))
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', label='Start/End'),
        mpatches.Patch(facecolor='#2196F3', label='Data Operations'),
        mpatches.Patch(facecolor='#FF9800', label='Loops'),
        mpatches.Patch(facecolor='#9C27B0', label='Training Steps'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/training_flowchart.png', dpi=300, bbox_inches='tight')
    print("  ✓ Training flowchart saved to 'results/training_flowchart.png'")
    plt.close()


def main():
    """Main execution"""
    import os
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "=" * 70)
    print("MODEL VISUALIZATION SCRIPT")
    print("=" * 70)
    
    # Print model summary
    print_model_summary()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    print("-" * 70)
    visualize_architecture()
    visualize_feature_maps_concept()
    create_training_flowchart()
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. results/model_architecture.png")
    print("  2. results/feature_learning_process.png")
    print("  3. results/training_flowchart.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
