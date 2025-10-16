"""
Quick Start Script

This script provides a quick way to train the model with reduced settings
for testing purposes.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import *


def quick_train():
    """Quick training with reduced epochs for testing"""
    
    print("\n" + "=" * 60)
    print("QUICK START - REDUCED TRAINING")
    print("=" * 60)
    print("\nThis will train the model for only 5 epochs for quick testing.")
    print("For full training, run: python main.py\n")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    train_loader, test_loader, classes = load_and_prepare_data()
    
    # Create model
    model = ImageClassificationNN(num_classes=len(classes))
    print_model_architecture(model)
    
    # Quick training (only 5 epochs)
    print("\n⚡ QUICK TRAINING MODE: 5 epochs only")
    history, trained_model = train_and_evaluate(
        model, train_loader, test_loader,
        num_epochs=5,  # Reduced from 30
        learning_rate=0.001,
        device=device
    )
    
    # Generate basic visualizations
    plot_training_history(history)
    cm, all_targets, all_preds = generate_confusion_matrix(
        trained_model, test_loader, classes, device
    )
    generate_classification_report(all_targets, all_preds, classes)
    
    print("\n" + "=" * 60)
    print("QUICK START COMPLETED!")
    print("=" * 60)
    print("\nFor full training (30 epochs), run: python main.py")
    print("For predictions, run: python predict.py <image_path>")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    quick_train()
