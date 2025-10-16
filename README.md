# REMOVED

This file was removed from the repository at the request of the user on October 16, 2025.
If you need the original README restored, please check the repository's version control history.
- **Testing Samples**: 10,000 images
- **Image Dimensions**: 32×32 pixels (RGB - 3 channels)
- **Number of Classes**: 10
- **Classes**: 
  1. Airplane
  2. Automobile
  3. Bird
  4. Cat
  5. Deer
  6. Dog
  7. Frog
  8. Horse
  9. Ship
  10. Truck

---

## 🧠 Neural Network Architecture

### Model Design: ImageClassificationNN

Our neural network is a **Convolutional Neural Network (CNN)** specifically designed for image classification tasks. The architecture consists of multiple layers organized as follows:

#### **1. Input Layer**
- **Input Shape**: 3 × 32 × 32 (RGB image)
- Accepts color images with 3 channels (Red, Green, Blue)

#### **2. Convolutional Block 1**
- **Conv2D Layer 1**: 32 filters, 3×3 kernel, padding=1
- **Batch Normalization**: Normalizes activations
- **ReLU Activation**: Introduces non-linearity
- **Conv2D Layer 2**: 32 filters, 3×3 kernel, padding=1
- **Batch Normalization**
- **ReLU Activation**
- **Max Pooling**: 2×2, stride=2 (reduces dimension from 32×32 → 16×16)

#### **3. Convolutional Block 2**
- **Conv2D Layer 3**: 64 filters, 3×3 kernel, padding=1
- **Batch Normalization**
- **ReLU Activation**
- **Conv2D Layer 4**: 64 filters, 3×3 kernel, padding=1
- **Batch Normalization**
- **ReLU Activation**
- **Max Pooling**: 2×2, stride=2 (reduces dimension from 16×16 → 8×8)

#### **4. Convolutional Block 3**
- **Conv2D Layer 5**: 128 filters, 3×3 kernel, padding=1
- **Batch Normalization**
- **ReLU Activation**
- **Conv2D Layer 6**: 128 filters, 3×3 kernel, padding=1
- **Batch Normalization**
- **ReLU Activation**
- **Max Pooling**: 2×2, stride=2 (reduces dimension from 8×8 → 4×4)

#### **5. Fully Connected Layers**
- **Flatten**: Converts 3D feature maps to 1D vector (128 × 4 × 4 = 2,048 features)
- **Dense Layer 1**: 512 neurons, ReLU activation
- **Dropout**: 0.5 (prevents overfitting)
- **Dense Layer 2**: 256 neurons, ReLU activation
- **Dropout**: 0.3
- **Output Layer**: 10 neurons (one for each class)

#### **Total Parameters**: ~1.5 million trainable parameters

### Architecture Diagram

```
Input (3×32×32)
      ↓
[Conv 32] → [BatchNorm] → [ReLU] → [Conv 32] → [BatchNorm] → [ReLU] → [MaxPool]
      ↓ (16×16)
[Conv 64] → [BatchNorm] → [ReLU] → [Conv 64] → [BatchNorm] → [ReLU] → [MaxPool]
      ↓ (8×8)
[Conv 128] → [BatchNorm] → [ReLU] → [Conv 128] → [BatchNorm] → [ReLU] → [MaxPool]
      ↓ (4×4)
[Flatten] → [Dense 512] → [ReLU] → [Dropout 0.5]
      ↓
[Dense 256] → [ReLU] → [Dropout 0.3]
      ↓
[Dense 10] → Output (10 classes)
```

### Why This Architecture?

1. **Convolutional Layers**: Extract spatial features from images (edges, textures, patterns)
2. **Batch Normalization**: Stabilizes training and allows higher learning rates
3. **ReLU Activation**: Introduces non-linearity, allowing the network to learn complex patterns
4. **Max Pooling**: Reduces spatial dimensions while preserving important features
5. **Dropout**: Prevents overfitting by randomly deactivating neurons during training
6. **Progressive Filters**: Increasing filter counts (32→64→128) allows learning from simple to complex features

---

## 🚀 Training Process

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 (with ReduceLROnPlateau scheduler) |
| **Loss Function** | CrossEntropyLoss |
| **Batch Size** | 128 |
| **Epochs** | 30 |
| **Weight Decay** | None |

### Training Steps

1. **Data Preprocessing**:
   - Normalize images using CIFAR-10 mean and standard deviation
   - Mean: (0.4914, 0.4822, 0.4465)
   - Std: (0.2470, 0.2435, 0.2616)

2. **Forward Pass**:
   - Input batch passes through all network layers
   - Model outputs 10 scores (logits) for each image

3. **Loss Calculation**:
   - CrossEntropyLoss measures difference between predictions and true labels
   - Combines softmax activation and negative log-likelihood

4. **Backward Pass**:
   - Gradients computed via backpropagation
   - Weights updated using Adam optimizer

5. **Learning Rate Scheduling**:
   - Reduces learning rate by 50% if validation loss doesn't improve for 3 epochs
   - Helps model converge to better solutions

### Training Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│                  START TRAINING                      │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Load CIFAR-10 Dataset (50,000 train, 10,000 test) │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│          Normalize Images (mean, std)               │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│        Initialize Model with Random Weights         │
└────────────────────┬────────────────────────────────┘
                     ↓
         ┌───────────────────────┐
         │   FOR EACH EPOCH      │
         └───────┬───────────────┘
                 ↓
    ┌────────────────────────────┐
    │  FOR EACH BATCH (128 imgs) │
    └────────┬───────────────────┘
             ↓
    ┌─────────────────────┐
    │   Forward Pass      │ ← Input batch → CNN → Output predictions
    └────────┬────────────┘
             ↓
    ┌─────────────────────┐
    │  Calculate Loss     │ ← Compare predictions vs true labels
    └────────┬────────────┘
             ↓
    ┌─────────────────────┐
    │  Backward Pass      │ ← Compute gradients
    └────────┬────────────┘
             ↓
    ┌─────────────────────┐
    │  Update Weights     │ ← Adam optimizer
    └────────┬────────────┘
             ↓
    ┌─────────────────────┐
    │   End of Batch      │
    └────────┬────────────┘
             ↓
         [Repeat for all batches]
             ↓
    ┌─────────────────────┐
    │ Evaluate on Test Set│ ← Calculate test accuracy
    └────────┬────────────┘
             ↓
    ┌─────────────────────┐
    │ Adjust Learning Rate│ ← ReduceLROnPlateau
    └────────┬────────────┘
             ↓
    ┌─────────────────────┐
    │  Save Best Model    │ ← If test accuracy improved
    └────────┬────────────┘
             ↓
         [Repeat for 30 epochs]
             ↓
┌─────────────────────────────────────────────────────┐
│              TRAINING COMPLETED                      │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Expected Results

Based on the architecture and training configuration, you can expect:

- **Training Accuracy**: ~85-90%
- **Testing Accuracy**: ~75-80%
- **Training Time**: 20-40 minutes (depending on hardware)
  - With GPU: ~20-30 minutes
  - With CPU: ~30-40 minutes

### Evaluation Metrics

The project generates comprehensive evaluation metrics:

1. **Accuracy**: Overall percentage of correct predictions
2. **Precision**: Ratio of correct positive predictions
3. **Recall**: Ratio of actual positives correctly identified
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Visualization of correct vs incorrect predictions per class

---

## 🛠️ Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- torchvision (computer vision library)
- NumPy (numerical computing)
- Matplotlib & Seaborn (visualization)
- scikit-learn (machine learning utilities)
- Pillow (image processing)

---

## ▶️ Usage

### 1. Train the Model

Run the main script to train the neural network:

```bash
python main.py
```

This will:
- Download the CIFAR-10 dataset automatically
- Train the neural network for 30 epochs
- Save the best model to `results/best_model.pth`
- Generate various visualizations and reports

**Generated Output Files**:
- `results/best_model.pth` - Trained model weights
- `results/sample_images.png` - Sample dataset images
- `results/training_history.png` - Loss and accuracy curves
- `results/confusion_matrix.png` - Confusion matrix heatmap
- `results/classification_report.txt` - Detailed performance metrics
- `results/predictions.png` - Sample predictions visualization
- `results/training_history.pkl` - Training data for further analysis

### 2. Visualize Model Architecture

```bash
python visualize_model.py
```

Generates a detailed diagram of the neural network architecture.

### 3. Test on Custom Images

```bash
python predict.py path/to/your/image.jpg
```

Or run interactively:

```bash
python predict.py
```

Then follow the prompts to select an image.

---

## 📁 Project Structure

```
MLcoursework2/
├── main.py                          # Main training script
├── predict.py                       # Custom image prediction script
├── visualize_model.py               # Model architecture visualization
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data/                           # Auto-downloaded CIFAR-10 dataset
│   └── cifar-10-batches-py/
└── results/                        # Generated outputs
    ├── best_model.pth
    ├── sample_images.png
    ├── training_history.png
    ├── confusion_matrix.png
    ├── classification_report.txt
    ├── predictions.png
    └── training_history.pkl
```

---

## 🎓 Coursework Requirements Checklist

### ✅ Required Features

- [x] **1. Public Image Dataset**: Using CIFAR-10 dataset with 60,000 images
- [x] **2. Data Import**: Automated download and loading via torchvision
- [x] **3. Input/Output Definition**: 
  - Input: 3×32×32 RGB images
  - Output: 10-class predictions
- [x] **4. Train/Test Split**: 50,000 training, 10,000 testing samples
- [x] **5. Neural Network Design**: Multi-layer CNN with convolutional and fully connected layers
- [x] **6. Model Training**: Complete training process with Adam optimizer
- [x] **7. Model Evaluation**: Test accuracy and comprehensive metrics
- [x] **8. Training Process Explanation**: Detailed documentation with diagrams
- [x] **9. Model Structure Explanation**: Architecture diagrams and descriptions

### ✅ Optional Features

- [x] **Quick Rerun**: Script can be rerun without errors
- [x] **Custom Image Testing**: `predict.py` allows user to test their own images
- [x] **Additional Features**:
  - Confusion matrix visualization
  - Per-class performance metrics
  - Training history plots
  - Sample prediction visualization
  - Model architecture visualization
  - Learning rate scheduling
  - Batch normalization
  - Dropout for regularization
  - Progress tracking during training

---

## 🔬 Technical Details

### Key Technologies

- **PyTorch**: Deep learning framework for building and training neural networks
- **torchvision**: Provides easy access to CIFAR-10 dataset and image transformations
- **CUDA Support**: Automatically uses GPU if available for faster training

### Design Decisions

1. **Why CNN over Fully Connected?**
   - CNNs preserve spatial relationships in images
   - Significantly fewer parameters than fully connected networks
   - Better generalization on image data

2. **Batch Normalization**:
   - Stabilizes training by normalizing layer inputs
   - Allows higher learning rates
   - Acts as regularization

3. **Dropout Regularization**:
   - Prevents overfitting by randomly dropping neurons
   - Different dropout rates in different layers (0.5 and 0.3)

4. **Data Normalization**:
   - Using CIFAR-10 specific mean and std ensures better convergence
   - Brings all features to similar scale

---

## 📈 Performance Tips

### To Improve Accuracy:

1. **Increase Epochs**: Train for 50-100 epochs
2. **Data Augmentation**: Add random flips, rotations, crops
3. **Learning Rate Tuning**: Experiment with different learning rates
4. **Architecture Changes**: Add more convolutional layers or residual connections
5. **Ensemble Methods**: Train multiple models and average predictions

### To Speed Up Training:

1. **Use GPU**: Ensure PyTorch can access CUDA-enabled GPU
2. **Increase Batch Size**: Use larger batches if memory allows
3. **Mixed Precision**: Use torch.cuda.amp for faster training
4. **Reduce Epochs**: Start with fewer epochs for testing

---

## 📝 Citation

If using CIFAR-10 dataset, please cite:

```
Learning Multiple Layers of Features from Tiny Images
Alex Krizhevsky, 2009
https://www.cs.toronto.edu/~kriz/cifar.html
```

---

## 🤝 Support

For questions or issues:
1. Check that all dependencies are installed correctly
2. Ensure Python version is 3.8+
3. Verify sufficient disk space (~200MB for dataset)
4. Check GPU availability: `torch.cuda.is_available()`

---

## 📄 License

This project is created for educational purposes as part of Machine Learning Coursework 2.

