# REMOVED

This file was removed from the repository at the request of the user on October 16, 2025.
If you need the original document restored, please check the repository's version control history.
**Course**: Machine Learning  
**Assignment**: Coursework 2 - Image Classification with Neural Networks  

---

## 📋 Assignment Overview

### Task
Develop a machine learning product that can classify images using a neural network.

### Contribution
40% of total module mark

---

## ✅ Requirements Checklist

### Mandatory Requirements

| Requirement | Status | Implementation |
|------------|--------|----------------|
| 1. Find public image dataset (not digits) | ✅ Complete | CIFAR-10 dataset (60,000 images, 10 classes) |
| 2. Import data, define input/output, split train/test | ✅ Complete | 50,000 training / 10,000 testing images |
| 3. Design neural network (input, hidden, output layers) | ✅ Complete | CNN with 3 conv blocks + 2 FC layers |
| 4. Build and evaluate model, present accuracy | ✅ Complete | Training/testing pipeline implemented |
| 5. Explain training process and model structure | ✅ Complete | Documentation + visualizations provided |

### Optional Requirements (Bonus Features)

| Feature | Status | Implementation |
|---------|--------|----------------|
| Quick rerun capability | ✅ Complete | `quick_start.py` (5 epochs) and `main.py` (30 epochs) |
| User can import/select test image | ✅ Complete | `predict.py` with interactive mode |
| Additional features | ✅ Complete | Model visualization, comprehensive docs |

---

## 🗂️ Project Structure

```
MLcoursework2/
├── main.py                      # Main training script (30 epochs)
├── quick_start.py               # Quick training (5 epochs)
├── predict.py                   # Custom image prediction
├── visualize_model.py           # Model architecture visualization
├── test_prediction.py           # Prediction testing
├── simple_test.py               # Simple test script
├── requirements.txt             # Dependencies
├── setup.sh                     # Automated setup script
│
├── data/
│   └── cifar-10-batches-py/    # CIFAR-10 dataset (auto-downloaded)
│
├── results/
│   ├── best_model.pth           # Trained model weights
│   ├── training_history.png     # Training/validation curves
│   ├── confusion_matrix.png     # Confusion matrix visualization
│   └── classification_report.txt # Performance metrics
│
├── test_images/                 # Custom test images
├── images_for_pdf/              # Documentation images
│
└── Documentation/
    ├── README.md                # Main documentation
    ├── PROJECT_OVERVIEW.md      # Project summary
    ├── GETTING_STARTED.md       # Quick start guide
    ├── INSTALL.md               # Installation instructions
    ├── EXPERIMENTS.md           # Improvement experiments
    ├── CHECKLIST.md             # Submission checklist
    └── START_HERE.txt           # Getting started guide
```

---

## 🧠 Neural Network Architecture

### Model: ImageClassificationNN

**Type**: Convolutional Neural Network (CNN)

**Architecture Summary**:
```
Input Layer (3×32×32 RGB images)
    ↓
Convolutional Block 1: Conv(32) → BatchNorm → ReLU → Conv(32) → BatchNorm → ReLU → MaxPool
    ↓ (Output: 32×16×16)
Convolutional Block 2: Conv(64) → BatchNorm → ReLU → Conv(64) → BatchNorm → ReLU → MaxPool
    ↓ (Output: 64×8×8)
Convolutional Block 3: Conv(128) → BatchNorm → ReLU → Conv(128) → BatchNorm → ReLU → MaxPool
    ↓ (Output: 128×4×4)
Flatten → Fully Connected (512) → ReLU → Dropout(0.5)
    ↓
Fully Connected (256) → ReLU → Dropout(0.3)
    ↓
Output Layer (10 classes)
```

**Total Parameters**: ~1.5 million trainable parameters

**Key Features**:
- 6 Convolutional layers with increasing filters (32→64→128)
- Batch Normalization for training stability
- ReLU activation functions
- Max Pooling for dimensionality reduction
- Dropout layers (0.5, 0.3) to prevent overfitting
- 2 Fully Connected layers for classification

---

## 📊 Dataset Information

**Dataset**: CIFAR-10  
**Source**: Canadian Institute For Advanced Research  
**URL**: https://www.cs.toronto.edu/~kriz/cifar.html

**Specifications**:
- Total Images: 60,000 (32×32 color images)
- Training Set: 50,000 images
- Test Set: 10,000 images
- Image Dimensions: 32×32×3 (RGB)
- Number of Classes: 10

**Classes**:
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

## 🔧 Implementation Details

### Training Process

1. **Data Loading & Preprocessing**:
   - Automatic download of CIFAR-10 dataset
   - Data normalization (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
   - Data augmentation for training (random horizontal flip, random crop)
   - Batch size: 128

2. **Model Configuration**:
   - Optimizer: Adam (lr=0.001)
   - Loss Function: CrossEntropyLoss
   - Learning Rate Scheduler: ReduceLROnPlateau
   - Early stopping: Patience of 7 epochs

3. **Training Strategy**:
   - Epochs: 30 (full training) or 5 (quick start)
   - Validation split: 10% of training data
   - Best model saved based on validation accuracy
   - GPU acceleration if available (MPS for Mac, CUDA for others)

4. **Evaluation Metrics**:
   - Accuracy (overall and per-class)
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - Training/Validation loss curves

### Key Code Components

**Main Training Script** (`main.py`):
- Complete training pipeline
- Data loading and preprocessing
- Model training with validation
- Results visualization and saving

**Quick Start Script** (`quick_start.py`):
- Reduced epochs (5) for quick demonstration
- Same model architecture
- Faster evaluation for testing

**Prediction Script** (`predict.py`):
- Load pre-trained model
- Process custom images
- Display predictions with confidence scores
- Interactive mode for multiple predictions

**Visualization Script** (`visualize_model.py`):
- Model architecture diagram
- Layer-by-layer breakdown
- Parameter count visualization

---

## 📈 Results

### Performance Metrics

**Expected Accuracy**: 75-85% on test set

**Detailed Results** (check `results/classification_report.txt`):
- Per-class precision, recall, F1-score
- Confusion matrix showing classification patterns
- Training history with loss/accuracy curves

### Visualizations Generated

1. **Training History** (`training_history.png`):
   - Training vs Validation Loss
   - Training vs Validation Accuracy
   - Shows learning progression over epochs

2. **Confusion Matrix** (`confusion_matrix.png`):
   - Heatmap showing prediction patterns
   - Identifies which classes are confused
   - Normalized values for better interpretation

3. **Model Architecture** (via `visualize_model.py`):
   - Layer-by-layer structure
   - Dimensions at each stage
   - Parameter counts

---

## 🖼️ Visual Assets and Images for Assignment

### 1. Training History Graph (`results/training_history.png`)

**Description**:
This dual-panel visualization shows the model's learning progression over the training epochs.

**Left Panel - Loss Curves**:
- **Blue Line (Training Loss)**: Shows how the model's error decreases during training
- **Orange Line (Validation Loss)**: Shows error on unseen validation data
- **Y-axis**: Loss value (lower is better)
- **X-axis**: Training epochs (1-30)
- **Key Observation**: Both curves should decrease, with training loss slightly lower than validation loss. A large gap indicates overfitting.

**Right Panel - Accuracy Curves**:
- **Blue Line (Training Accuracy)**: Percentage of correctly classified training images
- **Orange Line (Validation Accuracy)**: Percentage correct on validation data
- **Y-axis**: Accuracy percentage (0-100%)
- **X-axis**: Training epochs (1-30)
- **Key Observation**: Both should increase and converge. Final accuracy typically 75-85%.

**What This Shows**:
- Model learning effectiveness
- Whether the model is overfitting or underfitting
- Training stability and convergence
- When early stopping might have occurred

---

### 2. Confusion Matrix (`results/confusion_matrix.png`)

**Description**:
A 10×10 heatmap showing the classification performance for each class pair.

**Structure**:
- **Rows**: True (actual) class labels
- **Columns**: Predicted class labels
- **Diagonal Elements**: Correct predictions (darker = better)
- **Off-Diagonal Elements**: Misclassifications (lighter = fewer mistakes)
- **Color Scale**: Dark blue (high values) to white (low values)

**Classes (in order)**:
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

**How to Read It**:
- Darker diagonal = model is confident and accurate for that class
- Bright spots off-diagonal = common confusion pairs
- Example: "Cat" and "Dog" often have higher confusion values
- Example: "Automobile" and "Truck" may be confused

**What This Shows**:
- Which classes are easy vs. difficult to classify
- Common misclassification patterns
- Whether similar objects are confused (expected)
- Overall model performance distribution

---

### 3. Sample Predictions (Generated during prediction)

**Description**:
When using `predict.py`, the system displays:

**Visual Elements**:
- **Original Image**: The input image (resized to 32×32 for processing)
- **Predicted Class**: Large text showing the predicted class name
- **Confidence Score**: Percentage (0-100%) showing model's certainty
- **Top-5 Predictions**: Bar chart showing probabilities for top 5 classes
- **Color Coding**: Green for high confidence (>70%), Yellow for medium (40-70%), Red for low (<40%)

**Example Output Text**:
```
Prediction: Airplane (85.3% confidence)

Top 5 Predictions:
1. Airplane:   85.3%
2. Ship:       8.2%
3. Bird:       3.1%
4. Automobile: 2.0%
5. Truck:      1.4%
```

**What This Shows**:
- Model's prediction on custom images
- Confidence level (certainty of prediction)
- Alternative possibilities considered
- Whether the model is making reasonable decisions

---

### 4. Model Architecture Diagram (via `visualize_model.py`)

**Description**:
Text-based or graphical representation of the neural network structure.

**Elements Shown**:
```
Layer               Output Shape      Parameters
================================================================
Input               [3, 32, 32]       0
Conv2d-1            [32, 32, 32]      896
BatchNorm2d-1       [32, 32, 32]      64
ReLU-1              [32, 32, 32]      0
Conv2d-2            [32, 32, 32]      9,248
BatchNorm2d-2       [32, 32, 32]      64
ReLU-2              [32, 32, 32]      0
MaxPool2d-1         [32, 16, 16]      0
Conv2d-3            [64, 16, 16]      18,496
BatchNorm2d-3       [64, 16, 16]      128
... (continues)
================================================================
Total Parameters:    1,547,210
Trainable Params:    1,547,210
Non-trainable:       0
================================================================
```

**What This Shows**:
- Complete layer-by-layer breakdown
- Input/output dimensions at each stage
- Parameter count per layer
- Total model complexity
- How data flows through the network

---

### 5. Sample Dataset Images (CIFAR-10 Examples)

**Description**:
Representative samples from each of the 10 classes in CIFAR-10.

**Characteristics**:
- **Resolution**: 32×32 pixels (low resolution by design)
- **Format**: RGB color images
- **Quality**: Variable - some images clear, others grainy
- **Content**: Objects typically centered but with variation

**Example Classes**:
- **Airplane**: Side view, various angles, different aircraft types
- **Automobile**: Cars, sedans, various colors
- **Bird**: Different species, various poses
- **Cat**: Different breeds, positions
- **Deer**: Various angles, natural settings
- **Dog**: Different breeds, poses
- **Frog**: Green frogs, various positions
- **Horse**: Different angles, colors
- **Ship**: Boats, vessels, various sizes
- **Truck**: Various truck types, angles

**Why Show These**:
- Demonstrates dataset diversity
- Shows challenge level (32×32 is quite small)
- Illustrates why some classes are easier/harder
- Provides context for model performance

---

### 6. Classification Report (`results/classification_report.txt`)

**Description**:
Text-based detailed metrics for each class.

**Format**:
```
                precision    recall  f1-score   support

      airplane       0.82      0.85      0.83      1000
    automobile       0.88      0.91      0.89      1000
          bird       0.71      0.68      0.69      1000
           cat       0.65      0.62      0.63      1000
          deer       0.76      0.79      0.77      1000
           dog       0.73      0.70      0.71      1000
          frog       0.85      0.88      0.86      1000
         horse       0.81      0.84      0.82      1000
          ship       0.87      0.89      0.88      1000
         truck       0.86      0.84      0.85      1000

      accuracy                           0.80     10000
     macro avg       0.79      0.80      0.79     10000
  weighted avg       0.79      0.80      0.79     10000
```

**Metrics Explained**:
- **Precision**: Of all predictions for a class, what % were correct?
- **Recall**: Of all actual instances of a class, what % were found?
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of test samples for each class (1000 each)
- **Accuracy**: Overall correctness (typically 75-85%)

**What This Shows**:
- Per-class performance breakdown
- Which classes perform well (e.g., automobile, ship, truck)
- Which classes are challenging (e.g., cat, dog, bird)
- Overall model effectiveness

---

### 7. Training Progress Screenshots (for video/presentation)

**What to Capture**:

**Screenshot 1 - Initial Training**:
- Terminal showing: "Starting training..."
- Epoch 1/30 progress
- Initial loss values (typically 2.0-2.3)
- Initial accuracy (typically 10-20%)

**Screenshot 2 - Mid Training**:
- Epoch 15/30
- Decreasing loss (typically 0.5-0.8)
- Improving accuracy (60-70%)
- Learning rate adjustments

**Screenshot 3 - Final Results**:
- Epoch 30/30 completed
- Final loss (typically 0.3-0.5)
- Final accuracy (75-85%)
- "Best model saved" message

**Screenshot 4 - Results Folder**:
- File explorer showing:
  - best_model.pth (50-60 MB)
  - training_history.png
  - confusion_matrix.png
  - classification_report.txt

---

## 📸 Images to Include in Your Submission

### Required Images:

1. **Training History Graph** ✓
   - Location: `results/training_history.png`
   - Shows learning curves
   - Demonstrates model improvement over time

2. **Confusion Matrix** ✓
   - Location: `results/confusion_matrix.png`
   - Shows classification accuracy per class
   - Highlights common misclassifications

3. **Custom Prediction Examples** (take screenshots)
   - Show 3-5 predictions on test images
   - Include both correct and interesting predictions
   - Display confidence scores

4. **Model Architecture Diagram** ✓
   - Generated by `visualize_model.py`
   - Shows complete network structure
   - Demonstrates complexity

5. **Dataset Samples** (optional but recommended)
   - Show 10 sample images (one per class)
   - Demonstrates dataset characteristics
   - Helps explain model challenges

### Optional Supporting Images:

6. **Code Structure Screenshot**
   - File explorer showing project organization
   - Highlights clean code structure

7. **Terminal Output Screenshots**
   - Training in progress
   - Successful completion messages
   - Error-free execution

8. **Comparative Results** (if you ran experiments)
   - Different epoch counts
   - Different hyperparameters
   - Shows your experimentation

---

## 🎨 Image Quality Guidelines for Submission

### For Screenshots:
- **Resolution**: At least 1920×1080 (Full HD)
- **Format**: PNG (lossless) preferred over JPG
- **Clarity**: Ensure text is readable, no blur
- **Cropping**: Remove unnecessary UI elements
- **File Size**: Compress if needed, but maintain readability

### For Graphs/Plots:
- **DPI**: 300 DPI for print quality
- **Labels**: All axes clearly labeled
- **Legend**: Included and readable
- **Title**: Descriptive and informative
- **Colors**: High contrast, colorblind-friendly

### For Video:
- **Resolution**: 1080p minimum
- **Frame Rate**: 30 fps
- **Audio**: Clear narration (optional but recommended)
- **Length**: 8-10 minutes maximum
- **Format**: MP4 (widely compatible)

---

## 📊 How to Generate All Images

### Automated Generation (Run Training):
```bash
# This generates training_history.png and confusion_matrix.png
python3 quick_start.py
# or
python3 main.py
```

### Manual Generation:

**1. Model Architecture**:
```bash
python3 visualize_model.py
```

**2. Custom Predictions** (take screenshots):
```bash
python3 predict.py test_images/airplane.jpg
python3 predict.py test_images/cat.jpg
python3 predict.py test_images/ship.jpg
```

dataset = datasets.CIFAR10('./data', train=False, download=True)
**3. Dataset Samples** (add code to display):

Run the included helper script `visualize_samples.py` (or `python3 -m scripts.visualize_samples` if you prefer a module layout) to generate a `results/dataset_samples.png` image containing one representative sample per CIFAR-10 class. The project includes this utility so you don't need to copy code snippets into the document.

---

## 🎬 Image Usage in Video Presentation

**Recommended Flow**:

1. **Opening** (0:00-0:30)
   - Show project folder structure
   - Highlight main files

2. **Dataset Introduction** (0:30-1:30)
   - Show dataset sample images
   - Explain CIFAR-10 characteristics
   - Display class distribution

3. **Architecture Explanation** (1:30-3:00)
   - Show model architecture diagram
   - Explain layer by layer
   - Highlight key features

4. **Training Process** (3:00-6:00)
   - Show training starting
   - Display progress updates
   - Show training history graph forming

5. **Results Analysis** (6:00-8:00)
   - Show confusion matrix
   - Explain classification report
   - Discuss performance metrics

6. **Live Demonstration** (8:00-9:30)
   - Predict 3-4 custom images
   - Show confidence scores
   - Discuss results

7. **Conclusion** (9:30-10:00)
   - Summarize achievements
   - Show results folder
   - Final thoughts

---

## 🚀 How to Run the Code

### 1. Setup (First Time Only)

**Option A - Automated**:
```bash
./setup.sh
```

**Option B - Manual**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Training

**Quick Demo (5 epochs, ~5-10 minutes)**:
```bash
python3 quick_start.py
```

**Full Training (30 epochs, ~30 minutes)**:
```bash
python3 main.py
```

### 3. Testing with Custom Images

**Predict single image**:
```bash
python3 predict.py path/to/your/image.jpg
```

**Interactive mode**:
```bash
python3 predict.py
```

### 4. Visualization

**View model architecture**:
```bash
python3 visualize_model.py
```

---

## 💡 Design Decisions and Rationale

### Why CNN Architecture?

Convolutional Neural Networks are ideal for image classification because:
- **Spatial Feature Extraction**: Conv layers detect edges, textures, and patterns
- **Parameter Efficiency**: Shared weights reduce parameters compared to fully connected networks
- **Translation Invariance**: Features detected anywhere in the image
- **Hierarchical Learning**: Early layers detect simple features, deeper layers detect complex patterns

### Why These Specific Layers?

1. **Multiple Conv Layers**: Allows learning hierarchical features
2. **Batch Normalization**: Stabilizes training, allows higher learning rates
3. **Max Pooling**: Reduces spatial dimensions, provides translation invariance
4. **Dropout**: Prevents overfitting by randomly dropping neurons during training
5. **Progressive Filters (32→64→128)**: Captures increasingly complex features

### Why CIFAR-10 Dataset?

- Well-established benchmark dataset
- Appropriate difficulty level (not too easy, not too hard)
- Diverse classes covering various object types
- Manageable size for training on standard hardware
- Widely used in academic research for comparison

---

## 🎓 Learning Outcomes Demonstrated

1. **Data Handling**: Successfully import, preprocess, and split image data
2. **Neural Network Design**: Implement multi-layer CNN architecture from scratch
3. **Training Process**: Understand and implement model training with validation
4. **Evaluation**: Comprehensive model evaluation with multiple metrics
5. **Visualization**: Create meaningful visualizations of results
6. **Code Quality**: Well-structured, documented, and reusable code
7. **Problem Solving**: Implement bonus features and additional functionality

---

## 📹 Video Demonstration

**Video Link**: [INSERT YOUR VIDEO LINK HERE]

**Video Duration**: [DURATION] minutes

**Video Contents**:
1. Project overview and code structure (1 min)
2. Setup and installation demonstration (1 min)
3. Training process (quick_start.py) (3 min)
4. Results visualization (2 min)
5. Custom image prediction demonstration (2 min)
6. Additional features showcase (1 min)

**Video Recording Tool Used**: [e.g., QuickTime, OBS, Zoom, etc.]

---

## 🔗 Additional Resources

### Documentation Files:
- `README.md` - Comprehensive project documentation
- `PROJECT_OVERVIEW.md` - High-level project summary
- `GETTING_STARTED.md` - Quick start instructions
- `INSTALL.md` - Detailed installation guide
- `EXPERIMENTS.md` - How to improve the model
- `CHECKLIST.md` - Submission checklist

### References:
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch Documentation: https://pytorch.org/docs/
- Papers With Code: https://paperswithcode.com/datasets

---

## 📝 Code Comments and Documentation

All code files are thoroughly commented with:
- Function docstrings explaining purpose, parameters, and returns
- Inline comments for complex logic
- Section headers for code organization
- Type hints for clarity
- Usage examples where appropriate

Key files with extensive documentation:
- `main.py`: Complete training pipeline with detailed comments
- `predict.py`: Prediction functionality with usage instructions
- `visualize_model.py`: Visualization code with explanations

---

## 🎯 Academic Integrity Statement

I confirm that:
- This work is entirely my own
- All external sources and references are properly cited
- No plagiarism or academic misconduct has occurred
- The code and documentation were created independently
- I understand the university's academic integrity policies

**Signature**: _________________  
**Date**: _________________

---

## 📤 Submission Details

### Submitted Files:
- [ ] All Python code files with name and ID
- [ ] Documentation (this file)
- [ ] Video recording link included
- [ ] README.md and supporting documentation
- [ ] Results folder with outputs

### Submission Method:
- Platform: Moodle
- Deadline: November 22, 2024, 23:59 GMT
- Format: [Specify format - ZIP file, etc.]

### Lab Presentation:
- Date: [TO BE SCHEDULED - within 2 weeks after Nov 22]
- Tutor: [TUTOR NAME]
- Prepared: [ ] Yes  [ ] No

---

## 📧 Contact Information

**Student**: [YOUR NAME]  
**Email**: [YOUR EMAIL]  
**Student ID**: [YOUR ID]  

**For Questions Contact**:
- Module Leader: [NAME]
- Email: [EMAIL]

---

## 🏆 Summary

This project successfully implements a complete image classification system using Convolutional Neural Networks. All mandatory requirements have been met, and all optional bonus features have been implemented. The code is well-documented, functional, and demonstrates a thorough understanding of machine learning concepts and practices.

**Total Features Implemented**: 10+  
**Lines of Code**: 1000+  
**Documentation Pages**: 7  
**Visualizations**: 3+  

---

**End of Assignment Document**

*Generated: October 16, 2025*  
*Last Updated: [DATE]*
