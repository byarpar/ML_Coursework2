# REMOVED

This guide will help you set up the project and install all necessary dependencies.

## Step 1: Check Python Version

Make sure you have Python 3.8 or higher installed:

```bash
python3 --version
```

If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/).

## Step 2: Create Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other projects:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

## Step 3: Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and torchvision (deep learning framework)
- NumPy (numerical computing)
- Matplotlib and Seaborn (visualization)
- scikit-learn (machine learning utilities)
- Pillow (image processing)
- Other supporting libraries

### Alternative: Install PyTorch with CUDA (for GPU support)

If you have an NVIDIA GPU and want to use it for faster training:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install other requirements
pip install numpy pandas scikit-learn matplotlib seaborn Pillow tqdm h5py
```

Visit [PyTorch's website](https://pytorch.org/get-started/locally/) to find the correct command for your system.

## Step 4: Verify Installation

Verify that PyTorch is installed correctly:

```bash
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Step 5: Run the Project

### Option A: Full Training (30 epochs, ~30-40 minutes)

```bash
python3 main.py
```

### Option B: Quick Test (5 epochs, ~5-10 minutes)

```bash
python3 quick_start.py
```

### Option C: Visualize Model Architecture

```bash
python3 visualize_model.py
```

### Option D: Test on Custom Image

```bash
# With specific image
python3 predict.py path/to/your/image.jpg

# Interactive mode
python3 predict.py
```

## Step 6: Check Results

After training, you'll find the following files in the `results/` directory:

- `best_model.pth` - Trained model weights
- `sample_images.png` - Sample dataset images
- `training_history.png` - Training curves (loss and accuracy)
- `confusion_matrix.png` - Confusion matrix heatmap
- `classification_report.txt` - Detailed performance metrics
- `predictions.png` - Sample predictions visualization

## Troubleshooting

### Issue: "No module named 'torch'"

Solution: Install PyTorch:
```bash
pip install torch torchvision
```

### Issue: "CUDA out of memory"

Solution: Reduce batch size in main.py (line 35):
```python
batch_size = 64  # Change from 128 to 64 or 32
```

### Issue: Training is very slow

Solutions:
1. Use GPU if available (see CUDA installation above)
2. Reduce number of epochs for testing (use quick_start.py)
3. Reduce batch size to use less memory

### Issue: "FileNotFoundError: Model file not found"

Solution: Train the model first:
```bash
python3 main.py
```

### Issue: Image prediction fails

Solution: Make sure the image is in a supported format (jpg, png, etc.) and the path is correct.

## System Requirements

### Minimum Requirements:
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- CPU: Any modern processor

### Recommended Requirements:
- Python 3.9+
- 8GB+ RAM
- 5GB free disk space
- GPU: NVIDIA GPU with 4GB+ VRAM (for faster training)

## Next Steps

Once installed and running successfully:

1. Review the generated visualizations in `results/`
2. Read the classification report to understand model performance
3. Test with your own images using `predict.py`
4. Experiment with different hyperparameters in `main.py`
5. Review the code to understand the neural network architecture

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Convolutional Neural Networks Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## Getting Help

If you encounter any issues:

1. Check that all dependencies are installed: `pip list`
2. Verify Python version: `python3 --version`
3. Check available disk space
4. Review error messages carefully
5. Search for the error message online

## Deactivating Virtual Environment

When you're done working on the project:

```bash
deactivate
```

This will return you to your system's default Python environment.
