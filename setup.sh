#!/bin/bash

# Setup script for Machine Learning Coursework 2
# This script will set up the environment and install dependencies

echo "========================================"
echo "ML Coursework 2 - Setup Script"
echo "========================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed."
    echo "Please install Python 3.8 or higher from https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created successfully"
else
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment activated"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All dependencies installed successfully!"
else
    echo ""
    echo "❌ Failed to install dependencies"
    echo "Please try manually: pip install -r requirements.txt"
    exit 1
fi

# Create results directory
echo ""
echo "Creating results directory..."
mkdir -p results
echo "✓ Results directory created"

# Verify installation
echo ""
echo "Verifying PyTorch installation..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

if [ $? -eq 0 ]; then
    echo "✓ PyTorch installed correctly"
else
    echo "⚠️  Warning: PyTorch verification failed"
fi

echo ""
echo "========================================"
echo "Setup completed successfully! 🎉"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run quick test: python3 quick_start.py"
echo "3. Run full training: python3 main.py"
echo "4. Visualize model: python3 visualize_model.py"
echo "5. Make predictions: python3 predict.py"
echo ""
echo "To deactivate virtual environment: deactivate"
echo ""
