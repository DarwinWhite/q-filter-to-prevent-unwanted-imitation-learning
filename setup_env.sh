#!/bin/bash
# Setup script for Q-filter project environment

set -e  # Exit on error

echo "Setting up Q-filter project environment..."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3.7."
    exit 1
fi

# Check Python version compatibility
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION 3.7" | awk '{print ($1 > $2)}') == 1 ]]; then
    echo "Warning: This codebase requires TensorFlow 1.x which only supports Python 3.7 and earlier."
    echo "   Your current Python version is $PYTHON_VERSION."
    echo ""
    echo "Options:"
    echo "1. Install Python 3.7 and use that (recommended for exact compatibility)"
    echo "2. Continue with TensorFlow 2.x in compatibility mode (experimental)"
    echo ""
    read -p "Continue with TensorFlow 2.x compatibility mode? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Setup cancelled. Please install Python 3.7 for best compatibility."
        exit 1
    fi
    USE_TF2_COMPAT=1
else
    USE_TF2_COMPAT=0
fi

# Create virtual environment
VENV_NAME="env"
echo "Creating virtual environment: $VENV_NAME"

if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment already exists. Removing it..."
    rm -rf $VENV_NAME
fi

python3 -m venv $VENV_NAME

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Upgrade pip
echo "⬆Upgrading pip..."
pip install --upgrade pip

# Install basic requirements first
echo "Installing basic requirements..."
pip install numpy==1.21.0  # Specific version that works well with TF 1.x

# Install TensorFlow (version depends on Python compatibility)
if [ $USE_TF2_COMPAT -eq 1 ]; then
    echo "Installing TensorFlow 2.x with v1 compatibility..."
    pip install tensorflow==2.13.1
    echo "Note: Using TensorFlow 2.x in v1 compatibility mode."
    echo "   You may need to add 'import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()' to scripts."
else
    echo "Installing TensorFlow 1.x..."
    pip install tensorflow==1.15.0
fi

# Install PyTorch (for our new demo generation scripts)
echo "Installing PyTorch..."
pip install torch==1.12.1 --index-url https://download.pytorch.org/whl/cpu

# Install MuJoCo and gym
echo "Installing MuJoCo and Gym..."
# Note: We skip mujoco-py as it requires system MuJoCo libraries
# The modern mujoco package will be installed via OpenAI Baselines
pip install gym==0.26.2  # Use newer compatible version

# Install other requirements (skip problematic ones)
echo "Installing remaining requirements..."
# Skip mpi4py as it requires system MPI libraries - not needed for basic usage
pip install matplotlib==3.5.3
pip install seaborn==0.11.2
pip install click==8.1.3
pip install glob2==0.7

# Install OpenAI Baselines (specific commit)
echo "Installing OpenAI Baselines (specific commit)..."
pip install git+https://github.com/openai/baselines.git@a6b1bc70f156dc45c0da49be8e80941a88021700

echo "Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "    source env/bin/activate"
echo ""
echo "To test the setup, run:"
echo "    python test/test_policy_loading.py"
echo ""
echo "To deactivate when done:"
echo "    deactivate"