#!/bin/bash
# Setup script for Q-filter project environment with original dependencies
# This script sets up Python 3.7 + TensorFlow 1.15.0 for exact compatibility

set -e  # Exit on error

echo "Setting up Q-filter project environment with original dependencies..."

# Check if pyenv is available
if ! command -v pyenv &> /dev/null; then
    echo "Installing pyenv for Python version management..."
    curl https://pyenv.run | bash
    
    # Add pyenv to PATH for this session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init - bash)"
    
    echo "Please restart your terminal and run this script again for pyenv to be fully activated."
    echo "Or run: source ~/.bashrc"
    exit 0
fi

# Ensure pyenv is properly loaded
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# Check if Python 3.7.17 is installed
if ! pyenv versions | grep -q "3.7.17"; then
    echo "Installing Python 3.7.17 (required for TensorFlow 1.x)..."
    
    # Install system dependencies for building Python
    echo "Installing Python build dependencies..."
    sudo apt update
    sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
        libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git
    
    pyenv install 3.7.17
fi

# Set Python 3.7.17 as local version
echo "Setting Python 3.7.17 as local version..."
pyenv local 3.7.17

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Using Python: $PYTHON_VERSION"

if [[ ! $PYTHON_VERSION =~ "Python 3.7.17" ]]; then
    echo "Error: Expected Python 3.7.17, but got: $PYTHON_VERSION"
    echo "Please ensure pyenv is properly configured."
    exit 1
fi

# Create virtual environment
VENV_NAME="env"
echo "Creating virtual environment: $VENV_NAME"

if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment already exists. Removing it..."
    rm -rf $VENV_NAME
fi

python -m venv $VENV_NAME

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies in correct order for TensorFlow 1.x compatibility
echo "Installing NumPy (TensorFlow 1.x compatible version)..."
pip install numpy==1.18.5

echo "Installing TensorFlow 1.15.0..."
pip install tensorflow==1.15.0

echo "Fixing protobuf compatibility..."
pip install protobuf==3.20.3

echo "Installing OpenAI Baselines and dependencies..."
pip install git+https://github.com/openai/baselines.git@a6b1bc70f156dc45c0da49be8e80941a88021700

echo "Installing visualization and utility libraries..."
pip install matplotlib==3.5.3 seaborn==0.12.2

echo "Environment setup complete!"
echo ""
echo "Configuration:"
echo "  - Python: 3.7.17 (via pyenv)"
echo "  - TensorFlow: 1.15.0 (original compatibility)"
echo "  - MuJoCo: 2.2 (modern version)"
echo "  - OpenAI Baselines: Original commit for HER"
echo ""
echo "To activate the environment in the future, run:"
echo "    source env/bin/activate"
echo ""
echo "To test the setup, run:"
echo "    python test/test_env_status.py"
echo ""
echo "To deactivate when done:"
echo "    deactivate"