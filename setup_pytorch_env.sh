#!/bin/bash
# Setup script for PyTorch Q-filter environment with Python 3.8+
# This script creates a modern PyTorch environment for HPRC compatibility

set -e  # Exit on error

echo "Setting up PyTorch Q-filter environment with Python 3.8+..."

# Check if pyenv is available
if ! command -v pyenv &> /dev/null; then
    echo "Installing pyenv..."
    echo "Run this manually first:"
    echo "  curl https://pyenv.run | bash"
    echo "  # Add to ~/.bashrc:"
    echo "  export PYENV_ROOT=\"\$HOME/.pyenv\""
    echo "  export PATH=\"\$PYENV_ROOT/bin:\$PATH\""
    echo "  eval \"\$(pyenv init - bash)\""
    exit 0
fi

# Ensure pyenv is properly loaded
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# Check if Python 3.8.18 is installed
if ! pyenv versions | grep -q "3.8.18"; then
    echo "Installing Python 3.8.18..."
    echo "This may take several minutes..."
    
    pyenv install 3.8.18
fi

# Set Python 3.8.18 as local version
echo "Setting Python 3.8.18 as local version..."
pyenv local 3.8.18

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Using Python: $PYTHON_VERSION"

if [[ ! $PYTHON_VERSION =~ "Python 3.8.18" ]]; then
    echo "❌ Error: Expected Python 3.8.18, got $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
VENV_NAME="pytorch_env"
echo "Creating virtual environment: $VENV_NAME"

if [ -d "$VENV_NAME" ]; then
    echo "Removing existing pytorch_env..."
    rm -rf $VENV_NAME
fi

python -m venv $VENV_NAME

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch dependencies in correct order
echo "Installing PyTorch (CPU version for compatibility)..."
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

echo "Installing core dependencies..."
pip install -r pytorch_requirements.txt

echo "PyTorch environment setup complete!"
echo ""
echo "Configuration:"
echo "  - Python: 3.8.18 (via pyenv)"
echo "  - PyTorch: 2.1.0+cpu (HPRC compatible)"
echo "  - Gymnasium: 0.29+ (modern Gym)"
echo "  - MuJoCo: 2.3+ (latest version)"
echo ""
echo "To use this environment:"
echo "  source pytorch_env/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"