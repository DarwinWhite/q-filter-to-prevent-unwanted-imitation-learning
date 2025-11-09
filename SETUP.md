# Environment Setup Guide

## Quick Setup (Recommended)

Run the automated setup script:
```bash
./setup_env.sh
```

This will:
- Install Python 3.7.17 via pyenv (required for TensorFlow 1.x)
- Create a virtual environment with original dependencies
- Install TensorFlow 1.15.0, OpenAI Baselines, and all compatible packages

## Current Environment Configuration

### Core Dependencies
- **Python**: 3.7.17 (via pyenv)
- **TensorFlow**: 1.15.0 (original version)
- **MuJoCo**: 2.2.0 (modern version, backward compatible)
- **OpenAI Baselines**: Original commit (a6b1bc70...)
- **Gym**: 0.26.2 (working with deprecation warnings)

### Key Packages
```
numpy==1.18.5              # TensorFlow 1.x compatible
tensorflow==1.15.0         # Original version  
protobuf==3.20.3           # Compatibility fix
mujoco==2.2                # Modern MuJoCo
gym==0.26.2               # Gym with MuJoCo support
matplotlib==3.5.3          # Plotting
mpi4py==4.0.3             # Distributed computing
```

## Using the Environment

### Activate
```bash
source env/bin/activate
```

### Test Setup
```bash
python test/test_env_status.py          # Quick status check
python test/test_policy_loading.py      # Detailed validation
```

### Deactivate
```bash
deactivate
```

## Validation Results

All critical components tested and working:

1. **TensorFlow 1.x**: Session-based API works correctly
2. **MuJoCo Environment**: HalfCheetah-v4 loads and runs (17D obs, 6D actions)
3. **OpenAI Baselines**: HER module imports successfully
4. **Episode Test**: Random policy completes 100 steps successfully

## Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Install Python 3.7
```bash
# Install pyenv if not available
curl https://pyenv.run | bash

# Install Python 3.7.17
pyenv install 3.7.17
pyenv local 3.7.17
```

### 2. Create Virtual Environment
```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

### 3. Install Dependencies (in order)
```bash
# Core dependencies
pip install numpy==1.18.5
pip install tensorflow==1.15.0
pip install protobuf==3.20.3

# OpenAI Baselines (includes MuJoCo, Gym, etc.)
pip install git+https://github.com/openai/baselines.git@a6b1bc70f156dc45c0da49be8e80941a88021700

# Visualization libraries
pip install matplotlib==3.5.3 seaborn==0.12.2
```

## Supported Environments

### Working Environment Versions
- **HalfCheetah-v4**: obs_dim=17, action_dim=6
- **Hopper-v4**: obs_dim=11, action_dim=3  
- **Walker2d-v4**: obs_dim=17, action_dim=6

## Important Notes

### Why Python 3.7 + TensorFlow 1.x?
- **Maximum Compatibility**: Original Q-filter codebase uses TF 1.x APIs
- **Reliable Reproduction**: Same dependency stack as original research
- **Easier Debugging**: Original session-based TensorFlow APIs work as expected
- **Better Stability**: Well-tested dependency combinations

### Expected Warnings
- Gym deprecation warnings - expected and doesn't affect functionality
- CUDA/NUMA warnings - expected for CPU-only TensorFlow setup

### Dependencies Intentionally Updated
- **MuJoCo**: Using modern `mujoco==2.2` instead of legacy `mujoco-py`
- **Protobuf**: Downgraded to 3.20.3 for TensorFlow 1.x compatibility
- **NumPy**: Using 1.18.5 for TensorFlow 1.x compatibility

## Troubleshooting

### TensorFlow Issues
If you encounter TensorFlow import errors, ensure protobuf version is correct:
```bash
pip install protobuf==3.20.3
```

### OpenAI Baselines Issues
- The specific commit is required for compatibility
- Ensure system dependencies are installed (handled by `setup_env.sh`)

### Missing System Dependencies
If manual setup fails, install build dependencies:
```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git
```

## Next Steps

**Environment Ready** - Proceed with Phase 1: MuJoCo environment adaptation

1. Adapt original Q-filter code to work with MuJoCo environments
2. Modify goal-conditioned structure to flat state observations  
3. Update training scripts for dense reward settings
4. Test DDPG training on HalfCheetah-v4

## Quick Commands Reference

```bash
# Setup (first time)
./setup_env.sh

# Activate environment
source env/bin/activate

# Test environment
python test/test_env_status.py

# Start Q-filter research
# Ready for Phase 1 implementation!
```