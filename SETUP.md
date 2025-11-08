# Environment Setup Guide

## Quick Setup (Recommended)

Run the automated setup script:
```bash
./setup_env.sh
```

This will create a virtual environment called `env` and install all dependencies.

## Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Create Virtual Environment
```bash
python3 -m venv env
source env/bin/activate
```

### 2. Upgrade pip
```bash
pip install --upgrade pip
```

### 3. Install Dependencies (in order)

#### Core Dependencies
```bash
# NumPy (specific version for TensorFlow compatibility)
pip install numpy==1.21.0

# TensorFlow (version depends on Python version)
# For Python 3.8+: Use TensorFlow 2.x with v1 compatibility
pip install tensorflow==2.13.1

# PyTorch for demo generation scripts
pip install torch==1.12.1 --index-url https://download.pytorch.org/whl/cpu
```

#### Gym and MuJoCo
```bash
# Gym (modern version)
pip install gym==0.26.2
```

#### Other Requirements
```bash
# Visualization and utilities
pip install matplotlib==3.5.3
pip install seaborn==0.11.2
pip install click==8.1.3
pip install glob2==0.7

# OpenAI Baselines (specific commit - this also installs modern mujoco package)
pip install git+https://github.com/openai/baselines.git@a6b1bc70f156dc45c0da49be8e80941a88021700
```

### 4. Install TensorFlow Compatibility Helper
The setup script automatically creates `src/utils/tf_compat.py` for TensorFlow 1.x/2.x compatibility.

## Using the Environment

### Activate
```bash
source env/bin/activate
```

### Test Setup
```bash
python test/test_env_status.py          # Quick status check  
python test/test_policy_loading.py # Detailed validation
```

### Deactivate
```bash
deactivate
```

## Environment Status

### Working Environment Versions
- **HalfCheetah-v4**: obs_dim=17, action_dim=6
- **Hopper-v4**: obs_dim=11, action_dim=3
- **Walker2d-v4**: obs_dim=17, action_dim=6

### Generated Helper Scripts
1. **`src/utils/tf_compat.py`**: TensorFlow compatibility layer for TF 1.x code on TF 2.x
2. **`test/test_policy_loading.py`**: Validates policy loading and environment compatibility
3. **`src/utils/generate_demos.py`**: Converts .pkl policy files to .npz demonstration trajectories
4. **`test/test_env_status.py`**: Quick environment status check

## Important Notes

### Dependencies Intentionally Skipped
- **mujoco-py**: Requires separate MuJoCo 2.1.0 system installation. We use the modern `mujoco` package instead.
- **mpi4py**: Requires system MPI libraries. Only needed for distributed training, which is optional.

### TensorFlow Compatibility
- This project was originally written for TensorFlow 1.x
- For Python 3.8+, we use TensorFlow 2.x with v1 compatibility mode
- The `tf_compat.py` file handles this automatically

### Key Compatibility Issues Resolved
1. **TensorFlow Version**: Use TF 2.x with v1 compatibility mode for Python 3.8+
2. **MuJoCo Installation**: Use modern `mujoco` package instead of legacy `mujoco-py`
3. **Gym API**: Handle both old and new gym `reset()` and `step()` return signatures
4. **Dependency Conflicts**: Skip problematic system-dependent packages

## Troubleshooting

### TensorFlow Issues
- If you get warnings about deprecated features, that's expected for this older codebase
- The compatibility layer in `src/utils/tf_compat.py` handles TF 1.x/2.x differences

### OpenAI Baselines Issues
- The specific commit is required for compatibility
- If installation fails, ensure all previous dependencies are installed first

### Missing System Dependencies
If you encounter issues, you may need:
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

## Alternative: Using Python 3.7
For exact compatibility with the original codebase, you can use Python 3.7:
```bash
# Install Python 3.7 if not available
sudo apt-get install python3.7 python3.7-venv

# Create environment with Python 3.7
python3.7 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install numpy==1.21.0
pip install tensorflow==1.15.0  # Original TF 1.x works with Python 3.7
# ... continue with other dependencies
```

## Next Steps

1. **Generate Demonstration Data**: Run `python src/utils/generate_demos.py` to create demo files
2. **Adapt Experiment Code**: Modify `src/experiment/config.py` to use MuJoCo environments  
3. **Test Q-Filter**: Run baseline experiments comparing DDPG vs DDPG+demos vs DDPG+demos+Q-filter
4. **Research Extension**: Evaluate Q-filter effectiveness in dense reward settings