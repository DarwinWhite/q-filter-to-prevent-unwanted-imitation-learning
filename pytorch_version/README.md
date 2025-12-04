# PyTorch Q-Filter Implementation - Standalone Version

This directory contains a complete, self-contained implementation of the Q-Filter for preventing unwanted imitation learning in PyTorch.

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv pytorch_env
   source pytorch_env/bin/activate  # Linux/Mac
   # or
   pytorch_env\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r pytorch_requirements.txt
   ```

   Or use the provided script:
   ```bash
   bash setup_pytorch_env.sh
   ```

## Usage

### Training with Demonstrations + Q-Filter
```bash
python src/experiment/train_mujoco.py \
    --env HalfCheetah-v4 \
    --n_epochs 200 \
    --demo_file demo_data/halfcheetah_expert_demos.npz \
    --bc_loss 1 \
    --q_filter 1 \
    --num_demo 100
```

### Training with Demonstrations (No Q-Filter)
```bash
python src/experiment/train_mujoco.py \
    --env HalfCheetah-v4 \
    --n_epochs 200 \
    --demo_file demo_data/halfcheetah_expert_demos.npz \
    --bc_loss 1 \
    --q_filter 0 \
    --num_demo 100
```

### Vanilla DDPG (No Demonstrations)
```bash
python src/experiment/train_mujoco.py \
    --env HalfCheetah-v4 \
    --n_epochs 200
```

### Test Trained Policy
```bash
python src/experiment/play.py logs/HalfCheetah-v4-YYYY-MM-DD-HH-MM-SS-mmm/policy_best.pt --render
```

### Plot Training Results
```bash
python src/experiment/plot.py logs/HalfCheetah-v4-YYYY-MM-DD-HH-MM-SS-mmm/
```

## Directory Structure
```
pytorch_version/
├── README_STANDALONE.md       # This file
├── pytorch_requirements.txt   # Python dependencies
├── setup_pytorch_env.sh      # Environment setup script
├── demo_data/                 # Expert demonstration data
│   ├── halfcheetah_expert_demos.npz
│   ├── hopper_expert_demos.npz
│   └── walker2d_expert_demos.npz
├── logs/                      # Training logs and saved policies
│   └── HalfCheetah-v4-YYYY-MM-DD-HH-MM-SS-mmm/
│       ├── policy_best.pt
│       ├── policy_latest.pt
│       ├── params.json
│       └── progress.csv
├── src/                       # Source code
│   ├── algorithms/            # DDPG and related algorithms
│   ├── experiment/            # Training and evaluation scripts
│   └── utils/                 # Utility functions
└── test/                      # Test files (optional)
```

## Key Features

- **Complete DDPG Implementation**: PyTorch-based DDPG with modern features
- **Behavioral Cloning Loss**: Learn from expert demonstrations
- **Q-Filter**: Prevent unwanted imitation by filtering low-Q actions
- **MuJoCo Integration**: Support for HalfCheetah, Hopper, Walker2D environments
- **Comprehensive Logging**: TensorBoard-compatible logging and plotting
- **Flexible Configuration**: Easy parameter tuning via command line

## Expected Performance

| Method | HalfCheetah-v4 Returns |
|--------|----------------------|
| Vanilla DDPG | ~0 (struggles to learn) |
| IL + BC Loss | ~800-1400 |
| IL + BC Loss + Q-Filter | ~800-1400 (more stable) |

## Dependencies

All required packages are listed in `pytorch_requirements.txt`. Key dependencies include:
- PyTorch
- Gymnasium (with MuJoCo)
- NumPy
- Click (for CLI)
- Matplotlib (for plotting)