# Setup Instructions

This guide will help you set up the Q-Filter PyTorch implementation for MuJoCo continuous control environments.

## Prerequisites

- Python 3.8 or higher
- Linux, macOS, or Windows with WSL
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/DarwinWhite/q-filter-to-prevent-unwanted-imitation-learning.git
cd q-filter-to-prevent-unwanted-imitation-learning
```

### 2. Create Virtual Environment

```bash
python -m venv pytorch_env
source pytorch_env/bin/activate  # Linux/Mac
# or
pytorch_env\Scripts\activate     # Windows
```

### 3. Install Dependencies

Use the provided setup script for automatic installation:

```bash
bash setup_pytorch_env.sh
```

Or install manually:

```bash
pip install --upgrade pip
pip install -r pytorch_requirements.txt
```

### 4. Verify Installation

Run the test script to ensure everything is installed correctly:

```bash
python test_installation.py
```

All tests should pass with ✅ indicators.

## Directory Structure

```
q-filter-to-prevent-unwanted-imitation-learning/
├── src/                       # Source code
│   ├── algorithms/            # DDPG and related algorithms
│   │   ├── ddpg.py           # Main DDPG implementation
│   │   ├── ddpg_mujoco.py    # MuJoCo adapter for DDPG
│   │   ├── actor_critic.py   # Neural network architectures
│   │   ├── replay_buffer.py  # Experience replay with Q-filter
│   │   ├── her.py            # Hindsight Experience Replay
│   │   ├── rollout_mujoco.py # MuJoCo rollout worker
│   │   └── rollout.py        # Original rollout worker
│   ├── experiment/            # Training and evaluation scripts
│   │   ├── train_mujoco.py   # Main training script
│   │   ├── play.py           # Policy visualization
│   │   ├── plot.py           # Basic plotting utilities
│   │   ├── plot_hprc_results.py # Advanced results visualization
│   │   ├── mujoco_config.py  # MuJoCo-specific configuration
│   │   └── config.py         # General configuration
│   └── utils/                 # Utility functions
│       ├── normalizer.py     # State normalization
│       └── util.py           # Helper functions
├── demo_data/                 # Expert demonstration data
│   ├── halfcheetah_expert_demos.npz
│   ├── hopper_expert_demos.npz
│   └── walker2d_expert_demos.npz
├── test/                      # Test files
├── logs/                      # Training logs (created during training)
├── plots/                     # Generated plots
├── run.sh                     # SLURM batch script for HPC
├── test_installation.py       # Installation verification script
├── setup_pytorch_env.sh       # Automated setup script
└── pytorch_requirements.txt   # Python dependencies
```

## Usage

### Basic Training (Vanilla DDPG)

```bash
python src/experiment/train_mujoco.py \
    --env HalfCheetah-v4 \
    --n_epochs 200 \
    --seed 0
```

### Training with Demonstrations (Imitation Learning)

```bash
python src/experiment/train_mujoco.py \
    --env HalfCheetah-v4 \
    --n_epochs 200 \
    --demo_file demo_data/halfcheetah_expert_demos.npz \
    --bc_loss 1 \
    --q_filter 0 \
    --num_demo 100 \
    --seed 0
```

### Training with Q-Filter

```bash
python src/experiment/train_mujoco.py \
    --env HalfCheetah-v4 \
    --n_epochs 200 \
    --demo_file demo_data/halfcheetah_expert_demos.npz \
    --bc_loss 1 \
    --q_filter 1 \
    --num_demo 100 \
    --seed 0
```

### Visualize Trained Policy

```bash
python src/experiment/play.py logs/HalfCheetah-v4-*/policy_best.pt --render
```

### Generate Plots

```bash
python src/experiment/plot_hprc_results.py --data_dir hprc_logs/logs --output_dir plots
```

## Configuration Options

### Command Line Arguments

- `--env`: Environment name (HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4)
- `--n_epochs`: Number of training epochs (default: 200)
- `--seed`: Random seed for reproducibility (default: 0)
- `--demo_file`: Path to demonstration data file
- `--bc_loss`: Enable behavior cloning loss (0 or 1)
- `--q_filter`: Enable Q-filter (0 or 1)
- `--num_demo`: Number of demonstration episodes to use (default: 100)
- `--n_cycles`: Training cycles per epoch (default: 20)
- `--n_batches`: Batches per cycle (default: 40)
- `--log_name`: Custom name for log directory

### Environment-Specific Parameters

The implementation automatically configures parameters based on the environment:

- **HalfCheetah-v4**: 17D observations, 6D actions
- **Hopper-v4**: 11D observations, 3D actions
- **Walker2d-v4**: 17D observations, 6D actions
- **Ant-v4**: 27D observations, 8D actions

## Running on HPC (SLURM)

The `run.sh` script is configured for SLURM batch systems:

```bash
sbatch run.sh
```

Edit `run.sh` to configure:
- Environment
- Number of epochs
- Demonstration file
- BC loss and Q-filter settings
- Job parameters (time, memory, CPUs)

## Troubleshooting

### MuJoCo Installation Issues

If you encounter MuJoCo-related errors:

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libosmesa6 patchelf

# Verify MuJoCo installation
python -c "import gymnasium as gym; env = gym.make('HalfCheetah-v4'); print('✅ MuJoCo works!')"
```

### Import Errors

If you get import errors, ensure you're running from the repository root:

```bash
cd /path/to/q-filter-to-prevent-unwanted-imitation-learning
python src/experiment/train_mujoco.py --env HalfCheetah-v4
```

### CUDA/GPU Issues

The implementation defaults to CPU. To use GPU:

1. Install PyTorch with CUDA support
2. The code will automatically detect and use available GPUs

### Memory Issues

If you run out of memory:

- Reduce `--batch_size` (default: 1024)
- Reduce `--buffer_size` (default: 1000000)
- Reduce `--num_demo` (default: 100)

## Next Steps

1. **Run experiments**: Try the three training modes (vanilla, IL, IL+QF)
2. **Visualize results**: Use `play.py` to watch trained policies
3. **Analyze performance**: Generate plots with `plot_hprc_results.py`
4. **Experiment**: Try different environments, seeds, and hyperparameters

## Support

For issues or questions:
- Check existing GitHub issues
- Review the main README.md for project background
- Consult the original papers for algorithmic details
