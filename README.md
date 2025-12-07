# Q-Filter to Prevent Unwanted Imitation Learning

A PyTorch implementation of Q-Filter for preventing unwanted imitation learning in continuous control environments, adapted for MuJoCo dense reward tasks.

## Overview

This project implements the Q-filter mechanism from ["Overcoming Exploration in Reinforcement Learning with Demonstrations" (Nair et al.)](https://arxiv.org/pdf/1709.10089.pdf) using PyTorch and modern MuJoCo environments. The implementation bridges goal-conditioned DDPG+HER to dense reward continuous control tasks.

## Quick Start

```bash
# Setup environment
bash setup_pytorch_env.sh
source pytorch_env/bin/activate

# Train with Q-Filter
python src/experiment/train_mujoco.py \
    --env HalfCheetah-v4 \
    --demo_file demo_data/halfcheetah_expert_demos.npz \
    --bc_loss 1 \
    --q_filter 1 \
    --n_epochs 200

# Visualize policy
python src/experiment/play.py logs/HalfCheetah-v4-*/policy_best.pt --render
```

For detailed setup instructions, see **[SETUP.md](SETUP.md)**.

## Key Features

- **PyTorch Implementation**: Modern deep learning framework with clean, maintainable code
- **Q-Filter Mechanism**: Prevents learning from suboptimal demonstrations
- **MuJoCo Integration**: Support for HalfCheetah, Hopper, Walker2d, and Ant environments
- **Comprehensive Evaluation**: Tools for visualization, plotting, and analysis
- **HPC Support**: SLURM batch scripts for cluster computing


## Documentation

- **[SETUP.md](SETUP.md)**: Detailed installation and usage instructions
- **[Original Paper](https://arxiv.org/pdf/1709.10089.pdf)**: Theoretical foundation
- **Code Documentation**: Inline comments and docstrings throughout

## Project Structure

```
├── src/                       # Source code
│   ├── algorithms/            # DDPG, HER, Q-filter implementations
│   ├── experiment/            # Training, evaluation, and plotting scripts
│   └── utils/                 # Helper functions and normalizers
├── demo_data/                 # Expert demonstrations (.npz files)
├── test/                      # Test files
├── SETUP.md                   # Setup instructions
└── run.sh                     # SLURM batch script
```