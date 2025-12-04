# Q-Filter PyTorch Implementation

This directory contains the PyTorch adaptation of the Q-Filter algorithm for learning from demonstrations in continuous control tasks.

## Key Features

- **Modern PyTorch Implementation**: Compatible with Python 3.8+ and HPRC environments
- **Preserved Functionality**: Exact algorithmic behavior as TensorFlow version
- **MuJoCo Integration**: Full compatibility with continuous control environments
- **No Legacy Dependencies**: Removed TensorFlow 1.x, MPI, and OpenAI Baselines dependencies

## Quick Start

### Environment Setup
```bash
# Create PyTorch environment
./setup_pytorch_env.sh
source pytorch_env/bin/activate
```

### Train MuJoCo Agent
```bash
cd src/experiment
python train_mujoco.py --env HalfCheetah-v4 --n_epochs 200
```

### Test PyTorch Integration
```bash
python test/test_pytorch_integration.py
```

## Structure

### `src/algorithms/` - Core RL Algorithms
- `ddpg.py` - PyTorch DDPG implementation
- `her.py` - Hindsight Experience Replay  
- `actor_critic.py` - Neural network architectures
- `replay_buffer.py` - Experience replay buffer
- `rollout.py` - Environment rollout utilities

### `src/utils/` - Utilities and Support Modules
- `normalizer.py` - State/action normalization (no MPI)
- `util.py` - General utility functions
- `generate_demos.py` - Demo generation utilities

### `src/experiment/` - Experiment Framework
- `train_mujoco.py` - MuJoCo training script
- `config.py` - Configuration system
- `play.py` - Policy evaluation
- `plot.py` - Results visualization

### `test/` - Testing and Validation
- `test_pytorch_integration.py` - Basic functionality tests
- `test_equivalence.py` - TensorFlow vs PyTorch comparison

## Migration from TensorFlow Version

This implementation preserves:
- ✅ Exact DDPG algorithm behavior
- ✅ MuJoCo environment compatibility  
- ✅ Q-Filter demonstration learning
- ✅ Command-line interface
- ✅ Training convergence properties

Key improvements:
- 🚀 Modern PyTorch architecture
- 🔧 HPRC compatibility (Python 3.8+)
- 💾 Simplified dependencies
- 🧹 Removed legacy MPI/Baselines code

## Dependencies

- Python 3.8+
- PyTorch 2.1+
- Gymnasium (modern Gym)
- MuJoCo 2.3+
- See `pytorch_requirements.txt` for full list

## Usage

Same interface as TensorFlow version:

```bash
# DDPG training
python train_mujoco.py --env HalfCheetah-v4 --n_epochs 200

# DDPG + Behavior Cloning
python train_mujoco.py --env HalfCheetah-v4 --bc_loss 1 --demo_file ../../demo_data/halfcheetah_expert_demos.npz

# DDPG + BC + Q-Filter  
python train_mujoco.py --env HalfCheetah-v4 --bc_loss 1 --q_filter 1 --demo_file ../../demo_data/halfcheetah_expert_demos.npz
```