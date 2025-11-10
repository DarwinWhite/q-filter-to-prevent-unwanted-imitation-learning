# Source Code Organization

This directory contains the main Q-Filter implementation organized into logical modules.

## Structure

### `algorithms/` - Core RL Algorithms
- `ddpg.py` - Deep Deterministic Policy Gradient implementation
- `her.py` - Hindsight Experience Replay implementation  
- `actor_critic.py` - Actor-Critic network architectures
- `replay_buffer.py` - Experience replay buffer with Q-filter enhancements
- `rollout.py` - Environment rollout and trajectory collection utilities

### `utils/` - Utilities and Support Modules
- `tf_compat.py` - TensorFlow 1.x/2.x compatibility layer
- `generate_demos.py` - Convert .pkl parameter files to .npz demonstrations (TensorFlow-based)
- `normalizer.py` - State and action normalization utilities  
- `util.py` - General utility functions and helpers

### `experiment/` - Experiment Framework
- Configuration and execution scripts for running Q-Filter experiments
- Originally from OpenAI Baselines HER implementation

### `data_generation/` - Data Processing
- Data generation and processing modules for demonstrations and trajectories

## Usage

Import modules using the new structure:

```python
# Import core algorithms
from src.algorithms import ddpg, her, actor_critic

# Import utilities
from src.utils import generate_demos, normalizer

# Use TensorFlow compatibility
import src.utils.tf_compat
import tensorflow as tf
```

## Original Structure

This reorganization maintains all original functionality while providing better:
- **Modularity**: Clear separation of concerns
- **Maintainability**: Easier to find and modify specific components  
- **Extensibility**: Simple to add new algorithms or utilities
- **Testing**: Isolated testing of individual components