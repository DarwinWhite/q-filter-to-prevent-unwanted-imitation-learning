import os
import sys
import numpy as np
import gymnasium as gym

# Add project root for portable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.algorithms.ddpg import DDPG
from src.algorithms.ddpg_mujoco import DDPGMuJoCo

# Environment caching for efficiency
CACHED_ENVS = {}

def cached_make_env(make_env):
    """Cache environment to avoid repeated creation."""
    if make_env not in CACHED_ENVS:
        CACHED_ENVS[make_env] = make_env()
    return CACHED_ENVS[make_env]

# MuJoCo environment parameters - these are for continuous control with dense rewards
DEFAULT_MUJOCO_ENV_PARAMS = {
    'HalfCheetah-v4': {
        'n_cycles': 20,  # training cycles per epoch
    },
    'Ant-v4': {
        'n_cycles': 20,
    },
    'Hopper-v4': {
        'n_cycles': 20,
    },
    'Walker2d-v4': {
        'n_cycles': 20,
    },
}

# MuJoCo-specific parameters - adapted from original but without goal-conditioning
DEFAULT_MUJOCO_PARAMS = {
    # Environment
    'max_u': 1.,  # max absolute value of actions on different coordinates
    'T': 1000,  # horizon length (episode length)
    # Network
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient (changed from 0.8)
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',
    'relative_goals': False,
    # Training
    'n_cycles': 20,  # per epoch
    'rollout_batch_size': 1,  # per process
    'n_batches': 40,  # training batches per cycle  
    'batch_size': 1024,  # measured in transitions
    'n_test_rollouts': 10,  # number of test rollouts per epoch
    'test_with_polyak': False,  # run test episodes with the target network
    # Exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions
    # HER (disabled for MuJoCo dense rewards)
    'replay_strategy': 'none',  # no HER for dense reward environments
    'replay_k': 0,  # no additional goals
    # Normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    # Q-filter and demonstrations
    'bc_loss': 0,  # whether to use behavior cloning loss
    'q_filter': 0,  # whether to use q-filter
    'num_demo': 100,  # number of expert demo episodes
    'demo_batch_size': 128,  # number of demo samples per batch
    'prm_loss_weight': 0.001,  # weight for demo loss
    'aux_loss_weight': 0.0078,  # weight for auxiliary losses
    # PyTorch specific
    'device': 'cpu',  # PyTorch device
    'gamma': 0.99,  # discount factor for MuJoCo
}


def configure_mujoco_dims(params):
    """Configure dimensions for MuJoCo environments (flat state observations)."""
    env = cached_make_env(params['make_env'])
    
    # Reset to get observation dimensions
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle new Gymnasium API

    # MuJoCo environments return flat observations directly
    if isinstance(obs, dict):
        raise ValueError(f"Expected flat observations for MuJoCo, got dict: {obs.keys()}")
    
    dims = {
        'o': obs.shape[0],
        'u': env.action_space.shape[0],
        'g': 1,  # Minimal goal dimension for compatibility with goal-conditioned DDPG
        'info': 0,  # No additional info
    }
    
    # Set episode length for continuous control
    params['T'] = getattr(env, '_max_episode_steps', 1000)
    
    return dims


def prepare_mujoco_params(env_name, **kwargs):
    """Prepare parameters for MuJoCo environments."""
    # Start with MuJoCo defaults
    params = DEFAULT_MUJOCO_PARAMS.copy()
    
    # Add environment-specific parameters
    if env_name in DEFAULT_MUJOCO_ENV_PARAMS:
        params.update(DEFAULT_MUJOCO_ENV_PARAMS[env_name])
    
    # Add environment name and factory function
    params['env_name'] = env_name
    
    # Create environment factory with optional render mode
    def make_env_with_render(render_mode=None):
        import warnings
        # Suppress GLFW and rendering warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", message=".*GLFW.*")
            if render_mode:
                env = gym.make(env_name, render_mode=render_mode)
            else:
                env = gym.make(env_name)
        return env
    
    params['make_env'] = make_env_with_render
    
    # Override with any provided parameters
    params.update(kwargs)
    
    return params


def configure_mujoco_ddpg(dims, params):
    """Configure DDPG for MuJoCo environments."""
    from src.algorithms.her import make_sample_her_transitions
    
    # Simple reward function for dense rewards (no HER)
    def mujoco_reward_fun(ag_2, g, info):
        # For MuJoCo, rewards come from environment - return zeros for HER compatibility
        return np.zeros((ag_2.shape[0], 1))
    
    # Configure HER (disabled for MuJoCo)
    sample_her_transitions = make_sample_her_transitions(
        params['replay_strategy'], 
        params['replay_k'], 
        mujoco_reward_fun
    )
    
    # Prepare DDPG parameters (don't include input_dims here since it's passed separately)
    ddpg_params = {
        'T': params['T'],
        'clip_pos_returns': True,
        'clip_return': (1. / (1. - params['gamma'])) if params['gamma'] < 1. else np.inf,
        'subtract_goals': lambda x, y: x - y,  # Simple subtraction for dummy goals
        'sample_transitions': sample_her_transitions,
        'rollout_batch_size': params['rollout_batch_size'],
    }
    
    # Add all relevant DDPG parameters
    ddpg_keys = ['buffer_size', 'hidden', 'layers', 'polyak', 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u', 'action_l2', 'clip_obs', 'scope',
                 'relative_goals', 'bc_loss', 'q_filter', 'num_demo', 'demo_batch_size',
                 'prm_loss_weight', 'aux_loss_weight', 'device', 'gamma']
    
    for key in ddpg_keys:
        if key in params:
            ddpg_params[key] = params[key]
    
    return ddpg_params


def create_mujoco_ddpg(dims, params):
    """Create DDPG instance for MuJoCo environments."""
    ddpg_params = configure_mujoco_ddpg(dims, params)
    
    # Use MuJoCo adapter to handle flat state observations
    policy = DDPGMuJoCo(input_dims=dims, **ddpg_params)
    
    return policy


def log_mujoco_params(params, logger=None):
    """Log MuJoCo parameters."""
    if logger is None:
        print("MuJoCo DDPG Parameters:")
        for key in sorted(params.keys()):
            if not callable(params[key]):  # Skip function objects
                print(f'  {key}: {params[key]}')
    else:
        logger.info("MuJoCo DDPG Parameters:")
        for key in sorted(params.keys()):
            if not callable(params[key]):  # Skip function objects
                logger.info(f'{key}: {params[key]}')


# Environment dimension mapping for quick reference
MUJOCO_ENV_DIMS = {
    'HalfCheetah-v4': {'obs': 17, 'action': 6},
    'Hopper-v4': {'obs': 11, 'action': 3},
    'Walker2d-v4': {'obs': 17, 'action': 6},
    'Ant-v4': {'obs': 27, 'action': 8},
    'Humanoid-v4': {'obs': 376, 'action': 17},
}


def get_mujoco_dims(env_name):
    """Get dimensions for common MuJoCo environments."""
    if env_name in MUJOCO_ENV_DIMS:
        return MUJOCO_ENV_DIMS[env_name]
    else:
        # Query environment directly if not in predefined list
        env = gym.make(env_name)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return {'obs': obs.shape[0], 'action': env.action_space.shape[0]}