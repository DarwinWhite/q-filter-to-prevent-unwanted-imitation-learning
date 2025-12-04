import os
import sys
import numpy as np
import gymnasium as gym

# Add project root for portable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.algorithms.ddpg import DDPG
from src.algorithms.her import make_sample_her_transitions


DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
}


DEFAULT_PARAMS = {
    # Environment
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # Network
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'src.algorithms.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # Training
    'n_cycles': 20,  # per epoch
    'rollout_batch_size': 1,  # per thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 1024,  # per thread, measured in transitions
    'n_test_rollouts': 10,  # number of test rollouts per epoch
    'test_with_polyak': False,  # run test episodes with the target network
    # Exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # Normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    # Q-filter and demonstrations
    'bc_loss': 1,  # whether to use behavior cloning loss
    'q_filter': 1,  # whether to use q-filter
    'num_demo': 100,  # number of expert demo episodes
    'demo_batch_size': 128,  # number of demo samples per batch
    'prm_loss_weight': 0.001,  # weight for demo loss
    'aux_loss_weight': 0.0078,  # weight for auxiliary losses
    # Device
    'device': 'cpu',  # PyTorch device
}


CACHED_ENVS = {}


def cached_make_env(make_env):
    """Cache environment to avoid repeated creation."""
    if make_env not in CACHED_ENVS:
        CACHED_ENVS[make_env] = make_env()
    return CACHED_ENVS[make_env]


def prepare_params(kwargs):
    """Prepare parameters by merging defaults with provided kwargs."""
    # Default parameters
    default_params = DEFAULT_PARAMS.copy()
    
    # Environment-specific parameters
    env_name = kwargs.get('env_name', '')
    if env_name in DEFAULT_ENV_PARAMS:
        default_params.update(DEFAULT_ENV_PARAMS[env_name])
    
    # Override with provided parameters
    default_params.update(kwargs)
    
    return default_params


def configure_dims(params):
    """Configure input dimensions for the algorithm."""
    env = cached_make_env(params['make_env'])
    
    # Check if environment has goal-based observations
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle new Gymnasium API
        
    if isinstance(obs, dict):
        # Goal-conditioned environment
        dims = {
            'o': obs['observation'].shape[0],
            'u': env.action_space.shape[0],
            'g': obs['desired_goal'].shape[0],
        }
        params['T'] = env._max_episode_steps if hasattr(env, '_max_episode_steps') else 50
    else:
        # Flat observation environment (like MuJoCo)
        dims = {
            'o': obs.shape[0],
            'u': env.action_space.shape[0], 
            'g': 0,  # No goals for flat environments
        }
        params['T'] = 1000  # Default episode length for continuous control
    
    # Information dimension (usually empty)
    dims['info'] = 0
    
    return dims


def configure_her(params):
    """Configure Hindsight Experience Replay."""
    if params['replay_strategy'] == 'none':
        params['replay_k'] = 0
        
    sample_her_transitions = make_sample_her_transitions(
        params['replay_strategy'], 
        params['replay_k'], 
        her_reward_fun
    )
    
    return sample_her_transitions


def her_reward_fun(ag_2, g, info):
    """Reward function for HER (goal-conditioned environments)."""
    # Simple distance-based reward for goal-conditioned tasks
    return -(np.linalg.norm(ag_2 - g, axis=-1, keepdims=True) > 0.05).astype(np.float32)


def configure_ddpg(dims, params):
    """Configure DDPG algorithm."""
    sample_her_transitions = configure_her(params)
    
    # Prepare DDPG parameters
    ddpg_params = {
        'input_dims': dims,
        'T': params['T'],
        'clip_pos_returns': True,
        'clip_return': (1. / (1. - params['gamma'])) if params['gamma'] < 1. else np.inf,
        'subtract_goals': lambda x, y: x - y,  # Simple subtraction for goals
        'sample_transitions': sample_her_transitions,
        'gamma': params['gamma'] if 'gamma' in params else 0.98,
    }
    
    # Add all other DDPG-relevant parameters
    for key in ['buffer_size', 'hidden', 'layers', 'polyak', 'batch_size', 'Q_lr', 'pi_lr',
                'norm_eps', 'norm_clip', 'max_u', 'action_l2', 'clip_obs', 'scope', 
                'relative_goals', 'bc_loss', 'q_filter', 'num_demo', 'demo_batch_size',
                'prm_loss_weight', 'aux_loss_weight', 'device']:
        if key in params:
            ddpg_params[key] = params[key]
    
    return ddpg_params


def log_params(params, logger=None):
    """Log parameters."""
    if logger is None:
        print("Parameters:")
        for key in sorted(params.keys()):
            print(f'  {key}: {params[key]}')
    else:
        for key in sorted(params.keys()):
            logger.info(f'{key}: {params[key]}')


def make_env_function(env_name):
    """Create environment factory function."""
    def _make_env():
        return gym.make(env_name)
    return _make_env