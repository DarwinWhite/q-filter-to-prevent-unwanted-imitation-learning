# src/experiment/mujoco_config.py (patched for lazy DDPG imports and TF-free runs)
import numpy as np
import gym
import sys
import os

# Add project root for portable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Do NOT import DDPG at module import time (it may import TensorFlow).
# Lazy-import it inside configure_mujoco_ddpg to avoid TF initialization on HPRC.

# Environment caching for efficiency
CACHED_ENVS = {}

def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]

# MuJoCo environment parameters - these are for continuous control with dense rewards
DEFAULT_MUJOCO_ENV_PARAMS = {
    'HalfCheetah-v4': {
        'n_cycles': 20,  # Reduced for faster experimentation (was 50)
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
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E5),  # for experience replay
    'polyak': 0.8,  # polyak averaging coefficient
    'action_l2': 0.1,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,  # Not applicable for MuJoCo - no goals
    # training
    'n_cycles': 3,  # per epoch - higher for dense rewards
    'rollout_batch_size': 1,  # per mpi thread
    'n_batches': 20,  # training batches per cycle
    'batch_size': 1024,  # per mpi thread, measured in transitions
    'n_test_rollouts': 1,  # number of test rollouts per epoch
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.2,  # percentage of time a random action is taken
    'noise_eps': 0.1,  # std of gaussian noise added to actions as percentage of max_u
    # HER - disabled for MuJoCo dense reward environments
    'replay_strategy': 'none',  # No HER needed with dense rewards
    'replay_k': 0,  # No additional goals for replay
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    'bc_loss': 0,  # disabled for basic MuJoCo training (enable with demo data)
    'q_filter': 0,  # disabled for basic MuJoCo training (enable with demo data)
    'num_demo': 0  # no demos for basic MuJoCo training
}

CACHED_MUJOCO_ENVS = {}

def cached_make_mujoco_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_MUJOCO_ENVS:
        env = make_env()
        CACHED_MUJOCO_ENVS[make_env] = env
    return CACHED_MUJOCO_ENVS[make_env]

def prepare_mujoco_params(kwargs):
    """
    Prepare parameters for MuJoCo environments - adapted from original prepare_params
    but simplified for flat state observations (no goal conditioning)
    """
    # DDPG params
    ddpg_params = dict()

    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)
    
    kwargs['make_env'] = make_env
    tmp_env = cached_make_mujoco_env(kwargs['make_env'])
    
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps
    
    # Reset to get observation and action space info
    tmp_env.reset()
    
    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    
    # Extract DDPG parameters
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    
    kwargs['ddpg_params'] = ddpg_params

    return kwargs

def configure_mujoco_dims(params):
    """
    Configure dimensions for MuJoCo environments - flat state observations only
    No goals or achieved_goals like the original goal-conditioned version
    """
    env = cached_make_mujoco_env(params['make_env'])
    
    # Handle both old and new gym API
    obs_result = env.reset()
    if isinstance(obs_result, tuple):
        obs, info = obs_result  # New gym API
    else:
        obs = obs_result  # Old gym API
    
    # For MuJoCo environments, obs is directly the observation array (flat state)
    # Not a dictionary like {'observation': ..., 'achieved_goal': ..., 'desired_goal': ...}
    
    dims = {
        'o': obs.shape[0],  # observation dimension (e.g., 17 for HalfCheetah-v4)
        'u': env.action_space.shape[0],  # action dimension (e.g., 6 for HalfCheetah-v4)
        'g': 0,  # No goals in MuJoCo - set to 0
        'r': 1,  # Reward dimension (scalar reward)
    }
    
    # Note: No info handling needed for basic MuJoCo environments
    # They don't return structured info like goal-conditioned environments
    
    return dims

def simple_mujoco_sample_transitions(episode_batch, batch_size_in_transitions):
    """
    Simple transition sampler for MuJoCo (no HER).
    Just extracts all transitions from the episode batch.
    """
    T = episode_batch['u'].shape[1]
    rollout_batch_size = episode_batch['u'].shape[0]
    
    # Extract transitions in batch format 
    transitions = {}
    for key in episode_batch.keys():
        if key == 'o':
            # Split observations into o and o_2 (current and next state)
            transitions['o'] = episode_batch[key][:, :-1].reshape(-1, *episode_batch[key].shape[2:])
            transitions['o_2'] = episode_batch[key][:, 1:].reshape(-1, *episode_batch[key].shape[2:])
        elif key == 'ag':
            # Split achieved goals into ag and ag_2
            transitions['ag'] = episode_batch[key][:, :-1].reshape(-1, *episode_batch[key].shape[2:])
            transitions['ag_2'] = episode_batch[key][:, 1:].reshape(-1, *episode_batch[key].shape[2:])
        elif key in ['u', 'r', 'g']:
            # Actions, rewards, and goals have T timesteps (not T+1)
            transitions[key] = episode_batch[key].reshape(-1, *episode_batch[key].shape[2:])
        elif key in ['o_2', 'ag_2']:
            # Skip these - they're created by the replay buffer
            pass
    
    return transitions

def simple_mujoco_subtract(a, b):
    """Dummy subtract function for MuJoCo (goals not used)"""
    return a - b

def sample_her_transitions():
    """Function that creates the appropriate HER sampler"""
    # For MuJoCo without HER, return a simple transition sampler
    return simple_mujoco_sample_transitions

# Configure MuJoCo-specific parameters  
def prepare_mujoco_params(kwargs):
    """
    Prepare parameters for MuJoCo environment training.
    """
    # Basic environment setup
    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)
    
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps
    tmp_env.reset()
    
    # Compute gamma based on episode length
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    
    # Get observation and action dimensions for MuJoCo  
    obs_dims = tmp_env.observation_space.shape[0]
    action_dims = tmp_env.action_space.shape[0]
    
    # MuJoCo environments don't use goals, so set goal dimension to 0
    kwargs['goal_dim'] = 0
    
    return kwargs

def configure_mujoco_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=True):
    """
    Configure DDPG for MuJoCo environments - no HER since we have dense rewards
    """
    # Lazy import: prefer a PyTorch DDPG implementation if present, otherwise try TF one.
    # This avoids importing TensorFlow at module import time (which fails on HPRC).
    ddpg_impl = None
    ddpg_module_names_tried = []
    try:
        # Preferred name for a PyTorch implementation (you can create this file)
        ddpg_module_names_tried.append("src.algorithms.ddpg_torch")
        ddpg_impl_mod = __import__("src.algorithms.ddpg_torch", fromlist=["DDPG"])
        ddpg_impl = getattr(ddpg_impl_mod, "DDPG")
    except Exception:
        # fallback: try the legacy ddpg (may import TF and fail)
        try:
            ddpg_module_names_tried.append("src.algorithms.ddpg")
            ddpg_impl_mod = __import__("src.algorithms.ddpg", fromlist=["DDPG"])
            ddpg_impl = getattr(ddpg_impl_mod, "DDPG")
        except Exception as e:
            msg = ("Could not import a DDPG implementation. "
                   "Tried modules: {}. Import error: {}. \n\n"
                   "If you have a PyTorch DDPG implementation, place it at "
                   "'src/algorithms/ddpg_torch.py' exposing class DDPG. "
                   "If you only have the old TensorFlow-based DDPG, it will fail on this system "
                   "because TensorFlow and protobuf are not compatible with the current environment. "
                   "Please install a PyTorch DDPG or ensure TensorFlow/protobuf compatibility.")
            raise ImportError(msg.format(ddpg_module_names_tried, e))

    # Extract relevant parameters
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = {}
    for key in ['buffer_size', 'hidden', 'layers', 'network_class', 'polyak',
                'batch_size', 'Q_lr', 'pi_lr', 'norm_eps', 'norm_clip', 'max_u',
                'action_l2', 'clip_obs', 'scope', 'relative_goals']:
        if key in params:
            ddpg_params[key] = params[key]

    input_dims = dims.copy()

    # DDPG agent configuration (the constructor signature expected mirrors the
    # previous configure_mujoco_ddpg usage; adjust your ddpg_torch.DDPG to accept these)
    ddpg_params.update({
        'input_dims': input_dims,  # agent takes flat state observations
        'T': params['T'],
        'clip_pos_returns': True,  # clip positive returns
        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
        'rollout_batch_size': rollout_batch_size,
        'subtract_goals': simple_mujoco_subtract,
        'sample_transitions': simple_mujoco_sample_transitions,
        'gamma': gamma,
        'bc_loss': params.get('bc_loss', 0),
        'q_filter': params.get('q_filter', 0),
        'num_demo': params.get('num_demo', 0),
    })

    ddpg_params['info'] = {'env_name': params.get('env_name', 'unknown')}

    # Create policy instance using the discovered implementation
    policy = ddpg_impl(**ddpg_params, use_mpi=use_mpi)
    return policy

def log_mujoco_params(params, logger=None):
    """Log MuJoCo parameters"""
    if logger is None:
        # fallback to simple print
        for key in sorted(params.keys()):
            print(f'{key}: {params[key]}')
    else:
        for key in sorted(params.keys()):
            try:
                logger.info('{}: {}'.format(key, params[key]))
            except Exception:
                pass

# Environment-specific configurations (unchanged)
HALFCHEETAH_CONFIG = {
    'env_name': 'HalfCheetah-v4',
    'observation_dim': 17,  # HalfCheetah state space
    'action_dim': 6,        # HalfCheetah action space
    'max_episode_steps': 1000,
    'reward_type': 'dense',  # MuJoCo provides dense rewards
}

ANT_CONFIG = {
    'env_name': 'Ant-v4',
    'observation_dim': 27,  # Ant state space
    'action_dim': 8,        # Ant action space
    'max_episode_steps': 1000,
    'reward_type': 'dense',
}

HOPPER_CONFIG = {
    'env_name': 'Hopper-v4',
    'observation_dim': 11,  # Hopper state space
    'action_dim': 3,        # Hopper action space
    'max_episode_steps': 1000,
    'reward_type': 'dense',
}

WALKER2D_CONFIG = {
    'env_name': 'Walker2d-v4',
    'observation_dim': 17,  # Walker2d state space
    'action_dim': 6,        # Walker2d action space
    'max_episode_steps': 1000,
    'reward_type': 'dense',
}
