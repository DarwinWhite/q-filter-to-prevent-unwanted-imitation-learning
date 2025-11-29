import numpy as np
import gym
import os
import sys
import warnings

# Add project root for portable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Optional gazebo import (keep optional; many users won't have gym_gazebo installed)
try:
    import gym_gazebo  # noqa: F401
except Exception:
    # not required for MuJoCo experiments
    pass

# OpenAI Baselines logger is convenient, but provide a fallback if unavailable
try:
    from baselines import logger
except Exception:
    class _SimpleLogger:
        def info(self, msg): print(msg)
        def warn(self, msg): print("WARN:", msg)
    logger = _SimpleLogger()

# DDPG implementation (PyTorch version you added)
from src.algorithms.ddpg import DDPG

# Try to import HER sampler; if unavailable, we'll treat HER as disabled
try:
    from src.algorithms.her import make_sample_her_transitions  # optional
except Exception:
    make_sample_her_transitions = None

DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
    'GazeboWAMemptyEnv-v2': {
        'n_cycles': 20,
    },
}

DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.8,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 20,  # per epoch
    'rollout_batch_size': 1,  # per mpi thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 1024,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.2,  # percentage of time a random action is taken
    'noise_eps': 0.1,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    'bc_loss': 1, # whether or not to use the behavior cloning loss as an auxilliary loss
    'q_filter': 1, # whether or not a Q value filter should be used on the Actor outputs
    'num_demo': 100 # number of expert demo episodes
}

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


def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()

    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps
    # warm-up reset (handles new and old gym API)
    try:
        reset_res = tmp_env.reset()
        if isinstance(reset_res, tuple):
            _obs, _info = reset_res
        else:
            _obs = reset_res
    except Exception:
        tmp_env.reset()

    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
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


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    """
    Configure HER sampling function if HER is enabled and the helper is available.

    Returns:
      sample_her_transitions function or None
    """
    # HER disabled explicitly
    if params.get('replay_strategy', 'future') == 'none':
        return None

    # If HER helper is not present, skip
    if make_sample_her_transitions is None:
        logger.warn("HER sampler not available (make_sample_her_transitions import failed). Disabling HER.")
        return None

    # Try to use env.compute_reward if present, otherwise disable HER.
    env = cached_make_env(params['make_env'])
    env.reset()
    if not hasattr(env, 'compute_reward'):
        logger.warn("Environment has no compute_reward function. Disabling HER.")
        return None

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    her_params = {
        'reward_fun': reward_fun,
        'replay_strategy': params.get('replay_strategy', 'future'),
        'replay_k': params.get('replay_k', 4)
    }

    # Return the HER sampler
    return make_sample_her_transitions(**her_params)


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def configure_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=True):
    """
    Prepare parameters and construct a DDPG agent instance.
    """
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']

    input_dims = dims.copy()

    # DDPG agent
    env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        'bc_loss': params.get('bc_loss', 0),
                        'q_filter': params.get('q_filter', 0),
                        'num_demo': params.get('num_demo', 0),
                        })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)
    return policy


def configure_dims(params):
    """
    Determine dimensions for the environment. Works with both dict-observation (goal-conditioned)
    and flat-observation (MuJoCo) environments, and is robust to both old and new Gym APIs.
    """
    env = cached_make_env(params['make_env'])
    # ensure env is usable
    try:
        reset_res = env.reset()
        if isinstance(reset_res, tuple):
            obs, _ = reset_res
        else:
            obs = reset_res
    except Exception:
        # If reset fails for some reason, try a sample step
        try:
            step_res = env.step(env.action_space.sample())
            if len(step_res) == 5:
                obs = step_res[0]
            else:
                obs = step_res[0]
        except Exception:
            raise

    # If obs is a dict -> goal-conditioned environment
    if isinstance(obs, dict):
        dims = {
            'o': int(np.array(obs['observation']).shape[0]),
            'u': int(env.action_space.shape[0]),
            'g': int(np.array(obs['desired_goal']).shape[0]),
        }

        # include info keys if present (info shape detection)
        # perform a single step to fetch info if possible
        try:
            step_res = env.step(env.action_space.sample())
            if len(step_res) == 5:
                _, _, _, _, info = step_res
            else:
                _, _, _, info = step_res
            for key, value in info.items():
                value = np.array(value)
                if value.ndim == 0:
                    value = value.reshape(1)
                dims['info_{}'.format(key)] = int(value.shape[0])
        except Exception:
            # no info available or step failed; skip info keys
            pass
    else:
        # Flat state environment (MuJoCo, etc.)
        obs_arr = np.asarray(obs)
        dims = {
            'o': int(obs_arr.shape[0]),  # Direct observation dimension
            'u': int(env.action_space.shape[0]),  # Action dimension
            'g': 0,  # No goals in flat state environments
        }

    return dims
