import click
import numpy as np
import pickle
import sys
import os
import gym

# Add project root for portable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# OpenAI Baselines should be available via pip install - no hardcoded path needed

from baselines import logger
from baselines.common import set_global_seeds
import src.experiment.mujoco_config as config  # Use MuJoCo config instead
from src.algorithms.rollout_mujoco import RolloutWorkerMuJoCo  # Use MuJoCo rollout worker


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
def main(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        loaded_data = pickle.load(f)
    
    # Handle different policy save formats
    if hasattr(loaded_data, 'info'):
        # Policy object with info attribute
        policy = loaded_data
        env_name = policy.info['env_name']
    elif isinstance(loaded_data, dict) and 'env_name' in loaded_data:
        # Dictionary format
        env_name = loaded_data['env_name']
        policy = loaded_data.get('policy', loaded_data)
    else:
        # Try to extract from object or use default
        policy = loaded_data
        try:
            env_name = getattr(policy, 'env_name', 'HalfCheetah-v4')
        except:
            env_name = 'HalfCheetah-v4'  # Default fallback
    
    print(f"Loading policy for environment: {env_name}")
    print(f"Running {n_test_rollouts} test episodes with render={bool(render)}")

    # Prepare params for MuJoCo.
    params = config.DEFAULT_MUJOCO_PARAMS.copy()
    params['env_name'] = env_name
    if env_name in config.DEFAULT_MUJOCO_ENV_PARAMS:
        params.update(config.DEFAULT_MUJOCO_ENV_PARAMS[env_name])  # merge env-specific parameters in
    
    params = config.prepare_mujoco_params(params)
    
    # Override make_env for rendering support after config preparation
    if render:
        def make_env_with_render():
            return gym.make(env_name, render_mode="human")
        params['make_env'] = make_env_with_render
    
    config.log_mujoco_params(params, logger=logger)

    dims = config.configure_mujoco_dims(params)

    # Configure evaluation parameters for MuJoCo
    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
        'T': params['T'],
    }

    for name in ['gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    # Create MuJoCo evaluator
    evaluator = RolloutWorkerMuJoCo(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    print("Starting evaluation...")
    evaluator.clear_history()
    episode_returns = []
    
    for episode in range(n_test_rollouts):
        episode_data = evaluator.generate_rollouts()
        episode_return = episode_data['r'].sum() if isinstance(episode_data['r'], list) else episode_data['r'].mean()
        episode_returns.append(episode_return)
        print(f"Episode {episode + 1}: Return = {episode_return:.2f}")

    # Print summary
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    print(f"\nEvaluation Summary:")
    print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Min Return: {np.min(episode_returns):.2f}")
    print(f"Max Return: {np.max(episode_returns):.2f}")

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
