import click
import numpy as np
import torch
import pickle
import sys
import os
import gymnasium as gym
import json

# Add project root for portable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import src.experiment.mujoco_config as config
from src.algorithms.rollout_mujoco import RolloutWorkerMuJoCo


def try_load_policy(policy_file, dims, params, env_name):
    """Try to load PyTorch policy with fallback strategies."""
    print(f"Attempting to load policy from: {policy_file}")
    
    try:
        # Try loading as PyTorch state dict
        policy_data = torch.load(policy_file, map_location='cpu')
        
        # Create policy and load state
        policy = config.create_mujoco_ddpg(dims, params)
        
        if 'actor_state_dict' in policy_data:
            policy.ddpg.actor.load_state_dict(policy_data['actor_state_dict'])
            policy.ddpg.critic.load_state_dict(policy_data['critic_state_dict'])
            
            # Load normalizer stats if available
            if 'o_stats' in policy_data:
                policy.ddpg.o_stats.mean = torch.tensor(policy_data['o_stats']['mean'])
                policy.ddpg.o_stats.std = torch.tensor(policy_data['o_stats']['std'])
            if 'g_stats' in policy_data:
                policy.ddpg.g_stats.mean = torch.tensor(policy_data['g_stats']['mean'])
                policy.ddpg.g_stats.std = torch.tensor(policy_data['g_stats']['std'])
                
            print("✅ Successfully loaded PyTorch policy")
            return policy
        else:
            raise ValueError("Invalid policy format")
            
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        print("   Using random policy for demonstration.")
        
        # Create random policy
        class RandomPolicy:
            def __init__(self, action_dim):
                self.action_dim = action_dim
                
            def get_actions(self, obs, ag=None, g=None, noise_eps=0., random_eps=0.,
                          use_target_net=False, compute_Q=False):
                """Random policy for testing."""
                batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
                actions = np.random.uniform(-1, 1, (batch_size, self.action_dim))
                Q_values = np.zeros((batch_size, 1)) if compute_Q else None
                return (actions, Q_values) if compute_Q else actions
                
        # Get action dimensions for environment
        env_action_dims = {
            'HalfCheetah-v4': 6,
            'Hopper-v4': 3,
            'Walker2d-v4': 6,
            'Ant-v4': 8
        }
        action_dim = env_action_dims.get(env_name, 6)
        return RandomPolicy(action_dim)


class SimpleLogger:
    def __init__(self):
        pass
    def info(self, msg):
        print(f"INFO: {msg}")
    def warn(self, msg):
        print(f"WARN: {msg}")
    def record_tabular(self, key, val):
        pass
    def dump_tabular(self):
        pass

logger = SimpleLogger()


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=5)
@click.option('--render', is_flag=True)
@click.option('--record_gif', is_flag=True)
@click.option('--output_dir', type=str, default='visualization_output')
def main(policy_file, seed, n_test_rollouts, render, record_gif, output_dir):
    """Play policy in MuJoCo environment."""
    
    # Suppress warnings at the start
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*GLFW.*")
    warnings.filterwarnings("ignore", message=".*out of date.*")
    
    # Extract environment name from policy path
    env_name = None
    if 'HalfCheetah' in policy_file:
        env_name = 'HalfCheetah-v4'
    elif 'Hopper' in policy_file:
        env_name = 'Hopper-v4' 
    elif 'Walker2d' in policy_file:
        env_name = 'Walker2d-v4'
    elif 'Ant' in policy_file:
        env_name = 'Ant-v4'
    else:
        env_name = 'HalfCheetah-v4'  # Default
    
    print(f"🎯 Environment: {env_name}")
    
    # Setup parameters
    params = config.prepare_mujoco_params(env_name)
    
    # Update environment factory to include render mode if needed
    if render:
        original_make_env = params['make_env']
        params['make_env'] = lambda: original_make_env(render_mode='human')
    
    dims = config.configure_mujoco_dims(params)
    
    # Load policy
    policy = try_load_policy(policy_file, dims, params, env_name)

    # Rollout worker (kept same)
    evaluator = RolloutWorkerMuJoCo(params['make_env'], policy, dims, logger,
                                    exploit=True, render=render, T=params['T'])

    if record_gif:
        os.makedirs(output_dir, exist_ok=True)

    # Run episodes
    returns = []
    for ep in range(n_test_rollouts):
        result = evaluator.generate_rollouts()
        
        # Get actual episode return - use stored episode returns if available
        if 'episode_returns' in result and len(result['episode_returns']) > 0:
            ep_return = result['episode_returns'][0]  # Get first episode return
        else:
            # Fallback: sum rewards (though they might be dummy)
            ep_return = result['r'].sum()
            
        returns.append(ep_return)
        print(f"Episode {ep + 1}: Return = {ep_return:.2f}")
        
        if record_gif:
            # Record episode as GIF (placeholder - would need environment rendering)
            print(f"📹 Episode {ep + 1} recorded (GIF recording not implemented)")

    # Cleanup environments to prevent cleanup errors
    try:
        for env in evaluator.envs:
            env.close()
    except Exception:
        pass  # Ignore cleanup errors

    # Summary statistics
    print(f"\n📊 Results after {n_test_rollouts} episodes:")
    print(f"   Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"   Min Return: {np.min(returns):.2f}")
    print(f"   Max Return: {np.max(returns):.2f}")
    
    # Success rate (environment dependent)
    if env_name == 'HalfCheetah-v4':
        success_threshold = 4000
    elif env_name == 'Hopper-v4':
        success_threshold = 3500
    elif env_name == 'Walker2d-v4':  
        success_threshold = 4000
    else:
        success_threshold = 1000
        
    success_rate = np.mean(np.array(returns) > success_threshold)
    print(f"   Success Rate: {success_rate:.2%} (threshold: {success_threshold})")


if __name__ == '__main__':
    main()