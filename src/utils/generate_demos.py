#!/usr/bin/env python3
"""
PyTorch-based demo generation utility for MuJoCo parameter files.
Converts Berkeley policy pickle files to demonstration .npz files.

This script is used to generate demonstration data from pre-trained policy
parameter files (in .pkl format). The demonstrations are saved in .npz format
compatible with the PyTorch DDPG implementation.

Note: This script is not needed for normal usage since good demonstration
files are already provided in demo_data/. It's included for reference and
to enable regeneration of demos if needed.

Usage:
    python src/utils/generate_demos.py

Requirements:
    - Berkeley policy parameter files in params/ directory
    - PyTorch and Gymnasium installed

The script will:
    1. Load policy parameters from .pkl files
    2. Create a PyTorch neural network matching the policy structure
    3. Roll out the policy in the environment
    4. Save trajectories in the format expected by DDPG training
"""

import os
import sys
import pickle
import numpy as np
import gymnasium as gym
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

class PolicyNetwork(nn.Module):
    """
    PyTorch neural network policy that matches the structure in the pickle files.
    2-layer feedforward network with ReLU activations.
    """
    
    def __init__(self, obs_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Network layers
        self.fc0 = nn.Linear(obs_dim, 300)
        self.fc1 = nn.Linear(300, 300)
        self.last_fc = nn.Linear(300, action_dim)
        self.last_fc_log_std = nn.Linear(300, action_dim)
        
    def forward(self, obs):
        """Forward pass through the network"""
        h1 = torch.relu(self.fc0(obs))
        h2 = torch.relu(self.fc1(h1))
        
        # Mean actions
        action_mean = self.last_fc(h2)
        
        # For deterministic policy (expert demonstrations), use tanh on mean
        action = torch.tanh(action_mean)
        
        return action
    
    def load_parameters(self, policy_params):
        """Load parameters from the pickle file into the network"""
        try:
            # Map parameter names to PyTorch layers
            # Berkeley format has weights as (out_dim, in_dim), PyTorch uses (out_dim, in_dim)
            # But the pickle files might be transposed, so we need to check
            
            with torch.no_grad():
                # fc0 layer
                self.fc0.weight.copy_(torch.from_numpy(policy_params['fc0/weight']))
                self.fc0.bias.copy_(torch.from_numpy(policy_params['fc0/bias']))
                
                # fc1 layer
                self.fc1.weight.copy_(torch.from_numpy(policy_params['fc1/weight']))
                self.fc1.bias.copy_(torch.from_numpy(policy_params['fc1/bias']))
                
                # Output layer - mean actions
                self.last_fc.weight.copy_(torch.from_numpy(policy_params['last_fc/weight']))
                self.last_fc.bias.copy_(torch.from_numpy(policy_params['last_fc/bias']))
                
                # Output layer - log std (not used for deterministic policy)
                self.last_fc_log_std.weight.copy_(torch.from_numpy(policy_params['last_fc_log_std/weight']))
                self.last_fc_log_std.bias.copy_(torch.from_numpy(policy_params['last_fc_log_std/bias']))
            
            print("Loaded policy parameters successfully")
            return True
        except Exception as e:
            print(f"Error loading parameters: {e}")
            print("   Available keys:", list(policy_params.keys()))
            return False
    
    def get_action(self, obs):
        """Get action from the policy given observation"""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            action = self.forward(obs)
        
        return action.squeeze(0).numpy()

def load_policy_parameters(pkl_file_path):
    """
    Load policy parameters from a .pkl file.
    
    Args:
        pkl_file_path: Path to the .pkl policy file
        
    Returns:
        Dictionary containing policy parameters
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            params = pickle.load(f)
        print(f"Successfully loaded policy parameters from {pkl_file_path}")
        print(f"   Parameter keys: {list(params.keys())}")
        return params
    except Exception as e:
        print(f"Error loading policy parameters from {pkl_file_path}: {e}")
        return None

def collect_demonstrations_with_params(env_name, pkl_file_path, 
                                     num_episodes=100, max_steps=1000):
    """
    Collect demonstrations using a pre-trained policy from parameter files.
    
    Args:
        env_name: Name of the environment (e.g., 'HalfCheetah-v4')
        pkl_file_path: Path to the policy parameter file
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary containing demonstration data
    """
    print(f"\nCollecting demonstrations for {env_name} using {pkl_file_path}")
    
    # Load policy parameters
    policy_params = load_policy_parameters(pkl_file_path)
    if policy_params is None:
        return None
    
    # Create environment
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create policy network
    policy = PolicyNetwork(obs_dim, action_dim)
    policy.eval()  # Set to evaluation mode
    
    if not policy.load_parameters(policy_params):
        return None
    
    # Storage for demonstrations
    episodes = []
    
    for episode in range(num_episodes):
        # Reset environment
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        
        episode_data = {
            'observations': [obs.copy()],
            'actions': [],
            'rewards': [],
            'done': False
        }
        
        total_reward = 0
        for step in range(max_steps):
            # Get action from the loaded policy
            action = policy.get_action(obs)
            
            # Take action in environment
            step_result = env.step(action)
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            # Store transition
            episode_data['actions'].append(action.copy())
            episode_data['rewards'].append(reward)
            episode_data['observations'].append(next_obs.copy())
            
            obs = next_obs
            total_reward += reward
            
            if done:
                episode_data['done'] = True
                break
        
        episodes.append(episode_data)
        
        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes, last episode reward: {total_reward:.2f}")
    
    env.close()
    
    # Convert to the format expected by the DDPG system
    demo_data = format_demonstrations_for_ddpg(episodes, env_name)
    
    return demo_data

def format_demonstrations_for_ddpg(episodes, env_name):
    """
    Format collected episodes into the format expected by DDPG for MuJoCo.
    
    Args:
        episodes: List of episode dictionaries
        env_name: Name of the environment
        
    Returns:
        Formatted demonstration data
    """
    # Convert episodes to the format expected by DDPG
    all_obs = []
    all_actions = []
    
    for episode_idx, episode in enumerate(episodes):
        # For MuJoCo, we need to adapt to the goal-conditioned format
        # even though we don't have real goals
        episode_obs = []
        episode_actions = []
        
        obs_array = np.array(episode['observations'])
        actions_array = np.array(episode['actions'])
        
        # Create goal-conditioned observations for compatibility
        for i in range(len(obs_array)):
            # Create dummy goal-conditioned observation
            obs_dict = {
                'observation': obs_array[i],
                'achieved_goal': np.zeros(1),  # Dummy achieved goal
                'desired_goal': np.zeros(1)    # Dummy desired goal
            }
            episode_obs.append(obs_dict)
        
        # Actions (exclude last observation since it has no corresponding action)
        for i in range(len(actions_array)):
            episode_actions.append(actions_array[i])
        
        all_obs.append(episode_obs)
        all_actions.append(episode_actions)
    
    return {
        'obs': all_obs,
        'acs': all_actions,
        'info': [[{} for _ in range(len(ep_actions))] for ep_actions in all_actions]  # Empty info
    }

def save_demonstrations(demo_data, save_path):
    """
    Save demonstration data to .npz file format.
    
    Args:
        demo_data: Demonstration data dictionary
        save_path: Path to save the .npz file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to numpy arrays for saving
    save_dict = {}
    for key, value in demo_data.items():
        if isinstance(value, list):
            save_dict[key] = np.array(value, dtype=object)
        else:
            save_dict[key] = value
    
    np.savez_compressed(save_path, **save_dict)
    print(f"Saved demonstrations to {save_path}")

def main():
    """
    Main function to generate demonstrations for all available parameter files.
    """
    print("=" * 70)
    print("PyTorch-based Demo Generation for Q-Filter Implementation")
    print("=" * 70)
    
    # Environment configurations
    env_configs = {
        'HalfCheetah-v4': {'obs_dim': 17, 'action_dim': 6},
        'Hopper-v4': {'obs_dim': 11, 'action_dim': 3},
        'Walker2d-v4': {'obs_dim': 17, 'action_dim': 6}
    }
    
    # Parameter file mappings
    param_files = {
        'HalfCheetah-v4': {
            'expert': 'params/cheetah_params.pkl',
            'medium_high': 'params/cheetah_medium_high_params.pkl',
            'medium': 'params/cheetah_medium_params.pkl',
            'random': 'params/cheetah_random_params.pkl'
        },
        'Hopper-v4': {
            'expert': 'params/hopper_params.pkl',
            'medium_high': 'params/hopper_medium_high_params.pkl',
            'medium': 'params/hopper_medium_params.pkl',
            'random': 'params/hopper_random_params.pkl'
        },
        'Walker2d-v4': {
            'expert': 'params/walker2d_params.pkl',
            'medium': 'params/walker2d_medium_params.pkl',
            'medium_low': 'params/walker2d_medium_low_params.pkl',
            'random': 'params/walker2d_random_params.pkl'
        }
    }
    
    # Create output directory
    os.makedirs('demo_data', exist_ok=True)
    
    # Generate demonstrations
    generated_count = 0
    skipped_count = 0
    failed_count = 0
    
    for env_name, qualities in param_files.items():
        print(f"\n{'=' * 70}")
        print(f"Processing {env_name}...")
        print(f"{'=' * 70}")
        
        for quality, pkl_path in qualities.items():
            if not os.path.exists(pkl_path):
                print(f"Warning: {pkl_path} not found, skipping...")
                skipped_count += 1
                continue
            
            output_path = f"demo_data/{env_name.lower().replace('-v4', '')}_{quality}_demos.npz"
            
            if os.path.exists(output_path):
                print(f"{quality}: {output_path} already exists, skipping...")
                skipped_count += 1
                continue
            
            print(f"\nGenerating {quality} demonstrations...")
            
            try:
                demo_data = collect_demonstrations_with_params(
                    env_name, 
                    pkl_path, 
                    num_episodes=50,  # Standard number of demos
                    max_steps=1000    # Full episode length
                )
                
                if demo_data is not None:
                    save_demonstrations(demo_data, output_path)
                    generated_count += 1
                else:
                    failed_count += 1
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
    
    print(f"\n{'=' * 70}")
    print("Demo Generation Summary")
    print(f"{'=' * 70}")
    print(f"Generated: {generated_count} files")
    print(f"Skipped:   {skipped_count} files (already exist or missing params)")
    print(f"Failed:    {failed_count} files")
    print(f"\nDemonstrations are ready for use with DDPG + BC loss and Q-filtering!")
    print(f"\nExample usage:")
    print(f"  python src/experiment/train_mujoco.py \\")
    print(f"    --env HalfCheetah-v4 \\")
    print(f"    --demo_file demo_data/halfcheetah_expert_demos.npz \\")
    print(f"    --bc_loss 1 --q_filter 1 --num_demo 100")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()