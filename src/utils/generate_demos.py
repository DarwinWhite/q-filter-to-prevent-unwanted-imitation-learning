#!/usr/bin/env python3
"""
Script to generate demonstration trajectories from pre-trained policy parameters.
Converts .pkl policy files to .npz demonstration files compatible with the codebase.
"""

import pickle
import numpy as np
import gym
import torch
import torch.nn as nn
import os
from typing import Dict, List, Tuple, Any

class PolicyNetwork(nn.Module):
    """Neural network policy matching the architecture from the .pkl files"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 300):
        super().__init__()
        self.fc0 = nn.Linear(obs_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.last_fc = nn.Linear(hidden_size, action_dim)  # mean
        self.last_fc_log_std = nn.Linear(hidden_size, action_dim)  # log std
        
    def forward(self, obs):
        x = torch.relu(self.fc0(obs))
        x = torch.relu(self.fc1(x))
        mean = self.last_fc(x)
        log_std = self.last_fc_log_std(x)
        return mean, log_std
    
    def act(self, obs, deterministic=False):
        """Sample action from policy"""
        mean, log_std = self.forward(obs)
        if deterministic:
            return mean
        else:
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            return action

def load_policy_from_pkl(pkl_path: str, obs_dim: int, action_dim: int) -> PolicyNetwork:
    """Load policy network from .pkl parameter file"""
    
    # Load parameters
    with open(pkl_path, 'rb') as f:
        params = pickle.load(f)
    
    # Create network
    policy = PolicyNetwork(obs_dim, action_dim)
    
    # Load parameters into network
    state_dict = {}
    for key, value in params.items():
        # Convert numpy arrays to torch tensors
        tensor_value = torch.from_numpy(value)
        
        # Map parameter names to PyTorch conventions
        # The weights are already in PyTorch format (out_features, in_features)
        clean_key = key.replace('/', '.')
        
        state_dict[clean_key] = tensor_value
    
    policy.load_state_dict(state_dict)
    policy.eval()
    
    return policy

def collect_trajectories(env_name: str, policy: PolicyNetwork, 
                        num_episodes: int = 100, max_steps: int = 1000) -> Dict[str, List]:
    """Collect demonstration trajectories using the policy"""
    
    env = gym.make(env_name)
    action_space = env.action_space
    
    episodes_obs = []
    episodes_actions = []
    episodes_info = []
    
    print(f"Collecting {num_episodes} episodes from {env_name}...")
    print(f"Action space bounds: [{action_space.low}, {action_space.high}]")
    
    for episode in range(num_episodes):
        obs_list = []
        action_list = []
        info_list = []
        
        # Handle modern gym API
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        obs_list.append(obs)
        
        for step in range(max_steps):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Get action from policy (deterministic for demonstrations)
            with torch.no_grad():
                action_tensor = policy.act(obs_tensor, deterministic=True)
                action = action_tensor.squeeze(0).numpy()
            
            # Clip action to environment bounds
            action = np.clip(action, action_space.low, action_space.high)
            
            # Take step in environment
            step_result = env.step(action)
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            action_list.append(action)
            info_list.append(info)
            obs_list.append(next_obs)
            
            obs = next_obs
            
            if done:
                break
        
        episodes_obs.append(obs_list)
        episodes_actions.append(action_list)
        episodes_info.append(info_list)
        
        if (episode + 1) % 20 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes")
    
    env.close()
    
    return {
        'obs': episodes_obs,
        'acs': episodes_actions,
        'info': episodes_info
    }

def save_demonstrations(demo_data: Dict[str, List], save_path: str):
    """Save demonstration data in .npz format"""
    
    # Convert to format expected by the codebase
    # The current codebase expects observations to be dictionaries with specific keys
    # For MuJoCo environments, we need to adapt this
    
    # For now, save raw observations - we'll adapt the loader later
    np.savez_compressed(
        save_path,
        obs=demo_data['obs'],
        acs=demo_data['acs'],
        info=demo_data['info']
    )
    
    print(f"Saved demonstrations to {save_path}")

def main():
    """Generate demonstrations for all environments and quality levels"""
    
    # Environment configurations (using modern versions)
    env_configs = {
        'HalfCheetah-v4': {'obs_dim': 17, 'action_dim': 6},
        'Hopper-v4': {'obs_dim': 11, 'action_dim': 3},
        'Walker2d-v4': {'obs_dim': 17, 'action_dim': 6}
    }
    
    # Policy file mappings
    policy_files = {
        'HalfCheetah-v4': {
            'expert': 'demos/cheetah_params.pkl',
            'medium': 'demos/cheetah_medium_params.pkl',
            'random': 'demos/cheetah_random_params.pkl'
        },
        'Hopper-v4': {
            'expert': 'demos/hopper_params.pkl',
            'medium': 'demos/hopper_medium_params.pkl',
            'random': 'demos/hopper_random_params.pkl'
        },
        'Walker2d-v4': {
            'expert': 'demos/walker2d_params.pkl',
            'medium': 'demos/walker2d_medium_params.pkl',
            'random': 'demos/walker2d_random_params.pkl'
        }
    }
    
    # Create output directory
    os.makedirs('demo_data', exist_ok=True)
    
    # Generate demonstrations for each environment and quality level
    for env_name, config in env_configs.items():
        for quality, pkl_path in policy_files[env_name].items():
            
            if not os.path.exists(pkl_path):
                print(f"Warning: {pkl_path} not found, skipping...")
                continue
            
            print(f"\nProcessing {env_name} - {quality}")
            
            try:
                # Load policy
                policy = load_policy_from_pkl(
                    pkl_path, 
                    config['obs_dim'], 
                    config['action_dim']
                )
                
                # Collect demonstrations
                demo_data = collect_trajectories(
                    env_name, 
                    policy, 
                    num_episodes=100,
                    max_steps=1000
                )
                
                # Save demonstrations
                save_path = f"demo_data/{env_name.lower().replace('-v2', '')}_{quality}_demos.npz"
                save_demonstrations(demo_data, save_path)
                
            except Exception as e:
                print(f"Error processing {env_name} - {quality}: {e}")
                continue

if __name__ == "__main__":
    main()