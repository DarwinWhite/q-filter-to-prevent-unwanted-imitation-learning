#!/usr/bin/env python3
"""
TensorFlow-based demo generation utility for MuJoCo parameter files.
This integrates with our existing TensorFlow 1.x setup.
"""

import os
import sys
import pickle
import numpy as np
import gym
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# OpenAI Baselines should be available via pip install - no hardcoded path needed

def load_policy_parameters(pkl_file_path: str) -> dict:
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
        return params
    except Exception as e:
        print(f"Error loading policy parameters from {pkl_file_path}: {e}")
        return None

def collect_demonstrations_with_params(env_name: str, pkl_file_path: str, 
                                     num_episodes: int = 100, max_steps: int = 1000) -> dict:
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
    print(f"Collecting demonstrations for {env_name} using {pkl_file_path}")
    
    # Load policy parameters
    policy_params = load_policy_parameters(pkl_file_path)
    if policy_params is None:
        return None
    
    # Create environment
    env = gym.make(env_name)
    
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
        
        for step in range(max_steps):
            # For now, use a simple policy based on the parameter structure
            # This is a placeholder - in practice, you'd reconstruct the neural network
            # and use the loaded parameters to generate actions
            
            # Simple heuristic action based on observation (placeholder)
            action = env.action_space.sample()  # Random action as placeholder
            
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
            
            if done:
                episode_data['done'] = True
                break
        
        episodes.append(episode_data)
        
        if (episode + 1) % 10 == 0:
            print(f"  Completed {episode + 1}/{num_episodes} episodes")
    
    env.close()
    
    # Convert to the format expected by the DDPG system
    demo_data = format_demonstrations_for_ddpg(episodes, env_name)
    
    return demo_data

def format_demonstrations_for_ddpg(episodes: list, env_name: str) -> dict:
    """
    Format collected episodes into the format expected by DDPG.
    
    Args:
        episodes: List of episode dictionaries
        env_name: Name of the environment
        
    Returns:
        Formatted demonstration data
    """
    # Convert episodes to batch format
    all_obs = []
    all_actions = []
    all_rewards = []
    all_info = []
    
    for episode_idx, episode in enumerate(episodes):
        obs_array = np.array(episode['observations'])
        actions_array = np.array(episode['actions'])
        rewards_array = np.array(episode['rewards']).reshape(-1, 1)  # Make sure rewards are 2D
        
        # Create info dictionary (dummy for MuJoCo)
        info_array = [{'episode': episode_idx, 'step': i} for i in range(len(episode['actions']))]
        
        all_obs.append(obs_array)
        all_actions.append(actions_array)
        all_rewards.append(rewards_array)
        all_info.append(info_array)
    
    return {
        'obs': all_obs,
        'acs': all_actions,
        'rewards': all_rewards,
        'info': all_info
    }

def save_demonstrations(demo_data: dict, save_path: str):
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
    print("TensorFlow-based Demo Generation")
    print("=" * 50)
    
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
    for env_name, qualities in param_files.items():
        print(f"\nProcessing {env_name}...")
        
        for quality, pkl_path in qualities.items():
            if not os.path.exists(pkl_path):
                print(f"  Warning: {pkl_path} not found, skipping...")
                continue
            
            output_path = f"demo_data/{env_name.lower().replace('-v4', '')}_{quality}_demos.npz"
            
            if os.path.exists(output_path):
                print(f"  {quality}: {output_path} already exists, skipping...")
                continue
            
            print(f"  Generating {quality} demonstrations...")
            
            try:
                demo_data = collect_demonstrations_with_params(
                    env_name, 
                    pkl_path, 
                    num_episodes=20,  # Smaller number for testing
                    max_steps=200
                )
                
                if demo_data is not None:
                    save_demonstrations(demo_data, output_path)
                    generated_count += 1
                
            except Exception as e:
                print(f"    Error: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"Demo generation complete! Generated {generated_count} demonstration files.")
    print("Note: This is a placeholder implementation using random actions.")
    print("For actual policy demonstrations, implement proper policy network reconstruction.")

if __name__ == "__main__":
    main()