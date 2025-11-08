#!/usr/bin/env python3
"""
Validation script to test policy loading and basic functionality.
Run this first to ensure policies load correctly before generating demonstrations.
"""

import pickle
import numpy as np
import gym
import torch
import torch.nn as nn
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.generate_demos import PolicyNetwork, load_policy_from_pkl

def test_policy_loading():
    """Test that policies can be loaded and produce reasonable outputs"""
    
    test_cases = [
        ('demos/cheetah_params.pkl', 'HalfCheetah-v4', 17, 6)
    ]
    
    for pkl_path, env_name, obs_dim, action_dim in test_cases:
        print(f"\nTesting {pkl_path} for {env_name}")
        
        try:
            # Load policy
            policy = load_policy_from_pkl(pkl_path, obs_dim, action_dim)
            print(f"Policy loaded successfully")
            
            # Test with random observation
            test_obs = torch.randn(1, obs_dim)
            with torch.no_grad():
                action = policy.act(test_obs, deterministic=True)
            
            print(f"Policy produces actions of shape: {action.shape}")
            print(f"  Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
            
            # Test with environment
            env = gym.make(env_name)
            
            # Handle modern gym API
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action = policy.act(obs_tensor, deterministic=True)
                action_np = action.squeeze(0).numpy()
            
            # Check if action is within environment bounds
            action_space = env.action_space
            if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
                in_bounds = np.all(action_np >= action_space.low) and np.all(action_np <= action_space.high)
                print(f"Action within environment bounds: {in_bounds}")
                if not in_bounds:
                    print(f"  Action: {action_np}")
                    print(f"  Bounds: [{action_space.low}, {action_space.high}]")
            
            env.close()
            
        except FileNotFoundError:
            print(f"File not found: {pkl_path}")
        except Exception as e:
            print(f"Error: {e}")

def test_environment_compatibility():
    """Test that MuJoCo environments work correctly"""
    
    print("\nTesting MuJoCo environment compatibility...")
    
    # Use newer environment versions that work with modern MuJoCo
    env_names = ['HalfCheetah-v4']
    
    for env_name in env_names:
        try:
            env = gym.make(env_name)
            
            # Handle modern gym API
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            
            action = env.action_space.sample()
            
            # Handle both old and new step API
            step_result = env.step(action)
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            print(f"{env_name}: obs_shape={obs.shape}, action_shape={action.shape}")
            env.close()
            
        except Exception as e:
            print(f"{env_name}: Error - {e}")

def run_quick_episode_test():
    """Run a quick episode with a loaded policy to test full pipeline"""
    
    print("\nRunning quick episode test...")
    
    try:
        # Test with HalfCheetah expert policy
        policy = load_policy_from_pkl('demos/cheetah_params.pkl', 17, 6)
        env = gym.make('HalfCheetah-v4')  # Use modern version
        
        # Handle modern gym API
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        
        total_reward = 0
        steps = 0
        
        for _ in range(100):  # Short episode
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action = policy.act(obs_tensor, deterministic=True)
                action_np = action.squeeze(0).numpy()
            
            # Handle both old and new step API
            step_result = env.step(action_np)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        print(f"Episode completed: {steps} steps, total reward: {total_reward:.2f}")
        env.close()
        
    except Exception as e:
        print(f"Episode test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Policy Loading Validation ===")
    test_policy_loading()
    
    print("\n=== Environment Compatibility Test ===")
    test_environment_compatibility()
    
    print("\n=== Quick Episode Test ===")
    run_quick_episode_test()
    
    print("\n=== Validation Complete ===")
    print("If all tests pass, you can run 'python src/utils/generate_demos.py' to create demonstrations.")