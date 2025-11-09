#!/usr/bin/env python3
"""
Validation script to test environment compatibility and basic functionality.
Run this to ensure the environment is correctly set up with original dependencies.
"""

import os
import sys
import numpy as np
import traceback

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_tensorflow_import():
    """Test that TensorFlow 1.x imports and works correctly"""
    print("\nTesting TensorFlow 1.x compatibility...")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Test basic TF 1.x functionality
        with tf.Session() as sess:
            # Test basic tensor operations
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([4.0, 5.0, 6.0])
            c = tf.add(a, b)
            result = sess.run(c)
            print(f"Basic tensor operations work: [1,2,3] + [4,5,6] = {result}")
        
        print("TensorFlow 1.x test successful")
        
    except Exception as e:
        print(f"TensorFlow test failed: {e}")
        traceback.print_exc()


def test_environment_compatibility():
    """Test that MuJoCo environments work correctly"""
    
    print("\nTesting MuJoCo environment compatibility...")
    
    # Use newer environment versions that work with modern MuJoCo
    env_names = ['HalfCheetah-v4']
    
    for env_name in env_names:
        try:
            import gym
            
            env = gym.make(env_name)
            print(f"Created {env_name} environment")
            
            # Handle modern gym API
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
            
            print(f"  Observation shape: {obs.shape}")
            print(f"  Action space: {env.action_space}")
            
            # Test a few random actions
            for i in range(3):
                action = env.action_space.sample()
                step_result = env.step(action)
                
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                print(f"  Step {i+1}: action_shape={np.array(action).shape}, reward={reward:.3f}")
                
                if done:
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple):
                        obs, info = reset_result
                    else:
                        obs = reset_result
                    break
            
            env.close()
            print(f"{env_name} environment test successful")
            
        except Exception as e:
            print(f"{env_name} environment test failed: {e}")
            traceback.print_exc()


def test_baselines_import():
    """Test that OpenAI Baselines imports correctly"""
    
    print("\nTesting OpenAI Baselines import...")
    
    try:
        import baselines
        print(f"OpenAI Baselines imported successfully")
        
        # Test HER import specifically
        from baselines import her
        print(f"HER module imported successfully")
        
        # Test other key modules
        from baselines.common import tf_util
        print(f"TF utilities imported successfully")
        
        print("OpenAI Baselines test successful")
        
    except Exception as e:
        print(f"OpenAI Baselines test failed: {e}")
        traceback.print_exc()


def run_quick_episode_test():
    """Run a quick episode with random actions to test full pipeline"""
    
    print("\nRunning quick episode test...")
    
    try:
        import gym
        import numpy as np
        
        env = gym.make('HalfCheetah-v4')  # Use modern version
        
        # Handle modern gym API
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
        
        total_reward = 0
        steps = 0
        
        for _ in range(100):  # Run for 100 steps max
            action = env.action_space.sample()  # Random action
            
            step_result = env.step(action)
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
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Q-Filter Environment Validation - Original Dependencies")
    print("=" * 60)
    
    test_tensorflow_import()
    test_environment_compatibility()
    test_baselines_import()
    run_quick_episode_test()
    
    print("\n" + "=" * 60)
    print("Environment validation complete!")
    print("If all tests pass, you can proceed with the Q-filter research.")
    print("=" * 60)

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
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Q-Filter Environment Validation - Original Dependencies")
    print("=" * 60)
    
    test_tensorflow_import()
    test_environment_compatibility()
    test_baselines_import()
    run_quick_episode_test()
    
    print("\n" + "=" * 60)
    print("Environment validation complete!")
    print("If all tests pass, you can proceed with the Q-filter research.")
    print("=" * 60)