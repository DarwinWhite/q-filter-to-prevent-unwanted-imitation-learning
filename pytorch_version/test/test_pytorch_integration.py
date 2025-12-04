#!/usr/bin/env python3
"""
PyTorch Integration Test for Q-Filter Implementation.
Tests basic functionality of the PyTorch DDPG MuJoCo implementation.
"""

import os
import sys
import numpy as np
import torch

# Add project root for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_pytorch_imports():
    """Test that all PyTorch modules import correctly."""
    print("\n1. Testing PyTorch Imports...")
    
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        
        import gymnasium as gym
        print(f"   ✅ Gymnasium: {gym.__version__}")
        
        import src.algorithms.ddpg
        print("   ✅ PyTorch DDPG")
        
        import src.algorithms.actor_critic
        print("   ✅ PyTorch Actor-Critic")
        
        import src.utils.normalizer
        print("   ✅ PyTorch Normalizer")
        
        import src.experiment.mujoco_config
        print("   ✅ MuJoCo Config")
        
        return True
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False


def test_environment_compatibility():
    """Test that MuJoCo environments work with PyTorch version."""
    print("\n2. Testing Environment Compatibility...")
    
    try:
        import gymnasium as gym
        import warnings
        warnings.filterwarnings('ignore')
        
        env_name = 'HalfCheetah-v4'
        env = gym.make(env_name)
        
        # Test reset
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # New Gymnasium API
        
        print(f"   ✅ Environment: {env_name}")
        print(f"   ✅ Observation shape: {obs.shape}")
        print(f"   ✅ Action shape: {env.action_space.shape}")
        
        # Test step
        action = env.action_space.sample()
        step_result = env.step(action)
        
        if len(step_result) == 4:
            obs_new, reward, done, info = step_result
        else:
            obs_new, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        print(f"   ✅ Step function works")
        print(f"   ✅ Reward type: {type(reward)}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Environment error: {e}")
        return False


def test_neural_networks():
    """Test PyTorch neural network creation and forward pass."""
    print("\n3. Testing Neural Networks...")
    
    try:
        from src.algorithms.actor_critic import ActorNetwork, CriticNetwork
        
        # Test actor network
        input_dim = 17 + 1  # HalfCheetah obs + dummy goal
        hidden = 256
        layers = 3
        output_dim = 6  # HalfCheetah actions
        max_u = 1.0
        
        actor = ActorNetwork(input_dim, hidden, layers, output_dim, max_u)
        
        # Test forward pass
        batch_size = 32
        dummy_input = torch.randn(batch_size, input_dim)
        actions = actor(dummy_input)
        
        print(f"   ✅ Actor network created")
        print(f"   ✅ Actor output shape: {actions.shape}")
        print(f"   ✅ Action range: [{actions.min().item():.2f}, {actions.max().item():.2f}]")
        
        # Test critic network
        critic_input_dim = input_dim + output_dim
        critic = CriticNetwork(critic_input_dim, hidden, layers)
        
        dummy_critic_input = torch.randn(batch_size, critic_input_dim)
        q_values = critic(dummy_critic_input)
        
        print(f"   ✅ Critic network created")
        print(f"   ✅ Critic output shape: {q_values.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Network error: {e}")
        return False


def test_ddpg_creation():
    """Test DDPG algorithm creation."""
    print("\n4. Testing DDPG Algorithm...")
    
    try:
        from src.algorithms.ddpg_mujoco import DDPGMuJoCo
        from src.experiment.mujoco_config import prepare_mujoco_params, configure_mujoco_dims
        
        # Setup environment
        env_name = 'HalfCheetah-v4'
        params = prepare_mujoco_params(env_name)
        dims = configure_mujoco_dims(params)
        
        # Configure DDPG parameters
        ddpg_params = {
            'buffer_size': 1000,
            'hidden': 64,  # Smaller for testing
            'layers': 2,
            'batch_size': 32,
            'device': 'cpu'
        }
        params.update(ddpg_params)
        
        # Create DDPG
        from src.experiment.mujoco_config import configure_mujoco_ddpg
        ddpg_config = configure_mujoco_ddpg(dims, params)
        policy = DDPGMuJoCo(input_dims=dims, **ddpg_config)
        
        print(f"   ✅ DDPG created")
        print(f"   ✅ Input dimensions: {dims}")
        
        # Test action generation
        batch_size = 5
        obs = np.random.randn(batch_size, dims['o'])
        actions, q_vals = policy.get_actions(obs, compute_Q=True)
        
        print(f"   ✅ Action generation works")
        print(f"   ✅ Action shape: {actions.shape}")
        print(f"   ✅ Q-value shape: {q_vals.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ DDPG error: {e}")
        return False


def test_rollout_worker():
    """Test MuJoCo rollout worker."""
    print("\n5. Testing Rollout Worker...")
    
    try:
        from src.algorithms.rollout_mujoco import RolloutWorkerMuJoCo
        from src.algorithms.ddpg_mujoco import DDPGMuJoCo  
        from src.experiment.mujoco_config import prepare_mujoco_params, configure_mujoco_dims
        
        # Simple logger
        class DummyLogger:
            def info(self, msg): pass
            def warn(self, msg): pass
        
        # Setup
        env_name = 'HalfCheetah-v4'
        params = prepare_mujoco_params(env_name)
        dims = configure_mujoco_dims(params)
        params.update({'buffer_size': 1000, 'hidden': 64, 'layers': 2, 'batch_size': 32})
        
        # Create policy
        from src.experiment.mujoco_config import configure_mujoco_ddpg
        ddpg_config = configure_mujoco_ddpg(dims, params)
        policy = DDPGMuJoCo(input_dims=dims, **ddpg_config)
        
        # Create rollout worker
        rollout_worker = RolloutWorkerMuJoCo(
            make_env=params['make_env'],
            policy=policy,
            dims=dims,
            logger=DummyLogger(),
            T=10,  # Short episodes for testing
            rollout_batch_size=1,
            render=False
        )
        
        print(f"   ✅ Rollout worker created")
        
        # Test rollout generation
        episode = rollout_worker.generate_rollouts()
        
        print(f"   ✅ Episode generated")
        print(f"   ✅ Episode keys: {list(episode.keys())}")
        print(f"   ✅ Episode shape: {episode['o'].shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Rollout error: {e}")
        return False


def test_training_step():
    """Test basic training step."""
    print("\n6. Testing Training Step...")
    
    try:
        from src.algorithms.ddpg_mujoco import DDPGMuJoCo
        from src.experiment.mujoco_config import prepare_mujoco_params, configure_mujoco_dims
        
        # Setup
        env_name = 'HalfCheetah-v4'
        params = prepare_mujoco_params(env_name)
        dims = configure_mujoco_dims(params)
        params.update({'buffer_size': 1000, 'hidden': 64, 'layers': 2, 'batch_size': 32})
        
        # Create DDPG
        from src.experiment.mujoco_config import configure_mujoco_ddpg
        ddpg_config = configure_mujoco_ddpg(dims, params)
        ddpg_config['T'] = 10  # Override T for testing to avoid huge episodes
        policy = DDPGMuJoCo(input_dims=dims, **ddpg_config)
        
        # Create dummy episode (with batch dimension as expected by replay buffer)
        T = 10  # Use smaller T for testing instead of params['T'] (which is 1000)
        batch_size = 1  # Single episode in batch
        episode = {
            'o': np.random.randn(batch_size, T+1, dims['o']),   # (batch, T+1, obs_dim)
            'u': np.random.randn(batch_size, T, dims['u']),     # (batch, T, action_dim)
            'g': np.random.randn(batch_size, T+1, dims['g']),   # (batch, T+1, goal_dim)
            'ag': np.random.randn(batch_size, T+1, dims['o']),  # (batch, T+1, obs_dim)
            'info': np.zeros(batch_size),  # Add info field with correct shape
        }
        
        # Store episode
        policy.store_episode(episode)
        
        print(f"   ✅ Episode stored")
        print(f"   ✅ Buffer size: {policy.get_current_buffer_size()}")
        
        # Try training step (should work even with minimal data)
        if policy.get_current_buffer_size() > 0:
            policy.train()
            print(f"   ✅ Training step completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        return False


def test_save_load():
    """Test policy save and load."""
    print("\n7. Testing Save/Load...")
    
    try:
        from src.algorithms.ddpg_mujoco import DDPGMuJoCo
        from src.experiment.mujoco_config import prepare_mujoco_params, configure_mujoco_dims
        import tempfile
        
        # Setup
        env_name = 'HalfCheetah-v4'
        params = prepare_mujoco_params(env_name)
        dims = configure_mujoco_dims(params)
        params.update({'buffer_size': 1000, 'hidden': 64, 'layers': 2, 'batch_size': 32})
        
        # Create DDPG
        from src.experiment.mujoco_config import configure_mujoco_ddpg
        ddpg_config = configure_mujoco_ddpg(dims, params)
        policy = DDPGMuJoCo(input_dims=dims, **ddpg_config)
        
        # Test save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            policy.save_policy(f.name)
            print(f"   ✅ Policy saved to {f.name}")
            
            # Test load
            policy.load_policy(f.name)
            print(f"   ✅ Policy loaded successfully")
            
            # Clean up
            os.unlink(f.name)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Save/load error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Q-Filter Integration Test")
    print("=" * 60)
    
    tests = [
        test_pytorch_imports,
        test_environment_compatibility,
        test_neural_networks,
        test_ddpg_creation,
        test_rollout_worker,
        test_training_step,
        test_save_load,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"PyTorch Integration Test Results: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! PyTorch implementation is ready.")
        print("\nNext steps:")
        print("1. Test with: python pytorch_version/src/experiment/train_mujoco.py --env HalfCheetah-v4 --n_epochs 10")
        print("2. Compare with TensorFlow version results")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)