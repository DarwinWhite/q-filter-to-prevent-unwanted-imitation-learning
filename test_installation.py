#!/usr/bin/env python3
"""
Quick test script to verify the PyTorch Q-filter implementation is working.
Run this after setting up the environment to ensure everything is installed correctly.
"""

import sys
import os
import tempfile

# Add src to path (from repository root)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
    except ImportError:
        print("[FAIL] PyTorch not installed")
        return False
    
    try:
        import gymnasium as gym
        env = gym.make('HalfCheetah-v4')
        env.close()
        print("[OK] Gymnasium with MuJoCo")
    except ImportError:
        print("[FAIL] Gymnasium not installed")
        return False
    except Exception as e:
        print(f"[FAIL] MuJoCo environment error: {e}")
        return False
    
    try:
        from src.experiment import train_mujoco, play, plot
        print("[OK] All experiment modules")
    except ImportError as e:
        print(f"[FAIL] Experiment modules: {e}")
        return False
    
    try:
        from src.algorithms import ddpg, rollout_mujoco
        print("[OK] All algorithm modules")
    except ImportError as e:
        print(f"[FAIL] Algorithm modules: {e}")
        return False
    
    return True

def test_demo_data():
    """Test that demo data files exist and are readable."""
    print("\nTesting demo data...")
    
    demo_files = [
        'demo_data/halfcheetah_expert_demos.npz',
        'demo_data/hopper_expert_demos.npz',
        'demo_data/walker2d_expert_demos.npz'
    ]
    
    for demo_file in demo_files:
        if os.path.exists(demo_file):
            try:
                import numpy as np
                data = np.load(demo_file, allow_pickle=True)
                print(f"[OK] {demo_file}: {data['obs'].shape[0]} trajectories")
            except Exception as e:
                print(f"[FAIL] {demo_file}: {e}")
                return False
        else:
            print(f"[FAIL] {demo_file}: File not found")
            return False
    
    return True

def test_training():
    """Test a very short training run."""
    print("\nTesting training (1 epoch)...")
    
    try:
        # Create temporary log directory
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.experiment.train_mujoco import launch
            
            # Run a very short training
            launch(
                env='HalfCheetah-v4',
                logdir=tmpdir,
                n_epochs=1,
                num_cpu=1,
                seed=42,
                replay_strategy='future',
                policy_save_interval=5,
                clip_return=5000,
                demo_file='',  # No demos for quick test
                bc_loss=0,
                q_filter=0,
                num_demo=0
            )
            print("[OK] Training test completed")
            return True
    
    except Exception as e:
        print(f"[FAIL] Training test failed: {e}")
        return False

if __name__ == '__main__':
    print("PyTorch Q-Filter Implementation Test")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_demo_data()
    
    if success:
        print("\n[TARGET] Basic tests passed! Attempting training test...")
        success &= test_training()
    
    print("\n" + "=" * 40)
    if success:
        print("[SUCCESS] All tests passed! The implementation is ready to use.")
        print("\nNext steps:")
        print("1. Run full training: python src/experiment/train_mujoco.py --env HalfCheetah-v4")
        print("2. With demos: python src/experiment/train_mujoco.py --env HalfCheetah-v4 --demo_file demo_data/halfcheetah_expert_demos.npz --bc_loss 1")
        print("3. With Q-filter: python src/experiment/train_mujoco.py --env HalfCheetah-v4 --demo_file demo_data/halfcheetah_expert_demos.npz --bc_loss 1 --q_filter 1")
    else:
        print("[FAIL] Some tests failed. Check the error messages above.")
        sys.exit(1)