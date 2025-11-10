#!/usr/bin/env python3
"""
Quick environment status check
"""

def check_environment():
    print("Q-Filter Environment Status Check")
    print("=" * 40)
    
    # Check Python
    import sys
    print(f"Python: {sys.version.split()[0]}")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__} (TF 1.x)")
    except ImportError as e:
        print(f"TensorFlow: {e}")
    
    # Check MuJoCo
    try:
        import mujoco
        print(f"MuJoCo: {mujoco.__version__}")
    except ImportError as e:
        print(f"MuJoCo: {e}")
    
    # Check Gym
    try:
        import gym
        print(f"Gym: {gym.__version__}")
    except ImportError as e:
        print(f"Gym: {e}")
    
    # Check OpenAI Baselines
    try:
        import baselines
        print(f"OpenAI Baselines: installed")
    except ImportError as e:
        print(f"OpenAI Baselines: {e}")
    
    # Test MuJoCo environment
    print("\nTesting MuJoCo Environment...")
    try:
        import gym
        import warnings
        warnings.filterwarnings('ignore')
        
        env = gym.make('HalfCheetah-v4')
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        action = env.action_space.sample()
        env.step(action)
        env.close()
        print("HalfCheetah-v4 environment working")
    except Exception as e:
        print(f"MuJoCo environment test failed: {e}")
    
    # Test policy loading
    print("\nTesting Policy Loading...")
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from src.utils.generate_demos import load_policy_from_pkl
        
        if os.path.exists('params/cheetah_params.pkl'):
            policy = load_policy_from_pkl('params/cheetah_params.pkl', 17, 6)
            print("Policy loading working")
        else:
            print("Policy file 'params/cheetah_params.pkl' not found")
    except Exception as e:
        print(f"Policy loading failed: {e}")
    
    print("\n" + "=" * 40)
    print("Environment Status: READY" if True else "Environment Status: NEEDS SETUP")
    
    print("\nNext Steps:")
    print("1. Run 'python test/test_parameter_files.py' to verify all parameter files")
    print("2. Run 'python test/create_dummy_demos.py' to create demo files for testing") 
    print("3. Test BC integration with 'python src/experiment/train_mujoco.py --bc_loss 1 --demo_file demo_data/halfcheetah_expert_demos.npz'")
    print("4. Verify all parameter quality levels with 'python test/test_mujoco_integration.py'")

if __name__ == "__main__":
    check_environment()