import click
import numpy as np
import pickle
import sys
import os
import gym
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

# Add project root for portable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# OpenAI Baselines should be available via pip install - no hardcoded path needed

from baselines import logger
from baselines.common import set_global_seeds
import src.experiment.mujoco_config as config  # Use MuJoCo config instead
from src.algorithms.rollout_mujoco import RolloutWorkerMuJoCo  # Use MuJoCo rollout worker


def load_policy_with_fallback(policy_file):
    """
    Load policy with multiple fallback strategies to handle HPRC compatibility issues.
    """
    print(f"Attempting to load policy from: {policy_file}")
    
    # Get the directory containing the policy file for params.json
    policy_dir = os.path.dirname(policy_file)
    params_file = os.path.join(policy_dir, 'params.json')
    
    # Strategy 1: Try standard pickle loading
    try:
        with open(policy_file, 'rb') as f:
            loaded_data = pickle.load(f)
        print("✓ Successfully loaded with standard pickle")
        return loaded_data, None
    except Exception as e:
        print(f"✗ Standard pickle failed: {e}")
    
    # Strategy 2: Try with protocol 2 (more compatible)
    try:
        with open(policy_file, 'rb') as f:
            loaded_data = pickle.load(f, encoding='latin1')
        print("✓ Successfully loaded with latin1 encoding")
        return loaded_data, None
    except Exception as e:
        print(f"✗ Latin1 encoding failed: {e}")
    
    # Strategy 3: Load params.json and create minimal policy object
    if os.path.exists(params_file):
        try:
            print(f"Attempting to load params from: {params_file}")
            with open(params_file, 'r') as f:
                params = json.load(f)
            print("✓ Successfully loaded params.json")
            
            # Create a minimal policy-like object with just the environment info
            class MinimalPolicy:
                def __init__(self, env_name):
                    self.env_name = env_name
                    self.info = {'env_name': env_name}
            
            return MinimalPolicy(params['env_name']), params
        except Exception as e:
            print(f"✗ Params.json loading failed: {e}")
    
    # Strategy 4: Last resort - return None and let caller handle
    print("✗ All loading strategies failed")
    return None, None


def visualize_training_progress(policy_file, output_dir):
    """Visualize training progress from tabular_log.json if available."""
    policy_dir = os.path.dirname(policy_file)
    tabular_file = os.path.join(policy_dir, 'tabular_log.json')
    
    if not os.path.exists(tabular_file):
        print(f"⚠️ No tabular_log.json found in {policy_dir}")
        return
    
    try:
        with open(tabular_file, 'r') as f:
            data = json.load(f)
        
        # Extract training metrics
        epochs = [d['epoch'] for d in data]
        test_returns = [d.get('test/mean_episode_return', 0) for d in data]
        train_returns = [d.get('train/mean_episode_return', 0) for d in data]
        
        # Create visualization
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        # Plot learning curves
        plt.subplot(2, 2, 1)
        plt.plot(epochs, test_returns, 'b-', label='Test Return', linewidth=2)
        plt.plot(epochs, train_returns, 'r--', label='Train Return', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Episode Return')
        plt.title('Learning Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot test return smoothed
        if len(test_returns) > 10:
            window = min(10, len(test_returns) // 5)
            smoothed = np.convolve(test_returns, np.ones(window)/window, mode='valid')
            plt.subplot(2, 2, 2)
            plt.plot(epochs[:len(smoothed)], smoothed, 'g-', linewidth=3)
            plt.xlabel('Epoch')
            plt.ylabel('Smoothed Test Return')
            plt.title(f'Smoothed Performance (window={window})')
            plt.grid(True, alpha=0.3)
        
        # Plot final performance comparison
        plt.subplot(2, 2, 3)
        final_performance = test_returns[-10:] if len(test_returns) >= 10 else test_returns
        plt.hist(final_performance, bins=min(10, len(final_performance)), alpha=0.7, color='skyblue')
        plt.xlabel('Episode Return')
        plt.ylabel('Frequency')
        plt.title('Final Performance Distribution')
        plt.axvline(np.mean(final_performance), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(final_performance):.1f}')
        plt.legend()
        
        # Training statistics
        plt.subplot(2, 2, 4)
        stats_text = f"""Training Statistics:
        
Final Test Return: {test_returns[-1]:.2f}
Best Test Return: {max(test_returns):.2f}
Mean (last 10): {np.mean(test_returns[-10:]):.2f}
Std (last 10): {np.std(test_returns[-10:]):.2f}
Total Epochs: {len(epochs)}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        plt.axis('off')
        
        plt.tight_layout()
        
        # Extract experiment info from path
        exp_name = os.path.basename(policy_dir)
        save_path = os.path.join(output_dir, f'{exp_name}_training_progress.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training progress saved to: {save_path}")
        plt.close()
        
        return {
            'final_return': test_returns[-1],
            'best_return': max(test_returns),
            'mean_final': np.mean(test_returns[-10:]),
            'epochs': len(epochs)
        }
        
    except Exception as e:
        print(f"❌ Failed to create training visualization: {e}")
        return None


def record_episode_gif(env, policy, output_path, max_steps=1000):
    """Record an episode as an animated GIF."""
    try:
        frames = []
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < max_steps:
            # Render frame with fallback for different gym versions
            try:
                frame = env.render(mode='rgb_array')
            except TypeError:
                # For newer gym/gymnasium versions
                frame = env.render()
            
            if frame is not None:
                frames.append(Image.fromarray(frame))
            
            # Get action
            if hasattr(policy, 'get_actions'):
                actions, _ = policy.get_actions(obs.reshape(1, -1), compute_Q=False)
                action = actions[0]
            else:
                # Random policy fallback
                action = env.action_space.sample()
            
            # Step environment
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            total_reward += reward
            step += 1
        
        # Save GIF
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=50,  # 50ms per frame = 20 FPS
                loop=0
            )
            print(f"🎬 Episode GIF saved to: {output_path} (Return: {total_reward:.2f})")
        
        return total_reward
        
    except Exception as e:
        print(f"❌ Failed to record GIF: {e}")
        return 0


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
@click.option('--visualize', is_flag=True, help='Create performance visualization plots')
@click.option('--record_gif', is_flag=True, help='Record episodes as animated GIFs')
@click.option('--output_dir', type=str, default='visualization_output', help='Directory for visualization outputs')
def main(policy_file, seed, n_test_rollouts, render, visualize, record_gif, output_dir):
    set_global_seeds(seed)

    # Load policy with fallback strategies
    loaded_data, params_from_json = load_policy_with_fallback(policy_file)
    
    if loaded_data is None:
        print("❌ Failed to load policy file. Please check the file format.")
        return
    
    # Extract environment name using various strategies
    env_name = None
    
    # Strategy 1: From loaded policy object
    if hasattr(loaded_data, 'info') and 'env_name' in loaded_data.info:
        env_name = loaded_data.info['env_name']
    elif hasattr(loaded_data, 'env_name'):
        env_name = loaded_data.env_name
    elif isinstance(loaded_data, dict) and 'env_name' in loaded_data:
        env_name = loaded_data['env_name']
    
    # Strategy 2: From params.json if available
    if env_name is None and params_from_json is not None:
        env_name = params_from_json.get('env_name', 'HalfCheetah-v4')
    
    # Strategy 3: Fallback default
    if env_name is None:
        env_name = 'HalfCheetah-v4'
        print(f"⚠️ Could not determine environment, using default: {env_name}")
    
    print(f"🎯 Environment: {env_name}")
    print(f"🎮 Running {n_test_rollouts} test episodes with render={bool(render)}")

    # Create visualization if requested
    if visualize:
        print("📊 Creating training progress visualization...")
        training_stats = visualize_training_progress(policy_file, output_dir)
        if training_stats:
            print(f"   Final Return: {training_stats['final_return']:.2f}")
            print(f"   Best Return: {training_stats['best_return']:.2f}")
            print(f"   Training Epochs: {training_stats['epochs']}")

    # Handle the case where we only have params but no actual policy
    if params_from_json is not None and not hasattr(loaded_data, '__call__'):
        print("⚠️ Policy object not usable, but we have parameters.")
        print("   This will run with a random policy for demonstration.")
        print("   The policy file may be corrupted or from an incompatible version.")
        
        # Create a simple random policy for testing
        class RandomPolicy:
            def __init__(self, action_dim):
                self.action_dim = action_dim
            
            def get_actions(self, obs, ag=None, g=None, noise_eps=0., random_eps=0., 
                          use_target_net=False, compute_Q=False):
                """Match the interface expected by rollout worker"""
                batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
                actions = np.random.uniform(-1, 1, (batch_size, self.action_dim))
                Q_values = np.zeros((batch_size, 1)) if compute_Q else None
                return actions, Q_values
        
        # Get action dimensions for the environment
        env_action_dims = {
            'HalfCheetah-v4': 6,
            'Hopper-v4': 3, 
            'Walker2d-v4': 6,
            'Ant-v4': 8
        }
        action_dim = env_action_dims.get(env_name, 6)
        policy = RandomPolicy(action_dim)
    else:
        policy = loaded_data

    # Prepare params for MuJoCo.
    params = config.DEFAULT_MUJOCO_PARAMS.copy()
    params['env_name'] = env_name
    if env_name in config.DEFAULT_MUJOCO_ENV_PARAMS:
        params.update(config.DEFAULT_MUJOCO_ENV_PARAMS[env_name])  # merge env-specific parameters in
    
    params = config.prepare_mujoco_params(params)
    
    # Override make_env for rendering support after config preparation
    if render or record_gif:
        def make_env_with_render():
            env = gym.make(env_name, render_mode="rgb_array" if record_gif else "human")
            return env
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
    
    # Setup for GIF recording
    if record_gif:
        os.makedirs(output_dir, exist_ok=True)
        exp_name = os.path.basename(os.path.dirname(policy_file))
        print(f"🎬 Recording episodes as GIFs...")
    
    for episode in range(n_test_rollouts):
        episode_data = evaluator.generate_rollouts()
        episode_return = episode_data['r'].sum() if isinstance(episode_data['r'], list) else episode_data['r'].mean()
        episode_returns.append(episode_return)
        print(f"Episode {episode + 1}: Return = {episode_return:.2f}")
        
        # Record GIF for first few episodes if requested
        if record_gif and episode < min(3, n_test_rollouts):
            gif_path = os.path.join(output_dir, f'{exp_name}_episode_{episode+1}.gif')
            # Create a separate environment for GIF recording
            gif_env = gym.make(env_name, render_mode='rgb_array')
            record_episode_gif(gif_env, policy, gif_path)
            gif_env.close()

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
