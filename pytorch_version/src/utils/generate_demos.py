#!/usr/bin/env python3
"""
Demo generation utility for MuJoCo environments.

This is a lightweight helper that collects rollouts using an existing policy
parameter file (pickle or PyTorch state dict). Currently it uses a placeholder
random-action policy unless you reconstruct a real policy from the parameter file.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import gym

# allow loading PyTorch state dicts optionally
try:
    import torch
except Exception:
    torch = None

# Add project root for imports (portable)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_policy_parameters(pkl_file_path: str):
    """
    Load policy parameters from a .pkl or PyTorch file.
    Returns the loaded object or None on failure.
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            params = pickle.load(f)
        return params
    except Exception:
        # Try torch.load as a fallback (for PyTorch state_dicts)
        if torch is not None:
            try:
                params = torch.load(pkl_file_path, map_location='cpu')
                return params
            except Exception:
                pass
    print(f"Warning: could not load policy parameters from {pkl_file_path}")
    return None


def collect_demonstrations_with_params(env_name: str, pkl_file_path: str,
                                       num_episodes: int = 100, max_steps: int = 1000) -> dict:
    """
    Collect demonstrations using a pre-trained policy from parameter files.

    Returns:
        Dictionary containing demonstration data in a simple list-of-episodes form.
    """
    print(f"Collecting demonstrations for {env_name} using {pkl_file_path}")

    # Load policy parameters (not used by placeholder random policy)
    policy_params = load_policy_parameters(pkl_file_path)
    if policy_params is None:
        print("Continuing with placeholder (random) policy since parameters could not be loaded.")

    env = gym.make(env_name)

    episodes = []

    for episode in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        episode_data = {
            'observations': [np.array(obs, copy=True)],
            'actions': [],
            'rewards': [],
            'done': False
        }

        for step in range(max_steps):
            # Placeholder action: random sample from action space
            action = env.action_space.sample()
            action_np = np.array(action, copy=True)

            step_result = env.step(action)
            # Support both old and new Gym APIs
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            episode_data['actions'].append(action_np)
            episode_data['rewards'].append(float(reward))
            episode_data['observations'].append(np.array(next_obs, copy=True))

            obs = next_obs

            if done:
                episode_data['done'] = True
                break

        episodes.append(episode_data)

        if (episode + 1) % 10 == 0:
            print(f"  Completed {episode + 1}/{num_episodes} episodes")

    env.close()

    demo_data = format_demonstrations_for_ddpg(episodes, env_name)
    return demo_data


def format_demonstrations_for_ddpg(episodes: list, env_name: str) -> dict:
    """
    Format collected episodes into a structure resembling the classic DDPG demo format.
    Note: episodes may have varying lengths; returned arrays are object arrays per-episode.
    """
    all_obs = []
    all_actions = []
    all_rewards = []
    all_info = []

    for episode_idx, episode in enumerate(episodes):
        obs_array = np.array(episode['observations'])
        actions_array = np.array(episode['actions'])
        rewards_array = np.array(episode['rewards']).reshape(-1, 1)  # (T, 1)

        info_array = [{'episode': episode_idx, 'step': i} for i in range(len(episode['actions']))]

        all_obs.append(obs_array)
        all_actions.append(actions_array)
        all_rewards.append(rewards_array)
        all_info.append(info_array)

    return {
        'obs': np.array(all_obs, dtype=object),
        'acs': np.array(all_actions, dtype=object),
        'rewards': np.array(all_rewards, dtype=object),
        'info': np.array(all_info, dtype=object),
        'env_name': env_name
    }


def save_demonstrations(demo_data: dict, save_path: str):
    """
    Save demonstration data to .npz file format.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert to numpy arrays (object arrays) for saving
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
    Simple CLI to generate demo files for a set of parameter files.
    This is purposely conservative — it does not attempt to reconstruct a network.
    """
    print("Demo Generation Utility")
    print("=" * 50)

    env_configs = {
        'HalfCheetah-v4': {'obs_dim': 17, 'action_dim': 6},
        'Hopper-v4': {'obs_dim': 11, 'action_dim': 3},
        'Walker2d-v4': {'obs_dim': 17, 'action_dim': 6}
    }

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

    os.makedirs('demo_data', exist_ok=True)

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
                    num_episodes=10,
                    max_steps=1000
                )

                if demo_data is not None:
                    save_demonstrations(demo_data, output_path)
                    generated_count += 1

            except Exception as e:
                print(f"    Error: {e}")

    print("\n" + "=" * 50)
    print(f"Demo generation complete! Generated {generated_count} demonstration files.")
    print("Note: This currently uses a placeholder random policy. To generate real demonstrations,")
    print("reconstruct the policy network from the parameter files and replace the action selection.")
    print("")

if __name__ == "__main__":
    main()
