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
import tensorflow as tf
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# OpenAI Baselines should be available via pip install - no hardcoded path needed

class PolicyNetwork:
    """
    Neural network policy that matches the structure in the pickle files.
    Based on the parameters, this appears to be a 2-layer feedforward network.
    """
    
    def __init__(self, obs_dim, action_dim, sess=None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sess = sess if sess is not None else tf.get_default_session()
        
        # Placeholders
        self.obs_ph = tf.placeholder(tf.float32, [None, obs_dim], name='obs')
        
        # Build network
        self._build_network()
        
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())
    
    def _build_network(self):
        """Build the policy network"""
        # First hidden layer (fc0)
        self.fc0_w = tf.Variable(tf.zeros([self.obs_dim, 300]), name='fc0_w')
        self.fc0_b = tf.Variable(tf.zeros([300]), name='fc0_b')
        h1 = tf.nn.relu(tf.matmul(self.obs_ph, self.fc0_w) + self.fc0_b)
        
        # Second hidden layer (fc1)
        self.fc1_w = tf.Variable(tf.zeros([300, 300]), name='fc1_w')
        self.fc1_b = tf.Variable(tf.zeros([300]), name='fc1_b')
        h2 = tf.nn.relu(tf.matmul(h1, self.fc1_w) + self.fc1_b)
        
        # Output layer - mean actions (last_fc)
        self.last_fc_w = tf.Variable(tf.zeros([300, self.action_dim]), name='last_fc_w')
        self.last_fc_b = tf.Variable(tf.zeros([self.action_dim]), name='last_fc_b')
        self.action_mean = tf.matmul(h2, self.last_fc_w) + self.last_fc_b
        
        # Output layer - log std (last_fc_log_std)
        self.last_fc_log_std_w = tf.Variable(tf.zeros([300, self.action_dim]), name='last_fc_log_std_w')
        self.last_fc_log_std_b = tf.Variable(tf.zeros([self.action_dim]), name='last_fc_log_std_b')
        self.action_log_std = tf.matmul(h2, self.last_fc_log_std_w) + self.last_fc_log_std_b
        
        # For deterministic policy (expert demonstrations), we use mean actions
        self.action = tf.tanh(self.action_mean)  # Assume bounded actions
    
    def load_parameters(self, policy_params):
        """Load parameters from the pickle file into the network"""
        param_assign_ops = []
        
        # Map parameter names to TF variables
        param_mapping = {
            'fc0/weight': (self.fc0_w, policy_params['fc0/weight'].T),  # Transpose for TF format
            'fc0/bias': (self.fc0_b, policy_params['fc0/bias']),
            'fc1/weight': (self.fc1_w, policy_params['fc1/weight'].T),
            'fc1/bias': (self.fc1_b, policy_params['fc1/bias']),
            'last_fc/weight': (self.last_fc_w, policy_params['last_fc/weight'].T),
            'last_fc/bias': (self.last_fc_b, policy_params['last_fc/bias']),
            'last_fc_log_std/weight': (self.last_fc_log_std_w, policy_params['last_fc_log_std/weight'].T),
            'last_fc_log_std/bias': (self.last_fc_log_std_b, policy_params['last_fc_log_std/bias'])
        }
        
        for param_name, (tf_var, param_value) in param_mapping.items():
            param_assign_ops.append(tf_var.assign(param_value))
        
        # Execute all assignments
        self.sess.run(param_assign_ops)
        print("Loaded policy parameters successfully")
    
    def get_action(self, obs):
        """Get action from the policy given observation"""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        action = self.sess.run(self.action, feed_dict={self.obs_ph: obs})
        return action.flatten()

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
        print("Successfully loaded policy parameters from", pkl_file_path)
        print("Parameter keys:", list(params.keys()))
        return params
    except Exception as e:
        print("Error loading policy parameters from {}: {}".format(pkl_file_path, e))
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
    print("Collecting demonstrations for {} using {}".format(env_name, pkl_file_path))
    
    # Load policy parameters
    policy_params = load_policy_parameters(pkl_file_path)
    if policy_params is None:
        return None
    
    # Create environment
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create TensorFlow session and policy network
    with tf.Session() as sess:
        policy = PolicyNetwork(obs_dim, action_dim, sess)
        policy.load_parameters(policy_params)
        
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
                print("  Completed {}/{} episodes, last episode reward: {:.2f}".format(
                    episode + 1, num_episodes, total_reward))
    
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
    print("Saved demonstrations to {}".format(save_path))

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
                    num_episodes=50,  # Reasonable number for testing
                    max_steps=1000    # Full episode length
                )
                
                if demo_data is not None:
                    save_demonstrations(demo_data, output_path)
                    generated_count += 1
                
            except Exception as e:
                print(f"    Error: {e}")
    
    print(f"\n" + "=" * 50)
    print("Demo generation complete! Generated {} demonstration files.".format(generated_count))
    print("Demonstrations are now ready for use with DDPG + BC loss and Q-filtering!")
    print("Use them with: python train_mujoco.py --env HalfCheetah-v4 --bc_loss 1 --demo_file demo_data/halfcheetah_expert_demos.npz")

if __name__ == "__main__":
    main()