#!/usr/bin/env python3
"""
Generate fresh expert trajectories using the policy directly for visualization.
This creates new demonstrations by running the policy in real-time, avoiding
the state-action mismatch issue of replaying recorded actions.
"""

import os
import sys
import pickle
import numpy as np
import gym
import tensorflow as tf
import click
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class PolicyNetwork:
    """Neural network policy for generating fresh trajectories"""
    
    def __init__(self, obs_dim, action_dim, sess=None, stochastic=False):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sess = sess if sess is not None else tf.get_default_session()
        self.stochastic = stochastic
        
        # Placeholders
        self.obs_ph = tf.placeholder(tf.float32, [None, obs_dim], name='obs')
        
        # Build network
        self._build_network()
        
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())
    
    def _build_network(self):
        """Build the policy network"""
        # First hidden layer
        self.fc0_w = tf.Variable(tf.zeros([self.obs_dim, 300]), name='fc0_w')
        self.fc0_b = tf.Variable(tf.zeros([300]), name='fc0_b')
        h1 = tf.nn.relu(tf.matmul(self.obs_ph, self.fc0_w) + self.fc0_b)
        
        # Second hidden layer
        self.fc1_w = tf.Variable(tf.zeros([300, 300]), name='fc1_w')
        self.fc1_b = tf.Variable(tf.zeros([300]), name='fc1_b')
        h2 = tf.nn.relu(tf.matmul(h1, self.fc1_w) + self.fc1_b)
        
        # Output layer - mean actions
        self.last_fc_w = tf.Variable(tf.zeros([300, self.action_dim]), name='last_fc_w')
        self.last_fc_b = tf.Variable(tf.zeros([self.action_dim]), name='last_fc_b')
        self.action_mean = tf.matmul(h2, self.last_fc_w) + self.last_fc_b
        
        # Output layer - log std (for stochastic policy)
        self.last_fc_log_std_w = tf.Variable(tf.zeros([300, self.action_dim]), name='last_fc_log_std_w')
        self.last_fc_log_std_b = tf.Variable(tf.zeros([self.action_dim]), name='last_fc_log_std_b')
        self.action_log_std = tf.matmul(h2, self.last_fc_log_std_w) + self.last_fc_log_std_b
        
        # Action generation
        if self.stochastic:
            # Sample from the distribution
            noise = tf.random_normal(tf.shape(self.action_mean))
            raw_action = self.action_mean + tf.exp(self.action_log_std) * noise
        else:
            # Use deterministic mean
            raw_action = self.action_mean
        
        # Apply tanh to bound actions
        self.action = tf.tanh(raw_action)
    
    def load_parameters(self, policy_params):
        """Load parameters from the pickle file into the network"""
        param_assign_ops = []
        
        param_mapping = {
            'fc0/weight': (self.fc0_w, policy_params['fc0/weight'].T),
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
        
        self.sess.run(param_assign_ops)
        print(f"Loaded policy parameters (stochastic={self.stochastic})")
    
    def get_action(self, obs):
        """Get action from the policy given observation"""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        action = self.sess.run(self.action, feed_dict={self.obs_ph: obs})
        return action.flatten()

@click.command()
@click.argument('policy_file', type=str)
@click.option('--env', type=str, default=None, help='Environment name (auto-detected if not specified)')
@click.option('--episodes', type=int, default=1, help='Number of episodes to run')
@click.option('--max_steps', type=int, default=1000, help='Maximum steps per episode')
@click.option('--stochastic', is_flag=True, help='Use stochastic policy (with exploration noise)')
@click.option('--render', is_flag=True, help='Enable visual rendering')
@click.option('--seed', type=int, default=0, help='Random seed')
def main(policy_file, env, episodes, max_steps, stochastic, render, seed):
    """
    Generate fresh expert trajectories using policy weights directly.
    
    This creates new demonstrations by running the loaded policy in real-time,
    which produces the correct high rewards and proper visualization.
    
    POLICY_FILE: Path to the .pkl policy parameters file
    """
    print("Fresh Expert Trajectory Generator")
    print("=" * 35)
    print(f"Policy mode: {'Stochastic' if stochastic else 'Deterministic'}")
    
    # Auto-detect environment from filename if not specified
    if env is None:
        if 'cheetah' in policy_file.lower():
            env = 'HalfCheetah-v4'
        elif 'hopper' in policy_file.lower():
            env = 'Hopper-v4'
        elif 'walker' in policy_file.lower():
            env = 'Walker2d-v4'
        else:
            env = 'HalfCheetah-v4'  # Default fallback
        print(f"Auto-detected environment: {env}")
    
    # Load policy parameters
    try:
        with open(policy_file, 'rb') as f:
            policy_params = pickle.load(f)
        print(f"Loaded policy parameters from {policy_file}")
    except Exception as e:
        print(f"Error loading policy file: {e}")
        return
    
    # Create environment
    if render:
        gym_env = gym.make(env, render_mode='human')
    else:
        gym_env = gym.make(env)
    
    obs_dim = gym_env.observation_space.shape[0]
    action_dim = gym_env.action_space.shape[0]
    
    print(f"Environment: {env}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Episodes to generate: {episodes}")
    print("-" * 40)
    
    # Set random seed
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    # Generate trajectories
    episode_rewards = []
    
    with tf.Session() as sess:
        policy = PolicyNetwork(obs_dim, action_dim, sess, stochastic=stochastic)
        policy.load_parameters(policy_params)
        
        for ep in range(episodes):
            print(f"\\nEpisode {ep + 1}/{episodes}")
            
            # Reset environment
            reset_result = gym_env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            
            total_reward = 0
            step_count = 0
            
            for step in range(max_steps):
                if render:
                    gym_env.render()
                
                # Get action from policy
                action = policy.get_action(obs)
                
                # Take step in environment
                step_result = gym_env.step(action)
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                else:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                total_reward += reward
                step_count += 1
                obs = next_obs
                
                # Show progress every 200 steps
                if (step + 1) % 200 == 0:
                    print(f"  Step {step + 1}: Reward = {total_reward:.1f}")
                
                if done:
                    print(f"  Episode ended early at step {step + 1}")
                    break
            
            episode_rewards.append(total_reward)
            print(f"  Final reward: {total_reward:.1f}")
            print(f"  Steps completed: {step_count}")
            
            # Brief pause between episodes for visual clarity
            if render and ep < episodes - 1:
                print("  Press Enter for next episode...")
                try:
                    input()
                except KeyboardInterrupt:
                    break
    
    gym_env.close()
    
    # Summary
    print(f"\\n" + "=" * 40)
    print("TRAJECTORY GENERATION SUMMARY")
    print(f"Episodes completed: {len(episode_rewards)}")
    print(f"Average reward: {np.mean(episode_rewards):.1f}")
    print(f"Reward range: [{np.min(episode_rewards):.1f}, {np.max(episode_rewards):.1f}]")
    print(f"Reward std: {np.std(episode_rewards):.1f}")
    
    print(f"\\nThese are the expert demonstration rewards!")
    print(f"   Use this script for accurate trajectory visualization.")

if __name__ == '__main__':
    main()