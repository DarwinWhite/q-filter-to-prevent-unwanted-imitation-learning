import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
from collections import OrderedDict

from src.utils.util import (store_args, to_tensor, to_numpy, polyak_update, hard_update, 
                           convert_episode_to_batch_major)
from src.utils.normalizer import Normalizer
from src.algorithms.replay_buffer import ReplayBuffer
from src.algorithms.actor_critic import ActorNetwork, CriticNetwork


def dims_to_shapes(input_dims, T):
    """Convert input dimensions to buffer shapes including time dimension."""
    shapes = {}
    for key, val in input_dims.items():
        if key == 'u':  # Actions have T timesteps
            shapes[key] = (T, val) if val > 0 else (T,)
        elif key in ['o', 'g']:  # Observations, goals have T+1 timesteps  
            shapes[key] = (T+1, val) if val > 0 else (T+1,)
        else:  # Other keys (like 'info') use original logic
            shapes[key] = (val,) if val > 0 else ()
    
    # Add 'ag' (achieved goals) - same shape as observations
    shapes['ag'] = shapes['o']
    
    return shapes


global demoBuffer


class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns,
                 clip_return, sample_transitions, gamma, bc_loss, q_filter, num_demo,
                 demo_batch_size, prm_loss_weight, aux_loss_weight, device='cpu', 
                 network_class=None, **kwargs):
        """Implementation of DDPG in PyTorch."""
        
        self.device = torch.device(device)
        self.create_actor_critic = True
        self.dimo = input_dims['o']
        self.dimg = input_dims['g']  
        self.dimu = input_dims['u']
        
        # Create actor and critic networks
        self.input_dim = self.dimo + self.dimg
        self.actor = ActorNetwork(self.input_dim, hidden, layers, self.dimu, max_u).to(self.device)
        self.critic = CriticNetwork(self.input_dim + self.dimu, hidden, layers).to(self.device)
        
        # Target networks
        self.target_actor = ActorNetwork(self.input_dim, hidden, layers, self.dimu, max_u).to(self.device)
        self.target_critic = CriticNetwork(self.input_dim + self.dimu, hidden, layers).to(self.device)
        
        # Initialize target networks
        hard_update(self.target_actor.parameters(), self.actor.parameters())
        hard_update(self.target_critic.parameters(), self.critic.parameters())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=pi_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=Q_lr)
        
        # Normalizers
        self.o_stats = Normalizer(self.dimo, eps=norm_eps, device=device)
        self.g_stats = Normalizer(self.dimg, eps=norm_eps, device=device)
        
        # Replay buffer
        buffer_shapes = dims_to_shapes(input_dims, T)
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, T, sample_transitions)
        
        # Demo buffer for Q-filter
        self.demo_buffer_size = num_demo
        self.demo_buffer = None
        
        # BC loss and Q-filter parameters (matching TensorFlow implementation)
        self.lambda1 = 0.001  # Weight for policy loss  
        self.lambda2 = 0.0078  # Weight for BC loss
        self.demo_batch_size = 128  # Size of demo batch
        
        # BC loss and Q-filter parameters
        self.lambda1 = 0.001  # Weight for policy loss
        self.lambda2 = 0.0078  # Weight for BC loss  
        self.demo_batch_size = 128  # Size of demo batch
        
    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        """Preprocess observations and goals."""
        if isinstance(o, torch.Tensor):
            o = to_numpy(o)
        if isinstance(ag, torch.Tensor):
            ag = to_numpy(ag)
        if isinstance(g, torch.Tensor):
            g = to_numpy(g)
            
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, ag, g

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        """Get actions from the actor network."""
        o, ag, g = self._preprocess_og(o, ag, g)
        
        # Ensure inputs have batch dimension
        single_input = len(o.shape) == 1
        if single_input:
            o = o.reshape(1, -1)
            ag = ag.reshape(1, -1)
            g = g.reshape(1, -1)
        
        # Convert to tensors
        o_tensor = to_tensor(o, self.device)
        g_tensor = to_tensor(g, self.device)
        
        # Normalize
        o_norm = self.o_stats.normalize(o_tensor)
        g_norm = self.g_stats.normalize(g_tensor)
        
        # Concatenate
        input_tensor = torch.cat([o_norm, g_norm], dim=-1)
        
        # Get actions
        if use_target_net:
            actor_net = self.target_actor
        else:
            actor_net = self.actor
            
        with torch.no_grad():
            u = actor_net(input_tensor)
            
        # Add noise
        u_numpy = to_numpy(u)
        
        # Random actions
        random_actions = self._random_action(u_numpy.shape[0])
        
        # Choose random vs policy actions
        if random_eps > 0:
            rand_mask = np.random.rand(u_numpy.shape[0]) < random_eps
            u_numpy[rand_mask] = random_actions[rand_mask]
        
        # Add noise
        if noise_eps > 0:
            noise = noise_eps * np.random.randn(*u_numpy.shape)
            u_numpy += noise
            
        # Clip actions
        u_numpy = np.clip(u_numpy, -self.max_u, self.max_u)
        
        # Compute Q values if requested
        q = None
        if compute_Q:
            u_tensor = to_tensor(u_numpy, self.device)
            critic_input = torch.cat([o_norm, g_norm, u_tensor / self.max_u], dim=-1)
            
            if use_target_net:
                critic_net = self.target_critic
            else:
                critic_net = self.critic
                
            with torch.no_grad():
                q = critic_net(critic_input)
            q = to_numpy(q)
            
        # Handle single input case - squeeze batch dimension
        if single_input:
            u_numpy = u_numpy.squeeze(0)
            if q is not None:
                q = q.squeeze(0)
            
        return u_numpy, q

    def initDemoBuffer(self, demoDataFile, update_stats=True):
        """Initialize demonstration buffer for Q-filter."""
        global demoBuffer
        
        # Load demonstration data
        demo_data = np.load(demoDataFile, allow_pickle=True)
            
        # Process demonstrations
        obs_data = demo_data['obs']
        acs_data = demo_data['acs']
        
        print(f"Loading {self.num_demo} demonstration episodes...")
        
        # Convert to episode format
        episodes = []
        num_episodes = min(self.num_demo, len(obs_data))
        
        for epsd in range(num_episodes):
            if epsd % 10 == 0:  # Progress indicator
                print(f"Loading episode {epsd+1}/{num_episodes}")
                
            episode_obs = obs_data[epsd]
            episode_actions = acs_data[epsd]
            
            # Extract observations from dict format
            obs = []
            for ep_obs in episode_obs:
                if isinstance(ep_obs, dict):
                    obs.append(ep_obs['observation'])
                else:
                    obs.append(ep_obs)
            obs = np.array(obs, dtype=np.float32)
            
            # Actions - ensure float32 dtype
            acs = np.array(episode_actions, dtype=np.float32)
            
            # Create episode
            episode = {}
            episode['o'] = obs
            episode['u'] = acs
            episode['ag'] = obs  # For goal-conditioned compatibility
            episode['g'] = np.zeros((len(obs), max(1, self.dimg)), dtype=np.float32)  # Dummy goals
            episode['info'] = np.empty((), dtype=np.float32)  # Empty scalar to match buffer shape
            episodes.append(episode)
            
        # Store demos
        episode_batch = {k: [] for k in episodes[0].keys()}
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key].append(episode[key])
                
        # Convert to batch format
        for key in episode_batch.keys():
            episode_batch[key] = np.array(episode_batch[key], dtype=np.float32)
            
        # Update statistics if requested
        if update_stats:
            for episode in episodes:
                # Update stats for observations
                for obs in episode['o']:
                    self.o_stats.update(obs)
                for goal in episode['g']:
                    self.g_stats.update(goal)
            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()
            
        # Store in demo buffer
        demoBuffer = episode_batch
        self.demo_buffer = episode_batch
        
        print(f"Loaded {len(episodes)} demonstration episodes")

    def store_episode(self, episode_batch, update_stats=True):
        """Store episode in replay buffer."""
        if update_stats:
            # Update normalizer statistics
            for key, values in episode_batch.items():
                if key == 'o':
                    self.o_stats.update(values.reshape(-1, values.shape[-1]))
                elif key == 'g':
                    self.g_stats.update(values.reshape(-1, values.shape[-1]))
                    
            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()
            
        self.buffer.store_episode(episode_batch)

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def sample_batch(self):
        """Sample batch from replay buffer."""
        if self.bc_loss and self.demo_buffer is not None:
            # Sample from both replay buffer and demo buffer  
            batch_size_rl = self.batch_size - self.demo_batch_size
            batch_size_demo = self.demo_batch_size
            
            # Sample from replay buffer
            if self.buffer.get_current_size() > 0:
                transitions_rl = self.buffer.sample(batch_size_rl)
            else:
                transitions_rl = None
                
            # Sample from demo buffer
            transitions_demo = self._sample_demo_batch(batch_size_demo)
            
            # Combine batches
            if transitions_rl is not None:
                transitions = {}
                for key in transitions_rl.keys():
                    transitions[key] = np.concatenate([
                        transitions_rl[key], transitions_demo[key]
                    ], axis=0)
            else:
                transitions = transitions_demo
                
            # Create mask: 0 for RL data, 1 for demo data
            mask = np.concatenate([
                np.zeros(batch_size_rl if transitions_rl is not None else 0),
                np.ones(batch_size_demo)
            ], axis=0)
            transitions['mask'] = mask
        else:
            # Sample only from replay buffer
            transitions = self.buffer.sample(self.batch_size)
            # No demo data, so mask is all zeros
            transitions['mask'] = np.zeros(self.batch_size)
            
        return transitions

    def _sample_demo_batch(self, batch_size):
        """Sample from demonstration buffer."""
        if self.demo_buffer is None:
            raise ValueError("Demo buffer not initialized")
            
        # Simple random sampling from demo buffer
        n_episodes = self.demo_buffer['o'].shape[0]
        T = self.demo_buffer['u'].shape[1]  # Use action time dimension (T, not T+1)
        
        episode_idxs = np.random.randint(0, n_episodes, batch_size)
        t_samples = np.random.randint(T, size=batch_size)  # 0 to T-1
        
        transitions = {}
        for key in self.demo_buffer.keys():
            if len(self.demo_buffer[key].shape) == 1:  # Handle 1D arrays (like 'info')
                transitions[key] = self.demo_buffer[key][episode_idxs]
            elif key in ['o', 'ag', 'g']:  # These have T+1 timesteps
                transitions[key] = self.demo_buffer[key][episode_idxs, t_samples]
            elif key == 'u':  # Actions have T timesteps
                transitions[key] = self.demo_buffer[key][episode_idxs, t_samples]
            else:
                transitions[key] = self.demo_buffer[key][episode_idxs, t_samples]
            
        # Add next state (t+1)
        t_next = np.minimum(t_samples + 1, self.demo_buffer['o'].shape[1] - 1)  # Clamp to valid range
        transitions['o_2'] = self.demo_buffer['o'][episode_idxs, t_next]
        transitions['ag_2'] = self.demo_buffer['ag'][episode_idxs, t_next]
        transitions['r'] = np.zeros((batch_size, 1))  # Dummy rewards
        
        return transitions

    def train(self, stage=True):
        """Train the DDPG agent."""
        if self.buffer.get_current_size() == 0:
            return
            
        # Sample batch
        batch = self.sample_batch()
        
        # Convert to tensors
        o = to_tensor(batch['o'], self.device)
        g = to_tensor(batch['g'], self.device) 
        u = to_tensor(batch['u'], self.device)
        o_2 = to_tensor(batch['o_2'], self.device)
        g_2 = to_tensor(batch['g'], self.device)  # Goals don't change
        r = to_tensor(batch['r'], self.device)
        mask = to_tensor(batch['mask'], self.device).bool()
        
        # Normalize observations
        o_norm = self.o_stats.normalize(o)
        g_norm = self.g_stats.normalize(g)
        o_2_norm = self.o_stats.normalize(o_2)
        g_2_norm = self.g_stats.normalize(g_2)
        
        # Critic loss
        with torch.no_grad():
            # Target actions
            target_input = torch.cat([o_2_norm, g_2_norm], dim=1)
            u_target = self.target_actor(target_input)
            
            # Target Q-values
            target_critic_input = torch.cat([o_2_norm, g_2_norm, u_target / self.max_u], dim=1)
            q_target = self.target_critic(target_critic_input)
            
            # Bellman target with clipping
            y = r + self.gamma * q_target
            if self.clip_return:
                y = torch.clamp(y, max=0.0)
            
        # Current Q-values
        critic_input = torch.cat([o_norm, g_norm, u / self.max_u], dim=1)
        q_current = self.critic(critic_input)
        
        # Critic loss
        critic_loss = F.mse_loss(q_current, y)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss with BC loss and Q-filtering
        actor_input = torch.cat([o_norm, g_norm], dim=1)
        u_policy = self.actor(actor_input)
        actor_critic_input = torch.cat([o_norm, g_norm, u_policy / self.max_u], dim=1)
        q_policy = self.critic(actor_critic_input)
        
        # Compute behavioral cloning loss (matching TensorFlow implementation)
        cloning_loss = torch.tensor(0.0, device=self.device)
        
        if self.bc_loss == 1 and self.q_filter == 1:
            # Q-filtering: only use demo actions where Q(s,a_demo) > Q(s,a_policy)
            if mask.any():
                demo_mask = mask
                q_demo = q_current[demo_mask]  # Q-values for demo actions
                q_policy_demo = q_policy[demo_mask]  # Q-values for policy actions on demo states
                
                # Q-filter: where demo action is better than policy action
                q_filter_mask = q_demo.squeeze() > q_policy_demo.squeeze()
                
                if q_filter_mask.any():
                    u_demo_filtered = u[demo_mask][q_filter_mask]
                    u_policy_filtered = u_policy[demo_mask][q_filter_mask]
                    cloning_loss = F.mse_loss(u_policy_filtered, u_demo_filtered, reduction='sum')
                    
        elif self.bc_loss == 1 and self.q_filter == 0:
            # Standard behavioral cloning on all demo data  
            if mask.any():
                u_demo = u[mask]
                u_policy_demo = u_policy[mask]
                cloning_loss = F.mse_loss(u_policy_demo, u_demo, reduction='sum')
        
        # Policy loss (matching TensorFlow weights and structure)
        actor_loss = -self.lambda1 * q_policy.mean()
        actor_loss += self.lambda1 * self.action_l2 * (u_policy / self.max_u).pow(2).mean()
        actor_loss += self.lambda2 * cloning_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        polyak_update(self.target_actor.parameters(), self.actor.parameters(), self.polyak)
        polyak_update(self.target_critic.parameters(), self.critic.parameters(), self.polyak)
        
        return critic_loss.item(), actor_loss.item()

    def update_target_net(self):
        """Update target networks using hard update."""
        hard_update(self.target_actor.parameters(), self.actor.parameters())
        hard_update(self.target_critic.parameters(), self.critic.parameters())

    def clear_buffer(self):
        """Clear the replay buffer."""
        self.buffer.clear_buffer()

    def save_policy(self, path):
        """Save the policy."""
        policy_data = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'o_stats': {
                'mean': to_numpy(self.o_stats.mean),
                'std': to_numpy(self.o_stats.std)
            },
            'g_stats': {
                'mean': to_numpy(self.g_stats.mean), 
                'std': to_numpy(self.g_stats.std)
            },
            'input_dims': {
                'o': self.dimo,
                'g': self.dimg,
                'u': self.dimu
            },
            'max_u': self.max_u
        }
        
        torch.save(policy_data, path)

    def load_policy(self, path):
        """Load a saved policy."""
        policy_data = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(policy_data['actor_state_dict'])
        self.critic.load_state_dict(policy_data['critic_state_dict'])
        
        # Load normalizer stats
        self.o_stats.mean = to_tensor(policy_data['o_stats']['mean'], self.device)
        self.o_stats.std = to_tensor(policy_data['o_stats']['std'], self.device)
        self.g_stats.mean = to_tensor(policy_data['g_stats']['mean'], self.device)
        self.g_stats.std = to_tensor(policy_data['g_stats']['std'], self.device)

    def logs(self):
        """Return logging information."""
        logs = {}
        logs.update(self.actor_optimizer.state_dict())
        logs.update(self.critic_optimizer.state_dict())
        return logs