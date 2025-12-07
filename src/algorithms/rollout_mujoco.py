from collections import deque
import numpy as np
import gymnasium as gym

# Handle optional mujoco_py import
try:
    from mujoco_py import MujocoException
except ImportError:
    # Create dummy exception if mujoco_py not available
    class MujocoException(Exception):
        pass

from src.utils.util import convert_episode_to_batch_major, store_args


class RolloutWorkerMuJoCo:
    """
    MuJoCo-specific rollout worker for flat state observations.
    Adapted from original RolloutWorker but simplified for non-goal-conditioned environments.
    PyTorch version with modern Gymnasium support.
    """

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """
        Initialize MuJoCo rollout worker.
        
        Args:
            make_env: Function to create environment
            policy: Policy object with get_actions method
            dims: Dictionary with dimensions
            logger: Logger object
            T: Episode length
            rollout_batch_size: Number of parallel rollouts
            exploit: Whether to use deterministic policy
            use_target_net: Whether to use target network
            compute_Q: Whether to compute Q-values
            noise_eps: Action noise level
            random_eps: Random action probability
            history_len: Length of success rate history
            render: Whether to render episodes
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        
        # Initialize observation arrays
        self.initial_o = np.empty((rollout_batch_size, dims['o']), dtype=np.float32)
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment and updates the `initial_o` array accordingly.
        For MuJoCo environments, obs is directly the observation array, not a dict.
        """
        try:
            obs_result = self.envs[i].reset()
            
            # Handle both old and new Gymnasium API
            if isinstance(obs_result, tuple):
                obs = obs_result[0]  # New API returns (observation, info)
            else:
                obs = obs_result     # Old API returns observation directly
                
            self.initial_o[i] = obs
        except Exception as e:
            self.logger.warn(f"Reset failed for environment {i}: {e}")
            # Create dummy observation if reset fails
            self.initial_o[i] = np.zeros(self.dims['o'])

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers."""
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly. Adapted for MuJoCo flat state observations.
        """
        self.reset_all_rollouts()

        # Initialize episode storage
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, dims['info']))
                      for dims in []]  # Empty for MuJoCo

        # Initialize current observations
        o = self.initial_o.copy()
        ag = o.copy()  # Use observation as achieved goal for compatibility
        g = np.zeros((self.rollout_batch_size, self.dims['g']))  # Proper goal dimensions for MuJoCo

        episode_rewards = np.zeros(self.rollout_batch_size)
        episode_lengths = np.zeros(self.rollout_batch_size)
        completed_episode_returns = []  # Track completed episode returns

        for t in range(self.T):
            # Get policy actions
            policy_output = self.policy.get_actions(
                o, ag, g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net
            )

            # DDPG always returns (actions, q_values) tuple
            u, Q = policy_output
            if self.compute_Q and Q is not None:
                self.Q_history.extend(Q.flatten())

            # Ensure actions are the right shape
            if u.ndim == 1:
                u = u.reshape(1, -1)

            # Step environments
            o_new = np.empty_like(o)
            rewards = np.zeros(self.rollout_batch_size)
            dones = np.zeros(self.rollout_batch_size, dtype=bool)

            for i in range(self.rollout_batch_size):
                try:
                    # Step environment
                    step_result = self.envs[i].step(u[i])
                    
                    # Handle both old and new Gymnasium API
                    if len(step_result) == 4:
                        # Old API: (obs, reward, done, info)
                        obs_new, r, done, info = step_result
                    elif len(step_result) == 5:
                        # New API: (obs, reward, terminated, truncated, info)
                        obs_new, r, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        raise ValueError(f"Unexpected step result length: {len(step_result)}")

                    o_new[i] = obs_new
                    rewards[i] = r
                    dones[i] = done

                    episode_rewards[i] += r
                    episode_lengths[i] = t + 1

                    if done and not self.exploit:
                        # Store completed episode return before reset
                        completed_episode_returns.append(episode_rewards[i])
                        # Reset environment if done (during training)
                        self.reset_rollout(i)
                        episode_rewards[i] = 0
                        episode_lengths[i] = 0

                    if self.render:
                        try:
                            self.envs[i].render()
                        except Exception as render_error:
                            # Silently ignore render errors to prevent crashes
                            pass

                except (MujocoException, Exception) as e:
                    self.logger.warn(f"Step failed for environment {i}: {e}")
                    # Use previous observation if step fails
                    o_new[i] = o[i]
                    rewards[i] = 0
                    dones[i] = False

            # Store transition
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            acts.append(u.copy())
            goals.append(g.copy())
            
            # For MuJoCo, success is based on reward (simplified)
            success = rewards > 0  # Simple success criterion
            successes.append(success.copy())

            # Update for next timestep
            o[:] = o_new
            ag[:] = o_new  # Use new observation as achieved goal
            # Goals stay the same (dummy goals)

        # Final observation
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        goals.append(g.copy())  # Add final goal to match observation timesteps

        # Convert to numpy arrays
        episode = dict(o=obs, u=acts, g=goals, ag=achieved_goals)
        for key, value in episode.items():
            episode[key] = np.array(value)

        # Store episode cumulative rewards
        episode['r'] = np.zeros((len(obs)-1, self.rollout_batch_size, 1))  # Buffer format requirement
        episode['info'] = np.zeros((self.rollout_batch_size,))  # Empty info array
        episode['success'] = np.array(successes).astype(np.float32)
        
        # Store actual episode returns for evaluation
        episode['episode_returns'] = episode_rewards

        # Convert to batch-major format
        episode = convert_episode_to_batch_major(episode)

        # Update history with completed episodes and final episode states
        all_episode_returns = list(completed_episode_returns) + list(episode_rewards)
        if all_episode_returns:  # Only extend if we have returns
            self.episode_rewards.extend(all_episode_returns)
        self.episode_lengths.extend(episode_lengths)

        # Compute success rate (simplified for MuJoCo)
        if all_episode_returns:
            success_rate = np.mean(np.array(all_episode_returns) > 1000)  # Threshold-based success
        else:
            success_rate = 0.0
        self.success_history.append(success_rate)

        return episode

    def clear_history(self):
        """Clear episode history."""
        self.success_history.clear()
        self.Q_history.clear()
        self.episode_rewards.clear()
        self.episode_lengths.clear()

    def current_success_rate(self):
        """Get current success rate."""
        return np.mean(self.success_history) if self.success_history else 0.0

    def current_mean_Q(self):
        """Get current mean Q-value."""
        return np.mean(self.Q_history) if self.Q_history else 0.0

    def save_policy(self, path):
        """Save policy."""
        if hasattr(self.policy, 'save_policy'):
            self.policy.save_policy(path)
        else:
            self.logger.warn("Policy does not support saving")

    def logs(self, prefix='worker'):
        """Get logging statistics."""
        logs = {}
        logs[prefix + '/success_rate'] = self.current_success_rate()
        logs[prefix + '/mean_Q'] = self.current_mean_Q()
        
        if self.episode_rewards:
            logs[prefix + '/mean_return'] = np.mean(self.episode_rewards[-100:])
            logs[prefix + '/std_return'] = np.std(self.episode_rewards[-100:])
            logs[prefix + '/max_return'] = np.max(self.episode_rewards[-100:])
            logs[prefix + '/min_return'] = np.min(self.episode_rewards[-100:])
        
        return logs

    def seed(self, seed):
        """Set random seed for environments."""
        for i, env in enumerate(self.envs):
            env.reset(seed=seed + i)