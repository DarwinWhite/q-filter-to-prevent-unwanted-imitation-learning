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


class RolloutWorker:
    """Goal-conditioned rollout worker for PyTorch implementation."""

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """
        Initialize goal-conditioned rollout worker.
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        
        # Initialize observation arrays
        self.initial_o = np.empty((rollout_batch_size, dims['o']), dtype=np.float32)
        self.g = np.empty((rollout_batch_size, dims['g']), dtype=np.float32)
        
        # Episode tracking
        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        try:
            obs_result = self.envs[i].reset()
            
            # Handle both old and new Gymnasium API
            if isinstance(obs_result, tuple):
                obs = obs_result[0]  # New API returns (observation, info)
            else:
                obs = obs_result     # Old API returns observation directly
                
            self.initial_o[i] = obs['observation']
            self.g[i] = obs['desired_goal']
        except Exception as e:
            self.logger.warn(f"Reset failed for environment {i}: {e}")
            # Create dummy observation if reset fails
            self.initial_o[i] = np.zeros(self.dims['o'])
            self.g[i] = np.zeros(self.dims['g'])

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers."""
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # Initialize episode storage
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []

        # Initialize current observations
        o = self.initial_o.copy()
        ag = o.copy()  # Assume achieved goal equals observation initially
        g = self.g.copy()

        for t in range(self.T):
            # Get policy actions
            policy_output = self.policy.get_actions(
                o, ag, g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net
            )

            if self.compute_Q:
                u, Q = policy_output
                self.Q_history.extend(Q.flatten())
            else:
                u = policy_output

            # Ensure actions are the right shape
            if u.ndim == 1:
                u = u.reshape(1, -1)

            # Step environments
            o_new = np.empty_like(o)
            ag_new = np.empty_like(ag)
            success = np.zeros(self.rollout_batch_size)

            for i in range(self.rollout_batch_size):
                try:
                    # Step environment
                    step_result = self.envs[i].step(u[i])
                    
                    # Handle both old and new Gymnasium API
                    if len(step_result) == 4:
                        obs_new, r, done, info = step_result
                    elif len(step_result) == 5:
                        obs_new, r, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        raise ValueError(f"Unexpected step result length: {len(step_result)}")

                    o_new[i] = obs_new['observation']
                    ag_new[i] = obs_new['achieved_goal']
                    success[i] = info['is_success']

                    if self.render:
                        self.envs[i].render()

                except (MujocoException, Exception) as e:
                    self.logger.warn(f"Step failed for environment {i}: {e}")
                    # Use previous observation if step fails
                    o_new[i] = o[i]
                    ag_new[i] = ag[i]
                    success[i] = 0.0

            # Store transition
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            acts.append(u.copy())
            goals.append(g.copy())
            successes.append(success.copy())

            # Update for next timestep
            o[:] = o_new
            ag[:] = ag_new

        # Final observation
        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        # Convert to numpy arrays
        episode = dict(o=obs, u=acts, g=goals, ag=achieved_goals)
        for key, value in episode.items():
            episode[key] = np.array(value)

        # Add success info
        episode['success'] = np.array(successes)

        # Convert to batch-major format
        episode = convert_episode_to_batch_major(episode)

        # Update success history
        final_success_rate = np.mean(episode['success'][-1])
        self.success_history.append(final_success_rate)

        return episode

    def clear_history(self):
        """Clear episode history."""
        self.success_history.clear()
        self.Q_history.clear()

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

    def logs(self, prefix='worker'):
        """Get logging statistics."""
        logs = {}
        logs[prefix + '/success_rate'] = self.current_success_rate()
        logs[prefix + '/mean_Q'] = self.current_mean_Q()
        return logs

    def seed(self, seed):
        """Set random seed for environments."""
        for i, env in enumerate(self.envs):
            env.reset(seed=seed + i)