from collections import deque

import numpy as np
import pickle

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
    """

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with MuJoCo environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        # For MuJoCo, we don't have info keys or goal structures
        self.info_keys = []

        self.Q_history = deque(maxlen=history_len)
        self.reward_history = deque(maxlen=history_len)  # Track episode returns instead of success

        self.n_episodes = 0
        # For MuJoCo: only observations, no goals or achieved goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment and updates the `initial_o` array accordingly.
        For MuJoCo environments, obs is directly the observation array, not a dict.
        """
        obs_result = self.envs[i].reset()  # This returns the observation array directly for MuJoCo
        if isinstance(obs_result, tuple):
            obs, info = obs_result  # New gym API
        else:
            obs = obs_result  # Old gym API
        self.initial_o[i] = obs

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly. Adapted for MuJoCo flat state observations.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        o[:] = self.initial_o

        # generate episodes
        obs, acts, rewards = [], [], []
        Qs = []
        episode_returns = np.zeros(self.rollout_batch_size)
        
        for t in range(self.T):
            # For MuJoCo flat state: pass observation directly, no goals
            # We need to create dummy goal arrays for compatibility with existing DDPG code
            dummy_goals = np.zeros((self.rollout_batch_size, max(1, self.dims['g'])), np.float32)
            
            policy_output = self.policy.get_actions(
                o, dummy_goals, dummy_goals,  # obs, achieved_goal, desired_goal (all dummies except obs)
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            rewards_step = np.zeros(self.rollout_batch_size)
            
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # For MuJoCo, we care about the reward (dense reward signal)
                    step_result = self.envs[i].step(u[i])
                    if len(step_result) == 5:  # New gym API
                        curr_o_new, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:  # Old gym API
                        curr_o_new, reward, done, info = step_result
                        
                    o_new[i] = curr_o_new  # Direct assignment for flat state
                    rewards_step[i] = reward
                    episode_returns[i] += reward
                    
                    if self.render:
                        self.envs[i].render()
                        
                    # If episode terminates early, reset for next episode
                    if done and t < self.T - 1:
                        obs_result = self.envs[i].reset()
                        if isinstance(obs_result, tuple):
                            curr_o_new, _ = obs_result
                        else:
                            curr_o_new = obs_result
                        o_new[i] = curr_o_new
                        self.reward_history.append(episode_returns[i])
                        episode_returns[i] = 0  # Reset for new episode
                        
                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            acts.append(u.copy())
            rewards.append(rewards_step.copy())
            o[...] = o_new
            
        obs.append(o.copy())
        self.initial_o[:] = o

        # Store final episode returns
        for i in range(self.rollout_batch_size):
            self.reward_history.append(episode_returns[i])

        # Create episode dict compatible with existing code structure
        # For MuJoCo, we create dummy goal arrays to maintain compatibility
        # Note: 'g' (goals) should have T timesteps, 'ag' (achieved goals) should have T+1 timesteps
        dummy_goals_episode = [np.zeros((self.rollout_batch_size, max(1, self.dims['g'])), np.float32) 
                               for _ in range(self.T)]
        dummy_achieved_goals_episode = [np.zeros((self.rollout_batch_size, max(1, self.dims['g'])), np.float32) 
                                        for _ in range(self.T + 1)]
        
        episode = dict(o=obs,
                       u=acts,
                       g=dummy_goals_episode,          # Dummy goals for compatibility
                       ag=dummy_achieved_goals_episode, # Dummy achieved goals for compatibility
                       r=rewards)                       # Include actual rewards for MuJoCo

        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
            
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.reward_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        """For MuJoCo, return mean episode return instead of success rate"""
        return np.mean(self.reward_history) if len(self.reward_history) > 0 else 0.0

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('mean_episode_return', np.mean(self.reward_history) if len(self.reward_history) > 0 else 0.0)]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            # Try new gymnasium style first, fall back to old gym style
            try:
                env.reset(seed=seed + 1000 * idx)
            except TypeError:
                # Fallback for older gym versions
                try:
                    env.seed(seed + 1000 * idx)
                except AttributeError:
                    # Environment doesn't support seeding, skip
                    pass