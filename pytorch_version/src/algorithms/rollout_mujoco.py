from collections import deque

import numpy as np
import pickle

# Handle optional mujoco_py import
try:
    from mujoco_py import MujocoException
except ImportError:
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

        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = []

        self.Q_history = deque(maxlen=history_len)
        self.reward_history = deque(maxlen=history_len)

        self.n_episodes = 0

        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        obs_result = self.envs[i].reset()
        if isinstance(obs_result, tuple):
            obs, _ = obs_result
        else:
            obs = obs_result
        self.initial_o[i] = obs

    def reset_all_rollouts(self):
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        self.reset_all_rollouts()

        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        o[:] = self.initial_o

        obs, acts, rewards = [], [], []
        Qs = []
        episode_returns = np.zeros(self.rollout_batch_size)

        for t in range(self.T):
            dummy_goals = np.zeros((self.rollout_batch_size, max(1, self.dims['g'])), np.float32)

            policy_output = self.policy.get_actions(
                o, dummy_goals, dummy_goals,
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
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            rewards_step = np.zeros((self.rollout_batch_size, 1))

            for i in range(self.rollout_batch_size):
                try:
                    step_result = self.envs[i].step(u[i])
                    if len(step_result) == 5:
                        curr_o_new, reward, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    else:
                        curr_o_new, reward, done, _ = step_result

                    o_new[i] = curr_o_new
                    rewards_step[i, 0] = reward
                    episode_returns[i] += reward

                    if self.render:
                        self.envs[i].render()

                    if done and t < self.T - 1:
                        obs_result = self.envs[i].reset()
                        if isinstance(obs_result, tuple):
                            curr_o_new, _ = obs_result
                        else:
                            curr_o_new = obs_result
                        o_new[i] = curr_o_new
                        self.reward_history.append(episode_returns[i])
                        episode_returns[i] = 0

                except MujocoException:
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

        for i in range(self.rollout_batch_size):
            self.reward_history.append(episode_returns[i])

        dummy_goals_episode = [
            np.zeros((self.rollout_batch_size, max(1, self.dims['g'])), np.float32)
            for _ in range(self.T)
        ]
        dummy_achieved_goals_episode = [
            np.zeros((self.rollout_batch_size, max(1, self.dims['g'])), np.float32)
            for _ in range(self.T + 1)
        ]

        episode = dict(
            o=obs,
            u=acts,
            g=dummy_goals_episode,
            ag=dummy_achieved_goals_episode,
            r=rewards
        )

        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))

        self.n_episodes += self.rollout_batch_size
        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        self.reward_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.reward_history) if len(self.reward_history) > 0 else 0.0

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Save policy in a safe way.

        Use the policy's own save_policy(path) if available to avoid pickling locks.
        Falls back to pickling only a minimal representation if necessary.
        """
        try:
            # If the policy implements a safe saver, prefer that.
            if hasattr(self.policy, 'save_policy'):
                self.policy.save_policy(path)
                return
            # Fallback: try to pickle only the policy's state_dict if available.
            import torch, pickle, os
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            if hasattr(self.policy, 'state_dict'):
                torch.save(self.policy.state_dict(), path)
                return
            # Last resort: attempt to pickle the object (may fail due to locks)
            with open(path, 'wb') as f:
                pickle.dump(self.policy, f)
        except Exception as e:
            # Surface the error via logger but don't crash the whole job
            try:
                self.logger.warn("Saving policy failed: ", e)
            except Exception:
                print("Saving policy failed:", e)


    def logs(self, prefix='worker'):
        logs = []
        logs += [('mean_episode_return',
                  np.mean(self.reward_history) if len(self.reward_history) > 0 else 0.0)]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        for idx, env in enumerate(self.envs):
            try:
                env.reset(seed=seed + 1000 * idx)
            except TypeError:
                try:
                    env.seed(seed + 1000 * idx)
                except AttributeError:
                    pass
