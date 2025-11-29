from collections import deque

import numpy as np
import pickle

# Optional torch import for tensor detection
try:
    import torch
except Exception:
    torch = None

# Handle optional mujoco_py import
try:
    from mujoco_py import MujocoException
except ImportError:
    # Create dummy exception if mujoco_py not available
    class MujocoException(Exception):
        pass

from src.utils.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, madeEnv, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            madeEnv : for gazebo envs multiple envs are not possible yet, so we pass the same environment
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
        #self.envs = [make_env() for _ in range(rollout_batch_size)] #comment out when a way to use multiple gazebo envs simulataneously is found
        self.envs = [madeEnv]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        reset_res = self.envs[i].reset()
        if isinstance(reset_res, tuple):
            obs, _ = reset_res
        else:
            obs = reset_res

        # Support both dict-returning envs and plain observation arrays
        if isinstance(obs, dict):
            self.initial_o[i] = obs.get('observation', obs)
            self.initial_ag[i] = obs.get('achieved_goal', np.zeros_like(self.initial_ag[i]))
            self.g[i] = obs.get('desired_goal', np.zeros_like(self.g[i]))
        else:
            # If environment returns plain arrays, just place them into observation slot
            # Achieved goal and goal remain zeros if not present
            self.initial_o[i] = obs

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def _to_numpy(self, x):
        """Convert torch Tensor -> numpy, otherwise return as-is."""
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Q = self._to_numpy(Q)
                Qs.append(Q)
            else:
                u = policy_output

            u = self._to_numpy(u)

            if u is None:
                raise RuntimeError("policy.get_actions returned None")

            if isinstance(u, np.ndarray) and u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            # Ensure u is array-like for indexing; handle lists as well
            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # step may return 4- or 5-element tuple depending on Gym version
                    step_result = self.envs[i].step(u[i])
                    if len(step_result) == 5:
                        curr_o_new, _, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        curr_o_new, _, done, info = step_result

                    # Support both dict observation and tuple obs
                    if isinstance(curr_o_new, dict):
                        if 'is_success' in info:
                            success[i] = info.get('is_success', 0.0)
                        o_new[i] = curr_o_new.get('observation', o_new[i])
                        ag_new[i] = curr_o_new.get('achieved_goal', ag_new[i])
                    else:
                        # if the env returns raw observations, try to fill observation slot
                        o_new[i] = curr_o_new

                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info.get(key, 0.0)

                    if self.render:
                        self.envs[i].render()
                except MujocoException:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                # Use logger if present else print
                try:
                    self.logger.warn('NaN caught during rollout generation. Trying again...')
                except Exception:
                    print('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy() if isinstance(u, np.ndarray) else np.array(u))
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q and len(Qs) > 0:
            # Qs is a list of arrays; store mean of flattened Qs
            qvals = np.array(Qs)
            try:
                self.Q_history.append(float(np.mean(qvals)))
            except Exception:
                # fallback: try to coerce
                self.Q_history.append(float(np.mean(self._to_numpy(Qs))))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history) if len(self.success_history) > 0 else 0.0

    def current_mean_Q(self):
        return np.mean(self.Q_history) if len(self.Q_history) > 0 else 0.0

    def save_policy(self, path):
        """Save policy via the policy's own save_policy method when available."""
        try:
            if hasattr(self.policy, 'save_policy'):
                self.policy.save_policy(path)
                return
            # fallback: attempt to save state_dict (PyTorch) or pickle
            import torch, pickle, os
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            if hasattr(self.policy, 'state_dict'):
                torch.save(self.policy.state_dict(), path)
                return
            with open(path, 'wb') as f:
                pickle.dump(self.policy, f)
        except Exception as e:
            try:
                self.logger.warn("Saving policy failed: ", e)
            except Exception:
                print("Saving policy failed:", e)


    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history) if len(self.success_history) > 0 else 0.0)]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history) if len(self.Q_history) > 0 else 0.0)]
        logs += [('episode', self.n_episodes)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            # Prefer new Gym API: reset(seed=...) then use that env
            try:
                env.reset(seed=seed + 1000 * idx)
            except TypeError:
                # Fallback: older gym envs
                try:
                    env.seed(seed + 1000 * idx)
                except AttributeError:
                    pass


class RolloutWorkerOriginal:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        reset_res = self.envs[i].reset()
        if isinstance(reset_res, tuple):
            obs, _ = reset_res
        else:
            obs = reset_res

        if isinstance(obs, dict):
            self.initial_o[i] = obs.get('observation', obs)
            self.initial_ag[i] = obs.get('achieved_goal', np.zeros_like(self.initial_ag[i]))
            self.g[i] = obs.get('desired_goal', np.zeros_like(self.g[i]))
        else:
            self.initial_o[i] = obs

    def reset_all_rollouts(self):
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def _to_numpy(self, x):
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def generate_rollouts(self):
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Q = self._to_numpy(Q)
                Qs.append(Q)
            else:
                u = policy_output

            u = self._to_numpy(u)

            if u is None:
                raise RuntimeError("policy.get_actions returned None")

            if isinstance(u, np.ndarray) and u.ndim == 1:
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            for i in range(self.rollout_batch_size):
                try:
                    step_result = self.envs[i].step(u[i])
                    if len(step_result) == 5:
                        curr_o_new, _, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        curr_o_new, _, done, info = step_result

                    if isinstance(curr_o_new, dict):
                        if 'is_success' in info:
                            success[i] = info.get('is_success', 0.0)
                        o_new[i] = curr_o_new.get('observation', o_new[i])
                        ag_new[i] = curr_o_new.get('achieved_goal', ag_new[i])
                    else:
                        o_new[i] = curr_o_new

                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info.get(key, 0.0)

                    if self.render:
                        self.envs[i].render()
                except MujocoException:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                try:
                    self.logger.warn('NaN caught during rollout generation. Trying again...')
                except Exception:
                    print('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy() if isinstance(u, np.ndarray) else np.array(u))
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q and len(Qs) > 0:
            qvals = np.array(Qs)
            try:
                self.Q_history.append(float(np.mean(qvals)))
            except Exception:
                self.Q_history.append(float(np.mean(self._to_numpy(Qs))))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history) if len(self.success_history) > 0 else 0.0

    def current_mean_Q(self):
        return np.mean(self.Q_history) if len(self.Q_history) > 0 else 0.0

    def save_policy(self, path):
        """Save policy via the policy's own save_policy method when available."""
        try:
            if hasattr(self.policy, 'save_policy'):
                self.policy.save_policy(path)
                return
            import torch, pickle, os
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            if hasattr(self.policy, 'state_dict'):
                torch.save(self.policy.state_dict(), path)
                return
            with open(path, 'wb') as f:
                pickle.dump(self.policy, f)
        except Exception as e:
            try:
                self.logger.warn("Saving policy failed: ", e)
            except Exception:
                print("Saving policy failed:", e)


    def logs(self, prefix='worker'):
        logs = []
        logs += [('success_rate', np.mean(self.success_history) if len(self.success_history) > 0 else 0.0)]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history) if len(self.Q_history) > 0 else 0.0)]
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
