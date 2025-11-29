# src/algorithms/ddpg.py
import os
import json
import math
import copy
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.util import store_args
from src.utils.normalizer import Normalizer
from src.algorithms.replay_buffer import ReplayBuffer
from src.algorithms.actor_critic import ActorCritic


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class DDPG:
    """
    Clean, corrected, production-ready PyTorch DDPG
    suitable for MuJoCo dense-reward tasks.
    """

    @store_args
    def __init__(
        self,
        input_dims,
        buffer_size=int(1e6),
        hidden=256,
        layers=3,
        network_class=None,
        polyak=0.995,
        batch_size=256,
        Q_lr=1e-3,
        pi_lr=1e-3,
        norm_eps=1e-2,
        norm_clip=5.0,
        max_u=1.0,
        action_l2=0.1,
        clip_obs=200.0,
        scope='ddpg',
        T=1000,
        rollout_batch_size=1,
        subtract_goals=None,
        relative_goals=False,
        clip_pos_returns=True,
        clip_return=None,
        bc_loss=0,
        q_filter=0,
        num_demo=0,
        sample_transitions=None,
        gamma=0.99,
        device='cpu',
        demo_batch_size=128,
        lambda1=0.001,
        lambda2=0.0078,
        l2_reg_coeff=0.005,
        **kwargs
    ):
        self.device = torch.device(device)

        # dims
        self.input_dims = input_dims
        self.dimo  = int(input_dims["o"])
        self.dimu  = int(input_dims["u"])
        self.dimg  = int(input_dims.get("g", 0))

        # settings
        self.max_u = float(max_u)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.bc_loss = int(bc_loss)
        self.q_filter = int(q_filter)
        self.num_demo = int(num_demo)
        self.demo_batch_size = int(demo_batch_size)
        self.lambda2 = float(lambda2)
        self.T = int(T)
        self.clip_return = float(clip_return) if clip_return is not None else np.inf

        # normalizers
        self.o_stats = Normalizer(self.dimo, eps=norm_eps, default_clip_range=norm_clip)
        self.g_stats = Normalizer(max(1, self.dimg), eps=norm_eps, default_clip_range=norm_clip)

        # replay buffers
        size_in_transitions = int(buffer_size)

        buffer_shapes = {
            "o":  (self.T+1, self.dimo),
            "u":  (self.T,   self.dimu),
            "r":  (self.T,   1),
            "g":  (self.T,   max(1,self.dimg)),
            "ag": (self.T+1, max(1,self.dimg)),
        }

        self.buffer = ReplayBuffer(buffer_shapes, size_in_transitions, self.T, sample_transitions or (lambda x,n: x))
        self.demo_buffer = ReplayBuffer(buffer_shapes, size_in_transitions, self.T, sample_transitions or (lambda x,n: x))
        self.demo_loaded = False

        # networks
        ac_kwargs = dict(
            dimo=self.dimo,
            dimg=max(1, self.dimg),
            dimu=self.dimu,
            hidden=hidden,
            layers=layers,
            max_u=self.max_u,
            o_stats=self.o_stats,
            g_stats=self.g_stats,
        )

        self.main = ActorCritic(**ac_kwargs).to(self.device)
        self.target = copy.deepcopy(self.main).to(self.device)
        self.target.eval()

        # optimizers
        self.Q_optimizer = optim.Adam(self.main.critic.parameters(), lr=Q_lr)
        self.pi_optimizer = optim.Adam(self.main.actor.parameters(), lr=pi_lr)

        # logs
        self._last_critic_loss = 0
        self._last_actor_loss = 0
        self._last_cloning_loss = 0
        self.total_steps = 0


    # ------------ helpers ------------
    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _preprocess_obs(self, o):
        return np.clip(o, -self.clip_obs, self.clip_obs)


    # ------------ exposed to rollout worker ------------
    def get_actions(self, o, ag=None, g=None, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        """
        o: numpy (batch, obs_dim)
        returns numpy action
        """
        o = self._preprocess_obs(o)
        o_t = self._to_tensor(o.reshape(-1, self.dimo))

        # dummy goal
        g_t = torch.zeros((o_t.shape[0], max(1,self.dimg)), device=self.device)

        net = self.target if use_target_net else self.main

        with torch.no_grad():
            pi = net.pi(o_t, g_t)                    # tensor
            Q_pi = net.Q(o_t, g_t, pi) if compute_Q else None

        a = to_numpy(pi)

        # noise
        if noise_eps > 0:
            a += noise_eps * self.max_u * np.random.randn(*a.shape)

        # random eps
        if random_eps > 0:
            mask = (np.random.rand(a.shape[0]) < random_eps)
            if mask.any():
                rnd = np.random.uniform(-self.max_u, self.max_u, size=a.shape)
                a[mask] = rnd[mask]

        a = np.clip(a, -self.max_u, self.max_u)

        if a.shape[0] == 1:
            a = a[0]

        if compute_Q:
            return a, to_numpy(Q_pi).reshape(-1)
        return a


    # ------------ demo buffer ------------
    def initDemoBuffer(self, demo_file: str, update_stats=True):
        data = np.load(demo_file, allow_pickle=True)
        obs_all = data["obs"]
        acs_all = data["acs"]

        stored = 0
        for i in range(len(obs_all)):
            o_ep = np.asarray(obs_all[i], dtype=np.float32)
            u_ep = np.asarray(acs_all[i], dtype=np.float32)

            if o_ep.ndim != 2: continue
            T_eps = min(self.T, o_ep.shape[0]-1, u_ep.shape[0])
            if T_eps <= 0: continue

            o = np.zeros((1,self.T+1,self.dimo), np.float32)
            u = np.zeros((1,self.T,  self.dimu), np.float32)
            r = np.zeros((1,self.T,1),           np.float32)

            o[0,:T_eps+1] = o_ep[:T_eps+1,:self.dimo]
            u[0,:T_eps]   = u_ep[:T_eps,:self.dimu]
            o[0,T_eps+1:] = o[0,T_eps]

            ep = dict(o=o, u=u, r=r,
                      g=np.zeros((1,self.T,max(1,self.dimg)),np.float32),
                      ag=np.zeros((1,self.T+1,max(1,self.dimg)),np.float32))

            self.demo_buffer.store_episode(ep)
            stored += 1

        self.demo_loaded = stored > 0

        if update_stats and stored > 0:
            trans = self.demo_buffer.sample(min(self.batch_size, self.demo_buffer.get_current_size()*self.T))
            o = trans["o"]
            o = self._preprocess_obs(o)
            # some Normalizer implementations expect numpy arrays for update()
            try:
                self.o_stats.update(o)
                self.o_stats.recompute_stats()
            except Exception:
                # best-effort: if normalizer needs other handling, skip update here
                pass

        return stored


    # ------------ store episode ------------
    def store_episode(self, episode, update_stats=True):
        self.buffer.store_episode(episode)
        if update_stats:
            o = episode["o"][:, :-1, :]
            o = self._preprocess_obs(o)
            try:
                self.o_stats.update(o)
                self.o_stats.recompute_stats()
            except Exception:
                pass


    # ------------ sample batch ------------
    def sample_batch(self):
        if self.bc_loss and self.demo_loaded and self.demo_buffer.get_current_size() > 0:
            demo_n = min(self.demo_batch_size, self.batch_size)
            env_n = self.batch_size - demo_n

            if env_n > 0:
                env = self.buffer.sample(env_n)
                demo = self.demo_buffer.sample(demo_n)
                return {k: np.concatenate([env[k], demo[k]], axis=0) for k in env}
            else:
                return self.demo_buffer.sample(self.batch_size)

        return self.buffer.sample(self.batch_size)


    # ------------ training step ------------
    def train(self, n_grad_steps=1):
        self.main.train()

        for _ in range(n_grad_steps):
            trans = self.sample_batch()

            o   = self._to_tensor(trans["o"])
            o2  = self._to_tensor(trans["o_2"])
            u   = self._to_tensor(trans["u"])
            r   = self._to_tensor(trans["r"]).reshape(-1,1)

            # flatten if needed
            if o.ndim == 3:
                o  = o.reshape(-1, o.shape[-1])
                o2 = o2.reshape(-1, o2.shape[-1])
                u  = u.reshape(-1, u.shape[-1])
                r  = r.reshape(-1, 1)

            # normalize
            # many Normalizer implementations in this project expect numpy inputs for update/normalize
            try:
                o_n  = self._to_tensor(self.o_stats.normalize(o.cpu().numpy()))
                o2_n = self._to_tensor(self.o_stats.normalize(o2.cpu().numpy()))
            except Exception:
                # fallback: if normalizer already returns torch tensors
                o_n  = self._to_tensor(self.o_stats.normalize(o))
                o2_n = self._to_tensor(self.o_stats.normalize(o2))

            g_dummy = torch.zeros((o_n.shape[0], max(1,self.dimg)), device=self.device)

            # ----- critic -----
            with torch.no_grad():
                a2 = self.target.pi(o2_n, g_dummy)
                Q2 = self.target.Q(o2_n, g_dummy, a2)
                y  = r + self.gamma * Q2
                if not math.isinf(self.clip_return):
                    y = torch.clamp(y, -self.clip_return, self.clip_return)

            Q_pred = self.main.Q(o_n, g_dummy, u)
            critic_loss = nn.MSELoss()(Q_pred, y)

            self.Q_optimizer.zero_grad()
            critic_loss.backward()
            self.Q_optimizer.step()

            # ----- actor -----
            self.pi_optimizer.zero_grad()

            a_pi = self.main.pi(o_n, g_dummy)
            Q_pi = self.main.Q(o_n, g_dummy, a_pi)
            policy_loss = -torch.mean(Q_pi)
            policy_loss += self.action_l2 * torch.mean((a_pi/self.max_u)**2)

            # ----- BC / Q-filter -----
            cloning_loss = torch.tensor(0.0, device=self.device)
            if self.bc_loss and self.demo_loaded:
                demo_n = min(self.demo_batch_size, self.batch_size)
                if demo_n > 0:
                    o_demo = o_n[-demo_n:]
                    u_demo = u[-demo_n:]

                    a_demo_pred = self.main.pi(o_demo, g_dummy[:demo_n])

                    if self.q_filter:
                        Q_demo   = self.main.Q(o_demo, g_dummy[:demo_n], u_demo)
                        Q_policy = self.main.Q(o_demo, g_dummy[:demo_n], a_demo_pred)
                        mask = (Q_demo > Q_policy).float()
                        if mask.sum() > 0:
                            cloning_loss = ((a_demo_pred - u_demo)**2 * mask).sum() / (mask.sum()+1e-8)
                    else:
                        cloning_loss = torch.mean((a_demo_pred - u_demo)**2)

                    policy_loss += self.lambda2 * cloning_loss

            policy_loss.backward()
            self.pi_optimizer.step()

            # ----- polyak -----
            self.update_target_net()

            # logs
            self._last_critic_loss  = critic_loss.item()
            self._last_actor_loss   = policy_loss.item()
            self._last_cloning_loss = cloning_loss.item()
            self.total_steps += 1

        return self._last_critic_loss, self._last_actor_loss


    # ------------ target update ------------
    def update_target_net(self):
        with torch.no_grad():
            for p, tp in zip(self.main.parameters(), self.target.parameters()):
                tp.data.mul_(self.polyak)
                tp.data.add_((1-self.polyak) * p.data)


    # ------------ save/load (robust) ------------
    def save_policy(self, path):
        """Safely save the actor, critic, and normalizer stats."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        checkpoint = {}
        try:
            # networks
            checkpoint["actor_state_dict"] = self.main.actor.state_dict()
            checkpoint["critic_state_dict"] = self.main.critic.state_dict()
            # target optionally
            try:
                checkpoint["target_actor_state_dict"] = self.target.actor.state_dict()
                checkpoint["target_critic_state_dict"] = self.target.critic.state_dict()
            except Exception:
                pass

            # normalizers: try to use state_dict if present, otherwise save mean/std arrays if available
            def _extract_norm_stats(norm):
                if norm is None:
                    return None
                # prefer state_dict
                if hasattr(norm, "state_dict") and callable(norm.state_dict):
                    try:
                        return {"state_dict": norm.state_dict()}
                    except Exception:
                        pass
                # try mean/std attributes (numpy arrays or tensors)
                if hasattr(norm, "mean") and hasattr(norm, "std"):
                    try:
                        mean = getattr(norm, "mean")
                        std = getattr(norm, "std")
                        # if tensors, convert to numpy
                        if hasattr(mean, "detach") or isinstance(mean, torch.Tensor):
                            mean = mean.detach().cpu().numpy()
                        if hasattr(std, "detach") or isinstance(std, torch.Tensor):
                            std = std.detach().cpu().numpy()
                        return {"mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist()}
                    except Exception:
                        pass
                # last resort: try to pick useful attributes
                d = {}
                for attr in ["_sum", "sum", "local_sum", "local_sumsq", "local_count"]:
                    if hasattr(norm, attr):
                        try:
                            val = getattr(norm, attr)
                            if isinstance(val, (np.ndarray, list)):
                                d[attr] = np.asarray(val).tolist()
                        except Exception:
                            pass
                return d or None

            o_stats_data = _extract_norm_stats(self.o_stats)
            g_stats_data = _extract_norm_stats(self.g_stats)
            if o_stats_data is not None:
                checkpoint["o_stats"] = o_stats_data
            if g_stats_data is not None:
                checkpoint["g_stats"] = g_stats_data

            # config/meta
            checkpoint["config"] = {"input_dims": self.input_dims, "max_u": self.max_u}

            torch.save(checkpoint, path)
        except Exception as e:
            # safe warning; some callers expect logger.warn; fallback to print
            try:
                if hasattr(self, "logger") and hasattr(self.logger, "warn"):
                    self.logger.warn(f"Saving policy failed inside DDPG.save_policy: {e}")
                else:
                    print("Saving policy failed inside DDPG.save_policy:", e)
            except Exception:
                print("Saving policy failed (unexpected error while reporting):", e)


    def load_policy(self, path, map_location=None):
        """Safely load actor, critic, and normalizer stats."""
        ckpt = torch.load(path, map_location=map_location or self.device)

        # networks
        if "actor_state_dict" in ckpt and "critic_state_dict" in ckpt:
            try:
                self.main.actor.load_state_dict(ckpt["actor_state_dict"])
                self.main.critic.load_state_dict(ckpt["critic_state_dict"])
            except Exception as e:
                print("Failed to load actor/critic state_dicts:", e)
        else:
            # try older keys
            for k in ["main", "target", "actor", "critic"]:
                if k in ckpt and isinstance(ckpt[k], dict):
                    # heuristic: if provided a raw state_dict for actor or critic
                    try:
                        if "actor" in ckpt[k]:
                            self.main.actor.load_state_dict(ckpt[k]["actor"])
                        if "critic" in ckpt[k]:
                            self.main.critic.load_state_dict(ckpt[k]["critic"])
                    except Exception:
                        pass

        # normalizers
        def _load_norm(norm, payload):
            if norm is None or payload is None:
                return
            # If payload contains 'state_dict' and norm exposes load_state_dict
            if isinstance(payload, dict) and "state_dict" in payload and hasattr(norm, "load_state_dict"):
                try:
                    norm.load_state_dict(payload["state_dict"])
                    return
                except Exception:
                    pass
            # If payload contains mean/std lists
            if isinstance(payload, dict) and "mean" in payload and "std" in payload:
                try:
                    mean = np.asarray(payload["mean"], dtype=np.float32)
                    std = np.asarray(payload["std"], dtype=np.float32)
                    # try common attribute names
                    if hasattr(norm, "mean"):
                        try:
                            setattr(norm, "mean", mean)
                        except Exception:
                            pass
                    if hasattr(norm, "std"):
                        try:
                            setattr(norm, "std", std)
                        except Exception:
                            pass
                    # some normalizers provide a set_stats/set_state method
                    for meth in ("set_stats", "set_state", "load_state"):
                        if hasattr(norm, meth) and callable(getattr(norm, meth)):
                            try:
                                getattr(norm, meth)({"mean": mean, "std": std})
                                break
                            except Exception:
                                pass
                    return
                except Exception:
                    pass
            # fallback: try to set attributes directly for some common names
            for attr in ["local_sum", "local_sumsq", "local_count", "sum", "sumsq", "count"]:
                if attr in payload and hasattr(norm, attr):
                    try:
                        setattr(norm, attr, np.asarray(payload[attr], dtype=np.float32))
                    except Exception:
                        pass

        if "o_stats" in ckpt:
            _load_norm(self.o_stats, ckpt["o_stats"])
        if "g_stats" in ckpt:
            _load_norm(self.g_stats, ckpt["g_stats"])

        # Update target to match main after loading
        try:
            self.target = copy.deepcopy(self.main)
            self.target.eval()
        except Exception:
            pass


    def logs(self):
        return [
            ("train/critic_loss", self._last_critic_loss),
            ("train/actor_loss",  self._last_actor_loss),
            ("train/cloning_loss",self._last_cloning_loss),
            ("buffer/size",       self.buffer.get_current_size()),
            ("total_steps",       self.total_steps)
        ]
    
    def get_current_buffer_size(self):
        return self.buffer.get_current_size()
