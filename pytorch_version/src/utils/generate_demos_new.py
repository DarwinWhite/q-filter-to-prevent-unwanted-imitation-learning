#!/usr/bin/env python3
"""
Generate .npz demonstration files from saved policy parameter files.

- Attempts to load a saved policy file (pickle or torch checkpoint).
- If it finds a dict of numpy arrays or a torch checkpoint, it will try to
  reconstruct a simple PyTorch MLP actor by inferring layer sizes from the
  weight matrices. It will also accept a torch.nn.Module or ScriptModule.
- Falls back to a random policy if reconstruction fails.
- Saves demonstrations in the same format as the existing demo files:
    obs: object array of shape (n_eps,) where each element is (T+1, obs_dim)
    acs: object array of shape (n_eps,) where each element is (T, action_dim)
    rewards: object array of shape (n_eps,) where each element is (T,1)
    info: object array (per-episode metadata)
    env_name: string
"""
import os
import sys
import json
import pickle
import argparse
import warnings
from typing import Optional, Any, Dict, Tuple
import numpy as np

# try to use torch if available (helps with torch.load and using nn.Module)
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

import gym

# add project root so imports from src.* work if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------
# Utilities: loader & adaptor
# ---------------------------
def safe_pickle_load(path: str):
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception as e:
            raise

def safe_torch_load(path: str):
    if torch is None:
        raise RuntimeError("torch not available")
    # map to CPU for portability
    return torch.load(path, map_location='cpu')

def try_load_policy_file(path: str) -> Tuple[Optional[Any], Optional[dict]]:
    """
    Try a few ways of loading the policy/params file.
    Returns (loaded_obj_or_None, params_dict_or_None)
    """
    # 1) try Python pickle
    try:
        loaded = safe_pickle_load(path)
        return loaded, None
    except Exception as e_pickle:
        # print("pickle failed:", e_pickle)  # we'll be quiet; caller can log
        pass

    # 2) try torch.load
    if torch is not None:
        try:
            loaded = safe_torch_load(path)
            return loaded, None
        except Exception:
            pass

    # 3) some pickles contain persistent_id -> can't load w/ pickle directly.
    #    nothing else to try here: caller will fallback to random.
    return None, None

def infer_mlp_from_weight_dict(wdict: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
    """
    Try to infer an MLP architecture (sequence of Linear layers) from a dict
    mapping param names -> numpy arrays (weights and biases).
    Approach:
      - find all 2D arrays (candidates for linear weight matrices)
      - sort them by key (stable best-effort ordering)
      - treat them as sequential Linear layers: for each weight W of shape (out, in)
        build Linear(in, out). Find matching bias arrays by searching for 1D arrays
        whose size == out.
    Returns:
      dict: {'layer_shapes': [ (in,out), ... ], 'weights': [W,...], 'biases':[b_or_None,...]}
      or None if not enough structure found.
    """
    # Collect 2D weight candidates
    weight_items = []
    bias_map = {}
    for k, v in wdict.items():
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.ndim == 2:
            weight_items.append((k, arr))
        elif arr.ndim == 1:
            bias_map.setdefault(arr.shape[0], []).append((k, arr))

    if not weight_items:
        return None

    # Sort weight items by key string (best-effort stable order)
    weight_items.sort(key=lambda t: t[0])

    # Extract shapes chain
    layer_shapes = []
    weights = []
    biases = []
    for k, W in weight_items:
        out, inp = W.shape
        layer_shapes.append((inp, out))
        weights.append(W.copy())
        # attempt to find bias of matching out dim (prefer same key with 'b' in name)
        found_bias = None
        # 1) key variants
        candidates = []
        for suffix in ['bias', 'b', 'biases', 'Bias', 'Biases']:
            bk = k + '.' + suffix
            if bk in wdict:
                candidates.append((bk, np.asarray(wdict[bk])))
        # 2) exact size matches
        if found_bias is None:
            list_by_size = bias_map.get(out, [])
            if list_by_size:
                # pick first unused bias with same size
                found_bias = list_by_size[0][1].copy()
            else:
                found_bias = None

        biases.append(found_bias)

    return {
        'layer_shapes': layer_shapes,
        'weights': weights,
        'biases': biases,
        'weight_keys': [k for k,_ in weight_items],
    }

class ReconstructedActor:
    """Simple wrapper around a reconstructed PyTorch MLP actor (if torch is available)."""
    def __init__(self, layer_shapes, weights, biases, device='cpu'):
        assert torch is not None, "PyTorch required to instantiate reconstructed actor"
        layers = []
        self.device = device
        for i,(inp,out) in enumerate(layer_shapes):
            lin = nn.Linear(inp, out)
            layers.append(lin)
            # add activation for all but last
            if i < len(layer_shapes)-1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers).to(device)
        # load weights/biases
        linear_modules = [m for m in self.net if isinstance(m, nn.Linear)]
        for i, lin in enumerate(linear_modules):
            W = weights[i]
            b = biases[i]
            # torch Linear expects weight shape (out, in) — W is expected in this shape
            try:
                lin.weight.data.copy_(torch.tensor(W, dtype=torch.float32))
            except Exception:
                # try transposed fallback
                lin.weight.data.copy_(torch.tensor(W.T, dtype=torch.float32))
            if b is not None:
                lin.bias.data.copy_(torch.tensor(np.asarray(b).reshape(-1), dtype=torch.float32))
        self.net.eval()

    def get_actions(self, obs: np.ndarray, ag=None, g=None, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        # obs: batch (B, obs_dim) or single (obs_dim,)
        arr = np.asarray(obs)
        if arr.ndim == 1:
            arr = arr[None, ...]
        x = torch.tensor(arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.net(x)
        a = out.cpu().numpy()
        # apply tanh to final layer to be safe (action scaling will be handled outside)
        a = np.tanh(a)
        # optionally add small gaussian noise for stochasticity
        if noise_eps and noise_eps > 0:
            a = a + noise_eps * np.random.randn(*a.shape)
        return a, None

# ---------------------------
# Demo collection helpers
# ---------------------------
def collect_rollouts(env, policy, n_episodes: int, max_steps: Optional[int] = None, render: bool = False):
    """
    Collect n_episodes using policy.get_actions(obs_batch, ...) or policy(obs).
    Returns: list of episode dicts consistent with your training code:
      episode['o']: list length T+1 of observations arrays  (we will return arrays for saving)
      episode['u']: list length T of actions arrays
      episode['r']: list length T of rewards (scalars)
    """
    episodes = []
    action_dim = env.action_space.shape[0]
    for ep in range(n_episodes):
        reset_res = env.reset()
        obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
        obs_list = [np.array(obs, copy=True)]
        ac_list = []
        rew_list = []
        Tcounter = 0
        done = False
        while True:
            # Prepare batch input for get_actions (RolloutWorker used batch format)
            # We'll send as (1, obs_dim)
            try:
                a_out = policy.get_actions(np.asarray(obs).reshape(1,-1), compute_Q=False)
            except Exception:
                # fallback if policy expects a different signature
                try:
                    a_out = policy(np.asarray(obs).reshape(1,-1))
                except Exception:
                    # final fallback: random
                    a_out = (np.random.uniform(-1,1,(1,action_dim)), None)
            # a_out may be tuple (actions, Q) or actions
            if isinstance(a_out, tuple) or isinstance(a_out, list):
                a = a_out[0]
            else:
                a = a_out
            a = np.asarray(a)
            if a.ndim == 2 and a.shape[0] == 1:
                a_step = a[0]
            else:
                # try flatten
                a_step = a.reshape(-1)
            # step environment (support old/new APIs)
            step_res = env.step(a_step)
            if len(step_res) == 4:
                next_obs, reward, done, info = step_res
            else:
                next_obs, reward, terminated, truncated, info = step_res
                done = bool(terminated or truncated)
            obs_list.append(np.array(next_obs, copy=True))
            ac_list.append(np.array(a_step, copy=True))
            rew_list.append(float(reward))
            Tcounter += 1
            obs = next_obs
            if render:
                # many gym envs require calling render() without args
                try:
                    env.render()
                except Exception:
                    pass
            # termination / max steps
            if done:
                break
            if max_steps is not None and Tcounter >= max_steps:
                break
        # convert to arrays with shapes compatible with your previous .npz format
        obs_arr = np.asarray(obs_list)           # (T+1, obs_dim)
        acs_arr = np.asarray(ac_list)           # (T, action_dim)
        rews_arr = np.asarray(rew_list).reshape(-1,1)  # (T,1)
        episodes.append({'obs': obs_arr, 'acs': acs_arr, 'rewards': rews_arr, 'info': None})
    return episodes

# def episodes_to_npz_dict(episodes: list, env_name: str) -> Dict[str, Any]:
#     obs_list = [ep['obs'] for ep in episodes]
#     acs_list = [ep['acs'] for ep in episodes]
#     rewards_list = [ep['rewards'] for ep in episodes]
#     info_list = [ep.get('info', None) for ep in episodes]
#     return {
#         'obs': np.array(obs_list, dtype=object),
#         'acs': np.array(acs_list, dtype=object),
#         'rewards': np.array(rewards_list, dtype=object),
#         'info': np.array(info_list, dtype=object),
#         'env_name': np.array(env_name)
#     }
def episodes_to_npz_dict(episodes: list, env_name: str) -> Dict[str, Any]:
    obs_list = []
    acs_list = []
    rewards_list = []
    info_list = []

    for ep_idx, ep in enumerate(episodes):
        obs_arr = ep['obs']               # (T+1, obs_dim)
        acs_arr = ep['acs']               # (T, act_dim)
        rewards_arr = ep['rewards']       # (T, 1)

        T = acs_arr.shape[0]

        # --- NEW: Build correct info array ---
        info_arr = [
            {'episode': ep_idx, 'step': t}
            for t in range(T)
        ]

        obs_list.append(obs_arr)
        acs_list.append(acs_arr)
        rewards_list.append(rewards_arr)
        info_list.append(info_arr)

    return {
        'obs': np.array(obs_list, dtype=object),
        'acs': np.array(acs_list, dtype=object),
        'rewards': np.array(rewards_list, dtype=object),
        'info': np.array(info_list, dtype=object),
        'env_name': np.array(env_name)
    }


# ---------------------------
# Main CLI
# ---------------------------
def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_file", type=str, help="Path to .pkl or torch file containing policy params")
    parser.add_argument("--env", type=str, default=None, help="Gym env id; if omitted, script will try to read params.json next to policy file")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="Output .npz path (defaults to <policy_basename>_demos.npz)")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    policy_path = args.policy_file
    if not os.path.exists(policy_path):
        print(f"Policy file not found: {policy_path}")
        return

    # try read params.json next to policy_file
    policy_dir = os.path.dirname(policy_path)
    env_name = args.env
    params = None
    if env_name is None:
        params_path = os.path.join(policy_dir, 'params.json')
        if os.path.exists(params_path):
            try:
                with open(params_path,'r') as f:
                    params = json.load(f)
                    env_name = params.get('env_name', None)
            except Exception:
                params = None
    if env_name is None:
        env_name = 'HalfCheetah-v4'
        print("No env specified and no params.json found, defaulting to HalfCheetah-v4")

    # Create env
    env = gym.make(env_name)
    print(f"Using env: {env_name}")

    # try load policy file
    loaded, _ = None, None
    try:
        loaded, _ = try_load_policy_file(policy_path)
    except Exception as e:
        print("Policy load attempts failed:", e)
        loaded = None

    policy_obj = None
    # If it's a torch module, use it (wrap if needed)
    if torch is not None and isinstance(loaded, torch.nn.Module):
        print("Loaded torch.nn.Module — using it directly")
        policy_obj = loaded
        # ensure method get_actions exists; if not, wrap
        if not hasattr(policy_obj, 'get_actions'):
            class ModuleWrapper:
                def __init__(self, mod):
                    self.mod = mod
                    try:
                        self.mod.eval()
                    except Exception:
                        pass
                def get_actions(self, obs, ag=None, g=None, **kwargs):
                    arr = np.asarray(obs)
                    if arr.ndim == 1:
                        arr = arr[None, ...]
                    with torch.no_grad():
                        out = self.mod(torch.tensor(arr, dtype=torch.float32))
                    if isinstance(out, torch.Tensor):
                        return out.cpu().numpy(), None
                    if isinstance(out, (list, tuple)):
                        return np.asarray(out[0]), None
                    return np.asarray(out), None
            policy_obj = ModuleWrapper(policy_obj)

    # If loaded is a dict, try to reconstruct DDPG actor or actor MLP
    if policy_obj is None and isinstance(loaded, dict):
        # If torch checkpoint-like dict with 'actor_state_dict' etc, try to handle that
        print("Loaded a dict from the policy file; attempting to reconstruct an MLP actor from numeric arrays.")
        # convert any torch tensors to numpy arrays
        numeric_map = {}
        for k,v in loaded.items():
            try:
                if torch is not None and isinstance(v, torch.Tensor):
                    numeric_map[k] = v.detach().cpu().numpy()
                else:
                    numeric_map[k] = np.asarray(v)
            except Exception:
                # skip non-array fields
                pass

        mlp_info = infer_mlp_from_weight_dict(numeric_map)
        if mlp_info is not None and torch is not None:
            try:
                actor = ReconstructedActor(mlp_info['layer_shapes'], mlp_info['weights'], mlp_info['biases'], device='cpu')
                policy_obj = actor
                print("Successfully reconstructed an MLP actor (best-effort).")
            except Exception as e:
                print("Failed to instantiate reconstructed actor:", e)
                policy_obj = None
        else:
            print("Could not infer an MLP structure from the checkpoint dict.")

    # If still no policy, fallback to RandomPolicy
    if policy_obj is None:
        print("⚠️ Could not build a usable policy from the file. Falling back to random policy.")
        class RandomPolicy:
            def __init__(self, action_dim):
                self.action_dim = action_dim
            def get_actions(self, obs, ag=None, g=None, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
                b = obs.shape[0] if hasattr(obs, 'shape') and obs.ndim>1 else 1
                actions = np.random.uniform(-1,1,(b,self.action_dim))
                q = np.zeros((b,1)) if compute_Q else None
                return actions, q
        policy_obj = RandomPolicy(env.action_space.shape[0])

    # collect rollouts
    episodes = collect_rollouts(env, policy_obj, args.n_episodes, max_steps=args.max_steps, render=args.render)
    # convert to .npz dict
    npz_dict = episodes_to_npz_dict(episodes, env_name)
    # output path
    out_path = args.out
    if out_path is None:
        base = os.path.basename(policy_path)
        name = os.path.splitext(base)[0]
        out_path = os.path.join(policy_dir, f"{name}_demos.npz")
    # save compressed .npz
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **npz_dict)
    print(f"Saved demos to: {out_path} (episodes: {len(episodes)})")

if __name__ == "__main__":
    main_cli()
