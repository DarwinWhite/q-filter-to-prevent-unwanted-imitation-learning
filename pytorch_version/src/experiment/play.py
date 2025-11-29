#!/usr/bin/env python3
"""
PyTorch-based policy viewer for MuJoCo (and other Gym) environments.

Features:

* Loads a saved PyTorch policy (either a pickled nn.Module, a TorchScript module,
  or a state_dict together with a model-class import).
* Runs a number of deterministic or stochastic rollouts and optionally renders.
* Flexible call interface: will attempt to call policy methods in this order:

  1. policy.act / policy.get_actions (numpy or torch-aware)
  2. policy.pi(o, g)  (actor method)
  3. policy(o)        (forward)
* Handles common Gym API differences (reset / step returning tuples).
* Prints summary statistics (mean/std return, min/max).
* Useful for quick visual checks / post-training analysis.

Usage examples:
python play.py path/to/policy.pt --env HalfCheetah-v4 --n_rollouts 20 --render
python play.py path/to/state_dict.pth --model-class src.algorithms.actor_critic:ActorCritic 
--model-kwargs '{"dimo":17,"dimg":0,"dimu":6,"max_u":1.0,"hidden":256,"layers":3}' 
--env HalfCheetah-v4 --render

Notes:

* If loading a state_dict, you must provide --model-class and --model-kwargs (JSON).
* The script makes reasonable guesses about how to call the policy; if your model
  has a custom API, adapt the small `get_action_from_policy` helper below.
  """

import os
import sys
import json
import time
import random
import argparse
from typing import Any, Dict, Optional

import numpy as np
import torch
import gym

# Add project root for portable imports

project_root = os.path.abspath(os.path.join(os.path.dirname(**file**), '..', '..'))
if project_root not in sys.path:
sys.path.insert(0, project_root)

# Utility to import a model class, if required

try:
from src.utils.util import import_function
except Exception:
# fallback simple importer
def import_function(spec: str):
mod_name, fn_name = spec.split(':')
mod = **import**(mod_name, fromlist=[fn_name])
return getattr(mod, fn_name)

def set_seed(seed: int):
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
torch.cuda.manual_seed_all(seed)

def try_load_policy(policy_path: str, device: torch.device, model_class_spec: Optional[str] = None,
model_kwargs: Optional[Dict[str, Any]] = None):
"""
Try to load the policy from file.


Returns:
  policy_obj, is_state_dict (bool)
- policy_obj will be:
    * an nn.Module (loaded directly or constructed then loaded state_dict)
    * a torch.jit.ScriptModule
    * an object (dict) if loading failed meaningfully
- is_state_dict indicates the file was a state dict (True means we constructed a model)
"""
# First try torch.load
loaded = None
try:
    loaded = torch.load(policy_path, map_location=device)
except Exception as e:
    raise RuntimeError(f"Failed to load policy file '{policy_path}' with torch.load: {e}")

# If it's already a ScriptModule or nn.Module instance -> use directly
if isinstance(loaded, torch.nn.Module):
    return loaded.to(device), False

# TorchScript objects may be of type ScriptModule
# They sometimes are returned as 'loaded' but not exactly a subclass; check presence of 'forward'
if hasattr(loaded, "__call__") and hasattr(loaded, "forward"):
    # treat as module-like
    try:
        loaded.to(device)
    except Exception:
        pass
    return loaded, False

# If it's a dict, several possibilities:
#  - {'state_dict': ...}
#  - plain state_dict
#  - a checkpoint with many keys ('actor_state_dict', 'critic_state_dict', etc.)
if isinstance(loaded, dict):
    # common key names
    if 'state_dict' in loaded and isinstance(loaded['state_dict'], dict):
        state_dict = loaded['state_dict']
    elif any(k.endswith('state_dict') for k in loaded.keys()):
        # pick actor_state_dict or first state dict-like field
        sdict_key = None
        for k in ['actor_state_dict', 'policy_state_dict', 'model_state_dict']:
            if k in loaded:
                sdict_key = k
                break
        if sdict_key is None:
            for k, v in loaded.items():
                if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                    sdict_key = k
                    break
        if sdict_key is None:
            # assume the whole dict is the state dict (best-effort)
            state_dict = loaded
        else:
            state_dict = loaded[sdict_key]
    else:
        # treat loaded as if it might be a pure state_dict
        state_dict = loaded

    # If user supplied a model class, instantiate and load the state dict
    if model_class_spec is None:
        raise RuntimeError(
            "Policy file contains a state-dict-like object. You must provide --model-class "
            "and --model-kwargs to reconstruct the model and load the state dict."
        )
    ModelClass = import_function(model_class_spec)
    kwargs = model_kwargs or {}
    try:
        model = ModelClass(**kwargs)
    except Exception as e:
        # Try instantiating without kwargs
        try:
            model = ModelClass()
        except Exception as e2:
            raise RuntimeError(f"Failed to instantiate model class '{model_class_spec}': {e} | {e2}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, True

# Unknown loaded type — return as-is (caller will need to handle)
return loaded, False


def get_action_from_policy(policy, observation: np.ndarray, device: torch.device,
deterministic: bool = True):
"""
Universal wrapper to request an action from a policy. Tries multiple common APIs.


observation: 1D numpy array or batch (we will convert to shape (1, d))
Returns: action as numpy array (1D)
"""
# Ensure 2D shape (batch of 1)
if observation.ndim == 1:
    obs_np = observation[None, ...]
else:
    obs_np = observation

# Convert to torch tensor
obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)

# Try common methods in order
# 1) policy.get_actions / policy.act (may accept numpy or torch)
for method_name in ('get_actions', 'act', 'forward_policy', 'step_policy'):
    if hasattr(policy, method_name):
        method = getattr(policy, method_name)
        try:
            # try numpy input first
            try:
                out = method(obs_np)
            except Exception:
                out = method(obs_t)
            # out may be (actions, info) or actions
            if isinstance(out, tuple) or isinstance(out, list):
                action = out[0]
            else:
                action = out
            # convert to numpy if torch tensor
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            return np.asarray(action).reshape(-1)
        except Exception:
            # fallthrough to next method
            pass

# 2) policy.pi(o, g) - actor that expects (o, g). We'll pass zeros for g if needed.
if hasattr(policy, 'pi'):
    pi_fn = policy.pi
    try:
        # try calling with (obs_t) only
        out = pi_fn(obs_t)
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().numpy().reshape(-1)
        elif isinstance(out, np.ndarray):
            return out.reshape(-1)
    except TypeError:
        # try (o, g) with dummy zeros for g
        try:
            g_dummy = torch.zeros((obs_t.shape[0], 0), dtype=obs_t.dtype, device=device)
            out = pi_fn(obs_t, g_dummy)
            if isinstance(out, torch.Tensor):
                return out.detach().cpu().numpy().reshape(-1)
        except Exception:
            pass

# 3) Call policy directly (policy(obs))
try:
    out = policy(obs_t)
    if isinstance(out, torch.Tensor):
        return out.detach().cpu().numpy().reshape(-1)
    if isinstance(out, tuple) or isinstance(out, list):
        a = out[0]
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().reshape(-1)
        return np.asarray(a).reshape(-1)
except Exception:
    pass

# 4) Last resort: try numpy-based call if policy expects numpy
try:
    out = policy(obs_np)
    if isinstance(out, torch.Tensor):
        return out.detach().cpu().numpy().reshape(-1)
    if isinstance(out, (np.ndarray, list, tuple)):
        arr = np.asarray(out)
        return arr.reshape(-1)
except Exception:
    pass

raise RuntimeError("Unable to query action from policy. Please adapt get_action_from_policy to your model API.")


def run_rollouts(env, policy, device, n_rollouts=10, max_steps=None, render=False, pause=0.0, deterministic=True):
returns = []
lengths = []


for r in range(n_rollouts):
    reset_res = env.reset()
    if isinstance(reset_res, tuple):
        obs, info = reset_res
    else:
        obs = reset_res

    ep_ret = 0.0
    ep_len = 0
    done = False
    step = 0
    while True:
        # get action
        try:
            a = get_action_from_policy(policy, np.asarray(obs), device, deterministic=deterministic)
        except Exception as e:
            raise RuntimeError(f"Policy invocation failed on rollout {r}, step {step}: {e}")

        # ensure correct shape / dtype for env
        a_clipped = np.asarray(a, dtype=np.float32)
        # step environment
        step_res = env.step(a_clipped)
        if len(step_res) == 4:
            next_obs, reward, done, info = step_res
        else:
            next_obs, reward, terminated, truncated, info = step_res
            done = terminated or truncated

        ep_ret += float(reward)
        ep_len += 1
        step += 1

        obs = next_obs

        if render:
            env.render()
            if pause and pause > 0.0:
                time.sleep(pause)

        # termination conditions
        if done:
            break
        if (max_steps is not None) and (ep_len >= max_steps):
            break

    returns.append(ep_ret)
    lengths.append(ep_len)
    print(f"Rollout {r+1}/{n_rollouts}  Return: {ep_ret:.3f}  Length: {ep_len}")

return returns, lengths


def main_cli():
parser = argparse.ArgumentParser(description="PyTorch policy viewer / play script")
parser.add_argument("policy_file", type=str, help="Path to policy file (torch .pt/.pth or state dict)")
parser.add_argument("--env", type=str, default="HalfCheetah-v4", help="Gym env id")
parser.add_argument("--n_rollouts", type=int, default=10, help="Number of rollouts to run")
parser.add_argument("--max_steps", type=int, default=None, help="Max steps per rollout (env default used if None)")
parser.add_argument("--render", action="store_true", help="Render environment during rollouts")
parser.add_argument("--device", type=str, default="cpu", help="torch device")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--model-class", type=str, default=None,
help="Optional import path for model class if loading a state_dict, e.g. 'src.algorithms.actor_critic:ActorCritic'")
parser.add_argument("--model-kwargs", type=str, default="{}",
help="JSON string of kwargs to pass to model constructor when --model-class is used")
parser.add_argument("--pause", type=float, default=0.0, help="Seconds to pause between frames when rendering")
parser.add_argument("--deterministic", action="store_true", help="Try to run policy deterministically if applicable")
args = parser.parse_args()


device = torch.device(args.device)
set_seed(args.seed)

# create env
env = gym.make(args.env)

# load policy
model_kwargs = {}
try:
    model_kwargs = json.loads(args.model_kwargs)
except Exception:
    print("Warning: --model-kwargs is not valid JSON; ignoring.")
    model_kwargs = {}

policy, _ = try_load_policy(args.policy_file, device, model_class_spec=args.model_class, model_kwargs=model_kwargs)
# If policy has eval() method, put in eval mode
try:
    if hasattr(policy, 'eval'):
        policy.eval()
except Exception:
    pass

print(f"Loaded policy from {args.policy_file}. Running {args.n_rollouts} rollouts on {args.env} (device={device})")
returns, lengths = run_rollouts(env, policy, device,
                                n_rollouts=args.n_rollouts,
                                max_steps=args.max_steps,
                                render=args.render,
                                pause=args.pause,
                                deterministic=args.deterministic)

if len(returns) > 0:
    arr = np.array(returns, dtype=np.float32)
    print("\n=== Summary ===")
    print(f"Rollouts: {len(returns)}")
    print(f"Return mean: {arr.mean():.3f}  std: {arr.std():.3f}  min: {arr.min():.3f}  max: {arr.max():.3f}")

env.close()


if **name** == "**main**":
main_cli()
