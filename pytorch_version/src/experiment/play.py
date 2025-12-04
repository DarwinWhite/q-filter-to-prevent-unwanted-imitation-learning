import click
import numpy as np
import sys
import os
import gym
import json
import pickle
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from baselines import logger
import src.experiment.mujoco_config as config
from src.algorithms.rollout_mujoco import RolloutWorkerMuJoCo

import torch
# # -------------------------------------------------------
# # 1. Direct copy: load policy exactly like working TF code
# # -------------------------------------------------------
# def load_policy_with_fallback(policy_file):
#     print(f"Attempting to load policy from: {policy_file}")

#     # Strategy 1: standard pickle
#     try:
#         with open(policy_file, 'rb') as f:
#             loaded_data = pickle.load(f)
#         print("✓ Successfully loaded with standard pickle")
#         return loaded_data
#     except Exception as e:
#         print(f"✗ Standard pickle failed: {e}")

#     # Strategy 2: latin1 fallback
#     try:
#         with open(policy_file, 'rb') as f:
#             loaded_data = pickle.load(f, encoding='latin1')
#         print("✓ Successfully loaded with latin1 encoding")
#         return loaded_data
#     except Exception as e:
#         print(f"✗ Latin1 encoding failed: {e}")

#     print("✗ All loading strategies failed")
#     return None
# ---------- loader helpers (paste into play.py) ----------
def _safe_torch_load(path):
    """Safe wrapper that attempts torch.load and returns the loaded object or raises."""
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        # re-raise with context for caller
        raise RuntimeError(f"torch.load failed for '{path}': {e}")


class DDPGAdapter:
    """
    Minimal adapter that exposes get_actions(...) by delegating to an inner DDPG instance.
    RolloutWorkerMuJoCo expects policy.get_actions(o, ag, g, **kwargs).
    We'll wrap the ddpg object (which implements get_actions).
    """
    def __init__(self, ddpg):
        self.ddpg = ddpg

    def get_actions(self, o, ag=None, g=None, **kwargs):
        # ddpg.get_actions expects numpy arrays and returns numpy actions (and optionally Q)
        return self.ddpg.get_actions(o, ag=ag, g=g, **kwargs)

    # Provide minimal other methods used elsewhere (not strictly needed for playback but safe)
    def eval(self):
        try:
            if hasattr(self.ddpg, "main"):
                self.ddpg.main.eval()
            if hasattr(self.ddpg, "target"):
                self.ddpg.target.eval()
        except Exception:
            pass

    def __getattr__(self, name):
        # Delegate everything else to wrapped ddpg if present
        return getattr(self.ddpg, name)


# def try_load_policy(policy_path, dims, params, env):
def try_load_policy(policy_path, dims, params, env_name):
    """
    Load policies saved by the training run. Handles:
      - torch.save(checkpoint_dict, path) where checkpoint contains 'main'/'target' or actor/critic dicts
      - torch.jit.ScriptModule or torch.nn.Module saved with torch.save
      - fallback to random policy if not loadable
    Returns: policy_object (with .get_actions signature)
    """
    print(f"🔍 Attempting to load policy via torch.load: {policy_path}")

    # 1) Try torch.load
    try:
        loaded = _safe_torch_load(policy_path)
    except Exception as e:
        print("❌ torch.load failed:", e)
        print("➡ Falling back to RandomPolicy")
        env_action_dims = {
            'HalfCheetah-v4': 6,
            'Hopper-v4': 3,
            'Walker2d-v4': 6,
            'Ant-v4': 8
        }
            
        # Check if env_name is valid before proceeding
        if env_name in env_action_dims:
            action_dim = env_action_dims[env_name]
        else:
            raise ValueError(f"Unknown environment: {env_name}. Please add it to env_action_dims.")
        # return RandomPolicy(env.action_space.shape[0])
        return RandomPolicy(action_dim)#google's version

    # 2) If it's an nn.Module (or script module), return it directly (wrap if needed)
    if isinstance(loaded, torch.nn.Module):
        print("✅ Loaded nn.Module from checkpoint.")
        try:
            loaded.to('cpu')
            loaded.eval()
        except Exception:
            pass
        # If module provides get_actions, return as-is; otherwise wrap to adapt API if possible
        if hasattr(loaded, "get_actions"):
            return loaded
        # If module uses forward to map obs->action, wrap into simple object
        class ModuleWrapper:
            def __init__(self, mod):
                self.mod = mod
                self.mod.eval()
            def get_actions(self, obs, ag=None, g=None, **kwargs):
                # obs is numpy batch shape (B, o_dim)
                import numpy as _np
                if isinstance(obs, _np.ndarray):
                    obs_t = torch.tensor(obs, dtype=torch.float32)
                else:
                    obs_t = torch.tensor(_np.asarray(obs), dtype=torch.float32)
                with torch.no_grad():
                    out = self.mod(obs_t)
                if isinstance(out, torch.Tensor):
                    return out.cpu().numpy(), None
                # If tuple returned
                if isinstance(out, (tuple, list)):
                    a = out[0]
                    if isinstance(a, torch.Tensor):
                        return a.cpu().numpy(), None
                    return _np.asarray(a), None
                return _np.asarray(out), None
        return ModuleWrapper(loaded)

    # 3) If it's a dict-like checkpoint, try to reconstruct DDPG
    if isinstance(loaded, dict):
        print("✅ Loaded checkpoint dict from torch.load — attempting to reconstruct DDPG agent.")
        # Try to import DDPG class from your code
        try:
            from src.algorithms.ddpg import DDPG as DDPGClass
        except Exception:
            # fallback: maybe the training used ddpg.py named DDPG in other module
            try:
                from src.algorithms.ddpg_torch import DDPG as DDPGClass
            except Exception:
                DDPGClass = None

        if DDPGClass is None:
            print("⚠ Could not import DDPG class from src.algorithms. Returning RandomPolicy.")
            # return RandomPolicy(env.action_space.shape[0])
            return RandomPolicy(action_dim) #using google's method again

        # Build input_dims expected by DDPG constructor
        input_dims = {"o": int(dims['o']), "u": int(dims['u']), "g": int(dims.get('g', 0))}
        # Construct DDPG with sensible params coming from params.json when available
        ddpg_kwargs = {
            "input_dims": input_dims,
            "hidden": int(params.get("hidden", 256)),
            "layers": int(params.get("layers", 3)),
            "max_u": float(params.get("max_u", 1.0)),
            "device": "cpu"
        }
        # optional args if present in constructor (DDPG in your repo has many defaults)
        try:
            ddpg = DDPGClass(**ddpg_kwargs)
        except Exception as e:
            # Try a minimal constructor attempt if above failed
            try:
                ddpg = DDPGClass(input_dims=input_dims)
            except Exception as e2:
                print("⚠ Failed to instantiate DDPG class:", e, "|", e2)
                print("➡ Falling back to RandomPolicy")
                # return RandomPolicy(env.action_space.shape[0])
                return RandomPolicy(action_dim) #using google's method again

        # Load weights into ddpg / ddpg.main / ddpg.target
        # Support checkpoints with keys: 'main','target' OR 'actor_state_dict','critic_state_dict' OR 'model' OR plain state_dict
        try:
            if "main" in loaded and isinstance(loaded["main"], dict):
                try:
                    ddpg.main.load_state_dict(loaded["main"])
                except Exception:
                    # maybe nested 'actor'/'critic' inside main
                    pass
                if "target" in loaded and isinstance(loaded["target"], dict):
                    try:
                        ddpg.target.load_state_dict(loaded["target"])
                    except Exception:
                        pass
            elif "actor_state_dict" in loaded and "critic_state_dict" in loaded:
                try:
                    ddpg.main.actor.load_state_dict(loaded["actor_state_dict"])
                    ddpg.main.critic.load_state_dict(loaded["critic_state_dict"])
                    # try to load targets if present
                    if "target_actor_state_dict" in loaded:
                        ddpg.target.actor.load_state_dict(loaded["target_actor_state_dict"])
                    if "target_critic_state_dict" in loaded:
                        ddpg.target.critic.load_state_dict(loaded["target_critic_state_dict"])
                except Exception:
                    pass
            elif "model" in loaded and isinstance(loaded["model"], dict):
                # some variants saved under key 'model'
                try:
                    ddpg.main.load_state_dict(loaded["model"])
                except Exception:
                    pass
            else:
                # maybe the whole dict is a state_dict for the main network
                try:
                    ddpg.main.load_state_dict(loaded)
                except Exception:
                    # last resort: try to set directly if saved as ddpg.state_dict()
                    try:
                        if hasattr(ddpg, "load_policy"):
                            # If your DDPG implementation exposes load_policy, call it
                            ddpg.load_policy(loaded)
                        else:
                            raise RuntimeError("No recognized loading route.")
                    except Exception as e:
                        print("⚠ Failed to load weights into DDPG:", e)

            # restore normalizer stats if present
            if "o_mean" in loaded and hasattr(ddpg, "o_stats"):
                try:
                    ddpg.o_stats.mean = np.array(loaded["o_mean"], dtype=np.float32)
                    ddpg.o_stats.std  = np.array(loaded.get("o_std", ddpg.o_stats.std), dtype=np.float32)
                except Exception:
                    pass
            if "o_stats" in loaded and hasattr(ddpg, "o_stats"):
                try:
                    # saved normalizer object might be a dict — try to apply safely
                    s = loaded["o_stats"]
                    if isinstance(s, dict) and "mean" in s:
                        ddpg.o_stats.mean = np.array(s["mean"], dtype=np.float32)
                        ddpg.o_stats.std = np.array(s.get("std", ddpg.o_stats.std), dtype=np.float32)
                except Exception:
                    pass

            print("✅ DDPG weights loaded (best-effort). Returning adapter for rollout usage.")
            return DDPGAdapter(ddpg)

        except Exception as e:
            print("❌ Exception while loading checkpoint into DDPG:", e)
            print("➡ Falling back to RandomPolicy.")
            # return RandomPolicy(env.action_space.shape[0])
            return RandomPolicy(action_dim) #using google's method again

    # 4) Unknown type — fallback
    print("⚠ Unrecognized policy file type returned by torch.load; falling back to RandomPolicy.")
    # return RandomPolicy(env.action_space.shape[0])
    return RandomPolicy(action_dim) #using google's method again




# -------------------------------------------------------
# 2. Same RandomPolicy used in TensorFlow version
# -------------------------------------------------------
class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def get_actions(self, obs, ag=None, g=None, noise_eps=0., random_eps=0.,
                    use_target_net=False, compute_Q=False):
        batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
        actions = np.random.uniform(-1, 1, (batch_size, self.action_dim))
        Q_values = np.zeros((batch_size, 1)) if compute_Q else None
        return actions, Q_values


# -------------------------------------------------------
# 3. Choose policy based on raw .pkl load
# -------------------------------------------------------
def get_policy(policy_file, env_name):
    loaded_data = load_policy_with_fallback(policy_file)

    # If loaded object is callable or policy-like → use it
    if loaded_data is not None and hasattr(loaded_data, 'get_actions'):
        print("✓ Loaded usable policy object from .pkl")
        return loaded_data

    print("⚠️ Policy not usable. Falling back to RandomPolicy.")

    env_action_dims = {
        'HalfCheetah-v4': 6,
        'Hopper-v4': 3,
        'Walker2d-v4': 6,
        'Ant-v4': 8
    }
    action_dim = env_action_dims.get(env_name, 6)
    return RandomPolicy(action_dim)


# -------------------------------------------------------
# 4. Record GIF helper
# -------------------------------------------------------
def record_episode_gif(env, policy, output_path, max_steps=1000):
    frames = []
    obs, _ = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        frame = env.render()
        frames.append(Image.fromarray(frame))

        action, _ = policy.get_actions(obs.reshape(1, -1))
        obs, reward, terminated, truncated, _ = env.step(action[0])
        total_reward += reward
        if terminated or truncated:
            break

    frames[0].save(output_path, save_all=True,
                   append_images=frames[1:], duration=40, loop=0)
    print(f"🎬 GIF saved: {output_path} | Return: {total_reward:.2f}")


# -------------------------------------------------------
# 5. Main
# -------------------------------------------------------
@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=5)
@click.option('--render', is_flag=True)
@click.option('--record_gif', is_flag=True)
@click.option('--output_dir', type=str, default='visualization_output')
def main(policy_file, seed, n_test_rollouts, render, record_gif, output_dir):
    np.random.seed(seed)

    policy_dir = os.path.dirname(policy_file)
    params_path = os.path.join(policy_dir, "params.json")

    # Load env name
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = json.load(f)
        env_name = params.get("env_name", "HalfCheetah-v4")
    else:
        print("⚠ params.json missing → defaulting to HalfCheetah-v4")
        env_name = "HalfCheetah-v4"

    print(f"🎯 Environment: {env_name}")

    # Build env
    # make_env = lambda: gym.make(env_name)
    if render:
        make_env = lambda: gym.make(env_name, render_mode="human")
    else:
        make_env = lambda: gym.make(env_name)


    # Load fallback or true policy
    # policy = get_policy(policy_file, env_name)
    # policy = try_load_policy(policy_file, env_name)
    

    # Load mujoco rollout dims from your existing config
    params = config.prepare_mujoco_params(params if "params" in locals() else None)
    dims = config.configure_mujoco_dims(params)

    policy = try_load_policy(policy_file, dims, params, env_name)

    # Rollout worker (kept same)
    evaluator = RolloutWorkerMuJoCo(make_env, policy, dims, logger,
                                    exploit=True, render=render, T=params['T'])

    if record_gif:
        os.makedirs(output_dir, exist_ok=True)

    # Run episodes
    returns = []
    for ep in range(n_test_rollouts):
        result = evaluator.generate_rollouts()
        ep_return = result['r'].sum()
        returns.append(ep_return)
        print(f"Episode {ep+1}: Return = {ep_return:.2f}")

        if record_gif and ep < 3:
            gif_path = os.path.join(
                output_dir,
                f"{os.path.basename(policy_file)}_ep{ep+1}.gif"
            )
            gif_env = make_env()# or maybe use this:#gif_env = gym.make(env_name, render_mode="rgb_array")
            record_episode_gif(gif_env, policy, gif_path)

    print("\nSUMMARY:")
    print(f"Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")


if __name__ == '__main__':
    main()
