#!/usr/bin/env python3
"""
PyTorch-compatible training launcher for MuJoCo/flat-state environments.
Supports optional behavior cloning + Q-filtering via the PyTorch DDPG class.

Usage (example):
python train.py --env HalfCheetah-v4 --logdir ./logs/cheetah --n_epochs 200 --demo_file demo_data/cheetah_expert.npz
"""

import os
import sys
import time
import json
import random
import click
import numpy as np
import resource

# Minimal MPI support if mpi4py is present; otherwise behave as single-process
try:
    from mpi4py import MPI
except Exception:
    MPI = None

# project root for portable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import rollout workers and config
from src.experiment import config
from src.algorithms.rollout_mujoco import RolloutWorkerMuJoCo
from src.algorithms.rollout import RolloutWorkerOriginal

# Use our PyTorch DDPG via configure_ddpg in config
# NOTE: config.configure_ddpg returns an instance of src.algorithms.ddpg.DDPG

# Set random seeds helper (numpy, python, torch)
def set_global_seeds(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# A tiny logger to mimic baselines.logger interface used in other scripts.
class SimpleLogger:
    _dir = None

    @classmethod
    def configure(cls, dir=None):
        if dir is None:
            # default to tmp/logs with timestamp
            dir = os.path.join(os.getcwd(), "logs", time.strftime("%Y%m%d-%H%M%S"))
        cls._dir = dir
        os.makedirs(cls._dir, exist_ok=True)

    @classmethod
    def get_dir(cls):
        return cls._dir

    @classmethod
    def record_tabular(cls, key, val):
        # Append a small CSV-style file per run
        if cls._dir is None:
            cls.configure()
        path = os.path.join(cls._dir, "progress.csv")
        header_path = os.path.join(cls._dir, "progress_header.txt")
        # We'll collect records in memory and write during dump_tabular
        if not hasattr(cls, "_records"):
            cls._records = {}
            cls._keys = []
        cls._records[key] = val
        if key not in cls._keys:
            cls._keys.append(key)

    @classmethod
    def dump_tabular(cls):
        if cls._dir is None:
            cls.configure()
        path = os.path.join(cls._dir, "progress.csv")
        # Write header if not exists
        if not hasattr(cls, "_keys") or len(cls._keys) == 0:
            return
        header_line = ",".join(cls._keys) + "\n"
        values_line = ",".join(str(cls._records.get(k, "")) for k in cls._keys) + "\n"
        # If file not exists, write header
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(header_line)
                f.write(values_line)
        else:
            with open(path, "a") as f:
                f.write(values_line)
        # flush memory records for next epoch
        cls._records = {}
        cls._keys = []

    @classmethod
    def info(cls, msg):
        print("[INFO]", msg)

    @classmethod
    def warn(cls, msg=""):
        print("[WARN]", msg)


logger = SimpleLogger

# safe average for single-process (MPI-less) setups
def safe_average(value):
    try:
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.array(value, dtype=float)
            if arr.size == 0:
                return 0.0
            return float(arr.mean())
        return float(value)
    except Exception:
        return 0.0

# Training loop
def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, demo_file, logdir=None, **kwargs):
    # rank if MPI available
    rank = 0
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pt')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pt')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pt')

    logger.info("Training (PyTorch DDPG)...")
    if rank == 0:
        print("\nStarting Training")
        print(f"  Logdir: {logger.get_dir()}")
        print(f"  Epochs: {n_epochs}, Cycles/epoch: {n_cycles}, Batches/cycle: {n_batches}")
        print(f"  Environment: {kwargs.get('env_name', 'Unknown')}")
        print("")

    best_metric = -np.inf
    best_epoch = 0

    # initialize demo buffer if needed
    if getattr(policy, "bc_loss", 0) and demo_file:
        try:
            stored = policy.initDemoBuffer(demo_file, update_stats=True)
            logger.info(f"Loaded {stored} demo episodes (demo_file={demo_file})")
        except Exception as e:
            logger.warn(f"Failed to initialize demo buffer: {e}")

    for epoch in range(n_epochs):
        epoch_start = time.time()
        if rank == 0:
            print(f"\n=== EPOCH {epoch+1}/{n_epochs} ===")

        # training cycles
        rollout_worker.clear_history()
        for cycle in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            # store into replay buffer
            policy.store_episode(episode)
            # training batches per cycle
            for batch in range(n_batches):
                critic_loss, actor_loss = policy.train()
            # ensure target networks are updated
            policy.update_target_net()

        # evaluation
        if rank == 0:
            print(f"  Testing policy with {n_test_rollouts} rollouts...")
        evaluator.clear_history()
        test_returns = []
        for _ in range(n_test_rollouts):
            test_res = evaluator.generate_rollouts()
            # get returns from evaluator's reward history if available, otherwise try to compute
            if hasattr(evaluator, "current_success_rate"):
                tr = evaluator.current_success_rate()
            else:
                tr = 0.0
            test_returns.append(tr)

        mean_test_return = float(np.mean(test_returns)) if len(test_returns) > 0 else 0.0

        # Logging
        logger.record_tabular('epoch', epoch)
        logger.record_tabular('test/mean_return', safe_average(mean_test_return))
        # rollout logs
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, safe_average(val))
        # evaluator logs
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, safe_average(val))
        # policy logs
        for key, val in policy.logs():
            logger.record_tabular(key, safe_average(val))

        if rank == 0:
            # Print summary
            buffer_size = policy.get_current_buffer_size() if hasattr(policy, "get_current_buffer_size") else -1
            print(f"  Epoch {epoch+1} Summary:")
            print(f"    Test Return (This Epoch): {mean_test_return:.4f}")
            print(f"    Buffer Size: {buffer_size}")
            logger.dump_tabular()

        # Save best / periodic policies
        if rank == 0 and mean_test_return >= best_metric and save_policies:
            best_metric = mean_test_return
            best_epoch = epoch
            logger.info(f'New best return: {best_metric:.4f}. Saving policy to {best_policy_path} ...')
            try:
                policy.save_policy(best_policy_path)
                policy.save_policy(latest_policy_path)
            except Exception as e:
                logger.warn(f"Failed to save policy: {e}")

        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            path = periodic_policy_path.format(epoch)
            logger.info(f'Saving periodic policy to {path} ...')
            try:
                policy.save_policy(path)
            except Exception as e:
                logger.warn(f"Failed to save periodic policy: {e}")

        # seed synchronization / tiny RNG check (like original)
        local_uniform = np.random.uniform(size=(1,))
        if MPI is not None:
            root_uniform = local_uniform.copy()
            MPI.COMM_WORLD.Bcast(root_uniform, root=0)
            if rank != 0:
                assert local_uniform[0] != root_uniform[0]

    # final summary
    if rank == 0:
        print("\nTraining Completed!")
        print(f"  Total epochs: {n_epochs}")
        print(f"  Best metric: {best_metric:.4f} at epoch {best_epoch+1}")
        print(f"  Final buffer size: {policy.get_current_buffer_size() if hasattr(policy, 'get_current_buffer_size') else 'N/A'}")


# CLI and launcher
@click.command()
@click.option('--env', type=str, default='HalfCheetah-v4', help='the name of the MuJoCo environment to train on')
@click.option('--logdir', type=str, default=None, help="the path to where logs and policy checkpoints should go. If not specified, a timestamped folder is created under ./logs/")
@click.option('--n_epochs', type=int, default=200, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (MPI not required; num_cpu>1 will be ignored)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed environment and training')
@click.option('--policy_save_interval', type=int, default=5, help='interval for saving periodic policies')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='none', help='HER replay strategy; use "none" for MuJoCo dense rewards')
@click.option('--clip_return', type=int, default=1, help='whether to clip returns (1) or not (0)')
@click.option('--demo_file', type=str, default=None, help='demo data file path (optional for behavior cloning)')
@click.option('--n_cycles', type=int, default=None, help='number of training cycles per epoch (overrides config default)')
@click.option('--n_batches', type=int, default=None, help='number of training batches per cycle (overrides config default)')
@click.option('--bc_loss', type=int, default=None, help='whether to use behavior cloning loss (0/1, overrides config default)')
@click.option('--q_filter', type=int, default=None, help='whether to use Q-value filtering (0/1, overrides config default)')
def main(env, logdir, n_epochs, num_cpu, seed, policy_save_interval, replay_strategy, clip_return, demo_file,
         n_cycles, n_batches, bc_loss, q_filter):
    # Basic multiprocessing/MPI note: we avoid re-forking via mpirun here to keep things simple and TF-free.
    if num_cpu > 1:
        logger.warn("num_cpu > 1 requested, but this script avoids MPI/TensorFlow complexity. Running single-process.")

    # Configure logging dir
    if logdir is None:
        logdir = os.path.join(os.getcwd(), "logs", time.strftime("%Y%m%d-%H%M%S"))
    logger.configure(dir=logdir)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    set_global_seeds(seed)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    # Prepare params
    params = config.DEFAULT_PARAMS.copy()
    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])
    # override params from CLI
    if n_cycles is not None:
        params['n_cycles'] = n_cycles
    if n_batches is not None:
        params['n_batches'] = n_batches
    if bc_loss is not None:
        params['bc_loss'] = bc_loss
    if q_filter is not None:
        params['q_filter'] = q_filter

    # Save params
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)

    # Prepare params for DDPG
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    # Configure dims & policy
    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=(clip_return == 1))

    # Rollout worker selection: MuJoCo flat-state -> RolloutWorkerMuJoCo; else use RolloutWorkerOriginal
    if dims.get('g', 0) == 0:
        rollout_params = {
            'exploit': False,
            'use_target_net': False,
            'compute_Q': False,
            'T': params['T'],
            'rollout_batch_size': params.get('rollout_batch_size', 1),
            'noise_eps': params.get('noise_eps', 0.0),
            'random_eps': params.get('random_eps', 0.0),
            'render': False,
        }
        eval_params = {
            'exploit': True,
            'use_target_net': params.get('test_with_polyak', False),
            'compute_Q': True,
            'T': params['T'],
            'rollout_batch_size': 1,
            'noise_eps': params.get('noise_eps', 0.0),
            'random_eps': params.get('random_eps', 0.0),
            'render': False,
        }

        rollout_worker = RolloutWorkerMuJoCo(params['make_env'], policy, dims, logger, **rollout_params)
        rollout_worker.seed(seed)

        evaluator = RolloutWorkerMuJoCo(params['make_env'], policy, dims, logger, **eval_params)
        evaluator.seed(seed)
    else:
        rollout_params = {
            'exploit': False,
            'use_target_net': False,
            'compute_Q': False,
            'T': params['T'],
            'rollout_batch_size': params.get('rollout_batch_size', 1),
            'noise_eps': params.get('noise_eps', 0.0),
            'random_eps': params.get('random_eps', 0.0),
            'render': False,
        }
        eval_params = {
            'exploit': True,
            'use_target_net': params.get('test_with_polyak', False),
            'compute_Q': True,
            'T': params['T'],
            'rollout_batch_size': 1,
            'noise_eps': params.get('noise_eps', 0.0),
            'random_eps': params.get('random_eps', 0.0),
            'render': False,
        }

        rollout_worker = RolloutWorkerOriginal(params['make_env'], policy, dims, logger, **rollout_params)
        rollout_worker.seed(seed)

        evaluator = RolloutWorkerOriginal(params['make_env'], policy, dims, logger, **eval_params)
        evaluator.seed(seed)

    # Start training
    train(
        policy=policy,
        rollout_worker=rollout_worker,
        evaluator=evaluator,
        n_epochs=n_epochs,
        n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'],
        n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval,
        save_policies=True,
        demo_file=demo_file,
        logdir=logdir,
        env_name=env
    )


if __name__ == "__main__":
    main()
