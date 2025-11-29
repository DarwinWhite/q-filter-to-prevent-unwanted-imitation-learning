# src/experiment/train_mujoco.py
import os
import sys
import time
import click
import numpy as np
import json
import random
import logging
import resource
from pathlib import Path

# Add project root for src imports (portable solution)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Local project imports (these modules will be adapted separately)
import src.experiment.mujoco_config as config  # Use MuJoCo-specific config
from src.algorithms.rollout_mujoco import RolloutWorkerMuJoCo  # Use MuJoCo-specific rollout worker
from src.algorithms.ddpg_mujoco import DDPGMuJoCo  # Use MuJoCo-specific DDPG

# ---------------------------
# Minimal local logger to replace baselines.logger usage
# ---------------------------
class SimpleLogger:
    def __init__(self, dirpath=None):
        self._dir = Path(dirpath) if dirpath is not None else None
        self._tabular = {}
        # Python logging for info/warn
        self._logger = logging.getLogger("ddpg_mujoco")
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def configure(self, dir=None):
        if dir is not None:
            self._dir = Path(dir)
            self._dir.mkdir(parents=True, exist_ok=True)

    def get_dir(self):
        return str(self._dir) if self._dir is not None else None

    def info(self, *args, **kwargs):
        self._logger.info(" ".join(str(a) for a in args))

    def warn(self, *args, **kwargs):
        self._logger.warning(" ".join(str(a) for a in args))

    def record_tabular(self, key, val):
        self._tabular[key] = val

    def dump_tabular(self):
        # Persist tabular to a small json file for convenience and also print to stdout
        dirpath = self.get_dir()
        try:
            self._logger.info("TABULAR: " + ", ".join(f"{k}: {v}" for k, v in self._tabular.items()))
            if dirpath:
                tab_path = Path(dirpath) / "tabular_log.json"
                # append per-epoch dictionary
                if tab_path.exists():
                    with open(tab_path, "r") as f:
                        existing = json.load(f)
                else:
                    existing = []
                existing.append(self._tabular.copy())
                with open(tab_path, "w") as f:
                    json.dump(existing, f, indent=2)
        finally:
            self._tabular.clear()

# instantiate a module-level logger (this mirrors the single-process assumption)
logger = SimpleLogger()

# ---------------------------
# Helpers that replaced MPI functionality
# ---------------------------
def safe_average(value):
    """Local-only average / safe extractor (replacement for MPI averaging)."""
    try:
        if value is None:
            return 0.0
        if isinstance(value, (int, float, np.number)):
            return float(value)
        # numpy array
        if hasattr(value, 'size'):
            if value.size == 0:
                return 0.0
            return float(np.mean(value))
        # list-like
        if hasattr(value, '__len__') and len(value) > 0:
            return float(np.mean(value))
        return 0.0
    except Exception:
        try:
            return float(value)
        except Exception:
            return 0.0

def set_global_seeds(seed):
    """Seed python, numpy and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch not available yet or not desired — ignore
        pass

# ---------------------------
# Training loop (single-process)
# ---------------------------
def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, demo_file, **kwargs):
    rank = 0  # single-process runner

    latest_policy_path = os.path.join(logger.get_dir() or ".", 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir() or ".", 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir() or ".", 'policy_{}.pkl')

    logger.info("Training MuJoCo DDPG...")
    if rank == 0:
        print("\nStarting DDPG Training on MuJoCo")
        print("Training Configuration:")
        print("   - Epochs: {}".format(n_epochs))
        print("   - Cycles per epoch: {}".format(n_cycles))
        print("   - Training batches per cycle: {}".format(n_batches))
        print("   - Test rollouts: {}".format(n_test_rollouts))
        print("   - Environment: {}".format(kwargs.get('env_name', 'Unknown')))
        print("")

    best_return = -np.inf  # Track best episode return instead of success rate
    best_return_epoch = 0

    if getattr(policy, 'bc_loss', 0) == 1 and demo_file:
        # policy should implement initDemoBuffer; leave call as-is so the policy code can handle it
        policy.initDemoBuffer(demo_file)

    for epoch in range(n_epochs):
        epoch_start = time.time()
        if rank == 0:
            print(f"\n=== EPOCH {epoch+1}/{n_epochs} ===")

        # train
        rollout_worker.clear_history()
        # We'll collect returns across cycles for this epoch
        epoch_returns = []
        for cycle in range(n_cycles):
            episode = rollout_worker.generate_rollouts()

            # Get episode return from episode data directly
            # Support both list and numpy arrays
            r = episode.get('r', None)
            if r is None:
                episode_return = 0.0
            else:
                if isinstance(r, list):
                    episode_return = float(np.sum(r))
                else:
                    try:
                        # numpy array or scalar
                        episode_return = float(np.sum(r))
                    except Exception:
                        episode_return = safe_average(r)

            epoch_returns.append(episode_return)

            if rank == 0:
                print(f"  Cycle {cycle+1}/{n_cycles} - Episode return: {episode_return:.2f}")

            policy.store_episode(episode)

            for batch in range(n_batches):
                policy.train()

            # update target networks twice to mimic original behavior
            policy.update_target_net()
            policy.update_target_net()

        # test
        if rank == 0:
            print(f"  Testing policy with {n_test_rollouts} rollouts...")
        logger.info("Testing")
        evaluator.clear_history()

        # Collect test episode returns for this epoch only
        test_episode_returns = []
        for test_episode in range(n_test_rollouts):
            test_result = evaluator.generate_rollouts()
            r = test_result.get('r', None)
            if r is None:
                test_return = 0.0
            else:
                if isinstance(r, list):
                    test_return = float(np.sum(r))
                else:
                    try:
                        test_return = float(np.sum(r))
                    except Exception:
                        test_return = safe_average(r)
            test_episode_returns.append(test_return)

        # Calculate epoch-specific statistics
        current_test_return = np.mean(test_episode_returns) if test_episode_returns else 0.0

        # record logs
        logger.record_tabular('epoch', epoch)

        # Override the test return with our epoch-specific calculation
        logger.record_tabular('test/mean_episode_return', safe_average(current_test_return))
        logger.record_tabular('test/episode', (epoch + 1) * n_test_rollouts)  # Total episodes tested so far

        # Get other logs but avoid any MPI-specific keys (rollout_worker/evaluator/policy should implement logs())
        for key, val in evaluator.logs('test'):
            if not key.startswith('mean_episode'):
                logger.record_tabular(key, safe_average(val))

        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, safe_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, safe_average(val))

        if rank == 0:
            # Print epoch summary with epoch-specific returns
            # For MuJoCo, get the mean episode return (which replaces success rate)
            train_return_raw = rollout_worker.current_success_rate()
            train_return = train_return_raw if isinstance(train_return_raw, (int, float)) else safe_average(train_return_raw)
            buffer_size = policy.get_current_buffer_size()
            print(f"  Epoch {epoch+1} Summary:")
            print(f"    Test Return (This Epoch): {current_test_return:.2f}")
            print(f"    Train Return (Rolling Avg): {train_return:.2f}")
            print(f"    Buffer Size: {buffer_size}")
            print(f"    Best Return So Far: {best_return:.2f} (Epoch {best_return_epoch+1})")

            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        if rank == 0 and current_test_return >= best_return and save_policies:
            best_return = current_test_return
            best_return_epoch = epoch
            logger.info('New best episode return: {}. Saving policy to {} ...'.format(best_return, best_policy_path))
            try:
                evaluator.save_policy(best_policy_path)
                evaluator.save_policy(latest_policy_path)
            except Exception as e:
                logger.warn("Saving policy failed: ", e)

        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            try:
                evaluator.save_policy(policy_path)
            except Exception as e:
                logger.warn("Saving periodic policy failed: ", e)

        if rank == 0:
            print(f"Best episode return so far: {best_return:.2f} (achieved in epoch {best_return_epoch+1})")
        local_uniform = np.random.uniform(size=(1,))

    # Final training summary
    if rank == 0:
        print(f"\nTraining Completed!")
        print(f"Final Results:")
        print(f"   • Total epochs completed: {n_epochs}")
        print(f"   • Best episode return: {best_return:.2f}")
        print(f"   • Best epoch: {best_return_epoch+1}")
        print(f"   • Final buffer size: {policy.get_current_buffer_size()}")
        print(f"")


# ---------------------------
# Launch wrapper (replaces MPI forking and baselines tf session code)
# ---------------------------
def launch(
    env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, demo_file,
    bc_loss=None, q_filter=None, num_demo=0, n_cycles=None, n_batches=None, override_params={}, save_policies=True
):
    # Multi-process / MPI removed. This runner is single-process.
    if num_cpu > 1:
        logger.warn("*** Warning ***")
        logger.warn('Multi-CPU MPI-based execution is disabled in this runner. Running single-process instead.')
        logger.warn('For multi-process training, consider using torch.multiprocessing or an external launcher (SLURM).')
        logger.warn('****************')

    # Configure logging directory
    if logdir:
        logger.configure(dir=logdir)
    else:
        # create a timestamped directory in /tmp if none specified
        base = Path("/tmp/ddpg_mujoco_runs")
        base.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = base / f"run-{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.configure(dir=str(run_dir))

    logdir_used = logger.get_dir()
    assert logdir_used is not None
    os.makedirs(logdir_used, exist_ok=True)

    # Seed everything.
    rank_seed = seed  # single-process
    set_global_seeds(rank_seed)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    # Prepare params for MuJoCo
    params = config.DEFAULT_MUJOCO_PARAMS.copy()
    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    # Add BC and Q-filter parameters
    params['bc_loss'] = bc_loss
    params['q_filter'] = q_filter
    params['num_demo'] = num_demo
    if env in config.DEFAULT_MUJOCO_ENV_PARAMS:
        params.update(config.DEFAULT_MUJOCO_ENV_PARAMS[env])  # merge env-specific parameters
    params.update(**override_params)  # makes it possible to override any parameter

    # Override with command-line parameters if provided
    if n_cycles is not None:
        params['n_cycles'] = n_cycles
        print(f"Override: n_cycles = {n_cycles}")
    if n_batches is not None:
        params['n_batches'] = n_batches
        print(f"Override: n_batches = {n_batches}")
    if bc_loss is not None:
        params['bc_loss'] = bc_loss
        print(f"Override: bc_loss = {bc_loss}")
    if q_filter is not None:
        params['q_filter'] = q_filter
        print(f"Override: q_filter = {q_filter}")

    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_mujoco_params(params)
    # config.log_mujoco_params used to accept baselines.logger; keep compatibility by passing our logger
    config.log_mujoco_params(params, logger=logger)

    # Configure dimensions and policy for MuJoCo
    dims = config.configure_mujoco_dims(params)
    policy = config.configure_mujoco_ddpg(dims=dims, params=params, clip_return=clip_return)

    # Configure rollout workers for MuJoCo
    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params.get('test_with_polyak', False),
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorkerMuJoCo(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorkerMuJoCo(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(
        logdir=logdir_used, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies, demo_file=demo_file)


# ---------------------------
# CLI (keeps same interface)
# ---------------------------
@click.command()
@click.option('--env', type=str, default='HalfCheetah-v4', help='the name of the MuJoCo environment to train on')
@click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=200, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (MPI disabled in this runner)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='none', help='the HER replay strategy to be used. For MuJoCo dense rewards, use "none".')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default=None, help='demo data file path (optional for behavior cloning)')
@click.option('--n_cycles', type=int, default=None, help='number of training cycles per epoch (overrides config default)')
@click.option('--n_batches', type=int, default=None, help='number of training batches per cycle (overrides config default)')
@click.option('--bc_loss', type=int, default=None, help='whether to use behavior cloning loss (0/1, overrides config default)')
@click.option('--q_filter', type=int, default=None, help='whether to use Q-value filtering (0/1, overrides config default)')
def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()
