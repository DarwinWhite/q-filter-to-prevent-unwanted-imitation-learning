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
from datetime import datetime
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
        self._training_log_file = None  # Add training log file
        
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
            
            # Initialize training log file
            self._training_log_file = self._dir / "training.log"
            with open(self._training_log_file, 'w') as f:
                f.write(f"Training Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

    def get_dir(self):
        return str(self._dir) if self._dir is not None else None

    def _write_to_log_file(self, message):
        """Write message to training log file."""
        if self._training_log_file:
            try:
                with open(self._training_log_file, 'a') as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
                    f.flush()
            except Exception:
                pass  # Silently continue if file write fails

    def info(self, *args, **kwargs):
        message = " ".join(str(a) for a in args)
        self._logger.info(message)
        self._write_to_log_file(f"INFO: {message}")

    def warn(self, *args, **kwargs):
        message = " ".join(str(a) for a in args)
        self._logger.warning(message)
        self._write_to_log_file(f"WARN: {message}")

    def record_tabular(self, key, val):
        self._tabular[key] = val

    def dump_tabular(self):
        if self._dir is None:
            return
        
        # Write to progress.csv
        csv_path = self._dir / "progress.csv"
        
        # Check if we need to write header
        write_header = not csv_path.exists()
        
        with open(csv_path, 'a') as f:
            if write_header:
                f.write(','.join(self._tabular.keys()) + '\n')
            f.write(','.join(str(self._tabular[k]) for k in self._tabular.keys()) + '\n')
        
        # Log only relevant metrics to console (filter out massive tensors)
        filtered_data = {k: v for k, v in self._tabular.items() 
                        if k != 'state' and not str(k).startswith('state/') and 
                        not isinstance(v, dict) and str(v).count('tensor') == 0}
        if filtered_data:
            log_message = f"Tabular data: {filtered_data}"
            self.info(log_message)
        
        self._tabular.clear()

# instantiate a module-level logger (this mirrors the single-process assumption)
logger = SimpleLogger()

# ---------------------------
# Helpers that replaced MPI functionality
# ---------------------------
def safe_average(value):
    """Simple averaging (no MPI in PyTorch version)."""
    if isinstance(value, (list, tuple)):
        return np.mean(value)
    else:
        return float(value)

def set_global_seeds(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    # PyTorch seeding will be handled in DDPG if using GPU
    import torch
    torch.manual_seed(seed)

# ---------------------------
# Main training function
# ---------------------------
def train(policy, rollout_worker, evaluator, n_epochs, n_test_rollouts, n_cycles, n_batches, 
          policy_save_interval, save_policies, demo_file, **kwargs):
    """Main training loop for DDPG MuJoCo."""
    
    rank = 0  # Single process
    
    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pt')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pt')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pt')
    
    logger.info("Training...")
    best_mean_return = -float('inf')
    
    # Load demonstrations if provided
    if demo_file is not None and demo_file != "":
        if os.path.exists(demo_file):
            logger.info(f"Loading demonstrations from: {demo_file}")
            policy.initDemoBuffer(demo_file, update_stats=True)
        else:
            logger.warn(f"Demo file not found: {demo_file}")
    
    for epoch in range(n_epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{n_epochs} ===")
        
        # Training rollouts and updates
        epoch_episode_returns = []
        for cycle in range(n_cycles):
            # Generate rollouts
            episode = rollout_worker.generate_rollouts()
            
            # Track episode returns for this cycle
            if hasattr(rollout_worker, 'episode_rewards') and rollout_worker.episode_rewards:
                # Get the most recent episode returns (could be multiple per cycle)
                num_recent = min(rollout_worker.rollout_batch_size * 2, len(rollout_worker.episode_rewards))
                recent_returns = rollout_worker.episode_rewards[-num_recent:]
                # Filter out zeros and very small values (incomplete episodes)
                meaningful_returns = [r for r in recent_returns if abs(r) > 0.01]
                if meaningful_returns:
                    cycle_mean_return = np.mean(meaningful_returns)
                    epoch_episode_returns.extend(meaningful_returns)
                    logger.info(f"  Cycle {cycle + 1:2d}/{n_cycles}: Episode returns = {meaningful_returns} (mean = {cycle_mean_return:.2f})")
                else:
                    logger.info(f"  Cycle {cycle + 1:2d}/{n_cycles}: Training... (no completed episodes)")
            else:
                logger.info(f"  Cycle {cycle + 1:2d}/{n_cycles}: Training...")
            
            policy.store_episode(episode)
            
            # Training updates
            for batch in range(n_batches):
                policy.train()
            
            # Update target networks periodically
            policy.update_target_net()
        
        # Print epoch summary
        if epoch_episode_returns:
            epoch_mean_return = np.mean(epoch_episode_returns)
            epoch_std_return = np.std(epoch_episode_returns)
            logger.info(f"  Epoch {epoch + 1} Training Summary: {len(epoch_episode_returns)} episodes, mean return = {epoch_mean_return:.2f} ± {epoch_std_return:.2f}")
        
        # Evaluation
        logger.info(f"  Running evaluation ({n_test_rollouts} episodes)...")
        evaluator.clear_history()
        eval_returns = []
        for eval_ep in range(n_test_rollouts):
            evaluator.generate_rollouts()
            if hasattr(evaluator, 'episode_rewards') and evaluator.episode_rewards:
                # Get meaningful evaluation returns
                num_recent = min(evaluator.rollout_batch_size, len(evaluator.episode_rewards))
                recent_eval_returns = evaluator.episode_rewards[-num_recent:]
                meaningful_eval_returns = [r for r in recent_eval_returns if abs(r) > 0.01]
                eval_returns.extend(meaningful_eval_returns)
        
        if eval_returns:
            eval_mean_return = np.mean(eval_returns)
            eval_std_return = np.std(eval_returns)
            logger.info(f"  Evaluation Summary: {len(eval_returns)} episodes, mean return = {eval_mean_return:.2f} ± {eval_std_return:.2f}")
        else:
            logger.info(f"  Evaluation Summary: No completed episodes with meaningful returns")
        
        # Logging
        logs = []
        logs += [rollout_worker.logs('train')]
        logs += [evaluator.logs('test')]
        logs += [policy.logs()]
        
        # Combine all logs
        combined_logs = {}
        for log_dict in logs:
            if isinstance(log_dict, dict):
                combined_logs.update(log_dict)
        
        # Record to tabular logger
        combined_logs['epoch'] = epoch
        for key, val in combined_logs.items():
            logger.record_tabular(key, val)
        
        # Dump tabular data
        logger.dump_tabular()
        
        # Save policies
        if save_policies:
            policy.save_policy(latest_policy_path)
            
            # Save best policy based on test return (MuJoCo environments don't use success rate)
            current_performance = combined_logs.get('test/mean_return', -float('inf'))
            if current_performance > best_mean_return:
                best_mean_return = current_performance
                policy.save_policy(best_policy_path)
                logger.info(f"  *** NEW BEST POLICY *** Saved with mean return: {current_performance:.4f}")
            
            # Periodic saves
            if epoch % policy_save_interval == 0 and epoch > 0:
                policy.save_policy(periodic_policy_path.format(epoch))
    
    logger.info("\n=== Training Completed Successfully! ===\n")


def launch(env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, demo_file,
           bc_loss=None, q_filter=None, num_demo=None, n_cycles=None, n_batches=None, override_params={}, save_policies=True):
    """Launch training with MuJoCo environment."""
    
    rank = 0  # Single process
    
    # Configure logging
    if rank == 0:
        if logdir is None:
            # Create timestamp-based logdir
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]
            # Use logs directory in current working directory (standalone version)
            logdir = os.path.join('logs', f'{env}-{timestamp}')
        
        logger.configure(dir=logdir)
    
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)
    
    # Prepare params
    params = config.prepare_mujoco_params(
        env, 
        replay_strategy=replay_strategy,
        clip_return=clip_return,
        **override_params
    )
    
    # Override with command line arguments
    if bc_loss is not None:
        params['bc_loss'] = bc_loss
    if q_filter is not None:
        params['q_filter'] = q_filter
    if num_demo is not None:
        params['num_demo'] = num_demo
    if n_cycles is not None:
        params['n_cycles'] = n_cycles
    if n_batches is not None:
        params['n_batches'] = n_batches
    
    # Configure dimensions
    dims = config.configure_mujoco_dims(params)
    
    # Log parameters
    config.log_mujoco_params(params, logger)
    
    # Save parameters
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump({k: v for k, v in params.items() if not callable(v)}, f, indent=2)
    
    # Create DDPG policy
    policy = config.create_mujoco_ddpg(dims, params)
    
    # Configure rollout workers
    rollout_params = {
        'T': params['T'],
        'rollout_batch_size': params['rollout_batch_size'],
        'exploit': False,
        'use_target_net': False,
        'compute_Q': True,
        'noise_eps': params['noise_eps'],
        'random_eps': params['random_eps'],
        'render': False,
    }
    
    eval_params = rollout_params.copy()
    eval_params.update({
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'render': False,
    })
    
    rollout_worker = RolloutWorkerMuJoCo(params['make_env'], policy, dims, logger, **rollout_params)
    evaluator = RolloutWorkerMuJoCo(params['make_env'], policy, dims, logger, **eval_params)
    
    # Set random seeds
    rollout_worker.seed(rank_seed)
    evaluator.seed(rank_seed)
    
    # Run training
    train(
        policy=policy,
        rollout_worker=rollout_worker,
        evaluator=evaluator,
        n_epochs=n_epochs,
        policy_save_interval=policy_save_interval,
        save_policies=save_policies,
        demo_file=demo_file,
        **params
    )


@click.command()
@click.option('--env', type=str, default='HalfCheetah-v4', help='the name of the MuJoCo environment')
@click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go')
@click.option('--n_epochs', type=int, default=200, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (always 1 for PyTorch version)')
@click.option('--seed', type=int, default=0, help='the random seed')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved')
@click.option('--replay_strategy', type=str, default='none', help='the HER replay strategy')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default='', help='demo data file path')
@click.option('--bc_loss', type=int, default=0, help='whether or not to use the behavior cloning loss as an auxiliary loss')
@click.option('--q_filter', type=int, default=0, help='whether or not to use the Q-filter')
@click.option('--num_demo', type=int, default=100, help='number of expert demo episodes')
@click.option('--n_cycles', type=int, default=None, help='number of cycles per epoch')
@click.option('--n_batches', type=int, default=None, help='number of batches per cycle')
def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()