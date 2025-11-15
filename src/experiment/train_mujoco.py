import os
import sys
import time
import click
import numpy as np
import json
from mpi4py import MPI
import resource

# Add project root for src imports (portable solution)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# OpenAI Baselines should be available via pip install - no hardcoded path needed
# If you need a specific baselines version, use: pip install git+https://github.com/openai/baselines.git@commit_hash

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import src.experiment.mujoco_config as config  # Use MuJoCo-specific config
from src.algorithms.rollout_mujoco import RolloutWorkerMuJoCo  # Use MuJoCo-specific rollout worker
from src.algorithms.ddpg_mujoco import DDPGMuJoCo  # Use MuJoCo-specific DDPG
from src.utils.util import mpi_fork

from subprocess import CalledProcessError


def safe_mpi_average(value):
    """Safely average a value across MPI workers, handling various data types"""
    try:
        # Handle empty arrays properly
        if hasattr(value, 'size'):
            if value.size == 0:  # Use .size for numpy arrays
                return 0.0
        elif value == [] or value is None:
            return 0.0
        
        # If it's already a scalar, just return it for single worker or broadcast it
        if isinstance(value, (int, float, np.number)):
            if MPI.COMM_WORLD.Get_size() == 1:
                return float(value)
            else:
                return mpi_moments(np.array([value]))[0]
        
        # Convert to list if not already
        if not isinstance(value, list):
            value = [value]
        
        return mpi_moments(np.array(value))[0]
    except Exception as e:
        # If MPI fails, return local value or 0.0
        if isinstance(value, (int, float, np.number)):
            return float(value)
        elif hasattr(value, '__len__') and len(value) > 0:
            return np.mean(value)
        else:
            return 0.0


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, demo_file, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

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

    if policy.bc_loss == 1 and demo_file: 
        policy.initDemoBuffer(demo_file)  # Initialize demo buffer if provided
        
    for epoch in range(n_epochs):
        epoch_start = time.time()
        if rank == 0:
            print(f"\n=== EPOCH {epoch+1}/{n_epochs} ===")
            
        # train
        rollout_worker.clear_history()
        for cycle in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            
            # Get episode return from episode data directly
            episode_return = episode['r'].sum() if isinstance(episode['r'], list) else episode['r'].mean()
            
            if rank == 0:
                print(f"  Cycle {cycle+1}/{n_cycles} - Episode return: {episode_return:.2f}")
                
            policy.store_episode(episode)
            
            for batch in range(n_batches):
                policy.train()
            
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
            # Calculate episode return directly from the episode data
            test_return = test_result['r'].sum() if isinstance(test_result['r'], list) else test_result['r'].mean()
            test_episode_returns.append(test_return)
        
        # Calculate epoch-specific statistics
        current_test_return = np.mean(test_episode_returns) if test_episode_returns else 0.0
        
        # For training return, we'll use the actual episode returns from this epoch's cycles
        # Note: rollout_worker.current_success_rate() gives accumulated history, which is misleading
        
        # record logs
        logger.record_tabular('epoch', epoch)
        
        # Override the test return with our epoch-specific calculation
        logger.record_tabular('test/mean_episode_return', safe_mpi_average(current_test_return))
        logger.record_tabular('test/episode', (epoch + 1) * n_test_rollouts)  # Total episodes tested so far
        
        # Get other logs but override the problematic accumulated statistics
        for key, val in evaluator.logs('test'):
            if not key.startswith('mean_episode'):  # Skip the accumulated episode return
                logger.record_tabular(key, safe_mpi_average(val))
                
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, safe_mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, safe_mpi_average(val))

        if rank == 0:
            # Print epoch summary with epoch-specific returns
            # For MuJoCo, get the mean episode return (which replaces success rate)
            train_return_raw = rollout_worker.current_success_rate()  # This returns mean episode return for MuJoCo
            train_return = train_return_raw if isinstance(train_return_raw, (int, float)) else safe_mpi_average(train_return_raw)
            buffer_size = policy.get_current_buffer_size()
            print(f"  Epoch {epoch+1} Summary:")
            print(f"    Test Return (This Epoch): {current_test_return:.2f}")
            print(f"    Train Return (Rolling Avg): {train_return:.2f}")  # Note: this is still accumulated
            print(f"    Buffer Size: {buffer_size}")
            print(f"    Best Return So Far: {best_return:.2f} (Epoch {best_return_epoch+1})")
            
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        # Use epoch-specific test return instead of accumulated history
        if rank == 0 and current_test_return >= best_return and save_policies:
            best_return = current_test_return
            best_return_epoch = epoch
            logger.info('New best episode return: {}. Saving policy to {} ...'.format(best_return, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        if rank == 0:
            print(f"Best episode return so far: {best_return:.2f} (achieved in epoch {best_return_epoch+1})")
        local_uniform = np.random.uniform(size=(1,))
        
    # Final training summary
    if rank == 0:
        print(f"\nTraining Completed!")
        print(f"📈 Final Results:")
        print(f"   • Total epochs completed: {n_epochs}")
        print(f"   • Best episode return: {best_return:.2f}")
        print(f"   • Best epoch: {best_return_epoch+1}")
        print(f"   • Final buffer size: {policy.get_current_buffer_size()}")
        print(f"")
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, demo_file,
    bc_loss=None, q_filter=None, num_demo=0, n_cycles=None, n_batches=None, override_params={}, save_policies=True
):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
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
    config.log_mujoco_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running DDPG on MuJoCo with just a single MPI worker. ' +
            'For better performance, consider using --num_cpu > 1.')
        logger.warn('****************')
        logger.warn()

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
        'use_target_net': params['test_with_polyak'],
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
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies, demo_file=demo_file)


@click.command()
@click.option('--env', type=str, default='HalfCheetah-v4', help='the name of the MuJoCo environment to train on')
@click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=200, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
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