import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from collections import defaultdict
import pandas as pd

plt.style.use('seaborn-v0_8')


def parse_run_name(run_dir):
    """Parse run directory name to extract environment, seed, and method info."""
    dirname = os.path.basename(run_dir)
    
    # Pattern: Environment_cpu1_epochs300_cycles20_seed_method_parts...
    parts = dirname.split('_')
    
    if len(parts) < 5:
        return None
    
    env_name = parts[0]  # HalfCheetah-v4, Hopper-v4, Walker2d-v4
    seed = int(parts[4])  # seed number at index 4
    
    # Reconstruct method from remaining parts (everything after seed)
    if len(parts) > 5:
        method_parts = parts[5:]
        method = '_'.join(method_parts)
    else:
        # Handle case where there might be no method parts (shouldn't happen with our data)
        method = 'unknown'
    
    return {
        'environment': env_name,
        'seed': seed,
        'method': method,
        'full_name': dirname
    }


def load_csv_results(csv_path):
    """Load results from CSV file, extracting only epoch and test return."""
    try:
        with open(csv_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # The structure is: header line, then data lines that start with numbers
        # followed by multi-line state data that we need to skip
        
        epochs = []
        returns = []
        
        # Skip header, process lines that start with numbers
        current_epoch = 0
        for line in lines[1:]:  # Skip header
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts with a number (actual data row)
            parts = line.split(',')
            if len(parts) >= 15:  # Should have enough columns for a complete row
                try:
                    # First part should be train/success_rate (0.0 for many cases)
                    train_success = float(parts[0])
                    
                    # Index 8 should be test/mean_return
                    test_return = float(parts[8])
                    
                    # This looks like a valid data row
                    epochs.append(float(current_epoch))
                    returns.append(test_return)
                    current_epoch += 1
                    
                    # We expect 300 epochs total
                    if current_epoch >= 300:
                        break
                        
                except (ValueError, IndexError):
                    # Not a valid data row, skip
                    continue
        
        if len(epochs) == 0:
            print("No valid data found in {}".format(csv_path))
            return None
            
        return {
            'epoch': np.array(epochs),
            'test_mean_return': np.array(returns)
        }
    except Exception as e:
        print("Error loading {}: {}".format(csv_path, e))
        return None


def smooth_curve(y, window=10):
    """Apply smoothing to a curve using moving average."""
    if len(y) < window:
        return y
    
    smoothed = []
    for i in range(len(y)):
        start = max(0, i - window // 2)
        end = min(len(y), i + window // 2 + 1)
        smoothed.append(np.mean(y[start:end]))
    
    return np.array(smoothed)


def plot_environment_all_methods_one_seed(data, env_name, seed, output_dir):
    """Plot all 7 methods for one environment and one seed."""
    plt.figure(figsize=(12, 8))
    
    methods_order = ['regular', 'IL_expert', 'IL_medium', 'IL_random', 
                    'IL_QF_expert', 'IL_QF_medium', 'IL_QF_random']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
             '#9467bd', '#8c564b', '#e377c2']
    
    method_labels = {
        'regular': 'Vanilla DDPG',
        'IL_expert': 'IL Expert',
        'IL_medium': 'IL Medium',
        'IL_random': 'IL Random',
        'IL_QF_expert': 'IL+QF Expert',
        'IL_QF_medium': 'IL+QF Medium',
        'IL_QF_random': 'IL+QF Random'
    }
    
    for i, method in enumerate(methods_order):
        key = (env_name, seed, method)
        if key in data:
            epochs = data[key]['epoch']
            returns = data[key]['test_mean_return']
            
            # Apply smoothing
            smoothed_returns = smooth_curve(returns, window=5)
            
            plt.plot(epochs, smoothed_returns, 
                    color=colors[i], linewidth=2, 
                    label=method_labels[method])
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Test Mean Return', fontsize=14)
    plt.title('{} - All Methods (Seed {})'.format(env_name.replace("-v4", ""), seed), fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = '{}_seed{}_all_methods.png'.format(env_name.replace("-v4", ""), seed)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(filepath))


def plot_il_qf_vs_il_comparison(data, env_name, seed, output_dir):
    """Plot IL+QF Expert vs IL Expert comparison using the same seed."""
    plt.figure(figsize=(10, 6))
    
    # Plot IL+QF Expert from same seed
    key1 = (env_name, seed, 'IL_QF_expert')
    if key1 in data:
        epochs = data[key1]['epoch']
        returns = data[key1]['test_mean_return']
        smoothed_returns = smooth_curve(returns, window=5)
        plt.plot(epochs, smoothed_returns, 
                color='#9467bd', linewidth=3, 
                label='IL+QF Expert')
    
    # Plot IL Expert from same seed
    key2 = (env_name, seed, 'IL_expert')
    if key2 in data:
        epochs = data[key2]['epoch']
        returns = data[key2]['test_mean_return']
        smoothed_returns = smooth_curve(returns, window=5)
        plt.plot(epochs, smoothed_returns, 
                color='#ff7f0e', linewidth=3, 
                label='IL Expert')
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Test Mean Return', fontsize=14)
    plt.title('{} - IL+QF vs IL Expert Comparison (Seed {})'.format(env_name.replace("-v4", ""), seed), 
             fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = '{}_IL_QF_vs_IL_seed{}_comparison.png'.format(env_name.replace("-v4", ""), seed)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(filepath))


def plot_learning_curves_by_demo_quality(data, output_dir):
    """Plot learning curves grouped by demonstration quality across all environments."""
    qualities = ['expert', 'medium', 'random']
    envs = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Learning Curves by Demonstration Quality', fontsize=20, fontweight='bold')
    
    for env_idx, env_name in enumerate(envs):
        for qual_idx, quality in enumerate(qualities):
            ax = axes[env_idx, qual_idx]
            
            # Plot IL vs IL+QF for this quality
            for seed in [0, 1, 2]:
                il_key = (env_name, seed, f'IL_{quality}')
                qf_key = (env_name, seed, f'IL_QF_{quality}')
                
                if il_key in data:
                    epochs = data[il_key]['epoch']
                    returns = data[il_key]['test_mean_return']
                    smoothed_returns = smooth_curve(returns, window=5)
                    ax.plot(epochs, smoothed_returns, 
                           color='orange', alpha=0.6, linewidth=1)
                
                if qf_key in data:
                    epochs = data[qf_key]['epoch']
                    returns = data[qf_key]['test_mean_return']
                    smoothed_returns = smooth_curve(returns, window=5)
                    ax.plot(epochs, smoothed_returns, 
                           color='purple', alpha=0.6, linewidth=1)
            
            # Add average lines
            il_returns_all = []
            qf_returns_all = []
            common_epochs = None
            
            for seed in [0, 1, 2]:
                il_key = (env_name, seed, f'IL_{quality}')
                qf_key = (env_name, seed, f'IL_QF_{quality}')
                
                if il_key in data:
                    if common_epochs is None:
                        common_epochs = data[il_key]['epoch']
                    il_returns_all.append(smooth_curve(data[il_key]['test_mean_return'], window=5))
                
                if qf_key in data:
                    qf_returns_all.append(smooth_curve(data[qf_key]['test_mean_return'], window=5))
            
            if il_returns_all:
                il_mean = np.mean(il_returns_all, axis=0)
                ax.plot(common_epochs, il_mean, 
                       color='orange', linewidth=3, label='IL')
            
            if qf_returns_all:
                qf_mean = np.mean(qf_returns_all, axis=0)
                ax.plot(common_epochs, qf_mean, 
                       color='purple', linewidth=3, label='IL+QF')
            
            ax.set_title('{} - {}'.format(env_name.replace("-v4", ""), quality.capitalize()))
            ax.grid(True, alpha=0.3)
            if env_idx == 2:  # Bottom row
                ax.set_xlabel('Epoch')
            if qual_idx == 0:  # Left column
                ax.set_ylabel('Test Mean Return')
            if env_idx == 0 and qual_idx == 2:  # Top right
                ax.legend()
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'learning_curves_by_demo_quality.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(filepath))


def plot_final_performance_summary(data, output_dir):
    """Plot final performance comparison across all methods and environments."""
    envs = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']
    methods = ['regular', 'IL_expert', 'IL_medium', 'IL_random', 
              'IL_QF_expert', 'IL_QF_medium', 'IL_QF_random']
    
    # Collect final performance (last 10 epochs average)
    final_perf = defaultdict(list)
    
    for env in envs:
        for method in methods:
            env_method_scores = []
            for seed in [0, 1, 2, 3, 4]:
                key = (env, seed, method)
                if key in data:
                    returns = data[key]['test_mean_return']
                    if len(returns) > 10:
                        # Average of last 10 epochs
                        final_score = np.mean(returns[-10:])
                        env_method_scores.append(final_score)
            
            # Remove extreme outliers (more than 4 standard deviations from median)
            if len(env_method_scores) > 2:
                median = np.median(env_method_scores)
                mad = np.median(np.abs(np.array(env_method_scores) - median))
                if mad > 0:
                    # Use a more lenient threshold (4*MAD instead of 3*MAD)
                    threshold = 4 * mad
                    filtered_scores = [s for s in env_method_scores 
                                     if abs(s - median) <= threshold]
                    # Only filter if we're removing clear outliers (less than half the data)
                    if len(filtered_scores) >= len(env_method_scores) // 2:
                        removed_count = len(env_method_scores) - len(filtered_scores)
                        if removed_count > 0:
                            print("Removed {} outliers for {} {}: {} -> {}".format(
                                removed_count, env, method, 
                                len(env_method_scores), len(filtered_scores)))
                        env_method_scores = filtered_scores
            
            if env_method_scores:
                final_perf[(env, method)] = env_method_scores
    
    # Create bar plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Final Performance Comparison (Last 10 Epochs Average)', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
             '#9467bd', '#8c564b', '#e377c2']
    
    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        
        means = []
        stds = []
        method_labels = []
        
        # Always include all methods to maintain consistent bar positions
        for method in methods:
            method_labels.append(method.replace('_', '\n'))
            
            if (env, method) in final_perf:
                scores = final_perf[(env, method)]
                means.append(np.mean(scores))
                stds.append(np.std(scores))
            else:
                # Missing data - show as zero with no error bar
                means.append(0)
                stds.append(0)
        
        if any(m > 0 for m in means):  # Only plot if we have some non-zero data
            bars = ax.bar(range(len(means)), means, yerr=stds, 
                         color=colors[:len(means)], alpha=0.8, capsize=5)
            
            # Color missing data bars differently (light gray)
            for i, (bar, mean_val) in enumerate(zip(bars, means)):
                if mean_val == 0:
                    bar.set_color('lightgray')
                    bar.set_alpha(0.3)
            
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(method_labels, rotation=45, ha='right')
            ax.set_title('{}'.format(env.replace("-v4", "")))
            ax.set_ylabel('Final Return')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'final_performance_summary.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(filepath))


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive plots from HPRC experiment data')
    parser.add_argument("--data_dir", type=str, 
                       default="hprc_logs/logs",
                       help="Directory containing experiment logs")
    parser.add_argument("--output_dir", type=str, 
                       default="plots",
                       help="Output directory for plots")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all progress.csv files
    csv_files = glob.glob(os.path.join(data_dir, "*/progress.csv"))
    
    if len(csv_files) == 0:
        print("No progress.csv files found in {}".format(data_dir))
        return
    
    print("Found {} experiment runs".format(len(csv_files)))

    # Load all data
    data = {}
    
    for csv_path in csv_files:
        run_dir = os.path.dirname(csv_path)
        run_info = parse_run_name(run_dir)
        
        if run_info is None:
            print("Could not parse run name: {}".format(run_dir))
            continue
        
        results = load_csv_results(csv_path)
        if results is None:
            continue
        
        key = (run_info['environment'], run_info['seed'], run_info['method'])
        data[key] = results
        
        print("Loaded: {} seed={} method={}".format(run_info['environment'], run_info['seed'], run_info['method']))

    print("\nLoaded data for {} runs".format(len(data)))

    # Generate plots
    envs = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']
    
    # 1. Plot all 7 methods for each environment (seed 0)
    print("\n=== Generating all-methods plots ===")
    for env in envs:
        plot_environment_all_methods_one_seed(data, env, 0, output_dir)
    
    # 2. Plot IL+QF vs IL comparisons (same seed comparisons)
    print("\n=== Generating IL+QF vs IL comparison plots ===")
    for env in envs:
        # Generate comparisons for multiple seeds to show consistency
        for seed in [0, 1, 2]:
            plot_il_qf_vs_il_comparison(data, env, seed, output_dir)
    
    # 3. Additional interesting plots
    print("\n=== Generating additional analysis plots ===")
    plot_learning_curves_by_demo_quality(data, output_dir)
    plot_final_performance_summary(data, output_dir)
    
    print("\n=== All plots saved to {} ===".format(output_dir))


if __name__ == "__main__":
    main()