import argparse
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def smooth_curve(x, y, window=100):
    """Apply smoothing to a curve."""
    if len(y) < window:
        return x, y
    
    # Simple moving average
    smoothed = []
    for i in range(len(y)):
        start = max(0, i - window // 2)
        end = min(len(y), i + window // 2 + 1)
        smoothed.append(np.mean(y[start:end]))
    
    return x, np.array(smoothed)


def load_csv_results(csv_path):
    """Load results from CSV file."""
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return None
            
        # Parse header
        header = lines[0].strip().split(',')
        
        # Parse data
        data = {}
        for line in lines[1:]:
            if line.strip():
                values = line.strip().split(',')
                if len(values) == len(header):
                    for i, key in enumerate(header):
                        if key not in data:
                            data[key] = []
                        try:
                            data[key].append(float(values[i]))
                        except ValueError:
                            data[key].append(0.0)
        
        return data
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="Root directory containing experiment logs")
    parser.add_argument("--smooth", type=int, default=1, help="Apply curve smoothing")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for progress.csv")
    args = parser.parse_args()

    root = args.root

    # Search for progress.csv
    if args.recursive:
        csv_paths = glob.glob(os.path.join(root, "**", "progress.csv"), recursive=True)
    else:
        csv_paths = glob.glob(os.path.join(root, "*", "progress.csv"))

    if len(csv_paths) == 0:
        print("No progress.csv found.")
        return

    # Data grouped by environment: env_id → list of runs
    env_data = {}

    for p in csv_paths:
        run_dir = os.path.dirname(p)
        params_path = os.path.join(run_dir, "params.json")

        if not os.path.exists(params_path):
            print(f"Skipping (missing params.json): {run_dir}")
            continue

        # Load CSV data
        results = load_csv_results(p)
        if results is None:
            print(f"Skipping invalid CSV: {run_dir}")
            continue

        with open(params_path, "r") as f:
            params = json.load(f)

        env_id = params.get("env_name", "UnknownEnv")

        # Required metrics
        if "test/success_rate" not in results or "test/mean_Q" not in results:
            print(f"Skipping (missing metrics): {run_dir}")
            continue

        epochs = results["epoch"]
        success_rate = results["test/success_rate"]
        mean_q = results["test/mean_Q"]

        # Smooth if needed
        if args.smooth:
            _, success_rate = smooth_curve(epochs, success_rate)
            _, mean_q = smooth_curve(epochs, mean_q)

        entry = (epochs, success_rate, mean_q)
        env_data.setdefault(env_id, []).append(entry)

        print(f"Loaded {run_dir} ({len(epochs)} steps)")

    # ------------------------------
    # Plotting
    # ------------------------------
    for env_id, runs in env_data.items():
        if len(runs) == 0:
            continue

        print(f"Plotting {env_id} with {len(runs)} run(s)")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Success rate
        ax1 = axes[0]
        for epochs, success_rate, _ in runs:
            ax1.plot(epochs, success_rate, alpha=0.7)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Test Success Rate")
        ax1.set_title(f"{env_id} - Success Rate")
        ax1.grid(True)

        # Plot 2: Mean Q
        ax2 = axes[1]
        for epochs, _, mean_q in runs:
            ax2.plot(epochs, mean_q, alpha=0.7)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Test Mean Q")
        ax2.set_title(f"{env_id} - Mean Q-Value")
        ax2.grid(True)

        plt.tight_layout()
        
        # Save plot
        plot_filename = f"plot_{env_id.replace('-', '_')}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot_filename}")
        
        plt.show()


if __name__ == "__main__":
    main()