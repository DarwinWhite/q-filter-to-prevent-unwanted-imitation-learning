#!/usr/bin/env python3
"""
Plot training curves from progress.csv files produced by the PyTorch MuJoCo experiments.

Outputs:
 - <env_id>_success_rate.png
 - <env_id>_mean_Q.png

The script searches recursively for progress.csv files, loads params.json
to determine the environment name, and groups runs accordingly.

Usage:
    python plot.py results/ --smooth 1
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import glob

# ------------------------------
# Utility functions
# ------------------------------

def smooth_curve(x, y, ratio=0.01):
    """Smooths a curve using convolution."""
    if len(y) < 3:
        return x, y
    halfwidth = int(np.ceil(len(x) * ratio))
    k = max(1, halfwidth)
    kernel = np.ones(2 * k + 1)
    y_smooth = np.convolve(y, kernel, mode='same') / np.convolve(np.ones_like(y), kernel, mode='same')
    return x, y_smooth


def load_csv_results(path):
    """Load progress.csv into dict of numpy arrays."""
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        header = f.readline().strip().split(",")

    try:
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
    except Exception:
        return None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    results = {header[i]: data[:, i] for i in range(len(header))}
    return results


def pad_to_same_length(arr_list, value=np.nan):
    """Pad (N, variable_len) arrays to same length."""
    maxlen = max(len(a) for a in arr_list)

    padded = []
    for a in arr_list:
        if len(a) == maxlen:
            padded.append(a)
            continue
        pad = np.full((maxlen - len(a),), value)
        padded.append(np.concatenate([a, pad], axis=0))
    return np.array(padded)


# ------------------------------
# Main plotting logic
# ------------------------------

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
        print(f"Generating plots for {env_id} ...")

        epochs_list = [r[0] for r in runs]
        sr_list = [r[1] for r in runs]
        q_list = [r[2] for r in runs]

        # Pad
        epochs_pad = pad_to_same_length(epochs_list)
        sr_pad = pad_to_same_length(sr_list)
        q_pad = pad_to_same_length(q_list)

        # Median + IQR
        sr_med = np.nanmedian(sr_pad, axis=0)
        sr_p25 = np.nanpercentile(sr_pad, 25, axis=0)
        sr_p75 = np.nanpercentile(sr_pad, 75, axis=0)

        q_med = np.nanmedian(q_pad, axis=0)
        q_p25 = np.nanpercentile(q_pad, 25, axis=0)
        q_p75 = np.nanpercentile(q_pad, 75, axis=0)

        # --- Success Rate Plot ---
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_pad[0], sr_med, label="Success Rate")
        plt.fill_between(epochs_pad[0], sr_p25, sr_p75, alpha=0.25)
        plt.xlabel("Epoch")
        plt.ylabel("Success Rate")
        plt.title(f"{env_id} – Success Rate")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(root, f"{env_id}_success_rate.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

        # --- Mean Q Plot ---
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_pad[0], q_med, label="Mean Q")
        plt.fill_between(epochs_pad[0], q_p25, q_p75, alpha=0.25)
        plt.xlabel("Epoch")
        plt.ylabel("Mean Q")
        plt.title(f"{env_id} – Mean Q Value")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(root, f"{env_id}_mean_Q.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
