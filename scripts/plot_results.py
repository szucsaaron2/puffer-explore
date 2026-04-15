#!/usr/bin/env python3
"""Generate comparison plots from MiniGrid benchmark results.

Usage:
    python scripts/plot_results.py --results results_minigrid --output docs/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Consistent colors per method
METHOD_COLORS = {
    "none": "#888888",
    "count_based": "#2196F3",
    "rnd": "#4CAF50",
    "icm": "#FF9800",
    "noveld": "#9C27B0",
    "ngu": "#F44336",
    "ride": "#00BCD4",
    "ensemble": "#795548",
}

METHOD_LABELS = {
    "none": "PPO (baseline)",
    "count_based": "Count-Based",
    "rnd": "RND",
    "icm": "ICM",
    "noveld": "NovelD",
    "ngu": "NGU",
    "ride": "RIDE",
    "ensemble": "Ensemble",
}


def plot_solve_rate_bar(results: list[dict], env_name: str, output_path: Path):
    """Bar chart of solve rates per method."""
    methods = [r["method"] for r in results]
    solve_rates = [r.get("mean_solve_rate", 0) * 100 for r in results]
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(methods)), solve_rates, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.7)

    # Add value labels on bars
    for bar, val in zip(bars, solve_rates):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=11,
                    fontweight="bold")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title(f"Exploration Methods on {env_name}\n5M steps, 2 seeds, RTX 3090",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(solve_rates) * 1.2 if max(solve_rates) > 0 else 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Add horizontal line at 0% for reference
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_reward_bar(results: list[dict], env_name: str, output_path: Path):
    """Bar chart of mean rewards per method."""
    methods = [r["method"] for r in results]
    rewards = [r.get("mean_reward", 0) for r in results]
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(methods)), rewards, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.7)

    for bar, val in zip(bars, rewards):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title(f"Mean Reward on {env_name}\n5M steps, 2 seeds, RTX 3090",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(rewards) * 1.3 if max(rewards) > 0 else 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_combined(results: list[dict], env_name: str, output_path: Path):
    """Combined plot: solve rate + reward side by side."""
    methods = [r["method"] for r in results]
    solve_rates = [r.get("mean_solve_rate", 0) * 100 for r in results]
    rewards = [r.get("mean_reward", 0) for r in results]
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Solve rate
    bars1 = ax1.bar(range(len(methods)), solve_rates, color=colors,
                    edgecolor="white", linewidth=1.5, width=0.7)
    for bar, val in zip(bars1, solve_rates):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=10,
                     fontweight="bold")
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax1.set_ylabel("Solve Rate (%)", fontsize=12)
    ax1.set_title("Solve Rate", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, max(max(solve_rates) * 1.25, 10))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    # Reward
    bars2 = ax2.bar(range(len(methods)), rewards, color=colors,
                    edgecolor="white", linewidth=1.5, width=0.7)
    for bar, val in zip(bars2, rewards):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                     fontweight="bold")
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax2.set_ylabel("Mean Reward", fontsize=12)
    ax2.set_title("Mean Reward", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, max(max(rewards) * 1.3, 0.05))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Exploration Methods on {env_name}\n5M steps | 2 seeds | RTX 3090 | PufferLib 3.0",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_reward_curves(results: list[dict], env_name: str, output_path: Path):
    """Training curves: reward vs timesteps for all methods."""
    fig, ax = plt.subplots(figsize=(12, 7))

    has_data = False
    for r in results:
        method = r["method"]
        color = METHOD_COLORS.get(method, "#888888")
        label = METHOD_LABELS.get(method, method)

        # Collect history across seeds
        seed_histories = []
        for seed_result in r.get("per_seed", []):
            steps = seed_result.get("history_steps", [])
            rewards = seed_result.get("history_reward", [])
            if steps and rewards:
                seed_histories.append((steps, rewards))

        if not seed_histories:
            continue

        has_data = True

        # Average across seeds (interpolate to common x-axis)
        max_step = max(s[-1] for s, _ in seed_histories)
        x_common = np.linspace(0, max_step, 100)
        y_all = []
        for steps, rewards in seed_histories:
            y_interp = np.interp(x_common, steps, rewards)
            y_all.append(y_interp)

        y_mean = np.mean(y_all, axis=0)

        # Smooth with rolling average
        window = 5
        if len(y_mean) > window:
            y_smooth = np.convolve(y_mean, np.ones(window) / window, mode="same")
        else:
            y_smooth = y_mean

        ax.plot(x_common, y_smooth, color=color, label=label,
                linewidth=2.5, alpha=0.9)

        # Show std band if multiple seeds
        if len(y_all) > 1:
            y_std = np.std(y_all, axis=0)
            if len(y_std) > window:
                y_std = np.convolve(y_std, np.ones(window) / window, mode="same")
            ax.fill_between(x_common, y_smooth - y_std, y_smooth + y_std,
                            color=color, alpha=0.15)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Training Steps", fontsize=13)
    ax.set_ylabel("Average Reward", fontsize=13)
    ax.set_title(f"Reward vs Training Steps on {env_name}\n"
                 f"2 seeds | RTX 3090 | PufferLib 3.0",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    # Format x-axis with M/K suffixes
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
    ))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MiniGrid benchmark results")
    parser.add_argument("--results", type=str, default="results_minigrid")
    parser.add_argument("--output", type=str, default="docs/plots")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)

    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json found in {results_dir}")
        return

    with open(summary_path) as f:
        all_results = json.load(f)

    # Group by environment
    envs = {}
    for r in all_results:
        env = r["env_name"]
        if env not in envs:
            envs[env] = []
        envs[env].append(r)

    for env_name, results in envs.items():
        print(f"\n  Plotting {env_name} ({len(results)} methods)")
        safe_name = env_name.replace("-", "_").replace("/", "_")

        plot_solve_rate_bar(results, env_name, output_dir / f"solve_rate_{safe_name}.png")
        plot_reward_bar(results, env_name, output_dir / f"reward_{safe_name}.png")
        plot_combined(results, env_name, output_dir / f"combined_{safe_name}.png")
        plot_reward_curves(results, env_name, output_dir / f"reward_curve_{safe_name}.png")

    print()


if __name__ == "__main__":
    main()
