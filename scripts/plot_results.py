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

    print()


if __name__ == "__main__":
    main()
