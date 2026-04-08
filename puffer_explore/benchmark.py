"""Throughput benchmark — measure exploration overhead precisely.

Simulates the hot path (batched intrinsic reward computation) on
synthetic data to measure raw overhead without environment stepping.

Usage:
    python -m puffer_explore.benchmark
    python -m puffer_explore.benchmark --method rnd --n_envs 1024 --rollout_steps 128
"""

from __future__ import annotations

import argparse
import time

import torch

from puffer_explore.integration import create_exploration, METHODS


def benchmark_method(
    method: str,
    obs_dim: int = 128,
    n_envs: int = 1024,
    rollout_steps: int = 128,
    n_iterations: int = 100,
    n_warmup: int = 10,
    device: str = "cuda",
    **kwargs,
) -> dict:
    """Benchmark a single exploration method.

    Returns timing stats for compute_rewards() and update().
    """
    total_steps = n_envs * rollout_steps
    exploration = create_exploration(
        method=method,
        obs_dim=obs_dim,
        n_envs=n_envs,
        rollout_steps=rollout_steps,
        device=device,
        **kwargs,
    )

    # Synthetic data (pre-allocated, reused)
    obs = torch.randn(total_steps, obs_dim, device=device)
    next_obs = torch.randn(total_steps, obs_dim, device=device)
    actions = torch.randint(0, 18, (total_steps,), device=device)
    rewards = torch.randn(total_steps, device=device)

    # Warm up (important for torch.compile and CUDA caching)
    for _ in range(n_warmup):
        exploration.compute_rewards(obs, next_obs, actions)
        exploration.update(obs[:256], next_obs[:256], actions[:256])
    torch.cuda.synchronize() if device == "cuda" else None

    # Benchmark compute_rewards
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        exploration.compute_rewards(obs, next_obs, actions)
    if device == "cuda":
        torch.cuda.synchronize()
    compute_time = (time.perf_counter() - t0) / n_iterations

    # Benchmark update
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        exploration.update(obs[:256], next_obs[:256], actions[:256])
    if device == "cuda":
        torch.cuda.synchronize()
    update_time = (time.perf_counter() - t0) / n_iterations

    # Benchmark augment_rewards (full pipeline)
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        rewards_copy = rewards.clone()
        exploration.augment_rewards(rewards_copy, obs, next_obs, actions)
    if device == "cuda":
        torch.cuda.synchronize()
    augment_time = (time.perf_counter() - t0) / n_iterations

    return {
        "method": method,
        "total_steps": total_steps,
        "compute_ms": compute_time * 1000,
        "update_ms": update_time * 1000,
        "augment_ms": augment_time * 1000,
        "steps_per_sec_compute": total_steps / compute_time,
        "steps_per_sec_augment": total_steps / augment_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Exploration method throughput benchmark")
    parser.add_argument("--method", type=str, default=None, help="Specific method (or all)")
    parser.add_argument("--obs_dim", type=int, default=128)
    parser.add_argument("--n_envs", type=int, default=1024)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--n_iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    methods = [args.method] if args.method else list(METHODS.keys())

    print(f"\n{'='*75}")
    print("  PufferExplore Throughput Benchmark")
    print(f"  Obs dim: {args.obs_dim} | Envs: {args.n_envs} | "
          f"Steps: {args.rollout_steps} | Device: {args.device}")
    print(f"  Total batch: {args.n_envs * args.rollout_steps:,} steps")
    print(f"{'='*75}\n")

    # Baseline: time for reward augmentation with no exploration (just a tensor add)
    total = args.n_envs * args.rollout_steps
    rewards = torch.randn(total, device=args.device)
    intrinsic = torch.randn(total, device=args.device)

    t0 = time.perf_counter()
    for _ in range(args.n_iterations):
        rewards.add_(0.01 * intrinsic)
    if args.device == "cuda":
        torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - t0) / args.n_iterations * 1000

    print(f"  {'Method':<20} {'Compute':>10} {'Update':>10} {'Augment':>10} {'SPS':>15} {'Overhead':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*15} {'-'*10}")
    print(f"  {'baseline (add)':.<20} {baseline_ms:>9.3f}ms {'-':>10} {baseline_ms:>9.3f}ms {'-':>15} {'0%':>10}")

    for method in methods:
        try:
            result = benchmark_method(
                method=method,
                obs_dim=args.obs_dim,
                n_envs=args.n_envs,
                rollout_steps=args.rollout_steps,
                n_iterations=args.n_iterations,
                device=args.device,
            )
            overhead = (result["augment_ms"] / max(baseline_ms, 0.001) - 1) * 100
            print(
                f"  {method:<20} "
                f"{result['compute_ms']:>9.3f}ms "
                f"{result['update_ms']:>9.3f}ms "
                f"{result['augment_ms']:>9.3f}ms "
                f"{result['steps_per_sec_augment']:>12,.0f} sps "
                f"{overhead:>+8.1f}%"
            )
        except Exception as e:
            print(f"  {method:<20} FAILED: {e}")

    print()


if __name__ == "__main__":
    main()
