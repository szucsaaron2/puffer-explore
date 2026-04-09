#!/usr/bin/env python3
"""MiniGrid benchmark: compare exploration methods via PufferLib.

Uses PufferLib's vectorized training with puffer-explore exploration
methods on MiniGrid environments.

Usage (in WSL with puffer-env):
    # Quick test (1 method, 1 env, 50K steps)
    python scripts/benchmark_minigrid.py --methods rnd --envs Empty-8x8 --steps 50000 --seeds 0

    # Tier 1: Core comparison (~30 min)
    python scripts/benchmark_minigrid.py --methods none rnd icm noveld count_based ngu ride \
        --envs Empty-8x8 DoorKey-6x6 --steps 300000 --seeds 0 1

    # Tier 2: Hard env (~1-2 hours)
    python scripts/benchmark_minigrid.py --methods none rnd icm noveld count_based ngu ride \
        --envs KeyCorridorS3R2 --steps 1000000 --seeds 0 1

    # Full run (all envs, all methods, 3 seeds)
    python scripts/benchmark_minigrid.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# MiniGrid environments supported
ENVS = {
    "Empty-8x8": {"full_name": "MiniGrid-Empty-8x8-v0", "difficulty": "easy"},
    "DoorKey-6x6": {"full_name": "MiniGrid-DoorKey-6x6-v0", "difficulty": "medium"},
    "KeyCorridorS3R2": {"full_name": "MiniGrid-KeyCorridorS3R2-v0", "difficulty": "hard"},
}

METHODS = ["none", "rnd", "icm", "noveld", "count_based", "ngu", "ride"]


class MiniGridPolicy(nn.Module):
    """PPO policy for MiniGrid with proper initialization and obs scaling.

    MiniGrid observations are 7x7x3 grid encoding (type, color, state)
    flattened to 147 dims, normalized to [0, 1/255*max_val] by the wrapper.
    Raw values are small integers (0-10), so after /255 they are ~0-0.04.
    We rescale by 255/10 ≈ 25.5 to get [0, 1] range for the network.
    """

    # MiniGrid obs are divided by 255 in the wrapper but raw values are 0-10
    OBS_SCALE = 25.5  # = 255/10, rescales [0, 0.04] → [0, 1]

    def __init__(self, obs_space, act_space):
        super().__init__()
        obs_size = obs_space.shape[0]
        n_actions = act_space.n
        self.net = nn.Sequential(
            self._layer_init(nn.Linear(obs_size, 256)),
            nn.ReLU(),
            self._layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(128, n_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(128, 1), std=1.0)

    @staticmethod
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, x, state=None):
        h = self.net(x * self.OBS_SCALE)
        return self.actor(h), self.critic(h)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)


def run_experiment(
    method: str,
    env_name: str,
    seed: int,
    total_steps: int,
    num_envs: int = 16,
    device: str = "cuda",
    batch_size: int = 4096,
    bptt_horizon: int = 256,
    beta: float = 0.01,
    verbose: bool = True,
) -> dict:
    """Run a single PufferLib + exploration experiment on MiniGrid."""
    from pufferlib.pufferl import PuffeRL, load_config
    import pufferlib
    import pufferlib.vector

    # Import MiniGrid wrapper from rl-exploration-lab
    from rl_exploration_lab.envs.minigrid_wrapper import make_wrapped_env

    # Seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)

    # PufferLib config
    config = load_config("breakout")
    train_cfg = config["train"]
    train_cfg["total_timesteps"] = total_steps
    train_cfg["batch_size"] = batch_size
    train_cfg["bptt_horizon"] = bptt_horizon
    train_cfg["minibatch_size"] = min(1024, batch_size)
    train_cfg["max_minibatch_size"] = min(1024, batch_size)
    train_cfg["update_epochs"] = 4
    train_cfg["learning_rate"] = 1e-3
    train_cfg["device"] = device
    train_cfg["compile"] = False
    train_cfg["use_rnn"] = False
    train_cfg["env"] = env_name
    train_cfg["data_dir"] = f"/tmp/puffer_minigrid/{method}_{env_name}_s{seed}"
    train_cfg["checkpoint_interval"] = 999999  # no checkpoints
    config["wandb"] = False
    config["neptune"] = False

    # Create PufferLib vectorized MiniGrid env
    def env_creator(buf=None, seed=seed, **kwargs):
        return pufferlib.vector.GymnasiumPufferEnv(
            env_creator=lambda: make_wrapped_env(env_name, seed=seed),
            buf=buf,
        )

    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=num_envs,
        num_workers=1,
        backend=pufferlib.vector.Serial,
    )

    # Policy
    policy = MiniGridPolicy(
        vecenv.single_observation_space,
        vecenv.single_action_space,
    ).to(device)

    # Create trainer
    trainer = PuffeRL(train_cfg, vecenv, policy)

    # Wrap with exploration
    if method != "none":
        from puffer_explore.integration import ExploreTrainer
        trainer = ExploreTrainer(
            trainer,
            method=method,
            device=device,
            beta=beta,
        )

    # Training loop
    start_time = time.time()
    n_epochs = 0
    target_epochs = total_steps // batch_size

    while n_epochs < target_epochs:
        trainer.evaluate()
        if method != "none":
            trainer.explore()
        trainer.train()
        n_epochs += 1

        # Print progress every 10 epochs
        if verbose and n_epochs % 10 == 0:
            elapsed = time.time() - start_time
            steps = n_epochs * batch_size
            sps = steps / elapsed
            print(f"    [{method}/{env_name}/s{seed}] "
                  f"epoch {n_epochs}/{target_epochs} | "
                  f"{steps:,} steps | {sps:,.0f} SPS | {elapsed:.0f}s")

    elapsed = time.time() - start_time
    total_actual_steps = n_epochs * batch_size

    # Get final stats from PufferLib's internal tracking
    base_trainer = trainer.pufferl if method != "none" else trainer
    base_trainer.close()

    return {
        "method": method,
        "env_name": env_name,
        "seed": seed,
        "total_steps": total_actual_steps,
        "elapsed_seconds": elapsed,
        "sps": total_actual_steps / elapsed,
        "n_epochs": n_epochs,
    }


def parse_args():
    p = argparse.ArgumentParser(description="MiniGrid Exploration Benchmark via PufferLib")
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--envs", nargs="+", default=["Empty-8x8", "DoorKey-6x6"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="results_minigrid")
    return p.parse_args()


def main():
    args = parse_args()

    # PufferLib's load_config parses sys.argv — clear it to avoid conflicts
    import sys
    sys.argv = [sys.argv[0]]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(args.methods) * len(args.envs) * len(args.seeds)
    print(f"\n{'='*70}")
    print("  PufferExplore MiniGrid Benchmark")
    print(f"  Methods:  {len(args.methods)} ({', '.join(args.methods)})")
    print(f"  Envs:     {len(args.envs)} ({', '.join(args.envs)})")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Steps:    {args.steps:,}")
    print(f"  Envs:     {args.num_envs} parallel")
    print(f"  Device:   {args.device}")
    print(f"  Total:    {total_runs} runs")
    print(f"{'='*70}\n")

    all_results = []
    run_idx = 0

    for env_name in args.envs:
        for method in args.methods:
            seed_results = []
            for seed in args.seeds:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] {method} | {env_name} | seed={seed}")

                try:
                    result = run_experiment(
                        method=method,
                        env_name=env_name,
                        seed=seed,
                        total_steps=args.steps,
                        num_envs=args.num_envs,
                        batch_size=args.batch_size,
                        beta=args.beta,
                        device=args.device,
                    )
                    seed_results.append(result)
                    print(f"  Done: {result['elapsed_seconds']:.1f}s, "
                          f"{result['sps']:,.0f} SPS\n")
                except Exception as e:
                    print(f"  FAILED: {e}\n")
                    import traceback
                    traceback.print_exc()

            # Aggregate across seeds
            if seed_results:
                agg = {
                    "method": method,
                    "env_name": env_name,
                    "n_seeds": len(seed_results),
                    "mean_sps": np.mean([r["sps"] for r in seed_results]),
                    "mean_elapsed": np.mean([r["elapsed_seconds"] for r in seed_results]),
                    "total_steps": args.steps,
                    "per_seed": seed_results,
                }
                all_results.append(agg)

                # Save per method-env
                fpath = output_dir / f"{method}_{env_name}.json"
                with open(fpath, "w") as f:
                    json.dump(agg, f, indent=2)

    # Print summary
    if all_results:
        print(f"\n{'='*70}")
        print("  RESULTS SUMMARY")
        print(f"{'='*70}\n")
        print(f"  {'Method':<15} {'Env':<20} {'Seeds':>5} {'SPS':>10} {'Time':>8}")
        print(f"  {'-'*15} {'-'*20} {'-'*5} {'-'*10} {'-'*8}")
        for r in all_results:
            print(f"  {r['method']:<15} {r['env_name']:<20} "
                  f"{r['n_seeds']:>5} {r['mean_sps']:>10,.0f} "
                  f"{r['mean_elapsed']:>7.1f}s")

        # Save full summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {output_dir}/")

    print()


if __name__ == "__main__":
    main()
