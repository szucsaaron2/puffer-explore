#!/usr/bin/env python3
"""MuJoCo benchmark: compare exploration methods on continuous control tasks.

Uses PufferLib's vectorized training with puffer-explore exploration methods
on MuJoCo environments. Continuous actions (Gaussian policy).

Usage (in WSL with puffer-env):
    # Quick sanity check
    python scripts/benchmark_mujoco.py --methods none rnd --envs Hopper-v4 \
        --seeds 0 --steps 100000

    # Full comparison (~2-3 hours)
    python scripts/benchmark_mujoco.py \
        --methods none rnd noveld count_based ngu \
        --envs HalfCheetah-v4 Hopper-v4 \
        --seeds 0 1 --steps 2000000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Classic MuJoCo envs
ENVS = {
    "HalfCheetah-v4": {"full_name": "HalfCheetah-v4", "difficulty": "medium"},
    "Hopper-v4": {"full_name": "Hopper-v4", "difficulty": "medium"},
    "Walker2d-v4": {"full_name": "Walker2d-v4", "difficulty": "medium"},
    "Ant-v4": {"full_name": "Ant-v4", "difficulty": "hard"},
    "InvertedPendulum-v4": {"full_name": "InvertedPendulum-v4", "difficulty": "easy"},
    "Humanoid-v4": {"full_name": "Humanoid-v4", "difficulty": "very_hard"},
}

# RND/NGU/NovelD/Count-Based work fine with continuous actions
# (they only look at observations, not actions)
# ICM/RIDE need actions — we skip them for continuous control by default
METHODS = ["none", "rnd", "noveld", "count_based", "ngu"]


class MuJoCoPolicy(nn.Module):
    """PPO policy for MuJoCo with Gaussian continuous actions."""

    def __init__(self, obs_space, act_space):
        super().__init__()
        obs_size = obs_space.shape[0]
        act_size = act_space.shape[0]

        self.net = nn.Sequential(
            self._layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_mean = self._layer_init(nn.Linear(64, act_size), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_size))
        self.critic = self._layer_init(nn.Linear(64, 1), std=1.0)

    @staticmethod
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, x, state=None):
        """Returns Normal distribution (for sample_logits) + value."""
        # MuJoCo obs are float64, cast to float32 for the network
        x = x.float()
        h = self.net(x)
        mean = self.actor_mean(h)
        std = self.actor_logstd.expand_as(mean).exp()
        value = self.critic(h)
        # PufferLib's sample_logits handles torch.distributions.Normal
        dist = torch.distributions.Normal(mean, std)
        return dist, value

    def forward_eval(self, x, state=None):
        return self.forward(x, state)


def run_experiment(
    method: str,
    env_name: str,
    seed: int,
    total_steps: int,
    num_envs: int = 16,
    device: str = "cuda",
    batch_size: int = 2048,
    bptt_horizon: int = 64,
    beta: float = 0.01,
    verbose: bool = True,
) -> dict:
    """Run a single PufferLib + exploration experiment on MuJoCo."""
    from pufferlib.pufferl import PuffeRL, load_config
    import pufferlib
    import pufferlib.vector
    import gymnasium as gym

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Resolve env name
    gym_id = ENVS[env_name]["full_name"] if env_name in ENVS else env_name

    # PufferLib config
    config = load_config("breakout")
    train_cfg = config["train"]
    train_cfg["total_timesteps"] = total_steps
    train_cfg["batch_size"] = batch_size
    train_cfg["bptt_horizon"] = bptt_horizon
    train_cfg["minibatch_size"] = min(256, batch_size)
    train_cfg["max_minibatch_size"] = min(256, batch_size)
    train_cfg["update_epochs"] = 10
    train_cfg["learning_rate"] = 3e-4
    train_cfg["gamma"] = 0.99
    train_cfg["gae_lambda"] = 0.95
    train_cfg["clip_coef"] = 0.2
    train_cfg["ent_coef"] = 0.0  # Entropy regularization harmful for continuous
    train_cfg["device"] = device
    train_cfg["compile"] = False
    train_cfg["use_rnn"] = False
    train_cfg["env"] = env_name
    train_cfg["data_dir"] = f"/tmp/puffer_mujoco/{method}_{env_name}_s{seed}"
    train_cfg["checkpoint_interval"] = 999999
    config["wandb"] = False
    config["neptune"] = False

    # PufferLib-wrapped MuJoCo env
    def env_creator(buf=None, seed=seed, **kwargs):
        return pufferlib.vector.GymnasiumPufferEnv(
            env_creator=lambda: gym.make(gym_id),
            buf=buf,
        )

    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=num_envs,
        num_workers=1,
        backend=pufferlib.vector.Serial,
    )

    policy = MuJoCoPolicy(
        vecenv.single_observation_space,
        vecenv.single_action_space,
    ).to(device)

    trainer = PuffeRL(train_cfg, vecenv, policy)
    base_trainer = trainer

    explore_trainer = None
    if method != "none":
        from puffer_explore.integration import ExploreTrainer
        explore_trainer = ExploreTrainer(
            trainer,
            method=method,
            device=device,
            beta=beta,
        )

    active_trainer = explore_trainer if explore_trainer else trainer

    start_time = time.time()
    n_epochs = 0
    target_epochs = total_steps // batch_size
    best_reward = -float("inf")

    history_steps = []
    history_reward = []
    log_interval = max(1, target_epochs // 100)

    mean_intrinsic = 0.0

    while n_epochs < target_epochs:
        active_trainer.evaluate()

        # Capture reward BEFORE train() clears stats
        reward = 0.0
        if "episode/r" in base_trainer.stats:
            vals = base_trainer.stats["episode/r"]
            if vals:
                reward = np.mean(vals) if isinstance(vals, list) else vals

        best_reward = max(best_reward, reward)

        if n_epochs % log_interval == 0:
            history_steps.append(n_epochs * batch_size)
            history_reward.append(reward)

        if explore_trainer is not None:
            active_trainer.explore()
            intrinsic = explore_trainer.exploration.compute_rewards(
                base_trainer.observations.reshape(-1, explore_trainer._obs_dim),
                explore_trainer._build_next_obs(
                    base_trainer.observations
                ).reshape(-1, explore_trainer._obs_dim),
                base_trainer.actions.reshape(-1)
                if base_trainer.actions.dim() == 2
                else base_trainer.actions.reshape(-1, base_trainer.actions.shape[-1]),
            )
            mean_intrinsic = intrinsic.mean().item()
            max_intrinsic = intrinsic.max().item()
            base_trainer.stats["explore/intrinsic_mean"].append(mean_intrinsic)
            base_trainer.stats["explore/intrinsic_max"].append(max_intrinsic)
            base_trainer.stats["explore/beta"].append(
                explore_trainer.exploration.beta
            )

        active_trainer.train()
        n_epochs += 1

        if verbose and n_epochs % 20 == 0:
            elapsed = time.time() - start_time
            steps = n_epochs * batch_size
            sps = steps / elapsed
            ir_str = ""
            if explore_trainer:
                ir_str = f" | intrinsic={mean_intrinsic:.4f}"
            print(f"    [{method}/{env_name}/s{seed}] "
                  f"epoch {n_epochs}/{target_epochs} | "
                  f"{steps:,} steps | {sps:,.0f} SPS | "
                  f"r={reward:.2f} (best={best_reward:.2f})"
                  f"{ir_str} | {elapsed:.0f}s")

    history_steps.append(n_epochs * batch_size)
    history_reward.append(reward)

    elapsed = time.time() - start_time
    total_actual_steps = n_epochs * batch_size
    base_trainer.close()

    return {
        "method": method,
        "env_name": env_name,
        "seed": seed,
        "total_steps": total_actual_steps,
        "elapsed_seconds": elapsed,
        "sps": total_actual_steps / elapsed,
        "n_epochs": n_epochs,
        "best_reward": best_reward,
        "final_reward": reward,
        "history_steps": history_steps,
        "history_reward": history_reward,
    }


def parse_args():
    p = argparse.ArgumentParser(description="MuJoCo Exploration Benchmark")
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--envs", nargs="+", default=["Hopper-v4", "HalfCheetah-v4"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="results_mujoco")
    return p.parse_args()


def main():
    args = parse_args()
    import sys
    sys.argv = [sys.argv[0]]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(args.methods) * len(args.envs) * len(args.seeds)
    print(f"\n{'='*70}")
    print("  PufferExplore MuJoCo Benchmark")
    print(f"  Methods:  {len(args.methods)} ({', '.join(args.methods)})")
    print(f"  Envs:     {len(args.envs)} ({', '.join(args.envs)})")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Steps:    {args.steps:,}")
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
                          f"{result['sps']:,.0f} SPS, "
                          f"best_r={result['best_reward']:.2f}, "
                          f"final_r={result['final_reward']:.2f}\n")
                except Exception as e:
                    print(f"  FAILED: {e}\n")
                    import traceback
                    traceback.print_exc()

            if seed_results:
                agg = {
                    "method": method,
                    "env_name": env_name,
                    "n_seeds": len(seed_results),
                    "mean_sps": np.mean([r["sps"] for r in seed_results]),
                    "mean_elapsed": np.mean([r["elapsed_seconds"] for r in seed_results]),
                    "mean_best_reward": np.mean([r["best_reward"] for r in seed_results]),
                    "mean_final_reward": np.mean([r["final_reward"] for r in seed_results]),
                    "total_steps": args.steps,
                    "per_seed": seed_results,
                }
                all_results.append(agg)

                fpath = output_dir / f"{method}_{env_name}.json"
                with open(fpath, "w") as f:
                    json.dump(agg, f, indent=2)

    if all_results:
        print(f"\n{'='*70}")
        print("  RESULTS SUMMARY")
        print(f"{'='*70}\n")
        print(f"  {'Method':<12} {'Env':<22} {'Seeds':>5} {'BestR':>10} {'FinalR':>10} {'SPS':>8}")
        print(f"  {'-'*12} {'-'*22} {'-'*5} {'-'*10} {'-'*10} {'-'*8}")
        for r in all_results:
            print(f"  {r['method']:<12} {r['env_name']:<22} "
                  f"{r['n_seeds']:>5} "
                  f"{r.get('mean_best_reward', 0):>10.2f} "
                  f"{r.get('mean_final_reward', 0):>10.2f} "
                  f"{r['mean_sps']:>8,.0f}")

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {output_dir}/")

    print()


if __name__ == "__main__":
    main()
