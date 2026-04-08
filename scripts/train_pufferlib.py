#!/usr/bin/env python3
"""Train with PufferLib + intrinsic exploration.

Drop-in replacement for `puffer train` that adds exploration methods.

Usage:
    # Standard PufferLib (no exploration)
    python scripts/train_pufferlib.py --env puffer_breakout

    # PufferLib + RND
    python scripts/train_pufferlib.py --env puffer_breakout --explore rnd --explore-beta 0.01

    # PufferLib + NovelD on a harder env
    python scripts/train_pufferlib.py --env puffer_squared --explore noveld

    # PufferLib + Count-based (fastest, zero NN overhead)
    python scripts/train_pufferlib.py --env puffer_breakout --explore count_based

    # All available methods: rnd, noveld, icm, ngu, ride, count_based, ensemble

Requires PufferLib to be installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="PufferLib + Exploration Training")
    p.add_argument("--env", type=str, default="puffer_breakout",
                    help="PufferLib environment name")

    # Exploration
    p.add_argument("--explore", type=str, default="none",
                    choices=["none", "rnd", "noveld", "icm", "ngu", "ride", "count_based", "ensemble"],
                    help="Exploration method")
    p.add_argument("--explore-beta", type=float, default=0.01,
                    help="Intrinsic reward coefficient")
    p.add_argument("--explore-beta-decay", type=float, default=0.9999,
                    help="Beta decay per epoch")
    p.add_argument("--explore-hidden", type=int, default=128,
                    help="Exploration network hidden dim")
    p.add_argument("--explore-lr", type=float, default=1e-3,
                    help="Exploration network learning rate")

    # PufferLib overrides
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-envs", type=int, default=2048)

    return p.parse_args()


def main():
    args = parse_args()

    try:
        from pufferlib import pufferl
        import pufferlib.ocean  # noqa: F401 — side-effect import
    except ImportError:
        print("PufferLib not installed. Install it first:")
        print("  git clone https://github.com/pufferai/puffertank && cd puffertank && ./docker.sh test")
        print("\nRunning in standalone demo mode instead...\n")
        run_standalone_demo(args)
        return

    # Load PufferLib config and env
    puffer_args = pufferl.load_config(args.env)
    puffer_args['train']['total_timesteps'] = args.total_timesteps
    puffer_args['train']['learning_rate'] = args.learning_rate
    puffer_args['env']['num_envs'] = args.num_envs

    vecenv = pufferl.load_env(args.env, puffer_args)
    policy = pufferl.load_policy(puffer_args, vecenv, args.env)
    trainer = pufferl.PuffeRL(puffer_args['train'], vecenv, policy)

    # Set up exploration
    if args.explore != "none":
        from puffer_explore.integration import ExploreTrainer
        trainer = ExploreTrainer(
            trainer,
            method=args.explore,
            beta=args.explore_beta,
            beta_decay=args.explore_beta_decay,
            hidden_dim=args.explore_hidden,
            lr=args.explore_lr,
        )

    print(f"\n{'='*60}")
    print("  PufferLib + Exploration")
    print(f"  Environment:  {args.env}")
    print(f"  Exploration:  {args.explore}")
    if args.explore != "none":
        print(f"  Beta:         {args.explore_beta}")
        print(f"  Beta decay:   {args.explore_beta_decay}")
    print(f"  Timesteps:    {args.total_timesteps:,}")
    print(f"{'='*60}\n")

    # Training loop
    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        if args.explore != "none" and hasattr(trainer, 'explore'):
            trainer.explore()
        trainer.train()
        trainer.print_dashboard()

    trainer.close()


def run_standalone_demo(args):
    """Fallback demo without PufferLib — uses gymnasium CartPole."""
    import torch
    import gymnasium as gym
    from puffer_explore.integration import StandaloneExploreTrainer, create_exploration

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    explore_kwargs = {
        "beta": args.explore_beta,
        "beta_decay": args.explore_beta_decay,
    }
    if args.explore in ("icm", "ride", "ensemble"):
        explore_kwargs["n_actions"] = n_actions
    if args.explore in ("rnd", "noveld", "icm", "ngu", "ride"):
        explore_kwargs["use_compile"] = False

    if args.explore == "none":
        print("No exploration method selected. Use --explore rnd/noveld/icm/etc.")
        return

    exploration = create_exploration(
        args.explore, obs_dim=obs_dim, n_envs=1, rollout_steps=128,
        device="cpu", **explore_kwargs,
    )

    policy = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, n_actions),
    )

    trainer = StandaloneExploreTrainer(env, exploration, device="cpu")

    print(f"Standalone demo: CartPole + {args.explore}")
    total_reward = 0
    for epoch in range(50):
        rollout = trainer.collect_rollout(policy, rollout_steps=128)
        metrics = exploration.update(
            rollout["obs"], rollout["next_obs"], rollout["actions"]
        )
        mean_r = rollout["raw_rewards"].mean().item()
        mean_ir = (rollout["rewards"] - rollout["raw_rewards"]).abs().mean().item()
        total_reward += rollout["raw_rewards"].sum().item()

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:>3} | "
                f"Reward: {mean_r:.2f} | "
                f"Intrinsic: {mean_ir:.4f} | "
                f"Beta: {exploration.beta:.6f} | "
                f"{metrics}"
            )

    env.close()
    print(f"\nTotal reward: {total_reward:.0f}")


if __name__ == "__main__":
    main()
