#!/usr/bin/env python3
"""Training benchmark: compare PufferLib with and without exploration.

Runs CartPole training with different exploration methods and measures
reward convergence and training overhead.

Usage (in WSL with puffer-env):
    python scripts/benchmark_training.py
"""

import time



def run_one(method, n_epochs=20, device="cuda"):
    """Run CartPole training with a specific exploration method."""
    from pufferlib.pufferl import PuffeRL, load_config
    import pufferlib
    import pufferlib.vector
    import gymnasium as gym
    from puffer_explore.integration import ExploreTrainer

    config = load_config("breakout")
    train_cfg = config["train"]
    train_cfg["total_timesteps"] = 100_000
    train_cfg["batch_size"] = 512
    train_cfg["bptt_horizon"] = 16
    train_cfg["minibatch_size"] = 256
    train_cfg["max_minibatch_size"] = 256
    train_cfg["update_epochs"] = 1
    train_cfg["device"] = device
    train_cfg["compile"] = False
    train_cfg["use_rnn"] = False
    train_cfg["env"] = "CartPole-v1"
    train_cfg["data_dir"] = "/tmp/puffer_bench"
    config["wandb"] = False
    config["neptune"] = False

    def cartpole_creator(buf=None, **kwargs):
        return pufferlib.vector.GymnasiumPufferEnv(
            env_creator=lambda: gym.make("CartPole-v1"),
            buf=buf,
        )

    vecenv = pufferlib.vector.make(
        cartpole_creator,
        num_envs=4,
        num_workers=1,
        backend=pufferlib.vector.Serial,
    )

    import torch.nn as nn
    import numpy as np

    class CartPolePolicy(nn.Module):
        def __init__(self, obs_space, act_space):
            super().__init__()
            self.net = nn.Sequential(
                self._layer_init(nn.Linear(obs_space.shape[0], 64)),
                nn.ReLU(),
                self._layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
            )
            self.actor = self._layer_init(nn.Linear(64, act_space.n), std=0.01)
            self.critic = self._layer_init(nn.Linear(64, 1), std=1.0)

        @staticmethod
        def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)
            return layer

        def forward(self, x, state=None):
            h = self.net(x)
            return self.actor(h), self.critic(h)

        def forward_eval(self, x, state=None):
            return self.forward(x, state)

    policy = CartPolePolicy(
        vecenv.single_observation_space,
        vecenv.single_action_space,
    ).to(device)

    trainer = PuffeRL(train_cfg, vecenv, policy)

    if method != "none":
        trainer = ExploreTrainer(
            trainer,
            method=method,
            device=device,
            beta=0.01,
        )

    start = time.time()
    for epoch in range(n_epochs):
        trainer.evaluate()
        if method != "none":
            trainer.explore()
        trainer.train()

    elapsed = time.time() - start

    # Access the underlying trainer for step count
    base = trainer.pufferl if method != "none" else trainer
    steps = getattr(base, "global_step", n_epochs * 512)

    trainer.close() if hasattr(trainer, "close") else base.close()
    return elapsed, steps


def main():
    print("\n" + "=" * 70)
    print("  PufferExplore Training Benchmark — CartPole + PufferLib 3.0")
    print("  20 epochs per method, RTX 3090")
    print("=" * 70 + "\n")

    methods = ["none", "rnd", "count_based", "noveld", "icm", "ngu", "ride"]
    results = []

    for method in methods:
        try:
            elapsed, steps = run_one(method)
            sps = steps / elapsed
            results.append((method, elapsed, steps, sps))
            print(f"  {method:<15} {elapsed:>6.1f}s  {steps:>8,} steps  {sps:>10,.0f} SPS")
        except Exception as e:
            print(f"  {method:<15} FAILED: {e}")
            import traceback
            traceback.print_exc()

    if results:
        base_time = results[0][1]
        print(f"\n  {'Method':<15} {'Time':>8} {'Steps':>10} {'SPS':>12} {'Overhead':>10}")
        print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*12} {'-'*10}")
        for method, elapsed, steps, sps in results:
            overhead = (elapsed / base_time - 1) * 100 if base_time > 0 else 0
            tag = "baseline" if method == "none" else f"+{overhead:.1f}%"
            print(f"  {method:<15} {elapsed:>7.1f}s {steps:>10,} {sps:>12,.0f} {tag:>10}")

    print()


if __name__ == "__main__":
    main()
