#!/usr/bin/env python3
"""End-to-end test: PufferLib + puffer-explore integration.

Runs a short Breakout training with RND exploration to verify
the integration wiring works with real PufferLib buffers.

Usage (in WSL with puffer-env):
    python scripts/test_e2e_pufferlib.py
"""

import sys
import time


def main():
    print("=== PufferLib + PufferExplore End-to-End Test ===\n")

    # 1. Import PufferLib
    try:
        from pufferlib.pufferl import PuffeRL, load_config
        import pufferlib
        import pufferlib.vector
        print(f"PufferLib {pufferlib.__version__} loaded")
    except ImportError as e:
        print(f"ERROR: PufferLib not available: {e}")
        sys.exit(1)

    # 2. Import puffer-explore
    from puffer_explore.integration import ExploreTrainer
    print("puffer-explore loaded")

    # 3. Set up PufferLib config for CartPole (short run, no Atari deps)
    config = load_config("breakout")  # base config for structure
    train_cfg = config["train"]
    train_cfg["total_timesteps"] = 20_000
    train_cfg["batch_size"] = 512
    train_cfg["bptt_horizon"] = 16
    train_cfg["minibatch_size"] = 256
    train_cfg["max_minibatch_size"] = 256
    train_cfg["update_epochs"] = 1
    train_cfg["device"] = "cuda"
    train_cfg["compile"] = False
    train_cfg["use_rnn"] = False
    train_cfg["env"] = "CartPole-v1"
    train_cfg["data_dir"] = "/tmp/puffer_e2e_test"
    config["wandb"] = False
    config["neptune"] = False

    print(f"\nConfig: {train_cfg['total_timesteps']} timesteps, "
          f"batch={train_cfg['batch_size']}, "
          f"device={train_cfg['device']}")

    # 4. Create a PufferLib-wrapped CartPole env (no Atari deps needed)
    print("\nCreating PufferLib CartPole environment + policy...")
    import gymnasium as gym

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

    # Simple policy for CartPole
    import torch.nn as nn
    import numpy as np

    class CartPolePolicy(nn.Module):
        def __init__(self, obs_space, act_space):
            super().__init__()
            obs_size = obs_space.shape[0]
            n_actions = act_space.n
            self.net = nn.Sequential(
                self._layer_init(nn.Linear(obs_size, 64)),
                nn.ReLU(),
                self._layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
            )
            self.actor = self._layer_init(nn.Linear(64, n_actions), std=0.01)
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
    ).to(train_cfg["device"])

    print(f"  Obs space: {vecenv.single_observation_space.shape}")
    print(f"  Act space: {vecenv.single_action_space}")

    # 5. Create PufferLib trainer
    print("\nCreating PufferLib trainer...")
    trainer = PuffeRL(config["train"], vecenv, policy)

    # Print buffer shapes
    print(f"  observations: {trainer.observations.shape}")
    print(f"  actions:      {trainer.actions.shape}")
    print(f"  rewards:      {trainer.rewards.shape}")

    # 6. Wrap with ExploreTrainer
    print("\nWrapping with ExploreTrainer (method=rnd)...")
    explore_trainer = ExploreTrainer(
        trainer,
        method="rnd",
        device=config["train"]["device"],
        beta=0.01,
    )
    print(f"  obs_dim inferred: {explore_trainer._obs_dim}")
    print(f"  Exploration method: {type(explore_trainer.exploration).__name__}")

    # 7. Run training loop
    print("\n--- Starting training loop ---\n")
    start = time.time()
    n_epochs = 0
    max_epochs = 5

    try:
        while n_epochs < max_epochs:
            # PufferLib evaluate (rollout collection)
            explore_trainer.evaluate()

            # puffer-explore: batch compute intrinsic rewards
            explore_trainer.explore()

            # PufferLib train (PPO update) + exploration network update
            logs = explore_trainer.train()

            n_epochs += 1
            elapsed = time.time() - start

            # Print progress
            if logs and isinstance(logs, dict):
                reward = logs.get("mean_reward", "N/A")
                expl_loss = logs.get("exploration_loss", "N/A")
                print(f"  Epoch {n_epochs}/{max_epochs} | "
                      f"reward={reward} | "
                      f"explore_loss={expl_loss} | "
                      f"time={elapsed:.1f}s")
            else:
                print(f"  Epoch {n_epochs}/{max_epochs} | time={elapsed:.1f}s")

    except Exception as e:
        print(f"\n  ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        trainer.close()
        sys.exit(1)

    elapsed = time.time() - start
    trainer.close()

    print("\n--- Done ---")
    print(f"  {n_epochs} epochs in {elapsed:.1f}s")
    print("  Integration test PASSED")


if __name__ == "__main__":
    main()
