"""PufferLib integration — hooks exploration into PufferLib's training loop.

The integration follows PufferLib's design philosophy:
- No per-step Python overhead during rollout collection
- All exploration computation is batched and runs AFTER rollout collection
- Rewards are augmented in-place in the rollout buffer (no allocation)
- Exploration network updates happen during the PPO minibatch loop

Usage with PufferLib:
    from puffer_explore.integration import ExploreTrainer

    trainer = ExploreTrainer(
        pufferl_trainer,       # existing PufferLib trainer
        method="rnd",          # exploration method
        obs_dim=obs_dim,
        beta=0.01,
    )

    for epoch in range(num_epochs):
        trainer.evaluate()      # PufferLib rollout collection (unchanged)
        trainer.explore()       # NEW: batch-compute intrinsic rewards, augment buffer
        logs = trainer.train()  # PufferLib PPO update + exploration network update
"""

from __future__ import annotations

from typing import Any

import torch

from puffer_explore.methods.base import BaseExploration
from puffer_explore.methods.rnd import RND
from puffer_explore.methods.count_based import CountBased
from puffer_explore.methods.noveld import NovelD
from puffer_explore.methods.icm import ICM
from puffer_explore.methods.ensemble import EnsembleDisagreement
from puffer_explore.methods.ngu import NGU
from puffer_explore.methods.ride import RIDE


# Registry: name → class
METHODS: dict[str, type[BaseExploration]] = {
    "rnd": RND,
    "count_based": CountBased,
    "noveld": NovelD,
    "icm": ICM,
    "ensemble": EnsembleDisagreement,
    "ngu": NGU,
    "ride": RIDE,
}


def create_exploration(
    method: str,
    obs_dim: int,
    n_envs: int = 1024,
    rollout_steps: int = 128,
    device: str = "cuda",
    **kwargs,
) -> BaseExploration:
    """Create an exploration method by name.

    Args:
        method: Method name (rnd, count_based, noveld, icm, ensemble).
        obs_dim: Observation dimension.
        n_envs: Number of parallel environments.
        rollout_steps: Steps per rollout.
        device: Torch device.
        **kwargs: Method-specific arguments (beta, lr, etc.).

    Returns:
        Initialized exploration method.
    """
    if method not in METHODS:
        available = ", ".join(METHODS.keys())
        raise ValueError(f"Unknown method '{method}'. Available: {available}")

    cls = METHODS[method]
    return cls(
        obs_dim=obs_dim,
        n_envs=n_envs,
        rollout_steps=rollout_steps,
        device=device,
        **kwargs,
    )


class ExploreTrainer:
    """Wraps a PufferLib trainer to add exploration.

    Designed as a thin wrapper that intercepts the evaluate→train flow
    and inserts exploration reward computation between them.

    Args:
        pufferl: A PufferLib PuffeRL trainer instance.
        method: Exploration method name.
        obs_dim: Observation dimension.
        n_envs: Number of parallel environments.
        rollout_steps: Steps per rollout.
        device: Torch device.
        **explore_kwargs: Additional exploration method arguments.
    """

    def __init__(
        self,
        pufferl: Any,
        method: str = "rnd",
        obs_dim: int | None = None,
        n_envs: int | None = None,
        rollout_steps: int | None = None,
        device: str = "cuda",
        **explore_kwargs,
    ):
        self.pufferl = pufferl

        # Infer dimensions from PufferLib trainer if not provided
        if obs_dim is None:
            obs_dim = self._infer_obs_dim()
        if n_envs is None:
            n_envs = self._infer_n_envs()
        if rollout_steps is None:
            rollout_steps = self._infer_rollout_steps()

        self.exploration = create_exploration(
            method=method,
            obs_dim=obs_dim,
            n_envs=n_envs,
            rollout_steps=rollout_steps,
            device=device,
            **explore_kwargs,
        )
        self.device = device
        self._obs_dim = obs_dim

    def _infer_obs_dim(self) -> int:
        """Infer obs_dim from PufferLib 3.0 trainer."""
        # PufferLib 3.0: self.observations has shape (segments, horizon, *obs_shape)
        try:
            obs_shape = self.pufferl.observations.shape[2:]
            result = 1
            for s in obs_shape:
                result *= s
            return result
        except AttributeError:
            pass
        # Fallback: try vecenv
        try:
            return self.pufferl.vecenv.single_observation_space.shape[0]
        except (AttributeError, IndexError):
            raise ValueError("Cannot infer obs_dim. Please provide it explicitly.")

    def _infer_n_envs(self) -> int:
        # PufferLib 3.0: segments = batch_size // horizon
        try:
            return self.pufferl.segments
        except AttributeError:
            try:
                return self.pufferl.total_agents
            except AttributeError:
                return 1024

    def _infer_rollout_steps(self) -> int:
        # PufferLib 3.0: horizon = bptt_horizon
        try:
            return self.pufferl.observations.shape[1]
        except (AttributeError, IndexError):
            return 128

    @property
    def epoch(self):
        return self.pufferl.epoch

    @property
    def total_epochs(self):
        return self.pufferl.total_epochs

    def evaluate(self):
        """PufferLib rollout collection -- unchanged."""
        return self.pufferl.evaluate()

    @torch.no_grad()
    def explore(self):
        """Batch-compute intrinsic rewards and augment the rollout buffer.

        Called AFTER evaluate(), BEFORE train().

        PufferLib 3.0 buffer layout:
          self.observations: (segments, horizon, *obs_shape)
          self.actions:      (segments, horizon, *atn_shape)
          self.rewards:      (segments, horizon)

        PufferLib does NOT store next_obs separately. We construct it
        by shifting observations: next_obs[t] = obs[t+1] within each
        segment, with the last step duplicated.
        """
        buf = self.pufferl

        try:
            # (segments, horizon, *obs_shape) -> (segments*horizon, obs_dim)
            obs_flat = buf.observations.reshape(-1, self._obs_dim)

            # Construct next_obs by shifting within each segment
            # obs[:, 1:] gives next obs for steps 0..horizon-2
            # For the last step, duplicate the last observation
            next_obs = torch.empty_like(buf.observations)
            next_obs[:, :-1] = buf.observations[:, 1:]
            next_obs[:, -1] = buf.observations[:, -1]
            next_obs_flat = next_obs.reshape(-1, self._obs_dim)

            actions_flat = buf.actions.reshape(-1)
            rewards_flat = buf.rewards.reshape(-1)
        except AttributeError:
            return

        # Augment rewards in-place (the core operation)
        self.exploration.augment_rewards(
            rewards_flat, obs_flat, next_obs_flat, actions_flat
        )

        # Write augmented rewards back (reshape is a view, so in-place
        # modification of rewards_flat already updates buf.rewards)

    def train(self):
        """PufferLib PPO update + exploration network update."""
        logs = self.pufferl.train()

        # Update exploration networks using a sample from the buffer
        try:
            obs = self.pufferl.observations.reshape(-1, self._obs_dim)
            # Sample a minibatch for the exploration update
            n = min(obs.shape[0], self.pufferl.minibatch_size)
            idx = torch.randperm(obs.shape[0], device=obs.device)[:n]
            mb_obs = obs[idx]

            next_obs = torch.empty_like(self.pufferl.observations)
            next_obs[:, :-1] = self.pufferl.observations[:, 1:]
            next_obs[:, -1] = self.pufferl.observations[:, -1]
            mb_next_obs = next_obs.reshape(-1, self._obs_dim)[idx]

            mb_actions = self.pufferl.actions.reshape(-1)[idx]
            explore_metrics = self.exploration.update(
                mb_obs, mb_next_obs, mb_actions
            )
        except AttributeError:
            explore_metrics = {}

        # Merge exploration metrics into PufferLib's logs
        if isinstance(logs, dict):
            logs.update(explore_metrics)
            logs.update(self.exploration.get_metrics())

        return logs

    def mean_and_log(self):
        """Pass through to PufferLib's logging."""
        return self.pufferl.mean_and_log()

    def print_dashboard(self):
        """Pass through to PufferLib's dashboard."""
        return self.pufferl.print_dashboard()

    def close(self):
        """Pass through to PufferLib's cleanup."""
        return self.pufferl.close()


class StandaloneExploreTrainer:
    """Standalone exploration training loop for testing without PufferLib.

    Uses a simple Gymnasium-compatible environment and vanilla PPO.
    Primarily for unit testing and development.
    """

    def __init__(
        self,
        env,
        exploration: BaseExploration,
        device: str = "cpu",
    ):
        self.env = env
        self.exploration = exploration
        self.device = device

    def collect_rollout(self, policy, rollout_steps: int = 128) -> dict:
        """Collect a rollout and compute intrinsic rewards."""
        obs_list, next_obs_list, actions_list, rewards_list = [], [], [], []

        obs, _ = self.env.reset()
        for _ in range(rollout_steps):
            obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_t).squeeze(0)

            if hasattr(action, 'sample'):
                action = action.sample()

            action_int = action.item() if action.dim() == 0 else action.argmax().item()
            next_obs, reward, terminated, truncated, info = self.env.step(action_int)

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            actions_list.append(action_int)
            rewards_list.append(reward)

            obs = next_obs
            if terminated or truncated:
                obs, _ = self.env.reset()

        # Stack and augment
        obs_t = torch.tensor(obs_list, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs_list, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions_list, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

        augmented = self.exploration.augment_rewards(rewards_t, obs_t, next_obs_t, actions_t)

        return {
            "obs": obs_t,
            "next_obs": next_obs_t,
            "actions": actions_t,
            "rewards": augmented,
            "raw_rewards": torch.tensor(rewards_list, dtype=torch.float32, device=self.device),
        }
