"""PufferLib integration — hooks exploration into PufferLib's training loop.

The integration follows PufferLib's design philosophy:
- No per-step Python overhead during rollout collection
- All exploration computation is batched and runs AFTER rollout collection
- Rewards are augmented in-place in the rollout buffer (no allocation)
- Exploration network updates happen during the PPO minibatch loop

Supports both PufferLib 3.0 and 4.0:
    | Aspect | 3.0 | 4.0 |
    |--------|-----|-----|
    | Module | pufferlib.pufferl | pufferlib.torch_pufferl |
    | Buffer | (segments, horizon, *) | (horizon, total_agents, *) |
    | Rollout | evaluate() | rollouts() |
    | Logs | self.stats (dict from info) | self.env_logs |

Usage with PufferLib:
    from puffer_explore.integration import ExploreTrainer

    trainer = ExploreTrainer(
        pufferl_trainer,       # existing PufferLib trainer (3.0 or 4.0)
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
        self._version = self._detect_version()

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

    def _detect_version(self) -> str:
        """Detect PufferLib version from API shape.

        PufferLib 4.0: observations is (horizon, total_agents, obs_size),
                       has rollouts() method.
        PufferLib 3.0: observations is (segments, horizon, *obs_shape),
                       has evaluate() method.
        """
        if hasattr(self.pufferl, "rollouts"):
            return "4.0"
        return "3.0"

    def _infer_obs_dim(self) -> int:
        """Infer obs_dim from PufferLib trainer buffers."""
        obs = self.pufferl.observations
        if self._version == "4.0":
            # (horizon, total_agents, obs_size)
            return obs.shape[-1]
        # 3.0: (segments, horizon, *obs_shape)
        obs_shape = obs.shape[2:]
        result = 1
        for s in obs_shape:
            result *= s
        return result

    def _infer_n_envs(self) -> int:
        if self._version == "4.0":
            # 4.0: total_agents is the canonical attribute
            if hasattr(self.pufferl, "total_agents"):
                return self.pufferl.total_agents
            return self.pufferl.observations.shape[1]
        try:
            return self.pufferl.segments
        except AttributeError:
            try:
                return self.pufferl.total_agents
            except AttributeError:
                return 1024

    def _infer_rollout_steps(self) -> int:
        if self._version == "4.0":
            return self.pufferl.observations.shape[0]
        try:
            return self.pufferl.observations.shape[1]
        except (AttributeError, IndexError):
            return 128

    @property
    def global_step(self):
        return self.pufferl.global_step

    def evaluate(self):
        """PufferLib rollout collection -- unchanged.

        Calls rollouts() on 4.0, evaluate() on 3.0.
        """
        if self._version == "4.0":
            return self.pufferl.rollouts()
        return self.pufferl.evaluate()

    def _build_next_obs(self, observations):
        """Construct next_obs by shifting along the time axis.

        PufferLib does NOT store next_obs. We approximate it:
        next_obs[t] = obs[t+1], with the last step duplicated.

        4.0: time axis is dim 0 — (horizon, agents, obs_size)
        3.0: time axis is dim 1 — (segments, horizon, *obs_shape)
        """
        next_obs = torch.empty_like(observations)
        if self._version == "4.0":
            next_obs[:-1] = observations[1:]
            next_obs[-1] = observations[-1]
        else:
            next_obs[:, :-1] = observations[:, 1:]
            next_obs[:, -1] = observations[:, -1]
        return next_obs

    def _flatten_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Flatten actions to (N,) for discrete or (N, act_dim) for continuous.

        PufferLib 3.0 actions: (segments, horizon) discrete, or
                               (segments, horizon, act_dim) continuous
        PufferLib 4.0 actions: (horizon, agents, num_atns) — always 3D
        """
        if actions.dim() <= 2:
            # Discrete: flatten to (N,)
            return actions.reshape(-1)
        # 3D: continuous or multi-discrete, last dim is action shape
        last_dim = actions.shape[-1]
        if last_dim == 1:
            # Single action per agent — squeeze last dim
            return actions.reshape(-1)
        # Continuous or multi-action: (N, act_dim)
        return actions.reshape(-1, last_dim)

    @torch.no_grad()
    def explore(self):
        """Batch-compute intrinsic rewards and augment the rollout buffer.

        Called AFTER evaluate(), BEFORE train().
        """
        buf = self.pufferl

        try:
            obs_flat = buf.observations.reshape(-1, self._obs_dim)
            next_obs_flat = self._build_next_obs(
                buf.observations
            ).reshape(-1, self._obs_dim)
            actions_flat = self._flatten_actions(buf.actions)
            rewards_flat = buf.rewards.reshape(-1)
        except AttributeError:
            return

        # Get terminal signals for per-episode episodic reset
        dones_flat = None
        try:
            dones_flat = buf.terminals.reshape(-1)
        except AttributeError:
            pass

        # Augment rewards in-place (reshape is a view, so
        # modification of rewards_flat updates buf.rewards)
        self.exploration.augment_rewards(
            rewards_flat, obs_flat, next_obs_flat, actions_flat, dones_flat
        )

    def train(self):
        """PufferLib PPO update + exploration network update."""
        logs = self.pufferl.train()

        # Update exploration networks using a sample from the buffer
        try:
            obs = self.pufferl.observations.reshape(-1, self._obs_dim)
            n_samples = min(obs.shape[0], 512)
            idx = torch.randperm(obs.shape[0], device=obs.device)[:n_samples]
            mb_obs = obs[idx]

            mb_next_obs = self._build_next_obs(
                self.pufferl.observations
            ).reshape(-1, self._obs_dim)[idx]

            mb_actions = self._flatten_actions(self.pufferl.actions)[idx]

            # Normalize obs before updating exploration networks
            # (matching the normalization applied in augment_rewards)
            norm_obs = self.exploration.normalize_obs(mb_obs)
            norm_next_obs = self.exploration.normalize_obs(mb_next_obs)

            explore_metrics = self.exploration.update(
                norm_obs, norm_next_obs, mb_actions
            )
        except AttributeError:
            explore_metrics = {}

        # Merge exploration metrics into PufferLib's logs
        if isinstance(logs, dict):
            logs.update(explore_metrics)
            logs.update(self.exploration.get_metrics())

        return logs

    def log(self):
        """Pass through to PufferLib's logging.

        4.0: returns dict from log() method
        3.0: returns dict from mean_and_log()
        """
        if hasattr(self.pufferl, "log"):
            return self.pufferl.log()
        if hasattr(self.pufferl, "mean_and_log"):
            return self.pufferl.mean_and_log()
        return {}

    def close(self):
        """Pass through to PufferLib's cleanup (works for both 3.0 and 4.0)."""
        if hasattr(self.pufferl, "close"):
            return self.pufferl.close()
        return None

    def print_dashboard(self, *args, **kwargs):
        """Pass through to dashboard if available (3.0 only; 4.0 has different interface)."""
        if hasattr(self.pufferl, "print_dashboard"):
            return self.pufferl.print_dashboard(*args, **kwargs)
        return None

    @property
    def epoch(self):
        """Current epoch (both versions have this)."""
        return getattr(self.pufferl, "epoch", 0)

    @property
    def total_epochs(self):
        """Total epochs (4.0 has this, 3.0 may not)."""
        return getattr(self.pufferl, "total_epochs", float("inf"))


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
