"""Base exploration interface — performance-first design.

Every exploration method follows this contract:
1. init(): pre-allocate ALL tensors. Nothing allocated after this.
2. compute_rewards(): ONE batched call on the full rollout buffer. Never per-step.
3. update(): train exploration networks on the minibatch. Called during PPO update.

The interface is designed so compute_rewards() can be called between
PufferLib's evaluate() and train() without breaking cudagraph traces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseExploration(ABC):
    """Base class for high-performance exploration methods.

    All tensors are pre-allocated at init. No dynamic allocation in the hot path.
    """

    def __init__(
        self,
        obs_dim: int,
        n_envs: int,
        rollout_steps: int,
        device: str = "cuda",
        beta: float = 0.01,
        reward_clip: float = 5.0,
        beta_decay: float = 1.0,
        normalize_rewards: bool = False,
    ):
        self.obs_dim = obs_dim
        self.n_envs = n_envs
        self.rollout_steps = rollout_steps
        self.device = device
        self.beta = beta
        self.initial_beta = beta
        self.reward_clip = reward_clip
        self.beta_decay = beta_decay
        self.normalize_rewards = normalize_rewards
        self._step = 0

        # Pre-allocate the intrinsic reward buffer (reused every epoch)
        self.total_steps = rollout_steps * n_envs
        self._intrinsic_rewards = torch.zeros(
            self.total_steps, device=device, dtype=torch.float32
        )

        # Running normalization for intrinsic rewards (only used if normalize_rewards=True)
        self._reward_running_mean = torch.zeros(1, device=device)
        self._reward_running_var = torch.ones(1, device=device)
        self._reward_count = torch.zeros(1, device=device, dtype=torch.long)

    @abstractmethod
    def compute_rewards(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute intrinsic rewards for an entire rollout buffer at once.

        This is called ONCE after rollout collection, not per-step.
        obs/next_obs/actions are the full flattened rollout:
            shape (rollout_steps * n_envs, dim)

        Must write results into self._intrinsic_rewards (pre-allocated).

        Returns:
            Intrinsic rewards, shape (rollout_steps * n_envs,).
        """
        ...

    @abstractmethod
    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor) -> dict:
        """Update exploration networks on a minibatch.

        Called during PPO's minibatch loop. Should be fast.
        Returns metrics dict for logging.
        """
        ...

    def augment_rewards(
        self,
        rewards: torch.Tensor,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute intrinsic rewards and add them to extrinsic rewards in-place.

        This is the main entry point called by the PufferLib integration hook.
        Handles beta scheduling and reward normalization.

        Args:
            rewards: Extrinsic rewards from env, shape (total_steps,). Modified in-place.
            obs: All observations, shape (total_steps, obs_dim).
            next_obs: All next observations, shape (total_steps, obs_dim).
            actions: All actions, shape (total_steps,) or (total_steps, act_dim).

        Returns:
            The augmented rewards tensor (same object, modified in-place).
        """
        # Compute intrinsic rewards (batched, one call)
        intrinsic = self.compute_rewards(obs, next_obs, actions)

        # Clip raw intrinsic rewards to [0, reward_clip]
        intrinsic.clamp_(0.0, self.reward_clip)

        # Add to extrinsic rewards: r_total = r_ext + beta * r_int
        # No normalization — beta is the user's control knob for
        # balancing exploration vs exploitation. Raw intrinsic rewards
        # are clipped but otherwise left at their natural scale.
        rewards.add_(self.beta * intrinsic)

        # Decay beta
        self._step += 1
        self.beta = self.initial_beta * (self.beta_decay ** self._step)

        return rewards

    @torch.no_grad()
    def _update_running_stats(self, values: torch.Tensor):
        """Update running mean/var for reward normalization. Welford's algorithm."""
        batch_mean = values.mean()
        batch_var = values.var()
        batch_count = values.numel()

        delta = batch_mean - self._reward_running_mean
        total_count = self._reward_count + batch_count
        new_mean = self._reward_running_mean + delta * batch_count / total_count.clamp(min=1)
        m_a = self._reward_running_var * self._reward_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self._reward_count * batch_count / total_count.clamp(min=1)
        new_var = m2 / total_count.clamp(min=1)

        self._reward_running_mean.copy_(new_mean)
        self._reward_running_var.copy_(new_var.clamp(min=1e-8))
        self._reward_count.copy_(total_count)

    def get_metrics(self) -> dict:
        """Return metrics for logging."""
        return {
            "explore/beta": self.beta,
            "explore/reward_mean": self._reward_running_mean.item(),
            "explore/reward_var": self._reward_running_var.item(),
            "explore/step": self._step,
        }
