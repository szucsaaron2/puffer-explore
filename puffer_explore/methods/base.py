"""Base exploration interface — performance-first design.

Every exploration method follows this contract:
1. init(): pre-allocate ALL tensors. Nothing allocated after this.
2. compute_rewards(): ONE batched call on the full rollout buffer. Never per-step.
3. update(): train exploration networks on the minibatch. Called during PPO update.

The interface is designed so compute_rewards() can be called between
PufferLib's evaluate() and train() without breaking cudagraph traces.

Observation normalization (following RND paper):
  Observations are normalized to mean=0, var=1 before feeding to
  exploration networks (target, predictor, encoder). This is critical
  because raw observations (e.g., MiniGrid values 0-0.04) produce
  nearly identical outputs from randomly initialized networks.

Reward normalization (following RND paper):
  Intrinsic rewards are divided by running std (NO mean subtraction).
  This keeps the reward scale stable as prediction errors change over
  training, without suppressing the mean signal.
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
    ):
        self.obs_dim = obs_dim
        self.n_envs = n_envs
        self.rollout_steps = rollout_steps
        self.device = device
        self.beta = beta
        self.initial_beta = beta
        self.reward_clip = reward_clip
        self.beta_decay = beta_decay
        self._step = 0

        # Pre-allocate the intrinsic reward buffer (reused every epoch)
        self.total_steps = rollout_steps * n_envs
        self._intrinsic_rewards = torch.zeros(
            self.total_steps, device=device, dtype=torch.float32
        )

        # Observation normalization (RunningMeanStd, per-dimension)
        self._obs_mean = torch.zeros(obs_dim, device=device)
        self._obs_var = torch.ones(obs_dim, device=device)
        self._obs_count = torch.tensor(1e-4, device=device)

        # Reward normalization (running var only, no mean subtraction)
        self._reward_running_var = torch.ones(1, device=device)
        self._reward_count = torch.tensor(1e-4, device=device)

    @torch.no_grad()
    def _update_obs_stats(self, obs: torch.Tensor):
        """Update running mean/var for observation normalization."""
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0)
        batch_count = obs.shape[0]

        delta = batch_mean - self._obs_mean
        total_count = self._obs_count + batch_count
        new_mean = self._obs_mean + delta * batch_count / total_count
        m_a = self._obs_var * self._obs_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self._obs_count * batch_count / total_count
        new_var = m2 / total_count

        self._obs_mean.copy_(new_mean)
        self._obs_var.copy_(new_var.clamp(min=1e-8))
        self._obs_count.copy_(total_count)

    @torch.no_grad()
    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations: (obs - mean) / sqrt(var), clipped to [-5, 5]."""
        return ((obs - self._obs_mean) / self._obs_var.sqrt()).clamp(-5.0, 5.0)

    @torch.no_grad()
    def _update_reward_stats(self, values: torch.Tensor):
        """Update running variance for reward normalization."""
        batch_var = values.var()
        batch_count = values.numel()

        total_count = self._reward_count + batch_count
        delta_var = batch_var - self._reward_running_var
        new_var = self._reward_running_var + delta_var * batch_count / total_count

        self._reward_running_var.copy_(new_var.clamp(min=1e-8))
        self._reward_count.copy_(total_count)

    @abstractmethod
    def compute_rewards(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute intrinsic rewards for an entire rollout buffer at once.

        IMPORTANT: obs and next_obs are already normalized by augment_rewards().
        Subclasses should NOT normalize again.

        This is called ONCE after rollout collection, not per-step.
        obs/next_obs/actions are the full flattened rollout:
            shape (rollout_steps * n_envs, dim)

        Args:
            dones: Terminal signals, shape (rollout_steps * n_envs,).
                Used by episodic methods (NovelD ERIR, NGU) to scope
                state counts per-episode. None if not available.

        Returns:
            Intrinsic rewards, shape (rollout_steps * n_envs,).
        """
        ...

    @abstractmethod
    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor) -> dict:
        """Update exploration networks on a minibatch.

        IMPORTANT: obs and next_obs should be pre-normalized by the caller
        (ExploreTrainer.train() handles this).

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
        dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute intrinsic rewards and add them to extrinsic rewards in-place.

        Pipeline (following RND paper):
        1. Update observation running stats from next_obs
        2. Normalize obs and next_obs to mean=0, var=1
        3. Compute intrinsic rewards using normalized observations
        4. Normalize intrinsic rewards by dividing by running std (no mean subtraction)
        5. Clip and add to extrinsic rewards: r_total = r_ext + beta * r_int

        Args:
            rewards: Extrinsic rewards from env, shape (total_steps,). Modified in-place.
            obs: All observations, shape (total_steps, obs_dim).
            next_obs: All next observations, shape (total_steps, obs_dim).
            actions: All actions, shape (total_steps,) or (total_steps, act_dim).
            dones: Terminal signals, shape (total_steps,). Used for per-episode
                episodic reset in NovelD/NGU. None if not available.

        Returns:
            The augmented rewards tensor (same object, modified in-place).
        """
        # Step 1-2: Normalize observations
        self._update_obs_stats(next_obs)
        norm_obs = self.normalize_obs(obs)
        norm_next_obs = self.normalize_obs(next_obs)

        # Step 3: Compute intrinsic rewards (batched, one call)
        intrinsic = self.compute_rewards(norm_obs, norm_next_obs, actions, dones)

        # Step 4: Reward normalization — divide by running std only (RND paper)
        # No mean subtraction: this keeps the exploration signal positive
        # while stabilizing the scale as prediction errors change over training
        self._update_reward_stats(intrinsic)
        intrinsic = intrinsic / (self._reward_running_var.sqrt() + 1e-8)

        # Step 5: Clip and add to extrinsic rewards
        intrinsic.clamp_(0.0, self.reward_clip)
        rewards.add_(self.beta * intrinsic)

        # Decay beta
        self._step += 1
        self.beta = self.initial_beta * (self.beta_decay ** self._step)

        return rewards

    def get_metrics(self) -> dict:
        """Return metrics for logging."""
        return {
            "explore/beta": self.beta,
            "explore/reward_var": self._reward_running_var.item(),
            "explore/obs_var_mean": self._obs_var.mean().item(),
            "explore/step": self._step,
        }
