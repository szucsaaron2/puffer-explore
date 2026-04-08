"""Count-Based exploration — pure tensor ops, zero neural network overhead.

Uses a fixed-size hash table stored as a GPU tensor. Observation hashing,
count incrementing, and reward computation are all elementwise tensor ops
that fuse naturally with torch.compile or can be replaced with a CUDA kernel.

This is the fastest exploration method: ~0% overhead because it's just
a hash + lookup + division, no neural network forward pass.
"""

from __future__ import annotations

import torch

from puffer_explore.methods.base import BaseExploration


class CountBased(BaseExploration):
    """Hash-based count exploration with GPU-resident hash table.

    Args:
        obs_dim: Observation dimension.
        n_envs: Number of parallel environments.
        rollout_steps: Steps per rollout.
        n_buckets: Hash table size. Larger = fewer collisions. Power of 2 recommended.
        count_beta: Scaling factor (r = beta / sqrt(count)).
        beta: Coefficient when adding to extrinsic reward.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        n_envs: int = 1024,
        rollout_steps: int = 128,
        n_buckets: int = 65536,
        count_beta: float = 1.0,
        beta: float = 0.01,
        reward_clip: float = 5.0,
        beta_decay: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__(obs_dim, n_envs, rollout_steps, device, beta, reward_clip, beta_decay)
        self.n_buckets = n_buckets
        self.count_beta = count_beta

        # GPU-resident hash table — pre-allocated, never reallocated
        self.hash_counts = torch.zeros(n_buckets, dtype=torch.int32, device=device)

        # Pre-allocate hash computation buffers
        self._hash_buf = torch.zeros(rollout_steps * n_envs, dtype=torch.long, device=device)

    @torch.no_grad()
    def _hash_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Hash observations to bucket indices. All tensor ops, no Python loops.

        Simple but effective: discretize → sum → mod n_buckets.
        Writes into pre-allocated buffer.
        """
        # Discretize to int (multiply by 100, cast to int, sum across features)
        hashes = (obs * 97.0).to(torch.int32).sum(dim=-1).abs() % self.n_buckets
        return hashes.long()

    @torch.no_grad()
    def compute_rewards(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Count-based reward: beta / sqrt(count(s')).

        All tensor operations — no Python loops, no per-sample branching.
        """
        buckets = self._hash_obs(next_obs)

        # Increment counts (scatter_add for batched atomic-like increment)
        ones = torch.ones_like(buckets, dtype=torch.int32)
        self.hash_counts.scatter_add_(0, buckets, ones)

        # Look up counts and compute reward
        counts = self.hash_counts[buckets].float().clamp(min=1.0)
        self._intrinsic_rewards = self.count_beta / counts.sqrt()

        return self._intrinsic_rewards

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor) -> dict:
        """No-op — count-based has no trainable parameters."""
        return {
            "explore/unique_buckets": (self.hash_counts > 0).sum().item(),
            "explore/total_counts": self.hash_counts.sum().item(),
        }

    def get_metrics(self) -> dict:
        return {
            **super().get_metrics(),
            "explore/unique_buckets": (self.hash_counts > 0).sum().item(),
        }
