"""Shared encoder — reuse PufferLib's policy features for exploration.

PufferLib's policy has an encode() function that maps raw observations to
a hidden feature vector. Instead of running a separate encoder in the
exploration network, we can reuse these features. This:

1. Eliminates a redundant forward pass (the biggest cost for ICM/RIDE)
2. Means the exploration method operates in feature space, which is
   often more semantically meaningful than raw observation space
3. Reduces the exploration network to just a tiny predictor head

Usage:
    # During rollout, PufferLib already computes policy features
    features = policy.encode(obs)  # shape (batch, hidden_dim)

    # Instead of: rnd.compute_rewards(obs, next_obs, actions)
    # We do:     rnd.compute_rewards(features, next_features, actions)
"""

from __future__ import annotations

import torch

from puffer_explore.methods.base import BaseExploration


class SharedEncoderExploration(BaseExploration):
    """Wraps any exploration method to use shared policy features.

    Instead of receiving raw observations, receives the output of
    the policy's encode() function. The inner exploration method
    is initialized with feature_dim instead of obs_dim.

    Args:
        inner: The wrapped exploration method.
        feature_dim: Dimension of the policy's feature output.
    """

    def __init__(
        self,
        inner: BaseExploration,
        feature_dim: int,
    ):
        # Don't call super().__init__ — we delegate everything to inner
        self._inner = inner
        self.feature_dim = feature_dim
        # Expose inner's attributes
        self.beta = inner.beta
        self.device = inner.device

    def compute_rewards(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """obs and next_obs are FEATURES from the policy encoder, not raw obs."""
        return self._inner.compute_rewards(obs, next_obs, actions)

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor) -> dict:
        return self._inner.update(obs, next_obs, actions)

    def augment_rewards(
        self,
        rewards: torch.Tensor,
        features: torch.Tensor,
        next_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self._inner.augment_rewards(rewards, features, next_features, actions)

    def get_metrics(self) -> dict:
        return self._inner.get_metrics()


def create_shared_encoder_exploration(
    method: str,
    feature_dim: int,
    n_envs: int = 1024,
    rollout_steps: int = 128,
    device: str = "cuda",
    **kwargs,
) -> SharedEncoderExploration:
    """Create an exploration method that uses shared policy features.

    Args:
        method: Method name (rnd, noveld, count_based, etc.).
        feature_dim: Dimension of policy encoder output.
        n_envs: Parallel environments.
        rollout_steps: Steps per rollout.
        device: Torch device.
        **kwargs: Method-specific arguments.

    Returns:
        SharedEncoderExploration wrapping the specified method.
    """
    from puffer_explore.integration import create_exploration

    # Create the inner method with feature_dim instead of obs_dim
    inner = create_exploration(
        method=method,
        obs_dim=feature_dim,  # Key: use feature dim, not obs dim
        n_envs=n_envs,
        rollout_steps=rollout_steps,
        device=device,
        **kwargs,
    )

    return SharedEncoderExploration(inner, feature_dim)
