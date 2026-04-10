"""ICM — Intrinsic Curiosity Module, batched implementation.

Three forward passes per compute_rewards call:
1. Encoder: obs → features, next_obs → next_features (batched as one pass)
2. Forward model: (features, action) → predicted_next_features
3. Inverse model: (features, next_features) → predicted_action (training only)

Intrinsic reward = forward prediction error in feature space.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.optim as optim

from puffer_explore.methods.base import BaseExploration
from puffer_explore.networks import DynamicsEncoder, ForwardDynamics, InverseDynamics, compile_network


class ICM(BaseExploration):
    """Intrinsic Curiosity Module — batched.

    Args:
        obs_dim: Observation dimension.
        n_actions: Number of discrete actions.
        n_envs: Parallel environments.
        rollout_steps: Steps per rollout.
        embed_dim: Feature embedding dimension.
        hidden_dim: Hidden width.
        lr: Learning rate for all ICM components.
        eta: Intrinsic reward scaling factor.
        icm_beta: Weight for forward vs inverse loss (0.2 = 80% inverse).
        beta: Coefficient for adding to extrinsic reward.
        use_compile: Whether to torch.compile.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 18,
        n_envs: int = 1024,
        rollout_steps: int = 128,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        eta: float = 1.0,
        icm_beta: float = 0.2,
        beta: float = 0.01,
        reward_clip: float = 5.0,
        beta_decay: float = 1.0,
        use_compile: bool = True,
        device: str = "cuda",
    ):
        super().__init__(obs_dim, n_envs, rollout_steps, device, beta, reward_clip, beta_decay)
        self.n_actions = n_actions
        self.eta = eta
        self.icm_beta = icm_beta

        self.encoder = DynamicsEncoder(obs_dim, embed_dim, hidden_dim).to(device)
        self.forward_model = ForwardDynamics(embed_dim, n_actions, hidden_dim).to(device)
        self.inverse_model = InverseDynamics(embed_dim, n_actions, hidden_dim).to(device)

        all_params = (
            list(self.encoder.parameters())
            + list(self.forward_model.parameters())
            + list(self.inverse_model.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=lr)

        if use_compile:
            example_obs = torch.randn(min(n_envs * rollout_steps * 2, 16384), obs_dim, device=device)
            self.encoder_compiled = compile_network(self.encoder, example_obs)
        else:
            self.encoder_compiled = self.encoder

        self._last_fwd_loss = 0.0
        self._last_inv_loss = 0.0

    @torch.no_grad()
    def compute_rewards(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Batched ICM: encode both, forward predict, compute error."""
        # Encode obs and next_obs in one batched pass
        combined = torch.cat([obs, next_obs], dim=0)
        features_all = self.encoder_compiled(combined)
        n = obs.shape[0]
        phi_s = features_all[:n]
        phi_s_next = features_all[n:]

        # Forward model prediction
        action_onehot = F.one_hot(actions.long(), self.n_actions).float()
        phi_s_next_pred = self.forward_model(phi_s, action_onehot)

        # Intrinsic reward = prediction error
        self._intrinsic_rewards = self.eta * (phi_s_next - phi_s_next_pred).pow(2).mean(dim=-1)
        return self._intrinsic_rewards

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor) -> dict:
        """Train encoder, forward model, and inverse model."""
        combined = torch.cat([obs, next_obs], dim=0)
        features_all = self.encoder(combined)
        n = obs.shape[0]
        phi_s = features_all[:n]
        phi_s_next = features_all[n:]

        # Forward loss
        action_onehot = F.one_hot(actions.long(), self.n_actions).float()
        phi_s_next_pred = self.forward_model(phi_s, action_onehot)
        fwd_loss = (phi_s_next.detach() - phi_s_next_pred).pow(2).mean()

        # Inverse loss
        action_logits = self.inverse_model(phi_s, phi_s_next)
        inv_loss = F.cross_entropy(action_logits, actions.long())

        # Combined loss
        loss = self.icm_beta * fwd_loss + (1 - self.icm_beta) * inv_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if self.encoder_compiled is not self.encoder:
            self.encoder_compiled.load_state_dict(self.encoder.state_dict())

        self._last_fwd_loss = fwd_loss.item()
        self._last_inv_loss = inv_loss.item()
        return {"explore/icm_fwd_loss": self._last_fwd_loss, "explore/icm_inv_loss": self._last_inv_loss}
