"""RIDE — Rewarding Impact-Driven Exploration (Raileanu et al., 2020).

Intrinsic reward = ||phi(s') - phi(s)|| / sqrt(count(s'))

Combines state-change magnitude (how much did the world change?) with
episodic visit counting (have we been in this new state before?).

Uses the same dynamics encoder as ICM but with a different reward formula.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from puffer_explore.methods.base import BaseExploration
from puffer_explore.networks import DynamicsEncoder, ForwardDynamics, InverseDynamics, compile_network


class RIDE(BaseExploration):
    """RIDE — impact * inverse-count.

    Args:
        obs_dim: Observation dimension.
        n_actions: Number of discrete actions.
        n_envs: Parallel environments.
        rollout_steps: Steps per rollout.
        embed_dim: Feature embedding dimension.
        hidden_dim: Hidden width.
        lr: Learning rate.
        epi_buckets: Hash table size for counting.
        beta: Intrinsic reward coefficient.
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
        epi_buckets: int = 65536,
        beta: float = 0.01,
        reward_clip: float = 5.0,
        beta_decay: float = 1.0,
        use_compile: bool = True,
        device: str = "cuda",
    ):
        super().__init__(obs_dim, n_envs, rollout_steps, device, beta, reward_clip, beta_decay)
        self.n_actions = n_actions
        self.epi_buckets = epi_buckets

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
            example = torch.randn(min(n_envs * rollout_steps * 2, 16384), obs_dim, device=device)
            self.encoder_compiled = compile_network(self.encoder, example)
        else:
            self.encoder_compiled = self.encoder

        # Episodic counting
        self._epi_counts = torch.zeros(epi_buckets, dtype=torch.int32, device=device)
        self._last_fwd_loss = 0.0
        self._last_inv_loss = 0.0

    @torch.no_grad()
    def compute_rewards(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """RIDE: ||phi(s') - phi(s)|| / sqrt(count(s'))."""
        # Batch encode obs and next_obs in one pass
        combined = torch.cat([obs, next_obs], dim=0)
        features_all = self.encoder_compiled(combined)
        n = obs.shape[0]
        phi_s = features_all[:n]
        phi_s_next = features_all[n:]

        # Impact: L2 distance in feature space
        impact = (phi_s_next - phi_s).pow(2).sum(dim=-1).sqrt()

        # Episodic count on next_obs
        hashes = (next_obs * 97.0).to(torch.int32).sum(dim=-1).abs() % self.epi_buckets
        ones = torch.ones_like(hashes, dtype=torch.int32)
        self._epi_counts.scatter_add_(0, hashes.long(), ones)
        counts = self._epi_counts[hashes.long()].float().clamp(min=1.0)

        # RIDE reward
        self._intrinsic_rewards = impact / counts.sqrt()
        return self._intrinsic_rewards

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor) -> dict:
        """Train encoder + forward/inverse dynamics. Reset episodic counts."""
        import torch.nn.functional as F

        combined = torch.cat([obs, next_obs], dim=0)
        features_all = self.encoder(combined)
        n = obs.shape[0]
        phi_s = features_all[:n]
        phi_s_next = features_all[n:]

        action_onehot = F.one_hot(actions.long(), self.n_actions).float()
        phi_pred = self.forward_model(phi_s, action_onehot)
        fwd_loss = (phi_s_next.detach() - phi_pred).pow(2).mean()

        action_logits = self.inverse_model(phi_s, phi_s_next)
        inv_loss = F.cross_entropy(action_logits, actions.long())

        loss = fwd_loss + inv_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if self.encoder_compiled is not self.encoder:
            self.encoder_compiled.load_state_dict(self.encoder.state_dict())

        self._epi_counts.zero_()
        self._last_fwd_loss = fwd_loss.item()
        self._last_inv_loss = inv_loss.item()
        return {"explore/ride_fwd_loss": self._last_fwd_loss, "explore/ride_inv_loss": self._last_inv_loss}
