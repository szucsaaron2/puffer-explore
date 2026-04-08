"""NovelD — batched novelty difference with ERIR.

Computes novelty(s') - alpha * novelty(s) in ONE forward pass by
concatenating obs and next_obs, running through RND together, then splitting.
This avoids two separate forward passes.

ERIR (Episodic Restriction) uses the same hash-table approach as count-based
but only tracks within-epoch visits, resetting each rollout.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from puffer_explore.methods.base import BaseExploration
from puffer_explore.networks import TinyMLP, compile_network


class NovelD(BaseExploration):
    """NovelD exploration with batched computation.

    Args:
        obs_dim: Observation dimension.
        n_envs: Number of parallel environments.
        rollout_steps: Steps per rollout.
        output_dim: RND embedding dimension.
        hidden_dim: Hidden layer width.
        lr: Predictor learning rate.
        alpha: Scaling for current-state novelty subtraction.
        use_erir: Whether to use Episodic Restriction on Intrinsic Reward.
        erir_buckets: Hash table size for ERIR tracking.
        beta: Intrinsic reward coefficient.
        reward_clip: Max intrinsic reward.
        use_compile: Whether to torch.compile networks.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        n_envs: int = 1024,
        rollout_steps: int = 128,
        output_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        alpha: float = 0.5,
        use_erir: bool = True,
        erir_buckets: int = 65536,
        beta: float = 0.01,
        reward_clip: float = 5.0,
        beta_decay: float = 1.0,
        use_compile: bool = True,
        device: str = "cuda",
    ):
        super().__init__(obs_dim, n_envs, rollout_steps, device, beta, reward_clip, beta_decay)
        self.alpha = alpha
        self.use_erir = use_erir

        # RND networks
        self.target = TinyMLP(obs_dim, output_dim, hidden_dim).to(device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = TinyMLP(obs_dim, output_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

        # Compile
        if use_compile:
            example = torch.randn(min(n_envs * rollout_steps * 2, 16384), obs_dim, device=device)
            self.target = compile_network(self.target, example)
            self.predictor_compiled = compile_network(
                TinyMLP(obs_dim, output_dim, hidden_dim).to(device), example
            )
            self.predictor_compiled.load_state_dict(self.predictor.state_dict())
        else:
            self.predictor_compiled = self.predictor

        # ERIR hash table (reset each rollout)
        if use_erir:
            self._erir_visited = torch.zeros(erir_buckets, dtype=torch.bool, device=device)
            self._erir_n_buckets = erir_buckets

        self._last_loss = 0.0

    @torch.no_grad()
    def _novelty(self, obs: torch.Tensor) -> torch.Tensor:
        """RND prediction error."""
        target_out = self.target(obs)
        pred_out = self.predictor_compiled(obs)
        return (target_out - pred_out).pow(2).mean(dim=-1)

    @torch.no_grad()
    def compute_rewards(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Batched NovelD: cat obs+next_obs, one forward pass, split, subtract."""
        # Stack for single forward pass
        combined = torch.cat([obs, next_obs], dim=0)  # (2N, D)

        target_out = self.target(combined)
        pred_out = self.predictor_compiled(combined)
        novelty_all = (target_out - pred_out).pow(2).mean(dim=-1)  # (2N,)

        n = obs.shape[0]
        novelty_s = novelty_all[:n]
        novelty_s_next = novelty_all[n:]

        # Novelty difference
        self._intrinsic_rewards = torch.clamp(
            novelty_s_next - self.alpha * novelty_s, min=0.0
        )

        # ERIR: zero out already-visited states (including duplicates within this batch)
        if self.use_erir:
            hashes = (next_obs * 97.0).to(torch.int32).sum(dim=-1).abs() % self._erir_n_buckets
            hashes_long = hashes.long()

            # Check which were already visited from previous batches
            already_visited = self._erir_visited[hashes_long]

            # Within this batch: find first occurrence of each unique hash.
            # unique(sorted=True) returns sorted unique values; the first occurrence
            # in the original tensor for each unique value can be found via inverse indices.
            _, inverse = torch.unique(hashes_long, return_inverse=True)
            n_unique = inverse.max().item() + 1

            # For each unique group, find the minimum index (= first occurrence)
            arange = torch.arange(n, device=self.device)
            first_per_group = torch.full((n_unique,), n, dtype=torch.long, device=self.device)
            first_per_group.scatter_reduce_(0, inverse, arange, reduce="amin", include_self=False)

            # An element is "first in batch" if its index equals the min index for its group
            is_first_in_batch = (first_per_group[inverse] == arange)

            # Zero out: already visited OR duplicate within batch
            mask = (~already_visited) & is_first_in_batch
            self._intrinsic_rewards *= mask.float()

            # Mark all as visited
            self._erir_visited[hashes_long] = True

        return self._intrinsic_rewards

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor) -> dict:
        """Train RND predictor."""
        with torch.no_grad():
            target_out = self.target(next_obs)
        pred_out = self.predictor(next_obs)
        loss = (target_out - pred_out).pow(2).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if self.predictor_compiled is not self.predictor:
            self.predictor_compiled.load_state_dict(self.predictor.state_dict())

        # Reset ERIR for next rollout
        if self.use_erir:
            self._erir_visited.zero_()

        self._last_loss = loss.item()
        return {"explore/noveld_loss": self._last_loss}
