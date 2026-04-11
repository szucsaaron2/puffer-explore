"""NGU — Never Give Up (Badia et al., 2020). Batched implementation.

Combines two novelty signals:
- Episodic: hash-based counting within the current rollout (resets each epoch)
- Lifelong: RND prediction error (persists across rollouts)

Combined: r_i = episodic(s') * clamp(lifelong(s'), min=1, max=L)

Both are computed in one batched pass after rollout collection.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from puffer_explore.methods.base import BaseExploration
from puffer_explore.networks import TinyMLP, compile_network


class NGU(BaseExploration):
    """Never Give Up — episodic + lifelong novelty.

    Args:
        obs_dim: Observation dimension.
        n_envs: Parallel environments.
        rollout_steps: Steps per rollout.
        output_dim: RND embedding dimension.
        hidden_dim: Hidden width.
        lr: RND predictor learning rate.
        max_reward_scale: Lifelong clamp max L.
        epi_buckets: Hash table size for episodic counting.
        beta: Intrinsic reward coefficient.
        reward_clip: Max intrinsic reward.
        use_compile: Whether to torch.compile.
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
        max_reward_scale: float = 5.0,
        epi_buckets: int = 65536,
        beta: float = 0.001,
        reward_clip: float = 5.0,
        beta_decay: float = 1.0,
        use_compile: bool = True,
        device: str = "cuda",
    ):
        super().__init__(obs_dim, n_envs, rollout_steps, device, beta, reward_clip, beta_decay)
        self.max_reward_scale = max_reward_scale
        self.epi_buckets = epi_buckets

        # Lifelong: RND
        self.target = TinyMLP(obs_dim, output_dim, hidden_dim).to(device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = TinyMLP(obs_dim, output_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

        if use_compile:
            example = torch.randn(min(n_envs * rollout_steps, 8192), obs_dim, device=device)
            self.target = compile_network(self.target, example)
            self.predictor_compiled = compile_network(
                TinyMLP(obs_dim, output_dim, hidden_dim).to(device), example
            )
            self.predictor_compiled.load_state_dict(self.predictor.state_dict())
        else:
            self.predictor_compiled = self.predictor

        # Episodic: hash-based counts (reset each rollout)
        self._epi_counts = torch.zeros(epi_buckets, dtype=torch.int32, device=device)

        # Running stats for lifelong normalization
        self._life_mean = torch.zeros(1, device=device)
        self._life_var = torch.ones(1, device=device)
        self._life_count = torch.zeros(1, device=device, dtype=torch.long)

        self._last_loss = 0.0

    @torch.no_grad()
    def compute_rewards(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """NGU reward: episodic(s') * clamp(lifelong(s'), 1, L)."""
        n = next_obs.shape[0]

        # Reset episodic counts
        self._epi_counts.zero_()

        # --- Lifelong: RND prediction error ---
        target_out = self.target(next_obs)
        pred_out = self.predictor_compiled(next_obs)
        lifelong = (target_out - pred_out).pow(2).mean(dim=-1)

        # Normalize lifelong using per-batch stats (not accumulating)
        batch_mean = lifelong.mean()
        batch_std = lifelong.std().clamp(min=1e-8)

        alpha = (lifelong - batch_mean) / batch_std
        modulated = alpha.clamp(min=1.0, max=self.max_reward_scale)

        # --- Episodic: hash-based counting (per-episode scoped) ---
        obs_hash = (next_obs * 97.0).to(torch.int32).sum(dim=-1).abs()

        # Per-episode scoping via composite hash
        if dones is not None and self.n_envs > 0:
            rollout_len = n // self.n_envs
            dones_2d = dones.reshape(rollout_len, self.n_envs)
            episode_ids = dones_2d.cumsum(dim=0).reshape(-1).long()
            env_ids = torch.arange(
                self.n_envs, device=self.device
            ).repeat(rollout_len)
            hashes = (
                obs_hash * 1000003 + episode_ids * 997 + env_ids * 31
            ) % self.epi_buckets
        else:
            hashes = obs_hash % self.epi_buckets

        ones = torch.ones_like(hashes, dtype=torch.int32)
        self._epi_counts.scatter_add_(0, hashes.long(), ones)
        counts = self._epi_counts[hashes.long()].float().clamp(min=1.0)
        episodic = 1.0 / counts.sqrt()

        # --- Combined ---
        self._intrinsic_rewards = episodic * modulated
        return self._intrinsic_rewards

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor) -> dict:
        """Train lifelong RND predictor + reset episodic counts."""
        with torch.no_grad():
            target_out = self.target(next_obs)
        pred_out = self.predictor(next_obs)
        loss = (target_out - pred_out).pow(2).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if self.predictor_compiled is not self.predictor:
            self.predictor_compiled.load_state_dict(self.predictor.state_dict())

        self._last_loss = loss.item()
        return {"explore/ngu_rnd_loss": self._last_loss}
