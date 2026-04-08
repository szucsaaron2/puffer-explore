"""Config system for exploration methods.

Provides default configurations and integrates with PufferLib's .ini config format.
Supports beta scheduling variants (constant, linear, cosine, adaptive).
"""

from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass
class ExploreConfig:
    """Configuration for an exploration method."""
    method: str = "rnd"
    beta: float = 0.01
    beta_schedule: str = "constant"  # constant, linear, cosine, adaptive
    beta_decay: float = 1.0  # For exponential decay
    beta_min: float = 0.0  # Floor for scheduled beta
    reward_clip: float = 5.0
    hidden_dim: int = 128
    output_dim: int = 64
    lr: float = 1e-3
    use_compile: bool = True
    # Method-specific
    n_actions: int = 18
    # NovelD
    noveld_alpha: float = 0.5
    use_erir: bool = True
    # NGU
    ngu_max_scale: float = 5.0
    # Ensemble
    n_models: int = 5
    # Go-Explore
    go_explore_steps: int = 100
    sticky_action_prob: float = 0.95


# Pre-built configs for common environments
PRESET_CONFIGS = {
    "breakout_rnd": ExploreConfig(
        method="rnd", beta=0.005, beta_schedule="cosine",
        hidden_dim=128, lr=1e-3,
    ),
    "breakout_count": ExploreConfig(
        method="count_based", beta=0.01,
    ),
    "nethack_rnd": ExploreConfig(
        method="rnd", beta=0.01, beta_schedule="linear",
        hidden_dim=256, lr=5e-4, reward_clip=10.0,
    ),
    "nethack_ngu": ExploreConfig(
        method="ngu", beta=0.01, beta_schedule="cosine",
        hidden_dim=256, ngu_max_scale=10.0,
    ),
    "montezuma_rnd": ExploreConfig(
        method="rnd", beta=0.01, beta_schedule="constant",
        hidden_dim=256, lr=1e-4, reward_clip=5.0,
    ),
    "montezuma_count": ExploreConfig(
        method="count_based", beta=0.1,
    ),
}


class BetaScheduler:
    """Schedules the intrinsic reward coefficient beta over training.

    Supported schedules:
    - constant: beta stays fixed
    - linear: beta decreases linearly to beta_min over total_steps
    - cosine: beta follows cosine annealing to beta_min
    - exponential: beta *= beta_decay each step
    - adaptive: beta adjusts based on the ratio of intrinsic to extrinsic reward
    """

    def __init__(
        self,
        initial_beta: float = 0.01,
        schedule: str = "constant",
        beta_min: float = 0.0,
        beta_decay: float = 0.9999,
        total_steps: int = 10_000_000,
    ):
        self.initial_beta = initial_beta
        self.schedule = schedule
        self.beta_min = beta_min
        self.beta_decay = beta_decay
        self.total_steps = total_steps
        self._step = 0
        self._adaptive_ratio = 1.0

    def step(self, intrinsic_mean: float = 0.0, extrinsic_mean: float = 0.0) -> float:
        """Advance one step and return the current beta value."""
        self._step += 1

        if self.schedule == "constant":
            return self.initial_beta

        elif self.schedule == "linear":
            progress = min(self._step / max(self.total_steps, 1), 1.0)
            return self.initial_beta * (1 - progress) + self.beta_min * progress

        elif self.schedule == "cosine":
            progress = min(self._step / max(self.total_steps, 1), 1.0)
            cos_val = 0.5 * (1 + math.cos(math.pi * progress))
            return self.beta_min + (self.initial_beta - self.beta_min) * cos_val

        elif self.schedule == "exponential":
            return max(
                self.initial_beta * (self.beta_decay ** self._step),
                self.beta_min,
            )

        elif self.schedule == "adaptive":
            # Adjust beta to maintain a target ratio of intrinsic/extrinsic reward
            target_ratio = 0.1  # Intrinsic should be ~10% of extrinsic
            if extrinsic_mean > 0 and intrinsic_mean > 0:
                current_ratio = intrinsic_mean / extrinsic_mean
                # If intrinsic is too large, reduce beta; if too small, increase beta
                adjustment = target_ratio / max(current_ratio, 1e-8)
                adjustment = max(0.5, min(2.0, adjustment))  # Clip adjustment
                self._adaptive_ratio *= adjustment
            return max(self.initial_beta * self._adaptive_ratio, self.beta_min)

        return self.initial_beta

    @property
    def current_beta(self) -> float:
        """Get current beta without advancing step."""
        return self.step.__wrapped__() if hasattr(self.step, '__wrapped__') else self.initial_beta


def load_explore_config(name: str) -> ExploreConfig:
    """Load a preset exploration config by name."""
    if name in PRESET_CONFIGS:
        return PRESET_CONFIGS[name]
    raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESET_CONFIGS.keys())}")


def write_ini_section(config: ExploreConfig) -> str:
    """Generate a PufferLib-compatible .ini section for exploration."""
    lines = ["[explore]"]
    for k, v in config.__dict__.items():
        lines.append(f"{k} = {v}")
    return "\n".join(lines)
