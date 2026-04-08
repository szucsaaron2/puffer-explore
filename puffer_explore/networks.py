"""Tiny neural networks for exploration — designed for torch.compile.

These are deliberately small (2-layer, 128-wide MLPs with ~33K params)
so they add negligible overhead to PufferLib's training loop.

All networks are designed to be:
1. torch.compile'd into fused kernels
2. Run in a single batched forward pass on the full rollout buffer
3. Never called per-step during rollout collection

The policy network in PufferLib is 150K-2M params. These are 33K.
The policy forward pass takes ~1ms on 128K obs. These take ~0.1ms.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """Minimal MLP for RND target/predictor. ~33K params at default settings.

    Architecture: Linear → ReLU → Linear → ReLU → Linear
    No normalization layers (unnecessary for this size, and they
    interfere with torch.compile static shapes).
    """
    __constants__ = ['input_dim', 'hidden_dim', 'output_dim']

    def __init__(self, input_dim: int, output_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2**0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsEncoder(nn.Module):
    """Shared state encoder for ICM/RIDE forward+inverse dynamics.

    Encodes raw observations into a compact feature space.
    The forward and inverse models operate on these features.
    """
    __constants__ = ['input_dim', 'embed_dim']

    def __init__(self, input_dim: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ForwardDynamics(nn.Module):
    """Predicts next state features from (state_features, action_onehot)."""
    __constants__ = ['embed_dim', 'n_actions']

    def __init__(self, embed_dim: int = 64, n_actions: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(embed_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, features: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([features, action_onehot], dim=-1))


class InverseDynamics(nn.Module):
    """Predicts action from (state_features, next_state_features)."""
    __constants__ = ['embed_dim', 'n_actions']

    def __init__(self, embed_dim: int = 64, n_actions: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, features: torch.Tensor, next_features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([features, next_features], dim=-1))


def compile_network(net: nn.Module, example_input: torch.Tensor) -> nn.Module:
    """Compile a network with torch.compile for fused execution.

    Falls back gracefully if torch.compile isn't available or fails.
    """
    try:
        compiled = torch.compile(net, mode="reduce-overhead", fullgraph=True)
        # Warm up
        with torch.no_grad():
            compiled(example_input)
        return compiled
    except Exception:
        return net
