# PufferExplore

**High-performance intrinsic exploration methods for [PufferLib](https://github.com/PufferAI/PufferLib) — designed to not be the bottleneck.**

[![Tests](https://img.shields.io/badge/tests-37%2F37-brightgreen.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

PufferLib trains PPO at 1-20M steps/sec. Adding exploration shouldn't slow that down. PufferExplore implements intrinsic motivation methods that add **<5% overhead** by following PufferLib's performance principles: batched post-rollout computation, pre-allocated buffers, torch.compile'd networks, zero per-step Python overhead.

## Methods

| Method | Type | Overhead Target | Parameters | Paper |
|--------|------|----------------|------------|-------|
| **RND** | Prediction error | <5% | ~33K | Burda et al., 2018 |
| **NovelD** | Novelty difference + ERIR | <5% | ~33K | Zhang et al., 2021 |
| **ICM** | Curiosity-driven | <8% | ~66K | Pathak et al., 2017 |
| **NGU** | Episodic + lifelong | <10% | ~66K | Badia et al., 2020 |
| **RIDE** | Impact-driven | <10% | ~66K | Raileanu et al., 2020 |
| **Count-Based** | Visit counting | <1% | 0 | Bellemare et al., 2016 |
| **Ensemble** | Model disagreement | <12% | ~165K | Pathak et al., 2019 |
| **Go-Explore** | Archive-based | N/A | 0 | Ecoffet et al., 2021 |

## Performance Design

```
PufferLib loop:          evaluate()  ──────────────────────────>  train()
                              │                                      │
PufferExplore:                │                                      │
                              ▼                                      ▼
                     Collect rollout               ┌─ PPO minibatch update
                     (unchanged,                   │  (unchanged)
                      full speed)                  ├─ Exploration network update
                              │                    │  (one backward pass, ~33K params)
                              ▼                    └──────────────────────────
                     ┌─ Batch-compute intrinsic
                     │  rewards for ALL steps
                     │  (one forward pass, torch.compile'd)
                     ├─ Augment rewards in-place
                     │  (no allocation)
                     └─ Recompute advantages
```

**The rule: nothing runs per-step during rollout collection.** All exploration computation is one batched pass after the full rollout is collected.

## Quick Start

```python
from puffer_explore.integration import ExploreTrainer

# Wrap your existing PufferLib trainer
trainer = ExploreTrainer(
    pufferl_trainer,
    method="rnd",
    obs_dim=128,
    beta=0.01,
)

for epoch in range(num_epochs):
    trainer.evaluate()    # PufferLib rollout (unchanged, full speed)
    trainer.explore()     # Batch-compute intrinsic rewards, augment buffer
    logs = trainer.train()  # PPO update + exploration network update
```

Or standalone without PufferLib:

```python
from puffer_explore.integration import create_exploration

rnd = create_exploration("rnd", obs_dim=128, n_envs=1024, rollout_steps=128, device="cuda")

# After collecting a rollout:
augmented_rewards = rnd.augment_rewards(rewards, obs, next_obs, actions)
metrics = rnd.update(obs_batch, next_obs_batch, actions_batch)
```

## Benchmarking Overhead

```bash
python -m puffer_explore.benchmark                          # All methods
python -m puffer_explore.benchmark --method rnd --device cuda  # Specific method on GPU
```

## Architecture

```
puffer_explore/
├── methods/           # 8 exploration methods
│   ├── base.py        #   BaseExploration (pre-allocated buffers, running normalization)
│   ├── rnd.py         #   RND (torch.compile'd target + predictor)
│   ├── noveld.py      #   NovelD (batched obs+next_obs concat, ERIR via scatter_reduce)
│   ├── icm.py         #   ICM (batched encoder + forward/inverse dynamics)
│   ├── ngu.py         #   NGU (episodic + lifelong novelty)
│   ├── ride.py        #   RIDE (impact-driven exploration)
│   ├── count_based.py #   Count-based (pure tensor ops, zero NN overhead)
│   ├── ensemble.py    #   Ensemble disagreement (5 stacked models)
│   └── go_explore.py  #   Go-Explore (archive-based, Phase 1)
├── networks.py        # TinyMLP (~33K params), DynamicsEncoder, torch.compile wrapper
├── integration.py     # PufferLib hook (ExploreTrainer) + StandaloneExploreTrainer
├── benchmark.py       # Throughput measurement tool
└── kernels/           # (Planned) CUDA C kernels for count-based + reward fusion
```

## Why This Exists

PufferLib has entropy-based exploration built in. What it doesn't have is *intrinsic motivation* — the family of methods (RND, ICM, NovelD, etc.) that reward the agent for visiting novel states. These are essential for hard exploration problems like Montezuma's Revenge, NetHack, and sparse-reward mazes where random exploration fails.

This project adds intrinsic motivation to PufferLib without sacrificing its speed.

## License

MIT
