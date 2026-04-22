# Plan: Contributing puffer-explore to PufferLib

## Summary

Goal: merge our intrinsic exploration methods (RND, NovelD, NGU, Count-Based, ICM) into the upstream PufferLib repo as a first-party module.

## Why this fits PufferLib

PufferLib currently has entropy-based exploration built in but no intrinsic motivation. Our methods:
- Add <5% wall-clock overhead (matches PufferLib's performance philosophy)
- Batched post-rollout computation (no per-step Python)
- Torch.compile'd networks
- **Real result**: NGU goes from 0% → 100% solve rate on KeyCorridorS6R3 where baseline PPO fails completely

## Contribution Strategy (Discord-first, then focused PR)

### Step 1: Discord intro (before coding)
Join https://discord.gg/puffer and post in `#contributions` (or DM `@jsuarez5341`):

```
Hi! I've been working on a puffer-explore module that adds intrinsic motivation
methods (RND, NovelD, NGU, Count-Based, ICM) to PufferLib training. All methods
hook into the evaluate() → train() loop post-rollout, keeping the hot path
unchanged. Benchmarks on RTX 3090 / PufferLib 3.0:

- Throughput: <5% SPS overhead across all methods
- MiniGrid KeyCorridorS6R3 (hard exploration):
  • PPO baseline: 0% solve
  • NGU: 100% solve on both seeds, 0.509 reward
- MiniGrid KeyCorridorS4R3 (medium):
  • PPO baseline: 3% solve
  • RND, NovelD, NGU: all 100% solve

Repo: https://github.com/szucsaaron2/puffer-explore

Would a PR against branch 4.0 be welcome? I'd start with RND only (smallest,
most impactful), then follow up with the others if the first one lands.
```

### Step 2: First PR — RND only
Keep the first PR tight. Maintainers prefer focused, benchmarkable PRs:

- Target branch: `4.0` (PufferLib's default)
- New module: `pufferlib/intrinsic.py` with `RND` class + `wrap_trainer()` helper
- One config in `config/` for a demo hard-exploration env (Craftax recommended — recently merged and needs strong exploration)
- PR body must include:
  - SPS overhead table (baseline vs RND)
  - Score delta on the chosen env
  - Architecture diagram (1 ASCII box)
  - Single-line usage example

### Step 3: Follow-up PRs
Once RND lands, propose NovelD → NGU → Count-Based → ICM in separate PRs.

## Expected PR structure

```
Title: "intrinsic exploration: RND"

## Summary
RND intrinsic rewards hook into the PufferL train loop without
changing rollout collection. ~33K extra params, <5% SPS overhead.

## Overhead (RTX 3090, Craftax, 16 envs)
| Method | SPS | vs baseline |
|--------|-----|-------------|
| baseline | 12.3k | - |
| + RND | 11.9k | -3.2% |

## Score on Craftax (5M steps, 2 seeds)
| Method | Score |
|--------|-------|
| baseline | XXX |
| + RND | XXX |

## Usage
trainer = intrinsic.wrap(trainer, method='rnd', beta=0.01)

## Reproducibility
python -m pufferlib.intrinsic.benchmark --env craftax --method rnd

## Files changed
- pufferlib/intrinsic.py (new, ~300 LOC)
- config/craftax_rnd.ini (new)
- tests/test_intrinsic.py (new, ~100 LOC)
```

## Key technical changes for upstream

Our current puffer-explore relies on PufferLib 3.0's `PuffeRL` class attributes
(`self.observations`, `self.actions`, `self.rewards`, `self.terminals`).
For the PR, we should:

1. Port the integration to PufferLib 4.0 (default branch) — the API is slightly different
2. Use the same buffer names (already work in 4.0)
3. Test on Craftax or another PufferLib ocean env (where we can show a clear score improvement)
4. Add the module inside `pufferlib/` (not as a separate package) so it's a first-party feature

## Files to adapt for upstream

From our `puffer-explore/` repo:
- `puffer_explore/methods/base.py` → `pufferlib/intrinsic/base.py`
- `puffer_explore/methods/rnd.py` → `pufferlib/intrinsic/rnd.py` (ship FIRST)
- `puffer_explore/methods/noveld.py` → second PR
- `puffer_explore/methods/ngu.py` → third PR
- `puffer_explore/integration.py` → absorbed into `pufferl.py` as a hook
- `puffer_explore/networks.py` → kept or merged into `pufferlib/models.py`

## Alternative: standalone package

If maintainers prefer a third-party package (less integration risk), we keep
puffer-explore as-is and list it in the PufferLib README as a community
extension. This is lower-commitment but less visible.

## Timeline

- **Week 1**: Discord pitch + wait for response
- **Week 2-3**: Port RND to PufferLib 4.0, benchmark on Craftax
- **Week 4**: Open first PR
- **Week 5+**: Iterate on review, then NovelD/NGU follow-ups

## What's already compelling (can show today)

Our current puffer-explore repo demonstrates:
- End-to-end PufferLib 3.0 integration (working)
- Reference-aligned implementations (obs norm, reward std-norm, per-episode reset)
- Real benchmarks showing PPO=0% vs NGU=100% on KeyCorridorS6R3
- Throughput overhead benchmarks
- 37 passing tests across Windows + Linux

Repo: https://github.com/szucsaaron2/puffer-explore
