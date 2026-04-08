"""Go-Explore Phase 1 — archive-based return-then-explore.

Go-Explore doesn't use intrinsic rewards or PPO. Instead:
1. Maintain an archive of visited states (cells)
2. Select a promising cell from the archive
3. Return to that cell (using the saved simulator state or trajectory)
4. Explore randomly from there
5. Add any new cells discovered to the archive

This runs as a SEPARATE mode from PufferLib's PPO loop.
Phase 2 (robustification) uses PufferLib's PPO normally to learn a
policy that can reliably reach the best trajectory found in Phase 1.

Performance: Phase 1 is CPU-bound (archive ops, env stepping).
No neural networks. Runs at env-stepping speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Cell:
    """A cell in the Go-Explore archive."""
    key: tuple  # Hash key for this cell
    trajectory: list[int] = field(default_factory=list)  # Actions to reach this cell
    score: float = 0.0  # Best score achieved from this cell
    visits: int = 0  # Number of times selected for exploration
    times_chosen: int = 0
    times_improved: int = 0


class GoExploreArchive:
    """Archive of visited cells with selection heuristics.

    Cells are identified by a hash of the observation (or a domain-specific
    representation like agent position + inventory).
    """

    def __init__(self, max_cells: int = 100_000):
        self.cells: dict[tuple, Cell] = {}
        self.max_cells = max_cells

    def _default_cell_key(self, obs: np.ndarray) -> tuple:
        """Hash observation to a cell key. Override for domain-specific hashing."""
        discretized = (obs * 10).astype(np.int32)
        return tuple(discretized.flatten()[:20])  # Use first 20 dims

    def add_or_update(
        self,
        obs: np.ndarray,
        trajectory: list[int],
        score: float,
        cell_key: tuple | None = None,
    ) -> bool:
        """Add a new cell or update if this trajectory is better.

        Returns True if the archive was modified.
        """
        key = cell_key or self._default_cell_key(obs)

        if key not in self.cells:
            if len(self.cells) >= self.max_cells:
                return False  # Archive full
            self.cells[key] = Cell(
                key=key,
                trajectory=list(trajectory),
                score=score,
                visits=1,
            )
            return True
        else:
            cell = self.cells[key]
            cell.visits += 1
            # Update if shorter trajectory or higher score
            if score > cell.score or (
                score == cell.score and len(trajectory) < len(cell.trajectory)
            ):
                cell.trajectory = list(trajectory)
                cell.score = score
                cell.times_improved += 1
                return True
            return False

    def select_cell(self) -> Cell | None:
        """Select a cell to explore from, weighted by novelty.

        Uses the Go-Explore selection heuristic:
        weight(cell) = 1 / (visits^0.5 * trajectory_length^0.5)

        Favors rarely-visited cells with short trajectories (easy to reach).
        """
        if not self.cells:
            return None

        cells = list(self.cells.values())
        weights = np.array([
            1.0 / (max(c.visits, 1) ** 0.5 * max(len(c.trajectory), 1) ** 0.5)
            for c in cells
        ])
        weights /= weights.sum()

        idx = np.random.choice(len(cells), p=weights)
        cell = cells[idx]
        cell.times_chosen += 1
        return cell

    @property
    def size(self) -> int:
        return len(self.cells)

    def stats(self) -> dict:
        if not self.cells:
            return {"archive_size": 0}
        cells = list(self.cells.values())
        return {
            "archive_size": len(cells),
            "best_score": max(c.score for c in cells),
            "mean_score": np.mean([c.score for c in cells]),
            "max_traj_len": max(len(c.trajectory) for c in cells),
            "mean_visits": np.mean([c.visits for c in cells]),
        }


class GoExplorePhase1:
    """Go-Explore Phase 1: archive-based exploration.

    Runs as a standalone loop, NOT integrated with PPO.

    Args:
        env: Gymnasium-compatible environment.
        archive: GoExploreArchive instance (or creates a new one).
        explore_steps: Random steps to take from each selected cell.
        sticky_action_prob: Probability of repeating the last action.
        cell_key_fn: Custom function to extract cell key from observation.
    """

    def __init__(
        self,
        env: Any,
        archive: GoExploreArchive | None = None,
        explore_steps: int = 100,
        sticky_action_prob: float = 0.95,
        cell_key_fn: Any = None,
    ):
        self.env = env
        self.archive = archive or GoExploreArchive()
        self.explore_steps = explore_steps
        self.sticky_action_prob = sticky_action_prob
        self.cell_key_fn = cell_key_fn
        self.n_actions = (
            env.action_space.n
            if hasattr(env.action_space, 'n')
            else env.action_space.shape[0]
        )

    def _get_cell_key(self, obs: np.ndarray) -> tuple:
        if self.cell_key_fn:
            return self.cell_key_fn(obs)
        return self.archive._default_cell_key(obs)

    def run(self, total_steps: int = 1_000_000, verbose: bool = True) -> dict:
        """Run Go-Explore Phase 1.

        Args:
            total_steps: Total environment steps to run.
            verbose: Print progress.

        Returns:
            Dict with archive stats, best trajectory, etc.
        """
        steps = 0
        episodes = 0
        best_score = 0.0
        best_trajectory: list[int] = []

        # Initial exploration from reset
        obs, _ = self.env.reset()
        self.archive.add_or_update(obs, [], 0.0, self._get_cell_key(obs))

        while steps < total_steps:
            # Select a cell from the archive
            cell = self.archive.select_cell()
            if cell is None:
                obs, _ = self.env.reset()
                current_trajectory = []
            else:
                # Return to the cell by replaying its trajectory
                obs, _ = self.env.reset()
                current_trajectory = list(cell.trajectory)
                for action in cell.trajectory:
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    steps += 1
                    if terminated or truncated:
                        obs, _ = self.env.reset()
                        current_trajectory = []
                        break

            # Explore randomly from the cell
            last_action = None
            episode_score = 0.0

            for _ in range(self.explore_steps):
                # Sticky action (repeat previous with high probability)
                if last_action is not None and np.random.random() < self.sticky_action_prob:
                    action = last_action
                else:
                    action = np.random.randint(self.n_actions)

                obs, reward, terminated, truncated, info = self.env.step(action)
                steps += 1
                current_trajectory.append(action)
                episode_score += reward
                last_action = action

                # Add discovered state to archive
                key = self._get_cell_key(obs)
                self.archive.add_or_update(obs, current_trajectory, episode_score, key)

                if terminated or truncated:
                    episodes += 1
                    if episode_score > best_score:
                        best_score = episode_score
                        best_trajectory = list(current_trajectory)
                    obs, _ = self.env.reset()
                    current_trajectory = []
                    episode_score = 0.0
                    last_action = None

                if steps >= total_steps:
                    break

            if verbose and steps % 10000 < self.explore_steps + 200:
                stats = self.archive.stats()
                print(
                    f"  Steps: {steps:>8,} | "
                    f"Archive: {stats['archive_size']:>6,} | "
                    f"Best: {stats['best_score']:.2f} | "
                    f"Episodes: {episodes}"
                )

        stats = self.archive.stats()
        stats["total_steps"] = steps
        stats["episodes"] = episodes
        stats["best_trajectory_length"] = len(best_trajectory)
        return stats
