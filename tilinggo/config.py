"""Project configuration: board-size classes and a seeded-RNG helper.

Kept deliberately small for Milestone 1. As later milestones land (komi tables, training
hyperparameters, the YAML experiment schema of §2), this is where their dataclasses live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SizeClass:
    """A board-size class (ARCHITECTURE.md §3.3), used for curriculum and komi tables."""

    name: str
    min_cells: int
    max_cells: int

    def contains(self, n: int) -> bool:
        return self.min_cells <= n <= self.max_cells


# §3.3: S 25–60, M 61–140, L 141–300.
SIZE_CLASSES: dict[str, SizeClass] = {
    "S": SizeClass("S", 25, 60),
    "M": SizeClass("M", 61, 140),
    "L": SizeClass("L", 141, 300),
}


def size_class_of(n: int) -> str | None:
    """Return the name of the size class containing ``n`` cells, or None if out of range."""
    for sc in SIZE_CLASSES.values():
        if sc.contains(n):
            return sc.name
    return None


def make_rng(seed: int) -> np.random.Generator:
    """Reproducible RNG. Every artifact derived from one should record ``seed`` in its meta."""
    return np.random.default_rng(seed)
