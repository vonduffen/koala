"""Shared test fixtures and small graph builders."""

from __future__ import annotations

import numpy as np
import pytest

from tilinggo.tilings.boardgraph import BoardGraph


@pytest.fixture
def triangle_graph() -> BoardGraph:
    """A minimal valid graph: 3 vertices in a cycle (each degree 2)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    return BoardGraph("tri3", 3, edges, coords, meta={"family": "test"})
