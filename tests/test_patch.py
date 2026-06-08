"""Tests for disc clipping, min-degree pruning, and largest-component extraction.

These now operate on a vertex graph ``(coords, edges)`` — no polygons.
"""

from __future__ import annotations

import numpy as np

from tilinggo.tilings import patch as patchmod


def _grid(n=5):
    """An n x n grid of vertices with 4-neighbour adjacency, centred on the origin."""
    keys = [(i, j) for i in range(n) for j in range(n)]
    index = {k: idx for idx, k in enumerate(keys)}
    off = (n - 1) / 2.0
    coords = np.array([[i - off, j - off] for i, j in keys], dtype=np.float32)
    edges = []
    for (i, j), a in index.items():
        for nb in [(i + 1, j), (i, j + 1)]:
            if nb in index:
                edges.append((a, index[nb]))
    return coords, np.array(edges, dtype=np.int32)


def test_clip_to_disc_keeps_only_inside():
    c, e = _grid(5)
    c2, e2 = patchmod.clip_to_disc(c, e, radius=1.5)
    norms = np.hypot(c2[:, 0], c2[:, 1])
    assert norms.max() <= 1.5 + 1e-6
    assert e2.max() < c2.shape[0]  # edges only reference surviving nodes


def test_largest_connected_component_drops_islands():
    c1, e1 = _grid(4)  # 16 nodes
    c2, e2 = _grid(2)  # 4 nodes, shifted far away
    c2 = c2 + np.array([100.0, 100.0], dtype=np.float32)
    e2 = e2 + c1.shape[0]
    coords = np.concatenate([c1, c2], axis=0)
    edges = np.concatenate([e1, e2], axis=0)

    c, e = patchmod.largest_connected_component(coords, edges)
    assert c.shape[0] == 16
    assert e.max() < 16


def test_prune_min_degree_removes_pendants():
    # a triangle (deg 2 each) plus a pendant vertex attached by a single edge (deg 1).
    coords = np.array([[0, 0], [1, 0], [0, 1], [5, 5]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [0, 2], [2, 3]], dtype=np.int32)
    c, e = patchmod.prune_min_degree(coords, edges, min_degree=2)
    assert c.shape[0] == 3  # the pendant (node 3) is gone
    deg = np.zeros(3, dtype=int)
    np.add.at(deg, e[:, 0], 1)
    np.add.at(deg, e[:, 1], 1)
    assert deg.min() >= 2


def test_clip_empty_when_radius_too_small():
    c, e = _grid(5)
    c2, e2 = patchmod.clip_to_disc(c, e, radius=0.1)
    assert c2.shape[0] == 1  # only the centre vertex survives
    assert e2.shape[0] == 0
