"""Tests for the periodic tiling generators (square / hex / tri)."""

from __future__ import annotations

import numpy as np
import pytest

from tilinggo.config import size_class_of
from tilinggo.tilings import periodic
from tilinggo.tilings.boardgraph import validate

# Expected interior vertex degree per family: the square grid's intersections have degree 4,
# the hexagonal tiling's (honeycomb) vertices degree 3, the triangular tiling's vertices 6.
INTERIOR_DEGREE = {"square": 4, "hex": 3, "tri": 6}


@pytest.mark.parametrize("family", ["square", "hex", "tri"])
def test_generated_graph_is_valid(family):
    bg = periodic.generate(family, cells=50, seed=0)
    validate(bg)  # must not raise


@pytest.mark.parametrize("family", ["square", "hex", "tri"])
def test_interior_degree_matches_family(family):
    bg = periodic.generate(family, cells=80, seed=0)
    deg = bg.degree()
    assert deg.max() == INTERIOR_DEGREE[family]
    # the modal (most common) degree should be the interior degree for a reasonably sized patch
    assert np.bincount(deg).argmax() == INTERIOR_DEGREE[family]


@pytest.mark.parametrize("family", ["square", "hex", "tri"])
def test_cell_count_close_to_target(family):
    # nearest-`cells` selection plus pruning lands near the target; allow generous slack.
    target = 50
    bg = periodic.generate(family, cells=target, seed=0)
    assert 0.7 * target <= bg.num_nodes <= 1.3 * target
    assert size_class_of(bg.num_nodes) == "S"


@pytest.mark.parametrize("family", ["square", "hex", "tri"])
def test_generation_is_deterministic(family):
    a = periodic.generate(family, cells=50, seed=0)
    b = periodic.generate(family, cells=50, seed=0)
    assert a.num_nodes == b.num_nodes
    assert np.array_equal(a.edges, b.edges)
    assert np.allclose(a.coords, b.coords)


@pytest.mark.parametrize("family", ["square", "hex", "tri"])
def test_radius_mode(family):
    bg = periodic.generate(family, radius=4.0, seed=1)
    validate(bg)
    assert bg.meta["radius_requested"] == 4.0
    # every kept centroid is within the clip radius
    norms = np.hypot(bg.coords[:, 0], bg.coords[:, 1])
    assert norms.max() <= bg.meta["clip_radius"] + 1e-3


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown family"):
        periodic.generate("octagon", cells=50)


def test_requires_exactly_one_size_arg():
    with pytest.raises(ValueError, match="exactly one"):
        periodic.generate("square")
    with pytest.raises(ValueError, match="exactly one"):
        periodic.generate("square", cells=50, radius=4.0)


def test_seed_recorded_in_meta():
    bg = periodic.generate("hex", cells=40, seed=7)
    assert bg.meta["seed"] == 7
    assert bg.meta["family"] == "hex"
    assert "s7" in bg.name


@pytest.mark.parametrize("side,V", [(2, 7), (3, 19), (5, 61), (9, 217)])
def test_triangular_hex_is_delta_go_board(side, V):
    # Reproduces the prior Delta-Go engine's board exactly: V = 3N²−3N+1, degree-6 interior.
    bg = periodic.triangular_hex(side)
    validate(bg)
    assert bg.num_nodes == V
    assert bg.meta["V"] == V
    assert bg.degree().max() == 6  # interior vertices have 6 neighbours


def test_triangular_hex_rejects_tiny():
    with pytest.raises(ValueError):
        periodic.triangular_hex(1)
