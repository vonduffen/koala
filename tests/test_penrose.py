"""Tests for the Penrose (P3 rhombus) aperiodic tiling compiler (ARCHITECTURE.md §3.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tilinggo.rules import Board
from tilinggo.tilings import penrose, symmetry
from tilinggo.tilings.boardgraph import validate


def _acute_angles(rhombs):
    """The smallest interior angle (degrees) of each rhomb."""
    out = []
    for poly in rhombs:
        a = poly[1] - poly[0]
        b = poly[3] - poly[0]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        ang = math.degrees(math.acos(np.clip(cos, -1, 1)))
        out.append(round(min(ang, 180 - ang)))
    return out


def test_only_thick_and_thin_rhombs():
    rhombs = penrose._rhombs(penrose._offsets(symmetric=False, seed=0), line_range=6)
    # Penrose P3 has exactly two rhombs: thick (acute 72°) and thin (acute 36°)
    angles = set(_acute_angles(rhombs))
    assert angles == {36, 72}
    # every rhomb has unit edges
    for poly in rhombs:
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            assert abs(np.linalg.norm(edge) - 1.0) < 1e-6


@pytest.mark.parametrize("kw", [dict(symmetric=True), dict(symmetric=False, seed=1)])
def test_generates_valid_intersection_graph(kw):
    bg = penrose.generate(radius=6, **kw)
    validate(bg)
    assert bg.meta["family"] == "penrose"
    # all graph edges are unit length (rhomb sides)
    a, b = bg.coords[bg.edges[:, 0]], bg.coords[bg.edges[:, 1]]
    el = np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])
    assert np.allclose(el, 1.0, atol=1e-3)
    assert 2 <= bg.degree().min() and bg.degree().max() <= 12


def test_sun_patch_has_fivefold_symmetry():
    bg = penrose.generate(radius=6, symmetric=True)
    # the equal-offset "sun" patch is 5-fold symmetric (D5 = 10 automorphisms)
    assert len(symmetry.symmetries(bg)) == 10
    # a generic patch is aperiodic with no exact symmetry
    assert len(symmetry.symmetries(penrose.generate(radius=6, seed=3, symmetric=False))) == 1


def test_determinism():
    a = penrose.generate(radius=5, seed=2, symmetric=False)
    b = penrose.generate(radius=5, seed=2, symmetric=False)
    assert a.num_nodes == b.num_nodes
    assert np.array_equal(a.edges, b.edges)


def test_rules_engine_plays_on_penrose():
    # the geometry-blind engine must run on an aperiodic board with no changes
    board = Board(penrose.generate(radius=5, symmetric=True), komi=0.5)
    s = board.new_game().play(0).play(1)
    assert s.legal_moves().any()
    assert s.move_num == 2
