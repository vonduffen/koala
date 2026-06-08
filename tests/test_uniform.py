"""Tests for the 11 uniform (Archimedean) Euclidean tilings, intersection model.

In the vertex model an interior intersection's degree equals the number of polygons meeting at
that vertex — i.e. the length of the tiling's vertex configuration.
"""

from __future__ import annotations

import numpy as np
import pytest

from tilinggo.tilings import uniform
from tilinggo.tilings.boardgraph import validate

# vertex configuration → interior vertex degree (number of polygons around a vertex)
INTERIOR_DEGREE = {
    "square": 4,         # 4.4.4.4
    "triangular": 6,     # 3.3.3.3.3.3
    "hexagonal": 3,      # 6.6.6
    "trunc_square": 3,   # 4.8.8
    "trunc_hex": 3,      # 3.12.12
    "trihexagonal": 4,   # 3.6.3.6
    "elongated_tri": 5,  # 3.3.3.4.4
    "rhombitrihex": 4,   # 3.4.6.4
    "trunc_trihex": 3,   # 4.6.12
    "snub_square": 5,    # 3.3.4.3.4
    "snub_hex": 5,       # 3.3.3.3.6
}


def test_all_eleven_present():
    assert set(uniform.TILING_NAMES) == set(INTERIOR_DEGREE)


@pytest.mark.parametrize("name", list(INTERIOR_DEGREE))
def test_generates_and_validates(name):
    bg = uniform.generate(name)
    validate(bg)  # connected, no dupes/self-loops, 2<=deg<=12, etc.
    assert bg.num_nodes > 0
    assert bg.meta["family"] == name


@pytest.mark.parametrize("name,expected", INTERIOR_DEGREE.items())
def test_interior_vertex_degree(name, expected):
    bg = uniform.generate(name)
    deg = bg.degree()
    d2b = bg.distance_to_boundary()
    interior = d2b > 0.5
    assert interior.any()
    # deep-interior intersections must all have the tiling's characteristic vertex degree
    assert np.all(deg[interior] == expected), (name, np.unique(deg[interior]))
    # and that degree is the most common one overall
    assert int(np.bincount(deg).argmax()) == expected


def test_determinism():
    a = uniform.generate("snub_hex")
    b = uniform.generate("snub_hex")
    assert a.num_nodes == b.num_nodes
    assert np.array_equal(a.edges, b.edges)


def test_unknown_tiling_raises():
    with pytest.raises(ValueError, match="unknown tiling"):
        uniform.generate("penrose")
