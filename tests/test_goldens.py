"""Golden tests: regenerating a fixed patch must exactly reproduce the checked-in artifacts.

The goldens are produced by ``scripts/make_goldens.py``. If a deliberate change to geometry
or serialization makes these fail, re-run that script and review the diff before committing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tilinggo.tilings import periodic
from tilinggo.tilings.boardgraph import BoardGraph
from tilinggo.ui import render

GOLDEN_DIR = Path(__file__).parent / "goldens"
CASES = [("square", 40), ("hex", 40), ("tri", 40)]


@pytest.mark.parametrize("family,cells", CASES)
def test_graph_matches_golden(family, cells):
    fresh = periodic.generate(family, cells=cells, seed=0)
    golden = BoardGraph.load(GOLDEN_DIR / fresh.name)
    assert fresh.num_nodes == golden.num_nodes
    assert np.array_equal(fresh.edges, golden.edges)
    assert np.allclose(fresh.coords, golden.coords, atol=1e-5)


@pytest.mark.parametrize("family,cells", CASES)
def test_svg_matches_golden(family, cells):
    fresh = periodic.generate(family, cells=cells, seed=0)
    svg = render.to_svg(fresh)
    golden_svg = (GOLDEN_DIR / f"{fresh.name}.svg").read_text()
    assert svg == golden_svg
