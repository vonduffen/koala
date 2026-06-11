"""The tiling compiler must produce a deterministic node ordering for a given (tiling, size).

Shareable game links (webapp/share.js) and game records serialize moves as node *indices*, so
two compilations of the same board must agree node-for-node — otherwise a shared link silently
replays a different game. The webapp additionally guards with a graph fingerprint baked into
each link, but determinism at the source is the real contract; this test pins it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tilinggo.tilings import penrose, periodic, uniform

BUILDERS = [
    ("square 8x8", lambda: periodic.rectangular(8, 8)),
    ("hexagonal", lambda: periodic.generate("hex", cells=64)),
    ("triangular", lambda: periodic.generate("tri", cells=64)),
    ("trihexagonal", lambda: uniform.generate("trihexagonal", radius=4.0)),
    ("snub_square", lambda: uniform.generate("snub_square", radius=4.0)),
    ("penrose r5", lambda: penrose.generate(radius=5.0, symmetric=True)),
]


@pytest.mark.parametrize("label,build", BUILDERS, ids=[l for l, _ in BUILDERS])
def test_node_ordering_is_deterministic(label, build):
    a, b = build(), build()
    assert a.num_nodes == b.num_nodes, f"{label}: node count differs between compilations"
    assert np.array_equal(np.asarray(a.coords), np.asarray(b.coords)), (
        f"{label}: node coordinate ordering differs between compilations")
    assert np.array_equal(np.asarray(a.edges), np.asarray(b.edges)), (
        f"{label}: edge list differs between compilations")
