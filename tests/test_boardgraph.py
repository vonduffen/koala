"""Validator and serialization tests for BoardGraph (§3.1 invariants), vertex/intersection model."""

from __future__ import annotations

import numpy as np
import pytest

from tilinggo.tilings.boardgraph import (
    BoardGraph,
    BoardGraphError,
    build_vertex_graph,
    canonical_edges,
    validate,
)


def test_validate_accepts_good_graph(triangle_graph):
    validate(triangle_graph)  # should not raise


def test_canonical_edges_dedupes_and_sorts():
    raw = np.array([[2, 1], [1, 2], [0, 3], [3, 0]], dtype=np.int64)
    canon = canonical_edges(raw)
    assert canon.dtype == np.int32
    # mirrored pairs collapse to (min, max); rows sorted lexicographically
    assert canon.tolist() == [[0, 3], [1, 2]]


def test_build_vertex_graph_from_two_squares():
    # Two unit squares sharing an edge → 6 distinct vertices, 7 edges (shared edge once).
    sq0 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    sq1 = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float)
    coords, edges = build_vertex_graph([sq0, sq1])
    assert coords.shape == (6, 2)
    assert edges.shape == (7, 2)            # 4 + 4 − 1 shared
    # the shared edge (the two stacked vertices at x=1) appears exactly once
    assert edges.shape[0] == len(set(map(tuple, edges.tolist())))


def _with_edges(bg: BoardGraph, edges: np.ndarray) -> BoardGraph:
    return BoardGraph(bg.name, bg.num_nodes, edges, bg.coords, bg.meta)


def test_validate_rejects_self_loop(triangle_graph):
    bad = _with_edges(triangle_graph, np.array([[0, 0], [1, 2], [0, 2]], dtype=np.int32))
    with pytest.raises(BoardGraphError, match="self-loop"):
        validate(bad)


def test_validate_rejects_duplicate_edges(triangle_graph):
    bad = _with_edges(
        triangle_graph, np.array([[0, 1], [1, 0], [1, 2], [0, 2]], dtype=np.int32))
    with pytest.raises(BoardGraphError, match="duplicate"):
        validate(bad)


def test_validate_rejects_out_of_range_index(triangle_graph):
    bad = _with_edges(triangle_graph, np.array([[0, 1], [1, 2], [0, 9]], dtype=np.int32))
    with pytest.raises(BoardGraphError, match="out of range"):
        validate(bad)


def test_validate_rejects_low_degree():
    # 3 nodes but only one edge → nodes 0,1 have degree 1, node 2 degree 0.
    coords = np.zeros((3, 2), dtype=np.float32)
    bg = BoardGraph("d", 3, np.array([[0, 1]], dtype=np.int32), coords)
    with pytest.raises(BoardGraphError, match="degree"):
        validate(bg)


def test_validate_rejects_disconnected():
    # Two triangles, no link between them: degree-OK but disconnected.
    coords = np.zeros((6, 2), dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int32)
    bg = BoardGraph("disc", 6, edges, coords)
    with pytest.raises(BoardGraphError, match="not connected"):
        validate(bg)


def test_validate_rejects_high_degree():
    # A star: center connected to 13 leaves → center degree 13 > 12.
    n = 14
    coords = np.zeros((n, 2), dtype=np.float32)
    edges = np.array([[0, k] for k in range(1, n)], dtype=np.int32)
    bg = BoardGraph("star", n, edges, coords)
    with pytest.raises(BoardGraphError, match="degree"):
        validate(bg)


def test_validate_node_count_bounds(triangle_graph):
    with pytest.raises(BoardGraphError, match="below minimum"):
        validate(triangle_graph, min_nodes=10)
    with pytest.raises(BoardGraphError, match="above maximum"):
        validate(triangle_graph, max_nodes=2)


def test_save_load_round_trip(triangle_graph, tmp_path):
    triangle_graph.save(tmp_path / "g")
    loaded = BoardGraph.load(tmp_path / "g")
    assert loaded.name == triangle_graph.name
    assert loaded.num_nodes == triangle_graph.num_nodes
    assert np.array_equal(loaded.edges, triangle_graph.edges)
    assert np.allclose(loaded.coords, triangle_graph.coords)
    assert loaded.meta == triangle_graph.meta


def test_distance_to_boundary_interior_is_deeper():
    # On a square patch an interior intersection must be farther from the boundary than the rim.
    from tilinggo.tilings import periodic

    bg = periodic.generate("square", radius=2.5)
    d2b = bg.distance_to_boundary()
    deg = bg.degree()
    interior = np.where(deg == deg.max())[0]
    assert d2b[interior].max() > 0.0
    assert d2b.min() == 0.0  # boundary vertices are at distance 0
