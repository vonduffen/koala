"""Regular periodic tiling generators: square, triangular, hexagonal.

Per ARCHITECTURE.md §3.2 these come first — they are the test bed and the easy end of the
curriculum. Playable points are the tiling's **intersections (vertices)**; adjacency runs
along tiling edges (the 1-skeleton). See ``boardgraph`` for the divergence from §3.1.

We analytically stamp out the tiling's face polygons over a region, then derive the vertex
graph from those polygons (shared corners collapse to one intersection), and hand off to
``patch`` for disc-clipping, min-degree pruning, and largest-component extraction. Generation
is fully deterministic, so the same parameters always yield the same graph (golden tests).
"""

from __future__ import annotations

import math

import numpy as np

from .boardgraph import BoardGraph, build_vertex_graph, canonical_edges, validate
from . import patch as patchmod

# Side/size length is 1 for every family. Cell-area densities (cells per unit area) used only
# to pick a generation radius that comfortably contains the requested number of cells.
_SQRT3 = math.sqrt(3.0)
_TRI_H = _SQRT3 / 2.0  # row height of a unit equilateral triangle
_DENSITY = {
    "square": 1.0,                 # unit squares, area 1
    "hex": 2.0 / (3.0 * _SQRT3),   # regular hexagon, area (3*sqrt(3)/2); density = 1/area
    "tri": 2.0 / _TRI_H,           # two unit triangles (area sqrt(3)/4 each) per row-cell
}


# ---- per-family geometry -------------------------------------------------------------------

def _square_cell(key):
    i, j = key
    cx, cy = float(i), float(j)
    poly = np.array(
        [[cx - 0.5, cy - 0.5], [cx + 0.5, cy - 0.5],
         [cx + 0.5, cy + 0.5], [cx - 0.5, cy + 0.5]],
        dtype=np.float32,
    )
    return (cx, cy), poly


def _square_neighbors(key):
    i, j = key
    return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]


def _hex_center(q, r):
    return _SQRT3 * (q + r / 2.0), 1.5 * r


def _hex_cell(key):
    q, r = key
    cx, cy = _hex_center(q, r)
    # Pointy-top hexagon: vertices at 60*k - 30 degrees, circumradius 1.
    angles = np.deg2rad(60.0 * np.arange(6) - 30.0)
    poly = np.stack([cx + np.cos(angles), cy + np.sin(angles)], axis=1).astype(np.float32)
    return (cx, cy), poly


def _hex_neighbors(key):
    q, r = key
    return [(q + 1, r), (q - 1, r), (q, r + 1), (q, r - 1), (q + 1, r - 1), (q - 1, r + 1)]


def _tri_vertex(i, j):
    """Triangular-lattice vertex (i, j) → cartesian position."""
    return i + 0.5 * j, j * _TRI_H


def _tri_cell(key):
    i, j, o = key
    if o == 0:  # up triangle: base at bottom, apex at top
        verts = [_tri_vertex(i, j), _tri_vertex(i + 1, j), _tri_vertex(i, j + 1)]
    else:       # down triangle: base at top, apex at bottom
        verts = [_tri_vertex(i + 1, j), _tri_vertex(i, j + 1), _tri_vertex(i + 1, j + 1)]
    poly = np.array(verts, dtype=np.float32)
    cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
    return (float(cx), float(cy)), poly


def _tri_neighbors(key):
    i, j, o = key
    if o == 0:  # up neighbors are the three adjacent down triangles
        return [(i, j - 1, 1), (i - 1, j, 1), (i, j, 1)]
    return [(i, j, 0), (i, j + 1, 0), (i + 1, j, 0)]  # down neighbors are three up triangles


# ---- candidate enumeration -----------------------------------------------------------------

def _enumerate(family, gen_radius):
    """Yield all cell keys of ``family`` whose centroid lies within ``gen_radius``."""
    if family == "square":
        b = int(math.ceil(gen_radius)) + 2
        for i in range(-b, b + 1):
            for j in range(-b, b + 1):
                (cx, cy), _ = _square_cell((i, j))
                if cx * cx + cy * cy <= gen_radius * gen_radius:
                    yield (i, j)
    elif family == "hex":
        b = int(math.ceil(gen_radius)) + 3
        for q in range(-b, b + 1):
            for r in range(-b, b + 1):
                cx, cy = _hex_center(q, r)
                if cx * cx + cy * cy <= gen_radius * gen_radius:
                    yield (q, r)
    elif family == "tri":
        b = int(math.ceil(gen_radius / _TRI_H)) + 3
        for i in range(-b, b + 1):
            for j in range(-b, b + 1):
                for o in (0, 1):
                    (cx, cy), _ = _tri_cell((i, j, o))
                    if cx * cx + cy * cy <= gen_radius * gen_radius:
                        yield (i, j, o)
    else:
        raise ValueError(f"unknown family {family!r}")


_CELL_FN = {"square": _square_cell, "hex": _hex_cell, "tri": _tri_cell}
_NBR_FN = {"square": _square_neighbors, "hex": _hex_neighbors, "tri": _tri_neighbors}


def _polys_over_region(family, gen_radius):
    """All face polygons whose centroid lies within ``gen_radius`` — the vertex-graph source."""
    cell_fn = _CELL_FN[family]
    return [cell_fn(k)[1] for k in _enumerate(family, gen_radius)]


# ---- public API ----------------------------------------------------------------------------

def generate(family: str, *, cells: int | None = None, radius: float | None = None,
             seed: int = 0) -> BoardGraph:
    """Compile a periodic tiling patch into a validated :class:`BoardGraph`.

    Specify exactly one of:
        cells:  approximate target number of intersections (the patch radius is chosen to fit).
        radius: explicit disc radius in edge-length units.

    ``seed`` is recorded in ``meta`` for reproducibility; periodic generation is deterministic
    so the seed does not perturb geometry. The returned graph passes :func:`validate`.
    """
    if family not in _DENSITY:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(_DENSITY)}")
    if (cells is None) == (radius is None):
        raise ValueError("specify exactly one of `cells` or `radius`")

    if radius is not None:
        gen_radius = float(radius)
        clip_radius = float(radius)
        size_tag = f"r{radius:g}"
    else:
        density = _DENSITY[family]
        # Over-generate (face count ≈ 2.4x target) so the region holds many more vertices than
        # the requested intersection count; nearest-`cells` selection then trims to size.
        gen_radius = math.sqrt(2.4 * max(cells, 1) / (math.pi * density)) + 2.0
        clip_radius = None
        size_tag = f"c{cells}"

    # +1.0 margin so vertices near the clip radius still get all their incident edges.
    coords, edges = build_vertex_graph(_polys_over_region(family, gen_radius + 1.0))

    if clip_radius is None:
        dists = np.sort(np.hypot(coords[:, 0], coords[:, 1]))
        k = min(int(cells), dists.shape[0])
        clip_radius = float(dists[k - 1]) + 1e-4

    coords, edges = patchmod.clip_to_disc(coords, edges, clip_radius)
    coords, edges = patchmod.prune_min_degree(coords, edges)
    coords, edges = patchmod.largest_connected_component(coords, edges)

    bg = BoardGraph(
        name=f"{family}_{size_tag}_s{seed}",
        num_nodes=coords.shape[0],
        edges=edges,
        coords=coords.astype(np.float32),
        meta={
            "family": family,
            "cells_requested": cells,
            "radius_requested": radius,
            "clip_radius": float(clip_radius),
            "seed": int(seed),
        },
    )
    validate(bg)
    return bg


def rectangular(rows: int, cols: int, *, seed: int = 0) -> BoardGraph:
    """A full (unclipped) rows x cols grid of intersections — a classic rectangular goban.

    Node ``r*cols + c`` is the intersection at (row r, col c); adjacency is the 4-neighbour
    rule. This is literally classical Go on a grid (the square tiling's vertices), with
    predictable node indices — which is what the rules-engine and differential tests need.
    Corner intersections have degree 2, edges 3, interior 4.
    """
    if rows < 1 or cols < 1 or rows * cols < 2:
        raise ValueError("rectangular grid needs at least 2 intersections")
    n = rows * cols
    coords = np.zeros((n, 2), dtype=np.float32)
    edge_list: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            coords[idx] = (float(c), float(rows - 1 - r))  # row 0 at the top when rendered
            if c + 1 < cols:
                edge_list.append((idx, idx + 1))
            if r + 1 < rows:
                edge_list.append((idx, idx + cols))

    bg = BoardGraph(
        name=f"rect_{rows}x{cols}_s{seed}",
        num_nodes=n,
        edges=canonical_edges(np.asarray(edge_list, dtype=np.int64)),
        coords=coords,
        meta={"family": "rectangular", "rows": rows, "cols": cols, "seed": int(seed)},
    )
    validate(bg)
    return bg


# Axial neighbour offsets on the triangular lattice (6 around each interior vertex).
_HEX_NBR = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def triangular_hex(side: int, *, seed: int = 0) -> BoardGraph:
    """Triangular lattice clipped to a regular hexagon — the **Delta-Go** board.

    Vertices are axial coords ``(q, r)`` with ``|q|, |r|, |q+r| <= side-1``; adjacency is the 6
    axial neighbours. This reproduces the board of the prior Delta-Go engine exactly (its
    ``V = 3·side² − 3·side + 1`` formula, degree-6 interior, D6 symmetry), so our graph-Go
    engine can be cross-checked against it. ``side`` is intersections per hexagon edge.
    """
    if side < 2:
        raise ValueError("side must be >= 2")
    s = side - 1
    keys = [(q, r) for q in range(-s, s + 1) for r in range(-s, s + 1) if abs(q + r) <= s]
    index = {k: i for i, k in enumerate(keys)}

    coords = np.array([[q + 0.5 * r, r * _TRI_H] for q, r in keys], dtype=np.float32)
    edge_list = []
    for (q, r), a in index.items():
        for dq, dr in _HEX_NBR:
            b = index.get((q + dq, r + dr))
            if b is not None and a < b:
                edge_list.append((a, b))

    bg = BoardGraph(
        name=f"deltahex_n{side}_s{seed}",
        num_nodes=len(keys),
        edges=canonical_edges(np.asarray(edge_list, dtype=np.int64)),
        coords=coords,
        meta={"family": "triangular_hex", "side": int(side),
              "V": 3 * side * side - 3 * side + 1, "seed": int(seed)},
    )
    validate(bg)
    return bg
