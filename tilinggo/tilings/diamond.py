"""Diamond-cubic lattice patch — Go in three dimensions.

Naive 3D Go on the cubic lattice is degenerate: degree 6 gives every stone six liberties and
surrounding a volume costs a 2D shell of stones, so capture economics collapse. The diamond
cubic lattice (carbon's crystal structure) is the natural fix: **4-regular** — every interior
point has exactly four neighbours, the same liberty budget as 2D square Go — yet genuinely
three-dimensional (smallest rings are the 6-cycles of the carbon "chair" hexagons; the graph
is bipartite and non-planar).

The geometry-blind engine plays it unchanged: rules, features (degree one-hot, BFS
boundary-distance, Laplacian PE) and search are all graph-native. ``coords`` carries an
isometric 2D *projection* used only for rendering; the topology lives in the edges.

    generate(cells=2)  -> ~50-60 nodes after trimming (whiskers pruned to keep degree >= 2)
    generate(cells=3)  -> ~180-200 nodes
"""

from __future__ import annotations

import numpy as np

from .boardgraph import BoardGraph, canonical_edges, validate

# conventional cell: FCC sites + the (1/4,1/4,1/4)-offset second sublattice
_FCC = np.array([(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)])
_BASIS = np.array([(0, 0, 0), (.25, .25, .25)])
_BOND = np.sqrt(3.0) / 4.0  # nearest-neighbour distance, lattice constant 1


def generate(*, cells: int = 2, name: str | None = None) -> BoardGraph:
    # all atom positions in a cells^3 patch
    pts = []
    for cx in range(cells):
        for cy in range(cells):
            for cz in range(cells):
                origin = np.array([cx, cy, cz], dtype=float)
                for f in _FCC:
                    for b in _BASIS:
                        pts.append(origin + f + b)
    pts = np.unique(np.round(np.asarray(pts), 6), axis=0)

    # bonds = pairs at the nearest-neighbour distance
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    tol = 1e-6
    ii, jj = np.where(np.abs(d2 - _BOND ** 2) < tol)
    edges = canonical_edges(np.stack([ii, jj], axis=1)[ii < jj])

    # trim whiskers: iteratively drop nodes with degree < 2 (boundary dangling bonds)
    keep = np.ones(len(pts), dtype=bool)
    while True:
        deg = np.zeros(len(pts), dtype=int)
        for a, b in edges:
            if keep[a] and keep[b]:
                deg[a] += 1
                deg[b] += 1
        drop = np.where(keep & (deg < 2))[0]
        if drop.size == 0:
            break
        keep[drop] = False
    idx = np.where(keep)[0]
    remap = -np.ones(len(pts), dtype=int)
    remap[idx] = np.arange(idx.size)
    pts = pts[idx]
    edges = np.asarray([(remap[a], remap[b]) for a, b in edges if keep[a] and keep[b]],
                       dtype=np.int32)

    # largest connected component (paranoia; the trimmed patch is normally connected)
    n = len(pts)
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    comp = -np.ones(n, dtype=int)
    c = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if comp[w] < 0:
                    comp[w] = c
                    stack.append(w)
        c += 1
    main = np.argmax(np.bincount(comp))
    sel = np.where(comp == main)[0]
    remap = -np.ones(n, dtype=int)
    remap[sel] = np.arange(sel.size)
    pts = pts[sel]
    edges = canonical_edges(np.asarray(
        [(remap[a], remap[b]) for a, b in edges if comp[a] == main and comp[b] == main]))

    # isometric projection for rendering only (topology is in the edges)
    iso_x = pts[:, 0] - pts[:, 2] * 0.52
    iso_y = pts[:, 1] - pts[:, 2] * 0.30 + pts[:, 0] * 0.12
    coords = np.stack([iso_x, iso_y], axis=1)
    # nudge any coincidentally-overlapping projections apart (render sanity)
    for _ in range(3):
        d = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(d, 1.0)
        a, b = np.where(d < 1e-4)
        if a.size == 0:
            break
        coords[a] += np.random.default_rng(0).normal(0, 0.02, size=(a.size, 2))

    bg = BoardGraph(name=name or f"diamond_c{cells}", num_nodes=int(len(pts)), edges=edges,
                    coords=coords.astype(np.float32),
                    meta={"family": "diamond_cubic", "cells": cells, "dim": 3,
                          "coords3": pts.tolist()})
    validate(bg)
    return bg
