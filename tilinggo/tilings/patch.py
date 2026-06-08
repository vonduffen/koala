"""Clip an (infinite) tiling's intersection graph down to a finite, connected, playable patch.

Per ARCHITECTURE.md §3.3: clip to a disc of radius ``r`` around the origin (here, keep the
*vertices* within ``r``), drop pendant vertices, then keep the largest connected component. No
wraparound, no torus. The resulting irregular boundary is a feature — the network learns edge
effects from the ``distance_to_boundary`` input rather than from any assumed shape.

These functions operate on a vertex graph ``(coords[N, 2], edges[E, 2])``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

VertexGraph = tuple[np.ndarray, np.ndarray]


def _remap(coords: np.ndarray, edges: np.ndarray, keep: np.ndarray) -> VertexGraph:
    """Restrict to the vertices in ``keep`` (a sorted index array) and re-index edges."""
    old_to_new = np.full(coords.shape[0], -1, dtype=np.int64)
    old_to_new[keep] = np.arange(keep.shape[0])
    new_coords = coords[keep]
    if edges.shape[0]:
        a = old_to_new[edges[:, 0]]
        b = old_to_new[edges[:, 1]]
        mask = (a >= 0) & (b >= 0)
        new_edges = np.stack([a[mask], b[mask]], axis=1).astype(np.int32)
    else:
        new_edges = np.zeros((0, 2), dtype=np.int32)
    return new_coords, new_edges


def clip_to_disc(coords: np.ndarray, edges: np.ndarray, radius: float,
                 tol: float = 1e-4) -> VertexGraph:
    """Keep only vertices within ``radius`` of the origin.

    A symmetry orbit of boundary vertices shares one exact radius, but float32 coordinate noise
    (~1e-6) can scatter their computed distances across a sharp threshold and split the orbit —
    silently breaking the patch's rotational symmetry. The ``tol`` slack keeps whole boundary
    orbits together (it is far below the inter-shell spacing, so it never pulls in an outer
    orbit), so a disc clipped about a symmetry centre stays exactly symmetric.
    """
    c = np.asarray(coords, dtype=np.float64)
    keep = np.where(np.hypot(c[:, 0], c[:, 1]) <= float(radius) + tol)[0]
    return _remap(np.asarray(coords, dtype=np.float32), edges, keep)


def prune_min_degree(coords: np.ndarray, edges: np.ndarray, min_degree: int = 2) -> VertexGraph:
    """Iteratively drop vertices whose degree is below ``min_degree``.

    Clipping leaves dangling boundary vertices (degree 1, or isolated) that violate the §3.1
    ``degree >= 2`` invariant; removing one can drop a neighbour below the threshold too, so
    this repeats until stable.
    """
    while True:
        n = coords.shape[0]
        if n == 0:
            return coords, edges
        deg = np.zeros(n, dtype=np.int64)
        if edges.shape[0]:
            np.add.at(deg, edges[:, 0], 1)
            np.add.at(deg, edges[:, 1], 1)
        keep = np.where(deg >= min_degree)[0]
        if keep.shape[0] == n:
            return coords, edges
        coords, edges = _remap(coords, edges, keep)


def largest_connected_component(coords: np.ndarray, edges: np.ndarray) -> VertexGraph:
    """Keep the largest connected component (ties broken by smallest member index)."""
    n = coords.shape[0]
    if n == 0:
        return coords, edges

    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    comp = np.full(n, -1, dtype=np.int64)
    components: list[list[int]] = []
    for start in range(n):
        if comp[start] != -1:
            continue
        members: list[int] = []
        comp[start] = len(components)
        dq = deque([start])
        while dq:
            u = dq.popleft()
            members.append(u)
            for w in adj[u]:
                if comp[w] == -1:
                    comp[w] = len(components)
                    dq.append(w)
        components.append(members)

    best = max(range(len(components)), key=lambda c: (len(components[c]), -min(components[c])))
    keep = np.array(sorted(components[best]), dtype=np.int64)
    return _remap(coords, edges, keep)
