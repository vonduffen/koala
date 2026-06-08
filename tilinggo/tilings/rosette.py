"""Rosette board: a hexagon split into 6 quad-mesh sectors meeting at one degree-6 centre.

Suggested by a GitHub user. It can't be produced by any periodic grid: the centre is a degree-6
"defect" (a disclination), while everything else is ordinary degree-4 quad topology. The
geometry-blind engine plays it unchanged — it's just another BoardGraph.

Construction (clean all-quad mesh): connect the centre O to the six hexagon EDGE-MIDPOINTS M_s.
That partitions the hexagon into six quad ("kite") regions  O–M_s–V_{s+1}–M_{s+1}.  Each region is
meshed with an n×n grid by bilinear interpolation and glued to its neighbours along the shared
spokes O–M_s.  Result: one degree-6 centre, six degree-2 corners, the rest degree 4 (interior) /
3 (boundary), with a properly connected hexagonal perimeter.
"""

from __future__ import annotations

import math

import numpy as np

from .boardgraph import BoardGraph, validate


def generate(*, n: int = 6, smooth_iters: int = 100) -> BoardGraph:
    """Compile the rosette board with ``n`` grid divisions per sector edge.

    ``smooth_iters`` relaxes interior vertex *positions* toward their neighbours' mean (boundary +
    centre pinned). This is rendering-only — it gives the gently curved cells of the reference
    picture without changing the graph at all (topology, edges, gameplay are identical)."""
    V = [(math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)) for k in range(6)]   # corners
    M = [((V[k][0] + V[(k + 1) % 6][0]) / 2, (V[k][1] + V[(k + 1) % 6][1]) / 2) for k in range(6)]
    O = (0.0, 0.0)

    ids: dict[tuple[float, float], int] = {}
    coords: list[tuple[float, float]] = []

    def vid(p):
        key = (round(p[0], 5), round(p[1], 5))            # dedup shared spoke/centre/corner points
        if key not in ids:
            ids[key] = len(coords)
            coords.append((float(p[0]), float(p[1])))
        return ids[key]

    def blerp(c00, c10, c11, c01, u, v):                  # bilinear over a quad (straight isolines)
        return ((1 - u) * (1 - v) * c00[0] + u * (1 - v) * c10[0] + u * v * c11[0] + (1 - u) * v * c01[0],
                (1 - u) * (1 - v) * c00[1] + u * (1 - v) * c10[1] + u * v * c11[1] + (1 - u) * v * c01[1])

    edges: set[tuple[int, int]] = set()
    for s in range(6):
        Ms, V1, Ms1 = M[s], V[(s + 1) % 6], M[(s + 1) % 6]   # region corners: O, Ms, V1, Ms1
        def P(i, j, Ms=Ms, V1=V1, Ms1=Ms1):
            return blerp(O, Ms, V1, Ms1, i / n, j / n)
        for i in range(n + 1):
            for j in range(n + 1):
                a = vid(P(i, j))
                if i + 1 <= n:
                    b = vid(P(i + 1, j)); edges.add((min(a, b), max(a, b)))
                if j + 1 <= n:
                    b = vid(P(i, j + 1)); edges.add((min(a, b), max(a, b)))

    coords_arr = np.asarray(coords, dtype=np.float64)
    edges_arr = np.asarray(sorted(edges), dtype=np.int32)

    # rendering-only: relax interior vertices for the reference's curved-cell look (topology fixed)
    if smooth_iters > 0:
        src = np.concatenate([edges_arr[:, 0], edges_arr[:, 1]])
        dst = np.concatenate([edges_arr[:, 1], edges_arr[:, 0]])
        deg = np.bincount(dst, minlength=coords_arr.shape[0]).astype(np.float64)
        pinned = (deg < 4) | (deg == 6)              # hexagon perimeter (deg 2/3) + the deg-6 centre
        free = ~pinned
        for _ in range(smooth_iters):
            acc = np.zeros_like(coords_arr)
            np.add.at(acc, dst, coords_arr[src])     # sum of each node's neighbour positions
            mean = acc / deg[:, None]
            coords_arr[free] = mean[free]

    coords_arr = coords_arr.astype(np.float32)
    bg = BoardGraph(name=f"rosette_n{n}", num_nodes=coords_arr.shape[0], edges=edges_arr,
                    coords=coords_arr, meta={"family": "rosette", "n": int(n), "vconf": "deg6-centre"})
    validate(bg)
    return bg
