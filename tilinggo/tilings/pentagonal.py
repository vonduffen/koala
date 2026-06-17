"""Laves pentagonal tilings — the edge-to-edge pentagon tilings, built as duals.

The three edge-to-edge convex pentagonal tilings on https://en.wikipedia.org/wiki/Pentagonal_tiling
are the Laves duals of Archimedean tilings we already generate:

    cairo      V3.3.4.3.4   dual of snub-square          (Type 4, p4g)
    prismatic  V3.3.3.4.4   dual of elongated-triangular (Type 1, cmm)
    floret     V3.3.3.3.6   dual of snub-hexagonal       (Types 1/5/6, p6)

Rather than hand-deriving each pentagon lattice, we build them generically: dualize the parent
Archimedean face soup. Each *interior* parent vertex becomes a dual face whose corners are the
centroids of the parent faces meeting at that vertex, ordered angularly around it. Every parent
vertex here has degree 5, so each dual face is a pentagon. The dual soup then feeds the same
``build_vertex_graph -> clip -> prune -> largest_component -> validate`` pipeline as every other
tiling (see :mod:`tilinggo.tilings.uniform`), so stones still play on the tiling's 1-skeleton.
"""

from __future__ import annotations

import math

import numpy as np

from . import patch as patchmod
from . import uniform
from .boardgraph import BoardGraph, build_vertex_graph, validate

# Edge length 1 in the parent ⇒ dual edges are O(1); 1e-3 cleanly separates distinct vertices.
_QEPS = 1e-3

# name -> (parent Archimedean tiling, dual vertex configuration, default patch radius)
_DUALS: dict[str, dict] = {
    "cairo":     dict(parent="snub_square",   vconf="V3.3.4.3.4", radius=6.0),
    "prismatic": dict(parent="elongated_tri", vconf="V3.3.3.4.4", radius=6.0),
    "floret":    dict(parent="snub_hex",      vconf="V3.3.3.3.6", radius=7.0),
}
TILING_NAMES = list(_DUALS)


def _dual_polys(parent_polys, qeps: float = _QEPS):
    """Dual face soup: one polygon per interior parent vertex (centroids of its incident faces).

    Boundary parent vertices (of the generation region) are missing faces, which shows up as a
    large angular gap between consecutive incident-face centroids; those are dropped so the soup
    contains only complete pentagons. The disc clip downstream removes whatever this leaves near
    the requested boundary.
    """
    centroids = [np.asarray(p, dtype=np.float64).mean(axis=0) for p in parent_polys]

    index: dict[tuple[int, int], int] = {}
    vcoord: list[tuple[float, float]] = []
    incident: list[list[int]] = []

    def vid(pt) -> int:
        key = (int(round(pt[0] / qeps)), int(round(pt[1] / qeps)))
        i = index.get(key)
        if i is None:
            i = len(vcoord)
            index[key] = i
            vcoord.append((float(pt[0]), float(pt[1])))
            incident.append([])
        return i

    for fi, p in enumerate(parent_polys):
        p = np.asarray(p, dtype=np.float64)
        for k in range(len(p)):
            incident[vid(p[k])].append(fi)

    duals: list[np.ndarray] = []
    for vi, faces in enumerate(incident):
        if len(faces) < 3:
            continue
        vx, vy = vcoord[vi]
        cs = [centroids[f] for f in faces]
        ang = [math.atan2(c[1] - vy, c[0] - vx) for c in cs]
        order = sorted(range(len(cs)), key=lambda k: ang[k])
        sa = [ang[o] for o in order]
        gaps = [(sa[(k + 1) % len(sa)] - sa[k]) % (2 * math.pi) for k in range(len(sa))]
        if max(gaps) > math.radians(150.0):  # a missing face ⇒ boundary vertex of the region
            continue
        duals.append(np.asarray([cs[o] for o in order], dtype=np.float64))
    return duals


def generate(name: str, *, radius: float | None = None, seed: int = 0) -> BoardGraph:
    """Compile a Laves pentagonal tiling patch into a validated :class:`BoardGraph`."""
    if name not in _DUALS:
        raise ValueError(f"unknown pentagonal tiling {name!r}; expected one of {TILING_NAMES}")
    spec = _DUALS[name]
    r = float(radius) if radius is not None else float(spec["radius"])

    # Generate the parent over a generous margin so dual cells near the requested disc are complete.
    parent_spec = uniform._SPECS[spec["parent"]]
    parent_polys = uniform._polys_for(parent_spec, r + 8.0)

    coords, edges = build_vertex_graph(_dual_polys(parent_polys), qeps=_QEPS)
    coords, edges = patchmod.clip_to_disc(coords, edges, r)
    coords, edges = patchmod.prune_min_degree(coords, edges)
    coords, edges = patchmod.largest_connected_component(coords, edges)

    bg = BoardGraph(
        name=f"{name}_r{r:g}_s{seed}",
        num_nodes=coords.shape[0],
        edges=edges,
        coords=coords.astype(np.float32),
        meta={"family": name, "vconf": spec["vconf"], "parent": spec["parent"],
              "radius_requested": r, "seed": int(seed)},
    )
    validate(bg)
    return bg
