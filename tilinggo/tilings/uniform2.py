"""The 2-uniform Euclidean tilings — tilings with exactly TWO vertex orbits (two vertex types).

There are 20 of them (Grünbaum–Shephard). Each is periodic, so — exactly like the 1-uniform
(Archimedean) tilings in ``uniform.py`` — we describe it by two lattice translation vectors and a
motif (the regular polygons in one fundamental domain, as ``(cx, cy, n_sides, rot_deg)``), stamp it
across a disc, and build the intersection graph. The engine is geometry-blind, so these play with
no changes; they're a strong showcase of "any tiling".

Numbered ``twouNN`` in the standard Grünbaum–Shephard reading order; the label carries the vertex
configuration ``(A; B)``. Built and verified incrementally (rendered against the reference plate).
"""

from __future__ import annotations

import math

import numpy as np

from . import patch as patchmod
from .boardgraph import BoardGraph, validate, build_vertex_graph
from .uniform import _reg

_S3 = math.sqrt(3.0)
_H = _S3 / 2.0          # height of a unit equilateral triangle
_QEPS = 1e-3


# Each spec: label (vertex configs), v1, v2 (lattice), motif (polygons), radius.
# Verified = rendered AND confirmed to be a connected 2D board (not a degenerate strip).
_SPECS2: dict[str, dict] = {
    # ---- square + triangle "stripe" family ------------------------------------------------
    # #3  (3^6 ; 3^3.4^2)_1 — square rows separated by DOUBLE triangle bands
    "twou03": dict(
        label="(3⁶; 3³.4²)₁", radius=6.0, v1=(1.0, 0.0), v2=(0.0, 1.0 + 2 * _H),
        motif=[(0.0, 0.0, 4, 45.0),
               (0.0, 0.5 + _H / 3, 3, 90.0), (0.5, 0.5 + 2 * _H / 3, 3, 270.0),
               (0.5, 0.5 + _H + _H / 3, 3, 90.0), (1.0, 0.5 + _H + 2 * _H / 3, 3, 270.0)],
    ),
    # #12 (3^3.4^2 ; 4^4)_1 — DOUBLE square rows + a triangle row
    "twou12": dict(
        label="(3³.4²; 4⁴)₁", radius=6.0, v1=(1.0, 0.0), v2=(0.5, 2.0 + _H),
        motif=[(0.0, 0.0, 4, 45.0), (0.0, 1.0, 4, 45.0),
               (0.0, 1.5 + _H / 3, 3, 90.0), (0.5, 1.5 + 2 * _H / 3, 3, 270.0)],
    ),
}

NAMES = list(_SPECS2)


def _polys_for(spec, span):
    v1 = np.asarray(spec["v1"], dtype=np.float64)
    v2 = np.asarray(spec["v2"], dtype=np.float64)
    reach = int(math.ceil(span / min(np.linalg.norm(v1), np.linalg.norm(v2)))) + 2
    polys = []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            sh = i * v1 + j * v2
            for (dx, dy, n, rot) in spec["motif"]:
                cx, cy = dx + sh[0], dy + sh[1]
                if cx * cx + cy * cy <= span * span:
                    polys.append(_reg(cx, cy, n, rot))
    return polys


def generate(name: str, *, radius: float | None = None, seed: int = 0) -> BoardGraph:
    """Compile a 2-uniform tiling patch into a validated intersection BoardGraph."""
    if name not in _SPECS2:
        raise ValueError(f"unknown 2-uniform tiling {name!r}; expected one of {NAMES}")
    spec = _SPECS2[name]
    r = float(radius) if radius is not None else float(spec["radius"])
    coords, edges = build_vertex_graph(_polys_for(spec, r + 4.0), qeps=_QEPS)
    coords, edges = patchmod.clip_to_disc(coords, edges, r)
    coords, edges = patchmod.prune_min_degree(coords, edges)
    coords, edges = patchmod.largest_connected_component(coords, edges)
    bg = BoardGraph(name=f"{name}_r{r:g}_s{seed}", num_nodes=coords.shape[0], edges=edges,
                    coords=coords.astype(np.float32),
                    meta={"family": name, "vconf": spec["label"], "radius_requested": r, "k_uniform": 2})
    validate(bg)
    return bg
