"""Penrose (P3 rhombus) aperiodic tiling via the de Bruijn pentagrid (ARCHITECTURE.md §3.2).

de Bruijn's construction: five families of equally-spaced parallel lines, family ``j`` with
normal direction ``e_j = (cos 2πj/5, sin 2πj/5)`` and an offset ``γ_j``. Every intersection of a
line from family ``r`` with one from family ``s`` maps to a rhombus whose four corners are
``Σ_j K_j e_j`` over the integer pentagrid coordinates ``K`` of the four surrounding meshes. The
result is a Penrose tiling of thick (72°) and thin (36°) rhombs, all edges length 1.

As everywhere else, we play on the tiling's **intersections**: the rhomb corners become vertices
and the rhomb sides become edges (via ``build_vertex_graph``), then we clip to a disc. With equal
offsets (``γ_j = 1/5``, sum = 1) the patch is the famous 5-fold symmetric "sun" Penrose; a seed
gives a generic non-symmetric patch. Either is genuinely aperiodic — the hardest held-out test
for the universal net.
"""

from __future__ import annotations

import math

import numpy as np

from .boardgraph import BoardGraph, build_vertex_graph, validate
from . import patch as patchmod

_E = np.array([[math.cos(2 * math.pi * j / 5), math.sin(2 * math.pi * j / 5)] for j in range(5)])


def _offsets(symmetric: bool, seed: int) -> np.ndarray:
    """Pentagrid offsets γ_j. Equal → 5-fold symmetric (sum 1); seeded → generic (sum 0)."""
    if symmetric:
        return np.full(5, 0.2)
    rng = np.random.default_rng(seed)
    g = rng.uniform(-0.4, 0.4, size=5)
    return g - g.mean()  # sum 0 → a valid (generic) Penrose


def _rhombs(gamma: np.ndarray, line_range: int) -> list[np.ndarray]:
    """All Penrose rhombs whose pentagrid lines fall within ``±line_range``."""
    rhombs: list[np.ndarray] = []
    for r in range(5):
        for s in range(r + 1, 5):
            det = _E[r, 0] * _E[s, 1] - _E[r, 1] * _E[s, 0]
            if abs(det) < 1e-9:
                continue
            for kr in range(-line_range, line_range + 1):
                ar = kr - gamma[r]
                for ks in range(-line_range, line_range + 1):
                    as_ = ks - gamma[s]
                    # intersection p of grid lines (r=kr, s=ks): e_r·p = ar, e_s·p = as_
                    px = (ar * _E[s, 1] - as_ * _E[r, 1]) / det
                    py = (as_ * _E[r, 0] - ar * _E[s, 0]) / det
                    # pentagrid index of the mesh at p; crossing line r=kr switches K_r between
                    # kr-1 and kr, so the rhomb spans {kr-1, kr} x {ks-1, ks}.
                    k = np.floor(_E @ (px, py) + gamma).astype(np.int64)
                    k[r], k[s] = kr - 1, ks - 1
                    # the four surrounding meshes → four rhomb corners t(K) = Σ K_j e_j
                    corners = []
                    for dr, ds in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        kk = k.copy()
                        kk[r] += dr
                        kk[s] += ds
                        corners.append(kk @ _E)
                    rhombs.append(np.array(corners))
    return rhombs


def generate(*, radius: float = 6.0, seed: int = 0, symmetric: bool = True) -> BoardGraph:
    """Compile a Penrose rhombus patch into a validated intersection :class:`BoardGraph`."""
    gamma = _offsets(symmetric, seed)
    # rhomb corners have magnitude ~ |K|·|e| ; generate a generous line range to cover the disc.
    line_range = int(math.ceil(radius)) + 3
    coords, edges = build_vertex_graph(_rhombs(gamma, line_range), qeps=1e-3)
    coords, edges = patchmod.clip_to_disc(coords, edges, radius)
    coords, edges = patchmod.prune_min_degree(coords, edges)
    coords, edges = patchmod.largest_connected_component(coords, edges)

    kind = "sun" if symmetric else f"s{seed}"
    bg = BoardGraph(
        name=f"penrose_{kind}_r{radius:g}",
        num_nodes=coords.shape[0],
        edges=edges,
        coords=coords.astype(np.float32),
        meta={"family": "penrose", "radius_requested": radius, "seed": int(seed),
              "symmetric": bool(symmetric), "vconf": "aperiodic"},
    )
    validate(bg)
    return bg
