"""Detect a board's geometric symmetry group as vertex permutations (graph automorphisms).

Although we play on irregularly-clipped patches, the patches are *not* asymmetric: clipping a
tiling with an origin-centered disc inherits the site-symmetry of the centre. So every board
has a nontrivial symmetry group — **dihedral** (rotations + mirrors) for achiral tilings,
**cyclic** (rotations only) for the chiral snubs. Each symmetry is an exact graph automorphism
(it permutes vertices and preserves adjacency), which is precisely the data-augmentation lever
AlphaZero engines use (cf. KataGo / the prior Delta-Go engine's 12× D6 augmentation).

This detector is geometric and tiling-agnostic: it tries rotations and reflections about the
patch centroid and keeps those that map the vertex set onto itself. It works for any patch,
including aperiodic (Penrose) ones later. The returned permutations include the identity.
"""

from __future__ import annotations

import numpy as np

from .boardgraph import BoardGraph

# Rotation orders worth trying: covers square(4), hex/tri(6), dodecagonal(12), and Penrose(5,10).
_CANDIDATE_ORDERS = (12, 10, 8, 6, 5, 4, 3, 2)


def _index_map(points: np.ndarray, qeps: float) -> dict[tuple[int, int], int]:
    return {(int(round(x / qeps)), int(round(y / qeps))): i for i, (x, y) in enumerate(points)}


def _permutation(base: np.ndarray, moved: np.ndarray, index, qeps: float):
    """If ``moved`` is a relabelling of ``base``, return the permutation; else None."""
    perm = np.empty(base.shape[0], dtype=np.int64)
    for i, (x, y) in enumerate(moved):
        j = index.get((int(round(x / qeps)), int(round(y / qeps))))
        if j is None:
            return None
        perm[i] = j
    if np.unique(perm).shape[0] != perm.shape[0]:
        return None  # not a bijection
    return perm


def symmetries(bg: BoardGraph, *, qeps: float = 1e-2,
               include_reflections: bool = True) -> list[np.ndarray]:
    """Return the board's symmetry automorphisms as vertex permutations (identity first).

    A permutation ``p`` means vertex ``i`` maps to vertex ``p[i]`` under the symmetry. ``qeps``
    is the position-matching tolerance (vertices are ≥ ~1 apart, so 1e-2 is safe against the
    floating-point error rotations introduce).
    """
    pts = np.asarray(bg.coords, dtype=np.float64)
    p = pts - pts.mean(axis=0)  # centre on the patch centroid (= the symmetry centre if any)
    index = _index_map(p, qeps)

    perms: list[np.ndarray] = []
    seen: set[bytes] = set()

    def add(perm):
        if perm is not None and perm.tobytes() not in seen:
            seen.add(perm.tobytes())
            perms.append(perm)

    add(np.arange(p.shape[0], dtype=np.int64))  # identity

    angles = {2.0 * np.pi * k / n for n in _CANDIDATE_ORDERS for k in range(1, n)}
    for a in angles:
        ca, sa = np.cos(a), np.sin(a)
        rot = p @ np.array([[ca, sa], [-sa, ca]])  # rotate by a (note: p @ R^T)
        add(_permutation(p, rot, index, qeps))

    if include_reflections:
        # A mirror axis through the centroid that maps vertex i onto vertex j (same radius) lies
        # along their angle bisector (θ_i+θ_j)/2; for i==j it runs through the vertex itself.
        # Enumerating these bisectors yields every candidate axis exactly — including ones that
        # pass between vertices (through edge midpoints), which a vertex-only scan would miss.
        radii = np.hypot(p[:, 0], p[:, 1])
        theta = np.arctan2(p[:, 1], p[:, 0])
        axis: set[float] = set()
        for i in range(p.shape[0]):
            if radii[i] <= 1e-9:
                continue
            same = np.where(np.abs(radii - radii[i]) < qeps)[0]
            for j in same:
                axis.add(round(((theta[i] + theta[j]) / 2.0) % np.pi, 5))
        for phi in axis:
            c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
            refl = p @ np.array([[c2, s2], [s2, -c2]])  # reflection about axis at angle phi
            add(_permutation(p, refl, index, qeps))

    return perms


def is_automorphism(bg: BoardGraph, perm: np.ndarray) -> bool:
    """True iff applying ``perm`` to vertex labels maps the edge set onto itself."""
    from .boardgraph import canonical_edges

    mapped = canonical_edges(perm[bg.edges])
    return mapped.shape == bg.edges.shape and np.array_equal(mapped, bg.edges)


def split_by_orientation(bg: BoardGraph, perms=None):
    """Partition symmetries into (rotations, reflections) by whether they preserve orientation.

    A board is *chiral* iff it has no reflections — then ``symmetries`` already returns rotations
    only, so using its full output for data augmentation is automatically safe (a snub tiling is
    never mirror-augmented). Achiral boards get the full dihedral set.
    """
    if perms is None:
        perms = symmetries(bg)
    p = np.asarray(bg.coords, dtype=np.float64)

    def area(P, i, j, k):
        return (P[j, 0] - P[i, 0]) * (P[k, 1] - P[i, 1]) - (P[j, 1] - P[i, 1]) * (P[k, 0] - P[i, 0])

    a = next(k for k in range(1, p.shape[0]) if abs(p[k, 0] - p[0, 0]) + abs(p[k, 1] - p[0, 1]) > 0)
    b = next(k for k in range(1, p.shape[0]) if abs(area(p, 0, a, k)) > 1e-3)
    s0 = np.sign(area(p, 0, a, b))

    rotations, reflections = [], []
    for perm in perms:
        q = p[perm]
        (rotations if np.sign(area(q, 0, a, b)) == s0 else reflections).append(perm)
    return rotations, reflections


def is_chiral(bg: BoardGraph) -> bool:
    """True iff the board has no mirror symmetry (only rotations)."""
    return len(split_by_orientation(bg)[1]) == 0
