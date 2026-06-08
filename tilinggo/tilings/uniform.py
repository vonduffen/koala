"""Uniform (Archimedean) Euclidean tilings — the 11 vertex-transitive tilings.

This covers the full list at https://en.wikipedia.org/wiki/List_of_Euclidean_uniform_tilings:
the 3 regular tilings plus the 8 semiregular (Archimedean) ones. Per ARCHITECTURE.md §3.2
these are Milestone 3 territory; they are geometry only (Component 1) and feed the same
``BoardGraph`` the engine consumes.

Construction strategy (general, so adjacency need not be hand-coded per family):
  1. each tiling is described by two lattice translation vectors and a *motif* — the polygons
     in one fundamental domain, given as (center_x, center_y, n_sides, rotation_degrees);
  2. the motif is stamped across a block of lattice translations covering the target disc;
  3. duplicate faces (same centroid) are dropped;
  4. two faces are adjacent iff they share an edge — detected by quantizing vertex coordinates
     and matching the unordered endpoint pair of each polygon edge. If a tiling's geometry is
     even slightly inconsistent (gaps/overlaps), shared edges fail to match and the resulting
     graph is disconnected — a built-in correctness check surfaced by ``validate``.

All tilings use edge length 1.
"""

from __future__ import annotations

import math

import numpy as np

from .boardgraph import BoardGraph, build_vertex_graph, validate
from . import patch as patchmod

_S2 = math.sqrt(2.0)
_S3 = math.sqrt(3.0)

# Quantization for vertex matching: edges have length 1, so 1e-3 cleanly separates distinct
# vertices while absorbing floating-point noise from independent polygon constructions.
_QEPS = 1e-3


def _reg(cx: float, cy: float, n: int, rot_deg: float) -> np.ndarray:
    """Regular ``n``-gon of edge length 1, centered at (cx, cy), first vertex at ``rot_deg``."""
    R = 1.0 / (2.0 * math.sin(math.pi / n))
    ang = np.deg2rad(rot_deg + 360.0 / n * np.arange(n))
    return np.stack([cx + R * np.cos(ang), cy + R * np.sin(ang)], axis=1).astype(np.float64)


# --------------------------------------------------------------------------------------------
# Tiling specifications.
#
# Each entry maps a name to a dict with:
#   v1, v2 : lattice translation vectors (np arrays)
#   motif  : list of (dx, dy, n_sides, rot_deg) polygons in the fundamental domain
#   radius : a sensible default patch radius for the demo
#   vconf  : the vertex configuration string (documentation)
# --------------------------------------------------------------------------------------------

def _specs() -> dict[str, dict]:
    specs: dict[str, dict] = {}

    # ---- regular tilings (also producible via periodic.py; included for completeness) ------
    specs["square"] = dict(
        vconf="4.4.4.4", v1=(1.0, 0.0), v2=(0.0, 1.0),
        motif=[(0.0, 0.0, 4, 45.0)], radius=5.0,
    )
    specs["triangular"] = dict(
        vconf="3.3.3.3.3.3", v1=(1.0, 0.0), v2=(0.5, _S3 / 2),
        motif=[(0.5, _S3 / 6, 3, 90.0), (1.0, _S3 / 3, 3, 270.0)], radius=5.0,
    )
    specs["hexagonal"] = dict(
        vconf="6.6.6", v1=(_S3, 0.0), v2=(_S3 / 2, 1.5),
        motif=[(0.0, 0.0, 6, 30.0)], radius=6.0,
    )

    # ---- truncated square 4.8.8: octagons + squares -----------------------------------------
    d = 1.0 + _S2
    specs["trunc_square"] = dict(
        vconf="4.8.8", v1=(d, 0.0), v2=(0.0, d),
        motif=[(0.0, 0.0, 8, 22.5), (d / 2, d / 2, 4, 0.0)], radius=7.0,
    )

    # ---- truncated hexagonal 3.12.12: dodecagons + triangles ---------------------------------
    D = 2.0 + _S3
    specs["trunc_hex"] = dict(
        vconf="3.12.12", v1=(D, 0.0), v2=(D / 2, D * _S3 / 2),
        motif=[
            (0.0, 0.0, 12, 15.0),
            (D / 2, D * _S3 / 6, 3, 270.0),
            (D, D * _S3 / 3, 3, 90.0),
        ],
        radius=9.0,
    )

    # ---- trihexagonal 3.6.3.6 (kagome): hexagons + triangles ---------------------------------
    specs["trihexagonal"] = dict(
        vconf="3.6.3.6", v1=(2.0, 0.0), v2=(1.0, _S3),
        motif=[
            (0.0, 0.0, 6, 0.0),
            (1.0, _S3 / 3, 3, 270.0),
            (2.0, 2.0 * _S3 / 3, 3, 90.0),
        ],
        radius=6.0,
    )

    # ---- elongated triangular 3.3.3.4.4: rows of squares + rows of triangles -----------------
    h = _S3 / 2
    period_y = 1.0 + h
    specs["elongated_tri"] = dict(
        vconf="3.3.3.4.4", v1=(1.0, 0.0), v2=(0.5, period_y),
        motif=[
            (0.0, 0.0, 4, 45.0),
            (0.0, 0.5 + h / 3.0, 3, 90.0),
            (0.5, 0.5 + 2.0 * h / 3.0, 3, 270.0),
        ],
        radius=6.0,
    )

    # ---- rhombitrihexagonal 3.4.6.4: hexagons + squares + triangles --------------------------
    Dr = 1.0 + _S3
    specs["rhombitrihex"] = dict(
        vconf="3.4.6.4",
        v1=(Dr * _S3 / 2, Dr / 2), v2=(0.0, Dr),
        motif=_rhombitrihex_motif(), radius=8.0,
    )

    # ---- truncated trihexagonal 4.6.12: dodecagons + hexagons + squares ----------------------
    Dt = 3.0 + _S3
    specs["trunc_trihex"] = dict(
        vconf="4.6.12", v1=(Dt, 0.0), v2=(Dt / 2, Dt * _S3 / 2),
        motif=_trunc_trihex_motif(Dt), radius=12.0,
    )

    # ---- snub square 3.3.4.3.4: squares + triangles (chiral) ---------------------------------
    specs["snub_square"] = dict(
        vconf="3.3.4.3.4", **_snub_square_lattice(), radius=6.0,
    )

    # ---- snub hexagonal 3.3.3.3.6: hexagons + triangles (chiral) -----------------------------
    specs["snub_hex"] = dict(
        vconf="3.3.3.3.6", **_snub_hex_lattice(), radius=7.0,
    )

    return specs


def _rhombitrihex_motif():
    """Hexagon at origin (rot 0), a square on each of its 6 edges, a triangle on each vertex."""
    motif = [(0.0, 0.0, 6, 0.0)]
    ap = _S3 / 2          # hexagon apothem; edge midpoints at angles 30,90,150,...
    sq_dist = ap + 0.5    # hexagon apothem + half a unit square
    for k in range(6):
        a = 30.0 + 60.0 * k
        ar = math.radians(a)
        motif.append((sq_dist * math.cos(ar), sq_dist * math.sin(ar), 4, a - 45.0))
    # triangles share a vertex with the hexagon (at R6=1, angles 0,60,...) and point inward.
    tri_dist = 1.0 + 1.0 / _S3  # hexagon vertex radius + triangle circumradius (apex at vertex)
    for k in range(6):
        a = 60.0 * k
        ar = math.radians(a)
        motif.append((tri_dist * math.cos(ar), tri_dist * math.sin(ar), 3, a + 180.0))
    return motif


def _trunc_trihex_motif(D):
    """Dodecagon at origin, with hexagons in the triangular holes and squares between dodecagons."""
    ap12 = 1.0 / (2.0 * math.tan(math.pi / 12))  # dodecagon apothem
    motif = [(0.0, 0.0, 12, -15.0)]  # edges face 0,30,60,... so squares sit along 0,60,...
    # hexagons at the two triangular-hole centroids of the dodecagon lattice (like trunc_hex)
    for (hx, hy) in [(D / 2, D * _S3 / 6), (D, D * _S3 / 3)]:
        motif.append((hx, hy, 6, 0.0))
    # squares between adjacent dodecagons, along the 6 directions 0,60,...
    sq_dist = ap12 + 0.5
    for k in range(6):
        a = 60.0 * k
        ar = math.radians(a)
        motif.append((sq_dist * math.cos(ar), sq_dist * math.sin(ar), 4, a - 45.0))
    return motif


def _snub_square_lattice():
    """Snub square 3.3.4.3.4 (chiral): squares in two orientations 15 deg apart, plus triangles.

    An axis-aligned square A at the origin has a triangle on each edge; each edge-triangle's
    side edge is shared with a second square B rotated -15 deg. Four B squares sit around A at
    radius |B|. The translation lattice mapping A to the next same-orientation square is square,
    spacing 1+sqrt(3), oriented at 60 deg.
    """
    tdist = 0.5 + (_S3 / 2) / 3.0          # square center -> edge-triangle centroid
    bdist = math.hypot(0.6830127, 1.1830127)  # |B - A|, from the shared-edge construction

    motif: list[tuple] = [(0.0, 0.0, 4, 45.0)]  # square A, axis-aligned
    for d in (0.0, 90.0, 180.0, 270.0):         # A's edge-triangles, apex outward
        dr = math.radians(d)
        motif.append((tdist * math.cos(dr), tdist * math.sin(dr), 3, d))

    for a in (60.0, 150.0, 240.0, 330.0):       # the four B squares + their edge-triangles
        ar = math.radians(a)
        bx, by = bdist * math.cos(ar), bdist * math.sin(ar)
        motif.append((bx, by, 4, -15.0))
        for d in (30.0, 120.0, 210.0, 300.0):   # B's edge normals = -15 + 45 + 90k
            dr = math.radians(d)
            motif.append((bx + tdist * math.cos(dr), by + tdist * math.sin(dr), 3, d))

    # primitive translation lattice: spacing sqrt(2+sqrt3) ~ 1.93185, at 15 and 105 degrees.
    return dict(v1=(1.8660254, 0.5), v2=(-0.5, 1.8660254), motif=motif)


def _snub_hex_polys(span):
    """Snub hexagonal 3.3.3.3.6 (chiral).

    Built from the unit triangular tiling: the 6 triangles around each vertex of the index-7
    (2,1) sublattice merge into a hexagon; every triangle not touching such a vertex stays.
    The chirality is inherent in choosing the (2,1) sublattice (rotated ~19.1 deg), so no
    explicit rotation is needed. Per √7 cell this yields 1 hexagon + 8 triangles, the correct
    3.3.3.3.6 ratio.
    """
    def pos(i, j):
        return np.array([i + 0.5 * j, j * _S3 / 2], dtype=np.float64)

    reach = int(math.ceil(span)) + 3
    in_hex = lambda i, j: (3 * i + j) % 7 == 0  # the index-7 sublattice of hexagon centers

    # CCW neighbor directions in (i, j) lattice coordinates, at 0,60,...,300 degrees.
    hex_dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    polys: list[np.ndarray] = []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            if in_hex(i, j):
                polys.append(np.array([pos(i + di, j + dj) for di, dj in hex_dirs]))
            # the two triangles of cell (i, j); keep only those with no hexagon-center vertex
            up = [(i, j), (i + 1, j), (i, j + 1)]
            if not any(in_hex(a, b) for a, b in up):
                polys.append(np.array([pos(a, b) for a, b in up]))
            dn = [(i + 1, j), (i, j + 1), (i + 1, j + 1)]
            if not any(in_hex(a, b) for a, b in dn):
                polys.append(np.array([pos(a, b) for a, b in dn]))

    return [p for p in polys if p.mean(axis=0) @ p.mean(axis=0) <= span * span]


def _snub_hex_lattice():
    """Snub hexagonal 3.3.3.3.6, built directly from polygons (see :func:`_snub_hex_polys`)."""
    return dict(polys_fn=_snub_hex_polys)


_SPECS = _specs()
TILING_NAMES = list(_SPECS.keys())


def _polys_for(spec, span):
    """The face-polygon soup for a tiling, covering a disc of ``span`` with margin."""
    if "polys_fn" in spec:
        # Tilings built directly from polygons (e.g. the chiral snubs) rather than a motif.
        return spec["polys_fn"](span)
    v1 = np.asarray(spec["v1"], dtype=np.float64)
    v2 = np.asarray(spec["v2"], dtype=np.float64)
    reach = int(math.ceil(span / min(np.linalg.norm(v1), np.linalg.norm(v2)))) + 2
    polys = []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            shift = i * v1 + j * v2
            for (dx, dy, n, rot) in spec["motif"]:
                cx, cy = dx + shift[0], dy + shift[1]
                if cx * cx + cy * cy <= span * span:
                    polys.append(_reg(cx, cy, n, rot))
    return polys


def generate(name: str, *, radius: float | None = None, seed: int = 0) -> BoardGraph:
    """Compile a uniform tiling patch into a validated :class:`BoardGraph` of intersections."""
    if name not in _SPECS:
        raise ValueError(f"unknown tiling {name!r}; expected one of {TILING_NAMES}")
    spec = _SPECS[name]
    r = float(radius) if radius is not None else float(spec["radius"])

    # Build the tiling's vertex graph from the face-polygon soup, then clip the vertices to the
    # disc (margin so boundary vertices keep their edges), prune pendants, take the main blob.
    coords, edges = build_vertex_graph(_polys_for(spec, r + 4.0), qeps=_QEPS)
    coords, edges = patchmod.clip_to_disc(coords, edges, r)
    coords, edges = patchmod.prune_min_degree(coords, edges)
    coords, edges = patchmod.largest_connected_component(coords, edges)

    bg = BoardGraph(
        name=f"{name}_r{r:g}_s{seed}",
        num_nodes=coords.shape[0],
        edges=edges,
        coords=coords.astype(np.float32),
        meta={"family": name, "vconf": spec["vconf"], "radius_requested": r, "seed": int(seed)},
    )
    validate(bg)
    return bg
