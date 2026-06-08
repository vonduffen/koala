"""BoardGraph: the geometry-blind adjacency structure that the whole engine runs on.

Stones are played on the **vertices (intersections)** of a tiling, and two intersections are
adjacent iff a tiling edge joins them — i.e. the board is the tiling's *1-skeleton* graph,
exactly like a traditional goban (lines, with stones on the crossings).

NOTE (divergence from ARCHITECTURE.md §3.1): the spec's DECIDED choice was to play on *faces
(cells)* of the tiling (the dual graph). That was overridden in favour of the intersection
model — Go is played on intersections. For the square tiling the two are identical (it is
self-dual); for other tilings they differ, and intersections are the canonical Go model.

The rules engine, search and neural network see only ``num_nodes`` and ``edges``; ``coords``
are vertex positions for RENDERING ONLY and must never leak into game logic.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# §3.1 invariant bound. Intersection degrees are small: 3 for most uniform tilings, 4 for the
# square grid, 6 for the triangular tiling. Boundary vertices are lower (pruned below 2).
MIN_DEGREE = 2
MAX_DEGREE = 12


class BoardGraphError(ValueError):
    """Raised when a BoardGraph violates a §3.1 invariant."""


def canonical_edges(edges: np.ndarray) -> np.ndarray:
    """Return edges as a deterministic [E, 2] int32 array.

    Each undirected pair is stored once as ``(min, max)``; rows are sorted lexicographically;
    duplicates are dropped. This makes graph equality and golden comparisons stable regardless
    of the order in which a generator happened to emit edges.
    """
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if e.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)  # sorts rows, removes duplicates
    return pairs.astype(np.int32)


def build_vertex_graph(polygons, *, qeps: float = 1e-4):
    """Derive the tiling's vertex/edge graph from a soup of face polygons.

    Each polygon contributes its corners as vertices and its sides as edges; vertices shared by
    several polygons collapse to one node (matched by quantizing coordinates to ``qeps``). This
    is how a tiling's 1-skeleton — the thing you actually play Go on — is built from the same
    polygons the compiler already generates.

    Returns ``(coords[N, 2] float32, edges[E, 2] int32)``.
    """
    index: dict[tuple[int, int], int] = {}
    coords: list[tuple[float, float]] = []

    def vid(pt) -> int:
        key = (int(round(pt[0] / qeps)), int(round(pt[1] / qeps)))
        idx = index.get(key)
        if idx is None:
            idx = len(coords)
            index[key] = idx
            coords.append((float(pt[0]), float(pt[1])))
        return idx

    eset: set[tuple[int, int]] = set()
    for poly in polygons:
        poly = np.asarray(poly)
        ids = [vid(poly[i]) for i in range(len(poly))]
        m = len(ids)
        for i in range(m):
            a, b = ids[i], ids[(i + 1) % m]
            if a != b:
                eset.add((a, b) if a < b else (b, a))

    coords_arr = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    edges = canonical_edges(np.asarray(list(eset), dtype=np.int64)) if eset else \
        np.zeros((0, 2), dtype=np.int32)
    return coords_arr, edges


@dataclass(frozen=True)
class BoardGraph:
    """The intersection graph of a tiling patch (its 1-skeleton).

    Attributes:
        name: stable identifier, e.g. ``"square_c81_s0"`` or ``"penrose_r6_seed3"``.
        num_nodes: N, the number of intersections (playable points).
        edges: [E, 2] int32, undirected, each pair once as (min, max), rows sorted.
        coords: [N, 2] float32 vertex positions — RENDERING ONLY.
        meta: tiling family, generation params, seed.
    """

    name: str
    num_nodes: int
    edges: np.ndarray
    coords: np.ndarray
    meta: dict = field(default_factory=dict)

    # ---- adjacency helpers (graph-only; safe for game logic) ----------------------------

    def neighbors(self) -> list[list[int]]:
        """Adjacency list: ``out[v]`` is the sorted list of v's neighbors."""
        adj: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for a, b in self.edges:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
        for nb in adj:
            nb.sort()
        return adj

    def degree(self) -> np.ndarray:
        """Per-node degree as int32[N]."""
        deg = np.zeros(self.num_nodes, dtype=np.int32)
        if self.edges.shape[0]:
            np.add.at(deg, self.edges[:, 0], 1)
            np.add.at(deg, self.edges[:, 1], 1)
        return deg

    def distance_to_boundary(self) -> np.ndarray:
        """Normalized graph BFS depth from the boundary, float32[N] in [0, 1].

        Boundary nodes are those with degree below the graph's maximum degree — intersections
        missing a neighbor because they sit at the patch edge. A multi-source BFS from the
        boundary gives each node its hop-distance to the nearest boundary, divided by the max
        depth. This is the ``distance_to_boundary`` static feature of §5.2.
        """
        deg = self.degree()
        if self.num_nodes == 0:
            return np.zeros(0, dtype=np.float32)
        max_deg = int(deg.max())
        boundary = np.where(deg < max_deg)[0]
        dist = np.full(self.num_nodes, -1, dtype=np.int64)
        adj = self.neighbors()
        dq: deque[int] = deque()
        sources = boundary if boundary.size else np.arange(self.num_nodes)
        for s in sources:
            dist[s] = 0
            dq.append(int(s))
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    dq.append(w)
        max_d = int(dist.max())
        if max_d <= 0:
            return np.zeros(self.num_nodes, dtype=np.float32)
        return (dist / max_d).astype(np.float32)

    # ---- serialization ------------------------------------------------------------------

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Write ``<path>.npz`` (arrays) and ``<path>.json`` (sidecar metadata)."""
        stem = Path(path)
        if stem.suffix in (".npz", ".json"):
            stem = stem.with_suffix("")
        npz_path = stem.with_suffix(".npz")
        json_path = stem.with_suffix(".json")
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(npz_path, edges=self.edges.astype(np.int32), coords=self.coords.astype(np.float32))
        sidecar = {
            "name": self.name,
            "num_nodes": int(self.num_nodes),
            "num_edges": int(self.edges.shape[0]),
            "meta": _jsonable(self.meta),
        }
        json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
        return npz_path, json_path

    @classmethod
    def load(cls, path: str | Path) -> BoardGraph:
        """Load a BoardGraph previously written by :meth:`save`."""
        stem = Path(path)
        if stem.suffix in (".npz", ".json"):
            stem = stem.with_suffix("")
        with np.load(stem.with_suffix(".npz")) as data:
            edges = data["edges"].astype(np.int32)
            coords = data["coords"].astype(np.float32)
        sidecar = json.loads(stem.with_suffix(".json").read_text())
        return cls(
            name=sidecar["name"],
            num_nodes=int(sidecar["num_nodes"]),
            edges=edges,
            coords=coords,
            meta=sidecar.get("meta", {}),
        )


def _jsonable(obj):
    """Best-effort conversion of meta values (numpy scalars/arrays) to JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def validate(bg: BoardGraph, *, min_nodes: int | None = None, max_nodes: int | None = None) -> None:
    """Enforce the §3.1 invariants; raise :class:`BoardGraphError` on the first violation."""
    n = bg.num_nodes
    if n <= 0:
        raise BoardGraphError(f"num_nodes must be positive, got {n}")
    if bg.coords.shape != (n, 2):
        raise BoardGraphError(f"coords must be [{n}, 2], got {bg.coords.shape}")

    e = bg.edges
    if e.ndim != 2 or e.shape[1] != 2:
        raise BoardGraphError(f"edges must be [E, 2], got shape {e.shape}")
    if e.shape[0] == 0:
        raise BoardGraphError("graph has no edges (cannot be connected for N>=2)")
    if e.min() < 0 or e.max() >= n:
        raise BoardGraphError("edge references a node index out of range")
    if np.any(e[:, 0] == e[:, 1]):
        raise BoardGraphError("graph contains a self-loop")

    canon = canonical_edges(e)
    if canon.shape[0] != e.shape[0]:
        raise BoardGraphError("graph contains duplicate edges")

    deg = bg.degree()
    bad = np.where((deg < MIN_DEGREE) | (deg > MAX_DEGREE))[0]
    if bad.size:
        v = int(bad[0])
        raise BoardGraphError(
            f"degree(node {v}) = {int(deg[v])} outside [{MIN_DEGREE}, {MAX_DEGREE}]")

    if not _is_connected(n, canon):
        raise BoardGraphError("graph is not connected")

    if min_nodes is not None and n < min_nodes:
        raise BoardGraphError(f"N={n} below minimum {min_nodes}")
    if max_nodes is not None and n > max_nodes:
        raise BoardGraphError(f"N={n} above maximum {max_nodes}")


def _is_connected(n: int, edges: np.ndarray) -> bool:
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen = np.zeros(n, dtype=bool)
    dq = deque([0])
    seen[0] = True
    count = 1
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if not seen[w]:
                seen[w] = True
                count += 1
                dq.append(w)
    return count == n
