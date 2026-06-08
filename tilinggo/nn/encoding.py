"""Turn game states into the padded tensors the GNN consumes (ARCHITECTURE.md §5.1–5.2).

Two halves:
  * **Static board features** depend only on the BoardGraph (degree, distance-to-boundary,
    Laplacian-eigenvector positional encoding) and the neighbour-index tensor for message
    passing. Computed once per board and cached.
  * **Game features** (stones, liberties, recent moves, legality) plus broadcast global scalars
    are recomputed each evaluation from a GoState.

Batching pads every graph to ``N_max`` with a node mask, and message passing runs over a dense
neighbour-index tensor ``nbr_index[N_max, D_max]`` (padded entries blocked by ``nbr_mask``) —
pure gather/reduce/MLP ops, MPS-friendly, no sparse kernels.

Feature layout (F = 42 per node):
    stones own/opp/empty .................. 3
    liberties of this chain, 1..6+ ........ 6
    played in last k moves, k=1..5 ........ 5
    legal for side to move ................ 1
    degree one-hot 3..9+ .................. 7
    normalized distance-to-boundary ....... 1
    Laplacian PE (16 eigenvectors) ....... 16
    global komi / move# / player .......... 3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..rules.gostate import BLACK, EMPTY, WHITE, GoState
from ..tilings.boardgraph import BoardGraph

PE_DIM = 16
FEATURE_DIM = 3 + 6 + 5 + 1 + 7 + 1 + PE_DIM + 3  # = 42

# Column ranges (for the optional feature ablation below).
_COL_DEGREE = (15, 22)   # degree one-hot
_COL_DIST = (22, 23)     # normalized distance-to-boundary
_COL_PE = (23, 23 + PE_DIM)  # Laplacian positional encoding

# Feature-ablation toggle (default OFF — zero geometry cues to test whether substrate-invariance
# survives without them). Set via ``set_feature_ablation`` before training/eval; the in_dim stays
# 42 (zeroed columns), so no architecture/C++ changes are needed and existing tests are unaffected.
_FEATURE_ABLATION: set[str] = set()


def set_feature_ablation(names) -> None:
    """Ablate input feature groups by zeroing their columns. ``names`` ⊆ {'pe','dist','degree'}."""
    global _FEATURE_ABLATION
    valid = {"pe", "dist", "degree"}
    names = {n.strip() for n in (names.split(",") if isinstance(names, str) else names) if n.strip()}
    bad = names - valid
    if bad:
        raise ValueError(f"unknown ablation features {bad}; valid: {sorted(valid)}")
    _FEATURE_ABLATION = names


def feature_ablation() -> set:
    return set(_FEATURE_ABLATION)


# --------------------------------------------------------------------------------------------
# Static, per-board features (cached on the BoardGraph's meta-independent identity).
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class StaticBoard:
    n: int
    nbr_index: np.ndarray   # [N, D_max] int64, neighbour node ids (padded with 0)
    nbr_mask: np.ndarray    # [N, D_max] float32, 1 for a real neighbour
    degree_onehot: np.ndarray  # [N, 7] float32, degree 3..9+
    dist: np.ndarray        # [N, 1] float32, normalized distance to boundary
    pe: np.ndarray          # [N, PE_DIM] float32, Laplacian eigenvector PE


_STATIC_CACHE: dict[int, StaticBoard] = {}


def laplacian_pe(bg: BoardGraph, dim: int = PE_DIM) -> np.ndarray:
    """First ``dim`` non-trivial eigenvectors of the normalized graph Laplacian, as [N, dim].

    Uses the symmetric normalized Laplacian L = I − D^{-1/2} A D^{-1/2}; the smallest eigenvalue
    is ~0 with the (trivial) constant eigenvector, which is dropped. Signs are arbitrary — the
    training loop sign-randomizes them, which is the intended augmentation (§5.4). If the board
    has fewer than ``dim+1`` nodes the extra columns are zero-padded.
    """
    n = bg.num_nodes
    a = np.zeros((n, n), dtype=np.float64)
    for u, v in bg.edges:
        a[u, v] = a[v, u] = 1.0
    deg = a.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    lap = np.eye(n) - (dinv[:, None] * a * dinv[None, :])
    # symmetric → eigh gives ascending real eigenvalues and orthonormal eigenvectors
    _, vecs = np.linalg.eigh(lap)
    pe = np.zeros((n, dim), dtype=np.float32)
    take = min(dim, max(n - 1, 0))
    pe[:, :take] = vecs[:, 1:1 + take].astype(np.float32)
    return pe


def _degree_onehot(deg: np.ndarray) -> np.ndarray:
    """Degree one-hot over buckets 3,4,5,6,7,8,9+ (7 columns); low degrees fold into bucket 0."""
    out = np.zeros((deg.shape[0], 7), dtype=np.float32)
    bucket = np.clip(deg - 3, 0, 6)
    out[np.arange(deg.shape[0]), bucket] = 1.0
    return out


def _cache_key(bg: BoardGraph):
    """Content key (node count + edge list) — stable across object identity.

    Caching by ``id(bg)`` is unsafe: once a graph is garbage-collected Python reuses its id, so a
    freshly-loaded graph can collide with a stale cache entry of a different size. Static features
    depend only on the adjacency, so two graphs with identical edges share them safely.
    """
    return (int(bg.num_nodes), bg.edges.astype(np.int32).tobytes())


def static_board(bg: BoardGraph) -> StaticBoard:
    """Compute (and cache) the static features and neighbour-index tensor for ``bg``."""
    key = _cache_key(bg)
    cached = _STATIC_CACHE.get(key)
    if cached is not None:
        return cached

    adj = bg.neighbors()
    n = bg.num_nodes
    d_max = max((len(a) for a in adj), default=1)
    nbr_index = np.zeros((n, d_max), dtype=np.int64)
    nbr_mask = np.zeros((n, d_max), dtype=np.float32)
    for i, a in enumerate(adj):
        nbr_index[i, : len(a)] = a
        nbr_mask[i, : len(a)] = 1.0

    sb = StaticBoard(
        n=n,
        nbr_index=nbr_index,
        nbr_mask=nbr_mask,
        degree_onehot=_degree_onehot(bg.degree()),
        dist=bg.distance_to_boundary().reshape(n, 1).astype(np.float32),
        pe=laplacian_pe(bg),
    )
    _STATIC_CACHE[key] = sb
    return sb


# --------------------------------------------------------------------------------------------
# Game features from a GoState.
# --------------------------------------------------------------------------------------------

def _per_node_liberties(state: GoState) -> np.ndarray:
    """Liberty count of each occupied node's chain (0 for empty nodes), as int [N]."""
    colors = state.colors
    adj = state.board.adj
    libs = np.zeros(state.board.n, dtype=np.int64)
    seen = np.zeros(state.board.n, dtype=bool)
    for v in range(state.board.n):
        if colors[v] == EMPTY or seen[v]:
            continue
        chain, liberties = state._group_and_liberties(colors, v)
        count = len(liberties)
        for c in chain:
            libs[c] = count
            seen[c] = True
    return libs


def game_features(state: GoState, recent_moves: list[int] | None = None) -> np.ndarray:
    """Per-node game features [N, 15]: stones(3) + liberties(6) + last-k(5) + legal(1)."""
    n = state.board.n
    colors = state.colors
    me, opp = state.to_move, (WHITE if state.to_move == BLACK else BLACK)
    f = np.zeros((n, 15), dtype=np.float32)

    f[colors == me, 0] = 1.0
    f[colors == opp, 1] = 1.0
    f[colors == EMPTY, 2] = 1.0

    libs = _per_node_liberties(state)
    occupied = colors != EMPTY
    bucket = np.clip(libs[occupied] - 1, 0, 5)
    f[np.where(occupied)[0], 3 + bucket] = 1.0  # liberties one-hot 1..6+

    for k, node in enumerate((recent_moves or [])[:5]):
        if node is not None and 0 <= node < n:
            f[node, 9 + k] = 1.0  # played k+1 moves ago

    f[:, 14] = state.legal_moves()[:n].astype(np.float32)
    return f


# --------------------------------------------------------------------------------------------
# Batch assembly → torch tensors.
# --------------------------------------------------------------------------------------------

@dataclass
class EncodedBatch:
    x: torch.Tensor            # [B, N_max, F] node features
    nbr_index: torch.Tensor    # [B, N_max, D_max] long, neighbour ids
    nbr_mask: torch.Tensor     # [B, N_max, D_max] float
    node_mask: torch.Tensor    # [B, N_max] float, 1 for real nodes
    legal: torch.Tensor        # [B, N_max + 1] bool, legal moves incl. pass (last column)

    def to(self, device) -> "EncodedBatch":
        return EncodedBatch(*(t.to(device) for t in
                              (self.x, self.nbr_index, self.nbr_mask, self.node_mask, self.legal)))


def encode_states(states: list[GoState],
                  recent_moves: list[list[int]] | None = None) -> EncodedBatch:
    """Encode a batch of GoStates into padded tensors. Graphs may have different sizes."""
    b = len(states)
    n_max = max(s.board.n for s in states)
    d_max = max(static_board(s.board.graph).nbr_index.shape[1] for s in states)

    x = np.zeros((b, n_max, FEATURE_DIM), dtype=np.float32)
    nbr_index = np.zeros((b, n_max, d_max), dtype=np.int64)
    nbr_mask = np.zeros((b, n_max, d_max), dtype=np.float32)
    node_mask = np.zeros((b, n_max), dtype=np.float32)
    legal = np.zeros((b, n_max + 1), dtype=bool)

    for bi, state in enumerate(states):
        sb = static_board(state.board.graph)
        n = sb.n
        gf = game_features(state, recent_moves[bi] if recent_moves else None)
        # global scalars broadcast to every node
        glob = np.array([state.board.komi / 15.0,
                         state.move_num / max(n, 1),
                         1.0 if state.to_move == BLACK else -1.0], dtype=np.float32)
        x[bi, :n, 0:15] = gf
        x[bi, :n, 15:22] = sb.degree_onehot
        x[bi, :n, 22:23] = sb.dist
        x[bi, :n, 23:23 + PE_DIM] = sb.pe
        x[bi, :n, 23 + PE_DIM:] = glob

        dn = sb.nbr_index.shape[1]
        nbr_index[bi, :n, :dn] = sb.nbr_index
        nbr_mask[bi, :n, :dn] = sb.nbr_mask
        node_mask[bi, :n] = 1.0
        lm = state.legal_moves()
        legal[bi, :n] = lm[:n]
        legal[bi, n_max] = lm[n]  # pass slot lives at the padded end

    if _FEATURE_ABLATION:  # zero ablated geometry-cue columns (default: no-op)
        if "pe" in _FEATURE_ABLATION:
            x[:, :, _COL_PE[0]:_COL_PE[1]] = 0.0
        if "dist" in _FEATURE_ABLATION:
            x[:, :, _COL_DIST[0]:_COL_DIST[1]] = 0.0
        if "degree" in _FEATURE_ABLATION:
            x[:, :, _COL_DEGREE[0]:_COL_DEGREE[1]] = 0.0

    return EncodedBatch(
        x=torch.from_numpy(x),
        nbr_index=torch.from_numpy(nbr_index),
        nbr_mask=torch.from_numpy(nbr_mask),
        node_mask=torch.from_numpy(node_mask),
        legal=torch.from_numpy(legal),
    )
