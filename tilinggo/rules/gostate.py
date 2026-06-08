"""Graph Go: exact rules on an arbitrary BoardGraph.

The engine never sees geometry — only the node/edge adjacency of a :class:`BoardGraph`
(ARCHITECTURE.md §1, §4). Classical Go on the square grid of intersections is exactly this
model on the grid's dual, so all standard rules carry over with neighbours = graph adjacency.

Rules implemented (§4.1, all DECIDED):
  * Black moves first; a move places a stone on an empty node, or passes.
  * Capture removes opponent chains with zero liberties; suicide (self-capture) is illegal.
  * Ko: **situational positional superko** — a move may not recreate any previous whole-board
    position *with the same player to move*. Tracked with incremental Zobrist hashing plus a
    set of visited position keys. Hash collisions are accepted (astronomically unlikely).
  * Game ends on two consecutive passes, or a hard cap of 3*N moves (scored as-is).
  * Area scoring: stones + territory; empty regions touching one colour score for it, regions
    touching both (or neither) are neutral.

States are immutable: :meth:`GoState.play` returns a new state, leaving the receiver unchanged
(this keeps MCTS simple). Perf target is ≥ 2000 moves/sec on a 100-node board (§4.2).
"""

from __future__ import annotations

import numpy as np

from ..tilings.boardgraph import BoardGraph

EMPTY = 0
BLACK = 1
WHITE = 2


def opponent(color: int) -> int:
    """The other player. ``opponent(BLACK) == WHITE`` and vice versa."""
    return WHITE if color == BLACK else BLACK


class IllegalMove(Exception):
    """Raised when :meth:`GoState.play` is given a move that isn't legal."""


class Board:
    """Immutable per-graph context shared by all states of one game.

    Holds the adjacency lists, the Zobrist key tables, komi, and the move cap. None of this
    changes during play, so it is created once and shared (never copied per state).
    """

    def __init__(self, graph: BoardGraph, *, komi: float = 0.0, zobrist_seed: int = 0,
                 max_moves: int | None = None):
        self.graph = graph
        self.n = int(graph.num_nodes)
        self.adj: list[np.ndarray] = [np.asarray(a, dtype=np.int32) for a in graph.neighbors()]
        self.komi = float(komi)
        self.max_moves = int(max_moves) if max_moves is not None else 3 * self.n
        self.pass_move = self.n  # index used to denote "pass" in move/policy vectors

        rng = np.random.default_rng(zobrist_seed)
        hi = np.iinfo(np.int64).max
        # zobrist[node, color-1] for color in {BLACK, WHITE}. The whole-board hash (XOR of the
        # keys of all stones) is the superko identity — *positional* superko (Tromp-Taylor):
        # a play may not recreate any previous board coloring, regardless of side to move.
        self._zobrist = rng.integers(1, hi, size=(self.n, 2), dtype=np.int64)

    def zobrist(self, node: int, color: int) -> int:
        return int(self._zobrist[node, color - 1])

    def new_game(self) -> "GoState":
        """The empty starting position, Black to move."""
        return GoState(self)


class GoState:
    """An immutable Go position over a :class:`Board`."""

    __slots__ = ("board", "colors", "to_move", "pass_count", "move_num",
                 "stone_hash", "history", "_legal_cache")

    def __init__(self, board: Board, colors: np.ndarray | None = None, to_move: int = BLACK,
                 pass_count: int = 0, move_num: int = 0, stone_hash: int = 0,
                 history: frozenset[int] | None = None):
        self.board = board
        self.colors = np.zeros(board.n, dtype=np.int8) if colors is None else colors
        self.to_move = to_move
        self.pass_count = pass_count
        self.move_num = move_num
        self.stone_hash = int(stone_hash)
        # history = the set of board hashes already seen on the path to here (incl. self). For
        # positional superko the identity is the stone arrangement alone (no side-to-move).
        self.history = frozenset({self.stone_hash}) if history is None else history
        self._legal_cache: np.ndarray | None = None

    def _group_and_liberties(self, colors: np.ndarray, start: int):
        """Return (chain nodes set, liberty nodes set) for the stone at ``start``."""
        color = colors[start]
        adj = self.board.adj
        chain = {start}
        liberties: set[int] = set()
        stack = [start]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                w = int(w)
                cw = colors[w]
                if cw == EMPTY:
                    liberties.add(w)
                elif cw == color and w not in chain:
                    chain.add(w)
                    stack.append(w)
        return chain, liberties

    def _simulate(self, node: int, player: int):
        """Try playing ``player`` at ``node``. Return (colors, stone_hash, captured) or None.

        None means the move is illegal by board logic (occupied node or suicide). Captures of
        zero-liberty opponent chains are applied first, then the suicide check on the mover's
        own resulting chain (so a move that captures is legal even if it would otherwise be
        self-capture). The new hash is computed incrementally.
        """
        if self.colors[node] != EMPTY:
            return None
        colors = self.colors.copy()
        colors[node] = player
        opp = opponent(player)

        captured: list[int] = []
        for w in self.board.adj[node]:
            w = int(w)
            if colors[w] == opp:
                chain, libs = self._group_and_liberties(colors, w)
                if not libs:
                    for c in chain:
                        colors[c] = EMPTY
                        captured.append(c)

        _, own_libs = self._group_and_liberties(colors, node)
        if not own_libs:
            return None  # suicide is illegal

        h = self.stone_hash ^ self.board.zobrist(node, player)
        for c in captured:
            h ^= self.board.zobrist(c, opp)
        return colors, h, captured

    # ---- queries ------------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.pass_count >= 2 or self.move_num >= self.board.max_moves

    def legal_moves(self) -> np.ndarray:
        """Boolean array of length N+1; index N is pass. Exact w.r.t. suicide and superko."""
        if self._legal_cache is not None:
            return self._legal_cache
        n = self.board.n
        legal = np.zeros(n + 1, dtype=bool)
        legal[n] = True  # passing is always legal
        if not self.is_terminal:
            for v in range(n):
                if self.colors[v] != EMPTY:
                    continue
                sim = self._simulate(v, self.to_move)
                if sim is None:
                    continue
                _, h, _ = sim
                if h not in self.history:  # positional superko: board hash must be new
                    legal[v] = True
        self._legal_cache = legal
        return legal

    def is_legal(self, move: int) -> bool:
        return bool(self.legal_moves()[move])

    # ---- transitions --------------------------------------------------------------------

    def play(self, move: int) -> "GoState":
        """Return the state after ``move`` (a node index, or ``board.pass_move`` to pass)."""
        n = self.board.n
        if move == n:  # pass — board unchanged, so the hash is already in history (a no-op add)
            nt = opponent(self.to_move)
            return GoState(self.board, self.colors, nt, self.pass_count + 1,
                           self.move_num + 1, self.stone_hash, self.history)

        sim = self._simulate(move, self.to_move)
        if sim is None:
            raise IllegalMove(f"move {move} is suicide or onto an occupied node")
        colors, h, _ = sim
        if h in self.history:
            raise IllegalMove(f"move {move} violates positional superko")
        nt = opponent(self.to_move)
        return GoState(self.board, colors, nt, 0, self.move_num + 1, h, self.history | {h})

    # ---- scoring ------------------------------------------------------------------------

    def score(self) -> tuple[int, int]:
        """Area score (black_area, white_area): stones plus single-colour territory."""
        colors = self.colors
        adj = self.board.adj
        black = int(np.count_nonzero(colors == BLACK))
        white = int(np.count_nonzero(colors == WHITE))

        visited = np.zeros(self.board.n, dtype=bool)
        for v in range(self.board.n):
            if colors[v] != EMPTY or visited[v]:
                continue
            region = []
            borders: set[int] = set()
            stack = [v]
            visited[v] = True
            while stack:
                u = stack.pop()
                region.append(u)
                for w in adj[u]:
                    w = int(w)
                    cw = colors[w]
                    if cw == EMPTY:
                        if not visited[w]:
                            visited[w] = True
                            stack.append(w)
                    else:
                        borders.add(cw)
            if borders == {BLACK}:
                black += len(region)
            elif borders == {WHITE}:
                white += len(region)
            # touching both colours, or an empty board, is neutral
        return black, white

    def score_difference(self) -> float:
        """Black area minus White area minus komi (positive ⇒ Black leads)."""
        black, white = self.score()
        return black - white - self.board.komi

    def winner(self) -> int:
        """BLACK or WHITE. With integer areas and a half-integer komi there are no ties."""
        return BLACK if self.score_difference() > 0 else WHITE

    def ownership(self) -> np.ndarray:
        """Per-node final ownership as int8[N]: 0=black, 1=white, 2=neutral (training target)."""
        colors = self.colors
        adj = self.board.adj
        own = np.full(self.board.n, 2, dtype=np.int8)  # default neutral
        own[colors == BLACK] = 0
        own[colors == WHITE] = 1

        visited = np.zeros(self.board.n, dtype=bool)
        for v in range(self.board.n):
            if colors[v] != EMPTY or visited[v]:
                continue
            region, borders, stack = [], set(), [v]
            visited[v] = True
            while stack:
                u = stack.pop()
                region.append(u)
                for w in adj[u]:
                    w = int(w)
                    if colors[w] == EMPTY:
                        if not visited[w]:
                            visited[w] = True
                            stack.append(w)
                    else:
                        borders.add(int(colors[w]))
            if borders == {BLACK}:
                for c in region:
                    own[c] = 0
            elif borders == {WHITE}:
                for c in region:
                    own[c] = 1
        return own
