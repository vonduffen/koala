"""A deliberately-simple, independent reference implementation of graph Go.

This exists only for the differential test (ARCHITECTURE.md §4.3): it is written in a different
style from the production engine — plain Python lists, naive flood-fill liberties recomputed
every time, and full board-tuple snapshots for superko instead of incremental Zobrist hashing.
Agreement between the two over thousands of random games is strong evidence both are correct.

It uses the same rules: capture-then-suicide-check, positional superko (Tromp-Taylor: no two
identical board colorings), area scoring, two-pass / move-cap termination.
"""

from __future__ import annotations

EMPTY, BLACK, WHITE = 0, 1, 2


def _opp(c):
    return WHITE if c == BLACK else BLACK


class RefGo:
    def __init__(self, graph, komi=0.0, max_moves=None):
        self.n = int(graph.num_nodes)
        self.adj = [[int(x) for x in a] for a in graph.neighbors()]
        self.komi = float(komi)
        self.max_moves = 3 * self.n if max_moves is None else int(max_moves)
        self.pass_move = self.n

        self.colors = [EMPTY] * self.n
        self.to_move = BLACK
        self.pass_count = 0
        self.move_num = 0
        self.seen = {tuple(self.colors)}  # positional superko: board colorings seen

    # -- helpers ------------------------------------------------------------------------
    def _group_libs(self, colors, start):
        color = colors[start]
        seen = {start}
        libs = set()
        stack = [start]
        while stack:
            u = stack.pop()
            for w in self.adj[u]:
                if colors[w] == EMPTY:
                    libs.add(w)
                elif colors[w] == color and w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen, libs

    def _apply(self, node, player):
        """Return resulting colors list, or None if illegal by board logic (occupied/suicide)."""
        if self.colors[node] != EMPTY:
            return None
        colors = list(self.colors)
        colors[node] = player
        opp = _opp(player)
        for w in self.adj[node]:
            if colors[w] == opp:
                grp, libs = self._group_libs(colors, w)
                if not libs:
                    for c in grp:
                        colors[c] = EMPTY
        _, own = self._group_libs(colors, node)
        if not own:
            return None
        return colors

    # -- queries ------------------------------------------------------------------------
    @property
    def is_terminal(self):
        return self.pass_count >= 2 or self.move_num >= self.max_moves

    def legal_moves(self):
        legal = [False] * (self.n + 1)
        legal[self.n] = True
        if self.is_terminal:
            return legal
        for v in range(self.n):
            colors = self._apply(v, self.to_move)
            if colors is None:
                continue
            if tuple(colors) not in self.seen:  # positional superko
                legal[v] = True
        return legal

    # -- transition (mutates in place; the reference plays one game line) ---------------
    def play(self, move):
        if move == self.n:  # pass: board unchanged, already in `seen`
            self.to_move = _opp(self.to_move)
            self.pass_count += 1
            self.move_num += 1
            return
        colors = self._apply(move, self.to_move)
        if colors is None:
            raise ValueError("illegal move (suicide/occupied)")
        snap = tuple(colors)
        if snap in self.seen:
            raise ValueError("illegal move (superko)")
        self.colors = colors
        self.to_move = _opp(self.to_move)
        self.pass_count = 0
        self.move_num += 1
        self.seen.add(snap)

    # -- scoring ------------------------------------------------------------------------
    def score(self):
        black = sum(1 for c in self.colors if c == BLACK)
        white = sum(1 for c in self.colors if c == WHITE)
        visited = [False] * self.n
        for v in range(self.n):
            if self.colors[v] != EMPTY or visited[v]:
                continue
            region, borders, stack = [], set(), [v]
            visited[v] = True
            while stack:
                u = stack.pop()
                region.append(u)
                for w in self.adj[u]:
                    if self.colors[w] == EMPTY:
                        if not visited[w]:
                            visited[w] = True
                            stack.append(w)
                    else:
                        borders.add(self.colors[w])
            if borders == {BLACK}:
                black += len(region)
            elif borders == {WHITE}:
                white += len(region)
        return black, white
