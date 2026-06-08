"""PUCT Monte-Carlo Tree Search with batched leaf evaluation (ARCHITECTURE.md §6).

AlphaZero-style, graph-agnostic: selection uses ``Q + c_puct·P·√ΣN / (1 + N)``; leaves are scored
by an evaluator's value head (no rollouts). To exploit batched matrix math, each search wave
collects up to ``eval_batch`` leaves using **virtual loss** (so concurrent descents diverge
instead of all taking the same path), evaluates them in one call, then expands and backs them up.

Other spec pieces: Dirichlet noise at the root (α = 10/avg-legal-moves, ε = 0.25), FPU reduction
for unvisited children, tree reuse between moves, and temperature control of move selection.
Single-process; no distributed infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..rules.gostate import GoState


@dataclass
class MCTSConfig:
    num_simulations: int = 160
    c_puct: float = 1.4
    eval_batch: int = 16
    dirichlet_eps: float = 0.25
    dirichlet_alpha: float | None = None  # default: 10 / avg legal moves
    fpu_reduction: float = 0.25
    virtual_loss: float = 1.0


class Node:
    """One position in the search tree. Edge statistics (N, W, P) live on the parent."""

    __slots__ = ("state", "terminal", "expanded", "children",
                 "legal", "P", "N", "W", "term_value")

    def __init__(self, state: GoState):
        self.state = state
        self.terminal = state.is_terminal
        self.expanded = False
        self.children: dict[int, "Node"] = {}
        self.legal = self.P = self.N = self.W = None
        self.term_value = None
        if self.terminal:
            # value from this node's side-to-move perspective (±1 win/loss)
            self.term_value = 1.0 if state.winner() == state.to_move else -1.0

    def expand(self, priors_full: np.ndarray) -> None:
        legal = np.flatnonzero(self.state.legal_moves())
        pr = np.asarray(priors_full, dtype=np.float64)[legal]
        total = pr.sum()
        self.P = pr / total if total > 0 else np.full(len(legal), 1.0 / len(legal))
        self.legal = legal.astype(np.int64)
        self.N = np.zeros(len(legal), dtype=np.float64)
        self.W = np.zeros(len(legal), dtype=np.float64)
        self.expanded = True


class MCTS:
    def __init__(self, evaluator, config: MCTSConfig = MCTSConfig(), rng=None):
        self.eval = evaluator
        self.cfg = config
        self.rng = rng if rng is not None else np.random.default_rng()

    # ---- selection ----------------------------------------------------------------------

    def _select_child(self, node: Node) -> int:
        N, W, P = node.N, node.W, node.P
        sqrt_total = np.sqrt(N.sum()) + 1e-8
        visited = N > 0
        if visited.any():
            parent_v = W[visited].sum() / N[visited].sum()
            fpu = parent_v - self.cfg.fpu_reduction * np.sqrt(P[visited].sum())
        else:
            fpu = 0.0
        q = np.where(visited, W / np.maximum(N, 1.0), fpu)
        u = self.cfg.c_puct * P * sqrt_total / (1.0 + N)
        return int(np.argmax(q + u))

    def _descend(self, root: Node):
        """Walk from root to an unexpanded/terminal leaf, applying virtual loss along the way."""
        node, path = root, []
        while node.expanded and not node.terminal:
            ai = self._select_child(node)
            node.N[ai] += self.cfg.virtual_loss
            node.W[ai] -= self.cfg.virtual_loss
            path.append((node, ai))
            move = int(node.legal[ai])
            child = node.children.get(move)
            if child is None:
                child = Node(node.state.play(move))
                node.children[move] = child
            node = child
        return node, path

    def _backup(self, path, value: float) -> None:
        """Negamax backup; ``value`` is from the leaf's side-to-move perspective."""
        v = value
        for node, ai in reversed(path):
            v = -v  # flip into this node's perspective
            node.N[ai] += 1.0 - self.cfg.virtual_loss  # +1 real, remove virtual
            node.W[ai] += v + self.cfg.virtual_loss

    # ---- root --------------------------------------------------------------------------

    def _add_dirichlet(self, root: Node) -> None:
        if self.cfg.dirichlet_eps <= 0:
            return
        k = len(root.legal)
        alpha = self.cfg.dirichlet_alpha if self.cfg.dirichlet_alpha is not None else 10.0 / k
        noise = self.rng.dirichlet([alpha] * k)
        root.P = (1 - self.cfg.dirichlet_eps) * root.P + self.cfg.dirichlet_eps * noise

    # ---- main loop ---------------------------------------------------------------------

    def run(self, root_state: GoState, reuse: Node | None = None) -> Node:
        """Run the configured simulations from ``root_state`` and return the root node.

        ``reuse`` may be a subtree (a child from a previous search) whose state matches, to
        carry over its statistics (tree reuse between moves).
        """
        root = reuse if (reuse is not None and reuse.expanded) else Node(root_state)
        if root.terminal:
            return root
        if not root.expanded:
            priors, _ = self.eval([root.state])
            root.expand(priors[0])
        self._add_dirichlet(root)

        sims = 0
        while sims < self.cfg.num_simulations:
            k = min(self.cfg.eval_batch, self.cfg.num_simulations - sims)
            backups, to_eval, seen = [], [], set()
            for _ in range(k):
                leaf, path = self._descend(root)
                backups.append((path, leaf))
                sims += 1
                if not leaf.terminal and id(leaf) not in seen:
                    seen.add(id(leaf))
                    to_eval.append(leaf)

            value_of: dict[int, float] = {}
            if to_eval:
                priors, values = self.eval([n.state for n in to_eval])
                for n, pr, v in zip(to_eval, priors, values):
                    n.expand(pr)
                    value_of[id(n)] = float(v)

            for path, leaf in backups:
                v = leaf.term_value if leaf.terminal else value_of[id(leaf)]
                self._backup(path, v)
        return root

    # ---- outputs -----------------------------------------------------------------------

    def visit_policy(self, root: Node) -> np.ndarray:
        """Normalized visit-count distribution over the full move space ``[0..N]`` (N = pass)."""
        pi = np.zeros(root.state.board.n + 1, dtype=np.float64)
        pi[root.legal] = root.N
        total = pi.sum()
        return pi / total if total > 0 else pi

    def select_move(self, root: Node, temperature: float = 1.0) -> int:
        """Pick a move from visit counts. ``temperature`` 0 → argmax; 1 → proportional to visits."""
        visits = root.N
        if temperature <= 1e-6 or visits.sum() == 0:
            ai = int(np.argmax(visits))
        else:
            weights = visits ** (1.0 / temperature)
            ai = int(self.rng.choice(len(visits), p=weights / weights.sum()))
        return int(root.legal[ai])
