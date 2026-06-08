"""Leaf evaluators for MCTS: map a batch of GoStates to (policy priors, value).

An evaluator returns, per state, a prior array indexed by that state's own move space
(``0..n-1`` for node moves, ``n`` for pass) plus a scalar value in (-1, 1) from the side-to-move's
perspective. This keeps MCTS independent of how the value/policy are produced — a trained net, a
cheap heuristic, or uniform priors for sanity checks all share the interface.
"""

from __future__ import annotations

import numpy as np

from ..rules.gostate import BLACK, GoState


class UniformEvaluator:
    """Uniform priors over legal moves and value 0 — a pure search-mechanism baseline."""

    def __call__(self, states: list[GoState]):
        priors = [np.ones(s.board.n + 1, dtype=np.float64) for s in states]
        values = np.zeros(len(states), dtype=np.float64)
        return priors, values


class ScoreHeuristicEvaluator:
    """Uniform priors, but value from the current area score — a weak yet real signal.

    Enough to show that search adds value over the raw (uniform) policy: MCTS using this looks
    ahead toward higher-scoring positions and crushes random play, validating the search.
    """

    def __init__(self, scale: float | None = None):
        self.scale = scale

    def __call__(self, states: list[GoState]):
        priors = [np.ones(s.board.n + 1, dtype=np.float64) for s in states]
        values = np.empty(len(states), dtype=np.float64)
        for i, s in enumerate(states):
            margin = s.score_difference()
            if s.to_move != BLACK:
                margin = -margin
            scale = self.scale if self.scale is not None else max(2.0, 0.15 * s.board.n)
            values[i] = np.tanh(margin / scale)
        return priors, values


class NetEvaluator:
    """Wrap a :class:`TilingGoNet` as an MCTS evaluator (legal-masked policy + value head)."""

    def __init__(self, net, device: str = "cpu"):
        import torch  # local import so the rules/search core needs no torch

        self.torch = torch
        self.net = net.to(device).eval()
        self.device = device

    def __call__(self, states: list[GoState]):
        from ..nn import encoding  # local import (torch-dependent)

        batch = encoding.encode_states(states).to(self.device)
        with self.torch.no_grad():
            probs, value = self.net.policy_value(batch)  # [B, N_max+1], [B]
        probs = probs.cpu().numpy()
        value = value.cpu().numpy().astype(np.float64)
        n_max = batch.x.shape[1]

        priors = []
        for i, s in enumerate(states):
            n = s.board.n
            arr = np.empty(n + 1, dtype=np.float64)
            arr[:n] = probs[i, :n]
            arr[n] = probs[i, n_max]  # pass prob lives at the padded end (see encoding)
            priors.append(arr)
        return priors, value
