"""Differential test: production GoState vs the independent reference (ARCHITECTURE.md §4.3).

Two implementations written in different styles (incremental Zobrist + sets vs naive recompute
+ board-tuple snapshots) play the *same* random games in lockstep. At every ply their legal-move
sets must match exactly, and at game end their area scores must agree. Disagreement on any board
in thousands of games would expose a rules bug in one of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from tilinggo.rules import Board
from tilinggo.tilings import periodic, uniform

from reference_go import RefGo


def _boards():
    return {
        "rect5x5": periodic.rectangular(5, 5),
        "rect4x6": periodic.rectangular(4, 6),
        "hex": periodic.generate("hex", cells=45, seed=7),
        "snub_square": uniform.generate("snub_square", radius=4),
    }


@pytest.mark.parametrize("name", list(_boards()))
def test_differential_random_games(name):
    graph = _boards()[name]
    n_games = 40 if name.startswith("rect") else 15

    for g in range(n_games):
        rng = np.random.default_rng(1000 * hash(name) % 7919 + g)
        board = Board(graph, komi=0.5)
        eng = board.new_game()
        ref = RefGo(graph, komi=0.5)

        ply = 0
        while not eng.is_terminal and ply < 600:
            le = eng.legal_moves()
            lr = ref.legal_moves()
            # exact agreement of the legal-move sets at every position
            assert list(map(bool, le)) == list(map(bool, lr)), (
                f"legal mismatch on {name} game {g} ply {ply}")

            legal_idx = np.flatnonzero(le)
            move = int(rng.choice(legal_idx))
            eng = eng.play(move)
            ref.play(move)

            # the two engines must agree on the resulting board, move by move
            assert np.array_equal(eng.colors, np.array(ref.colors, dtype=np.int8))
            ply += 1

        assert eng.is_terminal == ref.is_terminal
        assert eng.score() == ref.score(), f"score mismatch on {name} game {g}"
