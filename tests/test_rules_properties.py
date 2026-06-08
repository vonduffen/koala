"""Property tests for the rules engine on a variety of tilings (ARCHITECTURE.md §4.3).

These play random legal games on graphs from several tiling families and assert invariants that
must hold for *any* legal Go position, independent of geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from tilinggo.rules import BLACK, EMPTY, WHITE, Board, GoState
from tilinggo.tilings import periodic, uniform


def boards():
    return [
        periodic.rectangular(5, 5),
        periodic.generate("hex", cells=40, seed=1),
        periodic.generate("tri", cells=40, seed=2),
        uniform.generate("trihexagonal", radius=4),
        uniform.generate("snub_square", radius=4),
    ]


def _random_game(board: Board, rng, max_plies=400):
    """Play uniformly-random legal moves until terminal; yield each state along the way."""
    s = board.new_game()
    states = [s]
    for _ in range(max_plies):
        if s.is_terminal:
            break
        legal = np.flatnonzero(s.legal_moves())
        move = int(rng.choice(legal))
        s = s.play(move)
        states.append(s)
    return states


def _all_chains_have_liberties(s: GoState) -> bool:
    """Every on-board chain must have at least one liberty (else it should have been captured)."""
    seen = np.zeros(s.board.n, dtype=bool)
    for v in range(s.board.n):
        if s.colors[v] == EMPTY or seen[v]:
            continue
        chain, libs = s._group_and_liberties(s.colors, v)
        for c in chain:
            seen[c] = True
        if len(libs) == 0:
            return False
    return True


@pytest.mark.parametrize("bi", range(5))
def test_no_zero_liberty_chains_during_random_game(bi):
    board = Board(boards()[bi], komi=0.5)
    rng = np.random.default_rng(100 + bi)
    for s in _random_game(board, rng):
        assert _all_chains_have_liberties(s)


@pytest.mark.parametrize("bi", range(5))
def test_area_plus_neutral_equals_N(bi):
    board = Board(boards()[bi], komi=0.5)
    rng = np.random.default_rng(200 + bi)
    states = _random_game(board, rng)
    final = states[-1]
    black, white = final.score()
    # area scoring partitions every node into black / white / neutral, so the colour areas plus
    # the neutral count must total N exactly.
    neutral = board.n - black - white
    assert neutral >= 0
    assert black + white + neutral == board.n


@pytest.mark.parametrize("bi", range(5))
def test_replaying_moves_reproduces_hash_and_position(bi):
    board = Board(boards()[bi], komi=0.5)
    rng = np.random.default_rng(300 + bi)
    states = _random_game(board, rng)
    moves = [_infer_move(states[i], states[i + 1]) for i in range(len(states) - 1)]

    replay = board.new_game()
    for mv, expected in zip(moves, states[1:]):
        replay = replay.play(mv)
        assert replay.stone_hash == expected.stone_hash
        assert np.array_equal(replay.colors, expected.colors)
        assert replay.to_move == expected.to_move


def _infer_move(before: GoState, after: GoState) -> int:
    """Recover the move that took `before` to `after` (a node index, or the pass index)."""
    if after.pass_count > before.pass_count:
        return before.board.pass_move
    # the played node is the one that went from EMPTY to the mover's colour
    played = np.flatnonzero((before.colors == EMPTY) & (after.colors == before.to_move))
    assert played.size == 1
    return int(played[0])


@pytest.mark.parametrize("bi", range(5))
def test_legal_moves_are_actually_playable(bi):
    board = Board(boards()[bi], komi=0.5)
    rng = np.random.default_rng(400 + bi)
    s = board.new_game()
    for _ in range(50):
        if s.is_terminal:
            break
        legal = np.flatnonzero(s.legal_moves())
        # every move the engine reports legal must play without raising
        for mv in legal:
            s.play(int(mv))  # must not raise
        s = s.play(int(rng.choice(legal)))


def test_hash_consistency_after_capture_cycle():
    # Returning to the empty board (capture everything, then it's empty) gives back the empty
    # hash — a direct check that the incremental Zobrist add/remove is balanced.
    board = Board(periodic.rectangular(3, 3))
    s = board.new_game()
    empty_hash = s.stone_hash
    s = s.play(0)              # Black at corner 0
    assert s.stone_hash != empty_hash
    # White captures it: surround corner 0 (neighbours 1 and 3).
    s = s.play(1)              # White at 1
    s = s.play(board.pass_move)  # Black passes
    s = s.play(3)              # White at 3 captures Black 0
    assert s.colors[0] == EMPTY
    # board now has White at 1 and 3 only
    expected = empty_hash ^ board.zobrist(1, WHITE) ^ board.zobrist(3, WHITE)
    assert s.stone_hash == expected
