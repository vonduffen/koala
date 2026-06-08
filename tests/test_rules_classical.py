"""Classical Go rules cases ported onto the square-grid BoardGraph (ARCHITECTURE.md §4.3).

Positions are written as ASCII diagrams; `setup()` turns one into a GoState on a rectangular
grid. 'X' = Black, 'O' = White, '.' = empty. Row 0 is the top line.
"""

from __future__ import annotations

import numpy as np
import pytest

from tilinggo.rules import BLACK, EMPTY, WHITE, Board, GoState, IllegalMove
from tilinggo.tilings import periodic

CHAR = {".": EMPTY, "X": BLACK, "O": WHITE}


def setup(diagram: str, to_move: int = BLACK, komi: float = 0.5) -> tuple[GoState, Board]:
    rows = [ln.split() for ln in diagram.strip().splitlines()]
    rows = [r for r in rows if r]  # drop blank lines
    h, w = len(rows), len(rows[0])
    assert all(len(r) == w for r in rows), "ragged diagram"
    board = Board(periodic.rectangular(h, w), komi=komi)
    colors = np.zeros(board.n, dtype=np.int8)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            colors[r * w + c] = CHAR[ch]
    stone_hash = 0
    for v in range(board.n):
        if colors[v] != EMPTY:
            stone_hash ^= board.zobrist(v, int(colors[v]))
    return GoState(board, colors, to_move=to_move, stone_hash=stone_hash), board


def at(w, r, c):
    return r * w + c


def test_corner_capture():
    # White corner stone with one liberty at (0,1); Black fills it → capture.
    s, b = setup("""
        O . .
        X . .
        . . .
    """, to_move=BLACK)
    s2 = s.play(at(3, 0, 1))
    assert s2.colors[at(3, 0, 0)] == EMPTY
    assert s2.colors[at(3, 0, 1)] == BLACK


def test_multistone_chain_capture():
    # White chain (1,1),(1,2) has a single liberty at (2,1); Black there captures both.
    s, b = setup("""
        X X X .
        X O O X
        X . X .
        . . . .
    """, to_move=BLACK)
    s2 = s.play(at(4, 2, 1))
    assert s2.colors[at(4, 1, 1)] == EMPTY
    assert s2.colors[at(4, 1, 2)] == EMPTY


def test_suicide_is_illegal():
    s, b = setup("""
        . O .
        O . O
        . O .
    """, to_move=BLACK)
    assert not s.is_legal(at(3, 1, 1))
    with pytest.raises(IllegalMove):
        s.play(at(3, 1, 1))


def test_capture_overrides_suicide():
    # Throw-in: the White group (1,1),(1,2),(2,1) has a single liberty at (2,2). Black plays
    # there — that point is also Black's only liberty, but the move captures White first, so it
    # is legal (capture is resolved before the suicide check).
    s, b = setup("""
        X X X
        X O O
        X O .
    """, to_move=BLACK)
    assert s.is_legal(at(3, 2, 2))
    s2 = s.play(at(3, 2, 2))
    assert s2.colors[at(3, 1, 1)] == EMPTY
    assert s2.colors[at(3, 1, 2)] == EMPTY
    assert s2.colors[at(3, 2, 1)] == EMPTY
    assert s2.colors[at(3, 2, 2)] == BLACK


def test_basic_ko_recapture_forbidden_then_allowed_after_threat():
    # Textbook ko: Black captures one White stone; White's immediate recapture would recreate
    # the starting whole-board position (Black to move) and is forbidden by superko.
    s, b = setup("""
        . X O .
        X O . O
        . X O .
        . . . .
    """, to_move=BLACK)
    cap = s.play(at(4, 1, 2))                  # Black captures White (1,1)
    assert cap.colors[at(4, 1, 1)] == EMPTY
    assert cap.colors[at(4, 1, 2)] == BLACK
    assert not cap.is_legal(at(4, 1, 1))       # immediate recapture forbidden (ko)
    with pytest.raises(IllegalMove):
        cap.play(at(4, 1, 1))
    # After the board changes (a "ko threat" played elsewhere by each side), the recapture no
    # longer repeats a prior position, so superko allows it.
    after = cap.play(at(4, 3, 3)).play(at(4, 0, 0))  # White then Black play elsewhere
    assert after.is_legal(at(4, 1, 1))


def test_two_passes_end_game():
    s, b = setup(". . .\n. X .\n. . .", to_move=BLACK)
    assert not s.is_terminal
    s = s.play(b.pass_move)
    assert not s.is_terminal
    s = s.play(b.pass_move)
    assert s.is_terminal


def test_area_scoring_neutral_middle():
    # The gap column touches both colours (dame) → neutral; each side scores only its stones.
    s, b = setup("""
        X . O
        X . O
        X . O
    """)
    assert s.score() == (3, 3)


def test_area_scoring_territory():
    # White fully encloses one empty point → 8 stones + 1 territory = 9.
    s, b = setup("""
        O O O
        O . O
        O O O
    """)
    assert s.score() == (0, 9)


def test_dead_stone_captured_then_territory():
    # Area scoring requires playing it out: a lone White stone inside Black's wall is just a
    # stone until captured. Black fills its four liberties (White passing between) and captures
    # it, after which the interior is Black's.
    s, b = setup("""
        X X X X X
        X . . . X
        X . O . X
        X . . . X
        X X X X X
    """, to_move=BLACK)
    liberties = [at(5, 1, 2), at(5, 2, 1), at(5, 2, 3), at(5, 3, 2)]
    st = s
    for m in liberties[:-1]:
        st = st.play(m).play(st.board.pass_move)  # Black fills, White passes
    st = st.play(liberties[-1])                    # last liberty → capture
    assert st.colors[at(5, 2, 2)] == EMPTY


def test_move_cap_terminates():
    board = Board(periodic.rectangular(2, 2), max_moves=4)
    st = board.new_game().play(0).play(1).play(board.pass_move).play(2)
    assert st.move_num == 4
    assert st.is_terminal
