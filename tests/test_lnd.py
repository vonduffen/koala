"""Validate the life-and-death suite files (tests/lnd/) — structure, not engine accuracy.

Accuracy is a tracked metric (scripts/lnd_score.py, non-blocking in CI); these tests gate only
that the suite itself is well-formed: states are constructible, groups exist on the board,
expected moves are legal, and coverage meets the floor (≥40 positions, ≥3 substrates).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tilinggo.rules import GoState
from tilinggo.ui.server import _make_board

SUITE = Path(__file__).resolve().parent / "lnd"
FILES = sorted(SUITE.glob("*.json"))


def test_suite_coverage():
    assert len(FILES) >= 40, f"suite has {len(FILES)} positions; floor is 40"
    substrates = {json.loads(f.read_text())["board_key"] for f in FILES}
    assert len(substrates) >= 3, f"suite covers {substrates}; floor is 3 substrates"


@pytest.mark.parametrize("f", FILES, ids=[f.stem for f in FILES])
def test_position_well_formed(f):
    p = json.loads(f.read_text())
    board = _make_board(p["board_key"], komi=5.5)
    colors = np.asarray(p["colors"], dtype=np.int8)
    assert colors.shape == (board.n,)
    assert p["to_move"] in (1, 2)
    state = GoState(board, colors=colors, to_move=p["to_move"])
    assert p["group"], "empty target group"
    assert all(0 <= u < board.n for u in p["group"])
    exp = p["expected"]
    assert ("status" in exp) or ("moves" in exp)
    if "status" in exp:
        assert exp["status"] in ("alive", "dead")
    if "moves" in exp:
        legal = state.legal_moves()
        for m in exp["moves"]:
            assert 0 <= m < board.n and legal[m], f"expected move {m} is not legal"
