"""Game-record round-trip (tilinggo/records.py) — docs/game-record-format.md."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tilinggo import records
from tilinggo.ui.server import _make_board

SUBSTRATES = ["square_small", "hex_small", "tri_small", "trihex_small", "penrose_small"]
GAMES_PER = 20  # 5 × 20 = 100 round-trip games


def random_game(board, rng, max_len):
    state, moves = board.new_game(), []
    while len(moves) < max_len and not state.is_terminal:
        legal = np.flatnonzero(state.legal_moves())
        nodes = legal[legal != board.pass_move]
        mv = int(rng.choice(nodes)) if nodes.size and rng.random() >= 0.02 else board.pass_move
        state = state.play(mv)
        moves.append(mv)
    return state, moves


def sig(state):
    return (state.colors.tolist(), state.to_move, state.move_num, state.pass_count,
            state.stone_hash, tuple(sorted(state.history)))


@pytest.mark.parametrize("key", SUBSTRATES)
def test_round_trip_property(key, tmp_path):
    board = _make_board(key, komi=5.5)
    rng = np.random.default_rng(hash(key) % 2**32)
    for g in range(GAMES_PER):
        state, moves = random_game(board, rng, max_len=5 + int(rng.integers(0, 100)))
        p = records.save_record(tmp_path / f"{key}_{g}.json", key=key, moves=moves, board=board)
        back = records.load_record(p)
        assert sig(back) == sig(state), f"{key} game {g}: round-trip diverged"


def test_embedded_graph_accepted(tmp_path):
    board = _make_board("square_small", komi=5.5)
    state, moves = random_game(board, np.random.default_rng(0), 20)
    p = records.save_record(tmp_path / "g.json", key="square_small", moves=moves,
                            board=board, embed_graph=True)
    assert records.load_record(p).move_num == state.move_num


def test_embedded_graph_mismatch_rejected(tmp_path):
    board = _make_board("square_small", komi=5.5)
    p = records.save_record(tmp_path / "g.json", key="square_small", moves=[4, 9],
                            board=board, embed_graph=True)
    rec = json.loads(p.read_text())
    rec["board"]["graph"]["edges"][0] = [0, 99]          # corrupt one edge
    p.write_text(json.dumps(rec))
    with pytest.raises(records.RecordError) as e:
        records.load_record(p)
    assert e.value.code == "BAD_BOARD"


@pytest.mark.parametrize("mutate,code", [
    (lambda r: r.update(format="sgf"), "BAD_FORMAT"),
    (lambda r: r.update(version=99), "BAD_VERSION"),
    (lambda r: r["board"].update(key="atlantis_large"), "BAD_BOARD"),
    (lambda r: r["board"].update(fingerprint="zzzzzz"), "BAD_BOARD"),
    (lambda r: r.update(moves=[10**6]), "BAD_MOVE"),
    (lambda r: r.update(moves=[4, 4]), "ILLEGAL_MOVE"),
])
def test_malformed_records_rejected(tmp_path, mutate, code):
    board = _make_board("square_small", komi=5.5)
    p = records.save_record(tmp_path / "g.json", key="square_small", moves=[4, 9], board=board)
    rec = json.loads(p.read_text())
    mutate(rec)
    p.write_text(json.dumps(rec))
    with pytest.raises(records.RecordError) as e:
        records.load_record(p)
    assert e.value.code == code
