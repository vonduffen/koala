"""Game records — the JSON "graph-SGF" shared with the webapp (docs/game-record-format.md).

SGF assumes a coordinate grid, which only exists on the square board; Koala games are
sequences of *node indices* into a named board graph. A record stores the board key, a graph
fingerprint (and optionally the full edge list), metadata, and the move list. The same schema
is read/written by the webapp (webapp/records.js); cross-language identity is tested in
tests/test_records_cross.py.

Note an API asymmetry forced by the engine design: :class:`~tilinggo.rules.GoState` is an
immutable position and does not carry the move sequence that produced it, so ``save_record``
takes the move list explicitly; ``load_record`` replays the record and returns the final state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rules import Board, GoState

FORMAT = "euclidean-go-record"
VERSION = 1
PASS = "pass"


class RecordError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def fingerprint(board: Board) -> str:
    """FNV-1a over node count + flattened edge list — byte-for-byte the webapp's algorithm
    (webapp/share.js), so records and links interoperate."""
    h = 0x811C9DC5

    def mix_int(v: int) -> None:
        nonlocal h
        for shift in (0, 8, 16, 24):
            h ^= (v >> shift) & 0xFF
            h = (h * 0x01000193) & 0xFFFFFFFF

    mix_int(board.n)
    for v in np.asarray(board.graph.edges, dtype=np.int64).ravel():
        mix_int(int(v))
    out, n = "", h
    while n:
        out = "0123456789abcdefghijklmnopqrstuvwxyz"[n % 36] + out
        n //= 36
    return (out or "0")[:6]


def _resolve_board(key: str, komi: float) -> Board:
    from .ui.server import _make_board  # the families×sizes catalogue

    return _make_board(key, komi=komi)


def save_record(path: str | Path, *, key: str, moves: list[int], board: Board | None = None,
                players: dict | None = None, result: str | None = None,
                date: str | None = None, embed_graph: bool = False) -> Path:
    """Write a game record. ``moves`` are node indices in play order (``board.pass_move`` or
    the string "pass" for passes)."""
    board = board if board is not None else _resolve_board(key, komi=5.5)
    norm = [PASS if (m == PASS or int(m) == board.pass_move) else int(m) for m in moves]
    rec = {
        "format": FORMAT,
        "version": VERSION,
        "board": {"key": key, "fingerprint": fingerprint(board), "nodes": board.n},
        "komi": board.komi,
        "rules": "area",
        "players": players or {},
        "date": date,
        "result": result,
        "moves": norm,
    }
    if embed_graph:
        rec["board"]["graph"] = {
            "edges": np.asarray(board.graph.edges, dtype=int).reshape(-1, 2).tolist()}
    p = Path(path)
    p.write_text(json.dumps(rec, indent=1))
    return p


def load_record(path: str | Path) -> GoState:
    """Parse + validate a record, replay it through the rules engine, return the final state."""
    try:
        rec = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise RecordError("BAD_FILE", f"not a readable JSON record: {e}") from e
    if rec.get("format") != FORMAT:
        raise RecordError("BAD_FORMAT", "not a euclidean-go-record file")
    if rec.get("version") != VERSION:
        raise RecordError("BAD_VERSION", f"unsupported record version {rec.get('version')}")
    b = rec.get("board") or {}
    key = b.get("key")
    if not isinstance(key, str):
        raise RecordError("BAD_FORMAT", "record has no board key")
    try:
        board = _resolve_board(key, komi=float(rec.get("komi", 5.5)))
    except Exception as e:
        raise RecordError("BAD_BOARD", f"unknown board {key!r}") from e
    if b.get("fingerprint") and b["fingerprint"] != fingerprint(board):
        raise RecordError("BAD_BOARD",
                          f"record was made for a different build of board {key!r}")
    if "graph" in b:  # embedded graph must agree with the compiled substrate
        emb = np.asarray(b["graph"].get("edges", []), dtype=int).reshape(-1, 2)
        ours = np.asarray(board.graph.edges, dtype=int).reshape(-1, 2)
        if emb.shape != ours.shape or not np.array_equal(emb, ours):
            raise RecordError("BAD_BOARD",
                              "embedded graph does not match the named substrate's compiled graph")

    state = board.new_game()
    for ply, m in enumerate(rec.get("moves", [])):
        node = board.pass_move if m == PASS else m
        if not (node == board.pass_move or (isinstance(node, int) and 0 <= node < board.n)):
            raise RecordError("BAD_MOVE", f"move {m!r} out of range at ply {ply}")
        if not state.legal_moves()[node]:
            raise RecordError("ILLEGAL_MOVE", f"move {m!r} is illegal at ply {ply}")
        state = state.play(node)
    return state
