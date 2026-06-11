"""Cross-implementation record identity: JS-written records replay identically in Python,
and Python-written records replay identically in JS (docs/game-record-format.md)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tilinggo import records
from tilinggo.ui.server import _make_board

REPO = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not available")

SUBSTRATES = ["square_small", "hex_small", "tri_small", "trihex_small", "penrose_small"]


def test_js_written_records_load_in_python(tmp_path):
    out = tmp_path / "js"
    subprocess.run([NODE, str(REPO / "scripts" / "record_check.cjs"), "emit", str(out)],
                   check=True, capture_output=True)
    manifest = json.loads((out / "expected.json").read_text())
    assert len(manifest) >= 20
    for m in manifest:
        state = records.load_record(out / m["file"])
        assert state.colors.tolist() == m["colors"], f'{m["file"]}: colors differ'
        assert state.to_move == m["toMove"] and state.move_num == m["moveNum"]


def test_python_written_records_load_in_js(tmp_path):
    out = tmp_path / "py"
    out.mkdir()
    manifest = []
    for key in SUBSTRATES:
        board = _make_board(key, komi=5.5)
        rng = np.random.default_rng(hash(key) % 2**31)
        for g in range(4):
            state, moves = board.new_game(), []
            length = 30 + int(rng.integers(0, 60))
            while len(moves) < length and not state.is_terminal:
                legal = np.flatnonzero(state.legal_moves())
                nodes = legal[legal != board.pass_move]
                mv = int(rng.choice(nodes)) if nodes.size and rng.random() >= 0.02 else board.pass_move
                state = state.play(mv)
                moves.append(mv)
            name = f"{key}_{g}.json"
            records.save_record(out / name, key=key, moves=moves, board=board)
            manifest.append({"file": name, "colors": state.colors.tolist(),
                             "toMove": state.to_move, "moveNum": state.move_num})
    (out / "expected.json").write_text(json.dumps(manifest))
    r = subprocess.run([NODE, str(REPO / "scripts" / "record_check.cjs"), "verify", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"JS verify failed:\n{r.stdout}\n{r.stderr}"
