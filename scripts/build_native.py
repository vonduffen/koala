#!/usr/bin/env python3
"""Build the native macOS app bundle: a self-contained C++ server (Accelerate-fast) + the web UI +
the trained net + boards, in dist/native/. Copy that folder to any Apple-Silicon Mac and run the
launcher — no Python/torch/uv needed there.

Outputs dist/native/: TilingGo (binary), index.html, "Play TilingGo.command", data/{weights.bin,
boards.txt, <key>/{graph,static}.bin}.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tilinggo.sit.checkpoint import load_checkpoint  # noqa: E402
from tilinggo import export as cpp_bridge  # noqa: E402  — board/weight export to the C++ engine
from tilinggo.ui import server  # noqa: E402

CKPT = "results/universal/champion.pt"
KEYS = [k for k, _, _ in server.TILINGS]
b64 = lambda a, dt: base64.b64encode(np.asarray(a, dtype=dt).tobytes()).decode("ascii")


def families_struct():
    """Family → ordered [[key, size_label], ...] for the grouped picker (catalogue order)."""
    fams, cur = [], None
    for key, _, _ in server.TILINGS:
        fam, size = server._FAMILY_OF[key]
        if cur is None or cur["family"] != fam:
            cur = {"family": fam, "items": []}
            fams.append(cur)
        cur["items"].append([key, size])
    return fams


BODY = """
<div class="stage"><div class="glow"></div><div id="boardwrap"><div id="board"></div><div id="scan"></div></div></div>
<div class="winrail" title="Black win probability"><div class="winrail-fill" id="winfill"></div></div>
<div class="panel ctrl">
  <div class="brand"><div class="logo"></div><div><h1>KOALA</h1><p>native C++ engine</p></div></div>
  <label>Substrate</label>
  <select id="family"></select>
  <select id="variant" style="margin-top:6px"></select>
  <button id="random" style="margin-top:6px">🎲 Random board</button>
  <label>Opponent</label>
  <select id="opponent"><option value="engine">Neural engine (champion)</option><option value="off">Human (hot-seat)</option></select>
  <button id="analyze" class="cta">⌖ Analyze position</button>
  <div class="row"><button id="pass">Pass</button><button id="undo">Undo</button><button id="reset">New game</button></div>
  <label class="chk"><input type="checkbox" id="auto"> Auto-analyze each move</label>
  <label class="chk"><input type="checkbox" id="snd" checked> Stone sound</label>
  <label class="chk"><input type="checkbox" id="light"> ☀ Light mode</label>
</div>
<div class="panel info">
  <div class="turnrow"><span class="dot" id="turndot"></span><span id="turn">—</span><span class="spacer"></span><span class="muted" id="move">move 0</span></div>
  <div class="bignum"><div><div class="lbl">Score</div><div class="num" id="score">—</div></div><div><div class="lbl">Win&nbsp;B</div><div class="num acc" id="winpct">—</div></div></div>
  <div class="topmv" id="topmv"></div><div class="msg" id="msg"></div>
  <div class="lbl" style="margin-top:14px">Win-rate · Black</div>
  <div id="wrgraph"></div>
</div>
"""


def main() -> int:
    dist = REPO / "dist" / "native"
    data = dist / "data"
    data.mkdir(parents=True, exist_ok=True)
    net, _ = load_checkpoint(CKPT)

    cpp_bridge.export_weights(net, data / "weights.bin")
    geom, lines = {}, []
    for key in KEYS:
        board = server._make_board(key, komi=5.5)
        d = data / key; d.mkdir(exist_ok=True)
        cpp_bridge.export_graph(board, d / "graph.bin")
        cpp_bridge.export_static(board, d / "static.bin")
        lines.append(f"{key}\t{server._LABELS[key]}")
        g = board.graph
        geom[key] = {"n": int(g.num_nodes),
                     "coords": b64(np.asarray(g.coords).ravel(), "<f4"),
                     "edges": b64(np.asarray(g.edges).ravel(), "<i4")}
        print(f"  {key:14s} N={board.n}")
    (data / "boards.txt").write_text("\n".join(lines) + "\n")

    # assemble the client index.html (style + geometry + native UI)
    css = (REPO / "webapp" / "style.css").read_text()
    ui = (REPO / "webapp" / "native_ui.js").read_text()
    import json
    html = ("<!doctype html>\n<html lang=\"en\"><head>\n<meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>Koala</title>\n"
            "<style>\n" + css + "\n</style></head>\n<body>\n" + BODY +
            "\n<script>const BOARDSGEOM = " + json.dumps(geom) + ";\n"
            "const FAMILIES = " + json.dumps(families_struct()) + ";</script>\n"
            "<script>\n" + ui + "\n</script>\n</body></html>\n")
    (dist / "index.html").write_text(html)

    # compile the native server
    print("compiling cpp/server.cpp …", flush=True)
    subprocess.run(["clang++", "-O3", "-std=c++17", "-DACCELERATE_NEW_LAPACK", "-framework", "Accelerate",
                    str(REPO / "cpp" / "server.cpp"), "-o", str(dist / "TilingGo")], check=True)

    launcher = dist / "Play Koala.command"
    launcher.write_text(
        '#!/bin/bash\n'
        'D="$(cd "$(dirname "$0")" && pwd)"\n'
        'echo "Starting Koala (native C++ engine)…  close this window to stop."\n'
        '"$D/TilingGo" "$D" 8799 220 &\n'
        'SRV=$!\n'
        'sleep 1; open "http://127.0.0.1:8799/"\n'
        'wait $SRV\n')
    launcher.chmod(0o755)

    print(f"\nBuilt → {dist}")
    print("  Bundle contents: TilingGo (binary), index.html, data/, 'Play TilingGo.command'")
    print("  To play: open the folder and double-click 'Play TilingGo.command'")
    print("  To transfer: copy the whole 'native' folder to another Apple-Silicon Mac.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
