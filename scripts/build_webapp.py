#!/usr/bin/env python3
"""Inline style.css + data.js + engine.js + ui.js into ONE self-contained webapp/tilinggo.html.

Run scripts/export_webapp.py first (it regenerates data.js + the weights). This step is pure file
concatenation — no torch/scipy — so the result is a single double-clickable HTML that plays
Tiling-Go entirely in the browser, offline, no install.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "webapp"

BODY = """
<div class="stage"><div class="glow"></div>
  <div id="boardwrap"><div id="board"></div><div id="scan"></div></div>
</div>
<div class="winrail" title="Black win probability"><div class="winrail-fill" id="winfill"></div></div>
<div class="panel ctrl">
  <div class="brand"><div class="logo"></div><div><h1>EUCLIDEAN·GO</h1><p>plays in your browser</p></div></div>
  <label>Substrate</label><select id="tiling"></select>
  <label>Opponent</label>
  <select id="opponent">
    <option value="engine">Neural engine (champion)</option>
    <option value="off">Human (hot-seat)</option>
  </select>
  <label>Engine strength</label>
  <select id="strength"><option selected>Fast</option><option>Normal</option><option>Strong</option></select>
  <button id="analyze" class="cta">⌖ Analyze position</button>
  <div class="row"><button id="pass">Pass</button><button id="undo">Undo</button><button id="reset">New game</button></div>
  <label class="chk"><input type="checkbox" id="auto"> Auto-analyze each move</label>
  <label class="chk"><input type="checkbox" id="snd" checked> Stone sound</label>
</div>
<div class="panel info">
  <div class="turnrow"><span class="dot" id="turndot"></span><span id="turn">—</span>
    <span class="spacer"></span><span class="muted" id="move">move 0</span></div>
  <div class="bignum">
    <div><div class="lbl">Score</div><div class="num" id="score">—</div></div>
    <div><div class="lbl">Win&nbsp;B</div><div class="num acc" id="winpct">—</div></div>
  </div>
  <div class="topmv" id="topmv"></div>
  <div class="msg" id="msg"></div>
  <div class="lbl" style="margin-top:14px">Win-rate · Black</div>
  <div id="wrgraph"></div>
</div>
"""


def main() -> int:
    for f in ("style.css", "data.js", "engine.js", "ui.js"):
        if not (WEB / f).exists():
            print(f"missing {WEB/f} — run scripts/export_webapp.py first", file=sys.stderr)
            return 1
    css = (WEB / "style.css").read_text()
    data, engine, ui = (WEB / "data.js").read_text(), (WEB / "engine.js").read_text(), (WEB / "ui.js").read_text()
    html = (
        "<!doctype html>\n<html lang=\"en\"><head>\n<meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Euclidean Go</title>\n<style>\n" + css + "\n</style>\n</head>\n<body>\n"
        + BODY +
        "\n<script>\n" + data + "\n</script>\n"
        "<script>\n" + engine + "\n</script>\n"
        "<script>\n" + ui + "\n</script>\n"
        "</body></html>\n"
    )
    out = WEB / "tilinggo.html"
    out.write_text(html)
    print(f"wrote {out}  ({len(html)/1e6:.2f} MB — open it in any browser, no install)")
    # keep the GitHub Pages copy in sync (served at https://<user>.github.io/euclidean-go/)
    pages = WEB.parent / "docs" / "index.html"
    pages.parent.mkdir(exist_ok=True)
    pages.write_text(html)
    print(f"wrote {pages}  (GitHub Pages 'Play now' page)")
    # also drop a copy on the Desktop for convenience
    desktop = Path.home() / "Desktop" / "TilingGo.html"
    try:
        shutil.copyfile(out, desktop)
        print(f"copied to {desktop}")
    except Exception as e:
        print(f"(could not copy to Desktop: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
