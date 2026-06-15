#!/usr/bin/env python3
"""Assemble the single-file 3D page (diamond-cubic Go, rotatable three.js board).

Inlines: style.css + 3D extras, vendored three.js, data3d.js (boards + weights), engine.js,
the WASM worker bundle (same gated engine as the 2D app), wasm_glue, ui3d.js.
Run scripts/export_3d.py first. Outputs webapp/diamond3d.html + docs/3d.html.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
from brand_assets import LOGO_SVG, favicon_link  # noqa: E402
WEB = REPO / "webapp"

CSS3D = """
html, body { overflow:hidden; height:100%; }
#stage3d { position:fixed; inset:0; }
#stage3d canvas { display:block; touch-action:none; }   /* rotate ≠ scroll on touch */
#thinking { position:fixed; top:18px; right:22px; z-index:6; color:var(--accent); font-size:12px;
  letter-spacing:1.2px; text-transform:uppercase; opacity:0; transition:opacity .25s; }
.info3 { bottom:24px; right:24px; top:auto; left:auto; position:fixed; }

/* mobile: this is a fullscreen-canvas page — override the 2D app's stacked-panel rules.
   Compact pinned panels, board on top, everything reachable with a thumb. */
@media (max-width: 860px) {
  html, body { overflow:hidden; height:100%; }
  .panel.ctrl { position:fixed; top:10px; left:10px; right:10px; width:auto; margin:0;
    padding:10px 12px; max-height:44vh; overflow-y:auto; }
  .panel.ctrl .brand { margin-bottom:8px; }
  .panel.ctrl label { margin:8px 0 4px; }
  .panel.ctrl .perfnote, .panel.ctrl .ghlink { display:none; }   /* essentials only */
  .panel.info3 { position:fixed; top:auto; bottom:10px; left:10px; right:10px; width:auto;
    margin:0; padding:10px 14px; }
  .panel.info3 .bignum { display:none; }
  #hint { bottom:84px; }
}
"""

BODY = """
<div id="splash"><div class="splash-logo"></div><div class="splash-name">KOALA — 3D</div>
  <div class="splash-sub">loading the neural engine…</div></div>
<div id="stage3d"></div>
<div id="thinking">thinking…</div>
<div id="hint">drag to rotate · scroll to zoom · click a glowing point to play</div>
<div class="panel ctrl">
  <div class="brand"><div class="logo">__LOGO_SVG__</div><div><h1>KOALA <span style="color:var(--accent)">3D</span></h1><p>diamond-cubic lattice</p></div></div>
  <label>Board</label>
  <select id="board3"></select>
  <label>Opponent</label>
  <select id="opponent"><option value="engine">Neural engine (champion)</option><option value="off">Human (hot-seat)</option></select>
  <label>Engine strength</label>
  <select id="strength"><option selected>Fast</option><option>Normal</option><option>Strong</option></select>
  <div class="row"><button id="pass3">Pass</button><button id="undo3">Undo</button><button id="reset3">New game</button></div>
  <div class="perfnote" style="margin-top:14px">Every point has exactly <b>4 neighbours</b> — square-Go liberties, in three dimensions. The engine never trained on 3D: its planar knowledge is playing <b>zero-shot</b>.</div>
  <a class="ghlink" href="./index.html">← flat boards</a>
  <a class="ghlink" style="margin-left:12px" href="https://github.com/vonduffen/koala" target="_blank" rel="noopener">GitHub</a>
</div>
<div class="panel info info3">
  <div class="turnrow"><span class="dot" id="turndot3"></span><span id="turn3">—</span><span class="spacer"></span><span class="muted" id="move3">move 0</span></div>
  <div class="bignum"><div><div class="lbl">Score</div><div class="num" id="score3">—</div></div></div>
  <div class="msg" id="msg3"></div>
  <div class="lbl" id="engtag3" style="margin-top:8px">engine: JS</div>
</div>
"""


def main() -> int:
    for f in ("style.css", "data3d.js", "engine.js", "wasm_glue.js", "ui3d.js",
              "vendor/three.min.js"):
        if not (WEB / f).exists():
            print(f"missing {WEB/f}", file=sys.stderr)
            return 1
    css = (WEB / "style.css").read_text() + CSS3D
    three = (WEB / "vendor" / "three.min.js").read_text()
    data = (WEB / "data3d.js").read_text()
    engine = (WEB / "engine.js").read_text()
    glue = (WEB / "wasm_glue.js").read_text()
    ui = (WEB / "ui3d.js").read_text()
    tgwasm_p = WEB / "tgwasm.js"
    worker_src = ""
    if tgwasm_p.exists():
        worker_src = (tgwasm_p.read_text() + "\n" + glue + "\n"
                      + (WEB / "engine_worker.js").read_text()).replace("</script", "<\\/script")
    html = (
        "<!doctype html>\n<html lang=\"en\"><head>\n<meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        + favicon_link() +
        "<title>Koala — Go in 3D</title>\n<style>\n" + css + "\n</style>\n</head>\n<body>\n"
        + BODY.replace("__LOGO_SVG__", LOGO_SVG) +
        ("\n<script type=\"text/plain\" id=\"wasm-src\">\n" + worker_src + "\n</script>\n" if worker_src else "")
        + "\n<script>\n" + three + "\n</script>\n"
        "<script>\n" + data + "\n</script>\n"
        "<script>\n" + engine + "\n</script>\n"
        "<script>\n" + glue + "\n</script>\n"
        "<script>\n" + ui + "\n</script>\n"
        "</body></html>\n"
    )
    (WEB / "diamond3d.html").write_text(html)
    (REPO / "docs" / "3d.html").write_text(html)
    print(f"wrote webapp/diamond3d.html + docs/3d.html ({len(html)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
