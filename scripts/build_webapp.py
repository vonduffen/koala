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
<div id="splash"><div class="splash-logo"></div><div class="splash-name">EUCLIDEAN·GO</div>
  <div class="splash-sub">loading the neural engine…</div></div>
<div class="stage"><div class="glow"></div>
  <div id="boardwrap"><div id="board"></div><div id="scan"></div></div>
</div>
<div id="hint">tap any intersection to place a stone</div>
<div class="panel ctrl">
  <div class="brand"><div class="logo"></div><div><h1>EUCLIDEAN·GO</h1><p>plays in your browser</p></div></div>
  <label>Substrate</label>
  <select id="family"></select>
  <select id="variant" style="margin-top:6px"></select>
  <button id="random" style="margin-top:6px">🎲 Random board</button>
  <label>Opponent</label>
  <select id="opponent">
    <option value="engine">Neural engine (champion)</option>
    <option value="off">Human (hot-seat)</option>
  </select>
  <label>Your colour</label>
  <select id="playercolor">
    <option value="black">Black (move first)</option>
    <option value="white">White (engine opens)</option>
  </select>
  <label>Engine strength</label>
  <select id="strength"><option selected>Fast</option><option>Normal</option><option>Strong</option></select>
  <button id="analyze" class="cta">⌖ Analyze position</button>
  <div class="row"><button id="pass">Pass</button><button id="undo">Undo</button><button id="reset">New game</button></div>
  <button id="share" style="margin-top:8px">🔗 Copy game link</button>
  <div class="row"><button id="dlrec" title="Download the game as a JSON record">⬇ Record</button><button id="dlsgf" title="Download as SGF (square boards)">⬇ SGF</button><button id="ldrec" title="Load a saved record">⬆ Load</button></div>
  <input type="file" id="recfile" accept=".json,application/json" style="display:none">
  <label class="chk"><input type="checkbox" id="auto"> Auto-analyze each move</label>
  <label class="chk"><input type="checkbox" id="snd" checked> Stone sound</label>
  <label class="chk"><input type="checkbox" id="light"> ☀ Light mode</label>
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
  <div class="lbl" style="margin-top:14px">Engine performance</div>
  <div id="perf"></div>
  <div class="perfnote" title="Measured on a 9×9 board at equal MCTS: ≈870 sims/s native vs ≈19 sims/s in-browser. The web engine is pure JavaScript; the native one is C++/Accelerate.">⚡ This runs in pure JavaScript — the <b>native macOS build</b> plays the same engine <b>40× faster</b>.</div>
  <a class="ghlink" href="https://github.com/vonduffen/euclidean-go" target="_blank" rel="noopener">
    <svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
    open-source on GitHub</a>
</div>
"""


def main() -> int:
    for f in ("style.css", "data.js", "engine.js", "share.js", "ui.js"):
        if not (WEB / f).exists():
            print(f"missing {WEB/f} — run scripts/export_webapp.py first", file=sys.stderr)
            return 1
    css = (WEB / "style.css").read_text()
    data, engine, ui = (WEB / "data.js").read_text(), (WEB / "engine.js").read_text(), (WEB / "ui.js").read_text()
    share = (WEB / "share.js").read_text() + "\n" + (WEB / "records.js").read_text()
    html = (
        "<!doctype html>\n<html lang=\"en\"><head>\n<meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Euclidean Go</title>\n<style>\n" + css + "\n</style>\n</head>\n<body>\n"
        + BODY +
        "\n<script>\n" + data + "\n</script>\n"
        "<script>\n" + engine + "\n</script>\n"
        "<script>\n" + share + "\n</script>\n"
        "<script>\n" + ui + "\n</script>\n"
        "</body></html>\n"
    )
    out = WEB / "tilinggo.html"
    out.write_text(html)   # portable single-file: NO analytics (stays fully offline)
    print(f"wrote {out}  ({len(html)/1e6:.2f} MB — open it in any browser, no install)")
    # GitHub Pages copy: same app + a privacy-friendly (cookieless) GoatCounter snippet, so we can
    # see visits + a "game started" custom event. Only the HOSTED copy phones home, never the download.
    gc = ('<script data-goatcounter="https://vonduffen.goatcounter.com/count" '
          'async src="//gc.zgo.at/count.js"></script>\n')
    # social-share card (og/twitter) — without these a Reddit/HN/Discord link renders bare
    og = (
        '<meta name="description" content="Play Go on Penrose, hexagonal, snub and 13 families of '
        'Euclidean tilings — against a neural engine that runs entirely in your browser.">\n'
        '<meta property="og:title" content="Euclidean Go — Go on every tiling">\n'
        '<meta property="og:description" content="One geometry-blind neural net plays Go on Penrose, '
        'hexagonal, snub and 13 tiling families — entirely in your browser. No install.">\n'
        '<meta property="og:image" content="https://vonduffen.github.io/euclidean-go/og.png">\n'
        '<meta property="og:url" content="https://vonduffen.github.io/euclidean-go/">\n'
        '<meta property="og:type" content="website">\n'
        '<meta name="twitter:card" content="summary_large_image">\n')
    pages_html = html.replace("</head>", og + gc + "</head>", 1)
    pages = WEB.parent / "docs" / "index.html"
    pages.parent.mkdir(exist_ok=True)
    pages.write_text(pages_html)
    print(f"wrote {pages}  (GitHub Pages 'Play now' page, with analytics)")
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
