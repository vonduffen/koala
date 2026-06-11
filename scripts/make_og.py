#!/usr/bin/env python3
"""Render docs/og.png — the 1200x630 social-share card (og:image / twitter:card).

A real mid-game position on a Penrose board (random legal self-play, no net needed), composed
with the brand and tagline. Reproducible: python scripts/make_og.py

Requires rsvg-convert + Pillow.
"""

from __future__ import annotations

import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tilinggo.rules import Board  # noqa: E402
from tilinggo.tilings import penrose  # noqa: E402
from tilinggo.ui import render  # noqa: E402

W, H = 1200, 630
BG = (8, 9, 13)
INK = (232, 235, 242)
MUTED = (130, 140, 156)
ACCENT = (0, 255, 194)

_F = "/System/Library/Fonts/Helvetica.ttc"
_FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(_FB if bold else _F, size)
    except Exception:
        return ImageFont.load_default()


def main() -> int:
    board = Board(penrose.generate(radius=5.0, symmetric=True), komi=5.5)
    state = board.new_game()
    rng = random.Random(7)
    for _ in range(34):                                   # natural-looking random legal game
        legal = np.flatnonzero(state.legal_moves())
        legal = [m for m in legal if m != board.pass_move]
        if not legal:
            break
        state = state.play(rng.choice(legal))

    svg = render.interactive_svg(board.graph, state.colors, width=620, theme="dark")
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "b.svg"
        f.write_text(svg)
        png = Path(td) / "b.png"
        subprocess.run(["rsvg-convert", "-w", "620", "-o", str(png), str(f)], check=True)
        board_img = Image.open(png).convert("RGBA")

    img = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([-160, -40, 660, 720], fill=(18, 38, 34, 110))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")

    bw, bh = board_img.size
    img.paste(board_img, (34, (H - bh) // 2), board_img)
    d = ImageDraw.Draw(img)

    x = 710
    # logo chip
    d.rounded_rectangle([x, 158, x + 44, 202], radius=13, fill=(0, 224, 188))
    d.text((x + 60, 162), "EUCLIDEAN·GO", font=font(40, True), fill=INK)
    d.text((x, 232), "Go on every tiling.", font=font(30, True), fill=ACCENT)
    for i, line in enumerate(["One geometry-blind neural net plays",
                              "Penrose, hexagonal, snub and 13 tiling",
                              "families — entirely in your browser."]):
        d.text((x, 292 + i * 32), line, font=font(21), fill=MUTED)
    # CTA pill
    d.rounded_rectangle([x, 420, x + 290, 472], radius=26, fill=(0, 224, 188))
    d.polygon([(x + 38, 436), (x + 38, 456), (x + 54, 446)], fill=(4, 18, 14))   # play triangle
    d.text((x + 160, 446), "Play it now — free", font=font(20, True), fill=(4, 18, 14),
           anchor="mm")
    d.text((x, 496), "vonduffen.github.io/euclidean-go", font=font(16), fill=MUTED)

    out = REPO / "docs" / "og.png"
    img.save(out, optimize=True)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
