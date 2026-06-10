#!/usr/bin/env python3
"""Render the README hero GIF: a real champion self-play game on a Penrose (aperiodic, 5-fold)
tiling, with the live win-rate timeline graph and the engine-performance analyzer beside it.

Reproducible — no screen recording. Plays the actual trained net, rasterises each server-side
board SVG with `rsvg-convert`, composes the panel chrome with Pillow, and encodes the GIF with
ffmpeg (palettegen for small, clean output).

    python scripts/make_demo_gif.py            # -> docs/demo.gif
    python scripts/make_demo_gif.py --key square_medium --plies 30 --out docs/demo.gif

Requires: rsvg-convert, ffmpeg on PATH; Pillow.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tilinggo.ui import render, server  # noqa: E402

# ---- look -----------------------------------------------------------------
W, H = 980, 600
BOARD_PX = 540
PANEL_X = 596
PANEL_W = W - PANEL_X - 28
BG = (9, 11, 16)
CARD = (11, 13, 19)
EDGE = (42, 49, 60)
MUTED = (123, 134, 152)
INK = (228, 232, 240)
ACCENT = (0, 255, 194)
GRAPH_H = 122
SPARK_H = 46

_F = "/System/Library/Fonts/Helvetica.ttc"
_FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(_FB if bold else _F, size)
    except Exception:
        return ImageFont.load_default()


def rsvg(svg: str, px: int, tmp: Path) -> Image.Image:
    f = tmp / "b.svg"
    f.write_text(svg)
    out = tmp / "b.png"
    subprocess.run(["rsvg-convert", "-w", str(px), "-o", str(out), str(f)], check=True)
    return Image.open(out).convert("RGBA")


def text(d, xy, s, fnt, fill, anchor="la", spacing=0):
    if spacing and len(s) > 1:  # poor-man letter-spacing for small caps labels
        x, y = xy
        for ch in s:
            d.text((x, y), ch, font=fnt, fill=fill, anchor="la")
            x += d.textlength(ch, font=fnt) + spacing
        return
    d.text(xy, s, font=fnt, fill=fill, anchor=anchor)


def draw_graph(d, x, y, w, h, hist, cur):
    d.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=CARD, outline=EDGE, width=1)
    mid = y + h / 2
    for sx in range(int(x) + 8, int(x + w) - 4, 11):  # dashed 50% line
        d.line([sx, mid, sx + 5, mid], fill=EDGE, width=1)
    pts = [(i, v) for i, v in enumerate(hist) if v is not None]
    if len(pts) >= 2:
        n = max(len(hist) - 1, 1)
        px = lambda i: x + 8 + (i / n) * (w - 16)
        py = lambda v: y + 8 + (1 - v) * (h - 16)
        poly = [(px(i), py(v)) for i, v in pts]
        # soft fill to the midline
        fillpoly = poly + [(poly[-1][0], mid), (poly[0][0], mid)]
        d.polygon(fillpoly, fill=(0, 60, 48))
        d.line(poly, fill=ACCENT, width=2, joint="curve")
        lx, lv = pts[-1]
        d.ellipse([px(lx) - 3.5, py(lv) - 3.5, px(lx) + 3.5, py(lv) + 3.5], fill=ACCENT)
        pct = round(lv * 100)
        text(d, (x + w - 8, y + 6), f"{pct}%", font(15, True), ACCENT, anchor="ra")


def draw_spark(d, x, y, w, h, hist):
    d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=CARD, outline=EDGE, width=1)
    if len(hist) >= 2:
        mx = max(hist) or 1
        n = len(hist) - 1
        px = lambda i: x + 6 + (i / n) * (w - 12)
        py = lambda v: y + 5 + (1 - v / mx) * (h - 10)
        d.line([(px(i), py(v)) for i, v in enumerate(hist)], fill=ACCENT, width=2, joint="curve")


def compose(board_png, bwr_hist, perf, spark_hist, cur, label, tmp):
    img = Image.new("RGB", (W, H), BG)
    # subtle glow behind the board
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([-120, 60, 600, 720], fill=(20, 36, 33, 90))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    d = ImageDraw.Draw(img)

    bw, bh = board_png.size
    bx = 24 + (BOARD_PX - bw) // 2
    by = (H - bh) // 2
    img.paste(board_png, (bx, max(20, by)), board_png)
    d = ImageDraw.Draw(img)

    # brand
    text(d, (PANEL_X, 54), "EUCLIDEAN·GO", font(30, True), INK)
    text(d, (PANEL_X, 92), "Go on every Euclidean tiling", font(14), MUTED)

    y = 142
    text(d, (PANEL_X, y), "WIN-RATE  ·  BLACK", font(11, True), MUTED, spacing=1.5)
    draw_graph(d, PANEL_X, y + 20, PANEL_W, GRAPH_H, bwr_hist[: cur + 1], cur)

    y2 = y + 20 + GRAPH_H + 26
    text(d, (PANEL_X, y2), "ENGINE PERFORMANCE", font(11, True), MUTED, spacing=1.5)
    sps = (perf or {}).get("sps", 0)
    big = font(46, True)
    d.text((PANEL_X, y2 + 18), f"{sps:,}", font=big, fill=INK)
    nw = d.textlength(f"{sps:,}", font=big)
    d.text((PANEL_X + nw + 10, y2 + 44), "sims/s", font=font(16), fill=ACCENT, anchor="ls")
    sub = "—"
    if perf:
        sub = f"{perf['sims']} sims  ·  {perf['ms']} ms/move  ·  {perf['n']} nodes"
    text(d, (PANEL_X, y2 + 74), sub, font(13), MUTED)
    draw_spark(d, PANEL_X, y2 + 98, PANEL_W, SPARK_H, spark_hist)

    # footer
    text(d, (PANEL_X, H - 46), label, font(13, True), INK)
    text(d, (PANEL_X, H - 26), f"move {cur}", font(13), MUTED)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="penrose_medium")
    ap.add_argument("--plies", type=int, default=30)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--out", default="docs/demo.gif")
    a = ap.parse_args()

    label_map = {"penrose": "Penrose · 5-fold aperiodic", "square": "Square lattice",
                 "hex": "Hexagonal", "tri": "Triangular"}
    fam = a.key.split("_")[0]
    label = next((v for k, v in label_map.items() if a.key.startswith(k)), server._LABELS.get(a.key, a.key))

    g = server.Game(a.key)
    g.theme = "dark"
    print(f"[gif] {a.key}  N={g.board.n}  plies≤{a.plies}")

    bwr_hist, spark_hist, frames = [], [], []  # frames: (svg, perf, cur)

    def snap():
        bwr_hist.append(g._black_winrate())
        if g.last_perf:
            spark_hist.append(g.last_perf["sps"])
        svg = render.interactive_svg(g.board.graph, g.state.colors, last_move=g.last_move,
                                     width=BOARD_PX, theme="dark")
        frames.append((svg, g.last_perf, g.state.move_num if hasattr(g.state, "move_num") else len(frames)))

    snap()  # empty board
    for ply in range(a.plies):
        prev = len(g.history)
        g.neural_move(simulations=a.sims)
        if len(g.history) == prev or g.state.is_terminal:
            snap()
            break
        snap()
        print(f"  ply {ply + 1:2d}/{a.plies}  bwr={bwr_hist[-1]:.2f}  {g.last_perf['sps']:,} sims/s")

    print(f"[gif] composing {len(frames)} frames → rasterise + ffmpeg")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fdir = tmp / "f"; fdir.mkdir()
        for i, (svg, perf, _) in enumerate(frames):
            bpng = rsvg(svg, BOARD_PX, tmp)
            img = compose(bpng, bwr_hist, perf, spark_hist[: max(1, i)], i, label, tmp)
            img.save(fdir / f"{i:03d}.png")
        # hold the last frame
        last = Image.open(fdir / f"{len(frames) - 1:03d}.png")
        for j in range(6):
            last.save(fdir / f"{len(frames) + j:03d}.png")

        out = (REPO / a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pal = tmp / "pal.png"
        vf = "fps=4,scale=980:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(fdir / "%03d.png"),
                        "-vf", f"{vf},palettegen=max_colors=160:stats_mode=full", str(pal)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4",
                        "-i", str(fdir / "%03d.png"), "-i", str(pal),
                        "-lavfi", f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
                        "-loop", "0", str(out)], check=True)
    kb = out.stat().st_size / 1024
    print(f"[gif] wrote {out}  ({kb:.0f} KB, {len(frames)} frames + hold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
