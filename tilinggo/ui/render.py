"""Render a BoardGraph as a goban: the tiling's edges are the grid lines, stones sit on the
intersections (vertices).

This is the *only* place (besides the tiling compiler) allowed to look at geometry — see
ARCHITECTURE.md §1. ``to_svg`` draws the empty board (lines + optional vertex/index overlays);
``interactive_svg`` adds clickable intersections and stones for the play UI. Output is
deterministic given a BoardGraph (fixed coordinate formatting), so SVGs make stable goldens.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..tilings.boardgraph import BoardGraph

# y grows downward in SVG; tilings are defined with y up, so we flip vertically on output.


def _fmt(x: float) -> str:
    """Fixed 3-decimal formatting so golden SVGs are byte-stable across platforms."""
    return f"{x:.3f}"


def _layout(bg: BoardGraph, width: int, margin: float, pad_units: float = 0.0):
    """Compute the SVG transform (px functions, height, scale) from vertex coordinates.

    ``pad_units`` expands the data bounding box by that many board units on every side, leaving
    a wood border so stones drawn on edge intersections don't overflow the board.
    """
    pts = np.asarray(bg.coords, dtype=np.float64)
    min_x, min_y = pts.min(axis=0) - pad_units
    max_x, max_y = pts.max(axis=0) + pad_units
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    scale = (width - 2 * margin) / max(span_x, span_y)
    height = int(round(span_y * scale + 2 * margin))

    def tx(x):
        return margin + (x - min_x) * scale

    def ty(y):
        return margin + (max_y - y) * scale

    return tx, ty, height, scale


def _median_edge_len(bg: BoardGraph) -> float:
    """Typical edge length in board units (used to size stones)."""
    if bg.edges.shape[0] == 0:
        return 1.0
    a = bg.coords[bg.edges[:, 0]]
    b = bg.coords[bg.edges[:, 1]]
    return float(np.median(np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]))) or 1.0


def _min_spacing(bg: BoardGraph) -> float:
    """Smallest nearest-neighbour distance between *any* two vertices (robust low percentile).

    Stones must be sized by how close vertices actually get, not by edge length: on aperiodic
    tilings (e.g. thin Penrose rhombi) two *non-adjacent* vertices can sit much closer than an
    edge, so edge-length sizing makes stones overlap. The 2nd-percentile of per-vertex nearest-
    neighbour distances ignores rare near-coincident pairs while respecting systematic tight gaps.
    """
    pts = np.asarray(bg.coords, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        return _median_edge_len(bg)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.hypot(diff[:, :, 0], diff[:, :, 1])
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    spc = float(np.percentile(nn, 2))
    return spc if spc > 1e-9 else _median_edge_len(bg)


def to_svg(
    bg: BoardGraph,
    *,
    width: int = 800,
    margin: float = 24.0,
    background: str = "#e8c98a",
    line: str = "#5b4a2f",
    stroke_width: float = 1.4,
    show_vertices: bool = False,
    show_indices: bool = False,
) -> str:
    """Return an SVG of the empty board: tiling edges as grid lines on a wood background."""
    if bg.num_nodes == 0:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{width}"></svg>'

    tx, ty, height, scale = _layout(bg, width, margin)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{background}"/>',
    ]
    for a, b in bg.edges:
        ax, ay = bg.coords[int(a)]
        bx, by = bg.coords[int(b)]
        out.append(
            f'<line x1="{_fmt(tx(ax))}" y1="{_fmt(ty(ay))}" '
            f'x2="{_fmt(tx(bx))}" y2="{_fmt(ty(by))}" '
            f'stroke="{line}" stroke-width="{_fmt(stroke_width)}" stroke-linecap="round"/>'
        )
    if show_vertices or show_indices:
        for i, (x, y) in enumerate(bg.coords):
            px, py = tx(float(x)), ty(float(y))
            if show_vertices:
                out.append(f'<circle cx="{_fmt(px)}" cy="{_fmt(py)}" r="2.2" fill="{line}"/>')
            if show_indices:
                out.append(
                    f'<text x="{_fmt(px)}" y="{_fmt(py - 4)}" font-size="8" '
                    f'text-anchor="middle" fill="#202020">{i}</text>')
    out.append("</svg>")
    return "\n".join(out)


def render_to_file(bg: BoardGraph, path: str | Path, **kwargs) -> Path:
    """Render ``bg`` to ``path`` as SVG and return the path."""
    p = Path(path)
    if p.suffix != ".svg":
        p = p.with_suffix(".svg")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_svg(bg, **kwargs))
    return p


def interactive_svg(
    bg: BoardGraph,
    colors=None,
    *,
    last_move: int | None = None,
    legal=None,
    analysis=None,
    width: int = 760,
    margin: float = 26.0,
    theme: str = "dark",
) -> str:
    """Render a *playable* goban: grid lines, clickable intersections, and stones on vertices.

    ``colors`` is a length-N array (0=empty, 1=black, 2=white). Each intersection gets an
    invisible ``data-node`` hotspot for clicking; stones are drawn on occupied vertices, sized
    from the typical edge length so they look right on any tiling. ``last_move`` ringed,
    ``legal`` (bool over N) faintly marks playable points.
    """
    n = bg.num_nodes
    if colors is None:
        colors = np.zeros(n, dtype=np.int8)
    if n == 0:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{width}"></svg>'

    # theme palette (board is rendered server-side, so colours are chosen here)
    P = {
        "dark":  dict(bg="url(#bgs)", line="#454c5e", lineop="0.85", legal="#5a6478",
                      be="#000000", we="#aab0c0", last="#00ffc2", ownb="#0a0d10", ownw="#f4f4f2"),
        "light": dict(bg="#efe8d6", line="#3d4350", lineop="0.92", legal="#7a8290",
                      be="#14171d", we="#586071", last="#0a9c79", ownb="#2b3140", ownw="#9fb2d6"),
    }.get(theme, None) or {
        "bg": "url(#bgs)", "line": "#454c5e", "lineop": "0.85", "legal": "#5a6478",
        "be": "#000000", "we": "#aab0c0", "last": "#00ffc2", "ownb": "#0a0d10", "ownw": "#f4f4f2"}

    # size stones by the true nearest-neighbour spacing so they never overlap, even where
    # non-adjacent vertices crowd together (dense/aperiodic tilings like large Penrose patches)
    spc = _min_spacing(bg)
    tx, ty, height, scale = _layout(bg, width, margin, pad_units=0.55 * spc)
    r_stone = 0.46 * spc * scale
    r_hot = 0.5 * spc * scale

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<defs>'
        # black stone: cool specular highlight top-left → near-black (slate look)
        '<radialGradient id="bz" cx="0.36" cy="0.30" r="0.78">'
        '<stop offset="0%" stop-color="#4c5168"/><stop offset="40%" stop-color="#191c26"/>'
        '<stop offset="100%" stop-color="#04050a"/></radialGradient>'
        # white stone: bright highlight → soft shell grey (clamshell look)
        '<radialGradient id="wz" cx="0.36" cy="0.30" r="0.82">'
        '<stop offset="0%" stop-color="#ffffff"/><stop offset="56%" stop-color="#e6e9f1"/>'
        '<stop offset="100%" stop-color="#b9bfce"/></radialGradient>'
        # board surface: subtle dark vignette so the board reads as a material, not flat fill
        '<radialGradient id="bgs" cx="0.5" cy="0.40" r="0.85">'
        '<stop offset="0%" stop-color="#1a1f2c"/><stop offset="100%" stop-color="#0a0c12"/>'
        '</radialGradient>'
        '<filter id="sh" x="-40%" y="-40%" width="180%" height="180%">'
        '<feDropShadow dx="0" dy="1.4" stdDeviation="1.6" flood-color="#000" flood-opacity="0.6"/>'
        '</filter>'
        '<filter id="gl" x="-90%" y="-90%" width="280%" height="280%">'
        '<feDropShadow dx="0" dy="0" stdDeviation="3.2" flood-color="#00ffc2" flood-opacity="0.95"/>'
        '</filter>'
        '<filter id="grain"><feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" '
        'stitchTiles="stitch"/><feColorMatrix type="saturate" values="0"/></filter>'
        '</defs>'
        f'<rect width="{width}" height="{height}" rx="18" fill="{P["bg"]}"/>'
        f'<rect width="{width}" height="{height}" rx="18" filter="url(#grain)" opacity="0.045"/>',
    ]
    # grid lines (subtle cool-grey on the dark board)
    for a, b in bg.edges:
        ax, ay = bg.coords[int(a)]
        bx, by = bg.coords[int(b)]
        out.append(
            f'<line x1="{_fmt(tx(ax))}" y1="{_fmt(ty(ay))}" '
            f'x2="{_fmt(tx(bx))}" y2="{_fmt(ty(by))}" '
            f'stroke="{P["line"]}" stroke-width="1.1" stroke-linecap="round" opacity="{P["lineop"]}"/>'
        )
    # faint legal-move dots on empty intersections
    if legal is not None:
        for i in range(n):
            if colors[i] == 0 and legal[i]:
                x, y = bg.coords[i]
                out.append(
                    f'<circle class="legaldot" cx="{_fmt(tx(x))}" cy="{_fmt(ty(y))}" '
                    f'r="{_fmt(r_stone * 0.20)}" fill="{P["legal"]}" opacity="0.16"/>')
    # stones — gradient fills + soft drop shadow for depth
    for i in range(n):
        c = int(colors[i])
        if c == 0:
            continue
        x, y = bg.coords[i]
        fill = "url(#bz)" if c == 1 else "url(#wz)"
        edge = P["be"] if c == 1 else P["we"]
        out.append(
            f'<circle class="stone" cx="{_fmt(tx(x))}" cy="{_fmt(ty(y))}" r="{_fmt(r_stone)}" '
            f'fill="{fill}" stroke="{edge}" stroke-width="0.7" filter="url(#sh)"/>')
        if last_move == i:  # accent ring on the last move
            out.append(
                f'<circle cx="{_fmt(tx(x))}" cy="{_fmt(ty(y))}" r="{_fmt(r_stone * 0.40)}" '
                f'fill="none" stroke="{P["last"]}" stroke-width="2.2" filter="url(#gl)"/>')
    # analysis overlay (KataGo-style): ownership shading on empty points + candidate-move markers
    if analysis is not None:
        own = analysis.get("ownership")
        if own is not None:
            for i in range(n):
                if int(colors[i]) != 0:
                    continue
                o = float(own[i])
                if abs(o) < 0.18:
                    continue
                x, y = bg.coords[i]
                col = P["ownb"] if o > 0 else P["ownw"]   # black-leaning / white-leaning
                sq = r_stone * 1.15
                out.append(
                    f'<rect x="{_fmt(tx(x) - sq / 2)}" y="{_fmt(ty(y) - sq / 2)}" '
                    f'width="{_fmt(sq)}" height="{_fmt(sq)}" rx="2" fill="{col}" '
                    f'opacity="{_fmt(min(0.5, 0.55 * abs(o)))}"/>')
        best = analysis.get("best")
        for m in analysis.get("moves", []):
            i = int(m["node"])
            if i >= n:
                continue
            x, y = bg.coords[i]
            w = max(0.0, min(1.0, float(m["winrate"])))
            hue = int(round(135 * w))                      # 0=red (bad) → 135=green (good)
            is_best = (best == i)
            cls = ' class="bestmv"' if is_best else ""
            extra = ' filter="url(#gl)"' if is_best else ""
            out.append(
                f'<circle{cls} cx="{_fmt(tx(x))}" cy="{_fmt(ty(y))}" r="{_fmt(r_stone)}" '
                f'fill="hsl({hue},70%,45%)" opacity="0.94" '
                f'stroke="{"#ffffff" if is_best else "#0a0d10cc"}" '
                f'stroke-width="{_fmt(2.4 if is_best else 0.9)}"{extra}/>')
            out.append(
                f'<text x="{_fmt(tx(x))}" y="{_fmt(ty(y) + r_stone * 0.34)}" '
                f'font-size="{_fmt(r_stone * 0.82)}" fill="#ffffff" text-anchor="middle" '
                f'font-family="ui-monospace,Menlo,monospace" font-weight="700" '
                f'style="pointer-events:none">{int(round(w * 100))}</text>')
    # clickable hotspots on top (transparent), one per intersection
    for i in range(n):
        x, y = bg.coords[i]
        out.append(
            f'<circle class="hot" data-node="{i}" cx="{_fmt(tx(x))}" cy="{_fmt(ty(y))}" '
            f'r="{_fmt(r_hot)}" fill="transparent"/>')
    out.append("</svg>")
    return "\n".join(out)
