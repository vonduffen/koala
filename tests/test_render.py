"""Tests for the SVG renderer (line-drawing / goban model)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from tilinggo.tilings import periodic
from tilinggo.ui import render


def _count_tag(svg: str, tag: str) -> int:
    root = ET.fromstring(svg)
    return sum(1 for el in root.iter() if el.tag.split("}")[-1] == tag)


def test_svg_is_well_formed_xml():
    bg = periodic.generate("square", cells=50, seed=0)
    ET.fromstring(render.to_svg(bg))  # raises on malformed XML


def test_one_line_per_edge():
    for family in ["square", "hex", "tri"]:
        bg = periodic.generate(family, cells=50, seed=0)
        svg = render.to_svg(bg)
        assert _count_tag(svg, "line") == bg.edges.shape[0]


def test_vertex_and_index_overlays():
    bg = periodic.generate("hex", cells=40, seed=0)
    plain = render.to_svg(bg)
    decorated = render.to_svg(bg, show_vertices=True, show_indices=True)
    assert _count_tag(decorated, "circle") == bg.num_nodes
    assert _count_tag(decorated, "text") == bg.num_nodes
    assert len(decorated) > len(plain)


def test_interactive_svg_has_hotspots_and_stones():
    bg = periodic.generate("square", cells=40, seed=0)
    colors = np.zeros(bg.num_nodes, dtype=np.int8)
    colors[0] = 1  # black
    colors[1] = 2  # white
    svg = render.interactive_svg(bg, colors, last_move=0, legal=np.ones(bg.num_nodes, bool))
    root = ET.fromstring(svg)
    hotspots = [el for el in root.iter() if el.attrib.get("class") == "hot"]
    assert len(hotspots) == bg.num_nodes
    assert all("data-node" in el.attrib for el in hotspots)
    # at least the two stones plus the last-move ring are drawn as circles
    assert _count_tag(svg, "circle") >= bg.num_nodes + 2


def test_render_to_file(tmp_path):
    bg = periodic.generate("tri", cells=40, seed=0)
    out = render.render_to_file(bg, tmp_path / "board")
    assert out.exists() and out.suffix == ".svg"
    ET.fromstring(out.read_text())


def test_render_is_deterministic():
    bg = periodic.generate("square", cells=50, seed=0)
    assert render.to_svg(bg) == render.to_svg(bg)
