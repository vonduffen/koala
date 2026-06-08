"""Tests for board symmetry detection (graph automorphisms for data augmentation).

Every clipped patch is symmetric because it is clipped about a symmetry centre: achiral tilings
give dihedral groups (rotations + mirrors), the chiral snubs give cyclic groups (rotations only).
"""

from __future__ import annotations

import numpy as np
import pytest

from tilinggo.tilings import periodic, symmetry, uniform

# (rotations, reflections) expected for each tiling's default patch.
EXPECTED = {
    "square": (4, 4),         # D4
    "triangular": (6, 6),     # D6
    "hexagonal": (6, 6),      # D6
    "trunc_square": (4, 4),   # D4
    "trunc_hex": (6, 6),      # D6
    "trihexagonal": (6, 6),   # D6
    "elongated_tri": (2, 2),  # D2
    "rhombitrihex": (6, 6),   # D6
    "trunc_trihex": (6, 6),   # D6
    "snub_square": (4, 0),    # C4 — chiral
    "snub_hex": (6, 0),       # C6 — chiral
}


@pytest.mark.parametrize("name,counts", EXPECTED.items())
def test_symmetry_group(name, counts):
    bg = uniform.generate(name)
    rotations, reflections = symmetry.split_by_orientation(bg)
    assert (len(rotations), len(reflections)) == counts


@pytest.mark.parametrize("name", list(EXPECTED))
def test_symmetries_are_graph_automorphisms(name):
    bg = uniform.generate(name)
    perms = symmetry.symmetries(bg)
    assert len(perms) >= 2  # at least identity + one nontrivial symmetry
    for perm in perms:
        assert np.array_equal(np.sort(perm), np.arange(bg.num_nodes))  # a bijection
        assert symmetry.is_automorphism(bg, perm)                      # preserves adjacency


@pytest.mark.parametrize("name", ["snub_square", "snub_hex"])
def test_chiral_tilings_have_no_mirrors(name):
    # Augmenting these with reflections would manufacture invalid (mirror-world) data; the
    # detector must report them chiral so symmetries() yields rotations only.
    assert symmetry.is_chiral(uniform.generate(name))


@pytest.mark.parametrize("name", ["square", "hexagonal", "trihexagonal"])
def test_achiral_tilings_have_mirrors(name):
    assert not symmetry.is_chiral(uniform.generate(name))


def test_symmetry_consistent_across_radii():
    # The clip tolerance keeps boundary orbits whole, so the group order is radius-independent.
    for r in (3.0, 4.0, 5.0, 6.0, 7.0):
        bg = uniform.generate("triangular", radius=r)
        assert len(symmetry.symmetries(bg)) == 12


def test_identity_is_first():
    bg = periodic.rectangular(9, 9)
    perms = symmetry.symmetries(bg)
    assert np.array_equal(perms[0], np.arange(bg.num_nodes))
    # the 9x9 goban has the full D4 symmetry of the square
    rot, ref = symmetry.split_by_orientation(bg)
    assert (len(rot), len(ref)) == (4, 4)
