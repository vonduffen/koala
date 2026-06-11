#!/usr/bin/env python3
"""Generate the life-and-death regression suite (tests/lnd/*.json) — Task 7 Part A.

A 3-dan reviewer placed the engine's main weakness at eyes / life-and-death (~8-10k on
Penrose). This suite pins that with positions whose status is PROVABLE BY CONSTRUCTION on any
graph — no solver, no human labels:

  two_eyes      black group enclosing two separated one-point eyes → unconditionally ALIVE
  one_eye       black group fully sealed with a single one-point eye, black to move: every
                black board move is suicide (illegal), white captures next → DEAD
  capture_vital small white cluster inside black's area, in atari at ℓ, black to move:
                ℓ is the unique kill/clean-up move → expected move = ℓ
  eye3          black group enclosing a 3-node empty PATH (ends non-adjacent): the centre is
                the vital point BOTH ways — defender first → two eyes (ALIVE, move=centre);
                attacker first → one eye (DEAD, move=centre)

Each position records the target group, the expected status and/or expected move(s).
Deterministic (fixed seeds); run me again only to regenerate the committed suite.

    uv run python scripts/make_lnd_suite.py            # -> tests/lnd/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tilinggo.ui.server import _make_board  # noqa: E402

SUBSTRATES = ["hex_small", "trihex_small", "penrose_small"]
OUT = REPO / "tests" / "lnd"
EMPTY, BLACK, WHITE = 0, 1, 2


def neighbors(board):
    return [list(map(int, a)) for a in board.adj]


def grow_region(adj, seed, size, rng):
    """Randomized BFS growth — a connected region of ~size nodes."""
    region, frontier = {seed}, [seed]
    while len(region) < size and frontier:
        u = frontier[rng.integers(0, len(frontier))]
        nxt = [w for w in adj[u] if w not in region]
        if not nxt:
            frontier.remove(u)
            continue
        w = nxt[rng.integers(0, len(nxt))]
        region.add(w)
        frontier.append(w)
    return region


def interior(adj, region):
    return [u for u in region if all(w in region for w in adj[u])]


def boundary(adj, region):
    out = set()
    for u in region:
        out.update(w for w in adj[u] if w not in region)
    return out


def base_position(board, adj, region, eyes, wall_color=WHITE):
    """region−eyes black, boundary wall, rest empty. Returns colors or None if degenerate."""
    colors = np.zeros(board.n, dtype=np.int8)
    for u in region:
        colors[u] = BLACK
    for e in eyes:
        colors[e] = EMPTY
    for u in boundary(adj, region):
        colors[u] = wall_color
    # the black part must be ONE chain (otherwise "the group" is ill-defined)
    blacks = [u for u in region if colors[u] == BLACK]
    if not blacks:
        return None
    seen, stack = {blacks[0]}, [blacks[0]]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if colors[w] == BLACK and w not in seen:
                seen.add(w)
                stack.append(w)
    if len(seen) != len(blacks):
        return None
    return colors


def gen_two_eyes(board, adj, rng):
    for _ in range(400):
        region = grow_region(adj, int(rng.integers(0, board.n)), int(rng.integers(11, 16)), rng)
        ins = interior(adj, region)
        pairs = [(a, b) for i, a in enumerate(ins) for b in ins[i + 1:]
                 if b not in adj[a] and not (set(adj[a]) & set(adj[b]))]
        if not pairs:
            continue
        e1, e2 = pairs[rng.integers(0, len(pairs))]
        colors = base_position(board, adj, region, [e1, e2])
        if colors is None:
            continue
        return {"colors": colors.tolist(), "to_move": WHITE, "group": sorted(region),
                "expected": {"status": "alive"}, "eyes": [int(e1), int(e2)]}
    return None


def gen_one_eye(board, adj, rng):
    for _ in range(400):
        region = grow_region(adj, int(rng.integers(0, board.n)), int(rng.integers(9, 13)), rng)
        ins = interior(adj, region)
        if not ins:
            continue
        eye = ins[rng.integers(0, len(ins))]
        colors = base_position(board, adj, region, [eye])
        if colors is None:
            continue
        return {"colors": colors.tolist(), "to_move": BLACK, "group": sorted(region),
                "expected": {"status": "dead"}, "eyes": [int(eye)]}
    return None


def gen_capture_vital(board, adj, rng):
    """White cluster inside black's area, in atari at ℓ; black to move kills at ℓ."""
    for _ in range(600):
        region = grow_region(adj, int(rng.integers(0, board.n)), int(rng.integers(12, 17)), rng)
        ins = interior(adj, region)
        if len(ins) < 3:
            continue
        w0 = ins[rng.integers(0, len(ins))]
        cluster = {w0}
        for w in adj[w0]:                      # grow white to 2 nodes when possible
            if w in ins and rng.random() < 0.5:
                cluster.add(w)
                break
        libs = set()
        for u in cluster:
            libs.update(w for w in adj[u] if w not in cluster)
        if not libs.issubset(region):
            continue
        lib_list = sorted(libs)
        ell = lib_list[rng.integers(0, len(lib_list))]
        colors = np.zeros(board.n, dtype=np.int8)
        for u in region:
            colors[u] = BLACK
        for u in cluster:
            colors[u] = WHITE
        colors[ell] = EMPTY                    # white's single liberty
        for u in boundary(adj, region):
            colors[u] = EMPTY                  # open surroundings; capture is still the point
        # black chain(s) sanity: skip if the capture point is black's own last liberty drama
        return {"colors": colors.tolist(), "to_move": BLACK, "group": sorted(cluster),
                "expected": {"moves": [int(ell)]}}
    return None


def gen_eye3(board, adj, rng):
    """3-node empty path inside a black group, ends non-adjacent → centre is vital both ways."""
    for _ in range(800):
        region = grow_region(adj, int(rng.integers(0, board.n)), int(rng.integers(13, 18)), rng)
        ins = interior(adj, region)
        triples = []
        for c in ins:
            ends = [w for w in adj[c] if w in ins]
            for i, a in enumerate(ends):
                for b in ends[i + 1:]:
                    if b not in adj[a]:
                        triples.append((a, c, b))
        if not triples:
            continue
        a, c, b = triples[rng.integers(0, len(triples))]
        colors = base_position(board, adj, region, [a, c, b])
        if colors is None:
            continue
        return [
            {"colors": colors.tolist(), "to_move": BLACK, "group": sorted(region),
             "expected": {"status": "alive", "moves": [int(c)]}, "eyes": [int(a), int(c), int(b)],
             "variant": "defender_first"},
            {"colors": colors.tolist(), "to_move": WHITE, "group": sorted(region),
             "expected": {"status": "dead", "moves": [int(c)]}, "eyes": [int(a), int(c), int(b)],
             "variant": "attacker_first"},
        ]
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    total = 0
    for key in SUBSTRATES:
        board = _make_board(key, komi=5.5)
        adj = neighbors(board)
        rng = np.random.default_rng(0xEDA + len(key))
        made = {"two_eyes": 0, "one_eye": 0, "capture_vital": 0, "eye3": 0}
        idx = 0
        for kind, gen, want in (("two_eyes", gen_two_eyes, 3), ("one_eye", gen_one_eye, 3),
                                ("capture_vital", gen_capture_vital, 3), ("eye3", gen_eye3, 3)):
            got = 0
            while got < want:
                pos = gen(board, adj, rng)
                if pos is None:
                    print(f"  {key}/{kind}: generator exhausted at {got}/{want}", file=sys.stderr)
                    break
                for p in (pos if isinstance(pos, list) else [pos]):
                    p.update({"id": f"{key}_{kind}_{idx}", "board_key": key, "kind": kind})
                    (OUT / f"{p['id']}.json").write_text(json.dumps(p))
                    idx += 1
                    total += 1
                    made[kind] += 1
                got += 1
        print(f"{key}: {made}")
    print(f"\nwrote {total} positions → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
