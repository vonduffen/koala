#!/usr/bin/env python3
"""Score a checkpoint on the life-and-death suite (tests/lnd/) — Task 7 Part A.

Two question types:
  status  — does the net's ownership head call the target group correctly?
            signed ownership = P(black) − P(white), mean over the group's nodes;
            predicted ALIVE iff the sign favours the group's colour.
  move    — is the expected vital point the net's top policy choice? (top-1; top-3 also
            reported as a softer signal)

This is a tracked metric, not a gate: it always exits 0 unless --strict. Run:

    uv run python scripts/lnd_score.py [--ckpt results/universal/champion.pt] [--strict N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tilinggo.nn import encoding  # noqa: E402
from tilinggo.rules import GoState  # noqa: E402
from tilinggo.sit.checkpoint import load_checkpoint  # noqa: E402
from tilinggo.ui.server import _make_board  # noqa: E402

BLACK, WHITE = 1, 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="results/universal/champion.pt")
    ap.add_argument("--suite", default=str(REPO / "tests" / "lnd"))
    ap.add_argument("--strict", type=float, default=None,
                    help="exit 1 if overall accuracy falls below this fraction")
    args = ap.parse_args()

    net, _ = load_checkpoint(args.ckpt)
    net.eval()
    files = sorted(Path(args.suite).glob("*.json"))
    if not files:
        print("no suite files found", file=sys.stderr)
        return 1

    boards = {}
    stats = defaultdict(lambda: [0, 0])      # (substrate, kind, qtype) -> [correct, total]
    top3 = [0, 0]
    for f in files:
        p = json.loads(f.read_text())
        key = p["board_key"]
        if key not in boards:
            boards[key] = _make_board(key, komi=5.5)
        board = boards[key]
        state = GoState(board, colors=np.asarray(p["colors"], dtype=np.int8),
                        to_move=p["to_move"])
        with torch.no_grad():
            out = net.forward(encoding.encode_states([state]))

        exp = p["expected"]
        if "status" in exp:
            own = torch.softmax(out["ownership"][0], dim=-1).numpy()   # [N,3] B/W/neutral
            signed = float(np.mean(own[p["group"], 0] - own[p["group"], 1]))
            group_color_black = any(np.asarray(p["colors"])[p["group"]] == BLACK) and \
                not all(np.asarray(p["colors"])[p["group"]] == WHITE)
            favours_group = signed > 0 if group_color_black else signed < 0
            pred = "alive" if favours_group else "dead"
            s = stats[(key, p["kind"], "status")]
            s[1] += 1
            s[0] += int(pred == exp["status"])
        if "moves" in exp:
            logits = out["policy_logits"][0].numpy()
            legal = state.legal_moves()
            logits = np.where(legal, logits, -1e30)
            order = np.argsort(-logits)
            s = stats[(key, p["kind"], "move")]
            s[1] += 1
            s[0] += int(order[0] in exp["moves"])
            top3[1] += 1
            top3[0] += int(any(m in exp["moves"] for m in order[:3]))

    print(f"L&D suite — {len(files)} positions, checkpoint {args.ckpt}")
    print(f"{'substrate':14s}{'kind':16s}{'qtype':8s}{'acc':>10s}")
    tot = [0, 0]
    for (key, kind, qt), (c, n) in sorted(stats.items()):
        print(f"{key:14s}{kind:16s}{qt:8s}{c:>5d}/{n:<4d}{c/n*100:5.0f}%")
        tot[0] += c
        tot[1] += n
    print("-" * 52)
    print(f"{'OVERALL':38s}{tot[0]:>5d}/{tot[1]:<4d}{tot[0]/tot[1]*100:5.0f}%")
    if top3[1]:
        print(f"{'(move questions, top-3)':38s}{top3[0]:>5d}/{top3[1]:<4d}{top3[0]/top3[1]*100:5.0f}%")
    if args.strict is not None and tot[0] / tot[1] < args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
