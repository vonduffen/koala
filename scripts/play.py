#!/usr/bin/env python3
"""Launch the local web UI to play graph Go on any tiling (ARCHITECTURE.md §8).

    uv run python scripts/play.py [--port 8770] [--no-open]

Opens a browser pointed at a click-to-play board. There is no neural-net opponent yet
(Milestones 4–5); play human-vs-human, or tick "Opponent plays random moves".
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tilinggo.ui import server  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)

    httpd = server.serve(args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"Tiling-Go play UI running at {url}  (Ctrl-C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
