# Koala game-record format ("graph-SGF") — v1

SGF presumes a coordinate grid; most Koala boards have none. A game here is a sequence
of **node indices** into a named board graph, so the record names the board, fingerprints its
graph, and lists the moves. One JSON schema is read and written by both implementations:

- JavaScript: `webapp/records.js` (Download/Load buttons in the app)
- Python: `tilinggo/records.py` (`save_record` / `load_record`)

Cross-language identity is enforced by `tests/test_records_cross.py`: a record written by one
side replays on the other to the identical final position.

## Schema

```json
{
 "format": "euclidean-go-record",
 "version": 1,
 "board": {
  "key": "penrose_medium",
  "fingerprint": "psvkiz",
  "nodes": 86,
  "graph": { "edges": [[0, 1], [0, 5]] }
 },
 "komi": 5.5,
 "rules": "area",
 "players": { "black": "alex", "white": "engine (champion)" },
 "date": "2026-06-10",
 "result": "B+3.5",
 "moves": [34, 17, "pass", 52]
}
```

| field | meaning |
|---|---|
| `format`, `version` | literal `"euclidean-go-record"`, integer version. Unknown versions are rejected, never guessed at. |
| `board.key` | catalogue key (family_size, e.g. `square_small`, `penrose_medium`). |
| `board.fingerprint` | FNV-1a (32-bit) over node count + flattened edge list, base36, ≤6 chars. Identical algorithm in `webapp/share.js` and `tilinggo/records.py`; guards against a record made for a *different build* of the same-named board (node indices would silently shift). |
| `board.nodes` | node count, informational. |
| `board.graph` | *optional* embedded edge list (`[[u,v], …]` in canonical order) for forward-compatibility with custom graphs. When present, loaders must verify it equals the named substrate's compiled graph and fail clearly otherwise. |
| `komi`, `rules` | `5.5`, `"area"` (Tromp-Taylor area scoring, positional superko) in v1. |
| `players`, `date`, `result` | optional metadata; `result` like `"B+3.5"`, `null` if unfinished. |
| `moves` | plies in play order: integer node index (0-based, canonical node ordering) or the string `"pass"`. |

## Canonical node ordering

Node indices are only meaningful against the board's canonical node ordering, which the tiling
compiler guarantees is deterministic per (tiling, size) — pinned by
`tests/test_node_ordering.py`. The fingerprint exists because that guarantee spans *one build*:
regenerating the webapp's baked boards with a changed compiler may reorder nodes, and the
fingerprint turns that into an explicit `BAD_BOARD` error.

## SGF export (square boards only)

The webapp additionally exports standard SGF (FF[4], `RU[Chinese]`, area komi) for `square_*`
boards, mapping node *i* → column `round(x_i)`, row `round(y_i)` → SGF letter pair. Non-square
substrates have no SGF representation and none is attempted.

## Failure semantics

Loaders reject with a typed error (never silently mis-replay): `BAD_FORMAT` (not a record /
malformed move), `BAD_VERSION`, `BAD_BOARD` (unknown key, fingerprint mismatch, embedded-graph
mismatch), `BAD_MOVE` (index out of range), `ILLEGAL_MOVE` (violates the rules at its ply).
