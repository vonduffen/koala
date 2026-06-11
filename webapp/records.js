/* Game records — JSON "graph-SGF" (docs/game-record-format.md), shared schema with
   tilinggo/records.py. Plus standard SGF export for square boards only (SGF needs a
   coordinate grid; other substrates have none). */
(function (root) {
  "use strict";
  const FORMAT = "euclidean-go-record", VERSION = 1;

  function buildRecord(key, board, moves, meta) {
    meta = meta || {};
    return {
      format: FORMAT, version: VERSION,
      board: { key, fingerprint: root.SHARE.fingerprint(board), nodes: board.n },
      komi: board.komi, rules: "area",
      players: meta.players || {}, date: meta.date || null, result: meta.result || null,
      moves: moves.map(m => m === board.pass ? "pass" : m),
    };
  }

  /* Validate a parsed record object; returns a SHARE-style {key, fp, moves} ready for
     SHARE.replay / the app's installGame. Throws SHARE.ShareError with a typed code. */
  function toParsed(rec, TG, BOARDS) {
    const E = root.SHARE.ShareError;
    if (!rec || rec.format !== FORMAT) throw new E("BAD_FORMAT", "not a Euclidean Go record file");
    if (rec.version !== VERSION) throw new E("BAD_VERSION", `unsupported record version ${rec.version}`);
    const b = rec.board || {};
    if (typeof b.key !== "string" || !BOARDS[b.key])
      throw new E("BAD_BOARD", `this site doesn't have board "${b.key}"`);
    const board = TG.makeBoard(BOARDS[b.key]);
    if (b.fingerprint && b.fingerprint !== root.SHARE.fingerprint(board))
      throw new E("BAD_BOARD", "record was made for a different build of this board");
    if (b.graph && Array.isArray(b.graph.edges)) {           // embedded graph must agree
      const emb = b.graph.edges.flat();
      if (emb.length !== board.edges.length || emb.some((v, i) => v !== board.edges[i]))
        throw new E("BAD_BOARD", "embedded graph does not match the named substrate");
    }
    const moves = (rec.moves || []).map(m => {
      if (m === "pass") return root.SHARE.PASS;
      if (!Number.isInteger(m)) throw new E("BAD_FORMAT", `malformed move ${JSON.stringify(m)}`);
      return m;
    });
    return { version: 1, key: b.key, fp: root.SHARE.fingerprint(board), moves };
  }

  /* SGF (FF[4], Chinese/area rules) for square boards only. Coordinates come from the baked
     node positions: col = round(x), row = round(y); SGF letters are 'a'+index. */
  function toSGF(key, board, moves, meta) {
    const cols = board.coords.map(c => Math.round(c[0])), rows = board.coords.map(c => Math.round(c[1]));
    const w = Math.max(...cols) + 1, h = Math.max(...rows) + 1;
    const sz = (w === h) ? `${w}` : `${w}:${h}`;
    const L = i => String.fromCharCode(97 + i);
    let out = `(;FF[4]GM[1]CA[UTF-8]AP[euclidean-go]SZ[${sz}]RU[Chinese]KM[${board.komi}]`;
    if (meta && meta.result) out += `RE[${meta.result}]`;
    moves.forEach((m, i) => {
      const tag = (i % 2 === 0) ? "B" : "W";
      out += `;${tag}[` + (m === board.pass ? "" : L(cols[m]) + L(rows[m])) + "]";
    });
    return out + ")";
  }

  const REC = { FORMAT, VERSION, buildRecord, toParsed, toSGF };
  root.REC = REC;
  if (typeof module !== "undefined") module.exports = REC;
})(typeof globalThis !== "undefined" ? globalThis : this);
