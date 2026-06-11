/* Game serialization for shareable URLs — format v1.

   Fragment grammar (plain text, human-debuggable):

       #g=1.<boardKey>.<fp>.<moves>

   - 1          format version (integer)
   - boardKey   catalogue key, e.g. penrose_medium   [a-z0-9_]+
   - fp         board-graph fingerprint: FNV-1a over node count + edge list, base36.
                The webapp's graphs are baked into data.js, so node ordering is frozen per
                artifact; the fingerprint detects a *regenerated* site whose compiler produced
                a different ordering, turning silent corruption into a clear error.
   - moves      dot-separated plies in play order; node index in base36, or "-" for pass
                ("-" is outside the base36 alphabet, so it can never collide with a node index
                — "p" would silently alias node 25).

   Size: a 100-move game ≈ 330 chars, a 400-move 19×19 game ≈ 1.3k — under the ~1.5k URL
   budget. Plain text over binary packing is deliberate: the compactness target is already
   met, and a link you can read at a glance is worth more than the ~40% varint would save.

   Komi and rules are baked into the board data (komi 5.5, area scoring); a change there is a
   data/format change and must bump the version byte. */
(function (root) {
  "use strict";
  const VERSION = 1;
  const PASS = -1; // internal sentinel for a pass ply

  class ShareError extends Error {
    constructor(code, message) { super(message); this.code = code; }
  }

  function fingerprint(board) {
    let h = 0x811c9dc5;
    const mix = v => { h ^= v & 0xff; h = Math.imul(h, 0x01000193); h >>>= 0; };
    const mixInt = v => { mix(v); mix(v >> 8); mix(v >> 16); mix(v >> 24); };
    mixInt(board.n);
    for (let i = 0; i < board.edges.length; i++) mixInt(board.edges[i]);
    return h.toString(36).slice(0, 6);
  }

  function serialize(boardKey, board, moves) {
    const toks = moves.map(m => (m === PASS || m === board.pass) ? "-" : m.toString(36));
    return `g=${VERSION}.${boardKey}.${fingerprint(board)}` + (toks.length ? "." + toks.join(".") : "");
  }

  function parse(fragment) {
    const frag = (fragment || "").replace(/^#/, "");
    if (!frag.startsWith("g=")) return null;             // not a game link — caller ignores
    const parts = frag.slice(2).split(".");
    if (parts.length < 3) throw new ShareError("BAD_FORMAT", "link is truncated");
    const version = parseInt(parts[0], 10);
    if (version !== VERSION) throw new ShareError("BAD_VERSION", `unsupported format v${parts[0]}`);
    const key = parts[1];
    if (!/^[a-z0-9_]+$/.test(key)) throw new ShareError("BAD_FORMAT", "malformed board key");
    const fp = parts[2];
    const moves = [];
    for (let i = 3; i < parts.length; i++) {
      const t = parts[i];
      if (t === "") continue;                            // tolerate trailing dot
      if (t === "-") { moves.push(PASS); continue; }
      if (!/^[0-9a-z]+$/.test(t)) throw new ShareError("BAD_FORMAT", `malformed move "${t}"`);
      moves.push(parseInt(t, 36));
    }
    return { version, key, fp, moves };
  }

  /* Replay a parsed move list through the rules engine on a fresh board.
     Validates fingerprint, index range and legality of every ply; returns the final state
     plus one snapshot per ply so the caller can wire undo through the whole game. */
  function replay(TG, board, parsed) {
    if (parsed.fp !== fingerprint(board))
      throw new ShareError("BAD_BOARD", "link was made for a different version of this board");
    let s = TG.newGame(board);
    const snaps = [];
    for (const m of parsed.moves) {
      const node = m === PASS ? board.pass : m;
      if (!(node === board.pass || (Number.isInteger(node) && node >= 0 && node < board.n)))
        throw new ShareError("BAD_MOVE", `move ${m} is out of range for this board`);
      if (!TG.legalMoves(s, board)[node])
        throw new ShareError("ILLEGAL_MOVE", `move ${m} is illegal at ply ${s.moveNum + 1}`);
      snaps.push({ s, last: null });
      s = TG.play(s, node, board);
    }
    return { state: s, snaps };
  }

  const SHARE = { VERSION, PASS, ShareError, fingerprint, serialize, parse, replay };
  root.SHARE = SHARE;
  if (typeof module !== "undefined") module.exports = SHARE;
})(typeof globalThis !== "undefined" ? globalThis : this);
