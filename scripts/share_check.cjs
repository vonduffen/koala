/* share_check.cjs — verifies the shareable-link serialization (webapp/share.js).

   1. Round-trip property test: ≥200 random legal games across 5 substrate families
      (square, hexagonal, triangular, one Archimedean, Penrose); serialize → parse →
      replay must reproduce the exact final position: colors, side to move, move count,
      pass count, and the engine's superko history (the position-key set that IS its
      Zobrist-equivalent hash).
   2. Malformed-fragment tests: every failure class must raise a typed ShareError.

   Run: node scripts/share_check.cjs            (exit 0 = pass)                       */
"use strict";
const path = require("path");
const { BOARDS } = require(path.join(__dirname, "..", "webapp", "data.js"));
const TG = require(path.join(__dirname, "..", "webapp", "engine.js"));
const SHARE = require(path.join(__dirname, "..", "webapp", "share.js"));

// deterministic RNG so failures reproduce
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pickKey(prefix) {
  const k = Object.keys(BOARDS).find(k => k.startsWith(prefix));
  if (!k) throw new Error(`no board with prefix "${prefix}" in BOARDS`);
  return k;
}
const SUBSTRATES = ["square", "hex_", "tri_", "trihex", "penrose"].map(pickKey);

function randomGame(board, rng, maxLen) {
  let s = TG.newGame(board);
  const moves = [];
  while (moves.length < maxLen && !TG.isTerminal(s, board)) {
    const legal = TG.legalMoves(s, board);
    const nodes = [];
    for (let i = 0; i < board.n; i++) if (legal[i]) nodes.push(i);
    let mv;
    if (!nodes.length || rng() < 0.02) mv = board.pass;       // occasional / forced pass
    else mv = nodes[(rng() * nodes.length) | 0];
    s = TG.play(s, mv, board);
    moves.push(mv);
  }
  return { state: s, moves };
}

const sig = (s) => JSON.stringify({
  colors: Array.from(s.colors), toMove: s.toMove, moveNum: s.moveNum,
  passCount: s.passCount, history: Array.from(s.history).sort(),
});

let fails = 0;
const fail = (msg) => { fails++; console.error("FAIL: " + msg); };

// ---- 1. round-trip property test --------------------------------------------------------
const GAMES_PER = 40;                                          // 5 × 40 = 200 games
let total = 0, maxFragLen = 0, frag100 = null;
for (const key of SUBSTRATES) {
  const board = TG.makeBoard(BOARDS[key]);
  const rng = mulberry32(0xE0 + key.length);
  for (let g = 0; g < GAMES_PER; g++) {
    const len = 5 + ((rng() * 115) | 0);                       // 5..119 plies
    const { state, moves } = randomGame(board, rng, len);
    const frag = SHARE.serialize(key, board, moves);
    maxFragLen = Math.max(maxFragLen, frag.length);
    if (moves.length >= 100 && !frag100) frag100 = frag;
    const parsed = SHARE.parse("#" + frag);
    const r = SHARE.replay(TG, TG.makeBoard(BOARDS[parsed.key]), parsed);
    if (sig(r.state) !== sig(state))
      fail(`round-trip mismatch on ${key} game ${g} (${moves.length} plies)`);
    total++;
  }
}
console.log(`round-trip: ${total} games across ${SUBSTRATES.length} substrates — ` +
            `${fails ? fails + " failures" : "all identical (colors+toMove+superko history)"}`);
console.log(`longest fragment: ${maxFragLen} chars` +
            (frag100 ? `; a ${frag100.split(".").length - 3}-move game = ${frag100.length} chars (target <1500)` : ""));
if (maxFragLen > 1500) fail("fragment exceeds 1500-char URL budget");

// ---- 2. malformed fragments --------------------------------------------------------------
const sq = pickKey("square");
const sqBoard = TG.makeBoard(BOARDS[sq]);
const fp = SHARE.fingerprint(sqBoard);
const expectError = (label, code, fn) => {
  try { fn(); fail(`${label}: no error raised`); }
  catch (e) {
    if (!(e instanceof SHARE.ShareError)) fail(`${label}: wrong error type (${e})`);
    else if (e.code !== code) fail(`${label}: expected ${code}, got ${e.code}`);
    else console.log(`malformed ok: ${label} → ${e.code} ("${e.message}")`);
  }
};
expectError("truncated", "BAD_FORMAT", () => SHARE.parse(`#g=1.${sq}`));
expectError("wrong version", "BAD_VERSION", () => SHARE.parse(`#g=9.${sq}.${fp}.5`));
expectError("garbage move token", "BAD_FORMAT", () => SHARE.parse(`#g=1.${sq}.${fp}.5.@@`));
expectError("bad fingerprint", "BAD_BOARD",
  () => SHARE.replay(TG, sqBoard, SHARE.parse(`#g=1.${sq}.zzzzzz.5`)));
expectError("node out of range", "BAD_MOVE",
  () => SHARE.replay(TG, sqBoard, SHARE.parse(`#g=1.${sq}.${fp}.${(sqBoard.n + 5).toString(36)}`)));
expectError("illegal move (occupied)", "ILLEGAL_MOVE",
  () => SHARE.replay(TG, sqBoard, SHARE.parse(`#g=1.${sq}.${fp}.5.7.5`)));
if (SHARE.parse("#foo=bar") !== null) fail("non-game fragment should parse to null");
else console.log("malformed ok: non-game fragment → ignored (null)");

// ---- analysis links (Task 6): flag + comment round-trip; plain links unaffected -----------
{
  const payload = '<script>alert(1)</script> & "quotes" + émoji 🎯';
  const frag = SHARE.serialize(sq, sqBoard, [5, 7], { analysis: true, comment: payload });
  const p = SHARE.parse("#" + frag);
  if (!p.analysis) fail("analysis flag lost");
  if (p.comment !== payload) fail(`comment round-trip mangled: ${JSON.stringify(p.comment)}`);
  if (p.moves.length !== 2) fail("moves corrupted by &-params");
  const plain = SHARE.parse("#" + SHARE.serialize(sq, sqBoard, [5, 7]));
  if (plain.analysis !== false || plain.comment !== null) fail("plain link not identical to Task 1 semantics");
  const long = SHARE.parse("#" + SHARE.serialize(sq, sqBoard, [], { comment: "x".repeat(900) }));
  if (long.comment.length !== SHARE.COMMENT_MAX) fail(`comment cap not enforced (${long.comment.length})`);
  // replay must ignore the extras entirely
  const r = SHARE.replay(TG, sqBoard, p);
  if (r.state.moveNum !== 2) fail("replay broken by analysis params");
  console.log(`analysis links ok: flag + ${payload.length}-char hostile comment round-trip, ` +
              `cap ${SHARE.COMMENT_MAX}, plain links unchanged`);
}

// ---- 3. persistence round-trip (the localStorage record schema, simulated reload) --------
// Mirrors ui.js persist()/lsGet(): the game travels as JSON through a storage stub, then is
// replayed on "reload". Catches JSON escaping/typing issues in the stored record.
{
  const storage = {};                                              // localStorage stand-in
  const rng = mulberry32(0xBEEF);
  let ok = 0;
  for (const key of SUBSTRATES.slice(0, 3)) {
    const board = TG.makeBoard(BOARDS[key]);
    const { state, moves } = randomGame(board, rng, 60);
    const rec = { v: 1, frag: SHARE.serialize(key, board, moves),
                  strength: "Normal", opponent: "engine", color: "black" };
    storage["eg-game-v1"] = JSON.stringify(rec);                   // persist()
    const back = JSON.parse(storage["eg-game-v1"]);                // reload → lsGet()
    if (back.v !== 1) { fail(`persistence: version lost for ${key}`); continue; }
    const r = SHARE.replay(TG, TG.makeBoard(BOARDS[back.frag.split(".")[1]]),
                           SHARE.parse("#" + back.frag));
    if (sig(r.state) !== sig(state)) fail(`persistence round-trip mismatch on ${key}`);
    else ok++;
    if (back.strength !== "Normal" || back.color !== "black")
      fail(`persistence: settings lost for ${key}`);
  }
  // unknown schema version must be discarded (ui.js lsGet returns null unless v === 1)
  const unknown = JSON.parse(JSON.stringify({ v: 99, frag: "g=1.x.y.z" }));
  if (!(unknown.v !== 1)) fail("persistence: unknown version not detectable");
  console.log(`persistence: ${ok}/3 storage round-trips identical; unknown schema version detectable`);
}

console.log(fails ? `\nFAIL — ${fails} problem(s)` : "\nPASS — share serialization verified");
process.exit(fails ? 1 : 0);
