/* record_check.cjs — JS side of the cross-language record test + JS round-trip + SGF check.

   Modes:
     node scripts/record_check.cjs emit <dir>     write records + expected final positions
     node scripts/record_check.cjs verify <dir>   load records written by Python, replay,
                                                  compare against the bundled expectations
     node scripts/record_check.cjs                self-test (JS round-trip + SGF re-parse)   */
"use strict";
const fs = require("fs"), path = require("path");
const { BOARDS } = require(path.join(__dirname, "..", "webapp", "data.js"));
const TG = require(path.join(__dirname, "..", "webapp", "engine.js"));
const SHARE = require(path.join(__dirname, "..", "webapp", "share.js"));
const REC = require(path.join(__dirname, "..", "webapp", "records.js"));

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randomGame(board, rng, maxLen) {
  let s = TG.newGame(board); const moves = [];
  while (moves.length < maxLen && !TG.isTerminal(s, board)) {
    const legal = TG.legalMoves(s, board); const nodes = [];
    for (let i = 0; i < board.n; i++) if (legal[i]) nodes.push(i);
    const mv = (!nodes.length || rng() < 0.02) ? board.pass : nodes[(rng() * nodes.length) | 0];
    s = TG.play(s, mv, board); moves.push(mv);
  }
  return { state: s, moves };
}
const SUBSTRATES = ["square_small", "hex_small", "tri_small", "trihex_small", "penrose_small"];
const expect = (s) => ({ colors: Array.from(s.colors), toMove: s.toMove, moveNum: s.moveNum });

const mode = process.argv[2] || "self";
const dir = process.argv[3];

if (mode === "emit") {
  fs.mkdirSync(dir, { recursive: true });
  const manifest = [];
  for (const key of SUBSTRATES) {
    const board = TG.makeBoard(BOARDS[key]);
    const rng = mulberry32(0xC0FFEE + key.length);
    for (let g = 0; g < 4; g++) {
      const { state, moves } = randomGame(board, rng, 40 + ((rng() * 60) | 0));
      const name = `${key}_${g}`;
      fs.writeFileSync(path.join(dir, name + ".json"),
        JSON.stringify(REC.buildRecord(key, board, moves, {}), null, 1));
      manifest.push({ file: name + ".json", ...expect(state) });
    }
  }
  fs.writeFileSync(path.join(dir, "expected.json"), JSON.stringify(manifest));
  console.log(`emitted ${manifest.length} records → ${dir}`);
} else if (mode === "verify") {
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, "expected.json")));
  let fails = 0;
  for (const m of manifest) {
    const rec = JSON.parse(fs.readFileSync(path.join(dir, m.file)));
    const parsed = REC.toParsed(rec, TG, BOARDS);
    const r = SHARE.replay(TG, TG.makeBoard(BOARDS[parsed.key]), parsed);
    const got = expect(r.state);
    if (JSON.stringify(got) !== JSON.stringify({ colors: m.colors, toMove: m.toMove, moveNum: m.moveNum })) {
      console.error(`FAIL ${m.file}: final position differs`); fails++;
    }
  }
  console.log(fails ? `FAIL — ${fails}/${manifest.length}` : `PASS — ${manifest.length} python-written records replay identically in JS`);
  process.exit(fails ? 1 : 0);
} else {
  // self-test: JS record round-trip + SGF export sanity (re-parse our own SGF)
  let fails = 0;
  for (const key of SUBSTRATES) {
    const board = TG.makeBoard(BOARDS[key]);
    const rng = mulberry32(7 + key.length);
    for (let g = 0; g < 20; g++) {                                  // 5 × 20 = 100 games
      const { state, moves } = randomGame(board, rng, 5 + ((rng() * 100) | 0));
      const rec = JSON.parse(JSON.stringify(REC.buildRecord(key, board, moves, {})));
      const r = SHARE.replay(TG, TG.makeBoard(BOARDS[key]), REC.toParsed(rec, TG, BOARDS));
      if (JSON.stringify(expect(r.state)) !== JSON.stringify(expect(state))) {
        console.error(`FAIL round-trip ${key} game ${g}`); fails++;
      }
    }
  }
  // SGF: square only; re-parse the output and check move count + coordinates in range
  const key = "square_small", board = TG.makeBoard(BOARDS[key]);
  const { moves } = randomGame(board, mulberry32(99), 30);
  const sgf = REC.toSGF(key, board, moves, { result: "B+1.5" });
  const plies = sgf.match(/;[BW]\[[a-z]{0,2}\]/g) || [];
  if (plies.length !== moves.length) { console.error("FAIL sgf ply count"); fails++; }
  if (!/^\(;FF\[4\]GM\[1\].*SZ\[9\]RU\[Chinese\]KM\[5\.5\]/.test(sgf)) { console.error("FAIL sgf header"); fails++; }
  for (const p of plies) {
    const c = p.slice(3, -1);
    if (c && !(c[0] >= "a" && c[0] <= "i" && c[1] >= "a" && c[1] <= "i")) { console.error(`FAIL sgf coord ${c}`); fails++; }
  }
  console.log(fails ? `FAIL — ${fails} problems` : "PASS — 100 JS record round-trips identical; SGF export well-formed");
  process.exit(fails ? 1 : 0);
}
