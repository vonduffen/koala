/* duel_wasm_js.cjs — Gate 4: the WASM engine must actually be a STRONGER OPPONENT, not just a
   faster one (a previous port shipped weaker — never again). Head-to-head at equal wall-clock
   per move on the 9×9: each engine is first calibrated (sims/s on this board, this machine),
   then granted round(budget × sims_per_sec) sims per move — equal expected think time without
   restarting search trees mid-move. Colours alternate; openings are 6 shared random plies;
   area scoring; Wilson 95% CI reported. Pass bar: WASM wins ≥70% of ≥200 games.

   Run: node scripts/duel_wasm_js.cjs [games] [budgetMs]                                    */
"use strict";
const path = require("path");
const { CFG, WEIGHTS_B64, BOARDS } = require(path.join(__dirname, "..", "webapp", "data.js"));
const TG = require(path.join(__dirname, "..", "webapp", "engine.js"));
const GLUE = require(path.join(__dirname, "..", "webapp", "wasm_glue.js"));
const TGWasmFactory = require(path.join(__dirname, "..", "webapp", "tgwasm.js"));

const GAMES = parseInt(process.argv[2] || "200", 10);
const BUDGET_MS = parseInt(process.argv[3] || "400", 10);

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const wilson = (p, n, z = 1.96) => {
  if (!n) return [0, 1];
  const d = 1 + z * z / n, c = p + z * z / (2 * n);
  const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return [(c - h) / d, (c + h) / d];
};

(async () => {
  const key = "square_small";
  const board = TG.makeBoard(BOARDS[key]);
  const net = TG.loadNet(WEIGHTS_B64, CFG);
  const eng = await GLUE.makeWasmEngine(TGWasmFactory, board, net);

  // ---- calibration: sims/s for each engine on this board ---------------------------------
  let t0 = process.hrtime.bigint();
  { const s = TG.makeSearcher(TG.newGame(board), board, net); for (let i = 0; i < 200; i++) s.sim(); }
  const jsSps = 200 / (Number(process.hrtime.bigint() - t0) / 1e9);
  t0 = process.hrtime.bigint();
  eng.search([], 800, 0);
  const waSps = 800 / (Number(process.hrtime.bigint() - t0) / 1e9);
  const jsSims = Math.max(2, Math.round(jsSps * BUDGET_MS / 1000));
  const waSims = Math.max(2, Math.round(waSps * BUDGET_MS / 1000));
  console.log(`calibration: JS ${jsSps.toFixed(1)} sims/s → ${jsSims} sims/move; ` +
              `WASM ${waSps.toFixed(0)} sims/s → ${waSims} sims/move (${BUDGET_MS} ms budget)`);

  // ---- duel -------------------------------------------------------------------------------
  const rng = mulberry32(0xD0E1);
  let waWins = 0, done = 0;
  for (let g = 0; g < GAMES; g++) {
    let s = TG.newGame(board); const moves = [];
    for (let o = 0; o < 6 && !TG.isTerminal(s, board); o++) {     // shared random opening
      const legal = TG.legalMoves(s, board); const nodes = [];
      for (let i = 0; i < board.n; i++) if (legal[i]) nodes.push(i);
      if (!nodes.length) break;
      const mv = nodes[(rng() * nodes.length) | 0];
      s = TG.play(s, mv, board); moves.push(mv);
    }
    const waIsBlack = (g % 2 === 0);                              // colours alternate
    while (!TG.isTerminal(s, board)) {
      const blackToMove = s.toMove === TG.BLACK;
      let mv;
      if (blackToMove === waIsBlack) mv = eng.search(moves, waSims, 0).move;
      else {
        const sr = TG.makeSearcher(s, board, net);
        for (let i = 0; i < jsSims; i++) sr.sim();
        mv = sr.best();
      }
      s = TG.play(s, mv, board); moves.push(mv);
    }
    const blackWins = TG.winner(s, board) === TG.BLACK;
    if (blackWins === waIsBlack) waWins++;
    done++;
    if (done % 20 === 0) {
      const p = waWins / done, [lo, hi] = wilson(p, done);
      console.log(`  ${done}/${GAMES}: WASM ${(p * 100).toFixed(0)}% [${(lo * 100).toFixed(0)}-${(hi * 100).toFixed(0)}%]`);
    }
  }
  const p = waWins / done, [lo, hi] = wilson(p, done);
  console.log(`\nRESULT: WASM wins ${waWins}/${done} = ${(p * 100).toFixed(1)}% ` +
              `[95% CI ${(lo * 100).toFixed(1)}-${(hi * 100).toFixed(1)}%] at equal wall-clock (${BUDGET_MS} ms/move)`);
  const pass = p >= 0.70;
  console.log(pass ? "PASS — Gate 4 cleared (≥70%)" : "FAIL — below the 70% bar; DO NOT ship as default");
  process.exit(pass ? 0 : 1);
})().catch(e => { console.error("FATAL:", e); process.exit(1); });
