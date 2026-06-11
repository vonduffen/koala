/* wasm_check.cjs — Gates 2 & 3 for the WASM engine.

   Gate 2 (parity): WASM priors/value must match the JS engine (itself verified against
   PyTorch by webapp_check.cjs) within 1e-3 on random positions across 5 substrates incl.
   Penrose. The two engines share one weight source, so any drift is a porting bug.

   Gate 3 (speed): WASM must clear ≥10× the JS engine's sims/s on the 9×9 board (equal
   sims, same machine, both single-threaded in this node process).

   Run: node scripts/wasm_check.cjs            (exit 0 = both gates pass)                  */
"use strict";
const path = require("path");
const { CFG, WEIGHTS_B64, BOARDS } = require(path.join(__dirname, "..", "webapp", "data.js"));
const TG = require(path.join(__dirname, "..", "webapp", "engine.js"));
const GLUE = require(path.join(__dirname, "..", "webapp", "wasm_glue.js"));
const TGWasmFactory = require(path.join(__dirname, "..", "webapp", "tgwasm.js"));

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SUBSTRATES = ["square_small", "hex_small", "tri_small", "trihex_small", "penrose_medium"];
const TOL = 1e-3;

(async () => {
  const net = TG.loadNet(WEIGHTS_B64, CFG);
  let fails = 0;

  // ---- Gate 2: parity on random positions ------------------------------------------------
  for (const key of SUBSTRATES) {
    const board = TG.makeBoard(BOARDS[key]);
    const eng = await GLUE.makeWasmEngine(TGWasmFactory, board, net);
    const rng = mulberry32(0xA11CE + key.length);
    let dpMax = 0, dvMax = 0;
    for (let r = 0; r < 8; r++) {                       // 8 random positions per substrate
      let s = TG.newGame(board); const moves = [];
      const len = (rng() * 60) | 0;
      while (moves.length < len && !TG.isTerminal(s, board)) {
        const legal = TG.legalMoves(s, board); const nodes = [];
        for (let i = 0; i < board.n; i++) if (legal[i]) nodes.push(i);
        if (!nodes.length) break;
        const mv = nodes[(rng() * nodes.length) | 0];
        s = TG.play(s, mv, board); moves.push(mv);
      }
      const js = TG.evaluate(net, s, board);            // legal-masked priors + value
      const wa = eng.eval(moves);
      for (let i = 0; i <= board.n; i++) dpMax = Math.max(dpMax, Math.abs(js.priors[i] - wa.priors[i]));
      dvMax = Math.max(dvMax, Math.abs(js.value - wa.value));
    }
    const ok = dpMax <= TOL && dvMax <= TOL;
    if (!ok) fails++;
    console.log(`${ok ? "ok  " : "FAIL"} parity ${key.padEnd(16)} max|dPrior|=${dpMax.toExponential(2)}  max|dValue|=${dvMax.toExponential(2)}`);
  }

  // ---- Gate 3: speed on the 9×9 (square_small) -------------------------------------------
  const key = "square_small";
  const board = TG.makeBoard(BOARDS[key]);
  const eng = await GLUE.makeWasmEngine(TGWasmFactory, board, net);
  const SIMS = 400;
  // JS engine: time a search of SIMS sims from the empty board
  let t0 = process.hrtime.bigint();
  { const s = TG.makeSearcher(TG.newGame(board), board, net); for (let i = 0; i < SIMS; i++) s.sim(); }
  const jsMs = Number(process.hrtime.bigint() - t0) / 1e6;
  // WASM engine: same sims, same position, no noise
  t0 = process.hrtime.bigint();
  eng.search([], SIMS, 0);
  const waMs = Number(process.hrtime.bigint() - t0) / 1e6;
  const speedup = (SIMS / waMs) / (SIMS / jsMs);
  console.log(`speed 9x9 ${SIMS} sims: JS ${(SIMS / jsMs * 1000).toFixed(0)} sims/s, ` +
              `WASM ${(SIMS / waMs * 1000).toFixed(0)} sims/s → ${speedup.toFixed(1)}×`);
  if (speedup < 10) { fails++; console.error("FAIL: speedup below 10×"); }

  console.log(fails ? `\nFAIL — ${fails} gate(s) failed` : "\nPASS — WASM engine parity + ≥10× speed verified");
  process.exit(fails ? 1 : 0);
})().catch(e => { console.error("FATAL:", e); process.exit(1); });
