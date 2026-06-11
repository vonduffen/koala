/* Web Worker body for the WASM engine. The page assembles this worker from inline sources
   (tgwasm.js + wasm_glue.js + this file) via a Blob URL, so the site stays one static file.

   Protocol (all messages carry {type}):
     in : {type:"init", board:{n, komi, edges:Int32Array, static:Float32Array}, weights:Uint8Array}
     out: {type:"ready", sps}          — engine live; sps = calibrated sims/s on this board
     out: {type:"init-error", message} — WASM/SIMD unavailable or init failed → page falls back to JS
     in : {type:"search", id, moves:int[], sims, eps}
     out: {type:"result", id, move, pi, value, ms, sims}
     in : {type:"cancel", id}          — a finished search whose id was cancelled is discarded
                                         (searches are single WASM calls; cancellation discards
                                         the result rather than aborting mid-compute)           */
(function () {
  "use strict";
  let eng = null;
  const cancelled = new Set();

  self.onmessage = async (e) => {
    const m = e.data;
    if (m.type === "init") {
      try {
        const board = { n: m.board.n, komi: m.board.komi, edges: m.board.edges, static: m.board.static };
        eng = await WASMGLUE.makeWasmEngine(TGWasm, board, null, m.weights);
        // warm up first: the first searches run in wasm's baseline tier and measure 5-10×
        // slow, which would make the page under-budget the engine's thinking time
        eng.search([], 60, 0); eng.search([], 60, 0);
        const t0 = performance.now();
        eng.search([], 200, 0);
        const sps = 200 / ((performance.now() - t0) / 1000);
        self.postMessage({ type: "ready", sps });
      } catch (err) {
        self.postMessage({ type: "init-error", message: String(err && err.message || err) });
      }
      return;
    }
    if (m.type === "cancel") { cancelled.add(m.id); return; }
    if (m.type === "search") {
      if (!eng) { self.postMessage({ type: "result", id: m.id, move: -1 }); return; }
      const t0 = performance.now();
      let r;
      try { r = eng.search(m.moves, m.sims, m.eps || 0); }
      catch (err) { self.postMessage({ type: "result", id: m.id, move: -1, error: String(err) }); return; }
      if (cancelled.delete(m.id)) return;                       // discarded
      self.postMessage({ type: "result", id: m.id, move: r.move, pi: r.pi, value: r.value,
                         ms: performance.now() - t0, sims: m.sims });
    }
  };
})();
