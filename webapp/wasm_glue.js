/* Glue between the webapp and the WASM engine (cpp/wasm.cpp via webapp/tgwasm.js).

   No new data artifacts: the WASM engine is fed the SAME baked data the JS engine uses —
   edges + static features from BOARDS, and the weights re-serialized to the TGN1 stream from
   the already-parsed JS net (loadNet's tensor order is TGN1's order; the trailing aux heads
   the C++ never reads are simply not written). */
(function (root) {
  "use strict";

  // TGN1: magic, in_dim, hidden, blocks, then length-prefixed tensors in load_net order.
  function serializeTGN1(net) {
    const tensors = [];
    const lin = l => { tensors.push(l.W, l.b); };
    lin(net.in0); lin(net.in2);
    for (const b of net.blocks) { tensors.push(b.nw, b.nb); lin(b.mlp0); lin(b.mlp2); }
    tensors.push(net.fnW, net.fnB);
    lin(net.pol); lin(net.pas); lin(net.v0); lin(net.v2);
    let words = 4;                                   // magic + 3 dims
    for (const t of tensors) words += 1 + t.length;  // numel + data
    const buf = new ArrayBuffer(words * 4);
    const i32 = new Int32Array(buf), f32 = new Float32Array(buf);
    let p = 0;
    i32[p++] = 0x54474E31; i32[p++] = net.I; i32[p++] = net.H; i32[p++] = net.L;
    for (const t of tensors) { i32[p++] = t.length; f32.set(t, p); p += t.length; }
    return new Uint8Array(buf);
  }

  /* Instantiate the engine for one board. `TGWasmFactory` is the MODULARIZE export of
     tgwasm.js; `board` is the DECODED TG.makeBoard(...) result (typed arrays, not the raw
     base64 record). Throws if WASM (or SIMD) is unsupported — callers catch and fall back to
     the JS engine. Returns { n, search(moves, sims, eps) -> {move, pi, value}, eval(moves) }. */
  async function makeWasmEngine(TGWasmFactory, board, net, weightBytes) {
    const M = await TGWasmFactory();
    const edges = board.edges;                       // Int32Array, flat [u0,v0,u1,v1,...]
    const nE = edges.length / 2;
    const n = board.n;
    const statics = board.static;                    // Float32Array [n*24]
    const dim = statics.length / n;
    const weights = weightBytes || serializeTGN1(net);   // workers ship pre-serialized bytes

    const alloc = (bytes) => M._malloc(bytes);
    const pEdges = alloc(edges.length * 4);
    new Int32Array(M.HEAP32.buffer, pEdges, edges.length).set(edges);
    const pStatic = alloc(statics.length * 4);
    new Float32Array(M.HEAPF32.buffer, pStatic, statics.length).set(statics);
    const pW = alloc(weights.length);
    new Uint8Array(M.HEAPU8.buffer, pW, weights.length).set(weights);

    const rc = M._tg_init(n, pEdges, nE, board.komi, pStatic, dim, pW, weights.length);
    M._free(pEdges); M._free(pStatic); M._free(pW);
    if (rc !== 0) throw new Error("tg_init failed");

    // persistent IO buffers
    const pMoves = alloc(4 * 4096), pPi = alloc(4 * (n + 1)), pVal = alloc(4);
    const setMoves = (moves) => {
      new Int32Array(M.HEAP32.buffer, pMoves, moves.length).set(moves);
    };
    return {
      n,
      search(moves, sims, dirichletEps) {
        setMoves(moves);
        const mv = M._tg_search(pMoves, moves.length, sims, dirichletEps || 0, pPi, pVal);
        if (mv < 0) throw new Error("tg_search failed (illegal replay?)");
        return { move: mv,
                 pi: Array.from(new Float32Array(M.HEAPF32.buffer, pPi, n + 1)),
                 value: new Float32Array(M.HEAPF32.buffer, pVal, 1)[0] };
      },
      eval(moves) {
        setMoves(moves);
        if (M._tg_eval(pMoves, moves.length, pPi, pVal) !== 0) throw new Error("tg_eval failed");
        return { priors: Array.from(new Float32Array(M.HEAPF32.buffer, pPi, n + 1)),
                 value: new Float32Array(M.HEAPF32.buffer, pVal, 1)[0] };
      },
    };
  }

  const G = { serializeTGN1, makeWasmEngine };
  root.WASMGLUE = G;
  if (typeof module !== "undefined") module.exports = G;
})(typeof globalThis !== "undefined" ? globalThis : this);
