/* Tiling-Go — standalone browser engine. Pure JS port of the Python/C++ core:
   rules (capture / suicide / positional superko / area score / ownership), the 42-dim feature
   encoding, the TilingGoNet forward pass, and PUCT MCTS. No dependencies. Verified against PyTorch
   (scripts/webapp_check.cjs). Boards + trained weights are supplied by data.js.            */
(function (root) {
  "use strict";
  const EMPTY = 0, BLACK = 1, WHITE = 2;

  // ---- base64 → typed arrays (little-endian, matches numpy '<f4' / '<i4') ----
  function b64bytes(s) {
    const bin = atob(s), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  const b64f = s => new Float32Array(b64bytes(s).buffer);
  const b64i = s => new Int32Array(b64bytes(s).buffer);

  // ---- net loading (weight order mirrors scripts/export_webapp.py) ----
  function loadNet(b64, cfg) {
    const W = b64f(b64); let p = 0;
    const take = n => { const a = W.subarray(p, p + n); p += n; return a; };
    const H = cfg.hidden, I = cfg.in_dim, L = cfg.blocks;
    const lin = (out, inn) => ({ W: take(out * inn), b: take(out), out, in: inn });
    const net = { H, I, L, in0: lin(H, I), in2: lin(H, H), blocks: [] };
    for (let i = 0; i < L; i++)
      net.blocks.push({ nw: take(4 * H), nb: take(4 * H), mlp0: lin(2 * H, 4 * H), mlp2: lin(H, 2 * H) });
    net.fnW = take(H); net.fnB = take(H);
    net.pol = lin(1, H); net.pas = lin(1, H);
    net.v0 = lin(H, H); net.v2 = lin(1, H);
    net.own = lin(3, H);
    net.s0 = lin(H, H); net.s2 = lin(1, H);
    return net;
  }

  // reusable per-board scratch (one board per game ⇒ reused across the whole search; no GC churn)
  const _scratch = {};
  function scratch(N, H) {
    let s = _scratch[N];
    if (!s) s = _scratch[N] = {
      t: new Float32Array(N * H), h: new Float32Array(N * H), z: new Float32Array(N * 4 * H),
      m: new Float32Array(N * 2 * H), m2: new Float32Array(N * H),
      g: new Float32Array(H), mean: new Float32Array(H), mx: new Float32Array(H),
      v1: new Float32Array(H), s1: new Float32Array(H),
    };
    return s;
  }
  // Y[N,out] = X[N,in] @ Wᵀ + b into a provided buffer (k-loop unrolled ×4)
  function linInto(L, X, N, Y) {
    const W = L.W, b = L.b, IN = L.in, OUT = L.out, R = IN - (IN & 3);
    for (let i = 0; i < N; i++) {
      const xo = i * IN, yo = i * OUT;
      for (let o = 0; o < OUT; o++) {
        const wo = o * IN; let s = b[o], k = 0;
        for (; k < R; k += 4) s += X[xo + k] * W[wo + k] + X[xo + k + 1] * W[wo + k + 1] + X[xo + k + 2] * W[wo + k + 2] + X[xo + k + 3] * W[wo + k + 3];
        for (; k < IN; k++) s += X[xo + k] * W[wo + k];
        Y[yo + o] = s;
      }
    }
  }
  function relu(a) { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; }
  function layernorm(x, N, D, g, b) {           // per-row, biased variance, eps 1e-5
    for (let i = 0; i < N; i++) {
      const o = i * D; let m = 0;
      for (let j = 0; j < D; j++) m += x[o + j]; m /= D;
      let v = 0; for (let j = 0; j < D; j++) { const d = x[o + j] - m; v += d * d; } v /= D;
      const inv = 1 / Math.sqrt(v + 1e-5);
      for (let j = 0; j < D; j++) x[o + j] = (x[o + j] - m) * inv * g[j] + b[j];
    }
  }

  function forward(net, x, N, adj) {
    const H = net.H, sc = scratch(N, H), t = sc.t, h = sc.h;
    linInto(net.in0, x, N, t); relu(t);
    linInto(net.in2, t, N, h);
    const g = sc.g, mean = sc.mean, mx = sc.mx, z = sc.z, m = sc.m, m2 = sc.m2;
    for (const blk of net.blocks) {
      g.fill(0);
      for (let i = 0; i < N; i++) { const o = i * H; for (let j = 0; j < H; j++) g[j] += h[o + j]; }
      for (let j = 0; j < H; j++) g[j] /= N;
      for (let i = 0; i < N; i++) {
        const hi = i * H, zi = i * 4 * H, nb = adj[i];
        mean.fill(0); mx.fill(-1e9);
        for (let a = 0; a < nb.length; a++) {
          const ho = nb[a] * H;
          for (let j = 0; j < H; j++) { const v = h[ho + j]; mean[j] += v; if (v > mx[j]) mx[j] = v; }
        }
        const inv = nb.length > 0 ? 1 / nb.length : 1;
        for (let j = 0; j < H; j++) {
          z[zi + j] = h[hi + j]; z[zi + H + j] = mean[j] * inv;
          z[zi + 2 * H + j] = nb.length > 0 ? mx[j] : 0; z[zi + 3 * H + j] = g[j];
        }
      }
      layernorm(z, N, 4 * H, blk.nw, blk.nb);
      linInto(blk.mlp0, z, N, m); relu(m);
      linInto(blk.mlp2, m, N, m2);
      for (let k = 0; k < N * H; k++) h[k] += m2[k];
    }
    layernorm(h, N, H, net.fnW, net.fnB);
    g.fill(0);
    for (let i = 0; i < N; i++) { const o = i * H; for (let j = 0; j < H; j++) g[j] += h[o + j]; }
    for (let j = 0; j < H; j++) g[j] /= N;

    const policy = new Float32Array(N + 1);
    for (let i = 0; i < N; i++) { let s = net.pol.b[0]; const hi = i * H, Wp = net.pol.W; for (let j = 0; j < H; j++) s += h[hi + j] * Wp[j]; policy[i] = s; }
    { let s = net.pas.b[0]; for (let j = 0; j < H; j++) s += g[j] * net.pas.W[j]; policy[N] = s; }
    const v1 = sc.v1; linInto(net.v0, g, 1, v1); relu(v1);
    let vv = net.v2.b[0]; for (let j = 0; j < H; j++) vv += v1[j] * net.v2.W[j];
    const value = Math.tanh(vv);
    const ownSigned = new Float32Array(N), Wo = net.own.W, Bo = net.own.b;
    for (let i = 0; i < N; i++) {
      const hi = i * H; let a = Bo[0], b = Bo[1], c = Bo[2];
      for (let j = 0; j < H; j++) { const hv = h[hi + j]; a += hv * Wo[j]; b += hv * Wo[H + j]; c += hv * Wo[2 * H + j]; }
      const mm = Math.max(a, b, c), ea = Math.exp(a - mm), eb = Math.exp(b - mm), ec = Math.exp(c - mm);
      ownSigned[i] = (ea - eb) / (ea + eb + ec);          // P(black) − P(white)
    }
    const s1 = sc.s1; linInto(net.s0, g, 1, s1); relu(s1);
    let sco = net.s2.b[0]; for (let j = 0; j < H; j++) sco += s1[j] * net.s2.W[j];
    return { policy, value, ownSigned, score: sco };
  }

  // ---- board ----
  function makeBoard(rec) {
    const n = rec.n, coordsF = b64f(rec.coords), edgesI = b64i(rec.edges);
    const coords = []; for (let i = 0; i < n; i++) coords.push([coordsF[2 * i], coordsF[2 * i + 1]]);
    const adj = Array.from({ length: n }, () => []);
    for (let e = 0; e < edgesI.length; e += 2) { const a = edgesI[e], b = edgesI[e + 1]; adj[a].push(b); adj[b].push(a); }
    return { n, komi: rec.komi, label: rec.label, coords, adj, edges: edgesI, static: b64f(rec.static), pass: n, maxMoves: 3 * n };
  }

  // ---- rules ----
  function groupLib(colors, start, adj) {
    const color = colors[start], chain = [start], inchain = new Uint8Array(colors.length);
    inchain[start] = 1; let nlib = 0; const islib = new Uint8Array(colors.length); const stack = [start];
    while (stack.length) {
      const u = stack.pop();
      const nb = adj[u];
      for (let a = 0; a < nb.length; a++) {
        const w = nb[a], cw = colors[w];
        if (cw === EMPTY) { if (!islib[w]) { islib[w] = 1; nlib++; } }
        else if (cw === color && !inchain[w]) { inchain[w] = 1; chain.push(w); stack.push(w); }
      }
    }
    return { chain, nlib };
  }
  function simulate(state, node, player, board) {
    if (state.colors[node] !== EMPTY) return null;
    const nc = Int8Array.from(state.colors); nc[node] = player; const opp = player === BLACK ? WHITE : BLACK;
    const nb = board.adj[node];
    for (let a = 0; a < nb.length; a++) {
      const w = nb[a];
      if (nc[w] === opp) { const gl = groupLib(nc, w, board.adj); if (gl.nlib === 0) for (const c of gl.chain) nc[c] = EMPTY; }
    }
    if (groupLib(nc, node, board.adj).nlib === 0) return null;     // suicide
    return { colors: nc, key: nc.join("") };
  }
  function isTerminal(s, board) { return s.passCount >= 2 || s.moveNum >= board.maxMoves; }
  function legalMoves(s, board) {
    const n = board.n, legal = new Uint8Array(n + 1); legal[n] = 1;
    if (isTerminal(s, board)) return legal;
    for (let v = 0; v < n; v++) {
      if (s.colors[v] !== EMPTY) continue;
      const sim = simulate(s, v, s.toMove, board);
      if (sim && !s.history.has(sim.key)) legal[v] = 1;
    }
    return legal;
  }
  function newGame(board) {
    const colors = new Int8Array(board.n);
    return { colors, toMove: BLACK, passCount: 0, moveNum: 0, history: new Set([colors.join("")]) };
  }
  function play(s, move, board) {
    if (move === board.pass)
      return { colors: s.colors, toMove: s.toMove === BLACK ? WHITE : BLACK, passCount: s.passCount + 1, moveNum: s.moveNum + 1, history: s.history };
    const sim = simulate(s, move, s.toMove, board);
    const hist = new Set(s.history); hist.add(sim.key);
    return { colors: sim.colors, toMove: s.toMove === BLACK ? WHITE : BLACK, passCount: 0, moveNum: s.moveNum + 1, history: hist };
  }
  function regions(s, board, ownBuf) {              // area score + (optional) ownership fill
    const n = board.n, colors = s.colors; let black = 0, white = 0;
    for (let v = 0; v < n; v++) { if (colors[v] === BLACK) black++; else if (colors[v] === WHITE) white++; }
    const vis = new Uint8Array(n);
    if (ownBuf) for (let v = 0; v < n; v++) ownBuf[v] = colors[v] === BLACK ? 0 : colors[v] === WHITE ? 1 : 2;
    for (let v = 0; v < n; v++) {
      if (colors[v] !== EMPTY || vis[v]) continue;
      const region = [v]; vis[v] = 1; let border = 0, qi = 0;
      while (qi < region.length) {
        const u = region[qi++], nb = board.adj[u];
        for (let a = 0; a < nb.length; a++) {
          const w = nb[a];
          if (colors[w] === EMPTY) { if (!vis[w]) { vis[w] = 1; region.push(w); } }
          else border |= (colors[w] === BLACK ? 1 : 2);
        }
      }
      if (border === 1) { black += region.length; if (ownBuf) for (const c of region) ownBuf[c] = 0; }
      else if (border === 2) { white += region.length; if (ownBuf) for (const c of region) ownBuf[c] = 1; }
    }
    return { black, white };
  }
  function scoreDiff(s, board) { const r = regions(s, board); return r.black - r.white - board.komi; }
  function winner(s, board) { return scoreDiff(s, board) > 0 ? BLACK : WHITE; }

  // ---- encoding (42-dim features), mirrors tilinggo/nn/encoding.py ----
  function encode(s, board) {
    const n = board.n, x = new Float32Array(n * 42), colors = s.colors;
    const me = s.toMove, opp = me === BLACK ? WHITE : BLACK;
    const libs = new Int32Array(n), seen = new Uint8Array(n);
    for (let v = 0; v < n; v++) {
      if (colors[v] === EMPTY || seen[v]) continue;
      const gl = groupLib(colors, v, board.adj);
      for (const c of gl.chain) { libs[c] = gl.nlib; seen[c] = 1; }
    }
    const legal = legalMoves(s, board);
    for (let v = 0; v < n; v++) {
      const o = v * 42, c = colors[v];
      if (c === me) x[o] = 1; else if (c === opp) x[o + 1] = 1; else x[o + 2] = 1;
      if (c !== EMPTY) { let bkt = libs[v] - 1; if (bkt < 0) bkt = 0; if (bkt > 5) bkt = 5; x[o + 3 + bkt] = 1; }
      x[o + 14] = legal[v] ? 1 : 0;                       // cols 9..13 (recent) stay 0
      const so = v * 24; for (let j = 0; j < 24; j++) x[o + 15 + j] = board.static[so + j];
      x[o + 39] = board.komi / 15; x[o + 40] = s.moveNum / Math.max(n, 1); x[o + 41] = me === BLACK ? 1 : -1;
    }
    return { x, legal };
  }
  function evaluate(net, s, board) {                       // → {priors[n+1], value, out}
    const { x, legal } = encode(s, board);
    const out = forward(net, x, board.n, board.adj);
    const n = board.n, pr = new Float32Array(n + 1); let mx = -1e30;
    for (let i = 0; i <= n; i++) if (legal[i] && out.policy[i] > mx) mx = out.policy[i];
    let sum = 0;
    for (let i = 0; i <= n; i++) if (legal[i]) { const e = Math.exp(out.policy[i] - mx); pr[i] = e; sum += e; }
    if (sum > 0) for (let i = 0; i <= n; i++) pr[i] /= sum;
    return { priors: pr, value: out.value, out, legal };
  }

  // ---- PUCT MCTS (sequential, deterministic: no Dirichlet, best-move) ----
  const C_PUCT = 1.4, FPU = 0.25;
  function newNode(state, board) {
    const node = { state, terminal: isTerminal(state, board), expanded: false, termValue: 0,
      legal: null, P: null, N: null, W: null, children: new Map() };
    if (node.terminal) node.termValue = (winner(state, board) === state.toMove) ? 1 : -1;
    return node;
  }
  function expand(node, priors, board) {
    const lm = legalMoves(node.state, board), legal = [];
    for (let i = 0; i < lm.length; i++) if (lm[i]) legal.push(i);
    let tot = 0; for (const i of legal) tot += priors[i];
    const k = legal.length;
    node.legal = legal; node.P = new Float32Array(k); node.N = new Float32Array(k); node.W = new Float32Array(k);
    for (let j = 0; j < k; j++) node.P[j] = tot > 0 ? priors[legal[j]] / tot : 1 / k;
    node.expanded = true;
  }
  function selectChild(node) {
    const N = node.N, W = node.W, P = node.P, k = N.length;
    let sumN = 0; for (let i = 0; i < k; i++) sumN += N[i];
    const sq = Math.sqrt(sumN) + 1e-8;
    let vw = 0, vn = 0, vp = 0, any = false;
    for (let i = 0; i < k; i++) if (N[i] > 0) { vw += W[i]; vn += N[i]; vp += P[i]; any = true; }
    const fpu = any ? vw / vn - FPU * Math.sqrt(vp) : 0;
    let best = 0, bs = -1e30;
    for (let i = 0; i < k; i++) {
      const q = N[i] > 0 ? W[i] / N[i] : fpu;
      const u = C_PUCT * P[i] * sq / (1 + N[i]);
      if (q + u > bs) { bs = q + u; best = i; }
    }
    return best;
  }
  // stepped searcher so the UI can run sims in chunks (responsive); search() wraps it
  function makeSearcher(state, board, net) {
    const root = newNode(state, board);
    if (!root.terminal) { const e0 = evaluate(net, root.state, board); expand(root, e0.priors, board); }
    function sim() {
      if (root.terminal) return;
      let node = root; const path = [];
      while (node.expanded && !node.terminal) {
        const ai = selectChild(node); path.push([node, ai]);
        const mv = node.legal[ai];
        let child = node.children.get(mv);
        if (!child) { child = newNode(play(node.state, mv, board), board); node.children.set(mv, child); }
        node = child;
      }
      let v;
      if (node.terminal) { v = node.termValue; }
      else { const e = evaluate(net, node.state, board); expand(node, e.priors, board); v = e.value; }
      for (let k = path.length - 1; k >= 0; k--) { v = -v; const nd = path[k][0], ai = path[k][1]; nd.N[ai] += 1; nd.W[ai] += v; }
    }
    function best() {
      if (root.terminal || !root.N) return board.pass;
      let b = 0; for (let i = 1; i < root.N.length; i++) if (root.N[i] > root.N[b]) b = i;
      return root.legal[b];
    }
    return { root, terminal: root.terminal, sim, best };
  }
  function search(state, board, net, sims) {
    const s = makeSearcher(state, board, net);
    for (let i = 0; i < sims; i++) s.sim();
    return { move: s.best(), root: s.root };
  }

  root.TG = {
    EMPTY, BLACK, WHITE, loadNet, makeBoard, newGame, play, legalMoves, isTerminal,
    encode, forward, evaluate, search, makeSearcher, regions, scoreDiff, winner,
    ownership(s, board) { const o = new Int8Array(board.n); regions(s, board, o); return o; },
  };
  if (typeof module !== "undefined") module.exports = root.TG;
})(typeof globalThis !== "undefined" ? globalThis : this);
