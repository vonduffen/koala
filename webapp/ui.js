/* Tiling-Go webapp UI: SVG board renderer (port of tilinggo/ui/render.py) + game loop driving the
   local TG engine. Time-budgeted async MCTS keeps the page responsive. No server, no deps. */
(function () {
  "use strict";
  const $ = s => document.querySelector(s);
  const fmt = x => (Math.round(x * 100) / 100);

  let net = null, B = null, S = null, lastMove = null, prevMove = -1, busy = false, actx = null;
  let hist = [];   // undo stack: snapshots taken before each human turn / pass (state is immutable)
  let wrHist = []; // black win-rate per move index (drives the win-rate graph)
  let humanColor = TG.BLACK;  // which colour the human plays vs the engine (Black moves first)
  let countedStart = false;   // fire one analytics "game started" event per game (hosted site only)

  // GoatCounter custom event — a no-op unless the (hosted) page loaded the count.js script,
  // so the offline single-file build never phones home.
  function gcEvent(path, title) {
    try { if (window.goatcounter && window.goatcounter.count) window.goatcounter.count({ path, title, event: true }); } catch (e) { /* ignore */ }
  }
  const STRENGTH = { Fast: 700, Normal: 1800, Strong: 4500 };

  // ---------- renderer (mirrors render.py interactive_svg) ----------
  function minSpacing(coords) {
    const n = coords.length; if (n < 2) return 1;
    const nn = new Float64Array(n).fill(Infinity);
    for (let i = 0; i < n; i++) {
      const xi = coords[i][0], yi = coords[i][1];
      for (let j = 0; j < n; j++) { if (i === j) continue; const d = Math.hypot(coords[j][0] - xi, coords[j][1] - yi); if (d < nn[i]) nn[i] = d; }
    }
    const s = Array.from(nn).sort((a, b) => a - b);
    return s[Math.floor(0.02 * n)] || 1;
  }
  const DEFS =
    '<defs>' +
    '<radialGradient id="bz" cx="0.36" cy="0.30" r="0.78"><stop offset="0%" stop-color="#4c5168"/><stop offset="40%" stop-color="#191c26"/><stop offset="100%" stop-color="#04050a"/></radialGradient>' +
    '<radialGradient id="wz" cx="0.36" cy="0.30" r="0.82"><stop offset="0%" stop-color="#ffffff"/><stop offset="56%" stop-color="#e6e9f1"/><stop offset="100%" stop-color="#b9bfce"/></radialGradient>' +
    '<radialGradient id="bgs" cx="0.5" cy="0.40" r="0.85"><stop offset="0%" stop-color="#1a1f2c"/><stop offset="100%" stop-color="#0a0c12"/></radialGradient>' +
    '<filter id="sh" x="-40%" y="-40%" width="180%" height="180%"><feDropShadow dx="0" dy="1.4" stdDeviation="1.6" flood-color="#000" flood-opacity="0.6"/></filter>' +
    '<filter id="gl" x="-90%" y="-90%" width="280%" height="280%"><feDropShadow dx="0" dy="0" stdDeviation="3.2" flood-color="#00ffc2" flood-opacity="0.95"/></filter>' +
    '<filter id="grain"><feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" stitchTiles="stitch"/><feColorMatrix type="saturate" values="0"/></filter>' +
    '</defs>';

  function renderSVG(board, colors, last, legal, analysis) {
    const W = 760, margin = 26, coords = board.coords, n = board.n;
    const spc = minSpacing(coords), pad = 0.55 * spc;
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const [x, y] of coords) { if (x < minx) minx = x; if (y < miny) miny = y; if (x > maxx) maxx = x; if (y > maxy) maxy = y; }
    minx -= pad; miny -= pad; maxx += pad; maxy += pad;
    const spanx = Math.max(maxx - minx, 1e-9), spany = Math.max(maxy - miny, 1e-9);
    const scale = (W - 2 * margin) / Math.max(spanx, spany), H = Math.round(spany * scale + 2 * margin);
    const tx = x => margin + (x - minx) * scale, ty = y => margin + (maxy - y) * scale;
    const rS = 0.46 * spc * scale, rH = 0.5 * spc * scale;
    const o = [`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`, DEFS,
      `<rect width="${W}" height="${H}" rx="18" fill="url(#bgs)"/>`,
      `<rect width="${W}" height="${H}" rx="18" filter="url(#grain)" opacity="0.045"/>`];
    for (let e = 0; e < board.edges.length; e += 2) {
      const a = board.edges[e], b = board.edges[e + 1];
      o.push(`<line x1="${fmt(tx(coords[a][0]))}" y1="${fmt(ty(coords[a][1]))}" x2="${fmt(tx(coords[b][0]))}" y2="${fmt(ty(coords[b][1]))}" stroke="#454c5e" stroke-width="1.1" stroke-linecap="round" opacity="0.85"/>`);
    }
    if (legal) for (let i = 0; i < n; i++) if (colors[i] === 0 && legal[i])
      o.push(`<circle cx="${fmt(tx(coords[i][0]))}" cy="${fmt(ty(coords[i][1]))}" r="${fmt(rS * 0.2)}" fill="#5a6478" opacity="0.16"/>`);
    for (let i = 0; i < n; i++) {
      const c = colors[i]; if (c === 0) continue;
      const x = tx(coords[i][0]), y = ty(coords[i][1]);
      o.push(`<circle cx="${fmt(x)}" cy="${fmt(y)}" r="${fmt(rS)}" fill="${c === 1 ? "url(#bz)" : "url(#wz)"}" stroke="${c === 1 ? "#000" : "#aab0c0"}" stroke-width="0.7" filter="url(#sh)"/>`);
      if (last === i) o.push(`<circle cx="${fmt(x)}" cy="${fmt(y)}" r="${fmt(rS * 0.4)}" fill="none" stroke="#00ffc2" stroke-width="2.2" filter="url(#gl)"/>`);
    }
    if (analysis) {
      const own = analysis.ownership;
      if (own) for (let i = 0; i < n; i++) {
        if (colors[i] !== 0) continue; const v = own[i]; if (Math.abs(v) < 0.18) continue;
        const x = tx(coords[i][0]), y = ty(coords[i][1]), sq = rS * 1.15;
        o.push(`<rect x="${fmt(x - sq / 2)}" y="${fmt(y - sq / 2)}" width="${fmt(sq)}" height="${fmt(sq)}" rx="2" fill="${v > 0 ? "#0a0d10" : "#f4f4f2"}" opacity="${fmt(Math.min(0.5, 0.55 * Math.abs(v)))}"/>`);
      }
      for (const m of (analysis.moves || [])) {
        const i = m.node; if (i >= n) continue;
        const x = tx(coords[i][0]), y = ty(coords[i][1]), w = Math.max(0, Math.min(1, m.winrate));
        const hue = Math.round(135 * w), bestC = analysis.best === i;
        o.push(`<circle${bestC ? ' class="bestmv"' : ""} cx="${fmt(x)}" cy="${fmt(y)}" r="${fmt(rS)}" fill="hsl(${hue},70%,45%)" opacity="0.94" stroke="${bestC ? "#fff" : "#0a0d10cc"}" stroke-width="${bestC ? 2.4 : 0.9}"${bestC ? ' filter="url(#gl)"' : ""}/>`);
        o.push(`<text x="${fmt(x)}" y="${fmt(y + rS * 0.34)}" font-size="${fmt(rS * 0.82)}" fill="#fff" text-anchor="middle" font-family="ui-monospace,Menlo,monospace" font-weight="700" style="pointer-events:none">${Math.round(w * 100)}</text>`);
      }
    }
    for (let i = 0; i < n; i++)
      o.push(`<circle class="hot" data-node="${i}" cx="${fmt(tx(coords[i][0]))}" cy="${fmt(ty(coords[i][1]))}" r="${fmt(rH)}" fill="transparent"/>`);
    o.push("</svg>"); return o.join("");
  }

  // ---------- audio / fx ----------
  function clack() {
    if (!$("#snd").checked) return;
    try {
      actx = actx || new (window.AudioContext || window.webkitAudioContext)();
      const t = actx.currentTime, osc = actx.createOscillator(), g = actx.createGain();
      osc.type = "triangle"; osc.frequency.setValueAtTime(260, t); osc.frequency.exponentialRampToValueAtTime(92, t + 0.05);
      g.gain.setValueAtTime(0.16, t); g.gain.exponentialRampToValueAtTime(0.0007, t + 0.13);
      osc.connect(g).connect(actx.destination); osc.start(t); osc.stop(t + 0.14);
    } catch (e) {}
  }
  const setWin = bw => { $("#winfill").style.height = Math.round(bw * 100) + "%"; };
  const thinking = on => $("#scan").classList.toggle("on", on);
  const sleep = () => new Promise(r => setTimeout(r, 0));

  // ---------- win-rate timeline (KaTrain-style) ----------
  // black's win prob each move; red dot where the side that just moved lost ≥12% (a blunder).
  function drawWRGraph(cur) {
    const el = $("#wrgraph"); if (!el) return;
    const W = 256, H = 74, pad = 6, m = Math.max(cur, 1);
    const xs = i => pad + (i / m) * (W - 2 * pad), ys = w => pad + (1 - w) * (H - 2 * pad);
    let path = "", dots = "";
    for (let i = 0; i <= cur; i++) { const w = wrHist[i]; if (w == null) continue; path += (path ? "L" : "M") + xs(i).toFixed(1) + " " + ys(w).toFixed(1) + " "; }
    for (let i = 1; i <= cur; i++) {
      const w = wrHist[i], p = wrHist[i - 1]; if (w == null || p == null) continue;
      const moverDelta = (i % 2 === 1) ? (w - p) : (p - w);   // mover's own win-rate change
      if (moverDelta < -0.12) dots += `<circle cx="${xs(i).toFixed(1)}" cy="${ys(w).toFixed(1)}" r="3.4" fill="#e0796b" stroke="#1a0e0c" stroke-width="0.8"/>`;
    }
    const mid = ys(0.5).toFixed(1), cx = xs(cur).toFixed(1);
    el.innerHTML = `<svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="border-radius:8px;background:#0b0d13;border:1px solid #2a313c">
      <line x1="${pad}" y1="${mid}" x2="${W - pad}" y2="${mid}" stroke="#2a313c" stroke-dasharray="3 3"/>
      <path d="${path}" fill="none" stroke="#00ffc2" stroke-width="2" vector-effect="non-scaling-stroke"/>
      <line x1="${cx}" y1="${pad}" x2="${cx}" y2="${H - pad}" stroke="#ffffff22"/>${dots}</svg>`;
  }
  // one cheap net forward → black win-rate for the current position; record it and redraw the curve
  function recordWR() {
    if (!net) return;
    const out = TG.forward(net, TG.encode(S, B).x, B.n, B.adj);
    const bw = S.toMove === TG.BLACK ? 0.5 * (out.value + 1) : 0.5 * (1 - out.value);
    wrHist[S.moveNum] = bw;
    setWin(bw); $("#winpct").textContent = Math.round(bw * 100) + "%";
    drawWRGraph(S.moveNum);
  }

  // ---------- render state ----------
  function draw() {
    const legal = TG.legalMoves(S, B);
    $("#board").innerHTML = renderSVG(B, S.colors, lastMove, legal, null);
    const term = TG.isTerminal(S, B), sd = Math.round(TG.scoreDiff(S, B) * 10) / 10;
    $("#turndot").style.background = S.toMove === TG.BLACK ? "#111417" : "#f4f4f2";
    $("#turn").textContent = (S.toMove === TG.BLACK ? "Black" : "White") + " to move";
    $("#score").textContent = (sd > 0 ? "B+" : sd < 0 ? "W+" : "") + (sd ? Math.abs(sd) : "0");
    $("#move").textContent = "move " + S.moveNum + (S.passCount ? " · " + S.passCount + "p" : "");
    if (term) { const w = TG.winner(S, B) === TG.BLACK ? "Black" : "White"; $("#msg").innerHTML = `<span class="win">${w} wins by ${Math.abs(sd)}</span>`; }
    else $("#msg").textContent = "";
    if (S.moveNum > prevMove) clack(); prevMove = S.moveNum;
    $("#winpct").textContent = "—"; $("#topmv").innerHTML = "";
    bind();
  }
  function bind() {
    document.querySelectorAll("#board .hot").forEach(el =>
      el.addEventListener("click", () => onHuman(+el.dataset.node)));
  }

  // ---------- time-budgeted async search ----------
  async function searchMove(budgetMs, cap) {
    const s = TG.makeSearcher(S, B, net);
    if (s.terminal) return { move: B.pass, root: s.root };
    const t0 = performance.now(); let i = 0;
    while (i < cap && performance.now() - t0 < budgetMs) {
      const cs = performance.now();                       // ~45ms compute bursts → smooth scan
      do { s.sim(); i++; } while (i < cap && performance.now() - cs < 45 && performance.now() - t0 < budgetMs);
      await sleep();
    }
    return { move: s.best(), root: s.root, sims: i };
  }

  async function engineReply() {                    // engine plays the side to move (one move)
    if (TG.isTerminal(S, B)) return;
    thinking(true); await sleep();
    const r = await searchMove(STRENGTH[$("#strength").value], 600);
    S = TG.play(S, r.move, B); lastMove = (r.move === B.pass ? null : r.move);
    thinking(false); draw(); recordWR();
  }

  async function onHuman(node) {
    if (busy) return;
    const opp = $("#opponent").value;
    if (opp === "engine" && S.toMove !== humanColor) return;   // not your turn — engine to play
    const legal = TG.legalMoves(S, B);
    if (!legal[node]) { $("#msg").textContent = "Illegal move"; return; }
    busy = true;
    try {
      hist.push({ s: S, last: lastMove });          // snapshot before this turn (for undo)
      S = TG.play(S, node, B); lastMove = node; draw(); recordWR();
      if (!countedStart) {                            // count a real "game played" (first human move)
        countedStart = true; const k = $("#tiling").value;
        gcEvent("game/" + k, "Game started: " + (BOARDS[k] ? BOARDS[k].label : k));
      }
      if (opp === "engine") await engineReply();
      if ($("#auto").checked && !TG.isTerminal(S, B)) await analyze();
    } finally { busy = false; }
  }

  async function analyze() {
    if (TG.isTerminal(S, B)) { $("#msg").textContent = "game over"; return; }
    thinking(true); await sleep();
    const enc = TG.encode(S, B), out = TG.forward(net, enc.x, B.n, B.adj);   // value/own/score
    const r = await searchMove(STRENGTH[$("#strength").value], 800);          // visits
    thinking(false);
    const root = r.root, total = root.N.reduce((a, b) => a + b, 0) || 1;
    const order = Array.from(root.N.keys()).sort((a, b) => root.N[b] - root.N[a]);
    const moves = [], top = [];
    for (const ai of order) {
      const vis = root.N[ai]; if (vis <= 0) break;
      const node = root.legal[ai], q = root.W[ai] / vis, wr = 0.5 * (q + 1);
      if (node !== B.n) { moves.push({ node, winrate: wr }); top.push({ wr, vis }); }
    }
    const best = moves.length ? moves[0].node : null;
    $("#board").innerHTML = renderSVG(B, S.colors, lastMove, TG.legalMoves(S, B),
      { moves: moves.slice(0, 8), ownership: out.ownSigned, best }); bind();
    const bw = S.toMove === TG.BLACK ? 0.5 * (out.value + 1) : 0.5 * (1 - out.value);
    const lead = (S.toMove === TG.BLACK ? 1 : -1) * out.score * B.n;
    setWin(bw); $("#winpct").textContent = Math.round(bw * 100) + "%";
    $("#score").textContent = (lead >= 0 ? "B+" : "W+") + Math.abs(lead).toFixed(1);
    $("#topmv").innerHTML = top.slice(0, 5).map((m, i) => `<span class="k">${i + 1}</span> ${Math.round(m.wr * 100)}% &middot; ${m.vis}v`).join("<br>");
  }

  function newGame(key) {
    B = TG.makeBoard(BOARDS[key]); S = TG.newGame(B);
    lastMove = null; prevMove = 0; hist = []; wrHist = []; countedStart = false; setWin(0.5); draw(); recordWR();
    // if the human chose White, the engine (Black) opens the game
    if ($("#opponent").value === "engine" && humanColor === TG.WHITE) {
      busy = true;
      (async () => { try { await engineReply(); } finally { busy = false; } })();
    }
  }

  // ---------- board catalogue → family + size/variant groups ----------
  function buildFamilies(keys) {
    const fam = {}, order = [];
    const add = (f, key, sub) => { if (!fam[f]) { fam[f] = []; order.push(f); } fam[f].push([key, sub]); };
    const SUB = { rect9: ["Square", "9×9"], rect13: ["Square", "13×13"], rect19: ["Square", "19×19"], square: ["Square", "disc"],
      hexagonal: ["Hexagonal", "standard"], hex_big: ["Hexagonal", "large"],
      triangular: ["Triangular", "standard"], tri_big: ["Triangular", "large"],
      penrose_small: ["Penrose", "small"], penrose: ["Penrose", "medium"], penrose_med: ["Penrose", "large"], penrose_big: ["Penrose", "x-large"],
      rosette: ["Rosette", "standard"] };
    for (const k of keys) {
      if (SUB[k]) add(SUB[k][0], k, SUB[k][1]);
      else if (k.indexOf("twou") === 0) add("2-uniform", k, (BOARDS[k].label || k).replace(/^2-uniform\s*/, ""));
      else add("Archimedean", k, BOARDS[k].label || k);   // trihexagonal, trunc_*, rhombitrihex, snub_*
    }
    return order.map(f => ({ family: f, items: fam[f] }));
  }

  // ---------- wire up ----------
  function init() {
    net = TG.loadNet(WEIGHTS_B64, CFG);
    const FAMS = buildFamilies(Object.keys(BOARDS));
    const famSel = $("#family"), varSel = $("#variant");
    famSel.innerHTML = FAMS.map((g, i) => `<option value="${i}">${g.family}</option>`).join("");
    const fillVariants = (fi) => { varSel.innerHTML = FAMS[fi].items.map(([k, sub]) => `<option value="${k}">${sub}</option>`).join(""); };
    const currentKey = () => varSel.value;
    const selectKey = (key) => {                          // point both dropdowns at a board key
      for (let i = 0; i < FAMS.length; i++) if (FAMS[i].items.some(([k]) => k === key)) { famSel.value = i; fillVariants(i); varSel.value = key; return; }
    };
    $("#analyze").onclick = async () => { if (busy) return; busy = true; try { await analyze(); } finally { busy = false; } };
    $("#undo").onclick = () => {
      if (busy || !hist.length) return;
      const h = hist.pop(); S = h.s; lastMove = h.last; prevMove = S.moveNum; draw();
      wrHist.length = S.moveNum + 1; recordWR();      // trim the win-rate curve to match
    };
    $("#pass").onclick = async () => { if (busy) return; busy = true; try { hist.push({ s: S, last: lastMove }); S = TG.play(S, B.pass, B); lastMove = null; draw(); recordWR(); if ($("#opponent").value === "engine") await engineReply(); } finally { busy = false; } };
    $("#reset").onclick = () => { if (!busy) newGame(currentKey()); };
    famSel.onchange = () => { if (busy) return; fillVariants(+famSel.value); newGame(currentKey()); };
    varSel.onchange = () => { if (!busy) newGame(currentKey()); };
    $("#random").onclick = () => {                         // random board for a fresh game
      if (busy) return;
      const keys = Object.keys(BOARDS), cur = currentKey(); let k = cur;
      for (let t = 0; t < 25 && k === cur; t++) k = keys[(Math.random() * keys.length) | 0];
      selectKey(k); newGame(k);
    };
    // colour choice + opponent change take effect from a fresh game
    const applyColor = () => { humanColor = $("#playercolor").value === "white" ? TG.WHITE : TG.BLACK; };
    $("#playercolor").onchange = () => { if (busy) return; applyColor(); newGame(currentKey()); };
    $("#opponent").onchange = () => { if (!busy) newGame(currentKey()); };
    applyColor();
    selectKey(BOARDS["penrose"] ? "penrose" : Object.keys(BOARDS)[0]);
    newGame(currentKey());
  }
  if (document.readyState !== "loading") init(); else document.addEventListener("DOMContentLoaded", init);
})();
