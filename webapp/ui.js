/* Tiling-Go webapp UI: SVG board renderer (port of tilinggo/ui/render.py) + game loop driving the
   local TG engine. Time-budgeted async MCTS keeps the page responsive. No server, no deps. */
(function () {
  "use strict";
  const $ = s => document.querySelector(s);
  const fmt = x => (Math.round(x * 100) / 100);

  let net = null, B = null, S = null, lastMove = null, prevMove = -1, busy = false, actx = null;
  let hist = [];   // undo stack: snapshots taken before each human turn / pass (state is immutable)
  let moves = [];  // every committed ply in order (node index or B.pass) — drives share links;
                   // invariant: moves.length === S.moveNum
  let curKey = null;  // catalogue key of the current board (for share links + persistence)
  let wrHist = []; // black win-rate per move index (drives the win-rate graph)
  let perfHist = [], lastPerf = null; // engine sims/sec history (drives the performance analyzer)
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

  // board colour palettes per theme (the SVG is regenerated each draw, so it reads the live theme)
  const THEMES = {
    dark:  { boardBg: "url(#bgs)", line: "#454c5e", lineOp: 0.85, legal: "#5a6478", last: "#00ffc2",
             bStroke: "#78839b", wStroke: "#aab0c0", ownB: "#0a0d10", ownW: "#f4f4f2",
             gBg: "#0b0d13", gLine: "#2a313c", gPath: "#00ffc2" },
    light: { boardBg: "#efe8d6", line: "#3d4350", lineOp: 0.92, legal: "#7a8290", last: "#0a9c79",
             bStroke: "#14171d", wStroke: "#586071", ownB: "#2b3140", ownW: "#9fb2d6",
             gBg: "#f4efe3", gLine: "#cbc3b0", gPath: "#0a9c79" },
  };
  const theme = () => THEMES[document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark"];

  function renderSVG(board, colors, last, legal, analysis) {
    const T = theme();
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
      `<rect width="${W}" height="${H}" rx="18" fill="${T.boardBg}"/>`,
      `<rect width="${W}" height="${H}" rx="18" filter="url(#grain)" opacity="0.045"/>`];
    for (let e = 0; e < board.edges.length; e += 2) {
      const a = board.edges[e], b = board.edges[e + 1];
      o.push(`<line x1="${fmt(tx(coords[a][0]))}" y1="${fmt(ty(coords[a][1]))}" x2="${fmt(tx(coords[b][0]))}" y2="${fmt(ty(coords[b][1]))}" stroke="${T.line}" stroke-width="1.1" stroke-linecap="round" opacity="${T.lineOp}"/>`);
    }
    if (legal) for (let i = 0; i < n; i++) if (colors[i] === 0 && legal[i])
      o.push(`<circle cx="${fmt(tx(coords[i][0]))}" cy="${fmt(ty(coords[i][1]))}" r="${fmt(rS * 0.2)}" fill="${T.legal}" opacity="0.16"/>`);
    for (let i = 0; i < n; i++) {
      const c = colors[i]; if (c === 0) continue;
      const x = tx(coords[i][0]), y = ty(coords[i][1]);
      o.push(`<circle cx="${fmt(x)}" cy="${fmt(y)}" r="${fmt(rS)}" fill="${c === 1 ? "url(#bz)" : "url(#wz)"}" stroke="${c === 1 ? T.bStroke : T.wStroke}" stroke-width="0.9" filter="url(#sh)"/>`);
      if (last === i) o.push(`<circle cx="${fmt(x)}" cy="${fmt(y)}" r="${fmt(rS * 0.4)}" fill="none" stroke="${T.last}" stroke-width="2.2" filter="url(#gl)"/>`);
    }
    if (analysis) {
      const own = analysis.ownership;
      if (own) for (let i = 0; i < n; i++) {
        if (colors[i] !== 0) continue; const v = own[i]; if (Math.abs(v) < 0.18) continue;
        const x = tx(coords[i][0]), y = ty(coords[i][1]), sq = rS * 1.15;
        o.push(`<rect x="${fmt(x - sq / 2)}" y="${fmt(y - sq / 2)}" width="${fmt(sq)}" height="${fmt(sq)}" rx="2" fill="${v > 0 ? T.ownB : T.ownW}" opacity="${fmt(Math.min(0.5, 0.55 * Math.abs(v)))}"/>`);
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
  const thinking = on => $("#scan").classList.toggle("on", on);
  const sleep = () => new Promise(r => setTimeout(r, 0));
  function toast(text) {                              // transient top-center notice
    let t = $("#toast"); if (t) t.remove();
    t = document.createElement("div"); t.id = "toast"; t.textContent = text;
    document.body.appendChild(t);
    setTimeout(() => { t.classList.add("off"); setTimeout(() => t.remove(), 400); }, 6000);
  }
  function showComment(text) {                        // persistent note chip from a shared link.
    let c = $("#notechip"); if (c) c.remove();        // textContent only — never parsed as HTML,
    c = document.createElement("div");                // so script payloads stay inert text
    c.id = "notechip";
    const span = document.createElement("span"); span.textContent = text;
    const x = document.createElement("button"); x.textContent = "×"; x.onclick = () => c.remove();
    c.append(span, x);
    document.body.appendChild(c);
  }

  // ---------- local persistence (survives refresh; degrades silently without storage) ----------
  const LS_KEY = "eg-game-v1";
  const lsGet = () => { try { const o = JSON.parse(localStorage.getItem(LS_KEY) || "null"); return (o && o.v === 1) ? o : null; } catch (e) { return null; } };
  const lsSet = o => { try { localStorage.setItem(LS_KEY, JSON.stringify(o)); } catch (e) {} };
  const lsDel = () => { try { localStorage.removeItem(LS_KEY); } catch (e) {} };
  function persist() {                                // call after every committed ply
    if (!B || !S || !curKey) return;
    if (TG.isTerminal(S, B)) { lsDel(); return; }     // finished games clear the slot
    lsSet({ v: 1, frag: SHARE.serialize(curKey, B, moves), strength: $("#strength").value,
            opponent: $("#opponent").value, color: humanColor === TG.BLACK ? "black" : "white" });
  }

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
    const mid = ys(0.5).toFixed(1), cx = xs(cur).toFixed(1), T = theme();
    const tip = wrHist[cur] == null ? "" :
      `<circle cx="${cx}" cy="${ys(wrHist[cur]).toFixed(1)}" r="3.2" fill="${T.gPath}"/>`;
    el.innerHTML = `<svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="border-radius:8px;background:${T.gBg};border:1px solid ${T.gLine}">
      <line x1="${pad}" y1="${mid}" x2="${W - pad}" y2="${mid}" stroke="${T.gLine}" stroke-dasharray="3 3"/>
      <path d="${path}" fill="none" stroke="${T.gPath}" stroke-width="2" vector-effect="non-scaling-stroke"/>
      <line x1="${cx}" y1="${pad}" x2="${cx}" y2="${H - pad}" stroke="${T.line}" opacity="0.25"/>${dots}${tip}</svg>`;
  }
  // one cheap net forward → black win-rate for the current position; record it and redraw the curve
  function recordWR() {
    if (!net) return;
    const out = TG.forward(net, TG.encode(S, B).x, B.n, B.adj);
    const bw = S.toMove === TG.BLACK ? 0.5 * (out.value + 1) : 0.5 * (1 - out.value);
    wrHist[S.moveNum] = bw;
    $("#winpct").textContent = Math.round(bw * 100) + "%";
    drawWRGraph(S.moveNum);
  }

  // ---------- live engine-performance analyzer ----------
  // big sims/sec readout + sims/ms/nodes subline + a sparkline of the last ~40 engine searches.
  function updatePerf(p, push) {
    const el = $("#perf"); if (!el || !p) return;
    if (push) { perfHist.push(p.sps); if (perfHist.length > 40) perfHist.shift(); lastPerf = p; }
    const T = theme(), W = 256, H = 36, pad = 4, n = perfHist.length, mx = Math.max.apply(null, perfHist.concat(1));
    const xs = i => pad + (n < 2 ? 0 : (i / (n - 1)) * (W - 2 * pad)), ys = v => pad + (1 - v / mx) * (H - 2 * pad);
    let path = ""; perfHist.forEach((v, i) => { path += (path ? "L" : "M") + xs(i).toFixed(1) + " " + ys(v).toFixed(1) + " "; });
    el.innerHTML = `<div class="perfrow"><span class="perfbig">${p.sps.toLocaleString()}</span><span class="perfunit">sims/s</span></div>
      <div class="perfsub">${p.sims.toLocaleString()} sims &middot; ${p.ms} ms/move &middot; ${p.n} nodes</div>
      <svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="border-radius:8px;background:${T.gBg};border:1px solid ${T.gLine};margin-top:6px">
        <path d="${path}" fill="none" stroke="${T.gPath}" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>`;
  }
  // build a perf record from a finished search (sims + elapsed ms) and push it to the analyzer
  const pushPerf = r => { if (r && r.sims) updatePerf({ sps: Math.round(r.sims / Math.max(r.ms, 1) * 1000), sims: r.sims, ms: Math.round(r.ms * 10) / 10, n: B.n }, true); };

  // ---------- render state ----------
  function draw() {
    const legal = TG.legalMoves(S, B);
    const term = TG.isTerminal(S, B), over = term || resigned !== null;
    const sd = Math.round(TG.scoreDiff(S, B) * 10) / 10;
    // at the natural end, show the actual area result on the board (territory overlay)
    const overlay = term ? { moves: [], ownership: TG.ownership(S, B), best: null } : null;
    $("#board").innerHTML = renderSVG(B, S.colors, lastMove, legal, overlay);
    $("#turndot").style.background = S.toMove === TG.BLACK ? "#111417" : "#f4f4f2";
    $("#turn").textContent = over ? "game over"
                                  : (S.toMove === TG.BLACK ? "Black" : "White") + " to move";
    $("#score").textContent = (sd > 0 ? "B+" : sd < 0 ? "W+" : "") + (sd ? Math.abs(sd) : "0");
    $("#move").textContent = "move " + S.moveNum + (S.passCount ? " · " + S.passCount + "p" : "");
    if (term) { const w = TG.winner(S, B) === TG.BLACK ? "Black" : "White"; $("#msg").innerHTML = `<span class="win">${w} wins by ${Math.abs(sd)}</span>`; }
    else if (resigned !== null) { const w = resigned === TG.BLACK ? "White" : "Black"; $("#msg").innerHTML = `<span class="win">${w} wins by resignation</span>`; }
    else if (S.passCount === 1) $("#msg").textContent = "one pass — pass again to end and score";
    else $("#msg").textContent = "";
    if (over) showEndBanner(term, sd);
    else { const eb = $("#endbanner"); if (eb) eb.remove(); }
    if (S.moveNum > prevMove) clack(); prevMove = S.moveNum;
    $("#winpct").textContent = "—"; $("#topmv").innerHTML = "";
    const hint = $("#hint");                       // first-move hint: gone once a stone is down
    if (hint && S.moveNum > 0) hint.remove();
    const sg = $("#dlsgf");                        // SGF needs a coordinate grid: square only
    if (sg) sg.style.display = (curKey && curKey.startsWith("square")) ? "" : "none";
    bind();
  }
  function bind() {
    document.querySelectorAll("#board .hot").forEach(el =>
      el.addEventListener("click", () => onHuman(+el.dataset.node)));
  }
  function showEndBanner(term, sd) {               // unmissable result; no silent end states
    if ($("#endbanner")) return;
    const b = document.createElement("div"); b.id = "endbanner";
    const h = document.createElement("div"); h.className = "endtitle";
    const sub = document.createElement("div"); sub.className = "endsub";
    if (term) {
      h.textContent = (TG.winner(S, B) === TG.BLACK ? "Black" : "White") + ` wins by ${Math.abs(sd)}`;
      sub.textContent = `area scoring · komi ${B.komi} · territory shown on the board`;
    } else {
      h.textContent = (resigned === TG.BLACK ? "White" : "Black") + " wins";
      sub.textContent = "the engine resigned";
    }
    const row = document.createElement("div"); row.className = "endrow";
    const again = document.createElement("button"); again.textContent = "New game";
    again.onclick = () => { b.remove(); $("#reset").click(); };
    const look = document.createElement("button"); look.textContent = "Inspect board";
    look.className = "ghost"; look.onclick = () => b.remove();
    row.append(again, look);
    b.append(h, sub, row);
    document.body.appendChild(b);
  }

  // ---------- WASM engine (Web Worker; falls back to the JS engine) ----------
  // The worker is assembled from inline sources so the site stays one static file. Any
  // failure (no WASM, no SIMD, init error) leaves wasm = null and the JS engine plays.
  let wasm = null;            // { sps, search(moves, sims) -> Promise<{move,pi,value,ms,sims}> }
  let resigned = null;        // null | TG.BLACK | TG.WHITE — the side that resigned (engine only)
  let hopeless = 0;           // consecutive engine moves with a hopeless self-winrate
  // resignation is configurable: ?resign=wr,streak,minMove (or window.EG_RESIGN pre-load)
  const RESIGN = Object.assign({ wr: 0.04, streak: 4, minMove: 16 },
                               (typeof window !== "undefined" && window.EG_RESIGN) || {});
  try {
    const rs = new URLSearchParams(location.search).get("resign");
    if (rs) { const [w, s, m] = rs.split(",").map(Number);
      if (w > 0) RESIGN.wr = w; if (s > 0) RESIGN.streak = s; if (m >= 0) RESIGN.minMove = m; }
  } catch (e) {}
  function bootWasm() {
    try {
      if (new URLSearchParams(location.search).get("engine") === "js") return;  // forced fallback
      const srcEl = document.getElementById("wasm-src");
      if (!srcEl || typeof Worker === "undefined") return;
      const blob = new Blob([srcEl.textContent], { type: "text/javascript" });
      const w = new Worker(URL.createObjectURL(blob));
      const pend = new Map();
      let nextId = 1;
      w.onmessage = (e) => {
        const m = e.data;
        if (m.type === "ready") {
          wasm = {
            sps: m.sps,
            search(moves, sims, eps) {
              return new Promise((res, rej) => {
                const id = nextId++;
                pend.set(id, { res, rej });
                w.postMessage({ type: "search", id, moves, sims, eps: eps || 0 });
              });
            },
          };
          const tag = $("#engtag"); if (tag) tag.textContent = "engine: WASM ⚡ " + Math.round(m.sps) + " sims/s";
          return;
        }
        if (m.type === "init-error") { console.warn("WASM engine unavailable:", m.message); return; }
        if (m.type === "result") {
          const p = pend.get(m.id); if (!p) return;
          pend.delete(m.id);
          if (m.move < 0) p.rej(new Error(m.error || "search failed"));
          else {
            if (wasm && m.ms > 50) {                // live recalibration from real searches
              wasm.sps = 0.5 * wasm.sps + 0.5 * (m.sims / m.ms * 1000);
              const tag = $("#engtag"); if (tag) tag.textContent = "engine: WASM ⚡ " + Math.round(wasm.sps) + " sims/s";
            }
            p.res(m);
          }
        }
      };
      w.onerror = (e) => { console.warn("WASM worker error:", e.message); wasm = null; };
      const board = { n: B.n, komi: B.komi, edges: B.edges, static: B.static };
      w.postMessage({ type: "init", board, weights: WASMGLUE.serializeTGN1(net) });
      wasmWorker = w;
    } catch (e) { console.warn("WASM boot failed:", e); wasm = null; }
  }
  let wasmWorker = null;
  function rebootWasmForBoard() {                  // new board ⇒ new engine instance
    wasm = null;
    const tag = $("#engtag"); if (tag) tag.textContent = "engine: JS";
    if (wasmWorker) { wasmWorker.terminate(); wasmWorker = null; }
    bootWasm();
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
    return { move: s.best(), root: s.root, sims: i, ms: performance.now() - t0 };
  }

  async function engineReply() {                    // engine plays the side to move (one move)
    if (TG.isTerminal(S, B) || resigned !== null) return;
    thinking(true); await sleep();
    const budget = STRENGTH[$("#strength").value];
    let r = null;
    if (wasm) {                                     // same WAIT, ~10× the sims (gated: parity +
      try {                                         // head-to-head verified before default-on)
        const sims = Math.max(60, Math.min(4000, Math.round(wasm.sps * budget / 1000)));
        r = await wasm.search(moves, sims, 0);
      } catch (e) { console.warn("WASM search failed; falling back to JS:", e); r = null; }
    }
    if (!r) r = await searchMove(budget, 600);
    S = TG.play(S, r.move, B); moves.push(r.move); lastMove = (r.move === B.pass ? null : r.move);
    thinking(false); draw(); recordWR(); pushPerf(r); persist();
    if (r.move === B.pass && !TG.isTerminal(S, B))
      toast("The engine passes — pass too to end the game and score, or keep playing.");
    // resignation: hopeless self-winrate for RESIGN.streak consecutive engine moves
    const bw = wrHist[S.moveNum];
    if (typeof bw === "number" && S.moveNum >= RESIGN.minMove && !TG.isTerminal(S, B)) {
      const engineBlack = humanColor !== TG.BLACK;
      const engWr = engineBlack ? bw : 1 - bw;
      hopeless = engWr < RESIGN.wr ? hopeless + 1 : 0;
      if (hopeless >= RESIGN.streak) {
        resigned = engineBlack ? TG.BLACK : TG.WHITE;
        lsDel();                                    // a resigned game is finished
        draw();
        toast("The engine resigns. Good game!");
      }
    }
  }

  async function onHuman(node) {
    if (busy || resigned !== null) return;
    const opp = $("#opponent").value;
    if (opp === "engine" && S.toMove !== humanColor) return;   // not your turn — engine to play
    const legal = TG.legalMoves(S, B);
    if (!legal[node]) { $("#msg").textContent = "Illegal move"; return; }
    busy = true;
    try {
      hist.push({ s: S, last: lastMove });          // snapshot before this turn (for undo)
      S = TG.play(S, node, B); moves.push(node); lastMove = node; draw(); recordWR(); persist();
      if (!countedStart) {                            // count a real "game played" (first human move)
        countedStart = true; const k = $("#variant").value;
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
    thinking(false); pushPerf(r);
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
    $("#winpct").textContent = Math.round(bw * 100) + "%";
    $("#score").textContent = (lead >= 0 ? "B+" : "W+") + Math.abs(lead).toFixed(1);
    $("#topmv").innerHTML = top.slice(0, 5).map((m, i) => `<span class="k">${i + 1}</span> ${Math.round(m.wr * 100)}% &middot; ${m.vis}v`).join("<br>");
  }

  function newGame(key) {
    B = TG.makeBoard(BOARDS[key]); S = TG.newGame(B);
    curKey = key; lsDel();                            // a new game replaces the saved slot
    try { localStorage.setItem("eg-last-key", key); } catch (e) {}  // substrate preference
    lastMove = null; prevMove = 0; hist = []; moves = []; wrHist = []; perfHist = []; lastPerf = null; countedStart = false;
    resigned = null; hopeless = 0;
    $("#perf").innerHTML = ""; draw(); recordWR();
    rebootWasmForBoard();                             // engine instance is per-board-geometry
    // if the human chose White, the engine (Black) opens the game
    if ($("#opponent").value === "engine" && humanColor === TG.WHITE) {
      busy = true;
      (async () => { try { await engineReply(); } finally { busy = false; } })();
    }
  }

  // ---------- wire up ----------
  function init() {
    net = TG.loadNet(WEIGHTS_B64, CFG);
    // FAMILIES (baked by export_webapp from the catalogue) = [{family, items:[[key, size], ...]}, ...]
    const FAMS = (typeof FAMILIES !== "undefined" && FAMILIES.length)
      ? FAMILIES : [{ family: "Boards", items: Object.keys(BOARDS).map(k => [k, BOARDS[k].label || k]) }];
    const famSel = $("#family"), varSel = $("#variant");
    famSel.innerHTML = FAMS.map((g, i) => `<option value="${i}">${g.family}</option>`).join("")
      + `<option value="__3d">Diamond cubic (3D) →</option>`;   // lives on its own page
    const fillVariants = (fi) => { varSel.innerHTML = FAMS[fi].items.map(([k, sub]) => `<option value="${k}">${sub}</option>`).join(""); };
    const currentKey = () => varSel.value;
    const selectKey = (key) => {                          // point both dropdowns at a board key
      for (let i = 0; i < FAMS.length; i++) if (FAMS[i].items.some(([k]) => k === key)) { famSel.value = i; fillVariants(i); varSel.value = key; return; }
    };
    $("#analyze").onclick = async () => { if (busy) return; busy = true; try { await analyze(); } finally { busy = false; } };
    $("#undo").onclick = () => {
      if (busy || !hist.length) return;
      resigned = null; hopeless = 0;                  // undo reopens a resigned game
      const h = hist.pop(); S = h.s; lastMove = h.last; prevMove = S.moveNum; draw();
      moves.length = S.moveNum;                       // keep the share-link move list in sync
      wrHist.length = S.moveNum + 1; recordWR();      // trim the win-rate curve to match
      persist();
    };
    $("#pass").onclick = async () => { if (busy || resigned !== null) return; busy = true; try { hist.push({ s: S, last: lastMove }); S = TG.play(S, B.pass, B); moves.push(B.pass); lastMove = null; draw(); recordWR(); persist(); if ($("#opponent").value === "engine") await engineReply(); } finally { busy = false; } };
    $("#reset").onclick = () => { if (!busy) newGame(currentKey()); };
    famSel.onchange = () => {
      if (famSel.value === "__3d") { location.href = "./3d.html"; return; }
      if (busy) { selectKey(currentKey()); return; }
      fillVariants(+famSel.value); newGame(currentKey());
    };
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
    // theme (light/dark): a manual choice (persisted) wins; otherwise follow the device's setting,
    // so a phone in bright/outdoor light mode opens light, while a dark-mode system opens dark.
    const applyTheme = t => document.documentElement.setAttribute("data-theme", t);
    let savedTheme = null; try { savedTheme = localStorage.getItem("eg-theme"); } catch (e) {}
    const sysLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
    const startTheme = savedTheme || (sysLight ? "light" : "dark");
    applyTheme(startTheme); $("#light").checked = startTheme === "light";
    $("#light").onchange = () => {
      const t = $("#light").checked ? "light" : "dark";
      applyTheme(t); try { localStorage.setItem("eg-theme", t); } catch (e) {}
      draw(); if (S) drawWRGraph(S.moveNum); if (lastPerf) updatePerf(lastPerf, false);
    };
    // shareable links: copy serializes the CURRENT position into the fragment (lazy — we never
    // touch the URL on ordinary moves, so browser history stays clean)
    const copyShareLink = async (btn, opts) => {
      const frag = SHARE.serialize(currentKey(), B, moves, opts);
      history.replaceState(null, "", "#" + frag);
      let ok = true;
      try { await navigator.clipboard.writeText(location.href); }
      catch (e) {                                   // clipboard API can be unavailable (file://)
        try {
          const ta = document.createElement("textarea"); ta.value = location.href;
          document.body.appendChild(ta); ta.select(); ok = document.execCommand("copy"); ta.remove();
        } catch (e2) { ok = false; }
      }
      const old = btn.textContent;
      btn.textContent = ok ? "✓ Link copied" : "⚠ Copy failed — copy the URL bar";
      setTimeout(() => { btn.textContent = old; }, 1600);
    };
    $("#share").onclick = () => copyShareLink($("#share"));
    $("#sharea").onclick = () => {                  // analysis link, with an optional note
      let note = null;
      try { note = window.prompt("Add a note to this position? (optional, shown to the recipient)", ""); } catch (e) {}
      if (note === null) return;                    // cancelled
      copyShareLink($("#sharea"), { analysis: true, comment: note.trim() || undefined });
    };

    // install a replayed game (shared link or resumed save) into the live UI.
    // The engine never moves on its own right after install.
    function installGame(parsed, color) {
      const board = TG.makeBoard(BOARDS[parsed.key]);
      const r = SHARE.replay(TG, board, parsed);       // throws on any corruption
      selectKey(parsed.key);
      B = board; S = r.state; hist = r.snaps; curKey = parsed.key;
      moves = parsed.moves.map(m => m === SHARE.PASS ? B.pass : m);
      const lastPly = moves.length ? moves[moves.length - 1] : null;
      lastMove = (lastPly === null || lastPly === B.pass) ? null : lastPly;
      prevMove = S.moveNum; wrHist = []; perfHist = []; lastPerf = null; countedStart = false;
      resigned = null; hopeless = 0;
      humanColor = color != null ? color : S.toMove;
      $("#playercolor").value = humanColor === TG.BLACK ? "black" : "white";
      draw(); recordWR();
      rebootWasmForBoard();
    }

    // load a shared game from the URL fragment; any problem → friendly toast + fresh game
    function loadShared() {
      let parsed = null;
      try { parsed = SHARE.parse(location.hash); }
      catch (e) { toast(`Couldn't read the game link (${e.message}) — starting fresh.`); return false; }
      if (!parsed) return false;
      if (!BOARDS[parsed.key]) { toast(`This link uses a board this site doesn't have ("${parsed.key}") — starting fresh.`); return false; }
      try {
        installGame(parsed, null);                     // sharee plays the side to move
        toast(`Shared game loaded — move ${S.moveNum}, ${S.toMove === TG.BLACK ? "Black" : "White"} to play. That's you.`);
        if (parsed.comment) showComment(parsed.comment);
        if (parsed.analysis && !TG.isTerminal(S, B)) { // sender asked for the overlay open
          busy = true;
          analyze().finally(() => { busy = false; });  // async; board stays interactive after
        }
        return true;
      } catch (e) {
        toast(`Couldn't load the shared game (${e.message || e}) — starting fresh.`);
        return false;
      }
    }

    // non-blocking resume prompt for the previous (unfinished) game
    function offerResume(saved) {
      let parsed = null;
      try { parsed = SHARE.parse("#" + saved.frag); } catch (e) { return; }
      if (!parsed || !parsed.moves.length || !BOARDS[parsed.key]) return;   // empty/unknown: no prompt
      const bar = document.createElement("div"); bar.id = "resume";
      const label = (BOARDS[parsed.key].label || parsed.key);
      bar.innerHTML = `<span>Resume your previous game? <b>${label}</b>, move ${parsed.moves.length}</span>`;
      const yes = document.createElement("button"); yes.textContent = "Resume";
      const no = document.createElement("button"); no.textContent = "Discard"; no.className = "ghost";
      bar.append(yes, no); document.body.appendChild(bar);
      no.onclick = () => bar.remove();
      yes.onclick = () => {
        bar.remove();
        try {
          installGame(parsed, saved.color === "white" ? TG.WHITE : TG.BLACK);
          $("#strength").value = saved.strength || $("#strength").value;
          $("#opponent").value = saved.opponent || $("#opponent").value;
          persist();
          // refresh could have landed mid-engine-think: it's still the engine's turn — offer
          // a one-tap way to let it move (it never moves unprompted after a restore)
          if ($("#opponent").value === "engine" && S.toMove !== humanColor && !TG.isTerminal(S, B)) {
            const go = document.createElement("div"); go.id = "resume";
            go.innerHTML = "<span>It's the engine's turn.</span>";
            const btn = document.createElement("button"); btn.textContent = "▶ Let it move";
            go.appendChild(btn); document.body.appendChild(go);
            btn.onclick = async () => { go.remove(); if (!busy) { busy = true; try { await engineReply(); persist(); } finally { busy = false; } } };
          }
        } catch (e) { toast("Couldn't resume the saved game — it may be from an older version."); }
      };
    }

    // ---- game records (download / load) ----
    const download = (text, name, type) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name;
      a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    };
    const resultStr = () => {
      if (resigned !== null) return resigned === TG.BLACK ? "W+R" : "B+R";
      if (!TG.isTerminal(S, B)) return null;
      const sd = Math.round(TG.scoreDiff(S, B) * 10) / 10;
      return (sd > 0 ? "B+" : "W+") + Math.abs(sd);
    };
    $("#dlrec").onclick = () => {
      const rec = REC.buildRecord(curKey, B, moves,
        { date: new Date().toISOString().slice(0, 10), result: resultStr() });
      download(JSON.stringify(rec, null, 1), `euclidean-go-${curKey}-${rec.date}.json`, "application/json");
    };
    $("#dlsgf").onclick = () => {
      download(REC.toSGF(curKey, B, moves, { result: resultStr() }),
               `euclidean-go-${curKey}.sgf`, "application/x-go-sgf");
    };
    $("#ldrec").onclick = () => $("#recfile").click();
    $("#recfile").onchange = async () => {
      const f = $("#recfile").files[0]; $("#recfile").value = "";
      if (!f) return;
      try {
        const rec = JSON.parse(await f.text());
        const parsed = REC.toParsed(rec, TG, BOARDS);
        installGame(parsed, null);
        toast(`Record loaded — ${BOARDS[parsed.key].label || parsed.key}, move ${S.moveNum}.`);
      } catch (e) {
        toast(`Couldn't load that record (${e.message || "not valid JSON"}).`);
      }
    };

    const savedGame = lsGet();                         // read BEFORE newGame clears the slot
    if (!loadShared()) {
      // returning visitors get their last substrate; first-timers a small hex board — friendly
      // size, instantly playable, and unmistakably "not a normal Go board"
      let lastKey = null;
      try { lastKey = localStorage.getItem("eg-last-key"); } catch (e) {}
      const firstRun = !lastKey;
      const startKey = (lastKey && BOARDS[lastKey]) ? lastKey
                     : (BOARDS["hex_small"] ? "hex_small" : Object.keys(BOARDS)[0]);
      selectKey(startKey);
      newGame(currentKey());
      if (savedGame) offerResume(savedGame);
      if (firstRun) {
        const h = $("#hint");
        if (h) h.textContent = "tap any intersection to place a stone · more boards in the menu";
      }
    }
    const sp = $("#splash");                       // loading splash: engine is ready, fade it out
    if (sp) { sp.classList.add("off"); setTimeout(() => sp.remove(), 450); }
  }
  if (document.readyState !== "loading") init(); else document.addEventListener("DOMContentLoaded", init);
})();
