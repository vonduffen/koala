/* Native-app client: renders the board from embedded geometry and drives the C++ server over /api.
   The engine (net forward + MCTS) runs natively in C++ — this file is just UI + rendering. */
(function () {
  "use strict";
  const $ = s => document.querySelector(s);
  const fmt = x => Math.round(x * 100) / 100;
  const api = async (path, body) => (await fetch(path, { method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined })).json();

  function b64bytes(s) { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
  const b64f = s => new Float32Array(b64bytes(s).buffer);
  const b64i = s => new Int32Array(b64bytes(s).buffer);
  const GEOM = {};
  for (const k in BOARDSGEOM) {
    const g = BOARDSGEOM[k], cf = b64f(g.coords), coords = [];
    for (let i = 0; i < g.n; i++) coords.push([cf[2 * i], cf[2 * i + 1]]);
    GEOM[k] = { n: g.n, coords, edges: b64i(g.edges) };
  }

  let busy = false, prevMove = -1, actx = null, ST = null;
  let wrHist = [];   // black win-rate per move index (drives the win-rate graph)

  // KaTrain-style win-rate timeline: black's win prob each move; red dot where the side that just
  // moved lost ≥12% (a blunder). Drawn into #wrgraph.
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
    el.innerHTML = `<svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="border-radius:8px;background:#0b0d13;border:1px solid var(--line)">
      <line x1="${pad}" y1="${mid}" x2="${W - pad}" y2="${mid}" stroke="#2a313c" stroke-dasharray="3 3"/>
      <path d="${path}" fill="none" stroke="#00ffc2" stroke-width="2" vector-effect="non-scaling-stroke"/>
      <line x1="${cx}" y1="${pad}" x2="${cx}" y2="${H - pad}" stroke="#ffffff22"/>${dots}</svg>`;
  }

  function minSpacing(coords) {
    const n = coords.length; if (n < 2) return 1;
    const nn = new Float64Array(n).fill(Infinity);
    for (let i = 0; i < n; i++) { const xi = coords[i][0], yi = coords[i][1]; for (let j = 0; j < n; j++) { if (i === j) continue; const d = Math.hypot(coords[j][0] - xi, coords[j][1] - yi); if (d < nn[i]) nn[i] = d; } }
    return Array.from(nn).sort((a, b) => a - b)[Math.floor(0.02 * n)] || 1;
  }
  const DEFS = '<defs>' +
    '<radialGradient id="bz" cx="0.36" cy="0.30" r="0.78"><stop offset="0%" stop-color="#4c5168"/><stop offset="40%" stop-color="#191c26"/><stop offset="100%" stop-color="#04050a"/></radialGradient>' +
    '<radialGradient id="wz" cx="0.36" cy="0.30" r="0.82"><stop offset="0%" stop-color="#ffffff"/><stop offset="56%" stop-color="#e6e9f1"/><stop offset="100%" stop-color="#b9bfce"/></radialGradient>' +
    '<radialGradient id="bgs" cx="0.5" cy="0.40" r="0.85"><stop offset="0%" stop-color="#1a1f2c"/><stop offset="100%" stop-color="#0a0c12"/></radialGradient>' +
    '<filter id="sh" x="-40%" y="-40%" width="180%" height="180%"><feDropShadow dx="0" dy="1.4" stdDeviation="1.6" flood-color="#000" flood-opacity="0.6"/></filter>' +
    '<filter id="gl" x="-90%" y="-90%" width="280%" height="280%"><feDropShadow dx="0" dy="0" stdDeviation="3.2" flood-color="#00ffc2" flood-opacity="0.95"/></filter>' +
    '<filter id="grain"><feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" stitchTiles="stitch"/><feColorMatrix type="saturate" values="0"/></filter></defs>';

  function renderSVG(g, colors, last, legal, analysis) {
    const W = 760, margin = 26, coords = g.coords, n = g.n;
    const spc = minSpacing(coords), pad = 0.55 * spc;
    let mnx = Infinity, mny = Infinity, mxx = -Infinity, mxy = -Infinity;
    for (const [x, y] of coords) { if (x < mnx) mnx = x; if (y < mny) mny = y; if (x > mxx) mxx = x; if (y > mxy) mxy = y; }
    mnx -= pad; mny -= pad; mxx += pad; mxy += pad;
    const sx = Math.max(mxx - mnx, 1e-9), sy = Math.max(mxy - mny, 1e-9);
    const scale = (W - 2 * margin) / Math.max(sx, sy), H = Math.round(sy * scale + 2 * margin);
    const tx = x => margin + (x - mnx) * scale, ty = y => margin + (mxy - y) * scale;
    const rS = 0.46 * spc * scale, rH = 0.5 * spc * scale;
    const o = [`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`, DEFS,
      `<rect width="${W}" height="${H}" rx="18" fill="url(#bgs)"/>`, `<rect width="${W}" height="${H}" rx="18" filter="url(#grain)" opacity="0.045"/>`];
    for (let e = 0; e < g.edges.length; e += 2) { const a = g.edges[e], b = g.edges[e + 1];
      o.push(`<line x1="${fmt(tx(coords[a][0]))}" y1="${fmt(ty(coords[a][1]))}" x2="${fmt(tx(coords[b][0]))}" y2="${fmt(ty(coords[b][1]))}" stroke="#454c5e" stroke-width="1.1" stroke-linecap="round" opacity="0.85"/>`); }
    if (legal) for (let i = 0; i < n; i++) if (colors[i] === 0 && legal[i]) o.push(`<circle cx="${fmt(tx(coords[i][0]))}" cy="${fmt(ty(coords[i][1]))}" r="${fmt(rS * 0.2)}" fill="#5a6478" opacity="0.16"/>`);
    for (let i = 0; i < n; i++) { const c = colors[i]; if (!c) continue; const x = tx(coords[i][0]), y = ty(coords[i][1]);
      o.push(`<circle cx="${fmt(x)}" cy="${fmt(y)}" r="${fmt(rS)}" fill="${c === 1 ? "url(#bz)" : "url(#wz)"}" stroke="${c === 1 ? "#000" : "#aab0c0"}" stroke-width="0.7" filter="url(#sh)"/>`);
      if (last === i) o.push(`<circle cx="${fmt(x)}" cy="${fmt(y)}" r="${fmt(rS * 0.4)}" fill="none" stroke="#00ffc2" stroke-width="2.2" filter="url(#gl)"/>`); }
    if (analysis) for (const m of analysis.top) {
      const x = tx(coords[m.node][0]), y = ty(coords[m.node][1]), q = Math.max(0, Math.min(1, m.q));
      const best = analysis.best === m.node;
      o.push(`<circle${best ? ' class="bestmv"' : ""} cx="${fmt(x)}" cy="${fmt(y)}" r="${fmt(rS)}" fill="hsl(${Math.round(135 * q)},70%,45%)" opacity="0.94" stroke="${best ? "#fff" : "#0a0d10cc"}" stroke-width="${best ? 2.4 : 0.9}"${best ? ' filter="url(#gl)"' : ""}/>`);
      o.push(`<text x="${fmt(x)}" y="${fmt(y + rS * 0.34)}" font-size="${fmt(rS * 0.78)}" fill="#fff" text-anchor="middle" font-family="ui-monospace,Menlo,monospace" font-weight="700" style="pointer-events:none">${m.label}</text>`);
    }
    for (let i = 0; i < n; i++) o.push(`<circle class="hot" data-node="${i}" cx="${fmt(tx(coords[i][0]))}" cy="${fmt(ty(coords[i][1]))}" r="${fmt(rH)}" fill="transparent"/>`);
    o.push("</svg>"); return o.join("");
  }

  function clack() { if (!$("#snd").checked) return; try { actx = actx || new (window.AudioContext || window.webkitAudioContext)(); const t = actx.currentTime, o = actx.createOscillator(), g = actx.createGain(); o.type = "triangle"; o.frequency.setValueAtTime(260, t); o.frequency.exponentialRampToValueAtTime(92, t + 0.05); g.gain.setValueAtTime(0.16, t); g.gain.exponentialRampToValueAtTime(0.0007, t + 0.13); o.connect(g).connect(actx.destination); o.start(t); o.stop(t + 0.14); } catch (e) {} }
  const setWin = bw => { $("#winfill").style.height = Math.round(bw * 100) + "%"; };
  const thinking = on => $("#scan").classList.toggle("on", on);

  function render(st) {
    ST = st; const g = GEOM[st.key];
    if (st.moveNum === 0) wrHist = [];                  // fresh game → clear the curve
    if (typeof st.winrate === "number") wrHist[st.moveNum] = st.winrate;
    drawWRGraph(st.moveNum);
    $("#board").innerHTML = renderSVG(g, st.colors, st.last, st.legal, null);
    const sd = Math.round(st.scoreDiff * 10) / 10;
    $("#turndot").style.background = st.toMove === 1 ? "#111417" : "#f4f4f2";
    $("#turn").textContent = (st.toMove === 1 ? "Black" : "White") + " to move";
    $("#score").textContent = (sd > 0 ? "B+" : sd < 0 ? "W+" : "") + (sd ? Math.abs(sd) : "0");
    $("#move").textContent = "move " + st.moveNum + (st.passes ? " · " + st.passes + "p" : "");
    if (st.terminal) { const w = sd > 0 ? "Black" : "White"; $("#msg").innerHTML = `<span class="win">${w} wins by ${Math.abs(sd)}</span>`; } else $("#msg").textContent = "";
    if (st.moveNum > prevMove) clack(); prevMove = st.moveNum;
    $("#winpct").textContent = "—"; $("#topmv").innerHTML = "";
    bind();
  }
  function bind() { document.querySelectorAll("#board .hot").forEach(el => el.addEventListener("click", () => onHuman(+el.dataset.node))); }

  async function analyze() {
    if (ST.terminal) return; thinking(true);
    const a = await api("/api/analyze", {});
    thinking(false);
    if (a.error) { $("#msg").textContent = a.error; return; }
    const maxf = a.top.length ? a.top[0].frac : 1;
    const moves = a.top.map(m => ({ node: m.node, q: m.frac / maxf, label: Math.round(100 * m.frac / maxf) }));
    $("#board").innerHTML = renderSVG(GEOM[ST.key], ST.colors, ST.last, ST.legal, { top: moves, best: a.best }); bind();
    setWin(a.black_winrate); $("#winpct").textContent = Math.round(a.black_winrate * 100) + "%";
    $("#score").textContent = (a.score_lead >= 0 ? "B+" : "W+") + Math.abs(a.score_lead).toFixed(1);
    $("#topmv").innerHTML = a.top.slice(0, 5).map((m, i) => `<span class="k">${i + 1}</span> node ${m.node} · ${Math.round(m.frac * 100)}% visits`).join("<br>");
  }
  async function onHuman(node) {
    if (busy) return; if (!ST.legal[node]) { $("#msg").textContent = "Illegal move"; return; }
    busy = true;
    try {
      render(await api("/api/move", { node }));
      if ($("#opponent").value === "engine" && !ST.terminal) { thinking(true); render(await api("/api/engine", {})); thinking(false); }
      if ($("#auto").checked && !ST.terminal) await analyze();
    } finally { busy = false; }
  }

  async function init() {
    const tilings = await api("/api/tilings");
    $("#tiling").innerHTML = tilings.map(t => `<option value="${t.key}">${t.label}</option>`).join("");
    $("#analyze").onclick = async () => { if (busy) return; busy = true; try { await analyze(); } finally { busy = false; } };
    $("#pass").onclick = async () => { if (busy) return; busy = true; try { render(await api("/api/pass", {})); if ($("#opponent").value === "engine" && !ST.terminal) { thinking(true); render(await api("/api/engine", {})); thinking(false); } } finally { busy = false; } };
    $("#undo").onclick = async () => {
      if (busy) return; busy = true;
      try {
        // playing the engine ⇒ undo two states (your move + its reply); hot-seat ⇒ one
        const steps = $("#opponent").value === "engine" ? 2 : 1;
        let st = ST;
        for (let i = 0; i < steps; i++) { if ((st && st.moveNum > 0)) st = await api("/api/undo", {}); else break; }
        wrHist.length = (st ? st.moveNum + 1 : 0);   // trim the win-rate curve to match
        render(st);
      } finally { busy = false; }
    };
    $("#reset").onclick = async () => { if (busy) return; setWin(0.5); render(await api("/api/reset", { key: $("#tiling").value })); };
    $("#tiling").onchange = $("#reset").onclick;
    const st = await api("/api/state"); $("#tiling").value = st.key; setWin(0.5); render(st);
  }
  if (document.readyState !== "loading") init(); else document.addEventListener("DOMContentLoaded", init);
})();
