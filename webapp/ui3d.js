/* Euclidean Go — 3D: rotatable three.js board for the diamond-cubic lattice.
   Same verified engine as the 2D app (TG rules + WASM with JS fallback); only the renderer
   is new. The engine never trained on 3D — both players are visitors here. */
(function () {
  "use strict";
  const $ = s => document.querySelector(s);
  const T = window.THREE;

  // ---------- game state ----------
  let B = null, S = null, moves = [], hist = [], lastMove = null, busy = false, curKey = null;
  let humanColor = TG.BLACK, net = null, resigned = false;
  const STRENGTH = { Fast: 700, Normal: 1800, Strong: 4500 };

  // ---------- three.js scene ----------
  let scene, camera, renderer, raycaster, group;
  let stoneMeshes = [], hitMeshes = [], markerMeshes = [], lastRing = null, hoverIdx = -1;
  let yaw = 0.7, pitch = 0.45, dist = 4.2, panY = 0;
  let P3 = null;                                 // centered [N,3] positions (lattice units → scene)
  // win-rate HUD: drawn INTO the WebGL canvas (a CanvasTexture on an orthographic overlay),
  // so screenshots and canvas recordings capture the engine's live evaluation
  let hudScene, hudCam, hudTex, hudCanvas, hudPlane, wrHist = [];
  const HUD_W = 460, HUD_H = 132;

  const COL = { bg: 0x0b0d13, line: 0x39414f, marker: 0x4a5266, markerLegal: 0x5a6478,
                black: 0x14171d, white: 0xe8ebf2, accent: 0x00ffc2 };

  function buildScene() {
    const el = $("#stage3d");
    scene = new T.Scene();
    scene.background = new T.Color(COL.bg);
    scene.fog = new T.Fog(COL.bg, 6.0, 11.0);
    camera = new T.PerspectiveCamera(42, el.clientWidth / el.clientHeight, 0.1, 100);
    renderer = new T.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(renderer.domElement);
    raycaster = new T.Raycaster();
    scene.add(new T.AmbientLight(0xffffff, 0.55));
    const d1 = new T.DirectionalLight(0xffffff, 0.9); d1.position.set(3, 5, 4); scene.add(d1);
    const d2 = new T.DirectionalLight(0x88aaff, 0.25); d2.position.set(-4, -2, -3); scene.add(d2);
    group = new T.Group();
    scene.add(group);
    // HUD overlay pass (pixel-space orthographic scene sharing the same canvas)
    hudCanvas = document.createElement("canvas");
    hudCanvas.width = HUD_W * 2; hudCanvas.height = HUD_H * 2;       // 2× for crispness
    hudTex = new T.CanvasTexture(hudCanvas);
    hudScene = new T.Scene();
    hudCam = new T.OrthographicCamera(0, el.clientWidth, el.clientHeight, 0, -1, 1);
    hudPlane = new T.Mesh(new T.PlaneGeometry(HUD_W, HUD_H),
                          new T.MeshBasicMaterial({ map: hudTex, transparent: true }));
    hudScene.add(hudPlane);
    const placeHUD = () => {
      hudCam.right = el.clientWidth; hudCam.top = el.clientHeight;
      hudCam.updateProjectionMatrix();
      hudPlane.position.set(HUD_W / 2 + 24, HUD_H / 2 + 24, 0);   // bottom-left (info panel
                                                                  //  owns bottom-right)
    };
    placeHUD();
    renderer.autoClear = false;
    window.addEventListener("resize", () => {
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(el.clientWidth, el.clientHeight);
      placeHUD();
    });
    requestAnimationFrame(tick);
  }

  function tick() {
    const cp = Math.max(-1.35, Math.min(1.35, pitch));
    camera.position.set(dist * Math.cos(cp) * Math.sin(yaw),
                        dist * Math.sin(cp) + panY,
                        dist * Math.cos(cp) * Math.cos(yaw));
    camera.lookAt(0, panY, 0);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.clearDepth();
    renderer.render(hudScene, hudCam);
    requestAnimationFrame(tick);
  }

  // KaTrain-style win-rate timeline, drawn at 2× into the HUD texture
  function drawHUD() {
    const c = hudCanvas.getContext("2d");
    const W = hudCanvas.width, H = hudCanvas.height, s = 2;
    c.clearRect(0, 0, W, H);
    c.fillStyle = "rgba(13,16,23,0.88)"; c.strokeStyle = "#2a313c"; c.lineWidth = 2;
    const rr = (x, y, w, h, r) => { c.beginPath(); c.roundRect(x, y, w, h, r); };
    rr(1, 1, W - 2, H - 2, 16); c.fill(); c.stroke();
    c.fillStyle = "#828c9c"; c.font = `600 ${11 * s}px -apple-system, sans-serif`;
    c.fillText("W I N   R A T E   ·   B L A C K", 18 * s, 22 * s);
    const gx = 18 * s, gy = 30 * s, gw = W - 116 * s, gh = H - 44 * s;
    c.strokeStyle = "#39414f"; c.setLineDash([5, 5]); c.beginPath();
    c.moveTo(gx, gy + gh / 2); c.lineTo(gx + gw, gy + gh / 2); c.stroke(); c.setLineDash([]);
    const pts = wrHist.map((v, i) => v == null ? null : [
      gx + (wrHist.length < 2 ? 0 : (i / (wrHist.length - 1)) * gw),
      gy + (1 - v) * gh]).filter(Boolean);
    if (pts.length >= 2) {
      c.strokeStyle = "#00ffc2"; c.lineWidth = 2.5 * s; c.lineJoin = "round"; c.beginPath();
      pts.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.stroke();
    }
    if (pts.length) {
      const last = pts[pts.length - 1];
      c.fillStyle = "#00ffc2"; c.beginPath(); c.arc(last[0], last[1], 4 * s, 0, 7); c.fill();
    }
    const bw = wrHist.length ? wrHist[wrHist.length - 1] : null;
    if (bw != null) {
      c.fillStyle = "#00ffc2"; c.font = `700 ${30 * s}px ui-monospace, Menlo, monospace`;
      c.fillText(Math.round(bw * 100) + "%", W - 92 * s, H / 2 + 4 * s);
      c.fillStyle = "#828c9c"; c.font = `${10 * s}px -apple-system, sans-serif`;
      c.fillText("net eval, live", W - 92 * s, H / 2 + 20 * s);
    }
    hudTex.needsUpdate = true;
  }

  function buildBoardMeshes(rec) {
    group.clear();
    stoneMeshes = []; hitMeshes = []; markerMeshes = []; lastRing = null;
    const raw = rec.coords3;                              // Float32Array [N*3]
    const n = rec.n;
    // center + scale to a ~2.6-unit box
    let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) {
      mn[a] = Math.min(mn[a], raw[3 * i + a]); mx[a] = Math.max(mx[a], raw[3 * i + a]);
    }
    const span = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) || 1;
    const sc = 2.6 / span;
    P3 = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++)
      P3[3 * i + a] = (raw[3 * i + a] - (mn[a] + mx[a]) / 2) * sc;

    // bonds
    const pos = [];
    for (let e = 0; e < rec.edgesI.length; e += 2) {
      const a = rec.edgesI[e], b = rec.edgesI[e + 1];
      pos.push(P3[3 * a], P3[3 * a + 1], P3[3 * a + 2], P3[3 * b], P3[3 * b + 1], P3[3 * b + 2]);
    }
    const lg = new T.BufferGeometry();
    lg.setAttribute("position", new T.Float32BufferAttribute(pos, 3));
    group.add(new T.LineSegments(lg, new T.LineBasicMaterial({ color: COL.line, transparent: true, opacity: 0.6 })));

    // markers (empty points), stones (hidden until placed), hit spheres
    const mGeo = new T.SphereGeometry(0.020, 10, 8);
    const sGeo = new T.SphereGeometry(0.105, 24, 18);
    const rGeo = new T.SphereGeometry(0.109, 24, 18);     // black-stone rim shell (the 2D
    const hGeo = new T.SphereGeometry(0.16, 8, 6);        //  contrast lesson, in 3D)
    const blackMat = new T.MeshStandardMaterial({ color: 0x1b2029, roughness: 0.32, metalness: 0.35 });
    const whiteMat = new T.MeshStandardMaterial({ color: COL.white, roughness: 0.30, metalness: 0.05 });
    const rimMat = new T.MeshBasicMaterial({ color: 0x78839b, wireframe: true,
                                             transparent: true, opacity: 0.35 });
    const hideMat = new T.MeshBasicMaterial({ visible: false });
    for (let i = 0; i < n; i++) {
      const x = P3[3 * i], y = P3[3 * i + 1], z = P3[3 * i + 2];
      const m = new T.Mesh(mGeo, new T.MeshBasicMaterial({ color: COL.marker }));
      m.position.set(x, y, z); group.add(m); markerMeshes.push(m);
      const s = new T.Mesh(sGeo, blackMat.clone()); s.visible = false;
      s.position.set(x, y, z); group.add(s); stoneMeshes.push(s);
      const rim = new T.Mesh(rGeo, rimMat); rim.visible = false;
      rim.position.set(x, y, z); group.add(rim); s.userData.rim = rim;
      const h = new T.Mesh(hGeo, hideMat); h.position.set(x, y, z);
      h.userData.node = i; group.add(h); hitMeshes.push(h);
    }
    stoneMeshes.blackMat = blackMat; stoneMeshes.whiteMat = whiteMat;
    const ring = new T.Mesh(new T.SphereGeometry(0.135, 24, 18),
      new T.MeshBasicMaterial({ color: COL.accent, wireframe: true, transparent: true, opacity: 0.85 }));
    ring.visible = false; group.add(ring); lastRing = ring;
  }

  function syncBoard() {
    const legal = TG.legalMoves(S, B);
    const myTurn = $("#opponent").value !== "engine" || S.toMove === humanColor;
    for (let i = 0; i < B.n; i++) {
      const c = S.colors[i];
      stoneMeshes[i].visible = c !== 0;
      stoneMeshes[i].userData.rim.visible = c === 1;       // steel rim on black stones only
      if (c === 1) stoneMeshes[i].material = stoneMeshes.blackMat;
      if (c === 2) stoneMeshes[i].material = stoneMeshes.whiteMat;
      markerMeshes[i].visible = c === 0;
      const playable = !busy && !resigned && myTurn && c === 0 && legal[i];
      markerMeshes[i].material.color.setHex(playable ? COL.markerLegal : COL.marker);
      markerMeshes[i].scale.setScalar(playable ? 1.4 : 1.0);
      hitMeshes[i].userData.enabled = playable;
    }
    if (lastMove != null && lastMove < B.n) {
      lastRing.visible = true;
      lastRing.position.set(P3[3 * lastMove], P3[3 * lastMove + 1], P3[3 * lastMove + 2]);
    } else lastRing.visible = false;

    const term = TG.isTerminal(S, B);
    const sd = Math.round(TG.scoreDiff(S, B) * 10) / 10;
    // live evaluation for the HUD: one cheap net forward per position
    if (wrHist.length !== S.moveNum + 1 || wrHist[S.moveNum] == null) {
      wrHist.length = S.moveNum + 1;                       // handles undo trims too
      const out = TG.evaluate(net, S, B);
      wrHist[S.moveNum] = S.toMove === TG.BLACK ? 0.5 * (out.value + 1) : 0.5 * (1 - out.value);
      drawHUD();
    }
    $("#turn3").textContent = (term || resigned) ? "game over"
      : (S.toMove === TG.BLACK ? "Black" : "White") + " to move";
    $("#move3").textContent = "move " + S.moveNum + (S.passCount ? " · " + S.passCount + "p" : "");
    $("#score3").textContent = (sd > 0 ? "B+" : sd < 0 ? "W+" : "") + (sd ? Math.abs(sd) : "0");
    if (term) banner((TG.winner(S, B) === TG.BLACK ? "Black" : "White") + ` wins by ${Math.abs(sd)}`,
                     `area scoring · komi ${B.komi}`);
    else if (resigned) banner((humanColor === TG.BLACK ? "Black" : "White") + " wins", "the engine resigned");
    else if (S.passCount === 1) note("one pass — pass again to end and score");
    else note("");
  }
  const note = t => { $("#msg3").textContent = t; };
  function banner(title, sub) {
    if ($("#endbanner")) return;
    const b = document.createElement("div"); b.id = "endbanner";
    b.innerHTML = `<div class="endtitle"></div><div class="endsub"></div><div class="endrow"></div>`;
    b.querySelector(".endtitle").textContent = title;
    b.querySelector(".endsub").textContent = sub;
    const again = document.createElement("button"); again.textContent = "New game";
    again.onclick = () => { b.remove(); newGame(curKey); };
    b.querySelector(".endrow").appendChild(again);
    document.body.appendChild(b);
  }

  // ---------- WASM engine (same worker pattern + gates as the 2D app) ----------
  let wasm = null, wasmWorker = null;
  function bootWasm() {
    try {
      wasm = null; $("#engtag3").textContent = "engine: JS";
      if (wasmWorker) { wasmWorker.terminate(); wasmWorker = null; }
      const srcEl = document.getElementById("wasm-src");
      if (!srcEl || typeof Worker === "undefined") return;
      const w = new Worker(URL.createObjectURL(new Blob([srcEl.textContent], { type: "text/javascript" })));
      const pend = new Map(); let nextId = 1;
      w.onmessage = (e) => {
        const m = e.data;
        if (m.type === "ready") {
          wasm = { sps: m.sps, search: (mv, sims) => new Promise((res, rej) => {
            const id = nextId++; pend.set(id, { res, rej });
            w.postMessage({ type: "search", id, moves: mv, sims, eps: 0 });
          }) };
          $("#engtag3").textContent = "engine: WASM ⚡ " + Math.round(m.sps) + " sims/s";
        } else if (m.type === "result") {
          const p = pend.get(m.id); if (!p) return; pend.delete(m.id);
          if (m.move < 0) p.rej(new Error(m.error || "search failed"));
          else {
            if (wasm && m.ms > 50) {
              wasm.sps = 0.5 * wasm.sps + 0.5 * (m.sims / m.ms * 1000);
              $("#engtag3").textContent = "engine: WASM ⚡ " + Math.round(wasm.sps) + " sims/s";
            }
            p.res(m);
          }
        }
      };
      w.onerror = () => { wasm = null; };
      w.postMessage({ type: "init",
        board: { n: B.n, komi: B.komi, edges: B.edges, static: B.static },
        weights: WASMGLUE.serializeTGN1(net) });
      wasmWorker = w;
    } catch (e) { wasm = null; }
  }

  async function searchJS(budgetMs) {                       // fallback: time-budgeted JS MCTS
    const s = TG.makeSearcher(S, B, net);
    if (s.terminal) return { move: B.pass };
    const t0 = performance.now(); let i = 0;
    while (i < 600 && performance.now() - t0 < budgetMs) {
      const cs = performance.now();
      do { s.sim(); i++; } while (i < 600 && performance.now() - cs < 45 && performance.now() - t0 < budgetMs);
      await new Promise(r => setTimeout(r, 0));
    }
    return { move: s.best() };
  }

  async function engineReply() {
    if (TG.isTerminal(S, B) || resigned) return;
    $("#thinking").style.opacity = 1;
    const budget = STRENGTH[$("#strength").value];
    let r = null;
    if (wasm) {
      try { r = await wasm.search(moves, Math.max(60, Math.min(4000, Math.round(wasm.sps * budget / 1000)))); }
      catch (e) { r = null; }
    }
    if (!r) r = await searchJS(budget);
    S = TG.play(S, r.move, B); moves.push(r.move);
    lastMove = r.move === B.pass ? null : r.move;
    $("#thinking").style.opacity = 0;
    syncBoard();
    if (r.move === B.pass && !TG.isTerminal(S, B)) note("the engine passes — pass too to end and score");
  }

  async function onPick(node) {
    if (busy || resigned) return;
    const legal = TG.legalMoves(S, B);
    if (!legal[node]) return;
    if ($("#opponent").value === "engine" && S.toMove !== humanColor) return;
    busy = true;
    try {
      hist.push({ s: S, last: lastMove });
      S = TG.play(S, node, B); moves.push(node); lastMove = node;
      syncBoard();
      if ($("#opponent").value === "engine") await engineReply();
    } finally { busy = false; syncBoard(); }
  }

  function newGame(key) {
    curKey = key;
    const rec = BOARDS[key];
    B = TG.makeBoard(rec);
    B.coords3 = rec.coords3d;
    S = TG.newGame(B); moves = []; hist = []; lastMove = null; resigned = false; wrHist = [];
    const eb = $("#endbanner"); if (eb) eb.remove();
    buildBoardMeshes({ n: B.n, coords3: rec.coords3d, edgesI: B.edges });
    syncBoard();
    bootWasm();
  }

  // ---------- input: orbit + picking ----------
  function wireInput() {
    const el = renderer.domElement;
    let down = null, moved = 0, pinch0 = 0;
    el.addEventListener("pointerdown", e => { down = { x: e.clientX, y: e.clientY }; moved = 0; });
    window.addEventListener("pointermove", e => {
      if (!down) { hover(e); return; }
      const dx = e.clientX - down.x, dy = e.clientY - down.y;
      moved += Math.abs(dx) + Math.abs(dy);
      yaw -= dx * 0.0075; pitch += dy * 0.006;
      down = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener("pointerup", e => {
      if (down && moved < 6) pick(e);                      // a click, not a drag
      down = null;
    });
    el.addEventListener("wheel", e => {
      e.preventDefault();
      dist = Math.max(2.2, Math.min(9, dist * (1 + e.deltaY * 0.0011)));
    }, { passive: false });
    el.addEventListener("touchmove", e => {               // pinch zoom
      if (e.touches.length === 2) {
        const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX,
                             e.touches[0].clientY - e.touches[1].clientY);
        if (pinch0) dist = Math.max(2.2, Math.min(9, dist * pinch0 / d));
        pinch0 = d;
        e.preventDefault();
      } else pinch0 = 0;
    }, { passive: false });
    const cast = e => {
      const r = el.getBoundingClientRect();
      raycaster.setFromCamera(new T.Vector2(((e.clientX - r.left) / r.width) * 2 - 1,
                                            -((e.clientY - r.top) / r.height) * 2 + 1), camera);
      const hits = raycaster.intersectObjects(hitMeshes, false);
      return hits.find(h => h.object.userData.enabled);
    };
    const hover = e => {
      const h = cast(e);
      const idx = h ? h.object.userData.node : -1;
      if (idx === hoverIdx) return;
      if (hoverIdx >= 0 && markerMeshes[hoverIdx]) markerMeshes[hoverIdx].material.color.setHex(
        hitMeshes[hoverIdx].userData.enabled ? COL.markerLegal : COL.marker);
      if (idx >= 0) markerMeshes[idx].material.color.setHex(COL.accent);
      hoverIdx = idx;
      el.style.cursor = idx >= 0 ? "pointer" : "grab";
    };
    const pick = e => { const h = cast(e); if (h) onPick(h.object.userData.node); };
  }

  // ---------- boot ----------
  function init() {
    net = TG.loadNet(WEIGHTS_B64, CFG);
    // decode coords3 once per board record
    for (const k of Object.keys(BOARDS)) {
      const rec = BOARDS[k];
      const bin = atob(rec.coords3);
      const f = new Float32Array(bin.length / 4);
      const dv = new DataView(new ArrayBuffer(4));
      for (let i = 0; i < f.length; i++) {
        for (let b = 0; b < 4; b++) dv.setUint8(b, bin.charCodeAt(4 * i + b));
        f[i] = dv.getFloat32(0, true);
      }
      rec.coords3d = f;
    }
    $("#board3").innerHTML = Object.keys(BOARDS).map(k => `<option value="${k}">${BOARDS[k].label}</option>`).join("");
    $("#board3").onchange = () => { if (!busy) newGame($("#board3").value); };
    $("#opponent").onchange = () => { if (!busy) newGame(curKey); };
    $("#reset3").onclick = () => { if (!busy) newGame(curKey); };
    $("#undo3").onclick = () => {
      if (busy || !hist.length) return;
      resigned = false; const eb = $("#endbanner"); if (eb) eb.remove();
      const h = hist.pop(); S = h.s; lastMove = h.last; moves.length = S.moveNum; syncBoard();
    };
    $("#pass3").onclick = async () => {
      if (busy || resigned) return; busy = true;
      try {
        hist.push({ s: S, last: lastMove });
        S = TG.play(S, B.pass, B); moves.push(B.pass); lastMove = null; syncBoard();
        if ($("#opponent").value === "engine") await engineReply();
      } finally { busy = false; syncBoard(); }
    };
    buildScene();
    wireInput();
    newGame("diamond_c2");
    // minimal demo/debug hooks (drive the camera + place stones programmatically; used by
    // the demo recorder so motion is continuous instead of freezing during synthetic input)
    window.EG3D = {
      orbit(dy, dp) { yaw += dy; pitch += dp; },
      zoom(f) { dist = Math.max(2.2, Math.min(9, dist * f)); },
      place(node) { return onPick(node); },
      legal() {
        const l = TG.legalMoves(S, B), out = [];
        for (let i = 0; i < B.n; i++) if (l[i] && S.colors[i] === 0) out.push(i);
        return out;
      },
      pos(i) { return [P3[3 * i], P3[3 * i + 1], P3[3 * i + 2]]; },
      state() { return { move: S ? S.moveNum : 0, busy }; },
    };
    const sp = $("#splash"); if (sp) { sp.classList.add("off"); setTimeout(() => sp.remove(), 450); }
  }
  if (document.readyState !== "loading") init(); else document.addEventListener("DOMContentLoaded", init);
})();
