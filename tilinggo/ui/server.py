"""A tiny local web UI for playing graph Go on any tiling (ARCHITECTURE.md §8).

Deliberately dependency-free: it uses only the standard library's ``http.server`` (no Flask
yet) so you can ``uv run python scripts/play.py`` and start clicking. One game is held in
memory per process — this is a local debugging/play tool, not a multiplayer server.

There is no neural-net opponent yet (Milestones 4–5); you can play human-vs-human, or against
a random-legal-move opponent, which is enough to exercise the rules engine end to end.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from ..rules import BLACK, WHITE, Board, GoState, IllegalMove
from ..tilings import penrose, periodic, uniform
from . import render

# Tilings offered in the picker: (key, label, default radius).
TILINGS = [
    ("rect9", "Square 9×9 (classic)", None),
    ("rect13", "Square 13×13 (large)", None),
    ("rect19", "Square 19×19 (full board!)", None),
    ("square", "Square (disc)", 6.0),
    ("hexagonal", "Hexagonal", 6.0),
    ("hex_big", "Hexagonal (large)", 9.0),
    ("triangular", "Triangular", 5.0),
    ("tri_big", "Triangular (large)", 8.0),
    ("trihexagonal", "Trihexagonal 3.6.3.6", 4.5),
    ("trunc_square", "Truncated square 4.8.8", 7.0),
    ("trunc_hex", "Truncated hex 3.12.12", 9.0),
    ("rhombitrihex", "Rhombitrihexagonal 3.4.6.4", 6.5),
    ("snub_square", "Snub square 3.3.4.3.4", 5.0),
    ("snub_hex", "Snub hexagonal 3.3.3.3.6", 6.0),
    ("penrose_small", "Penrose (small, 5-fold)", 3.5),
    ("penrose", "Penrose (aperiodic, 5-fold)", 5.0),
    ("penrose_med", "Penrose (medium, 5-fold)", 6.5),
    ("penrose_big", "Penrose (large, aperiodic)", 8.0),
]
_LABELS = {k: lbl for k, lbl, _ in TILINGS}
_RADII = {k: r for k, _, r in TILINGS}
# periodic-family keys (incl. "large" variants) → generator family
_PERIODIC_FAM = {"square": "square", "triangular": "tri", "tri_big": "tri",
                 "hexagonal": "hex", "hex_big": "hex"}


def _make_board(key: str, komi: float = 5.5) -> Board:
    if key.startswith("rect"):
        n = int(key[len("rect"):])
        graph = periodic.rectangular(n, n)
    elif key.startswith("penrose"):
        graph = penrose.generate(radius=_RADII[key], symmetric=True)
    elif key in _PERIODIC_FAM:
        graph = periodic.generate(_PERIODIC_FAM[key], radius=_RADII[key])
    else:
        graph = uniform.generate(key, radius=_RADII[key])
    return Board(graph, komi=komi)


_NET_EVAL = None
_NET_TRIED = False
# Checkpoints the play UI will use for the "neural engine" opponent, in preference order
# (the strong champion if it has been trained, else the lightly-trained SIT nets).
_CHECKPOINTS = ["results/universal/champion.pt"]


def _trained_evaluator():
    """Load (once, cached) a trained-net evaluator, or None if no checkpoint is on disk."""
    global _NET_EVAL, _NET_TRIED
    if _NET_TRIED:
        return _NET_EVAL
    _NET_TRIED = True
    from pathlib import Path
    for path in _CHECKPOINTS:
        if Path(path).exists():
            from ..search.evaluators import NetEvaluator
            from ..sit.checkpoint import load_checkpoint
            net, meta = load_checkpoint(path)
            _NET_EVAL = NetEvaluator(net)
            print(f"[play] neural opponent: loaded {path} ({meta.get('kind','?')})", flush=True)
            return _NET_EVAL
    print("[play] no trained checkpoint found — neural opponent falls back to the heuristic bot",
          flush=True)
    return None


class Game:
    """A single in-memory game with an undo stack."""

    def __init__(self, key: str = "rect9"):
        self.lock = threading.Lock()
        self.reset(key)

    def reset(self, key: str):
        if key not in _LABELS:
            key = "rect9"
        self.key = key
        self.board = _make_board(key)
        self.history: list[GoState] = [self.board.new_game()]
        self.last_move: int | None = None

    @property
    def state(self) -> GoState:
        return self.history[-1]

    def play(self, node: int) -> str | None:
        try:
            self.history.append(self.state.play(node))
        except IllegalMove as e:
            return str(e)
        self.last_move = None if node == self.board.pass_move else node
        return None

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.last_move = None

    def random_move(self):
        legal = np.flatnonzero(self.state.legal_moves())
        # prefer not to pass unless forced
        non_pass = legal[legal != self.board.pass_move]
        choices = non_pass if non_pass.size else legal
        mv = int(np.random.default_rng().choice(choices))
        self.play(mv)

    def engine_move(self, simulations: int = 120):
        """Play one move via PUCT MCTS with the score-heuristic evaluator (a solid, net-free bot)."""
        from ..search.evaluators import ScoreHeuristicEvaluator
        from ..search.mcts import MCTS, MCTSConfig

        mcts = MCTS(ScoreHeuristicEvaluator(),
                    MCTSConfig(num_simulations=simulations, eval_batch=16))
        root = mcts.run(self.state)
        self.play(mcts.select_move(root, temperature=0.0))

    def neural_move(self, simulations: int = 200):
        """Play one move via PUCT MCTS guided by the trained neural net (the learned engine).

        This is the actual self-play-trained network — and because it's geometry-blind, it plays
        every tiling, including ones it never trained on (e.g. Penrose). It is only lightly
        trained, so it is a *real* but not a *strong* opponent.
        """
        from ..search.mcts import MCTS, MCTSConfig
        net = _trained_evaluator()
        if net is None:                       # no checkpoint → fall back to the heuristic bot
            return self.engine_move(simulations)
        # scale simulations down on larger boards so each move stays responsive
        # (keeps per-move net work ~constant; 9×9≈full sims, 19×19≈a quarter)
        n = self.state.board.n
        sims = max(40, min(simulations, int(simulations * 81 / max(n, 81))))
        mcts = MCTS(net, MCTSConfig(num_simulations=sims, eval_batch=16))
        root = mcts.run(self.state)
        self.play(mcts.select_move(root, temperature=0.0))

    def analyze(self, simulations: int = 320) -> dict:
        """KataGo-style read of the current position with the trained net (no move is played).

        Returns the board SVG with an analysis overlay (top candidate moves coloured by win-rate,
        ownership/territory shading), plus the position's win-rate and a score estimate.
        """
        import torch

        from ..nn import encoding
        from ..search.mcts import MCTS, MCTSConfig

        ev = _trained_evaluator()
        if ev is None:
            return {"error": "no trained network available for analysis"}
        s = self.state
        n = self.board.n
        if s.is_terminal:
            return {"error": "game is over — nothing to analyse"}

        # one net forward for value / score / ownership (the dense heads)
        batch = encoding.encode_states([s])
        with torch.no_grad():
            out = ev.net.forward(batch)
            ownp = torch.softmax(out["ownership"][0], dim=-1).cpu().numpy()  # [N,3] B/W/neutral
        value = float(out["value"][0])                       # side-to-move, in (-1,1)
        score_est = float(out["score"][0]) * max(n, 1)       # side-to-move margin estimate
        own_signed = (ownp[:, 0] - ownp[:, 1])               # +black .. -white (absolute)

        # PUCT search for visit counts / per-move win-rates (sims scaled down on big boards)
        sims = max(64, min(simulations, int(simulations * 81 / max(n, 81))))
        mcts = MCTS(ev, MCTSConfig(num_simulations=sims, eval_batch=16, dirichlet_eps=0.0),
                    rng=np.random.default_rng(0))
        root = mcts.run(s)
        total = float(root.N.sum()) or 1.0
        moves = []
        for ai in np.argsort(root.N)[::-1]:
            visits = float(root.N[ai])
            if visits <= 0:
                break
            mv = int(root.legal[ai])
            q = root.W[ai] / visits                          # side-to-move value of that move
            moves.append({"node": mv, "visits": int(visits), "frac": visits / total,
                          "winrate": 0.5 * (q + 1.0), "is_pass": mv == n})

        best = next((m["node"] for m in moves if not m["is_pass"]), None)
        analysis = {"moves": [m for m in moves[:8] if not m["is_pass"]],
                    "ownership": own_signed.tolist(), "best": best}
        svg = render.interactive_svg(self.board.graph, s.colors, last_move=self.last_move,
                                     legal=s.legal_moves()[:n], analysis=analysis)

        black_wr = (0.5 * (value + 1.0)) if s.to_move == BLACK else (0.5 * (1.0 - value))
        lead = score_est if s.to_move == BLACK else -score_est
        return {
            "svg": svg,
            "to_move": "Black" if s.to_move == BLACK else "White",
            "black_winrate": black_wr,
            "score_lead": round(lead, 1),
            "sims": sims,
            "top": [{"node": m["node"], "winrate": m["winrate"], "visits": m["visits"],
                     "is_pass": m["is_pass"]} for m in moves[:6]],
        }

    def snapshot(self) -> dict:
        s = self.state
        black, white = s.score()
        return {
            "key": self.key,
            "label": _LABELS[self.key],
            "svg": render.interactive_svg(
                self.board.graph, s.colors, last_move=self.last_move,
                legal=s.legal_moves()[: self.board.n]),
            "to_move": "Black" if s.to_move == BLACK else "White",
            "black": black,
            "white": white,
            "komi": self.board.komi,
            "score_diff": round(s.score_difference(), 1),
            "move_num": s.move_num,
            "passes": s.pass_count,
            "terminal": s.is_terminal,
            "winner": ("Black" if s.winner() == BLACK else "White") if s.is_terminal else None,
            "n": self.board.n,
            "can_undo": len(self.history) > 1,
        }


GAME = Game()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            with GAME.lock:
                self._json(GAME.snapshot())
        elif self.path == "/api/tilings":
            self._json([{"key": k, "label": lbl} for k, lbl, _ in TILINGS])
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or "{}")
        with GAME.lock:
            if self.path == "/api/move":
                err = GAME.play(int(body["node"]))
                self._json({"error": err, **GAME.snapshot()})
            elif self.path == "/api/pass":
                GAME.play(GAME.board.pass_move)
                self._json(GAME.snapshot())
            elif self.path == "/api/undo":
                GAME.undo()
                self._json(GAME.snapshot())
            elif self.path == "/api/random":
                GAME.random_move()
                self._json(GAME.snapshot())
            elif self.path == "/api/engine":
                GAME.engine_move()
                self._json(GAME.snapshot())
            elif self.path == "/api/neural":
                GAME.neural_move()
                self._json(GAME.snapshot())
            elif self.path == "/api/analyze":
                self._json(GAME.analyze())
            elif self.path == "/api/reset":
                GAME.reset(body.get("key", "rect9"))
                self._json(GAME.snapshot())
            else:
                self._send(404, "not found", "text/plain")


def serve(host: str = "127.0.0.1", port: int = 8770) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tiling-Go — play</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
  :root {
    --bg:#08090d; --panel:rgba(16,18,26,0.62); --line:rgba(255,255,255,0.08);
    --text:#e8ebf2; --muted:#828c9c; --accent:#00ffc2; --accent-dim:rgba(0,255,194,0.14);
    color-scheme:dark;
  }
  * { box-sizing:border-box; }
  html,body { height:100%; }
  body { margin:0; overflow:hidden; color:var(--text);
    font-family:"Space Grotesk","Inter",-apple-system,system-ui,Segoe UI,Roboto,sans-serif;
    background:radial-gradient(1200px 820px at 50% 42%, #14171f 0%, var(--bg) 62%), var(--bg);
    -webkit-font-smoothing:antialiased; }

  .stage { position:fixed; inset:0; display:flex; align-items:center; justify-content:center; }
  .glow { position:absolute; width:62vh; height:62vh; border-radius:50%; pointer-events:none;
    background:radial-gradient(closest-side, rgba(0,255,194,0.10), rgba(0,255,194,0) 70%);
    filter:blur(22px); }
  #boardwrap { position:relative; }
  #board svg { display:block; height:86vh; max-width:62vw; width:auto; border-radius:18px;
    box-shadow:0 30px 80px rgba(0,0,0,0.65), 0 0 0 1px rgba(255,255,255,0.04);
    animation:fade .26s ease; }
  @keyframes fade { from{ opacity:.5 } to{ opacity:1 } }
  .hot { cursor:pointer; transition:fill .08s; }
  .hot:hover { fill:rgba(0,255,194,0.22); }
  .bestmv { transform-box:fill-box; transform-origin:center; animation:pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{ opacity:.95 } 50%{ opacity:.5 } }

  .panel { position:fixed; z-index:5; background:var(--panel); border:1px solid var(--line);
    border-radius:18px; padding:18px; width:268px; box-shadow:0 16px 50px rgba(0,0,0,0.5);
    backdrop-filter:blur(22px) saturate(1.4); -webkit-backdrop-filter:blur(22px) saturate(1.4); }
  .ctrl { top:24px; left:24px; }
  .info { bottom:24px; right:24px; width:288px; }

  .brand { display:flex; align-items:center; gap:11px; margin-bottom:16px; }
  .logo { width:30px; height:30px; border-radius:9px; flex:none;
    background:linear-gradient(135deg, var(--accent), #2b8cff); box-shadow:0 0 18px var(--accent-dim); }
  .brand h1 { font-size:16px; margin:0; letter-spacing:1.6px; font-weight:600; }
  .brand p { margin:2px 0 0; font-size:9.5px; color:var(--muted); letter-spacing:.6px;
    text-transform:uppercase; }

  label { display:block; font-size:10px; color:var(--muted); margin:14px 0 6px;
    letter-spacing:1.4px; text-transform:uppercase; }
  select { width:100%; padding:10px 12px; border-radius:10px; border:1px solid var(--line);
    color:var(--text); font-size:13px; cursor:pointer; appearance:none; -webkit-appearance:none;
    background:#0e1016 url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%23828c9c' fill='none' stroke-width='1.4'/%3E%3C/svg%3E") no-repeat right 12px center; }
  select:focus { outline:none; border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-dim); }

  button { width:100%; padding:10px 12px; border-radius:10px; border:1px solid var(--line);
    background:#13151d; color:var(--text); font-size:13px; font-weight:500; cursor:pointer;
    font-family:inherit; transition:all .12s; }
  button:hover { border-color:rgba(0,255,194,0.5); background:#171a23; }
  button:active { transform:translateY(1px); }
  button:disabled { opacity:.4; cursor:not-allowed; }
  .cta { margin-top:14px; border:none; color:#04120e; font-weight:700;
    background:linear-gradient(135deg, var(--accent), #14d6ff); box-shadow:0 6px 20px var(--accent-dim); }
  .cta:hover { filter:brightness(1.08); }
  .row { display:flex; gap:8px; margin-top:8px; }
  .row button { flex:1; }
  .chk { display:flex; gap:8px; align-items:center; font-size:12px; color:#aeb6c2; margin-top:12px;
    cursor:pointer; text-transform:none; letter-spacing:0; }
  .chk input { width:auto; accent-color:var(--accent); }

  .turnrow { display:flex; align-items:center; gap:9px; font-size:13.5px; margin-bottom:15px; }
  .dot { width:13px; height:13px; border-radius:50%; border:1px solid #555;
    box-shadow:0 0 10px rgba(0,0,0,.5); }
  .spacer { flex:1; }
  .muted { color:var(--muted); font-size:12px; }
  .bignum { display:flex; gap:16px; }
  .bignum > div { flex:1; }
  .lbl { font-size:9px; color:var(--muted); letter-spacing:1.5px; text-transform:uppercase; }
  .num { font-size:30px; font-weight:600; margin-top:3px; line-height:1.05;
    font-family:"JetBrains Mono",ui-monospace,Menlo,monospace; }
  .num.acc { color:var(--accent); }
  .topmv { font-size:12px; color:#9aa3ad; margin-top:15px; line-height:1.7; min-height:16px;
    font-family:"JetBrains Mono",ui-monospace,Menlo,monospace; }
  .topmv .k { color:var(--accent); font-weight:700; }
  .msg { color:#e0796b; font-size:12px; min-height:15px; margin-top:10px; }
  .win { color:var(--accent); font-weight:600; }

  .winrail { position:fixed; left:26px; top:50%; transform:translateY(-50%); z-index:4; width:12px;
    height:60vh; border-radius:7px; overflow:hidden; border:1px solid var(--line);
    background:linear-gradient(#d7dbe4,#b3b9c6); box-shadow:inset 0 0 12px rgba(0,0,0,.4); }
  .winrail-fill { position:absolute; left:0; right:0; bottom:0; height:50%;
    background:linear-gradient(#222633,#05060a); transition:height .5s cubic-bezier(.2,.7,.2,1); }
  .winrail-fill::after { content:""; position:absolute; top:-1px; left:0; right:0; height:2px;
    background:var(--accent); box-shadow:0 0 8px var(--accent); }

  #scan { position:absolute; inset:0; border-radius:18px; overflow:hidden; pointer-events:none;
    opacity:0; transition:opacity .2s; }
  #scan.on { opacity:1; }
  #scan::before { content:""; position:absolute; left:0; right:0; height:36%; top:-36%;
    background:linear-gradient(transparent, rgba(0,255,194,0.13), transparent);
    animation:scan 1.1s linear infinite; }
  @keyframes scan { from{ top:-36% } to{ top:100% } }
</style></head>
<body>
<div class="stage"><div class="glow"></div>
  <div id="boardwrap"><div id="board"></div><div id="scan"></div></div>
</div>

<div class="winrail" title="Black win probability"><div class="winrail-fill" id="winfill"></div></div>

<div class="panel ctrl">
  <div class="brand"><div class="logo"></div>
    <div><h1>TILING·GO</h1><p>geometry-blind engine</p></div></div>
  <label>Substrate</label>
  <select id="tiling"></select>
  <label>Opponent</label>
  <select id="opponent">
    <option value="neural">Neural engine (champion)</option>
    <option value="off">Human (hot-seat)</option>
    <option value="engine">Heuristic engine</option>
    <option value="random">Random moves</option>
  </select>
  <button id="analyze" class="cta">⌖ Analyze position</button>
  <div class="row"><button id="pass">Pass</button><button id="undo">Undo</button></div>
  <div class="row"><button id="rand">Random</button><button id="reset">New game</button></div>
  <label class="chk"><input type="checkbox" id="auto"> Auto-analyze each move</label>
  <label class="chk"><input type="checkbox" id="snd" checked> Stone sound</label>
</div>

<div class="panel info">
  <div class="turnrow"><span class="dot" id="turndot"></span><span id="turn">—</span>
    <span class="spacer"></span><span class="muted" id="move">move 0</span></div>
  <div class="bignum">
    <div><div class="lbl">Score</div><div class="num" id="score">—</div></div>
    <div><div class="lbl">Win&nbsp;B</div><div class="num acc" id="winpct">—</div></div>
  </div>
  <div class="topmv" id="topmv"></div>
  <div class="msg" id="msg"></div>
</div>

<script>
const $ = s => document.querySelector(s);
let busy = false, prevMove = -1, actx;

function clack() {                         // synthesized stone "clack" — no audio assets needed
  if (!$("#snd").checked) return;
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    const t = actx.currentTime, o = actx.createOscillator(), g = actx.createGain();
    o.type = "triangle";
    o.frequency.setValueAtTime(260, t); o.frequency.exponentialRampToValueAtTime(92, t + 0.05);
    g.gain.setValueAtTime(0.16, t); g.gain.exponentialRampToValueAtTime(0.0007, t + 0.13);
    o.connect(g).connect(actx.destination); o.start(t); o.stop(t + 0.14);
  } catch (e) {}
}
const setWin = bw => { $("#winfill").style.height = Math.round(bw * 100) + "%"; };
const thinking = on => $("#scan").classList.toggle("on", on);

function render(st) {
  $("#board").innerHTML = st.svg;
  $("#turndot").style.background = st.to_move === "Black" ? "#111417" : "#f4f4f2";
  $("#turn").textContent = st.to_move + " to move";
  $("#score").textContent = (st.score_diff > 0 ? "B+" : st.score_diff < 0 ? "W+" : "")
    + (st.score_diff ? Math.abs(st.score_diff) : "0");
  $("#move").textContent = "move " + st.move_num + (st.passes ? " · " + st.passes + "p" : "");
  $("#undo").disabled = !st.can_undo;
  if (st.terminal) $("#msg").innerHTML = `<span class="win">${st.winner} wins by ${Math.abs(st.score_diff)}</span>`;
  else if (st.error) $("#msg").textContent = "Illegal: " + st.error;
  else $("#msg").textContent = "";
  if (st.move_num > prevMove && !st.error) clack();
  prevMove = st.move_num;
  $("#winpct").textContent = "—"; $("#topmv").innerHTML = "";
  bind();
}
function showAnalysis(a) {
  if (a.error) { $("#msg").textContent = a.error; return; }
  $("#board").innerHTML = a.svg; bind();
  setWin(a.black_winrate);
  $("#winpct").textContent = Math.round(a.black_winrate * 100) + "%";
  $("#score").textContent = (a.score_lead >= 0 ? "B+" : "W+") + Math.abs(a.score_lead).toFixed(1);
  $("#topmv").innerHTML = a.top.filter(m => !m.is_pass).slice(0, 5).map((m, i) =>
    `<span class="k">${i + 1}</span> ${Math.round(m.winrate * 100)}% &middot; ${m.visits}v`).join("<br>");
  $("#msg").textContent = "";
}
async function api(path, body) {
  const r = await fetch(path, {method: body ? "POST" : "GET",
    headers: {"Content-Type": "application/json"}, body: body ? JSON.stringify(body) : undefined});
  return r.json();
}
async function doAnalyze() {
  thinking(true);
  try { showAnalysis(await api("/api/analyze", {})); } finally { thinking(false); }
}
const OPP_ENDPOINT = {neural: "/api/neural", engine: "/api/engine", random: "/api/random"};
async function afterHuman(st) {
  render(st);
  const opp = $("#opponent").value;
  if (!st.error && !st.terminal && OPP_ENDPOINT[opp]) {
    if (opp !== "random") thinking(true);
    await new Promise(r => setTimeout(r, 50));
    render(await api(OPP_ENDPOINT[opp], {}));
    thinking(false);
  }
  if (!st.error && $("#auto").checked) await doAnalyze();
}
async function human(path, body) {
  if (busy) return; busy = true;
  try { await afterHuman(await api(path, body)); } finally { busy = false; }
}
function bind() {
  document.querySelectorAll("#board .hot").forEach(el =>
    el.addEventListener("click", () => human("/api/move", {node: +el.dataset.node})));
}
$("#analyze").onclick = async () => { if (busy) return; busy = true; try { await doAnalyze(); } finally { busy = false; } };
$("#pass").onclick = () => human("/api/pass", {});
$("#undo").onclick = async () => { render(await api("/api/undo", {})); };
$("#rand").onclick = () => human("/api/random", {});
$("#reset").onclick = async () => { setWin(0.5); render(await api("/api/reset", {key: $("#tiling").value})); };
$("#tiling").onchange = async () => { setWin(0.5); render(await api("/api/reset", {key: $("#tiling").value})); };

(async () => {
  const tilings = await api("/api/tilings");
  $("#tiling").innerHTML = tilings.map(t => `<option value="${t.key}">${t.label}</option>`).join("");
  const st = await api("/api/state");
  $("#tiling").value = st.key; setWin(0.5); render(st);
})();
</script>
</body></html>
"""
