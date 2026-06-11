// WebAssembly entry for the C++ engine (Task 5). STATELESS by design: every call passes the
// full move list and the engine replays from the empty board — a few microseconds — so the
// JS game state stays the single source of truth and engine/UI desync is impossible (the
// failure mode that sank a previous port attempt).
//
// Init takes raw arrays already present in the webapp's data.js (edges, static features) plus
// the TGN1 weight stream re-serialized in JS from the already-loaded net — no new artifacts.
//
// Build: scripts/build_wasm.sh  (emcc -O3 -msimd128, SINGLE_FILE → webapp/tgwasm.js)
#include "board.hpp"
#include "tgnet.hpp"
#include "encoder.hpp"
#include "mcts.hpp"
#include <memory>
#include <random>

using namespace tg;

static std::unique_ptr<Board> g_board;
static std::unique_ptr<StaticFeat> g_sf;
static std::unique_ptr<TGNet> g_net;
static std::mt19937_64 g_rng(0x7177AB1Eu);

extern "C" {

// returns 0 on success
int tg_init(int n, const int32_t* edges, int n_edges, float komi,
            const float* static_block, int static_dim,
            const uint8_t* weights, int weights_len) {
  try {
    std::vector<std::pair<int, int>> ev(n_edges);
    for (int e = 0; e < n_edges; ++e) ev[e] = {edges[2 * e], edges[2 * e + 1]};
    g_board = std::make_unique<Board>(n, ev, komi);
    g_net = std::make_unique<TGNet>(load_net_buffer((const char*)weights, weights_len));

    auto sf = std::make_unique<StaticFeat>();
    sf->n = n; sf->dim = static_dim;
    sf->block.assign(static_block, static_block + (size_t)n * static_dim);
    sf->dmax = 1;
    for (auto& a : g_board->adj) sf->dmax = std::max<int>(sf->dmax, (int)a.size());
    sf->nbr_index.assign((size_t)n * sf->dmax, 0);
    sf->nbr_mask.assign((size_t)n * sf->dmax, 0.0f);
    for (int i = 0; i < n; ++i)
      for (size_t d = 0; d < g_board->adj[i].size(); ++d) {
        sf->nbr_index[(size_t)i * sf->dmax + d] = g_board->adj[i][d];
        sf->nbr_mask[(size_t)i * sf->dmax + d] = 1.0f;
      }
    g_sf = std::move(sf);
    return 0;
  } catch (...) { return 1; }
}

// replay `moves` from the empty board; -1 if any move is illegal (desync guard)
static int replay(const int32_t* moves, int n_moves, GoState& out) {
  out = GoState::initial(*g_board);
  std::vector<char> legal;
  for (int i = 0; i < n_moves; ++i) {
    int m = moves[i];
    if (m < 0 || m > g_board->n) return -1;
    out.legal_moves(legal);
    if (!legal[m]) return -1;
    out = out.play(m);
  }
  return 0;
}

// MCTS from the position after `moves`. Writes visit distribution into out_pi (n+1 floats)
// and the root value into out_value. Returns the chosen move, or -1 on error.
int tg_search(const int32_t* moves, int n_moves, int sims, float dirichlet_eps,
              float* out_pi, float* out_value) {
  if (!g_board || !g_net || !g_sf) return -1;
  GoState s = GoState::initial(*g_board);
  if (replay(moves, n_moves, s) != 0) return -1;
  MCTSConfig cfg;
  cfg.num_simulations = sims;
  cfg.dirichlet_eps = dirichlet_eps;
  MCTS mcts(*g_net, *g_sf, cfg, g_rng);
  std::vector<float> pi;
  int mv = mcts.search(s, pi, /*temperature=*/0.0);
  for (size_t i = 0; i < pi.size(); ++i) out_pi[i] = pi[i];
  EvalResult ev = evaluate(*g_net, s, *g_sf);          // root value for the win-rate display
  *out_value = (float)ev.value;
  return mv;
}

// raw evaluation (legal-masked priors + value) — the parity-test surface vs JS/PyTorch
int tg_eval(const int32_t* moves, int n_moves, float* out_priors, float* out_value) {
  if (!g_board || !g_net || !g_sf) return -1;
  GoState s = GoState::initial(*g_board);
  if (replay(moves, n_moves, s) != 0) return -1;
  EvalResult ev = evaluate(*g_net, s, *g_sf);
  for (int i = 0; i <= g_board->n; ++i) out_priors[i] = ev.priors[i];
  *out_value = (float)ev.value;
  return 0;
}

int tg_n() { return g_board ? g_board->n : -1; }

}  // extern "C"
