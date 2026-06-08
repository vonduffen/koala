// Feature encoding + evaluator: turns a GoState into the 42-dim node features the GNN consumes,
// runs the forward pass, and returns legal-masked policy priors + value (mirrors NetEvaluator).
//
// Static feature columns (15:39 — degree one-hot, distance-to-boundary, Laplacian PE) are LOADED
// from static.bin (computed once in Python) so they match exactly. C++ computes only the per-move
// game features (cols 0:15) and global scalars (cols 39:42). recent-move features (cols 9:14) are
// always zero because NetEvaluator encodes states without a recent-move list.
#pragma once
#include "board.hpp"
#include "tgnet.hpp"
#include <vector>
#include <cmath>
#include <string>
#include <cstring>

namespace tg {

constexpr int FEATURE_DIM = 42;
constexpr int STATIC_OFF = 15, STATIC_DIM = 24, GLOBAL_OFF = 39;

struct StaticFeat {
  int n = 0, dim = 0;
  std::vector<float> block;  // [n*dim], dim==24
  // neighbour index/mask built from board adjacency (order-invariant for mean/max pooling)
  int dmax = 0;
  std::vector<int32_t> nbr_index;  // [n*dmax]
  std::vector<float> nbr_mask;     // [n*dmax]
};

inline StaticFeat load_static(const std::string& path, const Board& board) {
  Reader r(path);
  StaticFeat sf;
  sf.n = r.i32(); sf.dim = r.i32();
  r.floats(sf.block, sf.n * sf.dim);
  // build neighbour tensors from adjacency
  sf.dmax = 1;
  for (auto& a : board.adj) sf.dmax = std::max<int>(sf.dmax, (int)a.size());
  sf.nbr_index.assign((size_t)sf.n * sf.dmax, 0);
  sf.nbr_mask.assign((size_t)sf.n * sf.dmax, 0.0f);
  for (int i = 0; i < sf.n; ++i)
    for (size_t d = 0; d < board.adj[i].size(); ++d) {
      sf.nbr_index[(size_t)i*sf.dmax + d] = board.adj[i][d];
      sf.nbr_mask[(size_t)i*sf.dmax + d] = 1.0f;
    }
  return sf;
}

// Assemble the [N, 42] feature matrix for a state (recent-move features left at zero).
inline void encode(const GoState& s, const StaticFeat& sf, std::vector<float>& x,
                   std::vector<char>& legal) {
  const Board& B = *s.B;
  int n = B.n;
  x.assign((size_t)n * FEATURE_DIM, 0.0f);
  s.legal_moves(legal);  // length n+1
  int me = s.to_move, opp = opponent(me);

  // per-node liberties (chain liberty count) for occupied nodes
  static thread_local std::vector<int> libs; libs.assign(n, 0);
  static thread_local std::vector<char> seen; seen.assign(n, 0);
  static thread_local std::vector<int> chain; int nlib;
  for (int v = 0; v < n; ++v) {
    if (s.colors[v] == EMPTY || seen[v]) continue;
    s.group_and_liberties(s.colors, v, chain, nlib);
    for (int c : chain) { libs[c] = nlib; seen[c] = 1; }
  }

  for (int v = 0; v < n; ++v) {
    float* row = x.data() + (size_t)v * FEATURE_DIM;
    int c = s.colors[v];
    if (c == me) row[0] = 1.0f; else if (c == opp) row[1] = 1.0f; else row[2] = 1.0f;
    if (c != EMPTY) { int b = libs[v] - 1; if (b < 0) b = 0; if (b > 5) b = 5; row[3 + b] = 1.0f; }
    // cols 9..13 recent moves: zero (matches NetEvaluator)
    row[14] = legal[v] ? 1.0f : 0.0f;
    // static block cols 15:39
    std::memcpy(row + STATIC_OFF, sf.block.data() + (size_t)v * sf.dim, sizeof(float) * STATIC_DIM);
    // globals cols 39:42
    row[GLOBAL_OFF + 0] = (float)(B.komi / 15.0);
    row[GLOBAL_OFF + 1] = (float)((double)s.move_num / std::max(n, 1));
    row[GLOBAL_OFF + 2] = (s.to_move == BLACK) ? 1.0f : -1.0f;
  }
}

struct EvalResult {
  std::vector<float> priors;  // length n+1, legal-masked softmax (illegal = 0)
  float value = 0.0f;
};

// Full evaluator: encode -> forward -> legal-masked softmax. Matches NetEvaluator.
inline EvalResult evaluate(const TGNet& net, const GoState& s, const StaticFeat& sf) {
  int n = s.B->n;
  static thread_local std::vector<float> x; static thread_local std::vector<char> legal;
  encode(s, sf, x, legal);
  NetOut out = forward(net, x.data(), n, FEATURE_DIM, sf.nbr_index.data(), sf.nbr_mask.data(), sf.dmax);

  EvalResult res;
  res.priors.assign(n + 1, 0.0f);
  res.value = out.value;
  // softmax over legal entries only
  float mx = -1e30f;
  for (int i = 0; i <= n; ++i) if (legal[i] && out.policy_logits[i] > mx) mx = out.policy_logits[i];
  double sum = 0;
  for (int i = 0; i <= n; ++i) if (legal[i]) { double e = std::exp((double)out.policy_logits[i] - mx); res.priors[i] = (float)e; sum += e; }
  if (sum > 0) for (int i = 0; i <= n; ++i) res.priors[i] = (float)(res.priors[i] / sum);
  return res;
}

// Build a GoState from colours alone (fresh superko history) — for evaluator parity tests.
inline GoState state_from_colors(const Board& B, const int8_t* colors, int to_move, int move_num) {
  GoState s = GoState::initial(B);
  s.to_move = to_move; s.move_num = move_num;
  uint64_t h = 0;
  for (int v = 0; v < B.n; ++v) { s.colors[v] = colors[v]; if (colors[v] != EMPTY) h ^= B.zobrist(v, colors[v]); }
  s.stone_hash = h; s.history = {h};
  return s;
}

}  // namespace tg
