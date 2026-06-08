// PUCT MCTS — a C++ port of tilinggo/search/mcts.py. Sequential single-leaf search (cleaner than
// virtual-loss batching and equally valid for self-play): selection by Q + c_puct*P*sqrt(ΣN)/(1+N)
// with FPU reduction, Dirichlet root noise, negamax backup, temperature move selection.
#pragma once
#include "encoder.hpp"
#include <vector>
#include <unordered_map>
#include <deque>
#include <cmath>
#include <random>
#include <algorithm>

namespace tg {

struct MCTSConfig {
  int num_simulations = 160;
  double c_puct = 1.4;
  double dirichlet_eps = 0.25;
  double dirichlet_alpha = -1.0;  // <0 ⇒ 10 / num_legal
  double fpu_reduction = 0.25;
};

struct MNode {
  GoState state;
  bool terminal = false, expanded = false;
  double term_value = 0.0;
  std::vector<int> legal;        // legal move indices (0..n, n=pass)
  std::vector<double> P, N, W;   // priors / visit counts / value sums (per legal child)
  std::unordered_map<int, MNode*> children;  // move -> child
  explicit MNode(const GoState& s) : state(s) {
    terminal = s.is_terminal();
    if (terminal) {
      int b, w; s.score(b, w);
      int winner = (b - w - s.B->komi > 0) ? BLACK : WHITE;
      term_value = (winner == s.to_move) ? 1.0 : -1.0;
    }
  }
};

inline double sample_gamma(double a, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::normal_distribution<double> Norm(0.0, 1.0);
  if (a < 1.0) { double u = U(rng); return sample_gamma(a + 1.0, rng) * std::pow(u, 1.0 / a); }
  double d = a - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
  while (true) {
    double x = Norm(rng), v = 1.0 + c * x;
    if (v <= 0) continue;
    v = v * v * v;
    double u = U(rng);
    if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
    if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v;
  }
}

class MCTS {
 public:
  MCTS(const TGNet& net, const StaticFeat& sf, MCTSConfig cfg, std::mt19937_64& rng)
      : net_(net), sf_(sf), cfg_(cfg), rng_(rng) {}

  // Run search from `root_state`; fill visit-policy `pi` (length n+1) and return the chosen move.
  int search(const GoState& root_state, std::vector<float>& pi, double temperature) {
    std::deque<MNode> arena;  // stable pointers (deque push_back)
    arena.emplace_back(root_state);
    MNode* root = &arena.back();
    int n = root_state.B->n;
    pi.assign(n + 1, 0.0f);
    if (root->terminal) { pi[n] = 1.0f; return n; }

    EvalResult ev = evaluate(net_, root->state, sf_);
    expand(root, ev.priors);
    add_dirichlet(root);

    for (int sim = 0; sim < cfg_.num_simulations; ++sim) {
      std::vector<std::pair<MNode*, int>> path;
      MNode* node = root;
      while (node->expanded && !node->terminal) {
        int ai = select_child(node);
        path.emplace_back(node, ai);
        int move = node->legal[ai];
        auto it = node->children.find(move);
        if (it == node->children.end()) {
          arena.emplace_back(node->state.play(move));
          node->children[move] = &arena.back();
          node = &arena.back();
        } else node = it->second;
      }
      double v;
      if (node->terminal) v = node->term_value;
      else { EvalResult e = evaluate(net_, node->state, sf_); expand(node, e.priors); v = e.value; }
      // negamax backup
      for (auto it = path.rbegin(); it != path.rend(); ++it) {
        v = -v;
        it->first->N[it->second] += 1.0;
        it->first->W[it->second] += v;
      }
    }

    for (size_t i = 0; i < root->legal.size(); ++i) pi[root->legal[i]] = (float)root->N[i];
    double tot = 0; for (float p : pi) tot += p;
    if (tot > 0) for (float& p : pi) p = (float)(p / tot);
    return select_move(root, temperature);
  }

 private:
  void expand(MNode* node, const std::vector<float>& priors_full) {
    std::vector<char> legal; node->state.legal_moves(legal);
    int n1 = (int)legal.size();
    double total = 0;
    for (int i = 0; i < n1; ++i) if (legal[i]) { node->legal.push_back(i); total += priors_full[i]; }
    int k = (int)node->legal.size();
    node->P.resize(k); node->N.assign(k, 0.0); node->W.assign(k, 0.0);
    for (int j = 0; j < k; ++j)
      node->P[j] = total > 0 ? priors_full[node->legal[j]] / total : 1.0 / k;
    node->expanded = true;
  }

  int select_child(MNode* node) {
    const auto& N = node->N; const auto& W = node->W; const auto& P = node->P;
    int k = (int)N.size();
    double sumN = 0; for (double x : N) sumN += x;
    double sqrt_total = std::sqrt(sumN) + 1e-8;
    double vw = 0, vn = 0, vp = 0; bool any = false;
    for (int i = 0; i < k; ++i) if (N[i] > 0) { vw += W[i]; vn += N[i]; vp += P[i]; any = true; }
    double fpu = 0.0;
    if (any) fpu = vw / vn - cfg_.fpu_reduction * std::sqrt(vp);
    int best = 0; double best_score = -1e30;
    for (int i = 0; i < k; ++i) {
      double q = N[i] > 0 ? W[i] / std::max(N[i], 1.0) : fpu;
      double u = cfg_.c_puct * P[i] * sqrt_total / (1.0 + N[i]);
      double sc = q + u;
      if (sc > best_score) { best_score = sc; best = i; }
    }
    return best;
  }

  void add_dirichlet(MNode* root) {
    if (cfg_.dirichlet_eps <= 0) return;
    int k = (int)root->legal.size();
    double alpha = cfg_.dirichlet_alpha > 0 ? cfg_.dirichlet_alpha : 10.0 / k;
    std::vector<double> noise(k); double s = 0;
    for (int i = 0; i < k; ++i) { noise[i] = sample_gamma(alpha, rng_); s += noise[i]; }
    if (s <= 0) return;
    for (int i = 0; i < k; ++i)
      root->P[i] = (1 - cfg_.dirichlet_eps) * root->P[i] + cfg_.dirichlet_eps * (noise[i] / s);
  }

  int select_move(MNode* root, double temperature) {
    const auto& N = root->N; int k = (int)N.size();
    double sumN = 0; for (double x : N) sumN += x;
    if (temperature <= 1e-6 || sumN == 0) {
      int best = 0; for (int i = 1; i < k; ++i) if (N[i] > N[best]) best = i;
      return root->legal[best];
    }
    std::vector<double> w(k); double s = 0;
    for (int i = 0; i < k; ++i) { w[i] = std::pow(N[i], 1.0 / temperature); s += w[i]; }
    std::uniform_real_distribution<double> U(0.0, s);
    double r = U(rng_), acc = 0; int idx = k - 1;
    for (int i = 0; i < k; ++i) { acc += w[i]; if (r <= acc) { idx = i; break; } }
    return root->legal[idx];
  }

  const TGNet& net_;
  const StaticFeat& sf_;
  MCTSConfig cfg_;
  std::mt19937_64& rng_;
};

}  // namespace tg
