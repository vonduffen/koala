// C++ Go rules engine — a faithful port of tilinggo/rules/gostate.py.
// EMPTY=0, BLACK=1, WHITE=2; opponent = 3-c. Captures-before-suicide, suicide illegal,
// POSITIONAL superko (Tromp-Taylor: a play may not recreate any prior board colouring),
// area scoring. Geometry-blind: the board is just adjacency lists.
#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <string>

namespace tg {

constexpr int EMPTY = 0, BLACK = 1, WHITE = 2;
inline int opponent(int c) { return 3 - c; }

inline uint64_t splitmix64(uint64_t& s) {
  s += 0x9E3779B97F4A7C15ULL;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

struct Board {
  int n = 0;
  std::vector<std::vector<int>> adj;
  double komi = 0.0;
  int max_moves = 0;
  std::vector<uint64_t> zob;  // [n*2]: zob[node*2 + (color-1)]

  Board() = default;
  Board(int n_, const std::vector<std::pair<int,int>>& edges, double komi_, uint64_t seed = 0)
      : n(n_), adj(n_), komi(komi_), max_moves(3*n_), zob((size_t)n_*2) {
    for (auto& e : edges) { adj[e.first].push_back(e.second); adj[e.second].push_back(e.first); }
    uint64_t s = seed;
    for (auto& z : zob) z = splitmix64(s);
  }
  uint64_t zobrist(int node, int color) const { return zob[(size_t)node*2 + (color-1)]; }
  int pass_move() const { return n; }
};

struct GoState {
  const Board* B = nullptr;
  std::vector<int8_t> colors;
  int to_move = BLACK, pass_count = 0, move_num = 0;
  uint64_t stone_hash = 0;
  std::vector<uint64_t> history;  // sorted; arrangements seen on the path (incl. current)

  static GoState initial(const Board& b) {
    GoState s; s.B = &b; s.colors.assign(b.n, EMPTY); s.history = {0ULL}; return s;
  }
  bool is_terminal() const { return pass_count >= 2 || move_num >= B->max_moves; }
  bool seen(uint64_t h) const { return std::binary_search(history.begin(), history.end(), h); }

  // flood-fill the chain containing `start`; collect liberties. Returns chain via `chain`.
  void group_and_liberties(const std::vector<int8_t>& col, int start,
                           std::vector<int>& chain, int& nlib) const {
    int color = col[start];
    chain.clear();
    static thread_local std::vector<char> inchain;  // reused scratch
    inchain.assign(B->n, 0);
    static thread_local std::vector<char> islib;
    islib.assign(B->n, 0);
    nlib = 0;
    std::vector<int> stack{start};
    inchain[start] = 1; chain.push_back(start);
    while (!stack.empty()) {
      int u = stack.back(); stack.pop_back();
      for (int w : B->adj[u]) {
        int cw = col[w];
        if (cw == EMPTY) { if (!islib[w]) { islib[w] = 1; ++nlib; } }
        else if (cw == color && !inchain[w]) { inchain[w] = 1; chain.push_back(w); stack.push_back(w); }
      }
    }
  }

  // Try playing `player` at `node`. On success fill out_colors/out_hash and return true.
  // false ⇒ illegal by board logic (occupied or suicide). Superko is checked by the caller.
  bool simulate(int node, int player, std::vector<int8_t>& out_colors, uint64_t& out_hash) const {
    if (colors[node] != EMPTY) return false;
    out_colors = colors;
    out_colors[node] = (int8_t)player;
    int opp = opponent(player);
    uint64_t h = stone_hash ^ B->zobrist(node, player);
    std::vector<int> chain; int nlib;
    for (int w : B->adj[node]) {
      if (out_colors[w] == opp) {
        group_and_liberties(out_colors, w, chain, nlib);
        if (nlib == 0) {
          for (int c : chain) { out_colors[c] = EMPTY; h ^= B->zobrist(c, opp); }
        }
      }
    }
    group_and_liberties(out_colors, node, chain, nlib);
    if (nlib == 0) return false;  // suicide
    out_hash = h;
    return true;
  }

  // Boolean legal mask of length n+1 (index n = pass, always legal).
  void legal_moves(std::vector<char>& legal) const {
    int n = B->n;
    legal.assign(n + 1, 0);
    legal[n] = 1;
    if (is_terminal()) return;
    std::vector<int8_t> tmp; uint64_t h;
    for (int v = 0; v < n; ++v) {
      if (colors[v] != EMPTY) continue;
      if (!simulate(v, to_move, tmp, h)) continue;
      if (!seen(h)) legal[v] = 1;
    }
  }

  // Apply a move (node index, or B->pass_move() to pass). Assumes legality.
  GoState play(int move) const {
    GoState s = *this;
    int n = B->n;
    if (move == n) {  // pass
      s.to_move = opponent(to_move);
      s.pass_count = pass_count + 1;
      s.move_num = move_num + 1;
      return s;  // board + hash + history unchanged
    }
    std::vector<int8_t> nc; uint64_t h;
    bool ok = simulate(move, to_move, nc, h);
    (void)ok;  // caller guarantees legality
    s.colors = std::move(nc);
    s.stone_hash = h;
    s.to_move = opponent(to_move);
    s.pass_count = 0;
    s.move_num = move_num + 1;
    // insert h into sorted history (copy-on-play keeps each line independent for MCTS)
    auto it = std::lower_bound(s.history.begin(), s.history.end(), h);
    if (it == s.history.end() || *it != h) s.history.insert(it, h);
    return s;
  }

  // Area score (black_area, white_area): stones + single-colour territory.
  void score(int& black, int& white) const {
    int n = B->n;
    black = white = 0;
    for (int v = 0; v < n; ++v) { if (colors[v]==BLACK) ++black; else if (colors[v]==WHITE) ++white; }
    std::vector<char> vis(n, 0);
    for (int v = 0; v < n; ++v) {
      if (colors[v] != EMPTY || vis[v]) continue;
      std::vector<int> region{v}; vis[v] = 1; int border = 0; size_t qi = 0;
      while (qi < region.size()) {
        int u = region[qi++];
        for (int w : B->adj[u]) {
          if (colors[w] == EMPTY) { if (!vis[w]) { vis[w] = 1; region.push_back(w); } }
          else border |= (colors[w] == BLACK ? 1 : 2);
        }
      }
      if (border == 1) black += (int)region.size();
      else if (border == 2) white += (int)region.size();
    }
  }

  // Per-node ownership: 0=black, 1=white, 2=neutral.
  void ownership(std::vector<int8_t>& own) const {
    int n = B->n;
    own.assign(n, 2);
    for (int v = 0; v < n; ++v) { if (colors[v]==BLACK) own[v]=0; else if (colors[v]==WHITE) own[v]=1; }
    std::vector<char> vis(n, 0);
    for (int v = 0; v < n; ++v) {
      if (colors[v] != EMPTY || vis[v]) continue;
      std::vector<int> region{v}; vis[v] = 1; int border = 0; size_t qi = 0;
      while (qi < region.size()) {
        int u = region[qi++];
        for (int w : B->adj[u]) {
          if (colors[w] == EMPTY) { if (!vis[w]) { vis[w] = 1; region.push_back(w); } }
          else border |= (colors[w] == BLACK ? 1 : 2);
        }
      }
      if (border == 1) for (int c : region) own[c] = 0;
      else if (border == 2) for (int c : region) own[c] = 1;
    }
  }
};

}  // namespace tg
