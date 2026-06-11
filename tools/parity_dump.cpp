// Gate 1 for the WASM port: the portable GEMM must match Accelerate on the real champion
// weights. Loads weights + a board's graph/static bins, plays a deterministic move sequence,
// runs the full forward pass, prints policy logits + value. Compile twice — default
// (Accelerate) and with -DTG_PORTABLE_GEMM — and compare outputs with tools/parity_cmp.py.
// Same layer-for-layer discipline that validated the native engine against PyTorch.
#include "../cpp/board.hpp"
#include "../cpp/tgnet.hpp"
#include "../cpp/encoder.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

using namespace tg;

static Board loadBoard(const std::string& path) {           // graph.bin: N, komi, E, (u,v)*
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b{std::istreambuf_iterator<char>(f), {}};
  size_t p = 0;
  auto i32 = [&] { int32_t v; std::memcpy(&v, b.data() + p, 4); p += 4; return v; };
  auto f32 = [&] { float v; std::memcpy(&v, b.data() + p, 4); p += 4; return v; };
  int N = i32(); float komi = f32(); int E = i32();
  std::vector<std::pair<int, int>> edges(E);
  for (int e = 0; e < E; ++e) { int u = i32(), v = i32(); edges[e] = {u, v}; }
  return Board(N, edges, komi);
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s weights.bin board_dir\n", argv[0]); return 2; }
  TGNet net = load_net(argv[1]);
  std::string dir = argv[2];
  Board board = loadBoard(dir + "/graph.bin");
  StaticFeat sf = load_static(dir + "/static.bin", board);

  GoState s = GoState::initial(board);
  int played = 0;
  std::vector<char> legal;
  for (int t = 3; played < 24 && t < board.n * 4; t += 7) {  // deterministic pseudo-game
    int node = t % board.n;
    s.legal_moves(legal);
    if (!legal[node]) continue;
    s = s.play(node);
    ++played;
  }
  std::printf("# moves=%d n=%d\n", s.move_num, board.n);

  std::vector<float> x;
  encode(s, sf, x, legal);
  NetOut out = forward(net, x.data(), board.n, FEATURE_DIM,
                       sf.nbr_index.data(), sf.nbr_mask.data(), sf.dmax);
  for (int i = 0; i <= board.n; ++i) std::printf("p %d %.9e\n", i, out.policy_logits[i]);
  std::printf("v %.9e\n", out.value);
  return 0;
}
