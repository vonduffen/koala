// Tiling-Go native app: a tiny self-contained HTTP server that serves the web UI and runs the
// REAL C++ engine (Accelerate/AMX) — blazing fast, no Python/torch. Bundle = this binary + index.html
// + data/ (per-board graph/static + weights). Copy the folder to any Apple-Silicon Mac and run.
#include "mcts.hpp"            // pulls board.hpp, encoder.hpp, tgnet.hpp
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace tg;

// ---- small helpers ----
static std::string readFile(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static std::vector<char> readBytes(const std::string& p) {
  std::ifstream f(p, std::ios::binary); return {std::istreambuf_iterator<char>(f), {}};
}
static Board loadBoard(const std::string& path) {
  auto b = readBytes(path); size_t p = 0;
  auto i32 = [&]{ int32_t v; std::memcpy(&v, b.data()+p, 4); p+=4; return v; };
  auto f32 = [&]{ float v; std::memcpy(&v, b.data()+p, 4); p+=4; return v; };
  int N = i32(); float komi = f32(); int E = i32();
  std::vector<std::pair<int,int>> edges(E);
  for (int e = 0; e < E; ++e) { int u = i32(), v = i32(); edges[e] = {u, v}; }
  return Board(N, edges, komi);
}

struct BoardCtx { Board board; StaticFeat sf; std::string label; };

// ---- global app state ----
static std::map<std::string, BoardCtx> g_boards;
static std::vector<std::string> g_order;          // catalogue order
static TGNet g_net;
static std::mt19937_64 g_rng(12345);
static std::string g_key;
static GoState g_state;
static std::vector<GoState> g_hist;               // for undo
static int g_last = -1;

static BoardCtx& cur() { return g_boards.at(g_key); }

static void reset(const std::string& key) {
  g_key = g_boards.count(key) ? key : g_order.front();
  g_state = GoState::initial(cur().board);
  g_hist.clear(); g_last = -1;
}

// ---- JSON builders ----
static std::string jstate() {
  Board& b = cur().board; int n = b.n;
  std::vector<char> legal; g_state.legal_moves(legal);
  int bl, wh; g_state.score(bl, wh);
  double sd = bl - wh - b.komi;
  bool term = g_state.is_terminal();
  // black win-rate from the net value head (one cheap forward) — drives the win-rate graph
  double val = evaluate(g_net, g_state, cur().sf).value;       // side-to-move perspective
  double bwr = (g_state.to_move == BLACK) ? 0.5 * (val + 1) : 0.5 * (1 - val);
  std::ostringstream o;
  o << "{\"key\":\"" << g_key << "\",\"n\":" << n << ",\"toMove\":" << (int)g_state.to_move
    << ",\"moveNum\":" << g_state.move_num << ",\"passes\":" << g_state.pass_count
    << ",\"terminal\":" << (term ? "true" : "false") << ",\"winrate\":" << bwr
    << ",\"scoreDiff\":" << sd << ",\"last\":" << g_last << ",\"colors\":[";
  for (int i = 0; i < n; ++i) o << (i ? "," : "") << (int)g_state.colors[i];
  o << "],\"legal\":[";
  for (int i = 0; i <= n; ++i) o << (i ? "," : "") << (legal[i] ? 1 : 0);
  o << "]}";
  return o.str();
}

static int engineMove(int sims) {                  // returns move played
  MCTSConfig cfg; cfg.num_simulations = sims; cfg.dirichlet_eps = 0.0;
  MCTS m(g_net, cur().sf, cfg, g_rng);
  std::vector<float> pi;
  return m.search(g_state, pi, 0.0);
}

static std::string analyze(int sims) {
  Board& b = cur().board; int n = b.n;
  MCTSConfig cfg; cfg.num_simulations = sims; cfg.dirichlet_eps = 0.0;
  MCTS m(g_net, cur().sf, cfg, g_rng);
  std::vector<float> pi;
  m.search(g_state, pi, 0.0);                       // pi = visit distribution over n+1
  EvalResult ev = evaluate(g_net, g_state, cur().sf);   // value (side-to-move)
  double val = ev.value, bw = (g_state.to_move == BLACK) ? 0.5 * (val + 1) : 0.5 * (1 - val);
  int bl, wh; g_state.score(bl, wh); double lead = bl - wh - b.komi;
  // top moves by visit fraction (exclude pass)
  std::vector<std::pair<double,int>> mv;
  for (int i = 0; i < n; ++i) if (pi[i] > 0) mv.push_back({pi[i], i});
  std::sort(mv.rbegin(), mv.rend());
  std::ostringstream o;
  o << "{\"black_winrate\":" << bw << ",\"score_lead\":" << lead << ",\"best\":"
    << (mv.empty() ? -1 : mv[0].second) << ",\"top\":[";
  for (size_t i = 0; i < mv.size() && i < 8; ++i)
    o << (i ? "," : "") << "{\"node\":" << mv[i].second << ",\"frac\":" << mv[i].first << "}";
  o << "]}";
  return o.str();
}

// crude body field extractors ({"node":N} / {"key":"x"})
static int jint(const std::string& body, const std::string& k) {
  auto p = body.find("\"" + k + "\""); if (p == std::string::npos) return -999999;
  p = body.find(':', p); return std::atoi(body.c_str() + p + 1);
}
static std::string jstr(const std::string& body, const std::string& k) {
  auto p = body.find("\"" + k + "\""); if (p == std::string::npos) return "";
  p = body.find(':', p); p = body.find('"', p) + 1; auto e = body.find('"', p);
  return body.substr(p, e - p);
}

static std::string route(const std::string& method, const std::string& path, const std::string& body,
                         std::string& ctype, const std::string& webdir, int sims) {
  ctype = "application/json";
  if (method == "GET" && (path == "/" || path == "/index.html")) {
    ctype = "text/html; charset=utf-8"; return readFile(webdir + "/index.html");
  }
  if (path == "/api/tilings") {
    std::ostringstream o; o << "[";
    bool first = true;
    for (auto& k : g_order) { o << (first ? "" : ",") << "{\"key\":\"" << k << "\",\"label\":\""
                               << g_boards[k].label << "\"}"; first = false; }
    o << "]"; return o.str();
  }
  if (path == "/api/state") return jstate();
  if (path == "/api/reset") { reset(jstr(body, "key")); return jstate(); }
  if (path == "/api/move") {
    int node = jint(body, "node");
    std::vector<char> legal; g_state.legal_moves(legal);
    if (node >= 0 && node <= cur().board.n && legal[node]) {
      g_hist.push_back(g_state); g_state = g_state.play(node); g_last = (node == cur().board.n ? -1 : node);
    }
    return jstate();
  }
  if (path == "/api/pass") { g_hist.push_back(g_state); g_state = g_state.play(cur().board.n); g_last = -1; return jstate(); }
  if (path == "/api/undo") { if (!g_hist.empty()) { g_state = g_hist.back(); g_hist.pop_back(); g_last = -1; } return jstate(); }
  if (path == "/api/engine") {
    if (!g_state.is_terminal()) { int mv = engineMove(sims); g_hist.push_back(g_state); g_state = g_state.play(mv); g_last = (mv == cur().board.n ? -1 : mv); }
    return jstate();
  }
  if (path == "/api/analyze") {
    if (g_state.is_terminal()) return "{\"error\":\"game over\"}";
    return analyze(sims);
  }
  ctype = "text/plain"; return "not found";
}

static std::string httpRead(int fd) {               // read full request (headers + body)
  std::string req; char buf[8192];
  size_t hdr_end = std::string::npos, clen = 0; bool have_clen = false;
  while (true) {
    ssize_t k = recv(fd, buf, sizeof buf, 0);
    if (k <= 0) break;
    req.append(buf, k);
    if (hdr_end == std::string::npos) {
      hdr_end = req.find("\r\n\r\n");
      if (hdr_end != std::string::npos) {
        auto cl = req.find("Content-Length:");
        if (cl != std::string::npos) { clen = std::strtoul(req.c_str() + cl + 15, nullptr, 10); have_clen = true; }
      }
    }
    if (hdr_end != std::string::npos) {
      size_t body_have = req.size() - (hdr_end + 4);
      if (!have_clen || body_have >= clen) break;
    }
  }
  return req;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";   // bundle dir (data/, index.html live here)
  int port = argc > 2 ? std::atoi(argv[2]) : 8799;
  int sims = argc > 3 ? std::atoi(argv[3]) : 220;

  g_net = load_net(dir + "/data/weights.bin");
  std::ifstream bl(dir + "/data/boards.txt"); std::string line;
  while (std::getline(bl, line)) {
    auto tab = line.find('\t'); if (tab == std::string::npos) continue;
    std::string key = line.substr(0, tab), label = line.substr(tab + 1);
    if (key.empty()) continue;
    Board b = loadBoard(dir + "/data/" + key + "/graph.bin");
    StaticFeat sf = load_static(dir + "/data/" + key + "/static.bin", b);
    g_boards.emplace(key, BoardCtx{std::move(b), std::move(sf), label});
    g_order.push_back(key);
  }
  if (g_order.empty()) { std::fprintf(stderr, "no boards found in %s/data\n", dir.c_str()); return 1; }
  reset(g_order.front());

  int srv = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(srv, (sockaddr*)&addr, sizeof addr) < 0) { std::perror("bind"); return 1; }
  listen(srv, 16);
  std::printf("Tiling-Go (native C++ engine) serving on http://127.0.0.1:%d  — %zu boards, %d sims/move\n",
              port, g_order.size(), sims);
  std::fflush(stdout);

  while (true) {
    int fd = accept(srv, nullptr, nullptr);
    if (fd < 0) continue;
    std::string req = httpRead(fd);
    std::string method = req.substr(0, req.find(' '));
    size_t ps = req.find(' ') + 1, pe = req.find(' ', ps);
    std::string path = req.substr(ps, pe - ps);
    auto q = path.find('?'); if (q != std::string::npos) path = path.substr(0, q);  // strip query
    std::string body; auto he = req.find("\r\n\r\n"); if (he != std::string::npos) body = req.substr(he + 4);
    std::string ctype, out = route(method, path, body, ctype, dir, sims);
    std::ostringstream resp;
    resp << "HTTP/1.1 200 OK\r\nContent-Type: " << ctype << "\r\nContent-Length: " << out.size()
         << "\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n" << out;
    std::string r = resp.str();
    send(fd, r.data(), r.size(), 0);
    close(fd);
  }
}
