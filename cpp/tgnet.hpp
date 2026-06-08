// TilingGoNet forward pass in C++ using Apple Accelerate (BLAS sgemm runs on the AMX units).
// Mirrors tilinggo/nn/model.py exactly: input MLP -> L pre-norm residual message blocks
// (h += MLP(LayerNorm(h || mean_j || max_j || g))) -> final LayerNorm -> policy(node)+pass, value.
// Only the search-relevant heads (policy, value) are implemented; aux heads are training-only.
#pragma once
#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>

namespace tg {

struct Linear {            // y = x @ W^T + b ; W is [out, in] row-major (PyTorch layout)
  int out = 0, in = 0;
  std::vector<float> W, b;
};

struct Block {
  std::vector<float> norm_w, norm_b;   // LayerNorm over 4H
  Linear mlp0;                          // 4H -> 2H
  Linear mlp2;                          // 2H -> H
};

struct TGNet {
  int in_dim = 0, hidden = 0, blocks = 0;
  Linear in0, in2;                      // input MLP: Linear(in,H) ReLU Linear(H,H)
  std::vector<Block> block;
  std::vector<float> fnorm_w, fnorm_b;  // final LayerNorm over H
  Linear policy_node;                   // H -> 1
  Linear pass_head;                     // H -> 1
  Linear value0, value2;                // value MLP: Linear(H,H) ReLU Linear(H,1)
};

// ---- weight loading -------------------------------------------------------------------------

class Reader {
 public:
  explicit Reader(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    buf_.resize(sz);
    if (std::fread(buf_.data(), 1, sz, f) != (size_t)sz) { std::fclose(f); throw std::runtime_error("short read"); }
    std::fclose(f);
  }
  int32_t i32() { int32_t v; std::memcpy(&v, buf_.data() + pos_, 4); pos_ += 4; return v; }
  float f32() { float v; std::memcpy(&v, buf_.data() + pos_, 4); pos_ += 4; return v; }
  void floats(std::vector<float>& dst, int n) { dst.resize(n); std::memcpy(dst.data(), buf_.data() + pos_, 4L*n); pos_ += 4L*n; }
  void ints(std::vector<int32_t>& dst, int n) { dst.resize(n); std::memcpy(dst.data(), buf_.data() + pos_, 4L*n); pos_ += 4L*n; }
  // read a length-prefixed tensor (int32 numel, then floats) and assert the expected size
  void tensor(std::vector<float>& dst, int expect) {
    int n = i32();
    if (n != expect) throw std::runtime_error("tensor size mismatch: got " + std::to_string(n) + " expected " + std::to_string(expect));
    floats(dst, n);
  }
  size_t remaining() const { return buf_.size() - pos_; }
 private:
  std::vector<char> buf_;
  size_t pos_ = 0;
};

inline void load_linear(Reader& r, Linear& L, int out, int in) {
  L.out = out; L.in = in;
  r.tensor(L.W, out * in);
  r.tensor(L.b, out);
}

inline TGNet load_net(const std::string& path) {
  Reader r(path);
  if ((uint32_t)r.i32() != 0x54474E31u) throw std::runtime_error("bad magic");
  TGNet net;
  net.in_dim = r.i32(); net.hidden = r.i32(); net.blocks = r.i32();
  int H = net.hidden;
  load_linear(r, net.in0, H, net.in_dim);
  load_linear(r, net.in2, H, H);
  net.block.resize(net.blocks);
  for (auto& b : net.block) {
    r.tensor(b.norm_w, 4*H); r.tensor(b.norm_b, 4*H);
    load_linear(r, b.mlp0, 2*H, 4*H);
    load_linear(r, b.mlp2, H, 2*H);
  }
  r.tensor(net.fnorm_w, H); r.tensor(net.fnorm_b, H);
  load_linear(r, net.policy_node, 1, H);
  load_linear(r, net.pass_head, 1, H);
  load_linear(r, net.value0, H, H);
  load_linear(r, net.value2, 1, H);
  return net;
}

// ---- math kernels ---------------------------------------------------------------------------

// Y[N,out] = X[N,in] @ W^T + b   (W is [out,in])
inline void linear(const Linear& L, const float* X, int N, float* Y) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              N, L.out, L.in, 1.0f, X, L.in, L.W.data(), L.in, 0.0f, Y, L.out);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < L.out; ++j) Y[i*L.out + j] += L.b[j];
}

inline void relu(float* x, long n) { for (long i = 0; i < n; ++i) if (x[i] < 0) x[i] = 0; }

// per-row LayerNorm over D (biased variance, eps=1e-5 — PyTorch nn.LayerNorm default)
inline void layernorm(float* x, int N, int D, const float* g, const float* b) {
  const float eps = 1e-5f;
  for (int i = 0; i < N; ++i) {
    float* row = x + (long)i*D;
    double mean = 0; for (int j = 0; j < D; ++j) mean += row[j]; mean /= D;
    double var = 0; for (int j = 0; j < D; ++j) { double d = row[j]-mean; var += d*d; } var /= D;
    float inv = 1.0f / std::sqrt((float)var + eps);
    for (int j = 0; j < D; ++j) row[j] = ((float)(row[j]-mean)) * inv * g[j] + b[j];
  }
}

// ---- forward --------------------------------------------------------------------------------

struct NetOut {
  std::vector<float> policy_logits;  // length N+1, last is pass (pre-legal-mask)
  float value = 0.0f;
};

// nbr_index/nbr_mask: [N, dmax]; only entries with mask==1 are real neighbours of node i.
inline NetOut forward(const TGNet& net, const float* x, int N, int in_dim,
                      const int32_t* nbr_index, const float* nbr_mask, int dmax) {
  const int H = net.hidden;
  // thread_local scratch — reused across the millions of forward calls (resize keeps capacity)
  static thread_local std::vector<float> t, h, z, normed, m1, m2, g, mean, mx;
  t.resize((long)N*H); h.resize((long)N*H);
  z.resize((long)N*4*H); normed.resize((long)N*4*H);
  m1.resize((long)N*2*H); m2.resize((long)N*H);
  g.resize(H); mean.resize(H); mx.resize(H);
  linear(net.in0, x, N, t.data());
  relu(t.data(), (long)N*H);
  linear(net.in2, t.data(), N, h.data());   // h = input embedding [N,H]

  for (int bi = 0; bi < net.blocks; ++bi) {
    const Block& B = net.block[bi];
    // global mean-pool over nodes
    for (int j = 0; j < H; ++j) g[j] = 0;
    for (int i = 0; i < N; ++i) for (int j = 0; j < H; ++j) g[j] += h[(long)i*H+j];
    for (int j = 0; j < H; ++j) g[j] /= N;
    // per-node: concat [h | mean_j | max_j | g] into z[N,4H]
    for (int i = 0; i < N; ++i) {
      float* zi = z.data() + (long)i*4*H;
      const float* hi = h.data() + (long)i*H;
      float deg = 0;
      for (int j = 0; j < H; ++j) { mean[j] = 0.0f; mx[j] = -1e9f; }
      for (int d = 0; d < dmax; ++d) {
        if (nbr_mask[(long)i*dmax+d] == 0.0f) continue;
        int nb = nbr_index[(long)i*dmax+d];
        const float* hn = h.data() + (long)nb*H;
        deg += 1.0f;
        for (int j = 0; j < H; ++j) { mean[j] += hn[j]; if (hn[j] > mx[j]) mx[j] = hn[j]; }
      }
      float invdeg = 1.0f / (deg > 0 ? deg : 1.0f);
      for (int j = 0; j < H; ++j) {
        zi[j]       = hi[j];
        zi[H+j]     = mean[j] * invdeg;
        zi[2*H+j]   = (deg > 0) ? mx[j] : 0.0f;   // node_mask all 1 here
        zi[3*H+j]   = g[j];
      }
    }
    // normed = LayerNorm(z); m2 = MLP(normed); h += m2
    std::memcpy(normed.data(), z.data(), sizeof(float)*(long)N*4*H);
    layernorm(normed.data(), N, 4*H, B.norm_w.data(), B.norm_b.data());
    linear(B.mlp0, normed.data(), N, m1.data());
    relu(m1.data(), (long)N*2*H);
    linear(B.mlp2, m1.data(), N, m2.data());
    for (long k = 0; k < (long)N*H; ++k) h[k] += m2[k];
  }

  // final norm + global pool
  layernorm(h.data(), N, H, net.fnorm_w.data(), net.fnorm_b.data());
  for (int j = 0; j < H; ++j) g[j] = 0;
  for (int i = 0; i < N; ++i) for (int j = 0; j < H; ++j) g[j] += h[(long)i*H+j];
  for (int j = 0; j < H; ++j) g[j] /= N;

  NetOut out;
  out.policy_logits.resize(N+1);
  // per-node policy logit: h[i] . policy_node.W + b
  for (int i = 0; i < N; ++i) {
    float s = net.policy_node.b[0];
    const float* hi = h.data() + (long)i*H;
    for (int j = 0; j < H; ++j) s += hi[j]*net.policy_node.W[j];
    out.policy_logits[i] = s;
  }
  // pass logit: g . pass_head.W + b
  { float s = net.pass_head.b[0]; for (int j = 0; j < H; ++j) s += g[j]*net.pass_head.W[j]; out.policy_logits[N] = s; }
  // value: tanh( relu(g@value0^T+b) @ value2^T + b )
  static thread_local std::vector<float> v1;
  v1.resize(H);
  linear(net.value0, g.data(), 1, v1.data());
  relu(v1.data(), H);
  float vv = net.value2.b[0];
  for (int j = 0; j < H; ++j) vv += v1[j]*net.value2.W[j];
  out.value = std::tanh(vv);
  return out;
}

}  // namespace tg
