// TilingGoNet forward pass in C++. Mirrors tilinggo/nn/model.py exactly: input MLP -> L
// pre-norm residual message blocks (h += MLP(LayerNorm(h || mean_j || max_j || g))) -> final
// LayerNorm -> policy(node)+pass, value. Only the search-relevant heads are implemented.
//
// GEMM backend: Apple Accelerate (AMX) on macOS native builds; a portable kernel everywhere
// else (Emscripten/WASM, Linux CI). Define TG_PORTABLE_GEMM to force the portable path on
// macOS — tools/parity_native.cpp uses that to assert both backends agree on real weights.
#pragma once
#if defined(__APPLE__) && !defined(TG_PORTABLE_GEMM) && !defined(__EMSCRIPTEN__)
#define TG_USE_ACCELERATE 1
#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#endif
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstring>
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
  Reader(const char* data, size_t len) : buf_(data, data + len) {}   // from memory (WASM path)
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

inline TGNet load_net_from(Reader& r);

inline TGNet load_net(const std::string& path) {
  Reader r(path);
  return load_net_from(r);
}

inline TGNet load_net_buffer(const char* data, size_t len) {
  Reader r(data, len);
  return load_net_from(r);
}

inline TGNet load_net_from(Reader& r) {
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
#ifdef TG_USE_ACCELERATE
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              N, L.out, L.in, 1.0f, X, L.in, L.W.data(), L.in, 0.0f, Y, L.out);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < L.out; ++j) Y[i*L.out + j] += L.b[j];
#else
  // portable kernel: both X rows and W rows are contiguous, so Y[i,j] is a plain dot product
  // with unit-stride loads. On wasm we use explicit SIMD128 intrinsics (the autovectorizer
  // only reached ~6× the JS engine; intrinsics are needed to clear the 10× gate); elsewhere a
  // 4-way unrolled scalar loop. Both keep plain dot-product summation order, so results track
  // the BLAS path to ~1e-6 (verified by tools/parity_dump.cpp on the champion weights).
  const int in = L.in, out = L.out;
  for (int i = 0; i < N; ++i) {
    const float* xi = X + (long)i * in;
    float* yi = Y + (long)i * out;
    int j = 0;
#ifdef __wasm_simd128__
    // register-blocked: 4 output rows share each X load (the kernel is load-bound — wasm
    // SIMD128 has no FMA, so halving load traffic is the available win)
    const float* Wd = L.W.data();
    auto hsum = [](v128_t a) {
      return wasm_f32x4_extract_lane(a, 0) + wasm_f32x4_extract_lane(a, 1)
           + wasm_f32x4_extract_lane(a, 2) + wasm_f32x4_extract_lane(a, 3);
    };
    for (; j + 4 <= out; j += 4) {
      const float* w0 = Wd + (long)j * in,      *w1 = w0 + in;
      const float* w2 = w0 + 2 * (long)in,      *w3 = w0 + 3 * (long)in;
      v128_t a0 = wasm_f32x4_splat(0.f), a1 = a0, a2 = a0, a3 = a0;
      int k = 0;
      for (; k + 4 <= in; k += 4) {
        const v128_t x = wasm_v128_load(xi + k);
        a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(x, wasm_v128_load(w0 + k)));
        a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(x, wasm_v128_load(w1 + k)));
        a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(x, wasm_v128_load(w2 + k)));
        a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(x, wasm_v128_load(w3 + k)));
      }
      float s0 = hsum(a0), s1 = hsum(a1), s2 = hsum(a2), s3 = hsum(a3);
      for (; k < in; ++k) {
        const float x = xi[k];
        s0 += x * w0[k]; s1 += x * w1[k]; s2 += x * w2[k]; s3 += x * w3[k];
      }
      yi[j] = s0 + L.b[j]; yi[j+1] = s1 + L.b[j+1];
      yi[j+2] = s2 + L.b[j+2]; yi[j+3] = s3 + L.b[j+3];
    }
#endif
    for (; j < out; ++j) {                      // scalar path (+ j-tail on wasm)
      const float* wj = L.W.data() + (long)j * in;
      float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      int k = 0;
      for (; k + 4 <= in; k += 4) {
        s0 += xi[k] * wj[k];     s1 += xi[k+1] * wj[k+1];
        s2 += xi[k+2] * wj[k+2]; s3 += xi[k+3] * wj[k+3];
      }
      float s = (s0 + s1) + (s2 + s3);
      for (; k < in; ++k) s += xi[k] * wj[k];
      yi[j] = s + L.b[j];
    }
  }
#endif
}

inline void relu(float* x, long n) { for (long i = 0; i < n; ++i) if (x[i] < 0) x[i] = 0; }

// per-row LayerNorm over D (biased variance, eps=1e-5 — PyTorch nn.LayerNorm default).
// Optionally out-of-place (dst != src) so callers need no separate copy pass.
inline void layernorm_to(float* dst, const float* src, int N, int D,
                         const float* g, const float* b) {
  const float eps = 1e-5f;
  for (int i = 0; i < N; ++i) {
    const float* row = src + (long)i*D;
    float* out = dst + (long)i*D;
#ifdef __wasm_simd128__
    v128_t ms = wasm_f32x4_splat(0.f);
    int j = 0;
    for (; j + 4 <= D; j += 4) ms = wasm_f32x4_add(ms, wasm_v128_load(row + j));
    float mean = wasm_f32x4_extract_lane(ms, 0) + wasm_f32x4_extract_lane(ms, 1)
               + wasm_f32x4_extract_lane(ms, 2) + wasm_f32x4_extract_lane(ms, 3);
    for (; j < D; ++j) mean += row[j];
    mean /= D;
    const v128_t mv = wasm_f32x4_splat(mean);
    v128_t vs = wasm_f32x4_splat(0.f);
    for (j = 0; j + 4 <= D; j += 4) {
      const v128_t d = wasm_f32x4_sub(wasm_v128_load(row + j), mv);
      vs = wasm_f32x4_add(vs, wasm_f32x4_mul(d, d));
    }
    float var = wasm_f32x4_extract_lane(vs, 0) + wasm_f32x4_extract_lane(vs, 1)
              + wasm_f32x4_extract_lane(vs, 2) + wasm_f32x4_extract_lane(vs, 3);
    for (; j < D; ++j) { const float d = row[j] - mean; var += d * d; }
    var /= D;
    const v128_t inv = wasm_f32x4_splat(1.0f / std::sqrt(var + eps));
    for (j = 0; j + 4 <= D; j += 4) {
      const v128_t d = wasm_f32x4_sub(wasm_v128_load(row + j), mv);
      wasm_v128_store(out + j, wasm_f32x4_add(
          wasm_f32x4_mul(wasm_f32x4_mul(d, inv), wasm_v128_load(g + j)),
          wasm_v128_load(b + j)));
    }
    const float invs = 1.0f / std::sqrt(var + eps);
    for (; j < D; ++j) out[j] = (row[j] - mean) * invs * g[j] + b[j];
#else
    double mean = 0; for (int j = 0; j < D; ++j) mean += row[j]; mean /= D;
    double var = 0; for (int j = 0; j < D; ++j) { double d = row[j]-mean; var += d*d; } var /= D;
    float inv = 1.0f / std::sqrt((float)var + eps);
    for (int j = 0; j < D; ++j) out[j] = ((float)(row[j]-mean)) * inv * g[j] + b[j];
#endif
  }
}

inline void layernorm(float* x, int N, int D, const float* g, const float* b) {
  layernorm_to(x, x, N, D, g, b);
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
    layernorm_to(normed.data(), z.data(), N, 4*H, B.norm_w.data(), B.norm_b.data());
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
