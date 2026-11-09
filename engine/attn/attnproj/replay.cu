// attnproj: stock vs GR8_DS_ATTN_PROJ_FUSE=1 ST_ATTN_PROJ (q_a, q_a norm, q_b + rope, kv + norm + rope, window-KV write) at the decode GEMV, same process, ABBA,
// on synthetic operands with the real TP4-rank shapes (HID 5120, QA 1280, 16 heads/rank, window 128, 32-slot geometry, M = 1/3/6/8, ctx 1K/32K/128K).
// Both arms are the engine's own entry point: extern "C" ds41_attn_proj (engine/attn/ds41_attn.cu, compiled as-is from the production TU ds41_attn_dspark.cu), called as
// engine/attn/ds41_attn_ops.cuh op_proj calls it. The arm is chosen by the engine's own process-wide setter ds41_attn_set_proj_fuse(on, fn_dual), which the rank TU
// calls from GR8_DS_ATTN_PROJ_FUSE (ds41_attn_ops.cuh bind). Stock GEMV settings: ds41_attn_set_gv(1, 0, 0, gv_xfrag_launch, gv_gemv_launch_q) + q_b repacked to quad
// records (GR8_DS41_GV_QUAD=1), as the stock bind does. This replay never reads the knob library (knob_stub.cpp answers the ids the TUs read with their stock values).
//
// build: nvcc -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I<engine> replay.cu ds41_attn_dspark.o knob_stub.cpp   (see lane.sh)
// run:   replay [pairs]     (default 51 timed pairs after 8 warm-up pairs, per (arm pair, config); fewer than 51 is refused). One JSON line per record on stdout.
//
// Per config (M, ctx):
//   ref   = ds41_attn_set_proj_fuse(0)  (the stock 10 launches)       cand = ds41_attn_set_proj_fuse(1, gv_gemv_dual_launch)  (5 launches)
//   every timed launch: L2 flush (read 256 MiB) -> poison 0xFF over ALL proj outputs / scratch ("witnesses") -> event -> ONE ds41_attn_proj call -> event; after the event the
//   witnesses are hashed (not in the timed range) and compared with the clean reference of the config: exact must hold for EVERY timed launch of both arms.
//   ABBA: pair p runs (ref, cand) when p is even, (cand, ref) when odd. null = the same ABBA with ref against ref (the noise of delta). off_equals_stock: the reference hash of
//   every config was first taken BEFORE ds41_attn_set_proj_fuse was ever called (the process default), and the ref arm must reproduce it.
// Negative controls (each must report caught = true):
//   control_flip: run cand, XOR one byte of the q output on the device, hash through the same path: must differ from the reference.
//   control_skip: ds41_attn_set_proj_fuse(1, no-op dual hook): the cand path with its q_a + kv launch skipped: must differ from the reference.
// Timing is by stream launches, not a graph: launch-gap savings of a graph are NOT captured.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "../ds41_attn_gv.cuh"       // densev.cuh + the decode GEMV launchers (gv_xfrag_launch, gv_gemv_launch_q, gv_gemv_dual_launch, gv_quad_bind)
#include "../ds41_attn_types.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("{\"error\":\"CUDA %s at %s:%d: %s\"}\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); fflush(stdout); exit(2); } } while (0)

extern "C"
{
void ds41_attn_proj(const ds41a::Weights*, ds41a::Cache*, const ds41a::Scratch*, const void* y, int n, int start, int heads, void* st, const int* dstart);
void ds41_attn_inv_freqs(int compressed, float* out32);
void ds41_attn_set_gv(int on, int tpc, int ks, ds41a::XfragFn fn_xfrag, ds41a::GemvFn fn_gemv);
void ds41_attn_set_proj_fuse(int on, ds41a::GemvDualFn fn_dual);
size_t ds41_attn_scratch_bytes(int rows, int heads);
int ds41_attn_gv_ready(const ds41a::Weights*, int rows);
}

using ds41a::bf16;
namespace
{
constexpr int HIDN = 5120, QA = 1280, HD = 512, HEADS = 16, WIN = 128, MX = 48;
constexpr int RCAP = 4096;                                               // attention row cap (PW): the scratch is carved for it, as the stock first bind does
constexpr int RING_ROWS = WIN + MX - 1;                                  // per-slot decode ring (ns > 1 with PW > WIN), ds41_attn_bind
constexpr int NSLOT = 2;
constexpr size_t FLUSH_BYTES = 256ull << 20;
constexpr int MMAX = 8, L_MAIN = 40;

__device__ __forceinline__ uint32_t hash32(uint64_t i, uint32_t seed)
{
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed ^ static_cast<uint32_t>(i >> 32) * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return h;
}
__global__ void k_fill_gv(uint8_t* p, size_t n, uint32_t seed)       // MXFP8 GEMV records [..][528]: 512 B of e4m3 codes (sign/exp<8: finite), 16 B of E8M0 scales 124..128
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
    {
        const uint32_t h = hash32(i, seed);
        p[i] = (i % 528) < 512 ? static_cast<uint8_t>((h >> 24) & 0xbf) : static_cast<uint8_t>(124 + (h >> 24) % 5);
    }
}
__global__ void k_fill_rows(uint8_t* p, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x) p[i] = static_cast<uint8_t>(hash32(i, seed) >> 20);
}
__global__ void k_fill_bf16(bf16* p, size_t n, float amp, uint32_t seed)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = __float2bfloat16_rn(amp * (static_cast<float>(hash32(i, seed) >> 8) * (1.0f / 8388608.0f) - 1.0f));
}
__global__ void k_stream(const uint4* p, size_t n, uint32_t* sink)
{
    uint32_t x = 0;
    size_t i = blockIdx.x * 256ull + threadIdx.x;
    const size_t st = 256ull * gridDim.x;
    for (; i + 3 * st < n; i += 4 * st)
    {
        uint32_t a[4][4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) gv::ld_nc_v4(p + i + u * st, a[u]);
        #pragma unroll
        for (int u = 0; u < 4; ++u) x ^= a[u][0] ^ a[u][1] ^ a[u][2] ^ a[u][3];
    }
    for (; i < n; i += st) { uint32_t a[4]; gv::ld_nc_v4(p + i, a); x ^= a[0] ^ a[1] ^ a[2] ^ a[3]; }
    if (x == 0x9e3779b9u) sink[0] = x;
}
__global__ void k_flip(uint8_t* p) { p[0] ^= 0x01; }

int g_nsm = 0;
template <class T> T* dmal(size_t n) { T* p = nullptr; CK(cudaMalloc(&p, n * sizeof(T))); CK(cudaMemset(p, 0, n * sizeof(T))); return p; }
uint8_t* gv_mat(size_t bytes, uint32_t seed) { uint8_t* p = dmal<uint8_t>(bytes); k_fill_gv<<<g_nsm * 4, 256>>>(p, bytes, seed); CK(cudaGetLastError()); return p; }
bf16* bf_mat(size_t n, float amp, uint32_t seed) { bf16* p = dmal<bf16>(n); k_fill_bf16<<<g_nsm * 4, 256>>>(p, n, amp, seed); CK(cudaGetLastError()); return p; }
bf16* bf_vec_pos(size_t n, float lo, float hi, uint32_t seed)     // norm weights: lo .. hi
{
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; ++i) { const float f = lo + (hi - lo) * static_cast<float>((i * 2654435761u + seed) % 1000u) / 1000.0f; uint32_t u; memcpy(&u, &f, 4); h[i] = static_cast<uint16_t>((u + 0x8000u) >> 16); }
    bf16* p = dmal<bf16>(n); CK(cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice)); return p;
}
template <class T> T* to_dev(const std::vector<T>& v) { T* p = dmal<T>(v.size()); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
inline size_t gv_bytes(int N, int K) { return static_cast<size_t>(N / 16) * (K / 32) * 528; }

// ---- model state: one weight set (identical shape at every layer; the L2 flush makes every call cold, as in the real step where 40 layers of weights >> L2)
ds41a::Weights g_W;
ds41a::Cache g_C;
ds41a::Scratch g_S;
bf16 *g_y, *g_qnorm;
int *g_pos, *g_sess;
uint8_t *g_flush; uint32_t* g_sink; int g_sgrid;
cudaStream_t g_st;

struct Cfg { int M, ns, rps, ctx; int pos[MMAX], slot[MMAX]; };
Cfg make_cfg(int M, int ctx)
{
    Cfg c{}; c.M = M; c.ctx = ctx; c.ns = M == 8 ? 2 : 1; c.rps = M / c.ns;
    for (int r = 0; r < M; ++r) { const int j = r / c.rps, t = r % c.rps; c.slot[r] = j; c.pos[r] = ctx - c.rps + t; }
    return c;
}
void set_cfg(const Cfg& c)
{
    std::vector<int> p(64, 0), s(64, 0);
    for (int r = 0; r < c.M; ++r) { p[r] = c.pos[r]; s[r] = c.slot[r]; }
    CK(cudaMemcpy(g_pos, p.data(), 64 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_sess, s.data(), 64 * 4, cudaMemcpyHostToDevice));
    g_S.sess = g_sess; g_S.ns = c.ns;
}

// ---- witnesses: every buffer ST_ATTN_PROJ writes (outputs and scratch), per config
struct Wit { std::string name; void* p; size_t bytes; int kind; };   // kind 0 = bytes, 1 = bf16
std::vector<Wit> witnesses(const Cfg& c)
{
    const int M = c.M;
    std::vector<Wit> w;
    w.push_back({"yq", g_S.yq, (size_t)M * HIDN, 0}); w.push_back({"ys", g_S.ys, (size_t)M * HIDN / 32, 0});
    w.push_back({"qrq", g_S.qrq, (size_t)M * QA, 0}); w.push_back({"qrs", g_S.qrs, (size_t)M * QA / 32, 0});
    w.push_back({"qa", g_S.qa, (size_t)M * QA * 2, 1}); w.push_back({"qr", g_S.qr, (size_t)M * QA * 2, 1});
    w.push_back({"q", g_S.q, (size_t)M * HEADS * HD * 2, 1});
    w.push_back({"kv0", g_S.kv0, (size_t)M * HD * 2, 1}); w.push_back({"kvrows", g_S.kvrows, (size_t)M * HD * 2, 1});
    w.push_back({"xf_y", g_S.xf_y, (size_t)HIDN * 16, 0}); w.push_back({"xf_qr", g_S.xf_qr, (size_t)QA * 16, 0});
    for (int r = 0; r < M; ++r)
        w.push_back({"ring" + std::to_string(r), g_C.ring + ((size_t)c.slot[r] * RING_ROWS + c.pos[r] % RING_ROWS) * HD, (size_t)HD * 2, 1});
    return w;
}
void poison_all(const std::vector<Wit>& w) { for (const Wit& x : w) CK(cudaMemsetAsync(x.p, 0xFF, x.bytes, g_st)); }
uint64_t fnv(const std::vector<uint8_t>& v) { uint64_t h = 1469598103934665603ull; for (uint8_t c : v) { h ^= c; h *= 1099511628211ull; } return h; }
struct Snap { std::vector<std::vector<uint8_t>> d; std::vector<uint64_t> h; long unwritten = 0; };
Snap snap(const std::vector<Wit>& w)
{
    Snap s;
    CK(cudaDeviceSynchronize());
    for (const Wit& x : w)
    {
        std::vector<uint8_t> h(x.bytes);
        CK(cudaMemcpy(h.data(), x.p, x.bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i + 3 < h.size(); i += 4) { uint32_t v; memcpy(&v, &h[i], 4); s.unwritten += (v == 0xFFFFFFFFu); }
        s.h.push_back(fnv(h)); s.d.push_back(std::move(h));
    }
    return s;
}
std::string diff_names(const std::vector<Wit>& w, const Snap& a, const Snap& b)
{
    std::string r;
    for (size_t i = 0; i < w.size(); ++i) if (a.h[i] != b.h[i]) r += (r.empty() ? "" : ",") + w[i].name;
    return r;
}
float bf2f(const uint8_t* p) { uint16_t u; memcpy(&u, p, 2); uint32_t x = (uint32_t)u << 16; float f; memcpy(&f, &x, 4); return f; }
// per witness that differs: bytes differing, and for bf16 witnesses max abs / max rel error of b against a
std::string diff_stats_json(const std::vector<Wit>& w, const Snap& a, const Snap& b)
{
    std::string out = "[";
    bool first = true;
    for (size_t i = 0; i < w.size(); ++i)
    {
        if (a.h[i] == b.h[i]) continue;
        long nb = 0; double mabs = 0, mrel = 0;
        for (size_t k = 0; k < a.d[i].size(); ++k) nb += a.d[i][k] != b.d[i][k];
        if (w[i].kind == 1)
            for (size_t k = 0; k + 1 < a.d[i].size(); k += 2)
            {
                const double x = bf2f(&a.d[i][k]), y = bf2f(&b.d[i][k]);
                const double d = std::fabs(x - y);
                if (!(d <= mabs)) mabs = d;
                const double rel = d / std::max(std::fabs(x), 1e-30);
                if (!(rel <= mrel)) mrel = rel;
            }
        char buf[256];
        snprintf(buf, sizeof buf, "%s{\"witness\":\"%s\",\"bytes_differing\":%ld,\"max_abs\":%.6g,\"max_rel\":%.6g}", first ? "" : ",", w[i].name.c_str(), nb, mabs, mrel);
        out += buf; first = false;
    }
    return out + "]";
}
// fp64 host reference of the one stage whose arithmetic the variant re-wrote in a fused kernel and whose input is a device output: qr = bf16(w * (qa * rsqrt(mean(qa^2) + eps)))
// (S.qa is read back from the arm's own run). Returns max abs / max rel of the arm's qr against the fp64 value (rounding to bf16 is a relative error of 2^-8).
void qr_vs_fp64(const Cfg& c, const Snap& s, const std::vector<Wit>& w, double& mabs, double& mrel)
{
    size_t iqa = 0, iqr = 0;
    for (size_t i = 0; i < w.size(); ++i) { if (w[i].name == "qa") iqa = i; if (w[i].name == "qr") iqr = i; }
    std::vector<uint16_t> nw(QA);
    CK(cudaMemcpy(nw.data(), g_qnorm, QA * 2, cudaMemcpyDeviceToHost));
    mabs = mrel = 0;
    for (int t = 0; t < c.M; ++t)
    {
        double ss = 0;
        for (int i = 0; i < QA; ++i) { const double v = bf2f(&s.d[iqa][((size_t)t * QA + i) * 2]); ss += v * v; }
        const double r = 1.0 / std::sqrt(ss / QA + 1e-20);
        for (int i = 0; i < QA; ++i)
        {
            uint32_t x = (uint32_t)nw[i] << 16; float wf; memcpy(&wf, &x, 4);
            const double e = (double)wf * (bf2f(&s.d[iqa][((size_t)t * QA + i) * 2]) * r);
            const double g = bf2f(&s.d[iqr][((size_t)t * QA + i) * 2]);
            const double d = std::fabs(g - e);
            mabs = std::max(mabs, d); mrel = std::max(mrel, d / std::max(std::fabs(e), 1e-30));
        }
    }
}

// ---- arms
void noop_dual(const uint32_t*, const ds41a::Fp8Mat&, bf16*, int, const ds41a::Fp8Mat&, bf16*, int, int, cudaStream_t, int, int) {}   // control_skip: the q_a + kv launch is not made
enum Arm { REF = 0, CAND = 1, SKIP = 2 };
const char* arm_name(int a) { return a == REF ? "ref" : a == CAND ? "cand" : "skip"; }
void set_arm(int a)
{
    if (a == REF) ds41_attn_set_proj_fuse(0, ds41a::gv_gemv_dual_launch);
    else if (a == CAND) ds41_attn_set_proj_fuse(1, ds41a::gv_gemv_dual_launch);
    else ds41_attn_set_proj_fuse(1, noop_dual);
}
void launch_proj(const Cfg& c) { ds41_attn_proj(&g_W, &g_C, &g_S, g_y, c.M, 0, HEADS, g_st, g_pos); CK(cudaGetLastError()); }
void flush_l2() { k_stream<<<g_sgrid, 256, 0, g_st>>>(reinterpret_cast<const uint4*>(g_flush), FLUSH_BYTES / 16, g_sink); }

// one measured launch of arm `a`: flush, poison, events around ONE proj call; the snapshot is taken after the timed range. returns microseconds
double timed(int a, const Cfg& c, const std::vector<Wit>& w, cudaEvent_t e0, cudaEvent_t e1, Snap* out)
{
    set_arm(a);
    flush_l2();
    poison_all(w);
    CK(cudaEventRecord(e0, g_st));
    launch_proj(c);
    CK(cudaEventRecord(e1, g_st));
    CK(cudaStreamSynchronize(g_st));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    if (out) *out = snap(w);
    return ms * 1000.0;
}
double median_of(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
double mean_of(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return s / v.size(); }

struct AB { std::vector<double> ta, tb; std::vector<double> pair_delta; int exact_a = 0, exact_b = 0, n_a = 0, n_b = 0; long unwritten = 0; std::string bad; };
// ABBA over `pairs` timed pairs (+ warm unrecorded pairs): pair p runs (A, B) for even p, (B, A) for odd p. Every launch is compared with `ref`.
AB abba(int A, int B, int pairs, int warm, const Cfg& c, const std::vector<Wit>& w, const Snap& ref, cudaEvent_t e0, cudaEvent_t e1)
{
    AB r;
    for (int p = 0; p < warm + pairs; ++p)
    {
        double t[2]; Snap s[2];
        const int order[2] = {p % 2 == 0 ? A : B, p % 2 == 0 ? B : A};
        for (int k = 0; k < 2; ++k) t[k] = timed(order[k], c, w, e0, e1, &s[k]);
        const double ta = p % 2 == 0 ? t[0] : t[1], tb = p % 2 == 0 ? t[1] : t[0];
        const Snap& sa = p % 2 == 0 ? s[0] : s[1];
        const Snap& sb = p % 2 == 0 ? s[1] : s[0];
        if (p < warm) continue;
        r.ta.push_back(ta); r.tb.push_back(tb); r.pair_delta.push_back(ta - tb);
        const bool ea = sa.h == ref.h && sa.unwritten == 0, eb = sb.h == ref.h && sb.unwritten == 0;
        r.exact_a += ea; r.exact_b += eb; r.n_a++; r.n_b++;
        r.unwritten += sa.unwritten + sb.unwritten;
        if (!ea && r.bad.empty()) r.bad = std::string(arm_name(A)) + ":" + diff_names(w, ref, sa);
        if (!eb && r.bad.empty()) r.bad = std::string(arm_name(B)) + ":" + diff_names(w, ref, sb);
    }
    return r;
}

void peaks(uint8_t* dst, const char* tag, cudaEvent_t e0, cudaEvent_t e1)
{
    std::vector<double> tc, tr;
    for (int r = 0; r < 14; ++r)
    {
        CK(cudaEventRecord(e0, g_st)); CK(cudaMemcpyAsync(dst, g_flush, FLUSH_BYTES, cudaMemcpyDeviceToDevice, g_st)); CK(cudaEventRecord(e1, g_st)); CK(cudaStreamSynchronize(g_st));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (r >= 3) tc.push_back(ms * 1000.0);
    }
    for (int r = 0; r < 14; ++r)
    {
        CK(cudaEventRecord(e0, g_st)); k_stream<<<g_sgrid, 256, 0, g_st>>>(reinterpret_cast<const uint4*>(g_flush), FLUSH_BYTES / 16, g_sink); CK(cudaEventRecord(e1, g_st)); CK(cudaStreamSynchronize(g_st));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (r >= 3) tr.push_back(ms * 1000.0);
    }
    const double c = median_of(tc), rd = median_of(tr), n = static_cast<double>(FLUSH_BYTES);
    printf("{\"lane\":\"attnproj\",\"op\":\"d2d_copy\",\"tag\":\"%s\",\"bytes\":%zu,\"us\":%.1f,\"gbs_copied\":%.1f,\"gbs_traffic\":%.1f,\"reps\":11,\"note\":\"gbs_copied = bytes/t, gbs_traffic = 2*bytes/t (read + write)\"}\n",
           tag, FLUSH_BYTES, c, n / c / 1e3, 2 * n / c / 1e3);
    printf("{\"lane\":\"attnproj\",\"op\":\"read_stream\",\"tag\":\"%s\",\"bytes\":%zu,\"us\":%.1f,\"gbs\":%.1f,\"reps\":11,\"note\":\"read-only 16-byte-load stream kernel over the same buffer; the ceiling for a read-bound kernel\"}\n",
           tag, FLUSH_BYTES, rd, n / rd / 1e3);
    fflush(stdout);
}

void build_model()
{
    ds41a::Weights W{};
    W.wq_a.gv = gv_mat(gv_bytes(QA, HIDN), 1); W.wq_a.N = QA; W.wq_a.K = HIDN;
    W.wq_b.gv = gv_mat(gv_bytes(HEADS * HD, QA), 2); W.wq_b.N = HEADS * HD; W.wq_b.K = QA;
    W.wkv.gv = gv_mat(gv_bytes(HD, HIDN), 3); W.wkv.N = HD; W.wkv.K = HIDN;
    g_qnorm = bf_vec_pos(QA, 0.5f, 1.5f, 11);
    W.q_norm = g_qnorm; W.kv_norm = bf_vec_pos(HD, 0.5f, 1.5f, 12);
    float inv_c[32]; ds41_attn_inv_freqs(1, inv_c);
    W.inv = to_dev(std::vector<float>(inv_c, inv_c + 32));
    g_W = W;
    // stock bind: ds41_attn_set_gv(...) then the GV_QUAD repack of q_b (in place, outside any timing). q_a and kv stay in the stock record layout.
    ds41_attn_set_gv(1, 0, 0, ds41a::gv_xfrag_launch, ds41a::gv_gemv_launch_q);
    CK(cudaDeviceSynchronize());
    const int nq = ds41a::gv_quad_bind(g_W.wq_b, 1, 0, 0, 0);
    CK(cudaDeviceSynchronize());
    if (nq != 1) { printf("{\"error\":\"GV_QUAD repack count %d != 1\"}\n", nq); exit(2); }
    if (!ds41_attn_gv_ready(&g_W, 1)) { printf("{\"error\":\"decode GEMV not ready\"}\n"); exit(2); }
    const size_t sb = ds41_attn_scratch_bytes(RCAP, HEADS);
    uint8_t* base; CK(cudaMalloc(reinterpret_cast<void**>(&base), sb)); CK(cudaMemset(base, 0, sb));
    g_S.carve(base, RCAP, HEADS);
    g_y = bf_mat((size_t)MX * HIDN, 1.0f, 31);
    g_pos = dmal<int>(64); g_sess = dmal<int>(64);
    g_C = ds41a::Cache{};
    g_C.ring_rows = RING_ROWS;
    g_C.ring = bf_mat((size_t)NSLOT * RING_ROWS * HD, 1.5f, 40);
    CK(cudaMalloc(&g_flush, FLUSH_BYTES)); CK(cudaMalloc(&g_sink, 64));
    k_fill_rows<<<g_nsm * 4, 256>>>(g_flush, FLUSH_BYTES, 77);
    { int nb = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_stream, 256, 0)); g_sgrid = g_nsm * nb; }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}
}   // namespace

int main(int argc, char** argv)
{
    const int pairs = argc > 1 ? atoi(argv[1]) : 51, warm = 8;
    if (pairs < 51) { printf("{\"error\":\"pairs must be >= 51\"}\n"); return 2; }
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    CK(cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, 0));
    build_model();
    CK(cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    uint8_t* dst; CK(cudaMalloc(&dst, FLUSH_BYTES));
    size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
    printf("{\"lane\":\"attnproj\",\"op\":\"header\",\"device\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"l2_mb\":%zu,\"mem_total_mb\":%zu,\"mem_free_mb_after_alloc\":%zu,"
           "\"config\":{\"HID\":%d,\"QA\":%d,\"HEADS_PER_RANK\":%d,\"WIN\":%d,\"RCAP\":%d,\"RING_ROWS\":%d,\"GV\":1,\"GV_QUAD\":1,\"calls_per_step\":%d},"
           "\"attn_scratch_bytes\":%zu,\"pairs\":%d,\"warmup_pairs\":%d,\"arms\":{\"ref\":\"ds41_attn_set_proj_fuse(0): 10 launches\",\"cand\":\"ds41_attn_set_proj_fuse(1, gv_gemv_dual_launch): 5 launches\"},"
           "\"method\":\"ABBA, L2 flush, poison 0xFF over all proj outputs/scratch before every timed launch, events around ONE ds41_attn_proj call, byte-compare every launch against the clean reference, stream launches (not a graph)\"}\n",
           pr.name, pr.major, pr.minor, g_nsm, static_cast<size_t>(pr.l2CacheSize) >> 20, tot >> 20, fr >> 20, HIDN, QA, HEADS, WIN, RCAP, RING_ROWS, L_MAIN,
           ds41_attn_scratch_bytes(RCAP, HEADS), pairs, warm);
    fflush(stdout);
    peaks(dst, "start", e0, e1);

    const int Ms[4] = {1, 3, 6, 8}, ctxs[3] = {1024, 32768, 131072};
    // the process default first: no ds41_attn_set_proj_fuse call has been made yet, so these hashes are the stock proj stage as it ships
    std::vector<Snap> stock_ref;
    for (int mi = 0; mi < 4; ++mi)
        for (int ci = 0; ci < 3; ++ci)
        {
            const Cfg c = make_cfg(Ms[mi], ctxs[ci]);
            set_cfg(c);
            const auto w = witnesses(c);
            poison_all(w);
            launch_proj(c);
            stock_ref.push_back(snap(w));
        }
    int cfgn = 0;
    for (int mi = 0; mi < 4; ++mi)
        for (int ci = 0; ci < 3; ++ci, ++cfgn)
        {
            const Cfg c = make_cfg(Ms[mi], ctxs[ci]);
            set_cfg(c);
            const auto w = witnesses(c);
            printf("{\"lane\":\"attnproj\",\"op\":\"config\",\"M\":%d,\"ctx\":%d,\"sessions\":%d,\"rows_per_session\":%d,\"first_pos\":%d,\"last_pos\":%d,\"witnesses\":%zu}\n",
                   c.M, c.ctx, c.ns, c.rps, c.pos[0], c.pos[c.M - 1], w.size());
            // clean reference: ref arm, every witness poisoned first (also proves every witness is fully rewritten); twice for determinism
            set_arm(REF);
            poison_all(w); launch_proj(c); const Snap ref = snap(w);
            poison_all(w); launch_proj(c); const Snap ref2 = snap(w);
            const bool det = ref.h == ref2.h, off_eq = ref.h == stock_ref[cfgn].h;
            // single cand run: exactness with error statistics against ref
            set_arm(CAND);
            poison_all(w); launch_proj(c); const Snap cs = snap(w);
            const bool exact1 = cs.h == ref.h && cs.unwritten == 0;
            // controls
            poison_all(w); launch_proj(c); CK(cudaStreamSynchronize(g_st));
            k_flip<<<1, 1, 0, g_st>>>(reinterpret_cast<uint8_t*>(g_S.q)); CK(cudaGetLastError());
            const Snap fl = snap(w);
            const bool flip_caught = fl.h != ref.h;
            set_arm(SKIP);
            poison_all(w); launch_proj(c); const Snap sk = snap(w);
            const bool skip_caught = sk.h != ref.h || sk.unwritten != 0;
            double qr_abs_ref, qr_rel_ref, qr_abs_cand, qr_rel_cand;
            qr_vs_fp64(c, ref, w, qr_abs_ref, qr_rel_ref); qr_vs_fp64(c, cs, w, qr_abs_cand, qr_rel_cand);
            printf("{\"lane\":\"attnproj\",\"op\":\"check\",\"M\":%d,\"ctx\":%d,\"ref_deterministic\":%s,\"ref_unwritten_words\":%ld,\"off_equals_stock\":%s,\"cand_exact_single\":%s,\"cand_unwritten_words\":%ld,"
                   "\"inexact\":%s,\"qr_vs_fp64\":{\"ref_max_abs\":%.6g,\"ref_max_rel\":%.6g,\"cand_max_abs\":%.6g,\"cand_max_rel\":%.6g,\"note\":\"fp64 host rmsnorm of the arm's own qa; rel error of a bf16 rounding is <= 2^-8\"},"
                   "\"control_flip_caught\":%s,\"control_flip_witnesses\":\"%s\",\"control_skip_caught\":%s,\"control_skip_witnesses\":\"%s\"}\n",
                   c.M, c.ctx, det ? "true" : "false", ref.unwritten, off_eq ? "true" : "false", exact1 ? "true" : "false", cs.unwritten, diff_stats_json(w, ref, cs).c_str(),
                   qr_abs_ref, qr_rel_ref, qr_abs_cand, qr_rel_cand, flip_caught ? "true" : "false", diff_names(w, ref, fl).c_str(), skip_caught ? "true" : "false", diff_names(w, ref, sk).c_str());
            fflush(stdout);
            // timed: ABBA ref vs cand, then the null (ref vs ref)
            const AB ab = abba(REF, CAND, pairs, warm, c, w, ref, e0, e1);
            const AB nu = abba(REF, REF, pairs, warm, c, w, ref, e0, e1);
            const double rm = median_of(ab.ta), cm = median_of(ab.tb), nm = median_of(nu.ta) - median_of(nu.tb);
            const double d = rm - cm, dpair = median_of(ab.pair_delta), npair = median_of(nu.pair_delta);
            printf("{\"lane\":\"attnproj\",\"op\":\"proj_ab\",\"M\":%d,\"ctx\":%d,\"pairs\":%d,\"ref_us\":%.2f,\"ref_us_mean\":%.2f,\"cand_us\":%.2f,\"cand_us_mean\":%.2f,\"delta_us\":%.2f,\"delta_us_pair_median\":%.2f,"
                   "\"delta_ms_per_step\":%.4f,\"null_us\":%.2f,\"null_us_pair_median\":%.2f,\"null_ms_per_step\":%.4f,\"calls_per_step\":%d,"
                   "\"exact\":%s,\"exact_launches\":%d,\"launches\":%d,\"null_exact\":%s,\"unwritten_words\":%ld,\"bad_witness\":\"%s\","
                   "\"note\":\"delta = ref - cand, positive = variant faster; 5090 plumbing numbers, not GB10\"}\n",
                   c.M, c.ctx, pairs, rm, mean_of(ab.ta), cm, mean_of(ab.tb), d, dpair, d * L_MAIN / 1e3, nm, npair, nm * L_MAIN / 1e3, L_MAIN,
                   ab.exact_a == ab.n_a && ab.exact_b == ab.n_b ? "true" : "false", ab.exact_a + ab.exact_b, ab.n_a + ab.n_b,
                   nu.exact_a == nu.n_a && nu.exact_b == nu.n_b ? "true" : "false", ab.unwritten + nu.unwritten, ab.bad.c_str());
            fflush(stdout);
        }
    peaks(dst, "end", e0, e1);
    printf("{\"lane\":\"attnproj\",\"op\":\"done\"}\n");
    return 0;
}
