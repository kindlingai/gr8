// attnsplit: times the STOCK DS41 decode ATTENTION unit stage by stage (ST_ATTN_PROJ, ST_CA_ROW, ST_INDEX_TOPK, ST_CAND_BLOCKS, ST_ATTN_CORE, ST_ATTN_OUT) on synthetic
// operands with the real TP4-rank shapes (HID 5120, 16 heads/rank, 2 o groups, window 128, top-512 of 327680-token capacity, 32 slots geometry).
// Every launch is the engine's own: the extern "C" ds41_attn_* / ds41_idx_* entry points (engine/attn/ds41_attn.cu, ds41_bind.cu, ds41_idx.cu, compiled as-is and linked)
// called in the order and with the arguments of engine/attn/ds41_attn_ops.cuh and engine/attn/ds41_idx_ops.cuh (op_proj, op_comp_row, op_core, op_out, ds41_idx_op_*),
// and the decode GEMV hooks of engine/attn/ds41_attn_gv.cuh. No variant kernel. The stock production TU is attn/ds41_attn_dspark.cu (= ds41_attn.cu + the DSpark runtime header; same ds41a kernels).
// Stock launch settings (work-decode/mb/ds/stock-effective.txt): GR8_DS_ATTN_GV=1 (tpc 0, ks 0) + GR8_DS41_GV_QUAD=1, ATTN_TC=1, TC_MF=2, TC_TILE=1, COMP_FP8=0, SRC_FAST=3, ATTN_CPA=1,
// COMP_FP4=1 (packed 288 B compressed rows), ATTN_PW=4096, IDX_PW=4096, IDX_PWK=0, IDX_MAXK=0, PIPE_B/PIPE_QR/Q_TRIG off, REUSE off, cap 327680, 32 slots (S2), MX=48.
//
// build: nvcc -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I<engine> replay.cu ds41_attn.o ds41_bind.o ds41_idx.o knob_stub.cpp   (see lane.sh)
// run:   replay [reps [skip_stage_name]]      (default 51 timed reps after 8 warm-up reps, per series). One JSON line per series on stdout.
//
// Method per series [a, b) of a layer-class chain (stages are the launch groups of one op, e.g. klen / wq_b / weights_proj / topk are the four launches of ST_INDEX_TOPK):
//   L2 flush (read 256 MiB) -> stages [0, a) untimed -> poison (0xFF) every output ("witness") of stages >= a -> event -> stages [a, b) -> event.
// The last timed rep then runs [b, end) and compares the hash of EVERY witness of the chain with a clean full-chain reference (poison_ok), and no witness may contain unwritten words.
// Layer classes (one representative each, multiplied by the stock call counts in the sum lines):
//   l0 = layers 0,1 (ratio 0); l2 = a ratio-2 KV source (2, 8, 14); l20 = layer 20 (ratio 1 source, candidate blocks); l24 = a candidate index layer (24, 28, 32, 36).
// Sessions: M = 1, 3, 6 -> one session (rows at consecutive positions); M = 8 -> 2 sessions of 4 rows (S2 slots 0 and 1), as the engine splits rows over Model::ds_ns.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cuda_runtime.h>
#include "../ds41_attn_gv.cuh"       // densev.cuh + the decode GEMV launchers (gv_xfrag_launch, gv_gemv_launch[_q], gv_quad_bind)
#include "../ds41_attn_types.h"
#include "../ds41_idx_api.h"
#include "../ds41_idx_layout.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("{\"error\":\"CUDA %s at %s:%d: %s\"}\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); fflush(stdout); exit(2); } } while (0)

// the engine's extern "C" attention entry points (copied from attn/ds41_attn_ops.cuh; no header exposes them without the kernels)
extern "C"
{
void ds41_attn_proj(const ds41a::Weights*, ds41a::Cache*, const ds41a::Scratch*, const void* y, int n, int start, int heads, void* st, const int* dstart);
void ds41_attn_comp_row(const ds41a::Weights*, ds41a::Cache*, const ds41a::Scratch*, const void* y, int n, int start, void* st, const int* dstart);
void ds41_attn_core(const ds41a::Weights*, const ds41a::Cache*, const void* comp_src, const ds41a::Scratch*, int n, int start, int heads, const int* topk, int ldtk, int TK,
                    void* st, const int* dstart);
void ds41_attn_out(const ds41a::Weights*, const ds41a::Scratch*, int n, int start, int heads, void* out, void* st, const int* dstart, int direct_c2);
void ds41_attn_inv_freqs(int compressed, float* out32);
void ds41_attn_set_gv(int on, int tpc, int ks, ds41a::XfragFn fn_xfrag, ds41a::GemvFn fn_gemv);
void ds41_attn_set_src_fast(int bits);
void ds41_attn_set_tc(int on);
void ds41_attn_set_tc_mfast(int on);
void ds41_attn_set_tc_tile(int mode);
void ds41_attn_set_comp_fp8(int on);
void ds41_attn_set_cpasync(int on);
void ds41_attn_set_split(int on);
size_t ds41_attn_scratch_bytes(int rows, int heads);
int ds41_attn_gv_ready(const ds41a::Weights*, int rows);
}

using ds41a::bf16;
namespace
{
// ---- stock geometry (engine/model_geom_ds41.h, runtime/ds41_ops.h; sources are named in CALLS_* below)
constexpr int HIDN = 5120, QA = 1280, HD = 512, HEADS = 16, G = HEADS / 8, WIN = 128, OLORA = 1024, MX = 48, TKN = 512, CAND_BLK = 2048;
constexpr int IDX_HEADS = 32, IDX_DIM = 128, IDX_WQB_OUT = 4096;
constexpr int CAP = 327680, RCAP = 4096, MAXROWS = CAP + RCAP;          // capacity, indexer/attention row cap (PW), compressed-row bound of the ratio-1 layers
constexpr int RING_ROWS = WIN + MX - 1;                                  // per-slot decode ring (ns > 1 with PW > WIN), ds41_attn_bind
constexpr int COMP_ROWS = CAP / 2 + 2;                                   // C.compq rows per slot (ratio-2 sources)
constexpr int ROWS_KV2 = ds41i::ica_aligned_rows((CAP + RCAP) / 2 + 2);  // ica rows per slot of kv sources 2, 8, 14
constexpr int ROWS_KV3 = CAP + RCAP;                                     // layer 20 (ratio 1): ica and ca rows per slot
constexpr int SESS_CAP = 6;                                              // per-session index scratch rows (DSpark k + 1, MTP=5)
constexpr int NSLOT = 2;                                                 // slots allocated (the largest session count of the sweep)
constexpr size_t FLUSH_BYTES = 256ull << 20;
constexpr int MMAX = 8;

constexpr int L_MAIN = 40;
static const char* CALLS_ATTN = "engine/runtime/ds41_ops.h:144 loop over L_MAIN=40 trunk layers (engine/model_geom_ds41.h:20): ST_ATTN_PROJ/CORE/OUT every layer";
static const char* CALLS_CA = "engine/attn/ds41_idx_ops.cuh ds41_idx_op_ca_row at kv sources 2,8,14 (ratio 2: comp_row + ica_row) and 20 (ratio 1)";
static const char* CALLS_IDX = "ds41_idx_op_index_topk at index sources 2,8,14,20 (dense) and 24,28,32,36 (candidate); cand_blocks at 20 (model_geom_ds41.h is_index_source)";

// ---- fill kernels (finite values only)
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
__global__ void k_fill_rows(uint8_t* p, size_t n, int row, int codes, int lo, int span, uint32_t seed)   // rows of `row` bytes: `codes` random bytes (&0x77 = finite fp4/e4m3), then scale bytes lo .. lo+span-1
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
    {
        const uint32_t h = hash32(i, seed);
        p[i] = static_cast<int>(i % row) < codes ? static_cast<uint8_t>((h >> 20) & 0xff) : static_cast<uint8_t>(lo + (h >> 24) % span);
    }
}
__global__ void k_fill_bf16(bf16* p, size_t n, float amp, uint32_t seed)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = __float2bfloat16_rn(amp * (static_cast<float>(hash32(i, seed) >> 8) * (1.0f / 8388608.0f) - 1.0f));
}
__global__ void k_fill_f32(float* p, size_t n, float amp, uint32_t seed)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = amp * (static_cast<float>(hash32(i, seed) >> 8) * (1.0f / 8388608.0f) - 1.0f);
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

int g_nsm = 0, g_fill = 0;
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
inline size_t gv_bytes(int N, int K, int groups = 1) { return static_cast<size_t>(groups) * (N / 16) * (K / 32) * 528; }

// ---- model state, built once (one slab of NSLOT slots; the stock model has 32, the slot stride is the stock one)
struct Cls { ds41a::Weights W; ds41a::Cache C; };
struct Model
{
    // attention weights (identical shape every layer; ONE set shared by all classes: the L2 flush makes every call cold, as in the real step where 40 layers of weights >> L2)
    ds41a::Weights W0, W1, W2;      // ratio 0 (inv theta 10000), ratio 1 (no compressor), ratio-2 source (with compressor)
    ds41a::Cache C0, C2, C20, C24;
    ds41a::Scratch S;
    uint8_t* scr_base;
    bf16 *y, *out;                  // hc_y [MX][5120] input, out [MX][5120] (Model::sc.part[0])
    int *pos, *sess;
    // indexer
    const bf16 *idx_wk, *idx_knorm, *wproj, *cmp_wkv, *cmp_norm;
    uint8_t* wqb_gv;                // idx wq_b records N=4096 K=1280
    uint8_t *ica[4], *ca3;          // by kv source 2, 8, 14, 20 (ca: layer 20 only); NSLOT slabs
    int rows_kv[4];
    int* klen; bf16 *q_b, *iwproj, *y512, *pre; int32_t* topk;
    void* scratch;                  // ds41_idx_scratch_bytes(RCAP, MAXROWS)   (one session)
    void* sscr[NSLOT];              // ds41_idx_scratch_bytes_flat(SESS_CAP, MAXROWS)  (ns > 1)
    uint8_t *flush; uint32_t* sink; int sgrid;
    size_t scratch_bytes, sscr_bytes, attn_scratch_bytes;
};
Model g;

struct Cfg { int M, ns, rps, ctx; int pos[MMAX], slot[MMAX]; };
Cfg make_cfg(int M, int ctx)
{
    Cfg c{}; c.M = M; c.ctx = ctx; c.ns = M == 8 ? 2 : 1; c.rps = M / c.ns;
    for (int r = 0; r < M; ++r) { const int j = r / c.rps, t = r % c.rps; c.slot[r] = j; c.pos[r] = ctx - c.rps + t; }
    return c;
}

// ---- stages and chains
struct Wit { std::string name; const void* p; size_t bytes; };
struct Stage { std::string name; std::function<void(cudaStream_t)> run; std::vector<Wit> wit; };
struct Series { std::string name; int a, b; int calls; double bw, bkv; const char* calls_src; const char* note; };
struct Chain { std::string name; std::vector<Stage> st; std::vector<Series> ser; };

int g_skip_on = 0; std::string g_skip_name;
void run_stage(const Chain& ch, int s, cudaStream_t st)
{
    if (g_skip_on && ch.st[s].name == g_skip_name) return;
    ch.st[s].run(st);
    CK(cudaGetLastError());
}
void run_range(const Chain& ch, int a, int b, cudaStream_t st) { for (int s = a; s < b; ++s) run_stage(ch, s, st); }
void poison_stage(const Chain& ch, int s, cudaStream_t st) { for (const Wit& w : ch.st[s].wit) CK(cudaMemsetAsync(const_cast<void*>(w.p), 0xFF, w.bytes, st)); }

uint64_t fnv(const std::vector<uint8_t>& v) { uint64_t h = 1469598103934665603ull; for (uint8_t c : v) { h ^= c; h *= 1099511628211ull; } return h; }
struct Hash { std::vector<uint64_t> h; std::vector<long> unwritten; };
Hash hash_all(const Chain& ch)
{
    Hash r;
    CK(cudaDeviceSynchronize());
    for (const Stage& s : ch.st)
        for (const Wit& w : s.wit)
        {
            std::vector<uint8_t> h(w.bytes);
            CK(cudaMemcpy(h.data(), w.p, w.bytes, cudaMemcpyDeviceToHost));
            long un = 0; for (size_t i = 0; i + 3 < h.size(); i += 4) { uint32_t x; memcpy(&x, &h[i], 4); un += (x == 0xFFFFFFFFu); }
            r.h.push_back(fnv(h)); r.unwritten.push_back(un);
        }
    return r;
}
std::string wit_name(const Chain& ch, size_t idx)
{
    size_t k = 0;
    for (const Stage& s : ch.st) for (const Wit& w : s.wit) { if (k++ == idx) return s.name + ":" + w.name; }
    return "?";
}

// ---- builders for the witnesses
void add(Stage& s, const char* name, const void* p, size_t bytes) { s.wit.push_back({name, p, bytes}); }
// ica page of one slot (ds41_idx.cuh ica_map): rows_kv rows * 64 B of codes, then rows_kv * 4 B of scale words; page stride rows_kv * 68
void add_ica(Stage& s, const void* ica, int slot, int row, int rows_kv)
{
    const uint8_t* page = static_cast<const uint8_t*>(ica) + (size_t)slot * rows_kv * 68;
    add(s, "ica_codes", page + (size_t)row * 64, 64);
    add(s, "ica_sf", page + (size_t)rows_kv * 64 + (size_t)row * 4, 4);
}

Chain build_chain(int cls, const Cfg& c)      // cls: 0 = l0, 2 = l2, 20 = l20, 24 = l24
{
    Chain ch; ch.name = cls == 0 ? "l0" : cls == 2 ? "l2" : cls == 20 ? "l20" : "l24";
    const int M = c.M, ns = c.ns, rps = c.rps;
    ds41a::Scratch* S = &g.S;
    ds41a::Weights* W = cls == 0 ? &g.W0 : cls == 2 ? &g.W2 : &g.W1;
    ds41a::Cache* C = cls == 0 ? &g.C0 : cls == 2 ? &g.C2 : cls == 20 ? &g.C20 : &g.C24;
    const int ratio = cls == 0 ? 0 : cls == 2 ? 2 : 1;
    const int kix = cls == 2 ? 0 : 3;                              // the kv source whose ica rows the index stages of this class read (layer 2 stands for 2, 8, 14: identical shape)
    Model& m = g;
    const int* dpos = m.pos;
    const bf16* y = m.y;
    const int rows_kv = m.rows_kv[kix];
    // ---- ST_ATTN_PROJ
    {
        Stage s; s.name = "proj";
        s.run = [=](cudaStream_t st) { ds41_attn_proj(W, C, S, y, M, 0, HEADS, st, dpos); };
        add(s, "yq", S->yq, (size_t)M * HIDN); add(s, "ys", S->ys, (size_t)M * HIDN / 32); add(s, "qrq", S->qrq, (size_t)M * QA); add(s, "qrs", S->qrs, (size_t)M * QA / 32);
        add(s, "qa", S->qa, (size_t)M * QA * 2); add(s, "qr", S->qr, (size_t)M * QA * 2); add(s, "q", S->q, (size_t)M * HEADS * HD * 2);
        add(s, "kv0", S->kv0, (size_t)M * HD * 2); add(s, "kvrows", S->kvrows, (size_t)M * HD * 2);
        add(s, "xf_y", S->xf_y, (size_t)HIDN * 16); add(s, "xf_qr", S->xf_qr, (size_t)QA * 16);
        for (int r = 0; r < M; ++r)
            add(s, ("ring" + std::to_string(r)).c_str(), C->ring + ((size_t)c.slot[r] * RING_ROWS + c.pos[r] % RING_ROWS) * HD, HD * 2);
        ch.st.push_back(std::move(s));
    }
    if (cls == 2)       // ---- ST_CA_ROW at a ratio-2 source: ds41_attn_op_comp_row, then ds41_idx_ica_row per session (cnorm compacted j*((rps+1)/2))
    {
        Stage s; s.name = "comp_row";
        s.run = [=](cudaStream_t st) { ds41_attn_comp_row(W, C, S, y, M, 0, st, dpos); };
        add(s, "ckv", S->ckv, (size_t)M * HD * 4); add(s, "csc", S->csc, (size_t)M * HD * 4);
        for (int j = 0; j < ns; ++j)
        {
            int pairs = 0;
            for (int t = 0; t < rps; ++t)
            {
                const int p = c.pos[j * rps + t];
                if (p & 1) { ++pairs; add(s, "compq", C->compq + (size_t)j * C->comp_ss + (size_t)(p / 2) * 288, 288); }
            }
            if (pairs) add(s, "cnorm", S->cnorm + (size_t)j * ((rps + 1) / 2) * HD, (size_t)pairs * HD * 2);
        }
        ch.st.push_back(std::move(s));
        Stage t; t.name = "ica_row";
        t.run = [=](cudaStream_t st)
        {
            for (int j = 0; j < ns; ++j)
                ds41_idx_ica_row(S->cnorm + (size_t)j * ((rps + 1) / 2) * HD, m.idx_wk, m.idx_knorm, dpos + j * rps, 2, m.ica[kix], m.sess + j * rps, rows_kv, rps, st);
        };
        for (int r = 0; r < M; ++r) if (c.pos[r] & 1) add_ica(t, m.ica[kix], c.slot[r], c.pos[r] / 2, rows_kv);
        ch.st.push_back(std::move(t));
    }
    if (cls == 20)      // ---- ST_CA_ROW at layer 20 (ratio 1): gemv cmp_wkv, ca_row_r1, ica_row
    {
        Stage s; s.name = "cmp_gemv";
        s.run = [=](cudaStream_t st) { ds41_idx_gemv(y, HIDN, m.cmp_wkv, 512, HIDN, m.y512, 512, M, st); };
        add(s, "y512", m.y512, (size_t)M * 512 * 2);
        ch.st.push_back(std::move(s));
        Stage t; t.name = "ca_row_r1";
        t.run = [=](cudaStream_t st)
        {
            for (int j = 0; j < ns; ++j)
                ds41_idx_ca_row_r1(m.y512 + (size_t)j * rps * 512, m.cmp_norm, dpos + j * rps, m.ca3, m.sess + j * rps, rows_kv, m.pre + (size_t)j * rps * 512, nullptr, rps, st);
        };
        add(t, "pre", m.pre, (size_t)M * 512 * 2);
        for (int r = 0; r < M; ++r) add(t, "ca", m.ca3 + ((size_t)c.slot[r] * rows_kv + c.pos[r]) * 288, 288);
        ch.st.push_back(std::move(t));
        Stage u; u.name = "ica_row";
        u.run = [=](cudaStream_t st)
        {
            for (int j = 0; j < ns; ++j)
                ds41_idx_ica_row(m.pre + (size_t)j * rps * 512, m.idx_wk, m.idx_knorm, dpos + j * rps, 1, m.ica[kix], m.sess + j * rps, rows_kv, rps, st);
        };
        for (int r = 0; r < M; ++r) add_ica(u, m.ica[kix], c.slot[r], c.pos[r], rows_kv);
        ch.st.push_back(std::move(u));
    }
    if (cls != 0)       // ---- ST_INDEX_TOPK: klen, idx wq_b (GV, plain launcher, reusing xf_qr from proj), weights_proj, topk
    {
        const bool cand = cls == 24;
        auto stage_of = [=](int j)
        {
            Ds41IdxStage t{};
            t.q_b = m.q_b + (size_t)j * rps * IDX_WQB_OUT; t.wproj = m.iwproj + (size_t)j * rps * IDX_HEADS; t.dpos = dpos + j * rps;
            t.klen = m.klen + j * rps; t.max_klen = rows_kv; t.R = rps;
            t.ica = m.ica[kix]; t.pages = m.sess + j * rps; t.rpp = rows_kv;
            t.scratch = ns > 1 ? m.sscr[j] : m.scratch; t.Rcap = ns > 1 ? SESS_CAP : RCAP; t.max_rows = MAXROWS; t.use_cand = 0;
            t.flat = ns > 1; t.pipe_b = 0; t.q_trig = 0; t.pipe_qr = 0;
            t.out = m.topk + (size_t)j * rps * TKN; t.ldo = TKN;
            return t;
        };
        Stage k; k.name = "idx_klen";
        k.run = [=](cudaStream_t st) { ds41_idx_klen(dpos, ratio, m.klen, M, st); };
        add(k, "klen", m.klen, (size_t)M * 4);
        ch.st.push_back(std::move(k));
        Stage q; q.name = "idx_wqb";
        q.run = [=](cudaStream_t st)
        {
            ds41a::Fp8Mat Wq{}; Wq.gv = m.wqb_gv; Wq.N = IDX_WQB_OUT; Wq.K = QA;
            ds41a::gv_gemv_launch(S->xf_qr, Wq, m.q_b, IDX_WQB_OUT, M, 1, 0, st, 1, 0);
        };
        add(q, "q_b", m.q_b, (size_t)M * IDX_WQB_OUT * 2);
        ch.st.push_back(std::move(q));
        Stage p; p.name = "idx_wproj";
        p.run = [=](cudaStream_t st) { ds41_idx_gemv(y, HIDN, m.wproj, IDX_HEADS, HIDN, m.iwproj, IDX_HEADS, M, st); };
        add(p, "wproj", m.iwproj, (size_t)M * IDX_HEADS * 2);
        ch.st.push_back(std::move(p));
        Stage t; t.name = "idx_topk";
        t.run = [=](cudaStream_t st)
        {
            for (int j = 0; j < ns; ++j) { Ds41IdxStage d = stage_of(j); d.use_cand = cand; d.cand_src = cls == 20; ds41_idx_topk(&d, st); }
        };
        // rows with fewer than 512 keys pad their list tail with -1 (the poison pattern): only the first min(512, klen) entries of a row are witnesses
        for (int r = 0; r < M; ++r) add(t, "topk", m.topk + (size_t)r * TKN, (size_t)std::min(TKN, (c.pos[r] + 1) / ratio) * 4);
        ch.st.push_back(std::move(t));
        if (cls == 20)      // ---- ST_CAND_BLOCKS
        {
            Stage b; b.name = "cand_blocks";
            b.run = [=](cudaStream_t st)
            {
                for (int j = 0; j < ns; ++j) { Ds41IdxStage d = stage_of(j); d.cand_src = 1; ds41_idx_cand_blocks(&d, st); }
            };
            // the list pads its tail with -1, which is the poison pattern: only the valid prefix of a row is a witness (min(2048, ceil(klen / 8)) blocks, klen = pos + 1 at ratio 1)
            for (int j = 0; j < ns; ++j)
            {
                const int32_t* list = static_cast<const int32_t*>(ds41_idx_cand_list(ns > 1 ? m.sscr[j] : m.scratch, ns > 1 ? SESS_CAP : RCAP, MAXROWS));
                for (int rr = 0; rr < rps; ++rr)
                    add(b, "cand", list + (size_t)rr * CAND_BLK, (size_t)std::min(CAND_BLK, (c.pos[j * rps + rr] + 1 + 7) / 8) * 4);
            }
            ch.st.push_back(std::move(b));
        }
    }
    // ---- ST_ATTN_CORE (ratio 0: window + sink only; ratio 2: own packed compq rows; ratio 1: the layer-20 ca rows)
    {
        Stage s; s.name = "core";
        const void* comp = cls == 0 ? nullptr : cls == 2 ? (const void*)C->compq : (const void*)m.ca3;
        const int TK = cls == 0 ? 0 : TKN;
        const int* tk = m.topk;
        s.run = [=](cudaStream_t st) { ds41_attn_core(W, C, comp, S, M, 0, HEADS, tk, TKN, TK, st, dpos); };
        add(s, "o", S->o, (size_t)M * HEADS * HD * 2);
        ch.st.push_back(std::move(s));
    }
    if (cls == 0)       // ---- ST_ATTN_OUT (same weight shape at every layer: timed once, in the l0 chain)
    {
        Stage s; s.name = "out";
        s.run = [=](cudaStream_t st) { ds41_attn_out(W, S, M, 0, HEADS, m.out, st, dpos, 0); };
        add(s, "oq", S->oq, (size_t)M * HEADS * HD); add(s, "os", S->os, (size_t)M * HEADS * HD / 32);
        add(s, "oaq", S->oaq, (size_t)M * G * OLORA); add(s, "oas", S->oas, (size_t)M * G * OLORA / 32);
        add(s, "oa", S->oa, (size_t)M * G * OLORA * 2); add(s, "xf_o", S->xf_o, (size_t)HEADS * HD * 16); add(s, "xf_oa", S->xf_oa, (size_t)G * OLORA * 16);
        add(s, "out", m.out, (size_t)M * HIDN * 2);
        ch.st.push_back(std::move(s));
    }
    // ---- series with byte accounting (ESTIMATES: unique bytes the op reads, weights and KV separately; labelled in the report)
    const double wq_a = gv_bytes(QA, HIDN), wq_b = gv_bytes(HEADS * HD, QA), wkv = gv_bytes(HD, HIDN), wo_a = gv_bytes(OLORA, 4096, G), wo_b = gv_bytes(HIDN, G * OLORA);
    const double wproj_b = (double)IDX_HEADS * HIDN * 2, iwqb = gv_bytes(IDX_WQB_OUT, QA), cw = 2.0 * HD * HIDN * 2, cmpw = (double)HD * HIDN * 2, iwk = (double)IDX_DIM * HD * 2;
    const double klen = ratio ? (double)(c.ctx) / ratio : 0;             // keys of the last row of a session (position ~ ctx)
    const double ica_scan = ns * klen * 68;                              // unique index-K rows a dense logits pass reads (each session's rows share them)
    const double win_kv = ns * (double)(WIN + rps - 1) * HD * 2;         // unique window ring rows
    const double comp_kv = ratio ? (double)M * std::min<double>(TKN, klen) * 288 : 0;   // top-512 compressed rows per query row; UPPER BOUND (rows of one session may overlap)
    auto S_ = [&](const char* n, int a, int b, int calls, double bw, double bkv, const char* src, const char* note = "") { ch.ser.push_back({n, a, b, calls, bw, bkv, src, note}); };
    if (cls == 0)
    {
        S_("proj", 0, 1, L_MAIN, wq_a + wq_b + wkv, 0, CALLS_ATTN, "q_a + norm + q_b + rope + kv + norm + rope + ring write; same weight shape at every layer");
        S_("core_r0", 1, 2, 2, 0, win_kv, "layers 0,1 (ratio 0): ds41_ops.h:144 loop, compress_ratio(l) == 0", "window 128 + sink only");
        S_("out", 2, 3, L_MAIN, wo_a + wo_b, 0, CALLS_ATTN, "inverse rope + grouped o_a + o_b; same weight shape at every layer");
    }
    if (cls == 2)
    {
        S_("comp_row", 1, 2, 3, cw, 0, CALLS_CA, "ratio-2 compressor: two bf16 [512][5120] projections + pool + norm + rope + fp4 row (layers 2, 8, 14)");
        S_("ica_row_r2", 2, 3, 3, iwk, 0, CALLS_CA, "index-K row from the compressor's post-norm latent");
        S_("ca_row_r2", 1, 3, 3, cw + iwk, 0, CALLS_CA, "ST_CA_ROW at a ratio-2 source = comp_row + ica_row");
        S_("idx_klen", 3, 4, 0, 0, 0, CALLS_IDX, "part of index_topk");
        S_("idx_wqb", 4, 5, 0, iwqb, 0, CALLS_IDX, "part of index_topk");
        S_("idx_wproj", 5, 6, 0, wproj_b, 0, CALLS_IDX, "part of index_topk");
        S_("idx_topk_dense_r2", 6, 7, 0, 0, ica_scan, CALLS_IDX, "part of index_topk: dense logits over the whole capacity grid + top-512");
        S_("index_topk_dense_r2", 3, 7, 3, iwqb + wproj_b, ica_scan, CALLS_IDX, "ST_INDEX_TOPK at sources 2, 8, 14");
        S_("core_r2", 7, 8, 18, 0, win_kv + comp_kv, "layers 2..19 (ratio 2): 18 calls", "window 128 + sink + top-512 of the source's packed rows");
        S_("l2_chain", 0, (int)ch.st.size(), 0, 0, 0, CALLS_ATTN, "whole chain proj..core of a ratio-2 source; no call count (diagnostic)");
    }
    if (cls == 20)
    {
        S_("cmp_gemv", 1, 2, 0, cmpw, 0, CALLS_CA, "part of ca_row_r1: bf16 [512][5120] gemv");
        S_("ca_row_r1_k", 2, 3, 0, 0, 0, CALLS_CA, "part of ca_row_r1: norm + rope + fp4 row");
        S_("ica_row_r1", 3, 4, 0, iwk, 0, CALLS_CA, "part of ca_row_r1");
        S_("ca_row_r1", 1, 4, 1, cmpw + iwk, 0, CALLS_CA, "ST_CA_ROW at layer 20");
        S_("idx_klen_l20", 4, 5, 0, 0, 0, CALLS_IDX, "part of index_topk");
        S_("idx_wqb_l20", 5, 6, 0, iwqb, 0, CALLS_IDX, "part of index_topk");
        S_("idx_wproj_l20", 6, 7, 0, wproj_b, 0, CALLS_IDX, "part of index_topk");
        S_("idx_topk_dense_r1", 7, 8, 0, 0, ica_scan, CALLS_IDX, "part of index_topk: dense logits (cand_src=1: the tiled call also makes the blocks) + top-512");
        S_("index_topk_dense_r1", 4, 8, 1, iwqb + wproj_b, ica_scan, CALLS_IDX, "ST_INDEX_TOPK at layer 20");
        S_("cand_blocks", 8, 9, 1, 0, 0, CALLS_IDX, "ST_CAND_BLOCKS at layer 20: bytes not modelled (reads the scratch logits/block scores)");
        S_("core_r1_l20list", 9, 10, 4, 0, win_kv + comp_kv, "layers 20..23 read the layer-20 dense list", "window + top-512 of the layer-20 packed ca rows");
        S_("l20_chain", 0, (int)ch.st.size(), 0, 0, 0, CALLS_ATTN, "whole chain proj..core at layer 20; diagnostic");
    }
    if (cls == 24)
    {
        S_("idx_klen_c", 1, 2, 0, 0, 0, CALLS_IDX, "part of index_topk");
        S_("idx_wqb_c", 2, 3, 0, iwqb, 0, CALLS_IDX, "part of index_topk");
        S_("idx_wproj_c", 3, 4, 0, wproj_b, 0, CALLS_IDX, "part of index_topk");
        S_("idx_topk_cand", 4, 5, 0, 0, 0, CALLS_IDX, "part of index_topk: logits over the layer-20 candidate blocks (bytes not modelled)");
        S_("index_topk_cand", 1, 5, 4, iwqb + wproj_b, 0, CALLS_IDX, "ST_INDEX_TOPK at layers 24, 28, 32, 36");
        S_("core_r1_candlist", 5, 6, 16, 0, win_kv + comp_kv, "layers 24..39 read a candidate-path list", "window + top-512 of the layer-20 packed ca rows");
        S_("l24_chain", 0, (int)ch.st.size(), 0, 0, 0, CALLS_ATTN, "whole chain proj..core at a candidate layer; diagnostic");
    }
    return ch;
}

struct Res { double med, mn, mean; bool ok; std::string bad; long unwritten; };
void flush_l2(cudaStream_t st) { k_stream<<<g.sgrid, 256, 0, st>>>(reinterpret_cast<const uint4*>(g.flush), FLUSH_BYTES / 16, g.sink); }

Res time_series(const Chain& ch, const Series& sr, const Hash& ref, int warm, int reps, cudaStream_t st, cudaEvent_t e0, cudaEvent_t e1)
{
    std::vector<double> t;
    Res res{}; res.ok = true;
    const int ns = (int)ch.st.size();
    for (int r = 0; r < warm + reps; ++r)
    {
        flush_l2(st);
        run_range(ch, 0, sr.a, st);
        for (int s = sr.a; s < ns; ++s) poison_stage(ch, s, st);
        CK(cudaEventRecord(e0, st));
        run_range(ch, sr.a, sr.b, st);
        CK(cudaEventRecord(e1, st));
        if (r == warm + reps - 1) run_range(ch, sr.b, ns, st);
        CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (r >= warm) t.push_back(ms * 1000.0);
        if (r == warm + reps - 1)
        {
            const Hash h = hash_all(ch);
            for (size_t i = 0; i < h.h.size(); ++i)
            {
                if (h.h[i] != ref.h[i]) { res.ok = false; res.bad += (res.bad.empty() ? "" : ",") + wit_name(ch, i); }
                res.unwritten += h.unwritten[i];
            }
            if (res.unwritten) res.ok = false;
        }
    }
    std::sort(t.begin(), t.end());
    double sum = 0; for (double v : t) sum += v;
    res.med = t[t.size() / 2]; res.mn = t[0]; res.mean = sum / t.size();
    return res;
}
double median_of(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

void peaks(uint8_t* dst, const char* tag, cudaStream_t st, cudaEvent_t e0, cudaEvent_t e1)
{
    std::vector<double> tc, tr;
    for (int r = 0; r < 14; ++r)
    {
        CK(cudaEventRecord(e0, st)); CK(cudaMemcpyAsync(dst, g.flush, FLUSH_BYTES, cudaMemcpyDeviceToDevice, st)); CK(cudaEventRecord(e1, st)); CK(cudaStreamSynchronize(st));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (r >= 3) tc.push_back(ms * 1000.0);
    }
    for (int r = 0; r < 14; ++r)
    {
        CK(cudaEventRecord(e0, st)); k_stream<<<g.sgrid, 256, 0, st>>>(reinterpret_cast<const uint4*>(g.flush), FLUSH_BYTES / 16, g.sink); CK(cudaEventRecord(e1, st)); CK(cudaStreamSynchronize(st));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (r >= 3) tr.push_back(ms * 1000.0);
    }
    const double c = median_of(tc), rd = median_of(tr), n = static_cast<double>(FLUSH_BYTES);
    printf("{\"lane\":\"attnsplit\",\"op\":\"d2d_copy\",\"tag\":\"%s\",\"bytes\":%zu,\"us\":%.1f,\"gbs_copied\":%.1f,\"gbs_traffic\":%.1f,\"reps\":11,\"note\":\"gbs_copied = bytes/t, gbs_traffic = 2*bytes/t (read + write)\"}\n",
           tag, FLUSH_BYTES, c, n / c / 1e3, 2 * n / c / 1e3);
    printf("{\"lane\":\"attnsplit\",\"op\":\"read_stream\",\"tag\":\"%s\",\"bytes\":%zu,\"us\":%.1f,\"gbs\":%.1f,\"reps\":11,\"note\":\"read-only 16-byte-load stream kernel over the same buffer; the ceiling for a read-bound kernel\"}\n",
           tag, FLUSH_BYTES, rd, n / rd / 1e3);
    fflush(stdout);
}

void build_model()
{
    cudaStream_t st = 0;
    const size_t gq_a = gv_bytes(QA, HIDN), gq_b = gv_bytes(HEADS * HD, QA), gkv = gv_bytes(HD, HIDN), go_a = gv_bytes(OLORA, 4096, G), go_b = gv_bytes(HIDN, G * OLORA);
    ds41a::Weights W{};
    W.wq_a.gv = gv_mat(gq_a, 1); W.wq_a.N = QA; W.wq_a.K = HIDN;
    W.wq_b.gv = gv_mat(gq_b, 2); W.wq_b.N = HEADS * HD; W.wq_b.K = QA;
    W.wkv.gv = gv_mat(gkv, 3); W.wkv.N = HD; W.wkv.K = HIDN;
    W.wo_a.gv = gv_mat(go_a, 4); W.wo_a.N = OLORA; W.wo_a.K = 4096;
    W.wo_b.gv = gv_mat(go_b, 5); W.wo_b.N = HIDN; W.wo_b.K = G * OLORA;
    W.q_norm = bf_vec_pos(QA, 0.5f, 1.5f, 11); W.kv_norm = bf_vec_pos(HD, 0.5f, 1.5f, 12);
    { std::vector<float> s(HEADS); for (int h = 0; h < HEADS; ++h) s[h] = -1.0f + 2.0f * h / (HEADS - 1); W.sink = to_dev(s); }
    float inv_w[32], inv_c[32]; ds41_attn_inv_freqs(0, inv_w); ds41_attn_inv_freqs(1, inv_c);
    const float* d_inv_w = to_dev(std::vector<float>(inv_w, inv_w + 32)); const float* d_inv_c = to_dev(std::vector<float>(inv_c, inv_c + 32));
    g.W0 = W; g.W0.inv = d_inv_w;
    g.W1 = W; g.W1.inv = d_inv_c;
    g.W2 = W; g.W2.inv = d_inv_c;
    g.W2.c_wkv = bf_mat((size_t)HD * HIDN, 0.03f, 21); g.W2.c_wgate = bf_mat((size_t)HD * HIDN, 0.03f, 22); g.W2.c_norm = bf_vec_pos(HD, 0.5f, 1.5f, 13);
    // stock bind: ds41_attn_set_gv(...) + settings, then GV_QUAD repack of q_b / o_a / o_b records (in place, outside any timing)
    ds41_attn_set_gv(1, 0, 0, ds41a::gv_xfrag_launch, ds41a::gv_gemv_launch_q);
    ds41_attn_set_tc(1); ds41_attn_set_tc_mfast(2); ds41_attn_set_tc_tile(1); ds41_attn_set_comp_fp8(0); ds41_attn_set_src_fast(3); ds41_attn_set_cpasync(1); ds41_attn_set_split(0);
    CK(cudaDeviceSynchronize());
    const int nq = ds41a::gv_quad_bind(W.wq_b, 1, 0, 0, st) + ds41a::gv_quad_bind(W.wo_a, G, 0, 0, st) + ds41a::gv_quad_bind(W.wo_b, 1, 0, 0, st);
    CK(cudaDeviceSynchronize());
    g.W0.wq_a = g.W1.wq_a = g.W2.wq_a = W.wq_a;
    if (nq != 3) { printf("{\"error\":\"GV_QUAD repack count %d != 3\"}\n", nq); exit(2); }
    if (!ds41_attn_gv_ready(&g.W2, 1)) { printf("{\"error\":\"decode GEMV not ready\"}\n"); exit(2); }
    // scratch (one, shared by all layers, carved for rowcap = PW = 4096 rows as the stock first bind does)
    g.attn_scratch_bytes = ds41_attn_scratch_bytes(RCAP, HEADS);
    CK(cudaMalloc(reinterpret_cast<void**>(&g.scr_base), g.attn_scratch_bytes)); CK(cudaMemset(g.scr_base, 0, g.attn_scratch_bytes));
    g.S.carve(g.scr_base, RCAP, HEADS);
    g.y = bf_mat((size_t)MX * HIDN, 1.0f, 31); g.out = dmal<bf16>((size_t)MX * HIDN);
    g.pos = dmal<int>(64); g.sess = dmal<int>(64);
    // caches: window ring RING_ROWS per slot; ratio-2 source packed compq COMP_ROWS per slot, compressor state kvs/scs
    ds41a::Cache* cs[4] = {&g.C0, &g.C2, &g.C20, &g.C24};
    for (int i = 0; i < 4; ++i)
    {
        ds41a::Cache& C = *cs[i];
        C = ds41a::Cache{};
        C.ring_rows = RING_ROWS;
        C.ring = bf_mat((size_t)NSLOT * RING_ROWS * HD, 1.5f, 40 + i);
    }
    g.C2.comp_ss = (size_t)COMP_ROWS * ds41a::CQ_ROW;
    g.C2.compq = dmal<uint8_t>((size_t)NSLOT * g.C2.comp_ss);
    k_fill_rows<<<g_nsm * 4, 256>>>(g.C2.compq, (size_t)NSLOT * g.C2.comp_ss, 288, 256, 0x30, 16, 51);
    g.C2.kvs = dmal<float>((size_t)NSLOT * 2 * HD); g.C2.scs = dmal<float>((size_t)NSLOT * 2 * HD);
    k_fill_f32<<<g_nsm, 256>>>(g.C2.kvs, (size_t)NSLOT * 2 * HD, 1.0f, 52); k_fill_f32<<<g_nsm, 256>>>(g.C2.scs, (size_t)NSLOT * 2 * HD, 1.0f, 53);
    g.C2.src_q = 1; g.C2.src_ss = g.C2.comp_ss;                 // the source reads its own packed rows
    // indexer state
    g.rows_kv[0] = g.rows_kv[1] = g.rows_kv[2] = ROWS_KV2; g.rows_kv[3] = ROWS_KV3;
    for (int k = 0; k < 4; ++k)
    {
        const size_t n = (size_t)NSLOT * g.rows_kv[k] * 68;
        g.ica[k] = dmal<uint8_t>(n);
        for (int sl = 0; sl < NSLOT; ++sl)      // page layout (ds41_idx.cuh ica_map): rows_kv * 64 B of codes, then rows_kv * 4 B of scale bytes
        {
            uint8_t* page = g.ica[k] + (size_t)sl * g.rows_kv[k] * 68;
            k_fill_rows<<<g_nsm * 4, 256>>>(page, (size_t)g.rows_kv[k] * 64, 64, 64, 0, 1, 60 + k);
            k_fill_rows<<<g_nsm, 256>>>(page + (size_t)g.rows_kv[k] * 64, (size_t)g.rows_kv[k] * 4, 4, 0, 120, 4, 64 + k);
        }
    }
    {
        const size_t n = (size_t)NSLOT * ROWS_KV3 * 288;
        g.ca3 = dmal<uint8_t>(n);
        k_fill_rows<<<g_nsm * 4, 256>>>(g.ca3, n, 288, 256, 0x30, 16, 70);
    }
    for (int i : {2, 3}) { ds41a::Cache& C = i == 2 ? g.C20 : g.C24; C.src_q = 1; C.src_ss = (size_t)ROWS_KV3 * 288; C.compq = nullptr; }
    g.idx_wk = bf_mat((size_t)IDX_DIM * HD, 0.05f, 81); g.idx_knorm = bf_vec_pos(IDX_DIM, 0.5f, 1.5f, 82);
    g.wproj = bf_mat((size_t)IDX_HEADS * HIDN, 0.03f, 83);
    g.cmp_wkv = bf_mat((size_t)HD * HIDN, 0.03f, 84); g.cmp_norm = bf_vec_pos(HD, 0.5f, 1.5f, 85);
    g.wqb_gv = gv_mat(gv_bytes(IDX_WQB_OUT, QA), 86);
    g.klen = dmal<int>(RCAP); g.q_b = dmal<bf16>((size_t)RCAP * IDX_WQB_OUT); g.iwproj = dmal<bf16>((size_t)RCAP * IDX_HEADS);
    g.y512 = dmal<bf16>((size_t)RCAP * 512); g.pre = dmal<bf16>((size_t)RCAP * 512); g.topk = dmal<int32_t>((size_t)RCAP * TKN);
    CK(cudaMemset(g.topk, 0xFF, (size_t)RCAP * TKN * 4));
    g.scratch_bytes = ds41_idx_scratch_bytes(RCAP, MAXROWS);
    CK(cudaMalloc(&g.scratch, g.scratch_bytes)); CK(cudaMemset(g.scratch, 0, g.scratch_bytes));
    g.sscr_bytes = ds41_idx_scratch_bytes_flat(SESS_CAP, MAXROWS);
    for (int j = 0; j < NSLOT; ++j) { CK(cudaMalloc(&g.sscr[j], g.sscr_bytes)); CK(cudaMemset(g.sscr[j], 0, g.sscr_bytes)); }
    CK(cudaMalloc(&g.flush, FLUSH_BYTES)); CK(cudaMalloc(&g.sink, 64));
    k_fill_rows<<<g_nsm * 4, 256>>>(g.flush, FLUSH_BYTES, 256, 256, 0, 1, 77);
    { int nb = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_stream, 256, 0)); g.sgrid = g_nsm * nb; }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}

void set_cfg(const Cfg& c)
{
    std::vector<int> p(64, 0), s(64, 0);
    for (int r = 0; r < c.M; ++r) { p[r] = c.pos[r]; s[r] = c.slot[r]; }
    CK(cudaMemcpy(g.pos, p.data(), 64 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.sess, s.data(), 64 * 4, cudaMemcpyHostToDevice));
    g.S.sess = g.sess; g.S.ns = c.ns;
}
}   // namespace

int main(int argc, char** argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 51, warm = 8;
    if (argc > 2) { g_skip_on = 1; g_skip_name = argv[2]; }
    if (reps < 20) { printf("{\"error\":\"reps must be >= 20\"}\n"); return 2; }
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    CK(cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, 0));
    build_model();
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    uint8_t* dst; CK(cudaMalloc(&dst, FLUSH_BYTES));
    size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
    printf("{\"lane\":\"attnsplit\",\"op\":\"header\",\"device\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"l2_mb\":%zu,\"mem_total_mb\":%zu,\"mem_free_mb_after_alloc\":%zu,"
           "\"config\":{\"HID\":%d,\"HEADS_PER_RANK\":%d,\"O_GROUPS\":%d,\"WIN\":%d,\"TOPK\":%d,\"CAP\":%d,\"RCAP\":%d,\"RING_ROWS\":%d,\"COMP_ROWS\":%d,\"ROWS_KV_R2\":%d,\"ROWS_KV_R1\":%d,\"SESS_CAP\":%d,"
           "\"GV\":1,\"GV_QUAD\":1,\"ATTN_TC\":1,\"TC_MF\":2,\"TC_TILE\":1,\"COMP_FP8\":0,\"COMP_FP4\":1,\"SRC_FAST\":3,\"ATTN_CPA\":1,\"IDX_PIPE_B\":0,\"IDX_PIPE_QR\":0,\"IDX_Q_TRIG\":0,\"IDX_REUSE\":0,\"IDX_MAXK\":0},"
           "\"attn_scratch_bytes\":%zu,\"idx_scratch_bytes\":%zu,\"idx_sscr_bytes\":%zu,\"reps\":%d,\"warmup\":%d,\"skip_stage\":\"%s\","
           "\"method\":\"L2 flush, untimed prefix, poison 0xFF witnesses of timed+later stages, events around the range, stream launches (not a graph)\"}\n",
           pr.name, pr.major, pr.minor, g_nsm, static_cast<size_t>(pr.l2CacheSize) >> 20, tot >> 20, fr >> 20, HIDN, HEADS, G, WIN, TKN, CAP, RCAP, RING_ROWS, COMP_ROWS, ROWS_KV2, ROWS_KV3, SESS_CAP,
           g.attn_scratch_bytes, g.scratch_bytes, g.sscr_bytes, reps, warm, g_skip_on ? g_skip_name.c_str() : "");
    fflush(stdout);
    peaks(dst, "start", st, e0, e1);

    const int Ms[4] = {1, 3, 6, 8}, ctxs[3] = {1024, 32768, 131072};
    for (int mi = 0; mi < 4; ++mi)
    for (int ci = 0; ci < 3; ++ci)
    {
        const Cfg c = make_cfg(Ms[mi], ctxs[ci]);
        set_cfg(c);
        int pairs = 0; for (int r = 0; r < c.M; ++r) pairs += c.pos[r] & 1;
        printf("{\"lane\":\"attnsplit\",\"op\":\"config\",\"M\":%d,\"ctx\":%d,\"sessions\":%d,\"rows_per_session\":%d,\"first_pos\":%d,\"last_pos\":%d,\"ratio2_pairs_completed\":%d,"
               "\"note\":\"rows at consecutive positions ctx-rps..ctx-1 per session, session j in S2 slot j; compressor pending-half state (kvs/scs) is not poisoned\"}\n",
               c.M, c.ctx, c.ns, c.rps, c.pos[0], c.pos[c.M - 1], pairs);
        double med[64]; std::vector<std::string> mname;
        std::vector<std::pair<std::string, double>> meds;
        for (int cls : {0, 2, 20, 24})
        {
            const Chain ch = build_chain(cls, c);
            // reference: two clean full chains with every witness poisoned first (also proves every witness is fully rewritten)
            g_skip_on = 0;
            std::string saved = g_skip_name;
            for (size_t s = 0; s < ch.st.size(); ++s) poison_stage(ch, (int)s, st);
            run_range(ch, 0, (int)ch.st.size(), st); CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
            const Hash ref = hash_all(ch);
            for (size_t s = 0; s < ch.st.size(); ++s) poison_stage(ch, (int)s, st);
            run_range(ch, 0, (int)ch.st.size(), st); CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
            const Hash ref2 = hash_all(ch);
            long un = 0; std::string unnames;
            for (size_t i = 0; i < ref.h.size(); ++i) if (ref.unwritten[i]) { un += ref.unwritten[i]; unnames += (unnames.empty() ? "" : ",") + wit_name(ch, i); }
            printf("{\"lane\":\"attnsplit\",\"op\":\"ref\",\"class\":\"%s\",\"M\":%d,\"ctx\":%d,\"witnesses\":%zu,\"deterministic\":%s,\"ref_unwritten_words\":%ld,\"unwritten_in\":\"%s\"}\n",
                   ch.name.c_str(), c.M, c.ctx, ref.h.size(), ref.h == ref2.h ? "true" : "false", un, unnames.c_str());
            fflush(stdout);
            g_skip_on = !saved.empty();
            for (const Series& sr : ch.ser)
            {
                const Res r = time_series(ch, sr, ref, warm, reps, st, e0, e1);
                const double bytes = sr.bw + sr.bkv;
                printf("{\"lane\":\"attnsplit\",\"op\":\"%s\",\"class\":\"%s\",\"M\":%d,\"ctx\":%d,\"sessions\":%d,\"us_per_call\":%.2f,\"us_min\":%.2f,\"us_mean\":%.2f,\"reps\":%d,"
                       "\"bytes_weights_est\":%.0f,\"bytes_kv_est\":%.0f,\"bytes_read_est\":%.0f,\"gbs_est\":%.1f,\"calls_per_step\":%d,\"ms_per_step\":%.3f,\"calls_src\":\"%s\",\"note\":\"%s\","
                       "\"poison_ok\":%s,\"bad_witness\":\"%s\",\"unwritten_words\":%ld}\n",
                       sr.name.c_str(), ch.name.c_str(), c.M, c.ctx, c.ns, r.med, r.mn, r.mean, reps, sr.bw, sr.bkv, bytes, bytes > 0 ? bytes / r.med / 1e3 : 0.0, sr.calls, r.med * sr.calls / 1e3,
                       sr.calls_src, sr.note, r.ok ? "true" : "false", r.bad.c_str(), r.unwritten);
                fflush(stdout);
                meds.push_back({sr.name, r.med});
            }
        }
        auto get = [&](const char* n) { for (auto& p : meds) if (p.first == n) return p.second; return 0.0; };
        const double proj = get("proj"), out = get("out"), c0 = get("core_r0"), c2 = get("core_r2"), c1a = get("core_r1_l20list"), c1b = get("core_r1_candlist");
        const double ca2 = get("ca_row_r2"), ca1 = get("ca_row_r1"), ix2 = get("index_topk_dense_r2"), ix1 = get("index_topk_dense_r1"), cb = get("cand_blocks"), ixc = get("index_topk_cand");
        const double sum = proj * 40 + out * 40 + c0 * 2 + c2 * 18 + c1a * 4 + c1b * 16 + ca2 * 3 + ca1 + ix2 * 3 + ix1 + cb + ixc * 4;
        printf("{\"lane\":\"attnsplit\",\"op\":\"attn_sum\",\"M\":%d,\"ctx\":%d,\"us_proj\":%.2f,\"us_out\":%.2f,\"us_core_r0\":%.2f,\"us_core_r2\":%.2f,\"us_core_r1_l20list\":%.2f,\"us_core_r1_candlist\":%.2f,"
               "\"us_ca_row_r2\":%.2f,\"us_ca_row_r1\":%.2f,\"us_index_dense_r2\":%.2f,\"us_index_dense_r1\":%.2f,\"us_cand_blocks\":%.2f,\"us_index_cand\":%.2f,"
               "\"ms_per_step_sum\":%.3f,\"mhc_not_included\":true,\"calibration_target_ms\":11.2,\"note\":\"sum of the isolated medians x stock call counts (cold weights, op-to-op overlap excluded); ST_HC_ATTN and ST_CONSUME_ATTN are not in this sum\"}\n",
               c.M, c.ctx, proj, out, c0, c2, c1a, c1b, ca2, ca1, ix2, ix1, cb, ixc, sum / 1e3);
        fflush(stdout);
        (void)med; (void)mname;
    }
    peaks(dst, "end", st, e0, e1);
    printf("{\"lane\":\"attnsplit\",\"op\":\"done\"}\n");
    return 0;
}
