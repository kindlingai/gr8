// DeepSeek-V4.1-Flash-1017 attention A in the decode step: the op-table functions of runtime/ds41_ops.h for ST_ATTN_PROJ,
// ST_ATTN_CORE, ST_ATTN_OUT (and the Fn the indexer's ST_CA_ROW calls at the ratio-2 sources), plus the per-layer binder.
// Included by runtime/ds41_step.cuh inside namespace g6rt (Model, Layer, dmalloc, die, lname, MX, Find are visible); the attention
// kernels stay in the attention TU (ds41_attn.cu), reached through its extern "C" entry points; only the decode GEMV kernels (ds41_attn_gv.cuh) are here.
//
// Contract (docs/DEEPSEEK-STEP.md)
//   input   Model::hc_y  bf16 [rows][5120]  (hc_attn(i) output, already normed)
//   output  Model::sc.part[0]  bf16 [rows][5120]  this rank's o_b partial; the driver publishes it
//   rows    positions Model::pos[0] .. pos[0]+rows-1, read on the device (graph-safe); decode form only (window ring + top-k list)
//   state   Layer::ds41_attn -> Ds41AttnLayer, made by ds41_attn_bind (Table::bind_attn, once, rows = capacity)
// All 40 layers bind. Compressed rows: layers 2..19 read the C.comp of their source (2, 8, 14: bf16, fp4-valued, written by comp_row);
// layers 20..39 (ratio 1) read LayerState::comp_src, which the indexer sets to its bf16 shadow of the packed fp4 rows (there is no paged 288 B reader yet).
//
// For the indexer (ds41_idx_ops.cuh, ST_CA_ROW): Ds41AttnLayer::S.qr = q_a-normed bf16 [rows][1280] (written by ST_ATTN_PROJ); S.cnorm = post-norm
// pre-rope compressed rows completed by the last comp_row, bf16 [(rows+1)/2][512]; W.inv = the layer's rope table. The indexer sets
// Ds41AttnLayer::topk (int32 [MX][ldtk], ascending, -1 tail; a reuse layer points at its source's list) and ldtk in its own bind.
// ST_ATTN_CORE answers NOTSUP on a compressed layer whose topk is still null.
#pragma once
#if G6_MODEL_DS41
#include "ds41_attn_types.h"
#include "ds41_attn_plan.h"
#include "ds41_attn_gv.cuh"   // the decode GEMV kernels live in THIS (rank) TU: densev.cuh is already here; ds41_attn.cu reaches them by hooks

extern "C"
{
void ds41_attn_proj(const ds41a::Weights*, ds41a::Cache*, const ds41a::Scratch*, const void* y, int n, int start, int heads, void* st,
                    const int* dstart);
void ds41_attn_proj_q(const ds41a::Weights*, ds41a::Cache*, const ds41a::Scratch*, const void* y,
                      const uint8_t* codes, const uint8_t* scales, int n, int start, int heads, void* st, const int* dstart);
void ds41_attn_comp_row(const ds41a::Weights*, ds41a::Cache*, const ds41a::Scratch*, const void* y, int n, int start, void* st,
                        const int* dstart);
void ds41_attn_comp_commit(const ds41a::Cache*, const ds41a::Scratch*, const int* dq, const int* dacc, void* st);
void ds41_attn_comp_commit_ns(const ds41a::Cache*, const ds41a::Scratch*, const int* dq, const int* dacc, const int* drow0, const int* dslot, int ns,
                              void* st);
void ds41_attn_core(const ds41a::Weights*, const ds41a::Cache*, const void* comp_src, const ds41a::Scratch*, int n, int start, int heads,
                    const int* topk, int ldtk, int TK, void* st, const int* dstart);
void ds41_attn_out(const ds41a::Weights*, const ds41a::Scratch*, int n, int start, int heads, void* out, void* st, const int* dstart, int direct_c2);
void ds41_attn_inv_freqs(int compressed, float* out32);
void ds41_attn_set_gv(int on, int tpc, int ks, ds41a::XfragFn fn_xfrag, ds41a::GemvFn fn_gemv);
void ds41_attn_set_src_fast(int bits);
void ds41_attn_set_tc(int on);
void ds41_attn_set_tc_mfast(int on);
void ds41_attn_set_comp_fp8(int on);
void ds41_attn_comp_prepack(const ds41a::bf16* src, uint8_t* codes, uint8_t* scales, void* st);
void ds41_attn_set_cpasync(int on);
void ds41_attn_set_split(int on);
void ds41_attn_set_win_floor(int pos);
int ds41_attn_gv_ready(const ds41a::Weights*, int rows);
int ds41_attn_project(const ds41a::Fp8Mat*, const void* x, int rows, void* y, int ldy, uint8_t* codes, uint8_t* scales, uint32_t* xf, void* st);
}

// GR8_DS_SRC_FAST (default 0 = the generic launches): bit 1 = the ratio-2 compressor GEMV (bit-identical to k_f32gemm2), bit 2 = the indexer wq_b through the MXFP8 GEMV
// (summation order differs from k_gemm_mx; read by ds41_idx_ops.cuh). Set once by the first ds41_attn_bind.
inline int g_ds41_src_fast = 0;

// GR8_DS_ATTN_PW (default 0 = off): prefill row capacity PW of the attention ops, separate from MX (the decode / verify width). PW > MX gives every
// layer a window ring of WIN + PW - 1 slots and one scratch shared by all layers (sized for PW rows; the ops of a prefill run one after the other
// on one stream, so no two layers use it at once), and lifts the per-op row limit from MX to PW. The caller's buffers (Model::hc_y, Model::sc.part[0],
// the indexer's top-k list) must hold PW rows too. PW <= MX or unset: nothing changes (per-layer scratch, ring WIN + MX - 1, limit MX). Set by the first
// ds41_attn_bind from the knob, or earlier by ds41_attn_set_prefill_rows (which wins).
inline int g_ds41_attn_pw = 0;
inline int g_ds41_attn_split = 0;  // GR8_DS_ATTN_SPLIT=1, default off
inline void ds41_attn_set_prefill_rows(int pw) { g_ds41_attn_pw = pw > 0 ? pw : 0; }

namespace ds41a_ops
{
struct LayerState
{
    ds41a::Weights W{};
    ds41a::Cache C{};          // C.comp: own rows at a ratio-2 source (2, 8, 14), null elsewhere (readers use the source's)
    ds41a::Scratch S{};
    int l = 0, heads = 0, ratio = 0, src = -1;   // src: the layer whose compressed rows this layer reads (geom::kv_source), -1 at ratio 0
    int src_pos = -1;                            // its position in Model::layers, -1 when the subset has no such layer
    int rowcap = 0;                              // most rows one op takes: max(MX, GR8_DS_ATTN_PW); the ring and the scratch are sized for it
    size_t comp_rows = 0;                        // rows allocated in C.comp (capacity / 2 + 2); 0 where C.comp is null
    const int* topk = nullptr;                   // the indexer sets: this layer's top-512 list [MX][ldtk]
    int ldtk = geom::IDX_TOPK;
    const ds41a::bf16* comp_src = nullptr;       // the indexer sets (layers 20..39): bf16 shadow rows ST_ATTN_CORE reads instead of the source layer's C.comp
    __nv_bfloat16* ring_pf = nullptr;            // SESS > 1 with PW > WIN: one shared prefill ring (WIN + PW - 1 rows) across all layers & slots
    int ring_pf_rows = 0;
};
}   // namespace ds41a_ops
using Ds41AttnLayer = ds41a_ops::LayerState;

inline Ds41AttnLayer* ds41_attn_state(Model& m, int i) { return (Ds41AttnLayer*)m.layers[i].ds41_attn; }
// SESS > 1: the sessions of the step being launched (Model::ds_ns) into the layer's Scratch; rows must split evenly. 0 = ok.
inline int ds41_attn_sessions(const Model& m, Ds41AttnLayer* a, int rows)
{
    if (!a->S.sess) return 0;
    if (m.ds41_pack) {
        if (ds41::pack::attn_error(*m.ds41_pack, rows, a->rowcap, m.ds_slots)) return 1;
        a->S.ns = 1; // per-row slot maps remain in S.sess; only comp_row needs an explicit segment loop
        return 0;
    }
    if (m.ds_ns < 1 || rows % m.ds_ns) return 1;
    a->S.ns = m.ds_ns;
    return 0;
}
// Rebase only row-strided scratch arrays for one segment's ratio-2 compressor. The
// fragment tables are shared sequential scratch; the cache and slot-indexed slabs are not rebased.
inline ds41a::Scratch ds41_attn_slice(ds41a::Scratch s, int row0, int heads) {
    const size_t r = size_t(row0), g = size_t(heads / ds41a::HPG);
    s.yq += r * ds41a::HID; s.ys += r * ds41a::HID / 32;
    s.qrq += r * ds41a::QA; s.qrs += r * ds41a::QA / 32;
    s.oq += r * heads * ds41a::HD; s.os += r * heads * ds41a::HD / 32;
    s.oaq += r * g * ds41a::OLORA; s.oas += r * g * ds41a::OLORA / 32;
    s.qa += r * ds41a::QA; s.qr += r * ds41a::QA;
    s.q += r * heads * ds41a::HD; s.kv0 += r * ds41a::HD; s.kvrows += r * ds41a::HD;
    s.o += r * heads * ds41a::HD; s.oa += r * g * ds41a::OLORA;
    if (s.attn_split) s.attn_split += r * heads * 2 * (ds41a::HD + 2);
    s.ckv += r * ds41a::HD; s.csc += r * ds41a::HD; s.cnorm += r * ds41a::HD;
    if (s.sess) s.sess += row0;
    s.ns = 1;
    return s;
}

// The attention weight binding and its shape checks live in ds41_attn_plan.h (no CUDA calls: tests/ds41_attn_bind_test.cpp runs them on
// the planner's list). This is the Model's finder in the plan's tensor form.
inline auto ds41_plan_find(Model& m)
{
    return [&m](const std::string& n) {
        const TensorRef t = m.ds41_find(n);
        ds41a::PlanTensor p;
        p.p = t.p; p.nd = t.nd; p.bf16 = t.bf16; p.fp32 = t.fp32;
        for (int k = 0; k < 4; ++k) p.shape[k] = t.shape[k];
        return p;
    };
}
inline ds41a::Fp8Mat ds41_fp8(Model& m, int l, const char* nm, long long N, long long K)
{
    auto find = ds41_plan_find(m);
    ds41a::Planner<decltype(find)> P{find, l, std::string()};
    const ds41a::Fp8Mat w = P.fp8(nm, N, K);
    if (!P.err.empty()) die(P.err);
    return w;
}
// 1-D bf16 (or fp32 with want_f32) vector of a layer, shape-checked. Published helper: ds41_idx_ops.cuh and ds41_dspark_ops.cuh call it.
inline const __nv_bfloat16* ds41_vec(Model& m, int l, const char* nm, long long n, bool want_f32 = false)
{
    const std::string p = lname(l, nm);
    const TensorRef t = m.ds41_find(p);
    if (!t.p || t.nd != 1 || t.shape[0] != n || (want_f32 ? !t.fp32 : !t.bf16)) die("DS41 attention bind: bad tensor " + p);
    return (const __nv_bfloat16*)t.p;
}

// Engram, ST_ENGRAM's project hook: kv[rows][25600] = fp8gemm(act_quant(emb[rows][6144]), wkv), wkv = L001 / L014 .engram.wkv [25600][6144]
// (full on every rank). The dense MXFP8 GEMV of the attention projections: one pass over the weight for up to 8 rows, 8-row passes above.
// Bound with the layer (ds41_attn_bind at l = 1, 14) when the pack has the tensor; the scratch is carved there, so the hook allocates nothing.
struct Ds41EngramProj { ds41a::Fp8Mat W{}; uint8_t *codes = nullptr, *scales = nullptr; uint32_t* xf = nullptr; };
inline Ds41EngramProj g_ds41_eng[2];
inline void ds41_engram_bind(Model& m, int l)
{
    const int e = l == geom::ENGRAM_L0 ? 0 : 1;
    const std::string p = lname(l, "engram.wkv");
    const TensorRef c = m.ds41_find(p + ".mx.codes");
    if (!c.p) return;                                    // no Engram in this pack: the hook answers NOTSUP
    const long long K = geom::ENGRAM_DIM * geom::NGRAM_TABLES, N = c.nd == 2 ? c.shape[0] : 0;
    Ds41EngramProj& q = g_ds41_eng[e];
    q.W = ds41_fp8(m, l, "engram.wkv", N, K);
    if (!q.W.gv) die("DS41 Engram bind: " + p + ".mx.gv missing (the project hook needs the GEMV records)");
    q.codes = dmalloc<uint8_t>((size_t)8 * K);
    q.scales = dmalloc<uint8_t>((size_t)8 * K / 32);
    q.xf = dmalloc<uint32_t>((size_t)K * 4);
}
// EngramHooks::project (engine/engram/engram_dev.cuh): e = 0 (layer 1) or 1 (layer 14); 0 = ok.
inline int ds41_engram_project(void* model, int e, int rows, void* stream, const __nv_bfloat16* emb, __nv_bfloat16* kv)
{
    (void)model;
    if (e < 0 || e > 1 || rows < 1) return 1;
    const Ds41EngramProj& q = g_ds41_eng[e];
    if (!q.W.gv) return ds41::NOTSUP;
    return ds41_attn_project(&q.W, emb, rows, kv, q.W.N, q.codes, q.scales, q.xf, stream);
}

// The indexer's per-layer bind/prepare (engine/attn/ds41_idx_ops.cuh, included after this header in ds41_step.cuh): this file owns the table slots
// and calls them last, after the state they read is in Layer::ds41_attn. Absent from the tree -> the chain is a no-op.
#if __has_include("ds41_idx_ops.cuh")
inline int ds41_idx_bind(void*, int, int, void*);
inline int ds41_idx_prepare(void*, int, int, void*);
#define DS41_IDX_CHAIN 1
#endif

// The one scratch of the prefill build (GR8_DS_ATTN_PW > MX), carved by every layer. The first bind allocates it; a layer with another head count dies.
inline char* ds41_attn_shared_scratch(int rows, int heads)
{
    static char* base = nullptr;
    static int base_heads = 0;
    if (!base)
    {
        const size_t b = ds41a::Scratch::bytes(rows, heads);
        base = dmalloc<char>(b);
        base_heads = heads;
        // rows is the widest prefill chunk (PW); the ring that holds it is WIN + rows - 1. That ring is one shared ring for
        // every layer (ds41_attn_shared_prefill_ring below), not a per-layer ring.
        fprintf(stderr, "DS41 attention prefill rows: GR8_DS_ATTN_PW=%d (shared prefill ring %d rows, one shared scratch %.1f MiB)\n", rows, ds41a::WIN + rows - 1,
                (double)b / (1024.0 * 1024.0));
    }
    else if (base_heads != heads) die("DS41 attention bind: GR8_DS_ATTN_PW shares one scratch, but this layer has " + std::to_string(heads) + " heads, not " + std::to_string(base_heads));
    return base;
}

// One split-state buffer shared by all sequential prefill layers. Off => no allocation or change to Scratch::bytes.
inline float* ds41_attn_shared_split(int rows, int heads)
{
    static float* base = nullptr;
    static int base_rows = 0, base_heads = 0;
    if (!base) {
        base = dmalloc<float>(ds41a::Scratch::split_bytes(rows, heads) / sizeof(float));
        base_rows = rows; base_heads = heads;
    } else if (base_rows != rows || base_heads != heads) die("DS41 attention split scratch shape changed after bind");
    return base;
}

// Shared W-prefill window ring (WIN + PW - 1 rows) when SESS > 1 and PW > WIN: each layer keeps compact per-slot decode rings
// (WIN + MX - 1 = 175 rows/slot) and stages/commits the 128-token boundary through this shared prefill ring on wide chunks.
inline __nv_bfloat16* ds41_attn_shared_prefill_ring(int ring_rows)
{
    static __nv_bfloat16* ring = nullptr;
    static int base_rows = 0;
    if (!ring)
    {
        ring = dmalloc<__nv_bfloat16>((size_t)ring_rows * ds41a::HD);
        base_rows = ring_rows;
    }
    else if (base_rows != ring_rows) die("DS41 attention bind: shared prefill ring row mismatch");
    return ring;
}

__global__ inline void k_ds41_ring_stage(__nv_bfloat16* __restrict__ ring_pf, int r_pf,
                                         const __nv_bfloat16* __restrict__ ring_dec, int r_dec,
                                         const int* __restrict__ sess, const int* __restrict__ pos)
{
    const int p = pos[0] - (ds41a::WIN - 1) + (int)blockIdx.x;
    if (p < 0) return;
    const int slot = sess ? sess[0] : 0;
    ring_pf[(size_t)(p % r_pf) * ds41a::HD + threadIdx.x] =
        ring_dec[((size_t)slot * r_dec + (size_t)(p % r_dec)) * ds41a::HD + threadIdx.x];
}

__global__ inline void k_ds41_ring_commit(__nv_bfloat16* __restrict__ ring_dec, int r_dec,
                                          const __nv_bfloat16* __restrict__ kvrows, int n,
                                          const int* __restrict__ sess, const int* __restrict__ pos)
{
    const int row = n - ds41a::WIN + (int)blockIdx.x;
    if (row < 0) return;
    const int p = pos[0] + row;
    const int slot = sess ? sess[row] : 0;
    ring_dec[((size_t)slot * r_dec + (size_t)(p % r_dec)) * ds41a::HD + threadIdx.x] =
        kvrows[(size_t)row * ds41a::HD + threadIdx.x];
}

// Table::bind_attn (build, once per layer, rows = capacity). The one writer of the slot: binds this layer's attention, then
// the indexer's state.
inline int ds41_attn_bind(void* model, int i, int capacity, void* stream)
{
    Model& m = *(Model*)model;
    Layer& y = m.layers[i];
    const int l = y.l;
    const int ratio = geom::compress_ratio(l);
    {
        static bool gv_set = false;   // decode projections (<= 8 rows) go through the MXFP8 GEMV where the weight has .mx.gv records: default on, GR8_DS_ATTN_GV=0 turns it off; _TPC / _KS override the launch shape
        if (!gv_set)
        {
            gv_set = true;
            ds41_attn_set_gv(knob::ds_attn_gv(), (int)knob::ds_attn_gv_tpc(), (int)knob::ds_attn_gv_ks(), ds41a::gv_xfrag_launch, ds41a::gv_gemv_launch);
            const bool tc = knob::ds_attn_tc();   // 1 = the five projections at M >= 9 run on the block-scaled tensor-core GEMM (default 0)
            ds41_attn_set_tc(tc);
            if (tc) fprintf(stderr, "DS41 attention tensor-core GEMM: GR8_DS_ATTN_TC=1 (M >= 9)\n");
            ds41_attn_set_tc_mfast((int)knob::ds_attn_tc_mf());   // that GEMM's launch order: 0 = N tile fastest (default), 1 = M tile fastest, 2 = auto (M tile fastest only for a weight >= 100 MB)
            ds41_attn_set_comp_fp8(knob::ds_comp_fp8());   // default off; BF16 compressor weight prepack, lossy
            if (!g_ds41_attn_pw) ds41_attn_set_prefill_rows((int)knob::ds_attn_pw());
            g_ds41_src_fast = (int)knob::ds_src_fast();
            ds41_attn_set_src_fast(g_ds41_src_fast);
            const bool cpa = knob::ds_attn_cpa();
            ds41_attn_set_cpasync(cpa);
            g_ds41_attn_split = knob::ds_attn_split() ? 1 : 0;
            if (g_ds41_attn_split && !cpa) die("GR8_DS_ATTN_SPLIT=1 requires GR8_DS_ATTN_CPA=1");
            ds41_attn_set_split(g_ds41_attn_split);
            if (cpa) fprintf(stderr, "DS41 attention core: GR8_DS_ATTN_CPA=1 (cp.async K gather, k_attn_cpa)\n");
            if (g_ds41_src_fast) fprintf(stderr, "DS41 source-layer fast path: GR8_DS_SRC_FAST=%d (1 = compressor GEMV, 2 = idx wq_b GEMV)\n", g_ds41_src_fast);
        }
    }
    auto* a = new ds41a_ops::LayerState();
    a->l = l; a->ratio = ratio; a->src = ratio ? geom::kv_source(l) : -1;
    ds41a::Weights& W = a->W;
    // heads of this rank = q_b rows / 512 (the pack slices q_b, attn_sink, o_a and o_b by o group); the shape rules are ds41_attn_plan.h's
    ds41a::Plan plan;
    {
        auto find = ds41_plan_find(m);
        const std::string err = ds41a::plan_attn(find, l, geom::has_compressor_gate(l), plan);
        if (!err.empty()) die(err);
    }
    a->heads = plan.heads;
    W = plan.W;
    if (geom::has_compressor_gate(l))
    {
        if (knob::ds_comp_fp8())
        {
            auto prepack = [&](const ds41a::bf16* src, ds41a::Fp8Mat& dst) {
                dst.N = ds41a::HD; dst.K = ds41a::HID;
                uint8_t* codes = dmalloc<uint8_t>((size_t)dst.N * dst.K);
                uint8_t* scales = dmalloc<uint8_t>((size_t)dst.N * dst.K / 32);
                dst.w = codes; dst.s = scales;
                ds41_attn_comp_prepack(src, codes, scales, stream);
            };
            prepack(W.c_wkv, W.c_wkv8);
            prepack(W.c_wgate, W.c_wgate8);
            G6RT_CK(cudaStreamSynchronize((cudaStream_t)stream)); // bind must finish prepack before inference on another stream
        }
    }
    float inv[32];
    ds41_attn_inv_freqs(ratio != 0, inv);
    float* dinv = dmalloc<float>(32);
    G6RT_CK(cudaMemcpy(dinv, inv, sizeof inv, cudaMemcpyHostToDevice));
    W.inv = dinv;
    // window ring: 128 slots plus rowcap - 1 for the rows of a chunk written before the earliest row reads its window (verify chunk: MX; prefill: PW)
    // SESS > 1 (Model::ds_slots S2 slots): the ring, the compressed rows and the compressor state are one slab per slot (ds41a::Cache);
    // capacity is per slot. Scratch::sess = Model::sess tells the kernels each row's slot; ds_slots == 1 leaves it null (one session).
    const size_t ns = (size_t)m.ds_slots;
    a->rowcap = g_ds41_attn_pw > MX ? g_ds41_attn_pw : MX;
    if (m.ds41_draft) a->rowcap = MX;
    a->C.ring_rows = ds41a::WIN + a->rowcap - 1;
    if (ns > 1 && a->rowcap > ds41a::WIN)
    {
        a->ring_pf_rows = a->C.ring_rows;
        a->ring_pf = ds41_attn_shared_prefill_ring(a->ring_pf_rows);
        a->C.ring_rows = ds41a::WIN + MX - 1;
    }
    a->C.ring = dmalloc<__nv_bfloat16>(ns * a->C.ring_rows * ds41a::HD);
    // One boot line per process that names the ring geometry, so a log shows how the rings are wired: per-slot decode
    // ring rows, the slot count, the single shared prefill ring rows and the layer count.
    if (ns > 1 && a->ring_pf)
    {
        static bool rings_reported = false;
        if (!rings_reported)
        {
            rings_reported = true;
            fprintf(stderr, "# DS41 rings: per-slot decode ring %zu rows x %zu slots per layer, one shared prefill ring %zu rows, %d layers\n",
                    a->C.ring_rows, ns, a->ring_pf_rows, (int)m.nl);
        }
    }
    if (geom::has_compressor_gate(l))
    {
        a->comp_rows = (size_t)(capacity / 2 + 2);
        a->C.comp_ss = a->comp_rows * ds41a::HD;
        a->C.comp = dmalloc<__nv_bfloat16>(ns * a->C.comp_ss);             // bf16, fp4-valued; the packed 288 B form is a follow-up
        a->C.kvs = dmalloc<float>(ns * 2 * ds41a::HD);
        a->C.scs = dmalloc<float>(ns * 2 * ds41a::HD);
    }
    if (a->rowcap > MX) a->S.carve(ds41_attn_shared_scratch(a->rowcap, a->heads), a->rowcap, a->heads);
    else a->S.carve(dmalloc<char>(ds41a::Scratch::bytes(MX, a->heads)), MX, a->heads);
    if (g_ds41_attn_split && a->rowcap > MX) a->S.attn_split = ds41_attn_shared_split(a->rowcap, a->heads);
    a->S.sess = m.ds_slots > 1 ? m.sess : nullptr;
    y.ds41_attn = a;
    if (geom::is_engram(l)) ds41_engram_bind(m, l);
#ifdef DS41_IDX_CHAIN
    return ds41_idx_bind(model, i, capacity, stream);
#else
    return 0;
#endif
}
// Table::prepare_attn (per row count M): the scratch is carved once for rowcap rows (MX, or PW with GR8_DS_ATTN_PW; stable addresses for every graph), nothing to do.
inline int ds41_attn_prepare(void* model, int i, int rows, void* stream)
{
    const Ds41AttnLayer* a0 = ds41_attn_state(*(Model*)model, i);
    if (rows < 1 || rows > (a0 ? a0->rowcap : MX)) return 1;
#ifdef DS41_IDX_CHAIN
    return ds41_idx_prepare(model, i, rows, stream);
#else
    (void)model; (void)i; (void)stream;
    return 0;
#endif
}

inline int ds41_attn_op_proj(void* model, int i, int rows, void* stream)
{
    Model& m = *(Model*)model;
    Ds41AttnLayer* a = ds41_attn_state(m, i);
    if (!a) return ds41::NOTSUP;
    if (rows < 1 || rows > a->rowcap) return 1;
    if (ds41_attn_sessions(m, a, rows)) return 1;
    const int dec_cap = a->C.ring_rows - ds41a::WIN + 1;
    if (a->ring_pf && !m.ds41_pack && rows > dec_cap)
    {
        const cudaStream_t st = (cudaStream_t)stream;
        k_ds41_ring_stage<<<ds41a::WIN - 1, ds41a::HD, 0, st>>>(a->ring_pf, a->ring_pf_rows, a->C.ring, a->C.ring_rows, a->S.sess, m.pos);
        ds41a::Cache C_pf = a->C; C_pf.ring = a->ring_pf; C_pf.ring_rows = a->ring_pf_rows;
        ds41a::Scratch S_pf = a->S; S_pf.sess = nullptr; S_pf.ns = 1;
        if (m.ds41_y_quant_ready && m.ds41_rowshard_chunk && rows==4096) ds41_attn_proj_q(&a->W, &C_pf, &S_pf, m.hc_y,m.ds41_y_codes,m.ds41_y_scales,rows,0,a->heads,stream,m.pos);
        else ds41_attn_proj(&a->W, &C_pf, &S_pf, m.hc_y, rows, 0, a->heads, stream, m.pos);
        k_ds41_ring_commit<<<ds41a::WIN, ds41a::HD, 0, st>>>(a->C.ring, a->C.ring_rows, a->S.kvrows, rows, a->S.sess, m.pos);
        return 0;
    }
    if (m.ds41_y_quant_ready && m.ds41_rowshard_chunk && rows==4096) ds41_attn_proj_q(&a->W,&a->C,&a->S,m.hc_y,m.ds41_y_codes,m.ds41_y_scales,rows,0,a->heads,stream,m.pos);
    else ds41_attn_proj(&a->W, &a->C, &a->S, m.hc_y, rows, 0, a->heads, stream, m.pos);
    return 0;
}
// the ratio-2 compressed row of this token (layers 2, 8, 14); the indexer's ST_CA_ROW fn calls this, then does its index-K row from S.cnorm
inline int ds41_attn_op_comp_row(void* model, int i, int rows, void* stream)
{
    Model& m = *(Model*)model;
    Ds41AttnLayer* a = ds41_attn_state(m, i);
    if (!a || !a->W.c_wkv) return ds41::NOTSUP;
    if (rows < 1 || rows > a->rowcap) return 1;
    if (ds41_attn_sessions(m, a, rows)) return 1;
    if (m.ds41_pack) {
        for (int j = 0; j < m.ds41_pack->count; ++j) {
            const auto& seg = m.ds41_pack->seg[j];
            const ds41a::Scratch view = ds41_attn_slice(a->S, seg.row0, a->heads);
            ds41_attn_comp_row(&a->W, &a->C, &view, m.hc_y + size_t(seg.row0) * HID,
                               seg.rows, 0, stream, m.pos + seg.row0);
        }
        return 0;
    }
    ds41_attn_comp_row(&a->W, &a->C, &a->S, m.hc_y, rows, 0, stream, m.pos);
    return 0;
}
inline int ds41_attn_op_core(void* model, int i, int rows, void* stream)
{
    Model& m = *(Model*)model;
    Ds41AttnLayer* a = ds41_attn_state(m, i);
    if (!a) return ds41::NOTSUP;
    if (rows < 1 || rows > a->rowcap) return 1;
    const __nv_bfloat16* comp = nullptr;
    int TK = 0;
    if (a->ratio)
    {
        if (!a->topk) return ds41::NOTSUP;                // the indexer's list is not bound
        comp = a->comp_src;
        if (!comp)
        {
            if (a->src_pos < 0)
            {
                for (int j = 0; j < m.nl; ++j) if (m.layers[j].l == a->src) a->src_pos = j;
                if (a->src_pos < 0) return ds41::NOTSUP;  // the subset has no layer that owns these compressed rows
            }
            const Ds41AttnLayer* s = ds41_attn_state(m, a->src_pos);
            if (!s || !s->C.comp) return ds41::NOTSUP;
            comp = s->C.comp;
            a->C.src_ss = s->C.comp_ss;                   // SESS > 1: the source's slot stride (the indexer sets src_ss with comp_src)
        }
        TK = geom::IDX_TOPK;                              // the list is -1 padded past the visible rows
    }
    if (ds41_attn_sessions(m, a, rows)) return 1;
    const int dec_cap = a->C.ring_rows - ds41a::WIN + 1;
    if (a->ring_pf && !m.ds41_pack && rows > dec_cap)
    {
        ds41a::Cache C_pf = a->C; C_pf.ring = a->ring_pf; C_pf.ring_rows = a->ring_pf_rows;
        ds41a::Scratch S_pf = a->S; S_pf.sess = nullptr; S_pf.ns = 1;
        const __nv_bfloat16* comp_pf = (comp && a->S.sess) ? comp + (size_t)m.ds_pf_slot * a->C.src_ss : comp;
        ds41_attn_core(&a->W, &C_pf, comp_pf, &S_pf, rows, 0, a->heads, a->topk, a->ldtk, TK, stream, m.pos);
        return 0;
    }
    ds41_attn_core(&a->W, &a->C, comp, &a->S, rows, 0, a->heads, a->topk, a->ldtk, TK, stream, m.pos);
    return 0;
}
inline int ds41_attn_op_out(void* model, int i, int rows, void* stream)
{
    Model& m = *(Model*)model;
    Ds41AttnLayer* a = ds41_attn_state(m, i);
    if (!a) return ds41::NOTSUP;
    if (rows < 1 || rows > a->rowcap) return 1;
    if (ds41_attn_sessions(m, a, rows)) return 1;
    ds41_attn_out(&a->W, &a->S, rows, 0, a->heads, m.sc.part[0], stream, m.pos,
                    m.ds41_direct && m.ds41_dchunk && rows == 4096);
    return 0;
}

// Host-callable clear for the driver's reset(), every request after the first in one process: zero this layer's window ring, compressor
// state (kvs / scs) and compressed rows, i.e. the state ds41_attn_bind left (dmalloc zeroes). Async on `stream`; call it when no step is in
// flight and not inside a graph capture. 0 = ok, NOTSUP = layer has no attention state (not bound). Layers 20..39 read the indexer's bf16 shadow of
// the compressed rows (LayerState::comp_src): the indexer clears that buffer. A request that starts at position 0 overwrites ring rows, kvs / scs
// and compressed rows before it reads them, so this is cleanliness, not a correctness need, for the decode plan.
// slot: the S2 slot to clear (SESS > 1: the other slots' sessions keep their state), -1 = every slot.
inline int ds41_attn_clear_layer(Model& m, int i, void* stream, int slot = -1)
{
    Ds41AttnLayer* a = ds41_attn_state(m, i);
    if (!a) return ds41::NOTSUP;
    if (slot < -1 || slot >= m.ds_slots) return 1;
    const cudaStream_t st = (cudaStream_t)stream;
    const size_t k0 = slot < 0 ? 0 : (size_t)slot, nk = slot < 0 ? (size_t)m.ds_slots : 1;
    const size_t ring = (size_t)a->C.ring_rows * ds41a::HD;
    G6RT_CK(cudaMemsetAsync(a->C.ring + k0 * ring, 0, nk * ring * sizeof(__nv_bfloat16), st));
    if (a->C.comp)
    {
        G6RT_CK(cudaMemsetAsync(a->C.comp + k0 * a->C.comp_ss, 0, nk * a->C.comp_ss * sizeof(__nv_bfloat16), st));
        G6RT_CK(cudaMemsetAsync(a->C.kvs + k0 * 2 * ds41a::HD, 0, nk * 2 * ds41a::HD * sizeof(float), st));
        G6RT_CK(cudaMemsetAsync(a->C.scs + k0 * 2 * ds41a::HD, 0, nk * 2 * ds41a::HD * sizeof(float), st));
    }
    return 0;
}
// every bound layer of the Model (the subset's layers); returns the first non-zero code other than NOTSUP, else 0. slot as above.
inline int ds41_attn_clear(void* model, void* stream, int slot)   // slot default -1: rank.cuh's declaration
{
    Model& m = *(Model*)model;
    for (int i = 0; i < (int)m.layers.size(); ++i)
    {
        const int rc = ds41_attn_clear_layer(m, i, stream, slot);
        if (rc && rc != ds41::NOTSUP) return rc;
    }
    return 0;
}
// DSpark / MTP: after the accept of a verify chunk that can be partly rejected, re-park the compressor's pending half at every ratio-2
// source layer (2, 8, 14) the Model holds (k_comp_commit). dq / dacc: device words, the accept result (last accepted position, accepted
// drafts). Async on `stream`, graph-safe, 3 tiny launches; call it after the accept and before the next step's comp_row. 0 = ok.
inline int ds41_attn_comp_commit_all(void* model, const int* dq, const int* dacc, void* stream)
{
    Model& m = *(Model*)model;
    if (!dq || !dacc || m.ds_slots > 1) return 1;   // SESS > 1: ds41_attn_comp_commit_sessions
    for (int i = 0; i < (int)m.layers.size(); ++i)
    {
        Ds41AttnLayer* a = ds41_attn_state(m, i);
        if (a && a->W.c_wkv) ds41_attn_comp_commit(&a->C, &a->S, dq, dacc, stream);
    }
    return 0;
}
// SESS > 1 (Model::ds_slots > 1): the same re-park for the ns sessions of a draft step. Session j: dq[j] / dacc[j] (the accept result), drow0[j] = its
// first row in the verify (or prompt) chunk whose S.ckv / S.csc are still intact, dslot[j] = its S2 slot. Device words [ns]; async, graph-safe.
inline int ds41_attn_comp_commit_sessions(void* model, const int* dq, const int* dacc, const int* drow0, const int* dslot, int ns, void* stream)
{
    Model& m = *(Model*)model;
    if (!dq || !dacc || !drow0 || !dslot || ns < 1 || ns > m.ds_slots) return 1;
    for (int i = 0; i < (int)m.layers.size(); ++i)
    {
        Ds41AttnLayer* a = ds41_attn_state(m, i);
        if (a && a->W.c_wkv) ds41_attn_comp_commit_ns(&a->C, &a->S, dq, dacc, drow0, dslot, ns, stream);
    }
    return 0;
}
#endif
