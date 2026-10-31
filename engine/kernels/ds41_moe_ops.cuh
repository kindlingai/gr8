// MoE op-table adapters for the DS41 main-model decode step: ST_ROUTER, ST_SHARED, ST_EXPERTS + bind_moe / prepare_moe.
// Included inside namespace g6rt by ds41_step.cuh (the MoE line of ds41_table()); the kernels come from
// ds41_moe_kernels.h, which rank.cuh includes at global scope. Our own code (ds_router.cuh, ds_shared.cuh, mxfp4.cuh).
//
// Contract (docs/DEEPSEEK-STEP.md, ds41_ops.h Fn = int(void* model, int i, int rows, void* stream), 0 = ok):
//   in   Model::hc_y         bf16 [rows][HID], the ffn sub-layer input (ds_mhc::prepare output, already normed)
//   out  Model::sc.part[1]   bf16 [rows][HID], this rank's partial: routed (6 x weight x 1.5, the router applies the scale)
//                            + shared expert, summed in fp32 and rounded to bf16 once. The driver publishes / all-reduces it.
//   ST_ROUTER  gate GEMV + sqrtsoftplus + noaux_tc top-6 + the slot plan (+ the fp16 B fragment table of the input)
//   ST_SHARED  shared expert fc1 (clamp 10 before silu, weight 1) + fc2, fp32 [8][HID] kept for the combine
//   ST_EXPERTS routed fc1, fc2 over the plan's slots, then the combine (last launch, writes part[1])
// State: one LayerState per layer behind Layer::ds41_moe (weights, made by bind_moe at build); one Scratch per Model, shared by
// every layer (the three ops of a layer run back to back on the step stream, layers one after the other).
// M <= 8: one n=8 tile. M 9..32: the multi-tile sequence (ds_plan_mt.h): NT = ceil(M/8) token tiles, a routed expert's CTA reads its
// weights once and runs its tiles. Only configurations that tests/ds_shared_test.cu `check` ran: 384 experts / top-6 (GPU-verified),
// 128 / top-3 compiled the same way (DSpark draft layers; kernels verified at 128/3, this adapter not run there).
// Weight tensors: docs/DEEPSEEK-WEIGHTS.md.
// M 33..256 (prefill): the chunked path. k_plan_pw builds a plan with W-word expert masks whose slots are
// 32-token chunks of an expert (ds_plan_mt.h), so the unchanged NT = 4 multi-tile kernels run it; the shared expert runs in 32-row chunks on
// shifted pointers. Own scratch (PwScratch, allocated by prepare_moe when M > 32, never inside a capture), plain launches only (no PDL).
// M DS41_MOE_RO_FROM (default 257) .. ROMAX = 4096 (prefill): the read-once path. A multi-CTA planner (k_route_ro, k_scan_ro,
// k_place_ro: slot = one distinct expert, plan_ref's tiles) and k_fc1_ro / k_fc2_ro (a CTA group per expert, each weight record read by one CTA, walks the
// expert's tiles in groups of NTG). Own scratch (RoScratch, sized for ROMAX rows, allocated by prepare_moe, never inside a capture), plain launches only.
//
// PDL (programmatic dependent launch), off by default: GR8_DS_MOE_PDL=1 turns it on, 0 = plain launches.
// (Default off until the engine dump shows PDL=1 == PDL=0 on real layers.)
// The three ops launch one chain on the step stream: build_xfrag (plain, first of the layer: it waits for everything before it,
// so the scratch reuse across layers stays ordered) -> gate -> plan -> shared fc1 -> shared fc2 -> routed fc1 -> routed fc2 -> combine, the
// last seven with the programmatic-stream-serialization attribute. Rules the kernels follow (ds_router.cuh, ds_shared.cuh, mxfp4.cuh):
//   * every PDL-launched kernel executes griddepcontrol.wait on every path before it can finish, so a kernel finishing means all before it did;
//   * a kernel reads its predecessor's output only after its wait, or reads only data from kernels already complete (build_xfrag's xfrag
//     and the plan are read before the wait by kernels at least two steps later; each kernel triggers only after its own wait when its
//     dependents read such data);
//   * stores to scratch that an earlier kernel of the same layer reads happen after the wait.
// A launch between the ops from another source only delays a PDL launch (it then waits for full completion), it never breaks one.
#pragma once
#if G6_MODEL_DS41
#ifndef GR8_DS41_MOE_KERNELS
#error "ds41_moe_ops.cuh: include ../kernels/ds41_moe_kernels.h at global scope (rank.cuh, next to ds41_ops.h) first"
#endif
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "../knobs-rs/gr8_knobs.h"
#include "../runtime/ds41_moe_topk.h"
#include "ds_ro_cfg.h"                          // DS41_RO_* : the read-once expert loop shape

// GR8_DS_SH_TC=1: the shared expert of the read-once path on the attention code's tensor-core MXFP8 GEMM. The device code is in the attention TU
// (attn/ds41_attn.cu C entries, attn/ds41_sh_tc.cuh); WEAK, as engram_dev.cuh does, so a test bin without that object still links (the flag then dies at boot).
extern "C" int ds41_gemm_mx_tc(const uint8_t* A, int lda, const uint8_t* SA, int lsa, const uint8_t* W, const uint8_t* SW, void* Y, int ldy,
                               int M, int N, int K, void* st) __attribute__((weak));
extern "C" int ds41_sh_act_quant(const void* x, int ldx, uint8_t* codes, int ldc, uint8_t* scales, int lds, int M, int K, void* st) __attribute__((weak));
extern "C" int ds41_sh_unrecord(const uint8_t* rec, int T, int K, uint8_t* codes, int ldc, uint8_t* scales, int lds, void* st) __attribute__((weak));
extern "C" int ds41_sh_swiglu_q(const void* y13, int ldy, int I, float lim, uint8_t* codes, int ldc, uint8_t* scales, int lds, int M, int KP,
                                void* st) __attribute__((weak));

namespace ds41moe
{
constexpr int MMAX = dsr::MMAX;                       // 8 rows per step on the one-tile path
constexpr int MTMAX = dsr::MT_MAX;                    // 32 rows per step on the multi-tile path
constexpr int PWMAX = dsr::PW_MAX;                    // 256 rows per call on the chunked (prefill) path
constexpr int ROMAX = 4096;                           // rows per call on the read-once path (RoScratch is sized for this)
static_assert(ROMAX <= dsr::RO_MAX, "RoScratch bound");
#ifndef DS41_MOE_RO_FROM
#define DS41_MOE_RO_FROM (PWMAX + 1)                  // first row count on the read-once path; 33 sends every prefill call there (GPU test sweeps it)
#endif
constexpr int RO_FROM = DS41_MOE_RO_FROM;
static_assert(RO_FROM > MTMAX && RO_FROM <= ROMAX, "RO_FROM");
// read-once expert loop shape: DS41_RO_KS/CW/NTG1/W2/D2/NTG2 and the reason for their values are in ds_ro_cfg.h (shared with the GPU test)
constexpr int IW = geom::EXP_I / g6nr::NR;            // this rank's slice of the expert width (576 at TP4); shared and routed alike
static_assert(HID == dsr::HIDDEN && geom::SHARED_I == geom::EXP_I && IW == 576, "the verified tiling is hidden 5120, I-slice 576 (TP4)");
constexpr int SMAX = MTMAX * dsr::TOPK;               // routed tiles per step at the main model's top-6 (the largest case; slots <= tiles <= MTMAX * TK)

struct PwScratch                                      // chunked path, M 33..PWMAX; sized for the main model's top-6 / 384 experts (the DSpark layers need less)
{
    float* logits;          // [PWMAX][NE]
    char* plan;             // dsr::PlanMT<TK, 128 or 256>
    u32* xfrag;             // [PWMAX/8][HID/16*64]
    __half* act;            // [tiles][8][IW], tiles <= pw_tile_bound(PWMAX * TOPK, NEXP) = 528
    __half* ypart;          // [tiles][8][HID]
    __half* sh_act;         // [PWMAX][IW]
    float* sh_out;          // [PWMAX][HID]
};
struct RoScratch                                      // read-once path, M RO_FROM..ROMAX; sized for the main model's top-6 / 384 experts at ROMAX rows
{
    float* logits;          // [ROMAX][NE]
    char* plan;             // dsr::PlanMT<TK, ROMAX>
    unsigned* emask;        // [NE][ROMAX/32] token bits per expert, zeroed before every plan
    int* tile0_of;          // [NE] first global tile of the expert (k_scan_ro -> k_place_ro)
    int* order;             // [NE] GR8_DS_RO_ORDER: order[pos] = slot, the launch order of the expert loop (k_order_ro -> k_fc1_ro / k_fc2_ro)
    u32* xfrag;             // [ROMAX/8][HID/16*64]
    __half* act;            // [tiles][8][IW], tiles <= pw_tile_bound(ROMAX * TOPK, NEXP) = 3408 at ROMAX 4096
    __half* ypart;          // [tiles][8][HID]
    __half* sh_act;         // [ROMAX][IW]
    float* sh_out;          // [ROMAX][HID]; GR8_DS_SH_TC=1: holds the TC shared expert's bf16 Y [ROMAX][HID] instead (aliased, half the bytes)
    uint8_t* tc_xc = nullptr;     // GR8_DS_SH_TC=1 only: [ROMAX][HID] E4M3 codes of hc_y
    uint8_t* tc_xs = nullptr;     //   [ROMAX][HID/32] their E8M0 bytes
    __nv_bfloat16* tc_y13 = nullptr;   // [ROMAX][2 IW] fc1 out (gate | up), bf16
    uint8_t* tc_ac = nullptr;     //   [ROMAX][SH_KP] act codes (k >= IW: 0)
    uint8_t* tc_as = nullptr;     //   [ROMAX][SH_KP/32] their E8M0 bytes
};
struct Scratch
{
    dsr::ExpertLog* elog = nullptr;   // distinct-expert ring (ds_ro.h), written by every prefill router call (rows > 32); read with elog_read
    RoScratch* ro = nullptr;   // set by prepare_moe(M >= RO_FROM)
    PwScratch* pw = nullptr;   // set by prepare_moe(M > MTMAX)
    float* logits;          // [32][NE] gate logits
    char* plan;             // dsr::PlanT<TK> (M <= 8) or dsr::PlanMT<TK, 32> (M 9..32)
    u32* xfrag;             // [MTMAX/8][HID/16*64] fp16 B fragment tables of hc_y (fc1 of the routed and the shared expert); tile 0 = the M <= 8 table
    __half* act;            // [SMAX][8][IW] routed fc1 -> fc2 (indexed by slot at M <= 8, by global tile at M 9..32)
    __half* ypart;          // [SMAX][8][HID] routed fc2 -> combine
    __half* sh_act;         // [32][IW]
    float* sh_out;          // [32][HID] shared expert out, kept ST_SHARED -> ST_EXPERTS
    // trellis layers (docs/DS41-TRELLIS.md)
    __half* x16 = nullptr;  // [MTMAX][HID] the K1 input: hc_y rows in fp16
    int* tile_ex = nullptr; // [SMAX] expert of each tile of the M 9..32 plan, then [SMAX] its tile count
};
struct LayerState
{
    int ne = 0, tk = 0;
    int layer = 0;                           // checkpoint layer number (the counter's record)
    const __nv_bfloat16* gate = nullptr;     // [NE][HID]
    const float* bias = nullptr;             // [NE] e_score_correction_bias (selection only)
    const uint8_t *sh_w1 = nullptr, *sh_w3 = nullptr, *sh_w2 = nullptr;   // MXFP8 gv records of the shared expert's I-slice
    const uint8_t *fc1 = nullptr, *fc2 = nullptr;                          // MXFP4 mx4g records of all NE experts' I-slice
    Scratch* sc = nullptr;
    // GR8_DS_SH_TC=1 only (else null): plain copies of the shared expert for ds41_gemm_mx_tc, made once at bind from the records (which decode keeps)
    const uint8_t *tc_w13 = nullptr, *tc_sw13 = nullptr;   // [2 IW][HID] codes (rows 0..IW-1 gate, IW..2IW-1 up) + [2 IW][HID/32] E8M0
    const uint8_t *tc_w2 = nullptr, *tc_sw2 = nullptr;     // [HID][SH_KP] codes (k >= IW: 0) + [HID][SH_KP/32] E8M0 (k blocks >= IW/32: 0)
    // exlr8 trellis experts (the layer has tr3.table): the K1 kernels run its routed and shared experts, and the MXFP4 / FP8
    // records above stay null
    ::bind::LayerTensors* tr = nullptr;
};

inline bool elog_on() { return knob::ds_moe_elog(); }   // the distinct-expert counter (GR8_DS_MOE_ELOG, default on)
inline bool pdl_on() { return knob::ds_moe_pdl(); }
// GR8_DS_SH_TC=1 (default 0): the shared expert of the read-once path (rows >= RO_FROM) runs
// act_quant(hc_y) -> TC GEMM fc1 (M x 2IW x HID) -> k_sh_swiglu_q -> TC GEMM fc2 (M x HID x SH_KP) -> bf16 Y, and the combine reads that bf16 Y (k_combine_mt_shb).
// W8A8 (x and act E4M3 per 32 + E8M0), not the GL path's W8A16: not bit-identical to the flag-off path. Decode and the PW path keep the records and
// the GEMV. Costs per rank a plain copy of the shared weights (9,461,760 B per MoE layer) plus the RoScratch tc_* buffers. 1 without the attention
// TU's entries linked dies at boot.
constexpr int SH_KP = (IW + 127) / 128 * 128;          // fc2 K padded for tc_ok (K % 128): 576 -> 640; codes 0 in the pad add exactly 0
inline bool sh_tc_on() { return knob::ds_sh_tc(); }
// GR8_DS_MOE_PW_BOUND=1 (default off): the M 33..MMAX planner path (experts_pw) sizes its expert-slot grid from the routes the top-k cut keeps (M * kc) instead of
// M * TK. Slots only shrink when GR8_DS_MOE_TOPK < TOPK; CTAs of slots >= n_slots exit before any work, so outputs are bit-identical (the bound stays >= the plan's slot count).
inline bool pw_bound_on() { return knob::ds_moe_pw_bound(); }
// GR8_DS_MOE_SH_1PASS=1 (default off): the shared expert at M 33..SH1_MAX in one NT = 6 pass instead of the 32-row chunk loop (the 9.12 MB of weights is read once).
inline bool sh_1pass_on() { return knob::ds_moe_sh_1pass(); }
constexpr int SH1_NT = 6, SH1_MAX = SH1_NT * 8;   // the one-pass tile count and its row limit (prepare_moe below and shared_pw_1pass read them)
inline cudaError_t sh1_attr();                    // defined with shared_pw_1pass; prepare_moe calls it at host init
// GR8_DS_GATE_RO=1 (default off): the read-once prefill path (M >= RO_FROM) computes the main layers' gate logits with k_gate_logits_ro (8 experts x 8 tokens per warp,
// ds_router.cuh) instead of k_gate_logits_mt. Same bits (same fma chain per token and expert); the logits buffer is byte-identical. The 128-expert draft layers keep k_gate_logits_mt.
inline bool gate_ro_on() { return knob::ds_gate_ro(); }
// The 80 KB dynamic smem of k_gate_logits_ro: a host call without stream work, made once (prepare_moe makes the first call at host init, before any capture or launch).
inline cudaError_t gro_attr()
{
    static const cudaError_t attr = cudaFuncSetAttribute(dsr::k_gate_logits_ro<dsr::NEXP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsr::GRO_SMEM);
    return attr;
}
// GR8_DS_RO_ORDER=1 (default off): the read-once expert loop (k_fc1_ro / k_fc2_ro) launches its CTA groups in the order k_order_ro computes (ds_ro.h: heavy and
// light experts merged in proportion to their work, the heaviest first) instead of expert-id order. Only the start order of the CTAs changes: every (slot, chunk) CTA runs
// the same instructions on the same data, so every output byte is the same. The speed is not measured.
inline bool ro_order_on() { return knob::ds_ro_order(); }
// GR8_DS_RO_L2LA=<n> (default 0 = off): while CTA b of k_fc1_ro runs, its warp 0 prefetches into L2 the weights of CTA b + n
// (n = 48 is one wave at 1 CTA per SM on the 48 SMs); k_fc2_ro (2 CTAs per SM) looks 2 n ahead. A hint only: every value, order and write is the same. The speed is not measured,
// and 8.4 MB of lookahead (fc1, n = 48) sits next to the 21 MB xfrag in a 24 MB L2, so n is a runtime knob.
inline int ro_l2la() { return (int)knob::ds_ro_l2la(); }
// GR8_DS_MOE_L2HINT=0|1|2 (default 0 = off): an L2 evict_last hint on the routed expert weight loads of the read-once expert loop (Grp::loadh<true>, mxfp4.cuh):
// 1 = k_fc1_ro only (fc1_part_mt), 2 = k_fc1_ro and k_fc2_ro (the fc2 weight records). No x-fragment load, act load or store carries a hint, and the decode / M <= 32 kernels
// (k_fc1_mt, k_fc2_mt) and the shared expert are not touched. A hint only: every value, order and write is the same.
// The speed is measured on fc1 alone in a test harness, not on the whole layer.
inline int ro_l2hint() { return (int)knob::ds_moe_l2hint(); }
// GR8_DS_MOE_FC2STG=0|1 (default 0 = off): k_fc2_ro (the read-once fc2 expert loop only) reads the act tile of a group into registers in one batch, then stores it to smem
// (mx4::k_fc2_ro, template STG), instead of one load and one store per trip. Every address and value is the same; only the order of the loads and stores changes. The decode / M <= 32
// kernels, k_fc1_ro and the shared expert are not touched. The speed is measured on the fc2 kernel
// alone in a test harness, not on the whole layer.
inline int ro_fc2stg() { return knob::ds_moe_fc2stg() ? 1 : 0; }
// GR8_DS_MOE_FC1BSTG=0|1 (default off): the read-once fc1 kernel copies its
// FP16 B fragments to a two-record cp.async shared ring before the same MMA.
inline int ro_fc1bstg() { return knob::ds_moe_fc1bstg() ? 1 : 0; }
// The top-k cut GR8_DS_MOE_TOPK (quality not measured): routed experts kept per token on the main layers (384 experts). Default dsr::TOPK.
// The kept routes are the planner's first KC picks (best biased key first); the weights renormalize over
// them (sum of the kept scores, x ROUTE_SCALE). The DSpark draft layers (128 experts, top-3) keep their k. GR8_MOE_TOPK is the Flash model's knob, so it is not
// used here.
static_assert(dsr::TOPK == 6, "knobs/ds41-moe.yaml GR8_DS_MOE_TOPK: values 1..6, default 6");
inline int topk_cut() { return (int)knob::ds_moe_topk(); }
inline int topk_cut_env(std::string*) { return topk_cut(); }   // the knob table refuses a bad value at its first read
inline int topk_decode_env(std::string* err)
{
    return ds41_moe_topk::decode_env(topk_cut(), dsr::TOPK, err);
}
inline int topk_decode_cut()
{
    static const int k = [] { std::string e; const int v = topk_decode_env(&e); return v > 0 ? v : topk_cut(); }();
    return k;
}
inline int topk_for_step(const Model& m)
{
    return ds41_moe_topk::for_step(m.ds_moe_decode_step, topk_cut(), topk_decode_cut());
}
// f(std::integral_constant<int, kc>{}) with kc = the cut as a compile-time constant (the plan kernels are templates in it); returns f's int.
template <class F>
inline int with_topk_cut(int kc, F&& f)
{
    static_assert(dsr::TOPK == 6, "with_topk_cut: one case per cut 1..TOPK");
    switch (kc)
    {
        case 1: return f(std::integral_constant<int, 1>{});
        case 2: return f(std::integral_constant<int, 2>{});
        case 3: return f(std::integral_constant<int, 3>{});
        case 4: return f(std::integral_constant<int, 4>{});
        case 5: return f(std::integral_constant<int, 5>{});
        default: return f(std::integral_constant<int, 6>{});
    }
}
// One launch on `st`; pdl = with the programmatic-stream-serialization attribute (a plain launch otherwise).
template <typename... KP, typename... AP>
inline cudaError_t launch(bool pdl, void (*k)(KP...), dim3 grid, dim3 block, cudaStream_t st, AP... args)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    return cudaLaunchKernelEx(&cfg, k, static_cast<KP>(args)...);
}

inline Scratch* scratch_of(const Model* m)
{
    static std::mutex mu;
    static std::unordered_map<const Model*, Scratch*> map;
    std::lock_guard<std::mutex> lk(mu);
    Scratch*& s = map[m];
    if (!s)
    {
        s = new Scratch();
        constexpr size_t PLAN_B = sizeof(dsr::PlanT<dsr::TOPK>) > sizeof(dsr::PlanMT<dsr::TOPK, MTMAX>) ? sizeof(dsr::PlanT<dsr::TOPK>) : sizeof(dsr::PlanMT<dsr::TOPK, MTMAX>);
        s->logits = dmalloc<float>((size_t)MTMAX * dsr::NEXP);
        s->plan = dmalloc<char>(PLAN_B);
        s->xfrag = dmalloc<u32>((size_t)MTMAX / 8 * (HID / 16 * 64));
        s->act = dmalloc<__half>((size_t)SMAX * 8 * IW);
        s->ypart = dmalloc<__half>((size_t)SMAX * 8 * HID);
        s->sh_act = dmalloc<__half>((size_t)MTMAX * IW);
        s->sh_out = dmalloc<float>((size_t)MTMAX * HID);
        s->x16 = dmalloc<__half>((size_t)MTMAX * HID);
        s->tile_ex = dmalloc<int>(2 * (size_t)SMAX);
        s->elog = dmalloc<dsr::ExpertLog>(1);
        cudaMemset(s->elog, 0, sizeof(dsr::ExpertLog));
    }
    return s;
}

// ---------------------------------------------------------------------------------------- trellis layers
// docs/DS41-TRELLIS.md. The K1 kernels (k1.cuh) run the routed experts and the trellis shared expert (tr3.table row 384) of
// the 128-blocks this rank owns, and their finalize writes the BF16 partial to part[1] with the plan's weights.
__global__ void k_tr_x16(const __nv_bfloat16* x, __half* y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(__bfloat162float(x[i]));
}
// M 9..32: a K1 slot is a tile of the plan, so each tile needs its expert, and the slot count is the tile count
__global__ void k_tr_tiles(const dsr::PlanMT<dsr::TOPK, MTMAX>* P, int* tile_ex, int* n_tiles)
{
    for (int s = threadIdx.x; s < P->n_slots; s += blockDim.x)
        for (int j = 0; j < P->slot_nt[s]; ++j) tile_ex[P->slot_tile0[s] + j] = P->slot_expert[s];
    if (threadIdx.x == 0) *n_tiles = P->n_tiles;
}

inline const char* bind_trellis(LayerState& s, const Find& find, int l)
{
    s.tr = new ::bind::LayerTensors(::bind::layer_tensors(l, [&](const char* n) {
        const TensorRef r = find(n);
        ::bind::Tensor t;
        t.p = r.p; t.nd = r.nd; t.fmt = r.fmt;
        for (int k = 0; k < 4; ++k) t.shape[k] = r.shape[k];
        return t;
    }));
    if (const char* e = ::bind::scan_widths(*s.tr)) return e;
    if (!s.tr->sh_trellis) return "trellis layer without the trellis shared expert (tr3.table row 384)";
    static const cudaError_t attr = [] {
        cudaError_t err = cudaSuccess;
        ::bind::k1_each([&](auto E4, auto K2, auto SH, auto CB) {
            for (const void* f : {::bind::k1_fn_fc1<K1_V(E4), K1_V(K2), K1_V(SH), K1_V(CB)>(), ::bind::k1_fn_fc2<K1_V(E4), K1_V(K2), K1_V(SH), K1_V(CB)>()})
            {
                const cudaError_t e = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, KSH * 256);
                if (err == cudaSuccess) err = e;
            }
        });
        return err;
    }();
    return attr == cudaSuccess ? nullptr : "cannot set the K1 kernels' shared memory size";
}

// n <= MTMAX rows of x (bf16, routed by the plan in sc->plan) through K1, the partial to out (BF16 [n][HID])
inline int k1_rows(Model& m, const LayerState& s, const void* x, int n, void* out, cudaStream_t st)
{
    Scratch& sc = *s.sc;
    k_tr_x16<<<(n * HID + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)x, sc.x16, n * HID);
    ::bind::PlanPtrs pp;
    if (n <= MMAX) pp = ::bind::plan_ptrs((const dsr::PlanT<dsr::TOPK>*)sc.plan);
    else
    {
        using P = dsr::PlanMT<dsr::TOPK, MTMAX>;
        const P* p = (const P*)sc.plan;
        k_tr_tiles<<<1, 256, 0, st>>>(p, sc.tile_ex, sc.tile_ex + SMAX);
        pp.n_slots = sc.tile_ex + SMAX; pp.slot_expert = sc.tile_ex; pp.slot_mask = p->tile_mask;
        pp.slot_w = &p->tile_w[0][0]; pp.slot_tok = &p->tile_tok[0][0];
        pp.tok_slot = &p->tok_slot[0][0]; pp.tok_col = &p->tok_col[0][0]; pp.tok_w = &p->tok_w[0][0];
    }
    ::bind::K1 k;
    ::bind::Tensor sh_out[3];
    if (const char* e = ::bind::make_k1(*s.tr, n, m.sc, sc.x16, sc.xfrag, pp, out, 1, k, sh_out))
    {
        fprintf(stderr, "DS41 MoE layer %d, %d rows: %s\n", s.layer, n, e);
        return ds41::NOTSUP;
    }
    ::bind::launch_k1(k, st);
    return (int)cudaGetLastError();
}

// rows > MTMAX (prefill): 32-row passes, each routed on its own rows (one plan per pass)
inline int trellis_prefill(Model& m, const LayerState& s, int rows, cudaStream_t st)
{
    for (int r0 = 0; r0 < rows; r0 += MTMAX)
    {
        const int n = std::min(MTMAX, rows - r0);
        const __nv_bfloat16* x = (const __nv_bfloat16*)m.hc_y + (size_t)r0 * HID;
        const int r = with_topk_cut(topk_for_step(m), [&](auto kc) {
            constexpr int KC = decltype(kc)::value;
            launch(false, dsr::k_gate_logits_mt<dsr::NEXP, 0>, dim3(dsr::NEXP / 8), dim3(256), st, (const __nv_bfloat16*)s.gate, x, n, s.sc->logits);
            if (n <= MMAX) launch(false, dsr::k_plan<dsr::NEXP, dsr::TOPK, KC>, dim3(1), dim3(dsr::PLAN_THREADS), st, (const float*)s.sc->logits, s.bias, n,
                                  (dsr::PlanT<dsr::TOPK>*)s.sc->plan);
            else launch(false, dsr::k_plan_mt<dsr::NEXP, dsr::TOPK, MTMAX, 0, KC>, dim3(1), dim3(dsr::PLAN_THREADS), st, (const float*)s.sc->logits, s.bias, n,
                        (dsr::PlanMT<dsr::TOPK, MTMAX>*)s.sc->plan);
            return (int)cudaGetLastError();
        });
        if (r) return r;
        if (const int e = k1_rows(m, s, x, n, (uint16_t*)m.sc.part[1] + (size_t)r0 * HID, st)) return e;
    }
    return 0;
}

// bind_moe(model, i, 0, stream): resolve and shape-check the layer's tensors, once, at build.
inline int bind_moe(void* model, int i, int, void*)
{
    Model& m = *(Model*)model;
    Layer& y = m.layers[i];
    if (!y.moe) return 0;
    const Find& find = m.ds41_find;
    const std::string who = "DS41 MoE layer " + std::to_string(y.l) + ": ";
    auto get = [&](const char* n, int nd, std::initializer_list<long long> shape, int dt /*0 u8, 1 bf16, 2 f32*/) {
        const TensorRef t = find(lname(y.l, n));
        if (!t.p) die(who + "the pack has no " + lname(y.l, n));
        bool ok = t.nd == nd && (dt == 1 ? t.bf16 : dt == 2 ? t.fp32 : (!t.bf16 && !t.fp32));
        int k = 0;
        for (long long s : shape) ok = ok && t.shape[k++] == s;
        if (!ok) die(who + "bad dtype/shape: " + lname(y.l, n));
        return t;
    };
    const TensorRef g = find(lname(y.l, "router.w"));
    const int ne = g.p && g.nd == 2 ? (int)g.shape[0] : 0;
    if (ne != dsr::NEXP && ne != geom::DS_NEXP) die(who + "router.w must be [384 or 128][5120]");
    {   // GR8_DS_MOE_TOPK: the cut applies to the main layers only
        const int kc = topk_cut();
        static bool said = false;
        if (ne == dsr::NEXP && kc < dsr::TOPK && !said)
        {
            said = true;
            fprintf(stderr, "# GR8_DS_MOE_TOPK=%d: routed experts per token cut %d -> %d on the main layers, weights renormalized over the kept (quality not measured)\n",
                    kc, (int)dsr::TOPK, kc);
        }
    }
    {   // GR8_DS_MOE_L2HINT: every rank prints the value once, at its first MoE layer (one fixed format, for boot checks)
        const int h = ro_l2hint();
        bool first = true;
        for (int j = 0; j < i; ++j) if (m.layers[j].moe) first = false;
        if (first)
            fprintf(stderr, "# rank %d DS41 GR8_DS_MOE_L2HINT=%d k_fc1_ro_hint=%d k_fc2_ro_hint=%d\n", m.rank, h, h >= 1 ? 1 : 0, h == 2 ? 1 : 0);
    }
    {   // GR8_DS_MOE_FC2STG: same rule; the fixed boot line is separate from the L2HINT line
        const int g = ro_fc2stg();
        bool first = true;
        for (int j = 0; j < i; ++j) if (m.layers[j].moe) first = false;
        if (first)
            fprintf(stderr, "# rank %d DS41 GR8_DS_MOE_FC2STG=%d k_fc2_ro_stg=%d\n", m.rank, g, g);
    }
    {   // GR8_DS_MOE_FC1BSTG: same rule, its own boot line
        const int g = ro_fc1bstg();
        bool first = true;
        for (int j = 0; j < i; ++j) if (m.layers[j].moe) first = false;
        if (first)
            fprintf(stderr, "# rank %d DS41 GR8_DS_MOE_FC1BSTG=%d k_fc1_ro_bstg=%d\n", m.rank, g, g);
    }
    LayerState* s = new LayerState();
    s->ne = ne; s->tk = ne == dsr::NEXP ? dsr::TOPK : geom::DS_TOPK; s->layer = y.l;
    s->gate = (const __nv_bfloat16*)get("router.w", 2, {ne, HID}, 1).p;
    s->bias = (const float*)get("router.bias", 1, {ne}, 2).p;
    s->sc = scratch_of(&m);
    if (find(lname(y.l, "tr3.table")).p)
    {
        if (ne != dsr::NEXP) die(who + "trellis experts on a DSpark layer");
        if (const char* e = bind_trellis(*s, find, y.l)) die(who + e);
        y.ds41_moe = s;
        return 0;
    }
    s->sh_w1 = (const uint8_t*)get("sh.w1.mx.gv", 3, {IW / 16, HID / 32, 528}, 0).p;
    s->sh_w3 = (const uint8_t*)get("sh.w3.mx.gv", 3, {IW / 16, HID / 32, 528}, 0).p;
    s->sh_w2 = (const uint8_t*)get("sh.w2.mx.gv", 3, {HID / 16, IW / 32, 528}, 0).p;
    s->fc1 = (const uint8_t*)get("x.fc1", 4, {ne, IW / 16, HID / 64, 1088}, 0).p;
    s->fc2 = (const uint8_t*)get("x.fc2", 4, {ne, HID / 16, IW / 64, 544}, 0).p;
    {   // GR8_DS_SH_TC: 1 repacks the shared expert's records into the plain TC operands, once
        if (sh_tc_on())
        {
            if (!ds41_gemm_mx_tc || !ds41_sh_act_quant || !ds41_sh_unrecord || !ds41_sh_swiglu_q)
                die(who + "GR8_DS_SH_TC=1 needs the attention TU's ds41_gemm_mx_tc / ds41_sh_* entries (attn/ds41_attn.cu), not linked in this binary");
            constexpr int N13 = 2 * IW;
            uint8_t* w13 = dmalloc<uint8_t>((size_t)N13 * HID);             // dmalloc zeroes: the fc2 pad (k >= IW) stays code 0, byte 0
            uint8_t* sw13 = dmalloc<uint8_t>((size_t)N13 * (HID / 32));
            uint8_t* w2 = dmalloc<uint8_t>((size_t)HID * SH_KP);
            uint8_t* sw2 = dmalloc<uint8_t>((size_t)HID * (SH_KP / 32));
            int r = ds41_sh_unrecord(s->sh_w1, IW / 16, HID, w13, HID, sw13, HID / 32, nullptr);
            r |= ds41_sh_unrecord(s->sh_w3, IW / 16, HID, w13 + (size_t)IW * HID, HID, sw13 + (size_t)IW * (HID / 32), HID / 32, nullptr);
            r |= ds41_sh_unrecord(s->sh_w2, HID / 16, IW, w2, SH_KP, sw2, SH_KP / 32, nullptr);
            if (r) die(who + "GR8_DS_SH_TC repack: bad shape");
            const cudaError_t ce = cudaDeviceSynchronize();
            if (ce != cudaSuccess) die(who + "GR8_DS_SH_TC repack: " + cudaGetErrorString(ce));
            s->tc_w13 = w13; s->tc_sw13 = sw13; s->tc_w2 = w2; s->tc_sw2 = sw2;
            static bool said = false;
            if (!said)
            {
                said = true;
                fprintf(stderr, "# GR8_DS_SH_TC=1: prefill shared expert (rows >= %d) on the tensor-core GEMM, W8A8; +%zu B per MoE layer of plain weights\n",
                        RO_FROM, (size_t)N13 * HID + (size_t)N13 * (HID / 32) + (size_t)HID * SH_KP + (size_t)HID * (SH_KP / 32));
            }
        }
    }
    y.ds41_moe = s;
    return 0;
}
// prepare_moe(model, i, M, stream): per row count. The kernel arguments are plain values built in the ops from `rows`
// (masks and grids are functions of M), so this only refuses what the ops cannot do.
inline PwScratch* pw_alloc()
{
    PwScratch* p = new PwScratch();
    constexpr int TILES = (PWMAX * dsr::TOPK + 7 * dsr::NEXP) / 8;       // dsr::pw_tile_bound at the largest case
    p->logits = dmalloc<float>((size_t)PWMAX * dsr::NEXP);
    p->plan = dmalloc<char>(sizeof(dsr::PlanMT<dsr::TOPK, PWMAX>));      // the 128-row plan is a prefix-sized subset of this
    p->xfrag = dmalloc<u32>((size_t)PWMAX / 8 * (HID / 16 * 64));
    p->act = dmalloc<__half>((size_t)TILES * 8 * IW);
    p->ypart = dmalloc<__half>((size_t)TILES * 8 * HID);
    p->sh_act = dmalloc<__half>((size_t)PWMAX * IW);
    p->sh_out = dmalloc<float>((size_t)PWMAX * HID);
    return p;
}
inline RoScratch* ro_alloc()
{
    RoScratch* p = new RoScratch();
    constexpr int TILES = (ROMAX * dsr::TOPK + 7 * dsr::NEXP) / 8;       // dsr::pw_tile_bound at the largest case (3408 at ROMAX 4096)
    p->logits = dmalloc<float>((size_t)ROMAX * dsr::NEXP);
    p->plan = dmalloc<char>(sizeof(dsr::PlanMT<dsr::TOPK, ROMAX>));      // the 128-expert / top-3 plan is a prefix-sized subset of this
    p->emask = dmalloc<unsigned>((size_t)dsr::NEXP * (ROMAX / 32));
    p->tile0_of = dmalloc<int>((size_t)dsr::NEXP);
    p->order = dmalloc<int>((size_t)dsr::NEXP);
    p->xfrag = dmalloc<u32>((size_t)ROMAX / 8 * (HID / 16 * 64));
    p->act = dmalloc<__half>((size_t)TILES * 8 * IW);
    p->ypart = dmalloc<__half>((size_t)TILES * 8 * HID);
    p->sh_act = dmalloc<__half>((size_t)ROMAX * IW);
    p->sh_out = dmalloc<float>((size_t)ROMAX * HID);
    if (sh_tc_on())                                                       // the TC shared expert's operands (Y bf16 lives in sh_out)
    {
        p->tc_xc = dmalloc<uint8_t>((size_t)ROMAX * HID);
        p->tc_xs = dmalloc<uint8_t>((size_t)ROMAX * (HID / 32));
        p->tc_y13 = dmalloc<__nv_bfloat16>((size_t)ROMAX * 2 * IW);
        p->tc_ac = dmalloc<uint8_t>((size_t)ROMAX * SH_KP);
        p->tc_as = dmalloc<uint8_t>((size_t)ROMAX * (SH_KP / 32));
    }
    return p;
}
inline int prepare_moe(void* model, int i, int M, void*)
{
    Model& m = *(Model*)model;
    if (!m.layers[i].moe) return 0;
    const LayerState* s = (const LayerState*)m.layers[i].ds41_moe;
    if (!s || M < 1 || M > ROMAX) return ds41::NOTSUP;
    if (M > MTMAX)                                                        // the prefill paths' scratch exists before any capture / launch
    {
        static std::mutex mu;
        std::lock_guard<std::mutex> lk(mu);
        if (M >= RO_FROM) { if (!s->sc->ro) s->sc->ro = ro_alloc(); }
        else if (!s->sc->pw) s->sc->pw = pw_alloc();
    }
    if (M >= RO_FROM && gate_ro_on())                                    // the smem attribute of k_gate_logits_ro is set here (host init), never inside a capture
    {
        const cudaError_t a = gro_attr();
        if (a != cudaSuccess) return (int)a;
    }
    if (M > MTMAX && M <= SH1_MAX && sh_1pass_on())                     // the fc2 smem attribute is set here (host init), never inside a capture
    {
        const cudaError_t a = sh1_attr();
        if (a != cudaSuccess) return (int)a;
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------------ the ops
template <int NE, int TK, int PD, int KC = TK>
inline int router(Model& m, const LayerState& s, int M, cudaStream_t st)
{
    const int xg = (HID / 16 * 64 + 255) / 256;
    if constexpr (PD) mx4::build_xfrag_p<<<xg, 256, 0, st>>>(m.hc_y, M, HID, s.sc->xfrag);       // plain launch, the first of the chain
    else mx4::build_xfrag<<<xg, 256, 0, st>>>(m.hc_y, M, HID, s.sc->xfrag);
    launch(PD, dsr::k_gate_logits<NE, PD>, dim3(NE / 8), dim3(256), st, (const __nv_bfloat16*)s.gate, (const __nv_bfloat16*)m.hc_y, M, s.sc->logits);
    if constexpr (PD) launch(true, dsr::k_plan_p<NE, TK, KC>, dim3(1), dim3(dsr::PLAN_THREADS), st, (const float*)s.sc->logits, s.bias, M, (dsr::PlanT<TK>*)s.sc->plan);
    else launch(false, dsr::k_plan<NE, TK, KC>, dim3(1), dim3(dsr::PLAN_THREADS), st, (const float*)s.sc->logits, s.bias, M, (dsr::PlanT<TK>*)s.sc->plan);
    return (int)cudaGetLastError();
}
template <int PD>
inline int shared(const LayerState& s, int M, cudaStream_t st)
{
    dsh::Args SA{s.sh_w1, s.sh_w3, s.sh_w2, s.sc->xfrag, (1 << M) - 1, geom::SWIGLU_LIMIT, s.sc->sh_act, s.sc->sh_out};
    launch(PD, dsh::k_sh_fc1<HID, IW, 8, 1, 2, PD>, dim3(IW / 16), dim3(8 * 32), st, SA);
    launch(PD, dsh::k_sh_fc2<IW, HID, 8, 3, PD>, dim3(HID / 16 / 8), dim3(8 * 32), st, SA);
    return (int)cudaGetLastError();
}
template <int TK, int PD>
inline int experts(Model& m, const LayerState& s, int M, cudaStream_t st)
{
    using P = dsr::PlanT<TK>;
    const char* p = s.sc->plan;
    // the slot arrays live inside the device plan; their addresses are plain offsets (no device read here)
    mx4::Args RA{s.fc1, s.fc2, s.sc->xfrag,
                 (const int*)(p + offsetof(P, slot_expert)), (const int*)(p + offsetof(P, slot_mask)), (const float*)(p + offsetof(P, slot_w)),
                 (const int*)(p + offsetof(P, n_slots)), M * TK /* grid bound: CTAs of slots >= n_slots exit */, geom::SWIGLU_LIMIT,
                 s.sc->act, s.sc->ypart};
    launch(PD, mx4::k_fc1<HID, IW, 8, 1, 2, false, PD>, dim3(M * TK * (IW / 16)), dim3(8 * 32), st, RA);
    launch(PD, mx4::k_fc2<IW, HID, 8, 2, false, PD>, dim3(M * TK * (HID / 16 / 8)), dim3(8 * 32), st, RA);
    dsh::CombineArgsT<TK> CA{(const P*)p, s.sc->ypart, s.sc->sh_out, (void*)m.sc.part[1], M};
    launch(PD, dsh::k_combine<HID, 128, gv::O_BF16, TK>, dim3(M * HID / (8 * 128)), dim3(128), st, CA);
    return (int)cudaGetLastError();
}

// ---- M 9..32: NT = ceil(M/8) token tiles. Same op boundaries (router / shared / experts), the multi-tile kernels inside.
template <int NE, int TK, int PD, int KC = TK>
inline int router_mt(Model& m, const LayerState& s, int M, cudaStream_t st)
{
    const int nt = (M + 7) / 8;
    mx4::build_xfrag_mt<PD><<<(nt * HID * 4 + 255) / 256, 256, 0, st>>>(m.hc_y, M, HID, nt, s.sc->xfrag);      // plain launch, the first of the chain
    launch(PD, dsr::k_gate_logits_mt<NE, PD>, dim3(NE / 8), dim3(256), st, (const __nv_bfloat16*)s.gate, (const __nv_bfloat16*)m.hc_y, M, s.sc->logits);
    launch(PD, dsr::k_plan_mt<NE, TK, MTMAX, PD, KC>, dim3(1), dim3(dsr::PLAN_THREADS), st, (const float*)s.sc->logits, s.bias, M, (dsr::PlanMT<TK, MTMAX>*)s.sc->plan);
    return (int)cudaGetLastError();
}
template <int NT, int PD>
inline int shared_mt_nt(const LayerState& s, int M, cudaStream_t st)
{
    dsh::ArgsMT SA{s.sh_w1, s.sh_w3, s.sh_w2, s.sc->xfrag, M, geom::SWIGLU_LIMIT, s.sc->sh_act, s.sc->sh_out};
    launch(PD, dsh::k_sh_fc1_mt<HID, IW, 8, 1, 2, NT, PD>, dim3(IW / 16), dim3(8 * 32), st, SA);
    launch(PD, dsh::k_sh_fc2_mt<IW, HID, 8, 3, NT, PD>, dim3(HID / 16 / 8), dim3(8 * 32), st, SA);
    return (int)cudaGetLastError();
}
template <int PD>
inline int shared_mt(const LayerState& s, int M, cudaStream_t st)
{
    const int nt = (M + 7) / 8;
    return nt == 2 ? shared_mt_nt<2, PD>(s, M, st) : nt == 3 ? shared_mt_nt<3, PD>(s, M, st) : shared_mt_nt<4, PD>(s, M, st);
}
template <int TK, int NT, int PD>
inline int experts_mt_nt(Model& m, const LayerState& s, int M, cudaStream_t st)
{
    using P = dsr::PlanMT<TK, MTMAX>;
    const char* p = s.sc->plan;
    mx4::ArgsMT RA{s.fc1, s.fc2, s.sc->xfrag,
                   (const int*)(p + offsetof(P, slot_expert)), (const int*)(p + offsetof(P, slot_tile0)), (const int*)(p + offsetof(P, slot_nt)),
                   (const int*)(p + offsetof(P, tile_mask)), (const float*)(p + offsetof(P, tile_w)), (const int*)(p + offsetof(P, tile_tok)),
                   (const int*)(p + offsetof(P, n_slots)), M * TK /* grid bound: CTAs of slots >= n_slots exit */, geom::SWIGLU_LIMIT,
                   s.sc->act, s.sc->ypart};
    launch(PD, mx4::k_fc1_mt<HID, IW, 8, 1, mx4::FC1_MT_DEPTH<NT>, NT, 0, PD>, dim3(M * TK * (IW / 16)), dim3(8 * 32), st, RA);
    launch(PD, mx4::k_fc2_mt<IW, HID, 8, 2, NT, PD>, dim3(M * TK * (HID / 16 / 8)), dim3(8 * 32), st, RA);
    dsh::CombineArgsMT<TK, MTMAX> CA{(const P*)p, s.sc->ypart, s.sc->sh_out, (void*)m.sc.part[1], M};
    launch(PD, dsh::k_combine_mt<HID, 128, gv::O_BF16, TK, MTMAX>, dim3(M * HID / (8 * 128)), dim3(128), st, CA);
    return (int)cudaGetLastError();
}
template <int TK, int PD>
inline int experts_mt(Model& m, const LayerState& s, int M, cudaStream_t st)
{
    const int nt = (M + 7) / 8;
    return nt == 2 ? experts_mt_nt<TK, 2, PD>(m, s, M, st) : nt == 3 ? experts_mt_nt<TK, 3, PD>(m, s, M, st) : experts_mt_nt<TK, 4, PD>(m, s, M, st);
}

// ---- M 33..256 (prefill): the chunked path, plain launches. The routed kernels are the NT = 4 multi-tile ones, driven by the chunked plan's slots.
// xfrag holds ceil(M/32) * 4 tiles (rows >= M zero), shared by the shared expert's chunks and the routed fc1.
template <int NE, int TK, int MM, int KC = TK>
inline int router_pw(Model& m, const LayerState& s, int M, cudaStream_t st)
{
    PwScratch& q = *s.sc->pw;
    const int nt = (M + MTMAX - 1) / MTMAX * 4;
    mx4::build_xfrag_mt<0><<<(nt * HID * 4 + 255) / 256, 256, 0, st>>>(m.hc_y, M, HID, nt, q.xfrag);
    launch(false, dsr::k_gate_logits_mt<NE, 0>, dim3(NE / 8), dim3(256), st, (const __nv_bfloat16*)s.gate, (const __nv_bfloat16*)m.hc_y, M, q.logits);
    launch(false, dsr::k_plan_pw<NE, TK, MM, KC>, dim3(1), dim3(dsr::PLAN_THREADS), st, (const float*)q.logits, s.bias, M, (dsr::PlanMT<TK, MM>*)q.plan);
    if (elog_on()) launch(false, dsr::k_expert_log<NE, TK, MM, KC>, dim3(1), dim3(256), st, (const dsr::PlanMT<TK, MM>*)q.plan, M, s.layer, s.sc->elog);
    return (int)cudaGetLastError();
}
// M 33..SH1_MAX: the shared expert in one pass, NT = 6 tiles (the 32-row chunk loop below re-reads the 9.12 MB of weights once per chunk). 6 is the most
// fc1's ps tables take in the 48 KB static smem (6 * 8 KB); fc2's 6 tables are 55296 B, so it is the dynamic-smem copy k_sh_fc2_mt_dyn (ds_shared.cuh). Rows are independent
// inside a tile (GvGroup::run<NT> runs the same mma sequence per tile), so each row's bits equal the chunked path's.
// The dynamic-smem attribute of the NT = 6 fc2: a host call without stream work, made ONCE. prepare_moe makes the first call (host init, before any capture or launch);
// shared_pw_1pass reads the cached result, so no cudaFuncSetAttribute runs inside a stream capture.
inline cudaError_t sh1_attr()
{
    constexpr size_t SMEM2 = (size_t)SH1_NT * (IW / 16) * 64 * sizeof(u32);
    static const cudaError_t attr = cudaFuncSetAttribute(dsh::k_sh_fc2_mt_dyn<IW, HID, 8, 3, SH1_NT, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2);
    return attr;
}
inline int shared_pw_1pass(const LayerState& s, int M, cudaStream_t st)
{
    PwScratch& q = *s.sc->pw;
    constexpr size_t SMEM2 = (size_t)SH1_NT * (IW / 16) * 64 * sizeof(u32);
    dsh::ArgsMT SA{s.sh_w1, s.sh_w3, s.sh_w2, q.xfrag, M, geom::SWIGLU_LIMIT, q.sh_act, q.sh_out};
    auto* k2 = dsh::k_sh_fc2_mt_dyn<IW, HID, 8, 3, SH1_NT, 0>;
    const cudaError_t attr = sh1_attr();
    if (attr != cudaSuccess) return (int)attr;
    launch(false, dsh::k_sh_fc1_mt_m<HID, IW, 8, 1, 2, SH1_NT, 0>, dim3(IW / 16), dim3(8 * 32), st, SA);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(HID / 16 / 8); cfg.blockDim = dim3(8 * 32); cfg.dynamicSmemBytes = SMEM2; cfg.stream = st;
    cudaLaunchKernelEx(&cfg, k2, SA);
    return (int)cudaGetLastError();
}
inline int shared_pw(const LayerState& s, int M, cudaStream_t st)
{
    if (sh_1pass_on() && M <= SH1_MAX) return shared_pw_1pass(s, M, st);
    PwScratch& q = *s.sc->pw;
    for (int r0 = 0; r0 < M; r0 += MTMAX)             // 32-row chunks on shifted pointers: no kernel change; the shared weights are re-read per chunk
    {
        const int mc = M - r0 < MTMAX ? M - r0 : MTMAX;
        dsh::ArgsMT SA{s.sh_w1, s.sh_w3, s.sh_w2, q.xfrag + (size_t)(r0 / 8) * (HID * 4), mc, geom::SWIGLU_LIMIT,
                       q.sh_act + (size_t)r0 * IW, q.sh_out + (size_t)r0 * HID};
        launch(false, dsh::k_sh_fc1_mt<HID, IW, 8, 1, 2, 4, 0>, dim3(IW / 16), dim3(8 * 32), st, SA);
        launch(false, dsh::k_sh_fc2_mt<IW, HID, 8, 3, 4, 0>, dim3(HID / 16 / 8), dim3(8 * 32), st, SA);
    }
    return (int)cudaGetLastError();
}
template <int NE, int TK, int MM>
inline int experts_pw(Model& m, const LayerState& s, int M, int kc, cudaStream_t st)       // kc = the routes the planner kept per row (topk_cut() on the main layers, TK on the draft)
{
    using P = dsr::PlanMT<TK, MM>;
    PwScratch& q = *s.sc->pw;
    const char* p = q.plan;
    const int sb = dsr::pw_slot_bound(M * (pw_bound_on() ? kc : TK), NE);     // slots (32-token chunks) <= this; CTAs of slots >= n_slots exit (GR8_DS_MOE_PW_BOUND=1: R = M * kc live routes)
    mx4::ArgsMT RA{s.fc1, s.fc2, q.xfrag,
                   (const int*)(p + offsetof(P, slot_expert)), (const int*)(p + offsetof(P, slot_tile0)), (const int*)(p + offsetof(P, slot_nt)),
                   (const int*)(p + offsetof(P, tile_mask)), (const float*)(p + offsetof(P, tile_w)), (const int*)(p + offsetof(P, tile_tok)),
                   (const int*)(p + offsetof(P, n_slots)), sb, geom::SWIGLU_LIMIT, q.act, q.ypart};
    launch(false, mx4::k_fc1_mt<HID, IW, 8, 1, mx4::FC1_MT_DEPTH<4>, 4, 0, 0>, dim3(sb * (IW / 16)), dim3(8 * 32), st, RA);
    launch(false, mx4::k_fc2_mt<IW, HID, 8, 2, 4, 0>, dim3(sb * (HID / 16 / 8)), dim3(8 * 32), st, RA);
    dsh::CombineArgsMT<TK, MM> CA{(const P*)p, q.ypart, q.sh_out, (void*)m.sc.part[1], M};
    launch(false, dsh::k_combine_mt<HID, 128, gv::O_BF16, TK, MM>, dim3(M * HID / (8 * 128)), dim3(128), st, CA);
    return (int)cudaGetLastError();
}

// ---- M RO_FROM..ROMAX (prefill): the read-once path, plain launches. Planner = k_route_ro / k_scan_ro / k_place_ro (one slot per distinct expert),
// routed kernels = k_fc1_ro / k_fc2_ro (a CTA group per expert; the expert's weight records are read by one CTA each). xfrag holds ceil(M/32) * 4 tiles
// (rows >= M zero), shared by the shared expert's 32-row chunks and the routed fc1. The combine is the token-major k_combine_mt at MM = ROMAX.
template <int NE, int TK, int KC = TK>
inline int router_ro(Model& m, const LayerState& s, int M, cudaStream_t st)
{
    RoScratch& q = *s.sc->ro;
    using P = dsr::PlanMT<TK, ROMAX>;
    constexpr int MW = ROMAX / 32;
    const int nt = (M + MTMAX - 1) / MTMAX * 4;
    mx4::build_xfrag_mt<0><<<(nt * HID * 4 + 255) / 256, 256, 0, st>>>(m.hc_y, M, HID, nt, q.xfrag);
    if (NE == dsr::NEXP && gate_ro_on())                                   // GR8_DS_GATE_RO=1: the 8 x 8 register-tile gate (main layers only)
    {
        const cudaError_t a = gro_attr();
        if (a != cudaSuccess) return (int)a;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(NE / 8); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = dsr::GRO_SMEM; cfg.stream = st;
        cudaLaunchKernelEx(&cfg, dsr::k_gate_logits_ro<dsr::NEXP>, (const __nv_bfloat16*)s.gate, (const __nv_bfloat16*)m.hc_y, M, q.logits);
    }
    else launch(false, dsr::k_gate_logits_mt<NE, 0>, dim3(NE / 8), dim3(256), st, (const __nv_bfloat16*)s.gate, (const __nv_bfloat16*)m.hc_y, M, q.logits);
    cudaMemsetAsync(q.emask, 0, (size_t)NE * MW * sizeof(unsigned), st);
    launch(false, dsr::k_route_ro<NE, TK, ROMAX, KC>, dim3((M + 7) / 8), dim3(256), st, (const float*)q.logits, s.bias, M, (P*)q.plan, q.emask);
    launch(false, dsr::k_scan_ro<NE, TK, ROMAX>, dim3(1), dim3(NE), st, (P*)q.plan, (const unsigned*)q.emask, q.tile0_of);
    launch(false, dsr::k_place_ro<NE, TK, ROMAX, KC>, dim3((M * TK + 255) / 256), dim3(256), st, (P*)q.plan, (const unsigned*)q.emask, (const int*)q.tile0_of, M);
    if (ro_order_on()) launch(false, dsr::k_order_ro<NE, TK, ROMAX>, dim3(1), dim3(NE), st, (const P*)q.plan, q.order);
    if (elog_on()) launch(false, dsr::k_expert_log<NE, TK, ROMAX, KC>, dim3(1), dim3(256), st, (const P*)q.plan, M, s.layer, s.sc->elog);
    return (int)cudaGetLastError();
}
// GR8_DS_SH_TC=1: four launches, all on st (plain): act_quant(hc_y) -> fc1 GEMM -> swiglu + act_quant -> fc2 GEMM. Y bf16 [M][HID] into sh_out (aliased).
// Rows are independent (the GEMM's rows >= M read row M - 1 and are not stored), and nothing is split or atomic: deterministic, and a row's bits do not
// depend on M. A nonzero entry return (shape / alignment, never expected: the shapes are fixed) = NOTSUP, nothing after it launched.
inline int shared_ro_tc(const LayerState& s, const void* hc_y, int M, cudaStream_t st,
                        const uint8_t* gathered_codes=nullptr,const uint8_t* gathered_scales=nullptr)
{
    RoScratch& q = *s.sc->ro;
    constexpr int N13 = 2 * IW;
    if (!gathered_codes && ds41_sh_act_quant(hc_y, HID, q.tc_xc, HID, q.tc_xs, HID / 32, M, HID, st)) return ds41::NOTSUP;
    const uint8_t* xc=gathered_codes ? gathered_codes : q.tc_xc;
    const uint8_t* xs=gathered_scales ? gathered_scales : q.tc_xs;
    if (ds41_gemm_mx_tc(xc, HID, xs, HID / 32, s.tc_w13, s.tc_sw13, q.tc_y13, N13, M, N13, HID, st)) return ds41::NOTSUP;
    if (ds41_sh_swiglu_q(q.tc_y13, N13, IW, geom::SWIGLU_LIMIT, q.tc_ac, SH_KP, q.tc_as, SH_KP / 32, M, SH_KP, st)) return ds41::NOTSUP;
    if (ds41_gemm_mx_tc(q.tc_ac, SH_KP, q.tc_as, SH_KP / 32, s.tc_w2, s.tc_sw2, (void*)q.sh_out, HID, M, HID, SH_KP, st)) return ds41::NOTSUP;
    return (int)cudaGetLastError();
}
inline int shared_ro(const LayerState& s, const void* hc_y, int M, cudaStream_t st,
                     const uint8_t* gathered_codes=nullptr,const uint8_t* gathered_scales=nullptr)
{
    if (s.tc_w13) return shared_ro_tc(s, hc_y, M, st,gathered_codes,gathered_scales);   // GR8_DS_SH_TC=1
    RoScratch& q = *s.sc->ro;
#if DS41_SH_GL
    // the group loop (ds_shared_gl.cuh): two launches per layer, the same fc1 / fc2 bodies as the chunk loop below; the check in tests/ds_prefill_ro_test.cu is its gate
    static_assert(ROMAX % (DS41_SH_NT * 8) == 0, "group-loop chunks must tile RoScratch (xfrag, sh_act are sized for ROMAX rows)");
    dsh::ArgsGL GA{s.sh_w1, s.sh_w3, s.sh_w2, q.xfrag, M, geom::SWIGLU_LIMIT, q.sh_act, q.sh_out};
    return dsh::launch_shared_gl<HID, IW, HID, 8, 1, 2, 8, 3, DS41_SH_NT>(GA, DS41_SH_RG, st);
#else
    for (int r0 = 0; r0 < M; r0 += MTMAX)             // same 32-row chunks as shared_pw (functional, not fast: 2 launches per chunk, the shared weights re-read per chunk)
    {
        const int mc = M - r0 < MTMAX ? M - r0 : MTMAX;
        dsh::ArgsMT SA{s.sh_w1, s.sh_w3, s.sh_w2, q.xfrag + (size_t)(r0 / 8) * (HID * 4), mc, geom::SWIGLU_LIMIT,
                       q.sh_act + (size_t)r0 * IW, q.sh_out + (size_t)r0 * HID};
        launch(false, dsh::k_sh_fc1_mt<HID, IW, 8, 1, 2, 4, 0>, dim3(IW / 16), dim3(8 * 32), st, SA);
        launch(false, dsh::k_sh_fc2_mt<IW, HID, 8, 3, 4, 0>, dim3(HID / 16 / 8), dim3(8 * 32), st, SA);
    }
    return (int)cudaGetLastError();
#endif
}
// The two expert-loop launches (k_fc1_ro, k_fc2_ro) of one configuration; ORD = the k_order_ro launch order (RA.order), LA = the L2 lookahead (RA.la CTAs ahead for fc1, 2 x for fc2),
// HV = GR8_DS_MOE_L2HINT (0 off, 1 the fc1 weight loads carry the L2 evict_last hint, 2 fc1 and fc2); STG = GR8_DS_MOE_FC2STG (the fc2 act tile staged in one batch of loads).
template <bool ORD, bool LA, int KS, int CW, int NTG1, int W2, int D2, int NTG2, int HV = 0, bool STG = false>
inline void fc_ro_run(const mx4::ArgsMT& RA, int sb, int bstg, cudaStream_t st)
{
    constexpr size_t SMEM2 = (size_t)NTG2 * (IW / 16) * 64 * sizeof(u32);          // fc2 act staging: NTG2 * 9216 B
    constexpr size_t SMEM1 = 2u * KS * NTG1 * 4u * 64u * sizeof(u32);             // fc1 B ring: two k64 records per KS part and tile
    auto* k2 = mx4::k_fc2_ro<IW, HID, W2, D2, NTG2, ORD, LA, (HV == 2), STG>;
    if (SMEM2 > 48 * 1024) cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2);   // host-side attribute, no stream work
    if (bstg)
    {
        auto* k1 = mx4::k_fc1_ro_bstg<HID, IW, KS, CW, 1, NTG1, 0, ORD, LA, (HV >= 1)>;
        if (SMEM1 > 48 * 1024)
        {
            const cudaError_t ce = cudaFuncSetAttribute(k1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM1);
            if (ce != cudaSuccess) die(std::string("GR8_DS_MOE_FC1BSTG shared-memory attribute: ") + cudaGetErrorString(ce));
        }
        cudaLaunchConfig_t cfg1 = {};
        cfg1.gridDim = dim3(sb * (IW / 16 / CW)); cfg1.blockDim = dim3(KS * CW * 32); cfg1.dynamicSmemBytes = SMEM1; cfg1.stream = st;
        const cudaError_t ce = cudaLaunchKernelEx(&cfg1, k1, RA);
        if (ce != cudaSuccess) die(std::string("GR8_DS_MOE_FC1BSTG launch: ") + cudaGetErrorString(ce));
    }
    else launch(false, mx4::k_fc1_ro<HID, IW, KS, CW, 1, NTG1, 0, ORD, LA, (HV >= 1)>, dim3(sb * (IW / 16 / CW)), dim3(KS * CW * 32), st, RA);
    mx4::ArgsMT R2 = RA;
    R2.la = RA.la * 2;                                                              // fc2 runs 2 CTAs per SM
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(sb * (HID / 16 / W2)); cfg.blockDim = dim3(W2 * 32); cfg.dynamicSmemBytes = SMEM2; cfg.stream = st;
    cudaLaunchKernelEx(&cfg, k2, R2);
}
template <bool ORD, bool LA, int KS, int CW, int NTG1, int W2, int D2, int NTG2>
inline void fc_ro_hint(const mx4::ArgsMT& RA, int sb, int hv, int stg, int bstg, cudaStream_t st)
{
    if (stg)
    {
        if (hv == 2) fc_ro_run<ORD, LA, KS, CW, NTG1, W2, D2, NTG2, 2, true>(RA, sb, bstg, st);
        else if (hv == 1) fc_ro_run<ORD, LA, KS, CW, NTG1, W2, D2, NTG2, 1, true>(RA, sb, bstg, st);
        else fc_ro_run<ORD, LA, KS, CW, NTG1, W2, D2, NTG2, 0, true>(RA, sb, bstg, st);
    }
    else if (hv == 2) fc_ro_run<ORD, LA, KS, CW, NTG1, W2, D2, NTG2, 2>(RA, sb, bstg, st);
    else if (hv == 1) fc_ro_run<ORD, LA, KS, CW, NTG1, W2, D2, NTG2, 1>(RA, sb, bstg, st);
    else fc_ro_run<ORD, LA, KS, CW, NTG1, W2, D2, NTG2, 0>(RA, sb, bstg, st);
}
template <int NE, int TK>
inline int experts_ro(Model& m, const LayerState& s, int M, cudaStream_t st)
{
    using P = dsr::PlanMT<TK, ROMAX>;
    RoScratch& q = *s.sc->ro;
    const char* p = q.plan;
    constexpr int KS = DS41_RO_KS, CW = DS41_RO_CW, NTG1 = DS41_RO_NTG1, W2 = DS41_RO_W2, D2 = DS41_RO_D2, NTG2 = DS41_RO_NTG2;
    const int sb = dsr::ro_slots_bound(M * TK, NE);    // slots (distinct experts) <= this; CTAs of slots >= n_slots exit
    const bool ord = ro_order_on();
    const int la = ro_l2la();
    const int hv = ro_l2hint(), stg = ro_fc2stg(), bstg = ro_fc1bstg();
    mx4::ArgsMT RA{s.fc1, s.fc2, q.xfrag,
                   (const int*)(p + offsetof(P, slot_expert)), (const int*)(p + offsetof(P, slot_tile0)), (const int*)(p + offsetof(P, slot_nt)),
                   (const int*)(p + offsetof(P, tile_mask)), (const float*)(p + offsetof(P, tile_w)), (const int*)(p + offsetof(P, tile_tok)),
                   (const int*)(p + offsetof(P, n_slots)), sb, geom::SWIGLU_LIMIT, q.act, q.ypart, ord ? (const int*)q.order : nullptr, la};
    if (ord) { if (la) fc_ro_hint<true, true, KS, CW, NTG1, W2, D2, NTG2>(RA, sb, hv, stg, bstg, st); else fc_ro_hint<true, false, KS, CW, NTG1, W2, D2, NTG2>(RA, sb, hv, stg, bstg, st); }
    else { if (la) fc_ro_hint<false, true, KS, CW, NTG1, W2, D2, NTG2>(RA, sb, hv, stg, bstg, st); else fc_ro_hint<false, false, KS, CW, NTG1, W2, D2, NTG2>(RA, sb, hv, stg, bstg, st); }
    dsh::CombineArgsMT<TK, ROMAX> CA{(const P*)p, q.ypart, q.sh_out, (void*)m.sc.part[1], M};
    if (s.tc_w13) launch(false, dsh::k_combine_mt_shb<HID, 128, gv::O_BF16, TK, ROMAX>, dim3(M * HID / (8 * 128)), dim3(128), st, CA);   // GR8_DS_SH_TC=1: sh_out holds bf16 Y
    else launch(false, dsh::k_combine_mt<HID, 128, gv::O_BF16, TK, ROMAX>, dim3(M * HID / (8 * 128)), dim3(128), st, CA);
    return (int)cudaGetLastError();
}

inline const LayerState* state_of(void* model, int i, int rows)
{
    const LayerState* s = (const LayerState*)((Model*)model)->layers[i].ds41_moe;
    return s && rows >= 1 && (rows <= MTMAX || (rows < RO_FROM && s->sc->pw) || (rows >= RO_FROM && rows <= ROMAX && s->sc->ro)) ? s : nullptr;
}
// GR8_L2PF (kernels/l2pf.cuh, ds41_l2pf.h): layer i's gate and shared-expert records as prefetch targets (T_ROUTER, T_SHARED);
// nothing for a layer without MoE state. Host only, called by ds41_l2pf_window (ds41_step.cuh) while a collective is awaited.
inline void l2pf_add(const Model& m, int i, l2pf::Plan& P)
{
    const LayerState* s = (const LayerState*)m.layers[i].ds41_moe;
    if (s && !s->tr) ds41l2pf::add_moe(P, s->ne, HID, IW, s->gate, s->sh_w1, s->sh_w3, s->sh_w2);
}
inline int op_router(void* model, int i, int rows, void* stream)
{
    const LayerState* s = state_of(model, i, rows);
    if (!s) return ds41::NOTSUP;
    Model& m = *(Model*)model;
    cudaStream_t st = (cudaStream_t)stream;
    if (s->tr && rows > MTMAX) return 0;                                // trellis prefill routes each 32-row pass (trellis_prefill)
    const bool big = s->ne == dsr::NEXP, pd = pdl_on();
    // the main layers (384 experts) take the top-k cut GR8_DS_MOE_TOPK; the draft layers (128, top-3) keep their k
    auto cut = [&](auto f) { return with_topk_cut(topk_for_step(m), f); };
    if (rows >= RO_FROM)
        return big ? cut([&](auto kc) { return router_ro<dsr::NEXP, dsr::TOPK, decltype(kc)::value>(m, *s, rows, st); })
                   : router_ro<geom::DS_NEXP, geom::DS_TOPK>(m, *s, rows, st);
    if (rows > MTMAX)
        return big ? (rows <= 128 ? cut([&](auto kc) { return router_pw<dsr::NEXP, dsr::TOPK, 128, decltype(kc)::value>(m, *s, rows, st); })
                                  : cut([&](auto kc) { return router_pw<dsr::NEXP, dsr::TOPK, 256, decltype(kc)::value>(m, *s, rows, st); }))
                   : (rows <= 128 ? router_pw<geom::DS_NEXP, geom::DS_TOPK, 128>(m, *s, rows, st) : router_pw<geom::DS_NEXP, geom::DS_TOPK, 256>(m, *s, rows, st));
    if (rows > MMAX)
        return big ? (pd ? cut([&](auto kc) { return router_mt<dsr::NEXP, dsr::TOPK, 1, decltype(kc)::value>(m, *s, rows, st); })
                         : cut([&](auto kc) { return router_mt<dsr::NEXP, dsr::TOPK, 0, decltype(kc)::value>(m, *s, rows, st); }))
                   : (pd ? router_mt<geom::DS_NEXP, geom::DS_TOPK, 1>(m, *s, rows, st) : router_mt<geom::DS_NEXP, geom::DS_TOPK, 0>(m, *s, rows, st));
    return big ? (pd ? cut([&](auto kc) { return router<dsr::NEXP, dsr::TOPK, 1, decltype(kc)::value>(m, *s, rows, st); })
                     : cut([&](auto kc) { return router<dsr::NEXP, dsr::TOPK, 0, decltype(kc)::value>(m, *s, rows, st); }))
               : (pd ? router<geom::DS_NEXP, geom::DS_TOPK, 1>(m, *s, rows, st) : router<geom::DS_NEXP, geom::DS_TOPK, 0>(m, *s, rows, st));
}
inline int op_shared(void* model, int i, int rows, void* stream)
{
    const LayerState* s = state_of(model, i, rows);
    if (!s) return ds41::NOTSUP;
    cudaStream_t st = (cudaStream_t)stream;
    if (s->tr) return 0;                                                // the trellis shared expert runs inside K1
    if (rows >= RO_FROM) {
        Model& m=*(Model*)model;
        const bool direct=m.ds41_y_quant_ready && m.ds41_rowshard_chunk && rows==4096;
        return shared_ro(*s,m.hc_y,rows,st,direct?m.ds41_y_codes:nullptr,direct?m.ds41_y_scales:nullptr);
    }
    if (rows > MTMAX) return shared_pw(*s, rows, st);
    return pdl_on() ? (rows > MMAX ? shared_mt<1>(*s, rows, st) : shared<1>(*s, rows, st))
                    : (rows > MMAX ? shared_mt<0>(*s, rows, st) : shared<0>(*s, rows, st));
}
inline int op_experts(void* model, int i, int rows, void* stream)
{
    const LayerState* s = state_of(model, i, rows);
    if (!s) return ds41::NOTSUP;
    Model& m = *(Model*)model;
    cudaStream_t st = (cudaStream_t)stream;
    if (s->tr) return rows > MTMAX ? trellis_prefill(m, *s, rows, st) : k1_rows(m, *s, m.hc_y, rows, m.sc.part[1], st);
    const bool big = s->tk == dsr::TOPK, pd = pdl_on();
    if (rows >= RO_FROM)
        return big ? experts_ro<dsr::NEXP, dsr::TOPK>(m, *s, rows, st) : experts_ro<geom::DS_NEXP, geom::DS_TOPK>(m, *s, rows, st);
    if (rows > MTMAX)
        return big ? (rows <= 128 ? experts_pw<dsr::NEXP, dsr::TOPK, 128>(m, *s, rows, topk_for_step(m), st) : experts_pw<dsr::NEXP, dsr::TOPK, 256>(m, *s, rows, topk_for_step(m), st))
                   : (rows <= 128 ? experts_pw<geom::DS_NEXP, geom::DS_TOPK, 128>(m, *s, rows, geom::DS_TOPK, st) : experts_pw<geom::DS_NEXP, geom::DS_TOPK, 256>(m, *s, rows, geom::DS_TOPK, st));
    if (rows > MMAX)
        return big ? (pd ? experts_mt<dsr::TOPK, 1>(m, *s, rows, st) : experts_mt<dsr::TOPK, 0>(m, *s, rows, st))
                   : (pd ? experts_mt<geom::DS_TOPK, 1>(m, *s, rows, st) : experts_mt<geom::DS_TOPK, 0>(m, *s, rows, st));
    return big ? (pd ? experts<dsr::TOPK, 1>(m, *s, rows, st) : experts<dsr::TOPK, 0>(m, *s, rows, st))
               : (pd ? experts<geom::DS_TOPK, 1>(m, *s, rows, st) : experts<geom::DS_TOPK, 0>(m, *s, rows, st));
}

// Distinct-expert counter, host side. elog_read copies the device ring (one synchronous cudaMemcpy: call it between windows or at exit, never inside a
// capture) and returns the records in call order (oldest first; at most ELOG_N, the newest when the ring wrapped). elog_reset zeroes the ring.
// elog_print: one line per record + the mean distinct experts per (layer, rows). Records exist for prefill calls only (rows > 32).
inline int elog_read(const Model* m, std::vector<dsr::ExpertLogRec>& out, bool reset = true)      // reset = true: the next read starts a new chunk (cursor -> 0)
{
    dsr::ExpertLog* d = scratch_of(m)->elog;
    std::vector<char> h(sizeof(dsr::ExpertLog));
    if (cudaMemcpy(h.data(), d, h.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    const dsr::ExpertLog* L = (const dsr::ExpertLog*)h.data();
    const unsigned n = L->cursor < (unsigned)dsr::ELOG_N ? L->cursor : (unsigned)dsr::ELOG_N;
    out.assign(L->rec, L->rec + n);
    std::sort(out.begin(), out.end(), [](const dsr::ExpertLogRec& a, const dsr::ExpertLogRec& b) { return a.seq < b.seq; });
    if (reset) cudaMemset(d, 0, sizeof(unsigned));
    return (int)n;
}
// "a,b,c,..." distinct experts per record in call order (a 40-layer prefill chunk -> 40 numbers, layers in execution order); layer numbers are in the records.
inline std::string elog_csv(const std::vector<dsr::ExpertLogRec>& v)
{
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i].distinct);
    return s;
}
// elog_begin(model, first): optional, host-synchronous, before a window's launches: the window's first position, copied into every record of that window.
inline void elog_begin(const Model* m, int first) { cudaMemcpy((char*)scratch_of(m)->elog + offsetof(dsr::ExpertLog, win_start), &first, sizeof first, cudaMemcpyHostToDevice); }
inline void elog_reset(const Model* m) { cudaMemset(scratch_of(m)->elog, 0, sizeof(dsr::ExpertLog)); }
inline void elog_print(const Model* m, FILE* f)
{
    std::vector<dsr::ExpertLogRec> v;
    const int n = elog_read(m, v, false);
    fprintf(f, "moe distinct experts per prefill call (%d records): seq layer rows distinct maxn routes start\n", n);
    for (const auto& r : v) fprintf(f, "moe_elog %d %d %d %d %d %d %d\n", r.seq, r.layer, r.rows, r.distinct, r.maxn, r.routes, r.start);
    for (size_t i = 0; i < v.size(); ++i)             // mean per rows
    {
        double sum = 0; int cnt = 0;
        bool first = true;
        for (size_t j = 0; j < i; ++j) if (v[j].rows == v[i].rows) first = false;
        if (!first) continue;
        for (const auto& r : v) if (r.rows == v[i].rows) { sum += r.distinct; ++cnt; }
        fprintf(f, "moe_elog_mean rows %d: %.2f distinct experts over %d layer calls\n", v[i].rows, sum / cnt, cnt);
    }
}

// One call fills the three slots and the two hooks of ds41_table().
inline void fill(ds41::Table& t)
{
    t.fn[ds41::ST_ROUTER] = &op_router;
    t.fn[ds41::ST_SHARED] = &op_shared;
    t.fn[ds41::ST_EXPERTS] = &op_experts;
    t.bind_moe = &bind_moe;
    t.prepare_moe = &prepare_moe;
}
}   // namespace ds41moe
#endif
