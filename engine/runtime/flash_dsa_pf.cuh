// Flash prefill DSA block (INTERFACES.md §8.2-8.3, PREFILL-PLAN.md §2):
//   bind_dsa_prefill(m, P, st)
//   pf_dsa_block_gemm(m, P, i, M, context, st[, mark])
//
// Input:  P.xn [M][4096] FP32, P.x [M][4096] FP16, P.pos [M] int
// Output: P.part[0] [M][4096] BF16 (this rank's DSA o_proj partial; caller publishes & sums),
//         plus latent KV + kpool index keys + pool gates appended to y.ab's caches for all M rows.
//
// Projections:
//   - q_a, kv_a, idx.wk, idx.weights_proj, index_kpool_compress_gate, q_a_norm, idx.wq_b run in 32-row (NT=4)
//     multi-tile GEMV launches on the exact decode records (0-ULP bit-identical to pf_dsa_block_p1, so the persistent
//     KV cache, kpool cache and top-512 pool selection match decode bit-for-bit). By default the six input
//     projections run as one full-M grid launch each, with the same 32-row CTA math and reduction (GR8_PF_DSA_FULL_GV=0:
//     one launch per 32 rows).
//     GR8_PF_DSA_F16GEMM=1 (opt-in, needs full-GV): q_a, kv_a, idx_k, idx_q (MXFP8) run as cuBLAS fp16 GEMMs and
//     idx_w + idx_g (BF16W) as one bf16 GEMM, on matrices decoded at load (build_flash_prefill -> f16_prepare,
//     flash_dsa_f16.cuh); same operands, fp32 summation order differs (near-byte).
//   - W8d q_b.trellis and o_proj.trellis run k_gemm_tr_h16 (fp16-MMA on raw K4 tiles, pure-fp32 Hadamard epilogue)
//     across all M rows in a single launch (override GR8_PF_DSA_PROJ_P1=1 to force 32-row GEMVs everywhere).
//   - Lossless BF16 q_b.gv and o_proj.gv run 32-row (NT=4) multi-tile GEMV launches (0-ULP bit-identical to decode).
// Attention:
//   - Tensor-Core G6AttnPrefill (g6_attn_bind_prefill_run) over all M rows in a single call, sharing one
//     G6AttnPrefillScratch workspace across all DSA layers (override GR8_FPF_ATTN_P1=1 to force per-MX-chunk decode).
#pragma once
#if G6_MODEL_FLASH
#include <map>
#include <memory>
#include <type_traits>
#include <vector>
#include "rank.cuh"
#include "../attn/prefill.h"
#include "../attn/ops.h"
#include "flash_pool_owner.cuh"
#include "flash_dsa_full_schedule.h"
#include "../kernels/bind_prefill.cuh"
#include "flash_dsa_f16.cuh"
#include "../knobs-rs/gr8_knobs.h"

extern "C" void* g6_attn_bind_prefill_output(const G6AttnPrefill*, const G6AttnBinding*, int rows);
extern "C" int* g6_attn_bind_prefill_indices(const G6AttnPrefill*, const G6AttnBinding*, int rows);

namespace g6rt
{

namespace flash_dsa_pf
{

constexpr int GV_NT = 4;
constexpr int GV_ROWS = GV_NT * 8;

inline bool full_gv_on()
{
    // default ON (byte-equal to the 32-row launches, 8L and 45L TP2); GR8_PF_DSA_FULL_GV=0 restores the per-32-row launches
    return knob::pf_dsa_full_gv();
}

// One grid row reproduces one old 32-row launch: same GV decoder, MMA,
// split-K accumulation, and output conversion. Only host launch count changes.
template <class C, int THREADS>
__global__ void __launch_bounds__(THREADS) k_gv_full(const uint8_t* w, const u32* xfrag, int K, int ot,
                                                      int tpc, int ks, int M, void* y, int ld, int odt,
                                                      const float* rscale)
{
    const int m0 = flash_dsa_full_schedule::first_row(blockIdx.y);
    const int r = flash_dsa_full_schedule::active_rows(M, blockIdx.y);
    const u32* xf = xfrag + flash_dsa_full_schedule::fragment_words(blockIdx.y, K);
    __half* out = static_cast<__half*>(y) + flash_dsa_full_schedule::output_elements(blockIdx.y, ld);
    __shared__ float4 red[32][32];
    // Match the old NT choice on the final partial grid row. In particular,
    // do not read fragment tables for token rows past M.
    if (r <= 8) gv_gemv_body<C, 1>(w, xf, K, ot, tpc, ks, r, out, ld, odt, red, blockIdx.x, rscale);
    else if (r <= 16) gv_gemv_body<C, 2>(w, xf, K, ot, tpc, ks, r, out, ld, odt, red, blockIdx.x, rscale);
    else if (r <= 24) gv_gemv_body<C, 3>(w, xf, K, ot, tpc, ks, r, out, ld, odt, red, blockIdx.x, rscale);
    else gv_gemv_body<C, GV_NT>(w, xf, K, ot, tpc, ks, r, out, ld, odt, red, blockIdx.x, rscale);
}

inline void launch_gv_full(const bind::Gv& g, const u32* xfrag, int M, void* y, cudaStream_t st)
{
    if (!g.on || M <= 0) return;
    if (g.odt != gv::O_F16) die("DSA full GV requires FP16 output");
    const int ot = g.O / 16;
    const bool b512 = g.ks <= 16;
    auto go = [&](auto* kern) {
        const int tpc = bind::gv_fit_tpc(kern, g.tpc, g.ks);
        const dim3 grid((ot + tpc - 1) / tpc, flash_dsa_full_schedule::grid_rows(M));
        kern<<<grid, tpc * g.ks * 32, 0, st>>>(g.w, xfrag, g.K, ot, tpc, g.ks, M, y, g.ld, g.odt, g.rscale);
    };
    if (g.codec == bind::INT4) {
        if (b512) go(k_gv_full<Int4v, 512>); else go(k_gv_full<Int4v, 1024>);
    } else if (g.codec == bind::MXFP8) {
        if (b512) go(k_gv_full<Mxfp8, 512>); else go(k_gv_full<Mxfp8, 1024>);
    } else if (g.codec == bind::FP8R) {
        if (b512) go(k_gv_full<Fp8r, 512>); else go(k_gv_full<Fp8r, 1024>);
    } else {
        if (b512) go(k_gv_full<Bf16w, 512>); else go(k_gv_full<Bf16w, 1024>);
    }
}

inline dtr::Mat mat_from_gemv(const dtr::Gemv& gv)
{
    dtr::Mat m;
    if (!gv.on) return m;
    m.w = gv.a.w[0]; m.sB = gv.a.sB; m.sG = gv.a.sG;
    m.K = gv.a.K; m.N = gv.a.N;
    m.suh = gv.a.suh[0]; m.svh = gv.a.svh[0];
    return m;
}

inline void launch_xfrag(const __half* src, int K, int M, u32* f16, u32* bf16, cudaStream_t st)
{
    bind::Xfrag a{};
    a.on = 1; a.src = src; a.ld = K; a.K = K; a.M = M; a.f16 = f16; a.bf16 = bf16;
    bind::k_xfrag<<<dim3((K / 16 * 8 + 255) / 256, (M + 7) / 8), 256, 0, st>>>(a);
}

inline void launch_gv_rows(const bind::Gv& g, const u32* xfrag, int M, void* y, cudaStream_t st)
{
    if (!g.on || M <= 0) return;
    const int ot = g.O / 16;
    const bool b512 = g.ks <= 16;
    auto run_nt = [&](auto N) {
        constexpr int NTv = decltype(N)::value;
        auto go = [&](auto* kern) {
            const int tpc = NTv == 1 ? g.tpc : bind::gv_fit_tpc(kern, g.tpc, g.ks);
            kern<<<(ot + tpc - 1) / tpc, tpc * g.ks * 32, 0, st>>>(g.w, xfrag, g.K, ot, tpc, g.ks, M, y, g.ld, g.odt, g.rscale);
        };
        if (g.codec == bind::INT4) { if constexpr (NTv == 1) go(gv_gemv<Int4v, 1>); else if (b512) go(gv_gemv_lb512<Int4v, NTv>); else go(gv_gemv_lb<Int4v, NTv>); }
        else if (g.codec == bind::MXFP8) { if constexpr (NTv == 1) go(gv_gemv<Mxfp8, 1>); else if (b512) go(gv_gemv_lb512<Mxfp8, NTv>); else go(gv_gemv_lb<Mxfp8, NTv>); }
        else if (g.codec == bind::FP8R) { if constexpr (NTv == 1) go(gv_gemv<Fp8r, 1>); else if (b512) go(gv_gemv_lb512<Fp8r, NTv>); else go(gv_gemv_lb<Fp8r, NTv>); }
        else { if constexpr (NTv == 1) go(gv_gemv<Bf16w, 1>); else if (b512) go(gv_gemv_lb512<Bf16w, NTv>); else go(gv_gemv_lb<Bf16w, NTv>); }
    };
    const int nt = (M + 7) / 8;
    if (nt <= 1) run_nt(std::integral_constant<int, 1>{});
    else if (nt == 2) run_nt(std::integral_constant<int, 2>{});
    else if (nt == 3) run_nt(std::integral_constant<int, 3>{});
    else run_nt(std::integral_constant<int, 4>{});
}

struct State
{
    int MP = 0;
    __half *q_a = nullptr, *q_a_n = nullptr, *kv_a = nullptr, *q_b = nullptr;
    __half *idx_q = nullptr, *idx_k = nullptr, *idx_w = nullptr, *idx_g = nullptr;
    __half *attn_p1 = nullptr;
    uint8_t *x8_qa = nullptr, *x8_attn = nullptr;
    u32 *xf_x = nullptr, *xb_x = nullptr, *xf_qa = nullptr, *xb_qa = nullptr, *xf_attn = nullptr;
    u32 *xf_full = nullptr, *xb_full = nullptr, *xq_full = nullptr;
    std::vector<G6AttnPrefill*> pf;

    ~State()
    {
        for (G6AttnPrefill* p : pf)
            if (p) g6_attn_bind_prefill_release(p);
        if (q_a) cudaFree(q_a);
        if (q_a_n) cudaFree(q_a_n);
        if (kv_a) cudaFree(kv_a);
        if (q_b) cudaFree(q_b);
        if (idx_q) cudaFree(idx_q);
        if (idx_k) cudaFree(idx_k);
        if (idx_w) cudaFree(idx_w);
        if (idx_g) cudaFree(idx_g);
        if (attn_p1) cudaFree(attn_p1);
        if (x8_qa) cudaFree(x8_qa);
        if (x8_attn) cudaFree(x8_attn);
        if (xf_x) cudaFree(xf_x);
        if (xb_x) cudaFree(xb_x);
        if (xf_qa) cudaFree(xf_qa);
        if (xb_qa) cudaFree(xb_qa);
        if (xf_attn) cudaFree(xf_attn);
        if (xf_full) cudaFree(xf_full);
        if (xb_full) cudaFree(xb_full);
        if (xq_full) cudaFree(xq_full);
    }
};

// GR8_PF_DSA_F16GEMM: the decoded matrices of one model (shared by its prefill lanes) and, per prefill (lane), a
// cuBLAS handle with its own workspace plus the bf path's rows / fp32 out. A null Mat::w keeps that projection on GV.
struct F16Layer
{
    flash_dsa_f16::Mat qa, kva, ik, iq, wg;   // wg = idx_w stacked on idx_g
};
struct F16Lane
{
    cublasHandle_t blas = nullptr;
    void* ws = nullptr;
    __nv_bfloat16* rows = nullptr;   // [MP][HID] bf16
    float* y32 = nullptr;            // [MP][widest decoded matrix's O] fp32 cuBLAS output
    int MP = 0;
};
struct F16Set
{
    bool built = false;
    int on = 0, total = 0;
    std::vector<F16Layer> L;
    std::map<const void*, F16Lane> lanes;
    ~F16Set()
    {
        for (F16Layer& y : L)
            for (flash_dsa_f16::Mat* w : {&y.qa, &y.kva, &y.ik, &y.iq, &y.wg})
                if (w->w) cudaFree(w->w);
        for (auto& [p, ln] : lanes)
        {
            if (ln.blas) cublasDestroy(ln.blas);
            for (void* q : {ln.ws, static_cast<void*>(ln.rows), static_cast<void*>(ln.y32)})
                if (q) cudaFree(q);
        }
    }
};
inline std::map<const Model*, F16Set>& f16_sets()
{
    static std::map<const Model*, F16Set> s;
    return s;
}
constexpr size_t F16_WS = size_t(32) << 20;   // cuBLAS workspace per lane handle
static_assert(F16_WS == flash_dsa_f16::GEMMEX_WS, "the GR8_PF_DSA_LT heuristic query assumes the GemmEx workspace");

// load time (build_flash_prefill), and a no-op check per DSA block after that: decode every bound DSA layer's
// projections once per model, then the lane's handle and buffers for MP rows. Logs the device-memory delta.
inline F16Set* f16_prepare(Model& m, const void* lane, int MP, cudaStream_t st)
{
    if (!flash_dsa_f16::on() || !full_gv_on()) return nullptr;
    F16Set& f = f16_sets()[&m];
    auto it = f.lanes.find(lane);
    if (f.built && it != f.lanes.end() && it->second.MP >= MP) return &f;
    size_t free0 = 0, free1 = 0, total = 0;
    G6RT_CK(cudaStreamSynchronize(st));
    G6RT_CK(cudaMemGetInfo(&free0, &total));
    int wg_o = 0, o32 = 0;
    if (!f.built)
    {
        f.built = true;
        f.L.assign(m.nl, {});
        for (int i = 0; i < m.nl; ++i)
        {
            Layer& y = m.layers[i];
            if (y.kda || !y.built[1]) continue;
            const bind::Layer& b1 = y.L[1];
            F16Layer& d = f.L[i];
            f.on += flash_dsa_f16::build(b1.q_a, d.qa, "q_a", y.l, st);
            f.on += flash_dsa_f16::build(b1.kv_a, d.kva, "kv_a", y.l, st);
            f.on += flash_dsa_f16::build(b1.idx_k, d.ik, "idx_k", y.l, st);
            f.on += flash_dsa_f16::build(b1.idx_q, d.iq, "idx_q", y.l, st);
            f.on += flash_dsa_f16::build_bf_cat(b1.idx_w, b1.idx_g, d.wg, st);
            f.total += 5;
        }
    }
    for (const F16Layer& d : f.L) {
        wg_o = std::max(wg_o, d.wg.w ? d.wg.O : 0);
        for (const flash_dsa_f16::Mat* w : {&d.qa, &d.kva, &d.ik, &d.iq, &d.wg})
            o32 = std::max(o32, w->w ? w->O : 0);
    }
    F16Lane& ln = f.lanes[lane];
    if (!ln.blas)
    {
        if (cublasCreate(&ln.blas) != CUBLAS_STATUS_SUCCESS) die("flash_dsa_f16: cublasCreate failed");
        ln.ws = dmalloc<uint8_t>(F16_WS);
        if (cublasSetWorkspace(ln.blas, ln.ws, F16_WS) != CUBLAS_STATUS_SUCCESS) die("flash_dsa_f16: cublasSetWorkspace failed");
    }
    if (o32 && ln.MP < MP)
    {
        if (ln.rows) cudaFree(ln.rows);
        if (ln.y32) cudaFree(ln.y32);
        ln.rows = wg_o ? dmalloc<__nv_bfloat16>(static_cast<size_t>(MP) * HID) : nullptr;
        ln.y32 = dmalloc<float>(static_cast<size_t>(MP) * o32);
    }
    ln.MP = std::max(ln.MP, MP);
    flash_dsa_f16::lt_prepare(MP);   // GR8_PF_DSA_LT: every row-count plan now, none on the serve path
    G6RT_CK(cudaStreamSynchronize(st));
    G6RT_CK(cudaMemGetInfo(&free1, &total));
    std::fprintf(stderr, "# DSA f16 gemm: %d of %d projections on cuBLAS; lane %d MP %d; device memory +%.1f MiB\n", f.on,
                 f.total, static_cast<int>(f.lanes.size()), MP, (static_cast<double>(free0) - static_cast<double>(free1)) / 1048576.0);
    return &f;
}

inline std::map<std::pair<const Model*, const void*>, std::shared_ptr<State>>& states()
{
    static std::map<std::pair<const Model*, const void*>, std::shared_ptr<State>> s;
    return s;
}

template <class Prefill>
inline State& ensure_state(Model& m, Prefill& P, cudaStream_t st)
{
    auto& ptr = states()[{&m, static_cast<const void*>(&P)}];
    if (!ptr || ptr->MP < P.MP)
    {
        ptr = std::make_shared<State>();
        ptr->MP = P.MP;
        const size_t mp = static_cast<size_t>(P.MP);
        const size_t rb = (mp + 15) / 16;
        ptr->q_a = dmalloc<__half>(mp * QA);
        ptr->q_a_n = dmalloc<__half>(mp * QA);
        ptr->kv_a = dmalloc<__half>(mp * KVA);
        ptr->q_b = dmalloc<__half>(mp * g6nr::QBW);
        ptr->idx_q = dmalloc<__half>(mp * IDXQ);
        ptr->idx_k = dmalloc<__half>(mp * IDXK);
        ptr->idx_w = dmalloc<__half>(mp * IDXW);
        ptr->idx_g = dmalloc<__half>(mp * geom::IDX_DIM);
        ptr->x8_qa = dmalloc<uint8_t>(rb * (QA / 32) * 1024);
        ptr->x8_attn = dmalloc<uint8_t>(rb * (g6nr::QBW / 32) * 1024);
        ptr->xf_x = dmalloc<u32>(static_cast<size_t>(GV_NT) * (HID / 16) * 64);
        ptr->xb_x = dmalloc<u32>(static_cast<size_t>(GV_NT) * (HID / 16) * 64);
        ptr->xf_qa = dmalloc<u32>(static_cast<size_t>(GV_NT) * (QA / 16) * 64);
        ptr->xb_qa = dmalloc<u32>(static_cast<size_t>(GV_NT) * (QA / 16) * 64);
        ptr->xf_attn = dmalloc<u32>(static_cast<size_t>(GV_NT) * (g6nr::QBW / 16) * 64);
        if (full_gv_on()) {
            const size_t xwords = flash_dsa_full_schedule::full_fragment_words(P.MP, HID);
            ptr->xf_full = dmalloc<u32>(xwords);
            ptr->xb_full = dmalloc<u32>(xwords);
            ptr->xq_full = dmalloc<u32>(flash_dsa_full_schedule::full_fragment_words(P.MP, QA));
        }
        ptr->pf.assign(m.nl, nullptr);
        G6AttnPrefill* owner = nullptr;
        for (int i = 0; i < m.nl; ++i)
        {
            Layer& y = m.layers[i];
            if (y.kda) continue;
            const int old_mr = y.ab.max_rows;
            y.ab.max_rows = std::max(old_mr, P.MP);
            const int e = g6_attn_bind_prefill_create_shared(&ptr->pf[i], &y.ab, st, owner);
            y.ab.max_rows = old_mr;
            if (e) die("Flash DSA layer " + std::to_string(y.l) + ": g6_attn_bind_prefill_create_shared failed (" + std::to_string(e) + ")");
            if (!owner) owner = ptr->pf[i];
        }
        G6RT_CK(bind::init());
    }
    return *ptr;
}

}   // namespace flash_dsa_pf

template <class Prefill>
inline void bind_dsa_prefill(Model& m, Prefill& P, cudaStream_t st)
{
    flash_dsa_pf::ensure_state(m, P, st);
}

// xf_all / xb_all: f16 / bf16 fragment tables of all M input rows (bind::Xfrag layout) already built by the mHC
// consume; null = build each chunk's tables from P.xn here.
// rs.base set (GR8_PF_SP_ATTD): the GEMM-path o_proj writes its rows to that RS send slot instead of P.part[0]; returns
// whether it did (false: the chunked o_proj paths, P.part[0] as usual).
template <class Prefill, class Mark>
inline bool pf_dsa_block_gemm(Model& m, Prefill& P, int i, int M, int context, cudaStream_t st, Mark mark,
                              const u32* xf_all = nullptr, const u32* xb_all = nullptr,
                              FlashPoolTransport* transport = nullptr, const bindp::RsOut& ors = bindp::RsOut())
{
    if (M < 1 || M > P.MP) die("pf_dsa_block_gemm: M outside 1..MP");
    flash_dsa_pf::State& S = flash_dsa_pf::ensure_state(m, P, st);
    Layer& y = m.layers[i];
    if (!y.built[1]) die("pf_dsa_block_gemm: bind_dsa_layer did not run");
    const bind::Layer& b1 = y.L[1];
    const int h = y.ab.args.heads;
    const bool proj_p1 = knob::pf_dsa_proj_p1();
    const bool full_gv = flash_dsa_pf::full_gv_on();
    // GR8_PF_DSA_F16GEMM: prepared at load by build_flash_prefill; this call only finds it (or prepares a test's P)
    flash_dsa_pf::F16Set* f16 = full_gv ? flash_dsa_pf::f16_prepare(m, &P, P.MP, st) : nullptr;
    static const flash_dsa_pf::F16Layer f16_none;
    const flash_dsa_pf::F16Layer& fl = f16 ? f16->L[i] : f16_none;
    flash_dsa_pf::F16Lane* fn = f16 ? &f16->lanes[&P] : nullptr;

    // 1. Input projections: q_a, kv_a, idx_k, idx_w, idx_g (32-row multi-tile GEMVs, 0-ULP bit-identical to decode)
    if (full_gv)
    {
        const u32* xf = xf_all;
        const u32* xb = xb_all;
        if (!xf || !xb) {
            flash_mhc::fragments(P.xn, M, HID, S.xf_full, S.xb_full, st);
            xf = S.xf_full; xb = S.xb_full;
        }
        // MXFP8 on cuBLAS: P.x holds the same fp16 values as the f16 fragment table
        auto proj = [&](const flash_dsa_f16::Mat& w, const bind::Gv& g, __half* y) {
            if (w.w) flash_dsa_f16::gemm(fn->blas, w, P.x, HID, M, fn->y32, y, g.ld, st);
            else flash_dsa_pf::launch_gv_full(g, g.codec == bind::MXFP8 ? xf : xb, M, y, st);
        };
        proj(fl.qa, b1.q_a, S.q_a);
        proj(fl.kva, b1.kv_a, S.kv_a);
        proj(fl.ik, b1.idx_k, S.idx_k);
        if (fl.wg.w)
            flash_dsa_f16::gemm_bf_split(fn->blas, fl.wg, xb, M, fn->rows, fn->y32, S.idx_w, b1.idx_w.ld, S.idx_g,
                                         b1.idx_g.ld, st);
        else {
            flash_dsa_pf::launch_gv_full(b1.idx_w, xb, M, S.idx_w, st);
            flash_dsa_pf::launch_gv_full(b1.idx_g, xb, M, S.idx_g, st);
        }
    }
    else for (int m0 = 0; m0 < M; m0 += flash_dsa_pf::GV_ROWS)
    {
        const int r = std::min(flash_dsa_pf::GV_ROWS, M - m0);
        const u32* xf = S.xf_x;
        const u32* xb = S.xb_x;
        if (xf_all && xb_all)   // the chunk's tables: row block m0 / 8 of the full-M tables (GV_ROWS % 8 == 0)
        {
            const size_t o = static_cast<size_t>(m0 / 8) * (HID / 16) * 64;
            xf = xf_all + o; xb = xb_all + o;
        }
        else flash_mhc::fragments(P.xn + static_cast<size_t>(m0) * HID, r, HID, S.xf_x, S.xb_x, st);
        flash_dsa_pf::launch_gv_rows(b1.q_a, b1.q_a.codec == bind::MXFP8 ? xf : xb, r, S.q_a + static_cast<size_t>(m0) * QA, st);
        flash_dsa_pf::launch_gv_rows(b1.kv_a, b1.kv_a.codec == bind::MXFP8 ? xf : xb, r, S.kv_a + static_cast<size_t>(m0) * KVA, st);
        flash_dsa_pf::launch_gv_rows(b1.idx_k, b1.idx_k.codec == bind::MXFP8 ? xf : xb, r, S.idx_k + static_cast<size_t>(m0) * IDXK, st);
        flash_dsa_pf::launch_gv_rows(b1.idx_w, xb, r, S.idx_w + static_cast<size_t>(m0) * IDXW, st);
        flash_dsa_pf::launch_gv_rows(b1.idx_g, xb, r, S.idx_g + static_cast<size_t>(m0) * geom::IDX_DIM, st);
    }

    // 2. q_a_norm across all M rows
    glue::k_rmsnorm_rows<__half><<<M, 256, 0, st>>>(S.q_a, QA, QA, y.q_a_norm, geom::RMS_EPS_FLASH, S.q_a_n, QA, nullptr, nullptr);

    if (full_gv && b1.idx_q.on && fl.iq.w)
        flash_dsa_f16::gemm(fn->blas, fl.iq, S.q_a_n, QA, M, fn->y32, S.idx_q, b1.idx_q.ld, st);
    else if (full_gv && b1.idx_q.on) {
        const bool mx = b1.idx_q.codec == bind::MXFP8;
        flash_dsa_pf::launch_xfrag(S.q_a_n, QA, M, mx ? S.xq_full : nullptr, mx ? nullptr : S.xq_full, st);
        flash_dsa_pf::launch_gv_full(b1.idx_q, S.xq_full, M, S.idx_q, st);
    }

    // 3. q_b (W8d k_gemm_tr_h16 over M rows, or lossless BF16 32-row GEMV) + idx.wq_b (32-row GEMV)
    if (b1.q_b_t.on && !proj_p1)
    {
        const dtr::Mat qbm = flash_dsa_pf::mat_from_gemv(b1.q_b_t);
        bindp::Gemm g_qb;
        if (const char* e = bindp::make_gemm_tr(qbm, M, S.x8_qa, S.q_b, 0, g_qb))
            die(std::string("pf_dsa_block_gemm q_b: ") + e);
        g_qb.pre.on = 1; g_qb.pre.src = S.q_a_n; g_qb.pre.ld = QA; g_qb.pre.K = QA; g_qb.pre.M = M;
        g_qb.pre.dst = S.x8_qa; g_qb.pre.suh = qbm.suh; g_qb.pre.h16 = g_qb.h16;
        g_qb.osvh = qbm.svh;
        bindp::launch(g_qb, st);
    }
    if (!full_gv || b1.q_b.on || (b1.q_b_t.on && proj_p1))
    for (int m0 = 0; m0 < M; m0 += flash_dsa_pf::GV_ROWS)
    {
        const int r = std::min(flash_dsa_pf::GV_ROWS, M - m0);
        const bool wqb_mx = b1.idx_q.codec == bind::MXFP8;
        flash_dsa_pf::launch_xfrag(S.q_a_n + static_cast<size_t>(m0) * QA, QA, r,
                                   wqb_mx ? S.xf_qa : nullptr, (!wqb_mx || b1.q_b.on) ? S.xb_qa : nullptr, st);
        if (b1.q_b.on)
            flash_dsa_pf::launch_gv_rows(b1.q_b, S.xb_qa, r, S.q_b + static_cast<size_t>(m0) * (h * 256), st);
        else if (b1.q_b_t.on && proj_p1)
        {
            for (int c0 = 0; c0 < r; c0 += MX)
            {
                const int cr = std::min(MX, r - c0);
                dtr::Gemv gv = y.L[cr].q_b_t;
#ifdef DEC_SLOT
                gv.a.slot = dslot::Slot{};
#endif
                gv.a.x = S.q_a_n + static_cast<size_t>(m0 + c0) * QA;
                gv.a.y = S.q_b + static_cast<size_t>(m0 + c0) * (h * 256);
                dtr::launch(gv, st);
            }
        }
        if (!full_gv)
            flash_dsa_pf::launch_gv_rows(b1.idx_q, wqb_mx ? S.xf_qa : S.xb_qa, r, S.idx_q + static_cast<size_t>(m0) * IDXQ, st);
    }
    mark(3, st);

    // 4. Tensor-Core DSA prefill attention over all M rows (or GR8_FPF_ATTN_P1=1 per-MX-chunk decode fallback)
    const __half* attn_out = nullptr;
    const bool attn_p1 = knob::fpf_attn_p1();
    if (attn_p1)
    {
        if (knob::pf_pool_capture())
            die("Flash pool capture requires the prefill attention path");
        if (!S.attn_p1) S.attn_p1 = dmalloc<__half>(static_cast<size_t>(P.MP) * g6nr::QBW);
        for (int m0 = 0; m0 < M; m0 += MX)
        {
            const int r = std::min(MX, M - m0);
            G6AttnStep s{};
            s.rows = r; s.context = context - M + m0 + r;
            s.q_b = S.q_b + static_cast<size_t>(m0) * (h * 256);
            s.kv_a = S.kv_a + static_cast<size_t>(m0) * KVA;
            s.positions = P.pos + m0;
            s.index_q = S.idx_q + static_cast<size_t>(m0) * IDXQ;
            s.index_k = S.idx_k + static_cast<size_t>(m0) * IDXK;
            s.index_weight = S.idx_w + static_cast<size_t>(m0) * IDXW;
            s.index_gate = S.idx_g + static_cast<size_t>(m0) * geom::IDX_DIM;
            G6RT_CK(g6_attn_bind_run(&y.ab, &s, st));
            G6RT_CK(cudaMemcpyAsync(S.attn_p1 + static_cast<size_t>(m0) * (h * 256), y.ab.args.output,
                                    static_cast<size_t>(r) * h * 256 * sizeof(__half), cudaMemcpyDeviceToDevice, st));
        }
        attn_out = S.attn_p1;
    }
    else
    {
        G6AttnStep s{};
        s.rows = M; s.context = context;
        s.q_b = S.q_b; s.kv_a = S.kv_a; s.positions = P.pos;
        s.index_q = S.idx_q; s.index_k = S.idx_k; s.index_weight = S.idx_w; s.index_gate = S.idx_g;
        if (transport) {
            G6FlashPoolOwner owner{m.rank, transport, flash_pool_publish, flash_pool_receive};
            G6RT_CK(g6_attn_bind_prefill_run_owned(S.pf[i], &y.ab, &s, st, &owner));
        } else G6RT_CK(g6_attn_bind_prefill_run(S.pf[i], &y.ab, &s, st));
        attn_out = static_cast<const __half*>(g6_attn_bind_prefill_output(S.pf[i], &y.ab, M));
    }
    mark(4, st);

    // 5. o_proj -> P.part[0] [M][HID] BF16 (or the RS send slot ors)
    bool used_rs = false;
    if (b1.o_proj_t.on && !proj_p1)
    {
        const dtr::Mat opm = flash_dsa_pf::mat_from_gemv(b1.o_proj_t);
        bindp::Gemm g_o;
        if (const char* e = bindp::make_gemm_tr(opm, M, S.x8_attn, P.part[0], 1, g_o))
            die(std::string("pf_dsa_block_gemm o_proj: ") + e);
        g_o.pre.on = 1; g_o.pre.src = attn_out; g_o.pre.ld = h * 256; g_o.pre.K = h * 256; g_o.pre.M = M;
        g_o.pre.dst = S.x8_attn; g_o.pre.suh = opm.suh; g_o.pre.h16 = g_o.h16;
        g_o.osvh = opm.svh;
        bindp::launch(g_o, st, ors);
        used_rs = ors.base != nullptr;
    }
    else if (b1.o_proj.on)
    {
        for (int m0 = 0; m0 < M; m0 += flash_dsa_pf::GV_ROWS)
        {
            const int r = std::min(flash_dsa_pf::GV_ROWS, M - m0);
            flash_dsa_pf::launch_xfrag(attn_out + static_cast<size_t>(m0) * (h * 256), h * 256, r, nullptr, S.xf_attn, st);
            flash_dsa_pf::launch_gv_rows(b1.o_proj, S.xf_attn, r, P.part[0] + static_cast<size_t>(m0) * HID, st);
        }
    }
    else
    {
        for (int m0 = 0; m0 < M; m0 += MX)
        {
            const int r = std::min(MX, M - m0);
            dtr::Gemv gv = y.L[r].o_proj_t;
#ifdef DEC_SLOT
            gv.a.slot = dslot::Slot{};
#endif
            gv.a.x = attn_out + static_cast<size_t>(m0) * (h * 256);
            gv.a.y = P.part[0] + static_cast<size_t>(m0) * HID;
            dtr::launch(gv, st);
        }
    }
    mark(5, st);
    return used_rs;
}

template <class Prefill>
inline void pf_dsa_block_gemm(Model& m, Prefill& P, int i, int M, int context, cudaStream_t st)
{
    pf_dsa_block_gemm(m, P, i, M, context, st, [](int, cudaStream_t) {});
}

}   // namespace g6rt
#endif
