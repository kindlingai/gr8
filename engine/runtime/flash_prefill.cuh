// Flash main-model prefill (INTERFACES.md section 8, PREFILL-PLAN.md): every prompt token but the last, in sub-batches
// of M <= MP rows of one session at consecutive positions, layer by layer over a second fabric region (prefill.cuh:
// two collectives of MP x HID BF16, 0 = attention partial, 1 = MLP partial, the layer as the step). Same order and
// sums as the decode loop (flash_step.cuh): embed, then per layer attention block -> publish -> rank-ordered FP32 sum
// + mHC streams update + mHC(hc_ffn) -> MLP block -> publish -> sum + update + mHC(next hc_attn). After the last
// layer: final norm rows (P.x, P.xn, P.resid = stream mean), no head. Plain launches on one stream (no graph).
//
// P1 bodies (this file, the reference the P2 GEMM bodies are gated against):
//   pf_kda_block      chunked KDA scan (kda_prefill.cuh) over all M rows; projections and o_proj per MX-row chunk
//                     (P2: trellis q/k/v and o_proj as M-row GEMMs, pf_kda_proj_gemm / pf_kda_oproj_gemm)
//   pf_dsa_block_p1   the decode DSA block per MX-row chunk (decode GEMVs, decode attention binding, no publish)
//   pf_mlp_block_p1   the decode MLP / MoE block per MX-row chunk, no publish
//   pf_draft_flash    the MTP draft (layer 45) over the prompt rows: the decode draft step per MX-row chunk
// Decode buffers (m.sc fragment tables, m.pos, m.sc.part, m.q_* ...) are borrowed per chunk; the decode step does
// not run while a prefill sub-batch is in flight.
#pragma once
#if G6_MODEL_FLASH
#include <unistd.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include "../kda/kda_prefill.cuh"
#include "../kda/kda_g9.cuh"
#include "../kda/proj_in_prefill.cuh"
#include "flash_mlp_pf.cuh"
#include "flash_dsa_pf.cuh"
#include "flash_pool_owner.cuh"
#include "flash_route_split.cuh"

namespace g6rt
{
struct PfPeerStage {
    bool on = false;                            // G6_PF_STAGE_ASYNC=1 and the lane was built with a side stream
    cudaStream_t stream = nullptr;
    cudaEvent_t put = nullptr, ready = nullptr, used = nullptr;
    __nv_bfloat16* peer[g6nr::NR] = {};
};
// G6_PF_SP=1: sequence-parallel mHC. Per block output, the rank partials go reduce-scatter (collective 2c: rank p
// gets rows pf_r0(M, p).. of every partial), the owner runs the same mHC kernel on its rows only (same per-row op
// order, the partials summed in rank order: bit-identical rows), then all-gathers its FP32 normalized rows
// (collective 2c + 1) and every rank rounds them to the FP16 block input. The wire bytes per rank equal the
// all-gather form (2/3 M BF16 rows out + 1/3 M FP32 rows to each peer at TP3); the mHC work drops to M / nr rows.
inline bool pf_sp()
{
    static const bool on = getenv("G6_PF_SP") && atoi(getenv("G6_PF_SP")) == 1;
    return on;
}
// G6_PF_SP_X16=1 (with G6_PF_SP and G6_PF_ROUTER_SPLIT): an MoE layer's MLP input travels as the FP16 rows its
// consumers read (x2) instead of the FP32 rows (xn2): the router reads only this rank's FP32 rows (route split), the
// experts the FP16 ones. Half the bytes of that all-gather, and no FP16 conversion of the peers' rows. Byte-equal.
inline bool pf_sp_x16()
{
    static const bool on = getenv("G6_PF_SP_X16") && atoi(getenv("G6_PF_SP_X16")) == 1;
    return on;
}
// G6_PF_SP_AGK (default on with G6_PF_SP; =0 off): an FP32 all-gather whose consumer directly follows an owner pass
// (lane B's in the two-lane schedule, every one in the single-lane step) waits on the compute stream and moves the
// peers' rows with one SM pass that also writes their FP16 copy (k_sp_ag_f16), instead of two serial copy-engine copies
// on the lane's copy stream plus k_sp_f16. Same bytes in xn2 / xn and x2 / x (plain copy, the same __float2half_rn):
// byte-equal. TP3 8K A/B: prefill 2418 -> 2436 tok/s, rank-0 8K prefill -29 ms, T=0 hashes unchanged.
// G6_PF_SP_AGD (default on with G6_PF_SP; =0 off): the owner pass (k_mhc_out_inv_prepare_rr) also stores its rows
// straight into the all-gather send slot (FP32 rows, or the FP16 x2 rows of a G6_PF_SP_X16 layer), so the publish only
// rings the doorbell instead of copying the rows (k_put_publish_n). The same values in the same place: byte-equal.
// TP3 8K (nsys): k_put_publish_n -35 ms, rr +27 ms (its extra stores to the pinned slot); A/B rank-0 8K prefill -6 ms.
inline bool pf_sp_agd()
{
    static const bool on = !(getenv("G6_PF_SP_AGD") && atoi(getenv("G6_PF_SP_AGD")) == 0);
    return on;
}
inline bool pf_sp_agk()
{
    static const bool on = !(getenv("G6_PF_SP_AGK") && atoi(getenv("G6_PF_SP_AGK")) == 0);
    return on;
}
// G6_PF_SP_PUT_CTA=N (default 96, 1..1024; 32 = the PF_PUT grid): CTAs of the SP-path send copies (k_put_rs,
// k_put_publish_n). 32 x 128 threads leave the copy latency-bound: device -> send slot at M 4096, k_put_rs 291 -> 238 us,
// FP32 publish 127 -> 111 us (bench). Copy only, the same bytes to the same place: byte-equal.
inline int pf_sp_put_cta()
{
    static const int n = [] { const char* e = getenv("G6_PF_SP_PUT_CTA"); const int v = e ? atoi(e) : 96; return v >= 1 && v <= 1024 ? v : 96; }();
    return n;
}
inline size_t pf_sp_slot_bytes(int MP, int nr)
{
    return std::max(pf_slot_bytes(MP, true, nr), ((size_t)(MP + nr - 1) / nr * HID * 4 + 8 + 255) & ~size_t(255));
}
struct FlashPrefill
{
    FlashPoolTransport pool_transport;           // set up by g6m4 when the pool owner is on; independent of partial fabrics
    FlashRouteSplit route_split;                 // G6_PF_ROUTER_SPLIT=1 (g6m4): dedicated route-table fabric; fabric == null = off
    int MP = 0;
    g6::FabricDev f{};
    u64* stepp = nullptr;
    unsigned* tickets = nullptr;                 // [2]
    int *tok = nullptr, *pos = nullptr, *row_slot = nullptr;   // [MP]
    float* resid4 = nullptr;                     // [MP][HC][HID]
    float *hc_inv = nullptr, *hc_mix = nullptr, *hc_pre = nullptr, *hc_post = nullptr, *hc_comb = nullptr;
    unsigned* hc_done = nullptr;
    float *xn = nullptr, *xn2 = nullptr;         // [MP][HID] (xn: round_up(MP, 8) rows, see attn_frag)
    __nv_bfloat16* sp_peer[g6nr::NR] = {};       // G6_PF_SP: device copies of the peers' reduce-scatter slices (ceil(MP / nr) rows)
    cudaStream_t sp_st = nullptr;                // G6_PF_SP: per-lane copy stream (wait + copy-engine staging off the compute stream)
    cudaEvent_t sp_put[4] = {}, sp_ready[4] = {};
    bool sp_view = false;                        // G6_PF_SP owner-row view: no fragment tables (rows are a slice of M)
    __half* agd_x = nullptr;                     // G6_PF_SP_AGD: owner pass also writes its FP16 rows here (AG send slot)
    float* agd_xn = nullptr;                     // G6_PF_SP_AGD: owner pass also writes its FP32 rows here (AG send slot)
    bool agd_done = false;                       // G6_PF_SP_AGD: the owner pass wrote the send slot (pf_consume_launch)
    bool sp_agk[4] = {};                         // G6_PF_SP_AGK: all-gather c runs on the compute stream (pf_sp_gather)
    bool attn_frag = false;                      // xn holds the attention input as f16 + bf16 fragment tables (G6_PF_XFRAG)
    __half *x = nullptr, *x2 = nullptr;          // [MP][HID]
    __nv_bfloat16* part[2] = {};                 // [MP][HID]
    float* resid = nullptr;                      // [MP][HID]: final stream mean (after the last layer)
    kda::KdaScratch ksc;                         // MP rows
    kda::KdaPrefillScratch kps;                  // MP rows
    bool any_kda = false;
    unsigned long long* pxt = nullptr;           // G6_PF_XTRACE=1: [pxt::PX_STEPS][2][pxt::PX_W] (flash_step.cuh), else null
    unsigned long long* comm_xt = nullptr;      // XTRACE only: [PX_STEPS][2][3], before wait / after wait / after copy
    std::vector<uint4*> fnp;                     // [layer * 2 + hc]: mHC fn packed for k_mhc_out_inv_prepare_pk (unless G6_PF_MHC_PK=0)
    __nv_bfloat16* peer_dev = nullptr;           // NR == 2 (unless G6_PF_MHC_STAGE=0): [MP][HID] the peer's partial, copy engine
    PfPeerStage peer_stage;                       // G6_PF_STAGE_ASYNC=1 only; one stream and storage per lane
    bindp::RsOut fin_rs;                         // G6_PF_SP_FIND: the MoE final sum's send-slot target (base null = off)
    const char* fin_own = nullptr;               // G6_PF_SP_FIND: this rank's own rows of that slot (set = layer used it)
    const char* att_own = nullptr;               // G6_PF_SP_ATTD: this rank's own o_proj rows in the collective-0 slot
    u64 cur_step = 0;                            // host copy of the layer step pf_step_flash set (for the staging copy)
};
inline bool pf_mhc_stage()
{
    static const bool on = !(getenv("G6_PF_MHC_STAGE") && atoi(getenv("G6_PF_MHC_STAGE")) == 0);   // default on
    return on;
}
inline bool pf_mhc_pk()
{
    static const bool on = !(getenv("G6_PF_MHC_PK") && atoi(getenv("G6_PF_MHC_PK")) == 0);
    return on;
}
#ifndef PF_MHC_RR_R
#define PF_MHC_RR_R 2   // rows per CTA of k_mhc_out_inv_prepare_rr: each CTA streams the 786 KB packed fn once for R rows
#endif
inline bool pf_mhc_rr()   // k_mhc_out_inv_prepare_rr when the peer partial is staged; G6_PF_MHC_RR=0 = _pk
{
    static const bool on = !(getenv("G6_PF_MHC_RR") && atoi(getenv("G6_PF_MHC_RR")) == 0);
    return on;
}

// ------------------------------------------------------------------ G6_PF_XTRACE=1 (measurement only, default off)
inline int pf_fuse_level()
{
    static const int v = getenv("G6_PF_MHC_FUSE") ? atoi(getenv("G6_PF_MHC_FUSE")) : 2;
    return v;
}
// G6_PF_STAGE_ASYNC (default ON at TP2, default off at TP4): the wait for the peers' partials and the copy of each peer partial into
// device memory run on a per-lane side stream, under the other lane's block. Off = the code that was here before.
// Unset: on at NR=2 when the config allows it (a config that cannot run it, see the build gate, silently stays off);
// =1: on at any NR, a config that cannot run it dies; =0 (or any other value): off, the kill switch.
// Returns 0 off, 1 default-on (soft), 2 asked for (hard).
inline int pf_stage_async_mode()
{
    static const int v = []
    {
        const char* e = getenv("G6_PF_STAGE_ASYNC");
        if (e && *e) return atoi(e) == 1 ? 2 : 0;
        return g6nr::NR == 2 ? 1 : 0;
    }();
    return v;
}
inline bool pf_xtrace_env()
{
    static const bool on = getenv("G6_PF_XTRACE") && atoi(getenv("G6_PF_XTRACE")) == 1;
    return on;
}
// stream-ordered stamp: field PX_PROD also opens the record (min/max fields reset, rows, step tag)
__global__ void k_pxt_stamp(unsigned long long* p, const u64* stepp, int c, int field, int M)
{
    const u64 step = *stepp;
    unsigned long long* r = pxt::rec(p, step, c);
    const unsigned long long t = pxt::now();
    if (field == pxt::PX_PROD)
    {
        r[pxt::PX_ENTRY] = ~0ull; r[pxt::PX_WMIN] = ~0ull; r[pxt::PX_WMAX] = 0; r[pxt::PX_PUT] = 0; r[pxt::PX_END] = 0;
        r[pxt::PX_M] = (unsigned long long)M; r[pxt::PX_STEP] = step + 1;
    }
    r[field] = t;
}
inline void pf_xt(FlashPrefill& P, int c, int field, int M, cudaStream_t st)
{
    if (P.pxt) k_pxt_stamp<<<1, 1, 0, st>>>(P.pxt, P.stepp, c, field, M);
}
// A staged receive waits before the mHC grid. With two lanes, put->entry also
// contains useful compute from the other lane, so it cannot measure exposed wait.
__global__ void k_pf_comm_stamp(unsigned long long* p, const u64* stepp, int c, int field)
{
    p[((size_t)(*stepp % pxt::PX_STEPS) * 2 + c) * 3 + field] = pxt::now();
}
inline void pf_comm_xt(FlashPrefill& P, int c, int field, cudaStream_t st)
{
    if (P.comm_xt) k_pf_comm_stamp<<<1, 1, 0, st>>>(P.comm_xt, P.stepp, c, field);
}
__global__ void k_pxt_clock(const volatile unsigned* go, volatile unsigned long long* out)
{
    while (!*go) {}
    *out = pxt::now();
    __threadfence_system();
}
// GPU globaltimer vs host CLOCK_REALTIME: n handshakes, keep the tightest. Prints
// "PXTCLK <tag> rank R off_ns err_ns" with off = globaltimer - realtime at the midpoint, |error| <= err.
inline void pf_xt_clock(const char* tag, int rank, int n = 64)
{
    unsigned* go; unsigned long long* out;
    G6RT_CK(cudaHostAlloc(&go, 4, cudaHostAllocMapped));
    G6RT_CK(cudaHostAlloc(&out, 8, cudaHostAllocMapped));
    unsigned* go_d; unsigned long long* out_d;
    G6RT_CK(cudaHostGetDevicePointer(&go_d, go, 0));
    G6RT_CK(cudaHostGetDevicePointer(&out_d, out, 0));
    cudaStream_t s;
    G6RT_CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    long long best_off = 0, best_err = -1;
    for (int i = 0; i < n; ++i)
    {
        *(volatile unsigned*)go = 0; *(volatile unsigned long long*)out = 0;
        k_pxt_clock<<<1, 1, 0, s>>>(go_d, out_d);
        usleep(2000);   // the kernel is spinning by now
        timespec a, b;
        clock_gettime(CLOCK_REALTIME, &a);
        *(volatile unsigned*)go = 1;
        while (!*(volatile unsigned long long*)out) {}
        clock_gettime(CLOCK_REALTIME, &b);
        G6RT_CK(cudaStreamSynchronize(s));
        const long long ta = a.tv_sec * 1000000000LL + a.tv_nsec, tb = b.tv_sec * 1000000000LL + b.tv_nsec;
        const long long g = (long long)*(volatile unsigned long long*)out;
        const long long err = (tb - ta + 1) / 2;
        if (best_err < 0 || err < best_err) { best_err = err; best_off = g - (ta + tb) / 2; }
    }
    G6RT_CK(cudaStreamDestroy(s));
    cudaFreeHost(go); cudaFreeHost(out);
    fprintf(stderr, "PXTCLK %s rank %d off_ns %lld err_ns %lld\n", tag, rank, best_off, best_err);
}
// all records to stderr: "PXT step c M prod put entry wmin wmax end" (globaltimer ns, this rank's clock)
inline void pf_xt_dump(FlashPrefill& P, int rank)
{
    if (!P.pxt) return;
    const size_t n = (size_t)pxt::PX_STEPS * 2 * pxt::PX_W;
    std::vector<unsigned long long> h(n);
    G6RT_CK(cudaDeviceSynchronize());
    G6RT_CK(cudaMemcpy(h.data(), P.pxt, n * 8, cudaMemcpyDeviceToHost));
    std::vector<const unsigned long long*> rs;
    for (size_t i = 0; i < (size_t)pxt::PX_STEPS * 2; ++i)
        if (h[i * pxt::PX_W + pxt::PX_STEP]) rs.push_back(&h[i * pxt::PX_W]);
    std::sort(rs.begin(), rs.end(), [](const unsigned long long* a, const unsigned long long* b) { return a[pxt::PX_PROD] < b[pxt::PX_PROD]; });
    fprintf(stderr, "# PXT rank %d: step c M prod put entry wmin wmax end (globaltimer ns)\n", rank);
    for (const unsigned long long* r : rs)
        fprintf(stderr, "PXT %llu %d %llu %llu %llu %llu %llu %llu %llu\n", r[pxt::PX_STEP] - 1,
                (int)(((r - h.data()) / pxt::PX_W) % 2), r[pxt::PX_M], r[pxt::PX_PROD], r[pxt::PX_PUT], r[pxt::PX_ENTRY],
                r[pxt::PX_WMIN], r[pxt::PX_WMAX], r[pxt::PX_END]);
    if (P.comm_xt)
    {
        std::vector<unsigned long long> ch((size_t)pxt::PX_STEPS * 2 * 3);
        G6RT_CK(cudaMemcpy(ch.data(), P.comm_xt, ch.size() * 8, cudaMemcpyDeviceToHost));
        fprintf(stderr, "# PFCOMM rank %d: step c wait_begin wait_end copy_end (globaltimer ns)\n", rank);
        for (const unsigned long long* r : rs)
        {
            const size_t slot = (r - h.data()) / pxt::PX_W;
            const auto* q = ch.data() + slot * 3;
            if (q[0]) fprintf(stderr, "PFCOMM %llu %d %llu %llu %llu\n", r[pxt::PX_STEP] - 1,
                               (int)(slot % 2), q[0], q[1], q[2]);
        }
    }
    pf_xt_clock("end", rank);
}

// G6_PF_PROF=1: synchronized wall ms per Flash prefill stage, summed over sub-batches (the marks synchronize the stream,
// the DSA marks once per MX-row chunk). Wait+reduce includes the time this rank waits for the slower rank.
struct FlashPfProf
{
    static constexpr int NS = 11;
    double t[NS] = {}, last = 0;
    double a[9] = {}, alast = 0;          // G6_PF_AMARK=1: DSA attention ms by attention mark (flash_pf_amark)
    void mark(int k, cudaStream_t st)
    {
        if (!pf_prof().on) return;
        G6RT_CK(cudaStreamSynchronize(st));
        const double n = PfProf::now();
        if (k >= 0 && last > 0) t[k] += n - last;
        last = n;
    }
    void print(const char* tag) const
    {
        static const char* n[NS] = {"kda proj_in", "kda scan", "kda o_proj", "dsa projections", "dsa attention", "dsa o_proj",
                                    "dense MLP", "MoE", "put+publish", "wait+reduce+mHC", "embed/other"};
        double tot = 0;
        for (double v : t) tot += v;
        fprintf(stderr, "# %s Flash prefill stage profile (ms, synchronized):", tag);
        for (int i = 0; i < NS; ++i) fprintf(stderr, " %s %.1f (%.0f%%),", n[i], t[i], 100 * t[i] / (tot > 0 ? tot : 1));
        fprintf(stderr, " total %.1f\n", tot);
        double at = 0;
        for (double v : a) at += v;
        if (at > 0)
            fprintf(stderr, "# %s DSA attention split (ms, G6_PF_AMARK): append+index scores+select %.1f, reuse %.1f, absorb %.1f, "
                    "qk %.1f, normalize %.1f, pv %.1f, value proj %.1f, total %.1f\n",
                    tag, a[1], a[2], a[3], a[4], a[5], a[6] + a[7], a[8], at);
    }
};
inline FlashPfProf& flash_pf_prof() { static FlashPfProf p; return p; }
// g6_attn_mark hook during the prefill DSA call (G6_PF_PROF=1 G6_PF_AMARK=1): synchronized ms from the previous mark (or
// the stage mark before the attention) to attention mark k (flash_attn.cu AMARK in g6_attn_bind_prefill_run)
inline void flash_pf_amark(int k, void* s)
{
    FlashPfProf& p = flash_pf_prof();
    G6RT_CK(cudaStreamSynchronize(static_cast<cudaStream_t>(s)));
    const double n = PfProf::now(), ref = p.alast > p.last ? p.alast : p.last;
    if (k >= 0 && k < 9 && ref > 0) p.a[k] += n - ref;
    p.alast = n;
}

inline flash_mhc::Buffers mhc_buffers(FlashPrefill& P)
{
    flash_mhc::Buffers b{P.resid4, P.hc_inv, P.hc_mix, P.hc_pre, P.hc_post, P.hc_comb, P.hc_done};
    b.bf16 = flash_mhc::bf16_env() == 3;   // un-sided launches: only when both sides round
    return b;
}

inline void build_flash_prefill(Model& m, FlashPrefill& P, int MP, const g6::FabricDev& f, u64* stepp, cudaStream_t st)
{
    if (!m.resid4 || m.flash_draft) die("Flash prefill: main model only (the MTP draft prefill is P1b)");
    if (MP < 1 || MP % 64) die("Flash prefill: --prefill must be a positive multiple of 64");
    if (f.ncoll != (pf_sp() ? 4 : 2) || f.nr != g6nr::NR) die("Flash prefill: the prefill fabric needs 2 collectives (4 with G6_PF_SP=1)");
    if (pf_sp() && f.stride < pf_sp_slot_bytes(MP, f.nr)) die("Flash prefill: G6_PF_SP fabric slots are too small");
    if (f.stride < pf_slot_bytes(MP, false, f.nr) + 8) die("Flash prefill: fabric slots are smaller than MP rows");
    P = FlashPrefill();
    P.MP = MP; P.f = f; P.stepp = stepp;
    P.tickets = dmalloc<unsigned>(4);
    G6RT_CK(cudaMemset(P.tickets, 0, 4 * sizeof(unsigned)));
    P.tok = dmalloc<int>(MP); P.pos = dmalloc<int>(MP); P.row_slot = dmalloc<int>(MP);
    P.resid4 = dmalloc<float>((size_t)MP * geom::HC * HID);
    P.hc_inv = dmalloc<float>(MP); P.hc_mix = dmalloc<float>((size_t)MP * geom::HC_MIX);
    P.hc_pre = dmalloc<float>((size_t)MP * geom::HC); P.hc_post = dmalloc<float>((size_t)MP * geom::HC);
    P.hc_comb = dmalloc<float>((size_t)MP * geom::HC * geom::HC); P.hc_done = dmalloc<unsigned>(MP);
    P.xn = dmalloc<float>((size_t)(MP + 7) / 8 * 8 * HID); P.xn2 = dmalloc<float>((size_t)MP * HID);
    P.x = dmalloc<__half>((size_t)MP * HID); P.x2 = dmalloc<__half>((size_t)MP * HID);
    P.part[0] = dmalloc<__nv_bfloat16>((size_t)MP * HID); P.part[1] = dmalloc<__nv_bfloat16>((size_t)MP * HID);
    P.resid = dmalloc<float>((size_t)MP * HID);
    if (pf_sp())
    {
        for (int p = 0; p < g6nr::NR; ++p)
            if (p != f.rank) P.sp_peer[p] = dmalloc<__nv_bfloat16>((size_t)(MP + f.nr - 1) / f.nr * HID);
        int lo = 0, hi = 0;
        G6RT_CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        G6RT_CK(cudaStreamCreateWithPriority(&P.sp_st, cudaStreamNonBlocking, hi));
        for (int c = 0; c < 4; ++c)
        {
            G6RT_CK(cudaEventCreateWithFlags(&P.sp_put[c], cudaEventDisableTiming));
            G6RT_CK(cudaEventCreateWithFlags(&P.sp_ready[c], cudaEventDisableTiming));
        }
    }
    for (const Layer& y : m.layers) P.any_kda |= y.kda;
    if (pf_mhc_pk() && HID == 4096)
        for (const Layer& y : m.layers)
            for (int h = 0; h < 2; ++h)
            {
                uint4* q = dmalloc<uint4>((size_t)3 * 4 * 16 * flash_mhc::THREADS);
                flash_mhc::k_mhc_pack_fn<<<3 * 4 * 16, flash_mhc::THREADS, 0, st>>>(y.hc[h].fn, q);
                P.fnp.push_back(q);
            }
    if (pf_mhc_stage() && g6nr::NR == 2) P.peer_dev = dmalloc<__nv_bfloat16>((size_t)MP * HID);
    if (const int mode = pf_stage_async_mode())   // mode 2: asked for, run it or stop, never fall back to the OFF path unannounced
    {
        const char* no = !pf_mhc_stage() ? "G6_PF_MHC_STAGE=0" : pf_fuse_level() < 1 ? "G6_PF_MHC_FUSE < 1"
                       : pf_xtrace_env() ? "G6_PF_XTRACE=1 (the stamps need one stream)"
                       : (getenv("G6_PF_RS") && atoi(getenv("G6_PF_RS")) != 0) ? "G6_PF_RS (SP / reduce-scatter prefill: owner-only consume, not uniform slots)"
                       : pf_sp() ? "G6_PF_SP (owner rows read their reduce-scatter slices in place)" : nullptr;
        if (no && mode == 2)
        {
            char msg[256];
            snprintf(msg, sizeof msg, "G6_PF_STAGE_ASYNC=1 cannot run with %s", no);
            die(msg);
        }
        if (no) fprintf(stderr, "# rank %d: G6_PF_STAGE_ASYNC off (default-on skipped: %s)\n", f.rank, no);
        else
        {
            PfPeerStage& S = P.peer_stage;
            int lo = 0, hi = 0;
            G6RT_CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            G6RT_CK(cudaStreamCreateWithPriority(&S.stream, cudaStreamNonBlocking, hi));   // high: orders pending CTAs, never preempts
            G6RT_CK(cudaEventCreateWithFlags(&S.put, cudaEventDisableTiming));
            G6RT_CK(cudaEventCreateWithFlags(&S.ready, cudaEventDisableTiming));
            G6RT_CK(cudaEventCreateWithFlags(&S.used, cudaEventDisableTiming));
            for (int p = 0; p < g6nr::NR; ++p)
                if (p != f.rank) S.peer[p] = g6nr::NR == 2 ? P.peer_dev : dmalloc<__nv_bfloat16>((size_t)MP * HID);
            S.on = true;
            check_headroom(f.rank, MP, false);   // NR=4: 3 peer buffers of MP*HID*2 per lane just came out of the same memory
            static int lane_no = 0;   // build order: FP then FP2
            fprintf(stderr, "# rank %d: G6_PF_STAGE_ASYNC on (lane %d, peers %d)\n", f.rank, lane_no++, g6nr::NR - 1);
        }
    }
    if (pf_xtrace_env())
    {
        P.pxt = dmalloc<unsigned long long>((size_t)pxt::PX_STEPS * 2 * pxt::PX_W);
        G6RT_CK(cudaMemset(P.pxt, 0, (size_t)pxt::PX_STEPS * 2 * pxt::PX_W * 8));
        P.comm_xt = dmalloc<unsigned long long>((size_t)pxt::PX_STEPS * 2 * 3);
        G6RT_CK(cudaMemset(P.comm_xt, 0, (size_t)pxt::PX_STEPS * 2 * 3 * 8));
        pf_xt_clock("start", f.rank);
    }
    if (P.any_kda)
    {
        if (const char* e = kda::build_scratch(P.ksc, MP, g6nr::NR)) die(e);
        if (const char* e = kda::build_prefill_scratch(P.kps, MP, g6nr::NR)) die(e);
    }
    flash_dsa_pf::f16_prepare(m, &P, MP, st);   // G6_PF_DSA_F16GEMM only: decode + cuBLAS handle at load, not in a prefill
    if (flash_dsa_f16::on()) check_headroom(f.rank, MP, false);   // F16 decoded weights and per-lane workspace now count
    G6RT_CK(cudaStreamSynchronize(st));
}

// G6_PF_PREWARM (default ON at TP2, default off at TP4; =1 on at any NR, =0 or any other value off, the kill switch):
// create the per-lane DSA and MLP prefill states (the cudaMallocs, ~140 ms, and the layer tensor binds) at boot instead
// of inside the first chunk. Same calls the first chunk makes, so the states are identical.
inline bool pf_prewarm_env()
{
    static const bool on = []
    {
        const char* e = getenv("G6_PF_PREWARM");
        if (e && *e) return atoi(e) == 1;
        return g6nr::NR == 2;
    }();
    return on;
}
inline void pf_prewarm_flash(Model& m, FlashPrefill& P, cudaStream_t st)
{
    if (!pf_prewarm_env() || !P.MP) return;
    bind_dsa_prefill(m, P, st);
    bind_mlp_prefill(m, P, st);
    G6RT_CK(cudaStreamSynchronize(st));
    check_headroom(P.f.rank, P.MP, true);   // same floor as the KV pool: the states now count at boot, stop here and not in the OOM killer
    static int lane_no = 0;   // call order: FP then FP2; the launcher's verify-log counts these lines
    fprintf(stderr, "# rank %d: G6_PF_PREWARM on (lane %d)\n", P.f.rank, lane_no++);
}

// host: tokens and positions of one sub-batch; every row in KDA state slot `slot` (0: one offline session; S2: the
// session's first slot, FlashSlots::base)
inline void pf_inputs_flash(FlashPrefill& P, int M, const int* tok, const int* pos, int slot_id = 0)
{
    if (M < 1 || M > P.MP) die("Flash prefill: M outside 1..MP");
    for (int r = 1; r < M; ++r) if (pos[r] != pos[0] + r) die("Flash prefill: rows must be consecutive positions");
    G6RT_CK(g6h2d::copy(P.tok, tok, M * sizeof(int)));
    G6RT_CK(g6h2d::copy(P.pos, pos, M * sizeof(int)));
    const std::vector<int> slot(M, slot_id);
    G6RT_CK(g6h2d::copy(P.row_slot, slot.data(), M * sizeof(int)));
}

// host: the proxy's jobs of one sub-batch (layer-steps ls0 .. ls0 + nl - 1)
template <class Fab>
inline void pf_enqueue_flash(Fab& fab, u64 ls0, int nl, int M)
{
    if (pf_sp())
    {
        const g6::FabricDev f = fab.dev();
        std::vector<uint32_t> sizes(4, 0);
        sizes[1] = sizes[3] = (uint32_t)((size_t)pf_rows(M, f.rank, f.nr) * HID * 4);
        g6::Fabric::Rs rs;
        rs.mask = 0x5u;   // collectives 0 and 2
        rs.ss = pf_ss(f);
        for (int p = 0; p < f.nr; ++p) rs.bytes.push_back((uint32_t)((size_t)pf_rows(M, p, f.nr) * HID * 2));
        // G6_PF_SP_X16: an MoE layer's collective 1 carries FP16 rows (pf_sp_x16_layer; route split on at this M)
        const char* rse = getenv("G6_PF_ROUTER_SPLIT");
        const bool x16 = pf_sp_x16() && rse && atoi(rse) == 1 && flash_route_split::active(M, f.nr);
        std::vector<uint32_t> s16 = sizes;
        s16[1] = (uint32_t)((size_t)pf_rows(M, f.rank, f.nr) * HID * 2);
        for (int i = 0; i < nl; ++i) fab.enqueue_step(ls0 + i, x16 && geom::is_moe(i) ? s16 : sizes, rs);
        return;
    }
    const std::vector<uint32_t> sizes(2, (uint32_t)(M * HID * 2));
    for (int i = 0; i < nl; ++i) fab.enqueue_step(ls0 + i, sizes);
}

// ---------------------------------------------------------------------------------------------- P1 block bodies
// G6_FPF_P1=1 forces the P1 bodies everywhere (the reference the GEMM bodies are gated against)
inline bool pf_force_p1()
{
    static const bool f = getenv("G6_FPF_P1") && atoi(getenv("G6_FPF_P1")) == 1;
    return f;
}

// KDA q/k/v and o_proj as M-row GEMMs when they are dense trellis matrices (W8d): k_gemm_tr_h16 (bindp::make_gemm_tr)
// on fp16 A-fragments of the rotated rows (Xq h16), FP32 into P.ksc.P (q/k/v) and BF16 into P.part[0] (o_proj).
// f_a / g_a (MXFP8) and b_proj (BF16) stay the decode GEMVs per MX chunk. G6_FPF_KDA_P1=1 = the P1 projections.
inline bool pf_kda_gemm_on(const kda::KdaLayer& w)
{
    static const bool p1 = getenv("G6_FPF_KDA_P1") && atoi(getenv("G6_FPF_KDA_P1")) == 1;
    return !p1 && !pf_force_p1() && ((w.qt.w && w.kt.w && w.vt.w) || (w.mxc[0] && w.mxc[1] && w.mxc[2]));
}
// KDA f_a / g_a / b projections after the q/k/v GEMMs: 0 = old MX-row host loop, 1 = full-M 32-row tiles, 2 = 64-row
inline int pf_kda_tail()
{
    static const int t = getenv("G6_PF_KDA_TAIL") ? atoi(getenv("G6_PF_KDA_TAIL")) : 1;
    return t;
}
inline bool pf_kda_tail_check()
{
    static const bool c = getenv("G6_PF_KDA_TAIL_CHECK") && atoi(getenv("G6_PF_KDA_TAIL_CHECK")) == 1;
    return c;
}
inline bool pf_dsa_p1()
{
    static const bool p1 = getenv("G6_FPF_DSA_P1") && atoi(getenv("G6_FPF_DSA_P1")) == 1;
    return p1;
}
// G6_PF_XFRAG (default off; G6_PF_XFRAG=1 = on, byte-equal; clean 45L 32K A/B gain 0.08 s): when layer j's attention reads its input only as A-fragments (KDA GEMM + full-M tail, DSA
// GEMM projections), the mHC kernel that makes that input writes the f16 + bf16 fragment tables (bind::Xfrag layout,
// all M rows) into P.xn's memory instead of FP32 rows: same rounding of the same FP32 values as fragments(P.xn) ->
// bit-identical, and the FP32 write + re-read + fragment pass are gone. P1, control and check paths keep FP32 rows.
inline bool pf_xfrag_env()
{
    static const bool on = getenv("G6_PF_XFRAG") && atoi(getenv("G6_PF_XFRAG")) == 1;
    return on;
}
inline bool pf_attn_frag_ok(Model& m, int j)
{
    if (!pf_xfrag_env() || pf_force_p1() || HID != 4096) return false;
    const Layer& y = m.layers[j];
    if (y.kda) return pf_kda_gemm_on(y.kda_w) && (pf_kda_tail() == 1 || pf_kda_tail() == 2) && !pf_kda_tail_check();
    return !pf_dsa_p1();
}
inline unsigned* pf_attn_f16(FlashPrefill& P) { return reinterpret_cast<unsigned*>(P.xn); }
inline unsigned* pf_attn_bf16(FlashPrefill& P) { return reinterpret_cast<unsigned*>(P.xn) + (size_t)(P.MP + 7) / 8 * 8 * HID / 2; }
inline void pf_need_fp32_xn(const FlashPrefill& P, const char* who)
{
    if (P.attn_frag) die(std::string(who) + ": P.xn holds fragment tables (G6_PF_XFRAG), not FP32 rows");
}
inline void pf_kda_tail_control(Model& m, FlashPrefill& P, const kda::KdaLayer& w, const kda::Args& a, int M, cudaStream_t st)
{
    pf_need_fp32_xn(P, "pf_kda_tail_control");
    const kda::PLayout pl = a.pl();
    const int ld = pl.ld();
    for (int m0 = 0; m0 < M; m0 += MX)
    {
        const int r = std::min(MX, M - m0);
        float* y = P.ksc.P + (size_t)m0 * ld;
        flash_mhc::fragments(P.xn + (size_t)m0 * HID, r, HID, m.sc.xf_x, m.sc.xb_x, st);
        kda::ProjArgs pa;     // as kda::launch_proj_in without q/k/v
        pa.xfrag = m.sc.xb_x; pa.M = r; pa.ld = ld; pa.odt = gv::O_F32;
        if (w.fa_mx) pa.add(w.fa_mx, y + pl.fa(), HID, kda::DK, 1, 32, m.sc.xf_x, 1);
        else         pa.add(w.fa,    y + pl.fa(), HID, kda::DK, 1, 32);
        if (w.ga_mx) pa.add(w.ga_mx, y + pl.ga(), HID, kda::DK, 1, 32, m.sc.xf_x, 1);
        else         pa.add(w.ga,    y + pl.ga(), HID, kda::DK, 1, 32);
        pa.add(w.b, y + pl.b(), HID, kda::HEADS, 1, 32);
        kda::launch_proj(pa, st);
    }
}
inline void pf_kda_proj_gemm(Model& m, FlashPrefill& P, const kda::KdaLayer& w, const kda::Args& a, int M, cudaStream_t st)
{
    bindp::Scratch& sc = flash_mlp_pf::ensure_state(m, P, nullptr, st).sc;
    const kda::PLayout pl = a.pl();
    const int ld = pl.ld(), dl = a.dl();
    if (!w.qt.w)                        // MXFP8 q / k / v (the published release): one FP8 quantization of x feeds all three
    {
        for (int s = 0; s < 3; ++s)
        {
            bind::Tensor c, z;
            c.p = w.mxc[s]; c.nd = 2; c.shape[0] = dl; c.shape[1] = HID;
            z.p = w.mxs[s]; z.nd = 2; z.shape[0] = dl; z.shape[1] = HID / 32;
            bindp::Gemm g;
            if (const char* e = bindp::make_gemm(c, z, M, sc.x8_big, P.ksc.P + pl.q() + s * dl, 2, g))
                die(std::string("pf_kda_proj_gemm: ") + e);
            g.ldy = ld;
            if (s == 0)
            {
                bindp::Xq xq;
                xq.on = 1; xq.src = P.x; xq.ld = HID; xq.K = HID; xq.M = M; xq.dst = sc.x8_big;
                g.pre = xq;
            }
            bindp::launch(g, st);
        }
    }
    else
    {
        const dtr::Mat* m3[3] = {&w.qt, &w.kt, &w.vt};
        for (int s = 0; s < 3; ++s)
        {
            bindp::Gemm g;
            if (const char* e = bindp::make_gemm_tr(*m3[s], M, sc.x8_big, P.ksc.P + pl.q() + s * dl, 2, g))
                die(std::string("pf_kda_proj_gemm: ") + e);
            if (m3[s]->N != dl || m3[s]->K != HID) die("pf_kda_proj_gemm: q/k/v trellis shape is not [nh * 128][HID]");
            g.ldy = ld; g.osvh = m3[s]->svh;
            if (s == 0 || m3[s]->suh != m3[s - 1]->suh)     // q, k, v share the input rotation in the Flash import
            {
                bindp::Xq xq;
                xq.on = 1; xq.src = P.x; xq.ld = HID; xq.K = HID; xq.M = M; xq.dst = sc.x8_big; xq.suh = m3[s]->suh; xq.h16 = g.h16;
                g.pre = xq;
            }
            if (kda_proj_ws::enabled() && kda_proj_ws::supported(g)) G6RT_CK(kda_proj_ws::launch(g, st));
            else bindp::launch(g, st);
        }
    }
    // Reuse prefill scratch after q/k/v: each buffer holds >= round_up(M,8)*HID*2 bytes.
    // 0 = old MX-row host loop, 1 = full-M 32-row tiles, 2 = full-M 64-row tiles.
    const int tail = pf_kda_tail();
    if (tail == 1 || tail == 2)
    {
        const bool check = pf_kda_tail_check();
        constexpr int width = 2 * kda::DK + kda::HEADS;
        std::vector<float> ref;
        if (check) {
            ref.resize((size_t)M * width);
            pf_kda_tail_control(m, P, w, a, M, st);
            G6RT_CK(cudaStreamSynchronize(st));
            G6RT_CK(cudaMemcpy2D(ref.data(), width * sizeof(float), P.ksc.P + pl.fa(), ld * sizeof(float),
                                width * sizeof(float), M, cudaMemcpyDeviceToHost));
        }
        unsigned* xf = reinterpret_cast<unsigned*>(sc.x8_big);
        unsigned* xb = reinterpret_cast<unsigned*>(sc.x8_up);
        if (P.attn_frag) { xf = pf_attn_f16(P); xb = pf_attn_bf16(P); }   // built by the mHC kernel
        else flash_mhc::fragments(P.xn, M, HID, xf, xb, st);
        kda::ProjArgs pa;
        pa.xfrag = xb; pa.M = M; pa.ld = ld; pa.odt = gv::O_F32;
        if (w.fa_mx) pa.add(w.fa_mx, P.ksc.P + pl.fa(), HID, kda::DK, 1, 32, xf, 1);
        else         pa.add(w.fa,    P.ksc.P + pl.fa(), HID, kda::DK, 1, 32);
        if (w.ga_mx) pa.add(w.ga_mx, P.ksc.P + pl.ga(), HID, kda::DK, 1, 32, xf, 1);
        else         pa.add(w.ga,    P.ksc.P + pl.ga(), HID, kda::DK, 1, 32);
        pa.add(w.b, P.ksc.P + pl.b(), HID, kda::HEADS, 1, 32);
        kda::launch_tail_gemm(pa, tail == 2 ? 64 : 32, st);
        if (check) {
            std::vector<float> got(ref.size());
            G6RT_CK(cudaStreamSynchronize(st));
            G6RT_CK(cudaMemcpy2D(got.data(), width * sizeof(float), P.ksc.P + pl.fa(), ld * sizeof(float),
                                width * sizeof(float), M, cudaMemcpyDeviceToHost));
            for (int p = 0; p < 3; ++p) {
                const int first = p * kda::DK, n = p == 2 ? kda::HEADS : kda::DK;
                size_t diff = 0; double mx = 0, sq = 0;
                for (int row = 0; row < M; ++row) for (int j = 0; j < n; ++j) {
                    const size_t k = (size_t)row * width + first + j;
                    diff += std::memcmp(&ref[k], &got[k], sizeof(float)) != 0;
                    const double d = (double)ref[k] - got[k];
                    mx = std::max(mx, std::fabs(d)); sq += d * d;
                }
                fprintf(stderr, "# KTAIL_CHECK rank %d M %d matrix %d different %zu max %.9g rms %.9g\n",
                        m.rank, M, p, diff, mx, std::sqrt(sq / ((size_t)M * n)));
                if (diff) die("KDA prefill tail: per-matrix byte gate failed");
            }
        }
        return;
    }
    pf_kda_tail_control(m, P, w, a, M, st);
}
// G6_PF_SP_FIND (default on with G6_PF_SP; =0 off): an MoE layer's final sum (k_pf_fin RS form) writes every row into
// its owner's slice of the collective-2 send slot, so the put only rings the doorbell and the owner pass reads its own
// rows from the slot. Byte-equal (the same values, one copy fewer). TP3 8K A/B: prefill 2434 -> 2455 tok/s, rank-0 8K
// prefill -25 ms, T=0 hashes unchanged.
inline bool pf_sp_find()
{
    static const bool on = !(getenv("G6_PF_SP_FIND") && atoi(getenv("G6_PF_SP_FIND")) == 0);
    return on;
}
// G6_PF_SP_ATTD (default on with G6_PF_SP; =0 off): the attention o_proj GEMM (KDA and DSA, GEMM path) writes every row
// into its owner's slice of the collective-0 send slot (bindp::RsOut, the k_gemm_tr RS epilogue: the same pack of the
// same accumulators), so the put only rings the doorbell and the owner pass reads its own rows from the slot.
// Byte-equal. TP3 8K A/B (FIND on): prefill 2455 -> 2473 tok/s, rank-0 8K prefill -28 ms, T=0 hashes unchanged.
inline bool pf_sp_attd()
{
    static const bool on = !(getenv("G6_PF_SP_ATTD") && atoi(getenv("G6_PF_SP_ATTD")) == 0);
    return on;
}
// the RS send-slot target of collective c2 for an M-row lane (FIND / ATTD)
inline bindp::RsOut pf_rs_target(const FlashPrefill& P, int c2, int M)
{
    bindp::RsOut o;
    o.base = P.f.send + (size_t)c2 * P.f.stride; o.pstride = (size_t)P.f.ncoll * P.f.stride;
    o.ss = pf_ss(P.f); o.stepp = P.stepp; o.M = M; o.nr = P.f.nr;
    return o;
}
inline const char* pf_rs_own(const FlashPrefill& P, const bindp::RsOut& o)
{
    return o.base + (size_t)(P.cur_step & 1) * o.pstride + (size_t)P.f.rank * o.ss;
}
inline void pf_kda_oproj_gemm(Model& m, FlashPrefill& P, const kda::KdaLayer& w, int M, int dl, cudaStream_t st,
                              const bindp::RsOut& rs = bindp::RsOut())
{
    bindp::Scratch& sc = flash_mlp_pf::ensure_state(m, P, nullptr, st).sc;
    if (dl > g6nr::SMALLW || w.ot.K != dl || w.ot.N != HID) die("pf_kda_oproj_gemm: o_proj trellis shape is not [HID][nh * 128]");
    bindp::Gemm g;
    if (const char* e = bindp::make_gemm_tr(w.ot, M, sc.x8_small, P.part[0], 1, g)) die(std::string("pf_kda_oproj_gemm: ") + e);
    g.osvh = w.ot.svh;
    g.pre.on = 1; g.pre.src = reinterpret_cast<const __half*>(P.ksc.OH); g.pre.ld = dl; g.pre.K = dl; g.pre.M = M;
    g.pre.dst = sc.x8_small; g.pre.suh = w.ot.suh; g.pre.h16 = g.h16;
    bindp::launch(g, st, rs);
}

// G9_KDA_SCAN (default 0 = the base KDA scan, untouched): 1 = the CH16 bf16-MMA scan kda::g9::launch,
// a precision arm (not byte-exact). G9_KDA_VS in {1, 2, 4} (default 2) is read only when the scan is on.
inline int pf_g9_scan()
{
    static const int v = getenv("G9_KDA_SCAN") ? atoi(getenv("G9_KDA_SCAN")) : 0;
    if (v != 0 && v != 1) die("G9_KDA_SCAN: 0 or 1 (2..4 = fused F1..F3 are not built)");
    return v;
}
inline int pf_g9_vs()
{
    static const int v = getenv("G9_KDA_VS") ? atoi(getenv("G9_KDA_VS")) : 2;
    return v;
}

// KDA: projections (GEMMs for trellis q/k/v, else the decode GEMVs per MX chunk on the chunk's fragment tables), the
// chunked scan over all M rows (state of slot P.row_slot[0]), o_proj (GEMM for a trellis o_proj, else per MX chunk)
// -> P.part[0]. With the --mtp verify ring the prefill rows use slot 0 = the committed slot of the prompt steps
// (rank.cuh mtp_sequence: kcur 0; several S2 sessions: the session's first slot, pf_inputs_flash), carry mode off.
inline void pf_kda_block(Model& m, FlashPrefill& P, int i, int M, cudaStream_t st)
{
    const kda::KdaLayer& w = m.layers[i].kda_w;
    kda::Args a = kda::make_args(w, m.kda_st, P.ksc, P.row_slot, M);
    a.src = nullptr;
    const int ld = a.pl().ld(), dl = a.dl();
    const bool gemm = pf_kda_gemm_on(w);
    if (gemm) pf_kda_proj_gemm(m, P, w, a, M, st);
    else
        for (int m0 = (pf_need_fp32_xn(P, "pf_kda_block"), 0); m0 < M; m0 += MX)
        {
            const int r = std::min(MX, M - m0);
            flash_mhc::fragments(P.xn + (size_t)m0 * HID, r, HID, m.sc.xf_x, m.sc.xb_x, st);
            kda::launch_proj_in(w, a, P.ksc.P + (size_t)m0 * ld, m.sc.xb_x, r, st, P.x + (size_t)m0 * HID, m.sc.xf_x, &m.sc.dt);
        }
    flash_pf_prof().mark(0, st);
    if (pf_g9_scan())
    {
        static bool said = false;   // once per process: lets the A/B harness check the ON arm really ran the G9 scan
        if (!said) { said = true; fprintf(stderr, "# rank %d: G9_KDA_SCAN on (VS %d)\n", P.f.rank, pf_g9_vs()); }
        if (const char* e = kda::g9::launch(a, P.kps, pf_g9_vs(), st)) die(e);
    }
    else kda::launch_core_prefill(kda::make_pf_args(a, P.kps), st);
    flash_pf_prof().mark(1, st);
    P.att_own = nullptr;
    if (gemm && w.ot.w)
    {
        const bindp::RsOut rs = pf_sp() && pf_sp_attd() ? pf_rs_target(P, 0, M) : bindp::RsOut();
        pf_kda_oproj_gemm(m, P, w, M, dl, st, rs);
        if (rs.base) P.att_own = pf_rs_own(P, rs);
    }
    else
        for (int m0 = 0; m0 < M; m0 += MX)
        {
            const int r = std::min(MX, M - m0);
            kda::launch_oproj(w, P.ksc.OF + (size_t)(m0 / 8) * dl * 8, r, P.part[0] + (size_t)m0 * HID, gv::O_BF16, st,
                              P.ksc.OH + (size_t)m0 * dl, &m.sc.dt);
        }
    flash_pf_prof().mark(2, st);
}

// DSA: the decode block per MX chunk; o_proj into m.sc.part[0] (decode slot mode off), copied to P.part[0]
inline void pf_dsa_block_p1(Model& m, FlashPrefill& P, int i, int M, int context, cudaStream_t st)
{
    pf_need_fp32_xn(P, "pf_dsa_block_p1");
    Layer& y = m.layers[i];
    for (int m0 = 0; m0 < M; m0 += MX)
    {
        const int r = std::min(MX, M - m0);
        if (!y.built[r]) die("pf_dsa_block_p1: bind_dsa_layer did not run");
        bind::Layer b = y.L[r];
#ifdef DEC_SLOT
        bind::set_slots(b, dslot::Slot{}, dslot::Slot{});
#endif
        flash_mhc::fragments(P.xn + (size_t)m0 * HID, r, HID, m.sc.xf_x, m.sc.xb_x, st);
        G6RT_CK(cudaMemcpyAsync(m.pos, P.pos + m0, r * sizeof(int), cudaMemcpyDeviceToDevice, st));
        if (b.x_in.n) bind::launch(b.x_in, st);
        else { bind::launch(b.q_a, st); bind::launch(b.kv_a, st); bind::launch(b.idx_k, st); bind::launch(b.idx_w, st); bind::launch(b.idx_g, st); }
        glue::k_rmsnorm_rows<__half><<<r, 256, 0, st>>>(m.q_a, QA, QA, y.q_a_norm, geom::RMS_EPS_FLASH, m.q_a_n, QA,
                                                        b.xf_qa.on ? b.xf_qa.f16 : nullptr, b.xf_qa.on ? b.xf_qa.bf16 : nullptr);
        bind::launch_q_b(b, st);
        bind::launch(b.idx_q, st);
        flash_pf_prof().mark(3, st);
        // context = position after the chunk's last row (rows are consecutive; context - M = the first position)
        const G6AttnStep s = dsa_step(m, r, context - M + m0 + r);
        G6RT_CK(g6_attn_bind_run(&y.ab, &s, st));
        flash_pf_prof().mark(4, st);
        bind::launch(b.xf_attn, st);
        bind::launch_o_proj(b, st);
        G6RT_CK(cudaMemcpyAsync(P.part[0] + (size_t)m0 * HID, m.sc.part[0], (size_t)r * HID * 2, cudaMemcpyDeviceToDevice, st));
        flash_pf_prof().mark(5, st);
    }
}

// MLP (dense or MoE): a decode block per MX chunk reading P.xn2 / P.x2 rows in place, writing P.part[1] rows in place
inline void pf_mlp_block_p1(Model& m, FlashPrefill& P, int i, int M, cudaStream_t st)
{
    Layer& y = m.layers[i];
    for (int m0 = 0; m0 < M; m0 += MX)
    {
        const int r = std::min(MX, M - m0);
        flash_mlp::Block b;
        if (const char* e = flash_mlp::make(y.flash_mlp_w, r, P.xn2 + (size_t)m0 * HID,
                                            reinterpret_cast<uint16_t*>(P.part[1] + (size_t)m0 * HID), m.plan, m.flash_mlp_sc,
                                            m.sc, b, -1, P.x2 + (size_t)m0 * HID))
            die(std::string("pf_mlp_block_p1: ") + e);
        b.topk = moe_topk_cut(geom::TOPK);     // G6_MOE_TOPK (this path is main-model only)
        flash_mlp::launch(b, st);
    }
    flash_pf_prof().mark(y.moe ? 7 : 6, st);
}

// dispatch: P2 GEMM bodies where built. DSA layers: pf_dsa_block_gemm unless G6_FPF_DSA_P1=1. MLP layers:
// pf_mlp_block_gemm, dense too unless G6_FPF_DENSE_P1=1.
inline void pf_dsa_block(Model& m, FlashPrefill& P, int i, int M, int context, cudaStream_t st)
{
    if (!pf_force_p1() && !pf_dsa_p1())
    {
        static const bool amark = pf_prof().on && getenv("G6_PF_AMARK") && atoi(getenv("G6_PF_AMARK")) == 1;
        void (*const am0)(int, void*) = g6_attn_mark;
        if (amark) g6_attn_mark = flash_pf_amark;
        const bindp::RsOut rs = pf_sp() && pf_sp_attd() ? pf_rs_target(P, 0, M) : bindp::RsOut();
        P.att_own = nullptr;
        if (pf_dsa_block_gemm(m, P, i, M, context, st, [](int k, cudaStream_t s) { flash_pf_prof().mark(k, s); },
                              P.attn_frag ? pf_attn_f16(P) : nullptr, P.attn_frag ? pf_attn_bf16(P) : nullptr,
                              P.pool_transport.fabric ? &P.pool_transport : nullptr, rs) && rs.base)
            P.att_own = pf_rs_own(P, rs);
        if (amark) g6_attn_mark = am0;
    }
    else
    {
        P.att_own = nullptr;
        pf_dsa_block_p1(m, P, i, M, context, st);
    }
}
inline void pf_mlp_block(Model& m, FlashPrefill& P, int i, int M, cudaStream_t st)
{
    static const bool dense_p1 = getenv("G6_FPF_DENSE_P1") && atoi(getenv("G6_FPF_DENSE_P1")) == 1;
    if (!pf_force_p1() && (m.layers[i].moe || !dense_p1))
    {
        P.fin_own = nullptr;
        if (pf_sp() && pf_sp_find() && m.layers[i].moe)
        {
            P.fin_rs = pf_rs_target(P, 2, M);
            P.fin_own = pf_rs_own(P, P.fin_rs);
        }
        pf_mlp_block_gemm(m, P, i, M, st);
        P.fin_rs.base = nullptr;
        flash_pf_prof().mark(m.layers[i].moe ? 7 : 6, st);
    }
    else pf_mlp_block_p1(m, P, i, M, st);
}

// ---------------------------------------------------------------------------------------------- runtime
// G6_PF_STAGE_ASYNC: issue the wait for the peers' partials of collective c and their copies into device memory on the
// lane's side stream, right after our put. pf_consume_flash(c) waits ready and records used.
// Why the side stream reads the same state as the inline code did, and why the copies cannot overwrite a buffer still read:
//   step cell: the lane's next k_set_step follows consume(c) in compute-stream order, and consume(c) waits ready,
//     so k_wait_peers on the side stream reads the step this put used.
//   fabric slot parity: the peer rewrites our slot for (c, s) only at its step s + 2, which needs our put of (c, s + 1);
//     that put follows consume(c, s) on the compute stream, and consume(c, s) waits ready = the copy of (c, s) done.
//     This breaks if any put is moved before the consume that waits ready.
//   peer buffers: the copy waits used, recorded after the previous consume's mHC kernel. A wait on a never-recorded
//     event is a no-op.
// At most one issue per lane is outstanding (issue, consume, issue, consume ...); the schedule test checks it.
inline void pf_stage_issue(FlashPrefill& P, int c, int M, cudaStream_t st)
{
    PfPeerStage& S = P.peer_stage;
    const size_t payload = (size_t)M * HID * 2;
    G6RT_CK(cudaEventRecord(S.put, st));
    G6RT_CK(cudaStreamWaitEvent(S.stream, S.put, 0));
    G6RT_CK(cudaStreamWaitEvent(S.stream, S.used, 0));
    glue::k_wait_peers<<<1, 32, 0, S.stream>>>(P.f, P.stepp, c, payload, glue::ERR_AR);
    for (int p = 0; p < g6nr::NR; ++p)
    {
        if (p == P.f.rank) continue;
        const char* src = P.f.recv + (((size_t)(P.cur_step & 1) * P.f.ncoll + c) * P.f.nr + p) * P.f.stride;
        G6RT_CK(cudaMemcpyAsync(S.peer[p], src, payload, cudaMemcpyDefault, S.stream));
    }
    G6RT_CK(cudaEventRecord(S.ready, S.stream));
}
inline void pf_sp_issue(FlashPrefill& P, int c, int M, cudaStream_t st, bool x16 = false);
inline void pf_put_flash(FlashPrefill& P, int c, int M, cudaStream_t st)
{
    if (pf_sp())
    {
        if (c == 1 && P.fin_own) k_publish<<<1, 32, 0, st>>>(P.f, P.stepp, 2, P.tickets + 2);        // G6_PF_SP_FIND
        else if (c == 0 && P.att_own) k_publish<<<1, 32, 0, st>>>(P.f, P.stepp, 0, P.tickets + 0);   // G6_PF_SP_ATTD
        else k_put_rs<<<pf_sp_put_cta(), 128, 0, st>>>(reinterpret_cast<const uint4*>(P.part[c]), P.f, P.stepp, 2 * c, M, P.tickets + 2 * c);
        pf_sp_issue(P, 2 * c, M, st);
        return;
    }
    static const int ncta = getenv("G6_PF_PUT_CTA") ? atoi(getenv("G6_PF_PUT_CTA")) : 96;
    k_put_publish_n<<<ncta, 128, 0, st>>>(reinterpret_cast<const uint4*>(P.part[c]), P.f, P.stepp, c, M * (HID * 2 / 16),
                                          P.tickets + c);
    if (P.peer_stage.on) pf_stage_issue(P, c, M, st);
}
inline void pf_mhc_input(Model& m, FlashPrefill& P, int i, bool ffn, int M, cudaStream_t st)
{
    const bool frag = !ffn && !P.sp_view && pf_attn_frag_ok(m, i);
    if (!ffn) P.attn_frag = frag;
    flash_mhc::prepare(mhc_buffers(P), m.layers[i].hc[ffn ? 1 : 0], M, HID, geom::RMS_EPS_FLASH, ffn ? P.x2 : P.x, st,
                       ffn ? P.xn2 : frag ? nullptr : P.xn, frag ? pf_attn_f16(P) : nullptr, frag ? pf_attn_bf16(P) : nullptr);
}
// the mHC kernels that read the rank partials through blk (MhcFabricBlock: the code before G6_PF_STAGE_ASYNC;
// MhcFabricPeerBlock: every peer partial already in device memory). wp != 0 only for MhcFabricBlock.
template <class Blk>
inline void pf_consume_launch(Model& m, FlashPrefill& P, int i, int c, int M, cudaStream_t st, const Blk& blk, size_t wp)
{
    const int pf_fuse = pf_fuse_level();
    if (c == 0 || i + 1 < m.nl)
    {
        if (pf_fuse >= 1 && HID == 4096)
        {
            const int wl = c == 0 ? i : i + 1, wh = c == 0 ? 1 : 0;
            const flash_mhc::Weights& w = m.layers[wl].hc[wh];
            const bool rr = !P.fnp.empty() && !wp && pf_mhc_rr();
            const bool frag = c == 1 && rr && !P.sp_view && pf_attn_frag_ok(m, i + 1);
            if (c == 1) P.attn_frag = frag;
            if (rr)   // nothing to wait for: re-read rows, bit-identical
            {
                flash_mhc::k_mhc_out_inv_prepare_rr<Blk, PF_MHC_RR_R, 1><<<(M + PF_MHC_RR_R - 1) / PF_MHC_RR_R, flash_mhc::THREADS, 0, st>>>(
                    flash_mhc::side(mhc_buffers(P), c), blk, w, P.fnp[wl * 2 + wh], M, geom::RMS_EPS_FLASH, c == 0 ? P.x2 : P.x,
                    c == 0 ? P.xn2 : frag ? nullptr : P.xn, frag ? pf_attn_f16(P) : nullptr, frag ? pf_attn_bf16(P) : nullptr,
                    P.agd_x, P.agd_xn);
                P.agd_done = P.agd_x || P.agd_xn;
            }
            else if (!P.fnp.empty())   // packed fn, 2 rows per CTA: bit-identical to k_mhc_out_inv_prepare
                flash_mhc::k_mhc_out_inv_prepare_pk<Blk, 2><<<(M + 1) / 2, flash_mhc::THREADS, 0, st>>>(
                    flash_mhc::side(mhc_buffers(P), c), blk, w, P.fnp[wl * 2 + wh], M, geom::RMS_EPS_FLASH, c == 0 ? P.x2 : P.x,
                    c == 0 ? P.xn2 : P.xn);
            else
                flash_mhc::k_mhc_out_inv_prepare<<<M, flash_mhc::THREADS, 0, st>>>(
                    flash_mhc::side(mhc_buffers(P), c), blk, w, HID, geom::RMS_EPS_FLASH, c == 0 ? P.x2 : P.x, c == 0 ? P.xn2 : P.xn);
        }
        else
        {
            flash_mhc::k_mhc_out_inv<<<M, flash_mhc::THREADS, 0, st>>>(flash_mhc::side(mhc_buffers(P), c), blk, HID, geom::RMS_EPS_FLASH);
            if (c == 0) pf_mhc_input(m, P, i, true, M, st);
            else pf_mhc_input(m, P, i + 1, false, M, st);
        }
    }
    else
    {
        P.attn_frag = false;   // the final norm rows are FP32 (MTP draft, head)
        flash_mhc::k_mhc_out_inv_final<<<M, flash_mhc::THREADS, 0, st>>>(flash_mhc::side(mhc_buffers(P), c), blk, m.final_norm, HID, geom::RMS_EPS_FLASH,
                                                                         P.x, P.xn, P.resid);
    }
}
// recv slot of collective c from rank p, this lane's current layer-step parity (host address of the pinned region)
inline const char* pf_sp_slot(const FlashPrefill& P, int c, int p)
{
    return P.f.recv + (((size_t)(P.cur_step & 1) * P.f.ncoll + c) * P.f.nr + p) * P.f.stride;
}
// on the lane's copy stream, ordered after what st has queued: wait for collective c's peer data, copy it to device
// memory (RS: this rank's slices -> sp_peer; AG: peer rows -> xn2 / xn at their row offsets), then sp_ready[c].
// The compute stream runs the other lane's block meanwhile. Every buffer written here was last read by a kernel that
// st queued before this put, so the sp_put wait also orders the reuse.
inline void pf_sp_issue(FlashPrefill& P, int c, int M, cudaStream_t st, bool x16)
{
    const int nr = P.f.nr, me = P.f.rank;
    G6RT_CK(cudaEventRecord(P.sp_put[c], st));
    G6RT_CK(cudaStreamWaitEvent(P.sp_st, P.sp_put[c], 0));
    if (!(c & 1))
    {
        k_wait_rs<<<1, 32, 0, P.sp_st>>>(P.f, P.stepp, c, M, false);
        const int rows = pf_rows(M, me, nr);
        if (rows)
            for (int p = 0; p < nr; ++p)
                if (p != me) G6RT_CK(cudaMemcpyAsync(P.sp_peer[p], pf_sp_slot(P, c, p), (size_t)rows * HID * 2, cudaMemcpyDefault, P.sp_st));
    }
    else if (x16)   // G6_PF_SP_X16: the peers' FP16 MLP-input rows straight into x2
    {
        k_wait_ag<<<1, 32, 0, P.sp_st>>>(P.f, P.stepp, c, M, HID * 2);
        for (int p = 0; p < nr; ++p)
        {
            const int r0 = pf_r0(M, p, nr), rows = pf_rows(M, p, nr);
            if (p != me && rows) G6RT_CK(cudaMemcpyAsync(P.x2 + (size_t)r0 * HID, pf_sp_slot(P, c, p), (size_t)rows * HID * 2, cudaMemcpyDefault, P.sp_st));
        }
    }
    else if (P.sp_agk[c]) {}   // G6_PF_SP_AGK: pf_sp_gather waits and copies on the compute stream
    else
    {
        float* xn = c == 1 ? P.xn2 : P.xn;
        k_wait_ag<<<1, 32, 0, P.sp_st>>>(P.f, P.stepp, c, M, HID * 4);
        for (int p = 0; p < nr; ++p)
        {
            const int r0 = pf_r0(M, p, nr), rows = pf_rows(M, p, nr);
            if (p != me && rows) G6RT_CK(cudaMemcpyAsync(xn + (size_t)r0 * HID, pf_sp_slot(P, c, p), (size_t)rows * HID * 4, cudaMemcpyDefault, P.sp_st));
        }
    }
    G6RT_CK(cudaEventRecord(P.sp_ready[c], P.sp_st));
}
__global__ void k_sp_f16(const float* __restrict__ src, __half* __restrict__ dst, size_t n)
{
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) dst[i] = __float2half_rn(src[i]);
}
// G6_PF_SP_AGK: grid (PF_SP_AGK_CTA, nr), block 256, after k_wait_ag on the same stream: peer p's FP32 rows (offset 0
// of its recv slot) -> xn rows pf_r0(M, p).. and their __float2half_rn copy -> x rows (what the copy engine + k_sp_f16 write)
constexpr int PF_SP_AGK_CTA = 96;
__global__ void __launch_bounds__(256) k_sp_ag_f16(g6::FabricDev f, const u64* stepp, int c, int M, float* __restrict__ xn,
                                                   __half* __restrict__ x)
{
    const int p = blockIdx.y;
    if (p == f.rank) return;
    const size_t r0 = (size_t)pf_r0(M, p, f.nr), n4 = (size_t)pf_rows(M, p, f.nr) * HID / 4;
    const float4* s = reinterpret_cast<const float4*>(glue::peer_slot(f, *stepp, c, p));
    float4* d = reinterpret_cast<float4*>(xn + r0 * HID);
    uint2* h = reinterpret_cast<uint2*>(x + r0 * HID);
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n4; i += (size_t)gridDim.x * 256)
    {
        const float4 v = __ldcv(s + i);
        d[i] = v;
        const __half2 lo = __halves2half2(__float2half_rn(v.x), __float2half_rn(v.y));
        const __half2 hi = __halves2half2(__float2half_rn(v.z), __float2half_rn(v.w));
        h[i] = make_uint2(*reinterpret_cast<const unsigned*>(&lo), *reinterpret_cast<const unsigned*>(&hi));
    }
}
// G6_PF_SP_X16 applies to collective 1 (the MLP input) of an MoE layer whose router runs row-split
inline bool pf_sp_x16_layer(const Model& m, const FlashPrefill& P, int i, int c, int M)
{
    return pf_sp_x16() && c == 0 && m.layers[i].moe && P.route_split.fabric && flash_route_split::active(M, P.route_split.f.nr);
}
// G6_PF_SP, first half of a consume: this rank's rows of every partial have landed (collective 2c), mHC on them,
// then the FP32 normalized rows (xn2 after attention, xn after the MLP / the final norm) into collective 2c + 1
inline void pf_sp_own(Model& m, FlashPrefill& P, int i, int c, int M, cudaStream_t st)
{
    const int crs = 2 * c, cag = crs + 1, nr = P.f.nr, me = P.f.rank;
    const int r0 = pf_r0(M, me, nr), rows = pf_rows(M, me, nr);
    G6RT_CK(cudaStreamWaitEvent(st, P.sp_ready[crs], 0));   // the peers' slices are in sp_peer (pf_sp_issue)
    bool agd = false;
    if (rows)
    {
        FlashPrefill Q = P;   // the owner rows as a sub-batch of their own
        const size_t r = (size_t)r0;
        Q.resid4 += r * geom::HC * HID; Q.hc_inv += r; Q.hc_mix += r * geom::HC_MIX; Q.hc_pre += r * geom::HC;
        Q.hc_post += r * geom::HC; Q.hc_comb += r * geom::HC * geom::HC; Q.hc_done += r;
        Q.xn += r * HID; Q.xn2 += r * HID; Q.x += r * HID; Q.x2 += r * HID; Q.resid += r * HID;
        Q.sp_view = true;
        const char* slot_own = c == 1 ? P.fin_own : P.att_own;   // G6_PF_SP_FIND / G6_PF_SP_ATTD
        __nv_bfloat16* own = slot_own ? reinterpret_cast<__nv_bfloat16*>(const_cast<char*>(slot_own)) : P.part[c] + r * HID;
        MhcFabricPeerBlock blk{MhcFabricBlock{P.f, P.stepp, crs, 0, own}};
        for (int p = 0; p < nr; ++p) blk.peers[p] = P.sp_peer[p];
        if (pf_sp_agd())   // G6_PF_SP_AGD: this lane's AG send slot of the current step (glue::send_slot)
        {
            char* slot = P.f.send + ((size_t)(P.cur_step & 1) * P.f.ncoll + cag) * P.f.stride;
            if (pf_sp_x16_layer(m, P, i, c, M)) Q.agd_x = reinterpret_cast<__half*>(slot);
            else Q.agd_xn = reinterpret_cast<float*>(slot);
        }
        pf_consume_launch(m, Q, i, c, rows, st, blk, 0);
        agd = Q.agd_done;
    }
    P.attn_frag = false;
    if (pf_sp_x16_layer(m, P, i, c, M))
    {
        if (agd) k_publish<<<1, 32, 0, st>>>(P.f, P.stepp, cag, P.tickets + cag);   // G6_PF_SP_AGD: rows already in the slot
        else k_put_publish_n<<<pf_sp_put_cta(), 128, 0, st>>>(reinterpret_cast<const uint4*>(P.x2 + (size_t)r0 * HID), P.f, P.stepp, cag, rows * (HID * 2 / 16), P.tickets + cag);
        pf_sp_issue(P, cag, M, st, true);
        return;
    }
    const float* src = (c == 0 ? P.xn2 : P.xn) + (size_t)r0 * HID;
    if (agd) k_publish<<<1, 32, 0, st>>>(P.f, P.stepp, cag, P.tickets + cag);   // G6_PF_SP_AGD: rows already in the slot
    else k_put_publish_n<<<pf_sp_put_cta(), 128, 0, st>>>(reinterpret_cast<const uint4*>(src), P.f, P.stepp, cag, rows * (HID * 4 / 16), P.tickets + cag);
    pf_sp_issue(P, cag, M, st);
}
// second half: every owner's rows (collective 2c + 1) into xn2 / xn, then the FP16 block input of all M rows
inline void pf_sp_gather(FlashPrefill& P, int c, int M, cudaStream_t st, bool x16 = false)
{
    if (x16) { G6RT_CK(cudaStreamWaitEvent(st, P.sp_ready[2 * c + 1], 0)); return; }   // the peers' x2 rows are in place
    float* xn = c == 0 ? P.xn2 : P.xn;
    __half* x = c == 0 ? P.x2 : P.x;
    const int cag = 2 * c + 1;
    if (P.sp_agk[cag])   // G6_PF_SP_AGK: pf_sp_issue left the wait and the copies to this pass
    {
        k_wait_ag<<<1, 32, 0, st>>>(P.f, P.stepp, cag, M, HID * 4);
        k_sp_ag_f16<<<dim3(PF_SP_AGK_CTA, P.f.nr), 256, 0, st>>>(P.f, P.stepp, cag, M, xn, x);
        return;
    }
    G6RT_CK(cudaStreamWaitEvent(st, P.sp_ready[cag], 0));   // the peers' rows are in xn (pf_sp_issue)
    for (int p = 0; p < P.f.nr; ++p)   // peer rows only: the owner pass wrote this rank's xn / x rows
    {
        const int r0 = pf_r0(M, p, P.f.nr), rows = pf_rows(M, p, P.f.nr);
        if (p == P.f.rank || !rows) continue;
        k_sp_f16<<<256, 256, 0, st>>>(xn + (size_t)r0 * HID, x + (size_t)r0 * HID, (size_t)rows * HID);
    }
}
// wait for every rank's partial of collective c, sum in rank order (FP32), update the streams, then the next block's
// input (c = 0: mHC(hc_ffn) of layer i; c = 1: mHC(hc_attn) of layer i + 1, or the final norm rows after the last layer)
inline void pf_consume_flash(Model& m, FlashPrefill& P, int i, int c, int M, cudaStream_t st)
{
    if (pf_sp())
    {
        pf_sp_own(m, P, i, c, M, st);
        pf_sp_gather(P, c, M, st, pf_sp_x16_layer(m, P, i, c, M));
        return;
    }
    const int pf_fuse = pf_fuse_level();
    if (P.peer_stage.on)   // the wait and the copies were issued at the put: wait for them, read the device copies
    {
        PfPeerStage& S = P.peer_stage;
        G6RT_CK(cudaStreamWaitEvent(st, S.ready, 0));
        MhcFabricPeerBlock blk{MhcFabricBlock{P.f, P.stepp, c, 0, P.part[c]}};
        for (int p = 0; p < g6nr::NR; ++p) blk.peers[p] = S.peer[p];
        pf_consume_launch(m, P, i, c, M, st, blk, 0);
        G6RT_CK(cudaEventRecord(S.used, st));
        return;
    }
    const size_t payload = (size_t)M * HID * 2;
    const bool stage = P.peer_dev && pf_fuse >= 1;
    if (pf_fuse < 2 || stage)
    {
        pf_comm_xt(P, c, 0, st);
        glue::k_wait_peers<<<1, 32, 0, st>>>(P.f, P.stepp, c, payload, glue::ERR_AR);
        pf_comm_xt(P, c, 1, st);
    }
    if (stage)   // the peer's partial: pinned fabric slot -> device memory by the copy engine (GPU loads of pinned are slow)
    {
        const int p = 1 - P.f.rank;
        const char* src = P.f.recv + (((size_t)(P.cur_step & 1) * P.f.ncoll + c) * P.f.nr + p) * P.f.stride;
        G6RT_CK(cudaMemcpyAsync(P.peer_dev, src, payload, cudaMemcpyDefault, st));
    }
    if (pf_fuse < 2 || stage) pf_comm_xt(P, c, 2, st);
    const size_t wp = pf_fuse >= 2 && !stage ? payload : 0;
    MhcFabricBlock blk{P.f, P.stepp, c, wp, pf_fuse >= 1 ? P.part[c] : nullptr};
    blk.pxt = P.pxt;
    if (stage) blk.peer_part = P.peer_dev;
    pf_consume_launch(m, P, i, c, M, st, blk, wp);
}

// one sub-batch: rows P.pos[0..M), layer i is layer-step ls0 + i of the prefill fabric. residual_copy(slot, M), if set,
// is called at the decode dump slots (0 = after embed, 2i + 1 = after layer i's attention, 2i + 2 = after its MLP)
inline void pf_step_flash(Model& m, FlashPrefill& P, int M, int context, u64 ls0, cudaStream_t st,
                          const std::function<void(int, int)>& residual_copy = {})
{
    if (M < 1 || M > P.MP) die("pf_step_flash: M outside 1..MP");
    FlashPfProf& pr = flash_pf_prof();
    pr.last = 0;
    pr.mark(-1, st);
    P.sp_agk[1] = P.sp_agk[3] = pf_sp() && pf_sp_agk();   // single lane: every gather directly follows its owner pass
    flash_mhc::k_mhc_embed<<<M, flash_mhc::THREADS, 0, st>>>(mhc_buffers(P), P.tok, m.emb, HID, geom::RMS_EPS_FLASH);
    pf_mhc_input(m, P, 0, false, M, st);
    if (residual_copy) residual_copy(0, M);
    pr.mark(10, st);
    for (int i = 0; i < m.nl; ++i)
    {
        k_set_step<<<1, 1, 0, st>>>(P.stepp, ls0 + i);
        P.cur_step = ls0 + i;
        if (geom::is_kda(m.layers[i].l)) pf_kda_block(m, P, i, M, st);
        else pf_dsa_block(m, P, i, M, context, st);
        pf_xt(P, 0, pxt::PX_PROD, M, st);
        pf_put_flash(P, 0, M, st);
        pf_xt(P, 0, pxt::PX_PUT, M, st);
        pr.mark(8, st);
        pf_consume_flash(m, P, i, 0, M, st);
        pf_xt(P, 0, pxt::PX_END, M, st);
        if (residual_copy) residual_copy(2 * i + 1, M);
        pr.mark(9, st);
        pf_mlp_block(m, P, i, M, st);
        pf_xt(P, 1, pxt::PX_PROD, M, st);
        pf_put_flash(P, 1, M, st);
        pf_xt(P, 1, pxt::PX_PUT, M, st);
        pr.mark(8, st);
        pf_consume_flash(m, P, i, 1, M, st);
        pf_xt(P, 1, pxt::PX_END, M, st);
        if (residual_copy) residual_copy(2 * i + 2, M);
        pr.mark(9, st);
    }
}

// Two full-sized lanes on ONE compute stream. Each lane retains its row shape, fabric region,
// step cell and activations. All shared model scratch stays stream-ordered. A's earlier rows
// update each layer's KDA state / DSA cache before B's later rows. NIC/proxy progress is independent:
// a consume follows one producer block of the other lane, hiding the earlier lane's wire time.
inline void pf_step2_flash(Model& m, FlashPrefill& A, FlashPrefill& B, int MA, int MB,
                           int ctxA, int ctxB, u64 lsA, u64 lsB, cudaStream_t st)
{
    if (MA < 1 || MA > A.MP || MB < 1 || MB > B.MP || ctxB - MB != ctxA)
        die("pf_step2_flash: invalid or non-adjacent lane rows");
    FlashPfProf& pr = flash_pf_prof();
    pr.last = 0;
    pr.mark(-1, st);
    auto embed = [&](FlashPrefill& P, int M) {
        flash_mhc::k_mhc_embed<<<M, flash_mhc::THREADS, 0, st>>>(mhc_buffers(P), P.tok, m.emb, HID, geom::RMS_EPS_FLASH);
        pf_mhc_input(m, P, 0, false, M, st);
        pr.mark(10, st);
    };
    auto put = [&](FlashPrefill& P, int c, int M) {
        pf_xt(P, c, pxt::PX_PROD, M, st);
        pf_put_flash(P, c, M, st);
        pf_xt(P, c, pxt::PX_PUT, M, st);
        pr.mark(8, st);
    };
    auto consume = [&](FlashPrefill& P, int i, int c, int M) {
        pf_consume_flash(m, P, i, c, M, st);
        pf_xt(P, c, pxt::PX_END, M, st);
        pr.mark(9, st);
    };
    auto attn = [&](FlashPrefill& P, int i, int M, int context, u64 step) {
        k_set_step<<<1, 1, 0, st>>>(P.stepp, step);
        P.cur_step = step;
        if (geom::is_kda(m.layers[i].l)) pf_kda_block(m, P, i, M, st);
        else pf_dsa_block(m, P, i, M, context, st);
        put(P, 0, M);
    };
    embed(A, MA);
    embed(B, MB);
    // G6_PF_SP_AGK: lane B's gathers directly follow one of A's owner passes (exposed wire time); A's are covered by blocks
    A.sp_agk[1] = A.sp_agk[3] = false;
    B.sp_agk[1] = B.sp_agk[3] = pf_sp() && pf_sp_agk();
    if (pf_sp())
    {
        // each lane: attn -RS-> own0 -AG-> gather0, mlp -RS-> own1 -AG-> gather1, attn ... Every wire hop of one lane
        // is covered by a block of the other: RS by an owner pass, AG by an attention or MLP block.
        auto own = [&](FlashPrefill& P, int i, int c, int M) { pf_sp_own(m, P, i, c, M, st); pr.mark(9, st); };
        auto gather = [&](FlashPrefill& P, int c, int M, int i = -1) { pf_sp_gather(P, c, M, st, i >= 0 && pf_sp_x16_layer(m, P, i, c, M)); pr.mark(9, st); };
        for (int i = 0; i < m.nl; ++i)
        {
            attn(A, i, MA, ctxA, lsA + i);
            if (i > 0) own(B, i - 1, 1, MB);
            own(A, i, 0, MA);
            if (i > 0) gather(B, 1, MB);
            attn(B, i, MB, ctxB, lsB + i);
            gather(A, 0, MA, i);
            pf_mlp_block(m, A, i, MA, st);
            put(A, 1, MA);
            own(B, i, 0, MB);
            own(A, i, 1, MA);
            gather(B, 0, MB, i);
            pf_mlp_block(m, B, i, MB, st);
            put(B, 1, MB);
            gather(A, 1, MA);
        }
        own(B, m.nl - 1, 1, MB);
        gather(B, 1, MB);
        return;
    }
    for (int i = 0; i < m.nl; ++i)
    {
        attn(A, i, MA, ctxA, lsA + i);
        // B still has the previous step here: its MLP receive uses that parity/sequence.
        if (i > 0) consume(B, i - 1, 1, MB);
        attn(B, i, MB, ctxB, lsB + i);
        consume(A, i, 0, MA);
        pf_mlp_block(m, A, i, MA, st);
        put(A, 1, MA);
        consume(B, i, 0, MB);
        pf_mlp_block(m, B, i, MB, st);
        put(B, 1, MB);
        consume(A, i, 1, MA);
    }
    consume(B, m.nl - 1, 1, MB);
}

// Prompt draft outputs are discarded. Keep the decode input/eh/norm and projection arithmetic,
// but append only latent KV and raw/pooled index keys. Actual generation still uses the full draft step.
inline void pf_draft_cache_flash(Draft& D, int M, int context, cudaStream_t st)
{
    Model& d = D.d;
    Layer& y = d.layers[0];
    if (M < 1 || M > MX || !y.built[M]) die("Flash draft cache fill: row binding missing");
    launch_step_in(d, st);
    launch_draft_embed(D, M, false, st);
    const bind::Layer& b = y.L[M];
    bind::launch(b.kv_a, st);
    bind::launch(b.idx_k, st);
    bind::launch(b.idx_g, st);
    G6Attn a = y.ab.args;
    a.rows = M; a.context = context; a.positions = d.pos;
    a.kv_a = d.kv_a; a.index_k = d.idx_k; a.index_gate = d.idx_g;
    a.reuse_indices = nullptr;
    G6RT_CK(g6_attn_append(&a, st));
}

// MTP draft prefill (P1): the decode draft step (layer 45: eh_proj, DSA, MoE, single residual) over the sub-batch's
// rows in MX-row chunks, without graphs. Row r: hidden = main final-norm row r (P.xn), token = the prompt token after
// it (next[r]), position pos0 + r; it writes the MTP layer's KV rows of the prompt. run(di) enqueues the draft fabric
// step and launches it with D.main_xn pointing at the chunk's rows (the step's head output is not used).
inline void pf_draft_flash(Draft& D, FlashPrefill& P, int M, int pos0, const int* next, u64& dstep,
                           const std::function<void(const StepIn&)>& run)
{
    const float* main_xn = D.main_xn;
    for (int m0 = 0; m0 < M; m0 += MX)
    {
        const int r = std::min(MX, M - m0);
        D.main_xn = P.xn + (size_t)m0 * HID;
        run(draft_inputs(++dstep, r, pos0 + m0, next + m0, 0.f, 0u));
    }
    D.main_xn = main_xn;
}

// FNV-1a 64 of the sub-batch's final FP32 residual rows (P.resid [M][HID]); synchronizes the stream
inline unsigned long long pf_hash_flash(FlashPrefill& P, int M, cudaStream_t st)
{
    std::vector<uint32_t> h((size_t)M * HID);
    G6RT_CK(cudaStreamSynchronize(st));
    G6RT_CK(cudaMemcpy(h.data(), P.resid, h.size() * 4, cudaMemcpyDeviceToHost));
    unsigned long long x = 0xcbf29ce484222325ull;
    for (uint32_t w : h)
        for (int k = 0; k < 4; ++k) { x ^= (w >> (8 * k)) & 0xff; x *= 0x100000001b3ull; }
    return x;
}

// G6_PF_HEAD=1 (dumps only): the head over the sub-batch's final norm rows in MX-row chunks; out(r) after each chunk
// with m.hlog holding the chunk's r rows of this rank's vocab slice (stream synchronized)
inline void pf_head_flash(Model& m, FlashPrefill& P, int M, cudaStream_t st, const std::function<void(int)>& out)
{
    for (int m0 = 0; m0 < M; m0 += MX)
    {
        const int r = std::min(MX, M - m0);
        const bind::Head& H = head_for(m, r);
        const bool f16_in = H.lm_head.codec == bind::MXFP8 || H.lm_head.codec == bind::FP8R;
        flash_mhc::fragments(P.xn + (size_t)m0 * HID, r, HID, f16_in ? m.sc.xf_head : nullptr, f16_in ? nullptr : m.sc.xf_head, st);
        bind::launch(H.lm_head, st);
        G6RT_CK(cudaStreamSynchronize(st));
        out(r);
    }
}
}   // namespace g6rt
#endif
