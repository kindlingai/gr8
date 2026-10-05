// g6 rank step: one rank's decode step over a list of real layers + head, as launch helpers.
// Used unchanged by the NR-process runtime (engine/runtime/g6m4.cu, real fabric; NR = G6_NR, six or four) and by the
// one-process NR-simulated-rank driver (engine/runtime/m4sim.cu, loopback fabric), so their outputs must be bit-identical.
//
// Step, M rows of one sequence at consecutive positions (RUNTIME.md §4), collective c = 2*i for layer i's attention
// partial, 2*i+1 for its MLP partial, 2*NL for the head all-gather:
//   launch_embed
//   per layer i:  launch_attn_block (projections, attention, o_proj, partial -> send slot, doorbell)
//                 launch_consume(2i, mlp_norm)          wait for the peers, residual += NR partials, norm
//                 launch_mlp_block  (router, plan, experts + shared / dense MLP, partial -> send slot, doorbell)
//                 launch_consume(2i+1, next norm)
//   launch_head_local (lm_head slice, local argmax / LSE / Gumbel record -> send slot, doorbell), launch_head_merge
// Everything a kernel reads is device memory; the two partials are staged in device memory and copied into the
// (pinned) send slot by one CTA (k_put_publish), which also rings the doorbell.
// Step inputs come from one StepIn block (pinned host memory on the real fabric, device memory in the simulation),
// copied to device cells by k_step_in, the first kernel of the step; k_step_out, the last, writes the sampled ids,
// logprobs and the OR of the attention error words to a StepOut block. Nothing else crosses the host boundary, so
// the step is either plain launches (Model::graph = false, attention through g6_attn_bind_run with the host's
// context) or ONE captured graph per row count M (Model::graph = true, attention through the attention replay entry,
// which reads context / page counts from a device cell that k_step_in writes; engine/attn/REPLAY.md).
// Call prepare(m, M, st) outside a capture before the first step with M rows.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include "draft_shape.h"
#include "serve.h"
#include "../h2d.h"
#include "../kernels/glue.cuh"
#include "../kernels/bind.cuh"
#include "../attn/bind.h"
#if defined(G6_CP) && G6_CP
#include "../kernels/cpglue.cuh"
#endif

namespace g6rt
{

#define G6RT_CK(x) do { cudaError_t e_ = (cudaError_t)(x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(99); } } while (0)
[[noreturn]] inline void die(const std::string& m) { fprintf(stderr, "g6: %s\n", m.c_str()); exit(2); }

constexpr int NBK = 8;              // decode context buckets (graph form), at most
constexpr int MX = 8, QA = 2048, KVA = 576, IDXQ = 4096, IDXK = 128, IDXW = 32;

struct TensorRef
{
    const void* p = nullptr;          // device pointer; nullptr = the pack has no such tensor
    int nd = 0;
    long long shape[4] = {0, 0, 0, 0};
    bool bf16 = false;
    int fmt = 0;                      // routed-expert format of the manifest entry (g6::Entry::expert_format = bind::FMT_*)
};
using Find = std::function<TensorRef(const std::string&)>;

template <typename T> inline T* dmalloc(size_t n)
{
    T* p;
    G6RT_CK(cudaMalloc(&p, n * sizeof(T)));
    G6RT_CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
inline std::string lname(int l, const char* t) { char b[96]; snprintf(b, sizeof b, "L%03d.%s", l, t); return b; }

// device staging [M][6144] BF16 -> this rank's send slot of collective c (16-byte stores), then the doorbell.
// grid 1, block 128.
__global__ void __launch_bounds__(128) k_put_publish(const uint4* src, g6::FabricDev f, const u64* stepp, int c, int n16,
                                                     unsigned* ticket)
{
    const u64 step = *stepp;
    uint4* dst = reinterpret_cast<uint4*>(glue::send_slot(f, step, c));
    for (int i = threadIdx.x; i < n16; i += 128) dst[i] = src[i];
    glue::publish(f, c, g6::coll_seq(step, c, f.ncoll), ticket);
}

// host -> GPU, once per step. 336 bytes (<= 512), read as 32-bit words by one CTA.
struct StepIn
{
    u64 step;
    int M, context;                   // rows of this step; context = last row's position + 1
    int tok[MX], pos[MX];
    glue::SampleRow srow[MX];
    int tgt[MX];                      // teacher forcing: the id each row should predict (< 0 = none)
#ifdef DEC_S2
    int base_page, pad_s2;            // S2 (TP4 option (a), provisional): first pool page of the session (0 = today)
#endif
};
struct StepOut
{
    int tok[MX];
    float lp[MX];
    float lpt[MX];                    // logprob of tgt (-inf if none)
    unsigned attn_err, pad_;          // OR of every layer's attention error word (0 = clean)
};
struct StepCtl
{
    const uint32_t* in;               // StepIn
    u64* stepp;
    int *tok, *pos, *tgt;
    glue::SampleRow* srow;
    G6AttnStepCell* cells;            // [nl] attention step cells (graph form)
    const int* shared;                // [nl] 1 = the layer reuses another layer's index list
    int nl;
};
// grid 1, block 128
__global__ void __launch_bounds__(128) k_step_in(StepCtl c)
{
    __shared__ StepIn sh;
    const int t = threadIdx.x;
    if (t < (int)(sizeof(StepIn) / 4)) reinterpret_cast<uint32_t*>(&sh)[t] = __ldcv(c.in + t);
    __syncthreads();
    if (t == 0) *c.stepp = sh.step;
    if (t < MX) { c.tok[t] = sh.tok[t]; c.pos[t] = sh.pos[t]; c.srow[t] = sh.srow[t]; c.tgt[t] = sh.tgt[t]; }
    const int pages = (sh.context + 63) / 64;
    for (int l = t; l < c.nl; l += 128)
    {
        c.cells[l].context = sh.context;
#ifdef DEC_S2
        c.cells[l].page_count = sh.base_page + pages;   // the physical page bound of a run at base_page (table B + i)
#else
        c.cells[l].page_count = pages;
#endif
        c.cells[l].table_size = pages;
        c.cells[l].reuse_rows = c.shared[l] ? sh.M : 0;
    }
}
struct StepEnd
{
    const int* tok;
    const float *lp, *lpt;
    unsigned* const* errs;            // [nl] device pointers to the layers' attention error words
    int nl, M;
    StepOut* out;
};
// grid 1, block 128
__global__ void __launch_bounds__(128) k_step_out(StepEnd a)
{
    __shared__ unsigned e;
    const int t = threadIdx.x;
    if (t == 0) e = 0;
    __syncthreads();
    unsigned x = 0;
    for (int l = t; l < a.nl; l += 128) x |= *a.errs[l];
    if (x) atomicOr(&e, x);
    __syncthreads();
    if (t < a.M) { a.out->tok[t] = a.tok[t]; a.out->lp[t] = a.lp[t]; a.out->lpt[t] = a.lpt[t]; }
    if (t == 0) { a.out->attn_err = e; a.out->pad_ = 0; }
}

struct Layer
{
    int l = 0, index_src = -1;        // index_src: position in Model::layers of the full-index layer whose list is reused
    bool moe = false, indexer = false;
    bind::LayerTensors T;
    bind::Layer L[MX + 1];
    bool built[MX + 1] = {};
    G6AttnPack ap{};
    G6AttnBinding ab{};
    G6AttnReplay rp[NBK][MX + 1] = {};  // graph form: one replay entry per context bucket and row count
    bool rp_on[NBK][MX + 1] = {};
    const __nv_bfloat16 *attn_norm = nullptr, *mlp_norm = nullptr, *q_a_norm = nullptr;
    const float* router_bias = nullptr;
};

#if defined(G6_CP) && G6_CP
// J4 context-parallel decode (BUILD-CPE.md step 3). Set on, R, s3, colls BEFORE build (the driver sizes the CP fabric
// from colls); set vs (one kernel view per CP collective, fabc or the loopback) after build. A step runs the CP block
// when its context > R (plain form) / its bucket capacity > R (graph form; R is a bucket boundary, so the two agree).
struct CpState
{
    bool on = false;
    int R = 0, s3 = 1;                // s3: G6_CP_S3 (1 = head-group overlap order, 0 = S1 order; same arithmetic)
    g6::CpColls colls;                // numbering + capacities (colls.mmax >= any M of a CP step)
    std::vector<g6::VSlotDev> vs;     // [colls.ncoll()]
    unsigned* tickets = nullptr;      // [colls.ncoll()] publish tickets, device, zero
    uint64_t* cand = nullptr;         // [nr][MX][2048] candidate lists, own at [rank] (layout [s][M][2048] per step)
    float *qbuf = nullptr, *part_all = nullptr, *part_tmp = nullptr;   // [MX][HMAX][576], [nr][cp_blk(MX, HMAX)], [nr-1][same]
    // prefill staging (step 4, prefill.cuh cp_pf_*): two latent pools of one layer at capacity, the identity table, and
    // every layer's CP args while the prefill view is swapped in (pf_staged)
    char* pf_pool[2] = {};
    int* pf_ident = nullptr;
    std::vector<G6Attn> pf_saved;
    bool pf_staged = false;
    unsigned long long pf_g = 0;      // global pool-use counter: layer i of a sub-batch is use pf_g0 + i, pool (use % 2)
    unsigned long long pf_g0 = 0;     //   (not i % 2: 79 layers is odd); pf_g0 = pf_g at cp_pf_enter, pf_g += nl at exit
    void* pf_net = nullptr;           // g6m4: CpPfNet (prefill.cuh), the real-fabric gather; null = loopback (m4sim)
};
// binder rule (engine/attn/bind.cu): the layers that run their own indexer
inline bool cp_full_index(int l) { return l == 78 || l <= 2 || (l - 2) % 4 == 0; }
// floats of one partial block (c [M][H][512], m [M][H], l [M][H]), padded to 16 B as cp_lse (attn.cu) indexes it
inline size_t cp_blk(int M, int H) { return ((size_t)M * H * 514 + 3) & ~(size_t)3; }
// heads of rank d's group at model layer l (SPEC §9 / bind.cu: group (d - l) % NR, larger groups first)
inline int cp_heads(int d, int l) { return g6nr::width(g6nr::HEADS, (d - l % g6nr::NR + g6nr::NR) % g6nr::NR); }
#endif

#ifdef DEC_S2
struct S2Model;                       // s2.cuh: S2 batched decode state (sessions' rows in one step)
#endif
struct Model
{
    int rank = -1, nl = 0;
#ifdef DEC_S2
    S2Model* s2 = nullptr;            // s2_init; s2on: steps go through the S2 hooks (layer_for, attention, step in / out)
    bool s2on = false;
    int s2_pool_slots = 1;            // set before build: > 1 = the layers' own attention bindings hold the S2 pool
    int s2_run_pages = 0;             // build: the MOST pages one session's run can hold (the capacity argument; each view's table)
    int s2_pool_pages = 0;            // set before build: pages of the pool (0 = s2_pool_slots x s2_run_pages); >= s2_run_pages
    int s2_pf_base = 0;               // S2 below R: first page of the run being prefilled (CP staging index offset, prefill.cuh)
    int s2_below = 0;                 // set before s2_init: S2 below R (CP builds): every session's run <= s2_below rows on the
                                      // replicated (non-CP) path; the pool is the layer's own binding as built (s2.cuh)
#endif
    bool graph = false;               // attention through the replay entry (set before prepare)
    bool wide = false;                // set before build: tr3.fc1 K3 records into the wide order, in place (bind::widen_fc1)
    // graph form: captured attention capacities, ascending, the last = the binding's capacity. G6_DEC_BK=B (read at
    // build, default 2048): B, 2B, 4B, ... below the capacity, then the capacity; 0 = one bucket (the capacity). A step
    // runs the graph of the smallest bucket >= its context (bucket_of); cur = the bucket a capture is recording
    int nbk = 1, bk[NBK] = {}, cur = 0;
    const StepIn* in = nullptr;       // set by the driver: pinned host memory or device memory
    StepOut* out = nullptr;           //   same
    G6AttnStepCell* cells = nullptr;  // [nl] device
    int* shared = nullptr;            // [nl] device
    char* dscratch = nullptr;         // G6_DEC_SHARED_SCRATCH=1: the one decode attention scratch of all layers (build)
    size_t dscratch_bytes = 0;
    unsigned** errs = nullptr;        // [nl] device
    g6::FabricDev f{};
    u64* stepp = nullptr;             // device step counter
    unsigned* tickets = nullptr;      // [2*nl+1] publish tickets, device, zero
    float* resid = nullptr;           // [MX][6144]
    __half *x = nullptr, *x2 = nullptr, *q_a = nullptr, *q_a_n = nullptr, *kv_a = nullptr, *q_b = nullptr;
    __half *idx_q = nullptr, *idx_k = nullptr, *idx_w = nullptr;
    float *logits = nullptr, *hlog = nullptr, *lp = nullptr, *lpt = nullptr;
    glue::RoutePlan* plan = nullptr;
    int* rlog = nullptr;              // G6_ROUTE_LOG (measurement): pinned [nl][RLOG] copies of plan n_slots / pad / slot_expert
    glue::SampleRow* srow = nullptr;
    glue::TkRes* tk = nullptr;        // top lists of the step's rows with SampleRow::topk > 0 (main model only; null = none)
    int *tok = nullptr, *pos = nullptr, *tok_out = nullptr, *tgt = nullptr;
    void* scratch = nullptr;
    bind::Scratch sc;
    std::vector<Layer> layers;
    const __nv_bfloat16 *emb = nullptr, *final_norm = nullptr;
    bind::HeadTensors head_t;
    bind::Head head[MX + 1];
    __half *eh_in = nullptr, *eh_out = nullptr;   // MTP draft model only: eh_proj input [MX][12288] and output [MX][6144]
    int vlo = 0, vn = 0;
    int nc = 0;                       // top_p candidates per rank per row (0 = top_p off); set before the first step
    unsigned* topp_miss = nullptr;    // device counter: top_p rows whose nucleus went past the candidates
    void (*dfl_tap)(Model&, int, int, cudaStream_t) = nullptr;   // --dflash: after layer i, copy the residual rows the drafter reads (dflash.cuh)
    bool rs = false;                  // --spec-accept rejection (set before capture): the head accepts each row's draft by
                                      // rejection sampling (the glue.cuh _rs head kernels)
    int ncoll() const { return 2 * nl + 1; }
#if defined(G6_CP) && G6_CP
    CpState cp;
#endif
};

// optional decode stage profile (G6_DEC_PROF=1 in g6m4, plain launches only): synchronizes the stream at every mark, so
// it is a breakdown of where a step's time goes, never a speed number. Totals per row count M (set by the driver).
struct DecProf
{
    bool on = false;
    int M = 1;
    double t[MX + 1][8] = {}, last = 0;
    long steps[MX + 1] = {};
    static double now() { timespec s; clock_gettime(CLOCK_MONOTONIC, &s); return s.tv_sec * 1e3 + s.tv_nsec * 1e-6; }
    void print(const char* tag, int nl) const
    {
        static const char* n[8] = {"projections", "attention", "o_proj+put", "wait+reduce+norm", "router+experts+put", "head", "step in/out+embed", "other"};
        for (int M = 1; M <= MX; ++M)
        {
            if (!steps[M]) continue;
            double tot = 0;
            for (double v : t[M]) tot += v;
            fprintf(stderr, "# %s decode stage profile, M=%d, %ld steps (ms per step, synchronized):", tag, M, steps[M]);
            for (int i = 0; i < 8; ++i) fprintf(stderr, " %s %.3f (%.0f%%),", n[i], t[M][i] / steps[M], 100 * t[M][i] / (tot > 0 ? tot : 1));
            fprintf(stderr, " total %.3f; per layer us: attention %.1f, wait+reduce+norm %.1f (two collectives)\n", tot / steps[M],
                    t[M][1] * 1000 / steps[M] / nl, t[M][3] * 1000 / steps[M] / nl);
        }
    }
};
inline DecProf& dec_prof() { static DecProf p; return p; }
inline void dmark(int k, cudaStream_t st)
{
    DecProf& p = dec_prof();
    if (!p.on) return;
    G6RT_CK(cudaStreamSynchronize(st));
    const double n = DecProf::now();
    if (p.last > 0) p.t[p.M][k] += n - p.last;
    p.last = n;
}

inline const void* need(const Find& find, const std::string& name)
{
    const TensorRef t = find(name);
    if (!t.p) die("pack has no tensor " + name);
    return t.p;
}
inline bind::Tensor as_bind(const TensorRef& r)
{
    bind::Tensor b;
    b.p = r.p; b.nd = r.nd; b.fmt = r.fmt;
    for (int i = 0; i < 4; ++i) b.shape[i] = r.shape[i];
    return b;
}

// capacity = longest sequence (tokens) the attention caches of this model hold; max_rows = most rows one attention
// call may carry (MX for decode only, the prefill sub-batch size with prefill.cuh)
inline void build(Model& m, int rank, const std::vector<int>& ids, const Find& find, int capacity, const g6::FabricDev& f,
                  u64* stepp, cudaStream_t st, int max_rows = MX)
{
    m.rank = rank; m.nl = (int)ids.size(); m.f = f; m.stepp = stepp;
    {
        const char* e = getenv("G6_DEC_BK");
        int b = e ? atoi(e) : 2048;
        m.nbk = 0;
        if (b > 0) for (; b < capacity && m.nbk < NBK - 1; b *= 2) m.bk[m.nbk++] = b;
        m.bk[m.nbk++] = capacity;
#if defined(G6_CP) && G6_CP
        if (m.cp.on && m.cp.R > 0 && m.cp.R < capacity && std::find(m.bk, m.bk + m.nbk, m.cp.R) == m.bk + m.nbk)
        {   // R is a bucket boundary: buckets <= R run the replicated graphs, buckets > R the CP graphs
            if (m.nbk == NBK) die("G6_CP: no room for the R bucket (G6_DEC_BK)");
            m.bk[m.nbk++] = m.cp.R;
            std::sort(m.bk, m.bk + m.nbk);
        }
#endif
    }
    m.cells = dmalloc<G6AttnStepCell>(m.nl);
    m.shared = dmalloc<int>(m.nl);
    m.errs = dmalloc<unsigned*>(m.nl);
    if (f.ncoll != m.ncoll()) die("fabric ncoll must be 2 * layers + 1");
    m.tickets = dmalloc<unsigned>(m.ncoll());
    m.resid = dmalloc<float>((size_t)MX * HID);
    m.x = dmalloc<__half>((size_t)MX * HID);
    m.x2 = dmalloc<__half>((size_t)MX * HID);
    m.q_a = dmalloc<__half>((size_t)MX * QA);
    m.q_a_n = dmalloc<__half>((size_t)MX * QA);
    m.kv_a = dmalloc<__half>((size_t)MX * KVA);
    m.q_b = dmalloc<__half>((size_t)MX * g6nr::QBW);
    m.idx_q = dmalloc<__half>((size_t)MX * IDXQ);
    m.idx_k = dmalloc<__half>((size_t)MX * IDXK);
    m.idx_w = dmalloc<__half>((size_t)MX * IDXW);
    m.logits = dmalloc<float>((size_t)MX * glue::NEXP);
    m.plan = dmalloc<glue::RoutePlan>(1);
    m.srow = dmalloc<glue::SampleRow>(MX);
    m.tk = dmalloc<glue::TkRes>(1);
    m.topp_miss = dmalloc<unsigned>(1);
    m.tok = dmalloc<int>(MX);
    m.pos = dmalloc<int>(MX);
    m.tok_out = dmalloc<int>(MX);
    m.lp = dmalloc<float>(MX);
    m.lpt = dmalloc<float>(MX);
    m.tgt = dmalloc<int>(MX);
    m.scratch = dmalloc<char>(bind::scratch_bytes());
    m.sc = bind::scratch_at(m.scratch);
    m.layers = std::vector<Layer>(m.nl);      // sized once: the attention bindings must not move
    // G6_DEC_SHARED_SCRATCH=1 (PLAN-600K L1): ONE decode attention scratch for every layer of a main model (decode runs the
    // layers one after another on one stream; each run re-initializes the scratch). The draft (ids {78}) keeps its own.
    // The drivers die without --pfv2 (the v1 prefill lanes would run two layers' attention at once).
    const int attn_rows = max_rows < MX ? MX : max_rows;
#ifdef DEC_S2
    // S2 pool (decision on the S=1 transition contract): the binding holds s2_pool_slots runs of the reservation;
    // the context buckets stay those of one run, so the S=1 graphs keep today's buckets. Under CP the pool is the CP
    // binding itself (S2 below R, s2.cuh), so no slot multiple there.
    m.s2_run_pages = (capacity + 63) / 64;
#if defined(G6_CP) && G6_CP
    if (m.cp.on && (m.s2_pool_slots > 1 || m.s2_pool_pages)) die("DEC_S2 + G6_CP: s2_pool_slots > 1 is not the CP pool (S2 below R uses the CP binding)");
#endif
    if (m.s2_pool_pages && m.s2_pool_pages < m.s2_run_pages) die("s2: s2_pool_pages < one session's run");
    const int cap_ab = m.s2_pool_pages ? m.s2_pool_pages * 64 : m.s2_pool_slots > 1 ? m.s2_pool_slots * m.s2_run_pages * 64 : capacity;
#else
    const int cap_ab = capacity;
#endif
    {
        const char* e = getenv("G6_DEC_SHARED_SCRATCH");
        if (e && atoi(e) == 1 && !(m.nl == 1 && ids[0] == 78))
        {
            m.dscratch_bytes = std::max(g6_attn_bind_scratch_need(g6nr::HMIN, cap_ab, attn_rows, 16),
                                        g6_attn_bind_scratch_need(g6nr::HMAX, cap_ab, attn_rows, 16));
            if (!m.dscratch_bytes) die("G6_DEC_SHARED_SCRATCH: bad attention shape");
            m.dscratch = dmalloc<char>(m.dscratch_bytes);     // zeroed
        }
    }
    int last_index = -1;
    for (int i = 0; i < m.nl; ++i)
    {
        Layer& y = m.layers[i];
        y.l = ids[i];
        y.T = bind::layer_tensors(y.l, [&](const char* n) { return as_bind(find(n)); });
        if (m.wide)           // G6_NOWIDE2=1 keeps fc2 in the old order (A/B; outputs are bit-identical either way)
        {
            bind::widen_fc1(y.T, st);
            if (!getenv("G6_NOWIDE2")) bind::widen_fc2(y.T, st);
            G6RT_CK(cudaStreamSynchronize(st));
        }
#if DEC_I4PAIR
        bind::i4pair(y.T, st);
        G6RT_CK(cudaStreamSynchronize(st));
#endif
        if (const char* e = bind::check_shared_signs(y.T, st)) die("layer " + std::to_string(y.l) + ": " + e);
        y.moe = y.T.t[bind::T_TR3_TAB].p != nullptr;
        y.indexer = y.T.t[bind::T_IDX_WQ_B].p != nullptr;
        if (y.indexer) last_index = i;
        else if (last_index < 0) die("layer " + std::to_string(y.l) + " shares an index list but no full-index layer precedes it");
        else y.index_src = last_index;
        auto at = [&](const char* suffix) {
            G6AttnTensor t{};
            const std::string name = lname(y.l, suffix);
            const TensorRef r = find(name);
            if (!r.p) return t;               // only the index norms may be absent (shared-index layers)
            if (!r.bf16 || r.nd > 3) die("bad attention weight " + name);
            t.data = r.p; t.ndim = r.nd;
            for (int d = 0; d < r.nd; ++d) t.shape[d] = r.shape[d];
            return t;
        };
        y.ap = G6AttnPack{at("kv_b.uk"), at("kv_b.uv"), at("kv_a_norm"), at("idx.k_norm.w"), at("idx.k_norm.b")};
#if defined(G6_CP) && G6_CP
        if (m.cp.on)
        {
            if (y.indexer != cp_full_index(y.l)) die("G6_CP: layer " + std::to_string(y.l) + " indexer flag disagrees with the binder rule");
            if (m.cp.colls.q.size() != (size_t)m.nl || (m.cp.colls.cand[i] >= 0) != y.indexer) die("G6_CP: colls do not match the layer list");
            G6RT_CK(g6_attn_bind_create_cp(&y.ab, &y.ap, rank, y.l, capacity, attn_rows, 16, m.cp.R, g6nr::NR, st, m.dscratch, m.dscratch_bytes));
        }
        else
#endif
        G6RT_CK(g6_attn_bind_create_ext(&y.ab, &y.ap, rank, y.l, cap_ab, attn_rows, 16, st, m.dscratch, m.dscratch_bytes));
        y.attn_norm = (const __nv_bfloat16*)need(find, lname(y.l, "attn_norm"));
        y.mlp_norm = (const __nv_bfloat16*)need(find, lname(y.l, "mlp_norm"));
        y.q_a_norm = (const __nv_bfloat16*)need(find, lname(y.l, "q_a_norm"));
        if (y.moe) y.router_bias = (const float*)need(find, lname(y.l, "router.bias"));
    }
    {
        std::vector<int> sh(m.nl);
        std::vector<unsigned*> er(m.nl);
        for (int i = 0; i < m.nl; ++i) { sh[i] = !m.layers[i].indexer; er[i] = m.layers[i].ab.args.error; }
        G6RT_CK(g6h2d::copy(m.shared, sh.data(), m.nl * sizeof(int)));
        G6RT_CK(g6h2d::copy(m.errs, er.data(), m.nl * sizeof(unsigned*)));
    }
#if defined(G6_CP) && G6_CP
    if (m.cp.on)
    {
        // colls.mmax may exceed MX: it sizes the CP fabric only (A3: M up to 24 for S2); these buffers hold <= MX rows
        // NR=6: head groups of 10 / 11 heads rotate by layer (SPEC §9) -> size for the widest (HMAX); partial blocks are
        // padded to 16 B (cp_blk)
        const int H = g6nr::HMAX, NR = g6nr::NR;
        m.cp.tickets = dmalloc<unsigned>(m.cp.colls.ncoll());
        m.cp.cand = dmalloc<uint64_t>((size_t)NR * MX * 2048);
        m.cp.qbuf = dmalloc<float>((size_t)MX * H * 576);
        m.cp.part_all = dmalloc<float>((size_t)NR * cp_blk(MX, H));
        m.cp.part_tmp = dmalloc<float>((size_t)(NR - 1) * cp_blk(MX, H));
        fprintf(stderr, "# rank %d: G6_CP on, R %d, %s order, %d CP collectives (mmax %d)\n", rank, m.cp.R, m.cp.s3 ? "S3" : "S1",
                m.cp.colls.ncoll(), m.cp.colls.mmax);
    }
#endif
    size_t ab = 0;
    for (const Layer& y : m.layers) ab += y.ab.storage_bytes + y.ab.scratch_bytes;
    fprintf(stderr, "# rank %d: attention caches + scratch for %d layers at capacity %d: %.3f GB\n", rank, m.nl, capacity, ab / 1e9);
    if (m.dscratch)
    {
        size_t real = m.dscratch_bytes;
        for (const Layer& y : m.layers) real += y.ab.storage_bytes;
        fprintf(stderr, "# rank %d: shared decode scratch %.3f GB; attention allocated (caches + one scratch) %.3f GB\n", rank,
                m.dscratch_bytes / 1e9, real / 1e9);
    }
    {
        // per-layer expert format: a count on every rank, the layer list on rank 0. Ranks do not compare formats, so a
        // mixed set of packs shows only as different counts in the rank logs.
        int ne4 = 0, nmoe = 0;
        std::string e4, tr3;
        for (const Layer& y : m.layers)
        {
            if (!y.moe) continue;
            ++nmoe;
            ne4 += y.T.e4m3;
            (y.T.e4m3 ? e4 : tr3) += " " + std::to_string(y.l);
        }
        if (ne4) fprintf(stderr, "# rank %d: routed experts exlr8-e4m3 in %d of %d MoE layers\n", rank, ne4, nmoe);
        if (ne4 && rank == 0) fprintf(stderr, "# rank 0: exlr8-e4m3 layers:%s; TR3 layers:%s\n", e4.c_str(), tr3.empty() ? " none" : tr3.c_str());
    }
    m.emb = (const __nv_bfloat16*)need(find, "G.embed");
    m.final_norm = (const __nv_bfloat16*)need(find, "G.final_norm");
    m.head_t = bind::head_tensors([&](const char* n) { return as_bind(find(n)); });
    if (!m.head_t.lm_head.p) die("pack has no G.lm_head.mx.gv (run tools/convert/addgv.py)");
    if (!bind::vocab_slice(rank, m.vlo, m.vn)) die("rank must be 0.." + std::to_string(g6nr::NR - 1));
    m.hlog = dmalloc<float>((size_t)MX * m.vn);
}

#ifdef DEC_S2
inline const bind::Layer& s2_layer(Model& m, int i, int M);
inline const bind::Layer& layer_base(Model& m, int i, int M);
inline const bind::Layer& layer_for(Model& m, int i, int M) { return m.s2on ? s2_layer(m, i, M) : layer_base(m, i, M); }
#define G6RT_LAYER_BASE layer_base
#else
#define G6RT_LAYER_BASE layer_for
#endif
inline const bind::Layer& G6RT_LAYER_BASE(Model& m, int i, int M)
{
    Layer& y = m.layers[i];
    if (!y.built[M])
    {
        bind::Io io;
        io.x = m.x; io.q_a = m.q_a; io.q_a_n = m.q_a_n; io.kv_a = m.kv_a; io.q_b = m.q_b;
        io.idx_q = m.idx_q; io.idx_k = m.idx_k; io.idx_w = m.idx_w;
        io.attn = (const __half*)y.ab.args.output;
        io.attn_part = m.sc.part[0];
        io.x2 = m.x2; io.logits = m.logits; io.plan = bind::plan_ptrs(m.plan);
        io.mlp_part = m.sc.part[1];
        // G6_FUSE=1 (default): fused K1 launch (bytes equal); 0: separate launches; 2: fused + KS1=2 WG=2 split at M=2,3 (different rounding, not bit-equal)
        static const int fuse = getenv("G6_FUSE") ? atoi(getenv("G6_FUSE")) : 1;
        if (const char* e = bind::make_layer(y.T, M, m.sc, io, y.L[M], fuse))
            die("rank " + std::to_string(m.rank) + " layer " + std::to_string(y.l) + ": " + e);
        if (y.L[M].heads != y.ab.args.heads) die("kernel binder and attention binder disagree on the head count");
#ifdef DEC_SLOT
        // DEC_SLOT: o_proj and the K1 finalize / dense fc2 write the partial into the send slot (decslot.cuh)
        auto slot = [&](int c) {
            dslot::Slot q;
            q.send = m.f.send; q.bell = m.f.doorbell; q.stepp = m.stepp; q.stride = m.f.stride;
            q.ncoll = m.f.ncoll; q.c = c; q.ticket = m.tickets + c;
            return q;
        };
        y.L[M].o_proj.slot = slot(2 * i);
        y.L[M].k1.a.slot = slot(2 * i + 1);
#endif
        y.built[M] = true;
    }
    return y.L[M];
}

inline void launch_embed(Model& m, int M, cudaStream_t st)
{
    glue::EmbedArgs a{m.tok, m.emb, m.layers[0].attn_norm, m.resid, m.x, glue::RMS_EPS};
    glue::k_embed_norm<<<dim3(glue::RN_CL, M), glue::RN_T, 0, st>>>(a);
}

#ifdef DEC_SLOT
// DEC_SLOT < 2: the producer already wrote the send slot; one CTA rings the doorbell. DEC_SLOT >= 2: nothing.
inline void slot_publish(Model& m, int c, cudaStream_t st)
{
#if DEC_SLOT < 2
    k_put_publish<<<1, 128, 0, st>>>(nullptr, m.f, m.stepp, c, 0, m.tickets + c);
#endif
}
#endif
inline void put_publish(Model& m, const uint16_t* staging, int c, int M, cudaStream_t st)
{
    k_put_publish<<<1, 128, 0, st>>>(reinterpret_cast<const uint4*>(staging), m.f, m.stepp, c, M * (HID * 2 / 16), m.tickets + c);
}

#ifdef DEC_S2
inline void s2_attention(Model& m, int i, int M, cudaStream_t st);
inline void s2_launch_cells(Model& m, cudaStream_t st);
inline StepEnd s2_step_end(Model& m, StepEnd e);
#endif
inline G6AttnStep attn_step(Model& m, int i, int M, int context)
{
    Layer& y = m.layers[i];
    G6AttnStep s{};
    s.rows = M; s.context = context;
    s.q_b = m.q_b; s.kv_a = m.kv_a; s.positions = m.pos;
    if (y.indexer) { s.index_q = m.idx_q; s.index_k = m.idx_k; s.index_weight = m.idx_w; }
    else { s.reuse_indices = m.layers[y.index_src].ab.args.indices; s.reuse_rows = M; }
    return s;
}

#if defined(G6_CP) && G6_CP
inline bool cp_active(const Model& m, int context) { return m.cp.on && (m.graph ? m.bk[m.cur] : context) > m.cp.R; }
// The CP attention block (PLAN-S3 §1), in three phases so the loopback driver can deliver between them (phase -1 = all,
// in order, as the real fabric runs it):
//   0: projections, cp_front (append, absorb) -> q(i) put; cp_index -> cand(i) put (full layers); S3 + shared layer: my
//      own head group now (its q is local, the q all-gather is on the wire)
//   1: full layers: wait cand(i), merge; own group (if not done); wait q(i); remote groups d = (rank + k) % nr, k = 1..:
//      S3 publishes each partial as soon as it is computed, S1 computes all three, then publishes them
//   2: wait the nr-1 partials of my heads, LSE merge in shard order, value_project, then o_proj and publish(2i) as today
// S3 and S1 run the same kernels on the same inputs (only the order differs), so they are byte-identical (gate A).
inline void launch_attn_cp(Model& m, int i, int M, int context, int phase, cudaStream_t st)
{
    Layer& y = m.layers[i];
    CpState& C = m.cp;
    const bind::Layer& b = layer_for(m, i, M);
    const int r = m.rank, nr = g6nr::NR, H = y.ab.args.heads, G = 16;
    if (H != cp_heads(r, y.l)) die("G6_CP: layer " + std::to_string(y.l) + " binding heads disagree with the group rule");
    G6Attn* a = m.graph ? &y.rp[m.cur][M].fixed : &y.ab.args;
    const G6AttnStepCell* cell = m.graph ? y.rp[m.cur][M].cell : nullptr;
    void* scr = y.ab.scratch;
    const size_t sb = y.ab.scratch_bytes;
    const int cq = C.colls.q[i], cc = C.colls.cand[i];
    // NR=6: q payloads and partials are sized by the group they belong to (10 or 11 heads); the wire size (tail offset) of
    // every collective is M x its row bytes (CpColls, sized for the widest group)
    const size_t qn = (size_t)M * H * 512 * 4, rn = (size_t)M * H * 64 * 4, cb = (size_t)M * C.colls.cand_row;
    const size_t qw = (size_t)M * C.colls.q_row, pw = (size_t)M * C.colls.part_row;
    const size_t blk = cp_blk(M, H), tblk = cp_blk(M, g6nr::HMAX);
    auto part_of = [&](int s) { return C.part_all + s * blk; };
    auto cpart = [&](int k) { return C.colls.part[(size_t)i * (nr - 1) + k - 1]; };
    const float *q = nullptr, *rope = nullptr;
    if (phase <= 0)
    {
        if (m.graph && !y.rp_on[m.cur][M]) die("prepare(m, M) was not called before a graph step");
        dmark(7, st);
#ifdef DEC_SLOT
        if (i == 0) bind::launch(b.xf_x, st);
#else
        bind::launch(b.xf_x, st);
#endif
        bind::launch(b.q_a, st);
        glue::k_rmsnorm_rows<__half><<<M, 256, 0, st>>>(m.q_a, QA, QA, y.q_a_norm, glue::RMS_EPS, m.q_a_n, QA);
        bind::launch(b.kv_a, st);
        bind::launch(b.idx_k, st);
        bind::launch(b.idx_w, st);
        bind::launch(b.xf_qa, st);
        bind::launch(b.q_b, st);
        bind::launch(b.idx_q, st);
        dmark(0, st);
        if (!m.graph)
        {
            const G6AttnStep s = attn_step(m, i, M, context);
            G6RT_CK(g6_attn_bind_prepare(&y.ab, &s, st));
        }
        G6RT_CK(g6_attn_cp_front(a, cell, scr, sb, st));
        G6RT_CK(g6_attn_cp_qabs(a, scr, &q, &rope));
        cpglue::k_vput<<<G, 256, 0, st>>>(q, qn, rope, rn, C.vs[cq], m.stepp, C.tickets + cq);
        G6RT_CK(g6_attn_cp_index(a, cell, scr, sb, C.cand, st));
        if (cc >= 0) cpglue::k_vput<<<G, 256, 0, st>>>(C.cand + (size_t)r * M * 2048, cb, nullptr, 0, C.vs[cc], m.stepp, C.tickets + cc);
        if (C.s3 && cc < 0) G6RT_CK(g6_attn_cp_group(a, cell, scr, sb, q, rope, part_of(r), st));
    }
    if (phase == 0) return;
    if (phase <= 1)
    {
        if (!q) G6RT_CK(g6_attn_cp_qabs(a, scr, &q, &rope));
        if (cc >= 0)
        {
            const g6::VSlotDev& v = C.vs[cc];
            cpglue::k_vwait<<<1, 32, 0, st>>>(v, m.stepp, cb, -1, cpglue::ERR_CP);
            for (int j = 0; j < v.nsrc; ++j)
                cpglue::k_vget<<<G, 256, 0, st>>>(v, j, m.stepp, C.cand + (size_t)v.src[j] * M * 2048, cb);
            G6RT_CK(g6_attn_cp_merge_index(a, cell, C.cand, st));
        }
        if (!C.s3 || cc >= 0) G6RT_CK(g6_attn_cp_group(a, cell, scr, sb, q, rope, part_of(r), st));
        const g6::VSlotDev& vq = C.vs[cq];
        cpglue::k_vwait<<<1, 32, 0, st>>>(vq, m.stepp, qw, -1, cpglue::ERR_CP);
        for (int k = 1; k < nr; ++k)
        {
            const int d = (r + k) % nr, j = cpglue::vsrc_index(vq, d), hd = cp_heads(d, y.l);
            if (j < 0) die("G6_CP: q all-gather has no slot for rank " + std::to_string(d));
            const size_t qd = (size_t)M * hd * 512 * 4, rd = (size_t)M * hd * 64 * 4;
            G6Attn ad = *a;   // d's head group: hd heads of absorbed q over MY keys (no per-head weights in cp_group)
            ad.heads = hd;
            float* out = C.s3 ? C.part_tmp : C.part_tmp + (size_t)(k - 1) * tblk;
            cpglue::k_vget<<<G, 256, 0, st>>>(vq, j, m.stepp, C.qbuf, qd + rd);
            G6RT_CK(g6_attn_cp_group(&ad, cell, scr, sb, C.qbuf, C.qbuf + qd / 4, out, st));
            if (C.s3) cpglue::k_vput<<<G, 256, 0, st>>>(out, (size_t)M * hd * 514 * 4, nullptr, 0, C.vs[cpart(k)], m.stepp, C.tickets + cpart(k));
        }
        if (!C.s3)
            for (int k = 1; k < nr; ++k)
                cpglue::k_vput<<<G, 256, 0, st>>>(C.part_tmp + (size_t)(k - 1) * tblk, (size_t)M * cp_heads((r + k) % nr, y.l) * 514 * 4,
                                                   nullptr, 0, C.vs[cpart(k)], m.stepp, C.tickets + cpart(k));
    }
    if (phase == 1) return;
#ifdef CP_F1A
    {
        static_assert(g6nr::NR - 1 <= 7, "VParts holds 7 partials");
        cpglue::VParts p{};   // F1a: the nr-1 waits + gets below in one launch (same slots, same bytes, same destinations)
        for (int k = 1; k < nr; ++k)
        {
            p.v[k - 1] = C.vs[cpart(k)];
            if (p.v[k - 1].nsrc != 1) die("G6_CP F1a: partial collective with " + std::to_string(p.v[k - 1].nsrc) + " sources");
            p.dst[k - 1] = part_of(p.v[k - 1].src[0]);
        }
        cpglue::k_vwaitget<<<dim3(G, nr - 1), 256, 0, st>>>(p, m.stepp, pw, (size_t)M * H * 514 * 4, cpglue::ERR_CP);
    }
#else
    for (int k = 1; k < nr; ++k)
    {
        const g6::VSlotDev& v = C.vs[cpart(k)];
        cpglue::k_vwait<<<1, 32, 0, st>>>(v, m.stepp, pw, -1, cpglue::ERR_CP);
        cpglue::k_vget<<<G, 256, 0, st>>>(v, 0, m.stepp, part_of(v.src[0]), (size_t)M * H * 514 * 4);   // partials of MY heads
    }
#endif
    G6RT_CK(g6_attn_cp_finish(a, cell, scr, sb, C.part_all, st));
    dmark(1, st);
    bind::launch(b.xf_attn, st);
    bind::launch(b.o_proj, st);
#ifdef DEC_SLOT
    slot_publish(m, 2 * i, st);
#else
    put_publish(m, m.sc.part[0], 2 * i, M, st);
#endif
    dmark(2, st);
}
#endif

// rows at positions m.pos[0..M-1] (device), context = last position + 1 (plain form; the graph form reads the cell)
inline void launch_attn_block(Model& m, int i, int M, int context, cudaStream_t st)
{
#if defined(G6_CP) && G6_CP
#ifdef DEC_S2
    // S2 below R: every session's run is <= G6_CP_R rows (s2_init run_pages), so an S2 step never takes the CP block;
    // m.cur is not the S2 step's bucket (each slot has its own, S2Model::bk)
    if (!m.s2on)
#endif
    if (cp_active(m, context)) { launch_attn_cp(m, i, M, context, -1, st); return; }
#endif
    Layer& y = m.layers[i];
    const bind::Layer& b = layer_for(m, i, M);
    dmark(7, st);
#ifdef DEC_SLOT
    if (i == 0) bind::launch(b.xf_x, st);      // later layers: the fragment tables come from launch_consume(2i-1)
#else
    bind::launch(b.xf_x, st);
#endif
    bind::launch(b.q_a, st);
    glue::k_rmsnorm_rows<__half><<<M, 256, 0, st>>>(m.q_a, QA, QA, y.q_a_norm, glue::RMS_EPS, m.q_a_n, QA);
    bind::launch(b.kv_a, st);
    bind::launch(b.idx_k, st);
    bind::launch(b.idx_w, st);
    bind::launch(b.xf_qa, st);
    bind::launch(b.q_b, st);
    bind::launch(b.idx_q, st);
    dmark(0, st);
#ifdef DEC_S2
    if (m.s2on) s2_attention(m, i, M, st);
    else
#endif
    if (m.graph)
    {
        if (!y.rp_on[m.cur][M]) die("prepare(m, M) was not called before a graph step");
        G6RT_CK(g6_attn_replay_run(&y.rp[m.cur][M], st));
    }
    else
    {
        const G6AttnStep s = attn_step(m, i, M, context);
        G6RT_CK(g6_attn_bind_run(&y.ab, &s, st));
    }
    dmark(1, st);
    bind::launch(b.xf_attn, st);
    bind::launch(b.o_proj, st);
#ifdef DEC_SLOT
    slot_publish(m, 2 * i, st);
#else
    put_publish(m, m.sc.part[0], 2 * i, M, st);
#endif
    dmark(2, st);
}

inline void launch_consume(Model& m, int c, int M, const __nv_bfloat16* w, __half* out, cudaStream_t st)
{
    glue::ArNormArgs a{m.f, m.stepp, c, M, m.resid, w, out, glue::RMS_EPS};
#ifdef DEC_SLOT
    // fold: the normed rows also go into the fragment tables the next kernel reads (xf_x2 of this layer, xf_x of the
    // next; after the last layer the head's xf_h still runs)
    const bind::Xfrag* xf = c % 2 == 0 ? &layer_for(m, c / 2, M).xf_x2 : c / 2 + 1 < m.nl ? &layer_for(m, c / 2 + 1, M).xf_x : nullptr;
    a.xf16 = xf ? xf->f16 : nullptr;
    a.xbf16 = xf ? xf->bf16 : nullptr;
#endif
    dmark(7, st);
    // G6_WARM=1|2 (read at launch / graph capture): L2 warm-up of the next weights while the wait runs (bind.cuh warm_*,
    // loads only, no output change). 2i: this layer's router + shared fc1; 2i+1: the next layer's q_a (2: + kv_a, q_b)
    static const int warm = getenv("G6_WARM") ? atoi(getenv("G6_WARM")) : 0;
    if (warm && c < 2 * m.nl)
    {
        if (c % 2 == 0) bind::warm_mlp(layer_for(m, c / 2, M), st);
        else if (c / 2 + 1 < m.nl) bind::warm_proj(layer_for(m, c / 2 + 1, M), st, warm >= 2);
    }
    glue::k_wait_peers<<<1, 32, 0, st>>>(m.f, m.stepp, c, (size_t)M * HID * 2, glue::ERR_AR);
    glue::k_ar_add_norm<<<dim3(glue::RN_CL, M), glue::RN_T, 0, st>>>(a);
    dmark(3, st);
}

constexpr int RLOG = 4 + SMAX;           // RoutePlan head: n_slots, pad_[3], slot_expert[SMAX]
inline void launch_mlp_block(Model& m, int i, int M, cudaStream_t st)
{
    Layer& y = m.layers[i];
    const bind::Layer& b = layer_for(m, i, M);
#ifndef DEC_SLOT
    bind::launch(b.xf_x2, st);                 // DEC_SLOT: launch_consume(2i) wrote the fragment tables
#endif
    if (y.moe)
    {
        bind::launch(b.router, st);
        glue::k_router_plan<<<1, 256, 0, st>>>(m.logits, y.router_bias, M, m.plan);
        if (m.rlog) G6RT_CK(cudaMemcpyAsync(m.rlog + (size_t)i * RLOG, m.plan, RLOG * sizeof(int), cudaMemcpyDeviceToHost, st));
    }
    bind::launch_k1(b.k1, st);
#ifdef DEC_SLOT
    slot_publish(m, 2 * i + 1, st);
#else
    put_publish(m, m.sc.part[1], 2 * i + 1, M, st);
#endif
    dmark(4, st);
}

// G6_ROUTE_LOG=<file> (measurement only; set before prepare/capture so the graphs carry the copies): one JSON line per
// MTP main step, {"q","s","M","pos0","n","L":{"<layer>":[distinct routed expert ids ascending]}} over the MoE layers
inline FILE* route_log_open(Model& m)
{
    const char* e = getenv("G6_ROUTE_LOG");
    if (!e) return nullptr;
    FILE* f = fopen(e, "w");
    if (!f) die(std::string("G6_ROUTE_LOG: cannot open ") + e);
    G6RT_CK(cudaHostAlloc((void**)&m.rlog, (size_t)m.nl * RLOG * sizeof(int), cudaHostAllocDefault));
    return f;
}
inline void route_log_line(FILE* f, const Model& m, const std::vector<int>& ids, size_t q, int s, int M, int pos0, size_t n)
{
    fprintf(f, "{\"q\":%zu,\"s\":%d,\"M\":%d,\"pos0\":%d,\"n\":%zu,\"L\":{", q, s, M, pos0, n);
    bool first = true;
    for (int i = 0; i < m.nl; ++i)
    {
        if (!m.layers[i].moe) continue;
        const int* r = m.rlog + (size_t)i * RLOG;
        fprintf(f, "%s\"%d\":[", first ? "" : ",", ids[i]);
        for (int k = 0; k < r[0] && k < SMAX; ++k) fprintf(f, "%s%d", k ? "," : "", r[4 + k]);
        fputc(']', f);
        first = false;
    }
    fputs("}}\n", f);
}

inline const __nv_bfloat16* norm_after(const Model& m, int i) { return i + 1 < m.nl ? m.layers[i + 1].attn_norm : m.final_norm; }

inline glue::HeadArgs head_args(Model& m, int M)
{
    return glue::HeadArgs{m.f, m.stepp, 2 * m.nl, M, m.hlog, m.vn, m.vlo, m.vn, m.srow, m.tickets + 2 * m.nl, m.tok_out, m.lp,
                          m.tgt, m.lpt, m.nc, m.topp_miss, m.tk};
}
inline const bind::Head& head_for(Model& m, int M)
{
    bind::Head& H = m.head[M];
    if (!H.M)
    {
        bind::HeadIo io;
        io.h = m.x; io.logits = m.hlog;
        io.eh_in = m.eh_in; io.eh = m.eh_out;
        if (const char* e = bind::make_head(m.head_t, m.rank, M, m.sc, io, H)) die("rank " + std::to_string(m.rank) + " head: " + e);
    }
    return H;
}
// once per row count, outside any capture: kernel arguments and (graph form) the attention replay entries
inline void prepare(Model& m, int M, cudaStream_t st)
{
    head_for(m, M);
    for (int i = 0; i < m.nl; ++i)
    {
        Layer& y = m.layers[i];
        layer_for(m, i, M);
        if (!m.graph) continue;
        for (int k = 0; k < m.nbk; ++k)
        {
            if (y.rp_on[k][M]) continue;
            const G6AttnStep s = attn_step(m, i, M, M);
            G6RT_CK(g6_attn_bind_replay_init(&y.ab, &s, m.cells + i, m.bk[k], &y.rp[k][M], st));
            y.rp_on[k][M] = true;
        }
    }
}
// graph form: the bucket whose graph runs a step of this context (smallest captured capacity >= context)
inline int bucket_of(const Model& m, int context)
{
    for (int k = 0; k < m.nbk; ++k) if (m.bk[k] >= context) return k;
    return m.nbk - 1;
}
// after the stream is synchronized: attention error words of every layer (0 = clean), by host copies
inline unsigned attn_errors(Model& m)
{
    for (Layer& y : m.layers)
    {
        unsigned e = 0;
        G6RT_CK(cudaMemcpy(&e, y.ab.args.error, 4, cudaMemcpyDeviceToHost));
        if (e) return e;
    }
    return 0;
}
// new sequence: clear every layer's attention caches (load time / between requests, not inside a capture)
inline void reset(Model& m, cudaStream_t st)
{
    for (Layer& y : m.layers) G6RT_CK(g6_attn_bind_reset(&y.ab, st));
}
inline void launch_step_in(Model& m, cudaStream_t st)
{
    k_step_in<<<1, 128, 0, st>>>(StepCtl{reinterpret_cast<const uint32_t*>(m.in), m.stepp, m.tok, m.pos, m.tgt, m.srow, m.cells,
                                         m.shared, m.nl});
#ifdef DEC_S2
    if (m.s2on) s2_launch_cells(m, st);
#endif
}
inline void launch_step_out(Model& m, int M, cudaStream_t st)
{
#ifdef DEC_S2
    if (m.s2on) { k_step_out<<<1, 128, 0, st>>>(s2_step_end(m, StepEnd{m.tok_out, m.lp, m.lpt, m.errs, m.nl, M, m.out})); return; }
#endif
    k_step_out<<<1, 128, 0, st>>>(StepEnd{m.tok_out, m.lp, m.lpt, m.errs, m.nl, M, m.out});
}

// m.x holds the final-norm output of the last layer's second consume
inline void launch_head_local(Model& m, int M, cudaStream_t st)
{
    const bind::Head& H = head_for(m, M);
    bind::launch(H.xf_h, st);
    bind::launch(H.lm_head, st);
    if (m.rs) glue::k_head_local_rs<<<M, 1024, 0, st>>>(head_args(m, M), m.tok);
    else glue::k_head_local<<<M, 1024, 0, st>>>(head_args(m, M));
}
inline void launch_head_merge(Model& m, int M, cudaStream_t st)
{
    if (m.rs && m.nc > 0) glue::k_head_merge_tp_rs<<<M, 1024, 0, st>>>(head_args(m, M), m.tok);
    else if (m.rs) glue::k_head_merge_rs<<<1, 32, 0, st>>>(head_args(m, M), m.tok);
    else if (m.nc > 0) glue::k_head_merge_tp<<<M, 1024, 0, st>>>(head_args(m, M));
    else glue::k_head_merge<<<1, 32, 0, st>>>(head_args(m, M));
    dmark(5, st);
}
// host: top_p misses so far (rows whose nucleus reached past the candidates)
inline unsigned topp_misses(const Model& m)
{
    unsigned x = 0;
    G6RT_CK(cudaMemcpy(&x, m.topp_miss, 4, cudaMemcpyDeviceToHost));
    return x;
}

// the whole step of one rank in stream order (real fabric: the peers' partials arrive while this rank works).
// DEC_GSPLIT (G6_DEC_SPLIT=ls): part 0 = step_in, embed, layers [0, ls); part 1 = layers [ls, nl), head, step_out; the two
// parts are captured as two graphs and launched back to back on one stream (part -1 = the whole step, as before)
inline void launch_step(Model& m, int M, int context, cudaStream_t st, int part = -1, int ls = 0)
{
    if (part != 1)
    {
        dmark(7, st);
        launch_step_in(m, st);
        launch_embed(m, M, st);
        dmark(6, st);
    }
    const int l0 = part == 1 ? ls : 0, l1 = part == 0 ? ls : m.nl;
    for (int i = l0; i < l1; ++i)
    {
        launch_attn_block(m, i, M, context, st);
        launch_consume(m, 2 * i, M, m.layers[i].mlp_norm, m.x2, st);
        launch_mlp_block(m, i, M, st);
        launch_consume(m, 2 * i + 1, M, norm_after(m, i), m.x, st);
        if (m.dfl_tap) m.dfl_tap(m, i, M, st);
    }
    if (part == 0) return;
    launch_head_local(m, M, st);
    launch_head_merge(m, M, st);
    launch_step_out(m, M, st);
    dmark(6, st);
}

// ---------------------------------------------------------------- MTP draft pass (M6): layer 78 as a one-layer Model
// Draft row t (position t in layer 78's OWN caches) takes the token at t+1 and the target model's post-final-norm
// hidden of row t and predicts the token at t+2:
//   x = eh_proj([enorm(emb(tok[t+1])) (zeros at position 0) | hnorm(hidden[t])]);  h = layer 78(x);
//   logits = lm_head(shared_head_norm(h));  a further draft step takes shared_head_norm(h) as its hidden.
// Two forms of the step: "first" reads the hidden rows from the target model's x (rows 0..M-1 of the step the target
// just ran), "chain" (one row) reads the row the previous draft step kept. Layer 78 must see every row the target
// processed (prompt rows too), so its caches stay complete; rows written by chain steps are overwritten by the next
// first pass (attention appends overwrite). v1: every draft step computes its own index list (the old build reuses
// the first pass's list in chain steps).
// The draft Model has its own fabric region (3 collectives), its own step counter and its own StepIn / StepOut:
// StepIn::tok = the NEXT tokens (tok[t+1]), pos = t, srow.pos = t + 2 (the position the row predicts).
struct Draft
{
    Model d;                          // layers = {78}; final_norm = L078.shared_head_norm
    const __half* main_x = nullptr;   // the target model's x (final-norm output rows of its last step)
    const __nv_bfloat16 *enorm = nullptr, *hnorm = nullptr;
    __half* hprev = nullptr;          // [6144] shared_head_norm output of the last row of the previous draft step
};

struct MtpInArgs
{
    const int *tok, *pos;
    const __nv_bfloat16 *emb, *enorm, *hnorm;
    const __half* hid;                // [M][6144]
    float eps;
    __half* out;                      // [M][12288]
};
// grid (2, M), block 256: half 0 = enorm(emb[tok]) (zeros at position 0), half 1 = hnorm(hid). fp32 math, fixed order.
__global__ void __launch_bounds__(256) k_mtp_in(MtpInArgs a)
{
    __shared__ float wsum[8];
    const int m = blockIdx.y, half = blockIdx.x;
    const __nv_bfloat16* e = a.emb + (size_t)a.tok[m] * HID;
    const __half* h = a.hid + (size_t)m * HID;
    const __nv_bfloat16* w = half ? a.hnorm : a.enorm;
    __half* out = a.out + (size_t)m * 2 * HID + (size_t)half * HID;
    const bool zero = half == 0 && a.pos[m] == 0;
    float ss = 0.f;
    for (int i = threadIdx.x; i < HID; i += 256)
    {
        const float v = half ? __half2float(h[i]) : __bfloat162float(e[i]);
        ss = fmaf(v, v, ss);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) tot += wsum[i];
    const float inv = rsqrtf(tot * (1.f / HID) + a.eps);
    for (int i = threadIdx.x; i < HID; i += 256)
    {
        const float v = half ? __half2float(h[i]) : __bfloat162float(e[i]);
        out[i] = zero ? __float2half_rn(0.f) : __float2half_rn(v * inv * __bfloat162float(w[i]));
    }
}
// eh_proj output rows (fp16) -> the residual (fp32) and layer 78's input norm. grid (RN_CL, M), block RN_T.
__global__ void __cluster_dims__(glue::RN_CL, 1, 1) __launch_bounds__(glue::RN_T) k_mtp_resid_norm(const __half* in, const __nv_bfloat16* w,
                                                                                                  float* resid, __half* out, float eps)
{
    const int m = blockIdx.y;
    const int col0 = blockIdx.x * glue::RN_COLS + threadIdx.x * 8;
    const __half* src = in + (size_t)m * HID + col0;
    float x[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) x[k] = __half2float(src[k]);
    float4* r = reinterpret_cast<float4*>(resid + (size_t)m * HID + col0);
    r[0] = make_float4(x[0], x[1], x[2], x[3]);
    r[1] = make_float4(x[4], x[5], x[6], x[7]);
    glue::cluster_rms_store(x, w, eps, out + (size_t)m * HID, col0);
}
// grid 1, block 256
__global__ void __launch_bounds__(256) k_mtp_keep(const __half* src, __half* dst)
{
    for (int i = threadIdx.x; i < HID; i += 256) dst[i] = src[i];
}

// f: a fabric region with 3 collectives and decode-sized slots; stepp: the draft pass's own device step counter
// max_rows: MX for decode only, the prefill sub-batch size when prompts go through the prefill path
inline void build_draft(Draft& D, const Model& main, const Find& find, int capacity, const g6::FabricDev& f, u64* stepp,
                        cudaStream_t st, int max_rows = MX)
{
    if (!find("L078.eh.mx.gv").p) die("pack has no L078.eh.mx.gv (MTP)");
    D.d.wide = main.wide;
    D.d.eh_in = dmalloc<__half>((size_t)MX * 2 * HID);
    D.d.eh_out = dmalloc<__half>((size_t)MX * HID);
    build(D.d, main.rank, std::vector<int>{78}, find, capacity, f, stepp, st, max_rows);
    D.d.final_norm = (const __nv_bfloat16*)need(find, "L078.shared_head_norm");
    D.enorm = (const __nv_bfloat16*)need(find, "L078.enorm");
    D.hnorm = (const __nv_bfloat16*)need(find, "L078.hnorm");
    D.hprev = dmalloc<__half>(HID);
    D.main_x = main.x;
}
#ifdef DEC_S2
inline const __half* s2_draft_hid(Draft& D, bool chain, cudaStream_t st);
inline void s2_draft_keep(Draft& D, cudaStream_t st);
inline int s2_sessions(const Model& m);
#endif
// the draft step's "embedding": StepIn cells -> eh input -> eh_proj -> residual + layer 78 input norm
inline void launch_draft_embed(Draft& D, int M, bool chain, cudaStream_t st)
{
    Model& d = D.d;
    const bind::Head& H = head_for(d, M);
    if (!H.eh.on) die("the head binder has no eh_proj");
#ifdef DEC_S2
    const __half* hid = d.s2on ? s2_draft_hid(D, chain, st) : chain ? D.hprev : D.main_x;
    k_mtp_in<<<dim3(2, M), 256, 0, st>>>(MtpInArgs{d.tok, d.pos, d.emb, D.enorm, D.hnorm, hid, glue::RMS_EPS, d.eh_in});
#else
    k_mtp_in<<<dim3(2, M), 256, 0, st>>>(MtpInArgs{d.tok, d.pos, d.emb, D.enorm, D.hnorm, chain ? D.hprev : D.main_x, glue::RMS_EPS, d.eh_in});
#endif
    bind::launch(H.xf_eh, st);
    bind::launch(H.eh, st);
    k_mtp_resid_norm<<<dim3(glue::RN_CL, M), glue::RN_T, 0, st>>>(d.eh_out, d.layers[0].attn_norm, d.resid, d.x, glue::RMS_EPS);
}
inline void launch_draft_keep(Draft& D, int M, cudaStream_t st)
{
#ifdef DEC_S2
    if (D.d.s2on) { s2_draft_keep(D, st); return; }
#endif
    k_mtp_keep<<<1, 256, 0, st>>>(D.d.x + (size_t)(M - 1) * HID, D.hprev);
}
// the whole draft step of one rank in stream order (real fabric)
inline void launch_draft_step(Draft& D, int M, int context, bool chain, cudaStream_t st)
{
    Model& d = D.d;
#ifdef DEC_S2
    if (const char* e = draft_shape_error(M, chain, d.s2on, d.s2on ? s2_sessions(d) : 0)) die(e);
#else
    if (const char* e = draft_shape_error(M, chain, false, 0)) die(e);
#endif
    launch_step_in(d, st);
    launch_draft_embed(D, M, chain, st);
    launch_attn_block(d, 0, M, context, st);
    launch_consume(d, 0, M, d.layers[0].mlp_norm, d.x2, st);
    launch_mlp_block(d, 0, M, st);
    launch_consume(d, 1, M, d.final_norm, d.x, st);
    launch_head_local(d, M, st);
    launch_head_merge(d, M, st);
    launch_step_out(d, M, st);
    launch_draft_keep(D, M, st);
}
// StepIn of a draft step: rows at positions pos0.., next[r] = the token after row r, keyed to sample position + 2
inline StepIn draft_inputs(u64 step, int M, int pos0, const int* next, float inv_temp, uint32_t seed, float top_p = 0.f)
{
    StepIn in;
    memset(&in, 0, sizeof in);
    in.step = step; in.M = M;
    for (int r = 0; r < M; ++r)
    {
        in.tok[r] = next[r];
        in.pos[r] = pos0 + r;
        in.srow[r] = glue::SampleRow{inv_temp, seed, 0u, pos0 + r + 2, 0, top_p};
        in.tgt[r] = -1;
    }
    in.context = pos0 + M;
    return in;
}

// ---------------------------------------------------------------- the M4 test sequence, shared by both drivers
// The prompt goes in first, `chunk` rows per step at consecutive positions (the last prompt step may be shorter);
// then n_gen single-row steps, each feeding the token sampled from the previous step's last row.
struct Plan
{
    std::vector<int> prompt;
    int first = 0;                    // prompt tokens before this index were taken by the prefill path
    int chunk = 4;                    // rows per prompt step, 1..MX
    int n_gen = 16;
    float inv_temp = 0.f;             // 0 = greedy
    uint32_t seed = 1234u;
    float top_p = 0.f;                // (0,1) = nucleus sampling (needs Model::nc > 0); else off
    int lp = -1, plp = -1;            // --serve: top-k logprobs of the generated / the prompt tokens (k > 0 = top lists; 0 = only
                                      // the token's own logprob; -1 = off). Rows of a step with k > 0 carry SampleRow::topk.
    bool rs = false;                  // --spec-accept rejection: drafts are greedy, the target accepts them by rejection sampling
};
inline int prompt_steps(const Plan& p) { return ((int)p.prompt.size() - p.first + p.chunk - 1) / p.chunk; }
inline int total_steps(const Plan& p) { return prompt_steps(p) + p.n_gen; }
inline int step_rows(const Plan& p, int s)
{
    const int left = (int)p.prompt.size() - p.first - s * p.chunk;
    return s >= prompt_steps(p) ? 1 : left < p.chunk ? left : p.chunk;
}
// the distinct row counts of the plan (each needs prepare() and, in the graph form, its own graph)
inline std::vector<int> plan_rows(const Plan& p)
{
    std::vector<int> v;
    for (int s = 0; s < total_steps(p); ++s)
    {
        const int M = step_rows(p, s);
        bool seen = false;
        for (int x : v) seen |= x == M;
        if (!seen) v.push_back(M);
    }
    return v;
}
// step0: fabric steps used before this sequence (the fabric step counter never repeats within a process)
inline StepIn step_inputs(const Plan& p, int s, int last_tok, u64 step0 = 0)
{
    const int n = (int)p.prompt.size(), M = step_rows(p, s), ps = prompt_steps(p);
    StepIn in;
    memset(&in, 0, sizeof in);
    in.step = step0 + (u64)s + 1;
    in.M = M;
    for (int r = 0; r < M; ++r)
    {
        in.tok[r] = s < ps ? p.prompt[p.first + s * p.chunk + r] : last_tok;
        in.pos[r] = s < ps ? p.first + s * p.chunk + r : n + s - ps;
        in.tgt[r] = in.pos[r] + 1 < n ? p.prompt[in.pos[r] + 1] : -1;
        in.srow[r] = glue::SampleRow{p.inv_temp, p.seed, 0u, in.pos[r] + 1, 0, p.top_p, (in.tgt[r] >= 0 ? p.plp : p.lp) > 0 ? 1 : 0};
    }
    in.context = in.pos[M - 1] + 1;
    return in;
}
// payload bytes of the head collective of a step (the rows with a top list make every rank send it)
inline uint32_t head_size(const StepIn& si, int nc)
{
    bool any = false;
    for (int r = 0; r < si.M; ++r) any |= si.srow[r].topk > 0;
    return (uint32_t)glue::head_bytes_tk(si.M, nc, any);
}

// Host copy of what a step returned for logprobs (after the step's stream is synchronized): the rows' logprobs, and for
// rows with SampleRow::topk > 0 the top lists and ranks. Only fetched for --serve requests that ask.
static_assert(glue::LP_K == LP_MAX && glue::TK_ROWS == MX, "serve.h LP_MAX / rank.cuh MX follow glue::LP_K / TK_ROWS");
struct LpStep
{
    float lp[MX], lpt[MX];
    bool row_tk[MX];
    glue::TkRes tk;
};
inline void lp_fetch(Model& m, const StepOut& so, const StepIn& si, LpStep& L)
{
    bool any = false;
    for (int r = 0; r < MX; ++r) { L.lp[r] = so.lp[r]; L.lpt[r] = so.lpt[r]; L.row_tk[r] = r < si.M && si.srow[r].topk > 0; any |= L.row_tk[r]; }
    if (any) G6RT_CK(cudaMemcpy(&L.tk, m.tk, sizeof L.tk, cudaMemcpyDeviceToHost));
}
// row r of the fetched step as a response entry: its sampled token (target = false) or its teacher-forced target; k = entries wanted
inline LpRow lp_row(const LpStep& L, int r, bool target, int k)
{
    LpRow o;
    o.lp = target ? L.lpt[r] : L.lp[r];
    if (k > 0 && L.row_tk[r])
    {
        o.rank = target ? L.tk.rank_t[r] : L.tk.rank[r];
        for (int i = 0; i < k; ++i)
        {
            if (L.tk.id[r][i] < 0) break;
            o.id[o.n] = L.tk.id[r][i];
            o.tlp[o.n++] = L.tk.lp[r][i];
        }
    }
    return o;
}
inline uint64_t fnv(uint64_t h, const void* p, size_t n)
{
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001B3ull; }
    return h;
}
// after the step's stream is synchronized: the rank's result line (sampled ids, their logprobs and the logprobs of
// the teacher-forced targets as value + bits, hash of the M residual rows) from its StepOut (o: a host copy). Every rank of a run, and both drivers in both forms, must print
// the same lines. Returns the last row's token.
inline int step_line(Model& m, const StepOut& o, int s, int M, std::string& line)
{
    const int* tok = o.tok;
    const float* lp = o.lp;
    std::vector<float> r((size_t)M * HID);
    G6RT_CK(cudaMemcpy(r.data(), m.resid, r.size() * 4, cudaMemcpyDeviceToHost));
    char b[96];
    snprintf(b, sizeof b, "step %d M %d tok", s, M);
    line = b;
    for (int i = 0; i < M; ++i) { snprintf(b, sizeof b, " %d", tok[i]); line += b; }
    line += " lp";
    for (int i = 0; i < M; ++i)
    {
        uint32_t u;
        memcpy(&u, &lp[i], 4);
        snprintf(b, sizeof b, " %.5f/%08x", lp[i], u);
        line += b;
    }
    line += " lpt";
    for (int i = 0; i < M; ++i)
    {
        uint32_t u;
        memcpy(&u, &o.lpt[i], 4);
        snprintf(b, sizeof b, " %.5f/%08x", o.lpt[i], u);
        line += b;
    }
    snprintf(b, sizeof b, " resid %016llx", (unsigned long long)fnv(0xcbf29ce484222325ull, r.data(), r.size() * 4));
    line += b;
    return tok[M - 1];
}

// optional (--top K): for every row of the step, this rank's K largest raw logits of its vocab slice and the slice's
// log-sum-exp, by a host copy of m.hlog after the step's stream is synchronized. One "top" line per row; the ranks'
// lines differ (each holds its own slice), tools/m5/d8.py merges the six into the top-K logprobs of the position.
inline void top_lines(Model& m, int s, int M, int K, std::string& out)
{
    std::vector<float> lg((size_t)M * m.vn);
    G6RT_CK(cudaMemcpy(lg.data(), m.hlog, lg.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int> ix(m.vn);
    char b[64];
    for (int r = 0; r < M; ++r)
    {
        const float* row = lg.data() + (size_t)r * m.vn;
        for (int i = 0; i < m.vn; ++i) ix[i] = i;
        const int k = K < m.vn ? K : m.vn;
        std::partial_sort(ix.begin(), ix.begin() + k, ix.end(), [&](int x, int y) { return row[x] > row[y] || (row[x] == row[y] && x < y); });
        const double mx = row[ix[0]];
        double sm = 0;
        for (int i = 0; i < m.vn; ++i) sm += exp((double)row[i] - mx);
        snprintf(b, sizeof b, "top %d %d rank %d lse %.9g", s, r, m.rank, mx + log(sm));
        out += b;
        for (int i = 0; i < k; ++i) { snprintf(b, sizeof b, " %d:%.9g", m.vlo + ix[i], row[ix[i]]); out += b; }
        out += "\n";
    }
}

// ---------------------------------------------------------------- the MTP test sequence (M6), shared by both drivers
// Prompt: target steps of `chunk` rows, each followed by a first-form draft pass over the same rows (layer 78 sees
// every row). Then rounds until n_gen tokens are committed:
//   verify   one target step over [last committed token, d1..dK] at positions base..base+K
//   accept   a = the number of leading drafts that equal what the target sampled (exact: sampling is keyed by
//            position, and the drafts were sampled with the same keys); the a + 1 sampled tokens are committed.
//            With Plan::rs (--spec-accept rejection) the drafts are greedy and the target's head keeps a draft with
//            probability p(draft), else samples p without it (glue.cuh k_head_local_rs), so the same compare applies
//   draft    a first-form pass over the accepted rows 0..a (-> d1), then K - 1 chain steps (-> d2..dK)
// The rows of rejected drafts, and the chain steps' rows in layer 78, are overwritten by the next round.
// run_main(si, s, ms) runs a target step, prints its line(s) and returns the rank's StepOut; run_draft(di, chain, s, ms)
// runs a draft step, prints its line and returns the last row's token; both add their wall time to ms.
// main_step / draft_step: the fabric step counters (never repeat within a process), updated.
struct MtpStats
{
    long rounds = 0, tokens = 0, acc[MX] = {};
    std::vector<double> t_verify, t_first, t_chain;
    void print(int K, const char* tag) const
    {
        if (!rounds) return;
        auto q = [](std::vector<double> v, double f) { std::sort(v.begin(), v.end()); return v[(size_t)(f * (v.size() - 1))]; };
        fprintf(stderr, "# mtp %d drafts: %ld rounds, %ld tokens, %.3f tokens per round; rounds by accepted drafts", K, rounds, tokens,
                (double)tokens / rounds);
        for (int a = 0; a <= K; ++a) fprintf(stderr, " %d:%ld", a, acc[a]);
        std::vector<double> tot(t_verify.size());
        double sum = 0;
        for (size_t i = 0; i < tot.size(); ++i) { tot[i] = t_verify[i] + t_first[i] + t_chain[i]; sum += tot[i]; }
        fprintf(stderr, "\n# mtp round, %s: p50 verify %.3f ms + first draft %.3f ms + chain %.3f ms; round p50 %.3f ms p90 %.3f ms; "
                        "%.2f tok/s over the rounds\n", tag, q(t_verify, .5), q(t_first, .5), q(t_chain, .5), q(tot, .5), q(tot, .9),
                tokens * 1000.0 / sum);
    }
};
template <class RunMain, class RunDraft>
inline std::vector<int> mtp_sequence(const Plan& p, int K, u64& main_step, u64& draft_step, RunMain&& run_main, RunDraft&& run_draft,
                                     MtpStats& S, int& steps, const std::function<bool(const int*, int, int)>& emit = {})
{
    // emit (--serve): gets every batch of committed tokens and the row of the last run_main step each came from (the
    // rows are consecutive); true = the request is finished (stop id or max tokens)
    const int n = (int)p.prompt.size(), PS = prompt_steps(p);
    const u64 step0 = main_step;
    std::vector<int> tokens = p.prompt;          // committed tokens by position
    std::vector<int> d(K + 1, 0);                // d[j]: draft of position base + j
    const float dtemp = p.rs ? 0.f : p.inv_temp; // rejection sampling: greedy drafts (they must not use the target's noise)
    auto chain_steps = [&](int base) {           // d[1] is set; fills d[2..K]
        double ms = 0;
        for (int j = 1; j < K; ++j)
        {
            const StepIn ci = draft_inputs(++draft_step, 1, base - 1 + j, &d[j], dtemp, p.seed, p.top_p);
            d[j + 1] = run_draft(ci, true, j, ms);
        }
        return ms;
    };
    int s = 0;
    for (; s < PS; ++s)
    {
        const StepIn si = step_inputs(p, s, 0, step0);
        main_step = si.step;
        double ms = 0;
        const StepOut so = run_main(si, s, ms);
        bool fin = false;
        if (s == PS - 1)
        {
            tokens.push_back(so.tok[si.M - 1]);
            if (emit) fin = emit(&so.tok[si.M - 1], 1, si.M - 1);
        }
        if (fin) { steps += s + 1; return std::vector<int>(tokens.begin() + n, tokens.end()); }
        int next[MX];
        for (int r = 0; r < si.M; ++r) next[r] = tokens[si.pos[r] + 1];
        const StepIn di = draft_inputs(++draft_step, si.M, si.pos[0], next, dtemp, p.seed, p.top_p);
        d[1] = run_draft(di, false, s, ms);
    }
    chain_steps((int)tokens.size() - 1);
    while ((int)tokens.size() - n < p.n_gen)
    {
        const int base = (int)tokens.size() - 1;
        StepIn si;
        memset(&si, 0, sizeof si);
        si.step = ++main_step; si.M = K + 1;
        for (int r = 0; r <= K; ++r)
        {
            si.tok[r] = r ? d[r] : tokens[base];
            si.pos[r] = base + r;
            si.srow[r] = glue::SampleRow{p.inv_temp, p.seed, 0u, base + r + 1, 0, p.top_p, p.lp > 0 ? 1 : 0};
            si.tgt[r] = -1;
        }
        si.context = base + K + 1;
        double tv = 0, tf = 0;
        const StepOut so = run_main(si, s, tv);
        int a = 0;
        while (a < K && d[a + 1] == so.tok[a]) ++a;
        for (int r = 0; r <= a; ++r) tokens.push_back(so.tok[r]);
        printf("round %ld base %d accepted %d of %d\n", S.rounds, base, a, K);
        if (emit && emit(so.tok, a + 1, 0))
        {
            ++S.rounds; S.tokens += a + 1; ++S.acc[a];
            S.t_verify.push_back(tv); S.t_first.push_back(0); S.t_chain.push_back(0);
            ++s;
            steps += s;
            const size_t end = std::min(tokens.size(), (size_t)n + p.n_gen);
            return std::vector<int>(tokens.begin() + n, tokens.begin() + end);
        }
        const StepIn di = draft_inputs(++draft_step, a + 1, base, so.tok, dtemp, p.seed, p.top_p);
        d[1] = run_draft(di, false, s, tf);
        const double tc = chain_steps((int)tokens.size() - 1);
        ++S.rounds; S.tokens += a + 1; ++S.acc[a];
        S.t_verify.push_back(tv); S.t_first.push_back(tf); S.t_chain.push_back(tc);
        ++s;
    }
    steps += s;
    return std::vector<int>(tokens.begin() + n, tokens.begin() + n + p.n_gen);
}

}   // namespace g6rt
#ifdef DEC_S2
#include "s2.cuh"
#endif
