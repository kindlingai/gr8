// DeepSeek-V4.1-Flash-1017 attention B (the indexer) in the decode step: the op-table functions of runtime/ds41_ops.h for ST_CA_ROW, ST_INDEX_TOPK,
// ST_INDEX_BORROW, ST_CAND_BLOCKS, and ds41_idx_bind / ds41_idx_prepare, which ds41_attn_bind / ds41_attn_prepare (the owner of the bind_attn / prepare_attn slots) call at their end.
// Included by runtime/ds41_step.cuh inside namespace g6rt after attn/ds41_attn_ops.cuh (it reads that file's Ds41AttnLayer). No kernels here: they
// stay in ds41_bind.cu (ds41_idx.cuh) behind the C interface of ds41_idx_api.h.
//
// Per layer l, plan order (ds41_ops.h plan()): ATTN_PROJ -> [CA_ROW at kv sources 2, 8, 14, 20] -> [INDEX_TOPK at index sources 2, 8,
// 14, 20, 24, 28, 32, 36 (+ CAND_BLOCKS at 20); INDEX_BORROW at a reuse layer whose source is not in the layer subset] -> ATTN_CORE -> ATTN_OUT.
//   input    Model::hc_y bf16 [rows][5120]; positions Model::pos (device); the attention block's Scratch of the layer: S.qr/qrq/qrs (q_a normed, bf16 and mxfp8),
//            S.cnorm (post-norm pre-rope compressed rows the ratio-2 compressor completed in this call), W.inv
//   output   one shared int32 list [rcap][512] (rcap = max(MX, GR8_DS41_IDX_PW)) (ascending, -1 tail) written by every index source and read by every compressed layer after it
//            in plan order (a reuse layer reads the latest list = official model.py: the layer's own index source); Ds41AttnLayer::topk/ldtk
//            of every ratio>0 layer point at it. Layers 20..39 get comp_src = a bf16 shadow [capacity][512] of the layer-20 packed fp4 rows.
//   state    ca (288 B fp4 rows, layer 20) and ica (68 B fp4 index-K rows, layers 2, 8, 14, 20) in flat per-layer pools (one "page" of
//            capacity rows; there is no paged 64-token pool yet), the shared stage scratch, per-layer weight pointers.
//            Layers 2, 8, 14 keep their ca rows in the attention block's own bf16 C.comp (ds41_attn_op_comp_row); only their ica rows are stored here.
#pragma once
#if G6_MODEL_DS41
#include <algorithm>
#include "ds41_idx_api.h"
#include "ds41_idx_layout.h"

constexpr int DS41_CA_ROW = 288, DS41_ICA_ROW = ds41i::ICA_ROW_BYTES;   // bytes per token: fp4 e4m3/16 ca row, fp4 e8m0/32 index-K row (ds41i::CA_ROW / ICA_ROW)
struct Ds41IdxLayer
{
    bool bound = false;
    const __nv_bfloat16 *cmp_wkv = nullptr, *cmp_norm = nullptr;       // layer 20 compressor (ratio 1)
    const __nv_bfloat16 *idx_wk = nullptr, *idx_knorm = nullptr;       // index-K projection of the kv-source layers
    const uint8_t *wqb_codes = nullptr, *wqb_scales = nullptr;         // indexer wq_b (mxfp8) of the index sources
    const uint8_t* wqb_gv = nullptr;                                   // its GEMV records (.mx.gv) when the pack has them (GR8_DS_SRC_FAST bit 2)
    const __nv_bfloat16* wproj = nullptr;                              // indexer weights_proj of the index sources
};
struct Ds41IdxModel
{
    int cap = 0, max_rows = 0;                      // token capacity, compressed-row bound of the ratio-1 layers
    int rcap = 0;                                   // rows the indexer buffers are carved for: max(MX, GR8_DS41_IDX_PW); MX = decode
    int lim1 = 0;                                   // key-row limit of the scratch (ratio-1 rows): max_rows, or GR8_DS41_IDX_PWK when that is smaller
    int chunk_end = 0;                              // ds41_idx_set_chunk: 1 + highest position of the chunk being run (eager prefill only); 0 = off
    void* scratch = nullptr;                        // ds41_idx_scratch_bytes(rcap, lim1)
    int *klen = nullptr, *zero = nullptr;           // device int [rcap]; page table {0}
    __nv_bfloat16 *q_b = nullptr, *wproj = nullptr, *y512 = nullptr, *pre = nullptr, *shadow = nullptr;
    int32_t *topk = nullptr, *borrow = nullptr;
    int borrow_rows = 0;
    uint8_t *ca[4] = {}, *ica[4] = {};              // by kv source 2, 8, 14, 20
    int rows_kv[4] = {};
    std::vector<Ds41IdxLayer> lay;                  // by layer position in Model::layers
    bool cand_ready = false, borrow_warned = false;
    int dense_maxk = 0;                             // GR8_DS41_IDX_DENSE_MAXK=<keys> (default 0 = off): layers 24..36 run the dense stage instead of the candidate path while the chunk's key bound is <= this
    int wqb_tc = 0;                                 // GR8_DS41_IDX_WQB_TC=<min rows> (default 0 = off): wq_b on attention A's tensor-core GEMM when rows >= this (not bit-exact; rel-L2 about 1e-4)
    bool pipe_b = false;                             // GR8_DS41_IDX_PIPE_B=1: the pipelined dense logits on index-source layers 2/8/14/20 only
    bool q_trig = false;                             // GR8_DS41_IDX_Q_TRIG=1: once-per-row query rope sincos
    bool pipe_qr = false;                            // GR8_DS41_IDX_PIPE_QR=1: Q-stationary cp.async dense logits
    bool wqb_tc_warned = false;
    bool reuse = false;                             // GR8_DS41_IDX_REUSE=1 / ds41_idx_set_reuse: layers 24..36 keep the layer-20 top-512 list (no wq_b, logits, top-k there; layer 20 skips cand_blocks)
    // SESS > 1 (Model::ds_slots S2 slots): ca / ica / shadow hold one slab ("page") of rows_kv rows per slot; a call's page table is Model::sess
    // + row0 (the slot of the session's first row), so pages[0] = the slot. A step of ns > 1 sessions runs each index op once per session
    // (rows j*rps .. of Model::ds_ns sessions), session j on its own scratch sscr[j] (MX rows) so layer 20's candidate blocks stay until 24..36.
    int slots = 1;
    int sess_cap = MX;                             // Model::ds41_pack_enabled: 64 rows per segment; else MX, the decode allocation
    void* sscr[32] = {};
};

namespace ds41_idx_detail
{
inline int kvix(int l) { return l == 2 ? 0 : l == 8 ? 1 : l == 14 ? 2 : l == 20 ? 3 : -1; }
// registry Model* -> Ds41IdxModel (the per-layer pointer Layer::ds41_attn belongs to the attention block, so the shared state lives here)
struct Slot { const void* key; Ds41IdxModel* st; };
inline Slot* slots() { static Slot s[16] = {}; return s; }
inline int* lock_word() { static int w = 0; return &w; }
inline Ds41IdxModel* find(const Model& m, bool create)
{
    while (__atomic_exchange_n(lock_word(), 1, __ATOMIC_ACQUIRE)) {}
    Ds41IdxModel* r = nullptr;
    Slot* s = slots();
    for (int i = 0; i < 16 && !r; ++i) if (s[i].key == &m) r = s[i].st;
    if (!r && create)
        for (int i = 0; i < 16; ++i)
            if (!s[i].key) { s[i].key = &m; s[i].st = r = new Ds41IdxModel(); break; }
    __atomic_store_n(lock_word(), 0, __ATOMIC_RELEASE);
    return r;
}
inline const __nv_bfloat16* bf16_mat(Model& m, int l, const char* nm, long long n0, long long n1)
{
    const std::string p = lname(l, nm);
    const TensorRef t = m.ds41_find(p);
    if (!t.p || !t.bf16 || t.nd != 2 || t.shape[0] != n0 || t.shape[1] != n1) die("DS41 indexer bind: bad tensor " + p);
    return (const __nv_bfloat16*)t.p;
}
inline void* zalloc(size_t bytes)
{
    void* p = dmalloc<char>(bytes);
    G6RT_CK(cudaMemset(p, 0, bytes));
    return p;
}
inline bool ready(Model& m, int i, int rows, Ds41IdxModel*& S, Ds41AttnLayer*& a)
{
    S = find(m, false);
    a = ds41_attn_state(m, i);
    return S && a && i >= 0 && i < (int)S->lay.size() && S->lay[i].bound && rows >= 1 && rows <= S->rcap;
}
inline int src_ratio(int kix) { return geom::compress_ratio(kix == 0 ? 2 : kix == 1 ? 8 : kix == 2 ? 14 : 20); }
// Columns the dense logits / top-k run over for kv source kix, and the host bound of every klen[r] = (pos[r] + 1) / ratio.
//   default: the pool size rows_kv (capacity; what a graph captured at bind time needs). GR8_DS41_IDX_PWK (< capacity) lowers it with the scratch.
//   ds41_idx_set_chunk(pos_end) on: the real bound of the chunk, pos_end / ratio (prefill, eager launches only; the carve basis follows, see carve_rows).
inline int max_klen(const Ds41IdxModel& S, int kix)
{
    static const int lim = (int)knob::ds41_idx_maxk();   // bring-up: bound the dense grid below capacity
    int n = S.rows_kv[kix];
    if (S.lim1 < S.max_rows) n = std::min(n, S.lim1 / src_ratio(kix));
    if (S.chunk_end > 0) n = std::min(n, std::max(1, S.chunk_end / src_ratio(kix)));
    return lim > 0 && lim < n ? lim : n;
}
// Rows the scratch is carved for ("max_rows" of Scratch::carve): constant per decode step / prefill chunk, so L20's logits and candidates (written
// by its dense stage and cand_blocks) are where layers 24..36 expect them. Must be <= lim1 (the allocation).
inline int carve_rows(const Ds41IdxModel& S) { return S.chunk_end > 0 ? S.chunk_end : S.lim1; }
// Layers 24..36 take the dense stage (use_cand = 0) when the key bound of this call is <= dense_maxk. With the layer-20 candidate set (2048 blocks of
// 8 tokens = 16384 keys) dense_maxk <= 16384 selects over the same keys as the candidate path would (all of them), without the gather. Layer 20 skips
// cand_blocks under the same test (same kix, same chunk_end), so the two sides always agree.
inline bool cand_dense(const Ds41IdxModel& S, int kix) { return S.dense_maxk > 0 && max_klen(S, kix) <= S.dense_maxk; }
inline bool cand_skipped(const Ds41IdxModel& S, int kix) { return S.reuse || cand_dense(S, kix); }
// Uniform decode uses rps; an eager prefill chunk uses the validated ragged row0/len table.
struct Sess { int ns = 1; int row0[32] = {}, len[32] = {}, cnorm0[32] = {}; };
inline bool sessions(const Model& m, const Ds41IdxModel& S, int rows, Sess& q)
{
    if (m.ds41_pack) {
        if (ds41::pack::index_error(*m.ds41_pack, rows, S.rcap, S.sess_cap, S.slots)) return false;
        q.ns = m.ds41_pack->count;
        for (int j = 0; j < q.ns; ++j) {
            q.row0[j] = m.ds41_pack->seg[j].row0;
            q.len[j] = m.ds41_pack->seg[j].rows;
            q.cnorm0[j] = q.row0[j]; // the compressor's segment-sliced output is at its original row offset
        }
        return true;
    }
    q.ns = S.slots > 1 ? m.ds_ns : 1;
    if (q.ns < 1 || q.ns > S.slots || rows % q.ns) return false;
    const int rps = rows / q.ns;
    for (int j = 0; j < q.ns; ++j) {
        q.row0[j] = j * rps; q.len[j] = rps; q.cnorm0[j] = j * ((rps + 1) / 2);
    }
    return true;
}
// page table of session j's rows: SESS > 1 = its slot word in Model::sess; else the {0} table (one flat page)
inline const int* pages_of(const Model& m, const Ds41IdxModel& S, int row0) { return S.slots > 1 ? m.sess + row0 : S.zero; }
// stage of session j (rows j*rps .. (j+1)*rps-1); one session: j = 0, rps = rows, the whole stage
inline Ds41IdxStage stage(Model& m, Ds41IdxModel& S, int kix, const Sess& q, int j)
{
    const int row0 = q.row0[j];
    Ds41IdxStage t{};
    t.q_b = S.q_b + (size_t)row0 * geom::IDX_WQB_OUT; t.wproj = S.wproj + (size_t)row0 * geom::IDX_HEADS; t.dpos = m.pos + row0;
    t.klen = S.klen + row0; t.max_klen = max_klen(S, kix); t.R = q.len[j];
    t.ica = S.ica[kix]; t.pages = pages_of(m, S, row0); t.rpp = S.rows_kv[kix];
    t.scratch = q.ns > 1 ? S.sscr[j] : S.scratch; t.Rcap = q.ns > 1 ? S.sess_cap : S.rcap; t.max_rows = carve_rows(S); t.use_cand = 0;
    t.pipe_b = 0;  // set by index_topk only for source layers; candidate layers keep the plain logits even when GR8_DS41_IDX_DENSE_MAXK makes them dense
    t.q_trig = S.q_trig;
    t.pipe_qr = 0;
    t.out = S.topk + (size_t)row0 * geom::IDX_TOPK; t.ldo = geom::IDX_TOPK;
    return t;
}
}   // namespace ds41_idx_detail

// Table::bind_attn part (build, once per layer, rows = capacity; ds41_attn_bind has run): weights, row pools, shared buffers, and the
// topk / ldtk / comp_src pointers of the attention block's per-layer state. Idempotent per layer.
inline int ds41_idx_bind(void* model, int i, int capacity, void* stream)
{
    using namespace ds41_idx_detail;
    Model& m = *(Model*)model;
    const int l = m.layers[i].l;
    const int ratio = geom::compress_ratio(l);
    if (ratio == 0) return 0;
    Ds41AttnLayer* a = ds41_attn_state(m, i);
    if (!a) die("DS41 indexer bind: layer " + std::to_string(l) + " has no attention state (ds41_attn_bind did not run)");
    Ds41IdxModel& S = *find(m, true);
    if (S.lay.empty())
    {
        // GR8_DS41_IDX_PW=<rows> (prefill chunk width W, default = MX): every per-row buffer of the indexer is carved for max(MX, PW) rows. This is the
        // indexer's own row cap; the step driver's per-row arrays (Model::pos ...) must hold the same number of rows. GR8_DS41_IDX_PWK=<keys>: the
        // scratch holds logits for that many ratio-1 key rows only (default capacity + rows); it bounds the largest position the indexer serves.
        S.rcap = std::max<int>(MX, (int)knob::ds41_idx_pw().value_or(MX));
        if (S.rcap > 65536) die("DS41 indexer bind: GR8_DS41_IDX_PW " + std::to_string(S.rcap) + " > 65536");
        S.cap = capacity;
        S.max_rows = capacity + S.rcap;
        const int pwk = (int)knob::ds41_idx_pwk();
        S.lim1 = pwk > 0 && pwk < S.max_rows ? pwk : S.max_rows;
        S.reuse = knob::ds41_idx_reuse();
        S.dense_maxk = (int)knob::ds41_idx_dense_maxk();
        S.wqb_tc = (int)knob::ds41_idx_wqb_tc();
        S.pipe_b = m.ds41_idx_pipe_b;
        S.q_trig = knob::ds41_idx_q_trig();
        S.pipe_qr = knob::ds41_idx_pipe_qr();
        if (S.pipe_qr) {
            const int rc = ds41_idx_pipe_qr_prepare();
            if (rc) die("DS41 indexer QR dynamic shared-memory opt-in failed: CUDA code " + std::to_string(rc));
        }
        S.lay.resize(m.nl);
        S.slots = m.ds_slots;
        S.sess_cap = m.ds41_pack_enabled ? ds41::pack::MAX_SEGMENT_ROWS : MX;
        if (S.slots < 1 || S.slots > 32) die("DS41 indexer bind: ds_slots " + std::to_string(S.slots) + " outside 1..32");
        S.scratch = zalloc(ds41_idx_scratch_bytes(S.rcap, S.lim1));
        if (S.slots > 1) for (int j = 0; j < S.slots; ++j) S.sscr[j] = zalloc(ds41_idx_scratch_bytes(S.sess_cap, S.lim1));
        S.klen = (int*)zalloc((size_t)S.rcap * sizeof(int));
        S.zero = (int*)zalloc(sizeof(int));
        S.q_b = (__nv_bfloat16*)zalloc((size_t)S.rcap * geom::IDX_WQB_OUT * 2);
        S.wproj = (__nv_bfloat16*)zalloc((size_t)S.rcap * geom::IDX_HEADS * 2);
        S.y512 = (__nv_bfloat16*)zalloc((size_t)S.rcap * 512 * 2);
        S.pre = (__nv_bfloat16*)zalloc((size_t)S.rcap * 512 * 2);
        S.topk = (int32_t*)zalloc((size_t)S.rcap * geom::IDX_TOPK * sizeof(int32_t));
        G6RT_CK(cudaMemset(S.topk, 0xFF, (size_t)S.rcap * geom::IDX_TOPK * sizeof(int32_t)));   // until an index source runs the list reads "no key": -1
        if (const auto& f = knob::ds41_borrow_topk())   // raw int32 [rows][512] top-k lists standing in for an index source outside the subset
        {
            FILE* fp = fopen(f->c_str(), "rb");
            if (!fp) die("DS41 indexer bind: cannot open GR8_DS41_BORROW_TOPK " + *f);
            std::vector<int32_t> buf((size_t)S.rcap * geom::IDX_TOPK, -1);
            const size_t got = fread(buf.data(), sizeof(int32_t), buf.size(), fp);
            fclose(fp);
            S.borrow_rows = (int)(got / geom::IDX_TOPK < (size_t)S.rcap ? got / geom::IDX_TOPK : (size_t)S.rcap);
            S.borrow = (int32_t*)dmalloc<char>(buf.size() * sizeof(int32_t));
            G6RT_CK(cudaMemcpy(S.borrow, buf.data(), buf.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        }
    }
    Ds41IdxLayer& w = S.lay[i];
    if (w.bound) return 0;
    const int kix = kvix(l);   // kv sources: ica rows (and layer 20: ca rows, compressor)
    if (kix >= 0)
    {
        w.idx_wk = bf16_mat(m, l, "idx.wk", geom::IDX_DIM, 512);
        w.idx_knorm = ds41_vec(m, l, "idx.k_norm", geom::IDX_DIM);
        // The ratio-2 ICA's 68-byte rows need a 16-byte slot stride for pipe-B's uint4 loads.
        // Keep ratio-1 unchanged: its rpp also indexes the separate layer-20 BF16 shadow.
        // All ratio-2 ICA writers/readers and per-slot clear use this same padded rpp.
        const int nominal_kv_rows = ratio == 2 ? (capacity + S.rcap) / 2 + 2 : capacity + S.rcap;
        S.rows_kv[kix] = ratio == 2 ? ds41i::ica_aligned_rows(nominal_kv_rows) : nominal_kv_rows;
        S.ica[kix] = (uint8_t*)zalloc((size_t)S.slots * S.rows_kv[kix] * DS41_ICA_ROW);
        if (l == 20)
        {
            w.cmp_wkv = bf16_mat(m, l, "cmp.wkv", 512, HID);
            w.cmp_norm = ds41_vec(m, l, "cmp.norm", 512);
            S.ca[kix] = (uint8_t*)zalloc((size_t)S.slots * S.rows_kv[kix] * DS41_CA_ROW);
        }
    }
    if (geom::is_index_source(l))
    {
        const ds41a::Fp8Mat q = ds41_fp8(m, l, "idx.wq_b", geom::IDX_WQB_OUT, ds41a::QA);
        w.wqb_codes = q.w; w.wqb_scales = q.s; w.wqb_gv = q.gv;
        w.wproj = bf16_mat(m, l, "idx.weights_proj", geom::IDX_HEADS, HID);
    }
    if (l >= 20 && !S.shadow) S.shadow = (__nv_bfloat16*)zalloc((size_t)S.slots * (capacity + S.rcap) * 512 * 2);
    a->topk = S.topk;
    a->ldtk = geom::IDX_TOPK;
    if (l >= 20) { a->comp_src = S.shadow; a->C.src_ss = (size_t)(capacity + S.rcap) * 512; }   // the shadow's slot stride = the layer-20 ca slab (rows_kv)
    w.bound = true;
    (void)stream;
    return 0;
}
// Table::prepare_attn part (per row count M): everything is carved once for rcap = max(MX, GR8_DS41_IDX_PW) rows (stable addresses for every graph),
// nothing to do but the row check.
inline int ds41_idx_prepare(void* model, int i, int rows, void*)
{
    (void)i;
    const Ds41IdxModel* S = ds41_idx_detail::find(*(Model*)model, false);
    return rows >= 1 && rows <= (S ? S->rcap : MX) ? 0 : 1;
}
// Prefill (eager launches, not inside a graph capture): call before the first index op of each chunk with pos_end = 1 + the highest position the
// chunk's rows hold (pos0 + W). The dense logits / top-k / candidate stages then run over the real key count pos_end / ratio and the scratch is carved
// for pos_end ratio-1 rows (logits [rcap][ld_of(pos_end)]) instead of the capacity. Use one value for all layers of a chunk (layer 20's candidates are
// read by layers 24..36 through the same carve). pos_end = 0 returns to the capacity bound of the decode graphs. Returns 0, 1 = no indexer state,
// 2 = pos_end > the scratch key limit (GR8_DS41_IDX_PWK / capacity + rows).
inline int ds41_idx_set_chunk(void* model, int pos_end)
{
    Ds41IdxModel* S = ds41_idx_detail::find(*(Model*)model, false);
    if (!S || S->lay.empty()) return 1;
    if (pos_end < 0 || pos_end > S->lim1) return 2;
    S->chunk_end = pos_end;
    return 0;
}
// Reuse switch (eager launches only; a graph keeps the value it was captured with): on = layers 24/28/32/36 reuse the layer-20 top-512 list and layer 20
// skips cand_blocks. Off = the candidate path (needs layer 20's cand_blocks of the same step/chunk to have run with reuse off). GR8_DS41_IDX_REUSE=1
// sets the start value at bind. Returns 0, 1 = no indexer state.
inline int ds41_idx_set_reuse(void* model, int on)
{
    Ds41IdxModel* S = ds41_idx_detail::find(*(Model*)model, false);
    if (!S || S->lay.empty()) return 1;
    S->reuse = on != 0;
    if (S->reuse) S->cand_ready = false;
    return 0;
}
// ST_CA_ROW at the kv sources 2, 8, 14 (ratio 2: the attention block's compressor makes the ca row; the index-K row comes from its post-norm latent) and 20
// (ratio 1: wkv -> RMSNorm -> rope -> fp4 ca row + bf16 shadow; the same latent -> index-K row).
inline int ds41_idx_op_ca_row(void* model, int i, int rows, void* stream)
{
    using namespace ds41_idx_detail;
    Model& m = *(Model*)model;
    Ds41IdxModel* S; Ds41AttnLayer* a;
    if (!ready(m, i, rows, S, a)) return ds41::NOTSUP;
    const int l = m.layers[i].l, kix = kvix(l);
    if (kix < 0 || !S->ica[kix]) return ds41::NOTSUP;
    const Ds41IdxLayer& w = S->lay[i];
    const int ratio = geom::compress_ratio(l);
    Sess q;
    if (!sessions(m, *S, rows, q)) return 1;
    if (ratio == 2)
    {
        if (int e = ds41_attn_op_comp_row(model, i, rows, stream)) return e;
        for (int j = 0; j < q.ns; ++j)   // uniform decode compacts cnorm; ragged W keeps it at each segment's row0
        {
            const int row0 = q.row0[j];
            ds41_idx_ica_row(a->S.cnorm + (size_t)q.cnorm0[j] * 512, w.idx_wk, w.idx_knorm, m.pos + row0, 2, S->ica[kix],
                             pages_of(m, *S, row0), S->rows_kv[kix], q.len[j], stream);
        }
        return 0;
    }
    ds41_idx_gemv(m.hc_y, HID, w.cmp_wkv, 512, HID, S->y512, 512, rows, stream);
    for (int j = 0; j < q.ns; ++j)
    {
        const int row0 = q.row0[j];
        const int* pg = pages_of(m, *S, row0);
        ds41_idx_ca_row_r1(S->y512 + (size_t)row0 * 512, w.cmp_norm, m.pos + row0, S->ca[kix], pg, S->rows_kv[kix], S->pre + (size_t)row0 * 512,
                           S->shadow, q.len[j], stream);
        ds41_idx_ica_row(S->pre + (size_t)row0 * 512, w.idx_wk, w.idx_knorm, m.pos + row0, 1, S->ica[kix], pg, S->rows_kv[kix], q.len[j], stream);
    }
    return 0;
}
// ST_INDEX_TOPK at an index source: q_b = wq_b(q_a normed) (mxfp8 as attention A), w = weights_proj(x), logits over the visible index-K rows
// (layers 24/28/32/36: only over the layer-20 candidate blocks), top-512 -> the shared list.
inline int ds41_idx_op_index_topk(void* model, int i, int rows, void* stream)
{
    using namespace ds41_idx_detail;
    Model& m = *(Model*)model;
    Ds41IdxModel* S; Ds41AttnLayer* a;
    if (!ready(m, i, rows, S, a)) return ds41::NOTSUP;
    const int l = m.layers[i].l, ratio = geom::compress_ratio(l), kix = kvix(geom::kv_source(l));
    const Ds41IdxLayer& w = S->lay[i];
    if (!ratio || kix < 0 || !geom::is_index_source(l) || !S->ica[kix] || !w.wqb_codes) return ds41::NOTSUP;
    const bool cand = l > geom::CAND_SOURCE;
    if (cand && S->reuse)   // the shared list stays what layer 20 wrote (approximate: layers 24..36 would pick their own top-512 from the candidate blocks)
    {
        return 0;
    }
    const bool use_cand = cand && !cand_dense(*S, kix);   // dense stage at the candidate layers while the keys fit in dense_maxk
    if (use_cand && !S->cand_ready)
    {
        fprintf(stderr, "DS41 index_topk layer %d: no candidate blocks (layer %d did not run cand_blocks before it)\n", l, geom::CAND_SOURCE);
        return ds41::NOTSUP;
    }
    Sess q;
    if (!sessions(m, *S, rows, q)) return 1;
    ds41_idx_klen(m.pos, ratio, S->klen, rows, stream);   // per row: klen[r] = (pos[r] + 1) / ratio, every session at once
    // Keep SRC_FAST's decode GV path. Prefill can opt into the attention block's tensor-core
    // GEMM; when that entry refuses a shape, fall back to the scalar GEMM.
    if ((g_ds41_src_fast & 2) && w.wqb_gv && ds41_attn_gv_ready(&a->W, rows))
    {
        ds41a::Fp8Mat W{};
        W.gv = w.wqb_gv; W.N = geom::IDX_WQB_OUT; W.K = ds41a::QA;
        ds41a::gv_gemv_launch(a->S.xf_qr, W, (ds41a::bf16*)S->q_b, geom::IDX_WQB_OUT, rows, 1, 0, (cudaStream_t)stream, 1, 0);
    }
    else
    {
        int tc = -2;   // off or not linked/refused => scalar fallback
        if (S->wqb_tc > 0 && rows >= S->wqb_tc)
        {
            tc = ds41_idx_gemm_mx_tc(a->S.qrq, ds41a::QA, a->S.qrs, ds41a::QA / 32, w.wqb_codes, w.wqb_scales,
                                     geom::IDX_WQB_OUT, ds41a::QA, S->q_b, geom::IDX_WQB_OUT, rows, stream);
            if (tc != 0 && !S->wqb_tc_warned)
            {
                S->wqb_tc_warned = true;
                fprintf(stderr, "DS41 index_topk: GR8_DS41_IDX_WQB_TC set but the tensor-core wq_b did not run (rc %d%s), scalar gemm instead\n", tc,
                        tc == -1 ? ": ds41_gemm_mx_tc not linked" : "");
            }
        }
        if (tc != 0)
            ds41_idx_gemm_mx(a->S.qrq, ds41a::QA, a->S.qrs, ds41a::QA / 32, w.wqb_codes, w.wqb_scales,
                             geom::IDX_WQB_OUT, ds41a::QA, S->q_b, geom::IDX_WQB_OUT, rows, stream);
    }
    ds41_idx_gemv(m.hc_y, HID, w.wproj, geom::IDX_HEADS, HID, S->wproj, geom::IDX_HEADS, rows, stream);
    for (int j = 0; j < q.ns; ++j)
    {
        Ds41IdxStage t = stage(m, *S, kix, q, j);
        t.use_cand = use_cand;
        t.pipe_b = S->pipe_b && !cand;  // exactly the four native dense index sources (2/8/14/20)
        t.pipe_qr = S->pipe_qr && !cand && S->chunk_end > 0 && q.len[j] >= 16;
        ds41_idx_topk(&t, stream);
    }
    return 0;
}
// ST_CAND_BLOCKS (layer 20, right after its INDEX_TOPK): the top-2048 blocks of 8 tokens by block max of the logits it left in the scratch
inline int ds41_idx_op_cand_blocks(void* model, int i, int rows, void* stream)
{
    using namespace ds41_idx_detail;
    Model& m = *(Model*)model;
    Ds41IdxModel* S; Ds41AttnLayer* a;
    if (!ready(m, i, rows, S, a)) return ds41::NOTSUP;
    const int kix = kvix(geom::kv_source(m.layers[i].l));
    if (kix < 0 || !S->ica[kix]) return ds41::NOTSUP;
    if (cand_skipped(*S, kix)) return 0;   // reuse or dense: nothing reads the candidate blocks
    Sess q;
    if (!sessions(m, *S, rows, q)) return 1;
    for (int j = 0; j < q.ns; ++j)
    {
        const Ds41IdxStage t = stage(m, *S, kix, q, j);
        ds41_idx_cand_blocks(&t, stream);
    }
    S->cand_ready = true;
    return 0;
}
// Bounded replay (ST_PF_COMPACT, runtime/ds41_pf_split.h): the per-row state of the indexer that layers >= 21 read and nobody rewrites before
// that, moved from rows [rows - tail, rows) to [0, tail): the shared top-512 list (layer 20 wrote it; layers 21..23 read it, and 24..36 under
// GR8_DS41_IDX_REUSE) and layer 20's candidate blocks (layers 24..36 read them). shift(base, row_bytes) is the driver's overlap-safe row move.
// 0 = ok, 1 = no indexer state. klen, q_b, wproj, logits are recomputed by every index_topk; ca / ica / shadow rows are by position.
template <class Shift>
inline int ds41_idx_pf_compact(void* model, Shift&& shift)
{
    using namespace ds41_idx_detail;
    Ds41IdxModel* S = find(*(Model*)model, false);
    if (!S || S->lay.empty()) return 1;
    shift(S->topk, (size_t)geom::IDX_TOPK * sizeof(int32_t));
    if (S->cand_ready && !cand_skipped(*S, kvix(geom::CAND_SOURCE)))
        shift(ds41_idx_cand_list(S->scratch, S->rcap, carve_rows(*S)), (size_t)geom::CAND_TOPK_BLOCKS * sizeof(int32_t));
    return 0;
}
// New sequence (between requests, not inside a graph capture): returns every buffer of the indexer to its bind-time value. ica / ca rows, the
// layer-20 shadow, the stage inputs and the scratch go to zero, the shared top-k list to -1 ("no key"), cand_ready to false (the first index
// step of a request must run layer 20's cand_blocks before layers 24..36 read the candidates). Weights, addresses and the borrow list stay.
// A model that never ran ds41_idx_bind has nothing to clear (returns 0). The attention block's own caches (window ring, compressed rows, compressor state)
// are cleared by its own function. Called by rank.cuh reset() for DS41 (forward declare it there like ds41_attn_ops.cuh does for ds41_idx_bind).
// slot (SESS > 1): clear only that S2 slot's ca / ica / shadow slab (the other sessions keep theirs); -1 = every slot. The step buffers below
// (scratch, klen, q_b, ..., the shared list) are rewritten by every step before they are read, so they are cleared whole either way.
inline int ds41_idx_clear(void* model, void* stream, int slot)   // slot default -1: rank.cuh's declaration
{
    using namespace ds41_idx_detail;
    Model& m = *(Model*)model;
    Ds41IdxModel* S = find(m, false);
    if (!S || S->lay.empty()) return 0;
    if (slot < -1 || slot >= S->slots) return 1;
    const cudaStream_t st = (cudaStream_t)stream;
    auto clr = [&](void* p, size_t bytes) { if (p && bytes) G6RT_CK(cudaMemsetAsync(p, 0, bytes, st)); };
    const size_t k0 = slot < 0 ? 0 : (size_t)slot, nk = slot < 0 ? (size_t)S->slots : 1;
    for (int k = 0; k < 4; ++k)
    {
        if (S->ica[k]) clr(S->ica[k] + k0 * S->rows_kv[k] * DS41_ICA_ROW, nk * S->rows_kv[k] * DS41_ICA_ROW);
        if (S->ca[k]) clr(S->ca[k] + k0 * S->rows_kv[k] * DS41_CA_ROW, nk * S->rows_kv[k] * DS41_CA_ROW);
    }
    if (S->shadow) clr(S->shadow + k0 * (S->cap + S->rcap) * 512, nk * (S->cap + S->rcap) * 512 * 2);
    clr(S->scratch, ds41_idx_scratch_bytes(S->rcap, S->lim1));
    for (int j = 0; j < S->slots; ++j) clr(S->sscr[j], ds41_idx_scratch_bytes(S->sess_cap, S->lim1));
    clr(S->klen, (size_t)S->rcap * sizeof(int));
    clr(S->q_b, (size_t)S->rcap * geom::IDX_WQB_OUT * 2);
    clr(S->wproj, (size_t)S->rcap * geom::IDX_HEADS * 2);
    clr(S->y512, (size_t)S->rcap * 512 * 2);
    clr(S->pre, (size_t)S->rcap * 512 * 2);
    if (S->topk) G6RT_CK(cudaMemsetAsync(S->topk, 0xFF, (size_t)S->rcap * geom::IDX_TOPK * sizeof(int32_t), st));   // 0xFFFFFFFF = int32 -1
    S->chunk_end = 0;
    S->cand_ready = false;
    return 0;
}
// ST_INDEX_BORROW: a reuse layer whose index source is outside the layer subset. GR8_DS41_BORROW_TOPK=<raw int32 [rows][512]> (the CPU reference's dump of
// that source's list) is copied into the shared list; without it the list stays what the last index source left (a one-time warning).
inline int ds41_idx_op_index_borrow(void* model, int i, int rows, void* stream)
{
    using namespace ds41_idx_detail;
    Model& m = *(Model*)model;
    Ds41IdxModel* S; Ds41AttnLayer* a;
    if (!ready(m, i, rows, S, a)) return ds41::NOTSUP;
    if (S->borrow)
    {
        const int n = rows < S->borrow_rows ? rows : S->borrow_rows;
        G6RT_CK(cudaMemcpyAsync(S->topk, S->borrow, (size_t)n * geom::IDX_TOPK * sizeof(int32_t), cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
    }
    else if (!S->borrow_warned)
    {
        S->borrow_warned = true;
        fprintf(stderr, "DS41 index_borrow layer %d: index source %d is not in the subset and GR8_DS41_BORROW_TOPK is unset; the list is the last one written\n",
                m.layers[i].l, geom::index_source(m.layers[i].l));
    }
    return 0;
}
#endif
