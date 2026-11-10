// Included inside g6rt after the stock attention/indexer op headers. Include
// ds41_v2.h and ../attn/ds41_attn_v2_api.h at GLOBAL scope first. This adapter
// is called only by the v2 conductor; no stock or decode op is modified.
#pragma once
#if G6_MODEL_DS41
struct Ds41V2Attention {
    int rows = 0, q = 0, heads = 0;
    ds41a::v2::InputBuf in{};
    uint8_t *input_send = nullptr, *input_all = nullptr;
    uint8_t *output_send = nullptr, *output_recv = nullptr;
    __nv_bfloat16* parts = nullptr;
    int32_t *topk = nullptr, *candidates = nullptr;
    std::vector<std::vector<ds41a::Fp8Mat>> wob;
    std::vector<void*> allocations;
    template <class T> T* alloc(size_t n) {
        T* p = dmalloc<T>(n);
        allocations.push_back(p);
        return p;
    }
};

// Full wq_a/wkv/compressor/index weights are already replicated by the stock
// binder. Only wo_b is sharded. Keep remote slices in their original contiguous
// [HID][local K] layouts (no K-strided GEMM or boot repack is needed).
inline void ds41_v2_attn_boot(::ds41::v2::State& v, Model& m, int max_rows)
{
    if (v.attn) die("DS41 v2 attention: duplicate boot");
    if (max_rows < 1 || m.nl < 1) die("DS41 v2 attention: invalid row/layer capacity");
    auto* x = new Ds41V2Attention();
    x->rows = max_rows;
    x->q = (max_rows + g6nr::NR - 1) / g6nr::NR;
    x->wob.resize(m.nl);
    x->in.carve(x->alloc<char>(ds41a::v2::InputBuf::bytes(max_rows, x->q)), max_rows, x->q);
    x->input_send = x->alloc<uint8_t>(size_t(x->q) * ds41a::v2::input_stride(2));
    x->input_all = x->alloc<uint8_t>(size_t(max_rows) * ds41a::v2::input_stride(2));
    x->parts = x->alloc<__nv_bfloat16>(size_t(g6nr::NR) * x->q * ds41a::HID);
    x->topk = x->alloc<int32_t>(size_t(x->q) * geom::IDX_TOPK);
    x->candidates = x->alloc<int32_t>(size_t(x->q) * geom::CAND_TOPK_BLOCKS);
    for (int i = 0; i < m.nl; ++i) {
        if (m.layers[i].l >= ::ds41::v2::FIRST_LATE) continue;
        auto* a = ds41_attn_state(m, i);
        if (!a || a->rowcap < max_rows) die("DS41 v2 attention: attention capacity/binding missing");
        if (!x->heads) x->heads = a->heads;
        if (x->heads != a->heads) die("DS41 v2 attention: inconsistent head count");
        const ds41a::Fp8Mat& own = a->W.wo_b;
        const size_t bytes = size_t(own.N) * own.K;
        if (own.N != ds41a::HID || own.K != ds41a::v2::output_width(a->heads))
            die("DS41 v2 attention: wo_b shape mismatch");
        // Temporary full gather dies at boot; the local slice is never duplicated
        // permanently. Every rank makes the same two boot collectives per layer.
        uint8_t* all = dmalloc<uint8_t>(size_t(g6nr::NR) * bytes);
        ::ds41::v2::gather_weight(v, own.w, bytes, all);
        x->wob[i].resize(g6nr::NR);
        for (int p = 0; p < g6nr::NR; ++p) {
            auto& w = x->wob[i][p];
            w = own; w.gv = nullptr;  // no remote decode GV records required
            if (p == m.rank) continue;
            auto* dst = x->alloc<uint8_t>(bytes);
            G6RT_CK(cudaMemcpyAsync(dst, all + size_t(p) * bytes, bytes, cudaMemcpyDeviceToDevice, v.st));
            w.w = dst;
        }
        ::ds41::v2::gather_weight(v, own.s, bytes / 32, all);
        for (int p = 0; p < g6nr::NR; ++p) {
            if (p == m.rank) continue;
            auto* dst = x->alloc<uint8_t>(bytes / 32);
            G6RT_CK(cudaMemcpyAsync(dst, all + size_t(p) * bytes / 32, bytes / 32, cudaMemcpyDeviceToDevice, v.st));
            x->wob[i][p].s = dst;
        }
        G6RT_CK(cudaFree(all)); // boot only; drains copies before scratch is released
    }
    if (!x->heads) die("DS41 v2 attention: no range-A layers");
    const size_t output_bytes = size_t(g6nr::NR) * x->q * ds41a::v2::output_stride(x->heads);
    x->output_send = x->alloc<uint8_t>(output_bytes);
    x->output_recv = x->alloc<uint8_t>(output_bytes);
    v.attn = x;
}
inline void ds41_v2_attn_release(::ds41::v2::State& v)
{
    auto* x = static_cast<Ds41V2Attention*>(v.attn);
    if (!x) return;
    // Caller drains both conductor streams before release.
    for (void* p : x->allocations) G6RT_CK(cudaFree(p));
    delete x;
    v.attn = nullptr;
}

// After the input gather, keep the stock cache/slot semantics. In particular,
// ratio-2 pooling runs over ALL rows: a pair may cross an ownership boundary.
inline void ds41_v2_attn_project_rest(::ds41::v2::State& v, Ds41V2Attention& x, Model& m, int i)
{
    auto* a = ds41_attn_state(m, i);
    const int rows = v.g.M;
    if (ds41_attn_sessions(m, a, rows)) die("DS41 v2 attention: invalid single-session chunk");
    const int dec_cap = a->C.ring_rows - ds41a::WIN + 1;
    if (a->ring_pf && rows > dec_cap) {
        k_ds41_ring_stage<<<ds41a::WIN - 1, ds41a::HD, 0, v.st>>>(a->ring_pf, a->ring_pf_rows, a->C.ring, a->C.ring_rows, a->S.sess, m.pos);
        auto cache = a->C; cache.ring = a->ring_pf; cache.ring_rows = a->ring_pf_rows;
        auto scratch = a->S; scratch.sess = nullptr; scratch.ns = 1;
        ds41_v2_project_rest(&a->W, &cache, &scratch, &x.in, rows, a->heads, v.st, m.pos);
        k_ds41_ring_commit<<<ds41a::WIN, ds41a::HD, 0, v.st>>>(a->C.ring, a->C.ring_rows, a->S.kvrows, rows, a->S.sess, m.pos);
    } else ds41_v2_project_rest(&a->W, &a->C, &a->S, &x.in, rows, a->heads, v.st, m.pos);
    const int l = m.layers[i].l, kix = ds41_idx_detail::kvix(l);
    if (kix < 0) return;
    auto* s = ds41_idx_detail::find(m, false);
    if (!s || !s->ica[kix]) die("DS41 v2 attention: missing index key cache");
    const auto& w = s->lay[i];
    const int* pages = ds41_idx_detail::pages_of(m, *s, 0);
    if (geom::has_compressor_gate(l)) {
        ds41_v2_comp_rest(&a->W, &a->C, &a->S, &x.in, rows, v.st, m.pos);
        ds41_idx_ica_row(a->S.cnorm, w.idx_wk, w.idx_knorm, m.pos, 2, s->ica[kix], pages, s->rows_kv[kix], rows, v.st);
    } else {
        ds41_idx_ca_row_r1(x.in.ckv1, w.cmp_norm, m.pos, s->ca[kix], pages, s->rows_kv[kix], s->pre, s->shadow, rows, v.st);
        ds41_idx_ica_row(s->pre, w.idx_wk, w.idx_knorm, m.pos, 1, s->ica[kix], pages, s->rows_kv[kix], rows, v.st);
    }
}

// Dense index-source queries are owned-row work; caches are all-row/position
// addressed. Scratch uses the original carve capacity so the stock late path
// finds L20 candidate lists at the same address after its normal compaction.
inline void ds41_v2_attn_index(::ds41::v2::State& v, Ds41V2Attention& x, Model& m, int i, const __nv_bfloat16* own_y)
{
    using namespace ds41_idx_detail;
    using ::ds41::v2::Coll;
    const int l = m.layers[i].l;
    if (!geom::is_index_source(l)) return;
    auto* s = find(m, false);
    auto* a = ds41_attn_state(m, i);
    const int kix = kvix(l), n = v.g.owned, r0 = v.g.row0;
    if (!s || kix < 0 || l > geom::CAND_SOURCE || v.g.M > s->rcap) die("DS41 v2 index: unsupported range/binding");
    const auto& w = s->lay[i];
    const bool candidates = l == geom::CAND_SOURCE && !cand_skipped(*s, kix);
    if (n) {
        ds41_idx_klen(m.pos + r0, geom::compress_ratio(l), s->klen, n, v.st);
        const auto* q = a->S.qrq + size_t(r0) * ds41a::QA;
        const auto* qs = a->S.qrs + size_t(r0) * ds41a::QA / 32;
        int rc = ds41_idx_gemm_mx_tc(q, ds41a::QA, qs, ds41a::QA / 32, w.wqb_codes, w.wqb_scales,
                                    geom::IDX_WQB_OUT, ds41a::QA, s->q_b, geom::IDX_WQB_OUT, n, v.st);
        if (rc) ds41_idx_gemm_mx(q, ds41a::QA, qs, ds41a::QA / 32, w.wqb_codes, w.wqb_scales,
                                geom::IDX_WQB_OUT, ds41a::QA, s->q_b, geom::IDX_WQB_OUT, n, v.st);
        ds41_idx_gemv(own_y, ds41a::HID, w.wproj, geom::IDX_HEADS, ds41a::HID, s->wproj, geom::IDX_HEADS, n, v.st);
        Ds41IdxStage t{};
        t.q_b = s->q_b; t.wproj = s->wproj; t.dpos = m.pos + r0; t.klen = s->klen;
        t.max_klen = max_klen(*s, kix); t.R = n;
        t.ica = s->ica[kix]; t.pages = pages_of(m, *s, r0); t.rpp = s->rows_kv[kix];
        t.scratch = s->scratch; t.Rcap = s->rcap; t.max_rows = carve_rows(*s);
        t.pipe_b = s->pipe_b; t.q_trig = s->q_trig; t.pipe_qr = s->pipe_qr && s->chunk_end > 0 && n >= 16;
        t.out = x.topk; t.ldo = geom::IDX_TOPK; t.cand_src = candidates;
        ds41_idx_topk(&t, v.st);
        if (candidates) {
            ds41_idx_cand_blocks(&t, v.st);
            G6RT_CK(cudaMemcpyAsync(x.candidates, ds41_idx_cand_list(s->scratch, s->rcap, carve_rows(*s)),
                                    size_t(n) * geom::CAND_TOPK_BLOCKS * sizeof(int32_t), cudaMemcpyDeviceToDevice, v.st));
        }
    }
    ::ds41::v2::Segment topk{x.topk, s->topk, geom::IDX_TOPK * sizeof(int32_t)};
    ::ds41::v2::all_gather(v, Coll::Index, &topk, 1, v.st);
    if (candidates) {
        ::ds41::v2::Segment cand{x.candidates, ds41_idx_cand_list(s->scratch, s->rcap, carve_rows(*s)),
                                geom::CAND_TOPK_BLOCKS * sizeof(int32_t)};
        ::ds41::v2::all_gather(v, Coll::Candidates, &cand, 1, v.st);
        s->cand_ready = true;
    }
}

inline void ds41_v2_attention(::ds41::v2::State& v, Model& m, int i, const __nv_bfloat16* own_y, __nv_bfloat16* own_out)
{
    using ::ds41::v2::Coll;
    ::ds41::v2::TimingDetailSpan timer(v,::ds41::v2::TimingStage::Attention,v.st);
    auto* x = static_cast<Ds41V2Attention*>(v.attn);
    if (!x || i < 0 || i >= m.nl || x->wob[i].empty() || v.g.M > x->rows || v.g.Q > x->q ||
        v.g.nr != g6nr::NR || v.g.rank != m.rank || m.ds41_pack || m.ds_ns != 1)
        die("DS41 v2 attention: invalid range-A chunk");
    auto* a = ds41_attn_state(m, i);
    const int l = m.layers[i].l, n = v.g.owned;
    const int mode = geom::has_compressor_gate(l) ? 2 : l == geom::CAND_SOURCE ? 1 : 0;
    ds41_v2_input_project(&a->W, &x->in, own_y, n, mode == 2, v.st);
    if (mode == 1 && n) {
        const auto* s = ds41_idx_detail::find(m, false);
        if (!s) die("DS41 v2 attention: layer20 index binding missing");
        ds41_idx_gemv(own_y, ds41a::HID, s->lay[i].cmp_wkv, ds41a::HD, ds41a::HID, x->in.ckv1, ds41a::HD, n, v.st);
    }
    ds41_v2_input_pack(&x->in, x->input_send, n, v.g.Q, mode, v.st);
    ::ds41::v2::Segment in{x->input_send, x->input_all, size_t(ds41a::v2::input_stride(mode))};
    ::ds41::v2::all_gather(v, Coll::AttnIn, &in, 1, v.st);
    ds41_v2_input_unpack(x->input_all, &x->in, v.g.M, mode, v.st);
    ds41_v2_attn_project_rest(v, *x, m, i);
    ds41_v2_attn_index(v, *x, m, i, own_y);
    if (ds41_attn_op_core(&m, i, v.g.M, v.st)) die("DS41 v2 attention: core failed");
    ds41_v2_output_a(&a->W, &a->S, v.g.M, a->heads, v.st, m.pos);
    ds41_v2_output_pack(&a->S, x->output_send, v.g.M, v.g.padded(), a->heads, v.st);
    const size_t stride = ds41a::v2::output_stride(a->heads);
    ::ds41::v2::Segment out{x->output_send, x->output_recv, stride};
    ::ds41::v2::fork(v, Coll::AttnOut);
    ::ds41::v2::all_to_all(v, Coll::AttnOut, &out, 1, v.cs);
    ::ds41::v2::record_received(v, Coll::AttnOut);
    // Local source slice has no dependency on the exchange's receive buffer.
    {
        ::ds41::v2::TimingSpan timer(v, ::ds41::v2::TimingStage::WoBOwn, v.st, m.rank);
        ds41_v2_output_b(&x->wob[i][m.rank], x->output_send + size_t(v.g.row0) * stride,
                         x->parts + size_t(m.rank) * v.g.Q * ds41a::HID, n, a->S.xf_oa, v.st);
    }
    ::ds41::v2::join(v, Coll::AttnOut);
    for (int p = 0; p < g6nr::NR; ++p) {
        if (p == m.rank) continue;
        ::ds41::v2::TimingSpan timer(v, ::ds41::v2::TimingStage::WoBRemote, v.st, p);
        ds41_v2_output_b(&x->wob[i][p], x->output_recv + size_t(p) * v.g.Q * stride,
                         x->parts + size_t(p) * v.g.Q * ds41a::HID, n, a->S.xf_oa, v.st);
    }
    ds41_v2_output_sum(x->parts, own_out, n, v.g.Q, g6nr::NR, v.st);
}
#endif
