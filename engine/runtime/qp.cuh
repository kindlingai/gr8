// G6_PF_OWN=4, pair-owner QO prefill attention (TP4 only; docs/DESIGN-QO-PAIR.md). Included by prefill.cuh in a
// G6_NR=4 build, after the prefill helpers it uses; a TP6 build compiles none of it (prefill.cuh has stubs that exit).
//
// OWN=2 (row-owner projections + index) with, on the layers of G6_PF_QP_LAYERS (default: all), the attention block of
// OWN=3 split over two head pairs {0,1}, {2,3}: rank r holds its pair's 32 heads (QpW) and runs them on W_r = O_c then
// O_r, the owner rows of its counterpart c = r ^ 2 and its own, in ONE QO call. The o_proj partial (K = 8192, BF16) of O_c
// goes to c alone on c0 (Fabric::Rs::p2p); the owner sums its own and c's partial in a fixed per-layer order
// (k_pair_add_norm) and publishes c1. c3 / c4 are OWN=2's (every row's OWN_REC; full-index IDs all-gathered), so the
// q_a_n / idx_w and IDs of O_c are already on r, and layers outside the mask run OWN=2 unchanged.
// It is included inside namespace g6rt.
#pragma once

// k_own_add_norm with TWO BF16 o_proj partials of this rank's owner rows, one per head pair: `mine` (this rank's pair,
// row j at j * 6144) and the counterpart `peer`'s, read from its slice of RS collective a.c (= 0) like k_rs_add_norm.
// lo = the partial of the pair holding head group 0 of this layer's rotation (pair_lo), hi = the other pair's; mine_lo
// says which one `mine` is. The FP32 sum is 0 + lo + hi in that fixed order (the order depends on the layer, never on
// which rank owns the row), then + resid, then the k_own_add_norm tail (normed rows -> out and the send slot of
// c + 1 = c1, then its doorbell). Row j = owner row pf_r0(M, rank) + j. grid (RN_CL, max(rows, 1)), block RN_T.
__global__ void __cluster_dims__(glue::RN_CL, 1, 1) __launch_bounds__(glue::RN_T)
k_pair_add_norm(RsNormArgs a, const __nv_bfloat16* mine, int peer, bool mine_lo)
{
    const u64 step = *a.stepp;
    const int j = blockIdx.y;
    if (j < pf_rows(a.M, a.f.rank, a.f.nr))
    {
        const int m = pf_r0(a.M, a.f.rank, a.f.nr) + j;
        const int col0 = blockIdx.x * glue::RN_COLS + threadIdx.x * 8;
        const size_t off = (size_t)j * HID + col0;
        const __nv_bfloat16* other = reinterpret_cast<const __nv_bfloat16*>(rs_slice(a.f, step, a.c, peer, a.loop));
        const __nv_bfloat16* lo = mine_lo ? mine : other;
        const __nv_bfloat16* hi = mine_lo ? other : mine;
        float x[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float b[8];
        glue::bf16x8_to_f(__ldcv(reinterpret_cast<const uint4*>(lo + off)), b);
        #pragma unroll
        for (int k = 0; k < 8; ++k) x[k] += b[k];
        glue::bf16x8_to_f(__ldcv(reinterpret_cast<const uint4*>(hi + off)), b);
        #pragma unroll
        for (int k = 0; k < 8; ++k) x[k] += b[k];
        float4* r = reinterpret_cast<float4*>(a.resid + (size_t)m * HID + col0);
        const float4 r0 = r[0], r1 = r[1];
        x[0] += r0.x; x[1] += r0.y; x[2] += r0.z; x[3] += r0.w;
        x[4] += r1.x; x[5] += r1.y; x[6] += r1.z; x[7] += r1.w;
        r[0] = make_float4(x[0], x[1], x[2], x[3]);
        r[1] = make_float4(x[4], x[5], x[6], x[7]);
        __half* ag = reinterpret_cast<__half*>(glue::send_slot(a.f, step, a.c + 1)) + (size_t)j * HID;
        glue::cluster_rms_store(x, a.w, a.eps, a.out + (size_t)m * HID, col0, ag);
    }
    glue::publish(a.f, a.c + 1, g6::coll_seq(step, a.c + 1, a.f.ncoll), a.ticket);
}
// the head pair holding head group 0 of layer l (NR=4 rotation: rank m holds group (m - l) mod 4, so group 0 is on rank
// l mod 4, pair (l mod 4) / 2): its partial is k_pair_add_norm's lo
__host__ __device__ inline int pair_lo(int l) { return (l & 3) >> 1; }
// grid 1, block 32: peer p's slice of RS collective c for this rank has landed (k_wait_rs for one peer)
__global__ void __launch_bounds__(32) k_wait_rs1(g6::FabricDev f, const u64* stepp, int c, int M, bool loop, int p)
{
    if (threadIdx.x != 0) return;
    const u64 step = *stepp;
    const u64 seq = g6::coll_seq(step, c, f.ncoll);
    const u64* tail = reinterpret_cast<const u64*>(rs_slice(f, step, c, p, loop) + (size_t)pf_rows(M, f.rank, f.nr) * HID * 2);
    const long long t0 = clock64();
    unsigned it = 0;
    while (glue::ld_acquire_sys(tail) != seq)
        if ((++it & 1023u) == 0 && (*reinterpret_cast<volatile unsigned*>(f.error) != 0 || clock64() - t0 > G6_SPIN_TIMEOUT_CYC))
        {
            atomicCAS(f.error, 0u, glue::ERR_AR | ((unsigned)c << 8) | (unsigned)p);
            break;
        }
}
// n rows [n][6144] BF16 -> slice p of the send slot of RS collective c (the rows of rank p, at p * ss), then the
// doorbell (the proxy sends that slice to p only, Fabric::Rs::p2p). grid PF_PUT, block 128.
__global__ void __launch_bounds__(128) k_put_slice(const uint4* src, g6::FabricDev f, const u64* stepp, int c, int p, int n,
                                                   unsigned* ticket)
{
    const u64 step = *stepp;
    uint4* dst = reinterpret_cast<uint4*>(glue::send_slot(f, step, c) + p * pf_ss(f));
    const int n16 = n * (HID * 2 / 16);
    for (int i = blockIdx.x * 128 + threadIdx.x; i < n16; i += gridDim.x * 128) dst[i] = src[i];
    glue::publish(f, c, g6::coll_seq(step, c, f.ncoll), ticket);
}

// G6_PF_QP_LAYERS = comma list of model layers and ranges a-b (e.g. "0-20,40"); unset or "all" = every layer below 78.
// Per position of `layers` (the model's layer numbers): 1 = pair-owner attention, 0 = OWN=2.
inline std::vector<char> pf_qp_mask(const std::vector<int>& layers)
{
    std::vector<char> on(layers.size(), 0);
    const char* e = getenv("G6_PF_QP_LAYERS");
    const std::string v = e && *e ? e : "all";
    std::vector<char> want(78, 0);
    if (v == "all") std::fill(want.begin(), want.end(), 1);
    else
        for (size_t k = 0; k < v.size();)
        {
            char* end = nullptr;
            const long a = strtol(v.c_str() + k, &end, 10);
            long b = a;
            if (end == v.c_str() + k) die("G6_PF_QP_LAYERS: bad list " + v);
            if (*end == '-') { const char* q = end + 1; b = strtol(q, &end, 10); if (end == q) die("G6_PF_QP_LAYERS: bad list " + v); }
            if (a < 0 || b < a || b > 77) die("G6_PF_QP_LAYERS: layers must be 0..77 (the MTP layer keeps the head-split path): " + v);
            for (long l = a; l <= b; ++l) want[l] = 1;
            k = end - v.c_str();
            if (k < v.size() && v[k] != ',') die("G6_PF_QP_LAYERS: bad list " + v);
            if (k < v.size()) ++k;
        }
    for (size_t i = 0; i < layers.size(); ++i) on[i] = layers[i] >= 0 && layers[i] < 78 && want[layers[i]];
    return on;
}

// build_prefill, after the common buffers: the pair scratch of this lane (W rows: O_c then O_r)
inline void pf_qp_alloc(Prefill& P, Model& m, int Mmax, const g6::FabricDev& f, Prefill* share)
{
    if (!P.own || !P.ownix || P.qo || f.nr != 4 || !P.qpw || (int)P.qpw->size() != m.nl || (int)P.qpl.size() != m.nl)
        die("G6_PF_OWN=4 needs G6_PF_RS=1, --pfv2, four ranks and the pair weights of every masked layer");
    if (getenv("G6_PF_OWNPERM") && atoi(getenv("G6_PF_OWNPERM")) == 1) die("G6_PF_OWNPERM is OWN=3 only");
    P.qp_rows = 2 * ((Mmax + f.nr - 1) / f.nr + 16);
    std::string ee;
    if ((P.qp_early = env_choice("G6_PF_QP_EARLY", 2, 0, &ee)) < 0) die(ee);
    if (P.qp_early && f.rank == 0 && !share) fprintf(stderr, "# G6_PF_QP_EARLY=%d: O_c's partial goes out before O_r's %s\n",
                                                     P.qp_early, P.qp_early == 2 ? "attention" : "o_proj");
    const size_t R = (size_t)P.qp_rows;
    P.qp_qb = dmalloc<__half>(R * 8192);
    P.qp_out = dmalloc<__half>(R * 8192);
    P.qp_x8 = dmalloc<uint8_t>((R + 15) / 16 * (8192 / 32) * bindp::REC);
    P.qp_part = dmalloc<__nv_bfloat16>(R * HID);
    P.qp_pos = dmalloc<int>(R);
    P.qp_ids = dmalloc<int>(R * 2048);
    // two-stream lanes (pf_step2s): pf_attn_block_qp writes q_b / positions / IDs before wait_attn and reads qp_out after
    // used_attn, which is safe only because this scratch is per lane (only qa / av / the binding are shared)
    P.lane = share ? 1 : 0;
    P.qp_built = true;
}
// build_prefill, after the layer tensors: the 32-head QO states (one per masked layer, or one per roll slot)
inline void pf_qp_states(Prefill& P, Model& m, cudaStream_t st, Prefill* share)
{
    if (share)
    {
        if (P.roll != share->roll) die("G6_PF_QP_ROLL: both lanes must share one roll");
        P.qp_ctx = share->qp_ctx;
        P.qa = share->qa;
        return;
    }
    if (P.roll)              // rolling pack: S states (one shared workspace), layer position i -> state i % S
    {
        QpRoll& R = *P.roll;
        const char* cl = getenv("G6_PF_QP_CTX");     // key-table / RoPE capacity of the QO workspace (default: the cap)
        const int qcap = cl && *cl ? atoi(cl) : 0;
        if (cl && *cl && qcap <= 0) die("G6_PF_QP_CTX must be a positive position count");
        g6_attn_qo_cap(qcap);
        int i0 = -1;
        for (int i = 0; i < m.nl && i0 < 0; ++i) if (P.qpl[i]) i0 = i;
        if (i0 < 0) die("G6_PF_QP_ROLL: no pair-owner layer in G6_PF_QP_LAYERS");
        R.qa.assign(R.S, nullptr);
        size_t b = 0;
        for (int s = 0; s < R.S; ++s)
        {
            if (!qp_check(R.slot[s])) die("G6_PF_QP_ROLL: slot " + std::to_string(s) + " shapes");
            if (int e = g6_attn_qo_create(&R.qa[s], &m.layers[i0].ab, &R.slot[s].uk, &R.slot[s].uv, P.qp_rows, st, s ? R.qa[0] : nullptr))
                die("rank " + std::to_string(m.rank) + ": g6_attn_qo_create (pair, roll slot " + std::to_string(s) + ") failed (" + std::to_string(e) + ")");
            b += g6_attn_qo_bytes(R.qa[s]);
        }
        g6_attn_qo_cap(0);
        P.qp_ctx = qcap > 0 ? std::min(qcap, m.layers[i0].ab.capacity) : m.layers[i0].ab.capacity;
        if (R.wire) R.wire->ctx = P.qp_ctx;          // part of the ready tag (qp_wire.h tag): both members' tables must agree
        G6RT_CK(cudaStreamSynchronize(st));
        P.qa.assign(m.nl, nullptr);
        for (int i = 0; i < m.nl; ++i) if (P.qpl[i]) P.qa[i] = R.qa[i % R.S];
        const size_t sb = (size_t)R.S * ((size_t)8192 * 2048 + (size_t)8192 * 64 + (size_t)HID * 8192 + (size_t)HID * 256 +
                                         (size_t)QP_HEADS * (192 + 256) * 512 * 2);
        if (m.rank == 0) fprintf(stderr, "# rank %d: G6_PF_QP_ROLL=%d rolling pair pack: %d slots %.3f GB + QO states/workspace %.3f GB (QO cap %d)\n",
                                 m.rank, R.S, R.S, sb / 1e9, b / 1e9, P.qp_ctx);
        return;
    }
    size_t b = 0;
    int n = 0;
    G6AttnQo* owner = nullptr;
    P.qa.assign(m.nl, nullptr);
    for (int i = 0; i < m.nl; ++i)
    {
        if (!P.qpl[i]) continue;
        QpW& w = (*P.qpw)[i];
        if (!qp_check(w)) die("G6_PF_OWN=4: layer " + std::to_string(m.layers[i].l) + " has no pair weights");
        if (int e = g6_attn_qo_create(&P.qa[i], &m.layers[i].ab, &w.uk, &w.uv, P.qp_rows, st, owner))
            die("rank " + std::to_string(m.rank) + " layer " + std::to_string(m.layers[i].l) + ": g6_attn_qo_create (pair) failed (" + std::to_string(e) + ")");
        if (!owner) owner = P.qa[i];
        G6RT_CK(cudaStreamSynchronize(st));      // the FP16 copies are made: the BF16 pair UK / UV can go
        qp_free_kvb(w);
        b += g6_attn_qo_bytes(P.qa[i]);
        ++n;
    }
    P.qp_ctx = m.layers[0].ab.capacity;
    if (m.rank == 0) fprintf(stderr, "# rank %d: pair-owner QO attention states (G6_PF_OWN=4, 32 heads, kernel %s) for %d of %d layers: %.3f GB\n",
                             m.rank, getenv("G6_PF_QO") && *getenv("G6_PF_QO") ? getenv("G6_PF_QO") : "b", n, m.nl, b / 1e9);
}
// G6_PF_QP_WARM=1 (default 0 = none): the QO absorb + UV of the pair states at every W row count of an Mmax sub-batch
// over the ranks (nc + no; with G6_PF_QP_EARLY=2 also nc and no alone), then this rank's two q_b GEMMs (O_r's and O_c's
// aligned bindings) and its o_proj (W, or the two halves with EARLY >= 1) on the first pair layer's weights, on scratch;
// then a sync. cuBLAS's first call per (handle, shape) costs ~0.1-0.4 s on the host, paid by the first request otherwise.
inline void pf_qp_warm(Model& m, Prefill& P, cudaStream_t st)
{
    const int nr = P.f.nr, me = P.f.rank, M = P.Mmax;
    int il = -1;
    for (int i = 0; i < m.nl && il < 0; ++i) if (P.qpl[i] && P.qa[i]) il = i;
    if (il < 0) return;
    std::vector<int> rows;
    for (int r = 0; r < nr; ++r)
    {
        const int c = r ^ 2, no = pf_rows(M, r, nr), nc = pf_rows(M, c, nr);
        rows.push_back(nc + no);
        if (P.qp_early == 2) { rows.push_back(nc); rows.push_back(no); }
    }
    std::sort(rows.begin(), rows.end()); rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    for (int n : rows)
        if (n > 0 && n <= P.qp_rows)
            if (int e = g6_attn_qo_warm(P.qa[il], n, st)) die("g6_attn_qo_warm (pair) failed (" + std::to_string(e) + ")");
    const QpW& w = (*P.qpw)[il];
    const int cp = me ^ 2, r0 = pf_r0(M, me, nr), r1 = pf_r0(M, me + 1, nr), c0r = pf_r0(M, cp, nr), c1r = pf_r0(M, cp + 1, nr);
    const int nc = c1r - c0r, no = r1 - r0;
    auto gemm = [&](const bind::Tensor& c, const bind::Tensor& sc, int n, const uint8_t* x8, void* y, int bf16, const char* what) {
        if (n <= 0) return;
        bindp::Gemm g;
        if (const char* e = bindp::make_gemm(c, sc, n, x8, y, bf16, g)) die(std::string("pair warm ") + what + ": " + e);
        bindp::launch(g, st);
    };
    gemm(w.qb_c, w.qb_s, r1 - (r0 & ~15), P.sc.x8_small, P.qp_qb, 0, "q_b (O_r)");
    gemm(w.qb_c, w.qb_s, c1r - (c0r & ~15), P.sc.x8_small, P.qp_qb, 0, "q_b (O_c)");
    if (P.qp_early) { gemm(w.o_c, w.o_s, nc, P.qp_x8, P.qp_part, 1, "o_proj (O_c)"); gemm(w.o_c, w.o_s, no, P.qp_x8, P.qp_part, 1, "o_proj (O_r)"); }
    else gemm(w.o_c, w.o_s, nc + no, P.qp_x8, P.qp_part, 1, "o_proj (W)");
    G6RT_CK(cudaStreamSynchronize(st));
    if (m.rank == 0) fprintf(stderr, "# rank %d: G6_PF_OWN=4 warm-up (G6_PF_QP_WARM=1): absorb + UV at %zu row counts (%d..%d), q_b + o_proj\n",
                             m.rank, rows.size(), rows.front(), rows.back());
}
// G6_PF_QP_MIN (default 768): OWN=4 pays a fixed ~0.13-0.35 s per request (the pair weights rolled through every layer),
// so below ~770 prompt tokens OWN=2 prefills faster (TP4: 256 tokens 506 vs 401 tok/s, 1024 tokens 809 vs 846)
inline int pf_qp_min()
{
    static const int v = getenv("G6_PF_QP_MIN") ? atoi(getenv("G6_PF_QP_MIN")) : 768;
    return v;
}
// per request: a request whose prompt has at least pf_qp_min() tokens and whose prompt + gen fits the QO key / RoPE table
// (P.qp_ctx: G6_PF_QP_CTX with G6_PF_QP_ROLL, else the cap) runs ALL pair layers OWN=4, else ALL layers OWN=2. No
// per-layer mix; both lanes the same. Call before the request's first pf_inputs / pf_enqueue (pf_enqueue and pf_attn_block
// read P.qp). Returns the mode (true = OWN=4).
inline bool pf_qp_select(Prefill& P, Prefill* P2, int ctx, int prompt)
{
    if (P2 && P2->qp_built != P.qp_built) die("pf_qp_select: the two lanes were built with different G6_PF_OWN");
    const bool on = P.qp_built && ctx <= P.qp_ctx && prompt >= pf_qp_min();
    P.qp = on;
    if (P2) P2->qp = on;
    return on;
}
// layer i's pair-owner attention block (DESIGN-QO-PAIR section 2 steps 4-5) on W = O_c then O_r (c = rank ^ 2) for this
// rank's pair's 32 heads, as ONE QO call (one key-table prepass): q_b of each half (16-row aligned bindings; O_r's GEMM
// first, then O_c's, which overwrites O_r's alignment rows so W is contiguous in qp_qb), positions / IDs of W copied
// together, QO core, UV, o_proj (K = 8192, BF16) of W; O_c's partial -> c alone on c0. The pair norm of O_r
// (k_pair_add_norm, c1 publish) runs in the c0 consume (pf_qp_norm), after c's partial of O_r has landed, so a loopback
// driver can deliver c0 between the two. Layer 0 also projects every row and scores every row's top-2048 into idx_all
// (later full-index layers got theirs from pf_ix's c4 gather). wait_attn / used_attn as in pf_attn_block.
inline void pf_attn_block_qp(Model& m, Prefill& P, int i, int M, int context, cudaStream_t st, cudaEvent_t wait_attn,
                             cudaEvent_t used_attn)
{
    Layer& y = m.layers[i];
    const bindp::Layer& b = pf_layer(m, P, i, M);
    const int nr = P.f.nr, me = P.f.rank, cp = me ^ 2;
    const int r0 = pf_r0(M, me, nr), r1 = pf_r0(M, me + 1, nr), c0r = pf_r0(M, cp, nr), c1r = pf_r0(M, cp + 1, nr);
    const int nc = c1r - c0r, no = r1 - r0, W = nc + no;
    const QpW& w = (*P.qpw)[i];       // G6_PF_QP_ROLL: slot i % S (static pointers), filled by qp_roll_acquire
    // dense trellis q_b / o_proj (kernels/dtr.cuh): the pair codes are FP8 tile copies. The pair's suh rotates the inputs
    // (q_b: the shared full-width vector), its svh the outputs (o_proj: the shared full-width vector).
    const bindp::Tensors& T = P.T[i];
    const bool trq = T.tsuh[0] != nullptr, tro = T.tsuh[1] != nullptr;   // each may be trellis alone (layer 3's int4 o_proj)
    if ((trq || tro) && !w.qb_svh) die("G6_PF_OWN=4: layer " + std::to_string(y.l) + " is dense trellis but its pair weights have no vectors");
    pf_mark(7, st);
    if (P.roll) { qp_roll_acquire(*P.roll, P.qpl, m.nl, i, st); pf_mark(7, st); }   // exposed staging lands in "other"
    if (i == 0)                  // later layers: every row's q_a_n / kv_a / idx_k / idx_w came with c3 (OWN=2 records)
    {
        bindp::launch(b.xq_x, st);
        bindp::launch_pair(b.q_a, b.kv_a, st);
        pf_blas_x(m, P, i, M, st);
        glue::k_rmsnorm_rows<__half><<<M, 256, 0, st>>>(P.q_a, QA, QA, y.q_a_norm, glue::RMS_EPS, P.q_a_n, QA);
        if (y.indexer) { bindp::launch(b.xq_qa, st); bindp::launch(b.idx_q, st); }
    }
    // W = qp_qb rows [S - nc, S + no): O_r's GEMM (rows a0r..r1) at S - (r0 - a0r), then O_c's (rows a0c..c1r) at
    // S - nc - (c0r - a0c), which rewrites whatever of O_r's alignment rows lies inside W. S >= both offsets' needs.
    const int a0c = c0r & ~15, a0r = r0 & ~15, S = std::max(c1r - a0c, r0 - a0r), base = S - nc;
    auto qb = [&](int a0, int e, __half* y0) {
        if (e <= a0) return;
        const bindp::Layer& bq = pf_layer_rows(m, P, i, a0, e - a0);
        bindp::Xq xq = bq.xq_qa;
        xq.suh = trq ? T.tsuh[0] : nullptr;
        bindp::launch(xq, st);
        bindp::Gemm g;
        if (const char* er = bindp::make_gemm(w.qb_c, w.qb_s, e - a0, xq.dst, y0, 0, g)) die(std::string("pair q_b: ") + er);
        g.osvh = trq ? w.qb_svh : nullptr;
        bindp::launch(g, st);
    };
    if (no > 0) qb(a0r, r1, P.qp_qb + (size_t)(S - (r0 - a0r)) * 8192);
    if (nc > 0) qb(a0c, c1r, P.qp_qb + (size_t)(S - (c1r - a0c)) * 8192);
    if (W > 0)
    {
        G6RT_CK(cudaMemcpyAsync(P.qp_pos, P.pos + c0r, (size_t)nc * 4, cudaMemcpyDeviceToDevice, st));
        G6RT_CK(cudaMemcpyAsync(P.qp_pos + nc, P.pos + r0, (size_t)no * 4, cudaMemcpyDeviceToDevice, st));
    }
    pf_mark(0, st);
    if (wait_attn) G6RT_CK(cudaStreamWaitEvent(st, wait_attn, 0));
    for (int c0 = 0; c0 < M; c0 += P.attn_rows)          // every row's latent KV (+ index keys on layer 0) into the caches
    {
        const int r = std::min(P.attn_rows, M - c0);
        G6Attn a = y.ab.args;
        a.rows = r; a.context = context - (M - c0 - r);
        a.kv_a = P.kv_a + (size_t)c0 * KVA; a.positions = P.pos + c0;
        a.reuse_indices = P.idx_all;                        // non-null: latent KV only
        G6RT_CK(g6_attn_append(&a, st));
        if (y.indexer && i == 0)
        {
            G6AttnStep s{};
            s.rows = r; s.context = a.context; s.positions = P.pos + c0; s.index_k = P.idx_k + (size_t)c0 * IDXK;
            G6RT_CK(g6_attn_bind_index_append(&y.ab, &s, st));
        }
    }
    if (y.indexer && i == 0)     // every row's top-2048 into idx_all (reuse layers of source 0 read any row)
        for (int c0 = 0; c0 < M; c0 += P.attn_rows)
        {
            const int r = std::min(P.attn_rows, M - c0);
            G6AttnStep s{};
            s.rows = r; s.context = context - (M - c0 - r); s.positions = P.pos + c0;
            s.index_q = P.idx_q + (size_t)c0 * IDXQ; s.index_weight = P.idx_w + (size_t)c0 * IDXW;
            G6RT_CK(g6_attn_bind_prefill_index_rows(P.av[i], &y.ab, &s, P.idx_all + (size_t)c0 * 2048, nullptr, st));
        }
    if (W > 0)
    {
        G6RT_CK(cudaMemcpyAsync(P.qp_ids, P.idx_all + (size_t)c0r * 2048, (size_t)nc * 2048 * 4, cudaMemcpyDeviceToDevice, st));
        G6RT_CK(cudaMemcpyAsync(P.qp_ids + (size_t)nc * 2048, P.idx_all + (size_t)r0 * 2048, (size_t)no * 2048 * 4, cudaMemcpyDeviceToDevice, st));
    }
    pf_mark(y.indexer ? 8 : 1, st);
    // QO over rows [w0, w0 + n) of W (context = the last of those rows' positions + 1), into qp_out at row w0
    auto qo = [&](int w0, int n, int last) {
        if (n <= 0) return;
        G6AttnQoStep s{};
        s.rows = n; s.context = context - (M - last);
        s.q_b = P.qp_qb + (size_t)(base + w0) * 8192; s.ld_qb = 8192;
        s.positions = P.qp_pos + w0; s.ids = P.qp_ids + (size_t)w0 * 2048; s.out = P.qp_out + (size_t)w0 * 8192;
        if (int e = g6_attn_qo_run(P.qa[i], &y.ab, &s, st)) die("g6_attn_qo_run (pair) failed (" + std::to_string(e) + ")");
    };
    // xq + o_proj of rows [w0, w0 + n) of W: records at tile t0 (each part quantized on its own, so no tile straddles the
    // O_c / O_r boundary; xq is row-local, so the records equal the one-part ones), partials into qp_part at row w0
    auto oproj = [&](int w0, int n, int t0) {
        if (n <= 0) return;
        uint8_t* x8 = P.qp_x8 + (size_t)t0 * (8192 / 32) * bindp::REC;
        bindp::Xq xq;
        xq.on = 1; xq.src = P.qp_out + (size_t)w0 * 8192; xq.ld = 8192; xq.K = 8192; xq.M = n; xq.dst = x8;
        xq.suh = tro ? w.o_suh : nullptr;
        bindp::launch(xq, st);
        bindp::Gemm op;
        if (const char* er = bindp::make_gemm(w.o_c, w.o_s, n, x8, P.qp_part + (size_t)w0 * HID, 1, op)) die(std::string("pair o_proj: ") + er);
        op.osvh = tro ? T.tsvh[1] : nullptr;
        bindp::launch(op, st);
    };
    auto put = [&] { k_put_slice<<<PF_PUT, 128, 0, st>>>(reinterpret_cast<const uint4*>(P.qp_part), P.f, P.stepp, 0, cp, nc, P.tickets + 0); };
    const int tc = (nc + 15) / 16;                      // O_r's records start after O_c's tiles (qp_x8 holds (qp_rows + 15) / 16)
    if (P.qp_early == 0)
    {
        qo(0, W, std::max(r1, c1r));
        pf_adump(P, i, "attn", P.qp_out, W, 8192 * 2, st);
        pf_mark(y.indexer ? 8 : 1, st);
        if (W > 0)
        {
            bindp::Xq xq;
            xq.on = 1; xq.src = P.qp_out; xq.ld = 8192; xq.K = 8192; xq.M = W; xq.dst = P.qp_x8;
            xq.suh = tro ? w.o_suh : nullptr;
            bindp::launch(xq, st);
        }
        if (used_attn) G6RT_CK(cudaEventRecord(used_attn, st));   // after xq read qp_out, as pf_attn_block after xq_attn
        if (W > 0)
        {
            bindp::Gemm op;
            if (const char* er = bindp::make_gemm(w.o_c, w.o_s, W, P.qp_x8, P.qp_part, 1, op)) die(std::string("pair o_proj: ") + er);
            op.osvh = tro ? T.tsvh[1] : nullptr;
            bindp::launch(op, st);
        }
        pf_adump(P, i, "oproj", P.qp_part, W, HID * 2, st);
        pf_mark(2, st);
        put();
    }
    else
    {
        if (P.qp_early == 2) { qo(0, nc, c1r); oproj(0, nc, 0); put(); qo(nc, no, r1); }
        else { qo(0, W, std::max(r1, c1r)); oproj(0, nc, 0); put(); }
        pf_adump(P, i, "attn", P.qp_out, W, 8192 * 2, st);
        pf_mark(y.indexer ? 8 : 1, st);
        oproj(nc, no, tc);
        if (used_attn) G6RT_CK(cudaEventRecord(used_attn, st));   // every QO call done and qp_out read
        pf_adump(P, i, "oproj", P.qp_part, W, HID * 2, st);
        pf_mark(2, st);
    }
    if (P.roll) qp_roll_release(*P.roll, P.lane, i, st);   // this lane read the slot's q_b / o_proj / UK / UV for the last time
    P.qp_pend = y.l;
    P.qp_mine = P.qp_part + (size_t)nc * HID;
    pf_mark(3, st);
}
// the c0 consume after pf_attn_block_qp: the counterpart's partial of the owner rows, then the pair norm and c1 publish
inline void pf_qp_norm(Prefill& P, int M, const __nv_bfloat16* w, __half* out, cudaStream_t st)
{
    const int cp = P.f.rank ^ 2, rows = pf_rows(M, P.f.rank, P.f.nr);
    k_wait_rs1<<<1, 32, 0, st>>>(P.f, P.stepp, 0, M, P.loop, cp);
    RsNormArgs a{P.f, P.stepp, 0, M, P.loop, P.resid, w, out, glue::RMS_EPS, P.tickets + 1};
    k_pair_add_norm<<<dim3(glue::RN_CL, rows > 0 ? rows : 1), glue::RN_T, 0, st>>>(a, P.qp_mine, cp, (P.f.rank >> 1) == pair_lo(P.qp_pend));
    P.qp_pend = -1;
}
