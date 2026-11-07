// Flash DSA block (W5): NoPE MLA + kpool indexer on one rank, for the Flash main model (runtime/flash_step.cuh).
// Included by rank.cuh inside namespace g6rt, after the GLM helpers (attn_step, dmark, put_publish) and before
// flash_step.cuh. G6_MODEL_FLASH only; the attention itself is attn/flash_attn.cu + flash_bind.cu behind the GLM
// attention ABI (bind.h: G6AttnPack::index_ape, G6AttnStep::index_gate).
//
// Every projection is a BF16W .gv GEMV (lossless release, FLASH.md) on BF16 fragment tables:
//   on sc.xb_x (BF16 of m.xn, written by flash_step.cuh launch_mhc_input):
//     q_a -> m.q_a [M][1536], kv_a -> m.kv_a [M][512], idx.wk -> m.idx_k [M][128], idx.weights_proj -> m.idx_w [M][32],
//     index_kpool_compress_gate -> m.idx_g [M][128]                                         (FP16 outputs)
//   q_a_norm (RMSNorm, eps geom::RMS_EPS_FLASH) -> m.q_a_n; its BF16 table into sc.xf_qa (the Flash build has no
//   FP16 consumer of that table):
//     q_b (own heads) -> m.q_b [M][h*256], idx.wq_b -> m.idx_q [M][32*128]
//   attention -> y.ab.args.output FP16 [M][h][256]; its BF16 table into sc.xf_attn:
//     o_proj (own head columns) -> sc.part[0] BF16 [M][HID], then put_publish(2i) (the block publishes itself)
//
//   bind_dsa_layer(m, i, find, capacity, st)   build: the layer's attention binding (caches, scratch, error word), norms and
//                                              the GEMV bindings for every M = 1..MX (needs m.sc and the step buffers)
//   prepare_dsa_layer(m, i, M, st)             outside capture: graph form, the attention replay entries for M rows
//   launch_dsa_block(m, i, M, context, st)     the step: projections, attention, o_proj, publish of collective 2i
//                                              (S2 with several sessions: s2_attention on the layer's batch view, s2.cuh;
//                                              past R with GR8_CP: launch_dsa_cp on the CP fabric)
#pragma once
#if G6_MODEL_FLASH

#ifdef DEC_SLOT
inline dslot::Slot make_flash_slot(const Model& m, int c)
{
    dslot::Slot q;
    q.send = m.f.send; q.bell = m.f.doorbell; q.stepp = m.stepp; q.half = g6::fab_slot_base(m.f, m.f.ncoll); q.off = g6::fab_slot_base(m.f, c);
    q.ncoll = m.f.ncoll; q.c = c; q.ticket = m.tickets + c; q.xt = m.f.xt;
    return q;
}
#endif

inline const char* make_dsa(Model& m, int i, int M, const Find& find, bind::Layer& L);

inline void bind_dsa_layer(Model& m, int i, const Find& find, int capacity, cudaStream_t st)
{
    Layer& y = m.layers[i];
    if (geom::is_kda(y.l)) die("bind_dsa_layer: layer " + std::to_string(y.l) + " is a KDA layer");
    y.indexer = true; y.index_src = -1;
    auto at = [&](const char* suffix) {
        G6AttnTensor t{};
        const std::string name = lname(y.l, suffix);
        const TensorRef r = find(name);
        if (!r.p || !r.bf16 || r.nd > 3) die("Flash DSA: missing or non-BF16 tensor " + name);
        t.data = r.p; t.ndim = r.nd;
        for (int d = 0; d < r.nd; ++d) t.shape[d] = r.shape[d];
        return t;
    };
    y.ap = G6AttnPack{at("kv_b.uk"), at("kv_b.uv"), at("kv_a_norm"), at("idx.k_norm.w"), at("idx.k_norm.b")};
    y.ap.index_ape = at("self_attn.indexer.index_kpool_compress_ape");
    // decode only (build_flash: max_rows <= MX); the GLM attention scratch sharing (GR8_DEC_SHARED_SCRATCH) is not wired here
    // GR8_GLM53F_KV_CHUNKS=1: the MLA and index caches in kvchunks regions under the model's chunk map (flash_kv.h), every
    // chunk online at boot (FLASH_CAP_POLICY=protected: the capacity is never lent); 0 (default): one cudaMalloc
    const int chunked = flashkv::env_mode();
    if (chunked < 0) die("GR8_GLM53F_KV_CHUNKS is 0 or 1");
#if defined(G6_CP) && G6_CP
    if (chunked && m.cp.on) die("GR8_GLM53F_KV_CHUNKS=1 is not with GR8_CP");
#endif
    int e = 0;
    if (chunked)
    {
        const uint32_t pages = (uint32_t)((capacity + g6kv::PAGE - 1) / g6kv::PAGE);
        int dev = 0;
        G6RT_CK(cudaGetDevice(&dev));
        if (!m.fkv.pages && m.fkv.init(pages, UINT32_MAX, dev)) die("GR8_GLM53F_KV_CHUNKS: " + m.fkv.err);
        if (m.fkv.pages != pages) die("GR8_GLM53F_KV_CHUNKS: the DSA layers' capacities differ");
        G6KvChunks *mla = nullptr, *idx = nullptr;
        if (m.fkv.add(&mla, &idx)) die("GR8_GLM53F_KV_CHUNKS layer " + std::to_string(y.l) + ": " + m.fkv.err);
        e = g6_attn_bind_create_chunked(&y.ab, &y.ap, m.rank, y.l, capacity, MX, ATT_QT, st, mla, idx);
    }
#if defined(G6_CP) && G6_CP
    // GR8_CP: the replicated prefix and this rank's pages past R (flash_bind.cu)
    else if (m.cp.on)
    {
        if (m.cp.colls.q.size() != (size_t)m.nl || m.cp.colls.q[i] < 0 || m.cp.colls.cand[i] < 0)
            die("GR8_CP: the CP collectives do not match the Flash layer list");
        e = g6_attn_bind_create_cp(&y.ab, &y.ap, m.rank, y.l, capacity, MX, ATT_QT, m.cp.R, g6nr::NR, st, nullptr, 0);
    }
#endif
    else e = g6_attn_bind_create_ext(&y.ab, &y.ap, m.rank, y.l, capacity, MX, ATT_QT, st, nullptr, 0);
    if (e) die("Flash DSA layer " + std::to_string(y.l) + ": attention binding failed (" + std::to_string(e) + ")");
    y.q_a_norm = (const __nv_bfloat16*)need(find, lname(y.l, "q_a_norm"));
    const TensorRef qn = find(lname(y.l, "q_a_norm"));
    if (!qn.bf16 || qn.nd != 1 || qn.shape[0] != QA) die("Flash DSA: q_a_norm must be BF16 [1536]");
    for (int M = 1; M <= MX; ++M)
    {
        if (const char* e = make_dsa(m, i, M, find, y.L[M]))
            die("rank " + std::to_string(m.rank) + " layer " + std::to_string(y.l) + ": " + e);
#ifdef DEC_SLOT
        bind::set_slots(y.L[M], make_flash_slot(m, 2 * i), make_flash_slot(m, 2 * i + 1));
#endif
        y.built[M] = true;
    }
}

// the layer's GEMVs (W8 MXFP8 + trellis, or lossless BF16W) for M rows (host-side argument blocks only)
inline const char* make_dsa(Model& m, int i, int M, const Find& find, bind::Layer& L)
{
    Layer& y = m.layers[i];
    L = bind::Layer();
    L.M = M; L.indexer = 1;
    auto T = [&](const char* s) { return as_bind(find(lname(y.l, s))); };
    const bind::Scratch& sc = m.sc;
    const char* e = nullptr;
    using bind::BF16W;
    using bind::MXFP8;
    const int h = y.ab.args.heads;
    dtr::Mat qbm, opm;
    if ((e = dtr::make_mat(T("q_b.trellis"), T("q_b.suh"), T("q_b.svh"), dtr::SIDE_OUT, qbm))) return e;
    if ((e = dtr::make_mat(T("o_proj.trellis"), T("o_proj.suh"), T("o_proj.svh"), dtr::SIDE_IN, opm))) return e;
    const bool qa_mx = T("q_a.mx.gv").p != nullptr, kva_mx = T("kv_a.mx.gv").p != nullptr;
    const bool wk_mx = T("idx.wk.mx.gv").p != nullptr, wqb_mx = T("idx.wq_b.mx.gv").p != nullptr;
    const bool xin_ord = knob::glm53f_dsa_xin_order();
    const bool qb_rot  = knob::glm53f_dsa_qb_rot();
    const bool op_rot  = knob::glm53f_dsa_oproj_rot().value_or(M > 1);
    const int iq_tpc   = (int)knob::glm53f_dsa_idxq_tpc();
    //                       tensor                                                 codec  tpc ks  M  fragments  output          dtype
    if ((e = bind::make_gv(T(qa_mx ? "q_a.mx.gv" : "q_a.gv"),                       qa_mx ? MXFP8 : BF16W,  2, 16, M, qa_mx ? sc.xf_x : sc.xb_x,  m.q_a,   gv::O_F16, L.q_a))) return e;
    if ((e = bind::make_gv(T(kva_mx ? "kv_a.mx.gv" : "kv_a.gv"),                    kva_mx ? MXFP8 : BF16W, 1, 32, M, kva_mx ? sc.xf_x : sc.xb_x, m.kv_a,  gv::O_F16, L.kv_a))) return e;
    if ((e = bind::make_gv(T(wk_mx ? "idx.wk.mx.gv" : "idx.wk.gv"),                 wk_mx ? MXFP8 : BF16W,  1, 32, M, wk_mx ? sc.xf_x : sc.xb_x,  m.idx_k, gv::O_F16, L.idx_k))) return e;
    if ((e = bind::make_gv(T("idx.weights_proj.gv"),                                BF16W,                  1, 32, M, sc.xb_x,                    m.idx_w, gv::O_F16, L.idx_w))) return e;
    if ((e = bind::make_gv(T("self_attn.indexer.index_kpool_compress_gate.gv"),     BF16W,                  1, 32, M, sc.xb_x,                    m.idx_g, gv::O_F16, L.idx_g))) return e;
    if (qbm.w)
    {
        if (qb_rot)
        {
            if ((e = dtr::make_out(qbm, M, nullptr, 0, sc.dxd, m.q_b, h * 256, dtr::O_F16, sc.dt, 0, 0, 1, L.q_b_t))) return e;
            L.q_b_t.a.suh[0] = qbm.suh;
        }
        else if ((e = dtr::make_out(qbm, M, m.q_a_n, QA, nullptr, m.q_b, h * 256, dtr::O_F16, sc.dt, 0, 0, 1, L.q_b_t))) return e;
    }
    else if ((e = bind::make_gv(T("q_b.gv"),                                        BF16W,                  1, 16, M, sc.xf_qa,                   m.q_b,   gv::O_F16, L.q_b))) return e;
    if ((e = bind::make_gv(T(wqb_mx ? "idx.wq_b.mx.gv" : "idx.wq_b.gv"),            wqb_mx ? MXFP8 : BF16W, iq_tpc, 32, M, sc.xf_qa,              m.idx_q, gv::O_F16, L.idx_q))) return e;
    if (opm.w)
    {
        if (op_rot)
        {
            if ((e = dtr::make_out(opm, M, nullptr, 0, sc.xf_eh, m.sc.part[0], HID, dtr::O_BF16, sc.dt, 64, 0, 1, L.o_proj_t))) return e;
            L.o_proj_t.a.x = (const __half*)y.ab.args.output;
            L.o_proj_t.a.ldx = h * 256;
            L.o_proj_t.a.suh[0] = opm.suh;
        }
        else if ((e = dtr::make_out(opm, M, (const __half*)y.ab.args.output, h * 256, nullptr, m.sc.part[0], HID, dtr::O_BF16, sc.dt, 64, 0, 1, L.o_proj_t))) return e;
    }
    else if ((e = bind::make_gv(T("o_proj.gv"),                                     BF16W,                  2, 8,  M, sc.xf_attn,                 m.sc.part[0], gv::O_BF16, L.o_proj))) return e;
    const int qbK = qbm.w ? qbm.K : L.q_b.K, qbO = qbm.w ? qbm.N : L.q_b.O;
    const int opK = opm.w ? opm.K : L.o_proj.K, opO = opm.w ? opm.N : L.o_proj.O;
    if (!L.q_a.on || !L.kv_a.on || !L.idx_k.on || !L.idx_w.on || !L.idx_g.on || !(L.q_b.on || L.q_b_t.on) || !L.idx_q.on || !(L.o_proj.on || L.o_proj_t.on))
        return "Flash DSA: a projection record is missing";
    if (L.q_a.K != HID || L.q_a.O != QA || L.kv_a.K != HID || L.kv_a.O != KVA || L.idx_k.K != HID || L.idx_k.O != IDXK ||
        L.idx_w.K != HID || L.idx_w.O != IDXW || L.idx_g.K != HID || L.idx_g.O != geom::IDX_DIM ||
        qbK != QA || qbO != h * 256 || L.idx_q.K != QA || L.idx_q.O != IDXQ || opK != h * 256 || opO != HID)
        return "Flash DSA: projection shapes do not match the geometry and the attention binding's head count";
    L.heads = h;
    const bool multi = knob::gv_multi();
    if (multi)
    {
        if (xin_ord) bind::make_gv_multi({&L.idx_k, &L.q_a, &L.idx_w, &L.idx_g, &L.kv_a}, M, L.x_in);
        else         bind::make_gv_multi({&L.q_a, &L.kv_a, &L.idx_k, &L.idx_w, &L.idx_g}, M, L.x_in);
    }
    L.xf_qa.on = L.q_b.on || L.idx_q.on; L.xf_qa.src = m.q_a_n; L.xf_qa.ld = QA; L.xf_qa.K = QA; L.xf_qa.M = M;
    L.xf_qa.f16 = wqb_mx ? sc.xf_qa : nullptr;
    L.xf_qa.bf16 = wqb_mx ? nullptr : sc.xf_qa;
    L.xf_attn.on = L.o_proj.on; L.xf_attn.src = (const __half*)y.ab.args.output; L.xf_attn.ld = h * 256; L.xf_attn.K = h * 256;
    L.xf_attn.bf16 = sc.xf_attn; L.xf_attn.M = M;
    return nullptr;
}

// Fused DSA q_b (pre-rotated K4 trellis, K=1536, N=8192, ks=12, 768 sub-blocks = 384 CTAs of 512 threads)
// + idx.wq_b (MXFP8 .gv, K=1536, O=4096, tpc=1, ks=16, 256 CTAs of 512 threads) in one 640-CTA launch.
template <int NT>
__global__ void __launch_bounds__(512, NT <= 2 ? 2 : 1) k_dsa_qb_idxq(dtr::Args a, bind::Gv g, int idx_first)
{
    __shared__ float4 red[32][32];
    const int qb_blk = idx_first ? (static_cast<int>(blockIdx.x) - 256) : static_cast<int>(blockIdx.x);
    if (qb_blk < 0 || qb_blk >= 384)
    {
        const int iq_cta = idx_first ? static_cast<int>(blockIdx.x) : (static_cast<int>(blockIdx.x) - 384);
        gv_gemv_body<Mxfp8, NT>(g.w, g.xfrag, g.K, g.O / 16, g.tpc, g.ks, g.M, g.y, g.ld, g.odt, red, iq_cta, g.rscale);
        return;
    }
    constexpr int TB = 32 * dtr::BITS, DTR_UNR = 8;
    const int half = threadIdx.x >> 8, tid = threadIdx.x & 255;
    const int lane = tid & 31, j = tid >> 5;
    const int bidx = qb_blk * 2 + half;
    const int nb = bidx & 63, p = bidx >> 6;
    const int K16 = a.K / 16, sc = K16 / a.ks, s0 = p * sc;
    const uint8_t* gb = a.w[0] + nb * a.sB + (s0 >> 3) * a.sG + j * TB;
    Tr3Lane<dtr::BITS, true> ld;
    ld.init(lane);
    u32 wa[DTR_UNR];
    #pragma unroll
    for (int u = 0; u < DTR_UNR; ++u) wa[u] = ld.loadw(gb + u * (8 * TB), lane);
    const u32* xb = a.xa + static_cast<size_t>(s0) * 64 + lane * 2;
    const int xts = K16 * 64;
    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    #pragma unroll
    for (int u = 0; u < DTR_UNR; ++u)
    {
        u32 f0[2], f1[2];
        ld.decodew(wa[u], f0, f1);
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            const uint2 b = __ldg(reinterpret_cast<const uint2*>(xb + t * xts + u * 64));
            mma16816(acc[t], f0[0], f1[0], f0[1], f1[1], b.x, b.y);
        }
    }
    const int M = a.M;
    float* pp = a.part + (static_cast<size_t>(nb) * a.ks + p) * M * 128;
    const int col = j * 16 + (lane >> 2), c = lane & 3;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        const int m0 = t * 8 + 2 * c;
        if (m0 < M) { pp[m0 * 128 + col] = acc[t][0]; pp[m0 * 128 + col + 8] = acc[t][2]; }
        if (m0 + 1 < M) { pp[(m0 + 1) * 128 + col] = acc[t][1]; pp[(m0 + 1) * 128 + col + 8] = acc[t][3]; }
    }
    __threadfence();
    if (half == 0) asm volatile("bar.sync 1, 256;" ::: "memory");
    else           asm volatile("bar.sync 2, 256;" ::: "memory");
    int* last_p = reinterpret_cast<int*>(&red[half][0]);
    if (tid == 0) *last_p = (atomicAdd(a.cnt + nb, 1) == a.ks - 1);
    if (half == 0) asm volatile("bar.sync 1, 256;" ::: "memory");
    else           asm volatile("bar.sync 2, 256;" ::: "memory");
    if (!*last_p) return;
    __threadfence();
    const int n0 = nb * 128 + lane * 4;
    for (int m = j; m < M; m += 8)
    {
        float v[4] = {0.f, 0.f, 0.f, 0.f}, s[4];
        dtr::ld4h(a.svh[0] + n0, s);
        const float* pb = a.part + static_cast<size_t>(nb) * a.ks * M * 128 + m * 128 + lane * 4;
        #pragma unroll 4
        for (int r = 0; r < a.ks; ++r)
        {
            const float4 w = __ldcg(reinterpret_cast<const float4*>(pb + static_cast<size_t>(r) * M * 128));
            v[0] += w.x; v[1] += w.y; v[2] += w.z; v[3] += w.w;
        }
        had128(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] *= s[e];
        const size_t o = static_cast<size_t>(m) * a.ldy + n0;
        *reinterpret_cast<uint2*>(static_cast<__half*>(a.y) + o) = make_uint2(pack_h2(v[0], v[1]), pack_h2(v[2], v[3]));
    }
    if (tid == 0) a.cnt[nb] = 0;
}

inline G6AttnStep dsa_step(Model& m, int M, int context)
{
    G6AttnStep s{};
    s.rows = M; s.context = context;
    s.q_b = m.q_b; s.kv_a = m.kv_a; s.positions = m.pos;
    s.index_q = m.idx_q; s.index_k = m.idx_k; s.index_weight = m.idx_w; s.index_gate = m.idx_g;
    s.raw_sess = m.rsess;
    return s;
}

inline void prepare_dsa_layer(Model& m, int i, int M, cudaStream_t st)
{
    if (M < 1 || M > MX) die("prepare_dsa_layer: M outside 1..MX");
    Layer& y = m.layers[i];
    if (!y.ab.storage || !y.built[M]) die("prepare_dsa_layer: bind_dsa_layer did not run for layer " + std::to_string(y.l));
    if (!m.graph) return;
    for (int k = 0; k < m.nbk; ++k)
    {
        if (y.rp_on[k][M]) continue;
        const G6AttnStep s = dsa_step(m, M, M);
        G6RT_CK(g6_attn_bind_replay_init(&y.ab, &s, m.cells + i, m.bk[k], &y.rp[k][M], st));
        y.rp_on[k][M] = true;
    }
}

#if defined(G6_CP) && G6_CP
// The CP attention of DSA layer i: launch_attn_cp's stages and collectives (rank.cuh) at Flash's sizes, one call per
// layer (S2: on the layer's CP batch view, every row's table and bounds in its cell). In launch_attn_cp's phases, so m4sim's loopback can deliver between them (-1 = all, in
// order, as the real fabric runs it):
//   0: cp_front (append, absorb my heads) -> q(i) put; cp_index -> cand(i) put
//   1: wait cand(i), merge; my head group's partial; wait q(i); each other group's partial, put to its owner at once
//   2: wait the nr - 1 partials of my heads, LSE merge in shard order, value_project into the binding's output
inline void launch_dsa_cp(Model& m, int i, int M, int context, int phase, cudaStream_t st)
{
    Layer& y = m.layers[i];
    CpState& C = m.cp;
    const int r = m.rank, nr = g6nr::NR, H = y.ab.args.heads, G = 16, TP = g6kv::TOP_POOLS;
    if (H != cp_heads(r, y.l)) die("GR8_CP: layer " + std::to_string(y.l) + " binding heads disagree with the group rule");
    G6AttnReplay* srp = nullptr;
#ifdef DEC_S2
    if (m.s2on) srp = s2_cp_replay(m, i, M);
#endif
    if (!srp && m.graph && !y.rp_on[m.cur][M]) die("prepare_dsa_layer(m, i, M) was not called before a graph step");
    G6Attn* a = srp ? &srp->fixed : m.graph ? &y.rp[m.cur][M].fixed : &y.ab.args;
    const G6AttnStepCell* cell = srp ? srp->cell : m.graph ? y.rp[m.cur][M].cell : nullptr;
    void* scr = srp ? srp->scratch : y.ab.scratch;
    const size_t sb = srp ? srp->scratch_bytes : y.ab.scratch_bytes;
    const int cq = C.colls.q[i], cc = C.colls.cand[i];
    // every collective's wire size is M x its row bytes (CpColls, sized for the widest head group)
    const size_t qn = (size_t)M * H * KVA * 4, cb = (size_t)M * C.colls.cand_row;
    const size_t qw = (size_t)M * C.colls.q_row, pw = (size_t)M * C.colls.part_row;
    const size_t blk = cp_blk(M, H);
    auto part_of = [&](int s) { return C.part_all + s * blk; };
    auto cpart = [&](int k) { return C.colls.part[(size_t)i * (nr - 1) + k - 1]; };
    const float *q = nullptr, *rope = nullptr;
    if (phase <= 0)
    {
        if (!srp && !m.graph)
        {
            const G6AttnStep s = dsa_step(m, M, context);
            G6RT_CK(g6_attn_bind_prepare(&y.ab, &s, st));
        }
        G6RT_CK(g6_attn_cp_front(a, cell, scr, sb, st));
        G6RT_CK(g6_attn_cp_qabs(a, scr, &q, &rope));
        cpglue::k_vput<<<G, 256, 0, st>>>(q, qn, nullptr, 0, C.vs[cq], m.stepp, C.tickets + cq);
        G6RT_CK(g6_attn_cp_index(a, cell, scr, sb, C.cand, st));
        cpglue::k_vput<<<G, 256, 0, st>>>(C.cand + (size_t)r * M * TP, cb, nullptr, 0, C.vs[cc], m.stepp, C.tickets + cc);
    }
    if (phase == 0) return;
    if (phase <= 1)
    {
        if (!q) G6RT_CK(g6_attn_cp_qabs(a, scr, &q, &rope));
        const g6::VSlotDev& v = C.vs[cc];
        cpglue::k_vwait<<<1, 32, 0, st>>>(v, m.stepp, cb, -1, cpglue::ERR_CP);
        for (int j = 0; j < v.nsrc; ++j)
            cpglue::k_vget<<<G, 256, 0, st>>>(v, j, m.stepp, C.cand + (size_t)v.src[j] * M * TP, cb);
        G6RT_CK(g6_attn_cp_merge_index(a, cell, C.cand, st));
        G6RT_CK(g6_attn_cp_group(a, cell, scr, sb, q, nullptr, part_of(r), st));
        const g6::VSlotDev& vq = C.vs[cq];
        cpglue::k_vwait<<<1, 32, 0, st>>>(vq, m.stepp, qw, -1, cpglue::ERR_CP);
        for (int k = 1; k < nr; ++k)
        {
            const int d = (r + k) % nr, j = cpglue::vsrc_index(vq, d), hd = cp_heads(d, y.l);
            if (j < 0) die("GR8_CP: q all-gather has no slot for rank " + std::to_string(d));
            G6Attn ad = *a;   // d's head group: hd heads of its absorbed q over MY keys
            ad.heads = hd;
            cpglue::k_vget<<<G, 256, 0, st>>>(vq, j, m.stepp, C.qbuf, (size_t)M * hd * KVA * 4);
            G6RT_CK(g6_attn_cp_group(&ad, cell, scr, sb, C.qbuf, nullptr, C.part_tmp, st));
            cpglue::k_vput<<<G, 256, 0, st>>>(C.part_tmp, (size_t)M * hd * (KVA + 2) * 4, nullptr, 0, C.vs[cpart(k)], m.stepp,
                                               C.tickets + cpart(k));
        }
    }
    if (phase == 1) return;
    for (int k = 1; k < nr; ++k)
    {
        const g6::VSlotDev& v = C.vs[cpart(k)];
        cpglue::k_vwait<<<1, 32, 0, st>>>(v, m.stepp, pw, -1, cpglue::ERR_CP);
        cpglue::k_vget<<<G, 256, 0, st>>>(v, 0, m.stepp, part_of(v.src[0]), (size_t)M * H * (KVA + 2) * 4);   // partials of MY heads
    }
    G6RT_CK(g6_attn_cp_finish(a, cell, scr, sb, C.part_all, st));
}
// the step takes the CP block for its DSA layers (a graph step: its bucket is past R; S2: the step's bucket, s2_cp_step)
inline bool dsa_cp(const Model& m, int context) { return cp_block(m, context); }
#endif

// rows at positions m.pos[0..M-1], context = last position + 1 (plain form; the graph form reads the step cell).
// phase (a step past R with GR8_CP, m4sim's loopback): launch_dsa_cp's phase; 0 also runs the projections, 2 o_proj and the
// publish. -1 = the whole block
inline void launch_dsa_block(Model& m, int i, int M, int context, cudaStream_t st, int phase)
{
    Layer& y = m.layers[i];
    if (!y.built[M]) die("launch_dsa_block: bind_dsa_layer did not run for layer " + std::to_string(y.l));
    const bind::Layer& b = y.L[M];
#if defined(G6_CP) && G6_CP
    const bool cp = dsa_cp(m, context);
#else
    const bool cp = false;
#endif
    if (phase >= 0 && !cp) die("launch_dsa_block: a phase belongs to a CP step");
    if (phase <= 0)
    {
        const int qb_fuse = (int)knob::glm53f_dsa_qb_fuse();
        const bool use_qb_rot = b.q_b_t.on && !b.q_b_t.rot;
        dmark(DP_OTHER, st);
        if (b.x_in.n) bind::launch(b.x_in, st);
        else { bind::launch(b.q_a, st); bind::launch(b.kv_a, st); bind::launch(b.idx_k, st); bind::launch(b.idx_w, st); bind::launch(b.idx_g, st); }
        glue::k_rmsnorm_rows<__half><<<M, 256, 0, st>>>(m.q_a, QA, QA, y.q_a_norm, geom::RMS_EPS_FLASH, m.q_a_n, QA,
                                                        b.xf_qa.on ? b.xf_qa.f16 : nullptr, b.xf_qa.on ? b.xf_qa.bf16 : nullptr,
                                                        use_qb_rot ? b.q_b_t.a.suh[0] : nullptr,
                                                        use_qb_rot ? const_cast<u32*>(b.q_b_t.a.xa) : nullptr);
        dmark(DP_QA, st);
        if (qb_fuse && use_qb_rot && b.q_b_t.a.K == 1536 && b.q_b_t.a.N == 8192 && b.q_b_t.a.ks == 12 && b.q_b_t.sub == 1 &&
            b.idx_q.on && b.idx_q.codec == bind::MXFP8 && b.idx_q.tpc == 1 && b.idx_q.ks == 16 && b.idx_q.O == 4096)
        {
            g6mx::nt_dispatch(b.q_b_t.nt, [&](auto N) {
                k_dsa_qb_idxq<decltype(N)::value><<<640, 512, 0, st>>>(b.q_b_t.a, b.idx_q, qb_fuse == 2 ? 1 : 0);
            });
        }
        else
        {
            bind::launch_q_b(b, st);
            bind::launch(b.idx_q, st);
        }
        dmark(DP_QB, st);
    }
#if defined(G6_CP) && G6_CP
    if (cp)
    {
        launch_dsa_cp(m, i, M, context, phase, st);
        if (phase == 0 || phase == 1) return;
    }
    else
#endif
#ifdef DEC_S2
    if (m.s2on) s2_attention(m, i, M, st);   // the step's rows of every session on the batch view (s2.cuh)
    else
#endif
    if (m.graph)
    {
        if (!y.rp_on[m.cur][M]) die("prepare_dsa_layer(m, i, M) was not called before a graph step");
        G6RT_CK(g6_attn_replay_run(&y.rp[m.cur][M], st));
    }
    else
    {
        // GR8_GLM53F_KV_CHUNKS: no step reaches a page whose chunk is offline (admission keeps contexts inside)
        if (m.fkv.pages && (uint64_t)context > m.fkv.tokens())
            die("Flash DSA: context " + std::to_string(context) + " passes the " + std::to_string(m.fkv.tokens()) +
                " tokens online (GR8_GLM53F_KV_CHUNKS)");
        const G6AttnStep s = dsa_step(m, M, context);
        G6RT_CK(g6_attn_bind_run(&y.ab, &s, st));
    }
    dmark(DP_ATTN, st);
    if (b.o_proj_t.on && !b.o_proj_t.rot)
        dtr::launch_rot(b.o_proj_t.a.x, b.o_proj_t.a.ldx, b.o_proj_t.a.suh[0], const_cast<u32*>(b.o_proj_t.a.xa), b.o_proj_t.a.K, M, st);
    bind::launch(b.xf_attn, st);
    bind::launch_o_proj(b, st);
    dmark(DP_DSA_OPROJ, st);
#ifdef DEC_SLOT
    slot_publish(m, 2 * i, st);
#else
    put_publish(m, m.sc.part[0], 2 * i, M, st);
#endif
    dmark(DP_OPROJ, st);
}

#endif
