#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "runtime/rank.cuh"
#include "runtime/flash_mlp_pf.cuh"
#include "runtime/flash_dsa_pf.cuh"

using namespace g6rt;

struct SimplePrefill
{
    int MP = 0;
    int* pos = nullptr;
    float *xn = nullptr, *xn2 = nullptr;
    __half *x = nullptr, *x2 = nullptr;
    __nv_bfloat16* part[2] = {};
    FlashRouteSplit route_split;
};

inline void pf_dsa_block_p1_ref(Model& m, SimplePrefill& P, int i, int M, int context, cudaStream_t st)
{
    // Opt-in gate: compare the fused MLA with CURRENT GEMM prefill, not the older P1 math.
    if (getenv("G6_TEST_SPARSE_MLA"))
    {
        setenv("G6_PF_SPARSE_MLA", "0", 1);
        pf_dsa_block_gemm(m, P, i, M, context, st);
        setenv("G6_PF_SPARSE_MLA", "1", 1);
        return;
    }
    Layer& y = m.layers[i];
    for (int m0 = 0; m0 < M; m0 += MX)
    {
        const int r = std::min(MX, M - m0);
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
        const G6AttnStep s = dsa_step(m, r, context - M + m0 + r);
        G6RT_CK(g6_attn_bind_run(&y.ab, &s, st));
        bind::launch(b.xf_attn, st);
        bind::launch_o_proj(b, st);
        G6RT_CK(cudaMemcpyAsync(P.part[0] + (size_t)m0 * HID, m.sc.part[0], (size_t)r * HID * 2, cudaMemcpyDeviceToDevice, st));
    }
}

inline void pf_mlp_block_p1_ref(Model& m, SimplePrefill& P, int i, int M, cudaStream_t st)
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
        flash_mlp::launch(b, st);
    }
}

static uint32_t lcg(uint32_t& s)
{
    s = s * 1664525u + 1013904223u;
    return s;
}
static float rf(uint32_t& s, float scale = 0.2f)
{
    return ((int)(lcg(s) >> 8) - 8388608) * (scale / 8388608.f);
}

struct DevTensor
{
    void* p = nullptr;
    TensorRef ref{};
};

static DevTensor alloc_bf16(uint32_t& seed, long long d0, long long d1 = 0, long long d2 = 0, float scale = 0.15f)
{
    DevTensor t{};
    t.ref.bf16 = true;
    t.ref.nd = d2 ? 3 : (d1 ? 2 : 1);
    t.ref.shape[0] = d0; t.ref.shape[1] = d1; t.ref.shape[2] = d2;
    size_t n = (size_t)d0 * (d1 ? d1 : 1) * (d2 ? d2 : 1);
    std::vector<__nv_bfloat16> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = __float2bfloat16(rf(seed, scale));
    __nv_bfloat16* dp = dmalloc<__nv_bfloat16>(n);
    G6RT_CK(cudaMemcpy(dp, h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    t.p = dp; t.ref.p = dp;
    return t;
}

static DevTensor alloc_bf16_gv(uint32_t& seed, long long O, long long K, float scale = 0.05f)
{
    DevTensor t{};
    t.ref.bf16 = false;
    t.ref.nd = 3;
    t.ref.shape[0] = O / 16; t.ref.shape[1] = K / 16; t.ref.shape[2] = 512;
    size_t n = (size_t)(O / 16) * (K / 16) * 256;
    std::vector<__nv_bfloat16> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = __float2bfloat16(rf(seed, scale));
    __nv_bfloat16* dp = dmalloc<__nv_bfloat16>(n);
    G6RT_CK(cudaMemcpy(dp, h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    t.p = dp; t.ref.p = dp;
    return t;
}

static DevTensor alloc_mxgv(uint32_t& seed, long long O, long long K)
{
    DevTensor t{};
    t.ref.bf16 = false;
    t.ref.nd = 3;
    t.ref.shape[0] = O / 16; t.ref.shape[1] = K / 32; t.ref.shape[2] = 528;
    size_t n_tiles = (size_t)(O / 16) * (K / 32);
    std::vector<uint8_t> h(n_tiles * 528);
    for (size_t ti = 0; ti < n_tiles; ++ti)
    {
        uint8_t* tile = h.data() + ti * 528;
        for (int j = 0; j < 512; ++j)
        {
            __nv_fp8_e4m3 f(rf(seed, 0.5f));
            tile[j] = f.__x;
        }
        for (int j = 0; j < 16; ++j) tile[512 + j] = 122; // 2^-5
    }
    uint8_t* dp = dmalloc<uint8_t>(h.size());
    G6RT_CK(cudaMemcpy(dp, h.data(), h.size(), cudaMemcpyHostToDevice));
    t.p = dp; t.ref.p = dp;
    return t;
}

static DevTensor alloc_trellis(uint32_t& seed, long long K, long long N, int side)
{
    DevTensor t{};
    t.ref.bf16 = false;
    t.ref.nd = 2;
    const long long E = (side == dtr::SIDE_OUT ? N : K) / 128;
    const long long eb = (side == dtr::SIDE_OUT ? K : N) / 16 * 8 * 128;
    t.ref.shape[0] = E; t.ref.shape[1] = eb;
    size_t n = (size_t)E * eb;
    std::vector<uint8_t> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = (uint8_t)(lcg(seed) >> 24);
    uint8_t* dp = dmalloc<uint8_t>(n);
    G6RT_CK(cudaMemcpy(dp, h.data(), n, cudaMemcpyHostToDevice));
    t.p = dp; t.ref.p = dp;
    return t;
}

static DevTensor alloc_sign(uint32_t& seed, long long D, float mag = 0.015625f)
{
    DevTensor t{};
    t.ref.bf16 = false;
    t.ref.nd = 1;
    t.ref.shape[0] = D;
    std::vector<__half> h(D);
    for (long long i = 0; i < D; ++i) h[i] = __float2half_rn(((lcg(seed) & 0x10000) ? 1.f : -1.f) * mag);
    __half* dp = dmalloc<__half>(D);
    G6RT_CK(cudaMemcpy(dp, h.data(), D * sizeof(__half), cudaMemcpyHostToDevice));
    t.p = dp; t.ref.p = dp;
    return t;
}

int main()
{
    if (getenv("G6_TEST_SPARSE_MLA"))
    {
        setenv("G6_PF_SPARSE_MLA", "1", 1);
        puts("SPARSE MLA GATE: explicit opt-in hi/lo candidate; all P1-labelled reference columns use current GEMM prefill (fused MLA OFF)");
    }
    G6RT_CK(bind::init());
    cudaStream_t st;
    G6RT_CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

    const int rank = 0;
    const int layer_id = 3; // DSA layer
    const int h = g6nr::width(g6nr::HEADS, rank);
    const int cap = 2560;
    const int MP = 2048;

    printf("=== 1. W8d Dense MLP Prefill Gate (pf_mlp_block_gemm vs pf_mlp_block_p1, distinct gate.suh / up.suh) ===\n");
    {
        uint32_t seed = 777;
        std::map<std::string, DevTensor> tensors;
        const int ml_id = 0;
        auto add = [&](const char* suf, DevTensor dt) { tensors[lname(ml_id, suf)] = dt; };
        add("dense.gate.trellis", alloc_trellis(seed, HID, g6nr::DENSE_CH, dtr::SIDE_OUT));
        add("dense.gate.suh", alloc_sign(seed, HID, 0.088f));
        add("dense.gate.svh", alloc_sign(seed, g6nr::DENSE_CH, 0.5f));
        add("dense.up.trellis", alloc_trellis(seed, HID, g6nr::DENSE_CH, dtr::SIDE_OUT));
        add("dense.up.suh", alloc_sign(seed, HID, 0.094f)); // distinct suh scale (6.6% diff), as in release-w8!
        add("dense.up.svh", alloc_sign(seed, g6nr::DENSE_CH, 0.5f));
        add("dense.down.trellis", alloc_trellis(seed, g6nr::DENSE_CH, HID, dtr::SIDE_IN));
        add("dense.down.suh", alloc_sign(seed, g6nr::DENSE_CH, 0.088f));
        add("dense.down.svh", alloc_sign(seed, HID, 0.5f));

        Find find = [&](const std::string& name) -> TensorRef {
            auto it = tensors.find(name);
            return it != tensors.end() ? it->second.ref : TensorRef{};
        };

        Model m;
        m.rank = rank; m.nl = 1; m.layers.resize(1);
        m.layers[0].l = ml_id; m.layers[0].kda = false; m.layers[0].moe = false;
        m.scratch = dmalloc<char>(bind::scratch_bytes());
        m.sc = bind::scratch_at(m.scratch);
        void* mlp_ws = dmalloc<char>(flash_mlp::workspace_bytes(MX));
        m.flash_mlp_sc = flash_mlp::workspace_at(mlp_ws, MX);
        if (const char* e = flash_mlp::load(ml_id, find, m.layers[0].flash_mlp_w)) die(e);

        const int MP_MLP = 9216;
        SimplePrefill P_ref, P_pf;
        P_ref.MP = MP_MLP; P_pf.MP = MP_MLP;
        P_ref.xn2 = dmalloc<float>((size_t)MP_MLP * HID); P_pf.xn2 = P_ref.xn2;
        P_ref.x2 = dmalloc<__half>((size_t)MP_MLP * HID); P_pf.x2 = P_ref.x2;
        P_ref.part[1] = dmalloc<__nv_bfloat16>((size_t)MP_MLP * HID);
        P_pf.part[1] = dmalloc<__nv_bfloat16>((size_t)MP_MLP * HID);

        std::vector<float> h_xn2((size_t)MP_MLP * HID);
        std::vector<__half> h_x2((size_t)MP_MLP * HID);
        for (size_t i = 0; i < h_xn2.size(); ++i)
        {
            float v = rf(seed, 0.5f);
            h_xn2[i] = v;
            h_x2[i] = __float2half_rn(v);
        }
        G6RT_CK(cudaMemcpy(P_ref.xn2, h_xn2.data(), h_xn2.size() * 4, cudaMemcpyHostToDevice));
        G6RT_CK(cudaMemcpy(P_ref.x2, h_x2.data(), h_x2.size() * 2, cudaMemcpyHostToDevice));

        bind_mlp_prefill(m, P_pf, 0, find, st);

        for (int M : {64, 512, 2048, 9216})
        {
            cudaEvent_t e0, e1, e2;
            cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
            cudaEventRecord(e0, st);
            pf_mlp_block_p1_ref(m, P_ref, 0, M, st);
            cudaEventRecord(e1, st);
            pf_mlp_block_gemm(m, P_pf, 0, M, st);
            cudaEventRecord(e2, st);
            G6RT_CK(cudaStreamSynchronize(st));
            float ms_p1 = 0, ms_pf = 0;
            cudaEventElapsedTime(&ms_p1, e0, e1);
            cudaEventElapsedTime(&ms_pf, e1, e2);

            std::vector<__nv_bfloat16> out_ref((size_t)M * HID), out_pf((size_t)M * HID);
            G6RT_CK(cudaMemcpy(out_ref.data(), P_ref.part[1], out_ref.size() * 2, cudaMemcpyDeviceToHost));
            G6RT_CK(cudaMemcpy(out_pf.data(), P_pf.part[1], out_pf.size() * 2, cudaMemcpyDeviceToHost));
            double diff2 = 0, ref2 = 0, max_abs = 0;
            size_t u0 = 0, u1 = 0;
            for (size_t i = 0; i < out_ref.size(); ++i)
            {
                float a = __bfloat162float(out_ref[i]), b = __bfloat162float(out_pf[i]);
                if (!std::isfinite(a) || !std::isfinite(b)) { fprintf(stderr, "FAIL nonfinite output\n"); return 1; }
                double d = (double)a - (double)b;
                diff2 += d * d; ref2 += (double)a * (double)a;
                if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
                uint16_t ua, ub; std::memcpy(&ua, &out_ref[i], 2); std::memcpy(&ub, &out_pf[i], 2);
                int ulp = std::abs((int)ua - (int)ub);
                if (ulp == 0) ++u0;
                if (ulp <= 1) ++u1;
            }
            double rel_l2 = std::sqrt(diff2 / (ref2 > 0 ? ref2 : 1.0));
            printf("Dense W8d M=%4d: rel_l2=%.3e max_abs=%.3e 0-ULP=%.2f%% <=1-ULP=%.2f%% | P1=%.2fms PF=%.2fms (%.2fx)\n",
                   M, rel_l2, max_abs, 100.0 * u0 / out_ref.size(), 100.0 * u1 / out_ref.size(), ms_p1, ms_pf, ms_p1 / ms_pf);
            if (rel_l2 > 3e-3) { fprintf(stderr, "FAIL Dense W8d M=%d\n", M); return 1; }
        }
        flash_mlp_pf::states().clear();
        cudaFree(m.scratch); cudaFree(mlp_ws);
        cudaFree(P_ref.xn2); cudaFree(P_ref.x2); cudaFree(P_ref.part[1]); cudaFree(P_pf.part[1]);
        for (auto& kv : tensors) cudaFree(kv.second.p);
    }

    printf("\n=== 2. DSA Block Prefill Gate (pf_dsa_block_gemm vs pf_dsa_block_p1, BF16 and W8d) ===\n");
    for (int w8d = 0; w8d <= 1; ++w8d)
    {
        uint32_t seed = 42 + w8d * 1000;
        std::map<std::string, DevTensor> tensors;
        auto add = [&](const char* suf, DevTensor dt) { tensors[lname(layer_id, suf)] = dt; };

        add("kv_b.uk", alloc_bf16(seed, h, 256, 512, 0.08f));
        add("kv_b.uv", alloc_bf16(seed, h, 256, 512, 0.08f));
        add("kv_a_norm", alloc_bf16(seed, 512, 0, 0, 0.5f));
        add("idx.k_norm.w", alloc_bf16(seed, 128, 0, 0, 0.5f));
        add("idx.k_norm.b", alloc_bf16(seed, 128, 0, 0, 0.05f));
        add("self_attn.indexer.index_kpool_compress_ape", alloc_bf16(seed, 4, 128, 0, 0.1f));
        add("q_a_norm", alloc_bf16(seed, QA, 0, 0, 0.5f));
        add("idx.weights_proj.gv", alloc_bf16_gv(seed, IDXW, HID, 0.05f));
        add("self_attn.indexer.index_kpool_compress_gate.gv", alloc_bf16_gv(seed, geom::IDX_DIM, HID, 0.05f));

        if (!w8d)
        {
            add("q_a.gv", alloc_bf16_gv(seed, QA, HID, 0.05f));
            add("kv_a.gv", alloc_bf16_gv(seed, KVA, HID, 0.05f));
            add("idx.wk.gv", alloc_bf16_gv(seed, IDXK, HID, 0.05f));
            add("q_b.gv", alloc_bf16_gv(seed, h * 256, QA, 0.05f));
            add("idx.wq_b.gv", alloc_bf16_gv(seed, IDXQ, QA, 0.05f));
            add("o_proj.gv", alloc_bf16_gv(seed, HID, h * 256, 0.05f));
        }
        else
        {
            add("q_a.mx.gv", alloc_mxgv(seed, QA, HID));
            add("kv_a.mx.gv", alloc_mxgv(seed, KVA, HID));
            add("idx.wk.mx.gv", alloc_mxgv(seed, IDXK, HID));
            add("idx.wq_b.mx.gv", alloc_mxgv(seed, IDXQ, QA));
            add("q_b.trellis", alloc_trellis(seed, QA, h * 256, dtr::SIDE_OUT));
            add("q_b.suh", alloc_sign(seed, QA, 0.088f));
            add("q_b.svh", alloc_sign(seed, h * 256, 0.5f));
            add("o_proj.trellis", alloc_trellis(seed, h * 256, HID, dtr::SIDE_IN));
            add("o_proj.suh", alloc_sign(seed, h * 256, 0.088f));
            add("o_proj.svh", alloc_sign(seed, HID, 0.5f));
        }

        Find find = [&](const std::string& name) -> TensorRef {
            auto it = tensors.find(name);
            return it != tensors.end() ? it->second.ref : TensorRef{};
        };

        auto make_model = [&](Model& m) {
            m.rank = rank;
            m.nl = 1;
            m.layers.resize(1);
            m.layers[0].l = layer_id;
            m.layers[0].kda = false;
            m.q_a = dmalloc<__half>(MX * QA);
            m.q_a_n = dmalloc<__half>(MX * QA);
            m.kv_a = dmalloc<__half>(MX * KVA);
            m.q_b = dmalloc<__half>(MX * h * 256);
            m.idx_q = dmalloc<__half>(MX * IDXQ);
            m.idx_k = dmalloc<__half>(MX * IDXK);
            m.idx_w = dmalloc<__half>(MX * IDXW);
            m.idx_g = dmalloc<__half>(MX * geom::IDX_DIM);
            m.pos = dmalloc<int>(MX);
            m.scratch = dmalloc<char>(bind::scratch_bytes());
            m.sc = bind::scratch_at(m.scratch);
            bind_dsa_layer(m, 0, find, cap, st);
        };

        Model m_ref, m_pf;
        make_model(m_ref);
        make_model(m_pf);

        SimplePrefill P_ref, P_pf;
        P_ref.MP = MP; P_pf.MP = MP;
        P_ref.pos = dmalloc<int>(MP); P_pf.pos = dmalloc<int>(MP);
        P_ref.xn = dmalloc<float>((size_t)MP * HID); P_pf.xn = dmalloc<float>((size_t)MP * HID);
        P_ref.x = dmalloc<__half>((size_t)MP * HID); P_pf.x = dmalloc<__half>((size_t)MP * HID);
        P_ref.part[0] = dmalloc<__nv_bfloat16>((size_t)MP * HID);
        P_pf.part[0] = dmalloc<__nv_bfloat16>((size_t)MP * HID);

        std::vector<float> h_xn((size_t)MP * HID);
        std::vector<__half> h_x((size_t)MP * HID);
        for (size_t i = 0; i < h_xn.size(); ++i)
        {
            float v = rf(seed, 0.5f);
            h_xn[i] = v;
            h_x[i] = __float2half_rn(v);
        }
        G6RT_CK(cudaMemcpy(P_ref.xn, h_xn.data(), h_xn.size() * sizeof(float), cudaMemcpyHostToDevice));
        G6RT_CK(cudaMemcpy(P_pf.xn, h_xn.data(), h_xn.size() * sizeof(float), cudaMemcpyHostToDevice));
        G6RT_CK(cudaMemcpy(P_ref.x, h_x.data(), h_x.size() * sizeof(__half), cudaMemcpyHostToDevice));
        G6RT_CK(cudaMemcpy(P_pf.x, h_x.data(), h_x.size() * sizeof(__half), cudaMemcpyHostToDevice));

        bind_dsa_prefill(m_pf, P_pf, st);
        if (getenv("G6_TEST_SPARSE_MLA")) bind_dsa_prefill(m_ref, P_ref, st);

        for (int M : {8, 64, 512, 2048})
        {
            const int base_pos = (M == 512) ? 64 : 0;
            const int context = base_pos + M;
            if (base_pos > 0)
            {
                std::vector<int> pre_pos(base_pos);
                for (int r = 0; r < base_pos; ++r) pre_pos[r] = r;
                G6RT_CK(cudaMemcpy(P_ref.pos, pre_pos.data(), base_pos * sizeof(int), cudaMemcpyHostToDevice));
                G6RT_CK(cudaMemcpy(P_pf.pos, pre_pos.data(), base_pos * sizeof(int), cudaMemcpyHostToDevice));
                pf_dsa_block_p1_ref(m_ref, P_ref, 0, base_pos, base_pos, st);
                pf_dsa_block_gemm(m_pf, P_pf, 0, base_pos, base_pos, st);
                G6RT_CK(cudaStreamSynchronize(st));
            }
            std::vector<int> pos(M);
            for (int r = 0; r < M; ++r) pos[r] = base_pos + r;
            G6RT_CK(cudaMemcpy(P_ref.pos, pos.data(), M * sizeof(int), cudaMemcpyHostToDevice));
            G6RT_CK(cudaMemcpy(P_pf.pos, pos.data(), M * sizeof(int), cudaMemcpyHostToDevice));

            cudaEvent_t e0, e1, e2;
            cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
            cudaEventRecord(e0, st);
            pf_dsa_block_p1_ref(m_ref, P_ref, 0, M, context, st);
            cudaEventRecord(e1, st);
            pf_dsa_block_gemm(m_pf, P_pf, 0, M, context, st);
            cudaEventRecord(e2, st);
            G6RT_CK(cudaStreamSynchronize(st));
            float ms_p1 = 0, ms_pf = 0;
            cudaEventElapsedTime(&ms_p1, e0, e1);
            cudaEventElapsedTime(&ms_pf, e1, e2);

            std::vector<__nv_bfloat16> out_ref((size_t)M * HID), out_pf((size_t)M * HID);
            G6RT_CK(cudaMemcpy(out_ref.data(), P_ref.part[0], out_ref.size() * 2, cudaMemcpyDeviceToHost));
            G6RT_CK(cudaMemcpy(out_pf.data(), P_pf.part[0], out_pf.size() * 2, cudaMemcpyDeviceToHost));

            double diff2 = 0, ref2 = 0, max_abs = 0;
            size_t u0 = 0, u1 = 0;
            for (size_t i = 0; i < out_ref.size(); ++i)
            {
                float a = __bfloat162float(out_ref[i]);
                float b = __bfloat162float(out_pf[i]);
                if (!std::isfinite(a) || !std::isfinite(b)) { fprintf(stderr, "FAIL nonfinite output\n"); return 1; }
                double d = (double)a - (double)b;
                diff2 += d * d;
                ref2 += (double)a * (double)a;
                if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
                uint16_t ua, ub;
                std::memcpy(&ua, &out_ref[i], 2);
                std::memcpy(&ub, &out_pf[i], 2);
                int ulp = std::abs((int)ua - (int)ub);
                if (ulp == 0) ++u0;
                if (ulp <= 1) ++u1;
            }
            double rel_l2 = std::sqrt(diff2 / (ref2 > 0 ? ref2 : 1.0));

            const size_t kv_bytes = (size_t)m_ref.layers[0].ab.args.page_count * g6kv::PAGE * g6kv::ROW;
            const size_t ix_bytes = (size_t)m_ref.layers[0].ab.args.page_count * g6kv::IDX_PAGE;
            std::vector<uint8_t> kv_r(kv_bytes), kv_p(kv_bytes), ix_r(ix_bytes), ix_p(ix_bytes);
            G6RT_CK(cudaMemcpy(kv_r.data(), m_ref.layers[0].ab.args.mla_cache, kv_bytes, cudaMemcpyDeviceToHost));
            G6RT_CK(cudaMemcpy(kv_p.data(), m_pf.layers[0].ab.args.mla_cache, kv_bytes, cudaMemcpyDeviceToHost));
            G6RT_CK(cudaMemcpy(ix_r.data(), m_ref.layers[0].ab.args.index_cache, ix_bytes, cudaMemcpyDeviceToHost));
            G6RT_CK(cudaMemcpy(ix_p.data(), m_pf.layers[0].ab.args.index_cache, ix_bytes, cudaMemcpyDeviceToHost));
            size_t kv_diff = 0, ix_diff = 0;
            for (size_t i = 0; i < kv_bytes; ++i) if (kv_r[i] != kv_p[i]) ++kv_diff;
            for (size_t i = 0; i < ix_bytes; ++i) if (ix_r[i] != ix_p[i]) ++ix_diff;

            printf("DSA %s M=%4d ctx=%4d: rel_l2=%.3e max_abs=%.3e 0-ULP=%.2f%% <=1-ULP=%.2f%% kv_diff=%zu ix_diff=%zu | P1=%.2fms PF=%.2fms (%.2fx)\n",
                   w8d ? "W8d " : "BF16", M, context, rel_l2, max_abs,
                   100.0 * u0 / out_ref.size(), 100.0 * u1 / out_ref.size(),
                   kv_diff, ix_diff, ms_p1, ms_pf, ms_p1 / ms_pf);
            if (kv_diff != 0 || ix_diff != 0 || rel_l2 > 3e-3)
            {
                fprintf(stderr, "FAIL: mode=%d M=%d\n", w8d, M);
                return 1;
            }
        }

        if (w8d == 1)
        {
            printf("\n=== 3. Long-Context W8d DSA Prefill Scaling & Gate (M=2048 chunks up to 16K and 32K) ===\n");
            flash_dsa_pf::states().clear();
            g6_attn_bind_release(&m_ref.layers[0].ab);
            g6_attn_bind_release(&m_pf.layers[0].ab);
            const int cap_long = 32768;
            bind_dsa_layer(m_ref, 0, find, cap_long, st);
            bind_dsa_layer(m_pf, 0, find, cap_long, st);
            bind_dsa_prefill(m_pf, P_pf, st);
            if (getenv("G6_TEST_SPARSE_MLA")) bind_dsa_prefill(m_ref, P_ref, st);

            double cum_pf_ms = 0.0, cum_proj_ms = 0.0, cum_attn_ms = 0.0, cum_oproj_ms = 0.0;
            cudaEvent_t ev[4];
            for (int k = 0; k < 4; ++k) cudaEventCreate(&ev[k]);

            const int contexts[] = {1024,3072,5120,7168,9216,10240,12288,14336,16384,18432,20480,22528,24576,26624,28672,30720,32768};
            int previous_context = 0;
            for (const int context : contexts)
            {
                const int base_pos = previous_context;
                const int rows = context - base_pos;
                previous_context = context;
                std::vector<int> pos(rows);
                for (int r = 0; r < rows; ++r) pos[r] = base_pos + r;
                G6RT_CK(cudaMemcpy(P_pf.pos, pos.data(), rows * sizeof(int), cudaMemcpyHostToDevice));

                const bool do_gate = (context == 1024 || context == 3072 || context == 7168 || context == 9216 || context == 16384 || context == 32768);
                float ms_p1 = 0.f;
                if (do_gate)
                {
                    const size_t kv_bytes = (size_t)m_pf.layers[0].ab.args.page_count * g6kv::PAGE * g6kv::ROW;
                    const size_t ix_bytes = (size_t)m_pf.layers[0].ab.args.page_count * g6kv::IDX_PAGE;
                    G6RT_CK(cudaMemcpyAsync(m_ref.layers[0].ab.args.mla_cache, m_pf.layers[0].ab.args.mla_cache, kv_bytes, cudaMemcpyDeviceToDevice, st));
                    G6RT_CK(cudaMemcpyAsync(m_ref.layers[0].ab.args.index_cache, m_pf.layers[0].ab.args.index_cache, ix_bytes, cudaMemcpyDeviceToDevice, st));
                    G6RT_CK(cudaMemcpyAsync(P_ref.pos, P_pf.pos, rows * sizeof(int), cudaMemcpyDeviceToDevice, st));
                    cudaEventRecord(ev[0], st);
                    pf_dsa_block_p1_ref(m_ref, P_ref, 0, rows, context, st);
                    cudaEventRecord(ev[1], st);
                    G6RT_CK(cudaStreamSynchronize(st));
                    cudaEventElapsedTime(&ms_p1, ev[0], ev[1]);
                }

                cudaEventRecord(ev[0], st);
                int mark_idx = 0;
                pf_dsa_block_gemm(m_pf, P_pf, 0, rows, context, st, [&](int, cudaStream_t s) {
                    if (mark_idx < 3) cudaEventRecord(ev[++mark_idx], s);
                });
                G6RT_CK(cudaStreamSynchronize(st));
                float ms_proj = 0.f, ms_attn = 0.f, ms_oproj = 0.f, ms_pf = 0.f;
                cudaEventElapsedTime(&ms_proj, ev[0], ev[1]);
                cudaEventElapsedTime(&ms_attn, ev[1], ev[2]);
                cudaEventElapsedTime(&ms_oproj, ev[2], ev[3]);
                cudaEventElapsedTime(&ms_pf, ev[0], ev[3]);

                cum_proj_ms += ms_proj;
                cum_attn_ms += ms_attn;
                cum_oproj_ms += ms_oproj;
                cum_pf_ms += ms_pf;

                if (do_gate)
                {
                    std::vector<__nv_bfloat16> out_ref((size_t)rows * HID), out_pf((size_t)rows * HID);
                    G6RT_CK(cudaMemcpy(out_ref.data(), P_ref.part[0], out_ref.size() * 2, cudaMemcpyDeviceToHost));
                    G6RT_CK(cudaMemcpy(out_pf.data(), P_pf.part[0], out_pf.size() * 2, cudaMemcpyDeviceToHost));
                    double diff2 = 0, ref2 = 0, max_abs = 0;
                    size_t u0 = 0, u1 = 0;
                    for (size_t i = 0; i < out_ref.size(); ++i)
                    {
                        float a = __bfloat162float(out_ref[i]), b = __bfloat162float(out_pf[i]);
                        if (!std::isfinite(a) || !std::isfinite(b)) { fprintf(stderr, "FAIL nonfinite output\n"); return 1; }
                        double d = (double)a - (double)b;
                        diff2 += d * d; ref2 += (double)a * (double)a;
                        if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
                        uint16_t ua, ub; std::memcpy(&ua, &out_ref[i], 2); std::memcpy(&ub, &out_pf[i], 2);
                        int ulp = std::abs((int)ua - (int)ub);
                        if (ulp == 0) ++u0;
                        if (ulp <= 1) ++u1;
                    }
                    double rel_l2 = std::sqrt(diff2 / (ref2 > 0 ? ref2 : 1.0));
                    printf("DSA W8d chunk [%5d..%5d) ctx=%5d: proj=%.2fms attn=%.2fms o_proj=%.2fms total=%.2fms (P1=%.2fms, %.2fx) | cum_total=%.2fms (cum_attn=%.2fms) | rel_l2=%.3e 0-ULP=%.2f%% <=1-ULP=%.2f%%\n",
                           base_pos, context, context, ms_proj, ms_attn, ms_oproj, ms_pf, ms_p1, ms_p1 / ms_pf,
                           cum_pf_ms, cum_attn_ms, rel_l2, 100.0 * u0 / out_ref.size(), 100.0 * u1 / out_ref.size());
                    if (rel_l2 > 3e-3)
                    {
                        fprintf(stderr, "FAIL Long-Context DSA W8d ctx=%d rel_l2=%.3e\n", context, rel_l2);
                        return 1;
                    }
                }
            }
            for (int k = 0; k < 4; ++k) cudaEventDestroy(ev[k]);
        }

        flash_dsa_pf::states().clear();
        g6_attn_bind_release(&m_ref.layers[0].ab);
        g6_attn_bind_release(&m_pf.layers[0].ab);
        cudaFree(m_ref.scratch); cudaFree(m_pf.scratch);
        cudaFree(P_ref.pos); cudaFree(P_pf.pos);
        cudaFree(P_ref.xn); cudaFree(P_pf.xn);
        cudaFree(P_ref.x); cudaFree(P_pf.x);
        cudaFree(P_ref.part[0]); cudaFree(P_pf.part[0]);
        for (auto& kv : tensors) cudaFree(kv.second.p);
    }
    printf("ALL PASS\n");
    return 0;
}
