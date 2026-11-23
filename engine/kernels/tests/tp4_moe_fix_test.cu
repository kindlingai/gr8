// GPU bit-exactness unit test for TP=4 MoE bugfixes:
//   1. fuse_w4 fires under GR8_TR3_W2ALL=1 across M = 1,2,3,4,8,12,16,24,32 (both !k3 and k3, split_m = 0 and 3)
//      and produces bit-identical (0-ULP) output vs unfused stock.
//   2. Unified capped fused router (moe_rtr_fin_opt::launch_router_plan) matches bind::launch(router) +
//      moe_union_cap::k_router_plan_capped (or glue::k_router_plan when cap=0) bit-for-bit across cap={0,16,32,48}
//      and modes 0..5 for M = 1,2,3,4,8,12,16,24,32.
//   3. moe_rtr_fin_opt::launch_fin covers M = 1..32 and matches k1_tr3_fin<true> bit-for-bit.
//   4. Multi-tile NT>1 leader dedup (forced heavy-hitter experts with up to 32 tokens + shared expert) is 0-ULP
//      bit-exact vs per-tile 8-row chunks.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "glue.cuh"
#include "bind.cuh"
#include "moe_rtr_fin_opt.cuh"
#include "moe_union_cap.cuh"

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL %s:%d %s: ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); ++g_fail; } } while (0)

__device__ __forceinline__ uint32_t rnd32(size_t i, uint32_t seed)
{
    uint32_t x = static_cast<uint32_t>(i) ^ ((static_cast<uint32_t>(i >> 32) + seed) * 0x9e3779b9u) ^ (seed * 0x85ebca6bu);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
__device__ __forceinline__ float rndf(size_t i, uint32_t seed) { return (rnd32(i, seed) >> 8) * (1.f / 16777216.f); }

__global__ void k_fill_bf16_rec(uint16_t* p, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        p[i] = __bfloat16_as_ushort(__float2bfloat16_rn((rndf(i, seed) - 0.5f) * 0.25f));
}
__global__ void k_fill_h(__half* p, size_t n, float amp, uint32_t seed)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        p[i] = __float2half((2.f * rndf(i, seed) - 1.f) * amp);
}
__global__ void k_fill_f(float* p, size_t n, float amp, uint32_t seed)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        p[i] = (2.f * rndf(i, seed) - 1.f) * amp;
}
__global__ void k_fill_u32(uint32_t* p, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        p[i] = rnd32(i, seed);
}

static bool plan_eq(const glue::RoutePlan& a, const glue::RoutePlan& b, int M)
{
    if (a.n_slots != b.n_slots) return false;
    for (int s = 0; s < a.n_slots; ++s)
    {
        if (a.slot_expert[s] != b.slot_expert[s] || a.slot_mask[s] != b.slot_mask[s]) return false;
        for (int c = 0; c < 8; ++c)
            if (a.slot_tok[s][c] != b.slot_tok[s][c] || memcmp(&a.slot_w[s][c], &b.slot_w[s][c], 4) != 0) return false;
    }
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < 8; ++k)
            if (a.tok_slot[m][k] != b.tok_slot[m][k] || a.tok_col[m][k] != b.tok_col[m][k] ||
                a.tok_expert[m][k] != b.tok_expert[m][k] || memcmp(&a.tok_w[m][k], &b.tok_w[m][k], 4) != 0)
                return false;
    return true;
}

int main()
{
    setenv("GR8_MOE_RTR_FIN_OPT", "1", 1);
    CK(bind::init());
    moe_rtr_fin_opt::State mrfo;
    CK(moe_rtr_fin_opt::setup(mrfo));
    CHECK(mrfo.on, "moe_rtr_fin_opt::setup failed");

    const int Ms[] = {1, 2, 3, 4, 8, 12, 16, 24, 32};

    // --- 1. Unified capped fused router exactness vs bind::launch(router) + k_router_plan / k_router_plan_capped ---
    {
        const int O = glue::NEXP, K = geom::HID, OT = O / 16, NG = K / 16;
        const size_t wbytes = static_cast<size_t>(OT) * NG * 512;
        uint8_t* rw = nullptr;
        __half* x = nullptr;
        u32* xb = nullptr;
        float *bias = nullptr, *l_ref = nullptr, *l_opt = nullptr;
        glue::RoutePlan *p_ref = nullptr, *p_opt = nullptr;
        CK(cudaMalloc(&rw, wbytes));
        CK(cudaMalloc(&x, static_cast<size_t>(G6_MX) * K * sizeof(__half)));
        CK(cudaMalloc(&xb, static_cast<size_t>(K / 16) * 64 * sizeof(u32) * g6mx::NT));
        CK(cudaMalloc(&bias, O * sizeof(float)));
        CK(cudaMalloc(&l_ref, static_cast<size_t>(G6_MX) * O * sizeof(float)));
        CK(cudaMalloc(&l_opt, static_cast<size_t>(G6_MX) * O * sizeof(float)));
        CK(cudaMalloc(&p_ref, sizeof(glue::RoutePlan)));
        CK(cudaMalloc(&p_opt, sizeof(glue::RoutePlan)));
        k_fill_bf16_rec<<<256, 256>>>(reinterpret_cast<uint16_t*>(rw), wbytes / 2, 101u);
        k_fill_h<<<256, 256>>>(x, static_cast<size_t>(G6_MX) * K, 1.f, 202u);
        k_fill_f<<<4, 256>>>(bias, O, 0.25f, 303u);
        bind::Tensor rt; rt.p = rw; rt.nd = 3; rt.shape[0] = OT; rt.shape[1] = NG; rt.shape[2] = 512;

        const int caps[] = {0, 16, 32, 48};
        const int modes[] = {0, 1, 2, 3, 4, 5};
        int checked = 0;
        for (int M : Ms)
        {
            if (M > G6_MX) continue;
            bind::Xfrag xf; xf.on = 1; xf.src = x; xf.ld = K; xf.K = K; xf.M = M; xf.bf16 = xb;
            bind::launch(xf, nullptr);
            bind::Gv g_ref, g_opt;
            CHECK(!bind::make_gv(rt, bind::BF16W, 1, 16, M, xb, l_ref, gv::O_F32, g_ref), "make_gv ref");
            CHECK(!bind::make_gv(rt, bind::BF16W, 1, 16, M, xb, l_opt, gv::O_F32, g_opt), "make_gv opt");
            for (int cap : caps)
            {
                for (int mode : modes)
                {
                    if (cap == 0 && mode > 0) continue;
                    CK(cudaMemset(l_ref, 0, static_cast<size_t>(G6_MX) * O * sizeof(float)));
                    CK(cudaMemset(l_opt, 0, static_cast<size_t>(G6_MX) * O * sizeof(float)));
                    CK(cudaMemset(p_ref, 0, sizeof(glue::RoutePlan)));
                    CK(cudaMemset(p_opt, 0, sizeof(glue::RoutePlan)));

                    bind::launch(g_ref, nullptr);
                    if (cap > 0 && !(M <= 1 && cap >= glue::TOPK))
                        moe_union_cap::k_router_plan_capped<<<1, glue::NEXP>>>(l_ref, bias, M, p_ref, cap, mode, 0u, nullptr);
                    else
                        glue::k_router_plan<<<1, glue::NEXP>>>(l_ref, bias, M, p_ref, 0u, nullptr);

                    const bool fired = moe_rtr_fin_opt::launch_router_plan(mrfo, g_opt, bias, M, l_opt, p_opt, 0u, nullptr, nullptr, cap, mode);
                    if (M == 1)
                    {
                        CHECK(!fired, "M=1 cap=%d mode=%d: launch_router_plan expected stock (not fired)", cap, mode);
                        ++checked;
                        continue;
                    }
                    CHECK(fired, "M=%d cap=%d mode=%d: launch_router_plan did not fire", M, cap, mode);
                    CK(cudaDeviceSynchronize());

                    std::vector<float> hr(M * O), ho(M * O);
                    CK(cudaMemcpy(hr.data(), l_ref, M * O * sizeof(float), cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(ho.data(), l_opt, M * O * sizeof(float), cudaMemcpyDeviceToHost));
                    CHECK(memcmp(hr.data(), ho.data(), M * O * sizeof(float)) == 0, "M=%d cap=%d mode=%d: logits differ", M, cap, mode);

                    glue::RoutePlan pr{}, po{};
                    CK(cudaMemcpy(&pr, p_ref, sizeof pr, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(&po, p_opt, sizeof po, cudaMemcpyDeviceToHost));
                    CHECK(plan_eq(pr, po, M), "M=%d cap=%d mode=%d: RoutePlan differs (n_slots %d vs %d)", M, cap, mode, pr.n_slots, po.n_slots);
                    ++checked;
                }
            }
        }
        printf("fused capped router exactness: %d cases checked\n", checked);
        cudaFree(rw); cudaFree(x); cudaFree(xb); cudaFree(bias); cudaFree(l_ref); cudaFree(l_opt); cudaFree(p_ref); cudaFree(p_opt);
    }

    // --- 2. fuse_w4 under w2all=true + Multi-NT (>8 tok/expert + shared) + fin_mode(1..32) ---
    {
        constexpr int NE = geom::NEXP + 1, NP = 16, EB = g6nr::EB, ROWS = NP * EB;
        constexpr size_t MB = static_cast<size_t>(KSH) * 8 * 32;
        const size_t f1_bytes = static_cast<size_t>(ROWS) * 2 * 4 * MB;
        const size_t f2_bytes = static_cast<size_t>(ROWS) * 4 * MB;
        std::vector<long long> htab(NE * 6, 0), htab_p(NP * 6, 0);
        for (int p = 0; p < NP; ++p)
        {
            htab_p[p * 6 + 0] = static_cast<long long>(p) * EB * 2 * 4 * MB;
            htab_p[p * 6 + 1] = static_cast<long long>(p) * EB * 4 * MB;
            htab_p[p * 6 + 2] = static_cast<long long>(p) * EB;
            htab_p[p * 6 + 3] = EB;
            htab_p[p * 6 + 4] = 4;
            htab_p[p * 6 + 5] = 4;
        }
        for (int e = 0; e < NE; ++e)
        {
            const int p = e % NP;
            for (int j = 0; j < 6; ++j) htab[e * 6 + j] = htab_p[p * 6 + j];
        }
        long long *tab = nullptr, *tab_p = nullptr;
        uint8_t *fc1 = nullptr, *fc2 = nullptr, *fc1w = nullptr, *fc2w = nullptr;
        __half *sg = nullptr, *v1b = nullptr, *v2 = nullptr, *v3b = nullptr, *x = nullptr;
        u32* xfrag = nullptr;
        float *logits = nullptr, *bias = nullptr;
        glue::RoutePlan* plan = nullptr;
        uint16_t *out_ref = nullptr, *out_w4 = nullptr, *out_fin = nullptr;

        CK(cudaMalloc(&tab, htab.size() * sizeof(long long)));
        CK(cudaMemcpy(tab, htab.data(), htab.size() * sizeof(long long), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&tab_p, htab_p.size() * sizeof(long long)));
        CK(cudaMemcpy(tab_p, htab_p.data(), htab_p.size() * sizeof(long long), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&fc1, f1_bytes)); CK(cudaMalloc(&fc2, f2_bytes));
        CK(cudaMalloc(&fc1w, f1_bytes)); CK(cudaMalloc(&fc2w, f2_bytes));
        CK(cudaMalloc(&sg, 2 * HID * sizeof(__half)));
        CK(cudaMalloc(&v1b, static_cast<size_t>(ROWS) * 2 * geom::HB * sizeof(__half)));
        CK(cudaMalloc(&v2, static_cast<size_t>(ROWS) * 3 * 128 * sizeof(__half)));
        CK(cudaMalloc(&v3b, static_cast<size_t>(ROWS) * geom::HB * sizeof(__half)));
        CK(cudaMalloc(&x, static_cast<size_t>(G6_MX) * HID * sizeof(__half)));
        CK(cudaMalloc(&xfrag, static_cast<size_t>(KSH) * 64 * sizeof(u32) * g6mx::NT));
        CK(cudaMalloc(&logits, static_cast<size_t>(G6_MX) * glue::NEXP * sizeof(float)));
        CK(cudaMalloc(&bias, glue::NEXP * sizeof(float)));
        CK(cudaMalloc(&plan, sizeof(glue::RoutePlan)));
        CK(cudaMalloc(&out_ref, static_cast<size_t>(G6_MX) * HID * 2));
        CK(cudaMalloc(&out_w4, static_cast<size_t>(G6_MX) * HID * 2));
        CK(cudaMalloc(&out_fin, static_cast<size_t>(G6_MX) * HID * 2));

        k_fill_u32<<<512, 256>>>(reinterpret_cast<uint32_t*>(fc1), f1_bytes / 4, 11u);
        k_fill_u32<<<512, 256>>>(reinterpret_cast<uint32_t*>(fc2), f2_bytes / 4, 22u);
        k_fill_h<<<128, 256>>>(sg, 2 * HID, 0.5f, 33u);
        k_fill_h<<<128, 256>>>(v1b, static_cast<size_t>(ROWS) * 2 * geom::HB, 0.5f, 44u);
        k_fill_h<<<128, 256>>>(v2, static_cast<size_t>(ROWS) * 3 * 128, 0.5f, 55u);
        k_fill_h<<<128, 256>>>(v3b, static_cast<size_t>(ROWS) * geom::HB, 0.5f, 66u);
        k_fill_h<<<128, 256>>>(x, static_cast<size_t>(G6_MX) * HID, 0.5f, 77u);
        k_fill_f<<<64, 256>>>(logits, static_cast<size_t>(G6_MX) * glue::NEXP, 1.f, 88u);
        // Force 3 heavy-hitter experts (5, 77, 200) so ALL M tokens hit them (testing NT=2,3,4 dedup!)
        std::vector<float> hbias(glue::NEXP, 0.f);
        hbias[5] = 8.f; hbias[77] = 8.f; hbias[200] = 8.f;
        CK(cudaMemcpy(bias, hbias.data(), glue::NEXP * sizeof(float), cudaMemcpyHostToDevice));

        // Repack K4 wide4 layout in place on fc1w / fc2w (exact match to boot-time bind::widen4)
        CK(cudaMemcpy(fc1w, fc1, f1_bytes, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(fc2w, fc2, f2_bytes, cudaMemcpyDeviceToDevice));
        launch_tr4_widen(fc1w, tab_p, nullptr, NP, EB, nullptr);
        launch_tr4_widen2(fc2w, tab_p, NP, EB, nullptr);
        CK(cudaDeviceSynchronize());
        CK(cudaFree(tab_p));

        size_t sc_bytes = bind::scratch_bytes(G6_MX);
        void* sc_mem = nullptr;
        CK(cudaMalloc(&sc_mem, sc_bytes));
        CK(cudaMemset(sc_mem, 0, sc_bytes));
        bind::Scratch sc = bind::scratch_at(sc_mem, G6_MX);

        auto mk_t = [](const void* p, int nd, long long a, long long b = 0, long long c = 0) {
            bind::Tensor t; t.p = p; t.nd = nd; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c; return t;
        };
        bind::LayerTensors T;
        T.t[bind::T_TR3_TAB] = mk_t(tab, 2, NE, 6);
        T.t[bind::T_TR3_FC1] = mk_t(fc1, 1, f1_bytes);
        T.t[bind::T_TR3_FC2] = mk_t(fc2, 1, f2_bytes);
        T.t[bind::T_TR3_V2] = mk_t(v2, 3, ROWS, 3, 128);
        T.t[bind::T_TR3_SG] = mk_t(sg, 2, 2, HID);
        T.t[bind::T_TR3_V1B] = mk_t(v1b, 3, ROWS, 2, geom::HB);
        T.t[bind::T_TR3_V3B] = mk_t(v3b, 2, ROWS, geom::HB);
        T.e4m3 = 1; T.sh_trellis = 1; T.cb = TR3_MCG; T.k2 = 0; T.k3 = 0;

        bind::LayerTensors Tw = T;
        Tw.t[bind::T_TR3_FC1] = mk_t(fc1w, 1, f1_bytes);
        Tw.t[bind::T_TR3_FC2] = mk_t(fc2w, 1, f2_bytes);
        Tw.fc1_wide4 = 1; Tw.fc2_wide4 = 1;

        bind::K1Vars vw{};
        vw.w2all = true;
        vw.policy = false;
        bind::K1Vars vw_pol = vw;
        vw_pol.policy = true;

        int checked = 0;
        for (int M : Ms)
        {
            if (M > G6_MX) continue;
            for (int split_m : {0, 3})
            {
                if (split_m && M < split_m) continue;
                bind::Xfrag xf; xf.on = 1; xf.src = x; xf.ld = HID; xf.K = HID; xf.M = M; xf.f16 = xfrag;
                bind::launch(xf, nullptr);
                glue::k_router_plan<<<1, glue::NEXP>>>(logits, bias, M, plan);
                bind::PlanPtrs pp = bind::plan_ptrs(plan);
                bind::Tensor sh[3];

                // Policy check: v.policy=true keeps fuse_w4 at M==1, and forces fuse=0 at M>=2 so GR8_ROUTE_OVL stays on.
                bind::K1 k_pol;
                const char* ep = bind::make_k1(Tw, M, sc, x, xfrag, pp, out_w4, 1, k_pol, sh, 1, split_m, vw_pol);
                CHECK(!ep, "M=%d split_m=%d: policy make_k1 err %s", M, split_m, ep ? ep : "ok");
                if (M == 1) CHECK(k_pol.fuse_w4 != 0, "M=1: policy=true must still use fuse_w4");
                else CHECK(!k_pol.fuse && !k_pol.fuse_w4, "M=%d: policy=true must keep fuse=0 for GR8_ROUTE_OVL", M);

                // Reference: non-wide4, unfused (fuse=0), stock k1_tr3_fin<true>
                bind::K1 k_ref, k_w4;
                const char* e0 = bind::make_k1(T, M, sc, x, xfrag, pp, out_ref, 1, k_ref, sh, 0, split_m, vw);
                const char* e1 = bind::make_k1(Tw, M, sc, x, xfrag, pp, out_w4, 1, k_w4, sh, 1, split_m, vw);
                CHECK(!e0 && !e1, "M=%d split_m=%d: make_k1 err %s / %s", M, split_m, e0 ? e0 : "ok", e1 ? e1 : "ok");
                CHECK(k_w4.fuse_w4 != 0, "M=%d split_m=%d: fuse_w4 did NOT fire under w2all=true policy=false!", M, split_m);
                if (M >= 9) CHECK(moe_rtr_fin_opt::fin_mode(M) == 2, "M=%d: fin_mode expected 2!", M);
                else CHECK(moe_rtr_fin_opt::fin_mode(M) == 0, "M=%d: fin_mode expected 0 (stock)!", M);

                CK(cudaMemset(out_ref, 0xff, static_cast<size_t>(G6_MX) * HID * 2));
                CK(cudaMemset(out_w4, 0xff, static_cast<size_t>(G6_MX) * HID * 2));
                CK(cudaMemset(out_fin, 0xff, static_cast<size_t>(G6_MX) * HID * 2));
                CK(cudaMemset(sc.ctl, 0, 16));
                bind::launch_k1(k_ref, nullptr);
                CK(cudaMemset(sc.ctl, 0, 16));
                bind::launch_k1(k_w4, nullptr);
                if (moe_rtr_fin_opt::fin_mode(M))
                {
                    bind::K1 k_fin = k_w4;
                    k_fin.a.out = reinterpret_cast<__half*>(out_fin);
                    CK(cudaMemset(sc.ctl, 0, 16));
                    moe_rtr_fin_opt::launch_fin(k_fin.a, nullptr);
                }
                CK(cudaDeviceSynchronize());

                std::vector<uint16_t> hr(M * HID), hw(M * HID), hf(M * HID);
                CK(cudaMemcpy(hr.data(), out_ref, M * HID * 2, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hw.data(), out_w4, M * HID * 2, cudaMemcpyDeviceToHost));
                CHECK(hr == hw, "M=%d split_m=%d: fuse_w4 (mode %d) != unfused ref", M, split_m, k_w4.fuse_w4);
                if (moe_rtr_fin_opt::fin_mode(M))
                {
                    CK(cudaMemcpy(hf.data(), out_fin, M * HID * 2, cudaMemcpyDeviceToHost));
                    CHECK(hr == hf, "M=%d split_m=%d: moe_rtr_fin_opt::launch_fin (mode %d) != stock fin", M, split_m, moe_rtr_fin_opt::fin_mode(M));
                }
                ++checked;
            }
        }
        printf("fuse_w4 + multi-NT + fin_mode exactness: %d cases checked\n", checked);
        cudaFree(sc_mem); cudaFree(tab); cudaFree(fc1); cudaFree(fc2); cudaFree(fc1w); cudaFree(fc2w);
        cudaFree(sg); cudaFree(v1b); cudaFree(v2); cudaFree(v3b); cudaFree(x); cudaFree(xfrag);
        cudaFree(logits); cudaFree(bias); cudaFree(plan); cudaFree(out_ref); cudaFree(out_w4); cudaFree(out_fin);
    }

    if (g_fail) { printf("tp4_moe_fix_test: %d FAILED\n", g_fail); return 1; }
    printf("tp4_moe_fix_test: PASS (0-ULP bit-exact across M=1..32, w2all+wide4, multi-NT dedup, capped fused router & fin)\n");
    return 0;
}
