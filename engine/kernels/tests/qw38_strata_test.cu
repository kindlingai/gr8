// Bit-exact parity test for Qwen3.8 Strata-mode:
//   1. CPU widen_layer_experts_host vs GPU launch_tr3/tr4_widen{,2} (K2/K3/K4 mixed) -> 0 byte diffs!
//   2. StrataExpertCache decode (M=1, M=5, CUDA graphed, live async promotions/evictions) vs 100% VRAM -> 0 byte diffs!
//   3. StrataExpertCache double-buffered H2D prefill (M=64, M=256) vs 100% VRAM -> 0 byte diffs!
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

#define G6_MODEL_QW38 1
#define G6_NR 1
#include "../../runtime/qw38_strata.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s failed: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

int main() {
    constexpr int NE = geom::NEXP; // 512
    constexpr int EB = g6nr::EB;   // 5
    constexpr size_t MB = static_cast<size_t>(geom::KSH) * 8 * 32; // 40960 B/bit

    // Build 1 synthetic layer of 512 experts: mix K2 (e%7==0), K4 (e%11==0), K3 (rest) to test ALL widths!
    std::vector<long long> h_tab(NE * 6, 0);
    std::vector<long long> h_kmix(NE, 0);
    size_t fc1_tot = 0, fc2_tot = 0;
    for (int e = 0; e < NE; ++e) {
        int k = 3;
        if (e % 7 == 0) k = 2;
        else if (e % 11 == 0) k = 4;
        h_tab[e * 6 + 0] = static_cast<long long>(fc1_tot);
        h_tab[e * 6 + 1] = static_cast<long long>(fc2_tot);
        h_tab[e * 6 + 2] = static_cast<long long>(e) * EB;
        h_tab[e * 6 + 3] = EB;
        h_tab[e * 6 + 4] = k;
        h_tab[e * 6 + 5] = 0;
        fc1_tot += static_cast<size_t>(EB) * 2 * static_cast<size_t>(k) * MB;
        fc2_tot += static_cast<size_t>(EB) * static_cast<size_t>(k) * MB;
    }

    uint8_t *h_pin_fc1 = nullptr, *h_pin_fc2 = nullptr;
    uint8_t *d_uva_fc1 = nullptr, *d_uva_fc2 = nullptr;
    CK(cudaHostAlloc(&h_pin_fc1, fc1_tot, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&h_pin_fc2, fc2_tot, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostGetDevicePointer(&d_uva_fc1, h_pin_fc1, 0));
    CK(cudaHostGetDevicePointer(&d_uva_fc2, h_pin_fc2, 0));

    // Deterministic pseudo-random bytes
    uint32_t rng = 0x12345678u;
    auto fill_rand = [&](uint8_t* p, size_t n) {
        uint32_t* w = reinterpret_cast<uint32_t*>(p);
        for (size_t i = 0; i < n / 4; ++i) {
            rng = rng * 1664525u + 1013904223u;
            w[i] = rng;
        }
    };
    fill_rand(h_pin_fc1, fc1_tot);
    fill_rand(h_pin_fc2, fc2_tot);

    // Reference GPU copy for GPU widen & 100% VRAM resident MoE
    uint8_t *d_ref_fc1 = nullptr, *d_ref_fc2 = nullptr;
    long long* d_tab = nullptr;
    CK(cudaMalloc(&d_ref_fc1, fc1_tot));
    CK(cudaMalloc(&d_ref_fc2, fc2_tot));
    CK(cudaMalloc(&d_tab, NE * 6 * sizeof(long long)));
    CK(cudaMemcpy(d_ref_fc1, h_pin_fc1, fc1_tot, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ref_fc2, h_pin_fc2, fc2_tot, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tab, h_tab.data(), NE * 6 * sizeof(long long), cudaMemcpyHostToDevice));

    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

    // --- TEST 1: CPU widen_layer_experts_host vs GPU launch_tr3/tr4_widen{,2} ---
    launch_tr3_widen(d_ref_fc1, d_tab, nullptr, NE, EB, st);
    launch_tr4_widen(d_ref_fc1, d_tab, nullptr, NE, EB, st);
    launch_tr3_widen2(d_ref_fc2, d_tab, NE, EB, st);
    launch_tr4_widen2(d_ref_fc2, d_tab, NE, EB, st);
    CK(cudaStreamSynchronize(st));

    qw38rt::widen_layer_experts_host(h_pin_fc1, h_pin_fc2, h_tab.data(), nullptr, NE);

    std::vector<uint8_t> gpu_fc1(fc1_tot), gpu_fc2(fc2_tot);
    CK(cudaMemcpy(gpu_fc1.data(), d_ref_fc1, fc1_tot, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gpu_fc2.data(), d_ref_fc2, fc2_tot, cudaMemcpyDeviceToHost));
    size_t diff1 = 0, diff2 = 0;
    for (size_t i = 0; i < fc1_tot; ++i) if (gpu_fc1[i] != h_pin_fc1[i]) ++diff1;
    for (size_t i = 0; i < fc2_tot; ++i) if (gpu_fc2[i] != h_pin_fc2[i]) ++diff2;
    printf("[Test 1] CPU vs GPU wide-reorder (K2/K3/K4 mixed, 512 experts): fc1_diffs=%zu fc2_diffs=%zu -> %s\n",
           diff1, diff2, (diff1 == 0 && diff2 == 0) ? "PASS" : "FAIL");
    if (diff1 || diff2) return 1;

    // --- Setup shared/router/scales/scratch for MoE parity tests ---
    cublasHandle_t blas;
    cublasCreate(&blas);

    auto rand_bf16 = [&](size_t n, float scale) {
        std::vector<__nv_bfloat16> v(n);
        for (size_t i = 0; i < n; ++i) {
            rng = rng * 1664525u + 1013904223u;
            float f = (static_cast<int32_t>(rng) / 2147483648.0f) * scale;
            v[i] = __float2bfloat16(f);
        }
        __nv_bfloat16* d = nullptr;
        CK(cudaMalloc(&d, n * sizeof(__nv_bfloat16)));
        CK(cudaMemcpy(d, v.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        return d;
    };
    auto rand_f16 = [&](size_t n, float scale) {
        std::vector<__half> v(n);
        for (size_t i = 0; i < n; ++i) {
            rng = rng * 1664525u + 1013904223u;
            float f = (static_cast<int32_t>(rng) / 2147483648.0f) * scale;
            v[i] = __float2half_rn(f);
        }
        __half* d = nullptr;
        CK(cudaMalloc(&d, n * sizeof(__half)));
        CK(cudaMemcpy(d, v.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
        return d;
    };

    qw38::MoeWeights w_ref{};
    w_ref.router = rand_bf16(NE * geom::HID, 0.05f);
    w_ref.sh_gate = rand_bf16(geom::SHARED_I * geom::HID, 0.05f);
    w_ref.sh_up = rand_bf16(geom::SHARED_I * geom::HID, 0.05f);
    w_ref.sh_down = rand_bf16(geom::HID * geom::SHARED_I, 0.05f);
    w_ref.sh_exp_gate = rand_bf16(geom::HID, 0.05f);
    w_ref.sh_i = geom::SHARED_I;
    w_ref.is_tr3 = true;
    w_ref.t_fc1 = d_ref_fc1;
    w_ref.t_fc2 = d_ref_fc2;
    w_ref.t_tab = d_tab;
    w_ref.t_sg = rand_f16(2 * geom::HID, 1.0f);
    w_ref.t_v1b = rand_f16(NE * EB * 2 * geom::HB, 0.1f);
    w_ref.t_v2 = rand_f16(NE * EB * 3 * 128, 0.1f);
    w_ref.t_v3b = rand_f16(NE * EB * geom::HB, 0.1f);
    w_ref.t_rows = NE;
    w_ref.t_eb = EB;
    w_ref.fc1_wide = 1;
    w_ref.fc2_wide = 1;

    constexpr int MAX_M = 256;
    qw38::MoeScratch sc{};
    CK(cudaMalloc(&sc.router_logits, MAX_M * NE * sizeof(float)));
    CK(cudaMalloc(&sc.topk_idx, MAX_M * geom::TOPK * sizeof(int)));
    CK(cudaMalloc(&sc.topk_w, MAX_M * geom::TOPK * sizeof(float)));
    CK(cudaMalloc(&sc.d_n_groups, sizeof(int)));
    CK(cudaMalloc(&sc.grp_exp, MAX_M * geom::TOPK * sizeof(int)));
    CK(cudaMalloc(&sc.grp_cnt, MAX_M * geom::TOPK * sizeof(int)));
    CK(cudaMalloc(&sc.grp_off, MAX_M * geom::TOPK * sizeof(int)));
    CK(cudaMalloc(&sc.grp_routes, MAX_M * geom::TOPK * 8 * sizeof(int)));
    CK(cudaMalloc(&sc.x_had, MAX_M * geom::HID * sizeof(__half)));
    CK(cudaMalloc(&sc.fc1_part, 8 * geom::TOPK * EB * 2 * 2 * 128 * sizeof(float)));
    CK(cudaMalloc(&sc.fc1_act, MAX_M * geom::TOPK * EB * 128 * sizeof(__half)));
    CK(cudaMalloc(&sc.fc2_part, MAX_M * geom::TOPK * geom::HID * sizeof(float)));
    CK(cudaMalloc(&sc.sh_gate_buf, MAX_M * geom::SHARED_I * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&sc.sh_up_buf, MAX_M * geom::SHARED_I * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&sc.sh_out, MAX_M * geom::HID * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&sc.sh_scalar_gate, MAX_M * sizeof(float)));

    // Build StrataExpertCache with ONLY 160 / 512 experts in VRAM (~31%) + adaptive promotions!
    qw38rt::StrataConfig scfg;
    scfg.enabled = true;
    scfg.cache_slots_override = 160;
    scfg.profile_path = ""; // round-robin so many routed experts are cold!
    scfg.adaptive = true;
    scfg.async_promote = true;
    scfg.promote_max_experts = 8;
    scfg.min_promote_heat = 0.5f;
    scfg.pf_stage_experts = 16;

    qw38rt::StrataExpertCache cache;
    std::vector<int> lids = {0};
    std::vector<const uint8_t*> hf1 = {h_pin_fc1}, df1 = {d_uva_fc1}, hf2 = {h_pin_fc2}, df2 = {d_uva_fc2};
    std::vector<std::vector<long long>> htabs = {h_tab}, hkmixs = {{}};
    std::vector<int> rv = {NE};
    cache.init(scfg, 1, EB, MAX_M, lids, hf1, df1, hf2, df2, htabs, hkmixs, rv, st);

    qw38::MoeWeights w_str = w_ref;
    w_str.t_fc1 = d_uva_fc1;
    w_str.t_fc2 = d_uva_fc2;
    w_str.t_ptab = cache.layer_ptab(0);
    w_str.d_exp_hits = cache.layer_hits(0);
    w_str.strata_ctx = &cache;
    w_str.strata_li = 0;
    w_str.strata_pf_fn = &qw38rt::StrataExpertCache::prefill_moe_cb;

    __nv_bfloat16* d_x = rand_bf16(MAX_M * geom::HID, 0.5f);
    __nv_bfloat16 *d_out_ref = nullptr, *d_out_str = nullptr;
    CK(cudaMalloc(&d_out_ref, MAX_M * geom::HID * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_out_str, MAX_M * geom::HID * sizeof(__nv_bfloat16)));

    auto compare_out = [&](const char* label, int M) {
        std::vector<uint16_t> a(size_t(M) * geom::HID), b(size_t(M) * geom::HID);
        CK(cudaMemcpy(a.data(), d_out_ref, a.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), d_out_str, b.size() * 2, cudaMemcpyDeviceToHost));
        size_t diffs = 0;
        for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++diffs;
        printf("[%s] M=%3d: byte_diffs=%zu -> %s\n", label, M, diffs, diffs == 0 ? "PASS" : "FAIL");
        if (diffs) exit(1);
    };

    // --- TEST 2: Decode M=1 and M=5 inside CUDA Graph across 10 steps with live async swaps ---
    for (int M : {1, 5}) {
        qw38::moe_forward(blas, w_ref, sc, d_x, d_out_ref, M, st);
        CK(cudaStreamSynchronize(st));

        cudaGraph_t g = nullptr;
        cudaGraphExec_t ge = nullptr;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
        qw38::moe_forward(blas, w_str, sc, d_x, d_out_str, M, st);
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        CK(cudaGraphDestroy(g));

        for (int step = 0; step < 10; ++step) {
            CK(cudaGraphLaunch(ge, st));
            CK(cudaStreamSynchronize(st));
            cache.maintain_after_step(st);
            if (step == 0 || step == 9) {
                char lbl[64];
                snprintf(lbl, sizeof lbl, "Test 2 Graphed Decode step=%d", step);
                compare_out(lbl, M);
            }
        }
        CK(cudaGraphExecDestroy(ge));
    }

    // --- TEST 3: Prefill M=64 and M=256 (double-buffered H2D staging vs 100% VRAM) ---
    for (int M : {64, 256}) {
        qw38::moe_forward(blas, w_ref, sc, d_x, d_out_ref, M, st);
        CK(cudaStreamSynchronize(st));
        qw38::moe_forward(blas, w_str, sc, d_x, d_out_str, M, st);
        CK(cudaStreamSynchronize(st));
        compare_out("Test 3 Double-Buf H2D Prefill", M);
    }

    cache.print_stats();
    printf("ALL QW38 STRATA PARITY TESTS PASSED (0 BYTE DIFFS)!\n");
    return 0;
}
