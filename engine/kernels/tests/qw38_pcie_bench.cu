// Micro-benchmark: RTX 5090 sm_120a PCIe Gen4 x16 cold-expert mechanisms for Qwen3.8 K3 pmcgx MoE.
// Compares: (1) 100% VRAM, (2) 100% Zero-Copy UVA via t_ptab, (3) 80/20 Mixed UVA unordered vs cold-first,
// (4) 80/20 Mixed H2D DMA overlapped with hot VRAM experts, (5) Raw H2D cudaMemcpyAsync GB/s.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

#define G6_MODEL_QW38 1
#define G6_NR 1
#include "../qw38_moe.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s failed: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

int main() {
    constexpr int NE = geom::NEXP; // 512
    constexpr int EB = g6nr::EB;   // 5
    constexpr int KBITS = 3;
    constexpr size_t MB = static_cast<size_t>(geom::KSH) * 8 * 32; // 40960 B/bit
    constexpr size_t FC1_PIECE = 2 * KBITS * MB;                  // 245760 B
    constexpr size_t FC2_PIECE = KBITS * MB;                      // 122880 B
    constexpr size_t EXP_FC1 = EB * FC1_PIECE;                    // 1,228,800 B
    constexpr size_t EXP_FC2 = EB * FC2_PIECE;                    // 614,400 B
    constexpr size_t EXP_BYTES = EXP_FC1 + EXP_FC2;               // 1,843,200 B (1.758 MiB)

    // Allocate 64 experts in VRAM and 64 experts in mapped pinned host RAM
    constexpr int N_TEST = 64;
    uint8_t* d_vram_fc1 = nullptr;
    uint8_t* d_vram_fc2 = nullptr;
    CK(cudaMalloc(&d_vram_fc1, N_TEST * EXP_FC1));
    CK(cudaMalloc(&d_vram_fc2, N_TEST * EXP_FC2));
    CK(cudaMemset(d_vram_fc1, 0x5A, N_TEST * EXP_FC1));
    CK(cudaMemset(d_vram_fc2, 0xA5, N_TEST * EXP_FC2));

    uint8_t* h_pin_fc1 = nullptr;
    uint8_t* h_pin_fc2 = nullptr;
    CK(cudaHostAlloc(&h_pin_fc1, N_TEST * EXP_FC1, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&h_pin_fc2, N_TEST * EXP_FC2, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(h_pin_fc1, 0x5A, N_TEST * EXP_FC1);
    std::memset(h_pin_fc2, 0xA5, N_TEST * EXP_FC2);
    uint8_t* d_uva_fc1 = nullptr;
    uint8_t* d_uva_fc2 = nullptr;
    CK(cudaHostGetDevicePointer(&d_uva_fc1, h_pin_fc1, 0));
    CK(cudaHostGetDevicePointer(&d_uva_fc2, h_pin_fc2, 0));

    // Staging buffer in VRAM for H2D DMA test (up to 32 cold experts)
    uint8_t* d_stg_fc1 = nullptr;
    uint8_t* d_stg_fc2 = nullptr;
    CK(cudaMalloc(&d_stg_fc1, 32 * EXP_FC1));
    CK(cudaMalloc(&d_stg_fc2, 32 * EXP_FC2));

    cudaStream_t st, h2d_st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&h2d_st, cudaStreamNonBlocking));
    cudaEvent_t ev0, ev1, ev_dma;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));
    CK(cudaEventCreateWithFlags(&ev_dma, cudaEventDisableTiming));

    // 1. Raw H2D DMA bandwidth test (warm up PCIe link first!)
    for (int w = 0; w < 5; ++w)
        CK(cudaMemcpyAsync(d_vram_fc1, h_pin_fc1, 32 * EXP_FC1, cudaMemcpyHostToDevice, h2d_st));
    CK(cudaStreamSynchronize(h2d_st));
    {
        const size_t bytes = 32 * EXP_BYTES; // 59.0 MB
        CK(cudaEventRecord(ev0, h2d_st));
        for (int it = 0; it < 20; ++it) {
            CK(cudaMemcpyAsync(d_vram_fc1, h_pin_fc1, 32 * EXP_FC1, cudaMemcpyHostToDevice, h2d_st));
            CK(cudaMemcpyAsync(d_vram_fc2, h_pin_fc2, 32 * EXP_FC2, cudaMemcpyHostToDevice, h2d_st));
        }
        CK(cudaEventRecord(ev1, h2d_st));
        CK(cudaEventSynchronize(ev1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, ev0, ev1));
        printf("Raw H2D DMA (59 MB x 20): %.2f ms total -> %.2f GB/s\n", ms, (20.0 * bytes / 1e9) / (ms / 1e3));
    }

    // Build tr3.table, scales, scratch for M=5 verify window (35 unique experts = realistic M=5!)
    std::vector<long long> h_tab(NE * 6, 0);
    for (int e = 0; e < NE; ++e) {
        h_tab[e * 6 + 0] = 0;
        h_tab[e * 6 + 1] = 0;
        h_tab[e * 6 + 2] = static_cast<long long>(e) * EB;
        h_tab[e * 6 + 3] = EB;
        h_tab[e * 6 + 4] = KBITS;
        h_tab[e * 6 + 5] = 0;
    }
    long long* d_tab = nullptr;
    __half *d_sg = nullptr, *d_v1b = nullptr, *d_v2 = nullptr, *d_v3b = nullptr;
    CK(cudaMalloc(&d_tab, NE * 6 * sizeof(long long)));
    CK(cudaMemcpy(d_tab, h_tab.data(), NE * 6 * sizeof(long long), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_sg, 2 * geom::HID * sizeof(__half)));
    CK(cudaMalloc(&d_v1b, NE * EB * 2 * geom::HB * sizeof(__half)));
    CK(cudaMalloc(&d_v2, NE * EB * 3 * 128 * sizeof(__half)));
    CK(cudaMalloc(&d_v3b, NE * EB * geom::HB * sizeof(__half)));
    CK(cudaMemset(d_sg, 0, 2 * geom::HID * sizeof(__half)));
    CK(cudaMemset(d_v1b, 0, NE * EB * 2 * geom::HB * sizeof(__half)));
    CK(cudaMemset(d_v2, 0, NE * EB * 3 * 128 * sizeof(__half)));
    CK(cudaMemset(d_v3b, 0, NE * EB * geom::HB * sizeof(__half)));

    const int M = 5;
    const int n_routes = M * geom::TOPK; // 50 routes
    int *d_topk = nullptr, *d_ng = nullptr, *d_gexp = nullptr, *d_gcnt = nullptr, *d_grt = nullptr;
    float *d_tw = nullptr, *d_fc1p = nullptr, *d_fc2p = nullptr;
    __half *d_xhad = nullptr, *d_fc1a = nullptr;
    CK(cudaMalloc(&d_topk, n_routes * sizeof(int)));
    CK(cudaMalloc(&d_tw, n_routes * sizeof(float)));
    CK(cudaMalloc(&d_ng, sizeof(int)));
    CK(cudaMalloc(&d_gexp, n_routes * sizeof(int)));
    CK(cudaMalloc(&d_gcnt, n_routes * sizeof(int)));
    CK(cudaMalloc(&d_grt, n_routes * 8 * sizeof(int)));
    CK(cudaMalloc(&d_xhad, M * geom::HID * sizeof(__half)));
    CK(cudaMalloc(&d_fc1p, 8 * geom::TOPK * EB * 2 * 2 * 128 * sizeof(float)));
    CK(cudaMalloc(&d_fc1a, n_routes * EB * 128 * sizeof(__half)));
    CK(cudaMalloc(&d_fc2p, n_routes * geom::HID * sizeof(float)));
    CK(cudaMemset(d_xhad, 0, M * geom::HID * sizeof(__half)));

    const uint8_t** d_ptab = nullptr;
    CK(cudaMalloc(&d_ptab, NE * EB * 2 * sizeof(void*)));
    std::vector<const uint8_t*> h_ptab(NE * EB * 2, nullptr);

    // Helper: set up `n_groups` unique experts (e = 0..n_groups-1), first `n_cold` in UVA, rest in VRAM
    auto bench_case = [&](const char* label, int n_groups, int n_cold, bool cold_first, bool use_dma) {
        for (int e = 0; e < NE; ++e) {
            const int idx = e % N_TEST;
            const bool is_cold = (e < n_cold);
            const uint8_t* f1_base = (is_cold && !use_dma) ? (d_uva_fc1 + idx * EXP_FC1) : (d_vram_fc1 + idx * EXP_FC1);
            const uint8_t* f2_base = (is_cold && !use_dma) ? (d_uva_fc2 + idx * EXP_FC2) : (d_vram_fc2 + idx * EXP_FC2);
            if (is_cold && use_dma) {
                f1_base = d_stg_fc1 + (e % 32) * EXP_FC1;
                f2_base = d_stg_fc2 + (e % 32) * EXP_FC2;
            }
            for (int b = 0; b < EB; ++b) {
                h_ptab[(e * EB + b) * 2 + 0] = f1_base + b * FC1_PIECE;
                h_ptab[(e * EB + b) * 2 + 1] = f2_base + b * FC2_PIECE;
            }
        }
        CK(cudaMemcpy(d_ptab, h_ptab.data(), h_ptab.size() * sizeof(void*), cudaMemcpyHostToDevice));

        std::vector<int> gexp(n_groups), gcnt(n_groups, 1), grt(n_groups * 8, 0), topk(n_routes, 0);
        for (int g = 0; g < n_groups; ++g) {
            int e = cold_first ? g : (n_groups - 1 - g); // cold are 0..n_cold-1
            gexp[g] = e;
            grt[g * 8 + 0] = g % n_routes;
            if (g < n_routes) topk[g] = e;
        }
        CK(cudaMemcpy(d_ng, &n_groups, sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_gexp, gexp.data(), n_groups * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_gcnt, gcnt.data(), n_groups * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_grt, grt.data(), n_groups * 8 * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_topk, topk.data(), n_routes * sizeof(int), cudaMemcpyHostToDevice));

        auto run_once = [&]() {
            if (use_dma && n_cold > 0) {
                for (int c = 0; c < n_cold; ++c) {
                    CK(cudaMemcpyAsync(d_stg_fc1 + c * EXP_FC1, h_pin_fc1 + c * EXP_FC1, EXP_FC1, cudaMemcpyHostToDevice, h2d_st));
                    CK(cudaMemcpyAsync(d_stg_fc2 + c * EXP_FC2, h_pin_fc2 + c * EXP_FC2, EXP_FC2, cudaMemcpyHostToDevice, h2d_st));
                }
                CK(cudaEventRecord(ev_dma, h2d_st));
                CK(cudaStreamWaitEvent(st, ev_dma, 0));
            }
            qw38::k_qw38_moe_fc1_ks2<<<n_groups * EB * 4, 256, 0, st>>>(
                d_xhad, d_ng, d_gexp, d_gcnt, d_grt, d_tab, nullptr, nullptr, d_ptab, d_v1b, d_fc1p, EB, 1);
            qw38::k_qw38_moe_fc1_epilogue_ks2<<<n_routes * EB, 32, 0, st>>>(
                d_fc1p, d_topk, d_tab, d_v2, d_fc1a, M, EB);
            qw38::k_qw38_moe_fc2_grp<<<n_groups * geom::HB, 256, 0, st>>>(
                d_fc1a, d_ng, d_gexp, d_gcnt, d_grt, d_tw, d_tab, nullptr, d_ptab, d_sg, d_v3b, d_fc2p, EB, 1);
        };

        for (int w = 0; w < 5; ++w) run_once();
        CK(cudaStreamSynchronize(st));
        const int ITERS = 50;
        CK(cudaEventRecord(ev0, st));
        for (int it = 0; it < ITERS; ++it) run_once();
        CK(cudaEventRecord(ev1, st));
        CK(cudaEventSynchronize(ev1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, ev0, ev1));
        const double us = (ms * 1000.0) / ITERS;
        const double cold_gb = (double(n_cold) * EXP_BYTES) / 1e9;
        const double eff_pcie_gbs = n_cold > 0 ? (cold_gb / (us / 1e6)) : 0.0;
        const double full48_ms = us * 48.0 / 1000.0;
        printf("%-36s | G=%2d cold=%2d | %6.1f us/layer (%5.2f ms / 48L) | eff PCIe=%5.2f GB/s\n",
               label, n_groups, n_cold, us, full48_ms, eff_pcie_gbs);
    };

    printf("=== M=5 Verify Window (35 unique experts/layer, 1.758 MiB/expert) ===\n");
    bench_case("100% VRAM (GDDR7 roofline)",          35,  0, true,  false);
    bench_case("100% UVA (all 35 over PCIe)",         35, 35, true,  false);
    bench_case("80% hit: 7 cold UVA (hot-first)",     35,  7, false, false);
    bench_case("80% hit: 7 cold UVA (cold-first)",    35,  7, true,  false);
    bench_case("80% hit: 7 cold H2D DMA (sync-wait)", 35,  7, true,  true);
    bench_case("85% hit: 5 cold UVA (cold-first)",    35,  5, true,  false);
    bench_case("85% hit: 5 cold H2D DMA (sync-wait)", 35,  5, true,  true);
    bench_case("90% hit: 3 cold UVA (cold-first)",    35,  3, true,  false);
    bench_case("90% hit: 3 cold H2D DMA (sync-wait)", 35,  3, true,  true);
    bench_case("70% hit: 10 cold UVA (cold-first)",   35, 10, true,  false);
    bench_case("70% hit: 10 cold H2D DMA",            35, 10, true,  true);

    return 0;
}
