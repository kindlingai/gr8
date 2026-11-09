// One-GPU bench + bit gate of GR8_DS_ATTN_TC_TILE: the 128-row tile choices of launch_gemm_tc (ds41_attn.cuh) on random MXFP8 operands.
// For each shape it runs tile 0 (k_gemm_mx8t <128,128,2>, today), tile 1 (k_gemm_mx8t_sw <128,128,3>) and tile 2 (k_gemm_persist_ldm_epi_bid <128,128,3>,
// persistent blocks), compares tile 1's and tile 2's Y with tile 0's bit for bit (memcmp of the bf16 output) and prints the time of each (cudaEvent,
// REPS reps after a warm-up). The shapes are the DS41 TP4 attention projections at M = 4096 by default (q_a, kv, q_b, o_a in 4 groups, o_b); pass M
// to change it. A shape where tile 2 fell back to tile 1 (guard failed) is flagged "persist no". Exit 1 if any output differs.
// Build: nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I engine -o ds41_gemm_tc_bench engine/attn/tests/ds41_gemm_tc_bench.cu
// Run:   ds41_gemm_tc_bench [M]   (default 4096)
#include "../ds41_attn.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

__global__ void k_fill(uint8_t* p, size_t n, uint32_t seed, int scale)   // scale = 0: e4m3 codes, never NaN (|code| <= 0x7e); 1: E8M0 bytes 120..127
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
    {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        p[i] = scale ? (uint8_t)(120 + h % 8) : (uint8_t)((h & 0x80) | ((h >> 8) % 0x7f));
    }
}

int main(int argc, char** argv)
{
    const int M = argc > 1 ? atoi(argv[1]) : 4096;
    struct Shape { const char* name; int N, K, groups; };
    const Shape shapes[] = {{"q_a", 1280, 5120, 1}, {"kv", 512, 5120, 1}, {"q_b", 8192, 1280, 1}, {"o_a", 1024, 4096, 4}, {"o_b", 5120, 4096, 1}};
    int optin = 0;
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    printf("M %d; smem opt-in per block %d B; tile 0 <128,128,2> %zu B, tile 1 sw <128,128,3> %zu B, tile 2 persistent <128,128,3> same smem; %d SMs\n", M, optin,
           ds41a::TcCfg<128, 128, 2>::SMEM, ds41a::TcCfgSw<128, 128, 3>::SMEM, ds41a::tc_sm_count());
    if ((size_t)optin < ds41a::TcCfgSw<128, 128, 3>::SMEM) { printf("SKIP: this device cannot run tile 1 (launch_gemm_tc falls back to tile 0)\n"); return 0; }
    bool all = true;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (const Shape& s : shapes)
    {
        const size_t na = (size_t)s.groups * M * s.K, nw = (size_t)s.groups * s.N * s.K, ny = (size_t)s.groups * M * s.N;
        uint8_t *A, *SA, *W, *SW;
        ds41a::bf16* Y[3];
        CK(cudaMalloc(&A, na)); CK(cudaMalloc(&SA, na / 32)); CK(cudaMalloc(&W, nw)); CK(cudaMalloc(&SW, nw / 32));
        for (auto& y : Y) CK(cudaMalloc(&y, ny * 2));
        k_fill<<<1024, 256>>>(A, na, 1u, 0); k_fill<<<256, 256>>>(SA, na / 32, 2u, 1);
        k_fill<<<1024, 256>>>(W, nw, 3u, 0); k_fill<<<256, 256>>>(SW, nw / 32, 4u, 1);
        ds41a::GemmP p{};
        p.A = A; p.SA = SA; p.W = W; p.SW = SW; p.M = M; p.N = s.N; p.K = s.K; p.lda = s.K; p.lsa = s.K / 32; p.ldy = s.N;
        p.a_gs = (size_t)M * s.K; p.sa_gs = (size_t)M * s.K / 32; p.w_gs = (size_t)s.N * s.K; p.sw_gs = (size_t)s.N * s.K / 32; p.y_gs = (size_t)M * s.N;
        float us[3];
        std::vector<uint16_t> h0(ny), h(ny);
        bool eq[3] = {true, true, true};
        for (int tile = 0; tile < 3; ++tile)
        {
            p.Y = Y[tile];
            CK(cudaMemset(Y[tile], 0xff, ny * 2));
            ds41a::g_tc_tile = tile;
            auto run = [&] { ds41a::launch_gemm_tc(p, s.groups, 0); };
            run(); CK(cudaDeviceSynchronize());
            constexpr int REPS = 20;
            CK(cudaEventRecord(e0));
            for (int r = 0; r < REPS; ++r) run();
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1)); us[tile] = ms * 1000.f / REPS;
            CK(cudaMemcpy(tile ? h.data() : h0.data(), Y[tile], ny * 2, cudaMemcpyDeviceToHost));
            if (tile) eq[tile] = memcmp(h.data(), h0.data(), ny * 2) == 0;
            all = all && eq[tile];
        }
        p.Y = Y[2];
        const bool persist = ds41a::launch_tc_persist<128, 128, 3>(p, s.groups, 0);   // does tile 2 take the persistent kernel on this shape?
        CK(cudaDeviceSynchronize());
        const double tf = 2.0 * M * s.N * s.K * s.groups / 1e12;
        printf("%-4s N %5d K %5d g %d | tile0 %8.1f us %6.1f TF/s | tile1 %8.1f us %6.1f TF/s (%+.1f%%) bitEq %s | tile2 %8.1f us %6.1f TF/s (%+.1f%%) bitEq %s persist %s\n",
               s.name, s.N, s.K, s.groups, us[0], tf / us[0] * 1e6, us[1], tf / us[1] * 1e6, (us[0] / us[1] - 1.0) * 100.0, eq[1] ? "yes" : "NO",
               us[2], tf / us[2] * 1e6, (us[0] / us[2] - 1.0) * 100.0, eq[2] ? "yes" : "NO", persist ? "yes" : "no");
        CK(cudaFree(A)); CK(cudaFree(SA)); CK(cudaFree(W)); CK(cudaFree(SW));
        for (auto& y : Y) CK(cudaFree(y));
    }
    printf("%s\n", all ? "PASS: every tile gives tile 0's bits" : "FAIL: a tile differs from tile 0");
    return all ? 0 : 1;
}
