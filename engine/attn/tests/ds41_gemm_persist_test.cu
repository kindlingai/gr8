// One-GPU exactness gate of GR8_DS_ATTN_TC_TILE=2 (k_gemm_persist_ldm_epi_bid, ds41_gemm_persist.cuh): tile 2 against tile 1 AND tile 0, through launch_gemm_tc.
// Every case runs the three tiles on the same random MXFP8 operands, Y poisoned 0xff beforehand, and compares ALL bytes of the Y allocation (a row >= M written,
// a byte not written, or a wrong value is a mismatch against tiles 0 and 1, which write rows < M only). Tile 2 runs inside a captured CUDA graph (instantiated and
// launched), so the case also shows that the launcher is capture-safe; the captured kernel node says which kernel the launcher chose and with which grid, and that is
// checked against what the case expects (persistent kernel, or the fallback when the shape guard or the 128 x 128 wide-tile rule says so).
// Cases:
//   1. partial M tiles, short K and several tiles a CTA: M {200,129,255,130,128,1,384} x K {128,256,384,1024} x grid {SM count, 7, 13} (g_tc_persist_grid caps the grid so
//      each CTA walks several tiles and the cp.async ring and the staging slot cross tile boundaries), N 3072, 2 groups, dense layout.
//   2. the three served layouts: wq_b (M 8192, N 8192, K 1280), wo_a (M 8192, N 1024, K 4096, 2 groups, A / SA / Y interleaved by group), wo_b (M 2048, N 5120, K 2048,
//      lda = lsa = 2112, the scales at A + K), each at the SM count and at a forced grid of 13.
//   3. two guard cases that must take the fallback (tile 1) and stay exact: Y offset by 8 bytes, and ldy % 8 != 0.
// Exit 1 on any mismatch or unexpected kernel; prints "PERSIST PASS" / "PERSIST FAIL" with the case counts.
// Build: nvcc -O3 -std=c++17 -gencode=arch=compute_120a,code=sm_120a -DG6_MODEL_DS41=1 -I engine -o ds41_gemm_persist_test engine/attn/tests/ds41_gemm_persist_test.cu
//        (sm_121a on a GB10 box). Run: ds41_gemm_persist_test
#include "../ds41_attn.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

__device__ __forceinline__ uint8_t fill_byte(size_t i, uint32_t seed, int scale)   // scale = 0: e4m3 codes, never NaN (|code| <= 0x7e); 1: E8M0 bytes 120..127
{
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return scale ? (uint8_t)(120 + h % 8) : (uint8_t)((h & 0x80) | ((h >> 8) % 0x7f));
}
__global__ void k_fill(uint8_t* p, size_t n, uint32_t seed, int scale)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x) p[i] = fill_byte(i, seed, scale);
}
// scale bytes inside a row-interleaved buffer: bytes [off, off + n) of each of `rows` rows of `stride` bytes
__global__ void k_fill_rows(uint8_t* p, size_t rows, size_t stride, size_t off, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < rows * n; i += 256ull * gridDim.x) p[(i / n) * stride + off + i % n] = fill_byte(i, seed, 1);
}

struct Case
{
    const char* name;
    int M, N, K, groups, lda, lsa, ldy;
    size_t a_gs, sa_gs, y_gs;
    bool sa_in_a;       // SA = A + K (the wo_b layout: codes then scales in one row of lda bytes)
    int y_off;          // bytes added to the Y base (16-B alignment guard)
    int grid_cap;       // g_tc_persist_grid: 0 = the SM count
};

static int g_cases = 0, g_bad = 0, g_persist = 0, g_fallback = 0;

static void run_case(const Case& c)
{
    using namespace ds41a;
    const size_t G = c.groups;
    const size_t a_bytes = (size_t)(c.M - 1) * c.lda + (G - 1) * c.a_gs + c.K + (c.sa_in_a ? 0 : 0) + 64;
    const size_t sa_bytes = (size_t)(c.M - 1) * c.lsa + (G - 1) * c.sa_gs + c.K / 32 + 64;
    const size_t w_bytes = G * c.N * (size_t)c.K, sw_bytes = w_bytes / 32;
    const size_t y_elems = (size_t)(c.M - 1) * c.ldy + c.N + (G - 1) * c.y_gs + 8;
    const size_t y_bytes = y_elems * 2 + 16;
    uint8_t *A, *SAb, *W, *SW, *Yb[3];
    CK(cudaMalloc(&A, a_bytes + (c.sa_in_a ? 0 : 0))); CK(cudaMalloc(&SAb, sa_bytes)); CK(cudaMalloc(&W, w_bytes)); CK(cudaMalloc(&SW, sw_bytes));
    for (auto& y : Yb) CK(cudaMalloc(&y, y_bytes));
    k_fill<<<1024, 256>>>(A, a_bytes, 1u + c.M, 0);
    k_fill<<<1024, 256>>>(W, w_bytes, 3u + c.K, 0);
    k_fill<<<256, 256>>>(SW, sw_bytes, 4u, 1);
    const uint8_t* SA = SAb;
    if (c.sa_in_a)
    {
        for (size_t g = 0; g < G; g++) k_fill_rows<<<256, 256>>>(A + g * c.sa_gs, c.M, c.lsa, c.K, c.K / 32, 2u + (uint32_t)g);   // scales of row r at A + K + r * lsa (+ group)
        SA = A + c.K;
    }
    else k_fill<<<256, 256>>>(SAb, sa_bytes, 2u, 1);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());   // the fills run on the default stream, the tiles on a non-blocking one (no implicit ordering)
    GemmP p{};
    p.A = A; p.SA = SA; p.W = W; p.SW = SW; p.M = c.M; p.N = c.N; p.K = c.K; p.lda = c.lda; p.lsa = c.lsa; p.ldy = c.ldy;
    p.a_gs = c.a_gs; p.sa_gs = c.sa_gs; p.w_gs = (size_t)c.N * c.K; p.sw_gs = (size_t)c.N * c.K / 32; p.y_gs = c.y_gs;
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const void* ran[3] = {nullptr, nullptr, nullptr};
    unsigned grid2 = 0;
    for (int tile = 0; tile < 3; ++tile)
    {
        CK(cudaMemset(Yb[tile], 0xff, y_bytes));
        CK(cudaDeviceSynchronize());
        p.Y = (bf16*)(Yb[tile] + c.y_off);
        g_tc_tile = tile;
        g_tc_persist_grid = tile == 2 ? c.grid_cap : 0;
        // all three tiles through a captured graph: the node tells which kernel the launcher chose
        cudaGraph_t gr;
        cudaGraphExec_t ex;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        launch_gemm_tc(p, c.groups, s);
        CK(cudaStreamEndCapture(s, &gr));
        size_t nn = 0;
        CK(cudaGraphGetNodes(gr, nullptr, &nn));
        std::vector<cudaGraphNode_t> nodes(nn);
        CK(cudaGraphGetNodes(gr, nodes.data(), &nn));
        cudaKernelNodeParams kp{};
        if (nn != 1 || cudaGraphKernelNodeGetParams(nodes[0], &kp) != cudaSuccess) { printf("%s: the launcher put %zu graph nodes (want 1 kernel)\n", c.name, nn); exit(2); }
        ran[tile] = kp.func;
        if (tile == 2) grid2 = kp.gridDim.x;
        CK(cudaGraphInstantiate(&ex, gr, 0));
        CK(cudaGraphLaunch(ex, s));
        CK(cudaStreamSynchronize(s));
        CK(cudaGraphExecDestroy(ex)); CK(cudaGraphDestroy(gr));
    }
    g_tc_tile = 0; g_tc_persist_grid = 0;
    CK(cudaGetLastError());
    std::vector<uint8_t> h[3];
    for (int t = 0; t < 3; ++t) { h[t].resize(y_bytes); CK(cudaMemcpy(h[t].data(), Yb[t], y_bytes, cudaMemcpyDeviceToHost)); }
    const bool eq1 = memcmp(h[2].data(), h[1].data(), y_bytes) == 0, eq0 = memcmp(h[2].data(), h[0].data(), y_bytes) == 0, t10 = memcmp(h[1].data(), h[0].data(), y_bytes) == 0;
    size_t poison = 0;
    for (size_t i = 0; i < y_bytes; i++) poison += h[0][i] == 0xff;
    // what the launcher must choose for tile 2: the persistent kernel on the 128 x 128 wide tile when the guard passes, else what tile 1 runs
    const long ctas128 = (long)(c.N / 128) * ((c.M + 127) / 128) * c.groups;
    const bool wide128 = c.M >= 128 && c.N % 128 == 0 && ctas128 >= 96;
    const bool guard = c.ldy % 8 == 0 && (c.groups <= 1 || c.y_gs % 8 == 0) && c.y_off % 16 == 0;
    const bool want_persist = wide128 && guard;
    const bool is_persist = ran[2] == (const void*)k_gemm_persist_ldm_epi_bid<128, 128, 3>;
    const bool path_ok = is_persist == want_persist && (is_persist || ran[2] == ran[1]) && ran[1] != (const void*)k_gemm_persist_ldm_epi_bid<128, 128, 3>
                         && (!is_persist || (grid2 == (unsigned)(ctas128 < (c.grid_cap ? c.grid_cap : tc_sm_count()) ? ctas128 : (c.grid_cap ? c.grid_cap : tc_sm_count()))));
    const bool ok = eq1 && eq0 && t10 && path_ok;
    printf("%-24s M %5d N %5d K %5d g %zu grid %-3s: %s %s | vs tile1 %s vs tile0 %s (tile1 vs tile0 %s) | Y 0xff bytes %zu%s\n", c.name, c.M, c.N, c.K, G,
           c.grid_cap ? std::to_string(c.grid_cap).c_str() : "sm", is_persist ? "persist" : "fallback", path_ok ? "as expected" : "UNEXPECTED KERNEL/GRID",
           eq1 ? "EXACT" : "DIFF", eq0 ? "EXACT" : "DIFF", t10 ? "EXACT" : "DIFF", poison, ok ? "" : "  <-- FAIL");
    g_cases++;
    g_bad += !ok;
    g_persist += is_persist;
    g_fallback += !is_persist;
    CK(cudaStreamDestroy(s));
    CK(cudaFree(A)); CK(cudaFree(SAb)); CK(cudaFree(W)); CK(cudaFree(SW));
    for (auto& y : Yb) CK(cudaFree(y));
}

int main()
{
    int optin = 0;
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    printf("SMs %d; smem opt-in per block %d B; tile 2 needs %zu B\n", ds41a::tc_sm_count(), optin, ds41a::TcCfgSw<128, 128, 3>::SMEM);
    if ((size_t)optin < ds41a::TcCfgSw<128, 128, 3>::SMEM) { printf("SKIP: this device cannot run tile 2 (launch_gemm_tc falls back)\n"); return 0; }
    const int Ms[] = {200, 129, 255, 130, 128, 1, 384}, Ks[] = {128, 256, 384, 1024}, grids[] = {0, 7, 13};
    for (int K : Ks)
        for (int M : Ms)
            for (int gc : grids)
            {
                const int N = 3072;
                run_case({"dense", M, N, K, 2, K, K / 32, N, (size_t)M * K, (size_t)M * K / 32, (size_t)M * N, false, 0, gc});
            }
    for (int gc : {0, 13})
    {
        run_case({"wq_b", 8192, 8192, 1280, 1, 1280, 40, 8192, 0, 0, 0, false, 0, gc});
        run_case({"wo_a", 8192, 1024, 4096, 2, 8192, 256, 2048, 4096, 128, 1024, false, 0, gc});
        run_case({"wo_b", 2048, 5120, 2048, 1, 2112, 2112, 5120, 0, 0, 0, true, 0, gc});
    }
    run_case({"guard Y+8 bytes", 2048, 5120, 2048, 1, 2112, 2112, 5120, 0, 0, 0, true, 8, 0});
    run_case({"guard ldy % 8 != 0", 1000, 3072, 1024, 2, 1024, 32, 3072 + 2, (size_t)1000 * 1024, (size_t)1000 * 32, (size_t)1000 * (3072 + 2), false, 0, 0});
    printf("%d cases (%d took the persistent kernel, %d the fallback), %d bad\n", g_cases, g_persist, g_fallback, g_bad);
    printf(g_bad ? "PERSIST FAIL\n" : "PERSIST PASS\n");
    return g_bad ? 1 : 0;
}
