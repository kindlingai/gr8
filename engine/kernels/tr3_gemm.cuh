// Item 3b: TR3 prefill expert GEMM. Y[rows, N] = X[rows, K] @ What_e[K, N], trellis decoded straight
// into mma B fragments (no dequantized copy in memory), A rows streamed through smem with cp.async.
// One CTA = (128 output cols, one segment of <= 64 rows of one expert). 8 warps, one 16-col tile each.
// X is the already-rotated input (Hadamard/sign vectors are separate small passes, not timed here).
#pragma once
#include "tr3.cuh"

namespace pf
{
constexpr int MB = 4;                  // 16-row blocks per segment (64 rows)
constexpr int STAGES = 4;
constexpr int KS16 = 2;                // k16 steps per stage
constexpr int A_STAGE = MB * 16 * KS16 * 32;        // 4096 B
constexpr int B_STAGE_MAX = KS16 * 8 * 128;         // 2048 B (K4)
constexpr int SMEM = STAGES * (A_STAGE + B_STAGE_MAX);

__device__ __forceinline__ void cp16(void* smem, const void* gmem)
{
    const u32 s = static_cast<u32>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }
__device__ __forceinline__ void ldsm4(u32 (&r)[4], const void* smem)
{
    const u32 s = static_cast<u32>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

template <int BITS>
__device__ __forceinline__ void body(const __half* __restrict__ X, const uint8_t* __restrict__ tr, __half* __restrict__ Y,
                                     int K, int N, int row0, int rows, unsigned char* smem)
{
    constexpr int TB = 32 * BITS;
    constexpr int CPT = TB / 16;                       // 16-byte chunks per tile
    constexpr int B_CHUNKS = KS16 * 8 * CPT;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int ntile = blockIdx.x;                      // 128-col tile
    const int NT16 = N / 16;
    const int S = K / (16 * KS16);
    unsigned char* sA = smem;
    unsigned char* sB = smem + STAGES * A_STAGE;

    auto load = [&](int slot, int s)
    {
        const int r = t >> 2, ch = t & 3;              // 256 threads = 64 rows x 4 chunks
        if (r < rows)
            cp16(sA + slot * A_STAGE + r * 64 + ((ch ^ ((r >> 1) & 3)) << 4), X + static_cast<size_t>(row0 + r) * K + s * (16 * KS16) + ch * 8);
        if (t < B_CHUNKS)
        {
            const int j = t / (8 * CPT), rem = t % (8 * CPT);
            cp16(sB + slot * B_STAGE_MAX + t * 16,
                 tr + (static_cast<size_t>(s * KS16 + j) * NT16 + ntile * 8) * TB + rem * 16);
        }
    };

    #pragma unroll
    for (int p = 0; p < STAGES - 1; ++p)
    {
        if (p < S) load(p, p);
        cp_commit();
    }
    Tr3Lane<BITS> ld;
    ld.init(lane);
    float acc[MB][2][4];
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][h][i] = 0.f;
    const int arow = (lane & 7) + 8 * ((lane >> 3) & 1), achunk = lane >> 4;

    for (int s = 0; s < S; ++s)
    {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (s + STAGES - 1 < S) load((s + STAGES - 1) % STAGES, s + STAGES - 1);
        cp_commit();
        const unsigned char* a = sA + (s % STAGES) * A_STAGE;
        const unsigned char* b = sB + (s % STAGES) * B_STAGE_MAX;
        #pragma unroll
        for (int j = 0; j < KS16; ++j)
        {
            const u32* tile = reinterpret_cast<const u32*>(b + (j * 8 + warp) * TB);
            u32 f0[2], f1[2];
            ld.decode(tile[ld.i0], tile[ld.i1], f0, f1);
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                if (mb * 16 < rows)
                {
                    const int r = mb * 16 + arow, ch = j * 2 + achunk;
                    u32 fa[4];
                    ldsm4(fa, a + r * 64 + ((ch ^ ((r >> 1) & 3)) << 4));
                    mma16816(acc[mb][0], fa[0], fa[1], fa[2], fa[3], f0[0], f0[1]);
                    mma16816(acc[mb][1], fa[0], fa[1], fa[2], fa[3], f1[0], f1[1]);
                }
            }
        }
    }
    cp_wait<0>();

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const int col = ntile * 128 + warp * 16 + h * 8 + 2 * c;
            const int r0 = mb * 16 + r;
            if (r0 < rows) *reinterpret_cast<u32*>(Y + static_cast<size_t>(row0 + r0) * N + col) = pack_h2(acc[mb][h][0], acc[mb][h][1]);
            if (r0 + 8 < rows) *reinterpret_cast<u32*>(Y + static_cast<size_t>(row0 + r0 + 8) * N + col) = pack_h2(acc[mb][h][2], acc[mb][h][3]);
        }
}
}  // namespace pf

// segs: [nseg][3] = (expert, row0, rows)
__global__ __launch_bounds__(256) void tr3_gemm(const __half* X, const uint8_t* pool, size_t stride, int bits, const int* segs,
                                               __half* Y, int K, int N)
{
    extern __shared__ __align__(16) unsigned char smem_pf[];
    const int* sg = segs + blockIdx.y * 3;
    if (bits == 3) pf::body<3>(X, pool + sg[0] * stride, Y, K, N, sg[1], sg[2], smem_pf);
    else pf::body<4>(X, pool + sg[0] * stride, Y, K, N, sg[1], sg[2], smem_pf);
}

#ifdef HOST_CMDS
static int cmd_prefill(const Args& a)
{
    const int K = a.i("K", HID), N = a.i("N", 768), bits = a.i("bits", 3);
    const int E = a.i("experts", 256), rpe = a.i("rpe", 128), P = a.i("pool", 64), reps = a.i("reps", 5);
    const std::string dump = a.s("dump", "");
    const int TB = 32 * bits;
    if (K % 32 || N % 128) { fprintf(stderr, "bad shape\n"); return 1; }
    const size_t stride = (size_t)(K / 16) * (N / 16) * TB;
    const long R = (long)E * rpe;
    Rng rng(a.i("seed", 1));
    std::vector<uint8_t> pool((size_t)P * stride);
    rng.fill(pool.data(), pool.size());
    std::vector<uint16_t> x((size_t)R * K);
    {
        // cheap fill: small finite halves
        std::vector<uint16_t> pat = rand_half(rng, 65536, 0.f, 0.05f, true);
        for (size_t i = 0; i < x.size(); ++i) x[i] = pat[(i * 2654435761u + (i >> 16)) & 65535];
    }
    std::vector<int> segs;
    for (int e = 0; e < E; ++e)
        for (int r = 0; r < rpe; r += 64) { segs.push_back(e % P); segs.push_back(e * rpe + r); segs.push_back(std::min(64, rpe - r)); }
    const int nseg = (int)segs.size() / 3;
    const uint8_t* dpool = dev_upload(pool);
    const __half* dx = (const __half*)dev_upload(x);
    const int* dsegs = dev_upload(segs);
    __half* y = (__half*)dev_zero<uint16_t>((size_t)R * N);
    dim3 grid(N / 128, nseg);
    auto go = [&]() { tr3_gemm<<<grid, 256, pf::SMEM>>>(dx, dpool, stride, bits, dsegs, y, K, N); };
    go();
    CK(cudaDeviceSynchronize());
    if (!dump.empty())
    {
        FILE* f = fopen((dump + "/meta.txt").c_str(), "w");
        fprintf(f, "K=%d\nN=%d\nbits=%d\nexperts=%d\nrpe=%d\npool=%d\n", K, N, bits, E, rpe, P);
        fclose(f);
        dump_host(dump, "x.bin", x.data(), x.size());
        dump_host(dump, "pool.bin", pool.data(), pool.size());
        dump_host(dump, "segs.bin", segs.data(), segs.size());
        dump_dev(dump, "y.bin", (uint16_t*)y, (size_t)R * N);
        printf("{\"cmd\":\"prefill\",\"dumped\":\"%s\"}\n", dump.c_str());
        return 0;
    }
    std::vector<double> t;
    for (int r = 0; r < reps; ++r)
    {
        const double t0 = now_us();
        go();
        CK(cudaDeviceSynchronize());
        t.push_back(now_us() - t0);
    }
    const Stats s = stats(t);
    const double flop = 2.0 * R * K * N;
    printf("{\"cmd\":\"prefill\",\"K\":%d,\"N\":%d,\"bits\":%d,\"experts\":%d,\"rows_per_expert\":%d,\"rows\":%ld,\"ms_med\":%.3f,\"ms_min\":%.3f,"
           "\"tflops_med\":%.1f,\"tflops_best\":%.1f,\"weight_mb\":%.1f}\n",
           K, N, bits, E, rpe, R, s.med / 1e3, s.mn / 1e3, flop / s.med / 1e6, flop / s.mn / 1e6, (double)E * stride / 1e6);
    return 0;
}
#endif
