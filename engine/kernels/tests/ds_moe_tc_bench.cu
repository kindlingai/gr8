// Microbench + accuracy gate of the tensor-core read-once prefill MoE (ds_moe_tc.cuh) against the fp16 read-once kernels (ds_prefill_ro.cuh),
// one GPU, the main model's TP4 shapes (hidden 5120, I-slice 576, 384 experts, top-6), the production plan (k_route_ro / k_scan_ro / k_place_ro).
// Routing is synthetic: random logits over a chosen set of D experts (the others get a -1e9 selection bias), with a per-expert bias spread so some
// experts are heavy, the way a real prompt routes (docs/DS41-TRELLIS.md: 58-137 distinct experts per 1024-row chunk). Weights: random records of
// all 384 experts (1.8 GB), nibbles skewed to small codes; x: random bf16 in [-4, 4). Not a model: kernel against kernel.
//   fc1 / fc2 of each path are timed separately (cudaEvent, REPS reps after a warm-up), FLOP = 2 x routes x (1152 x 5120 | 5120 x 576);
//   accuracy = the TC path's act and ypart against the fp16 path's over the live columns: relative RMS and max |diff| / max |ref|.
// Build: nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -I engine/kernels -o ds_moe_tc_bench engine/kernels/tests/ds_moe_tc_bench.cu
// Run:   ds_moe_tc_bench [--stream] [--pool N] [M D]...   (default: 1024 x {60, 100, 140}, 2048 x {100, 140})
//        --stream: DRAM stream probes over the weight bytes first; --pool N: every slot's weights come from expert id % N (N <= 4: L2-resident)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../ds_ro_cfg.h"
#include "../ds_prefill_ro.cuh"
#include "../ds_shared.cuh"
#include "../ds_moe_tc.cuh"

// the TC kernels' shape: the production values (ds_ro_cfg.h DS41_TC_*), override with -DMXTC_*
#ifndef MXTC_CW
#define MXTC_CW DS41_TC_CW
#endif
#ifndef MXTC_NTG1
#define MXTC_NTG1 DS41_TC_NTG1
#endif
#ifndef MXTC_ST
#define MXTC_ST DS41_TC_ST1
#endif
#ifndef MXTC_GY1
#define MXTC_GY1 DS41_TC_GY1
#endif
#ifndef MXTC_W2
#define MXTC_W2 DS41_TC_W2
#endif
#ifndef MXTC_WT
#define MXTC_WT DS41_TC_WT
#endif
#ifndef MXTC_TPC
#define MXTC_TPC DS41_TC_TPC
#endif
#ifndef MXTC_ST2
#define MXTC_ST2 DS41_TC_ST2
#endif
#ifndef MXTC_GY2
#define MXTC_GY2 DS41_TC_GY2
#endif
#ifndef MXTC_OS2
#define MXTC_OS2 DS41_TC_OS2
#endif
#ifndef MXTC_FUSEQ
#define MXTC_FUSEQ 1              // 1: fc1's epilogue quantizes act (the production path; no fp16 act, so no act compare); 0: fp16 act + k_aq
#endif
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

constexpr int K1 = 5120, I = 576, NOUT = 5120, TK = 6, NE = 384, MM = 4096;
constexpr int TILES = (MM * TK + 7 * NE) / 8;
constexpr size_t EB1 = static_cast<size_t>(I / 16) * (K1 / 64) * 2 * mx4::GR, EB2 = static_cast<size_t>(NOUT / 16) * (I / 64) * mx4::GR;
constexpr size_t XFW = static_cast<size_t>(MM / 32 * 4) * (K1 / 16 * 64);
constexpr int REPS = 5;
using P = dsr::PlanMT<TK, MM>;

__device__ __forceinline__ uint32_t hash32(uint32_t i, uint32_t seed)
{
    uint32_t h = i * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return h;
}
__global__ void k_fill_mx4(uint8_t* p, size_t nrec, uint32_t seed)   // codes: two skewed nibbles (|code| small more often), scales 2^-10..2^-3
{
    const size_t tot = nrec * mx4::GR;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < tot; i += 256ull * gridDim.x)
    {
        const uint32_t h = hash32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(i >> 32) * 0x9e3779b1u, seed);
        if (static_cast<int>(i % mx4::GR) < 512)
        {
            const uint32_t m0 = ((h & 7) * ((h >> 3) & 7)) / 7, m1 = (((h >> 6) & 7) * ((h >> 9) & 7)) / 7;
            p[i] = static_cast<uint8_t>((m0 | ((h >> 12) & 8)) | ((m1 | ((h >> 13) & 8)) << 4));
        }
        else p[i] = static_cast<uint8_t>(117 + (h >> 24 & 7));
    }
}
__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = __float2bfloat16_rn((static_cast<float>(hash32(static_cast<uint32_t>(i), seed) & 0xffffu) / 65535.f - 0.5f) * 8.f);
}
__global__ void k_fill_f32(float* p, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = (static_cast<float>(hash32(static_cast<uint32_t>(i), seed) & 0xffffffu) / 16777215.f - 0.5f) * 16.f;
}
// live-column compare of two [tiles][8][N] fp16 buffers: sums of (a - b)^2, b^2, max |a - b|, max |b| over tiles < n_tiles
__global__ void k_cmp(const __half* a, const __half* b, const int* tile_mask, const int* n_tiles, int N, double* out)
{
    __shared__ double sh[4][8];
    const int tile = blockIdx.x;
    double d2 = 0, r2 = 0, md = 0, mr = 0;
    if (tile < *n_tiles)
    {
        const int mask = tile_mask[tile];
        for (int i = threadIdx.x; i < 8 * N; i += blockDim.x)
        {
            if (!((mask >> (i / N)) & 1)) continue;
            const double x = __half2float(a[static_cast<size_t>(tile) * 8 * N + i]), y = __half2float(b[static_cast<size_t>(tile) * 8 * N + i]);
            d2 += (x - y) * (x - y); r2 += y * y; md = fmax(md, fabs(x - y)); mr = fmax(mr, fabs(y));
        }
    }
    for (int o = 16; o; o >>= 1) { d2 += __shfl_xor_sync(0xffffffffu, d2, o); r2 += __shfl_xor_sync(0xffffffffu, r2, o); md = fmax(md, __shfl_xor_sync(0xffffffffu, md, o)); mr = fmax(mr, __shfl_xor_sync(0xffffffffu, mr, o)); }
    if ((threadIdx.x & 31) == 0) { sh[0][threadIdx.x >> 5] = d2; sh[1][threadIdx.x >> 5] = r2; sh[2][threadIdx.x >> 5] = md; sh[3][threadIdx.x >> 5] = mr; }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d2 = r2 = md = mr = 0;
        for (int w = 0; w < 8; ++w) { d2 += sh[0][w]; r2 += sh[1][w]; md = fmax(md, sh[2][w]); mr = fmax(mr, sh[3][w]); }
        atomicAdd(out, d2); atomicAdd(out + 1, r2);
        // max via CAS loop on the double bits (values are >= 0)
        unsigned long long* pm = reinterpret_cast<unsigned long long*>(out + 2);
        unsigned long long old = *pm;
        while (__longlong_as_double(old) < md) { const unsigned long long nv = __double_as_longlong(md); const unsigned long long p = atomicCAS(pm, old, nv); if (p == old) break; old = p; }
        pm = reinterpret_cast<unsigned long long*>(out + 3);
        old = *pm;
        while (__longlong_as_double(old) < mr) { const unsigned long long nv = __double_as_longlong(mr); const unsigned long long p = atomicCAS(pm, old, nv); if (p == old) break; old = p; }
    }
}

// DRAM stream probes over `bytes` of w: (a) flat: every thread loads 16 B at a grid stride, 8 loads in flight; (b) per-warp streams: warp w streams the
// contiguous 87 KB chunk record range w (as a fc1 CTA warp does), DEPTH x 512 B in flight, lane-contiguous 16-B loads.
__global__ void k_stream_flat(const uint8_t* w, size_t bytes, u32* sink)
{
    const uint4* p = reinterpret_cast<const uint4*>(w);
    const size_t n = bytes / 16;
    u32 acc = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x * 8)
    {
        uint4 v[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) { const size_t j = i + static_cast<size_t>(k) * gridDim.x * blockDim.x; v[k] = j < n ? __ldg(p + j) : make_uint4(0, 0, 0, 0); }
        #pragma unroll
        for (int k = 0; k < 8; ++k) acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
    }
    if (acc == 0x12345678u) sink[0] = acc;
}
template <int DEPTH>
__global__ void k_stream_warps(const uint8_t* w, int nstreams, size_t stream_bytes, u32* sink)
{
    const int lane = threadIdx.x & 31, ws = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (ws >= nstreams) return;
    const uint8_t* s = w + static_cast<size_t>(ws) * stream_bytes;
    const int nrec = static_cast<int>(stream_bytes / mx4::GR);
    u32 acc = 0;
    u32 v[DEPTH][4];
    #pragma unroll
    for (int d = 0; d < DEPTH; ++d) mx4::ld4(s + d * mx4::GR + lane * 16, v[d]);
    for (int r = 0; r < nrec; r += DEPTH)
    {
        u32 c[DEPTH][4];
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d) { c[d][0] = v[d][0]; c[d][1] = v[d][1]; c[d][2] = v[d][2]; c[d][3] = v[d][3]; }
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d) if (r + DEPTH + d < nrec) mx4::ld4(s + (r + DEPTH + d) * mx4::GR + lane * 16, v[d]);
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d) acc ^= c[d][0] ^ c[d][1] ^ c[d][2] ^ c[d][3];
    }
    if (acc == 0x12345678u) sink[0] = acc;
}
__global__ void k_pool_remap(int* slot_expert, const int* n_slots, int pool)       // --pool n: slot -> expert % n (weights L2-resident at n <= 4)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < *n_slots) slot_expert[s] = slot_expert[s] % pool;
}
__global__ void k_fill_sh(float* p, size_t n, uint32_t seed)                // a shared-expert term in [-1, 1)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = (static_cast<float>(hash32(static_cast<uint32_t>(i), seed) & 0xffffu) / 65535.f - 0.5f) * 2.f;
}
__global__ void k_hash(const uint32_t* p, size_t n, unsigned long long* h)   // order-independent 64-bit hash of a buffer (sum of mixed words)
{
    unsigned long long a = 0;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x) a += static_cast<unsigned long long>(hash32(p[i], static_cast<uint32_t>(i))) * 0x9e3779b97f4a7c15ull;
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(h, a);
}
// bf16 [M][N] compare: sums of (a - b)^2, b^2, max |a - b|, max |b|
__global__ void k_cmp_bf(const uint16_t* a, const uint16_t* b, size_t n, double* out)
{
    double d2 = 0, r2 = 0, md = 0, mr = 0;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
    {
        const double x = __uint_as_float(static_cast<uint32_t>(a[i]) << 16), y = __uint_as_float(static_cast<uint32_t>(b[i]) << 16);
        d2 += (x - y) * (x - y); r2 += y * y; md = fmax(md, fabs(x - y)); mr = fmax(mr, fabs(y));
    }
    for (int o = 16; o; o >>= 1) { d2 += __shfl_xor_sync(0xffffffffu, d2, o); r2 += __shfl_xor_sync(0xffffffffu, r2, o); md = fmax(md, __shfl_xor_sync(0xffffffffu, md, o)); mr = fmax(mr, __shfl_xor_sync(0xffffffffu, mr, o)); }
    if ((threadIdx.x & 31) == 0)
    {
        atomicAdd(out, d2); atomicAdd(out + 1, r2);
        unsigned long long* pm = reinterpret_cast<unsigned long long*>(out + 2);
        unsigned long long old = *pm;
        while (__longlong_as_double(old) < md) { const unsigned long long p = atomicCAS(pm, old, __double_as_longlong(md)); if (p == old) break; old = p; }
        pm = reinterpret_cast<unsigned long long*>(out + 3);
        old = *pm;
        while (__longlong_as_double(old) < mr) { const unsigned long long p = atomicCAS(pm, old, __double_as_longlong(mr)); if (p == old) break; old = p; }
    }
}
template <class T> static void dal(T*& p, size_t n) { CK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }

struct Timer
{
    cudaEvent_t e0, e1;
    Timer() { CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); }
    template <class F> float us(F&& f)       // mean of REPS after one warm-up
    {
        f(); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int r = 0; r < REPS; ++r) f();
        CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaGetLastError());
        return ms * 1000.f / REPS;
    }
};

struct Case { int M, D; };

int main(int argc, char** argv)
{
    std::vector<Case> cases;
    bool stream = false;
    int pool = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--stream")) stream = true;
        else if (!strcmp(argv[i], "--pool") && i + 1 < argc) pool = atoi(argv[++i]);
        else if (i + 1 < argc) { cases.push_back({atoi(argv[i]), atoi(argv[i + 1])}); ++i; }
    }
    if (cases.empty()) cases = {{1024, 60}, {1024, 100}, {1024, 140}, {2048, 100}, {2048, 140}};
    uint8_t *w1, *w2;
    dal(w1, EB1 * NE); dal(w2, EB2 * NE);
    k_fill_mx4<<<4096, 256>>>(w1, EB1 * NE / mx4::GR, 11u);
    k_fill_mx4<<<4096, 256>>>(w2, EB2 * NE / mx4::GR, 13u);
    __nv_bfloat16* x; float *logits, *bias;
    dal(x, static_cast<size_t>(MM) * K1); dal(logits, static_cast<size_t>(MM) * NE); dal(bias, NE);
    k_fill_bf16<<<4096, 256>>>(x, static_cast<size_t>(MM) * K1, 17u);
    k_fill_f32<<<4096, 256>>>(logits, static_cast<size_t>(MM) * NE, 19u);
    char* plan; unsigned* emask; int* tile0; u32* xf;
    dal(plan, sizeof(P)); dal(emask, static_cast<size_t>(NE) * (MM / 32)); dal(tile0, NE); dal(xf, XFW);
    __half *act_a, *act_b, *y_a, *y_b;
    dal(act_a, static_cast<size_t>(TILES) * 8 * I); dal(act_b, static_cast<size_t>(TILES) * 8 * I);
    dal(y_a, static_cast<size_t>(TILES) * 8 * NOUT); dal(y_b, static_cast<size_t>(TILES) * 8 * NOUT);
    uint8_t *xq, *xs, *aq, *as;
    dal(xq, static_cast<size_t>(MM) * K1); dal(xs, static_cast<size_t>(MM) * (K1 / 32));
    dal(aq, static_cast<size_t>(TILES) * 8 * I); dal(as, static_cast<size_t>(TILES) * 8 * mxtc::ASP);
    double* dcmp; dal(dcmp, 8);
    float* sh; dal(sh, static_cast<size_t>(MM) * NOUT);
    k_fill_sh<<<4096, 256>>>(sh, static_cast<size_t>(MM) * NOUT, 23u);
    uint16_t *out_a, *out_b, *out_c;
    dal(out_a, static_cast<size_t>(MM) * NOUT); dal(out_b, static_cast<size_t>(MM) * NOUT); dal(out_c, static_cast<size_t>(MM) * NOUT);
    int* cnt; dal(cnt, static_cast<size_t>(MM) * 64);
    unsigned long long* dh; dal(dh, 8);
    CK(cudaDeviceSynchronize());
    Timer T;
    if (stream)
    {   // DRAM stream probes over 83 experts of fc1 (260 MB)
        u32* sink; dal(sink, 4);
        const size_t bytes = EB1 * 83;
        const float tf = T.us([&] { k_stream_flat<<<48 * 8, 256>>>(w1, bytes, sink); });
        const size_t sbytes = static_cast<size_t>(K1 / 64) * 2 * mx4::GR;   // one chunk's records (87 KB)
        const int ns = static_cast<int>(bytes / sbytes);
        const float t2 = T.us([&] { k_stream_warps<2><<<(ns + 3) / 4, 128>>>(w1, ns, sbytes, sink); });
        const float t4 = T.us([&] { k_stream_warps<4><<<(ns + 3) / 4, 128>>>(w1, ns, sbytes, sink); });
        const float t8 = T.us([&] { k_stream_warps<8><<<(ns + 3) / 4, 128>>>(w1, ns, sbytes, sink); });
        const float t8w = T.us([&] { k_stream_warps<8><<<(ns + 7) / 8, 256>>>(w1, ns, sbytes, sink); });
        printf("DRAM stream %zu MB: flat %.0f GB/s | warp streams of 87 KB, depth 2 / 4 / 8 (4 warps/CTA): %.0f / %.0f / %.0f GB/s; depth 8 (8 warps/CTA) %.0f GB/s\n",
               bytes >> 20, bytes / tf / 1e3, bytes / t2 / 1e3, bytes / t4 / 1e3, bytes / t8 / 1e3, bytes / t8w / 1e3);
    }
    const char* p = plan;
    const int* n_slots_dev = reinterpret_cast<const int*>(p + offsetof(P, n_slots));
    const int* n_tiles_dev = reinterpret_cast<const int*>(p + offsetof(P, n_tiles));
    printf("%5s %4s %5s %5s %5s %4s | %8s %8s %8s %8s | %8s %8s %8s %8s %8s %8s | %10s %10s %10s %10s | %9s %9s %9s %9s %10s %10s %5s %5s\n", "M", "D", "dist", "maxn", "tiles", "pass",
           "fc1_us", "TF/s", "fc2_us", "TF/s", "xq_us", "fc1tc", "TF/s", "aq_us", "fc2tc", "TF/s", "act_rms", "act_max", "y_rms", "y_max",
           "f2+cmb", "tc+cmb", "fused", "cmb_us", "out_rms", "out_max", "bitEq", "det");
    for (const Case& cs : cases)
    {
        const int M = cs.M, D = cs.D;
        // bias: the chosen D experts (spread evenly over the ids) get a spread in [-0.6, 0.4) (the keys are sqrt_softplus(logit) in 0..2.8 plus this), the rest -1e9
        std::vector<float> hb(NE, -1e9f);
        for (int j = 0; j < D; ++j)
        {
            const int e = static_cast<int>(static_cast<long>(j) * NE / D);
            const uint32_t h = (e * 2654435761u) ^ 0xabcdu;
            hb[e] = (static_cast<float>(h & 0xffff) / 65535.f) * 1.f - 0.6f;
        }
        CK(cudaMemcpy(bias, hb.data(), NE * 4, cudaMemcpyHostToDevice));
        // the production planner
        CK(cudaMemset(emask, 0, static_cast<size_t>(NE) * (MM / 32) * 4));
        dsr::k_route_ro<NE, TK, MM, TK><<<(M + 7) / 8, 256>>>(logits, bias, M, reinterpret_cast<P*>(plan), emask);
        dsr::k_scan_ro<NE, TK, MM><<<1, NE>>>(reinterpret_cast<P*>(plan), emask, tile0);
        dsr::k_place_ro<NE, TK, MM, TK><<<(M * TK + 255) / 256, 256>>>(reinterpret_cast<P*>(plan), emask, tile0, M);
        if (pool > 0) k_pool_remap<<<2, 256>>>(reinterpret_cast<int*>(plan + offsetof(P, slot_expert)), n_slots_dev, pool);
        CK(cudaDeviceSynchronize());
        int n_slots, n_tiles;
        CK(cudaMemcpy(&n_slots, n_slots_dev, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&n_tiles, n_tiles_dev, 4, cudaMemcpyDeviceToHost));
        std::vector<int> snt(n_slots);
        CK(cudaMemcpy(snt.data(), p + offsetof(P, slot_nt), n_slots * 4, cudaMemcpyDeviceToHost));
        const int maxnt = *std::max_element(snt.begin(), snt.end());
        int passes = 0;
        for (int v : snt) passes += (v + MXTC_NTG1 - 1) / MXTC_NTG1;
        const int sb = dsr::ro_slots_bound(M * TK, NE);
        const double R = static_cast<double>(M) * TK, f1 = 2.0 * R * 2 * I * K1, f2 = 2.0 * R * NOUT * I;
        // ---- fp16 path (the production config, ds_ro_cfg.h)
        const int ntx = (M + 31) / 32 * 4;
        mx4::build_xfrag_mt<0><<<(ntx * K1 * 4 + 255) / 256, 256>>>(x, M, K1, ntx, xf);
        constexpr int KS = DS41_RO_KS, CW = DS41_RO_CW, NTG1 = DS41_RO_NTG1, W2 = DS41_RO_W2, D2 = DS41_RO_D2, NTG2 = DS41_RO_NTG2;
        const mx4::ArgsMT RA{w1, w2, xf,
                             reinterpret_cast<const int*>(p + offsetof(P, slot_expert)), reinterpret_cast<const int*>(p + offsetof(P, slot_tile0)), reinterpret_cast<const int*>(p + offsetof(P, slot_nt)),
                             reinterpret_cast<const int*>(p + offsetof(P, tile_mask)), reinterpret_cast<const float*>(p + offsetof(P, tile_w)), reinterpret_cast<const int*>(p + offsetof(P, tile_tok)),
                             n_slots_dev, sb, 10.f, act_a, y_a, nullptr, 0};
        constexpr size_t SMEM2 = static_cast<size_t>(NTG2) * (I / 16) * 64 * sizeof(u32);
        auto* kf1 = mx4::k_fc1_ro<K1, I, KS, CW, 1, NTG1, 0, false, false, false>;
        auto* kf2 = mx4::k_fc2_ro<I, NOUT, W2, D2, NTG2, false, false, false, false>;
        const float t1 = T.us([&] { kf1<<<sb * (I / 16 / CW), KS * CW * 32>>>(RA); });
        const float t2 = T.us([&] { kf2<<<sb * (NOUT / 16 / W2), W2 * 32, SMEM2>>>(RA); });
        // ---- tensor-core path
        mxtc::Args TA{w1, w2, xq, xs,
                      RA.slot_expert, RA.slot_tile0, RA.slot_nt, RA.tile_mask, RA.tile_w, RA.tile_tok, n_slots_dev, n_tiles_dev, sb, MXTC_GY1, MXTC_GY2, MXTC_OS2,
                      nullptr, TK, nullptr, nullptr, nullptr, 0, nullptr, 10.f, act_b, aq, as, y_b};
        CK(cudaMemset(aq, 0xee, static_cast<size_t>(TILES) * 8 * I)); CK(cudaMemset(as, 0xee, static_cast<size_t>(TILES) * 8 * mxtc::ASP));   // poison: every live byte must be written
        constexpr int CW1 = MXTC_CW, NT1 = MXTC_NTG1, ST1 = MXTC_ST, WW2 = MXTC_W2, WT2 = MXTC_WT, TPC2 = MXTC_TPC, ST2 = MXTC_ST2;
        constexpr size_t S1 = mxtc::fc1_smem<K1, CW1, NT1, ST1>(), S2 = mxtc::fc2_smem<WW2, TPC2, ST2>();
        auto* kt1 = mxtc::k_fc1_tc<K1, I, CW1, NT1, ST1, MXTC_FUSEQ != 0>;
        auto* kt2 = mxtc::k_fc2_tc<I, NOUT, WW2, WT2, TPC2, ST2>;
        CK(cudaFuncSetAttribute(kt1, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(S1)));
        CK(cudaFuncSetAttribute(kt2, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(S2)));
        {
            static bool said = false;
            if (!said)
            {
                said = true;
                int smem_sm = 0, b1 = 0, b2 = 0, o1 = 0, o2 = 0;
                CK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b1, kf1, KS * CW * 32, 0));
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b2, kf2, W2 * 32, SMEM2));
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o1, kt1, CW1 * 32, S1));
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o2, kt2, WW2 * 32, S2));
                printf("smem/SM %d B; CTAs per SM: fp16 fc1 %d (%d warps) fc2 %d (%d warps) | tc fc1 %d (%d warps, %zu B smem) fc2 %d (%d warps, %zu B smem)\n", smem_sm,
                       b1, b1 * KS * CW, b2, b2 * W2, o1, o1 * CW1, S1, o2, o2 * WW2, S2);
            }
        }
        const float tq = T.us([&] { mxtc::k_xq<<<(unsigned)((static_cast<long>(M) * (K1 / 32) + 255) / 256), 256>>>(x, M, K1, xq, xs); });
        const float t1t = T.us([&] { kt1<<<sb * (I / 16 / CW1) * MXTC_GY1, CW1 * 32, S1>>>(TA); });
        const long arows = static_cast<long>(TILES) * 8;
        const float ta = MXTC_FUSEQ ? 0.f : T.us([&] { mxtc::k_aq<<<(unsigned)((arows * (I / 32) + 255) / 256), 256>>>(act_b, static_cast<int>(arows), n_tiles_dev, I, aq, as); });
        const float t2t = T.us([&] { kt2<<<sb * MXTC_OS2 * MXTC_GY2, WW2 * 32, S2>>>(TA); });
        // ---- accuracy
        double ca[4], cy[4];
        CK(cudaMemset(dcmp, 0, 64));
        if (!MXTC_FUSEQ) k_cmp<<<n_tiles, 256>>>(act_b, act_a, RA.tile_mask, n_tiles_dev, I, dcmp);
        k_cmp<<<n_tiles, 256>>>(y_b, y_a, RA.tile_mask, n_tiles_dev, NOUT, dcmp + 4);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ca, dcmp, 32, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(cy, dcmp + 4, 32, cudaMemcpyDeviceToHost));
        // ---- the combine: fp16 path fc2 + k_combine_mt (out_a), TC fc2 + k_combine_mt (out_b), TC fc2 with the fused combine (out_c)
        const dsh::CombineArgsMT<TK, MM> CA_a{reinterpret_cast<const P*>(plan), y_a, sh, out_a, M}, CA_b{reinterpret_cast<const P*>(plan), y_b, sh, out_b, M};
        auto* kcmb = dsh::k_combine_mt<NOUT, 128, gv::O_BF16, TK, MM>;
        const float tc_a = T.us([&] { kf2<<<sb * (NOUT / 16 / W2), W2 * 32, SMEM2>>>(RA); kcmb<<<M * NOUT / (8 * 128), 128>>>(CA_a); });
        const float tc_b = T.us([&] { kt2<<<sb * MXTC_OS2 * MXTC_GY2, WW2 * 32, S2>>>(TA); kcmb<<<M * NOUT / (8 * 128), 128>>>(CA_b); });
        const float tcmb = T.us([&] { kcmb<<<M * NOUT / (8 * 128), 128>>>(CA_b); });
        mxtc::Args TF = TA;
        TF.cnt = cnt; TF.kc = TK; TF.tok_slot = reinterpret_cast<const int*>(p + offsetof(P, tok_slot)); TF.tok_col = reinterpret_cast<const int*>(p + offsetof(P, tok_col));
        TF.shared = sh; TF.shared_bf16 = 0; TF.out = out_c;
        auto fused = [&] { CK(cudaMemsetAsync(cnt, 0, static_cast<size_t>(M) * MXTC_OS2 * sizeof(int))); kt2<<<sb * MXTC_OS2 * MXTC_GY2, WW2 * 32, S2>>>(TF); };
        const float tc_c = T.us(fused);
        // the slots are poisoned (NaN halves) before each checked run: a route's slot that was not written, or was dropped before its sum, shows as NaN in out
        auto fused_poisoned = [&] { CK(cudaMemsetAsync(y_b, 0xff, static_cast<size_t>(TILES) * 8 * NOUT * 2)); fused(); };
        // determinism: 5 more runs, the hash of out_c each time; bit-equality with the unfused combine
        unsigned long long hs[6];
        for (int rr = 0; rr < 6; ++rr)
        {
            CK(cudaMemset(out_c, 0, static_cast<size_t>(M) * NOUT * 2));
            fused_poisoned(); CK(cudaDeviceSynchronize());
            CK(cudaMemset(dh, 0, 8));
            k_hash<<<256, 256>>>(reinterpret_cast<const uint32_t*>(out_c), static_cast<size_t>(M) * NOUT / 2, dh);
            CK(cudaMemcpy(&hs[rr], dh, 8, cudaMemcpyDeviceToHost));
        }
        bool det = true;
        for (int rr = 1; rr < 6; ++rr) det = det && hs[rr] == hs[0];
        std::vector<uint16_t> hb_(static_cast<size_t>(M) * NOUT), hc_(static_cast<size_t>(M) * NOUT);
        CK(cudaMemcpy(hb_.data(), out_b, hb_.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hc_.data(), out_c, hc_.size() * 2, cudaMemcpyDeviceToHost));
        const bool biteq = memcmp(hb_.data(), hc_.data(), hb_.size() * 2) == 0;
        double co[4];
        CK(cudaMemset(dcmp, 0, 32));
        k_cmp_bf<<<256, 256>>>(out_c, out_a, static_cast<size_t>(M) * NOUT, dcmp);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(co, dcmp, 32, cudaMemcpyDeviceToHost));
        printf("%5d %4d %5d %5d %5d %4d | %8.1f %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f | %10.3e %10.3e %10.3e %10.3e | %9.1f %9.1f %9.1f %9.1f %10.3e %10.3e %5s %5s\n",
               M, D, n_slots, maxnt * 8, n_tiles, passes,
               t1, f1 / t1 / 1e6, t2, f2 / t2 / 1e6, tq, t1t, f1 / t1t / 1e6, ta, t2t, f2 / t2t / 1e6,
               MXTC_FUSEQ ? 0. : sqrt(ca[0] / ca[1]), MXTC_FUSEQ ? 0. : ca[2] / ca[3], sqrt(cy[0] / cy[1]), cy[2] / cy[3],
               tc_a, tc_b, tc_c, tcmb, sqrt(co[0] / co[1]), co[2] / co[3], biteq ? "yes" : "NO", det ? "yes" : "NO");
    }
    printf("config: fuseq %d fc1 CW %d NTG %d ST %d GY %d | fc2 W %d TW %d TPC %d ST %d GY %d OS %d ; fp16 path fc1 KS%d CW%d NTG%d fc2 W%d D%d NTG%d\n", MXTC_FUSEQ, MXTC_CW, MXTC_NTG1, MXTC_ST, MXTC_GY1, MXTC_W2, MXTC_WT, MXTC_TPC, MXTC_ST2, MXTC_GY2, MXTC_OS2,
           DS41_RO_KS, DS41_RO_CW, DS41_RO_NTG1, DS41_RO_W2, DS41_RO_D2, DS41_RO_NTG2);
    return 0;
}
