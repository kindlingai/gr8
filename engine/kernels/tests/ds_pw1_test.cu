// Two options of the M 33..PWMAX MoE path (ds41_moe_ops.cuh): GPU bit-identity test. Not yet run on a GPU. main refuses anything that is
// not sm_121:
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -Xcompiler -Wall -o ds_pw1_test ds_pw1_test.cu
//
//   ./ds_pw1_test budget      no GPU call: device bytes
//   ./ds_pw1_test check       GPU. For every M 33..48:
//      B  the shared expert (GR8_DS_MOE_SH_1PASS), two-chunk path (k_sh_fc1_mt / k_sh_fc2_mt NT = 4 on 32-row chunks with shifted pointers, as shared_pw) vs the one-pass NT = 6 path
//         (k_sh_fc1_mt_m NT = 6 + k_sh_fc2_mt_dyn NT = 6, 55296 B dynamic smem, as shared_pw_1pass): sh_act and sh_out rows < M bit for bit (memcmp); rows >= M of the
//         one-pass path stay at the poison (the kernels write only rows < M).
//      For M in {33, 48, 64, 100, 128} and kc in {6, 4, 1}:
//      A  the routed pw chain (k_plan_pw KC = kc, k_fc1_mt, k_fc2_mt) with the slot grid bound M * TK vs M * kc (GR8_DS_MOE_PW_BOUND): the whole ypart buffer bit for bit
//         (poisoned before each run, so a write outside the live slots also shows), and the plan's n_slots <= the M * kc bound.
//   Exit 0 = every comparison identical.
// Weights are random records (fill kernels below); the check compares kernel against kernel on the same weights. Device memory stays under 200 MB.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../ds_shared.cuh"
#include "../ds_router.cuh"
#include "../mxfp4.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

constexpr int K1 = 5120, I = 576, NOUT = 5120, TK = 6, NE = 384, MMAXT = 128;           // the main model at TP4 (Iw = 576), top-6 of 384; planner MM = 128
constexpr int POOL = 16;                                                                   // routed expert weight pool; slot_expert is remapped modulo POOL
constexpr int SH1_NT = 6;
using PLAN = dsr::PlanMT<TK, MMAXT>;
constexpr size_t EB1 = static_cast<size_t>(I / 16) * (K1 / 64) * 2 * mx4::GR;             // one expert's fc1 bytes
constexpr size_t EB2 = static_cast<size_t>(NOUT / 16) * (I / 64) * mx4::GR;               // its fc2 bytes
constexpr size_t EBG = static_cast<size_t>(I / 16) * (K1 / 32) * dsh::GR;                 // shared gate (and up) bytes
constexpr size_t EBD = static_cast<size_t>(NOUT / 16) * (I / 32) * dsh::GR;               // shared down bytes

__device__ __forceinline__ u32 hash32(u32 i, u32 seed)
{
    u32 h = i * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return h;
}
__global__ void k_fill_mx4(uint8_t* p, size_t nrec, u32 seed)                              // 512 random bytes + 32 scale bytes in 117..124
{
    const size_t tot = nrec * mx4::GR;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < tot; i += 256ull * gridDim.x)
    {
        const u32 h = hash32(static_cast<u32>(i) ^ static_cast<u32>(i >> 32) * 0x9e3779b1u, seed);
        p[i] = static_cast<int>(i % mx4::GR) < 512 ? static_cast<uint8_t>(h) : static_cast<uint8_t>(117 + (h >> 24 & 7));
    }
}
__global__ void k_fill_mx8(uint8_t* p, size_t nrec, u32 seed)                              // 512 E4M3 codes without the NaN codes + 16 scale bytes in 117..124
{
    const size_t tot = nrec * dsh::GR;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < tot; i += 256ull * gridDim.x)
    {
        const u32 h = hash32(static_cast<u32>(i) ^ static_cast<u32>(i >> 32) * 0x9e3779b1u, seed);
        uint8_t b = static_cast<uint8_t>(h);
        if ((b & 0x7f) == 0x7f) b ^= 1;
        p[i] = static_cast<int>(i % dsh::GR) < 512 ? b : static_cast<uint8_t>(117 + (h >> 24 & 7));
    }
}
__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, u32 seed)                          // finite bf16 in [-2, 2)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = __float2bfloat16_rn((static_cast<float>(hash32(static_cast<u32>(i), seed) & 0xffffu) / 65535.f - 0.5f) * 4.f);
}
__global__ void k_fill_f32(float* p, size_t n, u32 seed, float scale)                      // [-scale/2, scale/2)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = (static_cast<float>(hash32(static_cast<u32>(i), seed) & 0xffffu) / 65535.f - 0.5f) * scale;
}
__global__ void k_pool_remap(int* slot_expert, const int* n_slots, int pool)               // test-only: expert id % pool (only slot_expert changes)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < *n_slots) slot_expert[s] = slot_expert[s] % pool;
}

static size_t g_dev = 0;
template <class T> static T* dalloc(size_t n, bool dry)
{
    g_dev += n * sizeof(T);
    if (dry) return nullptr;
    T* p = nullptr;
    CK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
    return p;
}

struct Buf
{
    uint8_t *sg, *su, *sd, *w1, *w2;
    __nv_bfloat16* x; u32* xf; float *logits, *bias;
    __half *sact, *act, *ypart; float* sout; char* plan;
    int maxw, tiles, slots;
};
static Buf make(bool dry)
{
    Buf b{};
    b.maxw = 128; b.slots = dsr::pw_slot_bound(b.maxw * TK, NE); b.tiles = dsr::pw_tile_bound(b.maxw * TK, NE);
    b.sg = dalloc<uint8_t>(EBG, dry); b.su = dalloc<uint8_t>(EBG, dry); b.sd = dalloc<uint8_t>(EBD, dry);
    b.w1 = dalloc<uint8_t>(EB1 * POOL, dry); b.w2 = dalloc<uint8_t>(EB2 * POOL, dry);
    b.x = dalloc<__nv_bfloat16>(static_cast<size_t>(b.maxw) * K1, dry);
    b.xf = dalloc<u32>(static_cast<size_t>(b.maxw / 32 * 4) * K1 * 4, dry);                // build_xfrag_mt: nt tiles of [K1/16][32][2] u32 x 4 groups -> K1 * 4 u32 per tile
    b.logits = dalloc<float>(static_cast<size_t>(b.maxw) * NE, dry); b.bias = dalloc<float>(NE, dry);
    b.sact = dalloc<__half>(static_cast<size_t>(64) * I, dry); b.sout = dalloc<float>(static_cast<size_t>(64) * NOUT, dry);
    b.act = dalloc<__half>(static_cast<size_t>(b.tiles) * 8 * I, dry); b.ypart = dalloc<__half>(static_cast<size_t>(b.tiles) * 8 * NOUT, dry);
    b.plan = dalloc<char>(sizeof(PLAN), dry);
    if (dry) return b;
    k_fill_mx8<<<256, 256>>>(b.sg, EBG / dsh::GR, 1u); k_fill_mx8<<<256, 256>>>(b.su, EBG / dsh::GR, 2u); k_fill_mx8<<<256, 256>>>(b.sd, EBD / dsh::GR, 3u);
    k_fill_mx4<<<256, 256>>>(b.w1, EB1 * POOL / mx4::GR, 11u); k_fill_mx4<<<256, 256>>>(b.w2, EB2 * POOL / mx4::GR, 12u);
    k_fill_bf16<<<256, 256>>>(b.x, static_cast<size_t>(b.maxw) * K1, 7u);
    k_fill_f32<<<256, 256>>>(b.logits, static_cast<size_t>(b.maxw) * NE, 9u, 8.f);
    CK(cudaMemset(b.bias, 0, NE * sizeof(float)));
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    return b;
}
static void xfrag(const Buf& b, int M)
{
    const int nt = (M + 31) / 32 * 4;
    mx4::build_xfrag_mt<0><<<(nt * K1 * 4 + 255) / 256, 256>>>(b.x, M, K1, nt, b.xf);
}

// ---- B: the shared expert, old chunk loop vs the one pass
static void sh_old(const Buf& b, int M)                                                     // shared_pw's 32-row chunk loop
{
    for (int r0 = 0; r0 < M; r0 += 32)
    {
        const int mc = M - r0 < 32 ? M - r0 : 32;
        dsh::ArgsMT SA{b.sg, b.su, b.sd, b.xf + static_cast<size_t>(r0 / 8) * (K1 * 4), mc, 10.f, b.sact + static_cast<size_t>(r0) * I, b.sout + static_cast<size_t>(r0) * NOUT};
        dsh::k_sh_fc1_mt<K1, I, 8, 1, 2, 4, 0><<<I / 16, 8 * 32>>>(SA);
        dsh::k_sh_fc2_mt<I, NOUT, 8, 3, 4, 0><<<NOUT / 16 / 8, 8 * 32>>>(SA);
    }
}
static void sh_new(const Buf& b, int M)                                                     // shared_pw_1pass
{
    constexpr size_t SMEM2 = static_cast<size_t>(SH1_NT) * (I / 16) * 64 * sizeof(u32);
    dsh::ArgsMT SA{b.sg, b.su, b.sd, b.xf, M, 10.f, b.sact, b.sout};
    auto* k2 = dsh::k_sh_fc2_mt_dyn<I, NOUT, 8, 3, SH1_NT, 0>;
    CK(cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(SMEM2)));
    dsh::k_sh_fc1_mt_m<K1, I, 8, 1, 2, SH1_NT, 0><<<I / 16, 8 * 32>>>(SA);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NOUT / 16 / 8); cfg.blockDim = dim3(8 * 32); cfg.dynamicSmemBytes = SMEM2; cfg.stream = 0;
    CK(cudaLaunchKernelEx(&cfg, k2, SA));
}
static void poison_sh(const Buf& b) { CK(cudaMemset(b.sact, 0xff, static_cast<size_t>(64) * I * 2)); CK(cudaMemset(b.sout, 0xff, static_cast<size_t>(64) * NOUT * 4)); }
static int check_B()
{
    int bad = 0;
    std::vector<uint16_t> a0(64 * static_cast<size_t>(I)), a1(a0.size());
    std::vector<float> o0(64 * static_cast<size_t>(NOUT)), o1(o0.size());
    Buf b = make(false);
    for (int M = 33; M <= 48; ++M)
    {
        xfrag(b, M);
        poison_sh(b); sh_old(b, M); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a0.data(), b.sact, a0.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(o0.data(), b.sout, o0.size() * 4, cudaMemcpyDeviceToHost));
        poison_sh(b); sh_new(b, M); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a1.data(), b.sact, a1.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(o1.data(), b.sout, o1.size() * 4, cudaMemcpyDeviceToHost));
        long dact = 0, dout = 0, wr = 0;
        for (int m = 0; m < M; ++m)
        {
            dact += memcmp(a0.data() + static_cast<size_t>(m) * I, a1.data() + static_cast<size_t>(m) * I, I * 2) != 0;
            dout += memcmp(o0.data() + static_cast<size_t>(m) * NOUT, o1.data() + static_cast<size_t>(m) * NOUT, NOUT * 4) != 0;
        }
        for (int m = M; m < 64; ++m)                                                        // rows >= M of the new path must still be the poison (0xffff / 0xffffffff)
            for (int j = 0; j < I; ++j) wr += a1[static_cast<size_t>(m) * I + j] != 0xffffu;
        for (int m = M; m < 64; ++m)
            for (int j = 0; j < NOUT; ++j) { uint32_t u; memcpy(&u, &o1[static_cast<size_t>(m) * NOUT + j], 4); wr += u != 0xffffffffu; }
        const bool ok = !dact && !dout && !wr;
        printf("B shared M=%2d: sh_act rows differ %ld, sh_out rows differ %ld, writes beyond M %ld -> %s\n", M, dact, dout, wr, ok ? "IDENTICAL" : "DIFFER");
        bad += !ok;
    }
    return bad;
}

// ---- A: the pw routed chain with the slot grid bound M * TK vs M * kc
template <int KC> static int check_A_kc(const Buf& b, int M)
{
    int bad = 0;
    xfrag(b, M);
    dsr::k_plan_pw<NE, TK, MMAXT, KC><<<1, dsr::PLAN_THREADS>>>(b.logits, b.bias, M, reinterpret_cast<PLAN*>(b.plan));
    const int sb_all = dsr::pw_slot_bound(M * TK, NE);
    k_pool_remap<<<(sb_all + 255) / 256, 256>>>(reinterpret_cast<int*>(b.plan + offsetof(PLAN, slot_expert)), reinterpret_cast<const int*>(b.plan + offsetof(PLAN, n_slots)), POOL);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    int n_slots = 0;
    CK(cudaMemcpy(&n_slots, b.plan + offsetof(PLAN, n_slots), 4, cudaMemcpyDeviceToHost));
    const int sbs[2] = {sb_all, dsr::pw_slot_bound(M * KC, NE)};
    const char* p = b.plan;
    size_t n = static_cast<size_t>(b.tiles) * 8 * NOUT;
    std::vector<uint16_t> y[2];
    for (int v = 0; v < 2; ++v)
    {
        const int sb = sbs[v];
        mx4::ArgsMT RA{b.w1, b.w2, b.xf,
                       reinterpret_cast<const int*>(p + offsetof(PLAN, slot_expert)), reinterpret_cast<const int*>(p + offsetof(PLAN, slot_tile0)), reinterpret_cast<const int*>(p + offsetof(PLAN, slot_nt)),
                       reinterpret_cast<const int*>(p + offsetof(PLAN, tile_mask)), reinterpret_cast<const float*>(p + offsetof(PLAN, tile_w)), reinterpret_cast<const int*>(p + offsetof(PLAN, tile_tok)),
                       reinterpret_cast<const int*>(p + offsetof(PLAN, n_slots)), sb, 10.f, b.act, b.ypart};
        CK(cudaMemset(b.act, 0xff, static_cast<size_t>(b.tiles) * 8 * I * 2)); CK(cudaMemset(b.ypart, 0xff, n * 2));
        mx4::k_fc1_mt<K1, I, 8, 1, mx4::FC1_MT_DEPTH<4>, 4, 0, 0><<<sb * (I / 16), 8 * 32>>>(RA);
        mx4::k_fc2_mt<I, NOUT, 8, 2, 4, 0><<<sb * (NOUT / 16 / 8), 8 * 32>>>(RA);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        y[v].resize(n);
        CK(cudaMemcpy(y[v].data(), b.ypart, n * 2, cudaMemcpyDeviceToHost));
    }
    const bool same = memcmp(y[0].data(), y[1].data(), n * 2) == 0;
    const bool fits = n_slots <= sbs[1];
    printf("A pw M=%3d kc=%d: plan n_slots %4d, grid bound M*TK %4d -> M*kc %4d (fc1 CTAs %5d -> %5d), n_slots <= new bound %s, ypart %s\n",
           M, KC, n_slots, sbs[0], sbs[1], sbs[0] * (I / 16), sbs[1] * (I / 16), fits ? "yes" : "NO", same ? "IDENTICAL" : "DIFFER");
    bad += !(same && fits);
    return bad;
}
static int check_A()
{
    int bad = 0;
    Buf b = make(false);
    for (int M : {33, 48, 64, 100, 128}) { bad += check_A_kc<6>(b, M); bad += check_A_kc<4>(b, M); bad += check_A_kc<1>(b, M); }
    return bad;
}

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "";
    if (!strcmp(mode, "budget"))
    {
        make(true);
        printf("budget: device %.1f MB (cap 200 MB)\n", g_dev / 1048576.0);
        return 0;
    }
    if (!strcmp(mode, "check"))
    {
        cudaDeviceProp pr{};
        CK(cudaGetDeviceProperties(&pr, 0));
        if (pr.major != 12 || pr.minor != 1) { printf("refusing to run: sm_%d%d is not sm_121\n", pr.major, pr.minor); return 3; }
        const int bad = check_B() + check_A();
        printf("%s: %d comparisons differ\n", bad ? "FAIL" : "PASS", bad);
        return bad ? 1 : 0;
    }
    printf("usage: %s budget | check\n", argv[0]);
    return 1;
}
