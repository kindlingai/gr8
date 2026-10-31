// W4096 (ROMAX = dsr::RO_MAX = 4096): GPU numeric gate of the read-once prefill MoE at 4096 rows a call.
// The chain is ds41_moe_ops.cuh router_ro + shared_ro (DS41_SH_GL group loop) + experts_ro, replicated with the same templates, the same config
// (ds_ro_cfg.h) and the same grids: build_xfrag_mt -> k_route_ro / k_scan_ro / k_place_ro (MM = ROMAX) -> launch_shared_gl -> k_fc1_ro / k_fc2_ro
// -> k_combine_mt (MM = ROMAX). Only the gate GEMV is not run: the logits are synthetic (hash), the same bytes for every call.
// Claim: a row's output does not depend on the other rows of its call (planner = per-token routes; fc kernels: a column's bits are the M <= 32
// kernels'; shared expert per 32-row chunk; combine per token in route order). So:
//   A  one 4096-row call == two 2048-row calls (rows 0..2047, rows 2048..4095), byte for byte (memcmp of the bf16 output) -- covers k_combine_mt,
//      the planner and the plan type at MM 4096, RoScratch-sized buffers at 3408 tiles;
//   B  the 2048-row calls at MM = 4096 == the same calls at MM = 2048 (the 2048-row instantiation): W2048 output bytes do not depend on ROMAX;
//   C  the 4096-row call run twice: identical (determinism);
//   D  no output word is the 0xFFFF poison (every row x column written; outputs are poisoned before every call);
//   CONTROL  one input row (3000) changed: that row must differ and every other row must stay identical (the compare can fail; row independence).
// Exit 0 = "W4096_RO_ALL_PASS". Weights: random records, a pool of 4 routed experts (expert id % 4, test-only remap of slot_expert),
// random shared-expert MXFP8 records; the values are not a model, the test compares kernel against kernel.
//   (no argument)  usage + memory need; no CUDA call.    gpu   run (refuses unless aarch64 and an sm_121 device).
// Build (compiles on any CUDA host; the bin runs on a spark only):
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -Xcompiler -Wall -I engine/kernels -o ds_ro_w4096_test engine/kernels/tests/ds_ro_w4096_test.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../ds_prefill_ro.cuh"
#include "../ds_shared_gl.cuh"
#include "../ds_ro_cfg.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

constexpr int K1 = 5120, I = 576, NOUT = 5120, TK = 6, NE = 384, POOL = 4;       // the main model at TP4 (ds41_moe_ops.cuh HID / IW), top-6 of 384
constexpr int MBIG = dsr::RO_MAX, MHALF = MBIG / 2, CTRL_ROW = 3000;
static_assert(MBIG == 4096, "this gate is the W4096 gate: dsr::RO_MAX must be 4096");
constexpr int TILES = (MBIG * TK + 7 * NE) / 8;                                   // ds41_moe_ops.cuh ro_alloc (3408)
constexpr size_t EB1 = static_cast<size_t>(I / 16) * (K1 / 64) * 2 * mx4::GR;     // one routed expert's fc1 bytes
constexpr size_t EB2 = static_cast<size_t>(NOUT / 16) * (I / 64) * mx4::GR;       // its fc2 bytes
constexpr size_t SH1 = static_cast<size_t>(I / 16) * (K1 / 32) * dsh::GR;         // shared gate / up bytes each
constexpr size_t SH2 = static_cast<size_t>(NOUT / 16) * (I / 32) * dsh::GR;       // shared down bytes
constexpr size_t XFW = static_cast<size_t>(MBIG / 32 * 4) * (K1 / 16 * 64);       // xfrag words (ro_alloc: ROMAX / 8 * (HID / 16 * 64))
constexpr size_t OUTN = static_cast<size_t>(MBIG) * NOUT;                         // output elements of a 4096-row call

__device__ __forceinline__ uint32_t hash32(uint32_t i, uint32_t seed)
{
    uint32_t h = i * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return h;
}
__global__ void k_fill_mx4(uint8_t* p, size_t nrec, uint32_t seed)                // 512 random fp4 bytes + 32 scales 2^-10..2^-3
{
    const size_t tot = nrec * mx4::GR;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < tot; i += 256ull * gridDim.x)
    {
        const uint32_t h = hash32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(i >> 32) * 0x9e3779b1u, seed);
        p[i] = static_cast<int>(i % mx4::GR) < 512 ? static_cast<uint8_t>(h) : static_cast<uint8_t>(117 + (h >> 24 & 7));
    }
}
__global__ void k_fill_mx8(uint8_t* p, size_t nrec, uint32_t seed)                // Mxfp8 records (densev.cuh, GR 528): 512 E4M3 codes |x| < 2 (no NaN) + 16 scales
{
    const size_t tot = nrec * dsh::GR;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < tot; i += 256ull * gridDim.x)
    {
        const uint32_t h = hash32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(i >> 32) * 0x9e3779b1u, seed);
        p[i] = static_cast<int>(i % dsh::GR) < 512 ? static_cast<uint8_t>((h & 0x80) | (h & 0x3f)) : static_cast<uint8_t>(117 + (h >> 24 & 7));
    }
}
__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, uint32_t seed)            // finite bf16 in [-4, 4)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = __float2bfloat16_rn((static_cast<float>(hash32(static_cast<uint32_t>(i), seed) & 0xffffu) / 65535.f - 0.5f) * 8.f);
}
__global__ void k_fill_f32(float* p, size_t n, uint32_t seed)                      // logits in [-8, 8)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = (static_cast<float>(hash32(static_cast<uint32_t>(i), seed) & 0xffffffu) / 16777215.f - 0.5f) * 16.f;
}
__global__ void k_pool_remap(int* slot_expert, const int* n_slots, int pool)       // test-only: slot -> weight pool expert
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < *n_slots) slot_expert[s] = slot_expert[s] % pool;
}

struct Buf
{
    uint8_t *w1 = nullptr, *w2 = nullptr, *s1 = nullptr, *s3 = nullptr, *s2 = nullptr;
    __nv_bfloat16* x = nullptr; float *logits = nullptr, *bias = nullptr;
    char* plan = nullptr; unsigned* emask = nullptr; int* tile0 = nullptr; uint32_t* xf = nullptr;
    __half *act = nullptr, *y = nullptr, *sh_act = nullptr; float* sh_out = nullptr;
};
template <class T> static void dal(T*& p, size_t n) { CK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }

// One read-once call: rows r0 .. r0 + M - 1 of x / logits -> out [M][NOUT] bf16. MM = the plan instantiation (ROMAX of the build).
template <int MM> static void ro_call(const Buf& b, const __nv_bfloat16* x, const float* logits, int M, __nv_bfloat16* out)
{
    using P = dsr::PlanMT<TK, MM>;
    static_assert(sizeof(P) <= sizeof(dsr::PlanMT<TK, MBIG>), "plan buffer");
    constexpr int MW = MM / 32, MTMAX = dsr::MT_MAX;
    CK(cudaMemset(b.act, 0xff, static_cast<size_t>(TILES) * 8 * I * 2)); CK(cudaMemset(b.y, 0xff, static_cast<size_t>(TILES) * 8 * NOUT * 2));
    CK(cudaMemset(b.sh_out, 0xff, static_cast<size_t>(MBIG) * NOUT * 4)); CK(cudaMemset(out, 0xff, static_cast<size_t>(M) * NOUT * 2));
    CK(cudaMemset(b.xf, 0, XFW * 4));
    // router_ro (gate GEMV replaced by the synthetic logits)
    const int nt = (M + MTMAX - 1) / MTMAX * 4;
    mx4::build_xfrag_mt<0><<<(nt * K1 * 4 + 255) / 256, 256>>>(x, M, K1, nt, b.xf);
    CK(cudaMemset(b.emask, 0, static_cast<size_t>(NE) * MW * sizeof(unsigned)));
    dsr::k_route_ro<NE, TK, MM, TK><<<(M + 7) / 8, 256>>>(logits, b.bias, M, reinterpret_cast<P*>(b.plan), b.emask);
    dsr::k_scan_ro<NE, TK, MM><<<1, NE>>>(reinterpret_cast<P*>(b.plan), b.emask, b.tile0);
    dsr::k_place_ro<NE, TK, MM, TK><<<(M * TK + 255) / 256, 256>>>(reinterpret_cast<P*>(b.plan), b.emask, b.tile0, M);
    const int sb = dsr::ro_slots_bound(M * TK, NE);
    k_pool_remap<<<(sb + 255) / 256, 256>>>(reinterpret_cast<int*>(b.plan + offsetof(P, slot_expert)), reinterpret_cast<const int*>(b.plan + offsetof(P, n_slots)), POOL);
    CK(cudaGetLastError());
    // shared_ro, DS41_SH_GL (ds_ro_cfg.h)
    static_assert(DS41_SH_GL == 1, "production config runs the shared group loop");
    dsh::ArgsGL GA{b.s1, b.s3, b.s2, b.xf, M, 10.f, b.sh_act, b.sh_out};
    CK(static_cast<cudaError_t>(dsh::launch_shared_gl<K1, I, NOUT, 8, 1, 2, 8, 3, DS41_SH_NT>(GA, DS41_SH_RG, 0)));
    // experts_ro: fc_ro_run<false, false> (no ORDER, no L2LA) + the combine
    constexpr int KS = DS41_RO_KS, CW = DS41_RO_CW, NTG1 = DS41_RO_NTG1, W2 = DS41_RO_W2, D2 = DS41_RO_D2, NTG2 = DS41_RO_NTG2;
    const char* p = b.plan;
    const mx4::ArgsMT RA{b.w1, b.w2, b.xf,
                         reinterpret_cast<const int*>(p + offsetof(P, slot_expert)), reinterpret_cast<const int*>(p + offsetof(P, slot_tile0)), reinterpret_cast<const int*>(p + offsetof(P, slot_nt)),
                         reinterpret_cast<const int*>(p + offsetof(P, tile_mask)), reinterpret_cast<const float*>(p + offsetof(P, tile_w)), reinterpret_cast<const int*>(p + offsetof(P, tile_tok)),
                         reinterpret_cast<const int*>(p + offsetof(P, n_slots)), sb, 10.f, b.act, b.y, nullptr, 0};
    constexpr size_t SMEM2 = static_cast<size_t>(NTG2) * (I / 16) * 64 * sizeof(uint32_t);
    auto* k2 = mx4::k_fc2_ro<I, NOUT, W2, D2, NTG2, false, false>;
    if (SMEM2 > 48 * 1024) CK(cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(SMEM2)));
    mx4::k_fc1_ro<K1, I, KS, CW, 1, NTG1, 0, false, false><<<sb * (I / 16 / CW), KS * CW * 32>>>(RA);
    k2<<<sb * (NOUT / 16 / W2), W2 * 32, SMEM2>>>(RA);
    dsh::CombineArgsMT<TK, MM> CA{reinterpret_cast<const P*>(p), b.y, b.sh_out, out, M};
    dsh::k_combine_mt<NOUT, 128, gv::O_BF16, TK, MM><<<M * NOUT / (8 * 128), 128>>>(CA);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    int ns = 0, ntl = 0;
    CK(cudaMemcpy(&ns, b.plan + offsetof(P, n_slots), 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&ntl, b.plan + offsetof(P, n_tiles), 4, cudaMemcpyDeviceToHost));
    printf("  call MM %d rows %d: slots %d tiles %d (capacity %d)\n", MM, M, ns, ntl, TILES);
    if (ntl > TILES) { printf("RED: tiles %d > capacity %d\n", ntl, TILES); exit(1); }
}
static std::vector<uint16_t> get(const __nv_bfloat16* d, size_t n)
{
    std::vector<uint16_t> h(n);
    CK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
    return h;
}
// rows of a vs b (row counts nrow, a from row ra, b from row rb) that differ; first = the first one
static long diff_rows(const std::vector<uint16_t>& a, int ra, const std::vector<uint16_t>& b, int rb, int nrow, int& first)
{
    long n = 0; first = -1;
    for (int r = 0; r < nrow; ++r)
        if (memcmp(a.data() + static_cast<size_t>(ra + r) * NOUT, b.data() + static_cast<size_t>(rb + r) * NOUT, NOUT * 2)) { if (first < 0) first = r; ++n; }
    return n;
}

int main(int argc, char** argv)
{
    const double mb = ((EB1 + EB2) * POOL + 2 * SH1 + SH2 + static_cast<size_t>(MBIG) * K1 * 2 * 2 + static_cast<size_t>(MBIG) * NE * 4 + sizeof(dsr::PlanMT<TK, MBIG>) + XFW * 4
                       + static_cast<size_t>(TILES) * 8 * (I + NOUT) * 2 + static_cast<size_t>(MBIG) * (I * 2 + NOUT * 4) + 4 * OUTN * 2) / 1048576.0;
    if (argc < 2 || strcmp(argv[1], "gpu"))
    {
        printf("usage: ds_ro_w4096_test gpu   (spark only, not next to a serve)\nneed: device %.0f MB + host %.0f MB; RO_MAX %d, tiles %d\n", mb, 4 * OUTN * 2 / 1048576.0, dsr::RO_MAX, TILES);
        return argc < 2 ? 0 : 2;
    }
#if defined(__aarch64__)
    constexpr bool ARM = true;
#else
    constexpr bool ARM = false;                                                    // compiled everywhere (an x86 compile checks every instantiation), runs on a spark only
#endif
    if (!ARM) { printf("REFUSED: gpu mode runs on a spark (aarch64) only\n"); return 3; }
    cudaDeviceProp pr{};
    CK(cudaGetDeviceProperties(&pr, 0));
    if (pr.major * 10 + pr.minor != 121) { printf("REFUSED: device sm_%d%d is not sm_121\n", pr.major, pr.minor); return 3; }
    printf("device %s, %d SMs; need %.0f MB device\n", pr.name, pr.multiProcessorCount, mb);
    Buf b;
    dal(b.w1, EB1 * POOL); dal(b.w2, EB2 * POOL); dal(b.s1, SH1); dal(b.s3, SH1); dal(b.s2, SH2);
    dal(b.x, static_cast<size_t>(MBIG) * K1 * 2); dal(b.logits, static_cast<size_t>(MBIG) * NE); dal(b.bias, NE);
    dal(b.plan, sizeof(dsr::PlanMT<TK, MBIG>)); dal(b.emask, static_cast<size_t>(NE) * (MBIG / 32)); dal(b.tile0, NE); dal(b.xf, XFW);
    dal(b.act, static_cast<size_t>(TILES) * 8 * I); dal(b.y, static_cast<size_t>(TILES) * 8 * NOUT); dal(b.sh_act, static_cast<size_t>(MBIG) * I); dal(b.sh_out, static_cast<size_t>(MBIG) * NOUT);
    __nv_bfloat16 *o_big, *o_half, *o_old, *o_x;
    dal(o_big, OUTN); dal(o_half, OUTN); dal(o_old, OUTN); dal(o_x, OUTN);
    const int g = pr.multiProcessorCount * 4;
    k_fill_mx4<<<g, 256>>>(b.w1, EB1 * POOL / mx4::GR, 11u); k_fill_mx4<<<g, 256>>>(b.w2, EB2 * POOL / mx4::GR, 12u);
    k_fill_mx8<<<g, 256>>>(b.s1, SH1 / dsh::GR, 21u); k_fill_mx8<<<g, 256>>>(b.s3, SH1 / dsh::GR, 22u); k_fill_mx8<<<g, 256>>>(b.s2, SH2 / dsh::GR, 23u);
    k_fill_bf16<<<g, 256>>>(b.x, static_cast<size_t>(MBIG) * K1, 31u); k_fill_f32<<<g, 256>>>(b.logits, static_cast<size_t>(MBIG) * NE, 32u);
    CK(cudaMemset(b.bias, 0, NE * 4)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    const size_t xr = static_cast<size_t>(MHALF) * K1, lr = static_cast<size_t>(MHALF) * NE, orow = static_cast<size_t>(MHALF) * NOUT;
    printf("big: one call of %d rows at MM %d\n", MBIG, MBIG);
    ro_call<MBIG>(b, b.x, b.logits, MBIG, o_big);
    printf("halves at MM %d\n", MBIG);
    ro_call<MBIG>(b, b.x, b.logits, MHALF, o_half); ro_call<MBIG>(b, b.x + xr, b.logits + lr, MHALF, o_half + orow);
    printf("halves at MM %d (the 2048-row instantiation)\n", MHALF);
    ro_call<MHALF>(b, b.x, b.logits, MHALF, o_old); ro_call<MHALF>(b, b.x + xr, b.logits + lr, MHALF, o_old + orow);
    const std::vector<uint16_t> big = get(o_big, OUTN), half = get(o_half, OUTN), old = get(o_old, OUTN);
    printf("big again (determinism)\n");
    ro_call<MBIG>(b, b.x, b.logits, MBIG, o_x);
    const std::vector<uint16_t> again = get(o_x, OUTN);
    printf("control: row %d of x changed\n", CTRL_ROW);
    k_fill_bf16<<<g, 256>>>(b.x + static_cast<size_t>(CTRL_ROW) * K1, K1, 99u); CK(cudaDeviceSynchronize());
    ro_call<MBIG>(b, b.x, b.logits, MBIG, o_x);
    const std::vector<uint16_t> ctrl = get(o_x, OUTN);
    int f = -1; bool ok = true;
    const long dA = diff_rows(big, 0, half, 0, MBIG, f);
    printf("A big 4096 vs two 2048 calls (MM %d): %ld rows differ%s\n", MBIG, dA, dA ? (" (first row " + std::to_string(f) + ")").c_str() : ""); ok &= dA == 0;
    const long dB = diff_rows(half, 0, old, 0, MBIG, f);
    printf("B 2048 calls MM %d vs MM %d: %ld rows differ%s\n", MBIG, MHALF, dB, dB ? (" (first row " + std::to_string(f) + ")").c_str() : ""); ok &= dB == 0;
    const long dC = diff_rows(big, 0, again, 0, MBIG, f);
    printf("C big twice: %ld rows differ\n", dC); ok &= dC == 0;
    long poison = 0, nonfin = 0;
    for (uint16_t v : big) { poison += v == 0xffff; nonfin += (v & 0x7f80) == 0x7f80; }
    printf("D big: %ld words 0xFFFF (poison), %ld non-finite\n", poison, nonfin); ok &= poison == 0 && nonfin == 0;
    const long dX = diff_rows(big, 0, ctrl, 0, MBIG, f);
    const bool rowx = memcmp(big.data() + static_cast<size_t>(CTRL_ROW) * NOUT, ctrl.data() + static_cast<size_t>(CTRL_ROW) * NOUT, NOUT * 2) != 0;
    printf("CONTROL: %ld rows differ, row %d differs: %s (expect exactly 1, yes)\n", dX, CTRL_ROW, rowx ? "yes" : "no"); ok &= dX == 1 && rowx;
    printf(ok ? "W4096_RO_ALL_PASS\n" : "W4096_RO_RED\n");
    return ok ? 0 : 1;
}
