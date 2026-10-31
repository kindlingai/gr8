// Bit-equality gate of the read-once expert loop flags GR8_DS_RO_ORDER (k_order_ro + ORD) and GR8_DS_RO_L2LA=<n> (in-kernel L2 lookahead, LAON) against the
// flag-off loop. Run on a spark before any boot with the flags.
//   (no argument)  prints the usage and the memory budget; makes no CUDA call.
//   budget         the same.
//   plan [DIR] [L ...] [rows=A,B,...] [--synthetic]   CPU only (no CUDA call): the host plan of every case (distinct experts, tiles, heaviest expert), re-derived from the routes alone.
//   gpu [DIR] [L ...] [pool=N] [rows=A,B,...] [--synthetic]
//                  GPU mode (a spark; refuses anything but sm_121). For each layer L (default 0 20 39) and each W in {2048, 846}: the production planner chain (k_route_ro, k_scan_ro,
//                  k_place_ro, then k_order_ro) runs on real p1200 routes (reference-model dumps, DIR/L<n>.router_idx.bin int32 [1200][6], W rows = rows 0..W-1 of layer L; rows 1200.. of W = 2048
//                  wrap to rows 0.. of the same layer, because the p1200 files hold 1200 rows and only layers 0, 2, 20, 39). A missing or short route file is a FAIL (exit 1, before
//                  any CUDA call), never a fallback. --synthetic runs every case on synthetic skewed routes instead (DIR is ignored); the output and the final line say which source
//                  ran. The production kernels (k_fc1_ro / k_fc2_ro, the config of ds_ro_cfg.h, the launch of ds41_moe_ops.cuh fc_ro_run)
//                  then run with the flags off (the baseline) and with ORDER, with L2LA n = 1, 48, 200, and with ORDER + L2LA. Before every run the whole act and ypart buffers
//                  (every tile of the capacity, the fc2 output) are filled with 0xFF bytes; after it, the whole buffers are compared with memcmp against the baseline.
//                  Also run: the baseline a second time (the kernels are deterministic, so a difference is the flag's), and a CONTROL, an order array with one entry duplicated
//                  (one slot never runs): it must differ, which shows the comparison can fail. Prints identical=yes/no per variant and, if every line is as expected,
//                  "ALL IDENTICAL". Exit 0 = ALL IDENTICAL.
//                  TIMING (after the memcmp lines of W 2048 at L0 and L20; the exit code does not depend on it): per variant (baseline, ORDER, L2LA n, ORDER + L2LA n for n in 1, 8, 48,
//                  200, baseline again as a drift check) the GPU time of fc1 + fc2 (cudaEvent around the two launches), median of 20 runs after 5 warm-up runs, with min / max. Weights
//                  in the timing phase are a pool of 32 experts (150 MB, 6 x the 24 MB L2), expert id by launch position (position p reads pool expert p % 32), so no expert is read
//                  twice within 150 MB, as 384 distinct experts would; act + ypart (170 MB) are poisoned by memset before every run (outside the timed region), which flushes the L2.
//                  The host baselines are freed for this phase so device + host stays under the cap (the budget of both phases is printed).
// rows=A,B,...: the row counts of the cases, default 2048,846; each must be <= RM, the capacity of the build (default 2048). The 2048 vs 8192 timing is
//   a second build, -DB6RO_RM=8192, run as `gpu DIR 20 0 rows=2048,8192` with GR8_TEST_DS41_RO_CAP_MB raised (the budget line prints the need). The GPU planner kernels static_assert rows <= dsr::RO_MAX (2048),
//   so a build with RM > RO_MAX builds the plan on the host (dsr::plan_ref, the specification of the device plan) and uploads it; the fc kernels read only the plan arrays. Before any case such
//   a build runs plan_selfcheck: at 2048 rows, on each layer's routes, the host plan and the GPU planner must have equal structural fields, else exit 1. Every timed layer with two or more
//   timed widths (W >= 2048) also gets a per-row line: the baseline median / W for each width and the change from the first to the last. The p1200 route files hold 1200 rows, so rows above
//   1200 repeat them: the experts per layer do not grow with W, and the output says so.
// Weights are a pool of random records (the pool size is the argument pool=N, default 4; expert id % pool): the flags change the order and the cache hints, not the values, so
// the check compares kernel against kernel on the same weights. Memory: device + host stays under GR8_TEST_DS41_RO_CAP_MB (default 400); the budget is printed.
// ntg / ntgprof (NTG test, test code only): k_fc1_ro alone (no fc2, no flags) at larger NTG, on the same real routes and the same planner chain.
//   ntg [DIR] [L ...] [pool=N] [rows=A] [--synthetic]    GPU mode (sm_121 only). Variants (NTG_VARS): the production fc1 (KS8 CW2 NTG4), CW2 NTG8, CW2 NTG16, CW1 NTG4, CW1 NTG8, CW1 NTG16.
//                  The register file bounds NTG (8 floats of accumulator per tile and thread; CW2 = 512 threads = 128 registers at most), so "all tiles of an expert in one group" cannot
//                  be built in this kernel; the per-variant registers and local-memory (spill) bytes are printed from cudaFuncGetAttributes. Per case: (1) model line (CPU arithmetic from the plan):
//                  groups = sum over experts of ceil(nt / NTG) and groups x one expert's fc1 bytes, the once-each floor; (2) bit-equality: act of every variant memcmp'd against the
//                  production variant's (whole act buffer, 0xFF-poisoned before each run), the production variant again (determinism) and a CONTROL (one CTA fewer than the
//                  slots need: must differ); (3) timing: round robin over the variants, 5 warm-up + 20 timed rounds, fc1 only, weights = pool of 32 experts by launch position, act + ypart
//                  poisoned before each run (flushes the L2); median / min / max per variant and the change against the production variant. Prints "NTG ALL IDENTICAL" or FAIL; exit 0 = identical.
//   ntgprof [DIR] [L ...] [pool=N] [rows=A]   one fc1 launch per variant per case (the timing phase's setup), nothing else matching k_fc1_ro: for ncu --kernel-name regex:k_fc1_ro
//                  --launch-count <number of variants>; the launch order is printed. No memcmp, no timing.
// eq / eqprof (L2-hint test, test code only): k_fc1_ro alone with L2 eviction hints, via a test-local copy roeq::k_fc1_ro_t<WPOL, XPOL, SPOL>. Variants: P0 = the engine kernel,
//   C0 = the copy with no hint (control), V1w = weight loads evict_last, V1 = weights evict_last + x loads evict_first + act stores evict_first. Same planner chain, routes and weight pool as ntg.
//   eq [DIR] [L ...] [pool=N] [rows=A]      GPU mode (spark only): bit-equality of C0 / V1w / V1 against P0 + determinism + CONTROL, then timing (5 warm-up + 20 rounds, P0 again = drift). "EQ ALL IDENTICAL" or FAIL.
//   eqprof [DIR] [L ...] [pool=N] [rows=A]  7 fc1 launches per case for ncu: 1-4 full grid P0, C0, V1w, V1; 5-7 one slot (the first slot with the most tiles copied to slot 0, grid 18) P0, V1w, V1.
// l2h / l2hprof (GR8_DS_MOE_L2HINT test, test code only): the engine kernels k_fc1_ro / k_fc2_ro (ds_prefill_ro.cuh) with their L2H template argument, which
//   ds41_moe_ops.cuh fc_ro_run instantiates for the flag: value 0 = no hint, 1 = fc1 weight loads evict_last, 2 = fc1 and fc2. Same planner chain, routes and weight pool as eq.
//   l2h [DIR] [L ...] [pool=N] rows=A[,B]   GPU mode (spark only). Per case: gate G = the chain (fc1 + fc2, la 0, no order) with value 0, then value 1, value 2 and value 0 again: the whole act and the whole
//                  ypart buffer (0xFF-poisoned before each run) memcmp'd against the value-0 run, plus two CONTROLS (fc1 one CTA fewer: act must differ; fc2 one CTA fewer: ypart must differ).
//                  Then timing, 5 warm-up + 20 rounds round robin over 10 items: fc1 P0 (L2H off), fc1 P1 (L2H on), fc1 V1w (the eq test copy, weights evict_last), fc2 Q0, fc2 Q1 (fc2 poisons ypart only),
//                  chain H0 / H1 / H2 (values 0 / 1 / 2, events around both launches), fc1 P0 again and fc2 Q0 again (drift); median / min / max and the change against P0, Q0, H0.
//                  Prints the registers and local-memory bytes of the 5 kernels from cudaFuncGetAttributes, "L2H ALL IDENTICAL" or FAIL. For a build with RM > 2048 the plan is the host plan (plan self-check first).
//   l2hprof [DIR] [L ...] [pool=N] rows=A[,B]   4 launches matching k_fc[12]_ro per case, for ncu: 1 fc1 P0, 2 fc1 P1, 3 fc2 Q0, 4 fc2 Q1 (fc2 reads the act of launch 2). No memcmp, no timing.
// plan mode also prints the MODEL line of every case (CPU only).
// Build (compiles on any CUDA host; the GPU modes run on a spark only): nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -Xcompiler -Wall -I engine/kernels -o ds_ro_flags_test engine/kernels/tests/ds_ro_flags_test.cu
// 8192-row build: the same command with -DB6RO_RM=8192 added.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <sys/utsname.h>
#include "../../knobs-rs/gr8_knobs.h"
#include <cuda_runtime.h>
#include "../ds_prefill_ro.cuh"
#include "../ds_ro_cfg.h"                                  // DS41_RO_*: the production config (the adapter includes the same file)

// The sparks (GB10, Grace CPU) are aarch64. The GPU modes check this in main before any CUDA call, so a run on an x86_64 host never starts the CUDA runtime
// (the sm_121 check inside the modes comes after cudaGetDeviceProperties, which is itself a CUDA call).
static bool host_is_spark() { struct utsname u; return uname(&u) == 0 && !strcmp(u.machine, "aarch64"); }

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

#ifndef DS_RO_RM
#define DS_RO_RM 2048                                                               // rows capacity of the build; -DB6RO_RM=8192 for the 2048 vs 8192 timing
#endif
constexpr int K1 = 5120, I = 576, NOUT = 5120, TK = 6, NE = 384, RM = DS_RO_RM;      // the main model at TP4 (Iw = 576), top-6 of 384, rows <= RM
constexpr bool GPU_PLAN = RM <= dsr::RO_MAX;                                       // the GPU planner kernels static_assert MM <= RO_MAX (2048); above that the plan is built on the host (plan_ref)
static_assert(RM % 64 == 0, "xfrag tiles");
using PR = dsr::PlanMT<TK, RM>;
constexpr size_t EB1 = static_cast<size_t>(I / 16) * (K1 / 64) * 2 * mx4::GR;     // one expert's fc1 bytes [I/16][K1/64][2][GR]
constexpr size_t EB2 = static_cast<size_t>(NOUT / 16) * (I / 64) * mx4::GR;       // its fc2 bytes [NOUT/16][I/64][GR]
constexpr int TILES = (RM * TK + 7 * NE) / 8;                                      // tiles of an RM-row plan <= this (dsr::pw_tile_bound(RM * TK, NE), checked in main)
constexpr size_t ACT_B = static_cast<size_t>(TILES) * 8 * I * 2;                   // act [tile][8][I] fp16
constexpr size_t Y_B = static_cast<size_t>(TILES) * 8 * NOUT * 2;                  // ypart [tile][8][NOUT] fp16 = the fc2 output
constexpr size_t XF_B = static_cast<size_t>(RM / 64 * 8) * K1 * 4 * 4;             // xfrag: RM / 32 * 4 tiles of K1 / 16 * 64 words
constexpr size_t SLICE = 8u << 20;                                                 // host compare slice
constexpr int TPOOL = 32;                                                         // weight pool of the timing phase (experts): 150 MB = 6 x L2, no reuse within a pool turn
constexpr int NSWEEP[] = {1, 8, 48, 200};                                          // L2LA distances (CTA groups) of the memcmp lines and of the timing
static_assert(sizeof(__nv_bfloat16) * static_cast<size_t>(RM) * K1 <= Y_B, "x is staged in the ypart buffer");

// ------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ uint32_t hash32(uint32_t i, uint32_t seed)
{
    uint32_t h = i * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return h;
}
// mx4 records (GR = 544): 512 random bytes (any fp4 nibble is a number) + 32 scale bytes in 117..124 (2^-10 .. 2^-3).
__global__ void k_fill_mx4(uint8_t* p, size_t nrec, uint32_t seed)
{
    const size_t tot = nrec * mx4::GR;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < tot; i += 256ull * gridDim.x)
    {
        const uint32_t h = hash32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(i >> 32) * 0x9e3779b1u, seed);
        p[i] = static_cast<int>(i % mx4::GR) < 512 ? static_cast<uint8_t>(h) : static_cast<uint8_t>(117 + (h >> 24 & 7));
    }
}
__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, uint32_t seed)              // finite bf16 in [-4, 4)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = __float2bfloat16_rn((static_cast<float>(hash32(static_cast<uint32_t>(i), seed) & 0xffffu) / 65535.f - 0.5f) * 8.f);
}
// Test-only: point the plan's slots at the weight pool (expert id % pool). Only slot_expert changes; no other kernel reads it, and k_fc1_ro / k_fc2_ro read the remapped ids
// for their loads and for their prefetches.
__global__ void k_pool_remap(int* slot_expert, const int* n_slots, int pool)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < *n_slots) slot_expert[s] = slot_expert[s] % pool;
}

// Timing phase: the expert at LAUNCH POSITION p reads pool expert p % tp (slot order[p] with the order, slot p without), so consecutive launches read consecutive pool experts.
__global__ void k_pool_remap_pos(int* slot_expert, const int* n_slots, const int* order, int tp)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < *n_slots) slot_expert[dsr::ro_pos_slot(order, p)] = p % tp;
}

// ------------------------------------------------------------------------------------------ eq / eqprof: L2 eviction hints on the routed fc1, test-local kernel copy
// k_fc1_ro_t is a copy of mx4::k_fc1_ro (ORD = false, LAON = false) whose weight loads, x loads and act stores can carry an L2 eviction-priority hint (WPOL / XPOL / SPOL, 0 = none).
// With all three 0 the code is the engine's, op for op (C0 of the eq test). No engine header is changed. Each hint asm makes its own policy (createpolicy ... 1.0 inside the asm
// string), so no 64-bit policy value is held in a register across the loop (production fc1 is at 128 registers and 8 B stack).
namespace roeq
{
__device__ __forceinline__ void ld4_last(const void* p, u32 (&v)[4])               // mx4::ld4 + L2::evict_last
{
    asm volatile("{\n\t.reg .b64 pol;\n\tcreatepolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
                 "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], pol;\n\t}"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "l"(p));
}
__device__ __forceinline__ u32 ld_nc_last(const void* p)                           // ld_nc (the scale word) + L2::evict_last
{
    u32 v;
    asm volatile("{\n\t.reg .b64 pol;\n\tcreatepolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
                 "ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%1], pol;\n\t}"
                 : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ld_x_first(const u32* p)                          // the plain 64-bit x-fragment load + L2::evict_first
{
    uint2 r;
    asm("{\n\t.reg .b64 pol;\n\tcreatepolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
        "ld.global.L2::cache_hint.v2.u32 {%0,%1}, [%2], pol;\n\t}"
        : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
__device__ __forceinline__ void st_first(__half* p, __half v)                      // 16-bit act store + L2::evict_first
{
    asm volatile("{\n\t.reg .b64 pol;\n\tcreatepolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
                 "st.global.L2::cache_hint.b16 [%0], %1, pol;\n\t}"
                 :: "l"(p), "h"(__half_as_ushort(v)));
}
template <int XPOL> __device__ __forceinline__ uint2 ldx(const u32* p)
{
    if constexpr (XPOL == 0) return *reinterpret_cast<const uint2*>(p);              // as mx4::Grp::run_mt
    else return ld_x_first(p);
}
template <int SPOL> __device__ __forceinline__ void stact(__half* p, __half v)
{
    if constexpr (SPOL == 0) *p = v;                                                 // as the plain a0[..] = ... of k_fc1_ro
    else st_first(p, v);
}
template <int WPOL> __device__ __forceinline__ void load_t(mx4::Grp& G, const uint8_t* g, int lane)
{
    if constexpr (WPOL == 0) G.load(g, lane);                                        // as mx4::Grp::load
    else { ld4_last(g + lane * 16, G.c); G.s = ld_nc_last(g + mx4::SOFF + (lane >> 2) * 4); }
}
// mx4::Grp::run_mt, copied: the b load goes through ldx<XPOL>; everything else (per tile zero g, 2 mma, scale fmaf) is the engine's loop.
template <int NT, int XPOL>
__device__ __forceinline__ void run_mt_t(const mx4::Grp& G, const u32* xb, const int (&boff)[NT], int ks0, int nt, float (&acc)[NT][4])
{
    #pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        float g[NT][4];
        #pragma unroll
        for (int t = 0; t < NT; ++t) g[t][0] = g[t][1] = g[t][2] = g[t][3] = 0.f;
        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const u32 w = G.c[2 * h + i];
            const u32 a0 = e2m1x2(w & 0xffu), a1 = e2m1x2((w >> 8) & 0xffu), a2 = e2m1x2((w >> 16) & 0xffu), a3 = e2m1x2(w >> 24);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (t < nt)
                {
                    const uint2 b = ldx<XPOL>(xb + boff[t] + (ks0 + 2 * h + i) * 64);
                    mma16816(g[t], a0, a1, a2, a3, b.x, b.y);
                }
        }
        const float sl = mx4::e8(G.s >> (16 * h)), sh = mx4::e8(G.s >> (16 * h + 8));
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            if (t < nt)
            {
                acc[t][0] = fmaf(sl, g[t][0], acc[t][0]); acc[t][1] = fmaf(sl, g[t][1], acc[t][1]);
                acc[t][2] = fmaf(sh, g[t][2], acc[t][2]); acc[t][3] = fmaf(sh, g[t][3], acc[t][3]);
            }
    }
}
// mx4::fc1_part_mt, copied: the loads go through load_t<WPOL>, the multiplies through run_mt_t<NT, XPOL>.
template <int WPOL, int XPOL, int NGP, int DEPTH, int NT>
__device__ __forceinline__ void fc1_part_mt_t(const uint8_t* s, const u32* xb, const int (&boff)[NT], int nt, int lane, int ks0, float (&ag)[NT][4], float (&au)[NT][4])
{
    mx4::Grp ng[DEPTH], nu[DEPTH];
    #pragma unroll
    for (int p = 0; p < DEPTH; ++p)
        if (p < NGP) { load_t<WPOL>(ng[p], s + p * 2 * mx4::GR, lane); load_t<WPOL>(nu[p], s + p * 2 * mx4::GR + mx4::GR, lane); }
    for (int g = 0; g < NGP; g += DEPTH)
    {
        mx4::Grp cg[DEPTH], cu[DEPTH];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p) { cg[p] = ng[p]; cu[p] = nu[p]; }
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + DEPTH + p < NGP) { load_t<WPOL>(ng[p], s + (g + DEPTH + p) * 2 * mx4::GR, lane); load_t<WPOL>(nu[p], s + (g + DEPTH + p) * 2 * mx4::GR + mx4::GR, lane); }
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + p < NGP)
            {
                run_mt_t<NT, XPOL>(cg[p], xb, boff, ks0 + (g + p) * 4, nt, ag);
                run_mt_t<NT, XPOL>(cu[p], xb, boff, ks0 + (g + p) * 4, nt, au);
            }
    }
}
// mx4::k_fc1_ro (ds_prefill_ro.cuh), ORD = false, LAON = false, copied.
template <int K1_, int I_, int KS, int CW, int DEPTH, int NTG, int MINB, int WPOL, int XPOL, int SPOL>
__global__ void __launch_bounds__(KS * CW * 32, MINB) k_fc1_ro_t(mx4::ArgsMT A)
{
    constexpr int NCH = I_ / 16, NG = K1_ / mx4::KG, NGP = NG / KS, UPS = NCH / CW, NW = KS * CW;
    constexpr int PMAX = 32768 / (NW * 1024), PST = KS == 1 ? 1 : (NTG < PMAX ? NTG : (PMAX > 0 ? PMAX : 1));
    static_assert(K1_ % mx4::KG == 0 && NG % KS == 0 && NCH % CW == 0, "fc1 tiling");
    static_assert(NTG % PST == 0 && (KS == 1 || NW * 1024 * PST <= 32768), "fc1 reduce rounds");
    __shared__ float4 ps[KS > 1 ? PST : 1][NW * 32 * 2];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int slot, chunk, part;
    dsr::ro_fc1_work(blockIdx.x, warp, UPS, KS, CW, slot, chunk, part);
    if (A.n_slots_dev && slot >= *A.n_slots_dev) return;
    const int nt = A.slot_nt[slot], tile0 = A.slot_tile0[slot], ng = dsr::ro_ngroups(nt, NTG);
    const uint8_t* s = A.fc1 + dsr::ro_fc1_rec(A.slot_expert[slot], chunk, part * NGP, 0, NCH, NG) * mx4::GR;
    #pragma unroll 1
    for (int g = 0; g < ng; ++g)
    {
        const int ntg = dsr::ro_group_nt(nt, NTG, g), t0 = dsr::ro_tile(tile0, g, NTG, 0);
        int boff[NTG];
        #pragma unroll
        for (int t = 0; t < NTG; ++t)
            boff[t] = t < ntg ? dsr::mt_xoff(A.tile_tok[(t0 + t) * 8 + (lane >> 2)], lane, K1_ * 4) : 0;
        float ag[NTG][4], au[NTG][4];
        #pragma unroll
        for (int t = 0; t < NTG; ++t) { ag[t][0] = ag[t][1] = ag[t][2] = ag[t][3] = 0.f; au[t][0] = au[t][1] = au[t][2] = au[t][3] = 0.f; }
        fc1_part_mt_t<WPOL, XPOL, NGP, DEPTH, NTG>(s, A.xfrag, boff, ntg, lane, part * NGP * 4, ag, au);
        if constexpr (KS > 1)
        {
            #pragma unroll
            for (int rb = 0; rb < NTG; rb += PST)
                if (rb < ntg)
                {
                    #pragma unroll
                    for (int tt = 0; tt < PST; ++tt)
                        if (rb + tt < ntg)
                        {
                            float4* p = ps[tt] + (warp * 32 + lane) * 2;
                            p[0] = make_float4(ag[rb + tt][0], ag[rb + tt][1], ag[rb + tt][2], ag[rb + tt][3]);
                            p[1] = make_float4(au[rb + tt][0], au[rb + tt][1], au[rb + tt][2], au[rb + tt][3]);
                        }
                    __syncthreads();
                    if (part == 0)
                        #pragma unroll
                        for (int tt = 0; tt < PST; ++tt)
                            if (rb + tt < ntg)
                                for (int q = 1; q < KS; ++q)      // fixed order
                                {
                                    const float4 gq = ps[tt][((warp + q) * 32 + lane) * 2], uq = ps[tt][((warp + q) * 32 + lane) * 2 + 1];
                                    ag[rb + tt][0] += gq.x; ag[rb + tt][1] += gq.y; ag[rb + tt][2] += gq.z; ag[rb + tt][3] += gq.w;
                                    au[rb + tt][0] += uq.x; au[rb + tt][1] += uq.y; au[rb + tt][2] += uq.z; au[rb + tt][3] += uq.w;
                                }
                    __syncthreads();                              // ps is rewritten by the next round / group
                }
        }
        if (part == 0)
        {
            const int r = lane >> 2, c = lane & 3;
            #pragma unroll
            for (int t = 0; t < NTG; ++t)
                if (t < ntg)
                {
                    __half* a0 = A.act + (static_cast<size_t>(t0 + t) * 8 + 2 * c) * I_ + chunk * 16 + r;
                    stact<SPOL>(a0, __float2half_rn(mx4::swiglu_lim(ag[t][0], au[t][0], A.limit)));
                    stact<SPOL>(a0 + I_, __float2half_rn(mx4::swiglu_lim(ag[t][1], au[t][1], A.limit)));
                    stact<SPOL>(a0 + 8, __float2half_rn(mx4::swiglu_lim(ag[t][2], au[t][2], A.limit)));
                    stact<SPOL>(a0 + I_ + 8, __float2half_rn(mx4::swiglu_lim(ag[t][3], au[t][3], A.limit)));
                }
        }
    }
}
}   // namespace roeq

// ------------------------------------------------------------------------------------------ memory budget (device allocations are counted; budget mode only counts)
static bool g_dry = false;
static size_t g_dev = 0, g_host = 0;
template <class T> static void dalloc(T*& p, size_t n)
{
    g_dev += n * sizeof(T);
    if (g_dry) { p = nullptr; return; }
    CK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
}
static double g_l2mb = 24.0;                                                          // L2 size printed by the timing (set from the device in gpu mode)
static size_t cap_bytes() { return static_cast<size_t>(knob::test_ds41_ro_cap_mb()) << 20; }   // GR8_TEST_DS41_RO_CAP_MB
static void print_budget(const char* what, int pool)
{
    printf("budget %s: device %.1f MB + host %.1f MB = %.1f MB (cap %.0f MB); pool %d experts\n", what, g_dev / 1048576.0, g_host / 1048576.0, (g_dev + g_host) / 1048576.0, cap_bytes() / 1048576.0, pool);
}

struct Ctx
{
    int nsm = 0, np = 0;
    uint8_t *w1 = nullptr, *w2 = nullptr;
    float *log = nullptr, *bias = nullptr;
    char* plan = nullptr; unsigned* emask = nullptr; int *tile0 = nullptr, *order = nullptr, *order_bad = nullptr; uint32_t* xf = nullptr;
    __half *act = nullptr, *y = nullptr;
};
static void setup(Ctx& c, int np, int nsm)
{
    g_dev = 0; g_host = 0;
    c.np = np; c.nsm = nsm;
    dalloc(c.w1, EB1 * np); dalloc(c.w2, EB2 * np);
    dalloc(c.xf, XF_B / 4); dalloc(c.plan, sizeof(PR)); dalloc(c.act, ACT_B / 2); dalloc(c.y, Y_B / 2);
    dalloc(c.log, static_cast<size_t>(RM) * NE); dalloc(c.bias, NE);
    dalloc(c.emask, static_cast<size_t>(NE) * (RM / 32)); dalloc(c.tile0, NE); dalloc(c.order, NE); dalloc(c.order_bad, NE);
    g_host += ACT_B + Y_B + SLICE + sizeof(PR);                                    // the baseline act / ypart on the host, one compare slice, the plan copy
    if (g_dry) return;
    k_fill_mx4<<<nsm * 4, 256>>>(c.w1, EB1 * np / mx4::GR, 11u); k_fill_mx4<<<nsm * 4, 256>>>(c.w2, EB2 * np / mx4::GR, 12u);
    CK(cudaMemset(c.bias, 0, NE * 4));
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}
static void teardown(Ctx& c)
{
    void* ps[] = {c.w1, c.w2, c.xf, c.plan, c.act, c.y, c.log, c.bias, c.emask, c.tile0, c.order, c.order_bad};
    for (void* p : ps) if (p) cudaFree(p);
    c = Ctx();
}
// New weight pool of np experts (same fill seeds, so the first experts are the same bytes). The timing phase swaps to TPOOL experts and back.
static void set_pool(Ctx& c, int np)
{
    g_dev -= (EB1 + EB2) * c.np;
    if (c.w1) cudaFree(c.w1);
    if (c.w2) cudaFree(c.w2);
    c.w1 = c.w2 = nullptr; c.np = np;
    dalloc(c.w1, EB1 * np); dalloc(c.w2, EB2 * np);
    if (g_dry) return;
    k_fill_mx4<<<c.nsm * 4, 256>>>(c.w1, EB1 * np / mx4::GR, 11u); k_fill_mx4<<<c.nsm * 4, 256>>>(c.w2, EB2 * np / mx4::GR, 12u);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}
// Budget of the timing phase: the weight pool is TPOOL experts, the host baselines (ACT_B + Y_B) are freed.
static size_t timing_dev(int pool) { return g_dev - (EB1 + EB2) * pool + (EB1 + EB2) * TPOOL; }
static size_t timing_host() { return g_host - ACT_B - Y_B; }

// ------------------------------------------------------------------------------------------ the chain (ds41_moe_ops.cuh router_ro / experts_ro, replicated)
static mx4::ArgsMT ro_args(const Ctx& c, int sb, const int* order, int la)
{
    const char* p = c.plan;
    return mx4::ArgsMT{c.w1, c.w2, c.xf,
                       reinterpret_cast<const int*>(p + offsetof(PR, slot_expert)), reinterpret_cast<const int*>(p + offsetof(PR, slot_tile0)), reinterpret_cast<const int*>(p + offsetof(PR, slot_nt)),
                       reinterpret_cast<const int*>(p + offsetof(PR, tile_mask)), reinterpret_cast<const float*>(p + offsetof(PR, tile_w)), reinterpret_cast<const int*>(p + offsetof(PR, tile_tok)),
                       reinterpret_cast<const int*>(p + offsetof(PR, n_slots)), sb, 10.f, c.act, c.y, order, la};
}
// The two launches of ds41_moe_ops.cuh fc_ro_run (same template arguments, same grids, fc2 looks 2 x la ahead).
template <bool ORD, bool LA> static void fc_run(const mx4::ArgsMT& RA, int sb)
{
    constexpr int KS = DS41_RO_KS, CW = DS41_RO_CW, NTG1 = DS41_RO_NTG1, W2 = DS41_RO_W2, D2 = DS41_RO_D2, NTG2 = DS41_RO_NTG2;
    constexpr size_t SMEM2 = static_cast<size_t>(NTG2) * (I / 16) * 64 * sizeof(uint32_t);
    auto* k2 = mx4::k_fc2_ro<I, NOUT, W2, D2, NTG2, ORD, LA>;
    if (SMEM2 > 48 * 1024) CK(cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(SMEM2)));
    mx4::k_fc1_ro<K1, I, KS, CW, 1, NTG1, 0, ORD, LA><<<sb * (I / 16 / CW), KS * CW * 32>>>(RA);
    mx4::ArgsMT R2 = RA;
    R2.la = RA.la * 2;
    k2<<<sb * (NOUT / 16 / W2), W2 * 32, SMEM2>>>(R2);
}
struct Variant { std::string name; bool ord; int la; };
static void run_variant(const Ctx& c, const Variant& v, const int* order_arr)
{
    const int sb = dsr::ro_slots_bound(RM * TK, NE);
    CK(cudaMemset(c.act, 0xff, ACT_B)); CK(cudaMemset(c.y, 0xff, Y_B));
    const mx4::ArgsMT RA = ro_args(c, sb, v.ord ? order_arr : nullptr, v.la);
    if (v.ord) { if (v.la) fc_run<true, true>(RA, sb); else fc_run<true, false>(RA, sb); }
    else { if (v.la) fc_run<false, true>(RA, sb); else fc_run<false, false>(RA, sb); }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}

// ------------------------------------------------------------------------------------------ routes
static std::string route_path(const std::string& dir, int layer) { return dir + "/L" + std::to_string(layer) + ".router_idx.bin"; }
static std::vector<int> read_routes(const std::string& dir, int layer)             // int32 [1200][6]; empty if the file is missing or short
{
    const std::string p = route_path(dir, layer);
    std::vector<int> r;
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return r;
    r.resize(1200 * 6);
    if (fread(r.data(), 4, r.size(), f) != r.size()) r.clear();
    fclose(f);
    return r;
}
// W rows of routes. Real (default): rows 0..W-1 of layer L's p1200 file, rows >= 1200 wrap to rows 0.. of the same file; a missing or short file returns EMPTY (the caller fails).
// synthetic: skewed (half of the picks in the first 48 experts), DIR is not read.
static std::vector<int> make_routes(const std::string& dir, int layer, int W, bool synthetic, std::string& src)
{
    std::vector<int> idx(static_cast<size_t>(W) * TK);
    if (!synthetic)
    {
        static_assert(TK == 6, "the p1200 route files are int32 [1200][6]");
        const std::vector<int> a = read_routes(dir, layer);
        if (a.empty()) { src = "MISSING route file " + route_path(dir, layer); return std::vector<int>(); }
        for (int m = 0; m < W; ++m) for (int k = 0; k < TK; ++k) idx[static_cast<size_t>(m) * TK + k] = a[(m % 1200) * 6 + k];
        src = "real L" + std::to_string(layer) + " rows 0.." + std::to_string(std::min(W, 1200) - 1) + (W > 1200 ? " (the p1200 file has 1200 rows: row m >= 1200 = file row m % 1200, " + std::to_string((W + 1199) / 1200) + " passes over the file)" : "");
        return idx;
    }
    uint32_t s = 12345u + layer * 7919u;
    auto rnd = [&s]() { s = s * 1664525u + 1013904223u; return s >> 8; };
    for (int m = 0; m < W; ++m)
        for (int k = 0; k < TK; ++k)
        {
            int e;
            bool dup;
            do { e = (rnd() & 1) ? static_cast<int>(rnd() % 48) : static_cast<int>(rnd() % NE); dup = false; for (int j = 0; j < k; ++j) dup |= idx[static_cast<size_t>(m) * TK + j] == e; } while (dup);
            idx[static_cast<size_t>(m) * TK + k] = e;
        }
    src = "SYNTHETIC skewed routes (--synthetic)";
    return idx;
}
// before any CUDA call: every layer's route file must be usable unless --synthetic
static bool check_routes(const std::string& dir, const std::vector<int>& layers, bool synthetic)
{
    bool ok = true;
    if (!synthetic)
        for (int L : layers)
            if (read_routes(dir, L).empty()) { printf("FAIL: no usable route file %s (int32 [1200][6]). Pass the p1200 route directory as DIR, or --synthetic to run on synthetic routes.\n", route_path(dir, L).c_str()); ok = false; }
    return ok;
}
// logits that make the planner pick exactly these routes in this selection order: the k-th pick gets 4 - 0.5 k, every other expert -20
static void upload_logits(float* dst, const std::vector<int>& idx, int M)
{
    std::vector<float> lg(static_cast<size_t>(M) * NE, -20.f);
    for (int m = 0; m < M; ++m) for (int k = 0; k < TK; ++k) lg[static_cast<size_t>(m) * NE + idx[static_cast<size_t>(m) * TK + k]] = 4.f - 0.5f * k;
    CK(cudaMemcpy(dst, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------------------------------ the plan
// Host plan = dsr::plan_ref (the sequential specification of the device plan, ds_plan_mt.h) of W rows of routes idx. Used where the GPU planner kernels do not exist for the row
// count: they static_assert MM <= dsr::RO_MAX (2048). tile_w is a constant (the fc kernels' timing does not depend on it).
template <int MM> static std::unique_ptr<dsr::PlanMT<TK, MM>> host_plan(const std::vector<int>& idx, int W)
{
    std::vector<int> ex(static_cast<size_t>(W) * 8, 0);
    std::vector<float> wt(static_cast<size_t>(W) * 8, 0.25f);
    for (int m = 0; m < W; ++m) for (int k = 0; k < TK; ++k) ex[static_cast<size_t>(m) * 8 + k] = idx[static_cast<size_t>(m) * TK + k];
    std::unique_ptr<dsr::PlanMT<TK, MM>> P(new dsr::PlanMT<TK, MM>);
    dsr::plan_ref<TK, MM>(W, reinterpret_cast<const int (*)[8]>(ex.data()), reinterpret_cast<const float (*)[8]>(wt.data()), *P, TK);
    return P;
}
// The host plan re-derived from the routes alone (counts per expert): slots ascending by expert id, slot_nt = ceil(n / 8), slot_tile0 = the tiles below, n_tiles, and every route in
// exactly one live column. Returns "" if right, else the first thing that is wrong. nslots_out = distinct experts, maxn_out = routes of the heaviest expert.
template <int MM> static std::string check_host_plan(const dsr::PlanMT<TK, MM>& P, const std::vector<int>& idx, int W, int& nslots_out, int& maxn_out)
{
    std::vector<int> cnt(NE, 0);
    for (int r = 0; r < W * TK; ++r) { if (idx[r] < 0 || idx[r] >= NE) return "route expert id out of range"; ++cnt[idx[r]]; }
    int ns = 0, tiles = 0, maxn = 0;
    for (int e = 0; e < NE; ++e)
    {
        if (!cnt[e]) continue;
        if (P.slot_expert[ns] != e) return "slot " + std::to_string(ns) + " is expert " + std::to_string(P.slot_expert[ns]) + ", expected " + std::to_string(e);
        if (P.slot_nt[ns] != (cnt[e] + 7) / 8) return "slot_nt of expert " + std::to_string(e);
        if (P.slot_tile0[ns] != tiles) return "slot_tile0 of expert " + std::to_string(e);
        tiles += (cnt[e] + 7) / 8; ++ns; maxn = std::max(maxn, cnt[e]);
    }
    if (P.n_slots != ns || P.n_tiles != tiles) return "n_slots / n_tiles";
    long live = 0;
    for (int t = 0; t < tiles; ++t) for (int c = 0; c < 8; ++c) live += (P.tile_mask[t] >> c & 1) && P.tile_tok[t][c] >= 0 && P.tile_tok[t][c] < W;
    if (live != static_cast<long>(W) * TK) return "live columns " + std::to_string(live) + " != routes " + std::to_string(static_cast<long>(W) * TK);
    nslots_out = ns; maxn_out = maxn;
    return "";
}
// MM <= RO_MAX: the production planner chain (k_route_ro, k_scan_ro, k_place_ro) on logits that pick exactly idx, then k_order_ro. Above: the host plan, uploaded, then k_order_ro
// (k_order_ro reads only slot_nt / n_slots and has no MM limit). The condition depends on MM, so the GPU kernels are not instantiated for MM > RO_MAX.
template <int MM> static void ro_plan_t(const Ctx& c, const std::vector<int>& idx, int M)
{
    using PM = dsr::PlanMT<TK, MM>;
    if constexpr (MM <= dsr::RO_MAX)
    {
        upload_logits(c.log, idx, M);
        CK(cudaMemset(c.emask, 0, static_cast<size_t>(NE) * (MM / 32) * sizeof(unsigned)));
        dsr::k_route_ro<NE, TK, MM><<<(M + 7) / 8, 256>>>(c.log, c.bias, M, reinterpret_cast<PM*>(c.plan), c.emask);
        dsr::k_scan_ro<NE, TK, MM><<<1, NE>>>(reinterpret_cast<PM*>(c.plan), c.emask, c.tile0);
        dsr::k_place_ro<NE, TK, MM><<<(M * TK + 255) / 256, 256>>>(reinterpret_cast<PM*>(c.plan), c.emask, c.tile0, M);
    }
    else
    {
        const std::unique_ptr<PM> hp = host_plan<MM>(idx, M);
        CK(cudaMemcpy(c.plan, hp.get(), sizeof(PM), cudaMemcpyHostToDevice));
    }
    dsr::k_order_ro<NE, TK, MM><<<1, NE>>>(reinterpret_cast<const PM*>(c.plan), c.order);
    const int sb = dsr::ro_slots_bound(MM * TK, NE);
    k_pool_remap<<<(sb + 255) / 256, 256>>>(reinterpret_cast<int*>(c.plan + offsetof(PM, slot_expert)), reinterpret_cast<const int*>(c.plan + offsetof(PM, n_slots)), c.np);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}
static void ro_plan(const Ctx& c, const std::vector<int>& idx, int M) { ro_plan_t<RM>(c, idx, M); }

// Proof that the host plan is the GPU plan: at RO_MAX rows (the largest the GPU planner takes) the production chain and plan_ref run on the same routes and the structural fields
// (n_slots, n_tiles, slot_expert / slot_tile0 / slot_nt of the live slots, tile_mask and tile_tok of the live tiles) must be equal. Run before any case of a build with RM > RO_MAX.
// Returns "" if equal. Its device buffers (about 6 MB) are freed before the cases.
static std::string plan_selfcheck(const std::vector<int>& idx)
{
    constexpr int MM = dsr::RO_MAX;
    using PM = dsr::PlanMT<TK, MM>;
    float *log = nullptr, *bias = nullptr; unsigned* emask = nullptr; int* tile0 = nullptr; PM* plan = nullptr;
    CK(cudaMalloc(&log, static_cast<size_t>(MM) * NE * 4)); CK(cudaMalloc(&bias, NE * 4)); CK(cudaMalloc(&emask, static_cast<size_t>(NE) * (MM / 32) * 4));
    CK(cudaMalloc(&tile0, NE * 4)); CK(cudaMalloc(&plan, sizeof(PM)));
    CK(cudaMemset(bias, 0, NE * 4)); CK(cudaMemset(emask, 0, static_cast<size_t>(NE) * (MM / 32) * 4));
    upload_logits(log, idx, MM);
    dsr::k_route_ro<NE, TK, MM><<<(MM + 7) / 8, 256>>>(log, bias, MM, plan, emask);
    dsr::k_scan_ro<NE, TK, MM><<<1, NE>>>(plan, emask, tile0);
    dsr::k_place_ro<NE, TK, MM><<<(MM * TK + 255) / 256, 256>>>(plan, emask, tile0, MM);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::unique_ptr<PM> g(new PM);
    CK(cudaMemcpy(g.get(), plan, sizeof(PM), cudaMemcpyDeviceToHost));
    cudaFree(log); cudaFree(bias); cudaFree(emask); cudaFree(tile0); cudaFree(plan);
    const std::unique_ptr<PM> h = host_plan<MM>(idx, MM);
    if (g->n_slots != h->n_slots || g->n_tiles != h->n_tiles) return "n_slots / n_tiles: GPU " + std::to_string(g->n_slots) + " / " + std::to_string(g->n_tiles) + ", host " + std::to_string(h->n_slots) + " / " + std::to_string(h->n_tiles);
    for (int s = 0; s < h->n_slots; ++s)
        if (g->slot_expert[s] != h->slot_expert[s] || g->slot_tile0[s] != h->slot_tile0[s] || g->slot_nt[s] != h->slot_nt[s]) return "slot " + std::to_string(s) + " differs";
    for (int t = 0; t < h->n_tiles; ++t)
    {
        if (g->tile_mask[t] != h->tile_mask[t]) return "tile_mask of tile " + std::to_string(t) + " differs";
        for (int c = 0; c < 8; ++c) if (g->tile_tok[t][c] != h->tile_tok[t][c]) return "tile_tok of tile " + std::to_string(t) + " column " + std::to_string(c) + " differs";
    }
    return "";
}

// MODEL line (CPU arithmetic only): if every tile group of an expert reads the expert's fc1 weights from DRAM again, the fc1 weight bytes are groups x EB1 with
// groups = sum over the experts of ceil(nt / NTG); if the first group's read stays in L2 for the later groups, they are experts x EB1 (the once-each floor).
static void print_model(int L, int W, const std::vector<int>& nts)
{
    long tiles = 0, maxnt = 0;
    for (int n : nts) { tiles += n; maxnt = std::max<long>(maxnt, n); }
    printf("MODEL (D) L%d W%d: %zu experts, %ld tiles, heaviest expert %ld tiles; once-each floor %.0f MB (= experts x %.3f MB); every group re-reads the expert:", L, W, nts.size(), tiles, maxnt,
           nts.size() * EB1 / 1e6, EB1 / 1e6);
    for (int g : {4, 8, 16, 32})
    {
        long s = 0;
        for (int n : nts) s += (n + g - 1) / g;
        printf("  NTG%d %ld groups = %.0f MB", g, s, s * EB1 / 1e6);
    }
    printf("  NTG>=%ld (all) %zu groups = %.0f MB\n", maxnt, nts.size(), nts.size() * EB1 / 1e6);
}

// ------------------------------------------------------------------------------------------ compare
static void capture(const void* dev, std::vector<uint8_t>& ref, size_t n)
{
    ref.resize(n);
    CK(cudaMemcpy(ref.data(), dev, n, cudaMemcpyDeviceToHost));
}
// The whole buffer (the capacity, not only the live tiles), a slice at a time, against the baseline copy. Returns the number of differing bytes; first = the first differing offset.
static size_t compare(const void* dev, const std::vector<uint8_t>& ref, std::vector<uint8_t>& slice, size_t& first)
{
    size_t nbad = 0;
    first = static_cast<size_t>(-1);
    for (size_t off = 0; off < ref.size(); off += SLICE)
    {
        const size_t len = std::min(SLICE, ref.size() - off);
        CK(cudaMemcpy(slice.data(), static_cast<const uint8_t*>(dev) + off, len, cudaMemcpyDeviceToHost));
        if (!memcmp(slice.data(), ref.data() + off, len)) continue;
        for (size_t i = 0; i < len; ++i) if (slice[i] != ref[off + i]) { if (first == static_cast<size_t>(-1)) first = off + i; ++nbad; }
    }
    return nbad;
}

// ------------------------------------------------------------------------------------------ timing
static float median_of(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
// One run of fc1 + fc2 of `v` timed with events. Before the run act + ypart are poisoned (170 MB of writes: flushes the L2; outside the timed region).
static float time_run(const Ctx& c, const Variant& v, int sb, const int* order_arr, cudaEvent_t a, cudaEvent_t b)
{
    CK(cudaMemset(c.act, 0xff, ACT_B)); CK(cudaMemset(c.y, 0xff, Y_B));
    CK(cudaDeviceSynchronize());
    const mx4::ArgsMT RA = ro_args(c, sb, v.ord ? order_arr : nullptr, v.la);
    CK(cudaEventRecord(a));
    if (v.ord) { if (v.la) fc_run<true, true>(RA, sb); else fc_run<true, false>(RA, sb); }
    else { if (v.la) fc_run<false, true>(RA, sb); else fc_run<false, false>(RA, sb); }
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    CK(cudaGetLastError());
    float t;
    CK(cudaEventElapsedTime(&t, a, b));
    return t;
}
// Timing of every variant for the plan that is in c (W rows, ns slots). Swaps the weights to a TPOOL pool and back; frees the host baselines. No memcmp here (the lines above did it).
static float time_variants(Ctx& c, int L, int W, int ns, const Variant& base, const std::vector<Variant>& vars, int pool, std::vector<uint8_t>& base_act, std::vector<uint8_t>& base_y)
{
    std::vector<uint8_t>().swap(base_act); std::vector<uint8_t>().swap(base_y);
    set_pool(c, TPOOL);
    const int sb = dsr::ro_slots_bound(RM * TK, NE);
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<Variant> list;
    list.push_back(base);
    for (size_t i = 0; i < vars.size(); ++i) if (i > 0) list.push_back(vars[i]);              // vars[0] is the determinism line
    list.push_back({"baseline (flags off) again, drift", false, 0});
    printf("TIMING L%d W%d (%d slots): fc1 + fc2 GPU time, median of 20 after 5 warm-up runs; weights = pool of %d experts (%.0f MB, L2 %.0f MB) by launch position; act + ypart poisoned (%.0f MB) before each run\n"
           "  %-34s %9s %9s %9s %10s\n", L, W, ns, TPOOL, (EB1 + EB2) * TPOOL / 1048576.0, g_l2mb, (ACT_B + Y_B) / 1048576.0, "variant", "median ms", "min ms", "max ms", "vs base");
    float base_med = 0.f, best_la = 1e30f, best_ord = 1e30f;
    std::string best_la_name, best_ord_name;
    for (const Variant& v : list)
    {
        // pool expert by launch position for THIS variant's order
        k_pool_remap_pos<<<(sb + 255) / 256, 256>>>(reinterpret_cast<int*>(c.plan + offsetof(PR, slot_expert)), reinterpret_cast<const int*>(c.plan + offsetof(PR, n_slots)), v.ord ? c.order : nullptr, TPOOL);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> ms;
        for (int it = 0; it < 25; ++it) { const float t = time_run(c, v, sb, c.order, a, b); if (it >= 5) ms.push_back(t); }
        const float med = median_of(ms), mn = *std::min_element(ms.begin(), ms.end()), mxv = *std::max_element(ms.begin(), ms.end());
        if (v.name == base.name) base_med = med;
        char rel[32] = "";
        if (base_med > 0.f && v.name != base.name) snprintf(rel, sizeof rel, "%+.1f%%", (med / base_med - 1.f) * 100.f);
        printf("  %-34s %9.3f %9.3f %9.3f %10s\n", v.name.c_str(), med, mn, mxv, rel);
        if (v.la > 0 && !v.ord && med < best_la) { best_la = med; best_la_name = v.name; }
        if (v.la > 0 && v.ord && med < best_ord) { best_ord = med; best_ord_name = v.name; }
    }
    printf("  fastest L2LA alone: %s %.3f ms; fastest ORDER + L2LA: %s %.3f ms; baseline (first) %.3f ms. A difference smaller than the min..max spread of its rows is not a result.\n",
           best_la_name.c_str(), best_la, best_ord_name.c_str(), best_ord, base_med);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    set_pool(c, pool);
    return base_med;
}

static int gpu_mode(const std::string& dir, const std::vector<int>& layers, const std::vector<int>& rows, int pool, bool synthetic)
{
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    if (pr.major != 12 || pr.minor != 1) { printf("refusing to run on %s (sm_%d%d): GPU modes run only on a spark (sm_121)\n", pr.name, pr.major, pr.minor); return 3; }
    if (dsr::pw_tile_bound(RM * TK, NE) != TILES) { printf("TILES constant != dsr::pw_tile_bound\n"); return 1; }
    g_l2mb = static_cast<double>(pr.l2CacheSize) / 1048576.0;
    if (!GPU_PLAN)                                            // RM > RO_MAX: the plan is the host plan; prove it equals the GPU plan at RO_MAX rows on every layer's routes
        for (int L : layers)
        {
            std::string src;
            const std::vector<int> idx = make_routes(dir, L, dsr::RO_MAX, synthetic, src);
            const std::string bad = plan_selfcheck(idx);
            printf("plan self-check L%d (%d rows, %s): host plan_ref vs GPU planner %s%s\n", L, dsr::RO_MAX, src.c_str(), bad.empty() ? "EQUAL" : "DIFFERENT: ", bad.c_str());
            if (!bad.empty()) return 1;
        }
    Ctx c;
    setup(c, pool, pr.multiProcessorCount);
    print_budget("gpu", pool);
    if (g_dev + g_host > cap_bytes()) { printf("over the cap\n"); return 1; }
    printf("budget timing phase: device %.1f MB + host %.1f MB = %.1f MB (cap %.0f MB); weight pool %d experts\n", timing_dev(pool) / 1048576.0, timing_host() / 1048576.0,
           (timing_dev(pool) + timing_host()) / 1048576.0, cap_bytes() / 1048576.0, TPOOL);
    printf("gpu on %s, %d SMs, L2 %zu MB; production config fc1 KS%d CW%d NTG%d, fc2 W%d D%d NTG%d; weight pool %d experts; layers:", pr.name, pr.multiProcessorCount, static_cast<size_t>(pr.l2CacheSize) >> 20,
           DS41_RO_KS, DS41_RO_CW, DS41_RO_NTG1, DS41_RO_W2, DS41_RO_D2, DS41_RO_NTG2, pool);
    for (int L : layers) printf(" %d", L);
    printf("\n");
    const Variant base{"baseline (flags off)", false, 0};
    std::vector<Variant> VARS = {{"baseline again (determinism)", false, 0}, {"ORDER", true, 0}};
    for (int n : NSWEEP) VARS.push_back({"L2LA " + std::to_string(n), false, n});
    for (int n : NSWEEP) VARS.push_back({"ORDER + L2LA " + std::to_string(n), true, n});
    if (timing_dev(pool) + timing_host() > cap_bytes()) { printf("timing phase over the cap\n"); return 1; }
    std::vector<uint8_t> base_act, base_y, slice(SLICE);
    int bad_lines = 0, lines = 0, cases = 0;
    long total_order_diff = 0;
    struct Timed { int L, W, ns; float med; };
    std::vector<Timed> timed;                                                          // baseline (flags off) median per timed case, for the per-row line
    for (int L : layers)
        for (int W : rows)
        {
            std::string src;
            const std::vector<int> idx = make_routes(dir, L, W, synthetic, src);
            if (idx.empty()) { printf("L%d W%d  routes: %s\n  FAIL: no routes\n", L, W, src.c_str()); ++bad_lines; ++lines; continue; }
            ++cases;
            // x -> xfrag; x is staged in the ypart buffer (free until the first run)
            auto* x = reinterpret_cast<__nv_bfloat16*>(c.y);
            k_fill_bf16<<<c.nsm * 4, 256>>>(x, static_cast<size_t>(W) * K1, 100u + L);
            CK(cudaMemset(c.xf, 0xff, XF_B));
            const int nt = (W + 31) / 32 * 4;
            mx4::build_xfrag_mt<0><<<(nt * K1 * 4 + 255) / 256, 256>>>(x, W, K1, nt, c.xf);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            ro_plan(c, idx, W);
            int ns = 0, nti = 0;
            CK(cudaMemcpy(&ns, c.plan + offsetof(PR, n_slots), 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&nti, c.plan + offsetof(PR, n_tiles), 4, cudaMemcpyDeviceToHost));
            std::vector<int> ord(NE, -1);
            CK(cudaMemcpy(ord.data(), c.order, NE * 4, cudaMemcpyDeviceToHost));
            std::vector<char> seen(NE, 0);
            long perm_bad = ns <= 0 || ns > NE, moved = 0;
            for (int p = 0; p < ns && !perm_bad; ++p) { if (ord[p] < 0 || ord[p] >= ns || seen[ord[p]]) ++perm_bad; else seen[ord[p]] = 1; moved += ord[p] != p; }
            total_order_diff += moved;
            printf("L%d W%d  routes: %s\n  n_slots %d, n_tiles %d (capacity %d); order is %s, differs from the expert-id order at %ld of %d positions\n", L, W, src.c_str(), ns, nti, TILES,
                   perm_bad ? "NOT A PERMUTATION" : "a permutation", moved, ns);
            if (perm_bad || nti > TILES) { ++bad_lines; ++lines; printf("  FAIL: plan or order invalid\n"); continue; }
            // baseline
            run_variant(c, base, nullptr);
            capture(c.act, base_act, ACT_B); capture(c.y, base_y, Y_B);
            size_t live_ff = 0;                                                    // sanity: the live tiles of the baseline are written (no 0xFFFF fp16 NaN-poison patterns everywhere)
            for (size_t i = 0; i < static_cast<size_t>(nti) * 8 * NOUT * 2 && i < base_y.size(); i += 2) live_ff += base_y[i] == 0xff && base_y[i + 1] == 0xff;
            printf("  %-30s captured act %.1f MB + ypart %.1f MB (live ypart %.1f MB, %zu of %zu fp16 values still the 0xFFFF poison)\n", base.name.c_str(), ACT_B / 1e6, Y_B / 1e6,
                   static_cast<double>(nti) * 8 * NOUT * 2 / 1e6, live_ff, static_cast<size_t>(nti) * 8 * NOUT);
            for (const Variant& v : VARS)
            {
                run_variant(c, v, c.order);
                size_t fa, fy;
                const size_t ba = compare(c.act, base_act, slice, fa), by = compare(c.y, base_y, slice, fy);
                const bool ok = !ba && !by;
                ++lines; bad_lines += !ok;
                printf("  %-30s act identical=%s%s  ypart identical=%s%s\n", v.name.c_str(), ba ? "no" : "yes", ba ? (" (" + std::to_string(ba) + " bytes differ, first at " + std::to_string(fa) + ")").c_str() : "",
                       by ? "no" : "yes", by ? (" (" + std::to_string(by) + " bytes differ, first at " + std::to_string(fy) + ")").c_str() : "");
            }
            // CONTROL: one order entry duplicated -> one slot never runs -> the output must differ from the baseline
            if (ns >= 2)
            {
                std::vector<int> bad(ord);
                bad[1] = bad[0];
                CK(cudaMemcpy(c.order_bad, bad.data(), NE * 4, cudaMemcpyHostToDevice));
                run_variant(c, {"control", true, 0}, c.order_bad);
                size_t fa, fy;
                const size_t ba = compare(c.act, base_act, slice, fa), by = compare(c.y, base_y, slice, fy);
                const bool ok = ba || by;                                           // expected: differs
                ++lines; bad_lines += !ok;
                printf("  %-30s act identical=%s  ypart identical=%s  -> %s\n", "CONTROL (order entry duplicated)", ba ? "no" : "yes", by ? "no" : "yes", ok ? "differs, as it must (the comparison can fail)" : "IDENTICAL: THE COMPARISON CANNOT FAIL");
            }
            if (W >= 2048 && (L == 0 || L == 20)) timed.push_back({L, W, ns, time_variants(c, L, W, ns, base, VARS, pool, base_act, base_y)});
        }
    if (total_order_diff == 0) { printf("ORDER never moved a slot in any case: the ORDER lines prove nothing\n"); ++bad_lines; }
    for (size_t i = 0; i < timed.size(); ++i)                                          // per-row baseline time of each layer's timed widths
        if (i == 0 || timed[i].L != timed[i - 1].L)
        {
            size_t j = i;
            while (j < timed.size() && timed[j].L == timed[i].L) ++j;
            if (j - i < 2) continue;
            printf("PER-ROW baseline (flags off) fc1+fc2, median, L%d:", timed[i].L);
            for (size_t k = i; k < j; ++k) printf("  W%d %.3f ms = %.3f us/row (%d experts)", timed[k].W, timed[k].med, timed[k].med * 1000.f / timed[k].W, timed[k].ns);
            const float f0 = timed[i].med / timed[i].W, f1 = timed[j - 1].med / timed[j - 1].W;
            printf("  ->  W%d vs W%d per-row change %+.1f%% (a fall of 25%% or more = the per-row part is more than the fixed part)\n", timed[j - 1].W, timed[i].W, (f1 / f0 - 1.f) * 100.f);
            if (timed[j - 1].W > 1200)
                printf("  note: rows above 1200 repeat the 1200-row route file, so the experts per layer did not grow with W; a FALL here is optimistic (weak evidence), NO fall is strong evidence\n");
        }
    teardown(c);
    const std::string route_note = (synthetic ? std::string("SYNTHETIC skewed routes (--synthetic), not real traffic") : "real routes from " + dir) + ", " + std::to_string(cases) + " cases";
    if (bad_lines) { printf("FAIL: %d of %d lines not as expected (routes: %s)\n", bad_lines, lines, route_note.c_str()); return 1; }
    printf("ALL IDENTICAL (%d lines, ORDER moved %ld slot positions in total; routes: %s)\n", lines, total_order_diff, route_note.c_str());
    return 0;
}

// ------------------------------------------------------------------------------------------ ntg / ntgprof modes: k_fc1_ro alone at larger NTG
using Fc1Fn = void (*)(const mx4::ArgsMT&, int);
template <int CW, int NTG> static void fc1_launch(const mx4::ArgsMT& RA, int grid)       // the fc1 launch of fc_run (no ORD, no LAON), with this CW and NTG; grid in CTAs
{
    mx4::k_fc1_ro<K1, I, DS41_RO_KS, CW, 1, NTG, 0, false, false><<<grid, DS41_RO_KS * CW * 32>>>(RA);
}
struct NtgVar { const char* name; int cw, ntg; Fc1Fn run; const void* fn; };
#define NTG_VAR(NAME, CWV, NTGV) {NAME, CWV, NTGV, &fc1_launch<CWV, NTGV>, reinterpret_cast<const void*>(&mx4::k_fc1_ro<K1, I, DS41_RO_KS, CWV, 1, NTGV, 0, false, false>)}
static const NtgVar NTG_VARS[] = {NTG_VAR("CW2 NTG4 (production)", DS41_RO_CW, DS41_RO_NTG1), NTG_VAR("CW2 NTG8", 2, 8), NTG_VAR("CW2 NTG16", 2, 16),
                                  NTG_VAR("CW1 NTG4", 1, 4), NTG_VAR("CW1 NTG8", 1, 8), NTG_VAR("CW1 NTG16", 1, 16)};
constexpr int NV = sizeof(NTG_VARS) / sizeof(NTG_VARS[0]);
static int ntg_grid(const NtgVar& v, int slots) { return slots * (I / 16 / v.cw); }

// The case set-up of gpu_mode: x -> xfrag, the production planner chain on the routes, then the slot tile counts (nts) of the plan. False (and src says why) if the routes are unusable.
static bool stage_case(const Ctx& c, const std::string& dir, int L, int W, bool synthetic, int& ns, int& nti, std::vector<int>& nts, std::string& src)
{
    const std::vector<int> idx = make_routes(dir, L, W, synthetic, src);
    if (idx.empty()) return false;
    auto* x = reinterpret_cast<__nv_bfloat16*>(c.y);
    k_fill_bf16<<<c.nsm * 4, 256>>>(x, static_cast<size_t>(W) * K1, 100u + L);
    CK(cudaMemset(c.xf, 0xff, XF_B));
    const int nt = (W + 31) / 32 * 4;
    mx4::build_xfrag_mt<0><<<(nt * K1 * 4 + 255) / 256, 256>>>(x, W, K1, nt, c.xf);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    ro_plan(c, idx, W);
    CK(cudaMemcpy(&ns, c.plan + offsetof(PR, n_slots), 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&nti, c.plan + offsetof(PR, n_tiles), 4, cudaMemcpyDeviceToHost));
    if (ns <= 0 || ns > NE || nti > TILES) { src += " (plan invalid: n_slots " + std::to_string(ns) + ", n_tiles " + std::to_string(nti) + ")"; return false; }
    nts.assign(ns, 0);
    CK(cudaMemcpy(nts.data(), c.plan + offsetof(PR, slot_nt), ns * 4, cudaMemcpyDeviceToHost));
    return true;
}

static int ntg_mode(const std::string& dir, const std::vector<int>& layers, const std::vector<int>& rows, int pool, bool synthetic, bool prof)
{
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    if (pr.major != 12 || pr.minor != 1) { printf("refusing to run on %s (sm_%d%d): GPU modes run only on a spark (sm_121)\n", pr.name, pr.major, pr.minor); return 3; }
    if (dsr::pw_tile_bound(RM * TK, NE) != TILES) { printf("TILES constant != dsr::pw_tile_bound\n"); return 1; }
    g_l2mb = static_cast<double>(pr.l2CacheSize) / 1048576.0;
    Ctx c;
    setup(c, pool, pr.multiProcessorCount);
    g_host -= Y_B;                                                                   // these modes keep no host copy of ypart (only act is compared)
    print_budget(prof ? "ntgprof (memcmp-phase pool)" : "ntg memcmp phase", pool);
    const size_t t_dev = timing_dev(pool);
    printf("budget timing phase: device %.1f MB + host %.1f MB = %.1f MB (cap %.0f MB); weight pool %d experts\n", t_dev / 1048576.0, g_host / 1048576.0, (t_dev + g_host) / 1048576.0,
           cap_bytes() / 1048576.0, TPOOL);
    if (g_dev + g_host > cap_bytes() || t_dev + g_host > cap_bytes()) { printf("over the cap\n"); return 1; }
    printf("gpu on %s, %d SMs, L2 %zu MB; fc1 only, KS%d; variants:\n", pr.name, pr.multiProcessorCount, static_cast<size_t>(pr.l2CacheSize) >> 20, DS41_RO_KS);
    for (const NtgVar& v : NTG_VARS)
    {
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, v.fn));
        printf("  %-24s CW%d NTG%-2d  %3d threads, %3d registers, %zu bytes local memory (spill + stack), %zu bytes static smem\n", v.name, v.cw, v.ntg, DS41_RO_KS * v.cw * 32, fa.numRegs,
               static_cast<size_t>(fa.localSizeBytes), static_cast<size_t>(fa.sharedSizeBytes));
    }
    int bad = 0, lines = 0, cases = 0;
    std::vector<uint8_t> base_act, slice(SLICE);
    const int sb = dsr::ro_slots_bound(RM * TK, NE);
    cudaEvent_t ea, eb; CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
    if (prof) set_pool(c, TPOOL);                                                    // ncu needs the timing phase's weights only
    for (int L : layers)
        for (int W : rows)
        {
            int ns = 0, nti = 0;
            std::vector<int> nts;
            std::string src;
            if (!stage_case(c, dir, L, W, synthetic, ns, nti, nts, src)) { printf("L%d W%d  routes: %s\n  FAIL: no usable routes or plan\n", L, W, src.c_str()); ++bad; ++lines; continue; }
            ++cases;
            printf("L%d W%d  routes: %s\n  n_slots %d, n_tiles %d (capacity %d)\n", L, W, src.c_str(), ns, nti, TILES);
            print_model(L, W, nts);
            if (prof)
            {
                k_pool_remap_pos<<<(sb + 255) / 256, 256>>>(reinterpret_cast<int*>(c.plan + offsetof(PR, slot_expert)), reinterpret_cast<const int*>(c.plan + offsetof(PR, n_slots)), nullptr, TPOOL);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                const mx4::ArgsMT RP = ro_args(c, sb, nullptr, 0);
                for (int k = 0; k < NV; ++k)
                {
                    CK(cudaMemset(c.act, 0xff, ACT_B)); CK(cudaMemset(c.y, 0xff, Y_B)); CK(cudaDeviceSynchronize());
                    NTG_VARS[k].run(RP, ntg_grid(NTG_VARS[k], sb));
                    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                    printf("  ncu k_fc1_ro launch #%d of this case = %s\n", k + 1, NTG_VARS[k].name);
                }
                continue;
            }
            // (2) bit-equality against the production variant, whole act buffer
            const mx4::ArgsMT RA = ro_args(c, sb, nullptr, 0);
            auto run_one = [&](const NtgVar& v, int grid) { CK(cudaMemset(c.act, 0xff, ACT_B)); v.run(RA, grid); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); };
            run_one(NTG_VARS[0], ntg_grid(NTG_VARS[0], sb));
            capture(c.act, base_act, ACT_B);
            size_t live_ff = 0;                                                      // sanity: the live tiles of the production run are written (no 0xFFFF poison left)
            for (size_t i = 0; i < static_cast<size_t>(nti) * 8 * I * 2 && i < base_act.size(); i += 2) live_ff += base_act[i] == 0xff && base_act[i + 1] == 0xff;
            printf("  %-30s captured act %.1f MB (live act %.1f MB, %zu of %zu fp16 values still the 0xFFFF poison)\n", NTG_VARS[0].name, ACT_B / 1e6, static_cast<double>(nti) * 8 * I * 2 / 1e6, live_ff,
                   static_cast<size_t>(nti) * 8 * I);
            for (int k = 0; k <= NV; ++k)                                            // k == NV: the production variant a second time (determinism)
            {
                const NtgVar& v = NTG_VARS[k % NV];
                if (k == 0) continue;
                run_one(v, ntg_grid(v, sb));
                size_t fa;
                const size_t ba = compare(c.act, base_act, slice, fa);
                ++lines; bad += ba != 0;
                printf("  %-30s act identical=%s%s\n", k == NV ? "production again (determinism)" : v.name, ba ? "no" : "yes", ba ? (" (" + std::to_string(ba) + " bytes differ, first at " + std::to_string(fa) + ")").c_str() : "");
            }
            {   // CONTROL: one CTA fewer than the slots need -> its chunks of the last slot are never written -> the act must differ
                run_one(NTG_VARS[0], ntg_grid(NTG_VARS[0], ns) - 1);
                size_t fa;
                const size_t ba = compare(c.act, base_act, slice, fa);
                ++lines; bad += ba == 0;
                printf("  %-30s act identical=%s  -> %s\n", "CONTROL (one CTA fewer)", ba ? "no" : "yes", ba ? "differs, as it must (the comparison can fail)" : "IDENTICAL: THE COMPARISON CANNOT FAIL");
            }
            // (3) timing: round robin over the variants; weights = TPOOL experts by launch position (no expert twice within 100 MB of fc1), act + ypart poisoned before each run
            std::vector<uint8_t>().swap(base_act);
            set_pool(c, TPOOL);
            k_pool_remap_pos<<<(sb + 255) / 256, 256>>>(reinterpret_cast<int*>(c.plan + offsetof(PR, slot_expert)), reinterpret_cast<const int*>(c.plan + offsetof(PR, n_slots)), nullptr, TPOOL);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            const mx4::ArgsMT RT = ro_args(c, sb, nullptr, 0);                       // after set_pool: the weight pointers changed
            std::vector<std::vector<float>> ms(NV + 1);
            for (int it = 0; it < 25; ++it)
                for (int k = 0; k <= NV; ++k)                                        // k == NV: the production variant again (drift)
                {
                    const NtgVar& v = NTG_VARS[k % NV];
                    CK(cudaMemset(c.act, 0xff, ACT_B)); CK(cudaMemset(c.y, 0xff, Y_B)); CK(cudaDeviceSynchronize());
                    CK(cudaEventRecord(ea));
                    v.run(RT, ntg_grid(v, sb));
                    CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
                    CK(cudaGetLastError());
                    float t;
                    CK(cudaEventElapsedTime(&t, ea, eb));
                    if (it >= 5) ms[k].push_back(t);
                }
            printf("TIMING NTG L%d W%d (%d slots): fc1 ONLY, median of 20 rounds after 5 warm-up rounds (round robin over the variants); weights = pool of %d experts (%.0f MB, L2 %.0f MB) by launch position;"
                   " act + ypart poisoned (%.0f MB) before each run\n  %-34s %9s %9s %9s %10s\n", L, W, ns, TPOOL, (EB1 + EB2) * TPOOL / 1048576.0, g_l2mb, (ACT_B + Y_B) / 1048576.0, "variant", "median ms", "min ms", "max ms", "vs prod");
            float prod_med = 0.f;
            for (int k = 0; k <= NV; ++k)
            {
                const float med = median_of(ms[k]), mn = *std::min_element(ms[k].begin(), ms[k].end()), mxv = *std::max_element(ms[k].begin(), ms[k].end());
                if (k == 0) prod_med = med;
                char rel[32] = "";
                if (k > 0) snprintf(rel, sizeof rel, "%+.1f%%", (med / prod_med - 1.f) * 100.f);
                printf("  %-34s %9.3f %9.3f %9.3f %10s\n", k == NV ? "production again (drift)" : NTG_VARS[k].name, med, mn, mxv, rel);
            }
            printf("  A difference smaller than the min..max spread of its rows is not a result.\n");
            set_pool(c, pool);
        }
    CK(cudaEventDestroy(ea)); CK(cudaEventDestroy(eb));
    teardown(c);
    const std::string route_note = (synthetic ? std::string("SYNTHETIC skewed routes (--synthetic), not real traffic") : "real routes from " + dir) + ", " + std::to_string(cases) + " cases";
    if (prof) { printf("ntgprof done (%s); %d variants per case\n", route_note.c_str(), NV); return bad ? 1 : 0; }
    if (bad) { printf("FAIL: %d of %d lines not as expected (routes: %s)\n", bad, lines, route_note.c_str()); return 1; }
    printf("NTG ALL IDENTICAL (%d lines; routes: %s)\n", lines, route_note.c_str());
    return 0;
}

// ------------------------------------------------------------------------------------------ eq / eqprof modes: L2 eviction hints on the routed fc1
// P0 = the engine k_fc1_ro (CW2 NTG4), C0 = roeq::k_fc1_ro_t with no hint, V1w = weights evict_last, V1 = weights evict_last + x evict_first + act-store evict_first. All 512 threads.
// A copy of the structure of ntg_mode on purpose: the gated ntg / ntgprof modes are not touched.
constexpr int EQ_UPS = I / 16 / DS41_RO_CW;                                                 // CTAs per slot (18)
template <int WP, int XP, int SP> static void fc1t_launch(const mx4::ArgsMT& RA, int grid)
{
    roeq::k_fc1_ro_t<K1, I, DS41_RO_KS, DS41_RO_CW, 1, DS41_RO_NTG1, 0, WP, XP, SP><<<grid, DS41_RO_KS * DS41_RO_CW * 32>>>(RA);
}
struct EqVar { const char* name; int wp, xp, sp; Fc1Fn run; const void* fn; };
#define EQ_VAR(NAME, W, X, S) {NAME, W, X, S, &fc1t_launch<W, X, S>, reinterpret_cast<const void*>(&roeq::k_fc1_ro_t<K1, I, DS41_RO_KS, DS41_RO_CW, 1, DS41_RO_NTG1, 0, W, X, S>)}
static const EqVar EQ_VARS[] = {{"P0 (engine production)", -1, -1, -1, &fc1_launch<DS41_RO_CW, DS41_RO_NTG1>,
                                  reinterpret_cast<const void*>(&mx4::k_fc1_ro<K1, I, DS41_RO_KS, DS41_RO_CW, 1, DS41_RO_NTG1, 0, false, false>)},
                                 EQ_VAR("C0 (copy, no hint)", 0, 0, 0), EQ_VAR("V1w (weights evict_last)", 1, 0, 0), EQ_VAR("V1 (w last, x first, act first)", 1, 1, 1)};
constexpr int EQV = sizeof(EQ_VARS) / sizeof(EQ_VARS[0]);
static int eq_grid(int slots) { return slots * EQ_UPS; }

// one-SLOT MODEL line (CPU arithmetic from the plan): the first slot with the most tiles; x* = its routes x one bf16 token row (K1 x 2 B), no sector over-fetch modelled.
// E1 = weights once + x*, E2 = weights re-read once per tile group + x*. Returns the slot.
static int print_slot_model(int L, int W, const std::vector<int>& nts, const std::vector<int>& tile0, const std::vector<int>& mask)
{
    int s = 0;
    for (size_t i = 1; i < nts.size(); ++i) if (nts[i] > nts[s]) s = static_cast<int>(i);
    long routes = 0;
    for (int t = tile0[s]; t < tile0[s] + nts[s]; ++t) routes += __builtin_popcount(static_cast<unsigned>(mask[t]));
    const long groups = (nts[s] + DS41_RO_NTG1 - 1) / DS41_RO_NTG1;
    const double x = static_cast<double>(routes) * K1 * 2;
    printf("ONE-SLOT MODEL (D) L%d W%d: slot %d of %zu (first with the most tiles), nt* %d tiles, routes* %ld, groups* %ld (NTG %d); x* = routes* x %d B = %.2f MB; weights once %.2f MB, re-read per group %.2f MB;"
           " E1 = once + x* = %.2f MB, E2 = re-read + x* = %.2f MB\n", L, W, s, nts.size(), nts[s], routes, groups, DS41_RO_NTG1, K1 * 2, x / 1e6, EB1 / 1e6, groups * EB1 / 1e6, (EB1 + x) / 1e6,
           (groups * static_cast<double>(EB1) + x) / 1e6);
    return s;
}

// one-SLOT launch set-up (eqprof only): slot 0 of the plan takes slot_expert / slot_tile0 / slot_nt of slot s (3 ints), n_slots is unchanged; grid EQ_UPS then runs that one slot.
static void make_one_slot(const Ctx& c, int s)
{
    for (size_t off : {offsetof(PR, slot_expert), offsetof(PR, slot_tile0), offsetof(PR, slot_nt)})
    {
        int v = 0;
        CK(cudaMemcpy(&v, c.plan + off + 4 * static_cast<size_t>(s), 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(c.plan + off, &v, 4, cudaMemcpyHostToDevice));
    }
    CK(cudaDeviceSynchronize());
}

static int eq_mode(const std::string& dir, const std::vector<int>& layers, const std::vector<int>& rows, int pool, bool synthetic, bool prof)
{
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    if (pr.major != 12 || pr.minor != 1) { printf("refusing to run on %s (sm_%d%d): GPU modes run only on a spark (sm_121)\n", pr.name, pr.major, pr.minor); return 3; }
    if (dsr::pw_tile_bound(RM * TK, NE) != TILES) { printf("TILES constant != dsr::pw_tile_bound\n"); return 1; }
    g_l2mb = static_cast<double>(pr.l2CacheSize) / 1048576.0;
    Ctx c;
    setup(c, pool, pr.multiProcessorCount);
    g_host -= Y_B;                                                                   // these modes keep no host copy of ypart (only act is compared)
    print_budget(prof ? "eqprof (memcmp-phase pool)" : "eq memcmp phase", pool);
    const size_t t_dev = timing_dev(pool);
    printf("budget timing phase: device %.1f MB + host %.1f MB = %.1f MB (cap %.0f MB); weight pool %d experts\n", t_dev / 1048576.0, g_host / 1048576.0, (t_dev + g_host) / 1048576.0,
           cap_bytes() / 1048576.0, TPOOL);
    if (g_dev + g_host > cap_bytes() || t_dev + g_host > cap_bytes()) { printf("over the cap\n"); return 1; }
    printf("gpu on %s, %d SMs, L2 %zu MB; fc1 only, KS%d CW%d NTG%d; variants:\n", pr.name, pr.multiProcessorCount, static_cast<size_t>(pr.l2CacheSize) >> 20, DS41_RO_KS, DS41_RO_CW, DS41_RO_NTG1);
    for (const EqVar& v : EQ_VARS)
    {
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, v.fn));
        printf("  %-34s WPOL %2d XPOL %2d SPOL %2d  %3d threads, %3d registers, %zu bytes local memory (spill + stack), %zu bytes static smem\n", v.name, v.wp, v.xp, v.sp, DS41_RO_KS * DS41_RO_CW * 32,
               fa.numRegs, static_cast<size_t>(fa.localSizeBytes), static_cast<size_t>(fa.sharedSizeBytes));
    }
    int bad = 0, lines = 0, cases = 0;
    std::vector<uint8_t> base_act, slice(SLICE);
    const int sb = dsr::ro_slots_bound(RM * TK, NE);
    cudaEvent_t ea, eb; CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
    if (prof) set_pool(c, TPOOL);                                                    // ncu needs the timing phase's weights only
    for (int L : layers)
        for (int W : rows)
        {
            int ns = 0, nti = 0;
            std::vector<int> nts;
            std::string src;
            if (!stage_case(c, dir, L, W, synthetic, ns, nti, nts, src)) { printf("L%d W%d  routes: %s\n  FAIL: no usable routes or plan\n", L, W, src.c_str()); ++bad; ++lines; continue; }
            ++cases;
            printf("L%d W%d  routes: %s\n  n_slots %d, n_tiles %d (capacity %d)\n", L, W, src.c_str(), ns, nti, TILES);
            print_model(L, W, nts);
            std::vector<int> t0s(ns), masks(nti);
            CK(cudaMemcpy(t0s.data(), c.plan + offsetof(PR, slot_tile0), ns * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(masks.data(), c.plan + offsetof(PR, tile_mask), nti * 4, cudaMemcpyDeviceToHost));
            const int s_star = print_slot_model(L, W, nts, t0s, masks);
            if (prof)
            {
                k_pool_remap_pos<<<(sb + 255) / 256, 256>>>(reinterpret_cast<int*>(c.plan + offsetof(PR, slot_expert)), reinterpret_cast<const int*>(c.plan + offsetof(PR, n_slots)), nullptr, TPOOL);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                const mx4::ArgsMT RP = ro_args(c, sb, nullptr, 0);
                int k = 0;
                for (int v = 0; v < EQV; ++v, ++k)                                   // launches 1-4: full grid
                {
                    CK(cudaMemset(c.act, 0xff, ACT_B)); CK(cudaMemset(c.y, 0xff, Y_B)); CK(cudaDeviceSynchronize());
                    EQ_VARS[v].run(RP, eq_grid(sb));
                    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                    printf("  ncu k_fc1_ro launch #%d of this case = FULL GRID (%d CTAs) %s\n", k + 1, eq_grid(sb), EQ_VARS[v].name);
                }
                make_one_slot(c, s_star);                                            // launches 5-7: one slot (EQ_UPS CTAs) of the heaviest slot, P0 / V1w / V1
                const mx4::ArgsMT R1 = ro_args(c, sb, nullptr, 0);
                for (int v : {0, 2, 3})
                {
                    CK(cudaMemset(c.act, 0xff, ACT_B)); CK(cudaMemset(c.y, 0xff, Y_B)); CK(cudaDeviceSynchronize());
                    EQ_VARS[v].run(R1, EQ_UPS);
                    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                    printf("  ncu k_fc1_ro launch #%d of this case = ONE SLOT (%d CTAs, slot %d copied to slot 0) %s\n", ++k, EQ_UPS, s_star, EQ_VARS[v].name);
                }
                continue;
            }
            // (2) bit-equality against P0, whole act buffer
            const mx4::ArgsMT RA = ro_args(c, sb, nullptr, 0);
            auto run_one = [&](const EqVar& v, int grid) { CK(cudaMemset(c.act, 0xff, ACT_B)); v.run(RA, grid); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); };
            run_one(EQ_VARS[0], eq_grid(sb));
            capture(c.act, base_act, ACT_B);
            size_t live_ff = 0;                                                      // sanity: the live tiles of the P0 run are written (no 0xFFFF poison left)
            for (size_t i = 0; i < static_cast<size_t>(nti) * 8 * I * 2 && i < base_act.size(); i += 2) live_ff += base_act[i] == 0xff && base_act[i + 1] == 0xff;
            printf("  %-34s captured act %.1f MB (live act %.1f MB, %zu of %zu fp16 values still the 0xFFFF poison)\n", EQ_VARS[0].name, ACT_B / 1e6, static_cast<double>(nti) * 8 * I * 2 / 1e6, live_ff,
                   static_cast<size_t>(nti) * 8 * I);
            for (int k = 1; k <= EQV; ++k)                                           // k == EQV: P0 a second time (determinism)
            {
                const EqVar& v = EQ_VARS[k % EQV];
                run_one(v, eq_grid(sb));
                size_t fa;
                const size_t ba = compare(c.act, base_act, slice, fa);
                ++lines; bad += ba != 0;
                printf("  %-34s act identical=%s%s\n", k == EQV ? "production again (determinism)" : v.name, ba ? "no" : "yes", ba ? (" (" + std::to_string(ba) + " bytes differ, first at " + std::to_string(fa) + ")").c_str() : "");
            }
            {   // CONTROL: one CTA fewer than the slots need -> its chunks of the last slot are never written -> the act must differ
                run_one(EQ_VARS[0], eq_grid(ns) - 1);
                size_t fa;
                const size_t ba = compare(c.act, base_act, slice, fa);
                ++lines; bad += ba == 0;
                printf("  %-34s act identical=%s  -> %s\n", "CONTROL (one CTA fewer)", ba ? "no" : "yes", ba ? "differs, as it must (the comparison can fail)" : "IDENTICAL: THE COMPARISON CANNOT FAIL");
            }
            // (3) timing: round robin over the variants; weights = TPOOL experts by launch position, act + ypart poisoned before each run
            std::vector<uint8_t>().swap(base_act);
            set_pool(c, TPOOL);
            k_pool_remap_pos<<<(sb + 255) / 256, 256>>>(reinterpret_cast<int*>(c.plan + offsetof(PR, slot_expert)), reinterpret_cast<const int*>(c.plan + offsetof(PR, n_slots)), nullptr, TPOOL);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            const mx4::ArgsMT RT = ro_args(c, sb, nullptr, 0);                       // after set_pool: the weight pointers changed
            std::vector<std::vector<float>> ms(EQV + 1);
            for (int it = 0; it < 25; ++it)
                for (int k = 0; k <= EQV; ++k)                                       // k == EQV: P0 again (drift)
                {
                    const EqVar& v = EQ_VARS[k % EQV];
                    CK(cudaMemset(c.act, 0xff, ACT_B)); CK(cudaMemset(c.y, 0xff, Y_B)); CK(cudaDeviceSynchronize());
                    CK(cudaEventRecord(ea));
                    v.run(RT, eq_grid(sb));
                    CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
                    CK(cudaGetLastError());
                    float t;
                    CK(cudaEventElapsedTime(&t, ea, eb));
                    if (it >= 5) ms[k].push_back(t);
                }
            printf("TIMING EQ L%d W%d (%d slots): fc1 ONLY, median of 20 rounds after 5 warm-up rounds (round robin over the variants); weights = pool of %d experts (%.0f MB, L2 %.0f MB) by launch position;"
                   " act + ypart poisoned (%.0f MB) before each run\n  %-34s %9s %9s %9s %10s\n", L, W, ns, TPOOL, (EB1 + EB2) * TPOOL / 1048576.0, g_l2mb, (ACT_B + Y_B) / 1048576.0, "variant", "median ms", "min ms", "max ms", "vs prod");
            float prod_med = 0.f;
            for (int k = 0; k <= EQV; ++k)
            {
                const float med = median_of(ms[k]), mn = *std::min_element(ms[k].begin(), ms[k].end()), mxv = *std::max_element(ms[k].begin(), ms[k].end());
                if (k == 0) prod_med = med;
                char rel[32] = "";
                if (k > 0) snprintf(rel, sizeof rel, "%+.1f%%", (med / prod_med - 1.f) * 100.f);
                printf("  %-34s %9.3f %9.3f %9.3f %10s\n", k == EQV ? "production again (drift)" : EQ_VARS[k].name, med, mn, mxv, rel);
            }
            printf("  A difference smaller than the min..max spread of its rows is not a result.\n");
            set_pool(c, pool);
        }
    CK(cudaEventDestroy(ea)); CK(cudaEventDestroy(eb));
    teardown(c);
    const std::string route_note = (synthetic ? std::string("SYNTHETIC skewed routes (--synthetic), not real traffic") : "real routes from " + dir) + ", " + std::to_string(cases) + " cases";
    if (prof) { printf("eqprof done (%s); 7 launches per case (4 full grid, 3 one slot)\n", route_note.c_str()); return bad ? 1 : 0; }
    if (bad) { printf("FAIL: %d of %d lines not as expected (routes: %s)\n", bad, lines, route_note.c_str()); return 1; }
    printf("EQ ALL IDENTICAL (%d lines; routes: %s)\n", lines, route_note.c_str());
    return 0;
}

// ------------------------------------------------------------------------------------------ l2h / l2hprof modes: the engine kernels with GR8_DS_MOE_L2HINT
// The engine's own k_fc1_ro / k_fc2_ro with the L2H template argument (ds_prefill_ro.cuh) that ds41_moe_ops.cuh fc_ro_run instantiates for the flag: HV = the flag value, 0 = no hint,
// 1 = fc1 weight loads evict_last, 2 = fc1 and fc2. A copy of the structure of eq_mode on purpose: the gated modes are not touched.
constexpr size_t L2H_SMEM2 = static_cast<size_t>(DS41_RO_NTG2) * (I / 16) * 64 * sizeof(uint32_t);     // fc2 act staging, as fc_run
template <int HV> static void l2h_fc1(const mx4::ArgsMT& RA, int grid)
{
    mx4::k_fc1_ro<K1, I, DS41_RO_KS, DS41_RO_CW, 1, DS41_RO_NTG1, 0, false, false, (HV >= 1)><<<grid, DS41_RO_KS * DS41_RO_CW * 32>>>(RA);
}
template <int HV> static void l2h_fc2(const mx4::ArgsMT& RA, int grid)
{
    mx4::k_fc2_ro<I, NOUT, DS41_RO_W2, DS41_RO_D2, DS41_RO_NTG2, false, false, (HV == 2)><<<grid, DS41_RO_W2 * 32, L2H_SMEM2>>>(RA);
}
static int l2h_grid1(int slots) { return slots * EQ_UPS; }                          // fc1 CTAs: 18 per slot
static int l2h_grid2(int slots) { return slots * (NOUT / 16 / DS41_RO_W2); }       // fc2 CTAs: 40 per slot
template <int HV> static void l2h_chain(const mx4::ArgsMT& RA, int sb) { l2h_fc1<HV>(RA, l2h_grid1(sb)); l2h_fc2<HV>(RA, l2h_grid2(sb)); }   // the two launches of fc_ro_run, la = 0
static void l2h_prepare()                                                           // the dynamic smem attribute of both fc2 instances, once, host side (not inside a timed region)
{
    if (L2H_SMEM2 > 48 * 1024)
    {
        CK(cudaFuncSetAttribute(mx4::k_fc2_ro<I, NOUT, DS41_RO_W2, DS41_RO_D2, DS41_RO_NTG2, false, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(L2H_SMEM2)));
        CK(cudaFuncSetAttribute(mx4::k_fc2_ro<I, NOUT, DS41_RO_W2, DS41_RO_D2, DS41_RO_NTG2, false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(L2H_SMEM2)));
    }
}
// kind 0 = fc1 alone (arg = grid), 1 = fc2 alone (arg = grid), 2 = both launches (arg = slots bound); base = the item this row is compared with
struct L2hItem { const char* name; int kind; Fc1Fn run; int base; };
static const L2hItem L2H_ITEMS[] = {
    {"fc1 P0 (engine, L2H off)", 0, &l2h_fc1<0>, 0},
    {"fc1 P1 (engine, L2H on)", 0, &l2h_fc1<1>, 0},
    {"fc1 V1w (test copy, w evict_last)", 0, &fc1t_launch<1, 0, 0>, 0},
    {"fc2 Q0 (engine, L2H off)", 1, &l2h_fc2<0>, 3},
    {"fc2 Q1 (engine, L2H on)", 1, &l2h_fc2<2>, 3},
    {"chain H0 (value 0)", 2, &l2h_chain<0>, 5},
    {"chain H1 (value 1: fc1)", 2, &l2h_chain<1>, 5},
    {"chain H2 (value 2: fc1 + fc2)", 2, &l2h_chain<2>, 5},
    {"fc1 P0 again (drift)", 0, &l2h_fc1<0>, 0},
    {"fc2 Q0 again (drift)", 1, &l2h_fc2<0>, 3}};
constexpr int L2HN = sizeof(L2H_ITEMS) / sizeof(L2H_ITEMS[0]);
static void l2h_go(const L2hItem& it, const mx4::ArgsMT& R, int sb) { it.run(R, it.kind == 0 ? l2h_grid1(sb) : it.kind == 1 ? l2h_grid2(sb) : sb); }
// fc2 reads the act that the fc1 (or chain) item before it wrote (all of them write the same bits, gate G); an fc2 item poisons ypart only.
static void l2h_poison(const Ctx& c, int kind) { if (kind != 1) CK(cudaMemset(c.act, 0xff, ACT_B)); CK(cudaMemset(c.y, 0xff, Y_B)); CK(cudaDeviceSynchronize()); }

static int l2h_mode(const std::string& dir, const std::vector<int>& layers, const std::vector<int>& rows, int pool, bool synthetic, bool prof)
{
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    if (pr.major != 12 || pr.minor != 1) { printf("refusing to run on %s (sm_%d%d): GPU modes run only on a spark (sm_121)\n", pr.name, pr.major, pr.minor); return 3; }
    if (dsr::pw_tile_bound(RM * TK, NE) != TILES) { printf("TILES constant != dsr::pw_tile_bound\n"); return 1; }
    g_l2mb = static_cast<double>(pr.l2CacheSize) / 1048576.0;
    if (!GPU_PLAN)                                            // RM > RO_MAX: the plan is the host plan; prove it equals the GPU plan at RO_MAX rows on every layer's routes (as gpu_mode)
        for (int L : layers)
        {
            std::string src;
            const std::vector<int> idx = make_routes(dir, L, dsr::RO_MAX, synthetic, src);
            const std::string bad = plan_selfcheck(idx);
            printf("plan self-check L%d (%d rows, %s): host plan_ref vs GPU planner %s%s\n", L, dsr::RO_MAX, src.c_str(), bad.empty() ? "EQUAL" : "DIFFERENT: ", bad.c_str());
            if (!bad.empty()) return 1;
        }
    Ctx c;
    setup(c, pool, pr.multiProcessorCount);
    if (prof) g_host -= ACT_B + Y_B;                                                 // l2hprof keeps no host baselines
    print_budget(prof ? "l2hprof (memcmp-phase pool)" : "l2h memcmp phase (host act + ypart baselines)", pool);
    const size_t t_dev = timing_dev(pool), t_host = prof ? g_host : timing_host();
    printf("budget timing phase: device %.1f MB + host %.1f MB = %.1f MB (cap %.0f MB); weight pool %d experts\n", t_dev / 1048576.0, t_host / 1048576.0, (t_dev + t_host) / 1048576.0,
           cap_bytes() / 1048576.0, TPOOL);
    if (g_dev + g_host > cap_bytes() || t_dev + t_host > cap_bytes()) { printf("over the cap\n"); return 1; }
    l2h_prepare();
    printf("gpu on %s, %d SMs, L2 %zu MB; engine kernels, fc1 KS%d CW%d NTG%d, fc2 W%d D%d NTG%d (la = 0, no order); kernels of this build:\n", pr.name, pr.multiProcessorCount, static_cast<size_t>(pr.l2CacheSize) >> 20,
           DS41_RO_KS, DS41_RO_CW, DS41_RO_NTG1, DS41_RO_W2, DS41_RO_D2, DS41_RO_NTG2);
    {
        struct KF { const char* name; const void* fn; int thr; };
        const KF kfs[] = {
            {"k_fc1_ro L2H off", reinterpret_cast<const void*>(&mx4::k_fc1_ro<K1, I, DS41_RO_KS, DS41_RO_CW, 1, DS41_RO_NTG1, 0, false, false, false>), DS41_RO_KS * DS41_RO_CW * 32},
            {"k_fc1_ro L2H on", reinterpret_cast<const void*>(&mx4::k_fc1_ro<K1, I, DS41_RO_KS, DS41_RO_CW, 1, DS41_RO_NTG1, 0, false, false, true>), DS41_RO_KS * DS41_RO_CW * 32},
            {"k_fc1_ro_t V1w (test copy)", reinterpret_cast<const void*>(&roeq::k_fc1_ro_t<K1, I, DS41_RO_KS, DS41_RO_CW, 1, DS41_RO_NTG1, 0, 1, 0, 0>), DS41_RO_KS * DS41_RO_CW * 32},
            {"k_fc2_ro L2H off", reinterpret_cast<const void*>(&mx4::k_fc2_ro<I, NOUT, DS41_RO_W2, DS41_RO_D2, DS41_RO_NTG2, false, false, false>), DS41_RO_W2 * 32},
            {"k_fc2_ro L2H on", reinterpret_cast<const void*>(&mx4::k_fc2_ro<I, NOUT, DS41_RO_W2, DS41_RO_D2, DS41_RO_NTG2, false, false, true>), DS41_RO_W2 * 32}};
        for (const KF& k : kfs)
        {
            cudaFuncAttributes fa;
            CK(cudaFuncGetAttributes(&fa, k.fn));
            printf("  %-30s %3d threads, %3d registers, %zu bytes local memory (spill + stack), %zu bytes static smem\n", k.name, k.thr, fa.numRegs, static_cast<size_t>(fa.localSizeBytes),
                   static_cast<size_t>(fa.sharedSizeBytes));
        }
    }
    int bad = 0, lines = 0, cases = 0;
    std::vector<uint8_t> base_act, base_y, slice(SLICE);
    const int sb = dsr::ro_slots_bound(RM * TK, NE);
    cudaEvent_t ea, eb; CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
    if (prof) set_pool(c, TPOOL);                                                    // ncu needs the timing phase's weights only
    for (int L : layers)
        for (int W : rows)
        {
            int ns = 0, nti = 0;
            std::vector<int> nts;
            std::string src;
            if (!stage_case(c, dir, L, W, synthetic, ns, nti, nts, src)) { printf("L%d W%d  routes: %s\n  FAIL: no usable routes or plan\n", L, W, src.c_str()); ++bad; ++lines; continue; }
            ++cases;
            printf("L%d W%d  routes: %s\n  n_slots %d, n_tiles %d (capacity %d)\n", L, W, src.c_str(), ns, nti, TILES);
            print_model(L, W, nts);
            if (prof)
            {
                k_pool_remap_pos<<<(sb + 255) / 256, 256>>>(reinterpret_cast<int*>(c.plan + offsetof(PR, slot_expert)), reinterpret_cast<const int*>(c.plan + offsetof(PR, n_slots)), nullptr, TPOOL);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                const mx4::ArgsMT RP = ro_args(c, sb, nullptr, 0);
                int k = 0;
                for (int v : {0, 1, 3, 4})                                           // launches 1-4: fc1 P0, fc1 P1, fc2 Q0, fc2 Q1 (the fc2 ones read the act of launch 2)
                {
                    l2h_poison(c, L2H_ITEMS[v].kind);
                    l2h_go(L2H_ITEMS[v], RP, sb);
                    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                    printf("  ncu k_fc[12]_ro launch #%d of this case = %s (full grid, %d CTAs)\n", ++k, L2H_ITEMS[v].name, L2H_ITEMS[v].kind == 0 ? l2h_grid1(sb) : l2h_grid2(sb));
                }
                continue;
            }
            // (2) gate G: chain with value 0 = the baseline; values 1, 2 and 0 again must leave the whole act and ypart buffers bit-identical
            const mx4::ArgsMT RA = ro_args(c, sb, nullptr, 0);
            auto chain = [&](int hv, int g1, int g2)
            {
                CK(cudaMemset(c.act, 0xff, ACT_B)); CK(cudaMemset(c.y, 0xff, Y_B));
                if (hv == 2) { l2h_fc1<2>(RA, g1); l2h_fc2<2>(RA, g2); }
                else if (hv == 1) { l2h_fc1<1>(RA, g1); l2h_fc2<1>(RA, g2); }
                else { l2h_fc1<0>(RA, g1); l2h_fc2<0>(RA, g2); }
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            };
            chain(0, l2h_grid1(sb), l2h_grid2(sb));
            capture(c.act, base_act, ACT_B);
            capture(c.y, base_y, Y_B);
            size_t live_a = 0, live_y = 0;                                           // sanity: the live tiles of the value-0 run are written (no 0xFFFF poison left)
            for (size_t i = 0; i < static_cast<size_t>(nti) * 8 * I * 2 && i < base_act.size(); i += 2) live_a += base_act[i] == 0xff && base_act[i + 1] == 0xff;
            for (size_t i = 0; i < static_cast<size_t>(nti) * 8 * NOUT * 2 && i < base_y.size(); i += 2) live_y += base_y[i] == 0xff && base_y[i + 1] == 0xff;
            printf("  %-34s captured act %.1f MB (live %.1f MB, %zu of %zu fp16 values still the 0xFFFF poison) and ypart %.1f MB (live %.1f MB, %zu of %zu still poison)\n", "value 0 (flag off)", ACT_B / 1e6,
                   static_cast<double>(nti) * 8 * I * 2 / 1e6, live_a, static_cast<size_t>(nti) * 8 * I, Y_B / 1e6, static_cast<double>(nti) * 8 * NOUT * 2 / 1e6, live_y, static_cast<size_t>(nti) * 8 * NOUT);
            for (int k = 1; k <= 3; ++k)                                             // k == 3: value 0 a second time (determinism)
            {
                const int hv = k == 3 ? 0 : k;
                chain(hv, l2h_grid1(sb), l2h_grid2(sb));
                size_t fa, fy;
                const size_t ba = compare(c.act, base_act, slice, fa), by = compare(c.y, base_y, slice, fy);
                lines += 2; bad += (ba != 0) + (by != 0);
                char nm[48];
                snprintf(nm, sizeof nm, k == 3 ? "value 0 again (determinism)" : "value %d", hv);
                printf("  %-34s act identical=%s%s  ypart identical=%s%s\n", nm, ba ? "no" : "yes", ba ? (" (" + std::to_string(ba) + " bytes differ, first at " + std::to_string(fa) + ")").c_str() : "",
                       by ? "no" : "yes", by ? (" (" + std::to_string(by) + " bytes differ, first at " + std::to_string(fy) + ")").c_str() : "");
            }
            {   // CONTROLS: one CTA fewer than the slots need -> the last chunks of the last slot are never written -> the buffer must differ
                size_t fa, fy;
                chain(0, l2h_grid1(ns) - 1, l2h_grid2(sb));
                const size_t ba = compare(c.act, base_act, slice, fa);
                ++lines; bad += ba == 0;
                printf("  %-34s act identical=%s  -> %s\n", "CONTROL fc1 (one CTA fewer)", ba ? "no" : "yes", ba ? "differs, as it must (the act comparison can fail)" : "IDENTICAL: THE ACT COMPARISON CANNOT FAIL");
                chain(0, l2h_grid1(sb), l2h_grid2(ns) - 1);
                const size_t by = compare(c.y, base_y, slice, fy);
                ++lines; bad += by == 0;
                printf("  %-34s ypart identical=%s  -> %s\n", "CONTROL fc2 (one CTA fewer)", by ? "no" : "yes", by ? "differs, as it must (the ypart comparison can fail)" : "IDENTICAL: THE YPART COMPARISON CANNOT FAIL");
            }
            // (3) timing: round robin over the items; weights = TPOOL experts by launch position
            std::vector<uint8_t>().swap(base_act); std::vector<uint8_t>().swap(base_y);
            set_pool(c, TPOOL);
            k_pool_remap_pos<<<(sb + 255) / 256, 256>>>(reinterpret_cast<int*>(c.plan + offsetof(PR, slot_expert)), reinterpret_cast<const int*>(c.plan + offsetof(PR, n_slots)), nullptr, TPOOL);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            const mx4::ArgsMT RT = ro_args(c, sb, nullptr, 0);                       // after set_pool: the weight pointers changed
            std::vector<std::vector<float>> ms(L2HN);
            for (int it = 0; it < 25; ++it)
                for (int k = 0; k < L2HN; ++k)
                {
                    l2h_poison(c, L2H_ITEMS[k].kind);
                    CK(cudaEventRecord(ea));
                    l2h_go(L2H_ITEMS[k], RT, sb);
                    CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
                    CK(cudaGetLastError());
                    float t;
                    CK(cudaEventElapsedTime(&t, ea, eb));
                    if (it >= 5) ms[k].push_back(t);
                }
            printf("TIMING L2H L%d W%d (%d slots): ENGINE kernels, median of 20 rounds after 5 warm-up rounds (round robin over the items in this order); weights = pool of %d experts (%.0f MB, L2 %.0f MB) by launch position;"
                   " fc1 and chain items poison act + ypart (%.0f MB) before each run, fc2 items poison ypart only (%.0f MB; the act is the one the item before wrote)\n  %-34s %9s %9s %9s %10s\n", L, W, ns, TPOOL,
                   (EB1 + EB2) * TPOOL / 1048576.0, g_l2mb, (ACT_B + Y_B) / 1048576.0, Y_B / 1048576.0, "item", "median ms", "min ms", "max ms", "vs base");
            for (int k = 0; k < L2HN; ++k)
            {
                const float med = median_of(ms[k]), mn = *std::min_element(ms[k].begin(), ms[k].end()), mxv = *std::max_element(ms[k].begin(), ms[k].end());
                char rel[32] = "";
                if (L2H_ITEMS[k].base != k) snprintf(rel, sizeof rel, "%+.1f%%", (med / median_of(ms[L2H_ITEMS[k].base]) - 1.f) * 100.f);      // the drift rows (P0, Q0 again) read against their first row
                printf("  %-34s %9.3f %9.3f %9.3f %10s\n", L2H_ITEMS[k].name, med, mn, mxv, rel);
            }
            printf("  A difference smaller than the min..max spread of its rows is not a result. fc1, fc2 and chain times of this harness are not layer times.\n");
            set_pool(c, pool);
        }
    CK(cudaEventDestroy(ea)); CK(cudaEventDestroy(eb));
    teardown(c);
    const std::string route_note = (synthetic ? std::string("SYNTHETIC skewed routes (--synthetic), not real traffic") : "real routes from " + dir) + ", " + std::to_string(cases) + " cases";
    if (prof) { printf("l2hprof done (%s); 4 k_fc[12]_ro launches per case (fc1 P0, fc1 P1, fc2 Q0, fc2 Q1)\n", route_note.c_str()); return bad ? 1 : 0; }
    if (bad) { printf("FAIL: %d of %d lines not as expected (routes: %s)\n", bad, lines, route_note.c_str()); return 1; }
    printf("L2H ALL IDENTICAL (%d lines; routes: %s)\n", lines, route_note.c_str());
    return 0;
}

// CPU-only (no CUDA call, like budget): the host plan of every (layer, rows) case, re-derived from the routes alone. What the cases of a gpu run would feed the fc kernels.
static int plan_mode(const std::string& dir, const std::vector<int>& layers, const std::vector<int>& rows, bool synthetic)
{
    int bad = 0;
    for (int L : layers)
        for (int W : rows)
        {
            std::string src;
            const std::vector<int> idx = make_routes(dir, L, W, synthetic, src);
            if (idx.empty()) { printf("L%d W%d  routes: %s\n  FAIL: no routes\n", L, W, src.c_str()); ++bad; continue; }
            const std::unique_ptr<PR> hp = host_plan<RM>(idx, W);
            int ns = 0, maxn = 0;
            const std::string err = check_host_plan<RM>(*hp, idx, W, ns, maxn);
            printf("L%d W%d  routes: %s\n  host plan: %d distinct experts, %d tiles (capacity %d), %d routes, heaviest expert %d routes, mean %.1f routes per expert -> %s\n", L, W, src.c_str(), ns, hp->n_tiles, TILES, W * TK,
                   maxn, ns ? static_cast<double>(W) * TK / ns : 0.0, err.empty() ? "plan matches the routes" : ("FAIL: " + err).c_str());
            if (err.empty())
            {
                print_model(L, W, std::vector<int>(hp->slot_nt, hp->slot_nt + ns));
                print_slot_model(L, W, std::vector<int>(hp->slot_nt, hp->slot_nt + ns), std::vector<int>(hp->slot_tile0, hp->slot_tile0 + ns), std::vector<int>(hp->tile_mask, hp->tile_mask + hp->n_tiles));
            }
            bad += !err.empty() || hp->n_tiles > TILES;
        }
    if (bad) printf("FAIL: %d plan(s) wrong\n", bad);
    else printf("all host plans right (RM %d)\n", RM);
    return bad ? 1 : 0;
}

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "";
    int pool = 4;
    std::string dir;                                                               // the p1200 route directory: required unless --synthetic
    std::vector<int> layers;
    bool synthetic = false;
    std::vector<int> rows = {2048, 846};
    for (int i = 2; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--synthetic")) synthetic = true;
        else if (argv[i][0] == '-') { printf("unknown option %s (only --synthetic)\n", argv[i]); return 1; }
        else if (!strncmp(argv[i], "pool=", 5)) pool = atoi(argv[i] + 5);
        else if (!strncmp(argv[i], "rows=", 5))                                     // rows=A,B,...: the row counts of the cases (default 2048,846)
        {
            rows.clear();
            for (const char* q = argv[i] + 5; *q;)
            {
                char* e = nullptr;
                const long v = strtol(q, &e, 10);
                if (e == q || v < 1 || (*e != ',' && *e != 0)) { printf("rows=A,B,...: positive integers separated by commas (got '%s')\n", argv[i] + 5); return 1; }
                rows.push_back(static_cast<int>(v));
                q = *e == ',' ? e + 1 : e;
            }
            if (rows.empty()) { printf("rows= needs a list\n"); return 1; }
        }
        else if (argv[i][0] == '/' || argv[i][0] == '.') dir = argv[i];
        else layers.push_back(atoi(argv[i]));
    }
    if (layers.empty()) layers = {0, 20, 39};
    if (pool < 2 || pool > 64) { printf("pool=N needs 2..64\n"); return 1; }
    for (int W : rows) if (W > RM) { printf("rows=%d is above this build's capacity RM %d (build with -DB6RO_RM=%d; the 400 MB cap then needs GR8_TEST_DS41_RO_CAP_MB, see the budget line)\n", W, RM, (W + 63) / 64 * 64); return 1; }
    if (!strcmp(mode, "plan")) return check_routes(dir, layers, synthetic) ? plan_mode(dir, layers, rows, synthetic) : 1;
    if ((!strcmp(mode, "gpu") || !strcmp(mode, "ntg") || !strcmp(mode, "ntgprof") || !strcmp(mode, "eq") || !strcmp(mode, "eqprof") || !strcmp(mode, "l2h") || !strcmp(mode, "l2hprof")) && !host_is_spark())
    {
        printf("refusing: this host is not a spark (uname -m is not aarch64), so %s mode makes no CUDA call here; GPU modes run only on a spark\n", mode);
        return 3;
    }
    if (!strcmp(mode, "gpu")) return check_routes(dir, layers, synthetic) ? gpu_mode(dir, layers, rows, pool, synthetic) : 1;
    if (!strcmp(mode, "ntg") || !strcmp(mode, "ntgprof")) return check_routes(dir, layers, synthetic) ? ntg_mode(dir, layers, rows, pool, synthetic, !strcmp(mode, "ntgprof")) : 1;
    if (!strcmp(mode, "eq") || !strcmp(mode, "eqprof")) return check_routes(dir, layers, synthetic) ? eq_mode(dir, layers, rows, pool, synthetic, !strcmp(mode, "eqprof")) : 1;
    if (!strcmp(mode, "l2h") || !strcmp(mode, "l2hprof")) return check_routes(dir, layers, synthetic) ? l2h_mode(dir, layers, rows, pool, synthetic, !strcmp(mode, "l2hprof")) : 1;
    g_dry = true;
    Ctx c;
    setup(c, pool, 1);
    print_budget("gpu", pool);
    printf("budget timing phase: device %.1f MB + host %.1f MB = %.1f MB (cap %.0f MB); weight pool %d experts\n", timing_dev(pool) / 1048576.0, timing_host() / 1048576.0,
           (timing_dev(pool) + timing_host()) / 1048576.0, cap_bytes() / 1048576.0, TPOOL);
    printf("one expert: fc1 %.2f MB + fc2 %.2f MB = %.2f MB; act %.1f MB, ypart %.1f MB (capacity %d tiles), xfrag %.1f MB\n", EB1 / 1e6, EB2 / 1e6, (EB1 + EB2) / 1e6, ACT_B / 1e6, Y_B / 1e6, TILES, XF_B / 1e6);
    printf("budget ntg / ntgprof (no host ypart copy): memcmp phase device %.1f MB + host %.1f MB = %.1f MB; timing phase device %.1f MB + host %.1f MB = %.1f MB (cap %.0f MB)\n",
           g_dev / 1048576.0, (g_host - Y_B) / 1048576.0, (g_dev + g_host - Y_B) / 1048576.0, timing_dev(pool) / 1048576.0, (g_host - Y_B) / 1048576.0,
           (timing_dev(pool) + g_host - Y_B) / 1048576.0, cap_bytes() / 1048576.0);
    printf("usage: %s budget | plan [DIR] [L ...] [rows=A,B] [--synthetic] | gpu|ntg|ntgprof|eq|eqprof|l2h|l2hprof [DIR] [L ...] [pool=N] [rows=A,B] [--synthetic]    (DIR = the p1200 route files, required unless --synthetic; default layers 0 20 39, pool 4, rows 2048,846; a missing route file fails unless --synthetic;\n"
           "       plan = the host plan of each case, no CUDA call; rows must be <= this build's RM %d; the GPU modes refuse anything but a spark (aarch64) with sm_121)\n", argv[0], RM);
    return strcmp(mode, "budget") && argc > 1 ? 1 : 0;
}
