// Decode row budget, the one compile-time definition the engine shares. Default 8 rows per step (one mma n=8 tile).
// -DDEC_S2 raises it to 32 rows (four n=8 tiles): dense GEMVs loop over ceil(M/8) tiles on the weights already loaded,
// the router splits an expert with more than 8 tokens into slots of at most 8. Every row's result is bit-identical to
// the M <= 8 path. The default build keeps every size and layout it has today.
#pragma once
#ifndef G6_MX
#ifdef DEC_S2
#define G6_MX 32
#else
#define G6_MX 8
#endif
#endif
static_assert(G6_MX >= 8 && G6_MX <= 32 && G6_MX % 8 == 0, "G6_MX: 8, 16, 24 or 32");

#ifdef __CUDACC__
#define G6MX_HD __host__ __device__
#else
#define G6MX_HD
#endif

namespace g6mx
{
constexpr int MX = G6_MX;                // rows per decode step, at most
constexpr int NT = G6_MX / 8;            // n=8 tiles at the row limit
G6MX_HD constexpr int tiles(int M) { return (M + 7) / 8; }
// routed-slot capacity of the K1 scratch at the row limit: at 8 rows 72 (today); beyond, slots <= routes <= 8 * M
constexpr int SLOTS = MX == 8 ? 72 : 8 * MX + 32;

// run f(NTc<N>()) with N = the tile count nt (1 .. NT) as a compile-time constant; only the N <= NT instantiate, so the
// default build (NT = 1) has the single N = 1 body. nt < 1 runs N = 1. Host (kernel launches) and device (in-kernel) forms.
template <int N> struct NTc { static constexpr int value = N; };
#define G6_NT_DISPATCH_BODY                                                                  \
    if constexpr (NT >= 4) { if (nt >= 4) { f(NTc<4>()); return; } }                        \
    if constexpr (NT >= 3) { if (nt == 3) { f(NTc<3>()); return; } }                        \
    if constexpr (NT >= 2) { if (nt == 2) { f(NTc<2>()); return; } }                        \
    f(NTc<1>());
template <class F> inline void nt_dispatch(int nt, F&& f) { G6_NT_DISPATCH_BODY }
#ifdef __CUDACC__
template <class F> __device__ __forceinline__ void nt_dispatch_dev(int nt, F&& f) { G6_NT_DISPATCH_BODY }
#endif
}
