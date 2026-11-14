// ink2 fabric: one collective of a step run by the device, with no host round trip.
//
// The stock exchange (ink2_m4.cu Wire::exchange_dev) is one fabric step per collective: the host copies this rank's
// share to the send slot, rings the doorbell, spins on the peers' tails and copies their shares back, with a stream
// synchronisation each time. Here a whole engine step is one fabric step (one doorbell word per collective), and
// each collective is three launches on the engine's stream:
//   1. k_ink2_ar_copy       this rank's share -> its send slot
//   2. k_ink2_ar_ring_wait  ring doorbell[c], then one thread per peer spins on that peer's tail word
//   3. k_ink2_ar_copy       each peer's receive slot -> its place in parts
// The sum that follows reads parts as it does on the stock path, so its bits are the stock path's.
// A peer that does not arrive in INK2_SPIN_TIMEOUT_CYC clock cycles sets the fabric's error word
// (0x61000000 | c << 8 | peer); the host reads it at the end of the step.
#pragma once
#include "../comms/fabric.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

#ifndef INK2_SPIN_TIMEOUT_CYC
#define INK2_SPIN_TIMEOUT_CYC 120000000000ll
#endif

namespace ink2
{
constexpr unsigned INK2_ERR_AR = 0x61000000u;
constexpr int INK2_AR_MAX_RANKS = 8;

__device__ __forceinline__ unsigned long long ar_ld_acquire_sys(const unsigned long long* p)
{
    unsigned long long v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ void ar_st_release_sys(unsigned long long* p, unsigned long long v)
{
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

// n16 cells of 16 bytes
__global__ void k_ink2_ar_copy(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n16)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n16) dst[i] = src[i];
}

struct ArWait
{
    g6::FabricDev f;
    size_t bytes;
    unsigned long long seq;
    unsigned code;   // INK2_ERR_AR | c << 8
    int par, c;
};
__global__ void k_ink2_ar_ring_wait(ArWait a)
{
    const int p = threadIdx.x;
    if (p == 0)
    {
        __threadfence_system();
        ar_st_release_sys(a.f.doorbell + a.c, a.seq);
    }
    if (p < a.f.nr && p != a.f.rank)
    {
        const unsigned long long* tail = reinterpret_cast<const unsigned long long*>(a.f.recv + g6::fab_recv_off(a.f, a.par, a.c, p) + a.bytes);
        const long long t0 = clock64();
        unsigned it = 0;
        while (ar_ld_acquire_sys(tail) != a.seq)
        {
            if ((++it & 1023u) == 0 && (*reinterpret_cast<volatile unsigned*>(a.f.error) != 0 || clock64() - t0 > INK2_SPIN_TIMEOUT_CYC))
            {
                atomicCAS(a.f.error, 0u, a.code | static_cast<unsigned>(p));
                break;
            }
        }
    }
}

// Collective c of fabric step `step`: parts [nr][bytes] on the device with this rank's share at index f.rank; when
// the launches have run, every rank's share is in place. bytes is a multiple of 16. No synchronisation.
inline void ar_exchange(const g6::FabricDev& f, unsigned long long step, int c, char* parts, size_t bytes, cudaStream_t st)
{
    if (bytes == 0 || bytes % 16 || f.nr > INK2_AR_MAX_RANKS || c < 0 || c >= f.ncoll) throw std::runtime_error("ink2::ar_exchange: bytes, ranks or collective");
    const int par = static_cast<int>(step & 1);
    const size_t n16 = bytes / 16;
    const unsigned grid = static_cast<unsigned>((n16 + 255) / 256);
    k_ink2_ar_copy<<<grid, 256, 0, st>>>(reinterpret_cast<const uint4*>(parts + static_cast<size_t>(f.rank) * bytes),
                                         reinterpret_cast<uint4*>(f.send + g6::fab_send_off(f, par, c)), n16);
    ArWait a{f, bytes, g6::coll_seq(step, c, f.ncoll), INK2_ERR_AR | (static_cast<unsigned>(c) << 8), par, c};
    k_ink2_ar_ring_wait<<<1, 32, 0, st>>>(a);
    for (int p = 0; p < f.nr; ++p)
        if (p != f.rank)
            k_ink2_ar_copy<<<grid, 256, 0, st>>>(reinterpret_cast<const uint4*>(f.recv + g6::fab_recv_off(f, par, c, p)),
                                                 reinterpret_cast<uint4*>(parts + static_cast<size_t>(p) * bytes), n16);
}
}   // namespace ink2
