// CP fabric (fabc) device helpers: publish / wait on a variable-layout collective (comms/cplayout.h VSlotDev).
// Same protocol as glue::publish / glue::wait_peers (seq tail after the payload, doorbell rung by the last CTA),
// with the slot addresses resolved on the host for both parities. Included only by CP builds (G6_CP).
#pragma once
#include "glue.cuh"
#include "../comms/cplayout.h"

namespace cpglue {

__device__ __forceinline__ char* vsend(const g6::VSlotDev& v, u64 step) { return v.send[step & 1]; }
__device__ __forceinline__ const char* vrecv(const g6::VSlotDev& v, u64 step, int j) { return v.recv[step & 1][j]; }

// all threads of every CTA, after the CTA's last store into vsend(v, step); ticket: device word, 0
__device__ __forceinline__ void vpublish(const g6::VSlotDev& v, u64 step, unsigned* ticket)
{
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const unsigned n = gridDim.x * gridDim.y * gridDim.z;
        if (atomicAdd(ticket, 1u) == n - 1)
        {
            *ticket = 0;
            __threadfence_system();
            glue::st_release_sys(reinterpret_cast<u64*>(v.doorbell), g6::coll_seq(step, v.c, v.ncoll));
        }
    }
}

// threads 0..nsrc-1 spin on source j's tail (only = -1: every source; else source index `only`)
__device__ __forceinline__ void vwait(const g6::VSlotDev& v, u64 step, size_t payload, int only, unsigned code)
{
    const int j = threadIdx.x;
    const u64 seq = g6::coll_seq(step, v.c, v.ncoll);
    if (j < v.nsrc && (only < 0 || j == only))
    {
        const u64* tail = reinterpret_cast<const u64*>(vrecv(v, step, j) + payload);
        const long long t0 = clock64();
        unsigned it = 0;
        while (glue::ld_acquire_sys(tail) != seq)
        {
            if ((++it & 1023u) == 0 &&
                (*reinterpret_cast<volatile unsigned*>(v.error) != 0 || clock64() - t0 > G6_SPIN_TIMEOUT_CYC))
            {
                atomicCAS(v.error, 0u, code | ((unsigned)(v.c & 0xffff) << 8) | (unsigned)v.src[j]);
                break;
            }
        }
    }
    __syncthreads();
}

// grid 1, block 32: wait for collective v of this step (all sources, or source index `only`)
__global__ void __launch_bounds__(32) k_vwait(g6::VSlotDev v, const u64* stepp, size_t payload, int only, unsigned code)
{
    vwait(v, *stepp, payload, only, code);
}

// grid 1, block 32: ring the doorbell of a send slot that a producer already filled (no ticket)
__global__ void __launch_bounds__(32) k_vbell(g6::VSlotDev v, const u64* stepp)
{
    if (threadIdx.x == 0)
    {
        __threadfence_system();
        glue::st_release_sys(reinterpret_cast<u64*>(v.doorbell), g6::coll_seq(*stepp, v.c, v.ncoll));
    }
}

// grid G, block 256: device buffers a (na bytes) then b (nb bytes) -> my send slot of v (packed a | b), then the doorbell
// (last CTA, ticket). nb > 0: na, nb multiples of 16; nb == 0: na a multiple of 4 (NR=6 partials of 10/11 heads).
__global__ void __launch_bounds__(256) k_vput(const void* a, size_t na, const void* b, size_t nb, g6::VSlotDev v,
                                              const u64* stepp, unsigned* ticket)
{
    const u64 step = *stepp;
    uint4* dst = reinterpret_cast<uint4*>(vsend(v, step));
    const size_t n1 = na / 16, n = n1 + nb / 16;
    for (size_t i = blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256)
        dst[i] = i < n1 ? reinterpret_cast<const uint4*>(a)[i] : reinterpret_cast<const uint4*>(b)[i - n1];
    if (!nb && blockIdx.x == 0 && threadIdx.x < (na % 16) / 4)
        reinterpret_cast<unsigned*>(dst + n1)[threadIdx.x] = reinterpret_cast<const unsigned*>(reinterpret_cast<const uint4*>(a) + n1)[threadIdx.x];
    vpublish(v, step, ticket);
}
// grid G, block 256: source j's recv slot of v (after k_vwait) -> device buffer dst (n bytes, multiple of 4; dst 16-B aligned
// up to n / 16 uint4, the rest by words, so nothing past dst + n is written)
__global__ void __launch_bounds__(256) k_vget(g6::VSlotDev v, int j, const u64* stepp, void* dst, size_t n)
{
    const uint4* src = reinterpret_cast<const uint4*>(vrecv(v, *stepp, j));
    for (size_t i = blockIdx.x * 256 + threadIdx.x; i < n / 16; i += (size_t)gridDim.x * 256)
        reinterpret_cast<uint4*>(dst)[i] = src[i];
    if (blockIdx.x == 0 && threadIdx.x < (n % 16) / 4)
        reinterpret_cast<unsigned*>(reinterpret_cast<uint4*>(dst) + n / 16)[threadIdx.x] = reinterpret_cast<const unsigned*>(src + n / 16)[threadIdx.x];
}
#ifdef CP_F1A
// F1a: the phase-2 partial waits + gets of launch_attn_cp (one k_vwait + one k_vget per partial) as ONE launch.
// grid (G, n), block 256: row y waits for partial collective v[y] (its one source; every CTA of the row spins on the
// same tail, thread 0, as vwait), then copies it to dst[y] (n bytes, as k_vget). Ordering is wait_peers' (k_head_merge*):
// acquire (sys) by thread 0, __syncthreads, then the other threads read the slot, with __ldcv like k_head_merge.
struct VParts
{
    g6::VSlotDev v[7];
    void* dst[7];
};
__global__ void __launch_bounds__(256) k_vwaitget(VParts p, const u64* stepp, size_t payload, size_t n, unsigned code)
{
    const u64 step = *stepp;
    const g6::VSlotDev& v = p.v[blockIdx.y];
    vwait(v, step, payload, 0, code);
    const uint4* src = reinterpret_cast<const uint4*>(vrecv(v, step, 0));
    void* dst = p.dst[blockIdx.y];
    for (size_t i = blockIdx.x * 256 + threadIdx.x; i < n / 16; i += (size_t)gridDim.x * 256)
        reinterpret_cast<uint4*>(dst)[i] = __ldcv(src + i);
    if (blockIdx.x == 0 && threadIdx.x < (n % 16) / 4)
        reinterpret_cast<unsigned*>(reinterpret_cast<uint4*>(dst) + n / 16)[threadIdx.x] =
            __ldcv(reinterpret_cast<const unsigned*>(src + n / 16) + threadIdx.x);
}
#endif
// host: index j of sender s in v.src (-1: s does not send v to me)
inline int vsrc_index(const g6::VSlotDev& v, int s)
{
    for (int j = 0; j < v.nsrc; ++j) if (v.src[j] == s) return j;
    return -1;
}

constexpr unsigned ERR_CP = 0x63000000u;   // error word code: CP wait timeout (low bits (c & 0xffff) << 8 | sender)

}  // namespace cpglue
