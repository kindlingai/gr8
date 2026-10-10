// DEC_SLOT: a decode partial producer (o_proj gv_gemv, K1 finalize, dense fc2)
// writes its BF16 partial straight into this rank's send slot of collective c and (DEC_SLOT >= 2) rings the doorbell
// from its own epilogue, so k_put_publish (copy + publish) is not launched. The slot address comes from the device step
// cell at run time, so one captured graph serves both slot parities. Only included under -DDEC_SLOT.
#pragma once
#include "common.cuh"
#include "../comms/xtrace.h"

namespace dslot
{

struct Slot
{
    char* send = nullptr;                 // FabricDev::send; nullptr = slot mode off (the producer writes its staging)
    unsigned long long* bell = nullptr;   // FabricDev::doorbell
    const u64* stepp = nullptr;           // device step cell
    size_t stride = 0;
    int ncoll = 0, c = 0;
    unsigned* ticket = nullptr;           // device, zero; one per collective
    unsigned long long* xt = nullptr;     // FabricDev::xt (G6_XTRACE records, xtrace.h); nullptr = off (default)
};

__device__ __forceinline__ char* ptr(const Slot& s)
{
    return s.send + ((size_t)(*s.stepp & 1) * s.ncoll + s.c) * s.stride;
}

// All threads of every CTA, after the CTA's last store into the slot (stored = this thread stored into it).
// Same protocol as glue::publish: system fence, CTA barrier, last CTA rings doorbell[c] = coll_seq (release, sys).
__device__ __forceinline__ void publish(const Slot& s, bool stored)
{
    if (stored) __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const unsigned n = gridDim.x * gridDim.y * gridDim.z;
        if (atomicAdd(s.ticket, 1u) == n - 1)
        {
            const u64 step = *s.stepp;
            // G6_XTRACE: same stamps as glue::publish ([PUB] before the fence, [DB] + [SEQ] after the doorbell store)
            u64* xr = s.xt ? reinterpret_cast<u64*>(s.xt) + ((size_t)(step % g6::XT_STEPS) * s.ncoll + s.c) * g6::XT_W : nullptr;
            if (xr) { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); xr[g6::XT_PUB] = t; }
            *s.ticket = 0;
            __threadfence_system();
            const u64 seq = step * (u64)s.ncoll + s.c + 1;          // g6::coll_seq
            asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(s.bell + s.c), "l"(seq) : "memory");
            if (xr) { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); xr[g6::XT_DB] = t; xr[g6::XT_SEQ] = seq; }
        }
    }
}

}   // namespace dslot
