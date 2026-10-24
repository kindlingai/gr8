// G6_PF_ROUTER_SPLIT transport (opt-in, default off): the prefill MoE router is row-split across the ranks and the route tables
// (tok_e, tok_w) are exchanged over a dedicated ncoll = 1 fabric per prefill lane (the flash_pool_owner.cuh pattern; not a
// collective on the prefill fabric: the proxy serves a step's collectives in order and c1 is published after the MoE block).
// One step per MoE block: publish (own rows) -> hx / shared work runs -> wait + copy peers' rows -> launch_experts with
// Expert::pre_routed. Layout and row split: flash_route_split.h. Parity-2 slots; the next use of a parity is ordered after
// every peer's c1 put of the following layer, which is after that peer's receive of this one (same argument as the pool
// owner, once per peer). Nothing here depends on the rank count or on mesh vs ring: the fabric moves a uniform slot.
#pragma once
#include "flash_route_split.h"
#include "flash_pool_owner.cuh"

namespace g6rt {
// G6_PF_ROUTER_SPLIT: unset = 0, "1" = on (g6m4 dies if the build or topology cannot run it), else die at boot.
inline int flash_route_split_mode()
{
    static const int mode = [] {
        std::string e;
        const int v = env_choice("G6_PF_ROUTER_SPLIT", 1, 0, &e);
        if (v < 0) die(e);
        return v;
    }();
    return mode;
}
struct FlashRouteSplit {
    g6::Fabric* fabric = nullptr;
    g6::FabricDev f{};
    flash_pool_owner::Flow flow;      // host-side send/receive pairing (one publish, then its receive)
    u64* step = nullptr;
    unsigned* ticket = nullptr;
    int max_rows = 0;
};
constexpr int RS_PUT_CTA = 32, RS_UNPACK_CTA = 32;
// grid RS_PUT_CTA, block 128: own rows -> the send slot (64-byte records), then the doorbell
__global__ void __launch_bounds__(128) k_rs_put(g6::FabricDev f, const u64* step, unsigned* ticket, const int* tok_e,
                                                const float* tok_w, int M)
{
    using namespace flash_route_split;
    const int lo = r0(M, f.rank, f.nr), n = rows(M, f.rank, f.nr);
    uint4* out = reinterpret_cast<uint4*>(glue::send_slot(f, *step, 0));
    const uint4* e = reinterpret_cast<const uint4*>(tok_e + size_t(lo) * SLOTS);
    const uint4* w = reinterpret_cast<const uint4*>(tok_w + size_t(lo) * SLOTS);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * 4; i += gridDim.x * blockDim.x)
        out[i] = (i & 3) < 2 ? e[(i >> 2) * 2 + (i & 3)] : w[(i >> 2) * 2 + (i & 3) - 2];
    glue::publish(f, 0, g6::coll_seq(*step, 0, f.ncoll), ticket);
}
// grid (RS_UNPACK_CTA, nr), block 256, after k_wait_peers: peer p's records -> tok_e / tok_w rows r0(M, p)..
// One instance for every rank count: blockIdx.y is the peer, the row split is flash_route_split.h's.
__global__ void __launch_bounds__(256) k_rs_unpack(g6::FabricDev f, const u64* step, int M, int* tok_e, float* tok_w)
{
    using namespace flash_route_split;
    const int p = blockIdx.y;
    if (p >= f.nr || p == f.rank || *reinterpret_cast<const volatile unsigned*>(f.error)) return;
    const uint4* src = reinterpret_cast<const uint4*>(glue::peer_slot(f, *step, 0, p));
    const int lo = r0(M, p, f.nr), n = rows(M, p, f.nr);
    uint4* e = reinterpret_cast<uint4*>(tok_e + size_t(lo) * SLOTS);
    uint4* w = reinterpret_cast<uint4*>(tok_w + size_t(lo) * SLOTS);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * 4; i += gridDim.x * blockDim.x) {
        const uint4 v = __ldcv(src + i);
        if ((i & 3) < 2) e[(i >> 2) * 2 + (i & 3)] = v; else w[(i >> 2) * 2 + (i & 3) - 2] = v;
    }
}
inline void init_flash_route_split(FlashRouteSplit& p, g6::Fabric& fabric, int max_rows)
{
    p.fabric = &fabric; p.f = fabric.dev(); p.max_rows = max_rows;
    if (p.f.ncoll != 1 || p.f.nr < 2 || p.f.nr > MAX_NR ||
        p.f.stride < flash_route_split::payload(max_rows, p.f.nr) + 8)
        die("G6_PF_ROUTER_SPLIT: invalid dedicated fabric geometry");
    p.step = dmalloc<u64>(1); p.ticket = dmalloc<unsigned>(1);
    G6RT_CK(cudaMemset(p.ticket, 0, sizeof(unsigned)));
}
// after the rows [r0, r1) of tok_e / tok_w are written on this stream
inline void flash_route_publish(FlashRouteSplit& p, const int* tok_e, const float* tok_w, int M, cudaStream_t st)
{
    if (!p.fabric || M < 1 || M > p.max_rows || !flash_route_split::active(M, p.f.nr) || !p.flow.publish(M))
        die("G6_PF_ROUTER_SPLIT: publish out of protocol");
    p.fabric->enqueue_step(p.flow.current, {uint32_t(flash_route_split::payload(M, p.f.nr))});
    k_pool_step<<<1, 1, 0, st>>>(p.step, p.flow.current);
    k_rs_put<<<RS_PUT_CTA, 128, 0, st>>>(p.f, p.step, p.ticket, tok_e, tok_w, M);
    G6RT_CK(cudaGetLastError());
}
// before launch_experts: all rows of tok_e / tok_w valid on this stream
inline void flash_route_receive(FlashRouteSplit& p, int* tok_e, float* tok_w, int M, cudaStream_t st)
{
    if (!p.flow.can_receive(M)) die("G6_PF_ROUTER_SPLIT: receive out of protocol");
    glue::k_wait_peers<<<1, 32, 0, st>>>(p.f, p.step, 0, flash_route_split::payload(M, p.f.nr), glue::ERR_AR);
    k_rs_unpack<<<dim3(RS_UNPACK_CTA, p.f.nr), 256, 0, st>>>(p.f, p.step, M, tok_e, tok_w);
    G6RT_CK(cudaGetLastError());
    if (!p.flow.received(M)) die("G6_PF_ROUTER_SPLIT: receive out of protocol");
}
// Call only after compute synchronization and checked proxy drain/stop.
inline void release_flash_route_split(FlashRouteSplit& p)
{
    if (p.flow.awaiting) die("G6_PF_ROUTER_SPLIT: unmatched publish at teardown");
    if (p.ticket) G6RT_CK(cudaFree(p.ticket));
    if (p.step) G6RT_CK(cudaFree(p.step));
    p = FlashRouteSplit{};
}
} // namespace g6rt
