// Separate pool-ID fabric (default on for TP2 Flash prefill). No aliases with prefill partial collectives.
#pragma once
#include "../attn/flash_pool_owner.h"
#include "../comms/fabric.h"

namespace g6rt {
// G6_PF_POOL_OWNER: unset/empty = auto (on for TP2 Flash prefill, off elsewhere), 0 = off, 1 = on (dies if unsupported).
inline int flash_pool_owner_mode() {
    static const int mode = [] {
        const char* s = getenv("G6_PF_POOL_OWNER");
        if (!s || !*s) return 2;
        if (!strcmp(s, "0")) return 0;
        if (strcmp(s, "1")) die("G6_PF_POOL_OWNER must be 0 or 1");
        return 1;
    }();
    return mode;
}
struct FlashPoolTransport {
    g6::Fabric* fabric = nullptr;
    g6::FabricDev f{};
    flash_pool_owner::Flow flow;
    u64* step = nullptr;
    unsigned* ticket = nullptr;
    int* received = nullptr;
    int max_rows = 0;
};
__global__ void k_pool_step(u64* p, u64 step) { *p = step; }
__global__ void k_pool_put(g6::FabricDev f, const u64* step, unsigned* ticket,
                          const int* indices, const int* positions, int rows) {
    using namespace flash_pool_owner;
    const int lo = begin(rows, f.rank), count = end(rows, f.rank) - lo;
    int* out = reinterpret_cast<int*>(glue::send_slot(f, *step, 0));
    const int words = padded_rows(rows) * POOLS;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += blockDim.x * gridDim.x) {
        const int row = i / POOLS, j = i % POOLS;
        out[i] = row < count ? pack(indices + size_t(lo + row) * TOKENS, positions[lo + row], j) : -1;
    }
    glue::publish(f, 0, g6::coll_seq(*step, 0, 1), ticket);
}
__global__ void k_pool_unpack(const int* received, int* indices, const int* positions, int rows, int rank,
                             const unsigned* error) {
    using namespace flash_pool_owner;
    if (*reinterpret_cast<const volatile unsigned*>(error)) return;
    const int lo = begin(rows, 1 - rank), count = end(rows, 1 - rank) - lo;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count * TOKENS; i += blockDim.x * gridDim.x) {
        const int row = i / TOKENS, j = i % TOKENS;
        indices[size_t(lo + row) * TOKENS + j] = unpack(received + size_t(row) * POOLS, positions[lo + row], j);
    }
}
inline void init_flash_pool_transport(FlashPoolTransport& p, g6::Fabric& fabric, int max_rows) {
    p.fabric = &fabric; p.f = fabric.dev(); p.max_rows = max_rows;
    if (p.f.nr != 2 || p.f.ncoll != 1 || p.f.stride < flash_pool_owner::payload(max_rows) + 8)
        die("Flash pool owner: invalid dedicated fabric geometry");
    p.step = dmalloc<u64>(1); p.ticket = dmalloc<unsigned>(1);
    p.received = dmalloc<int>(flash_pool_owner::payload(max_rows) / sizeof(int));
    G6RT_CK(cudaMemset(p.ticket, 0, sizeof(unsigned)));
}
inline int flash_pool_publish(void* opaque, const int* indices, const int* positions, int rows, void* stream) {
    auto& p = *static_cast<FlashPoolTransport*>(opaque);
    if (!p.fabric || rows < 1 || rows > p.max_rows) return int(cudaErrorInvalidValue);
    if (!p.flow.publish(rows)) return int(cudaErrorInvalidValue);
    // Parity s can be reused at s+2 only because each DSA receive precedes its
    // main c0 put. A later main c0 consume requires the peer's c0 put, ordered
    // after its receive copied out parity s; that also proves our send DMA ended.
    // Do not publish twice per DSA call or move receive after the main c0 put.
    p.fabric->enqueue_step(p.flow.current, {uint32_t(flash_pool_owner::payload(rows))});
    const auto st = static_cast<cudaStream_t>(stream);
    k_pool_step<<<1, 1, 0, st>>>(p.step, p.flow.current);
    k_pool_put<<<96, 128, 0, st>>>(p.f, p.step, p.ticket, indices, positions, rows);
    return int(cudaGetLastError());
}
inline int flash_pool_receive(void* opaque, int* indices, const int* positions, int rows, void* stream) {
    auto& p = *static_cast<FlashPoolTransport*>(opaque);
    if (!p.flow.can_receive(rows)) return int(cudaErrorInvalidValue);
    const auto st = static_cast<cudaStream_t>(stream);
    const size_t bytes = flash_pool_owner::payload(rows);
    glue::k_wait_peers<<<1, 32, 0, st>>>(p.f, p.step, 0, bytes, glue::ERR_AR);
    const char* src = p.f.recv + (((p.flow.current & 1) * p.f.ncoll) * p.f.nr + (1 - p.f.rank)) * p.f.stride;
    const auto e = cudaMemcpyAsync(p.received, src, bytes, cudaMemcpyDefault, st);
    if (e != cudaSuccess) return int(e);
    k_pool_unpack<<<96, 256, 0, st>>>(p.received, indices, positions, rows, p.f.rank, p.f.error);
    const auto result = cudaGetLastError();
    if (result == cudaSuccess && !p.flow.received(rows)) return int(cudaErrorInvalidValue);
    return int(result);
}
// Call only after compute synchronization and checked proxy drain/stop.
inline void release_flash_pool_transport(FlashPoolTransport& p) {
    if (p.flow.awaiting) die("Flash pool owner: unmatched publish at teardown");
    if (p.received) G6RT_CK(cudaFree(p.received));
    if (p.ticket) G6RT_CK(cudaFree(p.ticket));
    if (p.step) G6RT_CK(cudaFree(p.step));
    p = FlashPoolTransport{};
}
} // namespace g6rt
