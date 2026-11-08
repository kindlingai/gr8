#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Our harness-owned storage only. Red zones detect writes that persist until a
// check; they do not prove absence of invalid reads, races, or write-and-restore.
// Candidate-private allocations are not registered and are NOT checked.
namespace sweep_memory {
constexpr size_t redzone_bytes = 256;  // preserve cudaMalloc's >=256-byte alignment
constexpr unsigned char leading_pattern = 0xD3, trailing_pattern = 0xB7;
inline void checked(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s in memory guard\n", cudaGetErrorString(e));
        std::exit(e == cudaErrorMemoryAllocation ? 77 : 2);
    }
}
inline size_t allocation_bytes(size_t logical) {
    if (logical > std::numeric_limits<size_t>::max() - 2 * redzone_bytes)
        throw std::overflow_error("guarded allocation size overflow");
    return logical + 2 * redzone_bytes;
}
constexpr size_t hash_chunk_bytes = 64u << 10;
struct Digest { uint64_t first, second; };
inline size_t digest_count(size_t bytes) { return bytes / hash_chunk_bytes + (bytes % hash_chunk_bytes != 0); }
inline size_t digest_bytes(size_t bytes) { return digest_count(bytes) * sizeof(Digest); }
// Two order-sensitive 64-bit accumulators per 64 KiB chunk, covering EVERY
// logical input byte. This is a non-cryptographic mutation screen, not an exact
// comparison or a proof against adversarial hash collisions.
static __global__ void hash_input(const unsigned char* input, size_t bytes, Digest* output) {
    __shared__ uint64_t first[256], second[256];
    const unsigned t=threadIdx.x;
    const size_t begin=size_t(blockIdx.x)*hash_chunk_bytes;
    const size_t end=begin+hash_chunk_bytes<bytes ? begin+hash_chunk_bytes : bytes;
    uint64_t a=0xcbf29ce484222325ULL ^ t, b=0x84222325cbf29ce4ULL ^ (uint64_t(t)<<32);
    for (size_t i=begin+t;i<end;i+=blockDim.x) {
        const uint64_t value=input[i];
        a=(a ^ value)*0x100000001b3ULL;
        b=(b ^ (value+0x9e3779b97f4a7c15ULL))*0xbf58476d1ce4e5b9ULL;
        b=(b<<27)|(b>>37);
    }
    first[t]=a; second[t]=b;
    __syncthreads();
    for (unsigned stride=128;stride;stride>>=1) {
        if (t<stride) { first[t]^=first[t+stride]; second[t]+=second[t+stride]; }
        __syncthreads();
    }
    if (!t) output[blockIdx.x]={first[0],second[0]};
}
struct Allocation {
    unsigned char* logical;
    size_t bytes;
    std::string name;
    bool immutable = true;
    std::vector<Digest> expected;
};
struct Integrity {
    size_t guard_bytes = 0, immutable_chunks = 0, missing_snapshots = 0;
    size_t total() const { return guard_bytes + immutable_chunks + missing_snapshots; }
};
inline void* allocate(size_t bytes);
inline void deallocate(void* logical);
struct Registry {
    std::vector<Allocation> allocations;
    Digest* workspace = nullptr;
    size_t workspace_bytes = 0;
    bool snapshotted = false;
    Integrity last{};
    size_t device_bytes() const {
        size_t n = 0;
        for (const auto& a : allocations) n += allocation_bytes(a.bytes);
        return n;
    }
    size_t host_snapshot_bytes() const {
        size_t n = 0;
        for (const auto& a : allocations) if (a.immutable) n += digest_bytes(a.bytes);
        return n;
    }
    void describe(const void* p, const char* name, bool immutable = true) {
        if (!p) return;
        for (auto& a : allocations) if (a.logical == p) {
            a.name = name; a.immutable = immutable; return;
        }
        throw std::logic_error("unregistered fixture pointer");
    }
    // Called exactly once after the adapter labels all allocations, within its
    // allocation scope. The digest scratch is itself registered and guarded.
    void prepare() {
        if (workspace) throw std::logic_error("memory registry already prepared");
        for (const auto& a : allocations) if (a.immutable)
            workspace_bytes=std::max(workspace_bytes,digest_bytes(a.bytes));
        workspace=static_cast<Digest*>(allocate(workspace_bytes));
        describe(workspace,"integrity_hash_scratch",false);
    }
    void release_workspace() { deallocate(workspace); workspace=nullptr; }
    void invalidate_snapshot() { snapshotted = false; }
    std::vector<Digest> hashes(const Allocation& a, cudaStream_t st) {
        std::vector<Digest> result(digest_count(a.bytes));
        if (result.size()*sizeof(Digest)>workspace_bytes || !workspace)
            throw std::logic_error("integrity workspace too small or not prepared");
        if (!result.empty()) {
            hash_input<<<result.size(),256,0,st>>>(a.logical,a.bytes,workspace);
            checked(cudaGetLastError());
            checked(cudaMemcpyAsync(result.data(),workspace,result.size()*sizeof(Digest),cudaMemcpyDeviceToHost,st));
            checked(cudaStreamSynchronize(st));
        }
        return result;
    }
    void snapshot(cudaStream_t st) {
        checked(cudaStreamSynchronize(st));
        for (auto& a : allocations) if (a.immutable) a.expected=hashes(a,st);
        snapshotted = true;
    }
    size_t check(cudaStream_t st) {
        checked(cudaStreamSynchronize(st));
        last = {};
        if (!snapshotted) {
            ++last.missing_snapshots;
            fprintf(stderr, "INTEGRITY missing input snapshot (call snapshot_inputs after refresh)\n");
        }
        // Hash first, so even the scratch guards are checked after our helpers.
        if (snapshotted) for (const auto& a : allocations) if (a.immutable) {
            const auto actual=hashes(a,st);
            size_t bad=0;
            for (size_t i=0;i<actual.size();++i)
                bad+=actual[i].first!=a.expected[i].first || actual[i].second!=a.expected[i].second;
            last.immutable_chunks += bad;
            if (bad) fprintf(stderr, "INTEGRITY input name=%s changed_chunks=%zu logical_bytes=%zu\n",
                             a.name.c_str(), bad, a.bytes);
        }
        std::array<unsigned char, redzone_bytes> guard{};
        for (const auto& a : allocations) {
            for (int side = 0; side < 2; ++side) {
                const auto* p = side ? a.logical + a.bytes : a.logical - redzone_bytes;
                const auto pattern = side ? trailing_pattern : leading_pattern;
                checked(cudaMemcpy(guard.data(), p, guard.size(), cudaMemcpyDeviceToHost));
                size_t bad = 0;
                for (auto b : guard) bad += b != pattern;
                last.guard_bytes += bad;
                if (bad) fprintf(stderr, "INTEGRITY guard name=%s side=%s corrupt_bytes=%zu logical_bytes=%zu\n",
                                 a.name.c_str(), side ? "trailing" : "leading", bad, a.bytes);
            }
        }
        return last.total();
    }
};
// Allocation scope only, never the source of truth during checking/freeing.
// Registry addresses remain stable when a vector<Fixture> relocates its slots.
inline thread_local Registry* allocating_registry = nullptr;
struct AllocationScope {
    Registry* previous;
    explicit AllocationScope(Registry& r) : previous(allocating_registry) { allocating_registry = &r; }
    ~AllocationScope() { allocating_registry = previous; }
    AllocationScope(const AllocationScope&) = delete;
    AllocationScope& operator=(const AllocationScope&) = delete;
};
inline void* allocate(size_t bytes) {
    if (!allocating_registry) throw std::logic_error("fixture allocation without registry scope");
    unsigned char* raw = nullptr;
    checked(cudaMalloc(&raw, allocation_bytes(bytes)));
    auto* logical = raw + redzone_bytes;
    checked(cudaMemset(raw, leading_pattern, redzone_bytes));
    checked(cudaMemset(logical, 0, bytes));
    checked(cudaMemset(logical + bytes, trailing_pattern, redzone_bytes));
    // cudaMemset may return before completion. Fixtures may immediately move
    // to a nonblocking stream, which does not wait for the legacy stream.
    checked(cudaStreamSynchronize(nullptr));
    allocating_registry->allocations.push_back({logical, bytes, "unnamed", true, {}});
    allocating_registry->invalidate_snapshot();
    return logical;
}
inline void deallocate(void* logical) {
    if (logical) checked(cudaFree(static_cast<unsigned char*>(logical) - redzone_bytes));
}
} // namespace sweep_memory
