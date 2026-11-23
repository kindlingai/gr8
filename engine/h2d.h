// Host-to-device copies from pageable host memory, made safe on 64 KiB-page kernels: there a pageable copy
// whose source is not 64 KiB-aligned can hang (the process spins, the GPU sits idle). The bytes go through a pinned
// staging buffer instead, in chunks of at most 64 MiB. One buffer per thread, allocated on first use and kept.
// The copy has finished when the call returns (it waits for `st`), so the source may be reused at once, as with
// cudaMemcpy. Not for use inside a stream capture.
#pragma once
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>

namespace g6h2d
{
constexpr size_t kChunk = 64ull << 20;

inline cudaError_t stage(size_t want, char*& out)
{
    thread_local char* buf = nullptr;
    thread_local size_t cap = 0;
    if (want > cap)
    {
        if (buf) cudaFreeHost(buf);
        buf = nullptr;
        cap = 0;
        const size_t sz = want < 65536 ? 65536 : want;
        if (cudaError_t e = cudaHostAlloc(reinterpret_cast<void**>(&buf), sz, cudaHostAllocDefault)) return e;
        cap = sz;
    }
    out = buf;
    return cudaSuccess;
}

inline cudaError_t copy(void* dst, const void* src, size_t n, cudaStream_t st = nullptr)
{
    if (n == 0) return cudaSuccess;
    const size_t want = n < kChunk ? n : kChunk;
    char* buf = nullptr;
    if (cudaError_t e = stage(want, buf)) return e;
    for (size_t off = 0; off < n; off += kChunk)
    {
        const size_t k = n - off < kChunk ? n - off : kChunk;
        memcpy(buf, static_cast<const char*>(src) + off, k);
        if (cudaError_t e = cudaMemcpyAsync(static_cast<char*>(dst) + off, buf, k, cudaMemcpyHostToDevice, st)) return e;
        if (cudaError_t e = cudaStreamSynchronize(st)) return e;
    }
    return cudaSuccess;
}

// Broadcast the same host buffer [src, src + n) to `count` device destinations with one staging copy and one sync per chunk.
inline cudaError_t copy_broadcast(void* const* dsts, size_t count, const void* src, size_t n, cudaStream_t st = nullptr)
{
    if (count == 0 || n == 0) return cudaSuccess;
    const size_t want = n < kChunk ? n : kChunk;
    char* buf = nullptr;
    if (cudaError_t e = stage(want, buf)) return e;
    for (size_t off = 0; off < n; off += kChunk)
    {
        const size_t k = n - off < kChunk ? n - off : kChunk;
        memcpy(buf, static_cast<const char*>(src) + off, k);
        for (size_t i = 0; i < count; ++i)
            if (cudaError_t e = cudaMemcpyAsync(static_cast<char*>(dsts[i]) + off, buf, k, cudaMemcpyHostToDevice, st)) return e;
        if (cudaError_t e = cudaStreamSynchronize(st)) return e;
    }
    return cudaSuccess;
}

// Copy `count` distinct host buffers to `count` device destinations, packing into the pinned staging buffer so one sync
// covers all buffers that fit in kChunk.
struct BatchItem { void* dst; const void* src; size_t n; };
inline cudaError_t copy_many(const BatchItem* items, size_t count, cudaStream_t st = nullptr)
{
    if (count == 0) return cudaSuccess;
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) total += items[i].n;
    if (total == 0) return cudaSuccess;
    const size_t want = total < kChunk ? total : kChunk;
    char* buf = nullptr;
    if (cudaError_t e = stage(want, buf)) return e;
    size_t used = 0;
    for (size_t i = 0; i < count; ++i)
    {
        const BatchItem& it = items[i];
        if (it.n == 0) continue;
        if (it.n > kChunk)
        {
            if (used > 0)
            {
                if (cudaError_t e = cudaStreamSynchronize(st)) return e;
                used = 0;
            }
            if (cudaError_t e = copy(it.dst, it.src, it.n, st)) return e;
            continue;
        }
        if (used + it.n > want)
        {
            if (cudaError_t e = cudaStreamSynchronize(st)) return e;
            used = 0;
        }
        memcpy(buf + used, it.src, it.n);
        if (cudaError_t e = cudaMemcpyAsync(it.dst, buf + used, it.n, cudaMemcpyHostToDevice, st)) return e;
        used += it.n;
    }
    if (used > 0)
        if (cudaError_t e = cudaStreamSynchronize(st)) return e;
    return cudaSuccess;
}
}   // namespace g6h2d
