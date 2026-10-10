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

inline cudaError_t copy(void* dst, const void* src, size_t n, cudaStream_t st = nullptr)
{
    thread_local char* buf = nullptr;
    thread_local size_t cap = 0;
    const size_t want = n < kChunk ? n : kChunk;
    if (want > cap)
    {
        if (buf) cudaFreeHost(buf);
        buf = nullptr;
        cap = 0;
        const size_t sz = want < 65536 ? 65536 : want;
        if (cudaError_t e = cudaHostAlloc(reinterpret_cast<void**>(&buf), sz, cudaHostAllocDefault)) return e;
        cap = sz;
    }
    for (size_t off = 0; off < n; off += kChunk)
    {
        const size_t k = n - off < kChunk ? n - off : kChunk;
        memcpy(buf, static_cast<const char*>(src) + off, k);
        if (cudaError_t e = cudaMemcpyAsync(static_cast<char*>(dst) + off, buf, k, cudaMemcpyHostToDevice, st)) return e;
        if (cudaError_t e = cudaStreamSynchronize(st)) return e;
    }
    return cudaSuccess;
}
}   // namespace g6h2d
