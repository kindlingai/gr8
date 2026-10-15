// Weight self-check: is each device image still the bytes the loader read?
//
// The Rust loader fills an image in fixed chunks (G6_LOAD_CHUNK_MB, 32 MB), each through a pinned staging buffer. While a
// chunk is in staging, the copy callback (packfind.h RsSink) sums it on the CPU. After the load the GPU sums the same
// chunks from device memory, and a chunk that differs ends the boot (exit 3, "g6: weight check ..."): the copy to the
// GPU went wrong. Binding then rewrites some tensors in place (bind::widen_fc1, the i4pair repack), so once the engine
// is ready to serve the GPU sums every image again (seal). A later check (DIR/check.<n>, the supervisor's /health?deep)
// compares the GPU sums with the sealed ones, on its own low-priority stream while the engine serves: weights never
// change after the seal, so a difference is corruption.
//
// The sum of a chunk is sum over its 64-bit words w_i, i counted from the start of the image, of mix(w_i ^ i * phi):
// order-free, so the GPU adds it up in parallel and gets the CPU's value. A trailing partial word is zero-padded.
//
// G6_WEIGHT_CHECK=0 turns all of it off.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include "../loader/g6pack.h"

namespace wcheck {

__host__ __device__ inline uint64_t mix(uint64_t x)     // splitmix64's finalizer
{
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27; x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
__host__ __device__ inline uint64_t term(uint64_t w, uint64_t i) { return mix(w ^ (i * 0x9e3779b97f4a7c15ull)); }

// The sum of image bytes off .. off + n held at p.
inline uint64_t host_sum(const uint8_t* p, uint64_t off, uint64_t n)
{
    uint64_t s = 0, i0 = off / 8;
    const uint64_t nw = n / 8;
    for (uint64_t k = 0; k < nw; ++k)
    {
        uint64_t w;
        memcpy(&w, p + 8 * k, 8);
        s += term(w, i0 + k);
    }
    if (n % 8)
    {
        uint64_t w = 0;
        memcpy(&w, p + 8 * nw, n % 8);
        s += term(w, i0 + nw);
    }
    return s;
}

// out[c] += the sum of chunk c (bytes c * chunk .. min(+chunk, bytes) of the image at base). out starts at zero.
__global__ void k_sums(const uint8_t* __restrict__ base, uint64_t bytes, uint64_t chunk, unsigned long long* out)
{
    const uint64_t c = blockIdx.y, off = c * chunk, n = (bytes - off < chunk) ? bytes - off : chunk;
    const uint64_t* w = reinterpret_cast<const uint64_t*>(base + off);
    const uint64_t nw = n / 8, i0 = off / 8;
    uint64_t s = 0;
    for (uint64_t k = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; k < nw; k += uint64_t(gridDim.x) * blockDim.x)
        s += term(w[k], i0 + k);
    if (n % 8 && blockIdx.x == 0 && threadIdx.x == 0)
    {
        uint64_t t = 0;
        for (uint64_t b = 0; b < n % 8; ++b) t |= uint64_t(base[off + 8 * nw + b]) << (8 * b);
        s += term(t, i0 + nw);
    }
    for (int d = 16; d; d >>= 1) s += __shfl_down_sync(0xffffffffu, s, d);
    __shared__ uint64_t part[32];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        uint64_t t = 0;
        for (unsigned j = 0; j < (blockDim.x + 31) / 32; ++j) t += part[j];
        atomicAdd(out + c, (unsigned long long)t);
    }
}

// The chunk size the Rust loader uses (lib.rs env_u64: a positive integer, else the default).
inline uint64_t loader_chunk()
{
    const char* e = getenv("G6_LOAD_CHUNK_MB");
    const long long v = e ? atoll(e) : 0;
    return uint64_t(v > 0 ? v : 32) << 20;
}

inline bool enabled()
{
    const char* e = getenv("G6_WEIGHT_CHECK");
    return !(e && std::string(e) == "0");
}

// The GPU sums of every chunk of an image, on stream st.
inline std::vector<uint64_t> device_sums(const uint8_t* base, uint64_t bytes, uint64_t chunk, cudaStream_t st)
{
    const uint64_t nc = (bytes + chunk - 1) / chunk;
    std::vector<uint64_t> h(nc, 0);
    if (!nc) return h;
    unsigned long long* d = nullptr;
    if (cudaMallocAsync(&d, nc * 8, st) != cudaSuccess || cudaMemsetAsync(d, 0, nc * 8, st) != cudaSuccess)
    {
        fprintf(stderr, "g6: weight check: no device memory for %llu sums\n", (unsigned long long)nc);
        exit(3);
    }
    k_sums<<<dim3(16, unsigned(nc)), 256, 0, st>>>(base, bytes, chunk, d);     // record() keeps nc <= 65535
    cudaError_t e = cudaMemcpyAsync(h.data(), d, nc * 8, cudaMemcpyDeviceToHost, st);
    if (e == cudaSuccess) e = cudaStreamSynchronize(st);
    cudaFreeAsync(d, st);
    if (e != cudaSuccess)
    {
        fprintf(stderr, "g6: weight check: CUDA %s\n", cudaGetErrorString(e));
        exit(3);
    }
    return h;
}

// Every image the Rust loader filled in this process.
struct Image
{
    std::string what;
    const g6::Pack* pack = nullptr;
    const uint8_t* base = nullptr;
    uint64_t bytes = 0, chunk = 0;
    std::vector<uint64_t> loaded, sealed;   // CPU sums of the bytes read; GPU sums once the engine was ready
};
inline std::mutex& mu() { static std::mutex m; return m; }
inline std::vector<Image>& images() { static std::vector<Image> v; return v; }

// The loader's side, per image: the CPU sum of each chunk as it passes through staging.
struct Loaded
{
    uint64_t bytes = 0, chunk = 0;
    std::vector<uint64_t> sums;
    std::vector<uint8_t> seen;
    void begin(uint64_t b)
    {
        bytes = b; chunk = loader_chunk();
        const uint64_t nc = (b + chunk - 1) / chunk;
        sums.assign(nc, 0); seen.assign(nc, 0);
    }
    void add(uint64_t off, const void* src, uint64_t n)    // from loader workers: each chunk once
    {
        if (off % chunk || off / chunk >= sums.size()) return;
        sums[off / chunk] = host_sum((const uint8_t*)src, off, n);
        seen[off / chunk] = 1;
    }
};

inline std::string tensor_at(const g6::Pack* p, uint64_t off)
{
    if (!p) return "";
    for (const g6::Entry& e : p->entries())
        if (off >= e.offset && off < e.offset + e.nbytes) return e.name;
    return "(padding)";
}

// After a load: the GPU's sums must equal the CPU's. A difference ends the boot, exit 3.
inline void record(const char* what, const g6::Pack& pack, const uint8_t* base, const Loaded& l, int rank)
{
    if (!enabled()) return;
    if (l.sums.size() > 65535)     // gridDim.y of k_sums
    {
        fprintf(stderr, "g6: weight check: %zu chunks of %llu bytes: raise G6_LOAD_CHUNK_MB or set G6_WEIGHT_CHECK=0\n",
                l.sums.size(), (unsigned long long)l.chunk);
        exit(3);
    }
    for (size_t c = 0; c < l.seen.size(); ++c)
        if (!l.seen[c])
        {
            fprintf(stderr, "g6: weight check: rank %d: %s chunk %zu never passed through the loader's copy\n", rank, what, c);
            exit(3);
        }
    const auto t0 = std::chrono::steady_clock::now();
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    // G6_WEIGHT_CHECK_FLIP=RANK:OFFSET (a test of this check): flip one byte of rank RANK's first image before it runs
    if (const char* f = getenv("G6_WEIGHT_CHECK_FLIP"))
    {
        int r = -1;
        unsigned long long off = 0;
        if (sscanf(f, "%d:%llu", &r, &off) == 2 && r == rank && off < l.bytes && images().empty())
        {
            uint8_t b = 0;
            cudaMemcpy(&b, base + off, 1, cudaMemcpyDeviceToHost);
            b ^= 0x40;
            cudaMemcpy(const_cast<uint8_t*>(base) + off, &b, 1, cudaMemcpyHostToDevice);
            fprintf(stderr, "# rank %d: G6_WEIGHT_CHECK_FLIP: flipped the byte at offset %llu of the %s image\n", rank, off, what);
        }
    }
    const std::vector<uint64_t> dev = device_sums(base, l.bytes, l.chunk, st);
    cudaStreamDestroy(st);
    size_t bad = 0, first = 0;
    for (size_t c = 0; c < dev.size(); ++c)
        if (dev[c] != l.sums[c] && !bad++) first = c;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (bad)
    {
        fprintf(stderr, "g6: weight check: rank %d: %zu of %zu chunks of the %s image differ from the bytes read "
                "(the first is bytes %llu..%llu, which starts in %s)\n", rank, bad, dev.size(), what,
                (unsigned long long)(first * l.chunk), (unsigned long long)std::min((first + 1) * l.chunk, l.bytes),
                tensor_at(&pack, first * l.chunk).c_str());
        exit(3);
    }
    fprintf(stderr, "# rank %d: weight check: %s image, %zu chunks equal to the bytes read (%.0f ms)\n", rank, what,
            dev.size(), ms);
    std::lock_guard<std::mutex> lk(mu());
    images().push_back(Image{what, &pack, base, l.bytes, l.chunk, l.sums, {}});
}

inline uint64_t digest(const std::vector<uint64_t>& v)
{
    uint64_t d = 0;
    for (size_t i = 0; i < v.size(); ++i) d += term(v[i], i);
    return d;
}

// Once the engine is ready to serve: the GPU sums every image again; later checks compare with these.
inline void seal(int rank)
{
    if (!enabled()) return;
    std::lock_guard<std::mutex> lk(mu());
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    for (Image& im : images())
    {
        im.sealed = device_sums(im.base, im.bytes, im.chunk, st);
        size_t bound = 0;
        for (size_t c = 0; c < im.sealed.size(); ++c) bound += im.sealed[c] != im.loaded[c];
        fprintf(stderr, "# rank %d: weight check: %s image sealed, digest %016llx, %zu of %zu chunks rewritten at bind\n",
                rank, im.what.c_str(), (unsigned long long)digest(im.sealed), bound, im.sealed.size());
    }
    cudaStreamDestroy(st);
}

// One check against the sealed sums, as one JSON object.
inline std::string check_json(int rank, cudaStream_t st)
{
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mu());
    if (images().empty())
        return "{\"rank\":" + std::to_string(rank) + ",\"ok\":false,\"error\":\"no weight images recorded "
               "(the C++ loader, or G6_WEIGHT_CHECK=0)\"}\n";
    std::string imgs;
    bool ok = true;
    for (const Image& im : images())
    {
        const std::vector<uint64_t> now = device_sums(im.base, im.bytes, im.chunk, st);
        size_t bad = 0, first = 0;
        for (size_t c = 0; c < now.size(); ++c)
            if (now[c] != im.sealed[c] && !bad++) first = c;
        ok = ok && bad == 0;
        char b[512];
        snprintf(b, sizeof b, "%s{\"what\":\"%s\",\"bytes\":%llu,\"chunks\":%zu,\"changed\":%zu,\"digest\":\"%016llx\"", imgs.empty() ? "" : ",",
                 im.what.c_str(), (unsigned long long)im.bytes, now.size(), bad, (unsigned long long)digest(im.sealed));
        imgs += b;
        if (bad)
        {
            snprintf(b, sizeof b, ",\"first\":{\"offset\":%llu,\"end\":%llu,\"starts_in\":\"%s\"}",
                     (unsigned long long)(first * im.chunk), (unsigned long long)std::min((first + 1) * im.chunk, im.bytes),
                     tensor_at(im.pack, first * im.chunk).c_str());
            imgs += b;
        }
        imgs += "}";
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    char head[160];
    snprintf(head, sizeof head, "{\"rank\":%d,\"ok\":%s,\"ms\":%.0f,\"images\":[", rank, ok ? "true" : "false", ms);
    return head + imgs + "]}\n";
}

// Answers DIR/check.<n> (n = 0, 1, ... with no gap, written by the supervisor) with DIR/check.<n>.json, on a
// low-priority stream of its own while the engine serves. Polls twice a second.
inline void start_watcher(const std::string& dir, int rank, int device)
{
    if (!enabled()) return;
    std::thread([dir, rank, device] {
        cudaSetDevice(device);
        int lo = 0, hi = 0;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        cudaStream_t st;
        if (cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, lo) != cudaSuccess) return;
        for (uint64_t n = 0;; )
        {
            struct stat sb;
            const std::string req = dir + "/check." + std::to_string(n);
            if (::stat(req.c_str(), &sb) != 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            const std::string out = check_json(rank, st), tmp = req + ".json.tmp";
            if (FILE* f = fopen(tmp.c_str(), "w"))
            {
                fwrite(out.data(), 1, out.size(), f);
                fclose(f);
                rename(tmp.c_str(), (req + ".json").c_str());
            }
            fprintf(stderr, "# rank %d: weight check %llu: %s", rank, (unsigned long long)n, out.c_str());
            ++n;
        }
    }).detach();
}

}  // namespace wcheck
