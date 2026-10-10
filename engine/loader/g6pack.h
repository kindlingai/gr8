// g6 pack loader: rank manifest (JSON) + pack (sequential stream from a file or a pipe) -> one allocation.
//
// The allocation is a byte-for-byte image of the pack, so the manifest offsets ARE the engine's offset table:
// tensor i lives at base + entries[i].offset. Every tensor's sha256 (from the manifest) is checked while the
// bytes stream in (worker threads hash behind the reader); padding must be zero; the stream must end exactly at
// pack_bytes. File input uses O_DIRECT so the 60 GB image is not held a second time in the page cache (unified
// memory on GB10); "-" reads stdin (pack streamed over the fabric, e.g. `ssh node4 cat rank5.pack | g6 ... -`).
//
// Device visibility: the image is 2 MB-aligned host memory. g6_register_cuda() (g6pack_cuda.cpp, built only with
// CUDA) registers it with cudaHostRegister so kernels read it in place on GB10; not exercised by the host test.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace g6
{

enum class DType : uint8_t { U8, I16, F16, BF16, F32, I32, I64 };
size_t dtype_size(DType d);

struct Entry
{
    std::string name, layout;
    DType dtype;
    std::vector<int64_t> shape;
    uint64_t offset = 0, nbytes = 0;
    uint8_t sha256[32];
    int expert_format = 0;            // the entry's "expert_format": 0 = absent or "tr3", 1 = "exlr8-e4m3" (routed experts)
};

struct LoadOpts
{
    int hash_threads = 8;             // 0 = skip checksum (not recommended)
    size_t chunk = 64ull << 20;       // read size per syscall
    bool direct_io = true;            // O_DIRECT for file input (falls back to buffered + FADV_DONTNEED)
};

struct LoadStats
{
    double seconds = 0, read_seconds = 0, gb_per_s = 0;
    double copy_seconds = 0, stall_seconds = 0;   // DevImage only: inside cudaMemcpy / reader waiting for the hash worker
    uint64_t bytes = 0;
    bool used_direct_io = false;
};

class Pack
{
public:
    Pack() = default;
    Pack(const Pack&) = delete;
    Pack& operator=(const Pack&) = delete;
    ~Pack();

    // Parses the manifest only (no allocation). Returns false + err on any schema/layout problem.
    bool open_manifest(const std::string& manifest_path, std::string& err);
    // Streams the pack ("-" = stdin) into a fresh allocation and checks every sha256. Requires open_manifest.
    bool load(const std::string& pack_path, const LoadOpts& opts, LoadStats& stats, std::string& err);

    int rank() const { return rank_; }
    uint64_t bytes() const { return pack_bytes_; }
    size_t alloc_bytes() const { return alloc_bytes_; }
    uint8_t* base() const { return base_; }
    const std::vector<Entry>& entries() const { return entries_; }
    const Entry* find(const std::string& name) const;
    template <class T> const T* ptr(const std::string& name) const
    {
        const Entry* e = find(name);
        return e ? reinterpret_cast<const T*>(base_ + e->offset) : nullptr;
    }

private:
    int rank_ = -1;
    uint64_t pack_bytes_ = 0, align_ = 0;
    std::vector<Entry> entries_;            // sorted by offset
    std::unordered_map<std::string, size_t> index_;
    uint8_t* base_ = nullptr;
    size_t alloc_bytes_ = 0;
};

// SHA-256 (ARMv8 crypto extension when available, portable otherwise).
struct Sha256
{
    uint32_t h[8];
    uint8_t buf[64];
    uint64_t total = 0;
    size_t fill = 0;
    Sha256();
    void update(const uint8_t* p, size_t n);
    void final(uint8_t out[32]);
};
std::string hex(const uint8_t* p, size_t n);

#ifdef G6_WITH_CUDA
bool register_cuda(Pack& p, std::string& err);
#endif

}  // namespace g6
