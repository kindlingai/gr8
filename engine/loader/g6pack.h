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
    int expert_format = 0;            // the entry's "expert_format": 0 = absent or "tr3", 1 = "exlr8-e4m3" (routed experts, #19)
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

// Pack bytes kept in other files: exlr8 v2 slices (#65, tools/convert/convert.py --v2-refs). The pack is sparse there,
// and rank<r>.sources lists where each range lives. The loaders call fill() on every chunk as it arrives, before
// hashing, so the image and every sha256 check are exactly those of the full pack (tools/convert/sources.py is the
// reference implementation). Chunks must arrive in increasing pack order.
//   text format: a header "g6sources-v1 <root>", then one range per line, sorted by pack offset:
//   <pack_offset> <nbytes> <op> <file_offset> <file_nbytes> <path relative to root>
//   op copy:    pack bytes = file bytes file_offset.. (nbytes == file_nbytes)
//   op item128: one whole v2 down half, file order [col16 / 8][k16][col16 % 8][tile] -> g6 fc2 order
//               [col16][k16][tile] (tools/convert/v2.py down_to_g6), nbytes == file_nbytes == 98304 * width
class Sources
{
public:
    Sources() = default;
    Sources(const Sources&) = delete;
    Sources& operator=(const Sources&) = delete;
    ~Sources();

    // root: where the relative paths live ("" = the root the file names). Checks order, overlap, the pack size
    // and that every source file is long enough.
    bool open(const std::string& path, const std::string& root, uint64_t pack_bytes, std::string& err);
    bool fill(uint8_t* buf, uint64_t pos, size_t n, std::string& err);
    size_t ranges() const { return seg_.size(); }
    uint64_t bytes() const { return bytes_; }
    const std::string& root() const { return root_; }

private:
    struct Seg { uint64_t pack_off, nbytes, file_off, file_nbytes; int op, file; };
    std::vector<Seg> seg_;
    std::vector<int> fds_;
    std::string root_;
    size_t cur_ = 0, half_seg_ = SIZE_MAX;
    uint64_t bytes_ = 0;
    std::vector<uint8_t> raw_, half_;
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
    // src: fill its ranges from their files (a sparse pack, see Sources).
    bool load(const std::string& pack_path, const LoadOpts& opts, LoadStats& stats, std::string& err,
              Sources* src = nullptr);

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
