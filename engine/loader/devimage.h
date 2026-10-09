// g6 device image: rank manifest (an opened g6::Pack) + pack stream (file or pipe) -> ONE cudaMalloc allocation that is
// a byte-for-byte image of the pack. Tensor i lives at base + entries[i].offset, the same offsets as the manifest.
//
// No host copy of the pack is kept: the stream is read in chunks into a small pinned staging ring (3 buffers,
// <= 256 MB in total), each chunk is cudaMemcpy'd to the device and then handed to one worker thread, which walks
// the stream in order, hashes every tensor (sha256 against the manifest) and checks that padding is zero. The
// reader only waits for the worker when all staging buffers are still being hashed. The stream must end exactly at
// pack_bytes. A failed load frees the device memory.
//
// Why device memory: on GB10, GPU reads of host-pinned memory are slow (results/m2/RESULTS-M2b.md), and a host image
// plus a device copy of a 62 GB pack do not fit in 128 GB of unified memory.
//
// Only the thread that calls load() touches CUDA; it must have (or be allowed to create) the context the kernels use.
#pragma once
#include "g6pack.h"

namespace g6
{

class DevImage
{
public:
    DevImage() = default;
    DevImage(const DevImage&) = delete;
    DevImage& operator=(const DevImage&) = delete;
    ~DevImage();

    // `pack` needs open_manifest() only (Pack::load is not called) and must outlive this image.
    // `path` is the pack file, or "-" for stdin. opts: chunk = staging buffer size (capped so the ring stays
    // <= 256 MB), hash_threads = 0 skips the sha256 and padding checks (not recommended), direct_io as in Pack::load.
    bool load(const Pack& pack, const std::string& path, const LoadOpts& opts, LoadStats& stats, std::string& err);
    void reset();   // frees the device memory
    // Takes ownership of a device image of pack's bytes that another loader filled (the Rust loader, G6_RS_LOADER).
    void adopt(const Pack& pack, uint8_t* dev, uint64_t bytes);

    const uint8_t* base() const { return base_; }
    uint64_t bytes() const { return bytes_; }
    const void* ptr(const Entry& e) const { return base_ + e.offset; }
    template <class T> const T* ptr(const std::string& name) const
    {
        const Entry* e = pack_ ? pack_->find(name) : nullptr;
        return e && base_ ? reinterpret_cast<const T*>(base_ + e->offset) : nullptr;
    }

private:
    const Pack* pack_ = nullptr;
    uint8_t* base_ = nullptr;   // device pointer
    uint64_t bytes_ = 0;
};

}  // namespace g6
