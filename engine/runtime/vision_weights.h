// A vision tower's safetensors file read whole into one device allocation (ds41_vision.cuh, flash_vision.cuh): the
// header (tensor name -> dtype, shape, byte range) and the data section, copied in 64 MiB pieces through pinned memory.
// Host only (the CUDA runtime API).
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.h"

namespace g6vis {

struct StTensor { std::string dtype; std::vector<int64_t> shape; size_t begin = 0, end = 0; };

// The safetensors header of `path`: tensor name -> dtype, shape, byte range inside the data section. data_off is
// where the data section starts in the file, and who names the caller in errors.
inline std::map<std::string, StTensor> st_header(const std::string& path, size_t& data_off, size_t& file_bytes,
                                                 const char* who = "vision")
{
    const std::string w = std::string(who) + ": ";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error(w + "cannot open " + path);
    uint64_t n = 0;
    if (std::fread(&n, 8, 1, f) != 1 || n > (64u << 20)) { std::fclose(f); throw std::runtime_error(w + "bad safetensors header in " + path); }
    std::string h(n, '\0');
    if (std::fread(&h[0], 1, n, f) != n) { std::fclose(f); throw std::runtime_error(w + "short header in " + path); }
    std::fseek(f, 0, SEEK_END);
    file_bytes = (size_t)std::ftell(f);
    std::fclose(f);
    data_off = 8 + n;
    const g6v::Json j = g6v::Json::parse(h);
    std::map<std::string, StTensor> out;
    for (const auto& kv : j.o) {
        if (kv.first == "__metadata__") continue;
        StTensor t;
        t.dtype = kv.second.at("dtype").str();
        for (const auto& d : kv.second.at("shape").a) t.shape.push_back(d.i64());
        t.begin = (size_t)kv.second.at("data_offsets").at(0).i64();
        t.end = (size_t)kv.second.at("data_offsets").at(1).i64();
        if (t.end < t.begin || data_off + t.end > file_bytes) throw std::runtime_error(w + "tensor " + kv.first + " past the file end");
        out[kv.first] = t;
    }
    return out;
}

// The file's data section in device memory, and its header.
struct Blob {
    void* dev = nullptr;
    size_t bytes = 0;
    std::map<std::string, StTensor> hdr;
    // tensor `name` with the given dtype and shape, as a device pointer into the blob
    const void* get(const std::string& name, const char* dtype, const std::vector<int64_t>& shape, const std::string& path,
                    const char* who) const
    {
        auto it = hdr.find(name);
        if (it == hdr.end()) throw std::runtime_error(std::string(who) + ": " + path + " has no tensor " + name);
        if (it->second.dtype != dtype || it->second.shape != shape)
            throw std::runtime_error(std::string(who) + ": tensor " + name + " has the wrong dtype or shape");
        return (const char*)dev + it->second.begin;
    }
};

inline Blob load_blob(const std::string& path, const char* who)
{
    const std::string w = std::string(who) + ": ";
    size_t off = 0, fbytes = 0;
    Blob b;
    b.hdr = st_header(path, off, fbytes, who);
    b.bytes = fbytes - off;
    if (cudaMalloc(&b.dev, b.bytes) != cudaSuccess) throw std::runtime_error(w + "cannot allocate " + std::to_string(b.bytes) + " device bytes");
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error(w + "cannot open " + path);
    std::fseek(f, (long)off, SEEK_SET);
    const size_t CH = 64u << 20;
    void* host = nullptr;
    if (cudaMallocHost(&host, CH) != cudaSuccess) { std::fclose(f); throw std::runtime_error(w + "cannot allocate pinned memory"); }
    size_t done = 0;
    while (done < b.bytes) {
        const size_t k = std::min(CH, b.bytes - done);
        if (std::fread(host, 1, k, f) != k || cudaMemcpy((char*)b.dev + done, host, k, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fclose(f); cudaFreeHost(host);
            throw std::runtime_error(w + "short read of " + path);
        }
        done += k;
    }
    std::fclose(f);
    cudaFreeHost(host);
    return b;
}

}  // namespace g6vis
