// Diagnostic-only, versioned selected-pool capture. No CUDA or communication changes.
// Each block: "FPIX0001", LE uint32 {layer, context, rows, 512}, then rows of
// LE int32 {query_position, 512 sorted pool IDs padded with -1}.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace flash_index_capture {
constexpr int POOLS = 512, KP = 4, STRIDE = 2304, ROW_WORDS = 1 + POOLS;

inline std::vector<int32_t> pack(int context, int rows, const int* positions, const int* indices) {
    if (context <= 0 || rows <= 0 || rows > context || !positions || !indices)
        throw std::runtime_error("invalid capture dimensions");
    std::vector<int32_t> out(static_cast<size_t>(rows) * ROW_WORDS, -1);
    for (int r = 0; r < rows; ++r) {
        const int pos = positions[r];
        if (pos != context - rows + r) throw std::runtime_error("non-contiguous capture positions");
        const int count = pos + 1, visible = count / KP, keep = std::min(visible, POOLS);
        const int* ids = indices + static_cast<size_t>(r) * STRIDE;
        int32_t* dst = out.data() + static_cast<size_t>(r) * ROW_WORDS;
        dst[0] = pos;
        int previous = -1;
        for (int j = 0; j < keep; ++j) {
            const int first = ids[j * KP];
            if (first < 0 || first % KP) throw std::runtime_error("invalid pool start");
            const int pool = first / KP;
            if (pool <= previous || pool >= visible) throw std::runtime_error("invalid pool order or visibility");
            for (int k = 0; k < KP; ++k)
                if (ids[j * KP + k] != first + k) throw std::runtime_error("invalid expanded pool");
            dst[1 + j] = pool;
            previous = pool;
        }
        const int tail_start = count - count % KP, selected = keep * KP;
        for (int j = selected; j < STRIDE; ++j) {
            const int tail = tail_start + j - selected;
            const int expected = tail < count ? tail : -1;
            if (ids[j] != expected) throw std::runtime_error("invalid tail or padding");
        }
    }
    return out;
}

class Writer {
    FILE* file_ = nullptr;
    std::string path_;
public:
    ~Writer() { if (file_ && fclose(file_) != 0) std::terminate(); }
    void write(const std::string& path, int layer, int context, int rows,
               const int* positions, const int* indices) {
        if (layer < 0) throw std::runtime_error("invalid capture layer");
        // Validate before emitting anything, so a bad row never looks like a complete block.
        const auto packed = pack(context, rows, positions, indices);
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian) != 1)
            throw std::runtime_error("capture requires a little-endian host");
        if (!file_) {
            path_ = path;
            file_ = fopen(path.c_str(), "wb");
            if (!file_) throw std::runtime_error("cannot open pool capture: " + path);
        } else if (path != path_) throw std::runtime_error("pool capture path changed");
        const char magic[8] = {'F','P','I','X','0','0','0','1'};
        const uint32_t hdr[4] = {static_cast<uint32_t>(layer), static_cast<uint32_t>(context),
                                 static_cast<uint32_t>(rows), POOLS};
        if (fwrite(magic, 1, sizeof magic, file_) != sizeof magic ||
            fwrite(hdr, sizeof hdr, 1, file_) != 1 ||
            fwrite(packed.data(), sizeof(int32_t), packed.size(), file_) != packed.size() ||
            fflush(file_) != 0)
            throw std::runtime_error("pool capture write failed: " + path);
    }
};
} // namespace flash_index_capture
