// DS41 image input in the rank agreement (g6m4.cu, the DS41 S2 runtime signature). Each rank looks for the tower file on its
// own, and a rank without it boots text-only, so the ranks agree on what they loaded, not on the GR8_DS41_VISION knob: a
// rank with the tower adds " vision <id>" (the file's byte size and the FNV-1a of its safetensors header), a rank without it
// adds nothing. Ranks that differ (one without the file, or another file) refuse at bootstrap. No CUDA types.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ds41v {

// "<bytes>-<fnv1a64 hex of the 8-byte header length + the header>"; "" when the file cannot be read or has no header
inline std::string file_id(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "";
    uint64_t n = 0;
    std::string id;
    if (std::fread(&n, 8, 1, f) == 1 && n > 0 && n <= (64u << 20)) {
        std::vector<unsigned char> h(size_t(n) + 8);
        for (int i = 0; i < 8; ++i) h[size_t(i)] = (unsigned char)(n >> (8 * i));
        if (std::fread(h.data() + 8, 1, size_t(n), f) == size_t(n) && std::fseek(f, 0, SEEK_END) == 0) {
            const long bytes = std::ftell(f);
            uint64_t x = 0xCBF29CE484222325ull;
            for (unsigned char c : h) x = (x ^ c) * 0x100000001B3ull;
            char buf[48];
            std::snprintf(buf, sizeof buf, "%ld-%016llx", bytes, (unsigned long long)x);
            if (bytes > 0) id = buf;
        }
    }
    std::fclose(f);
    return id;
}

// the runtime signature's vision words: " vision <id>" with the tower loaded, nothing without
inline std::string signature(bool loaded, const std::string& id) { return loaded ? " vision " + id : std::string(); }

} // namespace ds41v
