#pragma once
#include <cstddef>
#include <cerrno>
#include <climits>
#include <cstdlib>

// Caller-side row budget for the DS41 prefill hook. Zero means a bad
// explicit value; an unset value leaves the decode allocation unchanged.
constexpr int DS41_PF_OLD_MAX_ROWS = 4096;
constexpr int DS41_PF_V2_MAX_ROWS = 8192;
constexpr int DS41_PF_V2_W16K_MAX_ROWS = 16384;
inline int ds41_prefill_rowcap(const char* value, int decode_rows, bool v2 = false, bool v2_w16k = false) {
    if (!value) return decode_rows;
    if (!*value) return 0;
    errno = 0;
    char* end = nullptr;
    const long n = std::strtol(value, &end, 10);
    const int cap = v2 ? (v2_w16k ? DS41_PF_V2_W16K_MAX_ROWS : DS41_PF_V2_MAX_ROWS) : DS41_PF_OLD_MAX_ROWS;
    if (errno || end == value || *end || n < decode_rows || n > cap || n > INT_MAX) return 0;
    return static_cast<int>(n);
}

inline int ds41_prefill_main_fabric_rows(int pf_rows, bool v2, bool v2_w16k) {
    return v2 && v2_w16k && pf_rows > DS41_PF_OLD_MAX_ROWS ? DS41_PF_OLD_MAX_ROWS : pf_rows;
}

// The shared decode/prefill fabric's maximum payload, not its physical stride.
// Fabric::init appends the 8-byte sequence tail and rounds the stride to 256 B.
inline size_t ds41_prefill_fabric_payload(int rows, int hidden, int engram_dim) {
    return static_cast<size_t>(rows) * static_cast<size_t>(engram_dim ? engram_dim : hidden) * 2;
}
