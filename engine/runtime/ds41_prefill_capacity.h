#pragma once
#include <cstddef>
#include <cerrno>
#include <climits>
#include <cstdlib>

// Caller-side row budget for the DS41 prefill hook. Zero means a bad
// explicit value; an unset value leaves the decode allocation unchanged.
inline int ds41_prefill_rowcap(const char* value, int decode_rows) {
    if (!value) return decode_rows;
    if (!*value) return 0;
    errno = 0;
    char* end = nullptr;
    const long n = std::strtol(value, &end, 10);
    if (errno || end == value || *end || n < decode_rows || n > 4096 || n > INT_MAX) return 0;
    return static_cast<int>(n);
}

// The shared decode/prefill fabric's maximum payload, not its physical stride.
// Fabric::init appends the 8-byte sequence tail and rounds the stride to 256 B.
inline size_t ds41_prefill_fabric_payload(int rows, int hidden, int engram_dim) {
    return static_cast<size_t>(rows) * static_cast<size_t>(engram_dim ? engram_dim : hidden) * 2;
}
