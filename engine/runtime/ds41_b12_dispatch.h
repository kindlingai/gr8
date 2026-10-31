// mHC decomposition: a main-model W-row prefill switch. Host-only so the
// dispatch boundary can be tested without compiling or launching CUDA.
#pragma once
#include <cstring>

namespace ds41::b12 {
inline int parse(const char* value) {
    if (!value || std::strcmp(value, "0") == 0) return 0;
    if (std::strcmp(value, "1") == 0) return 1;
    return -1;
}
inline bool on(bool enabled, bool draft, int rows, int decode_rows, int hc_rows) {
    return enabled && !draft && rows > decode_rows && rows <= hc_rows;
}
} // namespace ds41::b12
