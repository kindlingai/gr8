// Host-only per-request bitmask selector for GR8_DS_PF_V2_BF16_TC (masks 0..7).
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

namespace ds41::v2 {
struct Bf16Tc {
    static constexpr int MIN_ROWS = 16;
    std::vector<int> masks;
    int mask = 0;
    std::map<unsigned, uint64_t> seen;

    static bool decode(const std::vector<int64_t>& values, std::vector<int>& out) {
        out.clear();
        if (values.empty()) return false;
        for (int64_t v : values) {
            if (v < 0 || v > 7) return false;
            out.push_back(static_cast<int>(v));
        }
        return true;
    }
    bool init(const std::vector<int64_t>& values) {
        seen.clear();
        if (!decode(values, masks)) { mask = 0; return false; }
        mask = masks[0];
        return true;
    }
    bool on() const { return mask != 0; }
    bool gate(int rows) const { return (mask & 1) != 0 && rows >= MIN_ROWS; }
    bool comp(int rows) const { return (mask & 2) != 0 && rows >= MIN_ROWS; }
    bool idx(int rows) const { return (mask & 4) != 0 && rows >= MIN_ROWS; }

    void select(uint64_t request, unsigned slot, int rank, FILE* log = stderr) {
        if (masks.empty()) { mask = 0; return; }
        mask = masks[request % masks.size()];
        if (masks.size() == 1 && masks[0] == 0) return;
        auto it = seen.find(slot);
        if (it != seen.end() && it->second == request) return;
        seen[slot] = request;
        if (mask != 0 && log) {
            std::fprintf(log, "# ds41_v2_bf16_tc rank=%d request=%llu slot=%u on=1 mask=%d\n",
                rank, (unsigned long long)request, slot, mask);
        }
    }
};
} // namespace ds41::v2
