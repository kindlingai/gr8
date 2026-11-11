// Request-local bitmask: direct attention send and dual-layout FFN receive.
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>
namespace ds41::v2 {
struct SendFfnOpt {
    std::vector<int> modes;
    int mode = 0;
    std::map<unsigned, uint64_t> seen;
    static bool decode(const std::vector<int64_t>& values, std::vector<int>& out) {
        out.clear();
        if (values.empty()) return false;
        for (auto value : values) {
            if (value < 0 || value > 3) return false;
            out.push_back(int(value));
        }
        return true;
    }
    bool init(const std::vector<int64_t>& values) {
        seen.clear();
        if (!decode(values, modes)) { mode = 0; return false; }
        mode = modes[0];
        return true;
    }
    bool attn_opt() const { return (mode & 1) != 0; }
    bool ffn_opt() const { return (mode & 2) != 0; }
    void select(uint64_t request, unsigned slot, int rank, FILE* log = stderr) {
        if (modes.empty()) { mode = 0; return; }
        mode = modes[request % modes.size()];
        if (modes.size() == 1 && modes[0] == 0) return;
        auto previous = seen.find(slot);
        if (previous != seen.end() && previous->second == request) return;
        seen[slot] = request;
        if (mode != 0 && log)
            std::fprintf(log, "# ds41_v2_send_ffn_opt rank=%d request=%llu slot=%u on=1 mode=%d\n",
                         rank, (unsigned long long)request, slot, mode);
    }
};
}
