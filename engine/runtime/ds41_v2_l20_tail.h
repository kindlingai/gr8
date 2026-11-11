// Request-local selection of the v2 layer-20 hoisted tail.
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>
namespace ds41::v2 {
struct L20Tail {
    std::vector<int> modes;
    bool on = false;
    std::map<unsigned, uint64_t> seen;
    static bool decode(const std::vector<int64_t>& values, std::vector<int>& out) {
        out.clear();
        if (values.empty()) return false;
        for (auto value : values) if (value != 0 && value != 1) return false;
        for (auto value : values) out.push_back(int(value));
        return true;
    }
    bool init(const std::vector<int64_t>& values) {
        seen.clear();
        if (!decode(values, modes)) { on = false; return false; }
        on = modes[0] != 0;
        return true;
    }
    void select(uint64_t request, unsigned slot, int rank, FILE* log = stderr) {
        if (modes.empty()) { on = false; return; }
        on = modes[request % modes.size()] != 0;
        if (modes.size() == 1 && modes[0] == 0) return;
        auto previous = seen.find(slot);
        if (previous != seen.end() && previous->second == request) return;
        seen[slot] = request;
        std::fprintf(log, "# ds41_v2_l20_tail request=%llu on=%d rank=%d\n",
                     (unsigned long long)request, int(on), rank);
    }
};
}
