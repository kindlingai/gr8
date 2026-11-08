// Bounded once-per-request diagnostics for the v2 attention pipeline.
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
namespace ds41::v2 {
struct AttnPipe {
    int on = 0;
    std::map<unsigned, uint64_t> seen;
    void select(uint64_t request, unsigned slot, FILE* log = stderr) {
        auto it = seen.find(slot);
        if (it != seen.end() && it->second == request) return;
        seen[slot] = request;
        std::fprintf(log, "# ds41_v2_attn_pipe request=%llu on=%d\n", (unsigned long long)request, int(on));
    }
};
}
