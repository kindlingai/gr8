// Park-miss attribution only: never changes admission, reuse or the pool digest.
#pragma once
#include "../prefix_reuse.h"
#include <cstdio>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace g6s2 {
using Ds41ParkMissKey = std::tuple<size_t, int, int, uint32_t, int>;

// The caller holds ck_mu. Repeated admission attempts for one state print only once.
template<class ParkedTable, class Namespace>
inline void ds41_park_miss(bool park, int q, const ParkedTable& parked, int slots, const Namespace& ns,
                          const std::vector<int>& prompt, int limit, int rank,
                          std::optional<Ds41ParkMissKey>& last_miss, FILE* out = stderr) {
    if (!park || q >= 0) return;
    int slot = -1, lcp = -1;
    for (size_t i = 0; i < size_t(slots); ++i) {
        const auto& k = parked[i];
        if (!k.on || k.ns != ns || !k.pos) continue;
        // Include limit-rejected slots: they explain why a longer parked prefix was not resumed.
        const int n = g6rt::PrefixReuse::common_prefix(k.tokens, prompt);
        if (slot < 0 || n > lcp || (n == lcp && k.pos > parked[size_t(slot)].pos)) {
            slot = int(i); lcp = n;  // ascending iteration retains the lowest slot on a full tie
        }
    }
    if (slot < 0) return;
    const auto pos = parked[size_t(slot)].pos;
    const Ds41ParkMissKey key{prompt.size(), slot, lcp, pos, limit};
    if (last_miss && *last_miss == key) return;
    last_miss = key;
    std::fprintf(out, "# rank %d ds41 park_miss slot %d lcp %d pos %u limit %d prompt %zu\n",
                 rank, slot, lcp, pos, limit, prompt.size());
}
} // namespace g6s2
