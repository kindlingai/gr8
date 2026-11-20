// Bootstrap conductor: unit callbacks are stock math until the A/B hooks are
// installed. The stock callback sees the original plan spans, including the
// boundary's hc_attn(21) and compaction; no missing tail state or index lists.
#pragma once
#include "ds41_ops.h"
#include <cstddef>
#include <vector>
namespace ds41::v2 {
struct Run { size_t begin = 0, end = 0; Unit unit = U_COUNT; int layer = -1; };
inline std::vector<Run> range_a_runs(const std::vector<Op>& ops,size_t end) {
    std::vector<Run> runs;
    if (end > ops.size()) return runs;
    for(size_t b=0;b<end;) {
        size_t e=b+1;
        while(e<end && ops[e].unit==ops[b].unit && ops[e].ui==ops[b].ui) ++e;
        runs.push_back({b,e,ops[b].unit,ops[b].ui}); b=e;
    }
    return runs;
}
template<class Execute>
inline Stop walk_stock(const std::vector<Op>& ops,size_t end,Execute&& execute) {
    for(const Run& r:range_a_runs(ops,end)) {
        Stop stop=execute(r);
        if(stop.stopped) return stop;
    }
    return {};
}
} // namespace ds41::v2
