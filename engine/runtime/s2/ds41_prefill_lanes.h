// DeepSeek-V4.1-Flash two-lane W prefill admission (GR8_DS_PF_LANES=2, ../ds41_lanes.h), host-only. Included by ds41_native.cuh only, so
// the S2 headers every model includes (sessions.h -> ds41_prefill.h) do not pull the lane contract in.
//
// Lanes 1 (unset, the default): ds41_prefill_single_lane, the Config unchanged. Lanes 2: the scheduler plans quanta of up to 2W
// rows of one session (Config::two_lanes, scheduler.h) and, while a session decodes, one lane (decode_prefill_rows = W). serve_ds41
// cuts each quantum into lane A (the first min(rows, W) rows, one id of Dispatch's main sequence, the main fabric) and lane B (the rest,
// positions A.context.., one id of lane B's own counter, lane B's fabric). Every rank runs the same Driver, so both ids agree.
#pragma once
#include "ds41_prefill.h"
#include "../ds41_lanes.h"

namespace g6s2 {
inline void ds41_prefill_lanes(Config& sc, uint32_t rows, int lanes) {
    require(lanes == 1 || lanes == 2, ds41::lanes::kParseError);
    if (lanes == 1) { ds41_prefill_single_lane(sc, rows); return; }
    require(rows > 0 && rows % PAGE == 0, "DS41 two-lane prefill width must be a positive multiple of 64");
    sc.prefill_rows = rows;
    sc.two_lanes = true;
    sc.decode_prefill_rows = ds41::lanes::decode_prefill_rows(2, rows);
}
// One quantum as the callees get it: b.rows == 0 = one lane (lane B's counter untouched).
struct Ds41LaneChunks { Ds41PrefillChunk a, b; };
inline Ds41LaneChunks ds41_make_lane_chunks(const std::vector<int>& prompt, uint32_t first, uint32_t rows, uint32_t width, bool two_lanes,
                                            uint64_t sequence, ds41::lanes::Counter& lane_b, uint64_t request, unsigned slot, const Run& run) {
    require(ds41::lanes::split_ok(rows, width) && (two_lanes || rows <= width), "DS41 prefill quantum outside 1..W (one lane) or 1..2W (two lanes)");
    const ds41::lanes::Split sp = ds41::lanes::split(rows, width);
    Ds41LaneChunks q;
    q.a = ds41_make_chunk(prompt, first, sp.a, width, sequence, request, slot, run);
    if (sp.b) {   // the id is taken only after lane B's chunk is valid
        q.b = ds41_make_chunk(prompt, uint32_t(q.a.context), sp.b, width, lane_b.next, request, slot, run);
        lane_b.take();
    }
    return q;
}
}
