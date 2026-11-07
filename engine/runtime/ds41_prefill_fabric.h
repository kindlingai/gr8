// Host-only DS41 serve main-fabric slot table. Shared by boot and the c8 pack slot fixture so the test checks the table
// installed by Fabric::init. The rule is engine/scheduler-rs src/plan_sizes.rs ds41_fabric_runs.
#pragma once
#include "../comms/fabric.h"
#include "../scheduler-rs/g6plan.h"
#include "ds41_rs.h"
#include "ds41_pf_split.h"
#include "ds41_pack_layout.h"
#include <cstddef>

namespace ds41::pf {
struct FabricRuns {
    g6::FabricConfig cfg;
    int rs_k = 0;
    int quarter = 0;
    size_t slice_stride = 0;
};

inline FabricRuns fabric_runs(int pf_rows, int tail, int i_late, int nl, int nx,
                              int mx, int hid, int nr, size_t head_payload,
                              bool rs_on, bool c8_pack, bool rowshard = false) {
    g6plan_runs r{};
    g6plan_ds41_runs(pf_rows, tail, i_late, nl, nx, mx, hid, nr, head_payload, rs_on, c8_pack, rowshard, ds41::pack::MAX_ROWS, &r);
    FabricRuns o;
    o.cfg.run1 = r.run1; o.cfg.run2 = r.run2; o.cfg.run3 = r.run3; o.cfg.ncoll = r.ncoll;
    o.cfg.slot_bytes = r.slot_bytes; o.cfg.slot_bytes1 = r.slot_bytes1; o.cfg.slot_bytes2 = r.slot_bytes2; o.cfg.slot_bytes3 = r.slot_bytes3;
    o.rs_k = r.rs_k; o.quarter = r.quarter; o.slice_stride = r.slice_stride;
    return o;
}
}
