// Host-only DS41 serve main-fabric slot table. Shared by boot and the c8 pack
// slot fixture so the test checks the table installed by Fabric::init.
#pragma once
#include "../comms/fabric.h"
#include "ds41_rs.h"
#include "ds41_pf_split.h"
#include "ds41_pack_layout.h"
#include <algorithm>
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
    FabricRuns o;
    const int nc = 2 * nl + 1 + nx;
    const size_t dec = size_t(mx) * (nx ? 6144 : hid) * 2;
    const size_t w_hid = size_t(pf_rows) * hid * 2;
    const size_t late_hid = size_t(ds41::br::late_rows(pf_rows, tail)) * hid * 2;
    o.cfg.run1 = i_late >= 0 ? 2 * i_late : 2 * nl;
    o.cfg.run2 = 2 * nl + 1;
    o.cfg.slot_bytes = std::max(dec, w_hid);
    // A ragged pack keeps every segment's <=64 rows through the late layers,
    // so the flattened late collective can carry all MAX_ROWS, not PF_TAIL.
    // OFF preserves the original solo table byte for byte.
    const size_t pack_late = c8_pack && i_late >= 0 ? size_t(ds41::pack::MAX_ROWS) * hid * 2 : 0;
    o.cfg.slot_bytes1 = std::max({dec, i_late >= 0 ? late_hid : size_t(0), head_payload, pack_late});
    o.cfg.slot_bytes2 = nx ? std::max(dec, size_t(pf_rows) * 6144 * 2) : 0;
    o.cfg.ncoll = nc;
    if (rs_on) {
        o.rs_k = 2 * i_late;
        o.quarter = ds41::rs::quarter(pf_rows, nr);
        o.slice_stride = ds41::rs::slice_stride(o.quarter, size_t(hid) * 2);
        o.cfg.slot_bytes = std::max(o.cfg.slot_bytes, size_t(nr) * o.slice_stride - 8);
        o.cfg.run3 = nc;
        o.cfg.slot_bytes3 = size_t(o.quarter) * hid * 2;
        o.cfg.ncoll = ds41::rs::main_ncoll(nc, o.rs_k, rowshard);
    }
    return o;
}
}
