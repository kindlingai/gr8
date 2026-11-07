// Row-sharded mHC prefill host rules. No CUDA: the CPU reference test uses these exact guards/offsets. boot_ok and the
// state sizes, which the fabric plan reads, are engine/scheduler-rs src/plan_sizes.rs.
#pragma once
#include "ds41_rs.h"
#include "../scheduler-rs/g6plan.h"
#include <cstddef>
namespace ds41::rowshard {
// Supported only for the 4096-row width (1024 rows per rank at NR 4) and the 128-row replay tail.
inline bool boot_ok(int width, int tail, int nr, bool rs, bool fuse) { return g6plan_ds41_rowshard_ok(width, tail, nr, rs, fuse); }
inline bool chunk_on(bool enabled, int rows, int late, int nr) {
    return enabled && nr == 4 && rows == 4096 && late == 128 && rs::quarter(rows,nr) == 1024 &&
           rs::owned(rows,nr,nr-1) >= late;
}
inline int row0(int rows, int nr, int rank) { return rank * rs::quarter(rows,nr); }
// Six tensors crossing the bounded-replay boundary: streams, mix, pre, post, comb, y.
inline size_t state_row_bytes(int hid, size_t trunk_bytes) { return g6plan_ds41_state_row_bytes(hid, trunk_bytes); }
inline size_t state_bytes(int late, int hid, size_t trunk_bytes) { return g6plan_ds41_state_bytes(late, hid, trunk_bytes); }
inline size_t y_mxfp8_bytes(int rows, int hid) { return size_t(rows)*hid + size_t(rows)*(hid/32); }
}
