// B12 row-sharded mHC prefill host rules. No CUDA: the CPU reference test uses these exact guards/offsets.
#pragma once
#include "ds41_rs.h"
#include <cstddef>
namespace ds41::rowshard {
// First source cut is deliberately limited to the ruled 4096/1024-row geometry and 128-row replay tail.
inline bool boot_ok(int width, int tail, int nr, bool rs, bool fuse) {
    return width == 4096 && tail == 128 && nr == 4 && rs && fuse;
}
inline bool chunk_on(bool enabled, int rows, int late, int nr) {
    return enabled && nr == 4 && rows == 4096 && late == 128 && rs::quarter(rows,nr) == 1024 &&
           rs::owned(rows,nr,nr-1) >= late;
}
inline int row0(int rows, int nr, int rank) { return rank * rs::quarter(rows,nr); }
// Six tensors crossing the bounded-replay boundary: streams, mix, pre, post, comb, y.
inline size_t state_row_bytes(int hid, size_t trunk_bytes) {
    return size_t(4)*hid*trunk_bytes + size_t(24+4+4+16)*sizeof(float) + size_t(hid)*2;
}
inline size_t state_bytes(int late, int hid, size_t trunk_bytes) { return size_t(late)*state_row_bytes(hid,trunk_bytes); }
inline size_t y_mxfp8_bytes(int rows, int hid) { return size_t(rows)*hid + size_t(rows)*(hid/32); }
}
