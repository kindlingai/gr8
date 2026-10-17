// G6_XTRACE proxy stamp layout (fabric.cpp): per (step % XT_STEPS, collective) one record of XT_RAILS rails x 4 ticks.
// A ring or hybrid fabric has up to 4 rails; the stamps keep two: rail 0, and every rail above 0 folded into 1. Every
// stamp store, record clear, and dump loop goes through these, so no rail index reaches another record or past the
// vector. The per-QP WC error counters keep all XT_WC_RAILS rails. No CUDA here: engine/comms/tests/xtrail_test.cpp.
#pragma once
#include <cstddef>

namespace g6 {

constexpr int XT_RAILS = 2, XT_TICKS = 4, XT_REC = XT_RAILS * XT_TICKS;   // u64 per proxy record
constexpr int XT_WC_RAILS = 4, XT_WC_PEERS = 8;
inline int xt_rail(int rail) { return rail > 0 ? 1 : 0; }
inline size_t xt_stamps(size_t records) { return records * XT_REC; }
inline size_t xt_at(size_t record, int rail, int tick) { return record * XT_REC + (size_t)xt_rail(rail) * XT_TICKS + (size_t)tick; }
inline int xt_dump_rails(int nrails) { return nrails < XT_RAILS ? nrails : XT_RAILS; }

}  // namespace g6
