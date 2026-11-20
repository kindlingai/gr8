// GR8_CP_STEP_ROWS=n (every rank, read at boot; default off): a step past R carries at most n rows. Host only, no CUDA;
// the rules are engine/scheduler-rs/src/cp_rows.rs.
// Since #94 a split run past R shares batched steps, and the CP fabric (and the CP graphs at every bucket past R) are
// sized for the widest S2 step: sessions x (mtp + 1) = 24 rows at 8 sessions, MTP 2. With the knob the fabric is sized
// for n rows and the round a past-R session takes holds at most n / (mtp + 1) sessions; the other sessions sit that round
// out (s2_batch.h Batch::round: their state does not change) and go in the next one, in cycles in which every decoding
// session gets exactly one round. A round without a past-R session (a step below R) keeps the full batch.
// GR8_CP_STEP_SEG=1 (needs the knob): a past-R round never takes a session below R, so a session below R never runs the CP
// block (whose LSE merge rounds differently from the replicated path) because of a long session it shares the pool with.
// Every input is replicated state (the scheduler's decoding sessions in request-id order, the driver's grow stalls,
// Batch lengths, boot constants), so every rank forms the same rounds without exchanging them.
#pragma once
#include "../../scheduler-rs/g6sched.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace g6s2 {
// The contexts that run the CP block: ctx > cp_after(...) exactly when bk[bucket_of(ctx)] > R, bucket_of = the smallest
// bucket >= ctx, clamped to the last (rank.cuh). bk ascending, nbk >= 1. R < 0: no CP (never past). Returns the largest
// bucket <= R (0 when every bucket is past R), or UINT64_MAX when no bucket is past R.
inline uint64_t cp_after(const int* bk, int nbk, int R) { return g6sch_cp_after(bk, nbk, R); }
// The boot value of v = GR8_CP_STEP_ROWS (0 = off), solo = the rows a CP step takes without S2 batching (the solo and
// prompt graphs' row counts: the fabric's size before #94), full = the widest S2 step (#94's size). A value below solo
// is refused (-1, *err); one above full is clamped to full.
inline int cp_step_rows(int64_t v, int solo, int full, std::string* err)
{
    int n = 0;
    if (!g6sch_cp_step_rows(v, solo, full, &n)) return n;
    *err = "GR8_CP_STEP_ROWS=" + std::to_string(v) + ": at least " + std::to_string(solo) +
           " (the rows of a solo or prompt step past R: the verify width and --chunk)";
    return -1;
}
struct CpCand { uint64_t id; bool past; };   // a decoding session: its immutable id, its next round may run past R
static_assert(sizeof(CpCand) == sizeof(g6sch_cand), "scheduler-rs layout");
// CpPick and CpRound: m4sim.cu and the tests only; the driver picks in Rust (driver.rs)
struct CpPick {
    std::vector<bool> in;   // per candidate: in this round
    bool capped = false;    // a cycle round (the others sit it out); false = everyone, and the cycle resets
    bool fresh = false;     // capped: this round starts a new cycle
};
class CpRound {
    std::vector<uint64_t> served_;   // ids served in the current cycle
public:
    // The round's sessions among cand (the decoding sessions that are not stalled, in request-id order). rows = n (0 =
    // off: everyone), per = the most rows one session's verify takes (mtp + 1), seg = GR8_CP_STEP_SEG. Pure: commit()
    // records it once the round really runs.
    CpPick pick(const std::vector<CpCand>& cand, uint32_t rows, uint32_t per, bool seg) const {
        CpPick p;
        std::unique_ptr<bool[]> in(new bool[std::max<size_t>(cand.size(), 1)]);
        g6sch_cp_pick(reinterpret_cast<const g6sch_cand*>(cand.data()), cand.size(), served_.data(), served_.size(), rows, per, seg,
                      in.get(), &p.capped, &p.fresh);
        p.in.assign(in.get(), in.get() + cand.size());
        return p;
    }
    void commit(const CpPick& p, const std::vector<CpCand>& cand) {
        if (!p.capped || p.fresh) served_.clear();
        if (!p.capped) return;
        for (size_t i = 0; i < cand.size(); ++i) if (p.in[i]) served_.push_back(cand[i].id);
    }
    const std::vector<uint64_t>& cycle() const { return served_; }
};
} // namespace g6s2
