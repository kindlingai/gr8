// GR8_CP_STEP_ROWS=n (every rank, read at boot; default off): a step past R carries at most n rows. Host only, no CUDA.
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
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace g6s2 {
// The contexts that run the CP block: ctx > cp_after(...) exactly when bk[bucket_of(ctx)] > R, bucket_of = the smallest
// bucket >= ctx, clamped to the last (rank.cuh). bk ascending, nbk >= 1. R < 0: no CP (never past). Returns the largest
// bucket <= R (0 when every bucket is past R), or UINT64_MAX when no bucket is past R.
inline uint64_t cp_after(const int* bk, int nbk, int R)
{
    if (R < 0 || nbk < 1 || bk[nbk - 1] <= R) return std::numeric_limits<uint64_t>::max();
    uint64_t t = 0;
    for (int k = 0; k < nbk && bk[k] <= R; ++k) t = uint64_t(bk[k]);
    return t;
}
// The boot value of v = GR8_CP_STEP_ROWS (0 = off), solo = the rows a CP step takes without S2 batching (the solo and
// prompt graphs' row counts: the fabric's size before #94), full = the widest S2 step (#94's size). A value below solo
// is refused (-1, *err); one above full is clamped to full.
inline int cp_step_rows(int64_t v, int solo, int full, std::string* err)
{
    if (v == 0) return 0;
    if (v < solo) {
        *err = "GR8_CP_STEP_ROWS=" + std::to_string(v) + ": at least " + std::to_string(solo) +
               " (the rows of a solo or prompt step past R: the verify width and --chunk)";
        return -1;
    }
    return int(std::min<int64_t>(v, std::max(full, solo)));
}
struct CpCand { uint64_t id; bool past; };   // a decoding session: its immutable id, its next round may run past R
struct CpPick {
    std::vector<bool> in;   // per candidate: in this round
    bool capped = false;    // a cycle round (the others sit it out); false = everyone, and the cycle resets
    bool fresh = false;     // capped: this round starts a new cycle
};
class CpRound {
    std::vector<uint64_t> served_;   // ids served in the current cycle
    bool served(uint64_t id) const { return std::find(served_.begin(), served_.end(), id) != served_.end(); }
public:
    // The round's sessions among cand (the decoding sessions that are not stalled, in request-id order). rows = n (0 =
    // off: everyone), per = the most rows one session's verify takes (mtp + 1), seg = GR8_CP_STEP_SEG. Pure: commit()
    // records it once the round really runs.
    CpPick pick(const std::vector<CpCand>& cand, uint32_t rows, uint32_t per, bool seg) const {
        CpPick p;
        p.in.assign(cand.size(), true);
        if (!rows || cand.empty()) return p;
        unsigned past = 0;
        for (const CpCand& c : cand) past += c.past;
        if (!past) return p;
        const bool mixed = past < cand.size();
        if (uint64_t(cand.size()) * per <= rows && !(seg && mixed)) return p;   // everyone fits in one step past R
        p.capped = true;
        std::vector<size_t> open;
        for (size_t i = 0; i < cand.size(); ++i) if (!served(cand[i].id)) open.push_back(i);
        if (open.empty()) { p.fresh = true; for (size_t i = 0; i < cand.size(); ++i) open.push_back(i); }
        p.in.assign(cand.size(), false);
        bool open_past = false;
        for (size_t i : open) open_past |= cand[i].past;
        if (!open_past) { for (size_t i : open) p.in[i] = true; return p; }   // a step below R: every open session
        const size_t most = std::max<size_t>(1, rows / std::max<uint32_t>(per, 1));
        size_t n = 0;
        for (size_t i : open) if (cand[i].past && n < most) { p.in[i] = true; ++n; }
        if (!seg) for (size_t i : open) if (!cand[i].past && n < most) { p.in[i] = true; ++n; }
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
