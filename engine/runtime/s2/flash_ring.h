// KDA rollback per session, independent of CUDA and the acceptance policy; the rules are
// engine/scheduler-rs/src/flash_ring.rs.
#pragma once
#include "../../scheduler-rs/g6sched.h"
#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>
namespace g6s2 {
struct FlashRing {
    int first = -1, rows = 0;
    bool ring = false;
    void reset() { first = -1; rows = 0; ring = false; }
    int source(int next) const {
        int src = 0;
        if (g6sch_ring_source(reinterpret_cast<const g6sch_ring*>(this), next, &src))
            throw std::runtime_error("Flash next main position outside previous step");
        return src;
    }
    void record(int pos, int n, bool verify) { first = pos; rows = n; ring = verify; }
};
static_assert(sizeof(FlashRing) == sizeof(g6sch_ring), "scheduler-rs layout");
// Every S2 session's KDA state slots (Model::kda_ring + (kda_sessions - 1) x kda_ring_rest of them): session 0 owns
// slots [0, first), session q > 0 owns [first + (q - 1) x per, first + q x per), each with a FlashRing of its own. per is
// the MTP ring (K + 1, or 1 without MTP); first is wider when session 0 is the one that may draft with DFlash2 (8 rows a
// round, Model::kda_ring), so only that session pays for the wide ring. In a main step a verify session's row r goes to
// slot base + r, other rows to base; the session's first row loads base + its ring's source, so sessions sharing a step
// never touch each other's state.
struct FlashSlots {
    int per = 1, first = 1;
    std::vector<FlashRing> rings;
    FlashSlots(int sessions, int per_session, int first_session = 0)
        : per(per_session), first(first_session ? first_session : per_session), rings(size_t(sessions)) {
        if (sessions < 1 || per_session < 1 || first < per) throw std::runtime_error("Flash KDA slots: no sessions or slots");
    }
    int sessions() const { return int(rings.size()); }
    int base(int q) const { check(q); return q ? first + (q - 1) * per : 0; }
    int width(int q) const { check(q); return q ? per : first; }
    int total() const { return first + (sessions() - 1) * per; }
    // a new session on q (its slots are reset on the device at the same time)
    void reset(int q) { at(q).reset(); }
    // session q's n rows from position pos in a main step: each row's slot into kslot[0..n), the slot the first row
    // loads from returned, the step recorded for the next one
    int rows(int q, int pos, int n, bool verify, int* kslot) {
        FlashRing& r = at(q);
        int src = 0;
        // the ring rule on the session's own slots (session 0 of a ring its width), then moved to its base
        const int rc = g6sch_ring_rows(reinterpret_cast<g6sch_ring*>(&r), width(q), 0, pos, n, verify, kslot, &src);
        if (rc == 2) throw std::runtime_error("Flash verify exceeds KDA ring");
        if (rc) throw std::runtime_error("Flash next main position outside previous step");
        for (int i = 0; i < n; ++i) kslot[i] += base(q);
        return src + base(q);
    }
    // the KDA inputs of a main step whose session j holds rows [row0[j], row0[j] + n[j]) (StepIn: kslot, kda_src for
    // row 0, krun for a later session's first row)
    template <class StepIn>
    void inputs(StepIn& si, const std::vector<int>& sess, const std::vector<int>& row0, const std::vector<int>& n,
                const std::vector<bool>& verify) {
        if (sess.empty() || row0.size() != sess.size() || n.size() != sess.size() || verify.size() != sess.size())
            throw std::runtime_error("Flash KDA inputs: bad step layout");
        for (size_t j = 0; j < sess.size(); ++j)
            if (row0[j] != (j ? row0[j - 1] + n[j - 1] : 0)) throw std::runtime_error("Flash KDA inputs: rows not in order");
        for (size_t j = 0; j < sess.size(); ++j) {
            const int src = rows(sess[j], si.pos[row0[j]], n[j], verify[j], si.kslot + row0[j]);
            if (j) si.krun[row0[j]] = src + 1;
            else si.kda_src = src;
        }
    }
    // the same from a main step's layout (s2::Step: sess, row0, rows) and its sessions' event classes: a session verifies
    // its rows unless they are its prompt tail (event == tail)
    template <class StepIn, class Step>
    void step(StepIn& si, const Step& sp, const std::vector<int>& events, int tail) {
        if (events.size() != sp.sess.size()) throw std::runtime_error("Flash KDA inputs: an event per session");
        std::vector<int> sess(sp.sess.begin(), sp.sess.end()), row0(sp.row0.begin(), sp.row0.end()),
            n(sp.rows.begin(), sp.rows.end());
        std::vector<bool> verify;
        for (int e : events) verify.push_back(e != tail);
        inputs(si, sess, row0, n, verify);
    }
private:
    void check(int q) const {
        if (q < 0 || q >= sessions()) throw std::runtime_error("Flash KDA slots: session out of range");
    }
    FlashRing& at(int q) { check(q); return rings[size_t(q)]; }
};
// The step shapes S2 can launch with several Flash sessions, each captured as a graph at every context bucket before
// serving (flash_native.cuh; the same set as the full model's NativeDispatch::capture_all, without the row classes,
// which Flash's bound layers do not have): main = prompt tails 1..chunk and rounds S x (k + 1), S = 1..sessions, k = 1..K
// with GR8_S2_ADAPT (else K); draft = 1..chunk and 1..sessions x (K + 1); chain = 1..sessions (K >= 2 only).
struct FlashShapes { std::set<int> mains, drafts, chains; };
inline FlashShapes flash_shapes(int sessions, int mtp, int chunk, bool adapt, int mx) {
    std::vector<uint8_t> m(size_t(std::max(mx, 0)) + 1), d(m.size()), c(m.size());
    const int rc = g6sch_flash_shapes(sessions, mtp, chunk, adapt, mx, m.data(), d.data(), c.data());
    if (rc == 1) throw std::runtime_error("Flash capture set: invalid sessions, mtp or chunk");
    if (rc) throw std::runtime_error("Flash capture set: sessions x (mtp + 1) rows exceed G6_MX");
    FlashShapes f;
    for (int M = 0; M <= mx; ++M) {
        if (m[size_t(M)]) f.mains.insert(M);
        if (d[size_t(M)]) f.drafts.insert(M);
        if (c[size_t(M)]) f.chains.insert(M);
    }
    return f;
}
}
