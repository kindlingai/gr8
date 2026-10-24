// KDA rollback per session, independent of CUDA and the acceptance policy.
#pragma once
#include <stdexcept>
#include <vector>
namespace g6s2 {
struct FlashRing {
    int first = -1, rows = 0;
    bool ring = false;
    void reset() { first = -1; rows = 0; ring = false; }
    int source(int next) const {
        if (first < 0) return 0;
        const int accepted = next - first;
        if (accepted < 1 || accepted > rows)
            throw std::runtime_error("Flash next main position outside previous step");
        return ring ? accepted - 1 : 0;
    }
    void record(int pos, int n, bool verify) { first = pos; rows = n; ring = verify; }
};
// Every S2 session's KDA state slots (Model::kda_sessions x kda_ring of them): session q owns slots
// [q * per, (q + 1) * per), per = the ring (K + 1, or 1 without MTP), with a FlashRing of its own. In a main step a
// verify session's row r goes to slot base + r, other rows to base; the session's first row loads base + its ring's
// source, so sessions sharing a step never touch each other's state.
struct FlashSlots {
    int per = 1;
    std::vector<FlashRing> rings;
    FlashSlots(int sessions, int per_session) : per(per_session), rings(size_t(sessions)) {
        if (sessions < 1 || per_session < 1) throw std::runtime_error("Flash KDA slots: no sessions or slots");
    }
    int sessions() const { return int(rings.size()); }
    int base(int q) const { check(q); return q * per; }
    // a new session on q (its slots are reset on the device at the same time)
    void reset(int q) { at(q).reset(); }
    // session q's n rows from position pos in a main step: each row's slot into kslot[0..n), the slot the first row
    // loads from returned, the step recorded for the next one
    int rows(int q, int pos, int n, bool verify, int* kslot) {
        FlashRing& r = at(q);
        if (n < 1 || (verify && n > per)) throw std::runtime_error("Flash verify exceeds KDA ring");
        const int b = q * per, src = b + r.source(pos);
        for (int i = 0; i < n; ++i) kslot[i] = b + (verify ? i : 0);
        r.record(pos, n, verify);
        return src;
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
private:
    void check(int q) const {
        if (q < 0 || q >= sessions()) throw std::runtime_error("Flash KDA slots: session out of range");
    }
    FlashRing& at(int q) { check(q); return rings[size_t(q)]; }
};
}
