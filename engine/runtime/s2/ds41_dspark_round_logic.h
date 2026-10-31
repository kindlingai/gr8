// Host rules of the DSpark round, the part of s2/ds41_dspark_round.cuh that needs no CUDA: what the DRAFT step feeds the draft block
// (q, golden, acc), which prompt drafts are skipped, what the DRAFT / CHAIN steps hand back to s2::Batch, and the fail-closed checks.
// engine/runtime/s2/tests/batch_ds41_dspark_test.cpp runs it against the real s2::Batch with a fake engine.
//   DRAFT  rows are the accepted rows of the verify (or a prompt / forced chunk): Row{tok = next token, pos}, positions consecutive. The last
//          row is the round's end: q = its pos (last accepted position), golden = its tok (the token at q + 1), acc = M - 1 (the row index in
//          the main step that wrote the taps; q - acc is that step's row 0). Returns M ids; only the last is read: u[M - 1] = draft[0] = d[1].
//   CHAIN  one row, pos = q + j, tok = d[j] = draft[j - 1]; returns d[j + 1] = draft[j]. Nothing is launched.
//   A prompt chunk's draft before the last (Batch::prompt: round id -1, event ROUND) is skipped: the draft block reads only what the main steps
//   wrote (the taps of the latest main step, the draft layers' key rings that ST_DS_MAIN fills in every main step), so the tail draft is the
//   first one that can differ. Rounds and forced chunks (round id >= 0) and the tail (event TAIL) draft.
#pragma once
#include "../s2_batch.h"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
namespace g6s2 {

class Ds41RoundLogic {
public:
    static constexpr int BLOCK = 5;                        // the draft block: 5 tokens in one pass
    struct DraftIn { int q = 0, golden = 0, acc = 0; };

    explicit Ds41RoundLogic(int k, bool skip_prompt = true) : k_(k), skip_prompt_(skip_prompt) {
        need(k >= 1 && k <= BLOCK, "DSpark round: k must be 1..5");
    }

    // a new request at position 0: the cached drafts are stale
    void reset() { valid_ = false; q_ = -1; }

    // Step::DRAFT: false = skipped (nothing to run, answer with skipped()); true = run the block with `in`
    bool draft_in(const s2::Step& s, DraftIn& in) const {
        need(s.kind == s2::Step::DRAFT && s.sess.size() == 1 && s.M() >= 1, "DSpark draft step is not one session's rows");
        if (skip_prompt_ && s.round < 0 && s.event != s2::Step::TAIL) return false;
        const int M = s.M();
        for (int i = 0; i < M; ++i) need(s.r[size_t(i)].pos == s.r[0].pos + i, "DSpark draft rows are not consecutive positions");
        in.q = s.r.back().pos; in.golden = s.r.back().tok; in.acc = M - 1;
        return true;
    }
    std::vector<int> skipped(const s2::Step& s) const { return std::vector<int>(size_t(s.M()), 0); }

    // after the block ran: its 5 draft ids for the positions q + 2 .. q + 6
    std::vector<int> draft_out(const s2::Step& s, int q, const int* draft) {
        for (int i = 0; i < BLOCK; ++i) draft_[i] = draft[i];
        q_ = q; valid_ = true;
        std::vector<int> u(size_t(s.M()), 0);
        u[size_t(s.M() - 1)] = draft_[0];
        return u;
    }

    // Step::CHAIN
    std::vector<int> chain(const s2::Step& s) const {
        need(s.kind == s2::Step::CHAIN && s.sess.size() == 1 && s.M() == 1, "DSpark chain step without a draft");
        return std::vector<int>{chain_row(s.r[0])};
    }

    // SESS > 1 (one Ds41RoundLogic per S2 slot): session j of a DRAFT step of several sessions. Its rows are row0[j] .. row0[j] + rows[j] - 1
    // (rows = acc + 1, not uniform across sessions); the skip rule reads the session's own event (Step::ev, else the step's).
    bool draft_in_sess(const s2::Step& s, size_t j, DraftIn& in) const {
        need(s.kind == s2::Step::DRAFT && j < s.sess.size() && j < s.row0.size() && j < s.rows.size() && s.rows[j] >= 1 &&
             s.row0[j] >= 0 && s.row0[j] + s.rows[j] <= s.M(), "DSpark draft step session layout");
        const int ev = j < s.ev.size() ? s.ev[j] : int(s.event);
        if (skip_prompt_ && s.round < 0 && ev != s2::Step::TAIL) return false;
        const int r0 = s.row0[j], n = s.rows[j];
        for (int i = 0; i < n; ++i)
            need(s.r[size_t(r0 + i)].sess == s.sess[j] && s.r[size_t(r0 + i)].pos == s.r[size_t(r0)].pos + i, "DSpark draft rows are not consecutive positions");
        in.q = s.r[size_t(r0 + n - 1)].pos; in.golden = s.r[size_t(r0 + n - 1)].tok; in.acc = n - 1;
        return true;
    }
    // after the block ran for this session: its 5 draft ids; returns draft[0] (the id of its last DRAFT row)
    int draft_set(int q, const int* draft) {
        for (int i = 0; i < BLOCK; ++i) draft_[i] = draft[i];
        q_ = q; valid_ = true;
        return draft_[0];
    }
    // one CHAIN row of this session
    int chain_row(const s2::Row& r) const {
        need(valid_, "DSpark chain step without a draft");
        const int j = r.pos - q_;                          // chain step j: pos = q + j, tok = d[j] = draft[j - 1]
        need(j >= 1 && j < k_ && r.tok == draft_[j - 1], "DSpark chain step does not continue the cached draft");
        return draft_[j];
    }

    int draft_id(int i) const { return draft_[i]; }
    int q() const { return q_; }

private:
    static void need(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
    int k_;
    bool skip_prompt_;
    int draft_[BLOCK] = {};
    int q_ = -1;
    bool valid_ = false;
};

// The text of the draft fabric's error word (the message of "DSpark draft fabric error"). Layouts: kernels/glue.cuh wait_peers sets
// code << 24 | c << 8 | peer (0x61 all-reduce wait, 0x62 head-merge wait; peer = the rank whose tail never arrived); comms/fabric.cpp's proxy
// sets 0x70 << 24 | c (doorbell wait of 30 s ran out) or 0x71 << 24 (work completion error). The prefix stays "DSpark draft fabric error".
inline std::string draft_fabric_error_text(unsigned e) {
    char b[160];
    const unsigned code = e >> 24;
    if (code == 0x61u || code == 0x62u)
        std::snprintf(b, sizeof b, "DSpark draft fabric error 0x%08x: %s wait timed out, collective %u, peer rank %u", e,
                      code == 0x61u ? "layer exchange" : "head merge", (e >> 8) & 0xffu, e & 0xffu);
    else if (code == 0x70u)
        std::snprintf(b, sizeof b, "DSpark draft fabric error 0x%08x: proxy doorbell timeout, collective %u", e, e & 0xffffu);
    else if (code == 0x71u)
        std::snprintf(b, sizeof b, "DSpark draft fabric error 0x%08x: work completion error", e);
    else
        std::snprintf(b, sizeof b, "DSpark draft fabric error 0x%08x", e);
    return b;
}
}
