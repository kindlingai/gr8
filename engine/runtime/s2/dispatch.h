// Host Step planning shared by native adapter and CPU tests. Native backend owns
// CUDA layout/graphs/fabric; this layer never recreates Batch accept decisions.
#pragma once
#include "sessions.h"
#include <functional>

namespace g6s2 {
struct Launch {
    s2::Step step;
    uint64_t sequence = 0;
    int solo = -1, base_page = 0, context = 0;
    // first-step timer: request ids + E1/E2 event class per session,
    // so repeated requests are grouped by immutable request ordinal, not slot reuse.
    std::vector<uint64_t> requests; // parallel to step.sess
    std::vector<int> req_slots;     // raw slot ids, for join with request ids
    std::vector<int> events;        // per-session E1/E2 class (s2::Step::Event), parallel to step.sess
    s2::Step::Event event = s2::Step::ROUND; // strongest (lowest ordinal) class among the sessions
    std::vector<int> bases, buckets, source_rows;   // per session: run base page, context bucket, first row in the last main step
    int bucket = 0;                                  // the step's bucket: the largest of its sessions'
    std::vector<Sampling> samples;
    // the graph key: kind, M and the step's bucket. Not the sessions, bases or source rows: those are data (S2In)
    std::string graph_key;
};
struct DispatchHooks {
    // Canonical native S2Model base, AFTER binding. No allocator lookup here.
    std::function<int(bool draft, int slot)> base;
    std::function<int(bool draft, int context)> bucket;
    // At a synchronized boundary: hprev handoff, solo tables B+i, s2on toggles.
    std::function<void(int old_solo, int new_solo)> mode;
    // All model work synchronous on return (including the draft).
    std::function<std::vector<int>(const Launch&)> launch;
    // Complete prefill including both lanes AND draft, no late closure.
    std::function<void(const Slot&, const Payload&, uint32_t, uint32_t)> prefill;
    // optional: (row, target, k) -> row's logprob entry in the last main step: its sampled token, or (target) the
    // teacher-forced next prompt token; k top entries (requests with lp / plp)
    std::function<g6rt::LpRow(int, bool, int)> logprob;
};
// the graph key of a step of shape `shape` (s2::Step::key: kind and M) whose largest context bucket is `bucket`
inline std::string launch_key(const std::string& shape, int bucket) { return shape + ":b" + std::to_string(bucket); }
class Dispatch {
    const Sessions& sessions_; DispatchHooks hooks_; uint32_t max_rows_;
    uint64_t main_sequence_ = 0, draft_sequence_ = 0;
    int solo_ = -1;
    bool mode_dirty_ = false;
    std::array<int, MAX_SESSIONS> source_{};
    std::array<bool, MAX_SESSIONS> have_main_{};
public:
    Dispatch(const Sessions& sessions, uint32_t rows, DispatchHooks h) : sessions_(sessions), hooks_(std::move(h)), max_rows_(rows) {
        require(rows && bool(hooks_.base) && bool(hooks_.bucket) && bool(hooks_.mode) && bool(hooks_.launch) && bool(hooks_.prefill), "incomplete dispatch hooks");
    }
    // Call at every safe boundary after local EOS/commit, not inside a Step.
    // Resident means Prefill OR Decode; a 1-session prompt step within S>=2 is
    // NOT the S=1 fast path. No callback if mode unchanged.
    void synchronize_mode(const Scheduler& scheduler) {
        int q = -1; unsigned live = 0;
        for (const auto& s : scheduler.slots()) if (s.phase == Phase::Prefill || s.phase == Phase::Decode) { q = int(s.key.slot); ++live; }
        const int next = live == 1 ? q : -1;
        if (next != solo_ || mode_dirty_) { hooks_.mode(solo_, next); solo_ = next; mode_dirty_ = false; }
    }
    void release(unsigned slot) {
        require(slot < MAX_SESSIONS, "invalid dispatch slot");
        have_main_[slot] = false; source_[slot] = 0;
        if (int(slot) == solo_) mode_dirty_ = true;
    }
    std::vector<int> run(const s2::Step& sp) {
        const bool draft = sp.kind != s2::Step::MAIN;
        require(sp.M() > 0 && unsigned(sp.M()) <= max_rows_ && !sp.sess.empty() && sp.sess.size() <= MAX_SESSIONS &&
                sp.row0.size() == sp.sess.size() && sp.rows.size() == sp.sess.size() && sp.ctx.size() == sp.sess.size(), "invalid native step layout");
        require(solo_ < 0 || (sp.sess.size() == 1 && sp.sess[0] == solo_), "solo launch contains another session");
        Launch l; l.step = sp; l.solo = solo_; l.event = sp.event;
        std::array<bool, MAX_SESSIONS> seen{};
        int next_row = 0;
        for (size_t j = 0; j < sp.sess.size(); ++j) {
            const int q = sp.sess[j]; require(q >= 0 && unsigned(q) < MAX_SESSIONS && !seen.at(q), "duplicate/invalid native session"); seen.at(q) = true;
            const auto& s = sessions_.at(unsigned(q));
            require(sp.row0[j] == next_row && sp.rows[j] > 0 && sp.rows[j] <= sp.M() - next_row && sp.ctx[j] > 0 &&
                    uint64_t(sp.ctx[j]) <= uint64_t(s.run.pages) * PAGE, "step outside session run");
            const int base = hooks_.base(draft, q), bucket = hooks_.bucket(draft, sp.ctx[j]);
            require(base >= 0 && unsigned(base) == s.run.base_page && bucket >= 0, "native/canonical binding mismatch");
            require(!draft || have_main_[q], "draft before session target step");
            l.bases.push_back(base); l.buckets.push_back(bucket); l.source_rows.push_back(source_[q]);
            // per-session event; hand-built Steps may predate the field (default ROUND)
            l.requests.push_back(s.request); l.req_slots.push_back(q);
            l.events.push_back(j < sp.ev.size() ? sp.ev[j] : int(sp.event));
            l.bucket = std::max(l.bucket, bucket);
            for (int r = next_row; r < next_row + sp.rows[j]; ++r) {
                require(sp.r[r].sess == q && sp.r[r].pos >= 0 && sp.r[r].pos < sp.ctx[j] &&
                        (r == next_row || sp.r[r].pos == sp.r[r-1].pos + 1), "row/session position mismatch");
                l.samples.push_back(s.sampling);
            }
            require(sp.r[next_row + sp.rows[j] - 1].pos + 1 == sp.ctx[j], "wrong session context");
            next_row += sp.rows[j]; l.context = std::max(l.context, sp.ctx[j]);
        }
        require(next_row == sp.M(), "unmapped native rows");
        l.graph_key = launch_key(sp.key(), l.bucket);
        auto& seq = draft ? draft_sequence_ : main_sequence_;
        require(seq != UINT64_MAX, "fabric sequence exhausted"); l.sequence = ++seq;
        if (solo_ >= 0) l.base_page = l.bases[0];
        auto ids = hooks_.launch(l); require(ids.size() == sp.r.size(), "native output row mismatch");
        if (!draft) for (size_t j = 0; j < sp.sess.size(); ++j) {
            source_[sp.sess[j]] = sp.row0[j]; have_main_[sp.sess[j]] = true;
        }
        return ids;
    }
    void prefill(const Slot& s, const Payload& p, uint32_t first, uint32_t rows) {
        require(rows && first <= p.prompt.size() && uint64_t(first) + rows < p.prompt.size() &&
                uint64_t(first) + rows <= uint64_t(s.run.pages) * PAGE, "invalid prefill quantum");
        hooks_.prefill(s, p, first, rows);
    }
};
// Direct mapping to fc2's native structs. Instantiate ONLY in DEC_S2 builds,
// whose StepIn has base_page. No kernels or sampling arithmetic changed here.
template<class StepIn, class S2In>
void native_inputs(const Launch& l, StepIn& si, S2In& h) {
    si = {}; h = {};
    require(size_t(l.step.M()) <= sizeof(si.tok) / sizeof(si.tok[0]), "native StepIn row capacity too small");
    require(l.step.sess.size() <= sizeof(h.sess) / sizeof(h.sess[0]), "native S2In session capacity too small");
    si.step = l.sequence; si.M = l.step.M(); si.context = l.context; si.base_page = l.base_page;
    for (int r = 0; r < si.M; ++r) {
        const auto& x = l.step.r[r]; const auto& sample = l.samples[r];
        si.tok[r] = x.tok; si.pos[r] = x.pos; si.tgt[r] = x.tgt;
        // top list: main-step rows of a request with lp > 0 (sampled rows) or plp > 0 (prompt rows; rank.cuh step_inputs)
        const int topk = l.step.kind == s2::Step::MAIN && (x.tgt >= 0 ? sample.plp : sample.lp) > 0;
        si.srow[r] = {sample.inv_temp, sample.seed, 0u, x.spos, 0, sample.top_p, topk};
    }
    h.ns = int(l.step.sess.size()); h.cls = l.step.cls;
    for (int j = 0; j < h.ns; ++j) {
        h.sess[j] = l.step.sess[j]; h.row0[j] = l.step.row0[j]; h.rows[j] = l.step.rows[j];
        h.ctx[j] = l.step.ctx[j]; h.base[j] = l.bases[j]; h.src0[j] = l.source_rows[j];
    }
}
} // namespace g6s2
