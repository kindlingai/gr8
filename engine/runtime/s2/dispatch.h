// Host Step planning shared by native adapter and CPU tests. Native backend owns
// CUDA layout/graphs/fabric; this layer never recreates Batch accept decisions.
#pragma once
#include "sessions.h"
#include "module.h"
#include "../prefix_reuse.h"
#include "../penalty.h"
#include "../ctx_index.h"
#include <functional>

namespace g6s2 {
struct Launch {
    s2::Step step;
    uint64_t sequence = 0;
    int solo = -1, context = 0;
    // first-step timer: request ids + E1/E2 event class per session,
    // so repeated requests are grouped by immutable request ordinal, not slot reuse.
    std::vector<uint64_t> requests; // parallel to step.sess
    std::vector<int> req_slots;     // raw slot ids, for join with request ids
    std::vector<int> events;        // per-session E1/E2 class (s2::Step::Event), parallel to step.sess
    s2::Step::Event event = s2::Step::ROUND; // strongest (lowest ordinal) class among the sessions
    std::vector<int> buckets, source_rows;          // per session: context bucket, first row in the last main step
    int bucket = 0;                                  // the step's bucket: the largest of its sessions'
    std::vector<Sampling> samples;
    std::vector<float> steering_scales;                 // per row; exact request FP32, target MAIN rows only
    std::vector<int> steering_slots;                    // per row; the request's loaded profile
    // the graph key: kind, M and the step's bucket. The sessions, source rows and steering scales are data (S2In)
    std::string graph_key;
};
struct DispatchHooks {
    std::function<int(bool draft, int context)> bucket;
    // At a synchronized boundary: hprev handoff, the solo session's tables, s2on toggles.
    std::function<void(int old_solo, int new_solo)> mode;
    // All model work synchronous on return (including the draft).
    std::function<std::vector<int>(const Launch&)> launch;
    // Complete prefill including both lanes AND draft, no late closure.
    std::function<void(const Slot&, const Payload&, uint32_t, uint32_t)> prefill;
    // optional: (row, target, k) -> row's logprob entry in the last main step: its sampled token, or (target) the
    // teacher-forced next prompt token; k top entries (requests with lp / plp)
    std::function<g6rt::LpRow(int, bool, int)> logprob;
    // optional, DFlash2 (s2_batch.h): does slot's session own the prefill's drafter taps; its drafter prompt context
    // (lowest position held); one drafter step with the session's sampling. Solo path only.
    std::function<bool(int slot)> dfl_ok;
    std::function<int(int slot)> dfl_start;
    std::function<s2::DflDrafts(const s2::DflReq&, const Sampling&)> dfl;
    // optional: row's log-probability in the last draft or chain step (StepOut::lp; s2_batch.h Batch::kstop)
    std::function<float(int row)> draft_lp;
    // optional, grammars (Dispatch::set_grammar): the host side of the mask arena the main head reads
    int* gm_on = nullptr; uint32_t* gm_bits = nullptr; int gm_words = 0;
    // optional, DS41 (ds41_prefill.h): in place of `prefill`, with the chunk's fabric step id: one id from the main sequence, the one
    // counter prefill and decode share on the fabric. The next MAIN step has sequence + 1.
    // LAST: tests build this struct positionally, so a new member goes at the end.
    std::function<void(const Slot&, const Payload&, uint32_t, uint32_t, uint64_t)> prefill_seq;
    // DS41 prefill pack: one main fabric sequence for a bounded ordered pack (append: tests use positional initialization).
    std::function<void(const std::vector<PrefillPart>&, uint64_t)> prefill_pack_seq;
    // optional, penalties (penalty.h, Dispatch::set_penalty): the host side of the row table (max_rows entries) and the
    // session table (MAX_SESSIONS entries) the main head reads
    g6rt::PenRow* pen_rows = nullptr; g6rt::PenSess* pen_sess = nullptr;
};
// The row of the step's j-th session that samples its next reply token: the one whose input is the last committed token,
// at position `last` (the prompt's length + the tokens emitted - 1). -1 when the step holds none (a prompt step before
// the prompt's last token). The session's rows after it hold drafts.
inline int grammar_row0(const s2::Step& sp, size_t j, int last) {
    for (int r = sp.row0[j]; r < sp.row0[j] + sp.rows[j]; ++r)
        if (sp.r[r].pos == last) return r;
    return -1;
}
// the graph key of a step of shape `shape` (s2::Step::key: kind and M) whose largest context bucket is `bucket`
inline std::string launch_key(const std::string& shape, int bucket) { return shape + ":b" + std::to_string(bucket); }
class Dispatch {
    const Sessions& sessions_;
    DispatchHooks hooks_;                                    // the std::function hooks a caller gave, if any
    std::unique_ptr<FnModule<DispatchHooks>> fn_;            // the table over them
    Module m_;
    uint32_t max_rows_;
    uint64_t main_sequence_ = 0, draft_sequence_ = 0;
    int solo_ = -1;
    bool mode_dirty_ = false;
    std::array<int, MAX_SESSIONS> source_{};
    std::array<bool, MAX_SESSIONS> have_main_{};
    std::array<g6rt::PrefixReuse, MAX_SESSIONS> rows_;   // prefix reuse: each session's written rows (main, draft, prefill)
    bool track_ = false;
    int* gm_on_ = nullptr; uint32_t* gm_bits_ = nullptr; int gm_words_ = 0;   // the grammar masks (set_grammar)
    g6rt::PenRow* pen_rows_ = nullptr; g6rt::PenSess* pen_sess_ = nullptr;    // the penalty tables (set_penalty)
    std::array<int, MAX_SESSIONS> pen_synced_{};                              // per slot: Session::out tokens the device counts hold
    std::array<bool, MAX_SESSIONS> pen_reset_{};                              // per slot: the counts are an earlier request's
    std::array<g6rt::CtxIndex, MAX_SESSIONS> ctx_;                             // per slot: prompt + reply indexed (the sequence penalty's thinking scope)
public:
    Dispatch(const Sessions& sessions, uint32_t rows, DispatchHooks h)
        : sessions_(sessions), hooks_(std::move(h)), fn_(new FnModule<DispatchHooks>(hooks_)), m_(fn_->ops()), max_rows_(rows) { init(); }
    // the steps of a module's table (ModuleOps). The table's module outlives the Dispatch
    Dispatch(const Sessions& sessions, uint32_t rows, const g6s2_module_ops& m) : sessions_(sessions), m_(m), max_rows_(rows) { init(); }
    Dispatch(const Dispatch&) = delete;
    Dispatch& operator=(const Dispatch&) = delete;
    void init() {
        const g6s2_module_ops& o = m_.o;
        require(max_rows_ && o.bucket && o.mode && o.launch && (o.prefill || o.prefill_seq), "incomplete dispatch hooks");
        if (o.gm_on) set_grammar(o.gm_on, o.gm_bits, o.gm_words);
        if (o.pen_rows) set_penalty(static_cast<g6rt::PenRow*>(o.pen_rows), static_cast<g6rt::PenSess*>(o.pen_sess));
    }
    // Call at every safe boundary after local EOS/commit, not inside a Step.
    // Resident means Prefill OR Decode; a 1-session prompt step within S>=2 is
    // NOT the S=1 fast path. No callback if mode unchanged.
    void synchronize_mode(const SchedulerView& scheduler) {
        int q = -1; unsigned live = 0;
        for (const auto& s : scheduler.slots()) if (s.phase == Phase::Prefill || s.phase == Phase::Decode) { q = int(s.key.slot); ++live; }
        const int next = live == 1 ? q : -1;
        if (next != solo_ || mode_dirty_) { m_.mode(solo_, next); solo_ = next; mode_dirty_ = false; }
    }
    void track_rows() { track_ = true; }
    // Grammars (grammar.h): before every step, on[r] = 1 where row r's sample follows its session's grammar, with the
    // row's mask at bits + r * words; the head reads both (rank.cuh launch_head_local). Draft steps are not masked.
    void set_grammar(int* on, uint32_t* bits, int words) { gm_on_ = on; gm_bits_ = bits; gm_words_ = words; }
    // Penalties and the think-end boost (penalty.h). Before every main step, rows[r] names row r's session, sampling
    // row and coefficients (or slot -1), with the think block's close while the session's block is open and the
    // thinking tokens committed, and sess[q] the tokens session q committed since the device counts were updated. The head reads both (rank.cuh
    // k_penalty, k_penalty_commit). A draft step has every row at slot -1. The counts live on the device, one slice
    // per slot. pen_synced_ tracks how many tokens they hold and pen_reset_ marks a slot with a new session.
    void set_penalty(g6rt::PenRow* rows, g6rt::PenSess* sess) { require(rows && sess, "penalty tables"); pen_rows_ = rows; pen_sess_ = sess; }
    // a new session on `slot`; skip: rows its run already holds (Grant::reuse)
    void begin(unsigned slot, int skip, const g6rt::CacheNamespace& ns = {}) {
        require(slot < MAX_SESSIONS, "invalid dispatch slot");
        rows_[slot].resume(skip, ns);
        pen_synced_[slot] = 0; pen_reset_[slot] = true; ctx_[slot].clear();
    }
    const g6rt::PrefixReuse& rows(unsigned slot) const { require(slot < MAX_SESSIONS, "invalid dispatch slot"); return rows_[slot]; }
    void release(unsigned slot) {
        require(slot < MAX_SESSIONS, "invalid dispatch slot");
        have_main_[slot] = false; source_[slot] = 0; rows_[slot].clear();
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
            const int bucket = m_.bucket(draft, sp.ctx[j]);
            require(bucket >= 0, "native bucket outside the captured ones");
            require(!draft || have_main_[q], "draft before session target step");
            l.buckets.push_back(bucket); l.source_rows.push_back(source_[q]);
            // per-session event; hand-built Steps may predate the field (default ROUND)
            l.requests.push_back(s.request); l.req_slots.push_back(q);
            l.events.push_back(j < sp.ev.size() ? sp.ev[j] : int(sp.event));
            l.bucket = std::max(l.bucket, bucket);
            for (int r = next_row; r < next_row + sp.rows[j]; ++r) {
                require(sp.r[r].sess == q && sp.r[r].pos >= 0 && sp.r[r].pos < sp.ctx[j] &&
                        (r == next_row || sp.r[r].pos == sp.r[r-1].pos + 1), "row/session position mismatch");
                l.samples.push_back(s.sampling);
                l.steering_scales.push_back(sp.kind == s2::Step::MAIN && s.payload.steering.enabled
                    ? g6rt::steering::bits_float(s.payload.steering.scale_bits) : 0.f);
                l.steering_slots.push_back(int(s.payload.steering.slot));
            }
            require(sp.r[next_row + sp.rows[j] - 1].pos + 1 == sp.ctx[j], "wrong session context");
            next_row += sp.rows[j]; l.context = std::max(l.context, sp.ctx[j]);
        }
        require(next_row == sp.M(), "unmapped native rows");
        l.graph_key = launch_key(sp.key(), l.bucket);
        if (gm_on_) grammar_masks(sp, draft);
        if (pen_rows_) penalty_rows(sp, draft);
        auto& seq = draft ? draft_sequence_ : main_sequence_;
        require(seq != UINT64_MAX, "fabric sequence exhausted"); l.sequence = ++seq;
        auto ids = m_.launch(l); require(ids.size() == sp.r.size(), "native output row mismatch");
        if (!draft) for (size_t j = 0; j < sp.sess.size(); ++j) {
            source_[sp.sess[j]] = sp.row0[j]; have_main_[sp.sess[j]] = true;
        }
        for (size_t j = 0; track_ && j < sp.sess.size(); ++j) {
            auto& w = rows_[sp.sess[j]];
            const int first = sp.r[sp.row0[j]].pos, n = sp.rows[j];
            if (!draft) {
                w.main_rows(first, n);
                if (g6mod::has(G6_CAP_DSPARK) && (sp.round < 0 || n == 1)) w.draft_rows(first, n, false);   // DSpark keys come from the main step's rows
            } else w.draft_rows(first, n, sp.kind == s2::Step::CHAIN);
        }
        return ids;
    }
    void grammar_masks(const s2::Step& sp, bool draft) {
        for (int r = 0; r < sp.M(); ++r) gm_on_[r] = 0;
        if (draft) return;
        for (size_t j = 0; j < sp.sess.size(); ++j) {
            const auto& s = sessions_.at(unsigned(sp.sess[j]));
            if (!s.grammar) continue;
            const int i0 = grammar_row0(sp, j, int(s.payload.prompt.size()) + s.emitter.emitted - 1);
            if (i0 < 0) continue;
            std::vector<int> drafts;
            for (int r = i0 + 1; r < sp.row0[j] + sp.rows[j]; ++r) drafts.push_back(sp.r[r].tok);
            g6rt::grammar_masks(*s.grammar, drafts.data(), int(drafts.size()), gm_on_ + i0, gm_bits_ + size_t(i0) * gm_words_);
        }
    }
    // the tokens of session `slot` the device counts do not hold yet (Session::out past pen_synced_)
    int penalty_new(const Session& s, unsigned slot) const {
        require(int(s.out.size()) == s.emitter.emitted && int(s.out.size()) >= pen_synced_[slot], "penalty output count");
        return int(s.out.size()) - pen_synced_[slot];
    }
    // the tables of a step. After the step the device counts hold every token its sessions had emitted before it: the
    // launch hook returns with the step done, and a session emits only between steps
    void penalty_rows(const s2::Step& sp, bool draft) {
        for (int r = 0; r < sp.M(); ++r) pen_rows_[r] = {-1, 0, 0, -1, 0.f, 0.f, 0.f, 0.f, 0, 0, 0.f, 0.f, {-1, -1, -1}, {0.f, 0.f, 0.f}};
        if (draft) return;
        for (size_t j = 0; j < sp.sess.size(); ++j) {
            const unsigned q = unsigned(sp.sess[j]); const auto& s = sessions_.at(q);
            if (!s.sampling.tracked()) continue;
            const int nnew = penalty_new(s, q);
            require(nnew <= g6rt::PEN_NEW_MAX, "too many tokens since the session's last penalty update");
            int first = grammar_row0(sp, j, int(s.payload.prompt.size()) + s.emitter.emitted - 1);
            if (first < 0) first = sp.M();   // a prompt step: no row samples a reply token
            auto& ps = pen_sess_[q]; ps.reset = pen_reset_[q]; ps.nnew = nnew;
            std::copy(s.out.end() - nnew, s.out.end(), ps.news);
            const auto& p = s.sampling;
            for (int r = sp.row0[j]; r < sp.row0[j] + sp.rows[j]; ++r)
                pen_rows_[r] = {int(q), first, r == sp.row0[j], s.think_open ? p.think_end : -1,
                                p.presence, p.frequency, p.tpresence, p.tfrequency,
                                s.think_tokens, p.boost_start, p.boost_per1k, p.boost_max, {-1, -1, -1}, {0.f, 0.f, 0.f}};
            if (first < sp.M()) sequence_rows(sp, j, first, s, q);
            pen_synced_[q] = int(s.out.size()); pen_reset_[q] = false;
        }
    }
    // The sequence penalty of session q's rows from `first` on (penalty.h pen_seq, ctx_index.h). Row first + k ends the
    // sequence prompt ++ reply ++ drafts of rows first + 1 .. first + k, as a step alone at its position would. The
    // thinking scope matches that sequence against itself while the row is inside the think block (no draft before it
    // is the close), the answer scope once it is not. The guard scope matches the reply's part against the avoided ids.
    void sequence_rows(const s2::Step& sp, size_t j, int first, const Session& s, unsigned q) {
        const auto& p = s.sampling;
        const bool thinking = p.seq_thinking() && s.think_open, answer = p.seq_answer(), guard = p.seq_guard();
        if (!thinking && !answer && !guard) return;
        const auto& prompt = s.payload.prompt;
        const int np = int(prompt.size()), no = int(s.out.size());
        const int end = sp.row0[j] + sp.rows[j];
        std::vector<int> tail; tail.reserve(end - first);   // the drafts: the tokens of rows first + 1 ..
        for (int r = first + 1; r < end; ++r) tail.push_back(sp.r[r].tok);
        const auto full = [&](int i) { return i < np ? prompt[i] : i < np + no ? s.out[i - np] : tail[i - np - no]; };
        const auto reply = [&](int i) { return i < no ? s.out[i] : tail[i - no]; };
        const auto ref = [&](int i) { return p.avoid[i]; };
        if (thinking || answer) ctx_[q].extend(full, np + no);
        for (int r = first; r < end; ++r) {
            const int k = r - first;
            auto& row = pen_rows_[r];
            // inside the block: the thinking scope, else the answer scope, over the whole context
            const bool in = s.think_open && g6rt::pen_in_think(p.think_end, tail.data(), k);
            const int scope = in ? 0 : 1;
            if (in ? thinking : answer) {
                const g6rt::CtxMatch m = ctx_[q].longest(full, np + no + k);
                const float pen = g6rt::pen_seq(m.len, p.seq[scope].mult, p.seq[scope].base, p.seq[scope].min, p.seq[scope].cap);
                if (pen > 0.f) { row.seq_tok[scope] = full(m.next); row.seq_pen[scope] = pen; }
            }
            if (guard) {
                const g6rt::CtxMatch m = g6rt::longest_in(reply, no + k, ref, int(p.avoid.size()));
                const float pen = g6rt::pen_seq(m.len, p.seq[2].mult, p.seq[2].base, p.seq[2].min, p.seq[2].cap);
                if (pen > 0.f) { row.seq_tok[2] = p.avoid[m.next]; row.seq_pen[2] = pen; }
            }
        }
    }
    bool dfl_ok(unsigned slot) const { return m_.o.dfl_ok && m_.dfl_ok(int(slot)); }
    int dfl_start(unsigned slot) { require(int(slot) == solo_, "DFlash2 outside the solo path"); return m_.dfl_start(int(slot)); }
    s2::DflDrafts dfl(const s2::DflReq& r) {
        require(r.sess == solo_ && have_main_[r.sess], "DFlash2 step outside the solo path or before a target step");
        return m_.dfl(r, sessions_.at(unsigned(r.sess)).sampling);
    }
    void prefill(const Slot& s, const Payload& p, uint32_t first, uint32_t rows) {
        require(rows && first <= p.prompt.size() && uint64_t(first) + rows < p.prompt.size() &&
                uint64_t(first) + rows <= uint64_t(s.run.pages) * PAGE, "invalid prefill quantum");
        if (m_.o.prefill_seq) {
            require(main_sequence_ != UINT64_MAX, "fabric sequence exhausted");
            m_.prefill_seq(s, p, first, rows, ++main_sequence_);
        } else m_.prefill(s, p, first, rows);
        if (track_) rows_[s.key.slot].prefill_rows(int(first), int(rows), true);
    }
    void prefill_pack(const std::vector<PrefillPart>& parts) {
        require(parts.size() >= 2 && parts.size() <= MAX_SESSIONS && m_.o.prefill_pack, "invalid/unwired prefill pack");
        require(main_sequence_ != UINT64_MAX, "fabric sequence exhausted");
        uint32_t total = 0;
        uint64_t last_id = 0;
        bool have_last = false; // native request ordinal zero is valid for the first member
        std::array<bool, MAX_SESSIONS> seen{};
        for (const auto& p : parts) {
            require(p.slot && p.payload, "null prefill pack member");
            require(p.slot->key.slot < MAX_SESSIONS && !seen[p.slot->key.slot] && (!have_last || p.slot->req.id > last_id) && p.slot->packable &&
                    p.slot->phase == Phase::Prefill && !p.slot->pack_tail && !p.payload->decode_prompt &&
                    p.payload->prompt.size() == p.slot->req.prompt_tokens && p.first == 0 && p.rows >= 1 && p.rows <= 64 &&
                    uint64_t(p.first) + p.rows + 1 == p.payload->prompt.size() &&
                    uint64_t(p.first) + p.rows <= uint64_t(p.slot->run.pages) * PAGE,
                    "invalid prefill pack segment");
            seen[p.slot->key.slot] = true; last_id = p.slot->req.id; have_last = true;
            total += p.rows;
        }
        require(total <= 512, "prefill pack exceeds 512 rows");
        m_.prefill_pack(parts, ++main_sequence_);
        if (track_) for (const auto& p : parts) rows_[p.slot->key.slot].prefill_rows(int(p.first), int(p.rows), true);
    }
};
// Direct mapping to fc2's native structs. Instantiate ONLY in DEC_S2 builds.
// No kernels or sampling arithmetic changed here.
template<class StepIn, class S2In>
void native_inputs(const Launch& l, StepIn& si, S2In& h) {
    si = {}; h = {};
    require(size_t(l.step.M()) <= sizeof(si.tok) / sizeof(si.tok[0]), "native StepIn row capacity too small");
    require(l.step.sess.size() <= sizeof(h.sess) / sizeof(h.sess[0]), "native S2In session capacity too small");
    si.step = l.sequence; si.M = l.step.M(); si.context = l.context;
    for (int r = 0; r < si.M; ++r) {
        const auto& x = l.step.r[r]; const auto& sample = l.samples[r];
        si.tok[r] = x.tok; si.pos[r] = x.pos; si.tgt[r] = x.tgt;
        si.steering_scale[r] = l.steering_scales.at(r);
        si.steering_slot[r] = l.steering_slots.at(r);
        // top list: main-step rows of a request with lp > 0 (sampled rows) or plp > 0 (prompt rows; rank.cuh step_inputs)
        const int topk = l.step.kind == s2::Step::MAIN && (x.tgt >= 0 ? sample.plp : sample.lp) > 0;
        si.srow[r] = {sample.inv_temp, sample.seed, 0u, x.spos, 0, sample.top_p, topk};
    }
    h.ns = int(l.step.sess.size()); h.cls = l.step.cls;
    for (int j = 0; j < h.ns; ++j) si.srow[l.step.row0[j] + l.step.rows[j] - 1].nodraft = 1;   // the next row is another session's
    for (int j = 0; j < h.ns; ++j) {
        h.sess[j] = l.step.sess[j]; h.row0[j] = l.step.row0[j]; h.rows[j] = l.step.rows[j];
        h.ctx[j] = l.step.ctx[j]; h.src0[j] = l.source_rows[j];
    }
}
} // namespace g6s2
