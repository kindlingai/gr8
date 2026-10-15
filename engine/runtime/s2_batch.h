// S2 batched decode driver, host only.
// Runs S sessions' MTP sequences with their rows sharing engine steps. Per session the step sequence is the one
// rank.cuh mtp_sequence runs for that session alone (same rows, positions, sample keys, contexts, in the same order):
//   prompt   each session's prompt steps run as that session's own step (one session per step, scheduler rule ii),
//            each followed by its first-form draft step, as in mtp_sequence
//   chain    K-1 chain steps, all sessions' rows in one step each (one row per session)
//   rounds   verify: ONE step, every active session's K+1 rows (M = S * (K+1) <= max rows); accept per session; emit
//            per session. First-form draft: ONE step, every surviving session's a+1 rows. Then the K-1 chain steps (all
//            survivors, one row each). No groups: the step count of a round does not depend on S or on the accepts.
//            The k1 split class of a main step is the class of a session's K+1 rows (as its solo step), not of M.
// A session leaves when its emit says finished (right after the verify, as mtp_sequence returns there) or at the start
// of the first round after it holds n_gen tokens (mtp_sequence runs that round's draft steps first, so this does too).
// Adaptive draft count (Batch::adapt): every session in a round verifies the same k <= K drafts. At the end of a
// round, before the chain steps, k for the next round is chosen from each session's accept rates (SpecLen, accept counts
// only, so every rank chooses the same k) and the S2 step costs (S2Cost): a verify row costs ~7 ms in a shared step, so
// at many sessions the K-th draft no longer pays. The main step's k1 class stays row_class(K+1), so a row's bits do not
// depend on k; only which drafts are verified changes.
// Sessions join at round boundaries (Batch::admit) and may be cancelled there (Batch::cancel); see class Batch.
// DFlash2 (Batch::dflash, Session::dflash): a session admitted while it is the only one runs the plain build's
// dflash_sequence (dflash.cuh) instead: each prompt step and each round feeds the drafter its target rows as context, a
// round verifies k <= 7 drafts (one SpecLen for every DFlash2 session, as the plain build keeps one per process). It also
// runs the MTP first-form draft over the same rows, so the MTP layer's cache and d[1] stay complete: when a second
// session arrives (demote) the K-1 chain steps hand it to the MTP rounds above. A demoted session stays on MTP.
// Drafter choice (Batch::dsel): with it on, a DFlash2 session drafts each round with DFlash2 or MTP, whichever is
// expected to commit more tokens per ms (SpecLen estimates from accept counts and the drafters' log-probabilities only, so
// every rank chooses the same). On MTP rounds the drafter still takes the accepted rows as context, so a switch back
// needs no catch-up; every DSel::probe-th of them also drafts, and its 7 drafts are scored against the tokens later
// rounds commit. On DFlash2 rounds the MTP first-form pass already drafts d[1], which scores MTP's first position. The
// estimates start afresh with each request.
// Insert rule (Session::ins, insert_rule.h): the rule's tokens take the place of the reply's after+1-th token, when none
// of the first after is `unless`. The verify's tokens after that one are dropped, as rejected drafts are. The inserted
// tokens are committed and emitted at once. Their own steps (forced, below) run at the start of the session's next
// round, after Driver::step has grown its run to the new length: one main step per chunk of them, teacher-forced as
// prompt steps are, each followed by its first-form draft, the last row sampling the token after them. Then the chain.
// Rows the verify wrote past the cut are rewritten later, as rejected drafts' rows are. The rule is a function of the
// committed tokens alone, so every rank cuts at the same token.
// Within a step the sessions are in ascending index and each session's rows are contiguous, in position order.
// No CUDA types: the engine adapter (m4sim / g6m4) turns a Step into its StepIn and per-session layout.
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "insert_rule.h"
#include "spec_len.h"

namespace s2
{

// the k1 split classes of bind::k1_split: 1 row / 2-3 rows / 4+ rows (chosen by the step's row count M)
inline int row_class(int rows) { return rows == 1 ? 1 : rows <= 3 ? 3 : 5; }

struct Row
{
    int sess, tok, pos, spos, tgt;    // spos: the position the row's sample is keyed to (SampleRow pos)
};
// one engine step. sess / row0 / rows / ctx: per session in the step (ctx = that session's last row position + 1)
struct Step
{
    enum Kind { MAIN, DRAFT, CHAIN } kind = MAIN;
    // first-step timer: what this step is, for E1/E2 grouping.
    // TAIL = the final PromptTail prompt MAIN; DECODE1 = the first Decode-kind round MAIN
    // after the tail; ROUND = every steady decode-round MAIN. DRAFT/CHAIN steps carry
    // the class of the MAIN they follow (TAIL/DECODE1/ROUND), so the readout pairs them.
    enum Event { TAIL, DECODE1, ROUND } event = ROUND;
    // first-step timer: per-session event class, parallel to sess.
    // A steady session in a mixed step keeps its own class; `event` = the strongest class in the step.
    std::vector<int> ev;
    int cls = 1;
    // G6_S2_ROUNDLOG: the Batch::round this step belongs to (-1 = not in a round, e.g. admit tail chains) and how many
    // sessions were active in that round (a draft or chain step holds only the survivors)
    long round = -1;
    int round_active = 0;
    std::vector<int> sess, row0, rows, ctx;
    std::vector<Row> r;
    int M() const { return (int)r.size(); }
    // the step's shape: kind and row count (the class follows M). The graph key adds the context bucket; the session
    // layout is data (S2In), not part of the key
    std::string key() const
    {   // a main step's k1 class is part of its graph: a 4-row prompt tail (class 5) and a 2-session round of 2 rows (class 3)
        if (kind == MAIN) return "m" + std::to_string(M()) + "c" + std::to_string(cls);
        return std::string(kind == DRAFT ? "d" : "c") + std::to_string(M());
    }
};

struct Session
{
    std::vector<int> prompt;
    int first = 0;                    // prompt tokens before this index were taken by a prefill path
    int chunk = 4;                    // rows per prompt step
    int n_gen = 16;
    bool dflash = false;              // the DFlash2 drafter while the session is alone (Batch::dflash)
    g6rt::InsertRule ins;             // off unless the request has one
};
struct Stats
{
    long rounds = 0, tokens = 0, drafted = 0;
    long mtp_rounds = 0, switches = 0;    // drafter choice: a DFlash2 session's rounds on MTP, drafter switches
    long inserted = 0;                // tokens the insert rule put in
    std::vector<long> acc;            // [K+1] rounds by accepted drafts (K = 7 for a DFlash2 session)
};
// one DFlash2 drafter step of session sess (dflash.cuh DflIn): context = its target rows at positions pos0 .. pos0 + rows - 1,
// lo = the lowest position the drafter holds; query: also draft the block after them, bonus token tok0
constexpr int DFL_B = 8;              // = dfl::B: the bonus row and 7 drafts
struct DflReq { int sess, rows, pos0, tok0, lo; bool query; };
struct DflDrafts { int tok[DFL_B]; float lp[DFL_B]; };

// run(step) -> the sampled token of every row of the step (step.M() ids), for MAIN, DRAFT and CHAIN steps
// round(sess, round, base, a, K): per-session round line (mtp_sequence prints "round N base B accepted a of K")
// prompt(sess, M, pos0): after each prompt MAIN step of a session (rows pos0 .. pos0 + M - 1, row r at pos0 + r)
// emit(sess, toks, k, row) -> true = the session is finished (stop id or max tokens); row: toks[0]'s row in the last MAIN step
// returns per session the generated tokens (as mtp_sequence returns them)
struct Hooks
{
    std::function<std::vector<int>(const Step&)> run;
    std::function<void(int, long, int, int, int)> round;
    std::function<void(int, int, int)> prompt;
    std::function<bool(int, const int*, int, int)> emit;
    // DFlash2 (optional): one drafter step; and before a DFlash2 session's prompt steps, the drafter's prompt context
    // (the prefill's taps), returning the lowest position it holds
    std::function<DflDrafts(const DflReq&)> dfl;
    std::function<int(int)> dfl_start;
};

// Persistent driver state for the scheduler: stable slots 0..S2N_SLOTS-1, all calls at round boundaries.
//   admit(slots, sessions)  prompt (tail) steps of each new session as its own steps (+ first-form drafts), in the order
//                           given, then one batched chain over those still live. A session finished by emit during its
//                           prompt is finished at once (finished(q), result(q)).
//   round(order)            one round over the sessions of `order` that are active (admitted, not finished, fewer than
//                           n_gen tokens), rows in that order; false = none active (nothing ran). An active session
//                           left out of `order` sits the round out: its state does not change, so its later rounds,
//                           rows and tokens are the ones it would have had (the driver does this when its pages
//                           cannot grow)
//   cancel(q)               the session leaves now (result(q) = what it committed); its slot can be admitted again
//   active(q) / finished(q) / result(q) / stats(q)
// Acceptance and draft state stay here; the scheduler keeps generations and pages. mtp_batch below = admit(all) and
// rounds in slot order until none is active.
constexpr int S2N_SLOTS = 8;
// G6_S2_ADAPT=1 (every rank, read at boot): choose each round's draft count. Off by default: on code and prose at 1-8
// sessions it measured within 0-3% of K drafts every round (the K-th draft is still accepted ~60% of the time), and
// its extra main-step graphs take ~1 GiB
inline bool adapt_env() { const char* e = getenv("G6_S2_ADAPT"); return e && atoi(e) != 0; }
// Step times at TP4, 2200 MHz, contexts < 2K (G6_DEC_EVPROF / G6_S2_ROUNDLOG), ms
struct S2Cost
{
    double main0 = 44, main_row = 7;            // verify step of M rows: main0 + main_row * M
    double draft0 = 2.1, draft_sess = 0.25;     // first-form draft step of S sessions' accepted rows
    double chain0 = 2.0, chain_sess = 0.15;     // chain step, one row per session
    double round(int S, int k) const
    {
        return main0 + main_row * S * (k + 1) + draft0 + draft_sess * S + (k - 1) * (chain0 + chain_sess * S);
    }
};
// G6_S2_DSEL (every rank, read at boot): drafter choice for DFlash2 sessions (Batch::dsel), on unless 0. RigMark TP4:
// prose 28.8 -> 30.1 tok/s, code 48.9 -> 49.0, C4 62.7 -> 63.8 (margin 0.15, dwell 8)
inline bool dsel_env() { const char* e = getenv("G6_S2_DSEL"); return !e || atoi(e) != 0; }
struct DSel
{
    bool on = false;
    int dwell = 8;                    // rounds on a drafter before the next switch (4 switched code to MTP too often)
    double margin = 0.15;             // switch when the other drafter's tokens per ms is this much higher (0.05: code -2.7%)
    int probe = 4;                    // every probe-th MTP round the drafter step also drafts (scored next round)
    g6rt::SpecLen d, m;               // DFlash2 and MTP estimates; draft_round / draft_each hold each round's drafting ms
};
class Batch
{
public:
    Batch(int K, int max_rows, Hooks h) : K_(K), max_rows_(max_rows), h_(std::move(h)), st_(S2N_SLOTS), stats_(S2N_SLOTS) {}

    void admit(const std::vector<int>& qs, const std::vector<Session>& ss)
    {
        if (qs.size() != ss.size()) throw std::runtime_error("s2: admit: slots and sessions differ in count");
        std::vector<int> live;
        for (size_t k = 0; k < qs.size(); ++k)
        {
            const int q = qs[k];
            if (q < 0 || q >= S2N_SLOTS || st_[q].on) throw std::runtime_error("s2: admit: slot " + std::to_string(q) + " is out of range or in use");
            if (prompt(q, ss[k]) && !st_[q].dfl && st_[q].force < 0) live.push_back(q);
        }
        std::vector<int> all;
        for (int q = 0; q < S2N_SLOTS; ++q) if (active(q)) all.push_back(q);
        { std::vector<int> evs; for (size_t k = 0; k < live.size(); ++k) evs.push_back(Step::TAIL);
          chain(live, evs, choose(all)); } // first-step timer: post-admit chains follow the tail steps
    }
    bool active(int q) const { const St& x = st_[q]; return x.on && !x.done && (int)x.tokens.size() - x.n < x.s.n_gen; }
    // the session's tokens so far, prompt included: its next round writes rows from length - 1 on
    int length(int q) const { return (int)st_[q].tokens.size(); }
    bool finished(int q) const { return st_[q].on && !active(q); }
    void cancel(int q) { St& x = st_[q]; x.done = true; x.on = false; }
    std::vector<int> result(int q) const
    {
        const St& x = st_[q];
        const size_t end = std::min(x.tokens.size(), (size_t)x.n + x.s.n_gen);
        return std::vector<int>(x.tokens.begin() + std::min((size_t)x.n, end), x.tokens.begin() + end);
    }
    const Stats& stats(int q) const { return stats_[q]; }
    // choose each round's draft count (k <= K); off: every round verifies K drafts
    void adapt(bool on) { adapt_ = on; }
    // DFlash2 sessions verify up to kv of the 7 drafts; sl: the plain build's --spec-adaptive policy (sl.on = false: kv)
    void dflash(int kv, const g6rt::SpecLen& sl) { dkv_ = kv; dsl_ = sl; }
    void dsel(const DSel& d) { sel_ = d; }
    // a second session is resident: every DFlash2 session goes to MTP (the K-1 chain steps), before the next round
    void demote()
    {
        for (int q = 0; q < S2N_SLOTS; ++q)
            if (st_[q].dfl && active(q))
            {
                st_[q].dfl = false;
                // on MTP already: d[2..K] are drafted. Inserted tokens waiting for their steps: forced() chains them
                if (!st_[q].mtp && st_[q].force < 0) chain({q}, {int(Step::ROUND)}, K_);
                st_[q].mtp = false;
            }
    }
    // the slot can be admitted again (after finished(q) or cancel(q))
    void release(int q) { st_[q].on = false; }

    // one decode round of every active session: one main step (all rows), one first-form draft step, K-1 chain steps.
    // A session's rows and steps do not depend on which other sessions are in the round (row invariance, A1).
    bool round(const std::vector<int>& order)
    {
        std::vector<int> act;
        for (int q : order) if (active(q)) act.push_back(q);
        if (act.empty()) return false;
        if (act.size() * (K_ + 1) > (size_t)max_rows_)
            throw std::runtime_error("s2: " + std::to_string(act.size()) + " sessions x " + std::to_string(K_ + 1) + " rows > max rows " + std::to_string(max_rows_));
        cur_round_ = round_no_++; cur_active_ = (int)act.size();
        for (int q : act) if (st_[q].force >= 0) forced(q);
        act.erase(std::remove_if(act.begin(), act.end(), [&](int q) { return !active(q); }), act.end());
        if (act.empty()) { cur_round_ = -1; cur_active_ = 0; return true; }
        if (act.size() == 1 && st_[act[0]].dfl) round_dfl(act[0]);
        else { demote(); round_all(act); }
        cur_round_ = -1; cur_active_ = 0;
        return true;
    }

private:
    void round_all(const std::vector<int>& act)
    {
        Step v;
        v.kind = Step::MAIN;
        v.round = cur_round_; v.round_active = cur_active_;
        // first-step timer: E1/E2 grouping. The first decode-round MAIN of a
        // session is DECODE1; every later round MAIN is ROUND. A step's label is the
        // LOWEST class among its sessions (an unlabeled steady session never masks another's E2).
        v.event = Step::ROUND;
        for (int q : act) if (st_[q].rounds == 0) { v.event = Step::DECODE1; break; } // strongest class in the step
        std::vector<int> base(S2N_SLOTS, 0);
        int k = K_;                                               // drafts verified: the fewest any session holds
        for (int q : act) k = std::min(k, st_[q].k);
        for (int q : act)
        {
            St& x = st_[q];
            base[q] = (int)x.tokens.size() - 1;
            std::vector<Row> rows;
            for (int r = 0; r <= k; ++r)
                rows.push_back(Row{q, r ? x.d[r] : x.tokens[base[q]], base[q] + r, base[q] + r + 1, -1});
            add_rows(v, q, rows, x.rounds == 0 ? Step::DECODE1 : Step::ROUND); // first-step timer: per-session E2
        }
        v.cls = row_class(K_ + 1);    // a session's own rows (K+1), as its solo step: the k1 split, so a row's bits do not depend on the batch
        check(v);
        const std::vector<int> t = h_.run(v);
        std::vector<int> on, acc(S2N_SLOTS, 0);
        for (size_t i = 0; i < v.sess.size(); ++i)
        {
            const int q = v.sess[i];
            St& x = st_[q];
            const int* so = &t[v.row0[i]];
            int a = 0;
            while (a < k && x.d[a + 1] == so[a]) ++a;
            int kept;
            const int c = commit(q, so, a + 1, &kept);
            sl_[q].observe(k, a, nullptr, 0);
            if (h_.round) h_.round(q, stats_[q].rounds, base[q], a, k);
            ++stats_[q].rounds; stats_[q].tokens += a + 1; ++stats_[q].acc[a]; stats_[q].drafted += k;
            ++st_[q].rounds;
            acc[q] = std::min(kept, a);   // the draft step's rows: those whose input the session committed
            if (h_.emit && h_.emit(q, &x.tokens[base[q] + 1], c, v.row0[i])) { x.done = true; continue; }
            if (x.force < 0) on.push_back(q);
        }
        {                                                         // first-form draft: one step, rows a+1 of each survivor
            Step d;
            d.kind = Step::DRAFT; d.event = v.event;
            d.round = cur_round_; d.round_active = cur_active_;
            for (size_t i = 0; i < v.sess.size(); ++i)
            {
                const int q = v.sess[i];
                if (st_[q].done) continue;
                add_rows(d, q, draft_rows(q, acc[q] + 1, base[q], &st_[q].tokens[base[q] + 1]), v.ev[i]);
            }
            if (!d.sess.empty())
            {
                d.cls = row_class(d.M());
                check(d);
                const std::vector<int> u = h_.run(d);
                for (size_t i = 0; i < d.sess.size(); ++i) st_[d.sess[i]].d[1] = u[d.row0[i] + d.rows[i] - 1];
            }
        }
        { // first-step timer: chain carries each survivor's own round class
            std::vector<int> evs;
            for (int q : on) { size_t i = 0; while (i < v.sess.size() && v.sess[i] != q) ++i;
                evs.push_back(i < v.sess.size() ? v.ev[i] : int(v.event)); }
            chain(on, evs, choose(on));
        }
    }

    // one round of solo DFlash2 session q (dflash.cuh dflash_sequence's round): verify k + 1 rows, accept, emit; then the
    // MTP first-form pass and the drafter step over the a + 1 accepted rows. On MTP (x.mtp, drafter choice) the round
    // verifies the K MTP drafts, the drafter step takes the rows as context only, and the K-1 chain steps follow.
    void round_dfl(int q)
    {
        St& x = st_[q];
        const int base = (int)x.tokens.size() - 1;
        const bool m = x.mtp;
        double pe = 0;
        const int k = m ? x.k : dsl_.on ? dsl_.choose(x.dl, dkv_, dkv_, &pe) : dkv_;
        const int* dr = m ? x.d.data() : x.dd;
        Step v;
        v.kind = Step::MAIN;
        v.round = cur_round_; v.round_active = cur_active_;
        v.event = x.rounds == 0 ? Step::DECODE1 : Step::ROUND;
        std::vector<Row> rows;
        for (int r = 0; r <= k; ++r) rows.push_back(Row{q, r ? dr[r] : x.tokens[base], base + r, base + r + 1, -1});
        add_rows(v, q, rows, v.event);
        v.cls = row_class(m ? K_ + 1 : k + 1);
        check(v);
        const std::vector<int> t = h_.run(v);
        int a = 0;
        while (a < k && dr[a + 1] == t[a]) ++a;
        int kept;
        const int c = commit(q, t.data(), a + 1, &kept);
        if (!m && dsl_.on) dsl_.observe(k, a, x.dl, pe);
        if (sel_.on) score(x, k, a, t.data());
        if (h_.round) h_.round(q, stats_[q].rounds, base, a, k);
        ++stats_[q].rounds; stats_[q].tokens += a + 1; ++stats_[q].acc[a]; stats_[q].drafted += k;
        ++x.rounds;
        if (m) ++stats_[q].mtp_rounds;
        if (h_.emit && h_.emit(q, &x.tokens[base + 1], c, 0)) { x.done = true; return; }
        Step d;
        d.kind = Step::DRAFT; d.event = v.event;
        d.round = cur_round_; d.round_active = cur_active_;
        const int dn = std::min(kept, a) + 1;   // rows whose input the session committed
        add_rows(d, q, draft_rows(q, dn, base, &x.tokens[base + 1]), v.event);
        d.cls = row_class(d.M());
        check(d);
        x.d[1] = h_.run(d)[dn - 1];
        if (x.force >= 0)                 // the insert cut this round: the drafter takes the rows as context, forced() drafts
        {
            drafts(q, DflReq{q, dn, base, 0, x.lo, false});
            x.pb = -1;
            return;
        }
        const bool nm = sel_.on && next_mtp(x);
        if (nm != m) { x.since = 0; ++stats_[q].switches; }
        ++x.since;
        x.mtp = nm;
        const bool probe = nm && x.pb < 0 && ++x.mn % sel_.probe == 0;
        drafts(q, DflReq{q, a + 1, base, t[a], x.lo, !nm || probe});
        if (probe) { x.pb = base + a + 1; std::copy(x.dd, x.dd + DFL_B, x.pd); std::copy(x.dl, x.dl + DFL_B, x.pl); }
        if (!nm) x.pb = -1;
        if (nm) chain({q}, {int(v.event)}, K_);
    }
    void drafts(int q, const DflReq& r)
    {
        const DflDrafts o = h_.dfl(r);
        if (!r.query) return;
        for (int i = 1; i < DFL_B; ++i) { st_[q].dd[i] = o.tok[i]; st_[q].dl[i] = o.lp[i]; }
    }

    struct St
    {
        Session s;
        std::vector<int> tokens, d;
        int n = 0;
        bool on = false, done = false;
        int rounds = 0; // first-step timer: per-session decode rounds; round 1 MAIN = DECODE1
        int k = 0;      // drafts held for the next verify (d[1..k])
        bool dfl = false;                 // on the DFlash2 drafter (Session::dflash, until demote)
        bool mtp = false;                 // drafter choice: this round drafts with MTP (d[1..K] held, x.k = K)
        int pb = -1;                      // drafter choice: a pending probe's base position (-1: none); pd/pl its drafts
        int pd[DFL_B] = {};
        float pl[DFL_B] = {};
        int since = 0, mn = 0;            // drafter choice: rounds on the current drafter; MTP rounds (probe count)
        int lo = 0;                       // DFlash2: the lowest position the drafter holds context for
        bool armed = false;               // the insert rule (s.ins) may still fire
        int force = -1;                   // inserted tokens from this position on wait for their steps (forced); -1: none
        int dd[DFL_B] = {};               // DFlash2 drafts d1..d7 and their log-probabilities under the drafter
        float dl[DFL_B] = {};
    };
    // drafter choice: teach the estimates what this round's tokens t[0..a] say about each drafter. DFlash2 rounds score
    // MTP's first draft (x.d[1], drafted for position base + 1 = t[0]); MTP rounds score a probe's DFlash2 drafts on the
    // positions the round committed
    void score(St& x, int k, int a, const int* t)
    {
        if (!x.mtp)
        {
            sel_.d.observe(k, a, x.dl, 0);
            sel_.m.observe(1, x.d[1] == t[0] ? 1 : 0, nullptr, 0);
            return;
        }
        sel_.m.observe(k, a, nullptr, 0);
        probe_score(x);
    }
    // a probe's drafts pd[j] are for positions pb + j; score them once a mismatch or all DFL_B - 1 are committed
    void probe_score(St& x)
    {
        if (x.pb < 0) return;
        const int have = (int)x.tokens.size() - 1 - x.pb, n = DFL_B - 1;
        int b = 0;
        while (b < n && b < have && x.pd[b + 1] == x.tokens[x.pb + b + 1]) ++b;
        if (b < n && b == have) return;             // every committed position matched so far: wait for more
        sel_.d.observe(std::min(b + 1, n), b, x.pl, 0);
        x.pb = -1;
    }
    // the drafter for the next round: switch after dwell rounds when the other one's tokens per ms is margin higher
    bool next_mtp(const St& x) const
    {
        if (x.since < sel_.dwell) return x.mtp;
        double rd = 0;
        if (dsl_.on) { for (int k : dsl_.levels) rd = std::max(rd, sel_.d.ratio(k, nullptr, k)); }
        else rd = sel_.d.ratio(dkv_, nullptr, dkv_);
        const double rm = sel_.m.ratio(K_, nullptr, K_);
        return x.mtp ? rm * (1 + sel_.margin) >= rd : rm > rd * (1 + sel_.margin);
    }
    static g6rt::SpecLen fresh(const g6rt::SpecLen& o)
    {
        g6rt::SpecLen f;
        f.on = o.on; f.levels = o.levels; f.draft_round = o.draft_round; f.draft_each = o.draft_each;
        return f;
    }
    // the draft count for the next round of sessions qs: the k with the most expected tokens per ms of the round
    int choose(const std::vector<int>& qs) const
    {
        if (!adapt_ || qs.empty()) return K_;
        int best = K_;
        double br = -1;
        for (int k = 1; k <= K_; ++k)
        {
            double e = 0;
            for (int q : qs)
            {
                double s = 1;
                e += 1;
                for (int j = 1; j <= k; ++j) { s *= sl_[q].rate(j); e += s; }
            }
            const double r = e / cost_.round((int)qs.size(), k);
            if (r > br * (1 + 1e-9)) { br = r; best = k; }
        }
        return best;
    }
    int K_, max_rows_;
    bool adapt_ = false;
    int dkv_ = DFL_B - 1;
    g6rt::SpecLen dsl_;
    DSel sel_;
    S2Cost cost_;
    std::vector<g6rt::SpecLen> sl_ = std::vector<g6rt::SpecLen>(S2N_SLOTS);
    long round_no_ = 0, cur_round_ = -1; // G6_S2_ROUNDLOG: id of the round in progress and its active-session count
    int cur_active_ = 0;
    Hooks h_;
    std::vector<St> st_;
    std::vector<Stats> stats_;

    static void add_rows(Step& s, int q, const std::vector<Row>& rows, int ev)
    {
        s.sess.push_back(q);
        s.row0.push_back(s.M());
        s.rows.push_back((int)rows.size());
        s.ctx.push_back(rows.back().pos + 1);
        s.r.insert(s.r.end(), rows.begin(), rows.end());
        s.ev.push_back(ev); // first-step timer: per-session event class
    }
    void check(const Step& s) const
    {
        if (s.M() > max_rows_) throw std::runtime_error("s2: step of " + std::to_string(s.M()) + " rows > max rows " + std::to_string(max_rows_));
    }
    // draft rows of session q at positions pos0.. with next tokens next[0..M-1] (rank.cuh draft_inputs)
    static std::vector<Row> draft_rows(int q, int M, int pos0, const int* next)
    {
        std::vector<Row> v;
        for (int r = 0; r < M; ++r) v.push_back(Row{q, next[r], pos0 + r, pos0 + r + 2, -1});
        return v;
    }
    // chain steps for the sessions in qs (each: base = its tokens.size() - 1; d[1] set; fills d[2..k])
    void chain(const std::vector<int>& qs, const std::vector<int>& evs, int k)
    {
        for (int q : qs) st_[q].k = k;
        // first-step timer: each session's chain carries THAT session's round class (evs, parallel to qs).
        for (int j = 1; j < k && !qs.empty(); ++j)
        {
            Step s;
            s.kind = Step::CHAIN;
            s.round = cur_round_; s.round_active = cur_active_;
            for (size_t k = 0; k < qs.size(); ++k)
            {
                const int q = qs[k];
                const int base = (int)st_[q].tokens.size() - 1;
                add_rows(s, q, draft_rows(q, 1, base - 1 + j, &st_[q].d[j]), evs[k]);
            }
            s.cls = row_class(s.M());
            check(s);
            const std::vector<int> t = h_.run(s);
            for (size_t i = 0; i < s.sess.size(); ++i) st_[s.sess[i]].d[j + 1] = t[s.row0[i]];
        }
    }
    // the prompt steps of session q as its own steps (rank.cuh mtp_sequence, prompt loop); false = finished by emit
    bool prompt(int q, const Session& p)
    {
        St& x = st_[q];
        x = St();
        x.s = p; x.on = true;
        x.tokens = p.prompt;
        x.n = (int)p.prompt.size();
        x.d.assign(K_ + 1, 0);
        sl_[q] = g6rt::SpecLen();
        stats_[q] = Stats();
        x.dfl = p.dflash;
        x.armed = p.ins.on();
        if (x.dfl && sel_.on)             // drafter choice learns each request afresh: the last request's text says little
        {
            sel_.d = fresh(sel_.d);
            sel_.m = fresh(sel_.m);
        }
        if (x.dfl && !(h_.dfl && h_.dfl_start)) throw std::runtime_error("s2: DFlash2 session without drafter hooks");
        stats_[q].acc.assign(std::max(K_, x.dfl ? DFL_B - 1 : 0) + 1, 0);
        for (float& v : x.dl) v = NAN;
        x.lo = x.dfl ? h_.dfl_start(q) : 0;
        const int n = x.n, PS = (n - p.first + p.chunk - 1) / p.chunk;
        for (int s = 0; s < PS; ++s)
        {
            const int left = n - p.first - s * p.chunk, M = left < p.chunk ? left : p.chunk;
            std::vector<Row> rows;
            for (int r = 0; r < M; ++r)
            {
                const int pos = p.first + s * p.chunk + r;
                rows.push_back(Row{q, p.prompt[pos], pos, pos + 1, pos + 1 < n ? p.prompt[pos + 1] : -1});
            }
            Step m;
            m.kind = Step::MAIN; m.cls = row_class(M); m.event = s == PS - 1 ? Step::TAIL : Step::ROUND; // single session: step event = session event
            add_rows(m, q, rows, s == PS - 1 ? Step::TAIL : Step::ROUND);
            check(m);
            const std::vector<int> t = h_.run(m);
            if (h_.prompt) h_.prompt(q, M, rows[0].pos);
            if (s == PS - 1)
            {
                int kept;
                const int c = commit(q, &t[M - 1], 1, &kept);
                if (h_.emit && h_.emit(q, &x.tokens[n], c, M - 1)) { x.done = true; return false; }
            }
            std::vector<int> next(M);
            for (int r = 0; r < M; ++r) next[r] = x.tokens[rows[r].pos + 1];
            Step d;
            d.kind = Step::DRAFT; d.cls = row_class(M); d.event = m.event;
            add_rows(d, q, draft_rows(q, M, rows[0].pos, next.data()), m.ev.back());
            check(d);
            const std::vector<int> u = h_.run(d);
            x.d[1] = u[M - 1];
            const bool query = s == PS - 1 && x.force < 0;
            if (x.dfl) drafts(q, DflReq{q, M, rows[0].pos, query ? t[M - 1] : 0, x.lo, query});
        }
        return true;
    }
    // the c tokens t a step sampled for session q (the verify's accepted ones and its bonus), committed with the insert
    // rule applied. Returns how many tokens were committed, inserted ones included. kept: how many of t
    int commit(int q, const int* t, int c, int* kept)
    {
        St& x = st_[q];
        const int len = (int)x.tokens.size() - x.n;
        int i = 0;
        for (; i < c; ++i)
        {
            if (x.armed && len + i == x.s.ins.after) break;
            if (t[i] == x.s.ins.unless) x.armed = false;
            x.tokens.push_back(t[i]);
        }
        *kept = i;
        if (i == c) return c;
        x.armed = false;
        const int m = std::max(0, std::min((int)x.s.ins.tokens.size(), x.s.n_gen - (len + i)));
        const int from = (int)x.tokens.size();
        x.tokens.insert(x.tokens.end(), x.s.ins.tokens.begin(), x.s.ins.tokens.begin() + m);
        stats_[q].inserted += m;
        if ((int)x.tokens.size() - x.n < x.s.n_gen) x.force = from;   // else the reply ends with them: no step follows
        return i + m;
    }
    // the inserted tokens (positions x.force on) as session q's own steps, as its prompt steps run: each main step followed
    // by its first-form draft and the DFlash2 drafter's context, the last main row sampling the token after them. Then the
    // chain, unless the session drafts with DFlash2. The emit may finish the session
    void forced(int q)
    {
        St& x = st_[q];
        const int from = x.force, end = (int)x.tokens.size(), ev = x.rounds == 0 ? Step::DECODE1 : Step::ROUND;
        x.force = -1;
        for (int p0 = from; p0 < end; p0 += x.s.chunk)
        {
            const int M = std::min(x.s.chunk, end - p0);
            const bool last = p0 + M == end;
            std::vector<Row> rows;
            for (int r = 0; r < M; ++r) rows.push_back(Row{q, x.tokens[p0 + r], p0 + r, p0 + r + 1, -1});
            Step m;
            m.kind = Step::MAIN; m.cls = row_class(M); m.event = Step::Event(ev);
            m.round = cur_round_; m.round_active = cur_active_;
            add_rows(m, q, rows, ev);
            check(m);
            const std::vector<int> t = h_.run(m);
            if (last)
            {
                x.tokens.push_back(t[M - 1]);
                if (h_.emit && h_.emit(q, &t[M - 1], 1, M - 1)) { x.done = true; return; }
            }
            Step d;
            d.kind = Step::DRAFT; d.cls = row_class(M); d.event = m.event;
            d.round = cur_round_; d.round_active = cur_active_;
            add_rows(d, q, draft_rows(q, M, p0, &x.tokens[p0 + 1]), ev);
            check(d);
            x.d[1] = h_.run(d)[M - 1];
            if (x.dfl) drafts(q, DflReq{q, M, p0, last ? t[M - 1] : 0, x.lo, last});
        }
        if (!x.dfl || x.mtp) chain({q}, {ev}, K_);
    }
};

inline std::vector<std::vector<int>> mtp_batch(const std::vector<Session>& S, int K, int max_rows, const Hooks& h,
                                               std::vector<Stats>& stats)
{
    const int NS = (int)S.size();
    if (NS > S2N_SLOTS) throw std::runtime_error("s2: more sessions than slots");
    Batch b(K, max_rows, h);
    std::vector<int> qs(NS);
    for (int q = 0; q < NS; ++q) qs[q] = q;
    b.admit(qs, S);
    while (b.round(qs)) {}
    std::vector<std::vector<int>> out(NS);
    stats.assign(NS, Stats());
    for (int q = 0; q < NS; ++q) { out[q] = b.result(q); stats[q] = b.stats(q); }
    return out;
}

}   // namespace s2
