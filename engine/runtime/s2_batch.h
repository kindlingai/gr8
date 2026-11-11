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
// estimates start afresh with each request. GR8_S2_DSEL_MODEL: the costs come from one frozen table (DSelTable), and MTP
// rounds are priced as the confidence stop runs them.
// Confidence stop (Batch::kstop, GR8_MTP_KSTOP): a solo DFlash2 session's MTP rounds draft and verify k <= KStop::kmax
// MTP drafts, chosen from the drafts' log-probabilities under the MTP head (StepOut::lp, Hooks::lp): the chain stops
// before a draft that could not be verified, and the verify takes the drafts that pass. The log-probabilities are the
// same bytes on every rank (spec_len.h), so every rank chooses the same k. The verify's row count then varies 2..kmax+1
// as a DFlash2 round's does. A demoted session's chain is completed to K before its first shared round.
// MTP-only engine (no drafter, KStop::solo, GR8_MTP_KSTOP=1 without --dflash-manifest: TP=6 / TP=3): a session that decodes
// alone runs round_all, and the same stop applies there: its verify takes kverify of the drafts it holds (main class
// row_class(k + 1), as the DFlash2 path's MTP rounds), its chain runs while kmore allows (kchain). Before a round of several
// sessions, a session whose chain stopped short is chained to K, so shared rounds verify as without the stop.
// Shared-round stop (KStop::b, GR8_MTP_KSTOPB): in round_all each session verifies its own k_q <= k drafts from the same
// log-probabilities, so the main step's width varies S..S * (K + 1); capture_all captures the widths the rule can reach.
// Insert rule (Session::ins, insert_rule.h): the rule's tokens take the place of the reply's after+1-th token, when none
// of the first after is `unless`. The verify's tokens after that one are dropped, as rejected drafts are. The inserted
// tokens are committed and emitted at once. Their own steps (forced, below) run at the start of the session's next
// round, after Driver::step has grown its run to the new length: one main step per chunk of them, teacher-forced as
// prompt steps are, each followed by its first-form draft, the last row sampling the token after them. Then the chain.
// Rows the verify wrote past the cut are rewritten later, as rejected drafts' rows are. The rule is a function of the
// committed tokens alone, so every rank cuts at the same token.
// N-gram lookup (Batch::ngram, GR8_NGRAM; S5, #56): before a solo DFlash2 session's verify (either drafter), the GLM-distilled
// table (ngram_lookup.h) proposes a chain after the committed tokens. When the chain's expected tokens per verify ms
// (its calibrated probabilities, the verify table) beat the drafter's by NgramSrc::margin, the round verifies the chain
// instead of the drafter's drafts; the rest of the round (first form, drafter step, chain) runs as before on the
// committed tokens. The drafter's drafts are still scored against the tokens the verify produced, so its estimates keep
// learning while the table serves. Drafting costs are left out on both sides: the drafter runs every round either way.
// Every input is the same on every rank.
// Within a step the sessions are in ascending index and each session's rows are contiguous, in position order.
// No CUDA types: the engine adapter (m4sim / g6m4) turns a Step into its StepIn and per-session layout.
#pragma once
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <atomic>
#include <cstring>
#include <exception>
#include <type_traits>
#include <memory>
#include <stdexcept>
#include <string>
#include "s2/ds41_sess_drafts.h"
#include "../module_info.h"
#include "s2_in.h"
#include <utility>
#include <vector>
#include <unistd.h>
#include "../knobs-rs/gr8_knobs.h"
#include "../scheduler-rs/g6sched.h"
#include "insert_rule.h"
#if defined(__GLIBC__)
#include <gnu/libc-version.h>
#endif
#include "s2_log.h"
#include "ngram_lookup.h"
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
    // Set by Driver from Scheduler::Kind, not inferred from Event or row shape.
    // Every MAIN step under Kind::Decode uses the decode MoE cut; prompt tails do not.
    bool scheduler_decode = false;
    // first-step timer: what this step is, for E1/E2 grouping.
    // TAIL = the final PromptTail prompt MAIN; DECODE1 = the first Decode-kind round MAIN
    // after the tail; ROUND = every steady decode-round MAIN. DRAFT/CHAIN steps carry
    // the class of the MAIN they follow (TAIL/DECODE1/ROUND), so the readout pairs them.
    enum Event { TAIL, DECODE1, ROUND } event = ROUND;
    // first-step timer: per-session event class, parallel to sess.
    // A steady session in a mixed step keeps its own class; `event` = the strongest class in the step.
    std::vector<int> ev;
    int cls = 1;
    // GR8_S2_ROUNDLOG: the Batch::round this step belongs to (-1 = not in a round, e.g. admit tail chains) and how many
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
    long ngram_rounds = 0, ngram_drafted = 0, ngram_acc = 0;   // GR8_NGRAM: rounds that verified a table chain, its tokens, accepted
    std::vector<long> acc;            // [K+1] rounds by accepted drafts (K = 7 for a DFlash2 session)
    long dfl_rounds = 0;              // rounds that verified DFlash2 drafts
    // GR8_DFL_REUSE (Batch::dfl_reuse, -1 = not set): where dfl_start filled the drafter from, the lowest position it held
    // context for, the position the MTP warm-up ran to (0 = none), the position the session took the drafter at (promote)
    int dfl_from = -1, dfl_lo = -1, dfl_warm = 0, promoted = -1;
};
// one DFlash2 drafter step of session sess (dflash.cuh DflIn): context = its target rows at positions pos0 .. pos0 + rows - 1,
// lo = the lowest position the drafter holds; query: also draft the block after them, bonus token tok0
constexpr int DFL_B = g6mod::INFO.dfl_block;   // the bonus row and 7 drafts
constexpr int DFL_WIN = 2048, DFL_RING = 4096;   // = dfl::WIN, dfl::RING: the drafter's window and its KV ring (positions)
// phase (GR8_DRAFT_PAR): 0 = launch and wait; 1 = launch on the drafter's own stream and return at once; 2 = wait for
// that launch and return its drafts
struct DflReq { int sess, rows, pos0, tok0, lo; bool query; int phase = 0; };
struct DflDrafts { int tok[DFL_B]; float lp[DFL_B]; };

// run(step) -> the sampled token of every row of the step (step.M() ids), for MAIN, DRAFT and CHAIN steps
// round(sess, round, base, a, K): per-session round line (mtp_sequence prints "round N base B accepted a of K")
// prompt(sess, M, pos0): after each prompt MAIN step of a session (rows pos0 .. pos0 + M - 1, row r at pos0 + r)
// emit(sess, toks, k, row) -> true = the session is finished (stop id or max tokens); row: toks[0]'s row in the last MAIN step
// returns per session the generated tokens (as mtp_sequence returns them)
// A hook must not call the Batch back, its getters included: the planner holds its state mutably while a hook runs, and
// every Batch call made from a hook throws std::logic_error.
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
    // optional: row r's log-probability in the last DRAFT or CHAIN step (StepOut::lp; Batch::kstop)
    std::function<float(int)> lp;
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
constexpr int S2N_SLOTS = g6mod::INFO.max_sessions;   // g6s2::MAX_SESSIONS
static_assert(S2N_SLOTS == g6rt::S2N, "the planner's slots and a step's S2In slots (s2_in.h) differ");
// GR8_S2_ADAPT=1 (every rank, read at boot): choose each round's draft count. Off by default: on code and prose at 1-8
// sessions it measured within 0-3% of K drafts every round (the K-th draft is still accepted ~60% of the time), and
// its extra main-step graphs take ~1 GiB
inline bool adapt_env() { return knob::s2_adapt(); }
// the C library the planner's exp() and strtod come from, for the round log's header: a replay under another one may round
// differently
inline std::string libc_version()
{
#if defined(__GLIBC__)
    return std::string("glibc-") + gnu_get_libc_version();
#else
    return "other";
#endif
}
// Step times at TP4, 2200 MHz, contexts < 2K (GR8_DEC_EVPROF / GR8_S2_ROUNDLOG), ms
struct S2Cost
{
    double main0 = 44, main_row = 7;            // verify step of M rows: main0 + main_row * M
    double draft0 = 2.1, draft_sess = 0.25;     // first-form draft step of S sessions' accepted rows
    double chain0 = 2.0, chain_sess = 0.15;     // chain step, one row per session
#if G6_MODEL_FLASH
    // A/B switch for Flash solo serving (GR8_GLM53F_S2_COST=1): spec_len.h's Flash table for this build's width, a host
    // build without one (the tests) TP2's. Never changes the full-GLM cost model.
    bool flash = knob::glm53f_s2_cost();
#ifdef G6_NR
    static constexpr int flash_tp = G6_NR;
#else
    static constexpr int flash_tp = 2;
#endif
#endif
};
// GR8_S2_DSEL (every rank, read at boot): drafter choice for DFlash2 sessions (Batch::dsel), on unless 0. RigMark TP4:
// prose 28.8 -> 30.1 tok/s, code 48.9 -> 49.0, C4 62.7 -> 63.8 (margin 0.15, dwell 8)
inline bool dsel_env() { return knob::s2_dsel(); }
// GR8_DRAFT_PAR (every rank, read at boot; g6m4 creates the drafter's stream): a solo DFlash2 round launches the drafter
// step before the MTP first form, on its own stream, and waits for it at the end of the round (after the chain on MTP).
// Both are fixed graphs on separate weights, scratch and fabrics, so each one's output bytes are as serial; 0 = serial
// (Batch::par_, unset = 0)
// GR8_S2_DSEL_MODEL=1 with GR8_S2_DSEL_TABLE (every rank, read at boot; DSel::model): the drafter choice prices both
// drafters from one frozen cost table in place of its drafting ms (built in, or GR8_SPEC_DRAFT) and its verify table
// (spec_len.h's, or GR8_SPEC_TABLE), and prices MTP rounds as the active confidence stop (KStop) runs them, not as K
// drafts. Only the choice reads it: the DFlash2 rounds' own draft count (--spec-adaptive, Batch::dflash) keeps the
// verify table it has (GR8_SPEC_TABLE prices that). The table is a list of key:value separated by commas or white
// space ('#' to the line's end is a comment), given inline or as the path of a file holding it; its canonical form goes
// into the S2 runtime signature, so ranks with different tables do not start:
//   v<M>     a verify step of M rows, ms (M 1..8; rows not given are interpolated, and extrapolated past the ends)
//   d        a DFlash2 round's drafting: the MTP first form and the drafter's query step
//   m        an MTP round's drafting: the first form and the drafter's context-only step
//   e        an MTP chain step
//   p        a probe's extra: the drafter's query step over its context-only one (p / DSel::probe per MTP round)
//   g k n    MTP rounds under the confidence stop, as the prior of a request's estimate: the drafts accepted after the
//            first when the first is (g), drafts verified (k) and drafted (n) per round. Default: K, K, the seeds
//   w        the acceptance estimates' prior in rounds (SpecLen::prior, default 4). With decay 0.9 the seed keeps
//            w / (w + 10) of every rate, and the positions of a round compound it: at 4 a DFlash2 block whose 7 drafts
//            are all accepted is expected to commit 6.1 tokens, not 8 (at 1: 7.3), which favours MTP's shorter rounds
//   r        price DFlash2 as MTP is: its tokens per ms = the request's DFlash2 rounds' committed tokens over their table
//            cost (verify k + 1 rows, d), a scored probe counting as a round of K drafts; r (0..1] ages that evidence
//            per MTP round, so the probes after a dip decide (0.95: about two). Not given: SpecLen's position rates
//            (spec_len.h), whose compounded prior caps a full block's estimate below MTP's three-draft rounds' price
// tools/s2_dsel_table.py writes it from a GR8_S2_TIMELINE (+ GR8_MTP_LPLOG) boot. Measured once and frozen: step times
// differ per rank, and a decision that differs between ranks hangs the fabric (spec_len.h).
struct DSelTable
{
    static constexpr int VX = g6rt::SpecLen::KX + 1;    // verify rows 1..VX
    double v[VX + 1] = {};
    double d = 0, m = 0, e = 0, p = 0;
    double g = -1, k = 0, n = 0;                        // the MTP prior (-1 / 0: not given)
    double w = 0;                                       // SpecLen::prior (0: not given)
    double r = 0;                                       // DFlash2 priced from its rounds, aged by r per MTP round (0: off)
    // the verify ms of a fractional row count (the prior's mean drafts), linear between rows
    double verify(double rows) const
    {
        rows = std::min(std::max(rows, 1.0), (double)VX);
        const int a = std::min((int)rows, VX - 1);
        return v[a] + (rows - a) * (v[a + 1] - v[a]);
    }
    // the table's text (an inline list, or a file's contents); false with *err on a bad key or value
    bool parse(const std::string& text, std::string* err)
    {
        *this = DSelTable();
        auto fail = [&](const std::string& w) { *err = "GR8_S2_DSEL_TABLE: " + w; return false; };
        bool have[VX + 1] = {}, hd = false, hm = false, he = false;
        std::string s;
        for (size_t i = 0; i < text.size(); ++i)
        {   // comments out, every separator a space
            if (text[i] == '#') { while (i < text.size() && text[i] != '\n') ++i; s += ' '; continue; }
            s += text[i] == ',' || std::isspace((unsigned char)text[i]) ? ' ' : text[i];
        }
        size_t a = 0;
        while ((a = s.find_first_not_of(' ', a)) != std::string::npos)
        {
            const size_t b = std::min(s.find(' ', a), s.size());
            const std::string item = s.substr(a, b - a);
            a = b;
            const size_t c = item.find(':');
            if (c == std::string::npos) return fail("'" + item + "' is not key:value");
            const std::string key = item.substr(0, c), val = item.substr(c + 1);
            char* end = nullptr;
            const double x = q(strtod(val.c_str(), &end));
            if (val.empty() || *end || !std::isfinite(x) || x < 0) return fail("'" + item + "': not a number >= 0");
            if (key.size() == 2 && key[0] == 'v' && key[1] >= '1' && key[1] <= '0' + VX)
            {
                const int r = key[1] - '0';
                if (!(x > 0)) return fail("'" + item + "': a verify step takes time");
                v[r] = x; have[r] = true;
            }
            else if (key == "d") { d = x; hd = true; }
            else if (key == "m") { m = x; hm = true; }
            else if (key == "e") { e = x; he = true; }
            else if (key == "p") p = x;
            else if (key == "g") g = x;
            else if (key == "k") k = x;
            else if (key == "n") n = x;
            else if (key == "w") { if (!(x > 0)) return fail("'" + item + "': the prior needs weight"); w = x; }
            else if (key == "r") { if (!(x > 0 && x <= 1)) return fail("'" + item + "': an aging factor in (0, 1]"); r = x; }
            else return fail("unknown key '" + key + "' (v1..v" + std::to_string(VX) + ", d, m, e, p, g, k, n, w, r)");
        }
        std::vector<int> rs;
        for (int r = 1; r <= VX; ++r) if (have[r]) rs.push_back(r);
        if (rs.size() < 2 || !hd || !hm || !he) return fail("needs two or more verify rows v<M>, and d, m and e");
        if ((k && (k < 1 || k > VX - 1)) || (n && (n < 1 || n > VX - 1)) || (k && n && n < k))
            return fail("k and n are mean draft counts, 1 <= k <= n");
        for (int r = 1; r <= VX; ++r)
        {
            if (have[r]) continue;
            size_t j = 1;                 // the pair of given rows around r (or the nearest pair past an end)
            while (j + 1 < rs.size() && rs[j] < r) ++j;
            const int lo = rs[j - 1], hi = rs[j];
            v[r] = q(v[lo] + (v[hi] - v[lo]) * (r - lo) / (hi - lo));
            if (!(v[r] > 0)) return fail("v" + std::to_string(r) + " extrapolates to " + std::to_string(v[r]) + " ms");
        }
        return true;
    }
    // every value on a 1e-4 grid, so the canonical form below is equal exactly when the doubles are
    static double q(double x) { return std::isfinite(x) ? std::round(x * 1e4) / 1e4 : x; }
    // canonical form: every rank that read the same table prints the same bytes
    std::string str() const
    {
        std::string s;
        char b[64];
        for (int r = 1; r <= VX; ++r) { snprintf(b, sizeof b, "v%d:%.10g,", r, v[r]); s += b; }
        snprintf(b, sizeof b, "d:%.10g,m:%.10g,e:%.10g,p:%.10g", d, m, e, p); s += b;
        if (g >= 0) { snprintf(b, sizeof b, ",g:%.10g", g); s += b; }
        if (k > 0) { snprintf(b, sizeof b, ",k:%.10g", k); s += b; }
        if (n > 0) { snprintf(b, sizeof b, ",n:%.10g", n); s += b; }
        if (w > 0) { snprintf(b, sizeof b, ",w:%.10g", w); s += b; }
        if (r > 0) { snprintf(b, sizeof b, ",r:%.10g", r); s += b; }
        return s;
    }
};
struct DSel
{
    bool on = false;
    int dwell = 8;                    // rounds on a drafter before the next switch (4 switched code to MTP too often)
    double margin = 0.15;             // switch when the other drafter's tokens per ms is this much higher (0.05: code -2.7%)
    int probe = 4;                    // every probe-th MTP round the drafter step also drafts (scored next round)
    g6rt::SpecLen d, m;               // DFlash2 and MTP estimates; draft_round / draft_each hold each round's drafting ms
    int model = 0;                    // GR8_S2_DSEL_MODEL (above): 0 = the constants in d and m; 1 = tab
    DSelTable tab;
    // model: the request's MTP rounds, decayed as m's positions (SpecLen::pos_decay) per MTP round: rounds and their
    // table cost (ms), rounds whose first draft was accepted and the drafts accepted after it. The table's prior counts
    // as `prior` rounds: it decides until the request has MTP rounds, then keeps ~9% (SpecLen's 4 would keep 29%)
    double rn = 0, rc = 0, an = 0, ag = 0;
    double prior = 1;
    double dn = 0, dt = 0, dc = 0;    // tab.r: DFlash2 rounds (and scored probes), their tokens and table cost, decayed
};
// GR8_MTP_KSTOP (every rank, read at boot; Batch::kstop): how many MTP drafts a solo DFlash2 session's MTP round verifies.
//   0  off: K every round
//   1  thresholds: draft j is verified while p_1 * .. * p_j >= tau[j] (p = exp(lp)), at least one; the chain stops at
//      the first draft that could not pass (p_1 * .. * p_j < tau[j + 1])
//   2  DSel's MTP estimates (DSel::m, levels 1..kmax): more() before each chain step, choose() at the verify
// kmax (GR8_MTP_KSTOP_KMAX, 1..3, default K) may exceed K: the solo path verifies with the plain build's graphs, which
// cover every row count up to the prompt chunk. log (GR8_MTP_LPLOG): one "# kstop" line per solo round (both drafters),
// the drafts' log-probabilities and the accept count, and one "# kstopb" line per shared round (round_all: every
// session's drafts' log-probabilities, verified and accepted counts, the round's step widths and host ms), for tuning
// tau offline; it changes nothing else.
// GR8_MTP_KSTOPB (every rank, read at boot; KStop::b): how many of its drafts each session verifies in a shared round.
//   0  off: every session verifies the same k (the fewest any holds: K unless GR8_S2_ADAPT)
//   1  session q verifies k_q: the most j <= k with p_1 * .. * p_i >= tau[S][i] for every i <= j (S = the round's
//      sessions), at least kmin. While the main width sum(k_q + 1) is not one the boot captured (widths, and every
//      multiple of K + 1), the unverified draft with the highest p_1 * .. * p_j joins. The chain still drafts k for every
//      session, and the main step keeps class row_class(K + 1), so the stop changes which drafts are verified only
// explore (measurement): every explore-th round, a shared round's sessions verify k_q from a hash of the round and slot
// instead, so one boot's log holds main steps of every width
struct KStopB
{
    int mode = 0;
    int kmin = 0;
    double tau[S2N_SLOTS + 1][8] = {};    // [S][j], j 1..7; past the configured entries 0 (the remaining drafts all verify)
    std::vector<int> widths;          // main widths captured besides the multiples of K + 1; empty: every width
    int explore = 0;
    // whether a shared round of S sessions may launch a main step of M rows (K: the MTP drafts)
    bool width(int S, int K, int M) const
    {
        if (M < S * (kmin + 1) || M > S * (K + 1)) return false;
        return widths.empty() || M % (K + 1) == 0 || std::find(widths.begin(), widths.end(), M) != widths.end();
    }
};
// the main widths of shared rounds of 1..N sessions with the stop on (capture_all: class row_class(K + 1))
inline std::vector<int> kstopb_widths(const KStopB& b, int N, int K)
{
    std::vector<int> v;
    if (!b.mode) return v;
    for (int M = 1; M <= N * (K + 1); ++M)
        for (int S = 1; S <= N; ++S) if (b.width(S, K, M)) { v.push_back(M); break; }
    return v;
}
// the policy's part of the S2 runtime signature (every rank must decide the same widths); empty when off
inline std::string kstopb_sig(const KStopB& b)
{
    if (!b.mode) return "";
    std::string s = " kstopb " + std::to_string(b.mode) + " kmin " + std::to_string(b.kmin) + " tau";
    char t[32];
    for (int S = 1; S <= S2N_SLOTS; ++S)
        for (int j = 1; j <= 7; ++j) { snprintf(t, sizeof t, "%s%.17g", S == 1 && j == 1 ? " " : j == 1 ? "/" : ",", b.tau[S][j]); s += t; }
    s += " widths";
    for (size_t i = 0; i < b.widths.size(); ++i) s += (i ? "," : " ") + std::to_string(b.widths[i]);
    return s + " explore " + std::to_string(b.explore);
}
struct KStop
{
    int mode = 0;
    int kmax = 0;                     // 0: K
    double tau[4] = {0, 0, 0.2, 0.5};
    double seed3 = 0.55;              // mode 2: MTP's prior acceptance of a third draft (SpecLen::seed, 0.70 for DFlash2)
    bool log = false;
    bool solo = false;                // mode 1 in an MTP-only engine: also a session alone in round_all (kmax <= K)
    KStopB b;                         // shared rounds (GR8_MTP_KSTOPB)
};
// GR8_DFL_REUSE (every rank, read at boot; g6m4): mode 1 = a DFlash2 session whose prefill a prefix-reuse hit cut short
// keeps the drafter (Batch::reuse_start); 2 = also a session that runs alone on MTP takes it (Batch::promote). min: the
// context rows the drafter must hold before it drafts (GR8_DFL_REUSE_MIN, 0..DFL_WIN); fewer = MTP rounds until then
struct DflReuse
{
    int mode = 0;
    int min = 256;
};
// GR8_NGRAM=<table file> (every rank, read at boot; Batch::ngram): t = the table (null: off). kmax: the longest chain
// verified (the engine captures verify widths 2..kmax + 1), kmin: the shortest worth a round, margin: how much more
// tokens per verify ms the chain must promise than the drafter's drafts. log: one "# ngram" line per round it takes
struct NgramSrc
{
    const g6rt::NgramTable* t = nullptr;
    int kmax = 7, kmin = 2;
    double margin = 0.1;
    bool log = false;
};
// the src's part of the S2 runtime signature (every rank must take the same rounds); empty when off
inline std::string ngram_sig(const NgramSrc& s)
{
    if (!s.t) return "";
    char b[160];
    snprintf(b, sizeof b, " ngram %016llx kmax %d kmin %d margin %.17g", (unsigned long long)s.t->digest, s.kmax, s.kmin, s.margin);
    return b;
}
// the round log's field lists (s2_log.h). The size check catches most new fields; one that fits in existing padding
// (a bool after Step::scheduler_decode) does not, so a new field also goes in its list by hand
template <class V> void fields(V& v, Row& r) { v(r.sess)(r.tok)(r.pos)(r.spos)(r.tgt); }
template <class V> void fields(V& v, Step& s)
{
    v(s.kind)(s.scheduler_decode)(s.event)(s.ev)(s.cls)(s.round)(s.round_active)(s.sess)(s.row0)(s.rows)(s.ctx)(s.r);
}
template <class V> void fields(V& v, Session& s) { v(s.prompt)(s.first)(s.chunk)(s.n_gen)(s.dflash)(s.ins); }
template <class V> void fields(V& v, DflReq& r) { v(r.sess)(r.rows)(r.pos0)(r.tok0)(r.lo)(r.query)(r.phase); }
template <class V> void fields(V& v, DflDrafts& d) { v(d.tok)(d.lp); }
template <class V> void fields(V& v, DSelTable& t) { v(t.v)(t.d)(t.m)(t.e)(t.p)(t.g)(t.k)(t.n)(t.w)(t.r); }
template <class V> void fields(V& v, DSel& d)
{
    v(d.on)(d.dwell)(d.margin)(d.probe)(d.d)(d.m)(d.model)(d.tab)(d.rn)(d.rc)(d.an)(d.ag)(d.prior)(d.dn)(d.dt)(d.dc);
}
template <class V> void fields(V& v, KStopB& b) { v(b.mode)(b.kmin)(b.tau)(b.widths)(b.explore); }
template <class V> void fields(V& v, KStop& k) { v(k.mode)(k.kmax)(k.tau)(k.seed3)(k.log)(k.solo)(k.b); }
template <class V> void fields(V& v, DflReuse& r) { v(r.mode)(r.min); }
template <class V> void fields(V& v, Stats& s)
{
    v(s.rounds)(s.tokens)(s.drafted)(s.mtp_rounds)(s.switches)(s.inserted)(s.ngram_rounds)(s.ngram_drafted)(s.ngram_acc)(s.acc)
     (s.dfl_rounds)(s.dfl_from)(s.dfl_lo)(s.dfl_warm)(s.promoted);
}
// the constants (Flash's table switch is the log header's flash_cost)
template <class V> void fields(V& v, S2Cost& c) { v(c.main0)(c.main_row)(c.draft0)(c.draft_sess)(c.chain0)(c.chain_sess); }
template <class V> void fields(V& v, NgramSrc& n) { v(n.kmax)(n.kmin)(n.margin)(n.log); }   // the table: its digest, logged beside
static_assert(sizeof(Row) == 20 && sizeof(Step) == 184 && sizeof(Session) == 72 && sizeof(DflReq) == 28 &&
              sizeof(DflDrafts) == 64 && sizeof(DSelTable) == 144 && sizeof(DSel) == 3136 &&
              sizeof(KStopB) == (S2N_SLOTS + 1) * 64 + 40 && sizeof(KStop) == sizeof(KStopB) + 56 &&
              sizeof(DflReuse) == 8 && sizeof(NgramSrc) == 32 && sizeof(Stats) == 120 &&
              sizeof(S2Cost) == (G6_MODEL_FLASH ? 56 : 48),
              "a field is missing from its fields() list (the round log, s2_log.h)");

// The planner itself is Rust (engine/scheduler-rs src/planner.rs, docs/HOST-CORE.md R3b) and owns its state. Batch is
// a facade over it: it passes the settings in as plain data (g6sched.h g6s2_*), answers the planner's hooks from the
// std::functions in Hooks (an exception a hook throws is held, the planner stops, and the call rethrows it), and keeps
// the round log (GR8_S2_REPLAY_LOG) here, in the same format as before. Hooks must not call back into the Batch.
class Batch
{
public:
    Batch(int K, int max_rows, Hooks h) : K_(K), max_rows_(max_rows), g_(new Glue(std::move(h))), stats_(S2N_SLOTS)
    {
        static const bool abi = g6sch_layout() == g6sch_layout_cpp();
        if (!abi) throw std::runtime_error("scheduler-rs: struct layouts differ from g6sched.h (the planner)");
        g6s2_cost c{cost_.main0, cost_.main_row, cost_.draft0, cost_.draft_sess, cost_.chain0, cost_.chain_sess, false, {}, 0, 0};
#if G6_MODEL_FLASH
        c.flash = cost_.flash;
        g6rt::SpecLen fc;
        fc.flash_cost(S2Cost::flash_tp);
        std::copy(fc.table, fc.table + g6rt::SpecLen::KX + 2, c.flash_table);
        c.flash_draft_round = fc.draft_round;
        c.flash_draft_each = fc.draft_each;
#endif
        const Hooks& hk = g_->h;
        const g6s2_hooks hooks{g_.get(), t_run, t_round, t_prompt, t_emit, t_dfl, t_dfl_start, t_lp, bool(hk.round), bool(hk.prompt),
                               bool(hk.emit), bool(hk.dfl), bool(hk.dfl_start), bool(hk.lp)};
        b_.reset(g6s2_batch_new(K, max_rows, S2N_SLOTS, par_, fresh_spec_, &c, &hooks));
        if (!b_) throw std::runtime_error("s2: the planner holds at most 32 slots");
        if (const auto& path = knob::s2_replay_log())
        {
            static std::atomic<int> made{0};   // m4sim builds a Batch per run: PATH.<pid>, then PATH.<pid>.1, ..
            const int n = made++;
            const std::string name = *path + "." + std::to_string((long)getpid()) + (n ? "." + std::to_string(n) : "");
            std::FILE* f = std::fopen(name.c_str(), "w");
            if (!f) throw std::runtime_error("s2: GR8_S2_REPLAY_LOG: cannot write " + name);
            g_->logf = std::shared_ptr<std::FILE>(f, [](std::FILE* x) { std::fclose(x); });
            record(f);
        }
    }
    Batch(Batch&&) noexcept = default;
    Batch& operator=(Batch&&) noexcept = default;

    // GR8_S2_REPLAY_LOG (s2_log.h): from now on, write every setting, driver call and hook result to f, which round_replay
    // reads back; after each outermost call, the state the driver reads back (state()). Call it before any other call. A
    // second call moves the log to f. f stays the caller's to close
    void record(std::FILE* f)
    {
        Glue* g = g_.get();
        g->log = f;
#if G6_MODEL_FLASH
        const bool flash_cost = cost_.flash;
        const int flash_tp = S2Cost::flash_tp;
#else
        const bool flash_cost = false;
        const int flash_tp = 0;
#endif
        Hooks& h = g->h;
        g->rec("g6s2log", 3, K_, max_rows_, S2N_SLOTS, par_, fresh_spec_, flash_cost, flash_tp, bool(h.dfl), bool(h.dfl_start),
               bool(h.lp), bool(h.round), bool(h.prompt), cost_, libc_version(), g6mod::INFO.model);
        if (g->wrapped) return;
        g->wrapped = true;
        if (h.run) h.run = [g, run = h.run](const Step& s) { std::vector<int> t = run(s); g->rec("run", s, t); return t; };
        if (h.dfl) h.dfl = [g, dfl = h.dfl](const DflReq& r) { const DflDrafts o = dfl(r); g->rec("dfl", r, o); return o; };
        if (h.dfl_start) h.dfl_start = [g, f0 = h.dfl_start](int q) { const int lo = f0(q); g->rec("dfl_start", q, lo); return lo; };
        if (h.lp) h.lp = [g, lp = h.lp](int r) { const float v = lp(r); g->rec("lp", r, v); return v; };
        if (h.round)
            h.round = [g, f = h.round](int q, long n, int base, int a, int k) { g->rec("on_round", q, n, base, a, k); f(q, n, base, a, k); };
        if (h.prompt) h.prompt = [g, f = h.prompt](int q, int M, int pos0) { g->rec("on_prompt", q, M, pos0); f(q, M, pos0); };
        if (h.emit)
            h.emit = [g, emit = h.emit](int q, const int* t, int k, int row) {
                const bool done = emit(q, t, k, row);
                g->rec("emit", q, std::vector<int>(t, t + k), row, done);
                return done;
            };
    }

    // Config::mtp_sessions: n > 0 = no drafting while more than n sessions are active. The cap keeps K_ + 1 in its
    // divisor, so a drafting round of mtp_max_sess() sessions always fits max_rows_ (round()'s guard counts sessions).
    void set_mtp_sessions(int n) { const Api api_(this, "mtp_sessions", n); mtp_sessions_ = n; g6s2_batch_set_mtp_sessions(b(), n); }
    // GR8_DS_SESS_DRAFTS=1 (s2/ds41_sess_drafts.h): a shared round of n sessions verifies at most sess_drafts_kv(K, max_rows, n)
    // drafts a session, and sessions draft while that is >= 1 (mtp_sessions_ is not read). Off: unchanged
    void set_sess_drafts(bool on) { const Api api_(this, "sess_drafts", on); sess_drafts_ = on; g6s2_batch_set_sess_drafts(b(), on); }
    int mtp_max_sess() const { return g6s2_batch_mtp_max_sess(get()); }

    void admit(const std::vector<int>& qs, const std::vector<Session>& ss)
    {
        const Api api_(this, "admit", qs, ss);
        const std::vector<g6s2_session> cs = sessions(ss);
        check(g6s2_batch_admit(b(), qs.data(), qs.size(), cs.data(), cs.size()));
    }
    // DS41 prefill pack (Config::c8_pack): every session of the pack owes exactly one prompt row. One MAIN step takes
    // all their tails, then one first DRAFT and the post-admit CHAIN. admit() stays the solo route; the driver owns the
    // pack's membership.
    void admit_packed_tail(const std::vector<int>& qs, const std::vector<Session>& ss)
    {
        const Api api_(this, "admit_packed_tail", qs, ss);
        const std::vector<g6s2_session> cs = sessions(ss);
        check(g6s2_batch_admit_packed_tail(b(), qs.data(), qs.size(), cs.data(), cs.size()));
    }
    bool active(int q) const { return g6s2_batch_active(get(), q); }
    // the round log's state record: per slot what the driver reads back (on, active, finished, length, the committed
    // tokens' FNV-1a, inserted tokens)
    std::string state() const
    {
        std::vector<uint64_t> v(6 * S2N_SLOTS);
        g6s2_batch_state(get(), v.data());
        return s2log::line("state", v);
    }
    // the round log's learn record: a hash of everything else the planner keeps between calls (its estimates, drafter
    // choice, confidence stop, ring and per-slot state), so a replay that drifts there fails at the call it drifts on
    uint64_t learn() const
    {
        s2log::Hash h;
        learn_fields(h);
        return h.value();
    }
    // the same state written out (round_replay --dump-learn), to find which field two learn records differ in
    std::string learn_text() const
    {
        s2log::Out o;
        o.line = "learn_text";
        learn_fields(o);
        return o.line;
    }
    // the settings passed in and the state the planner exports (g6s2_batch_spec_len, _slot, _learn, ...), in the learn
    // record's field order
    template <class V> void learn_fields(V& v) const
    {
        const Learned& l = pull();
        v(K_)(max_rows_)(mtp_sessions_)(sess_drafts_)(adapt_)(dkv_)(par_)(fresh_spec_)(l.dsl)(l.sel)(ks_)(reuse_)(ng_)(l.ngd)(l.ngm)
         (l.ring)(cost_)(l.sl)(l.round_no)(l.cur_round)(l.cur_active)(l.st)(l.stats);
        if (waterfill_) {
            float lp[32 * 8];
            g6s2_batch_waterfill_lp(get(), lp);
            v(waterfill_)(wf_margin_)(wf_ng_shared_only_)(lp);
        }
    }
    // the session's tokens so far, prompt included: its next round writes rows from length - 1 on
    int length(int q) const { return g6s2_batch_length(get(), q); }
    bool finished(int q) const { return g6s2_batch_finished(get(), q); }
    void cancel(int q) { const Api api_(this, "cancel", q); check(g6s2_batch_cancel(b(), q)); }
    std::vector<int> result(int q) const
    {
        const int32_t* p = nullptr;
        size_t n = 0;
        g6s2_batch_result(get(), q, &p, &n);
        return std::vector<int>(p, p + n);
    }
    // valid until the next call of stats(q)
    const Stats& stats(int q) const
    {
        if (q < 0 || q >= S2N_SLOTS) throw std::out_of_range("s2: stats of slot " + std::to_string(q));
        pull_stats(q, stats_[q]);
        return stats_[q];
    }
    // choose each round's draft count (k <= K); off: every round verifies K drafts
    void adapt(bool on) { const Api api_(this, "adapt", on); adapt_ = on; g6s2_batch_adapt(b(), on); }
    // a solo DFlash2 round's drafter step runs under the MTP first form (GR8_DRAFT_PAR, which sets the default)
    void draft_par(bool on) { const Api api_(this, "draft_par", on); par_ = on; g6s2_batch_draft_par(b(), on); }
    void waterfill(bool on, double margin, bool shared_only = false) {
        const Api api_(this, "waterfill", on, margin, shared_only);
        check(g6s2_batch_waterfill(b(), on, margin, shared_only));
        waterfill_ = on; wf_margin_ = margin; wf_ng_shared_only_ = shared_only;
    }
    void fresh_spec(bool on) { const Api api_(this, "fresh_spec", on); fresh_spec_ = on; g6s2_batch_fresh_spec(b(), on); }
    // DFlash2 sessions verify up to kv of the 7 drafts; sl: the plain build's --spec-adaptive policy (sl.on = false: kv)
    void dflash(int kv, const g6rt::SpecLen& sl)
    {
        const Api api_(this, "dflash", kv, sl);
        dkv_ = kv;
        const g6s2_spec_len c = to_c(sl);
        g6s2_batch_dflash(b(), kv, &c);
    }
    void dsel(const DSel& d)
    {
        const Api api_(this, "dsel", d);
        const g6s2_dsel c = to_c(d);
        g6s2_batch_dsel(b(), &c);
    }
    // the drafter choice's settings and the request's estimates, as the planner holds them (valid until the next call)
    const DSel& dsel() const
    {
        g6s2_dsel c;
        g6s2_batch_dsel_state(get(), &c);
        from_c(c, dsel_);
        return dsel_;
    }
    // the drafter choice's tokens per ms of an MTP round and of a DFlash2 round, as it compares them
    std::pair<double, double> dsel_rates() const
    {
        double rm = 0, rd = 0;
        g6s2_batch_dsel_rates(get(), &rm, &rd);
        return {rm, rd};
    }
    // solo MTP rounds' draft count (KStop; mode 2 sets DSel::m's levels and third-position seed, so call it after dsel)
    void kstop(const KStop& k)
    {
        const Api api_(this, "kstop", k);
        ks_ = k;
        const g6s2_kstop c{k.mode, k.kmax, {k.tau[0], k.tau[1], k.tau[2], k.tau[3]}, k.seed3, k.log, k.solo, k.b.mode, k.b.kmin,
                           k.b.explore, S2N_SLOTS + 1, &k.b.tau[0][0], k.b.widths.data(), (int32_t)k.b.widths.size()};
        g6s2_batch_kstop(b(), &c);
    }
    void dfl_reuse(const DflReuse& r) { const Api api_(this, "dfl_reuse", r); reuse_ = r; g6s2_batch_dfl_reuse(b(), r.mode, r.min); }
    void ngram(const NgramSrc& n)
    {
        const Api api_(this, "ngram", n, n.t ? n.t->digest : 0ull);
        g6s2_ngram c{};
        c.table = n.t != nullptr;
        c.kmax = n.kmax; c.kmin = n.kmin; c.margin = n.margin; c.log = n.log;
        std::vector<uint64_t> keys;
        std::vector<int32_t> toks;
        std::vector<double> ps;
        if (n.t)
        {
            c.n_ns = (int32_t)n.t->ns.size(); c.ns = n.t->ns.data();
            c.has_cal = n.t->has_cal;
            std::copy(n.t->cal, n.t->cal + g6rt::NgramTable::NB, c.cal);
            keys.reserve(n.t->t.size()); toks.reserve(n.t->t.size()); ps.reserve(n.t->t.size());
            for (const auto& [key, e] : n.t->t) { keys.push_back(key); toks.push_back(e.first); ps.push_back(e.second); }
            c.n = keys.size(); c.keys = keys.data(); c.toks = toks.data(); c.ps = ps.data();
        }
        check(g6s2_batch_ngram(b(), &c));
        ng_ = n;
    }
    // GR8_DFL_REUSE=2: session q, the only active one and drafting with MTP (it was not alone at its prompt tail, or was
    // demoted), takes the drafter before the next round: MTP rounds feed it the accepted rows as context from the
    // session's last token on, and once it holds reuse_.min rows the session drafts with DFlash2, as a new request starts.
    // The driver calls it while q is the only live session. Inserted tokens waiting for their steps: not this round
    void promote(int q) { const Api api_(this, "promote", q); check(g6s2_batch_promote(b(), q)); }
    // a second session is resident: every DFlash2 session goes to MTP (the K-1 chain steps), before the next round
    void demote() { const Api api_(this, "demote"); check(g6s2_batch_demote(b())); }
    // the slot can be admitted again (after finished(q) or cancel(q))
    void release(int q) { const Api api_(this, "release", q); check(g6s2_batch_release(b(), q)); }

    // one decode round of every active session: one main step (all rows), one first-form draft step, K-1 chain steps.
    // A session's rows and steps do not depend on which other sessions are in the round (row invariance, A1).
    bool round(const std::vector<int>& order)
    {
        const Api api_(this, "round", order);
        bool ran = false;
        check(g6s2_batch_round(b(), order.data(), order.size(), &ran));
        return ran;
    }

private:
    static_assert(sizeof(int) == sizeof(int32_t) && sizeof(long) == sizeof(int64_t), "the planner's C types");
    static_assert(sizeof(Row) == sizeof(g6s2_row) && std::is_trivially_copyable_v<Row>, "Row is g6s2_row");
    // what the hooks and the round log reach through the planner's context pointer: on the heap, so a moved Batch keeps it
    struct Glue
    {
        explicit Glue(Hooks hk) : h(std::move(hk)) {}
        Hooks h;
        Step step;                    // the step the run hook sees, refilled each time
        std::exception_ptr err;       // the exception a hook threw, rethrown when the planner returns
        bool busy = false;            // in a hook
        std::FILE* log = nullptr;     // record()
        std::shared_ptr<std::FILE> logf;   // the GR8_S2_REPLAY_LOG file, when the Batch opened it
        int depth = 0;                // public calls in progress: only the outermost one is logged
        bool wrapped = false;         // record() has wrapped the hooks
        template <class... T> void rec(const char* tag, const T&... x) { if (log) line(s2log::line(tag, x...)); }
        void line(const std::string& l)
        {
            std::fputs(l.c_str(), log);
            std::fputc('\n', log);
            if (l.compare(0, 6, "learn ") == 0 && depth == 0) std::fflush(log);   // a crash leaves the log whole to the last call
        }
        template <class F> int32_t guard(F&& f) noexcept
        {
            busy = true;
            try { f(); busy = false; return 0; }
            catch (...) { busy = false; err = std::current_exception(); return 1; }
        }
    };
    struct Api
    {
        Batch* b;
        template <class... T> Api(Batch* b, const char* tag, const T&... x) : b(b)
        {
            if (b->g_->busy) throw std::logic_error(std::string("s2: a hook called Batch::") + tag + ": hooks must not call back");
            if (b->g_->depth++ == 0) b->g_->rec(tag, x...);
        }
        ~Api()
        {
            Glue& g = *b->g_;
            if (--g.depth == 0 && g.log) { g.line(b->state()); g.line(s2log::line("learn", b->learn())); }
        }
    };
    struct Free { void operator()(g6s2_batch* p) const { g6s2_batch_free(p); } };
    g6s2_batch* b() const { return b_.get(); }
    // the handle for a getter: throws when a hook calls back (the planner holds its state mutably then)
    g6s2_batch* get() const
    {
        if (g_->busy) throw std::logic_error("s2: a hook read the Batch back: hooks must not call back");
        return b_.get();
    }
    // a call's result: 1 = a hook threw (rethrown here), 2 = the planner refused
    void check(int32_t rc)
    {
        if (rc == 1 && g_->err) { std::exception_ptr e = g_->err; g_->err = nullptr; std::rethrow_exception(e); }
        if (rc) throw std::runtime_error(g6s2_batch_error(b()));
    }

    static int32_t t_run(void* c, const g6s2_step* s, int32_t* out) noexcept
    {
        Glue& g = *static_cast<Glue*>(c);
        return g.guard([&] {
            Step& st = g.step;
            st.kind = Step::Kind(s->kind);
            st.scheduler_decode = false;
            st.event = Step::Event(s->event);
            st.ev.assign(s->ev, s->ev + s->n_sess);
            st.cls = s->cls;
            st.round = s->round;
            st.round_active = s->round_active;
            st.sess.assign(s->sess, s->sess + s->n_sess);
            st.row0.assign(s->row0, s->row0 + s->n_sess);
            st.rows.assign(s->rows, s->rows + s->n_sess);
            st.ctx.assign(s->ctx, s->ctx + s->n_sess);
            st.r.resize((size_t)s->n_rows);
            if (s->n_rows) std::memcpy(st.r.data(), s->r, sizeof(Row) * (size_t)s->n_rows);
            const std::vector<int> t = g.h.run(st);
            if ((int)t.size() < s->n_rows)
                throw std::runtime_error("s2: the run hook returned " + std::to_string(t.size()) + " tokens for " + std::to_string(s->n_rows) + " rows");
            std::copy(t.begin(), t.begin() + s->n_rows, out);
        });
    }
    static int32_t t_round(void* c, int32_t q, int64_t n, int32_t base, int32_t a, int32_t k) noexcept
    {
        Glue& g = *static_cast<Glue*>(c);
        return g.guard([&] { g.h.round(q, (long)n, base, a, k); });
    }
    static int32_t t_prompt(void* c, int32_t q, int32_t M, int32_t pos0) noexcept
    {
        Glue& g = *static_cast<Glue*>(c);
        return g.guard([&] { g.h.prompt(q, M, pos0); });
    }
    static int32_t t_emit(void* c, int32_t q, const int32_t* t, int32_t k, int32_t row, bool* done) noexcept
    {
        Glue& g = *static_cast<Glue*>(c);
        return g.guard([&] { *done = g.h.emit(q, t, k, row); });
    }
    static int32_t t_dfl(void* c, const g6s2_dfl_req* r, g6s2_dfl_drafts* out) noexcept
    {
        Glue& g = *static_cast<Glue*>(c);
        return g.guard([&] {
            const DflDrafts o = g.h.dfl(DflReq{r->sess, r->rows, r->pos0, r->tok0, r->lo, r->query, r->phase});
            std::copy(o.tok, o.tok + DFL_B, out->tok);
            std::copy(o.lp, o.lp + DFL_B, out->lp);
        });
    }
    static int32_t t_dfl_start(void* c, int32_t q, int32_t* lo) noexcept
    {
        Glue& g = *static_cast<Glue*>(c);
        return g.guard([&] { *lo = g.h.dfl_start(q); });
    }
    static int32_t t_lp(void* c, int32_t row, float* out) noexcept
    {
        Glue& g = *static_cast<Glue*>(c);
        return g.guard([&] { *out = g.h.lp(row); });
    }

    static std::vector<g6s2_session> sessions(const std::vector<Session>& ss)
    {
        std::vector<g6s2_session> v;
        v.reserve(ss.size());
        for (const Session& s : ss)
            v.push_back(g6s2_session{s.prompt.data(), s.prompt.size(), s.first, s.chunk, s.n_gen, s.dflash, s.ins.after, s.ins.unless,
                                     s.ins.tokens.data(), s.ins.tokens.size()});
        return v;
    }
    static g6s2_spec_len to_c(const g6rt::SpecLen& s)
    {
        g6s2_spec_len c{};
        if (s.levels.size() > sizeof c.levels / sizeof c.levels[0]) throw std::runtime_error("s2: a SpecLen with more than 16 levels");
        c.on = s.on;
        c.n_levels = (int32_t)s.levels.size();
        std::copy(s.levels.begin(), s.levels.end(), c.levels);
        c.draft_round = s.draft_round; c.draft_each = s.draft_each;
        std::copy(s.table, s.table + g6rt::SpecLen::KX + 2, c.table);
        std::copy(s.seed, s.seed + g6rt::SpecLen::KX + 1, c.seed);
        std::copy(s.pn, s.pn + g6rt::SpecLen::KX + 1, c.pn);
        std::copy(s.pk, s.pk + g6rt::SpecLen::KX + 1, c.pk);
        std::memcpy(c.bn, s.bn, sizeof c.bn);
        std::memcpy(c.bk, s.bk, sizeof c.bk);
        c.pos_decay = s.pos_decay; c.bucket_decay = s.bucket_decay; c.prior = s.prior;
        c.rounds = s.rounds;
        std::copy(s.chosen, s.chosen + g6rt::SpecLen::KX + 1, c.chosen);
        c.pred_sum = s.pred_sum; c.got_sum = s.got_sum;
        return c;
    }
    static void from_c(const g6s2_spec_len& c, g6rt::SpecLen& s)
    {
        s.on = c.on;
        s.levels.assign(c.levels, c.levels + c.n_levels);
        s.draft_round = c.draft_round; s.draft_each = c.draft_each;
        std::copy(c.table, c.table + g6rt::SpecLen::KX + 2, s.table);
        std::copy(c.seed, c.seed + g6rt::SpecLen::KX + 1, s.seed);
        std::copy(c.pn, c.pn + g6rt::SpecLen::KX + 1, s.pn);
        std::copy(c.pk, c.pk + g6rt::SpecLen::KX + 1, s.pk);
        std::memcpy(s.bn, c.bn, sizeof c.bn);
        std::memcpy(s.bk, c.bk, sizeof c.bk);
        s.pos_decay = c.pos_decay; s.bucket_decay = c.bucket_decay; s.prior = c.prior;
        s.rounds = c.rounds;
        std::copy(c.chosen, c.chosen + g6rt::SpecLen::KX + 1, s.chosen);
        s.pred_sum = c.pred_sum; s.got_sum = c.got_sum;
    }
    static g6s2_dsel to_c(const DSel& d)
    {
        const DSelTable& t = d.tab;
        g6s2_dsel c{d.on, d.dwell, d.margin, d.probe, d.model, to_c(d.d), to_c(d.m), {}, d.rn, d.rc, d.an, d.ag, d.prior, d.dn, d.dt, d.dc};
        std::copy(t.v, t.v + DSelTable::VX + 1, c.tab.v);
        c.tab.d = t.d; c.tab.m = t.m; c.tab.e = t.e; c.tab.p = t.p; c.tab.g = t.g; c.tab.k = t.k; c.tab.n = t.n; c.tab.w = t.w; c.tab.r = t.r;
        return c;
    }
    static void from_c(const g6s2_dsel& c, DSel& d)
    {
        d.on = c.on; d.dwell = c.dwell; d.margin = c.margin; d.probe = c.probe; d.model = c.model;
        from_c(c.d, d.d);
        from_c(c.m, d.m);
        DSelTable& t = d.tab;
        std::copy(c.tab.v, c.tab.v + DSelTable::VX + 1, t.v);
        t.d = c.tab.d; t.m = c.tab.m; t.e = c.tab.e; t.p = c.tab.p; t.g = c.tab.g; t.k = c.tab.k; t.n = c.tab.n; t.w = c.tab.w; t.r = c.tab.r;
        d.rn = c.rn; d.rc = c.rc; d.an = c.an; d.ag = c.ag; d.prior = c.prior; d.dn = c.dn; d.dt = c.dt; d.dc = c.dc;
    }
    void pull_stats(int q, Stats& s) const
    {
        g6s2_stats c;
        g6s2_batch_stats(get(), q, &c);
        s.rounds = c.rounds; s.tokens = c.tokens; s.drafted = c.drafted; s.mtp_rounds = c.mtp_rounds; s.switches = c.switches;
        s.inserted = c.inserted; s.ngram_rounds = c.ngram_rounds; s.ngram_drafted = c.ngram_drafted; s.ngram_acc = c.ngram_acc;
        s.acc.assign(c.acc, c.acc + c.n_acc);
        s.dfl_rounds = c.dfl_rounds; s.dfl_from = c.dfl_from; s.dfl_lo = c.dfl_lo; s.dfl_warm = c.dfl_warm; s.promoted = c.promoted;
    }

    // the learn record's view of the planner's state: a Span points into the planner's arrays
    template <class T> struct Span
    {
        const T* p = nullptr;
        size_t n = 0;
        template <class V> friend void fields(V& v, Span& s) { v((long long)s.n); for (size_t i = 0; i < s.n; ++i) v(s.p[i]); }   // as a vector
    };
    struct SlotView
    {
        int first, chunk, n_gen;
        bool dflash;
        int ins_after, ins_unless;
        Span<int32_t> ins_tokens, tokens, d;
        Span<float> dlm;
        int n;
        bool on, done;
        int rounds, k;
        bool dfl, mtp;
        int pb, pd[DFL_B];
        float pl[DFL_B];
        int since, mn, lo, warm;
        bool armed;
        int force, dd[DFL_B];
        float dl[DFL_B];
        // the C++ planner's St list (its Session's prompt left out, its insert rule flattened)
        template <class V> friend void fields(V& v, SlotView& x)
        {
            v(x.first)(x.chunk)(x.n_gen)(x.dflash)(x.ins_after)(x.ins_unless)(x.ins_tokens)(x.tokens)(x.d)(x.dlm)(x.n)(x.on)(x.done)(x.rounds)(x.k)
             (x.dfl)(x.mtp)(x.pb)(x.pd)(x.pl)(x.since)(x.mn)(x.lo)(x.warm)(x.armed)(x.force)(x.dd)(x.dl);
        }
    };
    struct RingView
    {
        int q = -1;
        Span<int32_t> tok;
        int lo = 0, end = 0, top = -1;
        template <class V> friend void fields(V& v, RingView& r) { v(r.q)(r.tok)(r.lo)(r.end)(r.top); }
    };
    struct Learned
    {
        g6rt::SpecLen dsl, ngd, ngm;
        DSel sel;
        RingView ring;
        std::vector<g6rt::SpecLen> sl = std::vector<g6rt::SpecLen>(S2N_SLOTS);
        long round_no = 0, cur_round = -1;
        int cur_active = 0;
        std::vector<SlotView> st = std::vector<SlotView>(S2N_SLOTS);
        std::vector<Stats> stats = std::vector<Stats>(S2N_SLOTS);
    };
    const Learned& pull() const
    {
        get();   // throws in a hook
        Learned& l = learned_;
        g6s2_spec_len c;
        g6s2_batch_spec_len(b(), 0, &c); from_c(c, l.dsl);
        g6s2_batch_spec_len(b(), 1, &c); from_c(c, l.ngd);
        g6s2_batch_spec_len(b(), 2, &c); from_c(c, l.ngm);
        g6s2_dsel d;
        g6s2_batch_dsel_state(b(), &d);
        from_c(d, l.sel);
        g6s2_learn r;
        g6s2_batch_learn(b(), &r);
        l.round_no = r.round_no; l.cur_round = r.cur_round; l.cur_active = r.cur_active;
        l.ring.q = r.ring_q; l.ring.tok = {r.ring_tok, r.n_ring_tok}; l.ring.lo = r.ring_lo; l.ring.end = r.ring_end; l.ring.top = r.ring_top;
        for (int q = 0; q < S2N_SLOTS; ++q)
        {
            g6s2_batch_spec_len(b(), 3 + q, &c);
            from_c(c, l.sl[q]);
            g6s2_slot x;
            g6s2_batch_slot(b(), q, &x);
            SlotView& v = l.st[q];
            v.first = x.first; v.chunk = x.chunk; v.n_gen = x.n_gen; v.dflash = x.dflash; v.ins_after = x.ins_after; v.ins_unless = x.ins_unless;
            v.ins_tokens = {x.ins_tokens, x.n_ins}; v.tokens = {x.tokens, x.n_tokens}; v.d = {x.d, x.n_d}; v.dlm = {x.dlm, x.n_dlm};
            v.n = x.n; v.on = x.on; v.done = x.done; v.rounds = x.rounds; v.k = x.k; v.dfl = x.dfl; v.mtp = x.mtp; v.pb = x.pb;
            std::copy(x.pd, x.pd + DFL_B, v.pd); std::copy(x.pl, x.pl + DFL_B, v.pl);
            v.since = x.since; v.mn = x.mn; v.lo = x.lo; v.warm = x.warm; v.armed = x.armed; v.force = x.force;
            std::copy(x.dd, x.dd + DFL_B, v.dd); std::copy(x.dl, x.dl + DFL_B, v.dl);
            pull_stats(q, l.stats[q]);
        }
        return l;
    }

    int K_, max_rows_;
    // the settings as passed in, for the round log (the planner keeps its own copy)
    int mtp_sessions_ = 0;            // Config::mtp_sessions
    bool sess_drafts_ = false;        // Config::sess_drafts (GR8_DS_SESS_DRAFTS)
    bool adapt_ = false;
    bool waterfill_ = false;
    double wf_margin_ = 0.0;
    bool wf_ng_shared_only_ = false;
    int dkv_ = DFL_B - 1;
    bool par_ = knob::draft_par().value_or(0) != 0;
    bool fresh_spec_ = knob::s2_fresh_spec();
    KStop ks_;
    DflReuse reuse_;
    NgramSrc ng_;
    S2Cost cost_;
    std::unique_ptr<Glue> g_;
    std::unique_ptr<g6s2_batch, Free> b_;
    mutable std::vector<Stats> stats_;   // stats(q)
    mutable DSel dsel_;                  // dsel()
    mutable Learned learned_;            // learn_fields()
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
