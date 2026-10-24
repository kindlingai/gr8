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
// estimates start afresh with each request. G6_S2_DSEL_MODEL: the costs come from one frozen table (DSelTable), and MTP
// rounds are priced as the confidence stop runs them.
// Confidence stop (Batch::kstop, G6_MTP_KSTOP): a solo DFlash2 session's MTP rounds draft and verify k <= KStop::kmax
// MTP drafts, chosen from the drafts' log-probabilities under the MTP head (StepOut::lp, Hooks::lp): the chain stops
// before a draft that could not be verified, and the verify takes the drafts that pass. The log-probabilities are the
// same bytes on every rank (spec_len.h), so every rank chooses the same k. The verify's row count then varies 2..kmax+1
// as a DFlash2 round's does. A demoted session's chain is completed to K before its first shared round.
// MTP-only engine (no drafter, KStop::solo, G6_MTP_KSTOP=1 without --dflash-manifest: TP=6 / TP=3): a session that decodes
// alone runs round_all, and the same stop applies there: its verify takes kverify of the drafts it holds (main class
// row_class(k + 1), as the DFlash2 path's MTP rounds), its chain runs while kmore allows (kchain). Before a round of several
// sessions, a session whose chain stopped short is chained to K, so shared rounds verify as without the stop.
// Shared-round stop (KStop::b, G6_MTP_KSTOPB): in round_all each session verifies its own k_q <= k drafts from the same
// log-probabilities, so the main step's width varies S..S * (K + 1); capture_all captures the widths the rule can reach.
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
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    long dfl_rounds = 0;              // rounds that verified DFlash2 drafts
    // G6_DFL_REUSE (Batch::dfl_reuse, -1 = not set): where dfl_start filled the drafter from, the lowest position it held
    // context for, the position the MTP warm-up ran to (0 = none), the position the session took the drafter at (promote)
    int dfl_from = -1, dfl_lo = -1, dfl_warm = 0, promoted = -1;
};
// one DFlash2 drafter step of session sess (dflash.cuh DflIn): context = its target rows at positions pos0 .. pos0 + rows - 1,
// lo = the lowest position the drafter holds; query: also draft the block after them, bonus token tok0
constexpr int DFL_B = 8;              // = dfl::B: the bonus row and 7 drafts
constexpr int DFL_WIN = 2048, DFL_RING = 4096;   // = dfl::WIN, dfl::RING: the drafter's window and its KV ring (positions)
// phase (G6_DRAFT_PAR): 0 = launch and wait; 1 = launch on the drafter's own stream and return at once; 2 = wait for
// that launch and return its drafts
struct DflReq { int sess, rows, pos0, tok0, lo; bool query; int phase = 0; };
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
#if G6_MODEL_FLASH
    // A/B switch for Flash solo serving; never changes the full-GLM cost model.
    bool flash_tp2 = [] { const char* e = getenv("G6_FLASH_S2_COST"); return e && atoi(e) == 1; }();
#endif
    double round(int S, int k) const
    {
#if G6_MODEL_FLASH
        if (flash_tp2 && S == 1) {
            static const g6rt::SpecLen cost = [] { g6rt::SpecLen x; x.flash_tp2(); return x; }();
            return cost.verify_ms(k + 1) + cost.draft_ms(k);
        }
#endif
        return main0 + main_row * S * (k + 1) + draft0 + draft_sess * S + (k - 1) * (chain0 + chain_sess * S);
    }
};
// G6_S2_DSEL (every rank, read at boot): drafter choice for DFlash2 sessions (Batch::dsel), on unless 0. RigMark TP4:
// prose 28.8 -> 30.1 tok/s, code 48.9 -> 49.0, C4 62.7 -> 63.8 (margin 0.15, dwell 8)
inline bool dsel_env() { const char* e = getenv("G6_S2_DSEL"); return !e || atoi(e) != 0; }
// G6_DRAFT_PAR (every rank, read at boot; g6m4 creates the drafter's stream): a solo DFlash2 round launches the drafter
// step before the MTP first form, on its own stream, and waits for it at the end of the round (after the chain on MTP).
// Both are fixed graphs on separate weights, scratch and fabrics, so each one's output bytes are as serial; 0 = serial
inline int draft_par_env() { const char* e = getenv("G6_DRAFT_PAR"); return e ? atoi(e) : 0; }
// G6_S2_DSEL_MODEL=1 with G6_S2_DSEL_TABLE (every rank, read at boot; DSel::model): the drafter choice prices both
// drafters from one frozen cost table in place of its drafting ms (built in, or G6_SPEC_DRAFT) and its verify table
// (spec_len.h's, or G6_SPEC_TABLE), and prices MTP rounds as the active confidence stop (KStop) runs them, not as K
// drafts. Only the choice reads it: the DFlash2 rounds' own draft count (--spec-adaptive, Batch::dflash) keeps the
// verify table it has (G6_SPEC_TABLE prices that). The table is a list of key:value separated by commas or white
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
// tools/s2_dsel_table.py writes it from a G6_S2_TIMELINE (+ G6_MTP_LPLOG) boot. Measured once and frozen: step times
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
        auto fail = [&](const std::string& w) { *err = "G6_S2_DSEL_TABLE: " + w; return false; };
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
    // a SpecLen's verify table, drafting ms and prior weight
    void price(g6rt::SpecLen& s, double round, double each) const
    {
        std::copy(v + 1, v + VX + 1, s.table + 1);
        s.draft_round = round; s.draft_each = each;
        if (w > 0) s.prior = w;
    }
};
struct DSel
{
    bool on = false;
    int dwell = 8;                    // rounds on a drafter before the next switch (4 switched code to MTP too often)
    double margin = 0.15;             // switch when the other drafter's tokens per ms is this much higher (0.05: code -2.7%)
    int probe = 4;                    // every probe-th MTP round the drafter step also drafts (scored next round)
    g6rt::SpecLen d, m;               // DFlash2 and MTP estimates; draft_round / draft_each hold each round's drafting ms
    int model = 0;                    // G6_S2_DSEL_MODEL (above): 0 = the constants in d and m; 1 = tab
    DSelTable tab;
    // model: the request's MTP rounds, decayed as m's positions (SpecLen::pos_decay) per MTP round: rounds and their
    // table cost (ms), rounds whose first draft was accepted and the drafts accepted after it. The table's prior counts
    // as `prior` rounds: it decides until the request has MTP rounds, then keeps ~9% (SpecLen's 4 would keep 29%)
    double rn = 0, rc = 0, an = 0, ag = 0;
    double prior = 1;
    double dn = 0, dt = 0, dc = 0;    // tab.r: DFlash2 rounds (and scored probes), their tokens and table cost, decayed
};
// G6_MTP_KSTOP (every rank, read at boot; Batch::kstop): how many MTP drafts a solo DFlash2 session's MTP round verifies.
//   0  off: K every round
//   1  thresholds: draft j is verified while p_1 * .. * p_j >= tau[j] (p = exp(lp)), at least one; the chain stops at
//      the first draft that could not pass (p_1 * .. * p_j < tau[j + 1])
//   2  DSel's MTP estimates (DSel::m, levels 1..kmax): more() before each chain step, choose() at the verify
// kmax (G6_MTP_KSTOP_KMAX, 1..3, default K) may exceed K: the solo path verifies with the plain build's graphs, which
// cover every row count up to the prompt chunk. log (G6_MTP_LPLOG): one "# kstop" line per solo round (both drafters),
// the drafts' log-probabilities and the accept count, and one "# kstopb" line per shared round (round_all: every
// session's drafts' log-probabilities, verified and accepted counts, the round's step widths and host ms), for tuning
// tau offline; it changes nothing else.
// G6_MTP_KSTOPB (every rank, read at boot; KStop::b): how many of its drafts each session verifies in a shared round.
//   0  off: every session verifies the same k (the fewest any holds: K unless G6_S2_ADAPT)
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
    KStopB b;                         // shared rounds (G6_MTP_KSTOPB)
};
// G6_DFL_REUSE (every rank, read at boot; g6m4): mode 1 = a DFlash2 session whose prefill a prefix-reuse hit cut short
// keeps the drafter (Batch::reuse_start); 2 = also a session that runs alone on MTP takes it (Batch::promote). min: the
// context rows the drafter must hold before it drafts (G6_DFL_REUSE_MIN, 0..DFL_WIN); fewer = MTP rounds until then
struct DflReuse
{
    int mode = 0;
    int min = 256;
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
            if (!prompt(q, ss[k]) || st_[q].force >= 0) continue;
            if (!st_[q].dfl) live.push_back(q);
            else if (st_[q].mtp) kchain(q, Step::TAIL);   // G6_DFL_REUSE warm-up: d[2..K] (KStop: d[2..k]) for its first round
        }
        std::vector<int> all;
        for (int q = 0; q < S2N_SLOTS; ++q) if (active(q)) all.push_back(q);
        if (ksolo(all.size()) && live.size() == 1) kchain(live[0], Step::TAIL);   // KStop::solo: the only session's chain stops early
        else { std::vector<int> evs; for (size_t k = 0; k < live.size(); ++k) evs.push_back(Step::TAIL);
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
    void dsel(const DSel& d) { sel_ = d; price(); kstop(ks_); }
    const DSel& dsel() const { return sel_; }
    // the drafter choice's tokens per ms of an MTP round and of a DFlash2 round, as next_mtp compares them
    std::pair<double, double> dsel_rates() const
    {
        double rd = 0;
        if (sel_.model && sel_.tab.r > 0 && sel_.dc > 0) rd = sel_.dt / sel_.dc;   // tab.r: the rounds as they ran
        else if (dsl_.on) { for (int k : dsl_.levels) rd = std::max(rd, sel_.d.ratio(k, nullptr, k)); }
        else rd = sel_.d.ratio(dkv_, nullptr, dkv_);
        return {sel_.model ? mtp_rate() : sel_.m.ratio(K_, nullptr, K_), rd};
    }
    // solo MTP rounds' draft count (KStop; mode 2 sets DSel::m's levels and third-position seed, so call it after dsel)
    void kstop(const KStop& k)
    {
        ks_ = k;
        if (ks_.mode != 2) return;
        std::vector<int> lv;
        for (int j = 1; j <= kmax(); ++j) lv.push_back(j);
        sel_.m.set_levels(lv);
        sel_.m.seed[3] = ks_.seed3;
    }
    void dfl_reuse(const DflReuse& r) { reuse_ = r; }
    // G6_DFL_REUSE=2: session q, the only active one and drafting with MTP (it was not alone at its prompt tail, or was
    // demoted), takes the drafter before the next round: MTP rounds feed it the accepted rows as context from the
    // session's last token on, and once it holds reuse_.min rows the session drafts with DFlash2, as a new request starts.
    // The driver calls it while q is the only live session. Inserted tokens waiting for their steps: not this round
    void promote(int q)
    {
        St& x = st_[q];
        if (reuse_.mode < 2 || !h_.dfl || !active(q) || x.dfl || x.force >= 0) return;
        for (int o = 0; o < S2N_SLOTS; ++o) if (o != q && active(o)) return;
        x.dfl = x.mtp = true;            // d[1..k] are drafted (the last round's chain): this round is an MTP one; with
                                         // KStop it verifies kverify() of those k (dlm[1..k] set by round_all and chain)
        x.lo = (int)x.tokens.size() - 1;
        x.warm = x.lo + std::max(1, reuse_.min);
        x.pb = -1; x.since = 0; x.mn = 0;
        if (sel_.on) restart();
        Stats& s = stats_[q];
        if ((int)s.acc.size() < DFL_B) s.acc.resize(DFL_B, 0);
        s.promoted = s.dfl_lo = x.lo; s.dfl_warm = x.warm;
        ring_ = Ring{q, {}, x.lo, x.lo, x.lo - 1};
    }
    // a second session is resident: every DFlash2 session goes to MTP (the K-1 chain steps), before the next round
    void demote()
    {
        for (int q = 0; q < S2N_SLOTS; ++q)
            if (st_[q].dfl && active(q))
            {
                st_[q].dfl = false;
                // on MTP already: d[2..K] are drafted (KStop: d[2..k], the rest chained here). Inserted tokens waiting for
                // their steps: forced() chains them
                if (!st_[q].mtp && st_[q].force < 0) chain({q}, {int(Step::ROUND)}, K_);
                else if (st_[q].k < K_ && st_[q].force < 0) chain({q}, {int(Step::ROUND)}, K_, st_[q].k);
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
        else { demote(); if (ks_.solo && act.size() > 1) complete(act); round_all(act); }
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
        std::vector<int> kq(S2N_SLOTS, k), aq(S2N_SLOTS, -1), nd(S2N_SLOTS, 0);   // KStopB: per session; log: accepted, drafts held
        const bool xr = ks_.b.mode && ks_.b.explore && cur_round_ % ks_.b.explore == 0;
        const bool solo = ksolo(act.size());       // KStop::solo: the session alone verifies kverify of its drafts
        double pe = 0;
        if (solo) kq[act[0]] = kverify(st_[act[0]], &pe);
        else if (ks_.b.mode) kshared(act, k, xr, kq);
        using clk = std::chrono::steady_clock;                    // G6_MTP_LPLOG: the round's host ms per step kind
        const clk::time_point t0 = ks_.log ? clk::now() : clk::time_point{};
        double ms[3] = {};
        for (int q : act)
        {
            St& x = st_[q];
            base[q] = (int)x.tokens.size() - 1;
            nd[q] = x.k;
            std::vector<Row> rows;
            for (int r = 0; r <= kq[q]; ++r)
                rows.push_back(Row{q, r ? x.d[r] : x.tokens[base[q]], base[q] + r, base[q] + r + 1, -1});
            add_rows(v, q, rows, x.rounds == 0 ? Step::DECODE1 : Step::ROUND); // first-step timer: per-session E2
        }
        v.cls = row_class(solo ? kq[act[0]] + 1 : K_ + 1);    // a session's own rows (K+1), as its solo step: the k1 split, so a row's bits do not depend on the batch (KStop::solo: k+1, as round_dfl's MTP rounds)
        check(v);
        const std::vector<int> t = h_.run(v);
        if (ks_.log) ms[0] = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        std::vector<int> on, acc(S2N_SLOTS, 0);
        for (size_t i = 0; i < v.sess.size(); ++i)
        {
            const int q = v.sess[i];
            St& x = st_[q];
            const int* so = &t[v.row0[i]];
            const int kv = kq[q];
            int a = 0;
            while (a < kv && x.d[a + 1] == so[a]) ++a;
            aq[q] = a;
            int kept;
            const int c = commit(q, so, a + 1, &kept);
            sl_[q].observe(kv, a, nullptr, 0);
            if (h_.round) h_.round(q, stats_[q].rounds, base[q], a, kv);
            ++stats_[q].rounds; stats_[q].tokens += a + 1; ++stats_[q].acc[a]; stats_[q].drafted += kv;
            ++st_[q].rounds;
            acc[q] = std::min(kept, a);   // the draft step's rows: those whose input the session committed
            if (h_.emit && h_.emit(q, &x.tokens[base[q] + 1], c, v.row0[i])) { x.done = true; continue; }
            if (x.force < 0) on.push_back(q);
        }
        if (ks_.log) klog(v, act, k, xr, kq, aq, nd, base);   // the drafts' lp before the draft step replaces them
        const clk::time_point t1 = ks_.log ? clk::now() : clk::time_point{};
        int dm = 0;
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
                dm = d.M();
                const std::vector<int> u = h_.run(d);
                for (size_t i = 0; i < d.sess.size(); ++i)
                {   // dlm[1]: KStop's input once G6_DFL_REUSE=2 promotes the session to the solo path (NaN when both are off)
                    st_[d.sess[i]].d[1] = u[d.row0[i] + d.rows[i] - 1];
                    st_[d.sess[i]].dlm[1] = dlp(d.row0[i] + d.rows[i] - 1);
                }
            }
        }
        { // first-step timer: chain carries each survivor's own round class
            std::vector<int> evs;
            for (int q : on) { size_t i = 0; while (i < v.sess.size() && v.sess[i] != q) ++i;
                evs.push_back(i < v.sess.size() ? v.ev[i] : int(v.event)); }
            const clk::time_point t2 = ks_.log ? clk::now() : clk::time_point{};
            const int kc = solo && on.size() == 1 ? 0 : choose(on);
            if (solo && on.size() == 1) kchain(on[0], evs[0]);   // KStop::solo: the chain stops where kmore says
            else chain(on, evs, kc);
            if (ks_.log)
            {
                const clk::time_point t3 = clk::now();
                ms[1] = std::chrono::duration<double, std::milli>(t2 - t1).count();
                ms[2] = std::chrono::duration<double, std::milli>(t3 - t2).count();
                fprintf(stderr, "# kstopb_ms round %ld main %.3f draft %.3f dM %d chain %.3f nch %d cM %d round %.3f\n", cur_round_, ms[0], ms[1], dm,
                        ms[2], on.empty() ? 0 : (solo && on.size() == 1 ? st_[on[0]].k : kc) - 1, (int)on.size(), std::chrono::duration<double, std::milli>(t3 - t0).count());
            }
        }
    }
    // KStopB: each session's verified drafts kq[q] <= k in a shared round of the sessions act (rank-invariant: the drafts'
    // log-probabilities, the round's sessions and its id)
    void kshared(const std::vector<int>& act, int k, bool xr, std::vector<int>& kq) const
    {
        const int S = (int)act.size(), lo = std::min(ks_.b.kmin, k);
        int M = 0;
        for (int q : act)
        {
            int j = 0;
            if (xr) j = lo + (int)(((uint64_t)cur_round_ * 0x9E3779B97F4A7C15ull + (uint64_t)q * 0xBF58476D1CE4E5B9ull) >> 33) % (k - lo + 1);
            else while (j < k && kprob(st_[q], j + 1) >= ks_.b.tau[S][j + 1]) ++j;
            kq[q] = std::max(j, lo);
            M += kq[q] + 1;
        }
        while (M != S * (k + 1) && !ks_.b.width(S, K_, M))
        {   // a width the boot did not capture: the unverified draft most likely right joins (ties: the first in act)
            int bq = -1;
            double bp = -1;
            for (int q : act)
                if (kq[q] < k) { const double p = kprob(st_[q], kq[q] + 1); if (p > bp) { bp = p; bq = q; } }
            ++kq[bq]; ++M;
        }
    }
    // G6_MTP_LPLOG: one "# kstopb" line per shared round, after its verify: the round's sessions S, main width M, the
    // common k, explore flag; per session slot:k_q:a:nd:lp_1,..,lp_nd (nd: the drafts it held, a: accepted of k_q); hash
    // of the decision's inputs (equal on every rank). "# kstopb_ms" follows with the round's host ms per step kind
    void klog(const Step& v, const std::vector<int>& act, int k, bool xr, const std::vector<int>& kq, const std::vector<int>& aq,
              const std::vector<int>& nd, const std::vector<int>& base) const
    {
        std::string l;
        char b[64];
        uint32_t hs = 0;
        for (int q : act)
        {
            const St& x = st_[q];
            snprintf(b, sizeof b, " %d:%d:%d:%d:", q, kq[q], aq[q], nd[q]);
            l += b;
            for (int j = 1; j <= nd[q]; ++j) { snprintf(b, sizeof b, "%s%.6g", j > 1 ? "," : "", x.dlm[j]); l += b; }
            snprintf(b, sizeof b, "@%d", base[q]);
            l += b;
            hs = hs * 0x01000193u ^ g6rt::SpecLen::hash(x.dlm.data(), nd[q], kq[q]);
        }
        fprintf(stderr, "# kstopb round %ld S %zu M %d k %d x %d sess%s hash %08x\n", cur_round_, act.size(), v.M(), k, xr ? 1 : 0, l.c_str(), hs);
    }

    // one round of solo DFlash2 session q (dflash.cuh dflash_sequence's round): verify k + 1 rows, accept, emit; then the
    // MTP first-form pass and the drafter step over the a + 1 accepted rows. On MTP (x.mtp, drafter choice) the round
    // verifies the K MTP drafts (KStop: kverify of the x.k drafted), the drafter step takes the rows as context only, and
    // the K-1 chain steps follow (KStop: kchain).
    void round_dfl(int q)
    {
        St& x = st_[q];
        const int base = (int)x.tokens.size() - 1;
        const bool m = x.mtp;
        double pe = 0;
        const int k = m ? (ks_.mode ? kverify(x, &pe) : x.k) : dsl_.on ? dsl_.choose(x.dl, dkv_, dkv_, &pe) : dkv_;
        const int* dr = m ? x.d.data() : x.dd;
        Step v;
        v.kind = Step::MAIN;
        v.round = cur_round_; v.round_active = cur_active_;
        v.event = x.rounds == 0 ? Step::DECODE1 : Step::ROUND;
        std::vector<Row> rows;
        for (int r = 0; r <= k; ++r) rows.push_back(Row{q, r ? dr[r] : x.tokens[base], base + r, base + r + 1, -1});
        add_rows(v, q, rows, v.event);
        v.cls = row_class(m && !ks_.mode ? K_ + 1 : k + 1);
        check(v);
        const std::vector<int> t = h_.run(v);
        int a = 0;
        while (a < k && dr[a + 1] == t[a]) ++a;
        int kept;
        const int c = commit(q, t.data(), a + 1, &kept);
        if (!m && dsl_.on) dsl_.observe(k, a, x.dl, pe);
        if (sel_.on) score(x, k, a, t.data(), pe);
        if (ks_.log)
        {   // G6_MTP_LPLOG: the verified drafts' log-probabilities (MTP: every drafted one), and MTP's d[1] with whether
            // the target took it (t[0]); round = G6_S2_ROUNDLOG's round id
            const int n = m ? x.k : k;
            const float* lp = m ? x.dlm.data() : x.dl;
            std::string l;
            char b[48];
            for (int j = 1; j <= n; ++j) { snprintf(b, sizeof b, "%s%.6g", j > 1 ? "," : "", lp[j]); l += b; }
            fprintf(stderr, "# kstop round %ld slot %d n %ld %s base %d k %d nd %d a %d lp %s m1 %.6g m1ok %d pe %.3f hash %08x\n",
                    cur_round_, q, stats_[q].rounds, m ? "mtp" : "dfl", base, k, n, a, l.c_str(), x.dlm[1], x.d[1] == t[0] ? 1 : 0, pe,
                    g6rt::SpecLen::hash(lp, n, k));
        }
        if (h_.round) h_.round(q, stats_[q].rounds, base, a, k);
        ++stats_[q].rounds; stats_[q].tokens += a + 1; ++stats_[q].acc[a]; stats_[q].drafted += k;
        ++x.rounds;
        if (m) ++stats_[q].mtp_rounds;
        else ++stats_[q].dfl_rounds;
        if (h_.emit && h_.emit(q, &x.tokens[base + 1], c, 0)) { x.done = true; return; }
        // G6_DRAFT_PAR: the drafter choice reads nothing the first form writes, so it is made first and the drafter step
        // runs under the first form
        const bool par = par_ && x.force < 0;
        bool nm = false, probe = false;
        DflReq r{};
        if (par)
        {
            r = choose_drafter(q, base, a, t[a], nm, probe);
            r.phase = 1;
            drafts(q, r);
        }
        Step d;
        d.kind = Step::DRAFT; d.event = v.event;
        d.round = cur_round_; d.round_active = cur_active_;
        const int dn = std::min(kept, a) + 1;   // rows whose input the session committed
        add_rows(d, q, draft_rows(q, dn, base, &x.tokens[base + 1]), v.event);
        d.cls = row_class(d.M());
        check(d);
        x.d[1] = h_.run(d)[dn - 1];
        x.dlm[1] = dlp(dn - 1);
        if (x.force >= 0)                 // the insert cut this round: the drafter takes the rows as context, forced() drafts
        {
            drafts(q, DflReq{q, dn, base, 0, x.lo, false});
            x.pb = -1;
            return;
        }
        if (!par)
        {
            r = choose_drafter(q, base, a, t[a], nm, probe);
            drafts(q, r);
        }
        else
        {   // the chain needs only d[1]: it runs before the wait, under the drafter step too
            if (nm) kchain(q, v.event);
            r.phase = 2;
            drafts(q, r);
        }
        if (probe) { x.pb = base + a + 1; std::copy(x.dd, x.dd + DFL_B, x.pd); std::copy(x.dl, x.dl + DFL_B, x.pl); }
        if (!nm) x.pb = -1;
        if (nm && !par) kchain(q, v.event);
    }
    // the drafter of the next round (drafter choice, G6_DFL_REUSE warm-up) and this round's drafter step: the a + 1
    // committed rows as context, drafting when the next round is on DFlash2 or probes
    DflReq choose_drafter(int q, int base, int a, int tok, bool& nm, bool& probe)
    {
        St& x = st_[q];
        nm = sel_.on && next_mtp(x);
        if (ks_.log && sel_.on && sel_.model)
        {   // G6_MTP_LPLOG with DSel::model: the tokens per ms the choice compares after this round, and its MTP means
            const auto [rm, rd] = dsel_rates();
            fprintf(stderr, "# dsel round %ld slot %d rm %.6f rd %.6f c1 %.4f mtp_rounds %.3f ms %.3f g %.4f next %s\n", cur_round_, q, rm, rd,
                    sel_.m.rate(1), sel_.rn, sel_.rn > 0 ? sel_.rc / sel_.rn : 0., sel_.an > 0 ? sel_.ag / sel_.an : 0., nm ? "mtp" : "dfl");
        }
        if (x.warm) { nm = base + a + 1 < x.warm; if (!nm) x.warm = 0; }   // G6_DFL_REUSE warm-up: DFlash2 once it is over
        if (nm != x.mtp) { x.since = 0; ++stats_[q].switches; }
        ++x.since;
        x.mtp = nm;
        probe = nm && !x.warm && x.pb < 0 && ++x.mn % sel_.probe == 0;
        return DflReq{q, a + 1, base, tok, x.lo, !nm || probe};
    }
    void drafts(int q, const DflReq& r)
    {
        const DflDrafts o = h_.dfl(r);
        if (reuse_.mode && r.phase != 2) ring_rows(q, r);
        if (!r.query || r.phase == 1) return;
        for (int i = 1; i < DFL_B; ++i) { st_[q].dd[i] = o.tok[i]; st_[q].dl[i] = o.lp[i]; }
    }

    struct St
    {
        Session s;
        std::vector<int> tokens, d;
        std::vector<float> dlm;           // KStop: d[j]'s log-probability under the MTP head (NaN: unknown)
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
        int warm = 0;                     // G6_DFL_REUSE: MTP rounds until the drafter's context reaches this position (0: none)
        bool armed = false;               // the insert rule (s.ins) may still fire
        int force = -1;                   // inserted tokens from this position on wait for their steps (forced); -1: none
        int dd[DFL_B] = {};               // DFlash2 drafts d1..d7 and their log-probabilities under the drafter
        float dl[DFL_B] = {};
    };
    // the chain of a solo session about to verify MTP drafts: K-1 steps, or with KStop while kmore() allows (x.k drafted)
    void kchain(int q, int ev)
    {
        if (!ks_.mode) { chain({q}, {ev}, K_); return; }
        St& x = st_[q];
        int n = 1;
        while (n < kmax() && kmore(x, n)) { chain({q}, {ev}, n + 1, n); ++n; }
        x.k = n;
    }
    int kmax() const { return ks_.kmax ? ks_.kmax : K_; }
    // KStop::solo: a round_all of n sessions is a solo MTP-only round (mode 1)
    bool ksolo(size_t n) const { return ks_.solo && ks_.mode == 1 && n == 1; }
    // KStop::solo, before a round of several sessions: a session whose chain stopped short drafts up to K (as demote)
    void complete(const std::vector<int>& act)
    {
        for (int q : act)
            if (st_[q].k < K_ && st_[q].force < 0) chain({q}, {int(Step::ROUND)}, K_, st_[q].k);
    }
    // KStop mode 1: the probability that drafts 1..j are all right, as the MTP head sees them (NaN, no lp hook: 1)
    double kprob(const St& x, int j) const
    {
        double p = 1;
        for (int i = 1; i <= j; ++i) p *= std::isnan(x.dlm[i]) ? 1.0 : std::exp((double)x.dlm[i]);
        return p;
    }
    // KStop: one more chain step after n drafts (mode 1: draft n + 1 could still pass, and drafts 1..n do)
    bool kmore(const St& x, int n) const
    {
        if (ks_.mode == 2) return sel_.m.more(x.dlm.data(), n);
        for (int j = 1; j <= n; ++j) if (kprob(x, j) < ks_.tau[j]) return false;
        return kprob(x, n) >= ks_.tau[n + 1];
    }
    // KStop: the drafts of x.d[1..x.k] to verify (at least one, at most kmax: a session G6_DFL_REUSE promoted holds the
    // shared chain's K drafts, which can be more than kmax, and tau has kmax + 1 entries)
    int kverify(const St& x, double* pe) const
    {
        const int n = std::min(x.k, kmax());
        if (ks_.mode == 2) return sel_.m.choose(x.dlm.data(), n, n, pe);
        int k = 0;
        while (k < n && kprob(x, k + 1) >= ks_.tau[k + 1]) ++k;
        return std::max(k, 1);
    }
    float dlp(int row) const { return h_.lp && (ks_.mode || ks_.b.mode || ks_.log) ? h_.lp(row) : NAN; }
    // G6_DFL_REUSE: what the drafter's KV ring holds. Its context rows at position p depend on the tokens 0..p alone, so
    // rows [lo, end) of the session that wrote it last (slot q, or after that slot took another session, its tokens in
    // tok) serve any prompt with those tokens. top: the highest position written, block rows included; a position below
    // top - DFL_RING + 1 has been written over
    struct Ring
    {
        int q = -1;
        std::vector<int> tok;
        int lo = 0, end = 0, top = -1;
    };
    void ring_rows(int q, const DflReq& r)
    {
        if (ring_.q != q) ring_ = Ring{q, {}, r.pos0, r.pos0, r.pos0 - 1};
        if (r.pos0 > ring_.end) ring_.lo = r.pos0;   // a gap: the rows below are not this context's
        ring_.end = r.pos0 + r.rows;
        ring_.top = std::max(ring_.top, ring_.end - 1 + (r.query ? DFL_B : 0));
    }
    // G6_DFL_REUSE, a DFlash2 session's start: dfl_start filled the drafter from x.lo, past the rows a prefix-reuse hit
    // skipped (the prefill taps none of those). The window's rows below x.lo are still in the ring when the last session
    // there had this prompt's tokens up to x.lo and wrote them after its ring last wrapped (turn N+1 of a conversation
    // whose turn N drafted with DFlash2): the context then starts at the lowest of them. With fewer than reuse_.min rows
    // the session drafts with MTP (the drafter taking the rows as context) until the drafter holds that many
    void reuse_start(int q)
    {
        St& x = st_[q];
        const int n = x.n, from = x.lo, wlo = std::max(0, n - 1 - DFL_WIN);
        if (from > wlo)
        {
            const std::vector<int>& t = ring_.q >= 0 ? st_[ring_.q].tokens : ring_.tok;
            const int a = std::max({wlo, ring_.lo, ring_.top - DFL_RING + 1});
            if (a < from && ring_.end >= from && (int)t.size() >= from && std::equal(t.begin(), t.begin() + from, x.tokens.begin()))
                x.lo = a;
        }
        ring_ = Ring{q, {}, x.lo, n - 1, n - 2};
        if (reuse_.min && x.lo > 0 && n - x.lo < reuse_.min) { x.warm = x.lo + reuse_.min; x.mtp = true; }
        stats_[q].dfl_from = from; stats_[q].dfl_lo = x.lo; stats_[q].dfl_warm = x.warm;
    }
    // drafter choice: teach the estimates what this round's tokens t[0..a] say about each drafter. DFlash2 rounds score
    // MTP's first draft (x.d[1], drafted for position base + 1 = t[0]); MTP rounds score a probe's DFlash2 drafts on the
    // positions the round committed
    void score(St& x, int k, int a, const int* t, double pe)
    {
        const float* ml = ks_.mode ? x.dlm.data() : nullptr;   // KStop: MTP's estimates learn its confidence buckets
        if (!x.mtp)
        {
            sel_.d.observe(k, a, x.dl, 0);
            sel_.m.observe(1, x.d[1] == t[0] ? 1 : 0, ml, 0);
            dfl_obs(a + 1, k);
            return;
        }
        sel_.m.observe(k, a, ml, pe);
        if (sel_.model)
        {   // the round as drafted (x.k) and verified (k); the cost is the table's, never this rank's clock
            const double f = sel_.m.pos_decay;
            sel_.rn = sel_.rn * f + 1; sel_.rc = sel_.rc * f + mtp_cost(k, x.k);
            sel_.an *= f; sel_.ag *= f;
            if (a >= 1) { sel_.an += 1; sel_.ag += a - 1; }
            sel_.dn *= sel_.tab.r; sel_.dt *= sel_.tab.r; sel_.dc *= sel_.tab.r;   // tab.r: DFlash2's evidence ages
        }
        probe_score(x);
    }
    // tab.r: a DFlash2 round (or a scored probe) that committed c tokens verifying k drafts, at the table's cost
    void dfl_obs(int c, int k)
    {
        if (!sel_.model || !(sel_.tab.r > 0)) return;
        const double f = sel_.d.pos_decay;
        sel_.dn = sel_.dn * f + 1; sel_.dt = sel_.dt * f + c; sel_.dc = sel_.dc * f + sel_.tab.verify(k + 1) + sel_.tab.d;
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
        dfl_obs(std::min(b, dkv_) + 1, dkv_);        // as a round verifying K drafts would have
        x.pb = -1;
    }
    // the drafter for the next round: switch after dwell rounds when the other one's tokens per ms is margin higher
    bool next_mtp(const St& x) const
    {
        if (x.since < sel_.dwell) return x.mtp;
        const auto [rm, rd] = dsel_rates();
        return x.mtp ? rm * (1 + sel_.margin) >= rd : rm > rd * (1 + sel_.margin);
    }
    // DSel::model: an MTP round's tokens per ms under the confidence stop. It commits 1 + c_1 (1 + g) tokens: c_1 = MTP's
    // first-position rate, which every round teaches (DFlash2 rounds score d[1]); g = the drafts accepted after the first
    // when the first is, which only MTP rounds teach, so before them the table's prior holds. It costs the mean table
    // cost of the request's MTP rounds (verify k + 1 rows, m, e per chain step), plus its share of a probe. Both means
    // shrink toward the table's prior (DSel::prior rounds)
    double mtp_rate() const
    {
        const DSelTable& t = sel_.tab;
        const double w = sel_.prior;
        double g0 = t.g;
        if (g0 < 0) { g0 = 0; double s = 1; for (int j = 2; j <= K_; ++j) { s *= sel_.m.seed[j]; g0 += s; } }
        const double k0 = t.k > 0 ? t.k : K_, n0 = t.n > 0 ? t.n : std::max<double>(K_, k0);
        const double c0 = t.verify(k0 + 1) + t.m + t.e * (n0 - 1);
        const double g = (sel_.ag + w * g0) / (sel_.an + w), c = (sel_.rc + w * c0) / (sel_.rn + w);
        return (1 + sel_.m.rate(1) * (1 + g)) / (c + t.p / sel_.probe);
    }
    double mtp_cost(int k, int nd) const { return sel_.tab.verify(k + 1) + sel_.tab.m + sel_.tab.e * std::max(0, nd - 1); }
    // DSel::model: the table's costs in the drafter choice's estimates
    void price()
    {
        if (!sel_.model) return;
        sel_.tab.price(sel_.d, sel_.tab.d, 0);
        sel_.tab.price(sel_.m, sel_.tab.m, sel_.tab.e);
    }
    // the drafter choice learns each request afresh; its settings (the verify table included) stay
    void restart()
    {
        sel_.d = fresh(sel_.d);
        sel_.m = fresh(sel_.m);
        sel_.rn = sel_.rc = sel_.an = sel_.ag = 0;
        sel_.dn = sel_.dt = sel_.dc = 0;
    }
    static g6rt::SpecLen fresh(const g6rt::SpecLen& o)
    {
        g6rt::SpecLen f;
        f.on = o.on; f.levels = o.levels; f.draft_round = o.draft_round; f.draft_each = o.draft_each;
        std::copy(o.table, o.table + g6rt::SpecLen::KX + 2, f.table);
        std::copy(o.seed, o.seed + g6rt::SpecLen::KX + 1, f.seed);
        f.pos_decay = o.pos_decay; f.bucket_decay = o.bucket_decay; f.prior = o.prior;
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
    bool par_ = draft_par_env() != 0;
    g6rt::SpecLen dsl_;
    DSel sel_;
    KStop ks_;
    DflReuse reuse_;
    Ring ring_;
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
    // chain steps for the sessions in qs (each: base = its tokens.size() - 1; d[1..j0] set; fills d[j0+1..k])
    void chain(const std::vector<int>& qs, const std::vector<int>& evs, int k, int j0 = 1)
    {
        for (int q : qs) st_[q].k = k;
        // first-step timer: each session's chain carries THAT session's round class (evs, parallel to qs).
        for (int j = j0; j < k && !qs.empty(); ++j)
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
            for (size_t i = 0; i < s.sess.size(); ++i) { st_[s.sess[i]].d[j + 1] = t[s.row0[i]]; st_[s.sess[i]].dlm[j + 1] = dlp(s.row0[i]); }
        }
    }
    // the prompt steps of session q as its own steps (rank.cuh mtp_sequence, prompt loop); false = finished by emit
    bool prompt(int q, const Session& p)
    {
        St& x = st_[q];
        if (ring_.q == q)                 // the slot's last session wrote the drafter's ring: keep its tokens
        {
            ring_.tok.assign(x.tokens.begin(), x.tokens.begin() + std::min((int)x.tokens.size(), ring_.end));
            ring_.q = -1;
        }
        x = St();
        x.s = p; x.on = true;
        x.tokens = p.prompt;
        x.n = (int)p.prompt.size();
        x.d.assign(std::max(K_, kmax()) + 1, 0);
        x.dlm.assign(x.d.size(), NAN);
        sl_[q] = g6rt::SpecLen();
        stats_[q] = Stats();
        x.dfl = p.dflash;
        x.armed = p.ins.on();
        if (x.dfl && sel_.on)             // drafter choice learns each request afresh: the last request's text says little
        {
            restart();
        }
        if (x.dfl && !(h_.dfl && h_.dfl_start)) throw std::runtime_error("s2: DFlash2 session without drafter hooks");
        stats_[q].acc.assign(std::max(K_, x.dfl ? DFL_B - 1 : 0) + 1, 0);
        for (float& v : x.dl) v = NAN;
        x.lo = x.dfl ? h_.dfl_start(q) : 0;
        if (x.dfl && reuse_.mode) reuse_start(q);
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
            x.dlm[1] = dlp(M - 1);
            const bool query = s == PS - 1 && x.force < 0 && !x.warm;
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
            x.dlm[1] = dlp(M - 1);
            if (x.dfl) drafts(q, DflReq{q, M, p0, last ? t[M - 1] : 0, x.lo, last});
        }
        if (!x.dfl) chain({q}, {ev}, K_);
        else if (x.mtp) kchain(q, ev);
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
