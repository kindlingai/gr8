// Presence / frequency penalties, host side (penalty.h, s2/dispatch.h). The head kernel's row logic (rank.cuh
// k_penalty) against a plain count of the whole output. Dispatch's tables before each main step: which rows
// carry a penalty, each session's sampling row, the tokens committed since the device counts were updated, and the reset
// of a slot a new request takes.
#include "s2/dispatch.h"
#include "ctx_index.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
template<class F> static bool throws(F f) { try { f(); return false; } catch (const std::exception&) { return true; } }

// the kernel's three passes over one row, as k_penalty runs them (every thread's share, in pass order)
static void kernel_row(float* row, int vlo, int vn, const int* counts, const int* pend, int n, float presence, float frequency) {
    std::vector<float> raw(std::max(n, 1));
    for (int t = 0; t < 256; ++t) g6rt::pen_pass0(row, vlo, vn, pend, n, raw.data(), t, 256);
    for (int t = 0; t < 256; ++t) g6rt::pen_pass1(row, vn, counts, presence, frequency, t, 256);
    for (int t = 0; t < 256; ++t) g6rt::pen_pass2(row, vlo, vn, counts, pend, n, raw.data(), presence, frequency, t, 256);
}
// the reference: the whole output counted, one subtraction per penalized id
static void reference_row(float* row, int vlo, int vn, const std::vector<int>& output, float presence, float frequency) {
    std::vector<int> c(vn, 0);
    for (int v : output) if (v >= vlo && v < vlo + vn) ++c[v - vlo];
    for (int i = 0; i < vn; ++i) if (c[i] > 0) row[i] = row[i] - (presence + frequency * float(c[i]));
}
static void row_logic() {
    std::mt19937 rng(7);
    const int VOCAB = 1000;
    for (int trial = 0; trial < 300; ++trial) {
        // a rank's slice, a committed output, then pending tokens (new since the counts, and drafts before the row)
        const int ranks = 1 + trial % 4, vn = VOCAB / ranks, vlo = vn * (trial / 4 % ranks);
        std::uniform_int_distribution<int> id(0, VOCAB - 1), len(0, 40), small(0, 20);
        std::vector<int> committed, pend;
        for (int i = len(rng); i > 0; --i) committed.push_back(id(rng));
        for (int i = small(rng) / 4; i > 0; --i) pend.push_back(small(rng) < 10 && !committed.empty() ? committed[rng() % committed.size()] : id(rng));
        if (!pend.empty() && small(rng) < 8) pend.push_back(pend[rng() % pend.size()]);   // a duplicate among the pending
        const float presence = (trial % 3) * 0.75f, frequency = (trial % 5) * 0.3f - 0.3f;
        std::vector<int> counts(vn, 0);
        for (int v : committed) if (v >= vlo && v < vlo + vn) ++counts[v - vlo];
        std::vector<float> logits(vn);
        std::normal_distribution<float> g(0.f, 5.f);
        for (float& x : logits) x = g(rng);
        std::vector<float> got = logits, want = logits;
        kernel_row(got.data(), vlo, vn, trial % 7 == 0 ? nullptr : counts.data(), pend.data(), int(pend.size()), presence, frequency);
        std::vector<int> all = trial % 7 == 0 ? std::vector<int>{} : committed;   // null counts read as zero (a reset slot)
        all.insert(all.end(), pend.begin(), pend.end());
        reference_row(want.data(), vlo, vn, all, presence, frequency);
        CHECK(std::memcmp(got.data(), want.data(), vn * sizeof(float)) == 0);
    }
    // the exactness rule itself: a draft counted through the pending list gives the bits a step alone would
    const int vn = 16;
    std::vector<float> a(vn, 0.f), b(vn, 0.f);
    for (int i = 0; i < vn; ++i) a[i] = b[i] = 1.f / (i + 3);
    std::vector<int> counts(vn, 0); counts[5] = 2; counts[9] = 1;
    const int pend[] = {5, 7, 7, 9, 20};
    kernel_row(a.data(), 0, vn, counts.data(), pend, 5, 1.1f, 0.3f);
    std::vector<int> later = counts; later[5] = 3; later[7] = 2; later[9] = 2;
    kernel_row(b.data(), 0, vn, later.data(), nullptr, 0, 1.1f, 0.3f);
    CHECK(std::memcmp(a.data(), b.data(), vn * sizeof(float)) == 0);
    CHECK(a[0] == 1.f / 3 && a[5] == 1.f / 8 - (1.1f + 0.3f * 3) && a[7] == 1.f / 10 - (1.1f + 0.3f * 2));
    // an id outside the slice (20) is another rank's
    CHECK(g6rt::pen_pending_total(pend, 5, 1, 0) == 2 && g6rt::pen_pending_total(pend, 5, 2, 0) == -1 && g6rt::pen_pending_total(pend, 5, 0, 2) == 3);
    // inside the think block: the block is open and no draft before the row is its close. The kernel then adds the
    // thinking coefficients, so a drafted close switches the rows after it to the whole-reply ones
    const int drafts[] = {3, 50, 4};
    CHECK(g6rt::pen_in_think(50, drafts, 0) && g6rt::pen_in_think(50, drafts, 1) && !g6rt::pen_in_think(50, drafts, 2) && !g6rt::pen_in_think(50, drafts, 3));
    CHECK(!g6rt::pen_in_think(-1, drafts, 0) && g6rt::pen_in_think(7, drafts, 3));
    // the think-end boost: 0 through start, per1k a thousand tokens past it, capped at max, and off at per1k 0
    CHECK(g6rt::pen_boost(0, 4096, 2.f, 10.f) == 0.f && g6rt::pen_boost(4096, 4096, 2.f, 10.f) == 0.f);
    CHECK(g6rt::pen_boost(4097, 4096, 2.f, 10.f) == 2.f * 1.f / 1000.f);
    CHECK(g6rt::pen_boost(4596, 4096, 2.f, 10.f) == 2.f * 500.f / 1000.f);
    CHECK(g6rt::pen_boost(9096, 4096, 2.f, 10.f) == 10.f && g6rt::pen_boost(100000, 4096, 2.f, 10.f) == 10.f);
    CHECK(g6rt::pen_boost(5000, 4096, 0.f, 10.f) == 0.f && g6rt::pen_boost(5, 0, 1.f, 0.f) == 0.f);
    // the kernel's n for row first + k: the committed thinking tokens plus k drafts, so a verify row at draft 3 sees
    // what a step alone at that position would (think_n + 3)
    CHECK(g6rt::pen_boost(4096 + 3, 4096, 2.f, 10.f) == g6rt::pen_boost(4099, 4096, 2.f, 10.f));
}

static const std::string header = "n_gen 20 temp 0 seed 1 min 0 stop 99 top_p 1";
static g6rt::ServeReq request(const std::string& suffix) { return parse_request(header + suffix + "\n1,2,3,4,5,6,7,8\n"); }
// a session's rows: positions pos0 .. pos0 + n - 1, token tok0 + i, teacher-forced to the next prompt token below `prompt`
static void add(s2::Step& s, int q, int pos0, int n, int tok0, int prompt = 0) {
    s.sess.push_back(q); s.row0.push_back(s.M()); s.rows.push_back(n); s.ctx.push_back(pos0 + n);
    for (int i = 0; i < n; ++i) s.r.push_back({q, tok0 + i, pos0 + i, pos0 + i + 1, pos0 + i + 1 < prompt ? 1 : -1});
}
static void tables() {
    const auto on = request(" pen 1.5 0.25 0 0 -1"), off = request(""), freq = request(" pen 0 -1 0 0 -1");
    CHECK(on.presence == 1.5f && on.frequency == 0.25f && off.presence == 0.f && off.frequency == 0.f && freq.frequency == -1.f);
    CHECK(throws([] { request(" pen 2.5 0 0 0 -1"); }) && throws([] { request(" pen 1 0 0 0"); }));
    Config c; c.epoch = 9; c.max_rows = 24;
    Sessions sessions(c, "", false, 4);
    Grant g0{{9, 0, 1}, Sessions::request(1, on), {0, 1, 1}, 0};
    Grant g1{{9, 1, 1}, Sessions::request(2, off), {1, 1, 2}, 0};
    sessions.bind(g0, on); sessions.bind(g1, off);
    CHECK(sessions.at(0).sampling.penalty() && !sessions.at(1).sampling.penalty());
    std::vector<g6rt::PenRow> rows(24); std::vector<g6rt::PenSess> sess(MAX_SESSIONS);
    int launches = 0;
    DispatchHooks h; h.bucket = [](bool, int) { return 0; }; h.mode = [](int, int) {};
    h.launch = [&](const Launch& l) { ++launches; return std::vector<int>(l.step.r.size(), 1); };
    h.prefill = [](const Slot&, const Payload&, uint32_t, uint32_t) {};
    h.pen_rows = rows.data(); h.pen_sess = sess.data();
    Dispatch d(sessions, 24, h);
    d.begin(0, 0); d.begin(1, 0);
    // the prompt's tail, rows 4..7 of an 8-token prompt. The last row samples the first reply token. The slot's counts
    // are another request's (reset) and nothing is new
    s2::Step tail; add(tail, 0, 4, 4, 5, 8); add(tail, 1, 4, 4, 5, 8);
    d.run(tail);
    CHECK(rows[0].slot == 0 && rows[0].first == 3 && rows[0].commit == 1 && rows[0].presence == 1.5f && rows[0].frequency == 0.25f);
    CHECK(rows[3].slot == 0 && rows[3].first == 3 && rows[3].commit == 0);
    for (int r = 4; r < 8; ++r) CHECK(rows[r].slot == -1);
    CHECK(sess[0].reset == 1 && sess[0].nnew == 0);
    // an earlier prompt chunk: first is the row count (nothing samples) and the counts still sync
    s2::Step early; add(early, 0, 0, 4, 1, 8);
    d.run(early);
    CHECK(rows[0].slot == 0 && rows[0].first == 4 && rows[0].commit == 1 && sess[0].reset == 0);
    // 3 tokens emitted (the verify committed 2 accepted drafts and the bonus), then the next verify: row 0 samples and
    // rows 1-2 carry drafts. The 3 tokens are new to the counts
    const int t1[] = {11, 12, 11};
    CHECK(sessions.emit({9, 0, 1}, t1, 3, 0, {}).count == 3);
    s2::Step verify; add(verify, 0, 10, 3, 11); add(verify, 1, 10, 3, 11);
    d.run(verify);
    CHECK(rows[0].slot == 0 && rows[0].first == 0 && rows[0].commit == 1 && rows[1].first == 0 && rows[2].first == 0 && !rows[2].commit);
    CHECK(sess[0].reset == 0 && sess[0].nnew == 3 && sess[0].news[0] == 11 && sess[0].news[1] == 12 && sess[0].news[2] == 11);
    CHECK(rows[3].slot == -1 && rows[5].slot == -1);
    // a draft step carries no penalty
    s2::Step draft = verify; draft.kind = s2::Step::DRAFT;
    d.run(draft);
    for (int r = 0; r < 6; ++r) CHECK(rows[r].slot == -1);
    // nothing committed since: the next verify has nothing new
    d.run(verify);
    CHECK(sess[0].nnew == 0 && rows[0].slot == 0);
    // a token, a stop id and a token after it: the count ends at the stop
    const int t2[] = {13, 99, 14};
    CHECK(sessions.emit({9, 0, 1}, t2, 3, 0, {}).finished);
    s2::Step next; add(next, 0, 15, 1, 99);
    d.run(next);
    CHECK(sess[0].nnew == 2 && sess[0].news[0] == 13 && sess[0].news[1] == 99 && sessions.at(0).out.size() == 5);
    // the slot's next request starts from zero
    Slot slot; slot.key = g0.key; slot.run = g0.run;
    sessions.retire(slot); d.release(0);
    g0.key.generation++; g0.req = Sessions::request(3, freq); sessions.bind(g0, freq); d.begin(0, 0);
    d.run(tail);
    CHECK(sess[0].reset == 1 && sess[0].nnew == 0 && rows[0].presence == 0.f && rows[0].frequency == -1.f);
    d.run(tail);
    CHECK(sess[0].reset == 0);
    // the unpenalized session keeps no output
    const int t3[] = {21};
    sessions.emit({9, 1, 1}, t3, 1, 0, {});
    CHECK(sessions.at(1).out.empty());
    CHECK(launches == 8);
}
// the thinking penalties: the row table carries the close while the block is open, and a committed close ends it
static void thinking() {
    const auto think = request(" pen 0.5 0 1.5 -0.25 50 boost 4 2 10"), closed = request(" pen 0 0 1.5 0 -1");
    CHECK(think.tpresence == 1.5f && think.tfrequency == -0.25f && think.think_end == 50 && closed.think_end == -1);
    CHECK(think.boost_start == 4 && think.boost_per1k == 2.f && think.boost_max == 10.f);
    CHECK(throws([] { request(" pen 0 0 2.5 0 50"); }) && throws([] { request(" boost 1 2 21"); }));
    Config c; c.epoch = 9; c.max_rows = 24;
    Sessions sessions(c, "", false, 4);
    Grant g0{{9, 0, 1}, Sessions::request(1, think), {0, 1, 1}, 0};
    Grant g1{{9, 1, 1}, Sessions::request(2, closed), {1, 1, 2}, 0};
    sessions.bind(g0, think); sessions.bind(g1, closed);
    CHECK(sessions.at(0).think_open && !sessions.at(1).think_open && sessions.at(1).sampling.penalty());
    std::vector<g6rt::PenRow> rows(24); std::vector<g6rt::PenSess> sess(MAX_SESSIONS);
    DispatchHooks h; h.bucket = [](bool, int) { return 0; }; h.mode = [](int, int) {};
    h.launch = [&](const Launch& l) { return std::vector<int>(l.step.r.size(), 1); };
    h.prefill = [](const Slot&, const Payload&, uint32_t, uint32_t) {};
    h.pen_rows = rows.data(); h.pen_sess = sess.data();
    Dispatch d(sessions, 24, h);
    d.begin(0, 0); d.begin(1, 0);
    s2::Step verify; add(verify, 0, 8, 3, 11); add(verify, 1, 8, 3, 11);
    d.run(verify);
    CHECK(rows[0].think_end == 50 && rows[2].think_end == 50 && rows[0].tpresence == 1.5f && rows[0].tfrequency == -0.25f && rows[0].presence == 0.5f);
    // the boost's n at the sampling row: no thinking token committed yet. The kernel adds the drafts before each row
    CHECK(rows[0].think_n == 0 && rows[2].think_n == 0 && rows[0].boost_start == 4 && rows[0].boost_per1k == 2.f && rows[0].boost_max == 10.f);
    // a session whose prompt left no block open: the thinking penalties never apply, so its rows carry no close
    CHECK(rows[3].slot == 1 && rows[3].think_end == -1 && rows[3].tpresence == 1.5f);
    // the reply commits the close (a drafted close the verify accepted): the rows after it are outside the block
    const int t1[] = {11, 50, 12};
    sessions.emit({9, 0, 1}, t1, 3, 0, {});
    CHECK(!sessions.at(0).think_open);
    s2::Step next; add(next, 0, 11, 3, 12);
    d.run(next);
    CHECK(rows[0].think_end == -1 && rows[0].presence == 0.5f && rows[0].tpresence == 1.5f && sess[0].nnew == 3 && sess[0].news[1] == 50);
    // the thinking tokens stop counting at the close: 11 and 50 were thinking, 12 was not
    CHECK(sessions.at(0).think_tokens == 2 && rows[0].think_n == 2);
}
// the sequence penalty: the longest earlier occurrence of a sequence's tail (ctx_index.h), the penalty's curve, and the
// rows Dispatch writes for the thinking scope (prompt + reply + drafts) and the guard scope (the reply against a call)
static void sequence() {
    using g6rt::CtxIndex; using g6rt::CtxMatch;
    // the curve: nothing below min, mult at min, times base per token after, and capped
    CHECK(g6rt::pen_seq(3, 0.8f, 1.75f, 4, 6.f) == 0.f && g6rt::pen_seq(4, 0.8f, 1.75f, 4, 6.f) == 0.8f);
    CHECK(g6rt::pen_seq(6, 0.8f, 1.75f, 4, 6.f) == 0.8f * 1.75f * 1.75f && g6rt::pen_seq(40, 0.8f, 1.75f, 4, 6.f) == 6.f);
    CHECK(g6rt::pen_seq(10, 0.f, 1.75f, 4, 6.f) == 0.f && g6rt::pen_seq(2, 0.5f, 2.f, 1, 8.f) == 1.f);
    // a transcribed log: the prompt holds three log lines (each: time, level, msg, newline); the reply copies the first
    // line's start, so the longest match ends where that line continues
    //                     0    1    2    3    4    5    6    7    8    9    10   11
    std::vector<int> ctx = {100, 200, 300, 9,   101, 201, 301, 9,   102, 202, 302, 9};
    std::vector<int> reply = {7, 7, 100, 200};   // "...", then the first line's time and level
    const auto t = [&](int i) { return i < int(ctx.size()) ? ctx[i] : reply[i - ctx.size()]; };
    CtxIndex ix;
    ix.extend(t, int(ctx.size()));
    CtxMatch m = ix.longest(t, int(ctx.size() + reply.size()));
    CHECK(m.len == 2 && m.next == 2 && t(m.next) == 300);          // the copy would go on with the message token
    reply.push_back(300);
    m = ix.longest(t, int(ctx.size() + reply.size()));
    CHECK(m.len == 3 && t(m.next) == 9);
    // the newline after the message: the latest of the three places wins the tie (len 1 run " 9" after 302 vs 300)
    // a run inside the unindexed tail: the reply repeats itself before anything is committed
    std::vector<int> ctx2 = {1};
    std::vector<int> tail = {5, 6, 7, 5, 6};
    const auto t2 = [&](int i) { return i < 1 ? ctx2[i] : tail[i - 1]; };
    CtxIndex ix2; ix2.extend(t2, 1);
    m = ix2.longest(t2, 6);
    CHECK(m.len == 2 && m.next == 3 && t2(m.next) == 7);
    // the index extended later gives the same answer as the scan did
    ix2.extend(t2, 6);
    CHECK(ix2.longest(t2, 6).len == 2 && ix2.longest(t2, 6).next == 3);
    // nothing to match: too short, or no earlier bigram
    CHECK(ix.longest(t, 1).len == 0);
    std::vector<int> fresh = {1, 2, 3, 4};
    CtxIndex ix3; ix3.extend([&](int i) { return fresh[i]; }, 4);
    CHECK(ix3.longest([&](int i) { return fresh[i]; }, 4).len == 0);
    // the guard scope: the retry's reply against the call it must not repeat
    std::vector<int> call = {50, 60, 61, 70, 71, 72, 51};   // <tool_call> read _file <arg..> path .. </tool_call>
    const auto r = [&](int i) { return call[i]; };
    std::vector<int> out = {50};
    const auto o = [&](int i) { return out[i]; };
    m = g6rt::longest_in(o, 1, r, int(call.size()));
    CHECK(m.len == 1 && m.next == 1 && call[m.next] == 60);
    out = {50, 60, 61};
    m = g6rt::longest_in(o, 3, r, int(call.size()));
    CHECK(m.len == 3 && call[m.next] == 70);
    out = {50, 99, 61};   // a different name: only the one-token run " 61" at its own place
    m = g6rt::longest_in(o, 3, r, int(call.size()));
    CHECK(m.len == 1 && call[m.next] == 70);
    out = {40, 41};
    CHECK(g6rt::longest_in(o, 2, r, int(call.size())).len == 0);

    // Dispatch: a session with both scopes. Prompt 1..8 (the test request's), think block open, the reply so far
    // copies prompt tokens 3 4 5; the step's rows carry drafts 6 7 (the copy going on) and then a row after a drafted
    // close
    const auto req = request(" pen 0 0 0 0 50 seqt 0.8:1.75:3:6 seqa 1:2:2:8 seqg 1:2:1:8:3,4,5,6,7");
    CHECK(req.seq[0].mult == 0.8f && req.seq[0].min == 3 && req.seq[1].min == 2 && req.seq[2].min == 1 && req.avoid.size() == 5);
    Config c; c.epoch = 9; c.max_rows = 24;
    Sessions sessions(c, "", false, 4);
    Grant g0{{9, 0, 1}, Sessions::request(1, req), {0, 1, 1}, 0};
    sessions.bind(g0, req);
    CHECK(sessions.at(0).sampling.seq_thinking() && sessions.at(0).sampling.seq_answer() && sessions.at(0).sampling.seq_guard());
    std::vector<g6rt::PenRow> rows(24); std::vector<g6rt::PenSess> sess(MAX_SESSIONS);
    DispatchHooks h; h.bucket = [](bool, int) { return 0; }; h.mode = [](int, int) {};
    h.launch = [&](const Launch& l) { return std::vector<int>(l.step.r.size(), 1); };
    h.prefill = [](const Slot&, const Payload&, uint32_t, uint32_t) {};
    h.pen_rows = rows.data(); h.pen_sess = sess.data();
    Dispatch d(sessions, 24, h);
    d.begin(0, 0);
    const int emitted[] = {3, 4, 5};
    sessions.emit({9, 0, 1}, emitted, 3, 0, {});
    // rows: 0 samples after "3 4 5" (input 5, at position 10), 1 after draft 6, 2 after draft 7, 3 after a drafted
    // close 50, 4 after draft 3, 5 after draft 4 (the copy starting again, outside the block)
    s2::Step v; v.sess = {0}; v.row0 = {0}; v.rows = {6}; v.ctx = {16};
    v.r = {{0, 5, 10, 11, -1}, {0, 6, 11, 12, -1}, {0, 7, 12, 13, -1}, {0, 50, 13, 14, -1}, {0, 3, 14, 15, -1}, {0, 4, 15, 16, -1}};
    d.run(v);
    // thinking scope: row 0's tail "3 4 5" matches prompt 3 4 5 (len 3, the copy would go on with 6): mult at min.
    // Row 1 (len 4) -> 7 at mult * base, row 2 (len 5) -> 8. Rows 3 to 5 are past the drafted close
    CHECK(rows[0].seq_tok[0] == 6 && rows[0].seq_pen[0] == 0.8f);
    CHECK(rows[1].seq_tok[0] == 7 && rows[1].seq_pen[0] == 0.8f * 1.75f);
    CHECK(rows[2].seq_tok[0] == 8 && rows[2].seq_pen[0] == 0.8f * 1.75f * 1.75f);
    CHECK(rows[3].seq_tok[0] == -1 && rows[4].seq_tok[0] == -1 && rows[5].seq_tok[0] == -1);
    // answer scope: off inside the block. Past the drafted close, row 5's tail "3 4" (after 50) matches the prompt's
    // and the reply's "3 4" (the latest, the reply's at positions 8-9, continues with 5): len 2 at its min 2 -> 5 at
    // mult 1; row 4's tail "50 3" matches nothing
    CHECK(rows[0].seq_tok[1] == -1 && rows[2].seq_tok[1] == -1 && rows[4].seq_tok[1] == -1);
    CHECK(rows[5].seq_tok[1] == 5 && rows[5].seq_pen[1] == 1.f);
    // guard scope: the reply "3 4 5" against the call 3 4 5 6 7: len 3 -> 6 at 1 * 2^2; with draft 6, len 4 -> 7;
    // with draft 7 the match reaches the call's end, which has no token after it: nothing. The rows after the close
    // still match (the guard scope does not care about the block): row 5's "3 4" -> 5 at 1 * 2
    CHECK(rows[0].seq_tok[2] == 6 && rows[0].seq_pen[2] == 4.f);
    CHECK(rows[1].seq_tok[2] == 7 && rows[1].seq_pen[2] == 8.f);
    CHECK(rows[2].seq_tok[2] == -1 && rows[3].seq_tok[2] == -1 && rows[5].seq_tok[2] == 5 && rows[5].seq_pen[2] == 2.f);
    // a legitimate short repeat below min: the reply "3" alone after the prompt matches 1 token, under min 3
    Sessions s2(c, "", false, 4);
    s2.bind(g0, req);
    Dispatch d2(s2, 24, h); d2.begin(0, 0);
    const int one[] = {3};
    s2.emit({9, 0, 1}, one, 1, 0, {});
    s2::Step w; w.sess = {0}; w.row0 = {0}; w.rows = {1}; w.ctx = {9}; w.r = {{0, 3, 8, 9, -1}};
    d2.run(w);
    CHECK(rows[0].seq_tok[0] == -1 && rows[0].seq_tok[1] == -1 && rows[0].seq_tok[2] == 4 && rows[0].seq_pen[2] == 1.f);   // the guard's min is 1: "3" -> 4
    // a prompt with no open block (think_end -1): the answer scope alone covers every row
    const auto ans = request(" seqa 1:2:2:8");
    CHECK(!ans.presence && ans.seq[1].mult == 1.f);
    Sessions s3(c, "", false, 4);
    s3.bind(g0, ans);
    CHECK(!s3.at(0).sampling.seq_thinking() && s3.at(0).sampling.seq_answer() && s3.at(0).sampling.tracked() && !s3.at(0).think_open);
    Dispatch d3(s3, 24, h); d3.begin(0, 0);
    const int two[] = {3, 4};
    s3.emit({9, 0, 1}, two, 2, 0, {});
    s2::Step y; y.sess = {0}; y.row0 = {0}; y.rows = {2}; y.ctx = {11}; y.r = {{0, 4, 9, 10, -1}, {0, 5, 10, 11, -1}};
    d3.run(y);
    CHECK(rows[0].seq_tok[1] == 5 && rows[0].seq_pen[1] == 1.f && rows[1].seq_tok[1] == 6 && rows[1].seq_pen[1] == 2.f && rows[0].seq_tok[0] == -1);
}
// a boost without a penalty: the session is tracked all the same, its rows carry the boost and no coefficients
static void boost_only() {
    const auto b = request(" pen 0 0 0 0 50 boost 2 1 5");
    CHECK(!b.presence && b.boost_per1k == 1.f);
    Config c; c.epoch = 9; c.max_rows = 24;
    Sessions sessions(c, "", false, 4);
    Grant g0{{9, 0, 1}, Sessions::request(1, b), {0, 1, 1}, 0};
    sessions.bind(g0, b);
    CHECK(!sessions.at(0).sampling.penalty() && sessions.at(0).sampling.boost() && sessions.at(0).think_open);
    std::vector<g6rt::PenRow> rows(24); std::vector<g6rt::PenSess> sess(MAX_SESSIONS);
    DispatchHooks h; h.bucket = [](bool, int) { return 0; }; h.mode = [](int, int) {};
    h.launch = [&](const Launch& l) { return std::vector<int>(l.step.r.size(), 1); };
    h.prefill = [](const Slot&, const Payload&, uint32_t, uint32_t) {};
    h.pen_rows = rows.data(); h.pen_sess = sess.data();
    Dispatch d(sessions, 24, h);
    d.begin(0, 0);
    const int t1[] = {11, 12, 13};
    sessions.emit({9, 0, 1}, t1, 3, 0, {});
    s2::Step verify; add(verify, 0, 11, 3, 14);
    d.run(verify);
    CHECK(rows[0].slot == 0 && rows[0].think_end == 50 && rows[0].think_n == 3 && rows[0].boost_start == 2 && rows[0].presence == 0.f && rows[0].tpresence == 0.f);
    CHECK(sess[0].nnew == 3);   // the counts still sync, so a penalty and a boost share one table
    // without an open block the boost never applies: the session is not tracked at all
    const auto nb = request(" pen 0 0 0 0 -1 boost 2 1 5");
    const Sampling nob{0, 0, 0, -1, -1, 0, 0, 0, 0, -1, 2, 1.f, 5.f, {}, {}};
    CHECK(!nb.presence && !nob.boost());
}

int main() {
    row_logic();
    tables();
    thinking();
    boost_only();
    sequence();
    std::printf(fails ? "penalty_test: %d FAIL\n" : "penalty_test: PASS\n", fails);
    return fails != 0;
}
