// Presence / frequency penalties, host side (penalty.h, s2/dispatch.h). The head kernel's row logic (rank.cuh
// k_penalty) against a plain count of the whole output. Dispatch's tables before each main step: which rows
// carry a penalty, each session's sampling row, the tokens committed since the device counts were updated, and the reset
// of a slot a new request takes.
#include "s2/dispatch.h"
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
    const auto think = request(" pen 0.5 0 1.5 -0.25 50"), closed = request(" pen 0 0 1.5 0 -1");
    CHECK(think.tpresence == 1.5f && think.tfrequency == -0.25f && think.think_end == 50 && closed.think_end == -1);
    CHECK(throws([] { request(" pen 0 0 2.5 0 50"); }));
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
    // a session whose prompt left no block open: the thinking penalties never apply, so its rows carry no close
    CHECK(rows[3].slot == 1 && rows[3].think_end == -1 && rows[3].tpresence == 1.5f);
    // the reply commits the close (a drafted close the verify accepted): the rows after it are outside the block
    const int t1[] = {11, 50, 12};
    sessions.emit({9, 0, 1}, t1, 3, 0, {});
    CHECK(!sessions.at(0).think_open);
    s2::Step next; add(next, 0, 11, 3, 12);
    d.run(next);
    CHECK(rows[0].think_end == -1 && rows[0].presence == 0.5f && rows[0].tpresence == 1.5f && sess[0].nnew == 3 && sess[0].news[1] == 50);
}

int main() {
    row_logic();
    tables();
    thinking();
    std::printf(fails ? "penalty_test: %d FAIL\n" : "penalty_test: PASS\n", fails);
    return fails != 0;
}
