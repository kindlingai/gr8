// DS41 two-lane W prefill admission (s2/ds41_prefill_lanes.h, GR8_DS_PF_LANES), host only. Build: scripts/host_tests.sh.
// 1. lanes 1 = ds41_prefill_single_lane exactly (Config fields and the scheduler digest); lanes 2 = two_lanes + decode cap W; bad lanes / W refused.
// 2. ds41_make_lane_chunks for rows 1..2W at SESS 8 and 32: lane A = the first min(rows, W) rows on the main id, lane B = the rest at
//    A.context on lane B's counter, taken only when lane B exists and its chunk is valid; one lane refuses rows > W.
// 3. A prompt walk on the real scheduler with ds41_prefill_lanes(2): the main ids (prefill quanta + MAIN steps) and lane B's ids are each
//    consecutive, the lane chunks tile the prompt's first N - 1 rows, and no quantum is two lanes while a session decodes.
#include "s2/ds41_prefill_lanes.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
template<class F> static std::string what(F f) { try { f(); } catch (const std::exception& e) { return e.what(); } return ""; }
static const uint32_t W = 4096;

static Config base(unsigned S) {
    Config c; c.pool_rows = 614400 * 4; c.max_rows = 48; c.max_sessions = S; c.mtp = 5;
    return c;
}
static void config() {
    for (unsigned S : {1u, 8u, 32u}) {
        Config a = base(S), b = base(S);
        ds41_prefill_single_lane(a, W);
        ds41_prefill_lanes(b, W, 1);
        CHECK(a.prefill_rows == b.prefill_rows && a.two_lanes == b.two_lanes && a.decode_prefill_rows == b.decode_prefill_rows);
        CHECK(!b.two_lanes && b.decode_prefill_rows == 0 && b.prefill_rows == W);
        CHECK(Scheduler(a).digest() == Scheduler(b).digest());
        Config t = base(S);
        ds41_prefill_lanes(t, W, 2);
        CHECK(t.two_lanes && t.prefill_rows == W && t.decode_prefill_rows == W);
        CHECK(what([&] { Scheduler s(t); }).empty());   // Config::validate accepts the decode cap
        CHECK(Scheduler(t).digest() != Scheduler(a).digest());
    }
    Config c = base(8);
    CHECK(what([&] { ds41_prefill_lanes(c, W, 0); }) == ds41::lanes::kParseError);
    CHECK(what([&] { ds41_prefill_lanes(c, W, 3); }) == ds41::lanes::kParseError);
    CHECK(!what([&] { ds41_prefill_lanes(c, 4000, 2); }).empty());   // not whole pages: the decode cap would fail validate
    CHECK(what([&] { ds41_prefill_lanes(c, 4000, 1); }).empty());     // one lane: the single-lane rule (any positive W)
    CHECK(!what([&] { ds41_prefill_lanes(c, 0, 1); }).empty());
    CHECK(ds41::lanes::parse(nullptr) == 1 && ds41::lanes::parse("") == 1 && ds41::lanes::parse("1") == 1 &&
          ds41::lanes::parse("2") == 2 && ds41::lanes::parse("0") == 0 && ds41::lanes::parse("2 ") == 0);
}
static void chunks() {
    long n = 0;
    for (unsigned S : {8u, 32u})
        for (unsigned slot : {0u, S / 2, S - 1})
            for (uint32_t first : {0u, 64u, 12345u})
                for (uint32_t rows = 1; rows <= 2 * W; rows += (rows < 70 || rows + 70 > 2 * W || (rows > W - 70 && rows < W + 70)) ? 1 : 97) {
                    std::vector<int> prompt(first + rows + 1 + (rows & 3));
                    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = int(i * 7 + 3);
                    const Run run{3, uint32_t((prompt.size() + PAGE - 1) / PAGE), 1, 77};
                    ds41::lanes::Counter cb; cb.next = 5;
                    const Ds41LaneChunks q = ds41_make_lane_chunks(prompt, first, rows, W, true, 11, cb, 99, slot, run);
                    const uint32_t a = std::min(rows, W), b = rows - a;
                    CHECK(q.a.first == first && q.a.rows == a && q.a.context == int(first + a) && q.a.sequence == 11);
                    CHECK(q.b.rows == b && cb.next == (b ? 6u : 5u));
                    CHECK(q.a.tok == prompt.data() + first && q.a.slot == slot && q.a.request == 99 && q.a.prompt_rows == prompt.size());
                    if (b) {
                        CHECK(q.b.first == uint32_t(q.a.context) && q.b.context == int(first + rows) && q.b.sequence == 5);
                        CHECK(q.b.tok == prompt.data() + q.b.first && q.b.slot == slot && q.b.request == 99 && q.b.prompt_rows == prompt.size());
                        CHECK(q.b.first_page == run.first_page && q.b.pages == run.pages);
                    }
                    if (rows > W) {   // one lane refuses a 2-lane quantum, counter untouched
                        ds41::lanes::Counter c1;
                        CHECK(!what([&] { ds41_make_lane_chunks(prompt, first, rows, W, false, 11, c1, 99, slot, run); }).empty() && c1.next == 1);
                    } else {
                        ds41::lanes::Counter c1;
                        const Ds41LaneChunks o = ds41_make_lane_chunks(prompt, first, rows, W, false, 11, c1, 99, slot, run);
                        const Ds41PrefillChunk r = ds41_make_chunk(prompt, first, rows, W, 11, 99, slot, run);
                        CHECK(o.b.rows == 0 && c1.next == 1 && o.a.first == r.first && o.a.rows == r.rows && o.a.context == r.context &&
                              o.a.sequence == r.sequence && o.a.tok == r.tok && o.a.prompt_rows == r.prompt_rows);
                    }
                    ++n;
                }
    // refusals: 0 rows, > 2W rows, lane B reaching the prompt tail (counter untouched), main id 0
    std::vector<int> p(3 * W);
    const Run run{0, uint32_t(3 * W / PAGE), 1, 1};
    ds41::lanes::Counter cb;
    CHECK(!what([&] { ds41_make_lane_chunks(p, 0, 0, W, true, 1, cb, 1, 0, run); }).empty());
    CHECK(!what([&] { ds41_make_lane_chunks(p, 0, 2 * W + 1, W, true, 1, cb, 1, 0, run); }).empty());
    CHECK(!what([&] { ds41_make_lane_chunks(p, uint32_t(3 * W - 2 * W), 2 * W, W, true, 1, cb, 1, 0, run); }).empty() && cb.next == 1);
    CHECK(!what([&] { ds41_make_lane_chunks(p, 0, W + 1, W, true, 0, cb, 1, 0, run); }).empty() && cb.next == 1);
    std::printf("lane chunks: %ld quanta checked (SESS 8 / 32, rows 1..2W)\n", n);
}
// the Driver's order on the real scheduler: Dispatch::prefill takes ++main for every prefill quantum, a MAIN step ++main
static void walk() {
    for (unsigned S : {8u, 32u}) {
        Config c = base(S);
        ds41_prefill_lanes(c, W, 2);
        Scheduler s(c);
        std::vector<std::vector<int>> prompts(3);
        auto admit = [&](uint64_t id, uint32_t n, uint32_t max_tokens) {
            prompts[id].resize(n);
            const Request r{id, n, max_tokens};
            Grant g; g.key = s.next_key(); g.req = r;
            g.run = Run{0, grant_pages(c, r), 1, id * 0x9E3779B97F4A7C15ull + 1};
            Commit cm; cm.epoch = c.epoch; cm.event = s.next_event(); cm.boundary = s.boundary(); cm.before_hash = s.digest(); cm.admit = g;
            s.apply(cm);
        };
        uint64_t main_seq = 0, last_a = 0, last_b = 0;
        ds41::lanes::Counter cb;
        std::vector<uint32_t> covered(3, 0);
        bool ok_ids = true, ok_tile = true, ok_decode = true; long two = 0;
        auto run = [&](int steps) {
            for (int i = 0; i < steps; ++i) {
                bool decoding = false;
                for (const auto& x : s.slots()) decoding |= x.phase == Phase::Decode;
                const Quantum q = s.next();
                if (q.kind == Kind::Prefill) {
                    const Slot* sl = nullptr;
                    for (const auto& x : s.slots()) if (x.key == q.keys[0]) sl = &x;
                    CHECK(sl);
                    const uint64_t id = sl->req.id;
                    const Ds41LaneChunks lc = ds41_make_lane_chunks(prompts[id], q.first, q.rows, W, c.two_lanes, ++main_seq, cb, id,
                                                                    sl->key.slot, sl->run);
                    ok_ids &= lc.a.sequence == last_a + 1 || last_a == 0; last_a = lc.a.sequence;
                    if (lc.b.rows) { ok_ids &= lc.b.sequence == last_b + 1; last_b = lc.b.sequence; ++two; }
                    ok_tile &= lc.a.first == covered[id] && uint32_t(lc.a.context) + lc.b.rows == covered[id] + q.rows;
                    covered[id] += q.rows;
                    ok_decode &= !(decoding && lc.b.rows);
                    s.complete_prefill();
                } else if (q.kind == Kind::PromptTail) {
                    ++main_seq; last_a = main_seq;
                    s.complete_tail(Completion{q.keys[0], 1, false, 0});
                } else if (q.kind == Kind::Decode) {
                    ++main_seq; last_a = main_seq;
                    std::vector<Completion> out;
                    for (unsigned k = 0; k < q.count; ++k) out.push_back(Completion{q.keys[k], 1, false, 0});
                    s.complete_decode(out);
                }
            }
        };
        admit(1, 3 * W + 50, 100000);
        run(12);
        admit(2, 6 * W + 9, 16);
        run(200);
        CHECK(ok_ids && ok_tile && ok_decode && two >= 2);
        CHECK(covered[1] == 3 * W + 49 && covered[2] == 6 * W + 8);
        std::printf("walk SESS %u: main ids %llu, lane-B ids %llu (%ld two-lane quanta), prompts tiled, none two-lane while decoding\n",
                    S, (unsigned long long)main_seq, (unsigned long long)last_b, two);
    }
}
int main() {
    config(); chunks(); walk();
    if (fails) { std::printf("ds41_prefill_lanes_test: %d FAIL\n", fails); return 1; }
    std::printf("ds41_prefill_lanes_test: ok\n");
    return 0;
}
