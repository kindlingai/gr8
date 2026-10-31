// DS41 two-lane W prefill contract (runtime/ds41_lanes.h, GR8_DS_PF_LANES=2). Host only. Build and run: scripts/host_tests.sh.
// 1. parse / split / decode cap; the lane chunks of one quantum at SESS 8 and 32 (first / slot of every lane).
// 2. The schedule of every chunk shape the callee makes (Engram, DSpark, bounded replay, late-final skip, rows 1..2W) passes the
//    happens-before / data-flow checker, and lane B skipping range B implies lane A skips it.
// 3. Lane B's fabric (the main table without the C8-pack late slots) carries every lane-B chunk collective (rows 1..W, skip on / off, RS +
//    AG + direct) and does not carry a 512-row pack's late collective, which the main table does.
// 4. Per fabric the step ids are consecutive with lane B on its own counter.
// 5. The real scheduler with decode_prefill_rows(2, W): no quantum above W while a session decodes; 2W when none does.
// Mutants (each must be refused): drop lane B's waits (ea) / lane A's waits (eb) / the fork / the join, share the index list / trunk /
// Engram step buffers / DSpark tap / MoE scratch, lane B on the main fabric, B before A, a wait before its record, a lane missing ops,
// lane B with lane A's window floor, one sequence for both lanes, a pack on lane B, two lanes while decoding.
#include "runtime/s2/ds41_prefill.h"
#include "runtime/ds41_lanes.h"
#include "runtime/ds41_dspark_prefill.h"
#include <cstdio>
#include <string>
#include <vector>

static int bad = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL line %d: %s\n", __LINE__, #x); ++bad; } } while (0)
namespace L = ds41::lanes;
namespace br = ds41::br;

static const int NL = 40, HID = 5120, NR = 4;
static const uint32_t W = 4096;
static std::vector<int> ids40() { std::vector<int> v; for (int l = 0; l < NL; ++l) v.push_back(l); return v; }

static br::Split make_split(bool engram, bool dspark, int tail)
{
    const std::vector<int> ids = ids40();
    const auto p = ds41::plan(ids, 0, NL, true, true, engram, dspark);
    const auto nf = ds41::dsp::pf::nonfinal_plan(p, dspark);
    return tail ? br::split_plan(nf, ids) : br::Split{nf, nf.size(), -1};
}
// the lanes of one quantum as the callee computes each chunk (g6m4.cu dc.prefill)
struct Q { L::Lanes ln; bool skip[2] = {false, false}; };
static Q lanes_for(const br::Split& sp, int tail, bool late_final, uint32_t first, uint32_t rows, uint32_t N)
{
    Q q;
    const L::Split s = L::split(rows, W);
    const uint32_t r[2] = {s.a, s.b}, f[2] = {first, first + s.a};
    for (int k = 0; k < 2; ++k)
    {
        if (!r[k]) continue;
        const int ctx = int(f[k] + r[k]), after = int(N) - 1 - ctx;
        q.skip[k] = late_final && sp.on() && br::late_skip(after, tail);
        const int late = sp.on() ? br::late_rows(int(r[k]), tail) : int(r[k]);
        q.ln.runs[k] = L::lane_runs(sp, int(r[k]), late, q.skip[k], br::win_floor(ctx, int(r[k]), tail));
        q.ln.first[k] = f[k];
    }
    return q;
}
static std::string run_check(const br::Split& sp, const Q& q, const L::Table& t, L::Schedule* out = nullptr)
{
    const std::vector<L::Run> r[2] = {q.ln.runs[0], q.ln.runs[1]};
    const L::Schedule s = L::schedule(sp.ops, r);
    if (out) *out = s;
    return L::check(sp.ops, s, q.ln, t);
}
static int mutants = 0, refused = 0;
static void mutant(const char* name, const std::string& why)
{
    ++mutants;
    if (!why.empty()) ++refused;
    std::printf("mutant %-34s %s%s\n", name, why.empty() ? "NOT REFUSED" : "refused: ", why.c_str());
    CHECK(!why.empty());
}

static void parse_split()
{
    CHECK(L::parse(nullptr) == 1 && L::parse("") == 1 && L::parse("1") == 1 && L::parse("2") == 2);
    CHECK(L::parse("0") == 0 && L::parse("3") == 0 && L::parse("2 ") == 0 && L::parse("two") == 0 && L::parse("12") == 0);
    CHECK(!L::split_ok(0, W) && !L::split_ok(2 * W + 1, W) && L::split_ok(1, W) && L::split_ok(2 * W, W));
    for (uint32_t rows = 1; rows <= 2 * W; ++rows)
    {
        const L::Split s = L::split(rows, W);
        CHECK(s.a + s.b == rows && s.a >= 1 && s.a <= W && s.b <= W && (s.b > 0) == (rows > W) && (s.b == 0 || s.a == W));
    }
    CHECK(L::decode_prefill_rows(2, W) == W && L::decode_prefill_rows(1, W) == 0);
    // the lane chunks of one quantum (ds41_make_chunk, the callee's input) at SESS 8 and 32
    long n = 0;
    for (unsigned S : {8u, 32u})
        for (unsigned slot : {0u, S / 2, S - 1})
            for (uint32_t rows : {1u, 64u, W - 1, W, W + 1, W + 64, 2 * W - 1, 2 * W})
                for (uint32_t first : {0u, 2 * W})
                {
                    const uint32_t N = first + rows + 1 + 7;
                    std::vector<int> prompt(N, 7);
                    g6s2::Run run; run.first_page = slot * 2048; run.pages = (N + g6s2::PAGE - 1) / g6s2::PAGE;
                    const L::Split s = L::split(rows, W);
                    L::Counter cb;
                    const auto a = g6s2::ds41_make_chunk(prompt, first, s.a, W, 11, 99, slot, run);
                    CHECK(a.first == first && a.rows == s.a && a.slot == slot && a.prompt_rows == N);
                    if (s.b)
                    {
                        const auto b = g6s2::ds41_make_chunk(prompt, first + s.a, s.b, W, cb.take(), 99, slot, run);
                        CHECK(b.first == uint32_t(a.context) && b.context == int(first + rows) && b.slot == a.slot && b.sequence == 1 && b.rows <= W);
                    }
                    ++n;
                }
    std::printf("parse/split: rows 1..%u ok; lane chunks %ld (SESS 8, 32)\n", 2 * W, n);
}

static void schedules()
{
    const L::Table t = L::table();
    long cases = 0, two = 0, both_late = 0, a_skips = 0, max_ev = 0, max_seg = 0;
    for (int engram : {0, 1})
        for (int dspark : {0, 1})
            for (int tail : {0, 128})
                for (int lf : {0, 1})
                {
                    if (lf && !tail) continue;
                    const br::Split sp = make_split(engram, dspark, tail);
                    for (uint32_t first : {0u, 2 * W})
                        for (uint32_t rows : {1u, 64u, 255u, W - 1, W, W + 1, W + 100, W + 127, W + 128, W + 300, 6000u, 2 * W - 1, 2 * W})
                            for (uint32_t extra : {0u, 1u, 100u, 255u, 256u, 300u, 5000u})
                            {
                                const uint32_t N = first + rows + 1 + extra;
                                const Q q = lanes_for(sp, tail, lf, first, rows, N);
                                L::Schedule s;
                                const std::string e = run_check(sp, q, t, &s);
                                if (!e.empty())
                                    std::printf("case engram %d dspark %d tail %d lf %d first %u rows %u N %u: %s\n", engram, dspark, tail, lf, first, rows, N, e.c_str());
                                CHECK(e.empty());
                                CHECK(!q.skip[1] || q.skip[0]);   // B skips range B => A skips it
                                ++cases;
                                if (!q.ln.runs[1].empty())
                                {
                                    ++two;
                                    both_late += !q.skip[0] && !q.skip[1] && sp.on();
                                    a_skips += q.skip[0] && !q.skip[1];
                                }
                                else CHECK(s.events == 0);
                                max_ev = std::max<long>(max_ev, s.events);
                                max_seg = std::max<long>(max_seg, long(s.seg.size()));
                            }
                }
    std::printf("schedules: %ld cases checked (%ld two-lane: %ld both run range B, %ld only B runs it); max events %ld, max segments %ld\n",
                cases, two, both_late, a_skips, max_ev, max_seg);
    CHECK(both_late > 0 && a_skips > 0);
}

static void schedule_mutants()
{
    const br::Split sp = make_split(true, true, 128);
    // M1: both lanes run range B, both with a window floor (rows W+200, B is the final chunk); M2: lane A skips range B (rows 2W)
    const Q m1 = lanes_for(sp, 128, true, 0, W + 200, W + 201), m2 = lanes_for(sp, 128, true, 0, 2 * W, 2 * W + 1);
    CHECK(!m1.skip[0] && !m1.skip[1] && m2.skip[0] && !m2.skip[1]);
    CHECK(m1.ln.runs[0].back().win_floor > 0 && m1.ln.runs[1].back().win_floor > m1.ln.runs[0].back().win_floor);
    const L::Table t = L::table();
    for (const Q* q : {&m1, &m2})
    {
        const char* tag = q == &m1 ? "[both late] " : "[A skips]   ";
        L::Schedule s;
        CHECK(run_check(sp, *q, t, &s).empty());
        auto chk = [&](const L::Schedule& x, const L::Lanes& ln, const L::Table& tb) { return L::check(sp.ops, x, ln, tb); };
        auto name = [&](const char* n) { static std::string b; b = std::string(tag) + n; return b.c_str(); };
        {   L::Schedule x = s; for (auto& g : x.seg) if (g.lane == 1 && g.kind == L::K_SEC) g.wait = -1;
            mutant(name("drop ea (B waits A)"), chk(x, q->ln, t)); }
        {   L::Schedule x = s; for (auto& g : x.seg) if (g.lane == 0 && g.kind == L::K_SEC) g.wait = -1;
            mutant(name("drop eb (A waits B)"), chk(x, q->ln, t)); }
        {   L::Schedule x = s; for (auto& g : x.seg) if (g.lane == 1 && g.kind == L::K_PRO) g.wait = -1;
            mutant(name("drop the fork"), chk(x, q->ln, t)); }
        {   L::Schedule x = s; x.seg.pop_back(); mutant(name("drop the join"), chk(x, q->ln, t)); }
        for (int b : {L::BF_IDX_LIST, L::BF_TRUNK, L::BF_ENG_STEP, L::BF_DSP_TAP, L::BF_MOE, L::BF_IDX_CAND, L::BF_HCY, L::BF_PART})
        {
            L::Table tb = t; tb.cls[b] = L::C_SERIAL;
            const std::string n = std::string("share ") + L::buf_name(b);
            // the DSpark tap (37..39 -> ds_main) and the layer-20 candidates (read by 24..36 and the compact) are range-B data: when lane A
            // skips range B it never reads them, so sharing them is no hazard there and the checker must accept it (precision, not a mutant)
            if (q == &m2 && (b == L::BF_DSP_TAP || b == L::BF_IDX_CAND)) { CHECK(chk(s, q->ln, tb).empty()); continue; }
            mutant(name(n.c_str()), chk(s, q->ln, tb));
        }
        {   L::Lanes ln = q->ln; ln.fabric[1] = 0; mutant(name("lane B on the main fabric"), chk(s, ln, t)); }
        {   // B before A: lane 0 carries B's rows (the later positions) and goes first in the chain
            Q r; r.ln.runs[0] = q->ln.runs[1]; r.ln.runs[1] = q->ln.runs[0]; r.ln.first[0] = q->ln.first[1]; r.ln.first[1] = q->ln.first[0];
            mutant(name("B before A"), run_check(sp, r, t));
        }
        {   L::Schedule x = s;   // swap a record with the wait right after it (the wait is enqueued before its record)
            bool done = false;
            for (size_t i = 0; i + 1 < x.seg.size() && !done; ++i)
                if (x.seg[i].record >= 0 && x.seg[i + 1].wait == x.seg[i].record && x.seg[i].lane != x.seg[i + 1].lane)
                { std::swap(x.seg[i], x.seg[i + 1]); done = true; }
            CHECK(done);
            mutant(name("wait before its record"), chk(x, q->ln, t)); }
        {   L::Schedule x = s; bool done = false;   // lane B drops the last op of one of its free segments
            for (auto& g : x.seg) if (!done && g.lane == 1 && g.kind == L::K_OPS && g.e - g.b >= 2) { --g.e; done = true; }
            CHECK(done);
            mutant(name("lane B missing ops"), chk(x, q->ln, t)); }
        if (q == &m1)
        {   L::Schedule x = s;
            for (auto& g : x.seg) if (g.lane == 1 && g.win_floor) g.win_floor = q->ln.runs[0].back().win_floor;
            mutant(name("lane B with lane A's floor"), chk(x, q->ln, t)); }
    }
}

// per fabric, consecutive step ids from 1 (Dispatch's main sequence for MAIN steps + lane A; the lane-B counter for lane B)
static std::string seq_check(bool one_sequence)
{
    std::vector<uint64_t> fab[2];
    uint64_t main_seq = 0;
    L::Counter cb;
    for (int round = 0; round < 40; ++round)
    {
        const uint32_t rows = round % 3 == 0 ? W / 2 : 2 * W;   // single- and two-lane quanta
        const L::Split s = L::split(rows, W);
        fab[0].push_back(++main_seq);                       // lane A: Dispatch::prefill's id
        if (s.b) fab[1].push_back(one_sequence ? ++main_seq : cb.take());
        fab[0].push_back(++main_seq);                       // a MAIN step
    }
    for (int f = 0; f < 2; ++f)
        for (size_t i = 0; i < fab[f].size(); ++i)
            if (fab[f][i] != i + 1) return "fabric " + std::to_string(f) + " step ids not consecutive at " + std::to_string(i);
    return "";
}

struct Tab { g6::FabricDev d{}; size_t region = 0; int nc = 0, rs_k = 0; };
static Tab table_of(const ds41::pf::FabricRuns& r, int pool)
{
    Tab t;
    g6::FabricConfig cfg = r.cfg; cfg.pool = pool;
    CHECK(g6::fab_slot_layout(cfg, cfg.ncoll, &t.d));
    const size_t half = g6::fab_slot_base(t.d, cfg.ncoll);
    t.d.nr = NR; t.d.compact = 1;
    t.region = ((2 * half + g6::fab_recv_bytes(t.d) + size_t(cfg.ncoll) * 8 + 64) + 4095) & ~size_t(4095);
    t.rs_k = r.rs_k;
    return t;
}
static void fabric()
{
    const std::vector<int> ids = ids40();
    const int tail = 128, i_late = br::late_start(ids);
    CHECK(i_late == geom::LATE_LAYER_START);
    long n = 0;
    for (int mx : {32, 48})
        for (int nx : {0, 2})
        {
            const int NC = 2 * NL + 1 + nx;
            const Tab tm = table_of(ds41::pf::fabric_runs(int(W), tail, i_late, NL, nx, mx, HID, NR, 0, true, true), 4);
            const Tab tb = table_of(L::fabric_runs_b(int(W), tail, i_late, NL, nx, mx, HID, NR, 0, true), 4);
            CHECK(tb.region < tm.region && tb.rs_k == tm.rs_k);
            for (uint32_t rows = 1; rows <= W; ++rows)
                for (bool skip : {false, true})
                {
                    const auto c = g6s2::ds41_chunk_collectives(ids, nx, rows, size_t(HID) * 2, 6144 * 2, i_late, uint32_t(br::late_rows(int(rows), tail)), skip);
                    const std::string e = L::fits(tb.d, c.sizes, c.order, NC, tb.rs_k, int(rows), mx, NR, size_t(HID) * 2, true);
                    if (!e.empty()) std::printf("lane B rows %u skip %d mx %d nx %d: %s\n", rows, int(skip), mx, nx, e.c_str());
                    CHECK(e.empty());
                    ++n;
                }
            // the ragged pack: 512 rows through the late layers (g6m4.cu dc.prefill_pack: late = rows, no skip, no RS)
            const auto p = g6s2::ds41_chunk_collectives(ids, nx, uint32_t(ds41::pack::MAX_ROWS), size_t(HID) * 2, 6144 * 2, i_late, uint32_t(ds41::pack::MAX_ROWS), false);
            CHECK(L::fits(tm.d, p.sizes, p.order, NC, 0, ds41::pack::MAX_ROWS, mx, NR, size_t(HID) * 2, false).empty());
            if (mx == 48 && nx == 2)
            {
                mutant("pack on lane B (512-row late coll)", L::fits(tb.d, p.sizes, p.order, NC, 0, ds41::pack::MAX_ROWS, mx, NR, size_t(HID) * 2, false));
                std::printf("fabric W %u tail %d pool 4 RS 1 MX %d Engram %d: main region %.1f MiB, lane B region %.1f MiB\n", W, tail, mx, nx,
                            tm.region / 1048576.0, tb.region / 1048576.0);
            }
            else CHECK(!L::fits(tb.d, p.sizes, p.order, NC, 0, ds41::pack::MAX_ROWS, mx, NR, size_t(HID) * 2, false).empty());
        }
    std::printf("fabric: %ld lane-B chunk shapes fit the lane-B table\n", n);
}

static void b12_fabric_both_lanes()
{
    const auto ids = ids40();
    const int tail = 128, il = br::late_start(ids), nx = 2, mx = 48;
    const int nc = 2 * NL + 1 + nx, k = 2 * il;
    const uint32_t state = uint32_t(ds41::rowshard::state_bytes(tail, HID, 2)); // BF16 trunk in the DS41 build
    for (int lane = 0; lane < 2; ++lane) {
        const auto runs = lane ? L::fabric_runs_b(int(W), tail, il, NL, nx, mx, HID, NR, 0, true, true)
                               : ds41::pf::fabric_runs(int(W), tail, il, NL, nx, mx, HID, NR, 0, true, true, true);
        const Tab t = table_of(runs, 4);
        CHECK(t.d.ncoll == ds41::rs::main_ncoll(nc, k, true));
        for (int rows : {int(W), int(W / 2)}) for (bool skip : {false, true}) {
            const int late = br::late_rows(rows, tail);
            const auto c = g6s2::ds41_chunk_collectives(ids, nx, uint32_t(rows), size_t(HID) * 2, 6144 * 2, il, uint32_t(late), skip);
            const auto p = L::b12_job(c.sizes, c.order, nc, k, rows, late, NR, size_t(HID) * 2, HID, 2, true, true, skip);
            CHECK(p.shard_chunk == (rows == int(W)));
            CHECK(p.shard_state == (rows == int(W) && !skip));
            CHECK(p.state_bytes == (p.shard_state ? state : 0));
            CHECK(p.job.sizes.size() <= size_t(t.d.ncoll));
            CHECK(p.job.sizes[size_t(nc)] == (rows == int(W) ? ds41::rowshard::y_mxfp8_bytes(W / NR, HID)
                                                                : size_t(ds41::rs::quarter(rows, NR)) * HID * 2));
            if (p.shard_state) {
                CHECK(p.job.sizes[size_t(nc + k)] == state);
                CHECK(std::find(p.job.order.begin(), p.job.order.end(), nc + k) != p.job.order.end());
                CHECK(g6::fab_slot_fits(t.d, nc + k, state));
            } else CHECK(std::find(p.job.order.begin(), p.job.order.end(), nc + k) == p.job.order.end());
            for (int q = 0; q < k; ++q) CHECK(g6::fab_slot_fits(t.d, nc + q, p.job.sizes[size_t(nc + q)]));
        }
        const auto old = L::fabric_runs_b(int(W), tail, il, NL, nx, mx, HID, NR, 0, true, false);
        mutant(lane ? "lane B missing row-shard state slot" : "lane A missing row-shard state slot",
               old.cfg.ncoll == t.d.ncoll ? "" : "state collective absent");
    }
    std::printf("lane A/B: row-shard state and MXFP8 AG geometry checked\n");
}

// the real scheduler: one decoder, then a long prompt; every Prefill quantum while a session decodes is at most W rows
static std::string sched(unsigned S, uint32_t dpr, long* quanta, uint32_t* idle_max)
{
    g6s2::Config c;
    c.pool_rows = 614400 * 4; c.max_rows = 48; c.max_sessions = S; c.mtp = 5;
    c.prefill_rows = W; c.two_lanes = true; c.decode_prefill_rows = dpr;
    g6s2::Scheduler s(c);
    auto admit = [&](uint64_t id, uint32_t prompt, uint32_t max_tokens) {
        const g6s2::Request r{id, prompt, max_tokens};
        g6s2::Grant g; g.key = s.next_key(); g.req = r;
        g.run = g6s2::Run{0, g6s2::grant_pages(c, r), 1, id * 0x9E3779B97F4A7C15ull + 1};
        g6s2::Commit cm; cm.epoch = c.epoch; cm.event = s.next_event(); cm.boundary = s.boundary(); cm.before_hash = s.digest(); cm.admit = g;
        s.apply(cm);
    };
    auto step = [&]() {
        const g6s2::Quantum q = s.next();
        if (q.kind == g6s2::Kind::Prefill) s.complete_prefill();
        else if (q.kind == g6s2::Kind::PromptTail) s.complete_tail(g6s2::Completion{q.keys[0], 1, false, 0});
        else if (q.kind == g6s2::Kind::Decode)
        {
            std::vector<g6s2::Completion> out;
            for (unsigned i = 0; i < q.count; ++i) out.push_back(g6s2::Completion{q.keys[i], 1, false, 0});
            s.complete_decode(out);
        }
        return q;
    };
    // alone: the first quantum of a 3W prompt is 2W (two lanes)
    admit(1, 3 * W + 50, 100000);
    *idle_max = step().rows;
    for (int i = 0; i < 8; ++i) step();   // session 1 reaches decode
    bool dec = false;
    for (const auto& x : s.slots()) dec |= x.phase == g6s2::Phase::Decode;
    if (!dec) return "session 1 never decodes";
    admit(2, 6 * W + 9, 16);
    for (int i = 0; i < 64; ++i)
    {
        bool decoding = false;
        for (const auto& x : s.slots()) decoding |= x.phase == g6s2::Phase::Decode;
        const g6s2::Quantum q = step();
        if (q.kind == g6s2::Kind::Prefill)
        {
            ++*quanta;
            if (decoding && q.rows > W) return "a " + std::to_string(q.rows) + "-row quantum (two lanes) while a session decodes";
        }
    }
    return "";
}
static void scheduler()
{
    for (unsigned S : {8u, 32u})
    {
        long qn = 0; uint32_t idle = 0;
        const std::string e = sched(S, L::decode_prefill_rows(2, W), &qn, &idle);
        if (!e.empty()) std::printf("scheduler SESS %u: %s\n", S, e.c_str());
        CHECK(e.empty() && idle == 2 * W && qn > 0);
        std::printf("scheduler SESS %u: alone quantum %u rows; %ld prefill quanta while decoding, all <= W\n", S, idle, qn);
        long q2 = 0; uint32_t i2 = 0;
        if (S == 32) mutant("two lanes while decoding (cap 0)", sched(S, 0, &q2, &i2));
    }
}

int main()
{
    parse_split();
    schedules();
    schedule_mutants();
    CHECK(seq_check(false).empty());
    mutant("one sequence for both lanes", seq_check(true));
    fabric();
    b12_fabric_both_lanes();
    scheduler();
    std::printf("mutants: %d of %d refused\n", refused, mutants);
    std::printf("test_ds41_lanes: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
