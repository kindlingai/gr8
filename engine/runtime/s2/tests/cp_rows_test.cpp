// G6_CP_STEP_ROWS / G6_CP_STEP_SEG (s2/cp_rows.h, Driver::step): a step past R carries at most n rows. Properties:
//   1. cp_after(bk, R) splits the contexts exactly as the engine's CP test does (bk[bucket_of(ctx)] > R, context_buckets.h).
//   2. the boot parse: off, the minimum (solo rows), the clamp to the widest S2 step, junk refused.
//   3. CpRound::pick on hand-made rounds: fits / no past-R session = everyone; cycles of past-R-first rounds of n / per
//      sessions and full-batch rounds below R; SEG never mixes; pick is pure; arrivals join the current cycle.
//   4. Driver::step with random requests (prompts below and past R, sessions crossing R while decoding, insert rules,
//      grow stalls): every request emits exactly its solo tokens (the cap only changes which sessions share a step), no
//      main step past R is wider than n, SEG never puts a session below R in such a step, two ranks fed the same events
//      agree on every digest and step, no decoding session waits more than a bounded number of rounds, and the trial
//      reaches steps past R wider than n with the knob off (so the cap is exercised).
//   5. not engaged = unchanged: no session past R, or n >= the widest round, gives the knob-off step sequence.
//   6. Config::validate refuses n < mtp + 1 and SEG without n; the digest and the bootstrap agreement name the knob only
//      when it is set (the default ones are unchanged), so ranks with different values refuse each other.
// Host only: no GPU, no CUDA, no pool. Build and run: see scripts/host_tests.sh.
#include "s2/bootstrap.h"
#include "s2/driver.h"
#include "context_buckets.h"
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <optional>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static uint32_t mix(uint32_t a, uint32_t b, uint32_t c)
{
    uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ (c + 0x1234567ull) * 0x94D049BB133111EBull;
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    return (uint32_t)x;
}
// the fake model of driver_grow_test: a pure function of (request, row position, row token)
constexpr int V = 2;
static int model(const s2::Row& r, int kind, uint32_t salt)
{
    int t = (int)(mix(salt, kind == s2::Step::MAIN ? r.pos : r.pos + 1, r.tok) % V);
    if (kind != s2::Step::MAIN && mix(salt, r.pos, r.tok + 99) % 3 == 0) t = (t + 1) % V;
    return t;
}

// ---- 1. cp_after against the engine's bucket rule ----
static void after_tests()
{
    long checked = 0;
    for (int cap : {4096, 65536 + 64, 212224, 262144})
        for (int base : {0, 2048, 4096})
            for (int R : {-1, 0, 64, 2048, 30016, 32768, 65536, 98304, 212224, 300000})
                for (int doublings : {7, 8}) {
                    int bk[9] = {};
                    const int nbk = context_buckets(bk, base, cap, R > 0 ? R : 0, doublings);
                    const uint64_t t = cp_after(bk, nbk, R);
                    auto bucket_of = [&](int ctx) { for (int k = 0; k < nbk; ++k) if (bk[k] >= ctx) return k; return nbk - 1; };
                    for (int ctx = 1; ctx <= cap + 64; ctx += (ctx < 4200 ? 1 : 61)) {
                        const bool engine = R >= 0 && bk[bucket_of(ctx)] > R;
                        CHECK((uint64_t(ctx) > t) == engine);
                        ++checked;
                    }
                    for (int k = 0; k < nbk; ++k) {   // the bucket boundaries themselves and one past them
                        CHECK((uint64_t(bk[k]) > t) == (R >= 0 && bk[bucket_of(bk[k])] > R));
                        CHECK((uint64_t(bk[k]) + 1 > t) == (R >= 0 && bk[bucket_of(bk[k] + 1)] > R));
                    }
                }
    CHECK(checked > 100000);
}

// ---- 2. the boot parse ----
static void parse_tests()
{
    std::string e;
    CHECK(cp_step_rows_env(nullptr, 8, 24, &e) == 0);
    CHECK(cp_step_rows_env("", 8, 24, &e) == 0);
    CHECK(cp_step_rows_env("0", 8, 24, &e) == 0);
    CHECK(cp_step_rows_env("8", 8, 24, &e) == 8);
    CHECK(cp_step_rows_env("12", 8, 24, &e) == 12);
    CHECK(cp_step_rows_env("24", 8, 24, &e) == 24);
    CHECK(cp_step_rows_env("32", 8, 24, &e) == 24);          // clamped to the widest S2 step
    e.clear(); CHECK(cp_step_rows_env("7", 8, 24, &e) == -1 && e.find("at least 8") != std::string::npos);
    for (const char* bad : {"x", "8x", "-8", "1e3", "99999"}) { e.clear(); CHECK(cp_step_rows_env(bad, 8, 24, &e) == -1 && !e.empty()); }
}

// ---- 3. CpRound on hand-made rounds ----
static std::vector<uint64_t> ids_in(const std::vector<CpCand>& c, const CpPick& p)
{
    std::vector<uint64_t> v;
    for (size_t i = 0; i < c.size(); ++i) if (p.in[i]) v.push_back(c[i].id);
    return v;
}
// runs `rounds` rounds over cand (fixed), returns the ids of each round
static std::vector<std::vector<uint64_t>> cycle(CpRound& r, const std::vector<CpCand>& cand, uint32_t rows, bool seg, int rounds)
{
    std::vector<std::vector<uint64_t>> out;
    for (int k = 0; k < rounds; ++k) {
        const CpPick p = r.pick(cand, rows, 3, seg);
        CHECK(p.in.size() == cand.size());
        const CpPick again = r.pick(cand, rows, 3, seg);    // pure
        CHECK(again.in == p.in && again.capped == p.capped && again.fresh == p.fresh);
        r.commit(p, cand);
        out.push_back(ids_in(cand, p));
    }
    return out;
}
using VV = std::vector<std::vector<uint64_t>>;
static void pick_tests()
{
    auto L = [](uint64_t id) { return CpCand{id, true}; };
    auto S = [](uint64_t id) { return CpCand{id, false}; };
    {   // off, no past-R session, everyone fits: everyone, never capped
        CpRound r;
        const std::vector<CpCand> eight{S(1), S(2), S(3), S(4), S(5), S(6), S(7), S(8)};
        CHECK(cycle(r, eight, 8, false, 2) == (VV{{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}}));
        const std::vector<CpCand> mixed{L(1), S(2), S(3), S(4)};
        CHECK(cycle(r, mixed, 0, false, 1) == (VV{{1, 2, 3, 4}}));
        CHECK(cycle(r, {L(1), S(2)}, 8, false, 2) == (VV{{1, 2}, {1, 2}}));       // 6 rows <= 8
        CHECK(cycle(r, mixed, 12, false, 1) == (VV{{1, 2, 3, 4}}));              // 12 rows <= 12
        CHECK(!r.pick(mixed, 12, 3, false).capped && r.cycle().empty());
    }
    {   // n = 8 (two sessions a round): the design note's examples
        CpRound r;
        CHECK(cycle(r, {L(1), S(2), S(3), S(4)}, 8, false, 4) == (VV{{1, 2}, {3, 4}, {1, 2}, {3, 4}}));
        CpRound r2;
        CHECK(cycle(r2, {L(1), S(2), S(3), S(4), S(5), S(6), S(7), S(8)}, 8, false, 4) ==
              (VV{{1, 2}, {3, 4, 5, 6, 7, 8}, {1, 2}, {3, 4, 5, 6, 7, 8}}));
        CpRound r3;
        CHECK(cycle(r3, {L(1), L(2), L(3), S(4), S(5), S(6), S(7), S(8)}, 8, false, 3) ==
              (VV{{1, 2}, {3, 4}, {5, 6, 7, 8}}));
        CpRound r4;   // past-R sessions first even when a session below R has the lower id
        CHECK(cycle(r4, {S(1), L(2), S(3), S(4)}, 8, false, 2) == (VV{{1, 2}, {3, 4}}));
        CpRound r5;
        CHECK(cycle(r5, {S(1), S(2), L(3), S(4)}, 8, false, 2) == (VV{{1, 3}, {2, 4}}));
        CpRound r6;   // n = 12: four a round
        CHECK(cycle(r6, {L(1), S(2), S(3), S(4), S(5), S(6), S(7), S(8)}, 12, false, 2) ==
              (VV{{1, 2, 3, 4}, {5, 6, 7, 8}}));
        CpRound r7;   // more past-R sessions than a round holds: they rotate, every one once per cycle
        CHECK(cycle(r7, {L(1), L(2), L(3), L(4), L(5)}, 8, false, 6) == (VV{{1, 2}, {3, 4}, {5}, {1, 2}, {3, 4}, {5}}));
    }
    {   // SEG: a past-R round never takes a session below R, even when everyone would fit
        CpRound r;
        CHECK(cycle(r, {L(1), S(2)}, 8, true, 4) == (VV{{1}, {2}, {1}, {2}}));
        CpRound r2;
        CHECK(cycle(r2, {L(1), L(2)}, 8, true, 2) == (VV{{1, 2}, {1, 2}}));     // all past R and they fit: one step
        CpRound r3;
        CHECK(cycle(r3, {L(1), L(2), L(3), S(4), S(5)}, 8, true, 3) == (VV{{1, 2}, {3}, {4, 5}}));
        CpRound r4;
        CHECK(cycle(r4, {S(1), S(2), S(3)}, 8, true, 1) == (VV{{1, 2, 3}}));    // no past-R session: everyone
    }
    {   // an arrival joins the current cycle; a session that left does not hold it open
        CpRound r;
        std::vector<CpCand> c{L(1), S(2), S(3), S(4)};
        CHECK(cycle(r, c, 8, false, 1) == (VV{{1, 2}}));
        c.push_back(S(5));
        CHECK(cycle(r, c, 8, false, 1) == (VV{{3, 4, 5}}));
        c.erase(c.begin() + 2);       // 3 finished
        CHECK(cycle(r, c, 8, false, 2) == (VV{{1, 2}, {4, 5}}));
        // a capped round followed by an uncapped one resets the cycle
        CHECK(cycle(r, {L(1), S(2)}, 8, false, 1) == (VV{{1, 2}}) && r.cycle().empty());
    }
}

// ---- 4 / 5. Driver::step ----
struct Req { uint64_t id; std::vector<int> prompt; uint32_t max_tokens; g6rt::InsertRule ins; };
struct Rank {
    Config c;
    std::unique_ptr<Driver> d;
    std::map<uint64_t, std::vector<int>> out;   // request id -> emitted tokens
    std::vector<uint64_t> trace;                // per step: a hash of its kind, sessions and rows (FNV)
    long cp_steps = 0, wide_cp = 0, mixed_cp = 0, outside = 0, grows = 0;
    int max_cp = 0;
    bool stall = false;
    uint64_t gcount = 0;
    Rank(const Config& cfg, bool stalls) : c(cfg), stall(stalls) {
        DriverHooks h;
        h.bind = [](const Grant&) {};
        h.retire = [](const Slot&) {};
        h.prefill = [this](const Slot& s, const Payload&, uint32_t first, uint32_t rows) {
            if (uint64_t(first) + rows > uint64_t(s.run.pages) * PAGE) ++outside;
        };
        h.run = [this](const s2::Step& st) {
            std::vector<int> ids;
            uint64_t hsh = 1469598103934665603ull;
            auto add = [&](uint64_t v) { hsh = (hsh ^ v) * 1099511628211ull; };
            add(uint64_t(st.kind)); add(uint64_t(st.M()));
            for (size_t j = 0; j < st.sess.size(); ++j) { add(uint64_t(st.sess[j])); add(uint64_t(st.rows[j])); add(uint64_t(st.ctx[j])); }
            for (const auto& r : st.r) {
                const Slot& s = d->state().slots()[r.sess];
                if (uint64_t(r.pos) >= uint64_t(s.run.pages) * PAGE) ++outside;
                ids.push_back(model(r, st.kind, uint32_t(s.req.id)));
            }
            if (st.kind == s2::Step::MAIN) {
                int ctx = 0;
                for (int x : st.ctx) ctx = std::max(ctx, x);
                if (uint64_t(ctx) > c.cp_after) {            // a step past R (Dispatch: the largest of its sessions' buckets)
                    ++cp_steps; max_cp = std::max(max_cp, st.M());
                    if (st.M() > 8) ++wide_cp;
                    // a round's main step: rows = kq + 1, ctx = length + kq. A session the cap counts below R
                    // (length + mtp <= cp_after) in a step past R
                    for (size_t j = 0; j < st.sess.size() && st.sess.size() > 1; ++j)
                        if (uint64_t(st.ctx[j] - st.rows[j] + 1 + int(c.mtp)) <= c.cp_after) { ++mixed_cp; break; }
                }
            }
            trace.push_back(hsh);
            return ids;
        };
        h.emit = [this](Key k, const int* ids, int n, int) {
            const Slot& s = d->state().slots()[k.slot];
            auto& o = out[s.req.id];
            const int cnt = std::min<int>(n, int(s.req.max_tokens - o.size()));
            o.insert(o.end(), ids, ids + cnt);
            return Emission{uint32_t(cnt), o.size() == s.req.max_tokens};
        };
        // the runs hold their worst case from the start; a stall trial sits sessions out now and then (both ranks alike)
        h.grow = [this](const Slot& s, uint64_t rows) -> std::optional<Run> {
            ++grows;
            if (rows > uint64_t(s.run.pages) * PAGE) ++outside;
            if (stall && mix(uint32_t(s.req.id), uint32_t(++gcount), 5) % 6 == 0) return std::nullopt;
            return s.run;
        };
        d = std::make_unique<Driver>(c, h);
    }
};
static std::vector<int> solo(const Req& r, int K)
{
    s2::Hooks h;
    h.run = [&](const s2::Step& st) {
        std::vector<int> ids;
        for (const auto& x : st.r) ids.push_back(model(x, st.kind, uint32_t(r.id)));
        return ids;
    };
    s2::Session s;
    s.prompt = r.prompt; s.first = int(r.prompt.size()) - 1; s.chunk = 4; s.n_gen = int(r.max_tokens); s.ins = r.ins;
    std::vector<s2::Stats> st;
    return s2::mtp_batch({s}, K, 8, h, st)[0];
}
struct Trial {
    long cp_steps = 0, wide_cp = 0, mixed_cp = 0, max_wait = 0, capped_rounds = 0, decode_rounds = 0;
    int max_cp = 0;
    std::vector<uint64_t> trace;
    std::map<uint64_t, std::vector<int>> out;
};
// one trial: the requests admitted FIFO as slots free up, `steps_per_event` steps between events, two ranks a / b
static Trial run_trial(Config c, const std::vector<Req>& reqs, bool stalls, uint64_t seed)
{
    Rank a(c, stalls), b(c, stalls);
    std::deque<Req> queue(reqs.begin(), reqs.end());
    std::map<uint64_t, Req> all;
    for (const auto& r : reqs) all[r.id] = r;
    std::map<uint64_t, long> waited;   // request id -> decode rounds in a row it sat out (cap only: no stall trials)
    Trial t;
    auto rnd = [&](int lo, int hi) {
        uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
        return lo + int(z % uint64_t(hi - lo + 1));
    };
    long guard = 0;
    for (;;) {
        const Scheduler& s = a.d->state();
        Commit cm;
        cm.epoch = c.epoch; cm.event = s.next_event(); cm.boundary = s.boundary();
        if (!queue.empty() && s.resident() < c.limit()) {
            const Req& r = queue.front();
            const Request rq{r.id, uint32_t(r.prompt.size()), r.max_tokens};
            cm.admit = Grant{s.next_key(), rq, Run{0, reserve_pages(c, rq), 1, r.id * 0x9E3779B97F4A7C15ull + 1}, 0};
        }
        for (Rank* x : {&a, &b}) {
            Commit mine = cm; mine.before_hash = x->d->state().digest();
            Payload p; p.prompt = cm.admit ? all[cm.admit->req.id].prompt : std::vector<int>{}; p.chunk = 4;
            if (cm.admit) p.ins = all[cm.admit->req.id].ins;
            x->d->apply(mine, cm.admit ? &p : nullptr);
        }
        CHECK(a.d->state().digest() == b.d->state().digest());
        if (cm.admit) queue.pop_front();
        if (!a.d->state().resident()) { if (queue.empty()) break; else continue; }
        for (int k = rnd(1, 6); k > 0 && a.d->state().resident(); --k) {
            const std::vector<Key> dec = a.d->state().decoding();
            const auto qa = a.d->step(), qb = b.d->step();
            CHECK(qa.kind == qb.kind && qa.count == qb.count);
            for (unsigned i = 0; i < qa.count && i < qb.count; ++i) CHECK(qa.keys[i] == qb.keys[i]);
            // a resident session always has something to run, unless every decoder drew a grow stall this time
            if (qa.kind == Kind::Idle && !stalls) { CHECK(!"idle with a resident session"); fails += 1000; break; }
            if (qa.kind == Kind::Decode) {
                ++t.decode_rounds;
                if (qa.count < dec.size()) ++t.capped_rounds;
                for (const Key& kk : dec) {
                    bool in = false;
                    for (unsigned i = 0; i < qa.count; ++i) in |= qa.keys[i] == kk;
                    const uint64_t id = a.d->state().slots()[kk.slot].req.id;
                    long& w = waited[id];
                    w = in ? 0 : w + 1;
                    if (!stalls) t.max_wait = std::max(t.max_wait, w);
                }
            }
        }
        CHECK(a.trace == b.trace);
        if (++guard > 400000 || fails > 100) { CHECK(!"no progress"); break; }
    }
    CHECK(a.outside == 0 && b.outside == 0);
    t.cp_steps = a.cp_steps; t.wide_cp = a.wide_cp; t.mixed_cp = a.mixed_cp; t.max_cp = a.max_cp;
    t.trace = a.trace; t.out = a.out;
    return t;
}
static void driver_tests()
{
    Config base;
    base.epoch = 1; base.mtp = 2; base.max_rows = 24; base.max_sessions = 8; base.prefill_rows = 64; base.two_lanes = false;
    base.pool_rows = 64 * 4096;
    uint64_t seed = 7;
    auto rnd = [&](int lo, int hi) {
        uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
        return lo + int(z % uint64_t(hi - lo + 1));
    };
    long cp_on = 0, wide_off = 0, capped = 0, crossed = 0, inserts = 0, max_wait = 0, seg_capped = 0, rounds = 0;
    for (int trial = 0; trial < 60; ++trial) {
        const uint64_t T = uint64_t(rnd(3, 8)) * 64;     // R: a bucket boundary, contexts past it run the CP block
        std::vector<Req> reqs;
        const int n = rnd(4, 14);
        for (int i = 0; i < n; ++i) {
            Req r; r.id = uint64_t(trial) * 100 + i + 1;
            const int kind = rnd(0, 3);                  // 0: past R from the start, 1: crosses R while decoding, else short
            const int plen = kind == 0 ? rnd(int(T) + 1, int(T) + 300) : kind == 1 ? rnd(int(T) - 120, int(T) - 10) : rnd(2, 120);
            for (int k = 0; k < plen; ++k) r.prompt.push_back(int(mix(uint32_t(r.id), k, 7) % V));
            r.max_tokens = uint32_t(kind == 1 ? rnd(150, 300) : rnd(1, 220));
            if (kind == 1) ++crossed;
            if (i % 3 == 1) {
                r.ins.after = rnd(0, int(r.max_tokens));
                r.ins.unless = rnd(0, 3) == 0 ? 1 : -1;
                for (int k = rnd(1, 9); k > 0; --k) r.ins.tokens.push_back(rnd(0, V - 1));
            }
            reqs.push_back(r);
        }
        const bool stalls = trial % 3 == 2;
        const uint64_t ev = uint64_t(rnd(1, 1 << 30));
        Config off = base; off.cp_after = T;              // knob off: cp_after is only read with it on
        Config on8 = base; on8.cp_step_rows = 8; on8.cp_after = T;
        Config on12 = base; on12.cp_step_rows = 12; on12.cp_after = T;
        Config seg = on8; seg.cp_seg = true;
        const Trial t_off = run_trial(off, reqs, stalls, ev), t8 = run_trial(on8, reqs, stalls, ev),
                    t12 = run_trial(on12, reqs, stalls, ev), ts = run_trial(seg, reqs, stalls, ev);
        auto got = [](const Trial& t, uint64_t id) { const auto it = t.out.find(id); return it == t.out.end() ? std::vector<int>{} : it->second; };
        for (const auto& r : reqs) {
            const auto want = solo(r, int(base.mtp));
            CHECK(got(t_off, r.id) == want);
            CHECK(got(t8, r.id) == want);
            CHECK(got(t12, r.id) == want);
            CHECK(got(ts, r.id) == want);
            if (r.ins.on() && int(r.ins.after) < int(want.size()) && want[size_t(r.ins.after)] == r.ins.tokens[0]) ++inserts;
        }
        CHECK(t8.max_cp <= 8 && ts.max_cp <= 8 && t12.max_cp <= 12);   // the invariant
        CHECK(ts.mixed_cp == 0);                                        // SEG: no session below R in a step past R
        cp_on += t8.cp_steps; wide_off += t_off.wide_cp; capped += t8.capped_rounds + t12.capped_rounds;
        seg_capped += ts.capped_rounds; rounds += t8.decode_rounds;
        max_wait = std::max({max_wait, t8.max_wait, t12.max_wait, ts.max_wait});
        CHECK(t_off.capped_rounds == 0 || stalls);
        // 5. not engaged: no session ever past R (cp_after past every context), or n = the widest round (non-SEG)
        Config far = on8; far.cp_after = 1u << 30;
        Config wide = base; wide.cp_step_rows = 24; wide.cp_after = T;
        Config off_far = base; off_far.cp_after = 1u << 30;
        CHECK(run_trial(far, reqs, stalls, ev).trace == run_trial(off_far, reqs, stalls, ev).trace);
        CHECK(run_trial(wide, reqs, stalls, ev).trace == t_off.trace);
    }
    // the trials reach what the cap is for: knob off, steps past R wider than 8; knob on, capped rounds
    CHECK(cp_on > 1000 && wide_off > 100 && capped > 100 && seg_capped > 100 && crossed > 20 && inserts > 10);
    CHECK(max_wait <= 2 * long(MAX_SESSIONS));   // a cycle: ceil(past / c) + 1 rounds, arrivals join it
    std::printf("cp_rows_test driver: %ld decode rounds at n=8, %ld main steps past R at n=8, %ld wider than 8 with the knob off, "
                "%ld capped rounds (n=8 + 12), %ld SEG capped, longest wait %ld rounds\n",
                rounds, cp_on, wide_off, capped, seg_capped, max_wait);
}

// ---- 6. config, digest, agreement ----
static void config_tests()
{
    Config c;
    c.mtp = 2;
    auto bad = [](Config x) { try { x.validate(); return false; } catch (const std::exception&) { return true; } };
    Config x = c; x.cp_step_rows = 2; CHECK(bad(x));
    x.cp_step_rows = 3; CHECK(!bad(x));
    x = c; x.cp_seg = true; CHECK(bad(x));
    x.cp_step_rows = 8; CHECK(!bad(x));
    // the default digest ignores cp_after (read only with the knob); a set knob changes it
    Config d0 = c, d1 = c; d1.cp_after = 123;
    CHECK(Scheduler(d0).digest() == Scheduler(d1).digest());
    Config e8 = c; e8.cp_step_rows = 8; e8.cp_after = 65536;
    Config e12 = e8; e12.cp_step_rows = 12;
    Config es = e8; es.cp_seg = true;
    Config ea = e8; ea.cp_after = 32768;
    const uint64_t h0 = Scheduler(d0).digest(), h8 = Scheduler(e8).digest();
    CHECK(h0 != h8 && h8 != Scheduler(e12).digest() && h8 != Scheduler(es).digest() && h8 != Scheduler(ea).digest());
    BootstrapConfig b;
    b.scheduler = c; b.ranks = 4; b.runtime_signature = "sig";
    const std::string a0 = b.agreement();
    CHECK(a0.find("cpsr") == std::string::npos);
    b.scheduler = e8;
    const std::string a8 = b.agreement();
    CHECK(a8.find(" cpsr 8 after 65536 ctl ") != std::string::npos);
    b.scheduler = es;
    CHECK(b.agreement().find(" cpsr 8 seg after 65536 ctl ") != std::string::npos);
    b.scheduler = e12;
    CHECK(b.agreement() != a8);
}

int main()
{
    after_tests();
    parse_tests();
    pick_tests();
    config_tests();
    driver_tests();
    if (fails) { std::fprintf(stderr, "cp_rows_test: %d FAILED\n", fails); return 1; }
    std::printf("cp_rows_test: PASS\n");
    return 0;
}
