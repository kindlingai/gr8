// Growth on demand through Driver::step (driver.h) and the Rust pool (s2/pool.h): runs start with their prompt + 1 +
// guard rows, grow before each round, sit a round out when the pool would be unsafe, and give their pages back when
// they finish. Every request must emit exactly what it emits alone, no step may touch a row past its session's pages,
// and two ranks driven by the same events must agree on every digest. A third of the requests carry an insert rule
// (s2_batch.h): their tokens must be the solo run's with the same rule, the growth must cover the inserted tokens' rows,
// and a rank that misses the rule must disagree at the next digest after it fires. Host only.
// The fake model is a pure function of (request, row position, row token), as batch_round_test's, so any difference
// from the solo run comes from the scheduler, the driver or the pool.
#include "s2/sessions.h"
#include "s2/pool.h"
#include <cstdio>
#include <deque>
#include <map>
#include <memory>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static uint32_t mix(uint32_t a, uint32_t b, uint32_t c)
{
    uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ (c + 0x1234567ull) * 0x94D049BB133111EBull;
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    return (uint32_t)x;
}
constexpr int V = 2;
static int model(const s2::Row& r, int kind, uint32_t salt)
{
    int t = (int)(mix(salt, kind == s2::Step::MAIN ? r.pos : r.pos + 1, r.tok) % V);
    if (kind != s2::Step::MAIN && mix(salt, r.pos, r.tok + 99) % 3 == 0) t = (t + 1) % V;
    return t;
}
struct Req { uint64_t id; std::vector<int> prompt; uint32_t max_tokens; g6rt::InsertRule ins; };

// one rank: a scheduler/driver over its own pool
struct Rank {
    Config c;
    RunPool pool;
    std::unique_ptr<Driver> d;
    std::map<uint64_t, std::vector<int>> out;   // request id -> emitted tokens
    long steps = 0, outside = 0;
    Rank(const Config& cfg, uint32_t pages) : c(cfg), pool(pages, pages, pages * 64) {
        DriverHooks h;
        h.bind = [](const Grant&) {};
        h.retire = [](const Slot&) {};
        h.prefill = [this](const Slot& s, const Payload&, uint32_t first, uint32_t rows) {
            if (uint64_t(first) + rows > uint64_t(s.run.pages) * PAGE) ++outside;
        };
        h.run = [this](const s2::Step& st) {
            std::vector<int> ids;
            for (const auto& r : st.r) {
                const Slot& s = d->state().slots()[r.sess];
                if (uint64_t(r.pos) >= uint64_t(s.run.pages) * PAGE) ++outside;   // a row past the session's pages
                ids.push_back(model(r, st.kind, uint32_t(s.req.id)));
            }
            ++steps;
            return ids;
        };
        h.emit = [this](Key k, const int* ids, int n, int) {
            const Slot& s = d->state().slots()[k.slot];
            auto& o = out[s.req.id];
            const int c = std::min<int>(n, int(s.req.max_tokens - o.size()));
            o.insert(o.end(), ids, ids + c);
            return Emission{uint32_t(c), o.size() == s.req.max_tokens};
        };
        h.grow = [this](const Slot& s, uint64_t rows) { return pool.grow(s.run, rows); };
        h.finished = [this](const Slot& s) { pool.release(s.run); };
        d = std::make_unique<Driver>(c, h);
    }
};

// the solo tokens of a request: its prompt tail and rounds alone in a Batch (prefilled up to the last prompt token)
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

// the request line's `ins` field (sessions.h parse_request)
static void parse_tests()
{
    auto req = [](const std::string& head) { return parse_request(head + "\n1,2,3\n"); };
    const auto r = req("n_gen 9 temp 0 seed 1 min 0 stop - top_p 1 plp 2 ins 4:154842:198,154842,271");
    CHECK(r.ins.after == 4 && r.ins.unless == 154842 && (r.ins.tokens == std::vector<int>{198, 154842, 271}) && r.plp == 2);
    CHECK(req("n_gen 9 temp 0 seed 1 min 0 stop - ins 0:-1:5").ins.unless == -1);
    CHECK(!req("n_gen 9 temp 0 seed 1 min 0 stop -").ins.on());
    for (const char* bad : {"ins 4:1", "ins 4:1:", "ins x:1:2", "ins 4:-2:1", "lp 1 ins 4:1:2", "ins 4:1:2 plp 1"}) {
        bool threw = false;
        try { req(std::string("n_gen 9 temp 0 seed 1 min 0 stop - ") + bad); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }
    std::string many = "ins 1:-1:1";
    for (int i = 1; i <= g6rt::INSERT_MAX; ++i) many += ",1";
    bool threw = false;
    try { req("n_gen 9 temp 0 seed 1 min 0 stop - " + many); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

int main()
{
    parse_tests();
    Config c;
    c.epoch = 1; c.mtp = 2; c.max_rows = 24; c.max_sessions = 8; c.prefill_rows = 64; c.two_lanes = false;
    uint64_t seed = 1;
    auto rnd = [&](int lo, int hi) {
        uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
        return lo + int(z % uint64_t(hi - lo + 1));
    };
    long total_stalls = 0, total_grows = 0, requests = 0, max_live = 0, blind_caught = 0, inserts = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const uint32_t pages = uint32_t(rnd(12, 40));
        c.pool_rows = pages * 64;
        Rank a(c, pages), b(c, pages), blind(c, pages);   // blind: the same events, its payloads without the rules
        std::deque<Req> queue;
        const int n = rnd(4, 14);
        for (int i = 0; i < n; ++i) {
            Req r; r.id = uint64_t(trial) * 100 + i + 1;
            const int plen = rnd(2, 200);
            for (int k = 0; k < plen; ++k) r.prompt.push_back(int(mix(uint32_t(r.id), k, 7) % V));
            r.max_tokens = uint32_t(rnd(1, 300));
            if (reserve_rows(c, Request{r.id, uint32_t(plen), r.max_tokens}) > uint64_t(pages) * 64) r.max_tokens = 8;
            if (i % 3 == 1) {                       // an insert rule, now and then longer than the guard rows
                r.ins.after = rnd(0, int(r.max_tokens));
                r.ins.unless = rnd(0, 3) == 0 ? 1 : -1;
                for (int k = rnd(1, 9); k > 0; --k) r.ins.tokens.push_back(rnd(0, V - 1));
            }
            queue.push_back(r);
        }
        std::map<uint64_t, Req> all;
        for (const auto& r : queue) all[r.id] = r;
        long guard = 0;
        for (;;) {
            // an event: retiring slots free, the FIFO head admitted when it fits; both ranks make the same calls
            const Scheduler& s = a.d->state();
            Commit cm;
            cm.epoch = 1; cm.event = s.next_event(); cm.boundary = s.boundary();
            if (!queue.empty() && s.resident() < c.limit()) {
                const Req& r = queue.front();
                const Request rq{r.id, uint32_t(r.prompt.size()), r.max_tokens};
                const uint64_t rows = grant_rows(c, rq), worst = reserve_rows(c, rq);
                CHECK(a.pool.fits(rows, worst) == b.pool.fits(rows, worst));
                if (a.pool.fits(rows, worst)) {
                    const Run run = a.pool.reserve(rows, worst);
                    CHECK(b.pool.reserve(rows, worst) == run);
                    cm.admit = Grant{s.next_key(), rq, run, 0};
                }
            }
            if (blind.d && cm.admit) CHECK(blind.pool.reserve(grant_rows(c, cm.admit->req), reserve_rows(c, cm.admit->req)) == cm.admit->run);
            for (Rank* x : {&a, &b, &blind}) {
                if (!x->d) continue;
                x->d->pool_digest(x->pool.digest());
                Commit mine = cm; mine.before_hash = x->d->state().digest();
                Payload p; p.prompt = cm.admit ? all[cm.admit->req.id].prompt : std::vector<int>{}; p.chunk = 4;
                if (cm.admit && x != &blind) p.ins = all[cm.admit->req.id].ins;
                x->d->apply(mine, cm.admit ? &p : nullptr);
            }
            CHECK(a.d->state().digest() == b.d->state().digest());
            if (cm.admit) { queue.pop_front(); ++requests; }
            max_live = std::max<long>(max_live, a.d->state().resident());
            if (!a.d->state().resident()) { if (queue.empty()) break; else continue; }
            // a few steps until the next event
            for (int k = rnd(1, 6); k > 0 && a.d->state().resident(); --k) {
                const auto qa = a.d->step(), qb = b.d->step();
                if (blind.d) {
                    blind.d->step();
                    // once a rule fired, the rank that missed it holds another state: the next event would refuse it
                    blind.d->pool_digest(blind.pool.digest()); a.d->pool_digest(a.pool.digest());
                    bool fired = false;
                    for (const auto& s : a.d->state().slots()) fired |= s.phase != Phase::Free && s.inserted > 0;
                    if (fired) { CHECK(blind.d->state().digest() != a.d->state().digest()); ++blind_caught; blind.d.reset(); }
                }
                CHECK(qa.kind == qb.kind && qa.count == qb.count);
                CHECK(qa.kind != Kind::Idle);       // a resident session always has something to run: no deadlock
                if (qa.kind == Kind::Idle) { fails += 1000; break; }
            }
            CHECK(a.pool.digest() == b.pool.digest());
            if (++guard > 200000 || fails > 100) { CHECK(!"no progress"); break; }
        }
        for (const auto& kv : all) {
            const auto& r = kv.second;
            const auto& o = a.out[kv.first];
            if (r.ins.on() && int(r.ins.after) < int(o.size()) && o[size_t(r.ins.after)] == r.ins.tokens[0]) ++inserts;
            CHECK(a.out[kv.first] == solo(kv.second, int(c.mtp)));
            CHECK(a.out[kv.first] == b.out[kv.first]);
        }
        CHECK(a.outside == 0);
        const auto st = a.pool.stats();
        CHECK(st.live_runs == 0 && st.held_pages == 0);
        total_stalls += long(st.stalls); total_grows += long(st.grows);
    }
    CHECK(total_stalls > 0 && total_grows > 0 && max_live >= 3 && inserts > 20 && blind_caught > 10);
    std::printf("driver_grow_test: %s (%ld requests, %ld growths, %ld rounds sat out, up to %ld sessions at once, %ld inserts, "
                "%ld blind ranks caught)\n", fails ? "FAIL" : "PASS", requests, total_grows, total_stalls, max_live, inserts, blind_caught);
    return fails ? 1 : 0;
}
