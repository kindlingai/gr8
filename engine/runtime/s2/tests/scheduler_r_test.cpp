// R per grant (Request::r_rows): a grant carries the R its run takes, the scheduler accepts only the engine's R or
// Config::r2_rows, and the digest names it, so ranks that disagree fail at the grant. A grant at the engine's R hashes
// as before r_rows existed. Host only. Build and run: see scripts/host_tests.sh.
#include "s2/scheduler.h"
#include <cstdio>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static Config cfg()
{
    Config c;
    c.window_pages = 40; c.r2_rows = 16384;
    return c;
}

// a grant of request 1 at R = r rows; false when apply refuses it
static bool admit(Scheduler& s, uint32_t r)
{
    const Config& c = s.config();
    Grant g;
    g.key = s.next_key();
    g.req = Request{1, 20000, 4000};
    g.req.r_rows = r;
    g.run = Run{0, reserve_pages(c, g.req) + c.window_pages, 1, 7};
    Commit cm;
    cm.epoch = c.epoch; cm.event = s.next_event(); cm.boundary = s.boundary(); cm.before_hash = s.digest(); cm.admit = g;
    try { s.apply(cm); } catch (const std::exception&) { return false; }
    return true;
}

int main()
{
    Scheduler big(cfg()), small(cfg()), odd(cfg());
    CHECK(admit(big, 0) && admit(small, 16384));
    CHECK(!admit(odd, 8192));
    CHECK(big.digest() != small.digest());
    CHECK(big.slots()[0].req.r_rows == 0 && small.slots()[0].req.r_rows == 16384);

    // a decoding session takes the smaller R: its slot's R and run change, and the digest with them
    {
        Scheduler a(cfg());
        CHECK(admit(a, 0));
        a.skip_prefill(a.slots()[0].key);
        const Quantum q = a.next();   // its prompt tail: then it decodes
        CHECK(q.kind == Kind::PromptTail);
        a.complete_tail(Completion{q.keys[0], 1, false, 0});
        CHECK(a.slots()[0].phase == Phase::Decode);
        const Slot was = a.slots()[0];
        auto shrink = [&](uint32_t r, uint32_t version) {
            Commit cm;
            cm.epoch = a.config().epoch; cm.event = a.next_event(); cm.boundary = a.boundary(); cm.before_hash = a.digest();
            Run run = was.run; run.list_hash ^= 1; run.table_version = version;
            cm.shrink = Shrink{was.key, r, run};
            try { a.apply(cm); } catch (const std::exception&) { return false; }
            return true;
        };
        const uint64_t before = a.digest();
        CHECK(!shrink(8192, was.run.table_version) && !shrink(16384, was.run.table_version + 1));
        CHECK(shrink(16384, was.run.table_version));
        CHECK(a.slots()[0].req.r_rows == 16384 && a.slots()[0].run.list_hash == (was.run.list_hash ^ 1) && a.digest() != before);
        CHECK(!shrink(16384, was.run.table_version));   // once only
    }

    // without r2_rows only the engine's R is granted, and r2_rows needs the window
    Config plain = cfg(); plain.r2_rows = 0;
    Scheduler p(plain), q(plain);
    CHECK(admit(p, 0) && !admit(q, 16384));
    Config bad = cfg(); bad.window_pages = 0;
    bool refused = false;
    try { bad.validate(); } catch (const std::exception&) { refused = true; }
    CHECK(refused);
    std::printf("scheduler_r_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
