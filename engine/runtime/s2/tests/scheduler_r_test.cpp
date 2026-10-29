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
