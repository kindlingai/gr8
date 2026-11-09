// The C++ facade over the Rust planner (s2_batch.h Batch, engine/scheduler-rs). The plain build's --spec-adaptive uses
// spec_len.h's SpecLen, and the planner keeps its own: a new SpecLen's settings and seeds, and the confidence buckets,
// must be the same in both. A hook that calls the Batch back throws, and a slot out of range is refused. Host
// only.
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <limits>
#include "s2_batch.h"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

template <class A, class B> static bool same(const A& a, const B& b) { return sizeof a == sizeof b && std::memcmp(&a, &b, sizeof a) == 0; }

int main()
{
    const g6rt::SpecLen s;
    g6s2_spec_len r;
    g6s2_spec_len_default(&r);
    CHECK(r.on == s.on && r.n_levels == (int)s.levels.size());
    CHECK(r.draft_round == s.draft_round && r.draft_each == s.draft_each);
    CHECK(same(r.table, s.table) && same(r.seed, s.seed) && same(r.pn, s.pn) && same(r.pk, s.pk));
    CHECK(same(r.bn, s.bn) && same(r.bk, s.bk));
    CHECK(r.pos_decay == s.pos_decay && r.bucket_decay == s.bucket_decay && r.prior == s.prior);
    CHECK(r.rounds == s.rounds && r.pred_sum == s.pred_sum && r.got_sum == s.got_sum);
    for (int k = 0; k <= g6rt::SpecLen::KX; ++k) CHECK(r.chosen[k] == s.chosen[k]);
    // every bucket edge from both sides, a sweep, and the non-finite values
    const double edge[] = {0.2, 0.4, 0.6, 0.75, 0.88, 0.95, 0.99};
    long n = 0;
    auto at = [&](float lp) { ++n; CHECK(g6s2_spec_len_bucket(lp) == g6rt::SpecLen::bucket(lp)); };
    for (double e : edge)
    {
        const float lp = (float)std::log(e);
        at(lp); at(std::nextafter(lp, 0.f)); at(std::nextafter(lp, -1.f));
    }
    for (float lp = -12.f; lp <= 0.5f; lp += 1.f / 1024) at(lp);
    at(std::numeric_limits<float>::quiet_NaN()); at(-INFINITY); at(INFINITY);
    // a hook that reads the Batch back throws, and the call that ran it rethrows
    {
        s2::Batch* self = nullptr;
        bool read_back = false;
        s2::Hooks h;
        h.run = [&](const s2::Step& st) { return std::vector<int>(st.r.size(), 7); };
        h.emit = [&](int q, const int*, int, int) {
            if (read_back) (void)self->stats(q);
            return false;
        };
        s2::Batch b(3, 32, h);
        self = &b;
        s2::Session x;
        x.prompt = {1, 2, 3, 4};
        x.first = 3;
        x.n_gen = 64;
        b.admit({0}, {x});
        read_back = true;
        bool threw = false;
        try { b.round({0}); } catch (const std::logic_error&) { threw = true; }
        CHECK(threw);
        read_back = false;
        CHECK(b.round({0}));   // the planner stopped at the emit and goes on
        // slots out of range: refused with a message, read as empty
        threw = false;
        try { b.cancel(s2::S2N_SLOTS); } catch (const std::runtime_error& e) { threw = std::strstr(e.what(), "out of range") != nullptr; }
        CHECK(threw);
        threw = false;
        try { b.promote(-1); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        CHECK(!b.active(99) && !b.finished(99) && b.length(99) == 0 && b.result(99).empty());
    }
    std::printf("planner_facade_test: %s (%ld bucket checks)\n", fails ? "FAIL" : "PASS", n);
    return fails ? 1 : 0;
}
