// RunPool page lists (pool.h): the lowest free pages first, the same lists on every rank given the same calls, a hit that
// adopts the start of a retained list, eviction of whole retained runs, and admission that needs only enough free pages
// anywhere. Host only.
#include "s2/pool.h"
#include <cstdio>
using namespace g6s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static std::vector<int> iota_list(int a, int n) { std::vector<int> v(n); for (int i = 0; i < n; ++i) v[i] = a + i; return v; }
static std::vector<int> seq(int n, int salt) { std::vector<int> v(n); for (int i = 0; i < n; ++i) v[i] = (i * 7 + salt) % 1000; return v; }
static g6rt::PrefixReuse finished(const std::vector<int>& prompt, int gen)
{
    g6rt::PrefixReuse st;
    const int n = int(prompt.size());
    const auto g = seq(gen, 77);
    st.prefill_rows(0, n - 1, true);
    st.main_rows(n - 1, 1 + gen);
    st.draft_rows(n - 1, 1 + gen);
    st.finish(prompt, g, gen, true);
    return st;
}
struct Rng {
    uint64_t s;
    uint64_t next() { uint64_t z = (s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
    int range(int lo, int hi) { return lo + int(next() % uint64_t(hi - lo + 1)); }
};

int main()
{
    const uint32_t below = 32768;
    g6rt::PrefixReuse rules; rules.grid = 1024;

    // lowest free first, and a scattered list where the old first-fit allocator stalled on fragmentation
    {
        RunPool p(100, 100, 100 * 64);
        std::vector<Run> r;
        for (int i = 0; i < 10; ++i) r.push_back(p.reserve(10 * 64));
        CHECK(p.list(r[3]) == iota_list(30, 10) && p.room() == 0);
        for (int i = 0; i < 10; i += 2) p.release(r[i]);     // 50 free pages in five gaps of 10
        AdmissionQueue q(Config{1, 100 * 64, 24, 8, 2, UINT32_MAX, 4096, true});
        q.push(Request{1, 30 * 64 - 4 - 10, 10});
        const auto s = q.check(1, true, p.fits(30 * 64, 30 * 64), p.room());
        CHECK(s && s->why == Stall::None && s->requested == 30 && p.room() == 50);
        const Run big = p.reserve(30 * 64);
        std::vector<int> want;
        for (int g = 0; g < 3; ++g) for (int i = 0; i < 10; ++i) want.push_back(g * 20 + i);
        CHECK(p.list(big) == want && big.first_page == 0 && p.room() == 20);
        q.push(Request{2, 30 * 64 - 4 - 10, 10});
        q.admitted(1);
        const auto s2 = q.check(2, true, p.fits(30 * 64, 30 * 64), p.room());
        CHECK(s2 && s2->why == Stall::Capacity);
    }

    // a hit adopts the first ceil(skip / 64) pages of the retained list and lets the rest go. The new pages are the lowest free.
    {
        RunPool p(400, 400, 400 * 64), w(400, 400, 400 * 64);
        p.keep(rules); w.keep(rules);
        Run f1 = p.reserve(20 * 64), f2 = p.reserve(40 * 64);
        CHECK(w.reserve(20 * 64) == f1 && w.reserve(40 * 64) == f2);
        p.release(f1); w.release(f1);                       // pages 0..19 free, 20..59 live
        const auto A = seq(4000, 3);
        Run a = p.reserve(4000 + 68);                         // 64 pages: 0..19, then 60..103
        CHECK(w.reserve(4000 + 68) == a);
        std::vector<int> la = iota_list(0, 20), tail = iota_list(60, 44);
        la.insert(la.end(), tail.begin(), tail.end());
        CHECK(p.list(a) == la);
        p.retain(a, finished(A, 8)); w.retain(a, finished(A, 8));
        p.release(f2); w.release(f2);                         // 20..59 free and spare: the next run takes them first
        std::vector<int> B = A; const auto b2 = seq(3000, 9); B.insert(B.end(), b2.begin(), b2.end());
        const PoolHit h = p.match(B, B.size() + 68, int(B.size()) - 1), hw = w.match(B, B.size() + 68, int(B.size()) - 1);
        CHECK(h.skip == 3072 && hw.skip == h.skip && hw.age == h.age);
        const Run b = p.take(h, B.size() + 68);
        CHECK(w.take(hw, B.size() + 68) == b);
        std::vector<int> lb(la.begin(), la.begin() + 48);    // 3072 / 64 = 48 adopted pages: 0..19, 60..87
        const auto rest = iota_list(20, 40); lb.insert(lb.end(), rest.begin(), rest.end());
        const auto more = iota_list(88, int(b.pages) - 88); lb.insert(lb.end(), more.begin(), more.end());
        CHECK(p.list(b) == lb && p.kept_runs() == 0 && p.digest() == w.digest());
    }

    // two ranks: the same calls in the same order give the same runs and the same digest. A divergent call shows.
    {
        RunPool p(1200, 900, below), w(1200, 900, below);
        p.keep(rules); w.keep(rules);
        Rng rng{42};
        struct L { Run run; std::vector<int> prompt; };
        std::vector<L> live;
        int hits = 0, grants = 0;
        for (int it = 0; it < 400; ++it) {
            const int op = rng.range(0, 9);
            if (op < 4 && !live.empty()) {                    // retire a session (retained when it has valid rows)
                const size_t k = size_t(rng.range(0, int(live.size()) - 1));
                const auto st = finished(live[k].prompt, rng.range(1, 30));
                p.retain(live[k].run, st); w.retain(live[k].run, st);
                live.erase(live.begin() + long(k));
            } else {                                          // admit: continue a recent prompt or start a new one
                std::vector<int> prompt = seq(rng.range(100, 9000), rng.range(0, 6));
                if (rng.range(0, 1)) { const auto t = seq(rng.range(10, 2000), 50 + it); prompt.insert(prompt.end(), t.begin(), t.end()); }
                const uint64_t rows = prompt.size() + 68;
                if (!p.admissible(rows) || p.room() < (rows + 63) / 64) continue;
                const PoolHit h = p.match(prompt, rows, int(prompt.size()) - 1);
                Run r, rw;
                if (h.skip > 0) {
                    const PoolHit hw = w.match(prompt, rows, int(prompt.size()) - 1);
                    CHECK(hw.skip == h.skip && hw.age == h.age);
                    r = p.take(h, rows); rw = w.take(hw, rows); ++hits;
                } else { r = p.reserve(rows); rw = w.reserve(rows); }
                CHECK(r == rw && p.list(r) == w.list(rw));
                live.push_back(L{r, prompt}); ++grants;
            }
            CHECK(p.digest() == w.digest() && p.free_pages() == w.free_pages());
        }
        CHECK(hits > 10 && grants > 100);
        // a worker whose pool took one more call than the head's has a different digest and grants a different run
        if (p.room() >= 3) {
            w.reserve(64);
            CHECK(w.digest() != p.digest());
            CHECK(p.reserve(64) != w.reserve(64));
        }
    }

    // an exclusive run is flagged, takes pages 0.. of the logical pool, and holds the whole pool while it lives
    {
        RunPool p(1000, 600, below);
        const Run x = p.reserve(40000);
        CHECK(p.exclusive_run(x) && p.list(x) == iota_list(0, 625) && p.room() == 0 && p.free_pages() == 0 && !p.admissible(64));
        p.release(x);
        const Run y = p.reserve(500 * 64);                    // below R: an ordinary run on the same first pages
        CHECK(!p.exclusive_run(y) && p.list(y) == iota_list(0, 500));
    }

    std::printf("pool_pages_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
