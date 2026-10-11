// The Rust pool (engine/kvman-rs through s2/pool.h) against the C++ pool it replaced (pool_ref.h): seeded random traces
// of reserve, release, retain, match + take, exclusive runs past R and calls that must fail, through both pools at once.
// After every call both must agree on the result (Run, PoolHit, or the error message), and on room, free pages, retained
// runs, digest and every live run's list. Some seeds also compare the G6_POOL_LOG lines. Host only.
//   pool_diff_test [seeds] [ops per seed]
#include "s2/pool.h"
#include "s2/tests/pool_ref.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
using namespace g6s2;

static long fails = 0;
#define CHECK(c) do { if (!(c)) { if (fails < 20) std::printf("FAIL seed %llu op %ld: %s\n", (unsigned long long)seed, op, #c); ++fails; } } while (0)

struct Rng {
    uint64_t s;
    uint64_t next() { uint64_t z = (s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
    int range(int lo, int hi) { return lo + int(next() % uint64_t(hi - lo + 1)); }
    bool one_in(int n) { return next() % uint64_t(n) == 0; }
};

// stderr into a file, so each pool's log lines can be read back and compared
struct Capture {
    int saved = -1, fd = -1;
    char path[64] = "/tmp/pool_diff_test.XXXXXX";
    void on() { fd = mkstemp(path); saved = dup(2); fflush(stderr); dup2(fd, 2); }
    std::string take() {
        fflush(stderr);
        std::string s;
        char buf[4096];
        lseek(fd, 0, SEEK_SET);
        for (ssize_t n; (n = read(fd, buf, sizeof buf)) > 0;) s.append(buf, size_t(n));
        lseek(fd, 0, SEEK_SET);
        if (ftruncate(fd, 0)) {}
        return s;
    }
    void off() { fflush(stderr); dup2(saved, 2); close(saved); close(fd); unlink(path); }
};

struct Live { Run run; std::vector<int> prompt; int skip; };
struct Stats { long ops = 0, grants = 0, hits = 0, exclusive = 0, retained = 0, evicted = 0, errors = 0, logs = 0; };

// both calls must throw the same message or neither
template <class A, class B> static bool same_error(A&& a, B&& b, std::string& why) {
    std::string ea, eb;
    try { a(); } catch (const std::exception& e) { ea = e.what(); if (ea.empty()) ea = "?"; }
    try { b(); } catch (const std::exception& e) { eb = e.what(); if (eb.empty()) eb = "?"; }
    why = ea;
    return ea == eb;
}

// a finished session as Service::retire_run builds it: prefill from skip (stopped early when `cut`), then decode
static g6rt::PrefixReuse finished(Rng& rng, const Live& s, bool cut) {
    g6rt::PrefixReuse st;
    const int n = int(s.prompt.size());
    if (s.skip) st.resume(s.skip);
    std::vector<int> gen(size_t(rng.range(0, 40)));
    for (auto& t : gen) t = rng.range(0, 30);
    if (cut) {   // cancelled during prefill: no decode rows, nothing emitted
        if (n - 1 > s.skip) st.prefill_rows(s.skip, rng.range(1, n - 1 - s.skip), true);
        st.finish(s.prompt, gen, 0, true);
        return st;
    }
    if (n - 1 > s.skip) st.prefill_rows(s.skip, n - 1 - s.skip, true);
    const int emitted = rng.range(0, int(gen.size()));
    st.main_rows(n - 1, 1 + int(gen.size()));
    st.draft_rows(n - 1, 1 + int(gen.size()));
    st.finish(s.prompt, gen, emitted, true);
    return st;
}

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? atoi(argv[1]) : 300, ops = argc > 2 ? atoi(argv[2]) : 3000;
    Stats total;
    uint64_t seed = 0; long op = 0;
    for (seed = 1; seed <= uint64_t(seeds); ++seed) {
        Rng rng{seed * 0x51ED27ull};
        const uint32_t pages = uint32_t(rng.range(8, 700));
        const uint32_t place = rng.one_in(3) ? pages : uint32_t(rng.range(1, int(pages)));
        const uint32_t below = 64u * uint32_t(rng.range(1, int(place)));
        RunPool rs(pages, place, below);
        g6s2ref::RunPool ref(pages, place, below);
        const bool keep = !rng.one_in(6);
        if (keep) {
            g6rt::PrefixReuse rules;
            rules.strict = rng.one_in(5); rules.noend = !rng.one_in(3);
            const int grids[] = {0, 0, 64, 128, 192, 1000, 4096};
            rules.grid = grids[rng.range(0, 6)];
            rs.keep(rules); ref.keep(rules);
        }
        Capture cap;
        const bool logged = seed % 25 == 0;
        if (logged) { rs.log(true); ref.log(true); cap.on(); }
        std::vector<Live> live;
        std::vector<Run> dead;                  // released runs, for stale calls
        std::vector<std::vector<int>> past;     // retired prompts, the conversations later prompts continue
        uint64_t age = 0;                       // the pools' retained age counter
        const int max_tokens = int(pages) * 64;
        auto prompt = [&] {
            std::vector<int> p;
            if (!past.empty() && !rng.one_in(4)) {
                p = past[size_t(rng.range(0, int(past.size()) - 1))];
                if (rng.one_in(3)) p.resize(size_t(rng.range(0, int(p.size()))));
            }
            const int extra = rng.one_in(4) ? rng.range(1, 8) : rng.range(1, std::max(1, max_tokens / (rng.one_in(3) ? 2 : 8)));
            for (int i = 0; i < extra; ++i) p.push_back(rng.range(0, 30));
            return p;
        };
        auto rows_for = [&](const std::vector<int>& p) {
            if (rng.one_in(10)) return uint64_t(rng.range(0, max_tokens + 200));   // anything, past R and past the pool
            return uint64_t(p.size()) + uint64_t(rng.range(1, 64)) + 4;
        };
        for (op = 0; op < ops; ++op) {
            const long before = fails;
            const int k = rng.range(0, 99);
            std::string why;
            if (k < 30) {                                           // admit: a hit when match finds one, else reserve
                const auto p = prompt();
                const uint64_t rows = rows_for(p);
                const int limit = rng.one_in(8) ? rng.range(-1, int(p.size()) + 5) : int(p.size()) - 1;
                const bool adm = rs.admissible(rows);
                CHECK(adm == ref.admissible(rows) && rs.exclusive(rows) == ref.exclusive(rows));
                PoolHit h;
                g6s2ref::PoolHit hr;
                if (rng.one_in(10) || adm) {
                    h = rs.match(p, rows, limit); hr = ref.match(p, rows, limit);
                    CHECK(h.skip == hr.skip && h.lcp == hr.lcp && h.age == hr.age && !strcmp(h.why, hr.why));
                }
                Run a{}, b{};
                const long kept = rs.kept_runs();
                const bool same = h.skip > 0
                    ? same_error([&] { a = rs.take(h, rows); }, [&] { b = ref.take(hr, rows); }, why)
                    : same_error([&] { a = rs.reserve(rows); }, [&] { b = ref.reserve(rows); }, why);
                CHECK(same && a == b);
                if (!why.empty()) ++total.errors;
                else {
                    ++total.grants; total.hits += h.skip > 0; total.exclusive += rs.exclusive_run(a);
                    total.evicted += kept - (h.skip > 0) - long(rs.kept_runs());
                    live.push_back(Live{a, p, h.skip > 0 ? h.skip : 0});
                }
            } else if (k < 55 && !live.empty()) {                   // retire: retained when it has valid rows
                const size_t i = size_t(rng.range(0, int(live.size()) - 1));
                const auto st = finished(rng, live[i], rng.one_in(8));
                CHECK(same_error([&] { rs.retain(live[i].run, st); }, [&] { ref.retain(live[i].run, st); }, why) && why.empty());
                if (keep && st.valid > 0) { ++total.retained; ++age; past.push_back(st.history); }
                if (past.size() > 64) past.erase(past.begin());
                dead.push_back(live[i].run); live.erase(live.begin() + long(i));
            } else if (k < 65 && !live.empty()) {                   // release without retaining
                const size_t i = size_t(rng.range(0, int(live.size()) - 1));
                CHECK(same_error([&] { rs.release(live[i].run); }, [&] { ref.release(live[i].run); }, why) && why.empty());
                dead.push_back(live[i].run); live.erase(live.begin() + long(i));
            } else if (k < 72) {                                    // calls that must fail, the same way
                Run r = !dead.empty() ? dead[size_t(rng.range(0, int(dead.size()) - 1))] : Run{0, 1, 999999, 1};
                if (!live.empty() && rng.one_in(2)) { r = live[size_t(rng.range(0, int(live.size()) - 1))].run; r.list_hash ^= 1; }
                // a take with no hit, or of an age no run was ever retained under
                PoolHit stale;
                stale.skip = rng.one_in(2) ? -rng.range(0, 3) : rng.range(1, 4096);
                stale.age = stale.skip > 0 ? age + uint64_t(rng.range(1, 3)) : uint64_t(rng.range(0, int(age) + 1));
                g6s2ref::PoolHit stale_ref; stale_ref.skip = stale.skip; stale_ref.age = stale.age;
                const int c = rng.range(0, 4);
                const bool same = c == 0 ? same_error([&] { rs.release(r); }, [&] { ref.release(r); }, why)
                                : c == 1 ? same_error([&] { rs.list(r); }, [&] { ref.list(r); }, why)
                                : c == 2 ? same_error([&] { rs.retain(r, g6rt::PrefixReuse{}); }, [&] { ref.retain(r, g6rt::PrefixReuse{}); }, why)
                                : c == 3 ? same_error([&] { rs.reserve(0); }, [&] { ref.reserve(0); }, why)
                                : same_error([&] { rs.take(stale, 64); }, [&] { ref.take(stale_ref, 64); }, why);
                CHECK(same && !why.empty());
                ++total.errors;
            } else {                                                // queries only
                const uint64_t rows = uint64_t(rng.range(0, max_tokens + 100));
                CHECK(rs.admissible(rows) == ref.admissible(rows) && rs.exclusive(rows) == ref.exclusive(rows));
            }
            ++total.ops;
            // the state after every call
            CHECK(rs.room() == ref.room() && rs.free_pages() == ref.free_pages() && rs.kept_runs() == ref.kept_runs());
            CHECK(rs.digest() == ref.digest() && rs.keeping() == ref.keeping());
            for (const auto& l : live) {
                std::vector<int> la, lb; bool xa = false, xb = false;
                CHECK(same_error([&] { la = rs.list(l.run); xa = rs.exclusive_run(l.run); },
                                 [&] { lb = ref.list(l.run); xb = ref.exclusive_run(l.run); }, why) && why.empty() && la == lb && xa == xb);
            }
            if (logged) {
                const std::string ls = cap.take();
                // the log lines of both calls arrive in one capture: the Rust call ran first, then the C++ one
                const size_t half = ls.size() / 2;
                CHECK(ls.size() % 2 == 0 && ls.compare(0, half, ls, half, half) == 0);
                total.logs += ls.empty() ? 0 : 1;
            }
            if (fails != before) break;   // the pools differ: the rest of this seed would only repeat it
        }
        if (logged) cap.off();
    }
    std::printf("pool_diff_test: %ld ops over %d seeds: %ld grants (%ld hits, %ld exclusive), %ld retained, %ld evicted, "
                "%ld matching errors, %ld logged claims\n", total.ops, seeds, total.grants, total.hits, total.exclusive, total.retained,
                total.evicted, total.errors, total.logs);
    std::printf("pool_diff_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
