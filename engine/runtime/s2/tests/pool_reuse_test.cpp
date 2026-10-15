// RunPool prefix reuse (pool.h keep / retain / match / take): the retained run a new prompt continues, the pages the
// grant takes, and the retained runs a reservation evicts. A mirror pool stands in for a worker and must grant the same runs. Host only.
#include "s2/pool.h"
#include <cstdio>
using namespace g6s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// a finished session: prompt 0..n-1 prefilled (rows 0..n-2), then `gen` emitted ids, as Dispatch + Service::retire_run build it
static g6rt::PrefixReuse finished(const std::vector<int>& prompt, const std::vector<int>& gen)
{
    g6rt::PrefixReuse st;
    const int n = int(prompt.size());
    st.prefill_rows(0, n - 1, true);
    st.main_rows(n - 1, 1 + int(gen.size()));
    st.draft_rows(n - 1, 1 + int(gen.size()));
    st.finish(prompt, gen, int(gen.size()), true);
    return st;
}
static std::vector<int> iota_list(int a, int n) { std::vector<int> v(n); for (int i = 0; i < n; ++i) v[i] = a + i; return v; }
static std::vector<int> seq(int n, int salt) { std::vector<int> v(n); for (int i = 0; i < n; ++i) v[i] = (i * 7 + salt) % 1000; return v; }

int main()
{
    const uint32_t pages = 1024, place = 1024, below = 32768;   // 64 Ki rows, R = 512 pages
    g6rt::PrefixReuse rules; rules.grid = 4096;
    RunPool pool(pages, place, below), mirror(pages, place, below);   // head and a worker
    pool.keep(rules); mirror.keep(rules);

    // A: 14,001-token prompt, 32 emitted, retained on pages 0..
    const auto A = seq(14001, 1), Agen = seq(32, 5);
    const uint64_t rowsA = 14001 + 64 + 4;
    Run a = pool.reserve(rowsA);
    CHECK(mirror.reserve(rowsA) == a && a.first_page == 0);
    pool.retain(a, finished(A, Agen)); mirror.retain(a, finished(A, Agen));
    CHECK(pool.kept_runs() == 1 && pool.free_pages() == pages && pool.digest() == mirror.digest());

    // B continues A (A + its output + a new turn, 20,001 tokens): a hit at floor(14000 / 4096) * 4096 = 12288
    std::vector<int> B = A; B.insert(B.end(), Agen.begin(), Agen.end());
    const auto tail = seq(20001 - int(B.size()), 9); B.insert(B.end(), tail.begin(), tail.end());
    const uint64_t rowsB = 20001 + 64 + 4;
    PoolHit h = pool.match(B, rowsB, 20000);
    CHECK(h.skip == 12288);
    const PoolHit hm = mirror.match(B, rowsB, 20000);     // the worker re-derives the head's decision
    CHECK(hm.skip == h.skip && hm.age == h.age);
    // an unrelated prompt does not hit
    CHECK(pool.match(seq(9001, 3), 9001 + 68, 9000).skip == 0);
    // a prompt that shares less than one grid cell is a miss
    std::vector<int> C(A.begin(), A.begin() + 3000); const auto ct = seq(2000, 4); C.insert(C.end(), ct.begin(), ct.end());
    CHECK(pool.match(C, C.size() + 68, int(C.size()) - 1).skip == 0);

    // a live run right after the retained one: still a hit (the new pages go anywhere); a miss only without room
    {
        RunPool p2(pages, place, below); p2.keep(rules);
        Run x = p2.reserve(rowsA); p2.retain(x, finished(A, Agen));
        p2.reserve(4 * 64);                                   // the four pages right after the retained run
        CHECK(p2.kept_runs() == 1 && p2.match(B, rowsB, 20000).skip == 12288);
        RunPool p6(400, 400, 400 * 64); p6.keep(rules);
        Run y = p6.reserve(rowsA); p6.retain(y, finished(A, Agen));
        p6.reserve(150 * 64);                                 // 250 pages left: B needs 314
        CHECK(p6.match(B, rowsB, 20000).skip == 0);
    }

    // two conversations seeded one after the other: the first one's next turn hits without evicting the second
    {
        RunPool p4(1920, 1920, below); p4.keep(rules);
        const auto Xs = seq(10001, 21), Ys = seq(10001, 22);
        Run x1 = p4.reserve(10001 + 68); p4.retain(x1, finished(Xs, seq(32, 1)));
        Run y1 = p4.reserve(10001 + 68); p4.retain(y1, finished(Ys, seq(32, 2)));
        std::vector<int> X2 = Xs; const auto xo = seq(32, 1), xt = seq(5968, 23);
        X2.insert(X2.end(), xo.begin(), xo.end()); X2.insert(X2.end(), xt.begin(), xt.end());
        const PoolHit hx = p4.match(X2, X2.size() + 68, int(X2.size()) - 1);
        CHECK(hx.skip == 8192);
        const Run x2 = p4.take(hx, X2.size() + 68);
        CHECK(p4.kept_runs() == 1 && x2.first_page == x1.first_page);   // Y's run stays
    }

    // take: the hit adopts A's first 192 pages (12288 rows), and the rest of A's pages count as free again
    Run b = pool.take(h, rowsB);
    CHECK(mirror.take(hm, rowsB) == b);
    CHECK(pool.list(b) == iota_list(0, b.pages) && pool.kept_runs() == 0 && mirror.kept_runs() == 0);
    CHECK(pool.free_pages() == pages - b.pages);

    // a second conversation D while B is live: on the lowest free pages, after B
    const auto D = seq(5000, 11);
    Run d = pool.reserve(5000 + 68);
    CHECK(mirror.reserve(5000 + 68) == d && d.first_page == b.pages);
    pool.retain(b, finished(B, seq(16, 2))); mirror.retain(b, finished(B, seq(16, 2)));
    pool.retain(d, finished(D, seq(8, 1))); mirror.retain(d, finished(D, seq(8, 1)));
    CHECK(pool.kept_runs() == 2 && pool.digest() == mirror.digest());
    // a new reservation prefers pages no retained run holds
    Run e = pool.reserve(64 * 10);
    CHECK(mirror.reserve(64 * 10) == e && e.first_page == b.pages + d.pages);
    CHECK(pool.kept_runs() == 2 && mirror.kept_runs() == 2);
    pool.release(e); mirror.release(e);
    // one that only fits over retained pages evicts the least recently retired run, whole
    {
        RunPool p3(600, 600, below); p3.keep(rules);
        Run r1 = p3.reserve(300 * 64); Run r2 = p3.reserve(100 * 64);
        p3.retain(r1, finished(A, Agen)); p3.retain(r2, finished(D, seq(8, 1)));
        Run r3 = p3.reserve(250 * 64);                       // 200 spare pages: r1 (retired first) goes
        CHECK(r3.first_page == 0 && p3.kept_runs() == 1 && p3.match(D, 5000 + 68, 4999).skip == 0);   // r2 stays (D < one grid cell)
        CHECK(p3.free_pages() == 600 - 250);
    }

    // no spare pages: the least recently retired run goes, whatever its address (PF-C6-CONC)
    {
        RunPool p5(600, 600, below); p5.keep(rules);
        const auto N = seq(9000, 31), O = seq(9000, 32);
        Run nw = p5.reserve(150 * 64), old = p5.reserve(150 * 64);
        p5.reserve(300 * 64);
        p5.retain(old, finished(O, seq(8, 3)));              // retired first: the older
        p5.retain(nw, finished(N, seq(8, 4)));               // retired last: the newer, at the lower address
        Run r = p5.reserve(150 * 64);
        CHECK(r.first_page == 150 && p5.kept_runs() == 1);
        std::vector<int> Nq = N; const auto ng = seq(8, 4), nt = seq(100, 40); Nq.insert(Nq.end(), ng.begin(), ng.end()); Nq.insert(Nq.end(), nt.begin(), nt.end());
        CHECK(p5.match(Nq, Nq.size() + 68, int(Nq.size()) - 1).skip == 8192 && p5.match(O, 9000 + 68, 8999).skip == 0);   // the newer one survives
    }

    // an exclusive (past-R) reservation evicts every retained run
    Run g = pool.reserve(rowsA); pool.retain(g, finished(A, Agen));
    CHECK(pool.kept_runs() == 3);                             // B, D and A
    Run x = pool.reserve(40000); CHECK(x.first_page == 0 && pool.exclusive_run(x) && pool.kept_runs() == 0);

    std::printf("pool_reuse_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
