// RunPool prefix reuse (pool.h keep / retain / match / take): what a new prompt continues, where the grant lands, and
// that claims evict exactly the retained runs they overlap. Host only.
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
static std::vector<int> seq(int n, int salt) { std::vector<int> v(n); for (int i = 0; i < n; ++i) v[i] = (i * 7 + salt) % 1000; return v; }

int main()
{
    const uint32_t pages = 1024, place = 1024, below = 32768;   // 64 Ki rows, R = 512 pages
    g6rt::PrefixReuse rules; rules.grid = 4096;
    RunPool pool(pages, place, below), mirror(pages, place, below);   // head and a worker
    pool.keep(rules); mirror.keep(rules);

    // A: 14,001-token prompt, 32 emitted, retained at base 0
    const auto A = seq(14001, 1), Agen = seq(32, 5);
    const uint64_t rowsA = 14001 + 64 + 4;
    Run a = pool.reserve(rowsA); mirror.claim(a, rowsA);
    CHECK(a.base_page == 0);
    pool.retain(a, finished(A, Agen)); mirror.retain(a, finished(A, Agen));
    CHECK(pool.kept_runs() == 1 && pool.free_pages() == pages);

    // B continues A (A + its output + a new turn, 20,001 tokens): a hit at floor(14000 / 4096) * 4096 = 12288
    std::vector<int> B = A; B.insert(B.end(), Agen.begin(), Agen.end());
    const auto tail = seq(20001 - int(B.size()), 9); B.insert(B.end(), tail.begin(), tail.end());
    const uint64_t rowsB = 20001 + 64 + 4;
    PoolHit h = pool.match(B, rowsB, 20000);
    CHECK(h.skip == 12288 && h.run.base_page == 0 && h.run.pages == (rowsB + 63) / 64);
    CHECK(mirror.match(B, rowsB, 20000).skip == h.skip);     // the worker re-derives the head's decision
    // an unrelated prompt does not hit
    CHECK(pool.match(seq(9001, 3), 9001 + 68, 9000).skip == 0);
    // a prompt that shares less than one grid cell is a miss
    std::vector<int> C(A.begin(), A.begin() + 3000); const auto ct = seq(2000, 4); C.insert(C.end(), ct.begin(), ct.end());
    CHECK(pool.match(C, C.size() + 68, int(C.size()) - 1).skip == 0);

    // a live run in the way of the hit's extension: no room, a miss
    {
        RunPool p2(pages, place, below); p2.keep(rules);
        Run x = p2.reserve(rowsA); p2.retain(x, finished(A, Agen));
        Run blocker{x.base_page + x.pages, 4, 99};
        p2.claim(blocker, 4 * 64);                            // live pages right after the retained run
        CHECK(p2.match(B, rowsB, 20000).skip == 0);
    }

    // two conversations seeded one after the other: the first one's next turn grows in place without evicting the second
    {
        RunPool p4(1920, 1920, below); p4.keep(rules);
        const auto Xs = seq(10001, 21), Ys = seq(10001, 22);
        Run x1 = p4.reserve(10001 + 68); p4.retain(x1, finished(Xs, seq(32, 1)));
        Run y1 = p4.reserve(10001 + 68); p4.retain(y1, finished(Ys, seq(32, 2)));
        std::vector<int> X2 = Xs; const auto xo = seq(32, 1), xt = seq(5968, 23);
        X2.insert(X2.end(), xo.begin(), xo.end()); X2.insert(X2.end(), xt.begin(), xt.end());
        const PoolHit hx = p4.match(X2, X2.size() + 68, int(X2.size()) - 1);
        CHECK(hx.skip == 8192 && hx.run.base_page == x1.base_page);
        p4.take(hx, X2.size() + 68);
        CHECK(p4.kept_runs() == 1);                           // Y's run is still there
    }

    // take: the hit becomes a live run at the same base; the retained entry is gone on both ranks
    Run b = pool.take(h, rowsB); mirror.claim(b, rowsB);
    CHECK(b.base_page == 0 && pool.kept_runs() == 0 && mirror.kept_runs() == 0);
    CHECK(pool.free_pages() == pages - b.pages);

    // a second conversation D while B is live: placed after B, retained there
    const auto D = seq(5000, 11);
    Run d = pool.reserve(5000 + 68); mirror.claim(d, 5000 + 68);
    CHECK(d.base_page == b.pages + (pages - b.pages - d.pages) / 2);   // centred in the gap after B, room for both to grow
    pool.retain(b, finished(B, seq(16, 2))); mirror.retain(b, finished(B, seq(16, 2)));
    pool.retain(d, finished(D, seq(8, 1))); mirror.retain(d, finished(D, seq(8, 1)));
    CHECK(pool.kept_runs() == 2);
    // a new reservation prefers pages no retained run holds
    Run e = pool.reserve(64 * 10); mirror.claim(e, 64 * 10);
    CHECK((e.base_page >= d.base_page + d.pages || e.base_page + e.pages <= d.base_page) && e.base_page >= b.pages);
    CHECK(pool.kept_runs() == 2 && mirror.kept_runs() == 2);
    pool.release(e); mirror.release(e);
    // one that only fits over retained pages evicts exactly the runs it overlaps
    {
        RunPool p3(600, 600, below); p3.keep(rules);
        Run r1 = p3.reserve(300 * 64); Run r2 = p3.reserve(100 * 64);
        p3.retain(r1, finished(A, Agen)); p3.retain(r2, finished(D, seq(8, 1)));
        Run r3 = p3.reserve(250 * 64);                       // 200 spare pages: first fit over r1's
        CHECK(r3.base_page == 0 && p3.kept_runs() == 1 && p3.match(D, 5000 + 68, 4999).skip == 0);   // r2 stays (D < one grid cell)
        CHECK(p3.free_pages() == 600 - 250);
    }

    // no gap fits: the window over the least recently retired run, not the lowest address (PF-C6-CONC)
    {
        RunPool p5(600, 600, below); p5.keep(rules);
        const auto N = seq(9000, 31), O = seq(9000, 32);
        Run nw{0, 150, 1}, old{150, 150, 2}, lv{300, 300, 3};
        p5.claim(nw, 150 * 64); p5.claim(old, 150 * 64); p5.claim(lv, 300 * 64);
        p5.retain(old, finished(O, seq(8, 3)));              // retired first: the older
        p5.retain(nw, finished(N, seq(8, 4)));               // retired last: the newer, at the lower address
        Run r = p5.reserve(150 * 64);
        CHECK(r.base_page == 150 && p5.kept_runs() == 1);
        std::vector<int> Nq = N; const auto ng = seq(8, 4), nt = seq(100, 40); Nq.insert(Nq.end(), ng.begin(), ng.end()); Nq.insert(Nq.end(), nt.begin(), nt.end());
        CHECK(p5.match(Nq, Nq.size() + 68, int(Nq.size()) - 1).skip == 8192 && p5.match(O, 9000 + 68, 8999).skip == 0);   // the newer one survives
    }

    // an exclusive (past-R) claim evicts every retained run
    Run g = pool.reserve(rowsA); pool.retain(g, finished(A, Agen));
    CHECK(pool.kept_runs() == 3);                             // B, D and A
    Run x = pool.reserve(40000); CHECK(x.base_page == 0 && pool.kept_runs() == 0);

    std::printf("pool_reuse_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
