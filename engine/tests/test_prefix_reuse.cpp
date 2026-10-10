// PrefixReuse unit cases (port of 18318ee's regression test to the K1/CP lineage: g6m4.cu has no PF_TAIL tail
// lane, so the replayed prefill loop is two lanes + a single-lane remainder). LCP (default), G6_REUSE_STRICT=1 (K2 rule),
// G6_REUSE_NOEND=1 (no prompt >= written_end rule), the miss reasons, MTP's one-row margin, chain rows, stops.
// build and run:  g++ -O2 -std=c++17 -I engine/runtime engine/tests/test_prefix_reuse.cpp -o /tmp/test_prefix_reuse \
//                 && /tmp/test_prefix_reuse        (scripts/build.sh builds and runs it; engine/tests/test_prefix_reuse.py
//                 runs it under ASan/UBSan and pins the g6m4.cu hook order)
#include "prefix_reuse.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
using g6rt::PrefixReuse;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// g6m4.cu prefill loop (two lanes), reuse bookkeeping only
static int prefill(PrefixReuse& r, int skip, int first, int MP, bool two, bool mtp)
{
    int done = skip;
    for (int a = skip; a < first; a += MP)
    {
        if (two && first - a > MP)
        {
            const int MA = MP, MB = first - a - MP < MP ? first - a - MP : MP;
            r.prefill_rows(a, MA + MB, mtp);
            done = a + MA + MB;
            a += MP;
            continue;
        }
        const int M = first - a < MP ? first - a : MP;
        r.prefill_rows(a, M, mtp);
        done = a + M;
    }
    return done;
}
static std::vector<int> toks(int n, int seed) { std::vector<int> v(n); for (int i = 0; i < n; ++i) v[i] = 1000 + (i * 7 + seed) % 50000; return v; }
static std::vector<int> cat(std::vector<int> a, const std::vector<int>& b) { a.insert(a.end(), b.begin(), b.end()); return a; }
static std::vector<int> head(const std::vector<int>& a, int n) { return std::vector<int>(a.begin(), a.begin() + n); }
// one served request: prefill (MP 4096, two lanes) to p.first = n - 1, then the prompt row + `gen` decode rows
// (MTP: verify rows of K + 1 with first-form drafts over accepted rows, chain rows K - 1 ahead)
static int request(PrefixReuse& r, const std::vector<int>& prompt, const std::vector<int>& gen, int emitted, int K)
{
    const int first = (int)prompt.size() - 1;
    const int skip = r.begin(prompt, first, K > 0);
    CHECK(prefill(r, skip, first, 4096, true, K > 0) == first || skip == first);
    r.main_rows(first, 1);
    if (K) r.draft_rows(first, 1);
    if (K) r.draft_rows(first + 1, K - 1, true);                   // chain after the prompt step
    int base = first + 1;                                          // position of the last committed token
    for (int left = (int)gen.size() - 1; left > 0;)
    {
        int acc = 1;
        if (K)                                                     // verify [last, d1..dK]; acc tokens committed
        {
            r.main_rows(base, K + 1);
            acc = std::min(left, 2);
            r.draft_rows(base, acc);                               // first-form over the accepted rows
            r.draft_rows(base + acc, K - 1, true);                 // chain rows ahead
        }
        else r.main_rows(base, 1);
        base += acc; left -= acc;
    }
    r.finish(prompt, gen, emitted, K > 0);
    return skip;
}

int main()
{
    for (int K : {0, 4})
    {
        const bool mtp = K > 0;
        // ---- turn 1 cold, 20,000-token prompt, 300 generated
        PrefixReuse r;
        const std::vector<int> p1 = toks(20000, 1), g1 = toks(300, 2);
        CHECK(request(r, p1, g1, 300, K) == 0 && !strcmp(r.why, "first"));
        CHECK(r.history.size() == 20300);
        CHECK(r.valid == 20299 || (mtp && r.valid <= 20299));
        CHECK(r.written_end >= 20299);
        // ---- strict append: everything valid is kept
        {
            PrefixReuse s = r;
            const std::vector<int> p = cat(r.history, toks(50, 3));
            const int skip = s.begin(p, (int)p.size() - 1, mtp);
            CHECK(skip == r.valid && !strcmp(s.why, "hit") && s.lcp == 20300);
        }
        // ---- agent re-template: previous prompt kept, generated text dropped (reasoning removed), new text appended
        {
            PrefixReuse s = r;
            const std::vector<int> p = cat(p1, toks(400, 9));      // 20,400 >= written_end: hit
            const int skip = s.begin(p, (int)p.size() - 1, mtp);
            CHECK(s.lcp >= 20000 && skip == (mtp ? s.lcp - 1 : s.lcp) && !strcmp(s.why, "hit"));
            PrefixReuse t = r; t.strict = true;                     // the K2 rule misses it
            CHECK(t.begin(p, (int)p.size() - 1, mtp) == 0 && !strcmp(t.why, "strict"));
            // short new text: prompt 20,050 < written_end (~20,300) -> "short" miss, NOEND hit
            const std::vector<int> q = cat(p1, toks(50, 9));
            PrefixReuse u = r;
            CHECK(u.begin(q, (int)q.size() - 1, mtp) == 0 && !strcmp(u.why, "short"));
            PrefixReuse v = r; v.noend = true;
            const int sk = v.begin(q, (int)q.size() - 1, mtp);
            CHECK(sk == (mtp ? v.lcp - 1 : v.lcp) && !strcmp(v.why, "hit"));
            CHECK(v.main_end == sk && v.draft_end == sk && v.written_end == sk);
        }
        // ---- a prompt SHORTER than the history (a prefix of it): miss "short" under the written_end rule (default),
        // HIT under G6_REUSE_NOEND=1 (skip = the safe LCP, here the prompt's p.first)
        {
            const std::vector<int> p = head(r.history, (int)r.history.size() - 5);
            PrefixReuse s = r;
            CHECK(s.begin(p, (int)p.size() - 1, mtp) == 0 && !strcmp(s.why, "short"));
            PrefixReuse t = r; t.noend = true;
            CHECK(t.begin(p, (int)p.size() - 1, mtp) == (int)p.size() - 1 && !strcmp(t.why, "hit"));
        }
        // ---- divergence inside the old prompt (system prompt edit at 5,000)
        {
            PrefixReuse s = r; s.noend = true;
            std::vector<int> p = cat(r.history, toks(10, 4)); p[5000] ^= 1;
            CHECK(s.begin(p, (int)p.size() - 1, mtp) == (mtp ? 4999 : 5000) && s.lcp == 5000);
        }
        // ---- divergence inside the generated part
        {
            PrefixReuse s = r;
            std::vector<int> p = cat(r.history, toks(10, 4)); p[20150] ^= 1;
            CHECK(s.begin(p, (int)p.size() - 1, mtp) == (mtp ? 20149 : 20150));
        }
        // ---- identical prompt: limit = p.first keeps the last row on the normal path
        {
            PrefixReuse s = r; s.noend = true;
            const int skip = s.begin(r.history, (int)r.history.size() - 1, mtp);
            CHECK(skip == std::min(r.valid, (int)r.history.size() - 1) && skip < (int)r.history.size());
        }
        // ---- first token differs: no-lcp; MTP with lcp 1: no-lcp (the draft row 0 depends on token 1)
        {
            PrefixReuse s = r;
            std::vector<int> p = cat(r.history, toks(10, 4)); p[0] ^= 1;
            CHECK(s.begin(p, (int)p.size() - 1, mtp) == 0 && !strcmp(s.why, "no-lcp"));
            PrefixReuse t = r; t.noend = true;
            std::vector<int> q = head(r.history, 30); q[1] ^= 1;
            CHECK(t.begin(q, 29, mtp) == (mtp ? 0 : 1) && !strcmp(t.why, mtp ? "no-lcp" : "hit"));
        }
        // ---- stop in the middle of an MTP batch: lookahead rows past the emitted ids never become valid
        {
            PrefixReuse s;
            const std::vector<int> p = toks(100, 5), g = toks(9, 6);
            request(s, p, g, 4, K);                                 // 9 sampled, 4 emitted (stop id)
            CHECK(s.history.size() == 104 && s.valid <= 103 && s.written_end > 104);
            const std::vector<int> q = cat(s.history, toks(5, 7));  // 109 >= written_end? written_end may be ~110
            PrefixReuse n = s; n.noend = true;
            CHECK(n.begin(q, (int)q.size() - 1, mtp) == s.valid);
        }
        // ---- an error-path request (rejected before begin) leaves the state as it was: nothing to do
        // ---- clear(): next request is a "first" miss
        {
            PrefixReuse s = r; s.clear();
            CHECK(s.begin(cat(r.history, toks(3, 1)), (int)r.history.size() + 2, mtp) == 0 && !strcmp(s.why, "first"));
        }
        // ---- 18318ee: a cold 8,193..8,258-token prompt after a short turn: prefill to p.first, no hole
        for (int n : {8193, 8194, 8257, 8258, 4097, 2})
        {
            PrefixReuse e; std::vector<int> q = toks(n, 4); const int f = n - 1;
            CHECK(e.begin(q, f, mtp) == 0);
            CHECK(prefill(e, 0, f, 4096, true, mtp) == f && e.main_end == f && (!mtp || e.draft_end == f));
            e.main_rows(f, 1);
        }
        // ---- a hit whose suffix starts mid-page and spans lanes: rows skip .. first, no hole
        {
            PrefixReuse s = r; s.noend = true;
            const std::vector<int> p = cat(head(r.history, 12345), toks(9000, 8));
            const int skip = s.begin(p, (int)p.size() - 1, mtp);
            CHECK(skip == (mtp ? 12344 : 12345) && skip % 64 != 0);
            CHECK(prefill(s, skip, (int)p.size() - 1, 4096, true, mtp) == (int)p.size() - 1);
            CHECK(s.main_end == (int)p.size() - 1);
        }
    }
    // ---- draft row t is built from main row t AND token t + 1. Divergence at
    // L = lcp: draft row L - 1 holds the OLD token L. Provenance check of every retained draft row's NEXT token; the
    // design-58b89d01 bound min(valid, lcp, limit) keeps the bad row (must fail), the shipped bound does not.
    for (int L : {1, 2, 5, 21, 22, 23})
    {
        PrefixReuse r;
        std::vector<int> prompt = toks(22, 11), gen = toks(4, 12);
        std::vector<int> next_of(64, -1);                         // draft row -> the token it consumed
        const int first = (int)prompt.size() - 1;
        r.begin(prompt, first, true);
        std::vector<int> full = cat(prompt, gen);
        for (int j = 0; j < first; ++j) next_of[j] = full[j + 1];  // prefill: pf_draft rows with prompt[j + 1]
        r.prefill_rows(0, first, true);
        r.main_rows(first, 1); r.draft_rows(first, 1); next_of[first] = full[first + 1];   // prompt step + first-form draft
        for (int pos = first + 1; pos < (int)full.size() - 1; ++pos) { r.main_rows(pos, 1); r.draft_rows(pos, 1); next_of[pos] = full[pos + 1]; }
        r.finish(prompt, gen, (int)gen.size(), true);
        std::vector<int> np = head(r.history, std::min(L, (int)r.history.size()));
        np.push_back(424242); np.push_back(1); np.push_back(2); np.push_back(3);   // diverges at L
        const PrefixReuse saved = r;
        r.noend = true;
        const int skip = r.begin(np, (int)np.size() - 1, true);
        CHECK(r.lcp == L);
        bool ok = true;
        for (int j = 0; j < skip; ++j) ok &= next_of[j] == np[j + 1];
        CHECK(ok);                                                // shipped bound: every retained draft row is valid
        const int old_bound = std::min({saved.valid, r.lcp, (int)np.size() - 1});
        bool old_ok = true;
        for (int j = 0; j < old_bound; ++j) old_ok &= next_of[j] == np[j + 1];
        CHECK(!old_ok || old_bound == 0);                         // the 58b89d01 bound retains the stale row L - 1
    }
    // ---- 2b G2: dirty_end over a NOEND hit chain. A long -> B short hit -> C hit: C's poison range [skip, dirty_end) must
    // cover A's leftover rows (written_end alone ends at B's rows); a miss resets it. Then random NOEND chains: dirty_end ==
    // the max row the driver wrote since the last miss, and >= written_end.
    for (int K : {0, 4})
    {
        PrefixReuse r; r.noend = true;
        const std::vector<int> a = toks(9000, 31);
        request(r, a, toks(64, 32), 64, K);
        const int a_end = r.dirty_end;
        CHECK(a_end >= 9000 + 64 - 1 && a_end == r.written_end);
        const std::vector<int> b = cat(head(a, 5000), toks(100, 33));
        CHECK(request(r, b, toks(64, 34), 64, K) > 0 && !strcmp(r.why, "hit"));
        CHECK(r.written_end < a_end && r.dirty_end == a_end);       // B's rows end before A's: written_end forgets A
        const std::vector<int> c = cat(cat(b, head(toks(64, 34), 10)), toks(20, 35));
        const int old_dirty = r.dirty_end, old_written = r.written_end;
        const int sc = request(r, c, toks(64, 36), 64, K);
        CHECK(sc > 0 && !strcmp(r.why, "hit") && sc < old_written && old_dirty == a_end);   // poison [sc, a_end) covers A
        CHECK(r.dirty_end == a_end);
        request(r, toks(300, 37), toks(8, 38), 8, K);              // miss (no lcp): reset
        CHECK(!strcmp(r.why, "no-lcp") && r.dirty_end == r.written_end && r.dirty_end < 400);
        PrefixReuse z = r; z.clear(); CHECK(z.dirty_end == 0);
        // random chains: the driver's own max row since the last miss
        unsigned s = 12345u + K;
        auto rnd = [&](int n) { s = s * 1103515245u + 12345u; return (int)((s >> 8) % (unsigned)n); };
        PrefixReuse q; q.noend = true;
        int maxw = 0;
        std::vector<int> prev;
        for (int t = 0; t < 400; ++t)
        {
            std::vector<int> p;
            const int kind = prev.empty() ? 0 : rnd(4);
            if (kind == 0) p = toks(50 + rnd(3000), 100 + t);                                   // fresh: miss
            else if (kind == 1) p = cat(q.history, toks(1 + rnd(200), 200 + t));             // append
            else p = cat(head(q.history, 2 + rnd((int)q.history.size() - 2)), toks(1 + rnd(100), 300 + t));   // shorter / divergent
            const int g = 1 + rnd(40);
            PrefixReuse before = q;
            const int skip = q.begin(p, (int)p.size() - 1, K > 0);
            q = before;                                             // request() calls begin() itself
            if (!skip) maxw = 0;
            request(q, p, toks(g, 400 + t), g, K);
            maxw = std::max(maxw, q.written_end);
            CHECK(q.dirty_end == maxw && q.dirty_end >= q.written_end);
            prev = p;
        }
    }
    // grid rule (G6_REUSE_ALIGN=1): served shape MP 4096 x two lanes -> grid 8192; the request shapes
    for (int K : {0, 4})
    {
        PrefixReuse r; r.grid = 8192;
        const std::vector<int> b1a = toks(6000, 51);                                 // r2a_b1
        request(r, b1a, toks(64, 52), 64, K);
        CHECK(r.pf_end == 5999);                                                     // prefill to n - 1 = 5999 rows
        const std::vector<int> b1 = cat(head(b1a, 3000), toks(3500, 53));            // r2b1: 3000 shared ids
        CHECK(request(r, b1, toks(64, 54), 64, K) == 0 && !strcmp(r.why, "grid") && r.raw == (K ? 2999 : 3000));
        CHECK(r.pf_end == 6499);                                                     // the miss re-prefilled 0..6498 from 0
        PrefixReuse q; q.grid = 8192;
        const std::vector<int> pi3 = toks(15087, 55);
        request(q, pi3, toks(64, 56), 64, K);
        CHECK(q.pf_end == 15086);
        const std::vector<int> pi4 = cat(q.history, toks(4517, 57));               // pi4 appends to pi3 + its output
        const int s4 = request(q, pi4, toks(64, 58), 64, K);
        CHECK(s4 == 8192 && !strcmp(q.why, "hit") && q.raw > 15086);                 // raw covers decode rows; pf_end caps it
        CHECK(q.pf_end == (int)pi4.size() - 1);                                      // 8192.. re-prefilled contiguously
        const std::vector<int> pi5 = cat(q.history, toks(9000, 59));
        CHECK(request(q, pi5, toks(8, 60), 8, K) == 16384 && !strcmp(q.why, "hit"));   // floor(19603 / 8192) * 8192
        PrefixReuse o; o.grid = 0;                                                   // grid off: the old rule unchanged
        request(o, pi3, toks(64, 56), 64, K);
        CHECK(request(o, cat(o.history, toks(4517, 57)), toks(64, 58), 64, K) == o.raw && o.raw > 15086);
        PrefixReuse z = q; z.clear(); CHECK(z.pf_end == 0 && z.raw == 0);
    }
    // the same prompt again (NOEND): its prefill would end where the retained one did, so the partial last cell is kept
    for (int K : {0, 4})
    {
        PrefixReuse q; q.grid = 4096; q.noend = true;
        const std::vector<int> p = toks(8192, 70);
        request(q, p, toks(64, 71), 64, K);
        CHECK(q.pf_end == 8191);
        CHECK(request(q, p, toks(64, 71), 64, K) == 8191 && !strcmp(q.why, "hit"));    // not floor(8191 / 4096) * 4096
        CHECK(request(q, head(p, 8000), toks(8, 72), 8, K) == 4096);                   // a shorter prompt: whole cells only
        PrefixReuse s; s.grid = 4096;                                                   // without NOEND the replay is 'short'
        request(s, p, toks(64, 71), 64, K);
        CHECK(request(s, p, toks(64, 71), 64, K) == 0 && !strcmp(s.why, "short"));
    }
    // g6m4.cu grid = MP = 4096 (the lane pairing does not change a row). Retained = floor(min(skip,
    // pf_end) / 4096) * 4096, so a hit can now keep an odd multiple of 4096 (retained mod 8192 = 4096)
    for (int K : {0, 4})
    {
        PrefixReuse q; q.grid = 4096;
        const std::vector<int> pi3 = toks(15087, 55);
        request(q, pi3, toks(64, 56), 64, K);
        const std::vector<int> pi4 = cat(q.history, toks(4517, 57));
        CHECK(request(q, pi4, toks(64, 58), 64, K) == 12288 && !strcmp(q.why, "hit"));   // floor(15086 / 4096) * 4096 (grid 8192: 8192)
        CHECK(q.pf_end == (int)pi4.size() - 1);                                      // 12288.. re-prefilled contiguously
        const std::vector<int> pi5 = cat(q.history, toks(9000, 59));
        CHECK(request(q, pi5, toks(8, 60), 8, K) == 16384 && !strcmp(q.why, "hit"));   // floor(19667 / 4096) * 4096
        // lcp just below / at / above the 12288 boundary (the MTP R1 rule: skip <= lcp - 1 when K > 0)
        for (int share : {12287, 12288, 12289, 12290})
        {
            PrefixReuse g; g.grid = 4096;
            const std::vector<int> a0 = toks(14000, 61);
            request(g, a0, toks(32, 62), 32, K);
            const std::vector<int> a1 = cat(head(a0, share), toks(2000, 63));
            const int raw = K ? share - 1 : share, want = raw / 4096 * 4096;
            const int got = request(g, a1, toks(32, 64), 32, K);
            CHECK(got == want && g.raw == raw && !strcmp(g.why, "hit"));
            CHECK(got % 4096 == 0 && g.pf_end == (int)a1.size() - 1);
        }
        // a lcp below one grid unit: a miss with why 'grid' (as the 8192 rule below 8192)
        PrefixReuse m; m.grid = 4096;
        const std::vector<int> b0 = toks(6000, 65);
        request(m, b0, toks(32, 66), 32, K);
        CHECK(request(m, cat(head(b0, 3000), toks(3500, 67)), toks(32, 68), 32, K) == 0 && !strcmp(m.why, "grid"));
    }
    printf("%s (%d failures)\n", fails ? "FAIL" : "PASS prefix_reuse unit cases", fails);
    return fails ? 1 : 0;
}
