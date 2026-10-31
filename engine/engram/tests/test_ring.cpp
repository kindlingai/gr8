// CPU test of the Engram token ring (engram_ring.h, the code k_engram_win runs) against the official reference's positional cache.
// Reference (ref-official/inference/engram.py NgramHashState.forward): cache[start_pos : start_pos + seqlen] = ids, then for each row
// source = cache[max(pos - shift, 0)], blocked once pos < shift; a blocked lookback is the pad id (here -1, as in the win[] contract).
// Three models must agree on every window of every step:
//   ring   - engram_ring.h, exactly the kernel's order: all rows put, then all rows get
//   ref    - an unbounded positional cache, the reference's own algorithm
//   truth  - from the real sequence T: positions below the step's first row hold accepted tokens T[q], this step's rows hold the fed tokens
// Scenarios: offline chunks + decode; serving (several requests back to back, the ring not cleared between them, chunk 1..8 prompt rows);
// DSpark verify chains (k = 1..5 drafts, random accept counts 0..k, the next step restarts at the first rejected position).
// A mutant (reads before writes) must fail, so the test is shown to discriminate. Usage: test_ring [seed]
#include "engram_ring.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
using namespace ds41::engram;

static const int kMaxPos = 4096;
static std::mt19937_64 rng;
static int rnd(int lo, int hi) { return lo + int(rng() % uint64_t(hi - lo + 1)); }

struct Rows { std::vector<int> pos, tok; };
struct Counts { long windows = 0, bad_ring = 0, bad_ref = 0; };

// one step through all three models; true_seq/accepted_to describe the sequence (see truth below)
static void step(int* ring, std::vector<int>& cache, const std::vector<int>& T, const Rows& rows, bool write_first, Counts& c)
{
    const int n = int(rows.pos.size());
    const int first = rows.pos[0];
    std::vector<int> win(size_t(n) * 4, -2);
    if (write_first) {
        for (int r = 0; r < n; ++r) ring_put(ring, rows.pos[r], rows.tok[r]);
        for (int r = 0; r < n; ++r) for (int k = 0; k < 4; ++k) win[size_t(r) * 4 + k] = ring_get(ring, rows.pos[r] - k);
    } else {   // mutant: the reads see the step's rows only after an earlier step wrote them
        for (int r = 0; r < n; ++r) for (int k = 0; k < 4; ++k) win[size_t(r) * 4 + k] = ring_get(ring, rows.pos[r] - k);
        for (int r = 0; r < n; ++r) ring_put(ring, rows.pos[r], rows.tok[r]);
    }
    for (int r = 0; r < n; ++r) cache[rows.pos[r]] = rows.tok[r];                      // reference: write the step's ids at start_pos...
    for (int r = 0; r < n; ++r) {
        bool blocked = false;
        for (int k = 0; k < 4; ++k) {
            const int p = rows.pos[r];
            blocked = blocked || p < k;                                                 // ...then gather cache[(pos - shift).clamp_min(0)]
            const int ref = blocked ? -1 : cache[p - k < 0 ? 0 : p - k];
            const int tru = (p - k < 0) ? -1 : (p - k >= first ? rows.tok[size_t(p - k - first)] : T[size_t(p - k)]);
            const int got = win[size_t(r) * 4 + k];
            ++c.windows;
            if (got != ref) ++c.bad_ref;
            if (got != tru || ref != tru) ++c.bad_ring;
        }
    }
}

static std::vector<int> make_seq(int len) { std::vector<int> T(size_t(len) + 16); for (int& t : T) t = rnd(0, 129279); return T; }

// positions [0, plen) in chunks of 1..cmax rows, then decode steps of 1 row up to len
static void plain(int* ring, std::vector<int>& cache, int len, int plen, int cmax, bool wf, Counts& c)
{
    const std::vector<int> T = make_seq(len);
    int p = 0;
    while (p < plen) { const int m = std::min(rnd(1, cmax), plen - p); Rows R; for (int i = 0; i < m; ++i) { R.pos.push_back(p + i); R.tok.push_back(T[size_t(p + i)]); } step(ring, cache, T, R, wf, c); p += m; }
    while (p < len) { Rows R; R.pos = {p}; R.tok = {T[size_t(p)]}; step(ring, cache, T, R, wf, c); ++p; }
}

// DSpark rounds after the prompt: rows = [golden, d1..dk] at q+1..q+k+1; a drafts accepted; the next step starts at q+a+2 with the bonus token
static void verify(int* ring, std::vector<int>& cache, int len, int plen, int kmax, bool wf, Counts& c)
{
    const std::vector<int> T = make_seq(len + 8);
    int p = 0;
    while (p < plen) { const int m = std::min(rnd(1, 8), plen - p); Rows R; for (int i = 0; i < m; ++i) { R.pos.push_back(p + i); R.tok.push_back(T[size_t(p + i)]); } step(ring, cache, T, R, wf, c); p += m; }
    int next = p;                                   // position of the next row 0 (its token is accepted: T[next])
    while (next < len) {
        const int k = rnd(1, kmax);
        Rows R; R.pos.push_back(next); R.tok.push_back(T[size_t(next)]);
        for (int i = 1; i <= k; ++i) {
            R.pos.push_back(next + i);
            R.tok.push_back(rnd(0, 99) < 55 ? T[size_t(next + i)] : T[size_t(next + i)] ^ 1);   // a right draft, or a wrong one (never equal to T)
        }
        int a = 0;                                  // accepted drafts: the leading ones equal to the sequence
        for (int i = 1; i <= k && R.tok[size_t(i)] == T[size_t(next + i)]; ++i) ++a;
        step(ring, cache, T, R, wf, c);
        next += a + 1;                              // rows 0..a were valid inputs; the bonus token T[next] starts the next step
    }
}

int main(int argc, char** argv)
{
    const uint64_t seed = argc > 1 ? strtoull(argv[1], nullptr, 10) : 20261025;
    static_assert(kRing >= 32 + 3, "ring must hold MX rows + 3");
    int fails = 0;
    for (int wf = 1; wf >= 0; --wf) {
        rng.seed(seed);                                                 // the mutant sees the same requests
        Counts c;
        int ring[kRing];
        for (int& x : ring) x = 7777;                                   // garbage: never cleared between requests
        std::vector<int> cache(kMaxPos + 64, 5555);
        for (int rep = 0; rep < 300; ++rep) {                           // requests back to back on one ring and one cache
            const int kind = rep % 3;
            const int len = rnd(1, 700), plen = rnd(0, len);
            if (kind == 0) plain(ring, cache, len, plen, 32, wf, c);   // offline loop: chunks up to MX rows
            else if (kind == 1) plain(ring, cache, len, plen, 8, wf, c);// serving: chunks of 1..8 then M = 1
            else verify(ring, cache, len + 8, plen, 5, wf, c);          // DSpark verify chains
        }
        printf("%s: %ld windows, ring!=truth/ref %ld, ring!=ref %ld\n", wf ? "ring (writes first)" : "mutant (reads first)", c.windows, c.bad_ring, c.bad_ref);
        if (wf && (c.bad_ring || c.bad_ref)) ++fails;
        if (!wf && !c.bad_ring) { printf("FAIL: the mutant was not caught\n"); ++fails; }
    }
    printf(fails ? "FAIL\n" : "PASS (seed %llu)\n", (unsigned long long)seed);
    return fails ? 1 : 0;
}
