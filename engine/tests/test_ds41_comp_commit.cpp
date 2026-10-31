// Host reference for k_comp_commit (engine/attn/ds41_attn.cuh): no CUDA. Build and run:
//   g++ -std=c++17 -O2 -Wall -Wextra -I engine engine/tests/test_ds41_comp_commit.cpp -o /tmp/test_ds41_comp_commit && /tmp/test_ds41_comp_commit
//
// What it checks. At a ratio-2 KV-source layer (2, 8, 14) the compressor pools token pairs (2g, 2g+1) into row g. The engine's dev-form step
// (ds41a::comp_row with dstart) is k_comp_pool_dev then k_comp_state: pair completions are produced at odd positions, and the pending half of a
// pair lives in state slot 0 (kvs / scs [0]). A verify chunk (start p, n = k + 1 rows) writes all its rows before acceptance, so after `a`
// accepted drafts (Pa = p + a = the last accepted position) the next step starts at Pa + 1, and when Pa is even that step's first row (odd
// position) pairs with slot 0, which must then hold the row of position Pa. k_comp_state parks row n - 1 (or nothing), so it is wrong unless
// a == n - 1. k_comp_commit fixes it: `if (!(Pa & 1)) slot0 = S.ckv[a]`.
//
// The model below copies the index arithmetic of the kernels (one "channel vector" is a few u64 words, pooling is an order-sensitive hash, so
// any wrong row or wrong pairing shows). Truth = feeding the accepted tokens one at a time (the n == 1 decode form of k_comp_pool: token at
// position t fills slot t & 1, an odd t completes row t >> 1 from slots 0 and 1). The test:
//   (1) with the commit, every compressed row 0 .. (Pa + 1) / 2 - 1 (those with 2g + 1 <= Pa) equals the truth after every round;
//   (2) with the commit, the row (Pa + 1) / 2 that the NEXT step completes equals the truth, for every start parity and every a in 0 .. k;
//   (3) negative control: WITHOUT the commit, (2) fails exactly in the rounds with Pa even and a < n - 1 (and only there), so the test sees the bug;
//   (4) rows written past Pa by the rejected tail never survive into a visible row (they are rewritten by the next step before any
//       query position can see them: row g is visible to a query at position t iff 2g + 1 <= t, the rule the verify chunk itself needs).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define EXPECT(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

namespace
{
constexpr int CH = 3;   // channel words per vector (the kernel has 512 threads; the copy is per channel, so any > 1 shows a half copy)
struct Vec
{
    uint64_t w[CH]{};
    bool operator==(const Vec& o) const { for (int i = 0; i < CH; ++i) if (w[i] != o.w[i]) return false; return true; }
};
static uint64_t mix(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
// the (kv, score) input of one position: tag 0 = the real token's row, otherwise a rejected-tail row of that round
struct Row { Vec kv, sc; };
static Row row_of(int pos, uint64_t tag)
{
    Row r;
    for (int i = 0; i < CH; ++i)
    {
        r.kv.w[i] = mix((uint64_t)pos * 1315423911ull + tag * 7919ull + (uint64_t)i * 31ull + 1);
        r.sc.w[i] = mix((uint64_t)pos * 2654435761ull + tag * 104729ull + (uint64_t)i * 17ull + 2);
    }
    return r;
}
static Vec pool(const Row& a, const Row& b)   // order-sensitive, stands for softmax-pool + norm + rope + fp4 of the real kernel
{
    Vec v;
    for (int i = 0; i < CH; ++i) v.w[i] = mix(a.kv.w[i] * 3 + b.kv.w[i] * 5 + mix(a.sc.w[i]) + 2 * mix(b.sc.w[i] + 9));
    return v;
}

// ---- the engine (ds41a::comp_row, device-start form) ----------------------------------------------------------------------------------
struct Engine
{
    std::vector<Vec> ck;   // C.comp rows
    Row slot0{};           // C.kvs / C.scs [0]; the dev kernels never touch slot 1
    std::vector<Row> scr;  // S.ckv / S.csc: the rows of the last chunk (the f32 gemm output), intact until the next comp_row
    void grow(size_t g) { if (ck.size() <= g) ck.resize(g + 1); }

    // prefill (start == 0): k_comp_pool start == 0 form
    void prefill(const std::vector<Row>& rows)
    {
        const int n = (int)rows.size();
        for (int j = 0; 2 * j + 1 < n; ++j) { grow(j); ck[j] = pool(rows[2 * j], rows[2 * j + 1]); }
        if (n & 1) slot0 = rows[n - 1];
        scr = rows;
    }
    // one chunk: k_f32gemm2 (scr = rows), k_comp_pool_dev, k_comp_state
    void step(int start, const std::vector<Row>& rows)
    {
        const int n = (int)rows.size();
        scr = rows;
        for (int k = 0; k < (n + 1) / 2; ++k)   // k_comp_pool_dev: block k
        {
            const int r = (start & 1) ? 2 * k : 2 * k + 1;
            if (r >= n) continue;
            const int grp = (start + r) >> 1;
            const Row& first = (r == 0) ? slot0 : scr[r - 1];
            grow(grp);
            ck[grp] = pool(first, scr[r]);
        }
        if (((start + n - 1) & 1) == 0) slot0 = scr[n - 1];   // k_comp_state
    }
    // k_comp_commit: dq = last accepted position Pa, dacc = accepted drafts a (row index of Pa in the chunk)
    void commit(int Pa, int a) { if (!(Pa & 1)) slot0 = scr[a]; }
};

// ---- the truth: one token at a time (k_comp_pool decode form, n == 1, start > 0) --------------------------------------------------------
struct Truth
{
    std::vector<Vec> ck;
    Row slot[2]{};
    void feed(int pos, const Row& r)
    {
        slot[pos & 1] = r;
        if (pos & 1) { const size_t g = pos >> 1; if (ck.size() <= g) ck.resize(g + 1); ck[g] = pool(slot[0], slot[1]); }
    }
};

struct Rng
{
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 7) {}
    uint32_t u32() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return (uint32_t)((s * 0x2545F4914F6CDD1Dull) >> 32); }
    int below(int n) { return (int)(u32() % (uint32_t)n); }
};
}   // namespace

int main()
{
    long rounds = 0, risky = 0, neg_wrong = 0, neg_pred = 0, tail_rows = 0;
    int by_parity[2][7] = {};   // [Pa & 1][a]: how many rounds
    for (int scen = 0; scen < 20000; ++scen)
    {
        Rng rng(1000 + scen);
        const int P0 = 1 + rng.below(9);                       // prompt length (prefill, start == 0)
        Engine with, without;
        Truth truth;
        std::vector<Row> pre;
        for (int t = 0; t < P0; ++t) { pre.push_back(row_of(t, 0)); truth.feed(t, pre.back()); }
        with.prefill(pre);
        without.prefill(pre);
        int next = P0;                                          // the next chunk's start = first position not yet in the cache
        long pending_check = -1;                                // the previous round's Pa when it was Pa even && a < n - 1, else -1
        bool pending_risky = false;
        for (int rd = 0; rd < 24; ++rd)
        {
            const int k = rng.below(6);                         // drafts 0..5 (k == 0: a plain decode step, n == 1)
            const int n = k + 1;
            const int a = rng.below(k + 1);                     // accepted drafts 0..k
            const int p = next;
            const int Pa = p + a;                               // = q, the last accepted position
            std::vector<Row> chunk;
            for (int r = 0; r < n; ++r) chunk.push_back(r <= a ? row_of(p + r, 0) : row_of(p + r, 1000 + (uint64_t)rd * 16 + r));   // tail = rejected
            tail_rows += n - 1 - a;
            with.step(p, chunk);
            without.step(p, chunk);

            // (2) the pair completed by THIS round's first row, when the previous round left Pa_prev even (this round starts odd)
            if (pending_check >= 0)
            {
                const size_t g = (size_t)(p >> 1);              // p = Pa_prev + 1 is odd: row 0 of this chunk completes row p >> 1
                Truth t2 = truth;                               // truth up to Pa_prev, plus this chunk's row 0 at position p, in a scratch copy
                t2.feed(p, chunk[0]);
                EXPECT(with.ck.size() > g && t2.ck.size() > g);
                EXPECT(with.ck[g] == t2.ck[g]);                 // (2) with the commit the next step's row is exact
                const bool ok_without = without.ck.size() > g && without.ck[g] == t2.ck[g];
                if (pending_risky) { ++neg_pred; if (!ok_without) ++neg_wrong; }
                else EXPECT(ok_without);                        // no fault was predicted: the uncommitted engine is exact too
            }
            // truth: feed the accepted tokens p .. Pa one at a time
            for (int t = p; t <= Pa; ++t) truth.feed(t, chunk[t - p]);
            with.commit(Pa, a);                                 // after the accept, before the next step; without.* skips it
            // (1) every row whose pair is complete at the accepted prefix
            const int visible = (Pa + 1) / 2;                   // rows g with 2g + 1 <= Pa
            for (int g = 0; g < visible; ++g)
            {
                EXPECT((size_t)g < with.ck.size() && (size_t)g < truth.ck.size());
                // rows completed by an earlier round's REJECTED tail were rewritten by the later step: the with-commit engine must be exact
                if (!(with.ck[g] == truth.ck[g]))
                {
                    // the only row that may still hold a rejected-tail value is the one the next step will rewrite: g == visible is excluded by the bound,
                    // so any mismatch here is a failure
                    std::printf("FAIL row %d scen %d round %d start %d n %d a %d Pa %d\n", g, scen, rd, p, n, a, Pa);
                    return 1;
                }
            }
            ++rounds;
            ++by_parity[Pa & 1][a];
            pending_check = ((Pa & 1) == 0) ? Pa : -1;
            pending_risky = ((Pa & 1) == 0) && a < n - 1;
            risky += pending_risky;
            next = Pa + 1;
        }
    }
    // (3) the negative control saw every predicted fault and nothing else, and the faulty case does occur
    EXPECT(risky > 1000);
    EXPECT(neg_pred > 0);
    EXPECT(neg_wrong == neg_pred);   // without the commit, a round with Pa even and a < n - 1 ALWAYS breaks the next step's row
    // (4) rejected tails existed in the run and none leaked (the (1) loop would have failed)
    EXPECT(tail_rows > 10000);
    EXPECT(by_parity[0][0] > 0 && by_parity[1][0] > 0 && by_parity[0][5] > 0 && by_parity[1][5] > 0);   // a = 0 and a = 5 at both parities
    std::printf("ok: %ld rounds, %ld rounds with Pa even and a < n-1 (the fault), %ld/%ld of them broke the next row without the commit, 0 with it; %ld rejected tail rows\n",
                rounds, risky, neg_wrong, neg_pred, tail_rows);
    return 0;
}
