// Host test of Engram at W-row prefill (a chunk of W = 1024 or 2048 rows through the window, hash, fetch, dequant, sum and gate of one step,
// where the decode path takes at most MX = 32 rows). Host only: no GPU. Every check runs the lines the kernels run (engram_ring.h
// win_wide_row / ring_put_tail, hash.h hash_row / hash_store_ids, engram_math.h dequant1 / gate_from_sums, rowstore.h RowStore) and compares a
// W-row call with what the decode path (32-row slabs) produces for the same rows. Four groups:
//   G1 window  the W-row window (all rows read the ring, then the tail rows enter it) against (a) the 32-row-slab path through ring_put/ring_get,
//              (b) the reference's unbounded positional cache, (c) the true sequence; the ring array itself must equal the slab path's after every
//              step. Prompt of 3W+7 positions in chunks of W, 40 decode steps, DSpark verify chains k = 1..5, 200 random chunk plans, a ring that
//              starts as garbage and survives across requests. Mutants (put before get; tail one row short; a row with a wrong position) must be caught.
//   G2 hash    hash_row + hash_store_ids over W rows (the grid of 128-thread blocks of the wide k_hash) == the same rows in slabs of <= 32, element by element.
//   G3 fetch   4 fake ranks x 2 tables on real files: one W*24-id fetch == 32-row slab fetches (staging buffers equal), owned rows == the file content,
//              k_dequant's formula, then the 4-rank sum in rank order (k_engram_sum) == the owner's dequantized row; W pipeline == slab pipeline bitwise.
//   G4 gate    a host mirror of k_gate's reduction order: 2048 rows (and 33) in one call == slabs of 32, bitwise.
// What this does not cover: the CUDA launch itself (grid, stream order, atomics: the GPU test covers those), and the BF16 trunk
// (test_engram_gpu.cu 7e). G4 shares one host implementation between its two sides, so it shows row independence of the mirror
// only; k_gate itself runs one block per (row, stream).
//   test_engram_wide [scratch dir for G3 files] [seed]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "engram_ring.h"
#include "engram_math.h"
#include "hash_consts.h"
#include "rowstore.h"
using namespace ds41::engram;

static std::mt19937_64 rng;
static int rnd(int lo, int hi) { return lo + int(rng() % uint64_t(hi - lo + 1)); }
static uint64_t splitmix(uint64_t& x)
{
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// ============================================================ G1 window
struct Step { std::vector<int> pos, tok; };
struct Stats { long steps = 0, windows = 0, new_vs_old = 0, new_vs_ref = 0, new_vs_truth = 0, ring_diff = 0, pos_false = 0; };
enum Mut { kNone = 0, kPutFirst = 1, kTailOff1 = 2 };

struct World
{
    int ring_old[kRing], ring_new[kRing];   // the slab path's ring and the wide path's ring: both start as garbage and live across requests
    std::vector<int> cache;                 // the reference's positional cache
    World() : cache(16384 + 64, 5555) { for (int& x : ring_old) x = 7777; for (int& x : ring_new) x = 7777; }
};

static std::vector<int> make_seq(int len) { std::vector<int> T(size_t(len) + 16); for (int& t : T) t = rnd(0, 129279); return T; }
static Step rows_of(const std::vector<int>& T, int p, int m) { Step s; for (int i = 0; i < m; ++i) { s.pos.push_back(p + i); s.tok.push_back(T[size_t(p + i)]); } return s; }

// One step through the four models. wide: the W-row path (win_wide_row for every row, then ring_put_tail for every row); else the decode
// path (every row put, then every row get: k_engram_win), only valid for <= 32 rows.
static void run_step(World& w, const std::vector<int>& T, const Step& s, bool wide, Mut mut, Stats& st)
{
    const int n = int(s.pos.size()), first = s.pos[0];
    std::vector<int> wo(size_t(n) * 4), wn(size_t(n) * 4);
    for (int a = 0; a < n; a += 32)                                   // the old path: slabs of <= 32 rows, put then get
    {
        const int b = std::min(n, a + 32);
        for (int r = a; r < b; ++r) ring_put(w.ring_old, s.pos[size_t(r)], s.tok[size_t(r)]);
        for (int r = a; r < b; ++r) for (int k = 0; k < 4; ++k) wo[size_t(r) * 4 + k] = ring_get(w.ring_old, s.pos[size_t(r)] - k);
    }
    if (wide)
    {
        auto get_all = [&] { for (int r = 0; r < n; ++r) if (!win_wide_row(s.tok.data(), s.pos.data(), w.ring_new, r, &wn[size_t(r) * 4])) ++st.pos_false; };
        auto put_tail = [&] {
            for (int r = 0; r < n; ++r)
            {
                if (mut == kTailOff1) { if (r >= n - kRing + 1) ring_put(w.ring_new, s.pos[size_t(r)], s.tok[size_t(r)]); }   // mutant: one row short
                else ring_put_tail(w.ring_new, s.tok.data(), s.pos.data(), n, r);
            }
        };
        if (mut == kPutFirst) { put_tail(); get_all(); } else { get_all(); put_tail(); }   // mutant: the tail enters the ring before the reads
    }
    else
    {
        for (int r = 0; r < n; ++r) ring_put(w.ring_new, s.pos[size_t(r)], s.tok[size_t(r)]);
        for (int r = 0; r < n; ++r) for (int k = 0; k < 4; ++k) wn[size_t(r) * 4 + k] = ring_get(w.ring_new, s.pos[size_t(r)] - k);
    }
    for (int r = 0; r < n; ++r) w.cache[size_t(s.pos[size_t(r)])] = s.tok[size_t(r)];   // reference: write the step's ids, then gather cache[(pos - shift).clamp_min(0)]
    for (int r = 0; r < n; ++r)
    {
        bool blocked = false;
        for (int k = 0; k < 4; ++k)
        {
            const int p = s.pos[size_t(r)];
            blocked = blocked || p < k;
            const int ref = blocked ? -1 : w.cache[size_t(p - k < 0 ? 0 : p - k)];
            const int tru = (p - k < 0) ? -1 : (p - k >= first ? s.tok[size_t(p - k - first)] : T[size_t(p - k)]);
            const int got = wn[size_t(r) * 4 + k];
            ++st.windows;
            if (got != wo[size_t(r) * 4 + k]) ++st.new_vs_old;
            if (got != ref) ++st.new_vs_ref;
            if (got != tru || ref != tru || wo[size_t(r) * 4 + k] != tru) ++st.new_vs_truth;
        }
    }
    if (memcmp(w.ring_old, w.ring_new, sizeof w.ring_old)) ++st.ring_diff;
    ++st.steps;
}

// DSpark verify rounds from position `next`: rows = [golden, d1..dk]; accepted drafts a; the next step restarts at the first rejected position (+ bonus)
static int verify_rounds(World& w, const std::vector<int>& T, int next, int rounds, int kmax, int small_wide, Mut mut, Stats& st)
{
    for (int i = 0; i < rounds; ++i)
    {
        const int k = rnd(1, kmax);
        Step R; R.pos.push_back(next); R.tok.push_back(T[size_t(next)]);
        for (int d = 1; d <= k; ++d) { R.pos.push_back(next + d); R.tok.push_back(rnd(0, 99) < 55 ? T[size_t(next + d)] : T[size_t(next + d)] ^ 1); }
        int a = 0;
        for (int d = 1; d <= k && R.tok[size_t(d)] == T[size_t(next + d)]; ++d) ++a;
        run_step(w, T, R, small_wide < 0 ? bool(rnd(0, 1)) : bool(small_wide), mut, st);
        next += a + 1;
    }
    return next;
}

// one request on `w`: prompt of plen positions in chunks of W, 40 decode steps, verify rounds k = 1..5 (8 each). small_wide: run the <= 32-row steps
// through the wide path as well (1), or through the decode path as the production driver does (0).
static void request(World& w, int W, int plen, Mut mut, int small_wide, Stats& st)
{
    const int ndec = 40, kmax = 5, rounds = 8;
    const std::vector<int> T = make_seq(plen + ndec + kmax * rounds * (kmax + 1) + 16);
    for (int p = 0; p < plen;)
    {
        const int m = std::min(W, plen - p);
        run_step(w, T, rows_of(T, p, m), m > 32 || small_wide, mut, st);
        p += m;
    }
    int p = plen;
    for (int i = 0; i < ndec; ++i, ++p) run_step(w, T, rows_of(T, p, 1), bool(small_wide), mut, st);
    for (int k = 1; k <= kmax; ++k) p = verify_rounds(w, T, p, rounds, k, small_wide, mut, st);
}

static const int kWs[] = {1, 2, 3, 4, 31, 32, 33, 63, 64, 65, 127, 128, 256, 1024, 2048};

static int g1_window()
{
    const int f0 = g_fail;
    printf("G1 window: wide path vs 32-row slabs vs reference cache vs truth, ring state compared after every step\n");
    for (int sw = 0; sw <= 1; ++sw)
    {
        Stats tot;
        for (int W : kWs)
        {
            World w; Stats st;
            for (int rep = 0; rep < 2; ++rep) request(w, W, 3 * W + 7, kNone, sw, st);      // two requests on one ring: position 0 again over stale slots
            CHECK(st.new_vs_old == 0 && st.new_vs_ref == 0 && st.new_vs_truth == 0 && st.ring_diff == 0 && st.pos_false == 0,
                  "W=%d small_wide=%d: %ld windows, new!=old %ld, new!=ref %ld, !=truth %ld, ring state differs %ld, pos_false %ld", W, sw, st.windows,
                  st.new_vs_old, st.new_vs_ref, st.new_vs_truth, st.ring_diff, st.pos_false);
            tot.steps += st.steps; tot.windows += st.windows;
        }
        printf("  W list %zu sizes, <=32-row steps on the %s path: %ld steps, %ld windows compared\n", sizeof kWs / sizeof kWs[0], sw ? "wide" : "decode", tot.steps, tot.windows);
    }
    {   // 200 random chunk plans on one world: chunk sizes 1..2048, random path for the small chunks, random verify chains
        World w; Stats st;
        for (int plan = 0; plan < 200; ++plan)
        {
            const int len = rnd(1, 6200), plen = rnd(0, len);
            const std::vector<int> T = make_seq(len + 300);
            for (int p = 0; p < plen;)
            {
                const int m = std::min(rnd(1, 2048), plen - p);
                run_step(w, T, rows_of(T, p, m), m > 32 || bool(rnd(0, 1)), kNone, st);
                p += m;
            }
            int p = plen;
            const int nd = std::min(len - plen, rnd(0, 40));
            for (int i = 0; i < nd; ++i, ++p) run_step(w, T, rows_of(T, p, 1), bool(rnd(0, 1)), kNone, st);
            verify_rounds(w, T, p, 6, 5, -1, kNone, st);
        }
        CHECK(st.new_vs_old == 0 && st.new_vs_ref == 0 && st.new_vs_truth == 0 && st.ring_diff == 0 && st.pos_false == 0,
              "random plans: %ld windows, new!=old %ld, new!=ref %ld, !=truth %ld, ring state differs %ld, pos_false %ld", st.windows, st.new_vs_old,
              st.new_vs_ref, st.new_vs_truth, st.ring_diff, st.pos_false);
        printf("  200 random chunk plans (chunks 1..2048): %ld steps, %ld windows compared\n", st.steps, st.windows);
    }
    // mutants: the test must discriminate
    for (int mut = kPutFirst; mut <= kTailOff1; ++mut)
    {
        printf("  mutant %s:", mut == kPutFirst ? "ring_put_tail before win_wide_row" : "tail bound r >= rows-kRing+1 (one row short)");
        for (int W : kWs)
        {
            World w; Stats st;
            for (int rep = 0; rep < 2; ++rep) request(w, W, 3 * W + 7, Mut(mut), 0, st);
            const long win = st.new_vs_ref + st.new_vs_truth + st.new_vs_old;
            printf(" W%d:%ld/%ld", W, win, st.ring_diff);
            if (W >= 64 && mut == kPutFirst) CHECK(win > 0, "mutant put-before-get not caught by the windows at W=%d", W);
            if (W >= 64 && mut == kTailOff1) CHECK(st.ring_diff > 0, "mutant tail-one-short not caught by the ring state at W=%d", W);
        }
        printf("   (window mismatches/ring-state mismatches per W; the checks need >0 from W=64)\n");
    }
    {   // rows that are not consecutive positions: win_wide_row must say so (the kernel counts it, status -2)
        long caught = 0, trials = 0, clean_false = 0;
        for (int W : {2, 3, 4, 31, 32, 33, 64, 65, 128, 1024, 2048})
            for (int trial = 0; trial < 20; ++trial)
            {
                const int p0 = rnd(0, 5000);
                Step s;
                int ring[kRing];
                for (int& x : ring) x = rnd(0, 129279);
                for (int r = 0; r < W; ++r) { s.pos.push_back(p0 + r); s.tok.push_back(rnd(0, 129279)); }
                int w4[4];
                for (int r = 0; r < W; ++r) if (!win_wide_row(s.tok.data(), s.pos.data(), ring, r, w4)) ++clean_false;
                const int j = rnd(1, W - 1);
                switch (trial % 5)
                {
                case 0: for (int r = j; r < W; ++r) s.pos[size_t(r)] += 1; break;                  // a gap from row j on
                case 1: s.pos[size_t(j)] = s.pos[size_t(j) - 1]; break;                           // a duplicate position
                case 2: std::swap(s.pos[size_t(j)], s.pos[size_t(j) - 1]); break;                 // two rows swapped
                case 3: s.pos[size_t(j)] = p0 - 5; break;                                         // a position going backwards
                default: s.pos[0] += 3; break;                                                    // row 0 anchors a different run
                }
                bool any = false;
                for (int r = 0; r < W; ++r) if (!win_wide_row(s.tok.data(), s.pos.data(), ring, r, w4)) any = true;
                ++trials; caught += any;
            }
        CHECK(clean_false == 0 && caught == trials, "position check: %ld clean rows flagged, %ld of %ld corrupted chunks caught", clean_false, caught, trials);
        printf("  position mutants (gap, duplicate, swap, backwards, shifted anchor): %ld of %ld corrupted chunks flagged, %ld clean rows flagged\n", caught, trials, clean_false);
    }
    return g_fail - f0;
}

// ============================================================ G2 hash
static int g2_hash()
{
    const int f0 = g_fail;
    printf("G2 hash: hash_row + hash_store_ids over W rows (grid of 128-thread blocks) vs 32-row slabs, real kHashParams, synthetic cmap\n");
    std::vector<uint32_t> cmap(kVocab);
    for (uint32_t& c : cmap) c = uint32_t(rng() % kCVocab);
    cmap[kPadTok] = kPadId;
    long checked = 0, bad_rows = 0;
    for (int W : {1, 2, 31, 32, 33, 127, 128, 129, 1000, 1024, 2048})
    {
        std::vector<int32_t> win(size_t(W) * 4);
        for (int r = 0; r < W; ++r)
        {
            const int kind = rnd(0, 9);
            const int valid = kind < 6 ? rnd(0, 4) : 4;                        // leading valid tokens; the rest -1 (before the start of the sequence)
            for (int k = 0; k < 4; ++k) win[size_t(r) * 4 + k] = k < valid ? rnd(0, int(kVocab) - 1) : -1;
            if (kind == 9) for (int k = 0; k < 4; ++k) if (rnd(0, 3) == 0) win[size_t(r) * 4 + k] = -1;   // -1 in the middle too
        }
        std::vector<uint32_t> ids_w(size_t(kNE) * W * kIds, 0xDEADBEEFu);
        int bad_w = 0;
        const int blocks = (W + 127) / 128;
        for (int b = 0; b < blocks; ++b)
            for (int t = 0; t < 128; ++t)                                 // k_hash<<<(rows+127)/128, 128>>> : r = blockIdx.x * blockDim.x + threadIdx.x
            {
                const int r = b * 128 + t;
                if (r >= W) continue;
                uint32_t loc[kNE][kIds];
                if (hash_row(kHashParams, cmap.data(), kVocab, win.data() + 4 * r, loc)) ++bad_w;
                hash_store_ids(W, ids_w.data(), r, loc);
            }
        int bad_s = 0;
        long diff = 0;
        for (int a = 0; a < W; a += 32)
        {
            const int n = std::min(32, W - a);
            std::vector<uint32_t> ids_s(size_t(kNE) * n * kIds, 0xDEADBEEFu);
            for (int r = 0; r < n; ++r)                                   // k_hash<<<1, n>>>
            {
                uint32_t loc[kNE][kIds];
                if (hash_row(kHashParams, cmap.data(), kVocab, win.data() + 4 * (a + r), loc)) ++bad_s;
                hash_store_ids(n, ids_s.data(), r, loc);
            }
            for (int e = 0; e < kNE; ++e)
                for (int r = 0; r < n; ++r)
                    for (int j = 0; j < kIds; ++j)
                    {
                        ++checked;
                        diff += ids_w[(size_t(e) * W + size_t(a + r)) * kIds + j] != ids_s[(size_t(e) * n + r) * kIds + j];
                    }
        }
        long unset = 0;
        for (uint32_t x : ids_w) unset += x == 0xDEADBEEFu;
        CHECK(diff == 0 && bad_w == bad_s && bad_w == 0 && unset == 0, "G2 W=%d: %ld ids differ, bad %d vs %d, %ld ids never written", W, diff, bad_w, bad_s, unset);
    }
    {   // a token outside the vocabulary is reported and treated as the pad (blocked) from there on, as in the 8-row path
        for (int trial = 0; trial < 200; ++trial)
        {
            int32_t win[4], wblk[4];
            for (int k = 0; k < 4; ++k) win[k] = rnd(0, int(kVocab) - 1);
            const int j = rnd(0, 3);
            win[j] = trial % 2 ? int32_t(kVocab) + rnd(0, 1000) : INT32_MAX - rnd(0, 5);
            for (int k = 0; k < 4; ++k) wblk[k] = k < j ? win[k] : -1;
            uint32_t a[kNE][kIds], b[kNE][kIds];
            const int ba = hash_row(kHashParams, cmap.data(), kVocab, win, a), bb = hash_row(kHashParams, cmap.data(), kVocab, wblk, b);
            CHECK(ba == 1 && bb == 0 && !memcmp(a, b, sizeof a), "G2 bad token at k=%d: bad=%d, ids %s the blocked window's", j, ba, memcmp(a, b, sizeof a) ? "differ from" : "equal");
            ++bad_rows;
        }
    }
    {   // mutant: the layout stride of a W-row call taken as 32 (the decode kernel's rows) must be caught for every W > 32
        for (int W : {33, 128, 2048})
        {
            std::vector<int32_t> win(size_t(W) * 4);
            for (int32_t& x : win) x = rnd(0, int(kVocab) - 1);
            std::vector<uint32_t> good(size_t(kNE) * W * kIds), mut(size_t(kNE) * W * kIds, 0);
            for (int r = 0; r < W; ++r)
            {
                uint32_t loc[kNE][kIds];
                hash_row(kHashParams, cmap.data(), kVocab, win.data() + 4 * r, loc);
                hash_store_ids(W, good.data(), r, loc);
                hash_store_ids(32, mut.data(), r, loc);
            }
            CHECK(good != mut, "G2 mutant (stride 32 at W=%d) was not caught", W);
        }
    }
    printf("  %ld ids compared wide vs slab (W 1..2048, ragged last block), %ld out-of-vocabulary windows flagged, stride mutant caught\n", checked, bad_rows);
    return g_fail - f0;
}

// ============================================================ G3 fetch + dequant + 4-rank sum
constexpr int kRanks = 4;
constexpr int64_t kV = 8192;   // rows per rank and table; rank R owns the global ids [R*kV, (R+1)*kV)

static void row_bytes(int table, int64_t gid, uint8_t* out)
{
    uint64_t s = ((uint64_t)table << 40) ^ (uint64_t)gid ^ 0x5151ull;
    for (int j = 0; j < 256; j += 8) { uint64_t v = splitmix(s); memcpy(out + j, &v, 8); }
}
static void scale_bytes(int table, int64_t gid, uint8_t* out)   // 8 E8M0 scales: mostly 110..145, a few 0 and 255 (NaN)
{
    uint64_t s = ((uint64_t)table << 40) ^ (uint64_t)gid ^ 0xA5A5ull;
    const uint64_t v = splitmix(s);
    for (int j = 0; j < 8; ++j)
    {
        const uint32_t b = (uint32_t)(v >> (8 * j)) & 255u;
        out[j] = b < 4 ? 0 : b >= 252 ? 255 : (uint8_t)(110 + b % 36);
    }
}
static bool make_file(const std::string& path, int table, int64_t first, int64_t rows)
{
    const uint64_t want = (uint64_t)rows * 256;
    struct stat st;
    if (stat(path.c_str(), &st) == 0 && (uint64_t)st.st_size == want)
    {
        uint8_t a[256], b[256], x[256], y[256];
        FILE* f = fopen(path.c_str(), "rb");
        const bool ok = f && fread(a, 1, 256, f) == 256 && fseek(f, (long)(want - 256), SEEK_SET) == 0 && fread(b, 1, 256, f) == 256;
        if (f) fclose(f);
        row_bytes(table, first, x);
        row_bytes(table, first + rows - 1, y);
        if (ok && !memcmp(a, x, 256) && !memcmp(b, y, 256)) return true;   // reuse: the content is a function of (table, global row)
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    std::vector<uint8_t> buf(256 * 4096);
    for (int64_t r = 0; r < rows; r += 4096)
    {
        const int64_t n = std::min<int64_t>(4096, rows - r);
        for (int64_t i = 0; i < n; ++i) row_bytes(table, first + r + i, &buf[size_t(i) * 256]);
        if (fwrite(buf.data(), 256, size_t(n), f) != size_t(n)) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

// k_dequant, host: stage[(r*24+g)*256 + c*8 + 0..7], scale s8[loc*8 + (c>>2)], emb[r*6144 + g*256 + c*8 + 0..7] = bf16 bits, 0 for ids this rank does not own
static void host_dequant(const uint32_t* ids, const uint8_t* stage, const uint8_t* s8, int64_t first, int64_t valid, int rows, uint16_t* emb)
{
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < kIds; ++g)
        {
            const int64_t loc = (int64_t)ids[size_t(r) * kIds + g] - first;
            for (int c = 0; c < 32; ++c)
            {
                uint16_t* o = emb + size_t(r) * kEmbDim + size_t(g) * kRowBytes + size_t(c) * 8;
                if (loc >= 0 && loc < valid)
                {
                    const uint8_t* w = stage + (size_t(r) * kIds + g) * kRowBytes + size_t(c) * 8;
                    const uint8_t s = s8[size_t(loc) * kScaleBytes + (c >> 2)];
                    for (int k = 0; k < 8; ++k) o[k] = dequant1(w[k], s);
                }
                else for (int k = 0; k < 8; ++k) o[k] = 0;
            }
        }
}
// k_engram_sum, host: fp32 sum of the 4 ranks' bf16 partials in rank order, rounded to bf16
static void host_sum(const std::vector<uint16_t>* part, size_t n, std::vector<uint16_t>& out)
{
    out.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        float s = 0.f;
        for (int p = 0; p < kRanks; ++p) s += bf16_to_f32(part[p][i]);
        out[i] = f32_to_bf16(s);
    }
}
static bool same_value(uint16_t a, uint16_t b)   // -0 == +0, NaN == NaN
{
    const float x = bf16_to_f32(a), y = bf16_to_f32(b);
    return x == y || (std::isnan(x) && std::isnan(y));
}

static int g3_fetch(const std::string& dir)
{
    const int f0 = g_fail;
    printf("G3 fetch: 4 ranks x 2 tables on real files in %s (%lld rows per rank and table), one W*24-id fetch vs 32-row slab fetches\n", dir.c_str(), (long long)kV);
    mkdir(dir.c_str(), 0755);
    std::vector<uint8_t> s8[kRanks][kNE];
    std::vector<TableFile> files[kRanks];
    for (int R = 0; R < kRanks; ++R)
        for (int t = 0; t < kNE; ++t)
        {
            const std::string path = dir + "/r" + std::to_string(R) + "_t" + std::to_string(t) + ".w8";
            if (!make_file(path, t, (int64_t)R * kV, kV)) { fprintf(stderr, "cannot write %s\n", path.c_str()); ++g_fail; return 1; }
            files[R].push_back({path, (int64_t)R * kV, kV});
            s8[R][t].resize(size_t(kV) * kScaleBytes);
            for (int64_t i = 0; i < kV; ++i) scale_bytes(t, (int64_t)R * kV + i, &s8[R][t][size_t(i) * kScaleBytes]);
        }
    StoreOpts o;
    o.cache_bytes = 64ull << 20;
    o.io_threads = 8;
    o.direct = false;   // buffered pread: the scratch dir may be tmpfs; the fetch/cache logic above the read is the same
    auto open_all = [&](std::vector<std::unique_ptr<RowStore>>& rs) {
        for (int R = 0; R < kRanks; ++R)
        {
            rs.emplace_back(new RowStore());
            std::string err;
            if (!rs.back()->open(files[R], o, &err)) { fprintf(stderr, "open rank %d: %s\n", R, err.c_str()); ++g_fail; return false; }
        }
        return true;
    };
    long total_ids = 0, owned_rows = 0, foreign_rows = 0, nobody = 0, sum_cmp = 0;
    for (int W : {33, 256, 1024, 2048})
        for (int t = 0; t < kNE; ++t)
        {
            const size_t n = size_t(W) * kIds;
            std::vector<uint32_t> ids(n);
            std::vector<uint32_t> hot(64);
            for (uint32_t& h : hot) h = uint32_t(rng() % uint64_t(kRanks * kV));
            for (uint32_t& id : ids)
            {
                const int pick = rnd(0, 99);
                id = pick < 2 ? uint32_t(kRanks * kV + rnd(0, 1000)) : pick < 50 ? hot[size_t(rnd(0, 63))] : uint32_t(rng() % uint64_t(kRanks * kV));
            }
            std::vector<std::unique_ptr<RowStore>> rw, rsl;
            if (!open_all(rw) || !open_all(rsl)) return 1;
            std::vector<uint8_t> stw[kRanks], sts[kRanks];
            for (int R = 0; R < kRanks; ++R)
            {
                stw[R].assign(n * kRowBytes, 0xEE);
                sts[R].assign(n * kRowBytes, 0xEE);
                FetchStats fs;
                CHECK(rw[R]->fetch(t, ids.data(), n, stw[R].data(), &fs) == 0, "G3 W=%d t=%d rank %d: W fetch error %d", W, t, R, fs.err);
                for (size_t a = 0; a < size_t(W); a += 32)                 // the decode path: 32 rows (768 ids) per fetch into the same buffer
                {
                    const size_t m = std::min<size_t>(32, size_t(W) - a);
                    FetchStats f2;
                    CHECK(rsl[R]->fetch(t, ids.data() + a * kIds, m * kIds, sts[R].data() + a * kIds * kRowBytes, &f2) == 0, "G3 slab fetch error %d", f2.err);
                }
                CHECK(stw[R] == sts[R], "G3 W=%d t=%d rank %d: W-row staging buffer differs from the 32-row slab buffer", W, t, R);
            }
            // the fetched bytes against the file content: owned rows exact, foreign rows untouched
            long bad_rows = 0;
            for (size_t i = 0; i < n; ++i)
            {
                const uint32_t id = ids[i];
                const int owner = id < uint32_t(kRanks * kV) ? int(id / kV) : -1;
                if (owner < 0) ++nobody;
                for (int R = 0; R < kRanks; ++R)
                {
                    const uint8_t* got = &stw[R][i * kRowBytes];
                    if (R == owner)
                    {
                        uint8_t want[256];
                        row_bytes(t, id, want);
                        bad_rows += memcmp(want, got, 256) != 0;
                        ++owned_rows;
                    }
                    else
                    {
                        bool untouched = true;
                        for (int b = 0; b < 256; ++b) untouched = untouched && got[b] == 0xEE;
                        bad_rows += !untouched;
                        ++foreign_rows;
                    }
                }
            }
            CHECK(bad_rows == 0, "G3 W=%d t=%d: %ld rows differ from the file content / foreign rows touched", W, t, bad_rows);
            total_ids += (long)n;
            // dequant per rank (W rows in one call, and slabs of 32), then the 4-rank sum
            std::vector<uint16_t> ew[kRanks], es[kRanks];
            for (int R = 0; R < kRanks; ++R)
            {
                ew[R].assign(size_t(W) * kEmbDim, 0xFFFF);
                es[R].assign(size_t(W) * kEmbDim, 0xFFFF);
                host_dequant(ids.data(), stw[R].data(), s8[R][t].data(), (int64_t)R * kV, kV, W, ew[R].data());
                for (int a = 0; a < W; a += 32)
                    host_dequant(ids.data() + size_t(a) * kIds, sts[R].data() + size_t(a) * kIds * kRowBytes, s8[R][t].data(), (int64_t)R * kV, kV, std::min(32, W - a),
                                 es[R].data() + size_t(a) * kEmbDim);
                CHECK(ew[R] == es[R], "G3 W=%d t=%d rank %d: W-row partial differs from the slab partial", W, t, R);
            }
            std::vector<uint16_t> sw, ss;
            host_sum(ew, size_t(W) * kEmbDim, sw);
            host_sum(es, size_t(W) * kEmbDim, ss);
            CHECK(sw == ss, "G3 W=%d t=%d: W-row sum differs from the slab sum", W, t);
            // independent expectation from the generators only (not the files, not the fetch): the owner's row dequantized, 0 for rows nobody owns
            long bad_val = 0;
            for (int r = 0; r < W; ++r)
                for (int g = 0; g < kIds; ++g)
                {
                    const uint32_t id = ids[size_t(r) * kIds + g];
                    uint8_t rb[256], sc[8];
                    const bool own = id < uint32_t(kRanks * kV);
                    if (own) { row_bytes(t, id, rb); scale_bytes(t, id, sc); }
                    for (int c = 0; c < 256; ++c)
                    {
                        const uint16_t want = own ? dequant1(rb[c], sc[c >> 5]) : 0;   // scale group = 32 values = c/32
                        ++sum_cmp;
                        bad_val += !same_value(want, sw[size_t(r) * kEmbDim + size_t(g) * kRowBytes + size_t(c)]);
                    }
                }
            CHECK(bad_val == 0, "G3 W=%d t=%d: %ld summed values differ from the owner's dequantized row", W, t, bad_val);
        }
    printf("  W 33/256/1024/2048 x 2 tables: %ld ids fetched per rank, %ld owned reads exact, %ld foreign untouched, %ld ids owned by nobody, %ld summed values checked\n",
           total_ids, owned_rows, foreign_rows, nobody, sum_cmp);
    return g_fail - f0;
}

// ============================================================ G5 shared-RowStore inter-chunk L001 lookahead warm
static int g5_lookahead(const std::string& dir)
{
    const int f0 = g_fail;
    printf("G5 lookahead: 4 ranks x 3 chunks (c0..c2) shared-RowStore L001 cache warm vs flag-OFF bitwise parity + warm repeat 0-read check\n");
    std::vector<uint8_t> s8[kRanks][kNE];
    std::vector<TableFile> files[kRanks];
    for (int R = 0; R < kRanks; ++R)
        for (int t = 0; t < kNE; ++t)
        {
            const std::string path = dir + "/r" + std::to_string(R) + "_t" + std::to_string(t) + ".w8";
            files[R].push_back({path, (int64_t)R * kV, kV});
            s8[R][t].resize(size_t(kV) * kScaleBytes);
            for (int64_t i = 0; i < kV; ++i) scale_bytes(t, (int64_t)R * kV + i, &s8[R][t][size_t(i) * kScaleBytes]);
        }
    StoreOpts o;
    o.cache_bytes = 64ull << 20;
    o.io_threads = 8;
    o.direct = false;

    HashParams hp = kHashParams;
    for (int e = 0; e < kNE; ++e)
        for (int g = 0; g < kIds; ++g)
        {
            hp.prime[e][g] = 1291u;
            hp.offset[e][g] = uint32_t(g * 1365);
        }
    std::vector<uint32_t> cmap(kVocab);
    for (uint32_t i = 0; i < kVocab; ++i) cmap[i] = uint32_t(rng() % kCVocab);

    for (int W : {256, 1024})
    {
        constexpr int nchunks = 3;
        const int prompt_rows = nchunks * W + 1;
        std::vector<int> prompt((size_t)prompt_rows);
        for (int& tok : prompt) tok = rnd(0, kVocab - 1);

        std::vector<std::unique_ptr<RowStore>> rs_off(kRanks), rs_on(kRanks);
        for (int R = 0; R < kRanks; ++R)
        {
            rs_off[R].reset(new RowStore());
            rs_on[R].reset(new RowStore());
            std::string err;
            CHECK(rs_off[R]->open(files[R], o, &err), "G5 off open rank %d: %s", R, err.c_str());
            CHECK(rs_on[R]->open(files[R], o, &err), "G5 on open rank %d: %s", R, err.c_str());
        }

        const size_t n = size_t(W) * kIds;
        // Build exact windows and IDs for each chunk c0..c2
        std::vector<int32_t> win[nchunks];
        std::vector<uint32_t> ids[nchunks];
        int ring[kRing];
        for (int& x : ring) x = 7777;
        for (int c = 0; c < nchunks; ++c)
        {
            const int first = c * W;
            std::vector<int> pos(W);
            for (int r = 0; r < W; ++r) pos[r] = first + r;
            win[c].resize(size_t(W) * kNGram);
            for (int r = 0; r < W; ++r)
                CHECK(win_wide_row(prompt.data() + first, pos.data(), ring, r, &win[c][size_t(r) * kNGram]), "G5 win_wide_row chunk %d row %d", c, r);
            for (int r = 0; r < W; ++r)
                ring_put_tail(ring, prompt.data() + first, pos.data(), W, r);
            ids[c].resize(size_t(kNE) * n);
            for (int r = 0; r < W; ++r)
            {
                uint32_t loc[kNE][kIds];
                CHECK(hash_row(hp, cmap.data(), kVocab, &win[c][size_t(r) * kNGram], loc) == 0, "G5 hash_row chunk %d row %d", c, r);
                hash_store_ids(W, ids[c].data(), r, loc);
            }
        }

        // Verify that Engram::plan_next_warm + warm_loop window/L001-hash logic produces identical L001 IDs for chunks c1..c2
        for (int c = 0; c + 1 < nchunks; ++c)
        {
            const int next_first = (c + 1) * W;
            const int rows = std::min(W, prompt_rows - 1 - next_first);
            CHECK(rows == W, "G5 next chunk rows=%d != W=%d", rows, W);
            std::vector<int32_t> warm_win(size_t(rows) * kNGram);
            std::vector<uint32_t> warm_ids(size_t(rows) * kIds);
            for (int r = 0; r < rows; ++r)
                for (int k = 0; k < kNGram; ++k)
                    warm_win[size_t(r) * kNGram + k] = (next_first + r - k >= 0) ? prompt[next_first + r - k] : -1;
            CHECK(warm_win == win[c + 1], "G5 chunk %d->%d warm_win differs from exact win_wide_row", c, c + 1);
            for (int r = 0; r < rows; ++r)
            {
                uint32_t loc[kNE][kIds];
                CHECK(hash_row(hp, cmap.data(), kVocab, &warm_win[size_t(r) * kNGram], loc) == 0, "G5 warm hash_row");
                memcpy(&warm_ids[size_t(r) * kIds], loc[0], kIds * sizeof(uint32_t));
            }
            CHECK(memcmp(warm_ids.data(), ids[c + 1].data(), n * sizeof(uint32_t)) == 0, "G5 chunk %d->%d warm L001 IDs differ from exact GPU hash IDs", c, c + 1);
        }

        // Run c0..c2 on both flag-OFF and flag-ON (shared RowStore L001 warm after each chunk's L014 fetch)
        for (int c = 0; c < nchunks; ++c)
        {
            for (int t = 0; t < kNE; ++t)
            {
                std::vector<uint16_t> emb_off[kRanks], emb_on[kRanks];
                for (int R = 0; R < kRanks; ++R)
                {
                    std::vector<uint8_t> st_off(n * kRowBytes, 0xEE), st_on(n * kRowBytes, 0xEE);
                    FetchStats fs_off, fs_on;
                    CHECK(rs_off[R]->fetch(t, ids[c].data() + size_t(t) * n, n, st_off.data(), &fs_off) == 0, "G5 off fetch err");
                    CHECK(rs_on[R]->fetch(t, ids[c].data() + size_t(t) * n, n, st_on.data(), &fs_on) == 0, "G5 on fetch err");
                    if (c > 0 && t == 0)
                    {
                        CHECK(fs_on.miss_uniq == 0 && fs_on.blocks == 0,
                              "G5 W=%d chunk %d rank %d: prefetched L001 exact fetch had miss_uniq=%u blocks=%u (want 0)",
                              W, c, R, fs_on.miss_uniq, fs_on.blocks);
                    }
                    CHECK(st_off == st_on, "G5 W=%d chunk %d table %d rank %d: fetched staging bytes differ between flag-OFF and flag-ON", W, c, t, R);
                    emb_off[R].assign(size_t(W) * kEmbDim, 0);
                    emb_on[R].assign(size_t(W) * kEmbDim, 0);
                    host_dequant(ids[c].data() + size_t(t) * n, st_off.data(), s8[R][t].data(), (int64_t)R * kV, kV, W, emb_off[R].data());
                    host_dequant(ids[c].data() + size_t(t) * n, st_on.data(), s8[R][t].data(), (int64_t)R * kV, kV, W, emb_on[R].data());
                    CHECK(emb_off[R] == emb_on[R], "G5 W=%d chunk %d table %d rank %d: dequant differs between flag-OFF and flag-ON", W, c, t, R);
                }
                std::vector<uint16_t> sum_off, sum_on;
                host_sum(emb_off, size_t(W) * kEmbDim, sum_off);
                host_sum(emb_on, size_t(W) * kEmbDim, sum_on);
                CHECK(sum_off == sum_on, "G5 W=%d chunk %d table %d: 4-rank sum differs between flag-OFF and flag-ON", W, c, t);
            }
            // At end of chunk c's L014 fetch (t == 1), warm chunk c+1's L001 rows into the shared RowStore (dst = nullptr)
            if (c + 1 < nchunks)
            {
                for (int R = 0; R < kRanks; ++R)
                {
                    FetchStats warm_st;
                    CHECK(rs_on[R]->fetch(0, ids[c + 1].data(), n, nullptr, &warm_st) == 0, "G5 warm fetch err");
                }
            }
        }

        // Warm 8K repeat check: re-running warm + exact fetch on already-cached chunks c0..c2 issues 0 NVMe reads
        for (int c = 0; c < nchunks; ++c)
        {
            for (int R = 0; R < kRanks; ++R)
            {
                if (c + 1 < nchunks)
                {
                    FetchStats warm_rep;
                    CHECK(rs_on[R]->fetch(0, ids[c + 1].data(), n, nullptr, &warm_rep) == 0, "G5 warm repeat err");
                    CHECK(warm_rep.blocks == 0 && warm_rep.miss_uniq == 0,
                          "G5 W=%d chunk %d rank %d: warm repeat lookahead issued %u reads (want 0)", W, c + 1, R, warm_rep.blocks);
                }
                std::vector<uint8_t> st_rep(n * kRowBytes, 0xEE);
                FetchStats exact_rep;
                CHECK(rs_on[R]->fetch(0, ids[c].data(), n, st_rep.data(), &exact_rep) == 0, "G5 exact repeat err");
                CHECK(exact_rep.blocks == 0 && exact_rep.miss_uniq == 0,
                      "G5 W=%d chunk %d rank %d: warm repeat exact fetch issued %u reads (want 0)", W, c, R, exact_rep.blocks);
            }
        }

        // Out-of-vocabulary speculative window is rejected by hash_row before touching RowStore
        int32_t bad_win[kNGram] = {kVocab, 0, 0, 0};
        uint32_t bad_loc[kNE][kIds];
        CHECK(hash_row(hp, cmap.data(), kVocab, bad_win, bad_loc) != 0, "G5 out-of-vocab speculative window was not rejected");

        // Two-lane prefill2 quantum (c0 + c1, e1_fetches = 2) warms c2 after lane B's L014 fetch
        std::vector<std::unique_ptr<RowStore>> rs_pf2(kRanks);
        for (int R = 0; R < kRanks; ++R)
        {
            rs_pf2[R].reset(new RowStore());
            std::string err;
            CHECK(rs_pf2[R]->open(files[R], o, &err), "G5 pf2 open rank %d: %s", R, err.c_str());
        }
        int e1_left = 2;
        for (int lane = 0; lane < 2; ++lane)
            for (int t = 0; t < kNE; ++t)
            {
                for (int R = 0; R < kRanks; ++R)
                {
                    std::vector<uint8_t> st_lane(n * kRowBytes, 0xEE);
                    FetchStats fs_lane;
                    CHECK(rs_pf2[R]->fetch(t, ids[lane].data() + size_t(t) * n, n, st_lane.data(), &fs_lane) == 0, "G5 pf2 lane %d t=%d fetch err", lane, t);
                }
                if (t == 1 && --e1_left == 0)
                {
                    CHECK(lane == 1, "G5 pf2 e1_fetches=2 armed on lane %d (want lane 1)", lane);
                    for (int R = 0; R < kRanks; ++R)
                    {
                        FetchStats warm_pf2;
                        CHECK(rs_pf2[R]->fetch(0, ids[2].data(), n, nullptr, &warm_pf2) == 0, "G5 pf2 warm c2 err");
                    }
                }
            }
        for (int R = 0; R < kRanks; ++R)
        {
            std::vector<uint8_t> st_c2(n * kRowBytes, 0xEE);
            FetchStats fs_c2;
            CHECK(rs_pf2[R]->fetch(0, ids[2].data(), n, st_c2.data(), &fs_c2) == 0, "G5 pf2 c2 exact fetch err");
            CHECK(fs_c2.miss_uniq == 0 && fs_c2.blocks == 0,
                  "G5 pf2 W=%d c2 rank %d: prefetched L001 exact fetch had miss_uniq=%u blocks=%u (want 0)",
                  W, R, fs_c2.miss_uniq, fs_c2.blocks);
        }
    }
    printf("  W 256/1024 x 3 chunks x 4 ranks: prefetched L001 exact misses=0 reads=0, staging & 4-rank sum bit-identical to flag-OFF, warm repeat reads=0\n");
    return g_fail - f0;
}

// ============================================================ G4 gate
struct HostMath
{
    static float rsqrt(float x) { return 1.f / sqrtf(x); }
    static float sqrt(float x) { return sqrtf(x); }
    static float exp(float x) { return expf(x); }
};

// One (row, stream) block of k_gate, in the kernel's order: 512 threads, element tid + q*512 (q = 0..9) accumulated in q order, an xor-shuffle tree over the
// 32 lanes of each warp (lane l adds lane l^o for o = 16..1), the 16 warp partials summed sequentially, the gate, then t = bf16(h + gate * value) unfused.
static void gate_block(float* t, const uint16_t* key, const uint16_t* val, const float* w)
{
    constexpr int NT = 512, PER = kHid / NT;
    float hv[PER][NT], sh[NT], sk[NT], shk[NT];
    for (int tid = 0; tid < NT; ++tid)
    {
        float a = 0.f, b = 0.f, c = 0.f;
        for (int q = 0; q < PER; ++q)
        {
            const int i = tid + q * NT;
            hv[q][tid] = t[i];
            const float kk = bf16_to_f32(key[i]);
            const float h2 = hv[q][tid] * hv[q][tid], k2 = kk * kk, hwk = (hv[q][tid] * w[i]) * kk;
            a += h2; b += k2; c += hwk;
        }
        sh[tid] = a; sk[tid] = b; shk[tid] = c;
    }
    float red[NT / 32][3];
    for (int wp = 0; wp < NT / 32; ++wp)
    {
        float x[3][32], y[3][32];
        for (int l = 0; l < 32; ++l) { x[0][l] = sh[wp * 32 + l]; x[1][l] = sk[wp * 32 + l]; x[2][l] = shk[wp * 32 + l]; }
        for (int o = 16; o > 0; o >>= 1)
        {
            for (int v = 0; v < 3; ++v) for (int l = 0; l < 32; ++l) y[v][l] = x[v][l] + x[v][l ^ o];
            memcpy(x, y, sizeof x);
        }
        red[wp][0] = x[0][0]; red[wp][1] = x[1][0]; red[wp][2] = x[2][0];
    }
    float a = 0.f, b = 0.f, c = 0.f;
    for (int k = 0; k < NT / 32; ++k) { a += red[k][0]; b += red[k][1]; c += red[k][2]; }
    const float gate = gate_from_sums<HostMath>(a, b, c, 1e-20f);
    for (int tid = 0; tid < NT; ++tid)
        for (int q = 0; q < PER; ++q)
        {
            const int i = tid + q * NT;
            const float prod = gate * bf16_to_f32(val[i]);
            t[i] = round_bf16(hv[q][tid] + prod);
        }
}
// k_gate<<<dim3(rows, 4), 512>>>(T, kv, wgt, rows): T [rows][4][5120] fp32, kv [rows][25600] bf16 (stream j's key at j*5120, the value at 4*5120)
static void gate_call(float* T, const uint16_t* kv, const float* wgt, int rows)
{
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < kStreams; ++j)
            gate_block(T + (size_t(r) * kStreams + j) * kHid, kv + size_t(r) * kKvDim + size_t(j) * kHid, kv + size_t(r) * kKvDim + size_t(kStreams) * kHid,
                       wgt + size_t(j) * kHid);
}
static float rnd_f(float scale) { return float(int64_t(rng() >> 40) - (1 << 23)) / float(1 << 23) * scale; }

static int g4_gate()
{
    const int f0 = g_fail;
    printf("G4 gate: host mirror of k_gate's reduction order, W rows in one call vs slabs of 32 rows\n");
    std::vector<float> wgt(size_t(kStreams) * kHid);
    for (float& x : wgt) x = rnd_f(0.1f);
    for (int W : {33, 2048})
    {
        std::vector<float> T(size_t(W) * kStreams * kHid);
        std::vector<uint16_t> kv(size_t(W) * kKvDim);
        for (float& x : T) x = round_bf16(rnd_f(2.f));
        for (uint16_t& x : kv) x = f32_to_bf16(rnd_f(0.5f));
        std::vector<float> Tw = T, Ts = T;
        gate_call(Tw.data(), kv.data(), wgt.data(), W);
        for (int a = 0; a < W; a += 32)
            gate_call(Ts.data() + size_t(a) * kStreams * kHid, kv.data() + size_t(a) * kKvDim, wgt.data(), std::min(32, W - a));
        long changed = 0, nonfinite = 0;
        for (size_t i = 0; i < T.size(); ++i) { changed += Tw[i] != T[i]; nonfinite += !std::isfinite(Tw[i]); }
        CHECK(!memcmp(Tw.data(), Ts.data(), Tw.size() * sizeof(float)), "G4 W=%d: W-row result differs bitwise from the slab result", W);
        CHECK(changed > 0 && nonfinite == 0, "G4 W=%d: gate changed %ld values, %ld non-finite", W, changed, nonfinite);
        printf("  W=%d: %zu values, bitwise equal to the 32-row slabs, %ld changed by the gate\n", W, Tw.size(), changed);
    }
    return g_fail - f0;
}

int main(int argc, char** argv)
{
    const std::string dir = argc > 1 ? argv[1] : "/tmp/engram-wide-test/rs";
    const uint64_t seed = argc > 2 ? strtoull(argv[2], nullptr, 10) : 20261026;
    static_assert(kRing >= 32 + 3, "ring must hold MX rows + 3");
    rng.seed(seed);
    int bad = 0;
    bad += g1_window() != 0;
    bad += g2_hash() != 0;
    bad += g3_fetch(dir) != 0;
    bad += g5_lookahead(dir) != 0;
    bad += g4_gate() != 0;
    printf("%s (seed %llu, %d failed checks)\n", bad || g_fail ? "FAIL" : "PASS", (unsigned long long)seed, g_fail);
    return bad || g_fail ? 1 : 0;
}
