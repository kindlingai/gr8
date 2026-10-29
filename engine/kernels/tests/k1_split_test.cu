// k1_split_test: the K1 split as bind.cuh's k1_split / k1_fuse_split choose it, host only (no GPU, no launches).
// (1) A per-row-count table in G6_K1_SPLIT ("1:/4,3:2x1,4-32:/4", #107) is not the "KS1,WG,NS2" triple: k1_split keeps
//     the release split at every M. Before, sscanf read its leading 1 as KS1 at every M; with the release WG = 1 at
//     M = 1 the fused remap then gave KS1 = 0 (no fc1 items, wrong output).
// (2) The triple still sets every M, and G6_K1_SPLIT_M1/M2/M4 still win over it.
// (3) k1_fuse_split refuses KS1 x WG < 2 and maps the release splits to (KS1*WG/2, 2).
// (4) The table reader (k1_split_env, #107): powers of two only (the kernels index units by __ffs shifts since #136),
//     malformed entries refused, a table with a G6_K1_SPLIT_M* triple refused.
// (5) k1_bind_split, the split make_k1 binds for a trellis-shared layer, at every M 1..G6_MX, separate and fused, with
//     the table in the environment as the launcher passes it: every fc1 partial covers the same k range as the release
//     split's (same KS1 x WG, so fc2 sums the same partials in the same order), NS2 is the table's, k1_split_check
//     passes; an fc1 entry with another KS1 x WG fails the bind (split_m 0) or keeps the class's shape (split_m set).
// (6) The kernels' unit decode (k1.cuh fc1 / fc2 / fused, tr3.cuh bodies; host copies of their shift expressions):
//     for every split k1_split_check accepts, slot / block / matrix / k part / n part == the division form, unit by
//     unit; for NS2 = 3 (which divides 384) the shift form differs, and k1_split_check refuses it.
//
//   k1_split_test            exit code 0 all PASS, 1 any FAIL
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "bind.cuh"

static int fails = 0;
static void check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAIL: %s\n", what); ++fails; }
}
static bool same(const bind::K1Split& a, const bind::K1Split& b)
{
    return a.KS1 == b.KS1 && a.WG == b.WG && a.NS2 == b.NS2 && a.s_ks == b.s_ks && a.s_ns2 == b.s_ns2;
}
static void clear_env()
{
    unsetenv("G6_K1_SPLIT"); unsetenv("G6_K1_SPLIT_M1"); unsetenv("G6_K1_SPLIT_M2"); unsetenv("G6_K1_SPLIT_M4");
}

int main()
{
    const int Ms[] = {1, 2, 3, 4, 5, 8, 12, 32};
    clear_env();
    bind::K1Split rel[8];
    for (int i = 0; i < 8; ++i) rel[i] = bind::k1_split(Ms[i]);

    // (1) tables: the release split at every M
    const char* tables[] = {"1:/4,3:2x1,4-32:/4", "1:/4,2-3:2x1,4-32:8x1/4", "1=/4"};
    for (const char* t : tables)
    {
        setenv("G6_K1_SPLIT", t, 1);
        for (int i = 0; i < 8; ++i)
        {
            char what[160];
            std::snprintf(what, sizeof what, "G6_K1_SPLIT=%s M=%d: KS1 %d WG %d NS2 %d, release %d %d %d", t, Ms[i],
                          bind::k1_split(Ms[i]).KS1, bind::k1_split(Ms[i]).WG, bind::k1_split(Ms[i]).NS2, rel[i].KS1, rel[i].WG, rel[i].NS2);
            check(same(bind::k1_split(Ms[i]), rel[i]), what);
            bind::K1Split f = bind::k1_split(Ms[i]);
            std::snprintf(what, sizeof what, "G6_K1_SPLIT=%s M=%d: fused split refused or KS1 < 1", t, Ms[i]);
            check(bind::k1_fuse_split(f) == nullptr && f.KS1 >= 1, what);
        }
    }

    // (2) the triple at every M; the per-M variables over it
    setenv("G6_K1_SPLIT", "2,1,4", 1);
    for (int i = 0; i < 8; ++i)
    {
        const bind::K1Split s = bind::k1_split(Ms[i]);
        check(s.KS1 == 2 && s.WG == 1 && s.NS2 == 4 && s.s_ks == rel[i].s_ks && s.s_ns2 == rel[i].s_ns2, "G6_K1_SPLIT=2,1,4 at every M");
    }
    setenv("G6_K1_SPLIT_M1", "8,1,2", 1);
    check(bind::k1_split(1).KS1 == 8 && bind::k1_split(1).NS2 == 2, "G6_K1_SPLIT_M1 over G6_K1_SPLIT at M=1");
    check(bind::k1_split(2).KS1 == 2 && bind::k1_split(2).NS2 == 4, "G6_K1_SPLIT at M=2 with only G6_K1_SPLIT_M1 set");
    clear_env();

    // (3) the fused remap
    bind::K1Split one = rel[0];
    one.KS1 = 1; one.WG = 1;
    check(bind::k1_fuse_split(one) != nullptr, "k1_fuse_split refuses KS1 x WG = 1 (G6_K1_SPLIT_M1=1,1,2)");
    setenv("G6_K1_SPLIT_M1", "1,1,2", 1);
    bind::K1Split m1 = bind::k1_split(1);
    check(bind::k1_fuse_split(m1) != nullptr, "G6_K1_SPLIT_M1=1,1,2: fused split refused");
    clear_env();
    for (int i = 0; i < 8; ++i)
    {
        bind::K1Split f = rel[i];
        const char* e = bind::k1_fuse_split(f);
        check(!e && f.WG == 2 && f.KS1 * 2 == rel[i].KS1 * rel[i].WG, "release split: fused (KS1*WG/2, 2)");
    }

    // #107's table, its last range clipped to the build's row limit (G6_MX 8 without DEC_S2: 4-32 is refused there)
    char tab107[64];
    std::snprintf(tab107, sizeof tab107, "1:/4,3:2x1,4-%d:/4", G6_MX);

    // (4) the table reader
    {
        struct { const char* t; bool ok; } T[] = {
            {tab107, true}, {"1=/4", true}, {"2-3:2x1/4", true}, {"4-8:8x1/32", true},
            {"1:/3", false}, {"1:/6", false}, {"1:/12", false}, {"1:/24", false}, {"1:/48", false}, {"1:/64", false},
            {"3:3x1", false}, {"3:6x1", false}, {"3:-2x1", false}, {"3:0x1", false}, {"1:/0", false}, {"1:/-4", false},
            {"3:2x3", false}, {"3:2x8", false}, {"0:/4", false}, {"3-2:/4", false}, {"1:/4;", false}, {"1:4", false},
            {"9999:/4", false}, {"1:0x1/4", false}, {"1:4x1/0", false}, {"1:4x1/", false}, {"1:x1", false}, {"1:4x", false},
            {"1:/", false}, {":/4", false}, {"1-:/4", false}, {"1:/4,", false}, {"1:/4,,3:/4", false}, {"1:+2x1", false},
            {"1:/4 ", false}, {"3:2x1/8", true}, {"1-4:/4", true}};
        for (auto& c : T)
        {
            const char* e = bind::k1_split_env(c.t).err;
            char what[200];
            std::snprintf(what, sizeof what, "table %s: %s (%s)", c.t, c.ok ? "accepted" : "refused", e ? e : "no error");
            check((e == nullptr) == c.ok, what);
        }
        // 16x1: 384 / 16 = 24 k16 steps, 3 groups of 8: accepted by the parser where KSH / 16 is a multiple of 8
        check((bind::k1_split_env("4-8:16x1").err == nullptr) == ((KSH / 16) % 8 == 0), "table 4-8:16x1 iff KSH / 16 is a multiple of 8");
        const bind::K1SplitEnv& o = bind::k1_split_env(tab107);
        check(o.ns2[1] == 4 && o.ks1[1] == 0 && o.ks1[3] == 2 && o.wg[3] == 1 && o.ns2[3] == 0 && o.ns2[4] == 4 && o.ns2[G6_MX] == 4 && o.ns2[2] == 0,
              "table 1:/4,3:2x1,4-G6_MX:/4 entries");
        setenv("G6_K1_SPLIT_M4", "4,2,2", 1);
        check(bind::k1_split_env("1:/4").err != nullptr, "a table with G6_K1_SPLIT_M4 set is refused");
        clear_env();
        // the triple form is k1_split's: the table reader sees no table and no error
        for (const char* t : {"2,1,4", "4,1,2", "4,2,2"})
        {
            const bind::K1SplitEnv& e = bind::k1_split_env(t);
            bool empty = !e.err;
            for (int M = 1; M <= G6_MX; ++M) empty = empty && !e.ks1[M] && !e.ns2[M];
            check(empty, "a triple is not a table (no entries, no error)");
        }
        bind::k1_split_env("");
    }

    // (5) the bound split with the table in the environment (both readers see it, as on a launch)
    {
        const char* tab = tab107;
        bind::K1Split relf[2][G6_MX + 1];
        for (int fu = 0; fu < 2; ++fu)
            for (int M = 1; M <= G6_MX; ++M)
                check(!bind::k1_bind_split(M, 0, 0, true, fu, relf[fu][M]), "release split binds at every M (separate and fused)");
        setenv("G6_K1_SPLIT", tab, 1);
        bind::k1_split_env(tab);
        for (int fu = 0; fu < 2; ++fu)
            for (int M = 1; M <= G6_MX; ++M)
            {
                bind::K1Split s;
                const char* e = bind::k1_bind_split(M, 0, 0, true, fu, s);
                const bind::K1Split& r = relf[fu][M];
                char what[200];
                std::snprintf(what, sizeof what, "table %s M=%d fuse=%d: %s, KS1 %d WG %d NS2 %d vs release %d %d %d", tab, M, fu, e ? e : "bound",
                              s.KS1, s.WG, s.NS2, r.KS1, r.WG, r.NS2);
                check(!e && s.KS1 * s.WG == r.KS1 * r.WG && s.NS2 == (M == 2 || M == 3 ? r.NS2 : 4), what);
                // the shape itself: 3 rows take 2x1 (fused: remapped to 1x2); elsewhere the release shape stays
                if (M == 3) check(fu ? (s.KS1 == 1 && s.WG == 2) : (s.KS1 == 2 && s.WG == 1), "table 3:2x1 taken at M=3 (fused: 1x2)");
                else check(s.KS1 == r.KS1 && s.WG == r.WG, "no fc1 entry: the release shape");
                // partial p = h * WG + q covers k16 steps [p * KSH / GP, (p + 1) * KSH / GP) in both (tr3_fc1_body ks0 + ksq)
                const int GP = s.KS1 * s.WG;
                for (int h = 0; h < s.KS1; ++h)
                    for (int q = 0; q < s.WG; ++q)
                    {
                        const int KSU = KSH / s.KS1, steps = KSU / s.WG, k0 = h * KSU + q * steps, p = h * s.WG + q;
                        const int hr = p / r.WG, qr = p % r.WG, KSUr = KSH / r.KS1, stepsr = KSUr / r.WG;
                        check(k0 == hr * KSUr + qr * stepsr && steps == stepsr && k0 == p * (KSH / GP), "table partial k range == release");
                    }
                // the sums of the S2 classes stay theirs (split_m: the M-row entry is not taken where KS1 x WG differs)
                for (int sm : {1, 3, 5})
                {
                    bind::K1Split c, cr;
                    unsetenv("G6_K1_SPLIT");
                    bind::k1_split_env("");
                    bind::k1_bind_split(M, sm, 0, true, fu, cr);
                    setenv("G6_K1_SPLIT", tab, 1);
                    bind::k1_split_env(tab);
                    const char* ec = bind::k1_bind_split(M, sm, 0, true, fu, c);
                    check(!ec && c.KS1 * c.WG == cr.KS1 * cr.WG, "table under an S2 class (split_m): the class's partial count");
                    const bool taken = M == 3 && cr.KS1 * cr.WG == 2;          // the only fc1 entry, and its product
                    if (!taken) check(c.KS1 == cr.KS1 && c.WG == cr.WG, "table under an S2 class: the class's shape where the entry is not taken");
                }
            }
        // an fc1 entry with another KS1 x WG: refused for the M-row step, the class's shape kept under split_m
        setenv("G6_K1_SPLIT", "4:2x1", 1);
        bind::k1_split_env("4:2x1");
        bind::K1Split s;
        check(bind::k1_bind_split(4, 0, 0, true, 0, s) != nullptr, "table 4:2x1 (KS1 x WG 2, release 8) refused at M=4");
        check(bind::k1_bind_split(4, 3, 0, true, 0, s) == nullptr && s.KS1 == 2 && s.WG == 1, "table 4:2x1 under split_m 3 (class KS1 x WG 2) taken");
        check(bind::k1_bind_split(4, 5, 0, true, 0, s) == nullptr && s.KS1 * s.WG == 8 && !(s.KS1 == 2 && s.WG == 1), "table 4:2x1 under split_m 5 (class 8): class shape");
        check(bind::k1_bind_split(4, 0, 0, false, 0, s) == nullptr && s.KS1 * s.WG == 8, "table not applied to a layer without the trellis shared expert");
        clear_env();
        bind::k1_split_env("");
        // fuse 2 (its own sums: KS1 2 where fc1 has 1 k part): the table leaves fuse 2's split as without it
        for (int M = 1; M <= G6_MX; ++M)
        {
            bind::K1Split a2, b2;
            clear_env();
            bind::k1_split_env("");
            const char* ea = bind::k1_bind_split(M, 0, 0, true, 2, a2);
            setenv("G6_K1_SPLIT", tab, 1);
            bind::k1_split_env(tab);
            const char* eb = bind::k1_bind_split(M, 0, 0, true, 2, b2);
            check(!ea && !eb && a2.KS1 == b2.KS1 && a2.WG == b2.WG, "fuse 2: the table keeps fuse 2's fc1 split");
        }
        // valid triples (k1_split's form) bind on trellis-shared layers, with the reader fresh from the environment
        for (const char* t : {"2,1,4", "4,1,2", "4,2,2", "8,1,2"})
        {
            clear_env();
            setenv("G6_K1_SPLIT", t, 1);
            bind::k1_split_env(getenv("G6_K1_SPLIT"));
            int ks, wg, ns;
            std::sscanf(t, "%d,%d,%d", &ks, &wg, &ns);
            for (int fu = 0; fu < 2; ++fu)
                for (int M : {1, 3, 8})
                {
                    char what[120];
                    std::snprintf(what, sizeof what, "triple G6_K1_SPLIT=%s binds at M=%d fuse=%d", t, M, fu);
                    const char* e = bind::k1_bind_split(M, 0, 0, true, fu, s);
                    check(!e && s.NS2 == ns && s.KS1 * s.WG == ks * wg && (fu || (s.KS1 == ks && s.WG == wg)), what);
                }
        }
        clear_env();
        setenv("G6_K1_SPLIT", "4,2,2", 1);
        setenv("G6_K1_SPLIT_M4", "8,1,2", 1);
        bind::k1_split_env(getenv("G6_K1_SPLIT"));
        check(!bind::k1_bind_split(8, 0, 0, true, 0, s) && s.KS1 == 8 && s.WG == 1, "triple with G6_K1_SPLIT_M4: the per-M triple at 4+ rows");
        check(!bind::k1_bind_split(1, 0, 0, true, 0, s) && s.KS1 == 4 && s.WG == 2, "triple with G6_K1_SPLIT_M4: the triple at 1 row");
        clear_env();
        bind::k1_split_env("");
        // the triples are checked too: KS1 3 / NS2 6 (they divide 384) and WG 8 are refused at bind
        for (const char* t : {"3,1,2", "4,2,6", "1,8,2", "4,1,48", "32,1,2"})
        {
            setenv("G6_K1_SPLIT", t, 1);
            char what[120];
            std::snprintf(what, sizeof what, "triple G6_K1_SPLIT=%s refused at bind", t);
            check(bind::k1_bind_split(4, 0, 0, true, 0, s) != nullptr, what);
        }
        clear_env();
    }

    // (6) the kernels' unit decode: shifts (k1.cuh, tr3.cuh) == division, for every accepted split
    {
        constexpr int EB = g6nr::EB;
        auto ffs = [](int x) { return __builtin_ffs(x); };
        auto decode_ok = [&](int KS1, int WG, int NS2) {
            const int nsl = 3;                                     // a few slots
            for (int unit = 0; unit < nsl * EB * 2 * KS1; ++unit)  // fc1 units (tr3_fc1_body, tr3_fc1_any, k1_tr3_fc1)
            {
                const int s_ks1 = ffs(KS1) - 1, s_wg = ffs(WG) - 1;
                const int h = unit & (KS1 - 1), u_ks1 = unit >> s_ks1, mat = u_ks1 & 1, u_b = u_ks1 >> 1;
                const int b = ((EB & (EB - 1)) == 0) ? (u_b & (EB - 1)) : (u_b % EB);
                const int slot = ((EB & (EB - 1)) == 0) ? (u_b >> (ffs(EB) - 1)) : (u_b / EB);
                const int ups = EB * 2 * KS1;
                const int kslot = ((EB & (EB - 1)) == 0) ? (unit >> (ffs(ups) - 1)) : (unit / ups);      // k1_tr3_fc1
                const int fslot = ((EB & (EB - 1)) == 0) ? ((unit >> ffs(KS1)) >> (ffs(EB) - 1)) : (unit / ups);  // fused
                const int fb = ((EB & (EB - 1)) == 0) ? ((unit >> ffs(KS1)) & (EB - 1)) : ((unit >> ffs(KS1)) % EB);
                if (h != unit % KS1 || mat != ((unit / KS1) & 1) || b != (unit / (KS1 * 2)) % EB || slot != unit / ups || kslot != slot ||
                    fslot != slot || fb != b || (KSH >> s_ks1) != KSH / KS1 || ((KSH >> s_ks1) >> s_wg) != KSH / KS1 / WG)
                    return false;
            }
            for (int unit = 0; unit < nsl * EB * NS2; ++unit)      // fc2 units (tr3_fc2_body, k1_tr3_fc2, fused)
            {
                const int s_ns2 = ffs(NS2) - 1;
                const int ns = unit & (NS2 - 1), u_b = unit >> s_ns2;
                const int b = ((EB & (EB - 1)) == 0) ? (u_b & (EB - 1)) : (u_b % EB);
                const int slot = ((EB & (EB - 1)) == 0) ? (u_b >> (ffs(EB) - 1)) : (u_b / EB);
                const int ups = EB * NS2;
                const int kslot = ((EB & (EB - 1)) == 0) ? (unit >> (ffs(ups) - 1)) : (unit / ups);
                if (ns != unit % NS2 || b != (unit / NS2) % EB || slot != unit / ups || kslot != slot || (KSH >> s_ns2) != KSH / NS2) return false;
            }
            return true;
        };
        int n = 0;
        for (int KS1 = 1; KS1 <= 16; ++KS1)
            for (int WG = 1; WG <= 4; ++WG)
                for (int NS2 = 1; NS2 <= 48; ++NS2)
                    for (int fu = 0; fu < 2; ++fu)
                    {
                        const bind::K1Split s = {KS1, WG, NS2, 8, 24};
                        if (bind::k1_split_check(s, fu)) continue;
                        ++n;
                        char what[120];
                        std::snprintf(what, sizeof what, "unit decode KS1 %d WG %d NS2 %d (EB %d): shifts == division", KS1, WG, NS2, EB);
                        check(decode_ok(KS1, WG, NS2), what);
                    }
        check(n > 0, "k1_split_check accepts some splits");
        check(!decode_ok(4, 1, 3), "NS2 = 3: the shift decode differs from division (why k1_split_check refuses it)");
        check(bind::k1_split_check(bind::K1Split{4, 1, 3, 8, 24}, 0) != nullptr, "k1_split_check refuses NS2 = 3");
        check(bind::k1_split_check(bind::K1Split{3, 1, 2, 8, 24}, 0) != nullptr, "k1_split_check refuses KS1 = 3");
        for (int M = 1; M <= G6_MX; ++M)
            for (int fu = 0; fu < 2; ++fu)
            {
                bind::K1Split f = bind::k1_split(M);
                check((!fu || !bind::k1_fuse_split(f)) && !bind::k1_split_check(f, fu), "the release split passes k1_split_check");
            }
    }

    if (fails) { std::printf("k1_split_test: %d FAIL\n", fails); return 1; }
    std::printf("k1_split_test: PASS (tables, triple, per-M variables, fused remap, table reader, bound split with the table, unit decode)\n");
    return 0;
}
