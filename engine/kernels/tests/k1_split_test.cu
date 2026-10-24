// k1_split_test: the K1 split as bind.cuh's k1_split / k1_fuse_split choose it, host only (no GPU, no launches).
// (1) A per-row-count table in G6_K1_SPLIT ("1:/4,3:2x1,4-32:/4", #107) is not the "KS1,WG,NS2" triple: k1_split keeps
//     the release split at every M. Before, sscanf read its leading 1 as KS1 at every M; with the release WG = 1 at
//     M = 1 the fused remap then gave KS1 = 0 (no fc1 items, wrong output).
// (2) The triple still sets every M, and G6_K1_SPLIT_M1/M2/M4 still win over it.
// (3) k1_fuse_split refuses KS1 x WG < 2 and maps the release splits to (KS1*WG/2, 2).
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

    if (fails) { std::printf("k1_split_test: %d FAIL\n", fails); return 1; }
    std::printf("k1_split_test: PASS (tables, triple, per-M variables, fused remap)\n");
    return 0;
}
