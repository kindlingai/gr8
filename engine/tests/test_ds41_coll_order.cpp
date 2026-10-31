// Host test of the DS41 decode step's collective order (engine/runtime/ds41_coll_order.h): no CUDA. Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_DS41=1 -I engine -I engine/runtime -I engine/comms engine/tests/test_ds41_coll_order.cpp -o /tmp/test_ds41_coll_order && /tmp/test_ds41_coll_order
// The reference order is derived from the REAL plan() (runtime/ds41_ops.h, the list ds41_step.cuh walks): ST_PUBLISH_ATTN is c = 2i,
// ST_PUBLISH_FFN c = 2i+1, ST_ENGRAM c = 2*nl+1+e (ds41_engram_ops.cuh), ST_HEAD_LOCAL c = 2*nl (rank.cuh head_args).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>
#include "runtime/ds41_coll_order.h"
#include "runtime/ds41_ops.h"
#include "runtime/ds41_dspark_prefill.h"
#include "runtime/ds41_pf_split.h"
#include "runtime/s2/ds41_prefill.h"
static std::string str(const std::vector<int>& v) { std::string s; for (int x : v) s += (s.empty() ? "" : " ") + std::to_string(x); return s; }
#define EXPECT_EQ(a, b) do { const std::string x = (a), y = (b); if (x != y) { std::printf("FAIL %s:%d\n  got  %s\n  want %s\n", __FILE__, __LINE__, x.c_str(), y.c_str()); std::exit(1); } } while (0)

// the order the step walk rings: first/last/prologue/epilogue as plan() takes them
static std::vector<int> walk(const std::vector<int>& ids, int first, int last, bool pro, bool epi)
{
    const int nl = (int)ids.size();
    std::vector<int> o;
    for (const ds41::Op& p : ds41::plan(ids, first, last, pro, epi, true))
    {
        if (p.stage == ds41::ST_PUBLISH_ATTN) o.push_back(2 * p.i);
        else if (p.stage == ds41::ST_PUBLISH_FFN) o.push_back(2 * p.i + 1);
        else if (p.stage == ds41::ST_ENGRAM) o.push_back(2 * nl + 1 + (ids[p.i] == geom::ENGRAM_L0 ? 0 : 1));
        else if (p.stage == ds41::ST_HEAD_LOCAL) o.push_back(2 * nl);
    }
    return o;
}
static void check(const char* name, const std::vector<int>& ids)
{
    const int nl = (int)ids.size();
    const std::vector<int> got = ds41::coll_exec_order(ids, 2), want = walk(ids, 0, nl, true, true);
    if (str(got) != str(want)) { std::printf("FAIL %s\n  got  %s\n  want %s\n", name, str(got).c_str(), str(want).c_str()); std::exit(1); }
    // a permutation of 0 .. 2*nl+2, each collective once
    std::vector<int> seen(2 * nl + 3, 0);
    for (int c : got) { if (c < 0 || c >= 2 * nl + 3 || seen[c]++) { std::printf("FAIL %s: c=%d out of range or twice\n", name, c); std::exit(1); } }
    if ((int)got.size() != 2 * nl + 3) { std::printf("FAIL %s: %zu entries, want %d\n", name, got.size(), 2 * nl + 3); std::exit(1); }
    // no head: the same order without c = 2*nl
    std::vector<int> nh = got; nh.erase(std::find(nh.begin(), nh.end(), 2 * nl));
    EXPECT_EQ(str(ds41::coll_exec_order(ids, 2, false)), str(nh));
    std::printf("ok  %-28s nl=%2d  %zu entries\n", name, nl, got.size());
}
int main()
{
    std::vector<int> all(40); std::iota(all.begin(), all.end(), 0);
    // the full model: the order the legate and B9 gave
    std::vector<int> want = {0, 1, 81};
    for (int c = 2; c <= 27; ++c) want.push_back(c);
    want.push_back(82);
    for (int c = 28; c <= 80; ++c) want.push_back(c);
    EXPECT_EQ(str(ds41::coll_exec_order(all, 2)), str(want));
    EXPECT_EQ(std::to_string(ds41::coll_exec_order(all, 2).size()), "83");
    // Engram off: no order at all (the proxy keeps its identity loop)
    EXPECT_EQ(std::to_string(ds41::coll_exec_order(all, 0).size()), "0");
    check("layers 0..39", all);
    std::vector<int> a19(20); std::iota(a19.begin(), a19.end(), 0);
    check("layers 0..19", a19);
    check("layers {1,14}", {1, 14});                 // the Engram stage of layer 1 sits in U_EMBED, before c=0
    check("layers {0,1,14,39}", {0, 1, 14, 39});
    check("layers {1,2,3,14}", {1, 2, 3, 14});
    check("layers {5,1,14}", {5, 1, 14});            // not ascending: e follows the layer id, not the position
    EXPECT_EQ(str(ds41::coll_exec_order({1, 14}, 2)), "5 0 1 6 2 3 4");   // nl=2: head c=4, Engram e=0 c=5 opens position 0 (before c=0), e=1 c=6 opens position 1 (after c=1)
    // Bounded replay (ds41_pf_split.h, s2/ds41_prefill.h): a non-final chunk's layer exchanges c < 2*i_late carry rows x 10,240 B, the ones of
    // range B (c >= 2*i_late) late_rows x 10,240 B; the Engram reduces keep rows x 12,288 B; the order does not move. The collectives a walk of
    // the split plan publishes, with the rows of the range each publish runs in, must be exactly those sizes.
    {
        const size_t RB = 5120 * 2, EB = 6144 * 2;
        for (const int ecoll : {0, 2})
            for (const int tail : {0, 128, 256, 1024})
                for (const uint32_t rows : {1024u, 700u, 100u})
                {
                    const int nl = 40, i_late = ds41::br::late_start(all);
                    EXPECT_EQ(std::to_string(i_late), "21");
                    const uint32_t late = (uint32_t)ds41::br::late_rows((int)rows, tail);
                    const g6s2::Ds41ChunkColl cc = g6s2::ds41_chunk_collectives(all, ecoll, rows, RB, EB, i_late, late);
                    const g6s2::Ds41ChunkColl base = g6s2::ds41_chunk_collectives(all, ecoll, rows, RB, EB);
                    EXPECT_EQ(str(cc.order), str(base.order));
                    EXPECT_EQ(std::to_string(cc.sizes.size()), std::to_string(base.sizes.size()));
                    for (int c = 0; c < 2 * nl; ++c)
                        EXPECT_EQ(std::to_string(cc.sizes[(size_t)c]), std::to_string((c < 2 * i_late ? rows : late) * RB));
                    if (ecoll) { EXPECT_EQ(std::to_string(cc.sizes[80]), "0"); EXPECT_EQ(std::to_string(cc.sizes[81]), std::to_string(rows * EB)); EXPECT_EQ(std::to_string(cc.sizes[82]), std::to_string(rows * EB)); }
                    if (late == rows) for (size_t c = 0; c < cc.sizes.size(); ++c) EXPECT_EQ(std::to_string(cc.sizes[c]), std::to_string(base.sizes[c]));
                    // the walk: every publish of the split plan, at the rows of its range
                    const auto nf = ds41::dsp::pf::nonfinal_plan(ds41::plan(all, 0, nl, true, true, ecoll != 0), false);
                    const ds41::br::Split sp = ds41::br::split_plan(nf, all);
                    std::vector<int> seen(2 * nl, 0);
                    for (size_t k = 0; k < sp.ops.size(); ++k)
                    {
                        const ds41::Op& o = sp.ops[k];
                        if (o.stage != ds41::ST_PUBLISH_ATTN && o.stage != ds41::ST_PUBLISH_FFN) continue;
                        const int c = 2 * o.i + (o.stage == ds41::ST_PUBLISH_FFN);
                        const uint32_t r = k < sp.b ? rows : late;
                        EXPECT_EQ(std::to_string(cc.sizes[(size_t)c]), std::to_string(r * RB));
                        ++seen[(size_t)c];
                    }
                    for (int c = 0; c < 2 * nl; ++c) EXPECT_EQ(std::to_string(seen[(size_t)c]), "1");
                }
        // subset {0, 2, 20, 21, 24, 39}: i_late = position 3 (layer 21) -> c >= 6 carry the tail
        const std::vector<int> sub6 = {0, 2, 20, 21, 24, 39};
        const g6s2::Ds41ChunkColl s6 = g6s2::ds41_chunk_collectives(sub6, 0, 1024, RB, EB, ds41::br::late_start(sub6), 128);
        std::vector<uint32_t> want6;
        for (int c = 0; c < 12; ++c) want6.push_back(uint32_t((c < 6 ? 1024 : 128) * RB));
        EXPECT_EQ(std::to_string(s6.sizes == want6), "1");
        // refusals: late rows 0 / > rows, i_late 0 / past the layers, an Engram layer past the boundary
        auto throws = [](auto f) { try { f(); } catch (const std::exception&) { return true; } return false; };
        EXPECT_EQ(std::to_string(throws([&] { g6s2::ds41_chunk_collectives(all, 0, 1024, RB, EB, 21, 0); })), "1");
        EXPECT_EQ(std::to_string(throws([&] { g6s2::ds41_chunk_collectives(all, 0, 1024, RB, EB, 21, 1025); })), "1");
        EXPECT_EQ(std::to_string(throws([&] { g6s2::ds41_chunk_collectives(all, 0, 1024, RB, EB, 0, 128); })), "1");
        EXPECT_EQ(std::to_string(throws([&] { g6s2::ds41_chunk_collectives(all, 0, 1024, RB, EB, 40, 128); })), "1");
        EXPECT_EQ(std::to_string(throws([&] { g6s2::ds41_chunk_collectives({0, 1, 14}, 2, 1024, RB, EB, 1, 128); })), "1");
        std::printf("ok  bounded replay sizes: c < 42 at rows x 10240, c >= 42 at late_rows x 10240 (Engram rows x 12288), every publish matches\n");
    }
    std::printf("test_ds41_coll_order: all checks passed\n");
}
