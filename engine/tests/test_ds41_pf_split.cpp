// Host test of the bounded-replay split (engine/runtime/ds41_pf_split.h), off and with GR8_DS_PF_L20_TAIL=1 (L1). No CUDA. Build and run:
//   g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_DS41=1 -I engine engine/tests/test_ds41_pf_split.cpp -o /tmp/test_ds41_pf_split
// 1. Off: range B starts at layer 21, range A ends with hc_attn(21) then ST_PF_COMPACT, and the plan is the non-final plan with the
//    compaction inserted (the plan before L1, rebuilt here by the old rule).
// 2. On: range B starts at layer 20; range A ends hc_attn(20), ca_row(20) (unit U_CONS_F(19), i = 20), ST_PF_COMPACT; range B opens with
//    attn_proj(20), index_topk(20), cand_blocks(20) and holds no ca row; the units stay contiguous (ds41::index).
// 3. Both, over Engram x DSpark and layer subsets: check_split passes; a plan made in one mode fails the other mode's check.
// 4. Mutants (each must be refused): the ca row left in range B, dropped, after the compaction, in U_ATTN(20) in range A, doubled;
//    cand_blocks(20) or attn_proj(20) moved into range A.
#include "runtime/ds41_pf_split.h"
#include "runtime/ds41_dspark_prefill.h"
#include <cstdio>
#include <string>
#include <vector>

static int bad = 0, mutants = 0, refused = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL line %d: %s\n", __LINE__, #x); ++bad; } } while (0)
using namespace ds41;
namespace br = ds41::br;

static std::vector<int> range(int a, int b) { std::vector<int> v; for (int l = a; l < b; ++l) v.push_back(l); return v; }
static std::vector<Op> nonfinal(const std::vector<int>& ids, bool engram, bool dspark)
{
    return dsp::pf::nonfinal_plan(plan(ids, 0, (int)ids.size(), true, true, engram, dspark), dspark);
}
static bool same(const Op& a, const Op& o)
{
    return a.stage == o.stage && a.unit == o.unit && a.ui == o.ui && a.i == o.i && a.l == o.l && a.src == o.src && a.rc == o.rc;
}
static bool same(const std::vector<Op>& a, const std::vector<Op>& b)
{
    if (a.size() != b.size()) return false;
    for (size_t k = 0; k < a.size(); ++k) if (!same(a[k], b[k])) return false;
    return true;
}
// the split before L1 (ds41_pf_split.h at 7b409ff3): the compaction before the first U_ATTN of a layer >= i_late, nothing moved
static br::Split old_split(const std::vector<Op>& nf, const std::vector<int>& ids)
{
    br::Split s;
    s.i_late = -1;
    for (int i = 0; i < (int)ids.size(); ++i) if (ids[i] >= geom::LATE_LAYER_START) { s.i_late = i > 0 ? i : -1; break; }
    if (s.i_late < 0) { s.ops = nf; s.b = nf.size(); return s; }
    for (const Op& o : nf)
    {
        if (s.b == 0 && o.unit == U_ATTN && o.ui >= s.i_late)
        {
            Op c; c.stage = ST_PF_COMPACT; c.unit = U_CONS_F; c.ui = s.i_late - 1; c.i = s.i_late - 1; c.l = ids[s.i_late - 1];
            s.ops.push_back(c);
            s.b = s.ops.size();
        }
        s.ops.push_back(o);
    }
    return s;
}
static void mutant(const char* name, const std::vector<Op>& nf, br::Split s, const std::vector<int>& ids)
{
    ++mutants;
    const std::string e = br::check_split(nf, s, ids);
    if (e.empty()) { std::printf("FAIL mutant %s: accepted\n", name); ++bad; }
    else { ++refused; std::printf("mutant %-40s refused: %s\n", name, e.c_str()); }
}
static size_t find(const std::vector<Op>& p, Stage st, int i, size_t from = 0)
{
    for (size_t k = from; k < p.size(); ++k) if (p[k].stage == st && p[k].i == i) return k;
    return p.size();
}

static void off_mode()
{
    br::l20_tail() = false;
    CHECK(br::late_layer_start() == 21);
    int cases = 0;
    for (const auto& ids : {range(0, 40), std::vector<int>{0, 2, 20, 39}, std::vector<int>{0, 1, 2, 14, 20, 21, 24, 39}, range(0, 21), range(0, 22)})
        for (bool engram : {false, true})
            for (bool dspark : {false, true})
            {
                if (dspark && ids.back() != 39) continue;
                const auto nf = nonfinal(ids, engram, dspark);
                const br::Split s = br::split_plan(nf, ids), o = old_split(nf, ids);
                CHECK(br::check_split(nf, s, ids).empty());
                CHECK(same(s.ops, o.ops) && s.b == o.b && s.i_late == o.i_late);   // off = the plan before L1, op for op
                for (const Op& x : s.ops) CHECK(!(x.stage == ST_CA_ROW && x.unit == U_CONS_F));
                ++cases;
            }
    const auto ids = range(0, 40);
    const br::Split s = br::split_plan(nonfinal(ids, true, true), ids);
    CHECK(s.i_late == 21 && s.ops[s.b - 1].stage == ST_PF_COMPACT && s.ops[s.b - 2].stage == ST_HC_ATTN && s.ops[s.b - 2].i == 21);
    CHECK(find(s.ops, ST_CA_ROW, 20) < s.b && find(s.ops, ST_CAND_BLOCKS, 20) < s.b && find(s.ops, ST_INDEX_TOPK, 20) < s.b);
    std::printf("off: %d plans = the split before L1, range B from layer 21\n", cases);
}

static void on_mode()
{
    br::l20_tail() = true;
    CHECK(br::late_layer_start() == 20);
    int cases = 0;
    for (const auto& ids : {range(0, 40), std::vector<int>{0, 2, 20, 39}, std::vector<int>{0, 1, 2, 14, 19, 20, 21, 24, 39}, range(0, 21), range(0, 20),
                            std::vector<int>{0, 21, 39}})
        for (bool engram : {false, true})
            for (bool dspark : {false, true})
            {
                if (dspark && ids.back() != 39) continue;
                const auto nf = nonfinal(ids, engram, dspark);
                const br::Split s = br::split_plan(nf, ids);
                const std::string e = br::check_split(nf, s, ids);
                if (!e.empty()) std::printf("ids %zu engram %d dspark %d: %s\n", ids.size(), engram, dspark, e.c_str());
                CHECK(e.empty());
                ++cases;
                if (!s.on()) { CHECK(br::late_start(ids) < 0); continue; }
                const int il = s.i_late;
                CHECK(ids[il] >= 20 && ids[il - 1] < 20);
                // the units stay contiguous: every op of a unit's span is that unit (the hoisted ca row is an op of U_CONS_F(il - 1))
                const Index x = index(s.ops);
                for (int u = 0; u < U_COUNT; ++u)
                    for (int ui = 0; ui < (int)x.u[u].size(); ++ui)
                        for (size_t k = x.u[u][ui].b; k < x.u[u][ui].e; ++k) CHECK(s.ops[k].unit == u && s.ops[k].ui == ui);
                if (ids[il] != 20) { CHECK(find(s.ops, ST_CA_ROW, il) == s.ops.size()); continue; }   // no layer 20: nothing to hoist
                // range A ends: hc_attn(20) [in U_CONS_F(il-1)], ca_row(20) [U_CONS_F(il-1), i = il], compact
                CHECK(s.ops[s.b - 1].stage == ST_PF_COMPACT);
                const Op& h = s.ops[s.b - 2];
                CHECK(h.stage == ST_CA_ROW && h.unit == U_CONS_F && h.ui == il - 1 && h.i == il && h.l == 20);
                CHECK(s.ops[s.b - 3].stage == ST_HC_ATTN && s.ops[s.b - 3].i == il);
                // range B opens with layer 20's attention: attn_proj, index_topk, cand_blocks, attn_core, attn_out; no ca row anywhere in B
                CHECK(s.ops[s.b].stage == ST_ATTN_PROJ && s.ops[s.b].i == il);
                CHECK(s.ops[s.b + 1].stage == ST_INDEX_TOPK && s.ops[s.b + 2].stage == ST_CAND_BLOCKS && s.ops[s.b + 3].stage == ST_ATTN_CORE);
                for (size_t k = s.b; k < s.ops.size(); ++k) CHECK(s.ops[k].stage != ST_CA_ROW);
                CHECK(s.ops.size() == nf.size() + 1);
                // every kv source's ca row (2, 8, 14, 20 in the subset) stays in range A: the rows every position needs
                for (int i = 0; i < (int)ids.size(); ++i)
                    if (geom::is_kv_source(ids[i])) CHECK(find(s.ops, ST_CA_ROW, i) < s.b);
                // the engram layers (1, 14) stay in range A
                for (size_t k = s.b; k < s.ops.size(); ++k) CHECK(s.ops[k].stage != ST_ENGRAM && s.ops[k].stage != ST_EMBED);
            }
    std::printf("on: %d plans checked, range B from layer 20, ca row(20) hoisted into range A\n", cases);
}

static void mutants_on()
{
    br::l20_tail() = true;
    const auto ids = range(0, 40);
    const auto nf = nonfinal(ids, true, true);
    const br::Split s = br::split_plan(nf, ids);
    CHECK(br::check_split(nf, s, ids).empty());
    const size_t ca = s.b - 2, cmp = s.b - 1;
    {   // the ca row left in range B (where it was in the non-final plan, after attn_proj(20)): only the tail rows' ca rows
        br::Split m = s; Op o = m.ops[ca]; o.unit = U_ATTN; o.ui = 20;
        m.ops.erase(m.ops.begin() + ca); m.b -= 1; m.ops.insert(m.ops.begin() + m.b + 1, o);
        mutant("ca row in range B", nf, m, ids);
    }
    { br::Split m = s; m.ops.erase(m.ops.begin() + ca); m.b -= 1; mutant("ca row dropped", nf, m, ids); }
    { br::Split m = s; std::swap(m.ops[ca], m.ops[cmp]); mutant("ca row after the compaction", nf, m, ids); }
    { br::Split m = s; m.ops[ca].unit = U_ATTN; m.ops[ca].ui = 20; mutant("ca row in range A as a U_ATTN(20) op", nf, m, ids); }
    { br::Split m = s; m.ops.insert(m.ops.begin() + ca, m.ops[ca]); m.b += 1; mutant("ca row twice", nf, m, ids); }
    {   // cand_blocks(20) into range A (before the hoisted ca row): it would read index-K rows not yet written
        br::Split m = s; const size_t cb = find(m.ops, ST_CAND_BLOCKS, 20, m.b); Op o = m.ops[cb]; o.unit = U_CONS_F; o.ui = 19;
        m.ops.erase(m.ops.begin() + cb); m.ops.insert(m.ops.begin() + ca, o); m.b += 1;
        mutant("cand_blocks(20) in range A", nf, m, ids);
    }
    {
        br::Split m = s; Op o = m.ops[m.b]; o.unit = U_CONS_F; o.ui = 19;
        m.ops.erase(m.ops.begin() + m.b); m.ops.insert(m.ops.begin() + ca, o); m.b += 1;
        mutant("attn_proj(20) in range A", nf, m, ids);
    }
    {   // the old split (boundary at 21) checked with the switch on
        mutant("off-mode split under GR8_DS_PF_L20_TAIL=1", nf, old_split(nf, ids), ids);
    }
    br::l20_tail() = false;
    mutant("on-mode split under GR8_DS_PF_L20_TAIL=0", nf, s, ids);
}

static void explicit_modes() {
    const auto ids=range(0,40);
    for(bool global:{false,true})for(bool engram:{false,true})for(bool dspark:{false,true}) {
        br::l20_tail()=global;
        const auto nf=nonfinal(ids,engram,dspark);
        for(bool l20:{false,true}) {
            const auto s=br::split_plan(nf,ids,l20);
            CHECK(s.i_late==(l20?20:21));
            CHECK(br::check_split(nf,s,ids,l20).empty());
            CHECK(!br::check_split(nf,s,ids,!l20).empty());
            CHECK(br::l20_tail()==global);
            const auto wrapper=br::split_plan(nf,ids);
            const auto explicit_global=br::split_plan(nf,ids,global);
            CHECK(same(wrapper.ops,explicit_global.ops));
        }
    }
    br::l20_tail()=false;
}
int main()
{
    explicit_modes();
    off_mode();
    on_mode();
    mutants_on();
    // the switch only moves the boundary: tail rows, window floor and the late-final rule do not depend on it
    br::l20_tail() = true;
    CHECK(br::late_rows(4096, 128) == 128 && br::win_floor(8192, 4096, 128) == 8192 - 128 && br::late_skip(256, 128) && !br::late_skip(255, 128));
    br::l20_tail() = false;
    std::printf("mutants: %d of %d refused\n", refused, mutants);
    std::printf("test_ds41_pf_split: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
