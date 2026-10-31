// Host test of the DS41 decode-step order (engine/runtime/ds41_ops.h): no CUDA. Build:
//   g++ -std=c++17 -O2 -Wall -DG6_MODEL_DS41=1 -I engine engine/tests/test_ds41_step_order.cpp
// Checks (a) the stage order of one layer and of the whole step, (b) that the optional stages follow geom::kv_source /
// index_source / compress_ratio for the subset {0, 2, 20, 39} and for all 40 layers, (c) that the walk over a table
// with every slot null stops at the first stage that needs an op (and names it), and with the slots filled one by one it
// moves on to the next one in plan order, (d) the residual-copy points, (e) the part split of the graph form, (f) the units
// (the groups the rank.cuh wrappers launch): contiguous, in the wrapper order, their sizes and their index.
//
// What this test is not: an oracle gate. In the subset {0, 2, 20, 39} the list of layer 39 comes from index source 36, which
// does not run there; the plan marks that stage ST_INDEX_BORROW (a fixture injected from the oracle's dump, a diag run). A
// true 4-layer oracle gate needs that injection; the all-40-layer plan has no borrowed stage.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>
#include "runtime/ds41_ops.h"
#include "runtime/ds41_dspark_prefill.h"
#include "runtime/ds41_pf_split.h"

using namespace ds41;

static std::vector<int> iota(int n) { std::vector<int> v(n); for (int i = 0; i < n; ++i) v[i] = i; return v; }
static std::vector<Stage> stages_of(const std::vector<Op>& p, int i)
{
    std::vector<Stage> s;
    for (const Op& o : p) if (o.i == i) s.push_back(o.stage);
    return s;
}
static int count(const std::vector<Op>& p, Stage s, int i = -2)
{
    int n = 0;
    for (const Op& o : p) n += o.stage == s && (i == -2 || o.i == i);
    return n;
}
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

// a recording table: every slot logs (stage, i) and answers 0; slots in `null_from` and later in stage number stay null
static std::vector<std::pair<int, int>> g_log;
template <int S> static int rec(void*, int i, int, void*) { g_log.push_back({S, i}); return 0; }
template <int... S> static void fill_all(Table& t, std::integer_sequence<int, S...>) { ((t.fn[S] = &rec<S>), ...); }
// the same, logging (stage, rows): the bounded-replay walk runs two row counts
static std::vector<std::pair<int, int>> g_rows_log;
template <int S> static int rec_rows(void*, int, int rows, void*) { g_rows_log.push_back({S, rows}); return 0; }
template <int... S> static void fill_rows(Table& t, std::integer_sequence<int, S...>) { ((t.fn[S] = &rec_rows<S>), ...); }

int main()
{
    // (a) one layer with every optional stage: layer 20 is a kv source, an index source and the candidate source
    {
        const std::vector<int> ids = {20};
        const auto p = plan(ids, 0, 1, true, false);
        const std::vector<Stage> want = {ST_EMBED, ST_HC_ATTN, ST_ATTN_PROJ, ST_CA_ROW, ST_INDEX_TOPK, ST_CAND_BLOCKS, ST_ATTN_CORE,
            ST_ATTN_OUT, ST_PUBLISH_ATTN, ST_CONSUME_ATTN, ST_HC_FFN, ST_ROUTER, ST_SHARED, ST_EXPERTS, ST_PUBLISH_FFN, ST_FINISH};
        std::vector<Stage> got;
        for (const Op& o : p) got.push_back(o.stage);
        EXPECT(got.front() == ST_STEP_IN);
        got.erase(got.begin());
        EXPECT(got == want);
        EXPECT(p[5].stage == ST_INDEX_TOPK && p[5].src == 20);
    }
    // layer 0 and 1: window only (no ca row, no index); a plain middle layer of ratio 2 reuses its source's list
    {
        const std::vector<int> ids = iota(40);
        const auto p = plan(ids, 0, 40, true, true);
        for (int l : {0, 1})
            EXPECT(count(p, ST_CA_ROW, l) == 0 && count(p, ST_INDEX_TOPK, l) == 0 && count(p, ST_ATTN_CORE, l) == 1);
        EXPECT(count(p, ST_CA_ROW, 5) == 0 && count(p, ST_INDEX_TOPK, 5) == 0);   // reuse of layer 2's list
        // (b) the optional stages follow the geometry over all 40 layers
        for (int l = 0; l < 40; ++l)
        {
            EXPECT(count(p, ST_CA_ROW, l) == (int)geom::is_kv_source(l));
            EXPECT(count(p, ST_INDEX_TOPK, l) == (int)(geom::compress_ratio(l) > 0 && geom::is_index_source(l)));
            EXPECT(count(p, ST_CAND_BLOCKS, l) == (int)(l == geom::CAND_SOURCE));
            EXPECT(count(p, ST_ATTN_CORE, l) == 1 && count(p, ST_EXPERTS, l) == 1);
            EXPECT(count(p, ST_CONSUME_FFN, l) == (int)(l != 39) && count(p, ST_FINISH, l) == (int)(l == 39));
        }
        EXPECT(count(p, ST_CA_ROW) == geom::N_KV_SRC && count(p, ST_INDEX_TOPK) == geom::N_IDX_SRC && count(p, ST_CAND_BLOCKS) == 1);
        EXPECT(count(p, ST_EMBED) == 1 && count(p, ST_FINISH) == 1 && count(p, ST_STEP_OUT) == 1);
        EXPECT(count(p, ST_INDEX_BORROW) == 0);   // the whole model has every index source
        EXPECT(p.front().stage == ST_STEP_IN && p[1].stage == ST_EMBED);
        EXPECT(p.back().stage == ST_STEP_OUT && p[p.size() - 2].stage == ST_HEAD_MERGE && p[p.size() - 3].stage == ST_HEAD_LOCAL);
        EXPECT(p[p.size() - 4].stage == ST_FINISH);
        // layers run in order, and within a layer the sub-layer stages never go backwards
        int last = -1;
        for (const Op& o : p) if (o.i >= 0) { EXPECT(o.i >= last); last = o.i; }
        EXPECT(count(p, ST_ENGRAM) == 0);   // off by default
        const auto pe = plan(ids, 0, 40, true, true, true);
        EXPECT(count(pe, ST_ENGRAM) == 2 && count(pe, ST_ENGRAM, 1) == 1 && count(pe, ST_ENGRAM, 14) == 1);
        EXPECT(stages_of(pe, 1).front() == ST_ENGRAM && stages_of(pe, 1)[1] == ST_HC_ATTN);
    }
    // the first-light subset {0, 2, 20, 39}: layer 39's list comes from index source 36, which is not in the subset
    const std::vector<int> sub = {0, 2, 20, 39};
    const auto ps = plan(sub, 0, 4, true, true);
    {
        EXPECT(count(ps, ST_CA_ROW) == 2 && count(ps, ST_CA_ROW, 1) == 1 && count(ps, ST_CA_ROW, 2) == 1);   // layers 2 and 20
        EXPECT(count(ps, ST_INDEX_TOPK, 0) == 0);
        EXPECT(count(ps, ST_INDEX_TOPK, 1) == 1 && count(ps, ST_INDEX_TOPK, 2) == 1 && count(ps, ST_INDEX_TOPK, 3) == 0);
        EXPECT(count(ps, ST_INDEX_BORROW) == 1 && count(ps, ST_INDEX_BORROW, 3) == 1);   // layer 39 only
        for (const Op& o : ps)
            if (o.stage == ST_INDEX_TOPK || o.stage == ST_INDEX_BORROW)
                EXPECT(o.src == (o.l == 39 ? 36 : o.l));
        EXPECT(synthetic(ps) && !synthetic(plan(iota(40), 0, 40, true, true)));
        EXPECT(synthetic_note(ps).find("layer 39 (index source 36)") != std::string::npos && synthetic_note(plan(iota(40), 0, 40, true, true)).empty());
        EXPECT(std::strstr(stop_message([&]{ Stop s; s.stopped = true; for (const Op& o : ps) if (o.stage == ST_INDEX_BORROW) s.op = o; s.null_slot = true; return s; }()).c_str(), "oracle fixture needed") != nullptr);
        EXPECT(count(ps, ST_CAND_BLOCKS, 2) == 1);
        EXPECT(count(ps, ST_FINISH, 3) == 1 && count(ps, ST_CONSUME_FFN) == 3);
        // (d) residual-copy points: 0 after embed, 2i+1 after consume_attn, 2i+2 after the ffn consume or finish
        std::vector<int> rc;
        for (const Op& o : ps) if (o.rc >= 0) rc.push_back(o.rc);
        EXPECT((rc == std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
    }
    // (c) all slots null: the first stage that needs an op is the embed (step_in is a driver op, but this table is empty too)
    {
        Table t;
        int calls = 0;
        const Stop s = run(ps, t, nullptr, 1, nullptr, [&](const Op&) { ++calls; });
        EXPECT(s.stopped && s.null_slot && s.at == 0 && s.op.stage == ST_STEP_IN && s.code == NOTSUP && calls == 0);
        // driver slots filled only: stops at the embed, in order
        t.fn[ST_STEP_IN] = &rec<ST_STEP_IN>;
        g_log.clear();
        const Stop s2 = run(ps, t, nullptr, 1, nullptr, [&](const Op&) { ++calls; });
        EXPECT(s2.stopped && s2.op.stage == ST_EMBED && s2.done == 1 && calls == 1 && g_log.size() == 1);
        EXPECT(std::strstr(stop_message(s2).c_str(), "stage embed (owner mHC)") != nullptr);
    }
    // fill the slots one stage at a time in plan order: each run stops exactly at the next unfilled stage, layer by layer
    {
        Table t;
        std::vector<Op> order;   // plan order of the distinct (stage) keys, first occurrence
        std::map<int, bool> seen;
        for (const Op& o : ps) if (!seen[o.stage]) { seen[o.stage] = true; order.push_back(o); }
        for (size_t k = 0; k < order.size(); ++k)
        {
            g_log.clear();
            const Stop s = run(ps, t, nullptr, 1, nullptr, [](const Op&) {});
            EXPECT(s.stopped && s.null_slot);
            // with k stage slots filled the stop is the first plan op whose stage is not filled
            size_t at = 0;
            while (at < ps.size() && t.fn[ps[at].stage]) ++at;
            EXPECT(s.at == at && s.op.stage == ps[at].stage && s.op.i == ps[at].i);
            // every op before the stop ran, in plan order
            EXPECT(g_log.size() == at);
            for (size_t j = 0; j < at; ++j) EXPECT(g_log[j].first == ps[j].stage && g_log[j].second == ps[j].i);
            switch (order[k].stage)   // fill the next stage in plan order (an explicit table of the instantiations)
            {
#define F(S) case S: t.fn[S] = &rec<S>; break;
            F(ST_STEP_IN) F(ST_EMBED) F(ST_ENGRAM) F(ST_HC_ATTN) F(ST_ATTN_PROJ) F(ST_CA_ROW) F(ST_INDEX_TOPK)
            F(ST_INDEX_BORROW) F(ST_CAND_BLOCKS) F(ST_ATTN_CORE) F(ST_ATTN_OUT) F(ST_PUBLISH_ATTN) F(ST_CONSUME_ATTN) F(ST_HC_FFN) F(ST_ROUTER)
            F(ST_SHARED) F(ST_EXPERTS) F(ST_PUBLISH_FFN) F(ST_CONSUME_FFN) F(ST_FINISH) F(ST_HEAD_LOCAL) F(ST_HEAD_MERGE)
            F(ST_STEP_OUT)
#undef F
            default: EXPECT(false);
            }
        }
        // all filled: the walk completes
        g_log.clear();
        const Stop done = run(ps, t, nullptr, 1, nullptr, [](const Op&) {});
        EXPECT(!done.stopped && done.done == ps.size() && g_log.size() == ps.size());
    }
    // the stop names a layer: the first layer-0 stage whose slot is empty is hc_attn at layer 0, then attn_proj, ...
    {
        Table t;
        t.fn[ST_STEP_IN] = &rec<ST_STEP_IN>; t.fn[ST_EMBED] = &rec<ST_EMBED>; t.fn[ST_HC_ATTN] = &rec<ST_HC_ATTN>;
        const Stop s = run(ps, t, nullptr, 1, nullptr, [](const Op&) {});
        EXPECT(s.op.stage == ST_ATTN_PROJ && s.op.l == 0 && s.op.i == 0);
        EXPECT(stop_message(s).find("stage attn_proj (owner attention), layer 0 (position 0)") != std::string::npos);
        // the same table over the 4-layer subset reaches layer 2 only if the layer-0 stages are all filled
        t.fn[ST_ATTN_PROJ] = &rec<ST_ATTN_PROJ>; t.fn[ST_ATTN_CORE] = &rec<ST_ATTN_CORE>; t.fn[ST_ATTN_OUT] = &rec<ST_ATTN_OUT>;
        t.fn[ST_PUBLISH_ATTN] = &rec<ST_PUBLISH_ATTN>; t.fn[ST_CONSUME_ATTN] = &rec<ST_CONSUME_ATTN>;
        t.fn[ST_HC_FFN] = &rec<ST_HC_FFN>; t.fn[ST_ROUTER] = &rec<ST_ROUTER>; t.fn[ST_SHARED] = &rec<ST_SHARED>;
        t.fn[ST_EXPERTS] = &rec<ST_EXPERTS>; t.fn[ST_PUBLISH_FFN] = &rec<ST_PUBLISH_FFN>; t.fn[ST_CONSUME_FFN] = &rec<ST_CONSUME_FFN>;
        t.fn[ST_HC_ATTN] = &rec<ST_HC_ATTN>;
        const Stop s2 = run(ps, t, nullptr, 1, nullptr, [](const Op&) {});
        EXPECT(s2.op.stage == ST_CA_ROW && s2.op.l == 2 && s2.op.i == 1);   // layer 0 was done; layer 2 needs its ca row
        // an op that answers NOTSUP is not a null slot but stops the same way; another code is a failure
        t.fn[ST_CA_ROW] = [](void*, int, int, void*) { return NOTSUP; };
        const Stop s3 = run(ps, t, nullptr, 1, nullptr, [](const Op&) {});
        EXPECT(s3.stopped && !s3.null_slot && s3.code == NOTSUP && s3.op.stage == ST_CA_ROW);
        t.fn[ST_CA_ROW] = [](void*, int, int, void*) { return 700; };
        const Stop s4 = run(ps, t, nullptr, 1, nullptr, [](const Op&) {});
        EXPECT(s4.code == 700 && stop_message(s4).find("code 700") != std::string::npos);
    }
    // (e) the part split of the graph form: part 0 = prologue + layers [0, ls), part 1 = layers [ls, nl) + epilogue
    {
        const auto a = plan(sub, 0, 2, true, false), b = plan(sub, 2, 4, false, true), whole = plan(sub, 0, 4, true, true);
        EXPECT(a.size() + b.size() == whole.size());
        std::vector<Op> cat = a;
        cat.insert(cat.end(), b.begin(), b.end());
        for (size_t k = 0; k < whole.size(); ++k)
            EXPECT(cat[k].stage == whole[k].stage && cat[k].i == whole[k].i && cat[k].rc == whole[k].rc);
    }
    // (f) the units: each (unit, ui) is one contiguous run, the runs come in the order the wrappers launch them, and rc sits on
    // the last op of a unit only
    {
        const Index ix = index(ps);
        auto sz = [&](Unit u, int ui) { const Span s = ix.get(u, ui); return s.e - s.b; };
        EXPECT(sz(U_STEP_IN, 0) == 1 && sz(U_EMBED, 0) == 2 && sz(U_HEAD_LOCAL, 0) == 1 && sz(U_HEAD_MERGE, 0) == 1 && sz(U_STEP_OUT, 0) == 1);
        // {0, 2, 20, 39}: layer 0 = window only (proj, core, out, publish); layer 2 adds the ca row and index_topk
        EXPECT(sz(U_ATTN, 0) == 4 && sz(U_ATTN, 1) == 6 && sz(U_ATTN, 2) == 7 && sz(U_ATTN, 3) == 5);   // 20: + cand_blocks; 39: borrow instead of index_topk, no ca row
        for (int i = 0; i < 4; ++i)
            EXPECT(sz(U_CONS_A, i) == 2 && sz(U_MLP, i) == 4 && sz(U_CONS_F, i) == (i < 3 ? 2u : 1u));
        EXPECT(ix.get(U_ATTN, 4).empty() && ix.get(U_ATTN, -1).empty());
        // order: step_in, embed, then per layer attn, consume_a, mlp, consume_f, then the head units
        std::vector<std::pair<int, int>> seq;
        for (const Op& o : ps) if (seq.empty() || seq.back() != std::make_pair((int)o.unit, o.ui)) seq.push_back({(int)o.unit, o.ui});
        std::vector<std::pair<int, int>> want = {{U_STEP_IN, 0}, {U_EMBED, 0}};
        for (int i = 0; i < 4; ++i) for (Unit u : {U_ATTN, U_CONS_A, U_MLP, U_CONS_F}) want.push_back({(int)u, i});
        for (Unit u : {U_HEAD_LOCAL, U_HEAD_MERGE, U_STEP_OUT}) want.push_back({(int)u, 0});
        EXPECT(seq == want);   // contiguous: a unit that came back would show up twice
        for (size_t k = 0; k < ps.size(); ++k)
        {
            const bool last = k + 1 == ps.size() || ps[k + 1].unit != ps[k].unit || ps[k + 1].ui != ps[k].ui;
            const bool rc_unit = ps[k].unit == U_EMBED || ps[k].unit == U_CONS_A || ps[k].unit == U_CONS_F;
            EXPECT((ps[k].rc >= 0) == (last && rc_unit));
        }
        // hc_attn of layer i+1 closes the consume unit of layer i (the prepare runs inside the consume group, as in Flash)
        EXPECT(ps[ix.get(U_CONS_F, 0).e - 1].stage == ST_HC_ATTN && ps[ix.get(U_CONS_F, 0).e - 1].i == 1);
        EXPECT(ps[ix.get(U_EMBED, 0).e - 1].stage == ST_HC_ATTN && ps[ix.get(U_EMBED, 0).e - 1].i == 0);
        // a part that starts at layer 2 has no prepare of its own for layer 2 (part 0 ended with it)
        const auto b = plan(sub, 2, 4, false, true);
        EXPECT(count(b, ST_HC_ATTN, 2) == 0 && b.front().stage == ST_ATTN_PROJ && b.front().i == 2);
        const auto a = plan(sub, 0, 2, true, false);
        EXPECT(a.back().stage == ST_HC_ATTN && a.back().i == 2 && a.back().rc == 4);
    }
    // DSpark: ST_DS_TAP opens U_ATTN of layers 37..39, ST_DS_MAIN follows finish and takes the last rc point; off = unchanged
    {
        const auto off = plan(iota(40), 0, 40, true, true), on = plan(iota(40), 0, 40, true, true, false, true);
        EXPECT(count(off, ST_DS_TAP) == 0 && count(off, ST_DS_MAIN) == 0);
        EXPECT(on.size() == off.size() + 4);                                  // 3 taps + 1 main-cache update
        for (const Op& o : on)
            if (o.stage == ST_DS_TAP) EXPECT(o.unit == U_ATTN && o.l >= 37 && o.l <= 39);
        for (size_t k = 0; k < on.size(); ++k)
        {
            if (on[k].stage == ST_DS_TAP) EXPECT(on[k + 1].stage == ST_ATTN_PROJ && on[k + 1].i == on[k].i);   // after hc_attn(i), before attn_proj
            if (on[k].stage == ST_FINISH) EXPECT(on[k].rc == -1 && on[k + 1].stage == ST_DS_MAIN && on[k + 1].rc == 80 && on[k + 2].stage == ST_HEAD_LOCAL);
        }
        const auto d = plan_draft({40, 41, 42});
        EXPECT(d.front().stage == ST_DS_DRAFT_IN && d[1].stage == ST_HC_ATTN && d.back().stage == ST_DS_MARKOV && d[d.size() - 2].stage == ST_DS_HEAD_LOCAL);
        EXPECT(count(d, ST_EMBED) == 0 && count(d, ST_CA_ROW) == 0 && count(d, ST_INDEX_TOPK) == 0 && count(d, ST_FINISH) == 1);
        EXPECT(!synthetic(d));
    }
    // Bounded replay (ds41_pf_split.h): the non-final chunk plan split at the first real layer id >= 21. Range A runs `rows` rows and ends with
    // ST_PF_COMPACT; range B runs late_rows. Without the compaction op the split plan is the non-final plan op for op (plan section 5 step 1).
    {
        EXPECT(geom::LATE_LAYER_START == 21 && geom::TAIL_ROWS == 128 && geom::is_kv_source(20) && !geom::is_kv_source(21));
        for (int l = 21; l < geom::L_MAIN; ++l) EXPECT(!geom::is_kv_source(l) && !geom::is_engram(l) && l != geom::CAND_SOURCE);
        std::vector<std::pair<int, int>>& log = g_rows_log;   // (stage, rows) of every op a recording walk ran
        Table rt;
        fill_rows(rt, std::make_integer_sequence<int, ST_COUNT>{});
        auto run_split = [&](const br::Split& sp, int rows, int tail)
        {
            log.clear();
            const Stop a = run(sp.ops, 0, sp.b, rt, nullptr, rows, nullptr, [](const Op&) {});
            const Stop b = run(sp.ops, sp.b, sp.ops.size(), rt, nullptr, br::late_rows(rows, tail), nullptr, [](const Op&) {});
            EXPECT(!a.stopped && !b.stopped && log.size() == sp.ops.size());
        };
        int n_cases = 0;
        for (const bool engram : {false, true})
            for (const bool dsp : {false, true})
                for (const bool rows_kept : {false, true})
                {
                    if (!dsp && rows_kept) continue;
                    const std::vector<int> ids = iota(40);
                    const auto verify = plan(ids, 0, 40, true, true, engram, dsp);
                    const auto nf = dsp::pf::nonfinal_plan(verify, dsp && rows_kept);
                    EXPECT(dsp::pf::check_nonfinal_plan(verify, nf, dsp && rows_kept).empty());
                    const br::Split sp = br::split_plan(nf, ids);
                    EXPECT(sp.on() && sp.i_late == 21 && br::check_split(nf, sp, ids).empty());
                    EXPECT(sp.ops.size() == nf.size() + 1 && sp.ops[sp.b - 1].stage == ST_PF_COMPACT && sp.ops[sp.b - 1].l == 20);
                    EXPECT(sp.ops[sp.b - 2].stage == ST_HC_ATTN && sp.ops[sp.b - 2].i == 21);       // the fused prepare of layer 21 is in range A
                    EXPECT(sp.ops[sp.b].stage == ST_ATTN_PROJ && sp.ops[sp.b].l == 21);
                    for (size_t k = sp.b; k < sp.ops.size(); ++k) EXPECT(sp.ops[k].l >= 21);
                    for (size_t k = 0; k + 1 < sp.b; ++k) EXPECT(sp.ops[k].unit == U_EMBED || sp.ops[k].ui <= 20);
                    if (dsp && rows_kept)
                    {
                        EXPECT(count(sp.ops, ST_DS_TAP) == 3 && count(sp.ops, ST_DS_MAIN) == 1 && sp.ops.back().stage == ST_DS_MAIN);
                        for (size_t k = 0; k < sp.b; ++k) EXPECT(sp.ops[k].stage != ST_DS_TAP && sp.ops[k].stage != ST_DS_MAIN);
                    }
                    for (const int tail : {0, 128, 256, 1024})
                    {
                        run_split(sp, 1024, tail);
                        // the walk minus ST_PF_COMPACT = the non-final plan, op for op; rows = 1024 before the compaction, late_rows after it
                        size_t k = 0;
                        for (size_t x = 0; x < log.size(); ++x)
                        {
                            EXPECT(log[x].first == sp.ops[x].stage);
                            EXPECT(log[x].second == (x < sp.b ? 1024 : (tail == 0 || tail >= 1024 ? 1024 : tail)));
                            if (log[x].first == ST_PF_COMPACT) continue;
                            EXPECT(log[x].first == nf[k].stage);
                            ++k;
                        }
                        EXPECT(k == nf.size());
                    }
                    ++n_cases;
                }
        // subsets: the split keys on the real layer id (Op::l >= 21), not on the position in --layers
        {
            const std::vector<int> sub6 = {0, 2, 20, 21, 24, 39};
            const auto nf = dsp::pf::nonfinal_plan(plan(sub6, 0, 6, true, true), false);
            const br::Split sp = br::split_plan(nf, sub6);
            EXPECT(sp.i_late == 3 && sp.ops[sp.b].l == 21 && sp.ops[sp.b - 1].l == 20 && br::check_split(nf, sp, sub6).empty());
            const std::vector<int> sub4 = {0, 2, 20, 39};
            const auto nf4 = dsp::pf::nonfinal_plan(plan(sub4, 0, 4, true, true), false);
            const br::Split s4 = br::split_plan(nf4, sub4);
            EXPECT(s4.i_late == 3 && s4.ops[s4.b].l == 39 && br::check_split(nf4, s4, sub4).empty());
            for (const std::vector<int>& none : {std::vector<int>{0, 2, 20}, std::vector<int>{21, 24}, std::vector<int>{39}})
            {
                const auto n = dsp::pf::nonfinal_plan(plan(none, 0, (int)none.size(), true, true), false);
                const br::Split x = br::split_plan(n, none);
                EXPECT(!x.on() && x.ops.size() == n.size() && x.b == n.size() && br::check_split(n, x, none).empty());
                for (const Op& o : x.ops) EXPECT(o.stage != ST_PF_COMPACT);
            }
            // not ascending: a layer < 21 after the first late layer would run tail rows -> refused
            const std::vector<int> bad = {0, 21, 2};
            const auto nb = dsp::pf::nonfinal_plan(plan(bad, 0, 3, true, true), false);
            EXPECT(!br::check_split(nb, br::split_plan(nb, bad), bad).empty());
        }
        // falsifiers: the compaction one unit early, an op dropped, an op swapped across the boundary
        {
            const std::vector<int> ids = iota(40);
            const auto nf = dsp::pf::nonfinal_plan(plan(ids, 0, 40, true, true), false);
            const br::Split good = br::split_plan(nf, ids);
            br::Split m1 = good;   // ST_PF_COMPACT before hc_attn(21)
            std::swap(m1.ops[m1.b - 1], m1.ops[m1.b - 2]);
            EXPECT(!br::check_split(nf, m1, ids).empty());
            br::Split m2 = good;   // drop layer 21's attn_proj
            m2.ops.erase(m2.ops.begin() + (long)m2.b);
            EXPECT(!br::check_split(nf, m2, ids).empty());
            br::Split m3 = good;   // the boundary one op late: layer 21's attn_proj in range A
            std::swap(m3.ops[m3.b - 1], m3.ops[m3.b]);
            EXPECT(!br::check_split(nf, m3, ids).empty());
            br::Split m4 = good;   // the boundary at layer 20 (range B would run layer 20's ca row / candidate blocks on tail rows)
            m4.i_late = 20;
            EXPECT(!br::check_split(nf, m4, ids).empty());
        }
        // the dial
        EXPECT(br::parse_tail(nullptr) == 128 && br::parse_tail("") == 128 && br::parse_tail("0") == 0 && br::parse_tail("128") == 128);
        EXPECT(br::parse_tail("1024") == 1024 && br::parse_tail("4096") == 4096 && br::parse_tail("127") == -1 && br::parse_tail("1") == -1);
        EXPECT(br::parse_tail("4097") == -1 && br::parse_tail("-1") == -1 && br::parse_tail("12x") == -1 && br::parse_tail("abc") == -1);
        EXPECT(br::late_rows(1024, 128) == 128 && br::late_rows(1024, 0) == 1024 && br::late_rows(1024, 1024) == 1024 && br::late_rows(100, 128) == 100);
        EXPECT(br::shift_rows(1024, 128) == 896 && br::shift_rows(1024, 1024) == 0 && br::shift_rows(1024, 0) == 0 && br::shift_rows(100, 128) == 0);
        EXPECT(br::win_floor(2048, 1024, 128) == 1920 && br::win_floor(2048, 1024, 1024) == 0 && br::win_floor(2048, 1024, 0) == 0 && br::win_floor(200, 100, 128) == 0);
        EXPECT(br::main_rows(1024, 128).lo == 0 && br::main_rows(1024, 128).hi == 128 && br::main_rows(1024, 0).hi == 1024);
        std::printf("ds41 bounded replay split ok: %d full-model cases (engram x dspark), boundary at layer 21, %zu ops in range B of 40 layers\n", n_cases,
                    br::split_plan(dsp::pf::nonfinal_plan(plan(iota(40), 0, 40, true, true), false), iota(40)).ops.size() -
                    br::split_plan(dsp::pf::nonfinal_plan(plan(iota(40), 0, 40, true, true), false), iota(40)).b);
    }
    // the cross-layer geometry the plan relies on (model_geom_ds41.h)
    EXPECT(geom::kv_source(39) == 20 && geom::index_source(39) == 36 && geom::is_kv_source(2) && !geom::is_kv_source(3));
    std::printf("ds41 step order ok: %zu ops for layers {0,2,20,39}, %zu for all 40\n", ps.size(), plan(iota(40), 0, 40, true, true).size());
    return 0;
}
