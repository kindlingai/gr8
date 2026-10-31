// DeepSeek-V4.1-Flash-1017 decode step: the order of its stages and the table of ops that fill them (host-only, no CUDA).
// runtime/ds41_step.cuh walks plan() and calls the table; engine/tests/test_ds41_step_order.cpp does the same on a recording
// table. A null slot is an op the build lacks (NOTSUP, as attn/ops.h): the walk stops there and names the stage and layer,
// it does not skip. runtime/ds41_step.cuh (ds41_table) points each slot at its function, one line per slot.
//
// One layer is two sub-layers; each is prepare -> block -> publish partial -> consume (the Flash shape, engine/mhc). The
// plan is cut into units, the pieces m4sim's per-rank wrappers (rank.cuh launch_embed / launch_attn_block / launch_consume /
// launch_mlp_block / launch_head_local / launch_head_merge / launch_step_out) launch. The mHC prepare of a sub-layer runs inside
// the consume unit before it, as in Flash (the consume and the next prepare are one launch group):
//   U_STEP_IN    step_in
//   U_EMBED      embed [engram(0)] hc_attn(0)                                                          rc 0
//   U_ATTN(i)    attn_proj [ca_row] [index_topk [cand_blocks]] attn_core attn_out publish_attn
//   U_CONS_A(i)  consume_attn hc_ffn                                                                   rc 2i+1
//   U_MLP(i)     router shared experts publish_ffn
//   U_CONS_F(i)  consume_ffn [engram(i+1)] hc_attn(i+1)   (the last layer: finish)                     rc 2i+2
//   U_HEAD_LOCAL / U_HEAD_MERGE / U_STEP_OUT
// finish also does the final norm (flash_mhc::k_mhc_out_inv_final). Layer i is the position in Model::layers, l = geom layer
// number (m4sim --layers 0,2,20,39 gives i = 0..3 and l = 0, 2, 20, 39).
#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include "../model_geom.h"

namespace ds41
{
constexpr int NOTSUP = 801;   // cudaErrorNotSupported, the value of G6_ATTN_NOTSUP

enum Stage : int
{
    ST_STEP_IN,        // driver: StepIn -> tok/pos/cells (k_step_in)
    ST_EMBED,          // mHC: ds_mhc::embed (4 copies of the embedding row, pre = [1,0,0,0])
    ST_ENGRAM,         // Engram, off by default (plan(engram=false)); layers 1, 14: n-gram rows added to the trunk
    ST_HC_ATTN,        // mHC: ds_mhc::prepare with hc_attn.*: sub-layer input y = BF16 [rows,5120] (norm folded in)
    ST_ATTN_PROJ,      // attention: q_a, q_a norm, q_b (+rope), kv (+norm, rope), window-KV write
    ST_CA_ROW,         // indexer: compressed-KV row (and index-K row) of this token; only at the kv sources 2, 8, 14, 20
    ST_INDEX_TOPK,     // indexer: top-512 keys, computed at an index source (2, 8, 14, 20, 24, 28, 32, 36)
    ST_INDEX_BORROW,   // indexer: a reuse layer whose index source is not in the layer subset (--layers 0,2,20,39: layer 39, source 36).
                       //        The list of that source is not computed. Without a fixture the op leaves the last list an index source
                       //        published in place (the reuse rule of the official model.py: layer 20's in that subset); with a dump of the
                       //        source's top-512 it loads it. Either way the run is synthetic against the full model (synthetic(plan)).
    ST_CAND_BLOCKS,    // indexer: candidate blocks (top 2048 blocks of 8 tokens); only at the candidate source, layer 20
    ST_ATTN_CORE,      // attention: window 128 + sink (layers 0, 1), + compressed rows via top-512 (2..39)
    ST_ATTN_OUT,       // attention: inverse rope, grouped o_a (8 groups) / o_b -> this rank's partial
    ST_PUBLISH_ATTN,   // driver: put_publish / slot_publish of collective c = 2i
    ST_CONSUME_ATTN,   // mHC: ds_mhc::consume (waits the peers' partials of c = 2i)
    ST_HC_FFN,         // mHC: ds_mhc::prepare with hc_ffn.*
    ST_ROUTER,         // MoE: sqrtsoftplus + bias, noaux_tc top-6, route weights, plan
    ST_SHARED,         // MoE: shared expert (MXFP8, 2304 wide)
    ST_EXPERTS,        // MoE: routed experts (MXFP4), swiglu limit 10, route scale 1.5
    ST_PUBLISH_FFN,    // driver: publish of collective c = 2i+1
    ST_CONSUME_FFN,    // mHC: ds_mhc::consume (not on the last layer)
    ST_FINISH,         // mHC: ds_mhc::finish = last consume + final norm -> lm_head input fragments (last layer only)
    ST_HEAD_LOCAL,     // driver: lm_head GEMV over this rank's vocab slice, local argmax / lse
    ST_HEAD_MERGE,     // driver: head all-gather and merge
    ST_STEP_OUT,       // driver: k_step_out
    // DSpark (off by default: plan(dspark=false), GR8_DS_MTP=0): docs/DEEPSEEK-DSPARK.md section 3
    ST_DS_TAP,         // DSpark: first op of U_ATTN(i) for l in 37..39 (after hc_attn(i)): dsp_tap[:, l-37] = bf16(mean of the 4 streams)
    ST_DS_MAIN,        // DSpark: after ST_FINISH in U_CONS_F(last): main_proj, main_norm, 3 wkv, ring write (every verify row)
    ST_DS_DRAFT_IN,    // DSpark + mHC, draft plan only: replaces step_in + embed (dsp_q / dsp_golden -> dsp_tok / dsp_pos, embed on MB rows)
    ST_DS_HEAD_LOCAL,  // DSpark, draft plan only: lm_head GEMV over the shard for all draft rows
    ST_DS_MARKOV,      // DSpark, draft plan only: 5 sequential markov add + confidence + head merge
    // Bounded replay (prefill chunk only, never in plan()): ds41_pf_split.h puts it last in U_CONS_F of the last layer before
    // geom::LATE_LAYER_START. Moves the last `tail` rows of the per-row state (trunk, mHC coefficients, hc_y, pos/tok/tgt, the
    // layer-20 top-k and candidate lists) to rows 0..tail-1; the layers after it run tail rows.
    ST_PF_COMPACT,
    ST_COUNT
};
inline const char* stage_name(Stage s)
{
    static const char* const n[ST_COUNT] = {"step_in", "embed", "engram", "hc_attn", "attn_proj", "ca_row", "index_topk",
        "index_borrow", "cand_blocks", "attn_core", "attn_out", "publish_attn", "consume_attn", "hc_ffn", "router", "shared", "experts",
        "publish_ffn", "consume_ffn", "finish", "head_local", "head_merge", "step_out", "ds_tap", "ds_main", "ds_draft_in",
        "ds_head_local", "ds_markov", "pf_compact"};
    return s >= 0 && s < ST_COUNT ? n[s] : "?";
}
// the component that writes the op ("driver" stages are filled by ds41_step.cuh itself)
inline const char* stage_owner(Stage s)
{
    switch (s)
    {
    case ST_EMBED: case ST_HC_ATTN: case ST_CONSUME_ATTN: case ST_HC_FFN: case ST_CONSUME_FFN: case ST_FINISH: return "mHC";
    case ST_ATTN_PROJ: case ST_ATTN_CORE: case ST_ATTN_OUT: return "attention";
    case ST_CA_ROW: case ST_INDEX_TOPK: case ST_INDEX_BORROW: case ST_CAND_BLOCKS: return "indexer";
    case ST_ROUTER: case ST_SHARED: case ST_EXPERTS: return "MoE";
    case ST_ENGRAM: return "Engram";
    case ST_DS_TAP: case ST_DS_MAIN: case ST_DS_DRAFT_IN: case ST_DS_HEAD_LOCAL: case ST_DS_MARKOV: return "DSpark";
    default: return "driver";
    }
}

enum Unit : int { U_STEP_IN, U_EMBED, U_ATTN, U_CONS_A, U_MLP, U_CONS_F, U_HEAD_LOCAL, U_HEAD_MERGE, U_STEP_OUT, U_COUNT };
inline const char* unit_name(Unit u)
{
    static const char* const n[U_COUNT] = {"step_in", "embed", "attn", "consume_attn", "mlp", "consume_ffn", "head_local",
                                           "head_merge", "step_out"};
    return u >= 0 && u < U_COUNT ? n[u] : "?";
}

// one step of the walk. i: position in Model::layers of the layer the op works on (-1 for a stage outside the layers), l: geom
// layer number (-1 likewise). unit/ui: the wrapper that launches the op and the argument that wrapper gets (the layer
// position; 0 for the units outside the layers) -- hc_attn(i+1) has i+1 but sits in U_CONS_F with ui = i.
// src: ST_INDEX_TOPK / ST_INDEX_BORROW only, the layer the list comes from (= l at an index source).
// rc: the residual-copy point after the unit (m4sim's hook: 0 after embed, 2i+1 after consume_attn, 2i+2 after the ffn
// consume or finish), -1 for none; set on the last op of the unit only.
struct Op
{
    Stage stage = ST_STEP_IN;
    Unit unit = U_STEP_IN;
    int i = -1, l = -1, ui = 0, src = -1, rc = -1;
};
inline bool in_subset(const std::vector<int>& ids, int l)
{
    for (int x : ids) if (x == l) return true;
    return false;
}
// first/last: the layer positions [first, last) of this part (the whole step is [0, nl)); prologue = step_in + embed (needs
// first == 0: the embed unit ends with the prepare of layer 0), epilogue = head_local + head_merge + step_out. A part that ends
// before nl also ends with the prepare of the next layer (U_CONS_F), so a part that starts there has no prepare of its own.
// engram: put the Engram stage at the entry of layers 1 and 14 (default off: rows made without it are "diag").
// dspark: ST_DS_TAP opens U_ATTN of layers 37..39; ST_DS_MAIN follows ST_FINISH in the last U_CONS_F (the rc point moves to it).
inline std::vector<Op> plan(const std::vector<int>& ids, int first, int last, bool prologue, bool epilogue, bool engram = false,
                            bool dspark = false)
{
    std::vector<Op> p;
    auto add = [&](Stage s, Unit u, int ui, int i, int rc = -1, int src = -1)
    {
        Op o; o.stage = s; o.unit = u; o.ui = ui; o.i = i; o.l = i >= 0 ? ids[i] : -1; o.src = src; o.rc = rc;
        p.push_back(o);
    };
    auto entry = [&](Unit u, int ui, int i, int rc)   // the stages that open layer i: [engram] hc_attn, rc on the unit's last op
    {
        if (engram && geom::is_engram(ids[i])) add(ST_ENGRAM, u, ui, i);
        add(ST_HC_ATTN, u, ui, i, rc);
    };
    const int nl = (int)ids.size();
    if (prologue)
    {
        add(ST_STEP_IN, U_STEP_IN, 0, -1);
        add(ST_EMBED, U_EMBED, 0, -1);
        if (nl > 0) entry(U_EMBED, 0, 0, 0);
        else p.back().rc = 0;
    }
    for (int i = first; i < last; ++i)
    {
        const int l = ids[i];
        if (dspark && l >= geom::DS_TARGET0 && l < geom::DS_TARGET0 + geom::N_NEXTN) add(ST_DS_TAP, U_ATTN, i, i);
        add(ST_ATTN_PROJ, U_ATTN, i, i);
        if (geom::is_kv_source(l)) add(ST_CA_ROW, U_ATTN, i, i);
        if (geom::compress_ratio(l) > 0)
        {
            if (geom::is_index_source(l))
            {
                add(ST_INDEX_TOPK, U_ATTN, i, i, -1, l);
                if (l == geom::CAND_SOURCE) add(ST_CAND_BLOCKS, U_ATTN, i, i);
            }
            else if (!in_subset(ids, geom::index_source(l)))
                add(ST_INDEX_BORROW, U_ATTN, i, i, -1, geom::index_source(l));
        }
        add(ST_ATTN_CORE, U_ATTN, i, i);
        add(ST_ATTN_OUT, U_ATTN, i, i);
        add(ST_PUBLISH_ATTN, U_ATTN, i, i);
        add(ST_CONSUME_ATTN, U_CONS_A, i, i);
        add(ST_HC_FFN, U_CONS_A, i, i, 2 * i + 1);
        add(ST_ROUTER, U_MLP, i, i);
        add(ST_SHARED, U_MLP, i, i);
        add(ST_EXPERTS, U_MLP, i, i);
        add(ST_PUBLISH_FFN, U_MLP, i, i);
        if (i + 1 < nl)
        {
            add(ST_CONSUME_FFN, U_CONS_F, i, i);
            entry(U_CONS_F, i, i + 1, 2 * i + 2);
        }
        else if (dspark)
        {
            add(ST_FINISH, U_CONS_F, i, i);
            add(ST_DS_MAIN, U_CONS_F, i, i, 2 * i + 2);
        }
        else
            add(ST_FINISH, U_CONS_F, i, i, 2 * i + 2);
    }
    if (epilogue)
    {
        add(ST_HEAD_LOCAL, U_HEAD_LOCAL, 0, -1);
        add(ST_HEAD_MERGE, U_HEAD_MERGE, 0, -1);
        add(ST_STEP_OUT, U_STEP_OUT, 0, -1);
    }
    return p;
}

// The DSpark draft block (a second Model, ids {40,41,42}): ST_DS_DRAFT_IN replaces step_in + embed, then the layer ops (no ca / index
// stages: compress_ratio is 0), ST_FINISH, ST_DS_HEAD_LOCAL, ST_DS_MARKOV. The five-stage contract: docs/DEEPSEEK-DSPARK.md section 3.
inline std::vector<Op> plan_draft(const std::vector<int>& ids)
{
    std::vector<Op> src = plan(ids, 0, (int)ids.size(), true, false), p;
    for (const Op& o : src)
    {
        if (o.stage == ST_EMBED) continue;               // inside ST_DS_DRAFT_IN
        Op x = o;
        if (o.stage == ST_STEP_IN) x.stage = ST_DS_DRAFT_IN;
        p.push_back(x);
    }
    Op h; h.stage = ST_DS_HEAD_LOCAL; h.unit = U_HEAD_LOCAL; p.push_back(h);
    Op k; k.stage = ST_DS_MARKOV; k.unit = U_HEAD_MERGE; p.push_back(k);
    return p;
}

// A plan with a borrowed index list is a synthetic subset run: the reuse layer's list is not what the full model gives, so the
// run is not an oracle gate for that layer unless the source's top-512 is injected (a dump) -- and then only for the layers
// whose every input is in the subset. The full gate is the all-40-layer prefix.
inline bool synthetic(const std::vector<Op>& p)
{
    for (const Op& o : p) if (o.stage == ST_INDEX_BORROW) return true;
    return false;
}
inline std::string synthetic_note(const std::vector<Op>& p)
{
    std::string s;
    for (const Op& o : p)
        if (o.stage == ST_INDEX_BORROW)
            s += (s.empty() ? "" : ", ") + std::string("layer ") + std::to_string(o.l) + " (index source " + std::to_string(o.src) + ")";
    return s.empty() ? s : "synthetic subset run, not an oracle gate: the index list of " + s + " is borrowed (source layer not in --layers)";
}

// Where each unit's ops sit in a plan (each unit instance is contiguous): get(unit, ui) = [b, e) in the plan, empty if the
// plan has no such unit (a part split away from it).
struct Span { size_t b = 0, e = 0; bool empty() const { return b == e; } };
struct Index
{
    std::vector<Span> u[U_COUNT];
    Span get(Unit un, int ui) const { return ui >= 0 && ui < (int)u[un].size() ? u[un][ui] : Span(); }
};
inline Index index(const std::vector<Op>& p)
{
    Index x;
    for (size_t k = 0; k < p.size(); ++k)
    {
        auto& v = x.u[p[k].unit];
        if ((int)v.size() <= p[k].ui) v.resize(p[k].ui + 1);
        Span& s = v[p[k].ui];
        if (s.empty()) s.b = k;
        s.e = k + 1;
    }
    return x;
}

// An op: model = the g6rt::Model*, i = Op::i, rows = M, stream = the cudaStream_t. 0 = ok, NOTSUP = the op cannot do this
// step (same as a null slot), anything else is an error code. The ops that need Op::src read it through the layer:
// src = geom::index_source(l) (no extra argument, one signature).
using Fn = int (*)(void* model, int i, int rows, void* stream);
struct Table
{
    Fn fn[ST_COUNT] = {};
    // build-time hooks, one per layer, called once from build / prepare; a null slot is skipped (the layer then stops
    // at its first stage that needs the binding, with the NOTSUP message of that stage)
    Fn bind_attn = nullptr;    // attention / indexer: bind the layer's attention, ca/ica and window caches (build, rows = capacity)
    Fn bind_moe = nullptr;     // MoE: load the layer's router, shared and routed experts (build, rows = 0)
    Fn prepare_attn = nullptr; // attention / indexer: per row count M: cells, scratch (prepare, rows = M)
    Fn prepare_moe = nullptr;  // MoE: per row count M: kernel arguments (prepare, rows = M)
};

struct Stop
{
    bool stopped = false;
    size_t at = 0, done = 0;   // index in the plan of the stop, ops that ran
    Op op;
    int code = 0;              // NOTSUP (null slot or the op's answer) or the op's error
    bool null_slot = false;
};
// Walk the plan ops [b, e) in order. After each op that returned 0, after(op) runs. The first null slot or non-zero answer ends
// the walk (Stop::at is the index in the plan).
template <class After>
inline Stop run(const std::vector<Op>& p, size_t b, size_t e, const Table& t, void* model, int rows, void* stream, After after)
{
    Stop s;
    for (size_t k = b; k < e; ++k)
    {
        const Op& o = p[k];
        const Fn f = t.fn[o.stage];
        int code = f ? f(model, o.i, rows, stream) : NOTSUP;
        if (code != 0)
        {
            s.stopped = true; s.at = k; s.op = o; s.code = code; s.null_slot = !f;
            return s;
        }
        ++s.done;
        after(o);
    }
    return s;
}
template <class After>
inline Stop run(const std::vector<Op>& p, const Table& t, void* model, int rows, void* stream, After after)
{
    return run(p, 0, p.size(), t, model, rows, stream, after);
}
inline std::string stop_message(const Stop& s)
{
    char b[320];
    std::snprintf(b, sizeof b, "DS41 step stopped at stage %s (owner %s), layer %d (position %d), op %zu of the step: %s",
                  stage_name(s.op.stage), stage_owner(s.op.stage), s.op.l, s.op.i, s.at,
                  s.null_slot ? "not in the op table (NOTSUP)" : s.code == NOTSUP ? "the op answered NOTSUP" : "the op failed");
    std::string m = b;
    if (!s.null_slot && s.code != NOTSUP) m += " (code " + std::to_string(s.code) + ")";
    if (s.op.stage == ST_INDEX_BORROW) m += " [index list of layer " + std::to_string(s.op.src) + " is not computed in this subset: oracle fixture needed, diag run]";
    return m;
}
}
