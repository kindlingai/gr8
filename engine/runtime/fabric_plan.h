// The fabric plan (docs/HOST-CORE.md, R6): every g6::Fabric g6m4 creates, in creation order, as a function of the
// options, the knobs the bring-up reads and a few build values (PlanInputs). Each entry is a whole FabricConfig: a
// fabric's config starts as a copy of the main config as the bring-up has changed it so far (compact, xtrace, DS41's W
// slots), so the plan carries every inherited field rather than re-deriving it at the init site. The three fabrics the
// built model completes (fabg, fabs, fabdg) are planned as far as the options go; their init sites set ncoll and the slot
// or bulk bytes. The plan is engine/scheduler-rs src/fabric_plan.rs; fabric_plan() is its facade: it passes the build's
// constants and the values of helpers the kernels share, and rebuilds the C++ types (the address lists stay here).
// Host only: s2/tests/fabric_plan_vectors_test.cpp runs it in each build's shape.
#pragma once
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "engine_options.h"
#include "fabric_dump.h"
#include "pf_slots.h"
#include "s2/cp_rows.h"
#include "../comms/cplayout.h"
#include "../comms/fabric.h"
#include "../knobs-rs/gr8_knobs.h"
#if G6_MODEL_FLASH
#include "../attn/flash_pool_owner.h"
#include "flash_route_split.h"
#endif
#if G6_NR == 4
#include "qp_wire.h"
#endif
#if G6_MODEL_DS41
#include "ds41_dspark.h"
#include "ds41_pack_layout.h"
#include "ds41_lanes.h"
#include "ds41_pf_split.h"
#include "ds41_prefill_capacity.h"
#include "ds41_prefill_fabric.h"
#include "ds41_rowshard.h"
#include "ds41_rs.h"
#endif

namespace g6rt
{

enum class FabKind
{
    Main, LaneB, Prefill, Prefill2, PoolIds, PoolIds2, RouteSplit, RouteSplit2, Draft, DSpark, Dflash, QpWire, Cp,
    Gather, CpStage, DraftGather,
};
constexpr int kFabKinds = 16;
// main()'s name for each
inline const char* fab_name(FabKind k)
{
    static const char* const n[kFabKinds] = {"fab", "fabB", "fabp", "fabp2", "fabix", "fabix2", "fabrs", "fabrs2", "fabd",
                                             "fabds", "fabf", "fabq", "fabc", "fabg", "fabs", "fabdg"};
    return n[(int)k];
}

struct PlannedFabric
{
    FabKind kind = FabKind::Main;
    bool late = false;         // created after the model build, which sets ncoll and the slot or bulk bytes
    int port_off = 0;          // cfg.port = the options' port + port_off
    g6::FabricConfig cfg;
};

// The knobs the bring-up reads, as plan_knobs() reads them from the table.
struct PlanKnobs
{
    bool xtrace_pf = false;
    std::optional<bool> pf_stripe;
    int64_t pf_stripe_min = 0;
    bool pf_rs = false;
    bool fpf_attn_p1 = false, fpf_dsa_p1 = false, fpf_p1 = false;
    std::optional<int64_t> qa_split_cpu;
    std::optional<bool> cp;
    int64_t cp_r = 0, cp_step_rows = 0;
    bool cp_s3 = false;
    std::optional<int64_t> cp_mmax, cp_r2;
    std::optional<std::string> ds_prefill_w, ds_pf_tail, ds_pf_pool, ds_pf_lanes;
    bool ds_pf_l20_tail = false, ds_pf_rs = false, ds_mhc_row_shard = false, ds_mhc_fuse = false, ds_mhc_y_mxfp8 = false;
    bool ds_pf_direct = false, ds_subdump = false, ds_stripe = false, ds_xtrace = false;
    int64_t ds_stripe_min = 0;
};

inline PlanKnobs plan_knobs()
{
    PlanKnobs k;
    k.xtrace_pf = knob::xtrace_pf();
    k.pf_stripe = knob::pf_stripe();
    k.pf_stripe_min = knob::pf_stripe_min();
    k.pf_rs = knob::pf_rs();
    k.fpf_attn_p1 = knob::fpf_attn_p1(); k.fpf_dsa_p1 = knob::fpf_dsa_p1(); k.fpf_p1 = knob::fpf_p1();
    k.qa_split_cpu = knob::qa_split_cpu();
    k.cp = knob::cp(); k.cp_r = knob::cp_r(); k.cp_step_rows = knob::cp_step_rows(); k.cp_s3 = knob::cp_s3();
    k.cp_mmax = knob::cp_mmax(); k.cp_r2 = knob::cp_r2();
    k.ds_prefill_w = knob::ds_prefill_w(); k.ds_pf_tail = knob::ds_pf_tail(); k.ds_pf_pool = knob::ds_pf_pool();
    k.ds_pf_lanes = knob::ds_pf_lanes();
    k.ds_pf_l20_tail = knob::ds_pf_l20_tail(); k.ds_pf_rs = knob::ds_pf_rs(); k.ds_mhc_row_shard = knob::ds_mhc_row_shard();
    k.ds_mhc_fuse = knob::ds_mhc_fuse(); k.ds_mhc_y_mxfp8 = knob::ds_mhc_y_mxfp8();
    k.ds_pf_direct = knob::ds_pf_direct(); k.ds_subdump = knob::ds_subdump().has_value(); k.ds_stripe = knob::ds_stripe();
    k.ds_xtrace = knob::ds_xtrace(); k.ds_stripe_min = knob::ds_stripe_min();
    return k;
}

// What the plan reads besides the options. main reads the lazy knob helpers other code reads too (pf_sp,
// flash_pool_owner_mode) and the device constants.
struct PlanInputs
{
    PlanKnobs k;
    bool pf_sp = false;             // Flash: pf_sp()
    int pool_owner_mode = 0;        // Flash: flash_pool_owner_mode()
    int dfl_ncoll = 0;              // dfl::NC
    size_t head_payload = 0;        // DS41: glue::head_bytes_tk(MX, nc, true)
    size_t trunk_bytes = 0;         // DS41: sizeof(ds_mhc::trunk_t)
    std::vector<bool> full_index;   // per --layers entry: cp_full_index
    bool cp_build = false;          // G6_CP
    bool gather = false;            // GR8_QA_SPLIT or GR8_IDXQ_SPLIT: fabg
    bool draft_gather = false;      // GR8_MTP_SPLIT: fabdg
};

struct FabricPlan
{
    std::string refusal;                  // non-empty: main dies with it, before any fabric
    std::vector<std::string> log;         // the lines the planning writes to stderr, newline included
    std::vector<PlannedFabric> fabrics;   // creation order
    // what the boot reads afterwards
    size_t dec_slot_bytes = 0;            // every decode collective of MX rows fits it
    int pf_rows = 0, tail = 0, i_late = -1;   // DS41 serve: W, GR8_DS_PF_TAIL, the bounded-replay split
    bool l20_tail = false;                // GR8_DS_PF_L20_TAIL as the serve took it: fabric_plan_apply sets ds41::br::l20_tail()
    int rs_k = 0, quarter = 0;            // GR8_DS_PF_RS: RS layer exchanges, rows a rank owns
    size_t slice_stride = 0;
    bool rowshard = false, y_mxfp8 = false, pf_direct = false;
    int pf_lanes_fab = 1;                 // GR8_DS_PF_LANES
    bool two = false, pool_owner = false, route_split = false;
    bool cp_on = false;
    int cp_R = 0, cp_R2 = 0, cp_s3 = 0, cp_mmax = 0, cp_step = 0, cp_step_R = 0;
    g6::CpColls cp_colls;

    const PlannedFabric* find(FabKind k) const
    {
        for (const PlannedFabric& f : fabrics) if (f.kind == k) return &f;
        return nullptr;
    }
};

namespace fplan
{
inline std::string fmt(const char* f, ...)
{
    char b[512];
    va_list a;
    va_start(a, f);
    vsnprintf(b, sizeof b, f, a);
    va_end(a);
    return b;
}
inline const char* txt(const std::optional<std::string>& v) { return v ? v->c_str() : nullptr; }

inline g6plan_build build()
{
    g6plan_build b{};
    b.model = g6mod::this_model();
    b.nr = g6nr::NR; b.mx = g6mx::MX; b.hid = geom::HID;
#if G6_MODEL_DS41
    b.ds_window = geom::WINDOW; b.ds_tail_rows = geom::TAIL_ROWS; b.ds_cand_source = geom::CAND_SOURCE;
    b.ds_late_start = geom::LATE_LAYER_START; b.ds_pack_rows = ds41::pack::MAX_ROWS; b.ds_draft_ncoll = ds41::dsp::DRAFT_NCOLL;
#endif
    b.coll_bits = (int)g6::CollMask::kBits;
    b.cp = {g6::CpColls::HEADS, g6::CpColls::KV_RANK, g6::CpColls::QK_DIM, g6::CpColls::TOPK};
#if G6_NR == 4
    b.qw_half = qw::HALF;
#endif
#if G6_MODEL_FLASH
    b.route_max_nr = flash_route_split::MAX_NR;
#endif
    return b;
}

inline g6plan_cfg cfg(const g6::FabricConfig& c)
{
    g6plan_cfg x{};
    x.rank = c.rank; x.stripe_min = c.stripe_min; x.slot_bytes = c.slot_bytes; x.run1 = c.run1; x.run2 = c.run2;
    x.slot_bytes1 = c.slot_bytes1; x.slot_bytes2 = c.slot_bytes2; x.run3 = c.run3; x.slot_bytes3 = c.slot_bytes3; x.pool = c.pool;
    x.sge2 = c.sge2; x.compact = c.compact; x.ncoll = c.ncoll; x.port = c.port; x.proxy_cpu = c.proxy_cpu;
    x.vcap = c.vcap.data(); x.n_vcap = c.vcap.size(); x.vdst_k = c.vdst_k.data(); x.n_vdst_k = c.vdst_k.size();
    x.bulk_bytes = c.bulk_bytes; x.bulk_flags = c.bulk_flags; x.xtrace = c.xtrace; x.ring = c.ring; x.relay_cpu = c.relay_cpu;
    return x;
}

inline g6::FabricConfig config(const g6plan_entry& e, const EngineOptions& eo)
{
    const g6plan_cfg& x = e.cfg;
    g6::FabricConfig c;
    c.rank = x.rank; c.ips = eo.cfg.ips; c.ips2 = e.use_pf_ips2 ? eo.pf_ips2 : eo.cfg.ips2; c.stripe_min = x.stripe_min;
    c.slot_bytes = x.slot_bytes; c.run1 = x.run1; c.run2 = x.run2; c.slot_bytes1 = x.slot_bytes1; c.slot_bytes2 = x.slot_bytes2;
    c.run3 = x.run3; c.slot_bytes3 = x.slot_bytes3; c.pool = x.pool; c.sge2 = x.sge2; c.compact = x.compact; c.ncoll = x.ncoll;
    c.port = x.port; c.proxy_cpu = x.proxy_cpu;
    c.vcap.assign(x.vcap, x.vcap + x.n_vcap); c.vdst_k.assign(x.vdst_k, x.vdst_k + x.n_vdst_k);
    c.bulk_bytes = x.bulk_bytes; c.bulk_flags = x.bulk_flags; c.xtrace = x.xtrace; c.ring = x.ring; c.ring_devs = eo.cfg.ring_devs;
    c.relay_cpu = x.relay_cpu;
    return c;
}

inline g6plan_knobs knobs(const PlanKnobs& k)
{
    g6plan_knobs n{};
    n.xtrace_pf = k.xtrace_pf; n.pf_stripe = k.pf_stripe ? (int)*k.pf_stripe : -1; n.pf_stripe_min = k.pf_stripe_min;
    n.pf_rs = k.pf_rs; n.fpf_attn_p1 = k.fpf_attn_p1; n.fpf_dsa_p1 = k.fpf_dsa_p1; n.fpf_p1 = k.fpf_p1;
    n.has_qa_split_cpu = k.qa_split_cpu.has_value(); n.qa_split_cpu = k.qa_split_cpu.value_or(0);
    n.cp = k.cp ? (int)*k.cp : -1; n.cp_r = k.cp_r; n.cp_step_rows = k.cp_step_rows; n.cp_s3 = k.cp_s3;
    n.has_cp_mmax = k.cp_mmax.has_value(); n.cp_mmax = k.cp_mmax.value_or(0);
    n.has_cp_r2 = k.cp_r2.has_value(); n.cp_r2 = k.cp_r2.value_or(0);
    n.ds_prefill_w = txt(k.ds_prefill_w); n.ds_pf_tail = txt(k.ds_pf_tail); n.ds_pf_pool = txt(k.ds_pf_pool);
    n.ds_pf_lanes = txt(k.ds_pf_lanes);
    n.ds_pf_l20_tail = k.ds_pf_l20_tail; n.ds_pf_rs = k.ds_pf_rs; n.ds_mhc_row_shard = k.ds_mhc_row_shard;
    n.ds_mhc_fuse = k.ds_mhc_fuse; n.ds_mhc_y_mxfp8 = k.ds_mhc_y_mxfp8; n.ds_pf_direct = k.ds_pf_direct;
    n.ds_subdump = k.ds_subdump; n.ds_stripe = k.ds_stripe; n.ds_xtrace = k.ds_xtrace; n.ds_stripe_min = k.ds_stripe_min;
    return n;
}
}   // namespace fplan

inline FabricPlan fabric_plan(const EngineOptions& eo, const PlanInputs& in)
{
    static_assert(sizeof(int) == sizeof(int32_t) && kFabKinds == sizeof(g6plan_plan::entries) / sizeof(g6plan_entry), "g6plan.h");
    FabricPlan p;
    if (g6sch_layout() != g6sch_layout_cpp()) {
        p.refusal = "scheduler-rs: struct layouts differ from g6sched.h (rebuild libg6host.a)";
        return p;
    }
    g6plan_in c{};
    c.b = fplan::build();
    c.cfg = fplan::cfg(eo.cfg);
    c.n_ips = eo.cfg.ips.size(); c.n_ips2 = eo.cfg.ips2.size(); c.n_pf_ips2 = eo.pf_ips2.size();
    c.nc = eo.NC; c.nx = eo.NX; c.nl = eo.NL; c.mp = eo.MP; c.mtp = eo.mtp; c.lanes = eo.lanes; c.drafts = eo.K;
    c.chunk = eo.plan.chunk; c.s2sess = eo.s2sess; c.rt_win = eo.rt_win; c.qp_roll_s = eo.qp_roll_s; c.qp_wire_e = eo.qp_wire_e;
    c.s2mode = eo.s2mode; c.dfl = eo.dfl;
#if G6_MODEL_FLASH
    c.route_split_mode = eo.route_split_mode;
    c.pool_payload = flash_pool_owner::payload(eo.MP);
    c.route_payload = flash_route_split::payload(eo.MP, (int)eo.cfg.ips.size());
#endif
#if G6_MODEL_DS41
    c.c8_pack_env = eo.c8_pack_env; c.ds_draft_cap = eo.ds_draft_cap; c.dsk_on = eo.dsk.on();
    for (int i = 0; i < ds41::dsp::DRAFT_NCOLL; ++i)
        c.dsp_coll_max = std::max(c.dsp_coll_max, ds41::dsp::coll_bytes(i, geom::HID, std::min(eo.s2sess, eo.ds_draft_cap)));
#endif
    c.layers = eo.layers.data(); c.n_layers = eo.layers.size();
    c.row_counts = eo.row_counts.data(); c.n_row_counts = eo.row_counts.size();
    c.knobs = fplan::knobs(in.k);
    c.pf_sp = in.pf_sp; c.pool_owner_mode = in.pool_owner_mode; c.dfl_ncoll = in.dfl_ncoll;
    c.head_payload = in.head_payload; c.trunk_bytes = in.trunk_bytes;
    const std::unique_ptr<bool[]> full(new bool[in.full_index.size() + 1]);
    for (size_t i = 0; i < in.full_index.size(); ++i) full[i] = in.full_index[i];
    c.full_index = full.get(); c.n_full_index = in.full_index.size();
    c.cp_build = in.cp_build; c.gather = in.gather; c.draft_gather = in.draft_gather;

    const std::unique_ptr<g6plan_result, void (*)(g6plan_result*)> r(g6plan_run(&c), g6plan_free);
    const g6plan_plan& v = *g6plan_view(r.get());
    p.refusal = v.refusal;
    for (size_t i = 0; i < v.n_log; ++i) p.log.push_back(v.log[i]);
    for (int i = 0; i < v.n; ++i) {
        const g6plan_entry& e = v.entries[i];
        p.fabrics.push_back(PlannedFabric{FabKind(e.kind), e.late, e.port_off, fplan::config(e, eo)});
    }
    p.dec_slot_bytes = v.dec_slot_bytes; p.pf_rows = v.pf_rows; p.tail = v.tail; p.i_late = v.i_late; p.l20_tail = v.l20_tail;
    p.rs_k = v.rs_k; p.quarter = v.quarter; p.slice_stride = v.slice_stride; p.rowshard = v.rowshard; p.y_mxfp8 = v.y_mxfp8;
    p.pf_direct = v.pf_direct; p.pf_lanes_fab = v.pf_lanes_fab; p.two = v.two; p.pool_owner = v.pool_owner;
    p.route_split = v.route_split; p.cp_on = v.cp_on; p.cp_R = v.cp_r; p.cp_R2 = v.cp_r2; p.cp_s3 = v.cp_s3;
    p.cp_mmax = v.cp_mmax; p.cp_step = v.cp_step; p.cp_step_R = v.cp_step_r;
    g6::CpColls& cc = p.cp_colls;
    const g6plan_cp& x = v.cp;
    cc.nr = x.nr; cc.nl = x.nl; cc.mmax = x.mmax; cc.q_row = x.q_row; cc.cand_row = x.cand_row; cc.part_row = x.part_row;
    cc.q.assign(x.q, x.q + x.n_q); cc.cand.assign(x.cand, x.cand + x.n_cand); cc.part.assign(x.part, x.part + x.n_part);
    cc.cap.assign(x.cap, x.cap + x.ncoll); cc.dst_k.assign(x.dst_k, x.dst_k + x.ncoll);
    return p;
}

// The process-wide state the plan decides, which step-time code reads: the bring-up calls it.
inline void fabric_plan_apply(const FabricPlan& p)
{
#if G6_MODEL_DS41
    ds41::br::l20_tail() = p.l20_tail;
#else
    (void)p;
#endif
}

// Creates the planned fabrics main builds before the model, in plan order, each with its proxy, and calls up(entry,
// fabric) after each. dump (GR8_FABRIC_DUMP): prints every entry instead and creates none.
template <class Up>
inline void fabric_bring_up(const FabricPlan& p, g6::Fabric* const (&fabs)[kFabKinds], bool dump, Up&& up)
{
    fabric_plan_apply(p);
    for (const PlannedFabric& e : p.fabrics) {
        if (dump) { printf("%s\n", describe_fabric(fab_name(e.kind), e.cfg).c_str()); continue; }
        if (e.late) continue;
        g6::Fabric& fb = *fabs[(int)e.kind];
        fb.init(e.cfg);
        fb.start_proxy();
        up(e, fb);
    }
}

// Every field of the plan, one line each, for the vectors test.
inline std::string describe(const FabricPlan& p)
{
    std::string s;
    for (const std::string& l : p.log) s += "log " + l;
    if (!p.refusal.empty()) return s + "refusal " + p.refusal + "\n";
    for (const PlannedFabric& f : p.fabrics)
        s += "late " + std::to_string(f.late) + " port_off " + std::to_string(f.port_off) + " " + describe_fabric(fab_name(f.kind), f.cfg) + "\n";
    s += fplan::fmt("scalars dec_slot_bytes=%zu pf_rows=%d tail=%d i_late=%d l20_tail=%d rs_k=%d quarter=%d slice_stride=%zu rowshard=%d "
                    "y_mxfp8=%d pf_direct=%d pf_lanes_fab=%d two=%d pool_owner=%d route_split=%d\n",
                    p.dec_slot_bytes, p.pf_rows, p.tail, p.i_late, p.l20_tail, p.rs_k, p.quarter, p.slice_stride, p.rowshard,
                    p.y_mxfp8, p.pf_direct, p.pf_lanes_fab, p.two, p.pool_owner, p.route_split);
    const g6::CpColls& c = p.cp_colls;
    s += fplan::fmt("cp on=%d R=%d R2=%d s3=%d mmax=%d step=%d step_R=%d colls nr=%d nl=%d mmax=%d rows=%u,%u,%u", p.cp_on, p.cp_R,
                    p.cp_R2, p.cp_s3, p.cp_mmax, p.cp_step, p.cp_step_R, c.nr, c.nl, c.mmax, c.q_row, c.cand_row, c.part_row);
    auto ints = [&](const char* k, const std::vector<int>& v) {
        s += std::string(" ") + k + "=";
        for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    };
    ints("q", c.q); ints("cand", c.cand); ints("part", c.part);
    return s + "\n";
}

}   // namespace g6rt
