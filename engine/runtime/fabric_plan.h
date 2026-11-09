// The fabric plan (docs/HOST-CORE.md, R6c): every g6::Fabric g6m4 creates, in creation order, as a function of the
// options, the knobs the bring-up reads and a few build values (PlanInputs). Each entry is a whole FabricConfig: a
// fabric's config starts as a copy of the main config as the bring-up has changed it so far (compact, xtrace, DS41's W
// slots), so the plan carries every inherited field rather than re-deriving it at the init site. The three fabrics the
// built model completes (fabg, fabs, fabdg) are planned as far as the options go; their init sites set ncoll and the slot
// or bulk bytes. Host only: s2/tests/fabric_plan_vectors_test.cpp runs it in each build's shape.
#pragma once
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "ds41_lanes.h"
#include "ds41_pf_split.h"
#include "ds41_prefill_capacity.h"
#include "ds41_prefill_fabric.h"
#include "ds41_rowshard.h"
#include "ds41_rs.h"
#include "ds41_v2.h"
#endif

namespace g6rt
{

enum class FabKind
{
    Main, LaneB, Prefill, Prefill2, PoolIds, PoolIds2, RouteSplit, RouteSplit2, Draft, DSpark, Dflash, QpWire, Cp,
    Gather, CpStage, DraftGather, Ds41V2,
};
constexpr int kFabKinds = 17;
// main()'s name for each
inline const char* fab_name(FabKind k)
{
    static const char* const n[kFabKinds] = {"fab", "fabB", "fabp", "fabp2", "fabix", "fabix2", "fabrs", "fabrs2", "fabd",
                                             "fabds", "fabf", "fabq", "fabc", "fabg", "fabs", "fabdg", "fabv2"};
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
    bool ds_pf_v2 = false, ds_pf_v2_xtrace = false;
    bool ds_pf_l20_tail = false, ds_pf_rs = false, ds_mhc_row_shard = false, ds_mhc_fuse = false, ds_mhc_y_mxfp8 = false;
    bool ds_pf_direct = false, ds_subdump = false, ds_stripe = false, ds_xtrace = false;
    int64_t ds_stripe_min = 0, ds_pf_v2_stripe_min = 0;
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
    k.ds_pf_lanes = knob::ds_pf_lanes(); k.ds_pf_v2 = knob::ds_pf_v2();
    k.ds_pf_v2_xtrace = knob::ds_pf_v2_xtrace();
    k.ds_pf_v2_stripe_min = knob::ds_pf_v2_stripe_min();
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
    std::vector<bool> attn;         // per --layers entry: cp_attn_layer (Flash: not the KDA layers); empty = every layer
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

// "" or the refusal; p holds what was planned before it
inline std::string plan(const EngineOptions& eo, const PlanInputs& in, FabricPlan& p)
{
    const PlanKnobs& k = in.k;
    const int MX = g6mx::MX, HID = geom::HID;
    const int NC = eo.NC, NX = eo.NX, MP = eo.MP, mtp = eo.mtp;
    const bool s2mode = eo.s2mode, dfl = eo.dfl;
    auto add = [&](FabKind kind, int port_off, const g6::FabricConfig& c, bool late = false) {
        PlannedFabric f;
        f.kind = kind; f.late = late; f.port_off = port_off; f.cfg = c;
        p.fabrics.push_back(f);
    };
    g6::FabricConfig cfg = eo.cfg;
    cfg.slot_bytes = NX ? (size_t)MX * 6144 * 2 : (size_t)MX * HID * 2;   // Engram: [rows][6144] BF16 slots (ds41_engram_ops.cuh)
    p.dec_slot_bytes = cfg.slot_bytes;
    g6::FabricConfig cm;                // the main fabric's own runs (DS41 serve); run1 = 0 = uniform
#if G6_MODEL_DS41
    g6::FabricConfig cmB;               // GR8_DS_PF_LANES=2: lane B's runs
    // A W-row headless prefill chunk uses the same fabric as decode; unset, the slot holds MX rows.
    if (s2mode) {
        const int pf_rows = ds41_prefill_rowcap(txt(k.ds_prefill_w), MX, k.ds_pf_v2);
        if (!pf_rows) return k.ds_pf_v2 ? "DS41 v2 serve: GR8_DS_PREFILL_W must be MX..8192" : "DS41 serve: GR8_DS_PREFILL_W must be MX..4096";
        p.pf_rows = pf_rows;
        cfg.slot_bytes = ds41_prefill_fabric_payload(pf_rows, HID, NX ? 6144 : 0);
        // the main fabric's slots sized per collective as a W chunk sends them (s2/ds41_prefill.h ds41_chunk_collectives):
        // [0, c1) layer exchanges of W rows; [c1, 2 NL + 1) the bounded-replay layers and the head; [2 NL + 1, NC) the
        // Engram reduces of W rows. Every run also holds the decode slot.
        const int tail = ds41::br::parse_tail(txt(k.ds_pf_tail));
        if (tail < 0) return ds41::br::tail_error(txt(k.ds_pf_tail), false);   // the boot has not set l20_tail() yet
        p.tail = tail;
        p.l20_tail = k.ds_pf_l20_tail;
        if (p.l20_tail && !tail) return "GR8_DS_PF_L20_TAIL=1 needs GR8_DS_PF_TAIL (layer 20 joins the bounded replay)";
        const int i_late = tail ? ds41::br::late_start(eo.layers, p.l20_tail) : -1;
        p.i_late = i_late;
        bool pf_rs_on = false;
        // GR8_DS_PF_RS=1: layer exchanges c < 2 i_late run as reduce-scatter + all-gather (ds41_rs.h). Run 0 holds NR
        // slices, and a 4th run [NC, NC + rs_k) holds the AG collectives.
        if (k.ds_pf_rs) {
            if (i_late < 0) return "GR8_DS_PF_RS=1 needs GR8_DS_PF_TAIL (the RS collectives are the layer exchanges before the bounded-replay split)";
            if (pf_rows <= MX || pf_rows < g6nr::NR) return "GR8_DS_PF_RS=1 needs GR8_DS_PREFILL_W > MX";
            pf_rs_on = true;
            if (NC + 2 * i_late > (int)g6::CollMask::kBits) return "GR8_DS_PF_RS=1: more than 128 collectives (NC + 2 i_late)";
        }
        if (k.ds_mhc_row_shard) {
            if (!ds41::rowshard::boot_ok(pf_rows, tail, g6nr::NR, pf_rs_on, k.ds_mhc_fuse))
                return "GR8_DS_MHC_ROW_SHARD=1 needs W=4096, tail=128, TP4, PF_RS=1, MHC_FUSE=1";
            if (NC + 2 * i_late + 1 > (int)g6::CollMask::kBits) return "GR8_DS_MHC_ROW_SHARD: more than 128 collectives";
            const size_t state = ds41::rowshard::state_bytes(tail, HID, in.trunk_bytes);
            if (state > (size_t)1024 * HID * 2) return "GR8_DS_MHC_ROW_SHARD: late state exceeds an AG slot";
            p.rowshard = true;
            p.log.push_back(fmt("# rank %d: GR8_DS_MHC_ROW_SHARD=1 Q=1024 late=128 state_bytes=%zu\n", cfg.rank, state));
        }
        if (k.ds_mhc_y_mxfp8) {
            if (!p.rowshard) return "GR8_DS_MHC_Y_MXFP8=1 needs GR8_DS_MHC_ROW_SHARD=1";
            p.y_mxfp8 = true;
            p.log.push_back(fmt("# rank %d: GR8_DS_MHC_Y_MXFP8=1 lossy y all-gather\n", cfg.rank));
        }
        const auto runs = ds41::pf::fabric_runs(pf_rows, tail, i_late, eo.NL, NX, MX, HID, g6nr::NR, in.head_payload, pf_rs_on,
                                                eo.c8_pack_env == 1, p.rowshard);
        cm = runs.cfg;
        p.rs_k = runs.rs_k;
        p.quarter = runs.quarter;
        p.slice_stride = runs.slice_stride;
        // GR8_DS_PF_POOL=P: run 0 and run 3 keep P slots each (comms/fabric.h FabricConfig::pool)
        if (const char* pv = txt(k.ds_pf_pool); pv && *pv && strcmp(pv, "0") != 0) {
            char* end = nullptr;
            const long P = strtol(pv, &end, 10);
            if (!end || *end || P < 2 || P > cm.run1) return "GR8_DS_PF_POOL must be 0 (off) or 2..the run 0 collectives (2 i_late with GR8_DS_PF_TAIL, else 2 NL)";
            cm.pool = (int)P;
            p.log.push_back(fmt("# rank %d DS41 fabric slot pool: GR8_DS_PF_POOL=%d (run 0: %d collectives, %s)\n", cfg.rank, cm.pool,
                                cm.run1, p.rs_k ? "run 3 AG pooled too" : "no run 3"));
        }
        // GR8_DS_PF_LANES=2 (ds41_lanes.h): lane B's slot table = the main one without the C8-pack late slots; the same pool
        p.pf_lanes_fab = ds41::lanes::parse(txt(k.ds_pf_lanes));
        if (!p.pf_lanes_fab) return ds41::lanes::kParseError;
        if (p.pf_lanes_fab == 2) {
            const auto rb = ds41::lanes::fabric_runs_b(pf_rows, tail, i_late, eo.NL, NX, MX, HID, g6nr::NR, in.head_payload, pf_rs_on,
                                                       p.rowshard);
            if (rb.rs_k != p.rs_k || rb.cfg.run1 != cm.run1 || rb.cfg.ncoll != cm.ncoll)
                return "DS41 lane B fabric: its runs or row-shard state collective differ from the main fabric's";
            cmB = rb.cfg;
            cmB.pool = cm.pool;
        }
        if (pf_rs_on)
            p.log.push_back(fmt("# rank %d DS41 prefill RS+AG: GR8_DS_PF_RS=1 (c < %d; Q %d rows, slice stride %zu B; AG collectives %d..%d, %zu B each)\n",
                                cfg.rank, p.rs_k, runs.quarter, runs.slice_stride, NC, NC + p.rs_k - 1, cm.slot_bytes3));
        // GR8_DS_PF_DIRECT=1: producers write the pinned send slots; RS slices go flat, so the QPs take 2 send SGEs
        if (k.ds_pf_direct) {
            if (pf_rows <= MX) return "GR8_DS_PF_DIRECT=1 needs GR8_DS_PREFILL_W > MX";
            if (k.ds_subdump) return "GR8_DS_PF_DIRECT=1: GR8_DS_SUBDUMP reads Model::sc.part after the publish (not supported)";
            p.pf_direct = true;
            p.log.push_back(fmt("# rank %d DS41 prefill direct publish: GR8_DS_PF_DIRECT=1 (send slots written by the producers; RS slices flat: %s)\n",
                                cfg.rank, p.rs_k ? "yes" : "no RS"));
        }
    }
    // The two --ips2 rails alone do not stripe: Fabric::init requires stripe_min > 0.
    if (k.ds_stripe) {
        if (cfg.ips2.size() != cfg.ips.size()) return "GR8_DS_STRIPE=1 needs --ips2 with one address per rank";
        if (cfg.ring || !cfg.vcap.empty() || cfg.bulk_bytes) return "GR8_DS_STRIPE=1 needs a uniform, non-bulk mesh fabric";
        cfg.stripe_min = (size_t)k.ds_stripe_min;
    }
    // no recv slot for a rank's own contribution: DS41 reads it from the send slot
    cfg.compact = true;
#endif
    cfg.ncoll = NC;
    cfg.relay_cpu = cfg.proxy_cpu > 7 ? cfg.proxy_cpu - 8 : -1;   // --ring: the relay thread's own core (N-1..N-7 are proxies)
    // GR8_XTRACE=1: stage-0 trace on the decode fabric, or with GR8_XTRACE_PF=1 on the two prefill fabrics instead
    const bool xtrace_pf = k.xtrace_pf;
    cfg.xtrace = xtrace_pf ? 0 : 1;
    {
        g6::FabricConfig mc = cfg;
        if (cm.run1) { mc.slot_bytes = cm.slot_bytes; mc.run1 = cm.run1; mc.run2 = cm.run2; mc.slot_bytes1 = cm.slot_bytes1; mc.slot_bytes2 = cm.slot_bytes2; mc.pool = cm.pool; }
#if G6_MODEL_DS41
        if (p.rs_k) { mc.run3 = cm.run3; mc.slot_bytes3 = cm.slot_bytes3; mc.ncoll = ds41::rs::main_ncoll(NC, p.rs_k, p.rowshard); }
#endif
        mc.sge2 = p.pf_direct;
        add(FabKind::Main, 0, mc);
    }
    cfg.xtrace = 0;
#if G6_MODEL_DS41
    if (k.ds_pf_v2) {
        if (!s2mode || !ds41::v2::boot_ok(p.pf_rows,p.tail,g6nr::NR,p.pf_lanes_fab,p.l20_tail?20:21))
            return "GR8_DS_PF_V2 needs serve, TP4, W=256..8192 multiple of 256, one lane, tail128 and layer21 boundary";
        const size_t state_row=4*HID*in.trunk_bytes+(24+4+4+16)*sizeof(float)+HID*2;
        auto vc=ds41::v2::fabric_config(cfg,p.pf_rows,HID,state_row,6352);
        vc.xtrace=k.ds_pf_v2_xtrace ? 1 : 0; // Fabric::init also requires GR8_XTRACE=1
        vc.port=cfg.port+104; vc.proxy_cpu=cfg.proxy_cpu>1?cfg.proxy_cpu-2:-1;
        // Zero preserves the inherited config, including an existing DS_STRIPE.
        // The proxy already handles compact runs, RS slices and ordered tails.
        if (k.ds_pf_v2_stripe_min > 0) vc.stripe_min=size_t(k.ds_pf_v2_stripe_min);
        const int rails=vc.ring ? 3 : (vc.ips2.empty() ? 1 : 2); // v2 is TP4
        const size_t effective=vc.stripe_min && !vc.ring && rails==2
            ? std::max(vc.stripe_min,size_t(8192)) : 0;
        p.log.push_back(fmt("# ds41_v2_stripe min=%zu rails=%d effective_min=%zu\n",
                            vc.stripe_min,rails,effective));
        add(FabKind::Ds41V2,104,vc);
    }
    // GR8_DS_PF_LANES=2: lane B's own fabric, port + 96, proxy core N-2 (DS41 builds no fabp / fabp2)
    if (p.pf_lanes_fab == 2) {
        if (!s2mode || !cmB.run1) return "GR8_DS_PF_LANES=2 needs --serve with W-row prefill (GR8_DS_PREFILL_W > MX)";
        g6::FabricConfig mc = cfg;
        mc.slot_bytes = cmB.slot_bytes; mc.run1 = cmB.run1; mc.run2 = cmB.run2; mc.slot_bytes1 = cmB.slot_bytes1; mc.slot_bytes2 = cmB.slot_bytes2;
        mc.pool = cmB.pool;
        if (p.rs_k) { mc.run3 = cmB.run3; mc.slot_bytes3 = cmB.slot_bytes3; mc.ncoll = ds41::rs::main_ncoll(NC, p.rs_k, p.rowshard); }
        mc.sge2 = p.pf_direct;
        mc.port = cfg.port + 96;
        mc.proxy_cpu = cfg.proxy_cpu > 1 ? cfg.proxy_cpu - 2 : -1;
        mc.xtrace = 0;
        add(FabKind::LaneB, 96, mc);
    }
#endif
    const bool pf_rs = k.pf_rs;
    // GR8_PF_STRIPE=1 (needs --pf-ips2 or --ips2, or a three-rank --ring): prefill payload writes >= GR8_PF_STRIPE_MIN go out
    // half on each rail; at a three-rank ring the second half takes the same port's device on the other PCIe root. Unset =
    // on when --pf-ips2 is given or at a three-rank ring; 0 = one rail.
    auto pf_stripe_cfg = [&](g6::FabricConfig& cp, const std::vector<std::string>& ips2) -> std::string {
        const bool ring3 = cp.ring && cp.ips.size() == 3;
        const bool on = k.pf_stripe.value_or(!ips2.empty() || ring3);
        if (!ips2.empty()) cp.ips2 = ips2;
        if (!on) return {};
        if (!ring3 && cp.ips2.size() != cp.ips.size()) return "GR8_PF_STRIPE=1 needs --pf-ips2 (or --ips2) with one address per rank";
        cp.stripe_min = (size_t)k.pf_stripe_min;
        return {};
    };
    auto prefill = [&](FabKind kind, int port_off, int cpu_off) -> std::string {
        g6::FabricConfig cp = cfg;
        cp.slot_bytes = pf_slot_bytes(MP, pf_rs, (int)cfg.ips.size());
        cp.ncoll = pf_rs ? 5 : 2;
#if G6_MODEL_FLASH
        if (in.pf_sp) { cp.slot_bytes = pf_sp_slot_bytes(MP, (int)cfg.ips.size()); cp.ncoll = 4; }
#endif
        cp.port = cfg.port + port_off;
        if (std::string e = pf_stripe_cfg(cp, eo.pf_ips2); !e.empty()) return e;
        cp.proxy_cpu = cfg.proxy_cpu > cpu_off - 1 ? cfg.proxy_cpu - cpu_off : -1;
        cp.xtrace = xtrace_pf ? 1 : 0;
        add(kind, port_off, cp);
        return {};
    };
    if (MP)
        if (std::string e = prefill(FabKind::Prefill, 8, 1); !e.empty()) return e;
    if (eo.lanes != 1 && eo.lanes != 2) return "--pf-lanes must be 1 or 2";
    const bool two = MP && eo.lanes == 2;
    p.two = two;
    if (two)
        if (std::string e = prefill(FabKind::Prefill2, 24, 2); !e.empty()) return e;
#if G6_MODEL_FLASH
    // a one-collective fabric of a prefill lane, striped as the lanes are. Its proxy cut-off is one core higher than the
    // prefill fabrics' (> cpu_off, not > cpu_off - 1), as the inline bring-up had it: at proxy_cpu == cpu_off it is unpinned
    auto lane_fabric = [&](FabKind kind, int port_off, int cpu_off, size_t slot_bytes) -> std::string {
        g6::FabricConfig cp = cfg;
        cp.ncoll = 1; cp.slot_bytes = slot_bytes;
        cp.port = cfg.port + port_off;
        cp.proxy_cpu = cfg.proxy_cpu > cpu_off ? cfg.proxy_cpu - cpu_off : -1;
        if (std::string e = pf_stripe_cfg(cp, eo.pf_ips2); !e.empty()) return e;
        add(kind, port_off, cp);
        return {};
    };
    // the pool-ID exchange: default on for TP2 Flash prefill (GR8_GLM53F_PF_POOL_OWNER unset = auto, 0 = off, 1 = on or refuse)
    const int pool_mode = in.pool_owner_mode;
    const bool pool_fits = MP && g6nr::NR == 2 && cfg.ips.size() == 2 && !cfg.ring;
    const char* pool_conflict = k.fpf_attn_p1 ? "GR8_FPF_ATTN_P1" : k.fpf_dsa_p1 ? "GR8_FPF_DSA_P1" : k.fpf_p1 ? "GR8_FPF_P1" : nullptr;
    if (pool_mode == 1 && !pool_fits) return "GR8_GLM53F_PF_POOL_OWNER=1 needs TP2 Flash prefill (not ring)";
    if (pool_mode == 1 && pool_conflict) return std::string("GR8_GLM53F_PF_POOL_OWNER=1 conflicts with ") + pool_conflict;
    p.pool_owner = pool_mode == 1 || (pool_mode == 2 && pool_fits && !pool_conflict);
    if (p.pool_owner) {
        if (std::string e = lane_fabric(FabKind::PoolIds, 64, 9, flash_pool_owner::payload(MP)); !e.empty()) return e;
        if (two)
            if (std::string e = lane_fabric(FabKind::PoolIds2, 72, 10, flash_pool_owner::payload(MP)); !e.empty()) return e;
    }
    // GR8_PF_ROUTER_SPLIT=1: one route-table fabric per prefill lane, ports +80 / +88, proxy CPU offsets 11 / 12
    p.route_split = eo.route_split_mode == 1;
    if (p.route_split) {
        if (!(MP && g6nr::NR >= 2 && g6nr::NR <= flash_route_split::MAX_NR && cfg.ips.size() == (size_t)g6nr::NR))
            return "GR8_PF_ROUTER_SPLIT=1 needs Flash prefill (--prefill) on 2 to 8 ranks";
        const size_t b = flash_route_split::payload(MP, (int)cfg.ips.size());
        if (std::string e = lane_fabric(FabKind::RouteSplit, 80, 11, b); !e.empty()) return e;
        if (two)
            if (std::string e = lane_fabric(FabKind::RouteSplit2, 88, 12, b); !e.empty()) return e;
    }
#endif
    if (mtp) {                          // MTP draft region: 3 collectives, decode-sized slots
        g6::FabricConfig cd = cfg;
        cd.ncoll = 3;
        cd.port = cfg.port + 16;
        cd.proxy_cpu = cfg.proxy_cpu > 5 ? cfg.proxy_cpu - 6 : -1;   // own core: fabp2 (prefill lane 2) has N-2
        add(FabKind::Draft, 16, cd);
    }
#if G6_MODEL_DS41
    if (eo.dsk.on()) {                  // DSpark draft block: DRAFT_NCOLL collectives, decode-sized slots (5 rows a session)
        g6::FabricConfig cd = cfg;
        cd.ncoll = ds41::dsp::DRAFT_NCOLL;
        cd.slot_bytes = p.dec_slot_bytes;
        for (int c = 0; c < cd.ncoll; ++c)
            if (ds41::dsp::coll_bytes(c, HID, std::min(eo.s2sess, eo.ds_draft_cap)) > cd.slot_bytes)
                return "DSpark draft fabric: a collective is larger than the decode slot";
        cd.port = cfg.port + 16;        // the MTP draft region's port (--mtp is refused with DSpark)
        cd.proxy_cpu = cfg.proxy_cpu > 5 ? cfg.proxy_cpu - 6 : -1;
        cd.xtrace = k.ds_xtrace ? 1 : 0;
        add(FabKind::DSpark, 16, cd);
    }
#endif
    if (dfl) {                          // --dflash: the drafter's region, dfl::NC collectives, decode-sized slots
        g6::FabricConfig cd = cfg;
        cd.ncoll = in.dfl_ncoll;
        cd.port = cfg.port + 40;
        cd.proxy_cpu = cfg.proxy_cpu > 6 ? cfg.proxy_cpu - 7 : -1;   // own core: not N-2 (fabp2) or N-6 (fabd)
        add(FabKind::Dflash, 40, cd);
    }
#if G6_NR == 4
    if (eo.qp_roll_s) {                 // GR8_PF_QP_ROLL: ring of [lo | hi] pair half images, one-peer jobs (qp_wire.h)
        g6::FabricConfig qc = cfg;
        qc.vcap.clear(); qc.vdst_k.clear();
        qc.bulk_bytes = qw::bulk_bytes(eo.qp_wire_e);
        qc.bulk_flags = qw::bulk_flags(eo.qp_wire_e);
        qc.port = cfg.port + 56;
        qc.proxy_cpu = cfg.proxy_cpu > 4 ? cfg.proxy_cpu - 5 : -1;
        add(FabKind::QpWire, 56, qc);
    }
#endif
    // CP (GR8_CP, GR8_CP_R, GR8_CP_S3, GR8_CP_MMAX): its own fabric, cplayout.h's variable layout, port + 32
    if (in.cp_build) {
        p.cp_on = k.cp.value_or(!G6_MODEL_FLASH);   // Flash: off unless asked for
        p.cp_R = (int)k.cp_r;
        p.cp_s3 = k.cp_s3 ? 1 : 0;
        int cp_mmax = eo.K + 1;
        for (int M : eo.row_counts) cp_mmax = std::max(cp_mmax, M);
        // S2: any step may run the CP block; GR8_CP_STEP_ROWS=n caps the steps past R at n rows (cp_rows.h)
        if (s2mode) {
            const int full = std::min(MX, std::max(eo.s2sess * (mtp + 1), eo.plan.chunk));
            std::string ce;
            p.cp_step = g6s2::cp_step_rows(k.cp_step_rows, cp_mmax, full, &ce);
            if (p.cp_step < 0) return ce;
            cp_mmax = std::max(cp_mmax, p.cp_step ? p.cp_step : full);
        }
        else if (k.cp_step_rows) return "GR8_CP_STEP_ROWS needs --serve";
        if (k.cp_mmax) cp_mmax = std::max(cp_mmax, (int)*k.cp_mmax);
#if G6_MODEL_FLASH
        cp_mmax = MX;   // a CP step may carry any decode step's rows (prompt chunks, MTP and DFlash2 verifies)
        if (p.cp_on && s2mode && eo.s2sess > 1 && (p.cp_R < 64 || p.cp_R >= eo.cap))
            return "Flash GR8_CP with several sessions: GR8_CP_R must be 64..--cap - 1";
#endif
        p.cp_mmax = cp_mmax;
        if (p.cp_R < 0 || p.cp_R % 64) return "GR8_CP_R must be a multiple of 64";
        // GR8_CP_R2: a session may take a smaller R; every step past it runs the CP block (CpState::step_R)
        p.cp_R2 = (int)k.cp_r2.value_or(0);
        if (p.cp_R2 && (p.cp_R2 < 64 || p.cp_R2 % 64 || p.cp_R2 >= p.cp_R || !p.cp_on || !s2mode || !eo.rt_win))
            return "GR8_CP_R2 must be a multiple of 64 below GR8_CP_R, with GR8_CP on, --serve and GR8_RT_LOCAL";
        p.cp_step_R = p.cp_R2 ? p.cp_R2 : p.cp_R;
        p.cp_colls.make(in.full_index, (int)cfg.ips.size(), cp_mmax, in.attn);
        if (p.cp_on) {
            g6::FabricConfig cc = cfg;
            cc.vcap = p.cp_colls.cap;
            cc.vdst_k = p.cp_colls.dst_k;
            cc.port = cfg.port + 32;
            cc.proxy_cpu = cfg.proxy_cpu > 2 ? cfg.proxy_cpu - 3 : -1;
            add(FabKind::Cp, 32, cc);
        }
    }
    // the fabrics the built model completes. The gathers (GR8_QA_SPLIT_CPU moves their proxy; main and draft steps never
    // run at once, so both take N-9)
    auto gather = [&](FabKind kind, int port_off) {
        g6::FabricConfig gc = cfg;
        gc.vcap.clear(); gc.vdst_k.clear();
        gc.port = cfg.port + port_off;
        gc.proxy_cpu = (int)k.qa_split_cpu.value_or(cfg.proxy_cpu > 8 ? cfg.proxy_cpu - 9 : -1);
        add(kind, port_off, gc, true);
    };
    if (in.gather) gather(FabKind::Gather, 64);
    if (p.cp_on && MP) {                // step 4d: CP prefill staging pools, a bulk-mode fabric of 2 cp_pf_pool_bytes(m)
        g6::FabricConfig sc = cfg;
        sc.vcap.clear(); sc.vdst_k.clear();
        sc.bulk_flags = 4;
        sc.port = cfg.port + 48;
        sc.proxy_cpu = cfg.proxy_cpu > 3 ? cfg.proxy_cpu - 4 : -1;
        add(FabKind::CpStage, 48, sc, true);
    }
    if (in.draft_gather) gather(FabKind::DraftGather, 72);
    return {};
}
}   // namespace fplan

inline FabricPlan fabric_plan(const EngineOptions& eo, const PlanInputs& in)
{
    FabricPlan p;
    p.refusal = fplan::plan(eo, in, p);
    if (!p.refusal.empty()) p.fabrics.clear();
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
