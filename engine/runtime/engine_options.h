// g6m4's command line and boot knobs: parse_engine_options reads argv and the knob table, makes every parse-time check
// a boot makes, and fills EngineOptions with what main reads afterwards. The parse is engine/scheduler-rs
// src/options.rs; this is its facade, which rebuilds the C++ types from the Rust's scalars. Host only, no CUDA, so
// s2/tests/options_vectors_test.cpp runs it in each model's shape. A refusal comes back as the message main dies with;
// the helpers that die themselves (parse_ints on a bad @file, ds41_engram_colls) still exit as before.
// The steering profile (steering::prepare_boot) and the grammar (grammar_init) read files, so main loads them after this.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "engine_opts.h"
#include "rank_host.h"
#include "spec_len.h"
#include "s2_batch.h"
#include "s2/flash_checkpoint.h"
#include "../comms/fabric.h"
#include "../module_info.h"
#include "../knobs-rs/gr8_knobs.h"
#include "../scheduler-rs/g6sched.h"
#if G6_MODEL_DS41
#include "ds41_dspark.h"
#include "ds41_knobs.h"
#include "ds41_coll_order.h"
#include "ds41_engram_boot.h"
#endif

namespace g6rt
{

struct EngineOptions
{
    g6::FabricConfig cfg;               // --rank, --ips, --ips2, --ring, --ring-devs, --proxy-cpu; the rest is set after the parse
    std::vector<std::string> pf_ips2;   // --pf-ips2: rail-1 addresses of the prefill fabrics only (GR8_PF_STRIPE dual-rail)
    std::string manifest, packfile;
    V2Src v2src;
    CommonOpts co;                      // every flag m4sim takes too (m4sim parses them in engine_opts.h)
    std::string qp_man, qp_file;        // --qp-manifest/--qp-pack: this rank's pair weights, GR8_PF_OWN=4
    std::string dfl_man, dfl_file;      // --dflash-manifest/--dflash-pack: the DFlash2 drafter (engine/loader-rs/src/dflash2.rs)
    int dfl_sel = -1;                   // --dflash-sel: 1 = selector walk, 0 = argmax, -1 = the pack decides
    int dfl_k = g6mod::INFO.dfl_drafts; // --dflash-k: drafts a round verifies (1..7; the drafter always drafts 7)
#if G6_MODEL_FLASH
    bool spec_reset_per_request = false; // offline evaluation: no adaptive history from earlier requests
    bool spec_fixed = false;            // --spec-fixed: Flash --mtp K verifies all K drafts every round
#endif
    Plan plan;                          // the defaults every --prompt sequence starts from
    bool image = false, hash = true, reuse_prefix = false;
    bool check_args = false;            // --check-args: parse and refuse as a boot would, then exit 0 before any CUDA work
    bool lanes_explicit = false;        // --pf-lanes given (common_arg parses it)
    std::string s2dir;                  // --serve DIR: serve through S2 (below R)
    std::string grammar_dir;            // --grammar-dir DIR: request grammars (grammar.h); cleared in a build without XGrammar
    std::vector<std::string> kv_tiers;  // --kv-store (and the deprecated --kv-tier), in flag order
    std::string kv_keep;                // --kv-keep system|sessions|all
    int kv_system_after = 2;            // --kv-system-after
    double kv_write_limit = 0;          // --kv-write-limit F
    std::string kv_format;              // --kv-format fp8|int4: which latent row this binary was built for (refused otherwise)
    bool kv_old = false, kv_old_all = false;   // a deprecated --kv-tier, and whether one kept evicted runs
    std::string set_id;                 // --set-id HEX: joins the S2 bootstrap agreement
    std::string steering_dir, steering_force_s;
    std::string steering_model_revision, steering_tokenizer_revision, steering_template_digest, steering_weight_set, steering_template_file;
    int s2port = 0, s2sess = G6_MODEL_FLASH ? 1 : 8;
    unsigned long long s2epoch = 1;

    // the shared flags, under the names main uses
    int cap = 0, top = 0, pfrows = 0, nc = 0, mtp = 0, MP = 0;
    int lanes = 1;                      // Flash: two lanes for a prefill of up to 8192 rows unless --pf-lanes
    float temp = 0.f;
    bool graph = false, wide = false, pfv2 = false;
    bool adapt = false;                 // Flash: --mtp turns it on unless --spec-fixed
    bool release = false, s2mode = false, dfl = false;
    int K = 0;                          // drafts per round
    std::vector<int> layers;
    bool progress_isolate = false;      // other threads stay off the proxy CPUs (progress_cpus.h)
#if G6_MODEL_FLASH
    int route_split_mode = 0;           // GR8_PF_ROUTER_SPLIT
#endif
#if G6_MODEL_DS41
    int c8_pack_env = 0;                // GR8_DS_C8PACK
#endif
    SpecLen spec;                       // --spec-adaptive. Estimates persist unless the offline reset option is on
#if G6_MODEL_FLASH
    SpecLen spec_initial;               // fully configured, before any observations; includes Flash costs and levels
#endif
    int rt_win = 0;                     // GR8_RT_LOCAL
    s2::KStop kstop;                    // GR8_MTP_KSTOP / GR8_MTP_KSTOPB
    g6s2::FlashCache flash_cache = g6s2::FlashCache::Off;   // GR8_GLM53F_CACHE
    long long flash_budget_mib = 0;     // GR8_GLM53F_STATE_BUDGET_MIB
#if G6_MODEL_DS41
    int ds_moe_main_topk = 0, ds_moe_decode_topk = 0;
    int ds_idx_pipe_b = 0, ds_idx_q_trig = 0, ds_idx_pipe_qr = 0, ds_admit_keep_scratch = 0;
    ds41::dsp::Knobs dsk;               // every DSpark knob, GR8_DS_VERIFY_CAP / GR8_DS_CONF_TEMP included
    bool ds_sess_drafts = false;        // GR8_DS_SESS_DRAFTS
    int ds_draft_cap = 0;               // sessions a DSpark draft launch holds
#endif
    std::vector<Plan> plans;            // one sequence per --prompt (steering applied by main)
    std::vector<int> row_counts;        // main step row counts to prepare and capture
    std::vector<int> draft_rows;        // row counts of first-form draft passes
    int NL = 0, NX = 0, NC = 0, R = -1; // layers, Engram collectives, decode collectives, rank
    std::vector<int> fab_order;         // DS41 with Engram: the step's collective order; else empty (identity)
    bool own4 = false;                  // GR8_PF_OWN=4 with --prefill
    int qp_roll_s = 0;                  // GR8_PF_QP_ROLL: 0 or 2..4
    bool qp_roll_ser = false;           // GR8_PF_QP_ROLL_SER
    int qp_wire_e = 1;                  // GR8_PF_QP_WIRE_E
};

namespace eopt
{
// the knob table's C API, which the Rust parse reads through
inline const g6opt_knobs& knob_api()
{
    static const g6opt_knobs k{knob::TABLE_HASH, gr8_knob_flag, gr8_knob_int, gr8_knob_float, gr8_knob_text,
                               gr8_knob_ints, gr8_knob_floats, gr8_knobs_tests_set};
    return k;
}
inline std::string str(const char* s) { return s ? s : ""; }
inline std::vector<std::string> strs(const g6opt_strs& v)
{
    std::vector<std::string> r;
    for (size_t i = 0; i < v.n; ++i) r.push_back(v.p[i]);
    return r;
}
inline std::vector<int> ints(const g6sch_ints& v) { return std::vector<int>(v.p, v.p + v.n); }
inline Plan plan(const g6opt_plan& c)
{
    Plan p;
    p.prompt = ints(c.prompt); p.first = c.first; p.chunk = c.chunk; p.n_gen = c.n_gen; p.inv_temp = c.inv_temp; p.seed = c.seed;
    p.top_p = c.top_p; p.lp = c.lp; p.plp = c.plp; p.rs = c.rs;
    return p;
}
#if G6_MODEL_DS41
inline int32_t engram_colls(void*, const int32_t* ids, size_t n) { return ds41_engram_colls(std::vector<int>(ids, ids + n)); }
#endif

inline void fill(const g6opt_options& c, EngineOptions& o)
{
    g6::FabricConfig& f = o.cfg;
    f.rank = c.rank; f.ips = strs(c.ips); f.ips2 = strs(c.ips2); f.ring = c.ring; f.ring_devs = strs(c.ring_devs);
    f.proxy_cpu = c.proxy_cpu;
    o.pf_ips2 = strs(c.pf_ips2);
    o.manifest = str(c.manifest); o.packfile = str(c.packfile);
    CommonOpts& co = o.co;
    co.v2_dir = str(c.v2_dir); co.v2_slices = str(c.v2_slices); co.v2_widths = str(c.v2_widths); co.v2_dense = str(c.v2_dense);
    co.v2_layout = str(c.v2_layout); co.v2_trellis = c.v2_trellis; co.qo_man = str(c.qo_man); co.qo_file = str(c.qo_file);
    co.layers_s = str(c.layers_s); co.levels_s = str(c.levels_s); co.prompts = strs(c.prompts);
    co.cap = c.co_cap; co.MP = c.co_mp; co.top = c.co_top; co.pfrows = c.co_pfrows; co.mtp = c.co_mtp; co.lanes = c.co_lanes;
    co.nc = c.co_nc; co.temp = c.co_temp; co.graph = c.co_graph; co.wide = c.co_wide; co.pfv2 = c.co_pfv2; co.adapt = c.co_adapt;
    if (c.has_n_gen) co.n_gen = c.n_gen;
    if (c.has_chunk) co.chunk = c.chunk;
    if (c.has_seed) co.seed = c.seed;
    if (c.has_top_p) co.top_p = c.top_p;
    if (c.has_rs) co.rs = c.rs;
    V2Src& v = o.v2src;
    v.dir = co.v2_dir; v.slices = co.v2_slices; v.widths = co.v2_widths; v.dense = co.v2_dense; v.trellis = co.v2_trellis;
    v.layout = co.v2_layout;
    o.qp_man = str(c.qp_man); o.qp_file = str(c.qp_file); o.dfl_man = str(c.dfl_man); o.dfl_file = str(c.dfl_file);
    o.dfl_sel = c.dfl_sel; o.dfl_k = c.dfl_k;
#if G6_MODEL_FLASH
    o.spec_reset_per_request = c.spec_reset_per_request; o.spec_fixed = c.spec_fixed;
#endif
    o.plan = plan(c.plan);
    o.image = c.image; o.hash = c.hash; o.reuse_prefix = c.reuse_prefix; o.check_args = c.check_args;
    o.lanes_explicit = c.lanes_explicit; o.s2dir = str(c.s2dir); o.grammar_dir = str(c.grammar_dir);
    o.kv_tiers = strs(c.kv_tiers); o.kv_keep = str(c.kv_keep); o.kv_system_after = c.kv_system_after;
    o.kv_write_limit = c.kv_write_limit; o.kv_format = str(c.kv_format); o.kv_old = c.kv_old; o.kv_old_all = c.kv_old_all;
    o.set_id = str(c.set_id); o.steering_dir = str(c.steering_dir); o.steering_force_s = str(c.steering_force_s);
    o.steering_model_revision = str(c.steering_model_revision);
    o.steering_tokenizer_revision = str(c.steering_tokenizer_revision);
    o.steering_template_digest = str(c.steering_template_digest); o.steering_weight_set = str(c.steering_weight_set);
    o.steering_template_file = str(c.steering_template_file);
    o.s2port = c.s2port; o.s2sess = c.s2sess; o.s2epoch = c.s2epoch;
    o.cap = c.cap; o.top = c.top; o.pfrows = c.pfrows; o.nc = c.nc; o.mtp = c.mtp; o.MP = c.mp; o.lanes = c.lanes;
    o.temp = c.temp; o.graph = c.graph; o.wide = c.wide; o.pfv2 = c.pfv2; o.adapt = c.adapt;
    o.release = c.release; o.s2mode = c.s2mode; o.dfl = c.dfl; o.K = c.k; o.layers = ints(c.layers);
    o.progress_isolate = c.progress_isolate;
#if G6_MODEL_FLASH
    o.route_split_mode = c.route_split_mode;
#endif
#if G6_MODEL_DS41
    o.c8_pack_env = c.c8_pack_env;
#endif
    SpecLen& s = o.spec;
    s.on = c.spec_on; s.levels = ints(c.spec_levels); s.draft_round = c.spec_draft_round; s.draft_each = c.spec_draft_each;
    std::copy(c.spec_table, c.spec_table + SpecLen::KX + 2, s.table);
#if G6_MODEL_FLASH
    o.spec_initial = s;
#endif
    o.rt_win = c.rt_win;
    s2::KStop& k = o.kstop;
    k.mode = c.kstop_mode; k.kmax = c.kstop_kmax; std::copy(c.kstop_tau, c.kstop_tau + 4, k.tau); k.seed3 = c.kstop_seed3;
    k.log = c.kstop_log; k.solo = c.kstop_solo;
    k.b.mode = c.kstopb_mode; k.b.kmin = c.kstopb_kmin;
    std::copy(c.kstopb_tau, c.kstopb_tau + std::min(c.kstopb_tau_n, sizeof k.b.tau / sizeof(double)), &k.b.tau[0][0]);
    k.b.widths = ints(c.kstopb_widths); k.b.explore = c.kstopb_explore;
    o.flash_cache = g6s2::FlashCache(c.flash_cache);
    o.flash_budget_mib = c.flash_budget_mib;
#if G6_MODEL_DS41
    o.ds_moe_main_topk = c.ds_moe_main_topk; o.ds_moe_decode_topk = c.ds_moe_decode_topk;
    o.ds_idx_pipe_b = c.ds_idx_pipe_b; o.ds_idx_q_trig = c.ds_idx_q_trig; o.ds_idx_pipe_qr = c.ds_idx_pipe_qr;
    o.ds_admit_keep_scratch = c.ds_admit_keep_scratch;
    o.ds_sess_drafts = c.ds_sess_drafts; o.ds_draft_cap = c.ds_draft_cap;
#endif
    o.plans.clear();
    for (size_t i = 0; i < c.n_plans; ++i) o.plans.push_back(plan(c.plans[i]));
    o.row_counts = ints(c.row_counts); o.draft_rows = ints(c.draft_rows);
    o.NL = c.nl; o.NX = c.nx; o.NC = c.ncoll; o.R = c.r;
    o.own4 = c.own4; o.qp_roll_s = c.qp_roll_s; o.qp_roll_ser = c.qp_roll_ser; o.qp_wire_e = c.qp_wire_e;
}
}   // namespace eopt

// Empty: the options pass; else the message main dies with.
inline std::string parse_engine_options(int argc, char** argv, EngineOptions& o)
{
#if !G6_MODEL_DS41
    static_assert(geom::TOPK == 8, "GR8_MOE_TOPK's default and range (knobs/decode.yaml) are the model's top-k");
#endif
    if (g6sch_layout() != g6sch_layout_cpp()) return "scheduler-rs: struct layouts differ from g6sched.h (rebuild libg6host.a)";
    g6opt_host h{};
#if G6_MODEL_DS41
    const ds41::dsp::Knobs dsk = ds41::dsp::parse_knobs(ds41_dspark_knob);
    h.engram_colls = eopt::engram_colls;
    h.dsk_mtp = dsk.mtp; h.dsk_relaxed = dsk.relaxed; h.dsk_relax_topp = dsk.relax_topp; h.dsk_err = dsk.err.c_str();
#endif
    const std::unique_ptr<g6opt_parsed, void (*)(g6opt_parsed*)> p(g6opt_parse(&g6mod::INFO, &eopt::knob_api(), &h, argc, argv),
                                                                    g6opt_free);
    const std::string refusal = g6opt_refusal(p.get());
    if (!refusal.empty()) return refusal;
    eopt::fill(*g6opt_options_of(p.get()), o);
#if G6_MODEL_DS41
    o.dsk = dsk;
    o.fab_order = ds41::coll_exec_order(o.layers, o.NX);   // the step's collective order with Engram on, else empty
#endif
    return {};
}

// Every field, one per line, for the vectors test.
inline std::string describe(const EngineOptions& o)
{
    std::string s;
    char b[64];
    auto num = [&](double v) { snprintf(b, sizeof b, "%.17g", v); return std::string(b); };
    auto line = [&](const char* k, const std::string& v) { s += k; s += ' '; s += v; s += '\n'; };
    auto i = [&](const char* k, long long v) { line(k, std::to_string(v)); };
    auto d = [&](const char* k, double v) { line(k, num(v)); };
    auto str = [&](const char* k, const std::string& v) { line(k, "[" + v + "]"); };
    auto strs = [&](const char* k, const std::vector<std::string>& v) {
        std::string t;
        for (const std::string& x : v) t += "[" + x + "]";
        line(k, t);
    };
    auto ints = [&](const char* k, const std::vector<int>& v) {
        std::string t;
        for (size_t j = 0; j < v.size(); ++j) t += (j ? "," : "") + std::to_string(v[j]);
        line(k, t);
    };
    auto dbls = [&](const char* k, const double* v, int n) {
        std::string t;
        for (int j = 0; j < n; ++j) t += (j ? "," : "") + num(v[j]);
        line(k, t);
    };
    auto bytes = [&](const char* k, const uint8_t* v, size_t n) {
        std::string t;
        for (size_t j = 0; j < n; ++j) { snprintf(b, sizeof b, "%02x", v[j]); t += b; }
        line(k, t);
    };
    auto plan = [&](const std::string& p, const Plan& x) {
        ints((p + ".prompt").c_str(), x.prompt);
        i((p + ".first").c_str(), x.first); i((p + ".chunk").c_str(), x.chunk); i((p + ".n_gen").c_str(), x.n_gen);
        d((p + ".inv_temp").c_str(), x.inv_temp); i((p + ".seed").c_str(), x.seed); d((p + ".top_p").c_str(), x.top_p);
        i((p + ".lp").c_str(), x.lp); i((p + ".plp").c_str(), x.plp); i((p + ".rs").c_str(), x.rs);
        d((p + ".steering_scale").c_str(), x.steering_scale); i((p + ".steering_slot").c_str(), x.steering_slot);
        const steering::Identity& y = x.steering;
        i((p + ".steering.enabled").c_str(), y.enabled);
        bytes((p + ".steering.profile_digest").c_str(), y.profile_digest.data(), y.profile_digest.size());
        i((p + ".steering.scope").c_str(), (int)y.scope);
        bytes((p + ".steering.schedule_digest").c_str(), y.schedule_digest.data(), y.schedule_digest.size());
        i((p + ".steering.scale_bits").c_str(), y.scale_bits); i((p + ".steering.slot").c_str(), y.slot);
        i((p + ".steering.images").c_str(), y.images);
        bytes((p + ".steering.image_digest").c_str(), y.image_digest.data(), y.image_digest.size());
    };
    auto spec = [&](const std::string& p, const SpecLen& x) {
        i((p + ".on").c_str(), x.on); ints((p + ".levels").c_str(), x.levels);
        d((p + ".draft_round").c_str(), x.draft_round); d((p + ".draft_each").c_str(), x.draft_each);
        dbls((p + ".table").c_str(), x.table, SpecLen::KX + 2); dbls((p + ".seed").c_str(), x.seed, SpecLen::KX + 1);
        dbls((p + ".pn").c_str(), x.pn, SpecLen::KX + 1); dbls((p + ".pk").c_str(), x.pk, SpecLen::KX + 1);
        dbls((p + ".bn").c_str(), &x.bn[0][0], (SpecLen::KX + 1) * SpecLen::NB);
        dbls((p + ".bk").c_str(), &x.bk[0][0], (SpecLen::KX + 1) * SpecLen::NB);
        d((p + ".pos_decay").c_str(), x.pos_decay); d((p + ".bucket_decay").c_str(), x.bucket_decay); d((p + ".prior").c_str(), x.prior);
        i((p + ".rounds").c_str(), x.rounds);
        std::vector<int> ch(x.chosen, x.chosen + SpecLen::KX + 1);
        ints((p + ".chosen").c_str(), ch);
        d((p + ".pred_sum").c_str(), x.pred_sum); d((p + ".got_sum").c_str(), x.got_sum);
    };
    const g6::FabricConfig& c = o.cfg;
    i("cfg.rank", c.rank); strs("cfg.ips", c.ips); strs("cfg.ips2", c.ips2); i("cfg.stripe_min", (long long)c.stripe_min);
    i("cfg.slot_bytes", (long long)c.slot_bytes); i("cfg.run1", c.run1); i("cfg.run2", c.run2);
    i("cfg.slot_bytes1", (long long)c.slot_bytes1); i("cfg.slot_bytes2", (long long)c.slot_bytes2); i("cfg.run3", c.run3);
    i("cfg.slot_bytes3", (long long)c.slot_bytes3); i("cfg.pool", c.pool); i("cfg.sge2", c.sge2); i("cfg.compact", c.compact);
    i("cfg.ncoll", c.ncoll); i("cfg.port", c.port); i("cfg.proxy_cpu", c.proxy_cpu);
    std::vector<int> vcap(c.vcap.begin(), c.vcap.end()), vdst(c.vdst_k.begin(), c.vdst_k.end());
    ints("cfg.vcap", vcap); ints("cfg.vdst_k", vdst); i("cfg.bulk_bytes", (long long)c.bulk_bytes); i("cfg.bulk_flags", c.bulk_flags);
    i("cfg.xtrace", c.xtrace); i("cfg.ring", c.ring); strs("cfg.ring_devs", c.ring_devs); i("cfg.relay_cpu", c.relay_cpu);
    strs("pf_ips2", o.pf_ips2); str("manifest", o.manifest); str("packfile", o.packfile);
    str("v2src.dir", o.v2src.dir); str("v2src.slices", o.v2src.slices); str("v2src.widths", o.v2src.widths);
    str("v2src.dense", o.v2src.dense); str("v2src.layout", o.v2src.layout); i("v2src.trellis", o.v2src.trellis); i("v2src.qo", o.v2src.qo);
    const CommonOpts& co = o.co;
    str("co.v2_dir", co.v2_dir); str("co.v2_slices", co.v2_slices); str("co.v2_widths", co.v2_widths); str("co.v2_dense", co.v2_dense);
    str("co.v2_layout", co.v2_layout); i("co.v2_trellis", co.v2_trellis); str("co.qo_man", co.qo_man); str("co.qo_file", co.qo_file);
    str("co.layers_s", co.layers_s); str("co.levels_s", co.levels_s); strs("co.prompts", co.prompts);
    i("co.cap", co.cap); i("co.MP", co.MP); i("co.top", co.top); i("co.pfrows", co.pfrows); i("co.mtp", co.mtp);
    i("co.lanes", co.lanes); i("co.nc", co.nc); d("co.temp", co.temp);
    i("co.graph", co.graph); i("co.wide", co.wide); i("co.pfv2", co.pfv2); i("co.adapt", co.adapt);
    str("co.n_gen", co.n_gen ? std::to_string(*co.n_gen) : "-"); str("co.chunk", co.chunk ? std::to_string(*co.chunk) : "-");
    str("co.seed", co.seed ? std::to_string(*co.seed) : "-"); str("co.top_p", co.top_p ? num(*co.top_p) : "-");
    str("co.rs", co.rs ? std::to_string(*co.rs) : "-");
    str("qp_man", o.qp_man); str("qp_file", o.qp_file); str("dfl_man", o.dfl_man); str("dfl_file", o.dfl_file);
    i("dfl_sel", o.dfl_sel); i("dfl_k", o.dfl_k);
#if G6_MODEL_FLASH
    i("spec_reset_per_request", o.spec_reset_per_request); i("spec_fixed", o.spec_fixed);
#endif
    plan("plan", o.plan);
    i("image", o.image); i("hash", o.hash); i("reuse_prefix", o.reuse_prefix); i("check_args", o.check_args);
    i("lanes_explicit", o.lanes_explicit); str("s2dir", o.s2dir); str("grammar_dir", o.grammar_dir);
    strs("kv_tiers", o.kv_tiers); str("kv_keep", o.kv_keep); i("kv_system_after", o.kv_system_after);
    d("kv_write_limit", o.kv_write_limit); str("kv_format", o.kv_format); i("kv_old", o.kv_old); i("kv_old_all", o.kv_old_all);
    str("set_id", o.set_id); str("steering_dir", o.steering_dir); str("steering_force_s", o.steering_force_s);
    str("steering_model_revision", o.steering_model_revision); str("steering_tokenizer_revision", o.steering_tokenizer_revision);
    str("steering_template_digest", o.steering_template_digest); str("steering_weight_set", o.steering_weight_set);
    str("steering_template_file", o.steering_template_file);
    i("s2port", o.s2port); i("s2sess", o.s2sess); i("s2epoch", (long long)o.s2epoch);
    i("cap", o.cap); i("top", o.top); i("pfrows", o.pfrows); i("nc", o.nc); i("mtp", o.mtp); i("MP", o.MP); i("lanes", o.lanes);
    d("temp", o.temp); i("graph", o.graph); i("wide", o.wide); i("pfv2", o.pfv2); i("adapt", o.adapt);
    i("release", o.release); i("s2mode", o.s2mode); i("dfl", o.dfl); i("K", o.K); ints("layers", o.layers);
    i("progress_isolate", o.progress_isolate);
#if G6_MODEL_FLASH
    i("route_split_mode", o.route_split_mode);
#endif
#if G6_MODEL_DS41
    i("c8_pack_env", o.c8_pack_env);
#endif
    spec("spec", o.spec);
#if G6_MODEL_FLASH
    spec("spec_initial", o.spec_initial);
#endif
    i("rt_win", o.rt_win);
    const s2::KStop& k = o.kstop;
    i("kstop.mode", k.mode); i("kstop.kmax", k.kmax); dbls("kstop.tau", k.tau, 4); d("kstop.seed3", k.seed3);
    i("kstop.log", k.log); i("kstop.solo", k.solo); i("kstop.b.mode", k.b.mode); i("kstop.b.kmin", k.b.kmin);
    dbls("kstop.b.tau", &k.b.tau[0][0], (s2::S2N_SLOTS + 1) * 8); ints("kstop.b.widths", k.b.widths); i("kstop.b.explore", k.b.explore);
    str("flash_cache", g6s2::flash_cache_name(o.flash_cache)); i("flash_budget_mib", o.flash_budget_mib);
#if G6_MODEL_DS41
    i("ds_moe_main_topk", o.ds_moe_main_topk); i("ds_moe_decode_topk", o.ds_moe_decode_topk);
    i("ds_idx_pipe_b", o.ds_idx_pipe_b); i("ds_idx_q_trig", o.ds_idx_q_trig); i("ds_idx_pipe_qr", o.ds_idx_pipe_qr);
    i("ds_admit_keep_scratch", o.ds_admit_keep_scratch);
    i("dsk.mtp", o.dsk.mtp); i("dsk.block", o.dsk.block); d("dsk.verify_cap", o.dsk.verify_cap); i("dsk.relaxed", o.dsk.relaxed);
    d("dsk.relax_ratio", o.dsk.relax_ratio); d("dsk.relax_topp", o.dsk.relax_topp); i("dsk.prefill_keep", o.dsk.prefill_keep);
    dbls("dsk.conf_temp", o.dsk.conf_temp.data(), (int)o.dsk.conf_temp.size()); i("dsk.engram", o.dsk.engram); str("dsk.err", o.dsk.err);
    i("ds_sess_drafts", o.ds_sess_drafts); i("ds_draft_cap", o.ds_draft_cap);
#endif
    i("plans", (long long)o.plans.size());
    for (size_t j = 0; j < o.plans.size(); ++j) plan("plans." + std::to_string(j), o.plans[j]);
    ints("row_counts", o.row_counts); ints("draft_rows", o.draft_rows);
    i("NL", o.NL); i("NX", o.NX); i("NC", o.NC); i("R", o.R); ints("fab_order", o.fab_order);
    i("own4", o.own4); i("qp_roll_s", o.qp_roll_s); i("qp_roll_ser", o.qp_roll_ser); i("qp_wire_e", o.qp_wire_e);
    return s;
}

}   // namespace g6rt
