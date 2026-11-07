// g6m4's command line and boot knobs: parse_engine_options reads argv and the knob table, makes every parse-time check
// a boot makes, and fills EngineOptions with what main reads afterwards. Host only, no CUDA, so
// s2/tests/options_vectors_test.cpp runs it in each model's shape. A refusal comes back as the message main dies with;
// the helpers that die themselves (parse_ints on a bad @file, ds41_engram_colls) still exit as before.
// The steering profile (steering::prepare_boot) and the grammar (grammar_init) read files, so main loads them after this.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "engine_opts.h"
#include "rank_host.h"
#include "spec_len.h"
#include "s2_batch.h"
#include "qp_wire.h"
#include "flash_route_split.h"
#include "s2/flash_checkpoint.h"
#include "../comms/fabric.h"
#include "../module_info.h"
#include "../knobs-rs/gr8_knobs.h"
#if G6_MODEL_DS41
#include "ds41_moe_topk.h"
#include "ds41_dspark.h"
#include "ds41_knobs.h"
#include "ds41_coll_order.h"
#include "ds41_engram_boot.h"
#include "s2/ds41_sess_drafts.h"
#include "s2/ds41_dspark_round_logic.h"
#endif

namespace g6rt
{

// device-header constants the checks read; g6m4.cu asserts each against its source
namespace eopt
{
constexpr int DFL_K = 7;                          // dflash.cuh dfl::K
constexpr bool DFL_REP = 8 % g6nr::NR != 0;       // dflash.cuh dfl::REP (NKV = 8)
constexpr int TP_SORT = 2048;                     // glue.cuh glue::TP_SORT
#if G6_MODEL_DS41
constexpr int DS_MARKOV_MAXS = 8;                 // ds41_dspark_api.h ::ds41dsp::MARKOV_MAXS
#endif
}

struct EngineOptions
{
    g6::FabricConfig cfg;               // --rank, --ips, --ips2, --ring, --ring-devs, --proxy-cpu; the rest is set after the parse
    std::vector<std::string> pf_ips2;   // --pf-ips2: rail-1 addresses of the prefill fabrics only (GR8_PF_STRIPE dual-rail)
    std::string manifest, packfile;
    V2Src v2src;
    CommonOpts co;                      // every flag m4sim takes too, parsed in engine_opts.h
    std::string qp_man, qp_file;        // --qp-manifest/--qp-pack: this rank's pair weights, GR8_PF_OWN=4
    std::string dfl_man, dfl_file;      // --dflash-manifest/--dflash-pack: the DFlash2 drafter (engine/loader-rs/src/dflash2.rs)
    int dfl_sel = -1;                   // --dflash-sel: 1 = selector walk, 0 = argmax, -1 = the pack decides
    int dfl_k = eopt::DFL_K;            // --dflash-k: drafts a round verifies (1..7; the drafter always drafts 7)
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
struct Refusal { std::string msg; };
[[noreturn]] inline void refuse(const std::string& m) { throw Refusal{m}; }
inline std::vector<std::string> split(const std::string& s, char c)
{
    std::vector<std::string> v;
    size_t a = 0, b;
    while ((b = s.find(c, a)) != std::string::npos) { v.push_back(s.substr(a, b - a)); a = b + 1; }
    if (a < s.size()) v.push_back(s.substr(a));
    return v;
}

inline void parse(int argc, char** argv, EngineOptions& o)
{
    // GR8_MOE_TOPK (model change, screen arm): unset = the model's top-k (8, the table's default)
#if !G6_MODEL_DS41
    static_assert(geom::TOPK == 8, "GR8_MOE_TOPK's default and range (knobs/decode.yaml) are the model's top-k");
#endif
#if G6_MODEL_DS41
    o.c8_pack_env = knob::ds_c8pack() ? 1 : 0;
    const int c8_pack_env = o.c8_pack_env;
#endif
#if G6_MODEL_FLASH
    if (const int tk = (int)knob::moe_topk(); tk < geom::TOPK)
        fprintf(stderr, "# GR8_MOE_TOPK=%d: routed experts per token cut %d -> %d, weights renormalized (screen arm)\n",
                tk, geom::TOPK, tk);
#elif G6_MODEL_DS41
    if (knob::moe_topk() != 8) refuse("GR8_MOE_TOPK is a Flash-model option: DeepSeek-V4.1 reads GR8_DS_MOE_TOPK");
    if (knob::pf_router_split()) refuse("GR8_PF_ROUTER_SPLIT is a Flash-model option");
#else
    if (knob::moe_topk() != geom::TOPK) refuse("GR8_MOE_TOPK is a Flash-model option (this build is the GLM model)");
    if (knob::pf_router_split()) refuse("GR8_PF_ROUTER_SPLIT is a Flash-model option (this build is the GLM model)");
#endif
#if G6_MODEL_FLASH
    o.route_split_mode = flash_route_split_mode();
#endif
    o.progress_isolate = knob::progress_isolate();
    g6::FabricConfig& cfg = o.cfg;
    std::vector<std::string>& pf_ips2 = o.pf_ips2;
    std::string &manifest = o.manifest, &packfile = o.packfile;
    V2Src& v2src = o.v2src;
    CommonOpts& co = o.co;
    std::string &qp_man = o.qp_man, &qp_file = o.qp_file, &dfl_man = o.dfl_man, &dfl_file = o.dfl_file;
    int &dfl_sel = o.dfl_sel, &dfl_k = o.dfl_k;
#if G6_MODEL_FLASH
    bool &spec_reset_per_request = o.spec_reset_per_request, &spec_fixed = o.spec_fixed;
#endif
    Plan& plan = o.plan;
    bool &image = o.image, &hash = o.hash, &reuse_prefix = o.reuse_prefix, &check_args = o.check_args;
    bool& lanes_explicit = o.lanes_explicit;
    std::string &s2dir = o.s2dir, &grammar_dir = o.grammar_dir;
    std::vector<std::string>& kv_tiers = o.kv_tiers;
    std::string& kv_keep = o.kv_keep;
    int& kv_system_after = o.kv_system_after;
    double& kv_write_limit = o.kv_write_limit;
    std::string& kv_format = o.kv_format;
    bool &kv_old = o.kv_old, &kv_old_all = o.kv_old_all;
    std::string& set_id = o.set_id;
    std::string &steering_dir = o.steering_dir, &steering_force_s = o.steering_force_s;
    std::string &steering_model_revision = o.steering_model_revision, &steering_tokenizer_revision = o.steering_tokenizer_revision,
                &steering_template_digest = o.steering_template_digest, &steering_weight_set = o.steering_weight_set,
                &steering_template_file = o.steering_template_file;
    int &s2port = o.s2port, &s2sess = o.s2sess;
    unsigned long long& s2epoch = o.s2epoch;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) refuse("missing value for " + a); return std::string(argv[++i]); };
        std::string oerr;
        if (a == "--pf-lanes") lanes_explicit = true;
        if (g6rt::common_arg(a, next, co, oerr)) { if (!oerr.empty()) refuse(oerr); continue; }
        if (a == "--rank") cfg.rank = atoi(next().c_str());
        else if (a == "--ips") cfg.ips = split(next(), ',');
        else if (a == "--ips2") cfg.ips2 = split(next(), ',');
        else if (a == "--pf-ips2") pf_ips2 = split(next(), ',');
        else if (a == "--ring") cfg.ring = true;
        else if (a == "--ring-devs") cfg.ring_devs = split(next(), ',');
        else if (a == "--proxy-cpu") cfg.proxy_cpu = atoi(next().c_str());
        else if (a == "--manifest") manifest = next();
        else if (a == "--pack") packfile = next();
        else if (a == "--qp-manifest") qp_man = next();
        else if (a == "--qp-pack") qp_file = next();
        else if (a == "--dflash-manifest") dfl_man = next();
        else if (a == "--dflash-pack") dfl_file = next();
        else if (a == "--dflash-sel") dfl_sel = atoi(next().c_str());
        else if (a == "--dflash-k") dfl_k = atoi(next().c_str());
#if G6_MODEL_FLASH
        else if (a == "--spec-fixed") spec_fixed = true;
        else if (a == "--spec-reset-per-request") spec_reset_per_request = true;
#endif
        else if (a == "--image") image = true;
        else if (a == "--no-hash") hash = false;
        else if (a == "--reuse-prefix") reuse_prefix = true;
        else if (a == "--check-args") check_args = true;
        else if (a == "--kv-store") kv_tiers.push_back(next());
        else if (a == "--kv-keep") kv_keep = next();
        else if (a == "--kv-system-after") kv_system_after = atoi(next().c_str());
        else if (a == "--kv-write-limit") kv_write_limit = atof(next().c_str());
        else if (a == "--kv-format") kv_format = next();
        else if (a == "--kv-tier")
        {
            const std::string v = next();
            kv_old = true;
            kv_old_all = kv_old_all || v.find("keep=evicted") != std::string::npos || v.find("keep=all") != std::string::npos ||
                         v.find("keep=") == std::string::npos;
            kv_tiers.push_back(v);
        }
        else if (a == "--kv-shared-after") { kv_old = true; kv_system_after = atoi(next().c_str()); }
        else if (a == "--serve") s2dir = next();
        else if (a == "--grammar-dir") grammar_dir = next();
        else if (a == "--s2-port") s2port = atoi(next().c_str());
        else if (a == "--s2-sessions") s2sess = atoi(next().c_str());
        else if (a == "--s2-epoch") s2epoch = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--set-id")
        {
            set_id = next();
            if (set_id.empty() || set_id.size() > 64 || set_id.find_first_not_of("0123456789abcdef") != std::string::npos)
                refuse("--set-id must be 1..64 lowercase hex digits");
        }
        else if (a == "--steering-profile") steering_dir = next();
        else if (a == "--steering-weight-set") steering_weight_set = next();
        else if (a == "--steering-force") steering_force_s = next();
        else if (a == "--steering-model-revision") steering_model_revision = next();
        else if (a == "--steering-tokenizer-revision") steering_tokenizer_revision = next();
        else if (a == "--steering-chat-template") steering_template_file = next();
        else if (a == "--steering-chat-template-sha256") steering_template_digest = next();
        else refuse("unknown argument " + a);
    }
    o.cap = co.cap; o.top = co.top; o.pfrows = co.pfrows; o.nc = co.nc; o.mtp = co.mtp; o.MP = co.MP;
    o.lanes = co.lanes;
    o.temp = co.temp;
    o.graph = co.graph; o.wide = co.wide; o.pfv2 = co.pfv2;
    o.adapt = co.adapt;
    const int cap = o.cap, pfrows = o.pfrows, nc = o.nc, mtp = o.mtp, MP = o.MP;
    const float temp = o.temp;
    const bool pfv2 = o.pfv2;
    bool& adapt = o.adapt;
    const std::string &layers_s = co.layers_s, &levels_s = co.levels_s;
    std::vector<std::string>& prompts = co.prompts;
    v2src.dir = co.v2_dir; v2src.slices = co.v2_slices; v2src.widths = co.v2_widths; v2src.dense = co.v2_dense;
    v2src.trellis = co.v2_trellis; v2src.layout = co.v2_layout;
    if (co.n_gen) plan.n_gen = *co.n_gen;          // Plan owns these defaults; apply only what the command line set
    if (co.seed) plan.seed = *co.seed;
    if (co.chunk) plan.chunk = *co.chunk;
    if (co.top_p) plan.top_p = *co.top_p;
    if (co.rs) plan.rs = *co.rs;
    const bool release = o.release = !v2src.dense.empty();
    if (knob::kvb() == "fp8" && !release) refuse("GR8_KVB=fp8 needs --v2-dense: packs hold kv_b in BF16");   // #50: read by the release planner only
    if (cfg.rank < 0 || cfg.ips.size() != (size_t)g6nr::NR || (!release && (manifest.empty() || packfile.empty())))
        refuse("usage: g6m4 --rank R --ips NR,addresses (this build: NR = " + std::to_string(g6nr::NR) + ") [--ips2 ..] --manifest M --pack P|- [--qo-manifest QM --qo-pack QP] [--qp-manifest PM --qp-pack PP] [--layers ..] [--gen N] [--temp T]");
#if G6_MODEL_DS41
    if (release && (!manifest.empty() || !packfile.empty() || !image))
        refuse("--v2-dense CHECKPOINT_DIR replaces --manifest / --pack and needs --image (DS41: no --v2 / --v2-slices / --v2-widths)");
#else
    if (release && (!manifest.empty() || !packfile.empty() || v2src.dir.empty()))
        refuse("--v2-dense replaces --manifest / --pack and needs --v2 (the image comes from the release alone)");
#endif
    if (!release && !v2src.layout.empty()) refuse("--layout needs --v2-dense (it places the release's pieces)");
    if (knob::dec_shared_scratch() && !pfv2) refuse("GR8_DEC_SHARED_SCRATCH=1 needs --pfv2");
    o.layers = parse_ints(layers_s);
    const std::vector<int>& layers = o.layers;
#if G6_MODEL_FLASH
    if (layers_s == "all") {
        if (layers.size() != (size_t)geom::L_MAIN) refuse("Flash --layers all did not select every main layer");
        for (int i = 0; i < geom::L_MAIN; ++i)
            if (layers[i] != i) refuse("Flash --layers all selected an unexpected layer");
    }
    fprintf(stderr, "# rank %d: Flash actual main layers %zu first %d last %d\n", cfg.rank, layers.size(),
            layers.empty() ? -1 : layers.front(), layers.empty() ? -1 : layers.back());
#endif
    if (prompts.empty()) prompts.push_back(g6rt::DEFAULT_PROMPT);
    plan.inv_temp = g6rt::inv_temp(temp);
    if (const std::string e = g6rt::common_check(co, g6mx::MX, plan.chunk); !e.empty()) refuse(e);
    if (nc < 0 || nc * g6nr::NR > TP_SORT) refuse("--top-p-cands must be 0.." + std::to_string(TP_SORT / g6nr::NR));
    // Flash lane default: the MP8192 memory gate passed; larger MPs remain opt-in.
#if G6_MODEL_FLASH
    if (!lanes_explicit && MP > 0 && MP <= 8192) o.lanes = 2;
#endif
    const bool s2mode = o.s2mode = !s2dir.empty();
    const bool dfl = o.dfl = !dfl_man.empty();
    if (s2mode && (knob::resid_dump() || knob::logit_dump()))
        refuse("GR8_RESID_DUMP / GR8_LOGIT_DUMP support offline main steps, not --serve");
    // S2 keeps --mtp next to the drafter: a session alone at its prompt tail runs DFlash2, the others MTP (s2_batch.h)
    if (dfl && (dfl_file.empty() || (mtp && !s2mode) || dfl_sel < -1 || dfl_sel > 1 || dfl_k < 1 || dfl_k > DFL_K))
        refuse("--dflash-manifest needs --dflash-pack, and replaces --mtp (7 drafts per round) except under S2; --dflash-sel 0|1; --dflash-k 1..7");
    // TP=6 runs the drafter body replicated (dflash.cuh dfl::REP), and so does Flash at TP=3; GLM-5.3's TP=3 would too, but is not measured
    if (dfl && DFL_REP && g6nr::NR != 6 && !(G6_MODEL_FLASH && g6nr::NR == 3))
        refuse("--dflash: TP=" + std::to_string(g6nr::NR) + " has no DFlash2 drafter (TP=4 splits it, TP=6 and Flash's TP=3 replicate it)");
    if (plan.rs && !mtp && !dfl) refuse("--spec-accept rejection needs --mtp or --dflash-manifest");
    o.K = dfl ? DFL_K : mtp;
    const int K = o.K;
#if G6_MODEL_FLASH
    if (spec_reset_per_request && s2mode) refuse("--spec-reset-per-request is offline only (no --serve)");
    // Flash: --mtp K is adaptive speculation (up to K drafts) unless --spec-fixed
    if (spec_fixed && (adapt || !levels_s.empty() || !mtp)) refuse("--spec-fixed needs --mtp, and not --spec-adaptive / --spec-levels");
    if (mtp && !spec_fixed && !s2mode && !dfl) adapt = true;
#endif
    if ((adapt || !levels_s.empty()) && ((s2mode && !dfl) || (!mtp && !dfl))) refuse("--spec-adaptive needs --mtp or --dflash-manifest; under S2 it is the DFlash2 sessions' policy");
    if (!levels_s.empty() && !adapt) refuse("--spec-levels needs --spec-adaptive");
    SpecLen& spec = o.spec;
    if (adapt)
    {
        const int kmax = dfl ? dfl_k : mtp;
        std::vector<int> lv;
        if (levels_s.empty()) for (int k = 1; k <= kmax; ++k) lv.push_back(k);
        else lv = parse_ints(levels_s);
        for (int k : lv) if (k < 1 || k > kmax) refuse("--spec-levels: each level must be 1.." + std::to_string(kmax));
        spec.on = true;
        if (!spec.set_levels(lv)) refuse("--spec-levels: no level");
        spec.draft_round = dfl ? 5.6 : 2.5;      // ms at 2200 MHz: the DFlash2 step, the MTP first-form pass
        spec.draft_each = dfl ? 0 : 2.1;         // an MTP chain step
#if G6_MODEL_FLASH
        spec.flash_cost(g6nr::NR);
        if (dfl) { spec.draft_round = 5.3; spec.draft_each = 0; }   // Flash's verify table; a DFlash2 round's drafting (the gap between main steps)
#endif
    }
#if G6_MODEL_FLASH
    o.spec_initial = spec;
#endif
#if G6_MODEL_DS41
    // DS41 serve (s2/ds41_native.cuh): no --prefill tensor prefill and no --mtp draft (the draft is DSpark, GR8_DS_MTP); --graph is optional
    if (s2mode && (s2port < 1 || s2port > 65535 || s2sess < 1 || s2sess > 32 || !s2epoch))
        refuse("DS41 serving (--serve) needs --s2-port 1..65535, --s2-sessions 1..32 and --s2-epoch > 0 (no --prefill, --mtp 0; GR8_DS_MTP 1..5 is the DSpark draft); --graph is optional (without it each step runs eagerly)");
#else
    if (s2mode && (!o.graph || !mtp || !MP || s2port < 1 || s2port > 65535 || s2sess < 1 || s2sess > 8 || !s2epoch))
        refuse("serving (--serve) needs --graph, --mtp, --prefill, --s2-port 1..65535, --s2-sessions 1..8 and --s2-epoch > 0");
#endif
    if (s2mode && !knob::tests_set().empty())
        refuse("serving (--serve) refuses test knobs (knobs/test.yaml): " + knob::tests_set());
    if (knob::rt_store() && !s2mode) refuse("GR8_RT_STORE needs --serve (S2 below R with GR8_CP on)");
    if (knob::test_rt_poison()) refuse("GR8_TEST_RT_POISON is an m4sim test: g6m4 reads the local copy of a row without a slot");
    // GR8_RT_LOCAL (release): each split run's tail cycles through that many window pages per rank; the pool grants the
    // drafter's whole tail apart from them (kvman-rs Pool::window, s2_cp_map). GR8_RT_PLACE: the main model's physical
    // pages per rank (m.cp.phys), so --cap (the logical length) may pass what the local pages hold
    const int rt_win = o.rt_win = (int)knob::rt_local();
    if (rt_win < 0 || (rt_win && !knob::rt_store())) refuse("GR8_RT_LOCAL needs GR8_RT_STORE and a page count >= 1");
    if (knob::rt_place() && !rt_win) refuse("GR8_RT_PLACE needs GR8_RT_LOCAL (the pages past it are reached through the rows tier only)");
    // --kv-store with it: the pool never retains a window run (kvman-rs Pool::window), so the block stores copy at most
    // its prefix, which stays on pool pages; runs below R are as without the tier
    if (rt_win && knob::rt_l() && *knob::rt_l() != knob::cp_r())
        refuse("GR8_RT_LOCAL needs GR8_RT_L = GR8_CP_R (the window holds every tail page past R)");
    // the profile itself is loaded by main (steering::prepare_boot), after the parse
    if (!steering_dir.empty() && !hash) refuse("--steering-profile requires weight hash checks");
    if (!steering_dir.empty() && (set_id.empty() || steering_weight_set.empty()))
        refuse("--steering-profile needs --set-id and --steering-weight-set");
    // GR8_MTP_KSTOP=1|2 (s2_batch.h KStop; default 0): a solo DFlash2 session's MTP rounds draft and verify 1..
    // GR8_MTP_KSTOP_KMAX (1..3, default --mtp) drafts, chosen from the drafts' log-probabilities: 1 by GR8_MTP_KSTOP_TAU=
    // t1,t2,t3 (draft j while p_1 * .. * p_j >= t_j; default 0,0.2,0.5), 2 by the drafter choice's MTP estimates
    // (GR8_MTP_KSTOP_SEED3: their prior for a third draft). GR8_MTP_LPLOG=1 (rank 0; 2: every rank): one "# kstop" line per
    // solo round, nothing else changes. The policy is in the S2 runtime signature
    s2::KStop& kstop = o.kstop;
    {
        kstop.mode = (int)knob::mtp_kstop().value_or((s2mode && dfl && mtp >= 2) ? 2 : 0);
        const int lplog = (int)knob::mtp_lplog();
        kstop.log = lplog == 2 || (lplog == 1 && cfg.rank == 0);
        kstop.kmax = (int)knob::mtp_kstop_kmax().value_or(mtp);
        const std::vector<double>& tau = knob::mtp_kstop_tau();
        if (tau.size() > 3) refuse("GR8_MTP_KSTOP_TAU: at most 3 thresholds");
        for (size_t j = 0; j < tau.size(); ++j) kstop.tau[j + 1] = tau[j];
        kstop.seed3 = knob::mtp_kstop_seed3();
        // Without --dflash-manifest (TP=6 / TP=3, MTP only): mode 1 stops the chain of a session that decodes alone
        // (s2_batch.h KStop::solo), with GR8_MTP_KSTOP_KMAX <= --mtp; mode 2 needs the drafter choice's estimates
        if (kstop.mode && (!s2mode || (!dfl && (kstop.mode != 1 || kstop.kmax > mtp)) || mtp < 2 || kstop.kmax < 1 || kstop.kmax > 3 ||
                           !(kstop.seed3 > 0 && kstop.seed3 < 1) || kstop.kmax + 1 > std::max(s2sess * (mtp + 1), plan.chunk)))
            refuse("GR8_MTP_KSTOP needs --serve and --mtp 2+ (without --dflash-manifest: mode 1 and GR8_MTP_KSTOP_KMAX <= --mtp); "
                   "GR8_MTP_KSTOP_KMAX 1..3 (verify rows within a step's), GR8_MTP_KSTOP_SEED3 in (0, 1)");
        if (!kstop.mode) kstop.kmax = 0;
        kstop.solo = kstop.mode == 1 && !dfl;
        // GR8_MTP_KSTOPB=1 (s2_batch.h KStopB; default 0): in shared rounds each session verifies its own k_q of the K
        // drafts: the most j with p_1 * .. * p_i >= t_i for every i <= j, at least GR8_MTP_KSTOPB_KMIN (0|1, default 0).
        // GR8_MTP_KSTOPB_TAU=t1[,t2[,...]] (up to 7; default 0.6,0.88) at every session count, GR8_MTP_KSTOPB_TAU_S<n> at n
        // sessions; past the configured entries there is no stop (the remaining drafts all verify).
        // GR8_MTP_KSTOPB_WIDTHS=M,..: the main widths captured besides the multiples of K + 1 (default every width the rule reaches;
        // a width outside them takes the next most likely drafts). GR8_MTP_KSTOPB_EXPLORE=n (measurement): every n-th
        // round's shared sessions verify a hashed k_q instead. In the S2 runtime signature
        kstop.b.mode = knob::mtp_kstopb() ? 1 : 0;
        kstop.b.kmin = (int)knob::mtp_kstopb_kmin();
        double bt[8] = {};                 // tau 0 past the entries: no stop there
        const std::vector<double>& btau = knob::mtp_kstopb_tau();
        if (btau.size() > 7) refuse("GR8_MTP_KSTOPB_TAU: at most 7 comma-separated numbers");
        std::copy(btau.begin(), btau.end(), bt + 1);
        for (int S = 1; S <= s2::S2N_SLOTS; ++S)
        {
            std::copy(bt, bt + 8, kstop.b.tau[S]);
            if (const auto t = knob::mtp_kstopb_tau_s(S))
            {
                if (t->size() > 7) refuse("GR8_MTP_KSTOPB_TAU_S" + std::to_string(S) + ": at most 7 comma-separated numbers");
                std::copy(t->begin(), t->end(), kstop.b.tau[S] + 1);
            }
        }
        if (const auto& widths = knob::mtp_kstopb_widths())
        {
            if (widths->size() > 64) refuse("GR8_MTP_KSTOPB_WIDTHS: at most 64 comma-separated numbers");
            for (int64_t w : *widths)
            {
                if (w < 1 || w > s2sess * (mtp + 1)) refuse("GR8_MTP_KSTOPB_WIDTHS: whole widths 1..sessions x (mtp + 1)");
                kstop.b.widths.push_back((int)w);
            }
        }
        kstop.b.explore = (int)knob::mtp_kstopb_explore();
        // the draft log-probabilities come from every MTP draft step (StepOut::lp), with or without the DFlash2 drafter
        if (kstop.b.mode && (!s2mode || kstop.b.explore < 0))
            refuse("GR8_MTP_KSTOPB needs --serve; GR8_MTP_KSTOPB_EXPLORE >= 0");
        if (!kstop.b.mode) kstop.b = s2::KStopB();
    }
    // GR8_GLM53F_CACHE (s2/flash_checkpoint.h): Flash session checkpoints under --serve, in place of prefix reuse. Its
    // store is the one --kv-store (a lease of its own), else this process's memory, GR8_GLM53F_STATE_BUDGET_MIB a rank
    g6s2::FlashCache& flash_cache = o.flash_cache;
    try { flash_cache = g6s2::flash_cache_mode(knob::glm53f_cache().c_str()); } catch (const std::exception& e) { refuse(e.what()); }
    const long long flash_budget_mib = o.flash_budget_mib = knob::glm53f_state_budget_mib();
#if G6_MODEL_FLASH
    // Flash: --s2-sessions 1 (the default) is the solo path; 2..8 share the --cap pool (s2/flash_native.cuh)
    if (s2mode && (reuse_prefix || plan.rs))
        refuse("Flash serve supports no prefix reuse or rejection sampling");
    // DFlash2 with GR8_GLM53F_CACHE: a checkpoint hit restores none of the drafter's context, so a hit inside the
    // drafter's window keeps the drafter through GR8_DFL_REUSE (s2/flash_dfl_window.h; on by default with the cache)
    if (s2mode && dfl && !mtp)
        refuse("Flash serve with DFlash2 needs --mtp (a second session moves the DFlash2 one to MTP)");
    // a terminal checkpoint's KDA state comes from the verify ring, not a cold prefill, so the drafter would resume on
    // state a cold run does not have: aligned checkpoints only (allowing terminal is a decision of its own)
    if (s2mode && dfl && flash_cache == g6s2::FlashCache::Terminal)
        refuse("Flash serve with DFlash2 needs GR8_GLM53F_CACHE=aligned or off (a terminal checkpoint's KDA state is not cold-exact)");
    if (s2mode && s2sess > 1 && s2sess * (mtp + 1) > g6mx::MX)
        refuse("Flash serve: --s2-sessions x (--mtp + 1) rows exceed G6_MX (every session's round is one step)");
    if (flash_cache != g6s2::FlashCache::Off) {
        if (!s2mode || !MP) refuse("GR8_GLM53F_CACHE needs --serve and tensor prefill (--prefill N)");
        if (kv_tiers.size() > 1) refuse("GR8_GLM53F_CACHE: one --kv-store at most (its checkpoint store)");
        if (kv_tiers.empty() && flash_budget_mib < 1) refuse("GR8_GLM53F_STATE_BUDGET_MIB: >= 1");
    }
    if (!kv_tiers.empty() && flash_cache == g6s2::FlashCache::Off) refuse("--kv-store on Flash needs GR8_GLM53F_CACHE (its checkpoint store)");
    if (s2mode && s2sess != 1 && knob::glm53f_kv_chunks())   // flash_kv.h flashkv::env_mode() == 1
        refuse("GR8_GLM53F_KV_CHUNKS needs --s2-sessions 1");
#else
    (void)flash_budget_mib;
    if (flash_cache != g6s2::FlashCache::Off) refuse("GR8_GLM53F_CACHE is GLM-5.3-Flash only (bin/g6m4_flash)");
    // --reuse-prefix: one conversation's caches (prefix_reuse.h), or under S2 every retired session's run (s2/pool.h).
    // The DFlash2 drafter's prompt-context taps (DF.pf_feat) are written by the prefill rows a hit skips.
    if (!kv_tiers.empty() && !(s2mode && reuse_prefix)) refuse("--kv-store needs --serve and --reuse-prefix (it keeps the runs prefix reuse retains)");
#endif
    if (kv_old)
    {
        if (kv_keep.empty()) kv_keep = kv_old_all ? "all" : "system";
        if (cfg.rank == 0)
            fprintf(stderr, "# warning: --kv-tier and --kv-shared-after are deprecated: --kv-store STORE,size=N, --kv-keep %s, "
                            "--kv-system-after %d\n", kv_keep.c_str(), kv_system_after);
    }
    if (kv_keep.empty()) kv_keep = "system";
    if (kv_keep != "system" && kv_keep != "sessions" && kv_keep != "all") refuse("--kv-keep system, sessions or all");
    if (kv_system_after < 1) refuse("--kv-system-after N: N >= 1");
    if (!(kv_write_limit >= 0) || kv_write_limit > 1000) refuse("--kv-write-limit F: 0 (off) to 1000 blocks per page computed");
    {   // the KV row is fixed at build time (attn/kvfmt.h): the launcher runs bin/g6m4 for fp8 and bin/g6m4-int4 for int4,
        // and each binary refuses the other's format. The int4 build needs the flag, so it never starts by accident.
#if defined(G6_KV4) && G6_KV4
        const std::string built = "int4";
#else
        const std::string built = "fp8";
#endif
#if G6_MODEL_FLASH
        const std::string bin = "bin/g6m4_flash";
#elif G6_MODEL_DS41
        const std::string bin = "bin/g6m4_ds41";
#else
        const std::string bin = "bin/g6m4";
#endif
        if (kv_format.empty() && built == "int4") refuse("this engine is the int4 KV build (" + bin + "-int4): it needs --kv-format int4");
        if (kv_format.empty()) kv_format = built;
        if (kv_format != "fp8" && kv_format != "int4") refuse("--kv-format fp8 or int4");
        if (kv_format != built)
            refuse("--kv-format " + kv_format + ": this engine is the " + built + " build (" + bin + " is fp8, " + bin + "-int4 is int4)");
        if (cfg.rank == 0) fprintf(stderr, "# rank 0: KV format %s (latent row %d B, %s)\n", kv_format.c_str(), g6kv::ROW, g6kv::NAME);
        if (cfg.rank == 0)   // module_info.h: what host code reads about this build
            fprintf(stderr, "# rank 0: module %s abi %u: TP%d, %d rows a step, %d session slots, KV %s %d B a row, caps %#x\n",
                    g6mod::INFO.name, g6mod::INFO.abi, g6mod::INFO.nr, g6mod::INFO.mx, g6mod::INFO.max_sessions, g6mod::INFO.kv_format,
                    g6mod::INFO.kv_row_bytes, g6mod::INFO.caps);
    }
#if G6_MODEL_DS41
    // Parse before CUDA, fabric or weights. Unset decode cut inherits the main cut.
    std::string moe_topk_error;
    const int ds_moe_main_topk = o.ds_moe_main_topk = (int)knob::ds_moe_topk();   // ds41_moe_ops.cuh topk_cut_env: the table refuses a bad value
    if (ds_moe_main_topk < 0) refuse(moe_topk_error);
    const int ds_moe_decode_topk = o.ds_moe_decode_topk = ds41_moe_topk::decode_env(ds_moe_main_topk, geom::TOPK, &moe_topk_error);
    if (ds_moe_decode_topk < 0) refuse(moe_topk_error);
    fprintf(stderr, "# rank %d ds41 moe topk main %d decode %d\n", cfg.rank, ds_moe_main_topk, ds_moe_decode_topk);
    o.ds_idx_pipe_b = knob::ds41_idx_pipe_b() ? 1 : 0;
    fprintf(stderr, "# rank %d ds41 idx pipe B %d\n", cfg.rank, o.ds_idx_pipe_b);
    o.ds_idx_q_trig = knob::ds41_idx_q_trig() ? 1 : 0;
    o.ds_idx_pipe_qr = knob::ds41_idx_pipe_qr() ? 1 : 0;
    fprintf(stderr, "# rank %d ds41 idx q trig %d pipe QR %d\n", cfg.rank, o.ds_idx_q_trig, o.ds_idx_pipe_qr);
    o.ds_admit_keep_scratch = knob::ds41_admit_keep_scratch() ? 1 : 0;
    fprintf(stderr, "# rank %d ds41 idx admit keep scratch %d\n", cfg.rank, o.ds_admit_keep_scratch);
    // Prefix reuse is served by the DS41 snapshot cache; the separate --kv-store tier is not implemented for DS41.
    if (reuse_prefix && !s2mode) refuse("DS41: --reuse-prefix needs --serve");
    if (!kv_tiers.empty()) refuse("DS41: --kv-store is not supported");
    // what DS41 cannot do dies here, at parse time, not mid-boot: prefill.cuh has no DS41 branch, --mtp N > 0 would run GLM's build_draft
    if (MP > 0) refuse("DS41: --prefill is not supported (there is no DS41 tensor prefill; the prompt runs as decode steps)");
    if (mtp > 0) refuse("DS41: --mtp N > 0 is not supported (the draft is DSpark: GR8_DS_MTP=1..5 with --mtp 0)");
    // DSpark (off unless GR8_DS_MTP = 1..5): served only, the round runner is the S2 DRAFT / CHAIN path (s2/ds41_dspark_round.cuh)
    ds41::dsp::Knobs& dsk = o.dsk;
    dsk = ds41::dsp::parse_knobs(ds41_dspark_knob);
    if (!dsk.err.empty()) refuse(dsk.err);
    if (dsk.on() && !s2mode) refuse("DS41: GR8_DS_MTP is served only (it needs --serve)");
    // GR8_DS_SESS_DRAFTS=1 (s2/ds41_sess_drafts.h): a round's drafts a session go by its active sessions (Batch), the DSpark draft step runs
    // launches of at most ds_draft_cap sessions (5 rows each within MX), and ds_ms_rps(ns) is the most rows a session of an ns-session main step
    // has (its captured graphs). Off: the cap is MARKOV_MAXS and k + 1 rows up to 8 sessions, as before
    const bool ds_sess_drafts = o.ds_sess_drafts = knob::ds_sess_drafts();
    if (ds_sess_drafts && !dsk.on()) refuse("DS41: GR8_DS_SESS_DRAFTS=1 needs --serve and GR8_DS_MTP = 1..5");
    o.ds_draft_cap = ds_sess_drafts ? s2::sess_drafts_chunk(g6mx::MX, g6s2::Ds41RoundLogic::BLOCK, DS_MARKOV_MAXS) : DS_MARKOV_MAXS;
    auto ds_ms_rps = [&](int ns) { return s2::ms_rps(dsk.on(), ds_sess_drafts, dsk.mtp, g6mx::MX, ns); };
    if (dsk.relaxed && !dsk.on()) refuse("DS41: GR8_DS_RELAXED_ACCEPT needs GR8_DS_MTP = 1..5 (there are no drafts to accept)");
    if (dsk.relax_topp > 0 && nc == 0) refuse("DS41: GR8_DS_RELAXED_ACCEPT topp:P needs --top-p-cands > 0 (the candidates carry the mass above the draft)");
    if (dfl) refuse("DS41: --dflash-manifest is not supported");
    if (!grammar_dir.empty()) refuse("DS41: --grammar-dir is not supported");
    // the DS41 serve runs the whole step as one graph (GR8_DEC_SPLIT=0); the split3 graphs are the GLM serve path (the GR8_DEC_SPLIT check below asks for them from every GLM serve)
    if (s2mode && knob::dec_split() != 0) refuse("DS41: serving runs GR8_DEC_SPLIT=0 (a split serve graph is not built for DS41 yet)");
    // Engram (GR8_DS_ENGRAM=1) under --serve: the token ring of ds41_engram_ops.cuh holds the n-gram window of each session, whose rows are
    // consecutive positions. A draft-and-verify round (GR8_DS_MTP) is not checked against the ring, so Engram with DSpark is refused unless
    // GR8_DS_ENGRAM_MTP=1. ds41_engram_colls is host-only: it reads the GR8_DS_ENGRAM knobs and dies on a bad config exactly as the boot
    // does (layers 1 and 14 needed, PREFIX set), so --check-args reports it.
    if (ds41_engram_colls(layers) && dsk.on() && !knob::ds_engram_mtp())
        refuse("DS41: GR8_DS_ENGRAM=1 with GR8_DS_MTP is not validated yet (Engram on serves at MTP 0)");
    // GR8_DS_C8PACK=1: a pack needs 2+ real S2 slots and the W-prefill buffers. Check before
    // loading weights: no zero-sized pf_pos/pf_tgt/pf_sess.
    if (c8_pack_env && (!s2mode || s2sess < 2 || knob::ds_prefill_min() < 2))
        refuse("GR8_DS_C8PACK=1 needs --serve, --s2-sessions >= 2 and GR8_DS_PREFILL_MIN >= 2");
    // SESS > 1 (s2/ds41_native.cuh): one attention / index / Engram slab per session slot, and every prompt of 2+ tokens on the W-row
    // prefill (GR8_DS_PREFILL_MIN >= 2), so a step of several sessions has the same rows per session (the captured ns-session graphs:
    // 1 row at MTP 0, k + 1 verify rows with DSpark up to 8 sessions; above 8 sessions MTP is dynamically off).
    if (s2mode && s2sess != 1 && (knob::ds_prefill_min() < 2))
        refuse("DS41: --s2-sessions > 1 needs GR8_DS_PREFILL_MIN >= 2 (with GR8_DS_PREFILL_W)");
    if (s2mode && s2sess != 1 && dsk.on() && !ds_sess_drafts && (std::min(s2sess, 8) * (dsk.mtp + 1) > g6mx::MX || std::min(s2sess, 8) * g6s2::Ds41RoundLogic::BLOCK > g6mx::MX))
        refuse("DS41: --s2-sessions x (GR8_DS_MTP + 1) verify rows and --s2-sessions x 5 draft rows must fit G6_MX " + std::to_string(g6mx::MX));
    // GR8_DS_PARK=1: a parked slot is evicted through the prefix cache (--reuse-prefix); one slot has nothing to park beside
    if (knob::ds_park() && (!s2mode || s2sess < 2 || !reuse_prefix)) refuse("DS41: GR8_DS_PARK=1 needs --serve, --s2-sessions >= 2 and --reuse-prefix");
#endif
    if (reuse_prefix && dfl && !s2mode) refuse("--reuse-prefix is not supported with --dflash-manifest (the drafter's prompt-context taps come from the prefill)");
    std::vector<Plan>& plans = o.plans;
    std::vector<int>& row_counts = o.row_counts;
    for (const std::string& ps : prompts)
    {
        Plan p = plan;
        p.prompt = parse_ints(ps);
        if (p.prompt.empty()) refuse("empty prompt");
        if ((int)p.prompt.size() + p.n_gen + 2 * K > cap) refuse("prompt + gen (+ 2 x drafts) exceeds --cap");
        if (MP) { const int pr = pfrows < 0 ? (int)p.prompt.size() + pfrows : pfrows;   // --pf-rows -N: all rows but the last N
                  p.first = pr > 0 && pr < (int)p.prompt.size() - 1 ? pr : (int)p.prompt.size() - 1; }   // --pf-rows: the rest is teacher-forced through decode steps
        for (int M : plan_rows(p))
            if (std::find(row_counts.begin(), row_counts.end(), M) == row_counts.end()) row_counts.push_back(M);
        plans.push_back(p);
    }
    if (s2mode || reuse_prefix)         // any prompt (reuse changes the offline prompt-step row counts too): 1..chunk
        for (int M = 1; M <= plan.chunk; ++M)
            if (std::find(row_counts.begin(), row_counts.end(), M) == row_counts.end()) row_counts.push_back(M);
    std::vector<int>& draft_rows = o.draft_rows;   // row counts of first-form draft passes: the prompt steps' and 1..K+1
    if (mtp)
    {
        draft_rows = row_counts;
        for (int M = 1; M <= mtp + 1; ++M)
            if (std::find(draft_rows.begin(), draft_rows.end(), M) == draft_rows.end()) draft_rows.push_back(M);
        if (std::find(row_counts.begin(), row_counts.end(), mtp + 1) == row_counts.end()) row_counts.push_back(mtp + 1);
    }
    if (dfl && std::find(row_counts.begin(), row_counts.end(), dfl_k + 1) == row_counts.end()) row_counts.push_back(dfl_k + 1);
    // a DFlash2 round's MTP first form runs on the accepted prefix + 1 rows (1 .. dfl_k + 1): capture every one of them
    // (the prompt chunk and the verify widths above cover them for a block-8 drafter, not for a wider one)
    if (dfl && mtp)
        for (int M = 1; M <= dfl_k + 1; ++M)
            if (std::find(draft_rows.begin(), draft_rows.end(), M) == draft_rows.end()) draft_rows.push_back(M);
    for (int M = 2; M <= kstop.kmax + 1; ++M)   // GR8_MTP_KSTOP: solo MTP verifies of 2..kmax + 1 rows, first forms of as many
    {
        if (std::find(row_counts.begin(), row_counts.end(), M) == row_counts.end()) row_counts.push_back(M);
        if (std::find(draft_rows.begin(), draft_rows.end(), M) == draft_rows.end()) draft_rows.push_back(M);
    }
    // GR8_MTP_KSTOPB: a session alone in round_all (MTP only) verifies the rule's widths of one session; its first forms
    // are within 1..mtp + 1 (draft_rows above)
    for (int M : s2::kstopb_widths(kstop.b, 1, mtp))
        if (std::find(row_counts.begin(), row_counts.end(), M) == row_counts.end()) row_counts.push_back(M);
    // GR8_S2_ADAPT: a session alone may verify any k <= mtp, also outside limited GR8_MTP_KSTOPB_WIDTHS (s2_batch.h kshared)
    if (s2mode && s2::adapt_env())
        for (int M = 2; M <= mtp + 1; ++M)
            if (std::find(row_counts.begin(), row_counts.end(), M) == row_counts.end()) row_counts.push_back(M);
    for (int k : spec.levels)           // --spec-adaptive: a verify graph per level
        if (std::find(row_counts.begin(), row_counts.end(), k + 1) == row_counts.end()) row_counts.push_back(k + 1);
#if G6_MODEL_FLASH
    if (s2mode) for (int n = 1; n <= mtp + 1; ++n)
        if (std::find(row_counts.begin(), row_counts.end(), n) == row_counts.end()) row_counts.push_back(n);
#endif
#if G6_MODEL_DS41
    o.NL = (int)layers.size(); o.NX = ds41_engram_colls(layers); o.NC = 2 * o.NL + 1 + o.NX; o.R = cfg.rank;   // NX: 2 Engram collectives after the head slot (GR8_DS_ENGRAM=1), else 0
    o.fab_order = ds41::coll_exec_order(layers, o.NX);   // the step's collective order with Engram on, else empty (identity)
    if (dsk.on())                       // DSpark verify steps of 2..k + 1 rows (the prompt chunk covers 1..chunk); SESS > 1: ns x (k + 1) for ns <= 8
        for (int ns = 1; ns <= std::min(s2mode ? s2sess : 1, ds_sess_drafts ? 32 : 8); ++ns)
            for (int M = ns; M <= ns * ds_ms_rps(ns) && M <= g6mx::MX; M += ns)
                if (std::find(row_counts.begin(), row_counts.end(), M) == row_counts.end()) row_counts.push_back(M);
#else
    o.NL = (int)layers.size(); o.NX = 0; o.NC = 2 * o.NL + 1; o.R = cfg.rank;
#endif
    if (!grammar_dir.empty())
    {
        if (!s2mode) refuse("--grammar-dir needs --serve");
#if !G6_GRAMMAR
        // GRAMMAR defaults to 1, so a build without XGrammar serves without grammars: its hello leaves out
        // `grammar`, and the front end sends none
        fprintf(stderr, "# rank %d: --grammar-dir ignored: this engine was built without XGrammar (GR8_XGRAMMAR)\n", o.R);
        grammar_dir.clear();
#endif
    }
    // GR8_PF_OWN=4 (TP4) env refusals before any CUDA / fabric work (qp_wire.h roll_env_refusal)
    const bool own4 = o.own4 = MP && knob::pf_own() == 4;
    const int qp_roll_s = o.qp_roll_s = (int)knob::pf_qp_roll();              // rolling pack: 0 or 2..4
    o.qp_roll_ser = knob::pf_qp_roll_ser();
    const int qp_wire_e = o.qp_wire_e = (int)knob::pf_qp_wire_e().value_or(1);   // wire ring entries (qp_wire.h)
    {
        const std::string qp_ctx = knob::pf_qp_ctx() ? std::to_string(*knob::pf_qp_ctx()) : "";
        const std::string why = qw::roll_env_refusal(own4, qp_roll_s, knob::pf_qp_wire_e().has_value(), qp_wire_e,
                                                     qp_ctx.c_str(), !qp_file.empty());
        if (!why.empty()) refuse(why);
        if (own4 && g6nr::NR != 4) refuse("GR8_PF_OWN=4 is TP4 only (G6_NR=4)");
        if (own4 && !qp_roll_s && (qp_man.empty() || qp_file.empty())) refuse("GR8_PF_OWN=4 needs --qp-manifest and --qp-pack (or GR8_PF_QP_ROLL)");
        if (!own4 && (!qp_man.empty() || !qp_file.empty())) refuse("--qp-manifest / --qp-pack need GR8_PF_OWN=4 and --prefill");
    }
}
}   // namespace eopt

// Empty: the options pass; else the message main dies with.
inline std::string parse_engine_options(int argc, char** argv, EngineOptions& o)
{
    try { eopt::parse(argc, argv, o); }
    catch (const eopt::Refusal& r) { return r.msg; }
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
