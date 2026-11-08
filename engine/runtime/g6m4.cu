// M4 runtime: ONE rank of the mini model (a list of real layers + the real head) on the real fabric, one
// process per Spark. Same rank step (rank.cuh) and same test sequence as m4sim; every rank prints the "step ..."
// lines, and all six (NR) must equal m4sim's on the same packs.
//
// Per step: host writes the StepIn block (pinned), hands the proxy the payload sizes of the step's 2*NL+1
// collectives, launches the whole step on one stream (plain launches, or with --graph one replayed graph), waits for
// the stream, checks the fabric and attention error words. The token every rank samples is the same, so the ranks stay in
// step without host messages; fab.barrier() after the load lines the first step up.
//
// g6m4 --rank R --ips a,b,.. [--ips2 a,b,..] [--proxy-cpu N] --manifest rankR.manifest.json --pack rankR.pack|-
//      [--ring [--ring-devs relay-out,next,relay-in,prev]]  switchless ring (engine/comms/fabric.cpp): QPs on four local
//                     devices to the cabled neighbours, the diagonal relayed through the next rank. --ips are then the LAN
//                     addresses, for the TCP bootstraps only
//      [--image]      stream the pack straight into one device image (g6::DevImage; needed for a full 60 GB pack)
//      [--v2 DIR --v2-slices DIR --v2-widths FILE]  with --image and the Rust loader: --manifest/--pack is a base pack
//                     whose routed experts the exlr8 v2 experts replace, read from the raw width files in DIR at the
//                     offsets of the index slices (tp<N>/k<K>/rank<r>.safetensors) and the widths FILE picks
//      [--v2-dense DIR]  with --v2: no --manifest / --pack. Every other tensor comes from the exlr8 release's dense and
//                     vocab files in DIR (or DIR/dense), planned at load (engine/loader-rs/src/release.rs). Rank 0
//                     prints which form each matrix took. GR8_PF_OWN=3 takes its full-head tensors from the same image
//                     (no --qo-manifest / --qo-pack)
//      [--v2-dense-form trellis|rounded]  trellis (default): q_b, o_proj and the dense MLP stored as trellis stay the
//                     release's entries for the dense trellis kernels (kernels/dtr.cuh). rounded: the fallback, which
//                     rounds them to the MXFP8 and int4 forms of a converted pack (not the release's weights)
//      [--layout FILE]  with --v2-dense: the pieces each rank holds come from a layout file (bin/g6layout,
//                     engine/loader-rs/src/layout.rs) instead of the formulas
//      [--layers 0,3,40,77] [--prompt 154822,154824,154826,25062] [--gen 16] [--temp 0] [--seed 1234] [--cap 64] [--chunk 4]
//      [--graph]      one captured graph per row count instead of plain launches (same lines)
//      [--prefill MP] every prompt token but the last goes through the prefill path in sub-batches of up to MP rows,
//                     on a second fabric region (two collectives of MP rows, the layer as its step; prefill.cuh)
//      --prompt may be repeated: each is its own sequence (attention caches reset in between); "@file" reads ids
//      [--reuse-prefix] keep the caches of the longest common prefix of the previous prompt + emitted ids (prefix_reuse.h;
//                     GR8_REUSE_STRICT=1: strict append only, GR8_REUSE_NOEND=1: no prompt >= written_end rule; default off;
//                     gate arm GR8_REUSE_POISON=1: a hit first sets the old sequence's stale rows to NaN, reuse_poison.cuh;
//                     GR8_REUSE_ALIGN=1: a hit retains only whole cold prefill grid cells, prefix_reuse.h.
//                     Under S2: every retired session's caches stay in the pool (s2/pool.h), ALIGN and NOEND default 1.
//                     No --dflash-manifest (the drafter's prompt-context taps are written by the prefill a hit skips)
//      GR8_RT_STORE=mem:|remote:HOST:PORT (environment; --serve with GR8_CP): remote attention rows (rt_rows.cuh). A
//                     session's whole pages past GR8_RT_L (default GR8_CP_R) go to a lease named after the boot and the
//                     layout, and CP decode reads them through GR8_RT_POOL (default 4096) resident slots per indexer group.
//                     A prefill of the session (a later turn) and its retirement drop them back to the local copies.
//                     GR8_RT_FMT=int4: the lease holds the rows as INT4 (416 B for 656), turned back into FP8 after each fetch.
//                     GR8_RT_LOCAL=N: release: a split run holds its prefix and N window pages per rank, its tail pages
//                     go back to the window once in the store, and the pool grants the drafter's whole tail apart from
//                     them; retained, it keeps its records for its next turn. GR8_RT_PLACE=P: the main model's physical pages per rank, so
//                     --cap may pass what they hold (the longest tail is then the drafter class, --cap / 64 - place pages)
//      [--kv-store STORE,size=N(G|T)[,keep=RULE]]...  under S2 with --reuse-prefix: block stores below the pool, tried in flag order
//                     (s2/kv_tier.cuh, kvman-rs src/stash.rs). A request that continues a stored entry reads its pages
//                     back and skips their prefill. STORE: ssd:PATH (a folder on this box: PATH/r<rank>; with --set-id
//                     its system prefixes and their counts survive restarts) or remote:HOST:PORT (a lease on a KV node,
//                     ./gr8 kv up; RDMA where both ends can, else TCP)
//      [--kv-keep system|sessions|all]  what the stores keep (default system). system: the system prompts and tool
//                     definitions (the span the front end sends, else every grid cell) that --kv-system-after N
//                     conversations started with (default 2), once each. sessions: those, plus the runs the pool evicts
//                     that continued their conversation at least once, stored on the system prefix they start with.
//                     all: every run the pool evicts (benchmarks). A store's keep=system|sessions|all overrides it for
//                     that store: ssd:...,keep=system then remote:...,keep=all puts system prefixes on the SSD and
//                     evicted runs on the KV node
//                     A store's spare_ram_cache=on puts a cache in spare RAM in front of it (kvman-rs src/ram.rs):
//                     blocks are copied into memory the kernel may take back (MADV_FREE) while MemAvailable is above
//                     spare_ram_floor=N(M|G) (default 2G), and a read whose copy the kernel took goes to the store
//      [--kv-write-limit F]  the stores write at most F page blocks per page of KV the engine computes (off by default)
//      [--kv-tier STORE,size=N[,keep=evicted|shared|all]] [--kv-shared-after N]  deprecated: --kv-store, with --kv-keep
//                     all when any keep is evicted or all, else system; --kv-system-after N
//      [--top-p F] [--top-p-cands N]  nucleus sampling at T (per request with --serve); N candidates per rank per row
//                     (0 = top_p off; 256 is the served setting; misses printed at the end)
//      [--serve DIR --s2-port N [--s2-sessions S] [--s2-epoch E]]  serve requests through S2 (a G6_CP build); --prompt is
//                     ignored. Up to S sessions share decode steps (runs of at most GR8_S2_BELOW rows, default GR8_CP_R), a
//                     longer request runs alone at base 0 of the pool (s2/pool.h); requests over the control sockets
//                     (s2/service.h, head = rank 0 on the bootstrap address, port N); DIR is this rank's fresh S2 directory
//                     (cmd.<n> / req.<n> / resp.<n>, ready.s2, status.s2). Needs --graph, --mtp, --prefill, GR8_DEC_SPLIT=3.
//                     Without --serve the --prompt sequences run once and the process exits.
//      [--grammar-dir DIR]  with --serve, in a build with XGrammar (GR8_XGRAMMAR): requests may carry a grammar (gram,
//                     grammar.h) compiled against DIR/tokenizer.json, its stop ids from DIR/generation_config.json. The
//                     head writes grammar.s2 next to ready.s2, so the shim offers it
//      [--qp-manifest PM --qp-pack PP]  (G6_NR=4) GR8_PF_OWN=4 pair weights resident; or
//                     GR8_PF_QP_ROLL=S + GR8_PF_QP_CTX=N: S rolling slots, the partner's half over RDMA (docs/DESIGN-QO-PAIR.md)
//      [--dflash-manifest DM --dflash-pack DP [--dflash-sel 0|1] [--dflash-k K]]  replaces MTP with the DFlash2 block
//                     drafter (dflash.cuh): 7 drafts per round in one forward; DP from engine/loader-rs/src/dflash2.rs. --dflash-sel 1 =
//                     the DFlash2 selector walk (default when the pack has it), 0 = argmax drafts; --dflash-k K = verify
//                     only the first K drafts (verify steps of K + 1 rows, default 7). No --mtp, except under S2: there a
//                     session alone at its prompt tail drafts with DFlash2 and goes to MTP when a second one arrives (s2_batch.h).
//      [--spec-accept coupled|rejection]  how a verify step accepts drafts at temperature > 0. coupled (default): drafts are
//                     sampled with the target's position-keyed noise and accepted when they equal the target's sample, so
//                     a seed gives the same text with and without speculation. rejection: greedy drafts, each accepted with
//                     probability p(draft) under the request's temperature / top_p, else replaced by a sample of p without
//                     it (glue.cuh rs_accept). Same output distribution, other text for a seed.
//                     Temperature 0 is the same in both. Needs --mtp or --dflash-manifest.
//      [--spec-adaptive [--spec-levels 7,5,3,2]]  each round verifies the level with the most expected tokens per ms
//                     (spec_len.h), from the drafter's own probabilities, recent acceptance and a fixed verify cost table.
//                     Levels default to 1..K (--mtp K, or --dflash-k K for the drafter). MTP also stops its chain early.
//                     Needs --mtp or --dflash-manifest. Under S2: the DFlash2 sessions' policy (needs --dflash-manifest).
//                     Flash: on by default with --mtp K (the width's Flash verify cost table, spec_len.h flash_cost).
//      [--spec-reset-per-request]  Flash offline: restore the configured adaptive policy before each request (default off).
//                     Does not reset transport counters or cumulative MTP statistics; no effect with --spec-fixed.
//      [--spec-fixed]  Flash: --mtp K verifies all K drafts every round (the adaptive default off).
//      [--set-id HEX]  the model set id from ./gr8 glm53 up (tools/launch/modelset.py). Under S2 it joins the bootstrap
//                     agreement, so ranks loaded from different model sets refuse to start. Otherwise unused.
#ifndef DEC_S2
#error "g6m4 needs a DEC_S2 build (scripts/build.sh defines it at every TP)"
#endif
#include <algorithm>
#include <chrono>
#include "engine_options.h"
#include "fabric_plan.h"
#include "../knobs-rs/gr8_knobs.h"
#include "packfind.h"
#include "weight_check.cuh"
#include "prefill.cuh"
#include "flash_prefill.cuh"
#include "dflash.cuh"
#include "prefix_reuse.h"
#include "reuse_poison.cuh"
#include "reuse_hdump.cuh"
#include "steering_boot.cuh"
#if G6_MODEL_FLASH
#endif
#include "../comms/fabric.h"
#include "../comms/progress_cpus.h"
#include "s2/native.cuh"
#include "s2/flash_native.cuh"
#if G6_MODEL_DS41
#include "s2/ds41_native.cuh"
// the C2 dispatch trace counter lives in attn/ds41_attn.cu: ds41_attn.cuh defines device kernels, so this second
// translation unit of the binary declares the two plain functions instead of including it
unsigned long long ds41_c2_dispatch_count();
bool ds41_c2_trace_on();
#include "ds41_dspark_prefill.h"
#include "ds41_pf_split.h"
#include "ds41_rs.h"
#include "ds41_prefill_fabric.h"
#include "ds41_v2.h"
#include "ds41_v2.cuh"
#include "ds41_v2_ffn.cuh"
#include "ds41_v2_engram.cuh"
#include "../attn/ds41_attn_v2_api.h"
namespace g6rt {
#include "ds41_v2_attn.cuh"
#include "ds41_v2_engine.cuh"
}
#include "ds41_knobs.h"
#endif
#include "s2/flash_checkpoint.h"   // GR8_GLM53F_CACHE parsed in every build, refused outside Flash
#include "s2/flash_dfl_window.h"   // the Flash serve's DFlash2 prompt context (with GR8_GLM53F_CACHE hits)
#include "s2/cp_rows.h"
#include "s2/signature.h"
#include "s2/pool.h"
#include "s2/kv_tier.cuh"
#include "rt_rows.cuh"
#include "s2/kv_share.h"
#include "host_reclaim.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <unistd.h>

using namespace g6rt;

static_assert(g6mod::INFO.dfl_drafts == dfl::K && g6mod::INFO.dfl_rep == int(dfl::REP) && g6mod::INFO.tp_sort == glue::TP_SORT,
              "module_info.h's copies of the device-header constants");
#if G6_MODEL_DS41
static_assert(g6mod::INFO.ds_markov_maxs == ::ds41dsp::MARKOV_MAXS && dsr::TOPK == geom::TOPK,
              "module_info.h's copies of the device-header constants");
#endif

int main(int argc, char** argv)
{
    if (knob::init()) return 2;        // every bad GR8_ value and unknown name at once (docs/ENV.md)
    const auto boot_t0 = std::chrono::steady_clock::now();
    int boot_rank = -1;
    auto boot = [&](const char* phase) {
        fprintf(stderr, "# rank %d: boot %.1f s %s\n", boot_rank,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - boot_t0).count(), phase);
    };
    setvbuf(stdout, nullptr, _IOLBF, 0);
    pf_env_check();                    // strict GR8_PF_* choices: a bad value exits before any work
    EngineOptions eo;                  // engine_options.h: argv and the knob table, every parse-time check
    if (const std::string e = parse_engine_options(argc, argv, eo); !e.empty()) die(e);
    const g6::FabricConfig& cfg = eo.cfg;
    const std::vector<std::string>& pf_ips2 = eo.pf_ips2;
    const std::string &manifest = eo.manifest, &packfile = eo.packfile;
    V2Src& v2src = eo.v2src;
    const std::string &qp_man = eo.qp_man, &qp_file = eo.qp_file, &dfl_man = eo.dfl_man, &dfl_file = eo.dfl_file;
    const int &dfl_sel = eo.dfl_sel, &dfl_k = eo.dfl_k;
#if G6_MODEL_FLASH
    const bool& spec_reset_per_request = eo.spec_reset_per_request;
#endif
    const Plan& plan = eo.plan;
    const bool &image = eo.image, &hash = eo.hash, &reuse_prefix = eo.reuse_prefix;
    const std::string& s2dir = eo.s2dir;
    const std::string& grammar_dir = eo.grammar_dir;
#if !G6_MODEL_DS41
    const std::vector<std::string>& kv_tiers = eo.kv_tiers;
#endif
#if !G6_MODEL_FLASH && !G6_MODEL_DS41
    const std::string& kv_keep = eo.kv_keep;
    const int& kv_system_after = eo.kv_system_after;
    const double& kv_write_limit = eo.kv_write_limit;
#endif
    const std::string& kv_format = eo.kv_format;
#if !G6_MODEL_FLASH && !G6_MODEL_DS41
    const bool& kv_old = eo.kv_old;
#endif
    const std::string& set_id = eo.set_id;
    const std::string& steering_force_s = eo.steering_force_s;
    const int &s2port = eo.s2port, &s2sess = eo.s2sess;
    const unsigned long long& s2epoch = eo.s2epoch;
    const int &cap = eo.cap, &top = eo.top, &nc = eo.nc, &mtp = eo.mtp, &MP = eo.MP, &lanes = eo.lanes;
    const float& temp = eo.temp;
    const bool &graph = eo.graph, &wide = eo.wide, &pfv2 = eo.pfv2;
    const std::string &layers_s = eo.co.layers_s, &qo_man = eo.co.qo_man, &qo_file = eo.co.qo_file;
    const bool &release = eo.release, &s2mode = eo.s2mode, &dfl = eo.dfl;
    const int& K = eo.K;
    const std::vector<int>& layers = eo.layers;
    const bool& progress_isolate = eo.progress_isolate;
#if G6_MODEL_FLASH
    const int& route_split_mode = eo.route_split_mode;
#endif
#if G6_MODEL_DS41
    const int& c8_pack_env = eo.c8_pack_env;
#endif
    SpecLen& spec = eo.spec;
#if G6_MODEL_FLASH
    const SpecLen& spec_initial = eo.spec_initial;
#endif
    const int& rt_win = eo.rt_win;
#if !G6_MODEL_DS41
    s2::KStop& kstop = eo.kstop;
#endif
#if G6_MODEL_FLASH
    const g6s2::FlashCache& flash_cache = eo.flash_cache;
    const long long& flash_budget_mib = eo.flash_budget_mib;
#endif
#if G6_MODEL_DS41
    const int &ds_moe_main_topk = eo.ds_moe_main_topk, &ds_moe_decode_topk = eo.ds_moe_decode_topk;
    const int &ds_idx_pipe_b = eo.ds_idx_pipe_b, &ds_idx_q_trig = eo.ds_idx_q_trig, &ds_idx_pipe_qr = eo.ds_idx_pipe_qr;
    const int& ds_admit_keep_scratch = eo.ds_admit_keep_scratch;
    const ds41::dsp::Knobs& dsk = eo.dsk;
    const bool& ds_sess_drafts = eo.ds_sess_drafts;
    const int& ds_draft_cap = eo.ds_draft_cap;
    auto ds_ms_rps = [&](int ns) { return s2::ms_rps(dsk.on(), ds_sess_drafts, dsk.mtp, MX, ns); };
#endif
    std::vector<Plan>& plans = eo.plans;
    std::vector<int>& row_counts = eo.row_counts;
    const std::vector<int>& draft_rows = eo.draft_rows;
    const int &NL = eo.NL, &NX = eo.NX, &NC = eo.NC, &R = eo.R;
    const std::vector<int>& fab_order = eo.fab_order;
    const bool& own4 = eo.own4;
    const int& qp_roll_s = eo.qp_roll_s;
    const bool& qp_roll_ser = eo.qp_roll_ser;
    const int& qp_wire_e = eo.qp_wire_e;
    // the files the options name: the steering profile and the request grammar
    steering::PreparedBoot steering_prepared;
    try {
        const steering::ExpectedModel expected{"glm-5.3", eo.steering_model_revision, eo.steering_weight_set,
            eo.steering_tokenizer_revision, eo.steering_template_digest, HID, 78};
        steering_prepared = steering::prepare_boot(eo.steering_dir, expected, eo.steering_template_file, layers, steering_force_s);
    } catch (const std::exception& e) { die(e.what()); }
    const auto& steering_profiles = steering_prepared.profiles;
    const auto& steering_boot = steering_prepared.config;
    if (steering_boot.forced)
        for (Plan& p : plans) {
            p.steering = steering_boot.selected(steering_boot.forced_scale_bits);
            p.steering_scale = steering::bits_float(steering_boot.forced_scale_bits);
        }
    if (!grammar_dir.empty())
    {
        const std::string e = g6rt::grammar_init(grammar_dir, geom::VOCAB);
        if (!e.empty()) die("--grammar-dir " + grammar_dir + ": " + e);
        fprintf(stderr, "# rank %d: request grammars from %s (%d mask words a row)\n", R, grammar_dir.c_str(), g6rt::grammar_words());
    }
    // --check-args: every check a boot makes before CUDA passed; stop before any CUDA, fabric or weight work
    if (eo.check_args) { fprintf(stderr, "g6m4: --check-args: the arguments pass the parse-time checks (no CUDA, no weights)\n"); return 0; }
    // GR8_FABRIC_DUMP=1: every planned fabric's config on stdout, then exit; no file, CUDA, fabric or network work
    const bool fabric_dump = knob::fabric_dump();
    if (!fabric_dump && !grammar_dir.empty() && R == 0)
        if (FILE* f = fopen((s2dir + "/grammar.s2").c_str(), "w")) { fputs("json json_schema\n", f); fclose(f); }

    if (!fabric_dump) G6RT_CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    // every fabric main creates, in creation order (fabric_plan.h); the plan below sets them up
    g6::Fabric fab, fabB, fabp, fabp2, fabix, fabix2, fabrs, fabrs2, fabd, fabds, fabf, fabq, fabc, fabg, fabs, fabdg, fabv2;
    boot_rank = cfg.rank;
    mem_min_gib();                      // a bad GR8_MEM_MIN stops the boot here, not after the weights have loaded
    boot("start");
    // DS41 with Engram: the step rings each Engram reduce (c = 2*nl+1+e) mid-step, not after the head, so the proxy must wait for the collectives in
    // the step's own order (ds41_coll_order.h, Fabric::Rs::order). Every rank builds the same vector. Empty (Engram off, every other model): the
    // plain enqueue_step(step, sizes) call in the proxy's numeric order.
    auto fab_enqueue = [&](unsigned long long step, const std::vector<uint32_t>& sizes) {
        if (fab_order.empty()) { fab.enqueue_step(step, sizes); return; }
        g6::Fabric::Rs rs;
        rs.order = fab_order;
        fab.enqueue_step(step, sizes, rs);
    };
    const bool pf_rs = knob::pf_rs();                                            // reduce-scatter form (prefill.cuh)
    const int pf_own = (int)knob::pf_own();                                      // 1: row-owner projections, 2: + index (prefill.cuh)
    const bool pf_qo = MP && pf_own == 3;                                        // 3: row-owner QO attention (QO-INTEGRATION.md)
    const bool pf_qp = own4;                                                     // 4: pair-owner QO attention (TP4, qp.cuh)
    if (pf_qo && !release && (qo_man.empty() || qo_file.empty())) die("GR8_PF_OWN=3 needs --qo-manifest and --qo-pack");
    if (release && (!qo_man.empty() || !qp_man.empty())) die("--v2-dense: the QO weights come from the release (no --qo- / --qp- packs)");
    v2src.qo = pf_qo;
#if G6_NR == 4
    QpWire qp_wire;                     // GR8_PF_QP_ROLL: the wire over fabq (qp_wire.h), set up after its init
    auto qp_err = [&]() -> unsigned { return qp_roll_s ? *fabq.bulk_dev().error : 0u; };
#else
    auto qp_err = [&]() -> unsigned { return 0u; };
#endif
    // GR8_DFL_REUSE=1|2 (s2_batch.h DflReuse; default 2 when serving DFlash2 with MTP, else 0): 1 = a DFlash2 session whose
    // prefill a prefix-reuse hit cut into the drafter's window keeps the drafter (the rows the prefill taps, and the ring's
    // rows of the last DFlash2 session with the same tokens); 2 = also a session alone on MTP takes it. GR8_DFL_REUSE_MIN=
    // <rows> (0..2048, default 256): MTP rounds until the drafter holds that many context rows. Drafts only. In the S2
    // runtime signature. GLM-5.3-Flash without MTP: 1 with GR8_GLM53F_CACHE (a checkpoint hit inside the drafter's window
    // keeps the drafter, s2/flash_dfl_window.h), else 0. A promoted Flash session takes the wide KDA ring at its first
    // DFlash2 round (s2/flash_native.cuh widen)
#if G6_MODEL_FLASH
    const int dfl_reuse_unset = (dfl && s2mode) ? (mtp ? 2 : flash_cache != g6s2::FlashCache::Off ? 1 : 0) : 0;
#else
    const int dfl_reuse_unset = (dfl && s2mode && mtp) ? 2 : 0;
#endif
    const int dfl_reuse = (int)knob::dfl_reuse().value_or(dfl_reuse_unset);
    if (dfl_reuse && !(dfl && s2mode)) die("GR8_DFL_REUSE needs --dflash-manifest and --serve");
    const int dfl_reuse_min = (int)knob::dfl_reuse_min();
    if (dfl_reuse_min > dfl::WIN) die("GR8_DFL_REUSE_MIN: 0.." + std::to_string(dfl::WIN) + " rows");
    // GR8_NGRAM=<file> (s2_batch.h NgramSrc, ngram_lookup.h; default off; S5, #56): the GLM-distilled n-gram table
    // (tools/ngram/ngram.py export) as a draft source for a solo DFlash2 session: a round verifies the table's chain
    // instead of the drafter's drafts when it promises more tokens per verify ms. GR8_NGRAM_KMAX (1..dflash-k, default
    // dflash-k): the longest chain; GR8_NGRAM_KMIN (default 2); GR8_NGRAM_MARGIN (default 0.1); GR8_NGRAM_LOG=1: one line per
    // round it takes (rank 0). Drafts only: the target decides every token. In the S2 runtime signature
    g6rt::NgramTable ngram_tab;
    s2::NgramSrc ngram;
    if (const auto& path = knob::ngram())
    {
        if (!(dfl && s2mode)) die("GR8_NGRAM needs --dflash-manifest and --serve");
        std::ifstream f(*path);
        if (!f) die("GR8_NGRAM: cannot read " + *path);
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::string err;
        if (!ngram_tab.parse(text, &err, geom::VOCAB)) die("GR8_NGRAM: " + err);
        ngram.t = &ngram_tab;
        ngram.kmax = (int)knob::ngram_kmax().value_or(dfl_k);
        ngram.kmin = (int)knob::ngram_kmin();
        ngram.margin = knob::ngram_margin();
        ngram.log = knob::ngram_log() && cfg.rank == 0;
        if (ngram.kmax < ngram.kmin || ngram.kmax > dfl_k)
            die("GR8_NGRAM: 1 <= GR8_NGRAM_KMIN <= GR8_NGRAM_KMAX <= --dflash-k");
        fprintf(stderr, "# rank %d: ngram table: %zu entries, ns", cfg.rank, ngram_tab.t.size());
        for (int n : ngram_tab.ns) fprintf(stderr, " %d", n);
        fprintf(stderr, ", %s, digest %016llx; chains %d..%d, margin %.3f\n", ngram_tab.has_cal ? "calibrated" : "raw probabilities",
                (unsigned long long)ngram_tab.digest, ngram.kmin, ngram.kmax, ngram.margin);
        for (int M = 2; M <= ngram.kmax + 1; ++M)   // a chain's solo verify of k + 1 rows, class row_class(k + 1)
            if (std::find(row_counts.begin(), row_counts.end(), M) == row_counts.end()) row_counts.push_back(M);
    }
    // GR8_DRAFT_PAR=1|2 (s2_batch.h Batch::par_; default 0): a solo DFlash2 round's drafter step runs on its own stream
    // under the MTP first form (and the chain on MTP rounds). 2: a drafting step goes on a high-priority stream (its CTAs
    // are dispatched first while both wait), a context-only step stays at the default priority. Each graph's bytes are as
    // serial. Progress: every kernel that waits for peers runs at most 8 CTAs (k_wait_peers 1, k_walk 1, head merges M),
    // so the other stream always finds SMs. In the S2 runtime signature
    const int draft_par = (int)knob::draft_par().value_or((dfl && s2mode && mtp) ? 2 : 0);
    if (draft_par && !(dfl && s2mode && mtp)) die("GR8_DRAFT_PAR needs --dflash-manifest, --mtp and --serve");
    // GR8_QA_SPLIT=1|2, GR8_IDXQ_SPLIT=1, GR8_MTP_SPLIT=1 (rank.cuh QSplit; default 0 = off): replicated GEMVs split by output
    // tiles across the ranks and all-gathered. The main model's gathers run on fabg (port + 64), the MTP draft's on fabdg
    // (port + 72), each created right after its model is built (the plan sets the collective count). Proxy CPU N-9 for
    // both (main and draft steps never run at once; inside the shim / front end's taskset range 0..N-8;
    // GR8_QA_SPLIT_CPU moves it). Every rank must run the same values (the S2 runtime signature carries them).
    // replicated-GEMV split needs peers to gather across (pointless at TP1); the Flash build plans no gathers
    // DS41 builds no QSplit geometry, so the split is off there and a request for it is refused
    const bool qs_def = g6nr::NR > 1 && !G6_MODEL_FLASH && !G6_MODEL_DS41;
    const int qs_mode = (int)knob::qa_split().value_or(qs_def ? 2 : 0);
    const bool iq_split = knob::idxq_split().value_or(qs_def), mtp_split = knob::mtp_split().value_or(qs_def && mtp);
    if ((G6_MODEL_FLASH || G6_MODEL_DS41) && (qs_mode || iq_split || mtp_split)) die("GR8_QA_SPLIT, GR8_IDXQ_SPLIT and GR8_MTP_SPLIT are GLM-5.3 paths");
    if (mtp_split && !mtp) die("GR8_MTP_SPLIT=1 needs --mtp");
    // GR8_S2_DSEL_MODEL=1 + GR8_S2_DSEL_TABLE=<key:value list | file> (s2_batch.h DSelTable; default 0): the drafter
    // choice prices both drafters from one frozen cost table and MTP rounds under the confidence stop. Every rank reads
    // the same table; its canonical form is in the S2 runtime signature
    const int dsel_model = knob::s2_dsel_model() ? 1 : 0;
    s2::DSelTable dsel_tab;
    if (dsel_model)
    {
        const auto& e = knob::s2_dsel_table();
        if (!e || !dfl || !s2mode) die("GR8_S2_DSEL_MODEL needs GR8_S2_DSEL_TABLE, --serve and --dflash-manifest");
        std::string text = *e, err;
        if (text.find(':') == std::string::npos)    // no key:value: the path of a file holding the table
        {
            std::ifstream f(text);
            if (!f) die("GR8_S2_DSEL_TABLE: cannot read " + text);
            text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        if (!dsel_tab.parse(text, &err)) die(err);
    }
    // GR8_DFL_SEL_ALPHA=<a> (dflash.cuh k_walk; default 1): the DFlash2 selector walk scores logit + a * correction. Drafts
    // only: the verify is unchanged. In the S2 runtime signature
    const float dfl_alpha = (float)knob::dfl_sel_alpha();
    if (!std::isfinite(dfl_alpha)) die("GR8_DFL_SEL_ALPHA: not a finite float");
    if (dfl_alpha != 1.f && !dfl) die("GR8_DFL_SEL_ALPHA needs --dflash-manifest");
    // GR8_DRAFT_HEAD_I4=1 (rank.cuh build_draft_head_i4; default 0): the MTP draft's head reads INT4 g32 records of the
    // rank's vocab slice; the target's head is unchanged. In the S2 runtime signature.
    const DraftHeadI4 dhead = draft_head_i4_env();
    if (dhead.on && !mtp) die("GR8_DRAFT_HEAD_I4=1 needs --mtp");
    // GR8_MTP_LASTROW=1 (rank.cuh mtp_lastrow_env; default 0): a first-form draft step routes only each session's last
    // row (the rows before it only write layer 78's caches). Same caches, drafts and log-probabilities. In the S2 runtime
    // signature
    const bool mtp_last = mtp_lastrow_env();
    if (mtp_last && !mtp) die("GR8_MTP_LASTROW=1 needs --mtp");
    // GR8_DRAFT_HEAD2=1|2 (rank.cuh draft_head2_env; default 0): the MTP draft's head in two stages (2-bit screen, exact
    // MXFP8 rescoring of its best). Drafts only. In the S2 runtime signature
    const int dhead2 = draft_head2_env();
    if (dhead2 && !mtp) die("GR8_DRAFT_HEAD2 needs --mtp");
    // GR8_DFL_HEAD_I4=1 (dflash.cuh dfl_head_i4; default 0): the DFlash2 drafter's head reads the same INT4 records (the MTP
    // draft's copy when there is one), with GR8_DRAFT_HEAD_KS / _TPC. In the S2 runtime signature.
    const bool dfl_i4 = knob::dfl_head_i4();
    if (dfl_i4 && !dfl) die("GR8_DFL_HEAD_I4=1 needs --dflash-manifest");
    // GR8_DFL_BODY_I4=1 (dflash.cuh dfl_body_i4; default 0): the DFlash2 drafter's body GEMVs read INT4 g32 copies of their
    // MXFP8 records. Drafts only; +382 MB per rank. In the S2 runtime signature
    const bool dfl_body_i4_on = knob::dfl_body_i4();
    if (dfl_body_i4_on && !dfl) die("GR8_DFL_BODY_I4=1 needs --dflash-manifest");
    // the fabric plan (fabric_plan.h): every fabric's config, and the bring-up's refusals, before the first fabric
    PlanInputs fpi;
    fpi.k = plan_knobs();
#if G6_MODEL_FLASH
    fpi.pf_sp = pf_sp();
    fpi.pool_owner_mode = flash_pool_owner_mode();
#endif
    fpi.dfl_ncoll = dfl::NC;
#if G6_MODEL_DS41
    fpi.head_payload = glue::head_bytes_tk(MX, nc, true);
    fpi.trunk_bytes = sizeof(ds_mhc::trunk_t);
#endif
    for (int l : layers) { fpi.full_index.push_back(cp_full_index(l)); fpi.attn.push_back(cp_attn_layer(l)); }
#if defined(G6_CP) && G6_CP
    fpi.cp_build = true;
#endif
    fpi.gather = qs_mode || iq_split;
    fpi.draft_gather = mtp_split;
    const FabricPlan fplan = fabric_plan(eo, fpi);
    for (const std::string& l : fplan.log) fputs(l.c_str(), stderr);
    if (!fplan.refusal.empty()) die(fplan.refusal);
    const bool& two = fplan.two;
#if G6_MODEL_DS41
    const bool &rowshard = fplan.rowshard, &y_mxfp8 = fplan.y_mxfp8, &pf_direct = fplan.pf_direct;
    const int &rs_k = fplan.rs_k, &pf_lanes_fab = fplan.pf_lanes_fab;
#endif
#if G6_MODEL_FLASH
    const bool &pool_owner = fplan.pool_owner, &route_split = fplan.route_split;
#endif
    [[maybe_unused]] const int& cp_step = fplan.cp_step;   // GR8_CP_STEP_ROWS (s2/cp_rows.h): the rows a step past R may carry; 0 = off
    g6::Fabric* const fabrics[kFabKinds] = {&fab, &fabB, &fabp, &fabp2, &fabix, &fabix2, &fabrs, &fabrs2,
                                            &fabd, &fabds, &fabf, &fabq, &fabc, &fabg, &fabs, &fabdg, &fabv2};
    fabric_bring_up(fplan, fabrics, fabric_dump, [&](const PlannedFabric& e, g6::Fabric& fb) {
        switch (e.kind)
        {
        case FabKind::LaneB:
            fprintf(stderr, "# rank %d DS41 two-lane prefill: lane B fabric port %d proxy cpu %d, slot table without the C8-pack late slots (run1 %d)\n",
                    cfg.rank, e.cfg.port, e.cfg.proxy_cpu, e.cfg.run1);
            break;
        case FabKind::RouteSplit:
        case FabKind::RouteSplit2:
            if (e.kind == FabKind::RouteSplit2 || !two)
                fprintf(stderr, "# rank %d: GR8_PF_ROUTER_SPLIT=1: prefill MoE router row-split, route tables over a dedicated fabric%s\n",
                        cfg.rank, two ? " per lane" : "");
            break;
#if G6_NR == 4
        case FabKind::QpWire:
            qp_wire.bd = fb.bulk_dev(); qp_wire.E = qp_wire_e; qp_wire.peer = qw::partner(cfg.rank);
            qp_wire.enqueue = [&fabq](g6::Fabric::BulkJob j) { fabq.enqueue_bulk(std::move(j)); };
            fprintf(stderr, "# rank %d: GR8_PF_QP_ROLL=%d wire fabric: %d entr%s x [lo | hi] %.2f MB = %.3f GB pinned + registered, partner rank %d\n",
                    cfg.rank, qp_roll_s, qp_wire_e, qp_wire_e == 1 ? "y" : "ies", qw::HALF / 1e6, qw::bulk_bytes(qp_wire_e) / 1e9, qp_wire.peer);
            break;
#endif
        case FabKind::Cp:
            fprintf(stderr, "# rank %d: GR8_CP fabric %d collectives, mmax %d, %.3f GB pinned + registered\n", cfg.rank,
                    fplan.cp_colls.ncoll(), fplan.cp_mmax, (fb.vlayout().send_bytes() + fb.vlayout().recv_bytes()) / 1e9);
            break;
        default:
            break;
        }
    });
    if (fabric_dump)
    {
        fprintf(stderr, "g6m4: GR8_FABRIC_DUMP: stopping before the first fabric\n");
        return 4;                       // not a clean stop, should the knob be left set
    }
    const g6::FabricDev f = fab.dev();
    // a late fabric's planned config; its init site sets what the built model decides
    auto late_fabric = [&](FabKind k) -> g6::FabricConfig {
        const PlannedFabric* e = fplan.find(k);
        if (!e) die(std::string("the fabric plan has no ") + fab_name(k));
        return e->cfg;
    };
    bool fabg_on = false, fabdg_on = false;
    auto qs_fabric = [&](g6::Fabric& fb, const Model& mm, FabKind kind, const char* what) {
        g6::FabricConfig gc = late_fabric(kind);
        gc.ncoll = mm.qs.ncoll();
        gc.slot_bytes = qs_slot_bytes(mm);
        fb.init(gc);
        fb.start_proxy();
        fprintf(stderr, "# rank %d: %s gather fabric: %d collectives of up to %zu B, %.1f MB pinned, proxy CPU %d\n", cfg.rank, what,
                gc.ncoll, gc.slot_bytes, 2.0 * gc.ncoll * (1 + cfg.ips.size()) * ((gc.slot_bytes + 8 + 255) & ~size_t(255)) / 1e6, gc.proxy_cpu);
    };
    auto qs_err = [&]() -> unsigned { return (fabg_on ? *fabg.dev().error : 0u) | (fabdg_on ? *fabdg.dev().error : 0u); };   // the gather proxies' own error words
#if defined(G6_CP) && G6_CP
    // CP: env as m4sim (GR8_CP, GR8_CP_R, GR8_CP_S3, GR8_CP_MMAX). Own fabric `fabc` (cplayout.h variable layout:
    // per-collective slots, single-destination partials), sized in the plan
    const bool& cp_on = fplan.cp_on;
    const int &cp_R = fplan.cp_R, &cp_R2 = fplan.cp_R2, &cp_s3 = fplan.cp_s3, &cp_step_R = fplan.cp_step_R;
    const g6::CpColls& cp_colls = fplan.cp_colls;
    // enqueue the CP collectives of a step that runs the CP block (context > CpState::step_R), before its GPU work
    auto cp_enqueue = [&](unsigned long long step, int M, int context) { if (cp_on && context > cp_step_R) fabc.enqueue_step(step, cp_colls.sizes(M)); };
    auto cp_error = [&]() -> unsigned { return (cp_on ? *fabc.dev().error : 0u) | qs_err(); };   // + the GR8_QA_SPLIT proxy
#else
    auto cp_enqueue = [&](unsigned long long, int, int) {};
    auto cp_error = [&]() -> unsigned { return qs_err(); };   // the GR8_QA_SPLIT proxy
#endif
    G6RT_CK(bind::init());
    G6RT_CK(l2pf::init_env());     // GR8_L2PF (kernels/l2pf.cuh); unset = off
    // GR8_MTP_L2PF=1|2|3 (kernels/l2pf.cuh; rank.cuh l2pf_window, l2pf_tail; default 0; needs GR8_L2PF and --mtp): 1 = the MTP
    // draft step arms the paced L2 prefetch (layer 78's windows, its eh_proj gather wait, its MLP wait: the draft head's
    // records); 2 = the head merge wait of the main step and of each draft step prefetches the next draft step's front
    // (this rank's eh_proj records, x_in), at most GR8_MTP_L2PF_US (60) us, ended by the head all-gather's arrival. A
    // prefetch writes nothing, so every output byte is unchanged; no collective is added (a window only reads the peers'
    // tails), so the ranks need not agree on it (as GR8_L2PF, not in the S2 runtime signature)
    {
        const int v = (int)knob::mtp_l2pf();
        if (v && !mtp) die("GR8_MTP_L2PF needs --mtp");
        if (v && !l2pf::cfg().win) die("GR8_MTP_L2PF needs GR8_L2PF (its side stream, pace and window bits)");
        l2pf::cfg().mtp = v;
        const double us = knob::mtp_l2pf_us();   // 0..1000 (table)
        if (!(us > 0)) die("GR8_MTP_L2PF_US=0: want us, 0 < x <= 1000");
        l2pf::cfg().mtp_us = (float)us;
        if (v && R == 0) fprintf(stderr, "# rank 0: GR8_MTP_L2PF=%d: %s%s\n", v, v & 1 ? "draft-step windows" : "",
                                 v & 2 ? (v & 1 ? ", head-merge tails" : "head-merge tails") : "");
    }
    pf_prof().on = knob::pf_prof() != 0;
    pf_prof().ev = knob::pf_prof() == 2;
    pf_prof().rank = R;
    pf_prof().tl_left = R == 0 ? (int)knob::pf_prof_timeline() : 0;
#if G6_MODEL_FLASH
    if (pf_prof().ev) die("GR8_PF_PROF=2 is GLM-5.3 only: the Flash prefill profile synchronizes (GR8_PF_PROF=1)");
#endif
    if (knob::dec_prof() && graph) die("GR8_DEC_PROF needs plain launches (no --graph)");
    cudaStream_t st;
    G6RT_CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

    g6::Pack pack;
    DevStore store;
    Model m;
#if G6_MODEL_DS41
    m.ds41_pack_enabled = c8_pack_env == 1;
    Model DS;                           // the DSpark draft Model (layers 40..42), built only with GR8_DS_MTP
    StepIn* in_ds = nullptr;
    StepOut* out_ds = nullptr;
    cudaGraphExec_t exec_ds[9] = {};    // its captured draft steps: [S] = S sessions of 5 rows (S = 1: the one-session block)
#endif
    g6::DevImage img;
    boot("fabric up");
    host_mem_log(R, "before the weights");     // #4: MemAvailable, page cache and RSS around the load
    if (!v2src.dir.empty() && (!image || v2src.slices.empty() || v2src.widths.empty()))
        die("--v2 needs --image, --v2-slices and --v2-widths");
    if (image) load_image(pack, img, manifest, packfile, R, hash, v2src);
    else load_pack(pack, manifest, packfile, R);
    u64* stepp = dmalloc<u64>(1);
    StepIn* in;
    StepOut* out;
    G6RT_CK(cudaHostAlloc((void**)&in, sizeof(StepIn), cudaHostAllocMapped));
    G6RT_CK(cudaHostAlloc((void**)&out, sizeof(StepOut), cudaHostAllocMapped));
    // dense trellis matrices: their FP8 copies for prefill under the MXFP8 names (dtr_find.h)
    const Find find = dtr_overlay(image ? image_find(pack, img) : pack_find(pack, store), MP > 0, st, R);
    g6::Pack qo_pack;
    g6::DevImage qo_img;                // --image: the qo pack streamed into one device allocation (no host copy)
    std::vector<QoW> qow;               // pf_qo: full-head weights per model layer
    if (pf_qo)
    {
        if (release) {}             // the rank image holds the full-head tensors (L%03d.qo.*)
        else if (image) load_image(qo_pack, qo_img, qo_man, qo_file, 0, hash);
        else load_pack(qo_pack, qo_man, qo_file, 0);
        const Find qfind = release ? find : image ? image_find(qo_pack, qo_img) : pack_find(qo_pack, store);
        for (int l : layers)
        {
            QoW w;
            if (!qo_from_pack(qfind, l, w)) die("--qo-pack has no full-head tensors for layer " + std::to_string(l));
            qow.push_back(w);
        }
        fprintf(stderr, "# rank %d: GR8_PF_OWN=3 full-head q_b / o_proj / UK / UV for %zu layers\n", R, qow.size());
    }
#if G6_NR == 4
    g6::Pack qp_pack;
    g6::DevImage qp_img;                // --image: the qp pack streamed into one device allocation (no host copy)
    std::vector<QpW> qpw(layers.size());   // pf_qp: pair weights per model layer (empty outside GR8_PF_QP_LAYERS)
    const std::vector<char> qpl = pf_qp ? pf_qp_mask(layers) : std::vector<char>(layers.size(), 0);
    QpRoll* qp_roll = nullptr;
    if (qp_roll_s)                      // S slots, own half from this rank's pack, partner half over fabq (no --qp-pack)
    {
        qp_roll = qp_roll_new(qp_roll_s, qp_roll_ser, std::vector<Find>{find}, R, layers, qpl, qpw, &qp_wire);
        int n = 0;
        for (char c : qpl) n += c != 0;
        fprintf(stderr, "# rank %d: GR8_PF_OWN=4 pair q_b / o_proj / UK / UV for %d of %zu layers (GR8_PF_QP_LAYERS), rolling: own pack + partner by RDMA\n", R, n, layers.size());
    }
    else if (pf_qp)
    {
        if (image) load_image(qp_pack, qp_img, qp_man, qp_file, R, hash);
        else load_pack(qp_pack, qp_man, qp_file, R);
        const Find pfind = image ? image_find(qp_pack, qp_img) : pack_find(qp_pack, store);
        int n = 0;
        for (size_t k = 0; k < layers.size(); ++k)
        {
            if (!qpl[k]) continue;
            if (!qp_from_pack(pfind, find, R, layers[k], qpw[k])) die("--qp-pack has no pair tensors for layer " + std::to_string(layers[k]));
            ++n;
        }
        fprintf(stderr, "# rank %d: GR8_PF_OWN=4 pair q_b / o_proj / UK / UV for %d of %zu layers (GR8_PF_QP_LAYERS)\n", R, n, layers.size());
    }
#endif
    m.wide = wide;
    m.qs.mode = qs_mode;
    m.qs.iq = iq_split;
#if defined(G6_CP) && G6_CP
    m.cp.on = cp_on; m.cp.R = cp_R; m.cp.R2 = cp_R2; m.cp.s3 = cp_s3; m.cp.colls = cp_colls;
    if (const auto place = knob::rt_place())
    {
        m.cp.phys = (int)*place;
        if (m.cp.phys < cp_R / 64 + rt_win) die("GR8_RT_PLACE must hold the prefix (GR8_CP_R / 64 pages) and one window");
        if (R == 0) fprintf(stderr, "# rank 0: GR8_RT_PLACE: %d physical pages per rank for the main model (--cap %d is the logical length)\n", m.cp.phys, cap);
    }
#endif
    host_reclaim(R, "after the pack");         // the loaders keep no page cache; this trims the heap they freed
    boot("pack loaded");
#if G6_MODEL_FLASH
    if (mtp || dfl) m.kda_ring = std::max(mtp + 1, dfl ? dfl::B : 0);   // KDA state slots for the verify ring (DFlash2: B rows)
    if (s2mode) m.kda_sessions = s2sess;   // each S2 session its own ring (s2/flash_ring.h FlashSlots)
    // one ring is B wide, for the session drafting with DFlash2 (FlashSlots::take_wide); the others hold an MTP round
    if (s2mode && mtp) m.kda_ring_rest = mtp + 1;
#endif
#if G6_MODEL_FLASH
    // Flash prefill (flash_prefill.cuh): the P1 bodies run the decode blocks in MX-row chunks, so the decode bindings
    // stay decode-sized. Combinations the Flash prefill does not support yet stop here.
    if (MP && (pf_rs || pf_own || !qp_man.empty() || !qo_man.empty() || reuse_prefix))
        die("Flash --prefill: not with GR8_PF_RS, GR8_PF_OWN, --qo-/--qp- packs or --reuse-prefix (P1)");

    if (two && knob::pf_streams() == 2)
        die("Flash --pf-lanes 2 uses one compute stream; GR8_PF_STREAMS=2 is not supported");
    if (two && knob::resid_dump())
        die("Flash --pf-lanes 2: GR8_RESID_DUMP is not supported (use GR8_PF_HASH / GR8_LOGIT_DUMP)");
    build(m, R, layers, find, cap, f, stepp, st, MX);
#else
#if G6_MODEL_DS41
    m.ds_slots = s2mode ? s2sess : 1;   // SESS: one attention / index / Engram slab per S2 session slot (cap rows each)
    m.ds41_idx_pipe_b = ds_idx_pipe_b == 1;
    m.ds41_admit_keep_scratch = ds_admit_keep_scratch == 1;
    // a step of several sessions gives each session 1 row, or k + 1 verify rows with DSpark (ds41_session_rows, the captured ns x rps graphs below)
    m.ds41_sess_rows = s2::index_sess_rows(dsk.on(), dsk.mtp);
    for (int ns = 2; ns <= m.ds_slots && ns <= MX; ++ns)
        if (ds_ms_rps(ns) > m.ds41_sess_rows)
            die("DS41: a " + std::to_string(ns) + "-session step has " + std::to_string(ds_ms_rps(ns)) + " rows a session, above the per-session index scratch (" +
                std::to_string(m.ds41_sess_rows) + " rows)");
    build_ds41(m, R, layers, find, cap, f, stepp, st, pf_attn_rows(MP), dsk.on() ? 1 : 0, rs_k, rowshard);
    // DS41 image input (GR8_DS41_VISION, default 0, opt-in): with --serve and W-row prefill, the release's vision tower loads here,
    // before the rest of the boot allocates, so the ready memory counts it. Every rank runs the tower on its own images
    // (each rank embeds every row). The head writes DIR/vision.s2, so the shim's hello offers `vision`.
    Ds41Vision vis;
    bool vis_on = false;
    std::string vis_id;   // the loaded tower file's id (vision_id.h: size + header, not tensor data): in the S2 runtime signature
    if (s2mode && release && knob::ds41_vision()) {
        std::string vpath;
        for (const char* rel : {"/exlr8-extra-vision.safetensors", "/dense/exlr8-extra-vision.safetensors"}) {
            const std::string c = v2src.dense + rel;
            if (FILE* fv = fopen(c.c_str(), "rb")) { fclose(fv); vpath = c; break; }
        }
        if (vpath.empty())
            fprintf(stderr, "# rank %d: ds41 vision off: no exlr8-extra-vision.safetensors in --v2-dense\n", R);
        else if (!knob::ds_prefill_min())
            fprintf(stderr, "# rank %d: ds41 vision off: image prompts prefill, and GR8_DS_PREFILL_MIN is 0\n", R);
        else {
            vis_id = g6vis::file_id(vpath);
            if (vis_id.empty()) die("DS41 vision: cannot read the safetensors header of " + vpath);
            vis.boot(vpath, 2 * m.hc_rows);
            vis.attach(m);
            vis_on = true;
            g6rt::vision_flag() = true;
            fprintf(stderr, "# rank %d: ds41 vision: tower %s, %.1f MiB on the device (weights, workspace, %d-row span pool)\n", R,
                    vpath.c_str(), vis.device_bytes() / (1024.0 * 1024.0), 2 * m.hc_rows);
            if (R == 0)
                if (FILE* fv = fopen((s2dir + "/vision.s2").c_str(), "w")) { fputs("1\n", fv); fclose(fv); }
        }
    }
    if (s2mode && R == 0 && !vis_on) std::remove((s2dir + "/vision.s2").c_str());   // a stale one from an earlier boot
    if (rs_k) ds41_rs_setup(m, R, NC, rs_k, rowshard, y_mxfp8);
    m.ds41_direct = pf_direct;                 // GR8_DS_PF_DIRECT=1
    if (ds41moe::ro_order_on())                // GR8_DS_RO_ORDER liveness, read through the same parser the expert loop launch uses (ds41_moe_ops.cuh)
        fprintf(stderr, "# rank %d DS41 read-once expert order: GR8_DS_RO_ORDER=1 (k_order_ro; L2 lookahead %d)\n", R, ds41moe::ro_l2la());
#else
    build(m, R, layers, find, cap, f, stepp, st, pf_attn_rows(MP));   // bindings sized for the attention chunk
#endif
#endif
    if (steering_boot.loaded)
    {
        try { materialize_steering(m, steering_profiles, st); }
        catch (const std::exception& e) { die(e.what()); }
        for (const auto& sp : steering_profiles)
        {
            fprintf(stderr, "# rank %d: steering profile %s manifest %s tensor %s scope %s layers %zu forced %s\n", R,
                    sp.name.c_str(), steering::digest_hex(sp.manifest_digest).c_str(),
                    steering::digest_hex(sp.tensor_digest).c_str(), steering::scope_name(sp.scope),
                    sp.layers.size(), steering_boot.forced ? steering_force_s.c_str() : "off");
            if (R == 0) fprintf(stderr, "# steering manifest %s\n", sp.manifest_json.c_str());
        }
    }
    boot("layers bound");
#if G6_MODEL_DS41
    if (dsk.on())                       // after the main build: ds41dsp_ops::build binds both Models (taps, projections, markov head, confidence)
    {
        G6RT_CK(cudaHostAlloc((void**)&in_ds, sizeof(StepIn), cudaHostAllocMapped));
        G6RT_CK(cudaHostAlloc((void**)&out_ds, sizeof(StepOut), cudaHostAllocMapped));
        DS.ds41_draft = true;           // build_ds41 sets it too; ncoll() must be DRAFT_NCOLL before the fabric check
        DS.ds_slots = m.ds_slots;       // SESS > 1: one slab of the draft layers' key rings per S2 slot
        build_ds41(DS, R, std::vector<int>{geom::L_MAIN, geom::L_MAIN + 1, geom::L_MAIN + 2}, find, cap, fabds.dev(), dmalloc<u64>(1), st, MX, 2);
        ds41dsp_ops::build(m, DS, /*nsess=*/std::min(m.ds_slots, ds_draft_cap), dsk.mtp, /*host_accept=*/true,
                          s2mode && knob::ds_prefill_min() ? m.hc_rows : 0,   // W-row tap/MainS when prefill is admitted
                          ds_draft_cap, /*verify_all=*/!ds_sess_drafts);
        DS.graph = graph; DS.in = in_ds; DS.out = out_ds;
    }
#endif
    if (m.qs.on())                      // every rank, at the same point: its TCP bootstrap is a barrier
    {
        qs_fabric(fabg, m, FabKind::Gather, "GR8_QA_SPLIT / GR8_IDXQ_SPLIT");
        fabg_on = true;
        qs_attach(m, fabg.dev(), f.error);   // GPU-side gather wait timeouts land in the error word the step loop checks
    }
    // enqueue the gathers of a main step (every main step runs all of them), before its GPU work
    auto qs_enqueue = [&](unsigned long long step, int M) { if (fabg_on) fabg.enqueue_step(step, qs_sizes(m, M)); };
    // the same for a draft step (GR8_MTP_SPLIT); D is built further down, before any step
    Draft D;
    auto qsd_enqueue = [&](unsigned long long step, int M) { if (fabdg_on) fabdg.enqueue_step(step, qs_sizes(D.d, M)); };
#if defined(G6_CP) && G6_CP
    if (cp_on)
        for (int c = 0; c < cp_colls.ncoll(); ++c)
        {
            g6::VSlotDev v = fabc.vslot(c);
            v.error = f.error;              // GPU-side CP wait timeouts land in the error word the step loop checks
            m.cp.vs.push_back(v);
        }
    // step 4d: CP prefill staging pools = the registered area of a bulk-mode fabric (prefill.cuh CpPfNet)
    const bool cp_pf = cp_on && MP;
    if (cp_pf)
    {
        g6::FabricConfig sc = late_fabric(FabKind::CpStage);
        sc.bulk_bytes = 2 * cp_pf_pool_bytes(m);
        fabs.init(sc);
        fabs.start_proxy();
        cp_pf_net_init(m, fabs, st);
    }
    auto cp_pf_in = [&]() { if (cp_pf) cp_pf_enter(m); };
    auto cp_pf_out = [&]() { if (cp_pf) cp_pf_exit(m); };
    auto cp_pf_err = [&]() -> unsigned { const unsigned e = cp_pf ? *fabs.bulk_dev().error : 0u; return e ? e : qp_err(); };   // + the qp wire
#else
    auto cp_pf_in = [&]() {};
    auto cp_pf_out = [&]() {};
    auto cp_pf_err = [&]() -> unsigned { return qp_err(); };   // the qp wire's error word (GR8_PF_QP_ROLL)
#endif
    FILE* rlf = R == 0 ? route_log_open(m) : nullptr;     // GR8_ROUTE_LOG=<file> (measurement, MTP main steps)
    if (s2mode && R == 0 && knob::s2_roundlog()) route_log_alloc(m);   // route copies for the "# s2_round" lines (s2/native.cuh)
    m.graph = graph; m.in = in; m.out = out;
    Prefill P;
    g6::FabricDev fp{};
    if (MP)
    {
        fp = fabp.dev();
        P.v2 = pfv2;
        P.own = pf_own >= 1; P.ownix = pf_own == 2 || pf_qp;
        P.qo = pf_qo; P.qow = &qow;
#if G6_NR == 4
        P.qp = pf_qp; P.qpw = &qpw; P.qpl = qpl; P.roll = qp_roll;
#endif
        P.attn_rows = pf_attn_rows(MP);
#if !G6_MODEL_FLASH
        build_prefill(P, m, find, MP, fp, dmalloc<u64>(1), st);
#endif
    }
#if G6_MODEL_FLASH
    FlashPrefill FP, FP2;
    if (MP) build_flash_prefill(m, FP, MP, fp, dmalloc<u64>(1), st);
    if (MP) pf_prewarm_flash(m, FP, st);
#endif
    Prefill P2;
    g6::FabricDev fp2{};
    if (two)
    {
        fp2 = fabp2.dev();
        P2.attn_rows = pf_attn_rows(MP);
        P2.own = pf_own >= 1; P2.ownix = pf_own == 2 || pf_qp;
        P2.qo = pf_qo; P2.qow = &qow;
#if G6_NR == 4
        P2.qp = pf_qp; P2.qpw = &qpw; P2.qpl = qpl; P2.roll = qp_roll;
#endif
#if G6_MODEL_FLASH
        build_flash_prefill(m, FP2, MP, fp2, dmalloc<u64>(1), st);
        pf_prewarm_flash(m, FP2, st);
#else
        build_prefill(P2, m, find, MP, fp2, dmalloc<u64>(1), st, &P);
#endif
    }
#if G6_MODEL_FLASH
    if (pool_owner) {
        init_flash_pool_transport(FP.pool_transport, fabix, MP);
        if (two) init_flash_pool_transport(FP2.pool_transport, fabix2, MP);
    }
    if (route_split) {
        init_flash_route_split(FP.route_split, fabrs, MP);
        if (two) init_flash_route_split(FP2.route_split, fabrs2, MP);
    }
    // GLM-5.3-Flash image input (GR8_GLM53F_VISION, default 0, opt-in): with --serve, --prefill and the release, the
    // vision tower loads here, with its workspace and a pool of image rows for one prefill quantum, so the ready memory
    // counts it. Every rank runs the tower on its own images (each rank embeds every row). The head writes
    // DIR/vision.s2 ("glm5v"), so the shim's hello offers `vision` and checks the Flash grids.
    FlashVision fvis;
    bool vis_on = false;
    std::string vis_id;   // the loaded tower file's id (vision_id.h: its size and header), in the S2 runtime signature
    if (s2mode && release && knob::glm53f_vision()) {
        std::string vpath;
        for (const char* rel : {"/dense/vision-bf16.safetensors", "/vision-bf16.safetensors"}) {
            const std::string c = v2src.dense + rel;
            if (FILE* fv = fopen(c.c_str(), "rb")) { fclose(fv); vpath = c; break; }
        }
        if (vpath.empty())
            fprintf(stderr, "# rank %d: flash vision off: no dense/vision-bf16.safetensors in --v2-dense\n", R);
        else if (!MP)
            fprintf(stderr, "# rank %d: flash vision off: image rows reach the model only through the prefill, and --prefill is 0\n", R);
        else {
            vis_id = g6vis::file_id(vpath);
            if (vis_id.empty()) die("Flash vision: cannot read the safetensors header of " + vpath);
            const int pool_rows = MP * (two ? 2 : 1);
            fvis.boot(vpath, pool_rows);
            fvis.attach(FP);
            if (two) fvis.attach(FP2);
            vis_on = true;
            g6rt::vision_flag() = true;
            fprintf(stderr, "# rank %d: flash vision: tower %s, %.1f MiB on the device (weights, workspace, %d-row pool, staging)\n",
                    R, vpath.c_str(), fvis.device_bytes() / (1024.0 * 1024.0), pool_rows);
            if (R == 0)
                if (FILE* fv = fopen((s2dir + "/vision.s2").c_str(), "w")) { fputs("glm5v\n", fv); fclose(fv); }
        }
    }
    if (s2mode && R == 0 && !vis_on) std::remove((s2dir + "/vision.s2").c_str());   // a stale one from an earlier boot
#endif
    u64 ls0b = 1;
    DraftPrefill DQ;
    g6::FabricDev fd{};
    StepIn* in_d = nullptr;
    StepOut* out_d = nullptr;
    if (mtp)
    {
        fd = fabd.dev();
        G6RT_CK(cudaHostAlloc((void**)&in_d, sizeof(StepIn), cudaHostAllocMapped));
        G6RT_CK(cudaHostAlloc((void**)&out_d, sizeof(StepOut), cudaHostAllocMapped));
#if G6_MODEL_FLASH
        build_draft(D, m, find, cap, fd, dmalloc<u64>(1), st, MX);   // Flash draft prefill: decode draft steps (flash_prefill.cuh)
        D.d.graph = graph; D.d.in = in_d; D.d.out = out_d;
#else
        if (mtp_split) { D.d.qs.mode = 1; D.d.qs.iq = true; D.d.qs.eh = true; }   // GR8_MTP_SPLIT: eh_proj, q_a / kv_a / idx, idx_q
        build_draft(D, m, find, cap, fd, dmalloc<u64>(1), st, MP ? pf_attn_rows(MP) : MX);
        if (D.d.qs.on())
        {
            qs_fabric(fabdg, D.d, FabKind::DraftGather, "GR8_MTP_SPLIT");
            fabdg_on = true;
            qs_attach(D.d, fabdg.dev(), fd.error);   // the draft fabric's error word, which the draft step checks
        }
        build_draft_head_i4(D, dhead, st);
        build_draft_head2(D, dhead2, st);
        D.d.route_last = mtp_last;
        D.d.graph = graph; D.d.in = in_d; D.d.out = out_d;
        DQ.P.attn_rows = pf_attn_rows(MP);
        if (MP) build_draft_prefill(DQ, D, find, MP, fp, pfv2, st, &P);
#endif
    }
    Dfl DF;                             // --dflash
    g6::Pack dpack;
    g6::DevImage dimg;
    DflIn* in_f = nullptr;
    DflOut* out_f = nullptr;
    if (dfl)
    {
        load_image(dpack, dimg, dfl_man, dfl_file, R, hash);
        const Find dfind = image_find(dpack, dimg);
        const DflMeta meta = dfl_read_meta(dfind);
        if (dfl_sel == 1 && !meta.sel) die("--dflash-sel 1: the drafter pack has no selector");
        G6RT_CK(cudaHostAlloc((void**)&in_f, sizeof(DflIn), cudaHostAllocMapped));
        G6RT_CK(cudaHostAlloc((void**)&out_f, sizeof(DflOut), cudaHostAllocMapped));
        DF.in = in_f; DF.out = out_f; DF.nc = nc;
        // GR8_DFL_PFCTX=0 (measurement): the drafter gets no prompt context from the prefill and starts at the last prompt token
        const bool pfctx = MP && knob::dfl_pfctx();
        dfl_build(DF, m, dfind, meta, fabf.dev(), dmalloc<u64>(1), dfl_sel < 0 ? meta.sel : dfl_sel == 1, st, pfctx);
        if (dfl_i4)
            dfl_head_i4(DF, D.d.dh_i4 ? D.d.dh_i4 : head_i4_records(m.head_t.lm_head, m.vn, dhead.ks, R, "GR8_DFL_HEAD_I4", st),
                        dhead.tpc, dhead.ks);
        if (dfl_body_i4_on) dfl_body_i4(DF, st);
        if (dfl_alpha != 1.f && !DF.sel) die("GR8_DFL_SEL_ALPHA needs the selector walk");
        DF.sel_alpha = dfl_alpha;
        dfl_attach(m, DF);              // before the target graphs are captured: they carry the residual taps
        if (pfctx) { P.dfl_tap = dfl_pf_tap_launch; P2.dfl_tap = dfl_pf_tap_launch; }
#if G6_MODEL_FLASH
        if (pfctx) { FP.dfl_tap = dfl_pf_tap_flash; FP2.dfl_tap = dfl_pf_tap_flash; }
#endif
    }
    // S2: below R on the CP binding. Session views on the pool, main and draft; today's graphs below
    // are captured with s2on off (the S = 1 path), the S2 graphs later by s2/native.cuh with s2on on.
    int s2_below = 0;
    std::unique_ptr<RtTier> rtt;       // GR8_RT_STORE (rt_rows.cuh); null = off
    int rt_rows = 0, rt_runs = 0;      // release: rows a session past R puts in one step, such sessions the slots cover
    if (s2mode && !G6_MODEL_FLASH && !G6_MODEL_DS41)
    {
#if defined(G6_CP) && G6_CP
        if (!cp_on || !cp_R) die("S2: serving below R needs GR8_CP on and GR8_CP_R");
        s2_below = (int)knob::s2_below().value_or(cp_R);
        if (s2_below < 64 || s2_below % 64 || s2_below > cp_R) die("S2: GR8_S2_BELOW (default GR8_CP_R) must be 64..GR8_CP_R, a multiple of 64");
#else
        die("S2: serving needs the CP binding (a G6_CP build)");
#endif
        m.s2_below = D.d.s2_below = s2_below;
        // GR8_RT_STORE: on the main model's bindings before s2_init and any prepare copy their arguments
        if (const auto& rs = knob::rt_store())
        {
            const int L = (int)knob::rt_l().value_or(cp_R);
            int S = (int)knob::rt_pool().value_or(4096);
            if (s2_below != cp_R) die("GR8_RT_STORE needs GR8_S2_BELOW = GR8_CP_R (runs past R on the pool's page classes)");
            char host[64] = "host";
            gethostname(host, sizeof host - 1);
            const std::string fmt = kv_format.empty() ? g6kv::NAME : kv_format;
            const bool int4 = rt_fmt_int4();
            const std::string tag = "R" + std::to_string(cp_R) + "-L" + std::to_string(L) + "-nr" + std::to_string(g6nr::NR) + "-" + fmt +
                                    rt_fmt_name(int4) + "-" + rt_name_part(set_id);
            const std::string layout = "R=" + std::to_string(cp_R) + " L=" + std::to_string(L) + " nr=" + std::to_string(g6nr::NR) + " row=" +
                                       g6kv::NAME + rt_fmt_layout(int4) + " model=" + rt_name_part(set_id) + " pages=" +
                                       std::to_string(m.layers[0].ab.args.page_count) + " layers=" + std::to_string(m.nl);
            if (set_id.empty() && R == 0) fprintf(stderr, "# rank 0: GR8_RT_STORE without --set-id: the lease names no model set\n");
            // release: a session's owned tail pages sum to at most the drafter class (the pages past place) over nr,
            // plus one partial page each
            int ids = 0;
            if (rt_win)
            {
                const G6Attn& a0 = m.layers[0].ab.args;
                // place as every rank computes it (the tail split differs by a page between ranks): the ids, and so the
                // pool's digest, must agree
                const int rp = cp_R / 64, fair = rp + (a0.table_size - rp) / g6nr::NR, place = m.cp.phys ? std::min(fair, m.cp.phys) : fair;
                ids = (a0.table_size - place + g6nr::NR - 1) / g6nr::NR + s2sess;
                const int need = 2 * ((MP + 63) / 64 + g6nr::NR - 1) / g6nr::NR + 2;   // a two-lane quantum's pages per rank, its partial ones, a decode page
                if (rt_win < need) die("GR8_RT_LOCAL " + std::to_string(rt_win) + ": a prefill quantum of --prefill " + std::to_string(MP) +
                                         " rows needs at least " + std::to_string(need) + " window pages");
                if (rp + rt_win > place) die("GR8_RT_LOCAL: the window does not fit the main model's pages past the prefix");
                // the slot pool: only sessions past R read released rows, each at most rt_rows rows a step (an MTP
                // verify, a DFlash2 verify, or a prompt chunk), and the window layout holds at most rt_runs of them
                // below place. Every row of theirs a step selects needs a slot: 2048 x rt_rows x rt_runs. A smaller
                // GR8_RT_POOL lowers rt_runs, and the pool then admits fewer sessions past R at once
                rt_rows = std::min<int>(MX, std::max({mtp + 1, dfl ? dfl::K + 1 : 0, plan.chunk}));   // DFlash2: its widest verify
                // sessions at GR8_CP_R2 hold fewer prefix pages: the slots cover as many of them as place holds
                rt_runs = std::min(s2sess, place / ((cp_R2 ? cp_R2 / 64 : rp) + rt_win));
                if (knob::rt_pool()) rt_runs = std::min(rt_runs, S / (2048 * rt_rows));
                else S = 2048 * rt_rows * rt_runs;
                if (rt_runs < 1) die("GR8_RT_POOL " + std::to_string(S) + ": less than one session past R (2048 x " + std::to_string(rt_rows) + " rows)");
                if (R == 0)
                    fprintf(stderr, "# rank 0: GR8_RT_LOCAL slot pool %d = 2048 x %d rows a session x %d sessions past R at once (place %d, %d prefix + "
                                    "%d window pages each)\n", S, rt_rows, rt_runs, place, rp, rt_win);
            }
            rtt = rt_build(m, *rs, std::string("g6m4-") + rt_name_part(host) + "-" + std::to_string(s2epoch) + "-rt-" + tag + (rt_win ? "-rel" : ""),
                           layout + (rt_win ? " released" : ""), L, S, rt_win, ids, rt_win ? rt_rows : 1, false, int4,
                           knob::rt_check(), st);
        }
        s2_init(m, s2sess, st);
        s2_init(D.d, s2sess, st);
        if (rtt && rtt->local)
        {
            m.s2->rt_window = D.d.s2->rt_window = rtt->local;
            RtTier* t = rtt.get();
            m.s2->rt_tail = [t](const std::vector<int>& list) { return rt_tail_of(*t, list); };
        }
        m.s2on = D.d.s2on = false;
    }
#if G6_MODEL_FLASH
    // Flash with several sessions: the S2 tables on the --cap pool, main and draft (S2 below R with GR8_CP). The solo graphs below
    // are captured with s2on off, the batched ones by s2/flash_native.cuh with s2on on
    if (s2mode && s2sess > 1)
    {
#if defined(G6_CP) && G6_CP
        // GR8_CP: S2 below R, the pool's page classes (serve_flash): runs below R on the replicated prefix, split runs past it
        if (cp_on) m.s2_below = D.d.s2_below = cp_R;
#endif
        s2_init(m, s2sess, st);
        s2_init(D.d, s2sess, st);
        m.s2on = D.d.s2on = false;
    }
#endif
    u64 ls0 = 1;
    double pf_ms = 0;
    long pf_rows = 0;
    m.nc = nc; D.d.nc = nc;             // before any capture: the head kernels take it by value
#if G6_MODEL_DS41
    DS.nc = nc;
    if (dsk.relaxed)                    // before any capture: the main (verify) head only; DS's draft head stays exact
    {
        m.rx.on = 1;
        m.rx.log_ratio = dsk.relax_ratio > 0 ? (float)std::log(dsk.relax_ratio) : -INFINITY;
        m.rx.topp = (float)dsk.relax_topp;
        if (cfg.rank == 0) fprintf(stderr, "# rank 0: GR8_DS_RELAXED_ACCEPT on ratio %g topp %g (not exact)\n", dsk.relax_ratio, dsk.relax_topp);
    }
#endif
    if (grammar_ready())                // before any capture: the main head's grammar mask (rank.cuh k_grammar_mask)
    {
        m.gm_words = grammar_words();
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&m.gm_on_host), sizeof(int) * MX, cudaHostAllocMapped));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&m.gm_bits_host), sizeof(uint32_t) * MX * m.gm_words, cudaHostAllocMapped));
        memset(m.gm_on_host, 0, sizeof(int) * MX);
        int* on;
        uint32_t* bits;
        G6RT_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&on), m.gm_on_host, 0));
        G6RT_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&bits), m.gm_bits_host, 0));
        m.gm_on = on;
        m.gm_bits = bits;
    }
    if (s2mode)                         // before any capture: the main head's penalties (penalty.h, rank.cuh k_penalty)
    {
        if (m.vn <= 0) die("penalty tables before the vocab slice");
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&m.pen_rows_host), sizeof(PenRow) * MX, cudaHostAllocMapped));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&m.pen_sess_host), sizeof(PenSess) * S2N, cudaHostAllocMapped));
        for (int r = 0; r < MX; ++r) m.pen_rows_host[r] = PenRow{-1, 0, 0, -1, 0.f, 0.f, 0.f, 0.f, 0, 0, 0.f, 0.f, {-1, -1, -1}, {0.f, 0.f, 0.f}};
        memset(m.pen_sess_host, 0, sizeof(PenSess) * S2N);
        PenRow* rows;
        PenSess* sess;
        G6RT_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&rows), m.pen_rows_host, 0));
        G6RT_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&sess), m.pen_sess_host, 0));
        m.pen_rows = rows;
        m.pen_sess = sess;
        m.pen_counts = dmalloc<int>((size_t)S2N * m.vn);
        G6RT_CK(cudaMemset(m.pen_counts, 0, sizeof(int) * (size_t)S2N * m.vn));
        // the head writes penalty.s2 next to ready.s2, so the shim offers the pen, boost and seq fields
        if (R == 0)
            if (FILE* f = fopen((s2dir + "/penalty.s2").c_str(), "w")) { fputs("presence frequency boost seq\n", f); fclose(f); }
    }
    m.rs = plan.rs;                     // the target's verify steps only; the MTP layer's head stays as it is
    if (!pf_qo && !pf_qp)               // QO attention builds the decode RoPE table; without it absorb would take FP64 sincos
        if (int e = g6_attn_rope_init(cap, st)) die("g6_attn_rope_init failed (" + std::to_string(e) + ")");
    boot("prefill and draft built");
    for (int M : row_counts) prepare(m, M, st);
    for (int M : draft_rows) prepare(D.d, M, st);
#if G6_MODEL_DS41
    for (int ns = 2; ns <= m.ds_slots && ns <= MX; ++ns) prepare(m, ns, st);
    if (dsk.on())                       // the draft step is the 5-row block of each of S = 1..dsp.nsess sessions
        for (int S = 1; S <= DS.dsp.nsess; ++S) prepare(DS, S * g6s2::Ds41Round::BLOCK, st);
#endif
#if G6_MODEL_FLASH
    if (mtp && MP) for (int M = 1; M <= MX; ++M) prepare(D.d, M, st);   // the draft prefill's MX-row chunks
#endif
    if (mtp && (l2pf::cfg().mtp & 2)) l2pf_set_front(D);   // before any capture: the head-merge tails read it
    G6RT_CK(cudaStreamSynchronize(st));
    boot("layers prepared");
    // Offline main-step dumps, compatible with m4sim after merging the rank logits.
    // The callback only enqueues D2D copies, so it also works inside split graphs.
    const bool rd_on = R == 0 && knob::resid_dump();
    const int rd_slots = 2 * NL + 1;
#if G6_MODEL_FLASH || G6_MODEL_DS41
    const int rd_hid = m.resid4 ? geom::HC * HID : HID;
#if G6_MODEL_DS41
    const float* rd_source = m.resid;
#else
    const float* rd_source = m.resid4 ? m.resid4 : m.resid;
#endif
#else
    const int rd_hid = HID;
    const float* rd_source = m.resid;
#endif
    float* rdump = rd_on ? dmalloc<float>((size_t)rd_slots * MX * rd_hid) : nullptr;
    std::function<void(int,int)> residual_copy;
    if (rd_on) residual_copy = [&](int slot, int M) {
#if G6_MODEL_DS41
        if (m.resid4) ds_mhc::trunk_to_f32(m.resid4, rdump + (size_t)slot * MX * rd_hid, (size_t)M * rd_hid, st);
        else
#endif
        G6RT_CK(cudaMemcpyAsync(rdump + (size_t)slot * MX * rd_hid, rd_source,
                               (size_t)M * rd_hid * sizeof(float), cudaMemcpyDeviceToDevice, st));
    };
    FILE* rdf = rd_on ? fopen(knob::resid_dump()->c_str(), "wb") : nullptr;
    if (rd_on && !rdf) die("GR8_RESID_DUMP: cannot open the file");
    FILE* ldump = nullptr;
    auto dump_write = [&](FILE* f, const void* data, size_t bytes) {
        if (fwrite(data, 1, bytes, f) != bytes) die("main-step dump: write failed");
    };
    if (const auto& path = knob::logit_dump()) {
        const std::string name = *path + ".rank" + std::to_string(R);
        ldump = fopen(name.c_str(), "wb");
        if (!ldump) die("GR8_LOGIT_DUMP: cannot open " + name);
        // Versioned rank header: magic, NR, rank, vocab, first vocab row, vocab rows.
        // bind::vocab_slice is contiguous and unrotated, including uneven TP widths.
        dump_write(ldump, "G6LGR001", 8);
        const int32_t hdr[] = {g6nr::NR, R, geom::VOCAB, m.vlo, m.vn};
        dump_write(ldump, hdr, sizeof hdr);
    }
#if G6_MODEL_FLASH
    // MTP first draft passes only, as m4sim: rank 0 writes GR8_DRAFT_IN_DUMP; every
    // rank writes its vocab slice to GR8_DRAFT_LOGIT_DUMP.rank<R> in the GR8_LOGIT_DUMP rank format (the step field is
    // the first-pass counter), so tools/cp/merge_logits.py gives m4sim's draft logit framing.
    FILE* didump = nullptr;
    FILE* dldump = nullptr;
    int dd_step = 0;
    if ((knob::draft_in_dump() || knob::draft_logit_dump()) && (!mtp || s2mode))
        die("GR8_DRAFT_IN_DUMP / GR8_DRAFT_LOGIT_DUMP need --mtp and no --serve");
    if (const auto& path = knob::draft_in_dump(); path && R == 0) {
        didump = fopen(path->c_str(), "wb");
        if (!didump) die("GR8_DRAFT_IN_DUMP: cannot open the file");
    }
    if (const auto& path = knob::draft_logit_dump()) {
        const std::string name = *path + ".rank" + std::to_string(R);
        dldump = fopen(name.c_str(), "wb");
        if (!dldump) die("GR8_DRAFT_LOGIT_DUMP: cannot open " + name);
        dump_write(dldump, "G6LGR001", 8);
        const int32_t hdr[] = {g6nr::NR, R, geom::VOCAB, D.d.vlo, D.d.vn};
        dump_write(dldump, hdr, sizeof hdr);
    }
    auto draft_dump = [&](const StepIn& di) {
        const int M = di.M;
        const int32_t hdr[] = {dd_step++, M};
        if (didump) {
            dump_write(didump, hdr, sizeof hdr);
            dump_write(didump, di.pos, M * sizeof(int));
            dump_write(didump, di.tok, M * sizeof(int));
            std::vector<float> h((size_t)M * HID);
            G6RT_CK(cudaMemcpy(h.data(), D.main_xn, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
            dump_write(didump, h.data(), h.size() * sizeof(float));
            if (fflush(didump)) die("GR8_DRAFT_IN_DUMP: flush failed");
        }
        if (dldump) {
            dump_write(dldump, hdr, sizeof hdr);
            std::vector<float> lg((size_t)M * D.d.vn);
            G6RT_CK(cudaMemcpy(lg.data(), D.d.hlog, lg.size() * sizeof(float), cudaMemcpyDeviceToHost));
            dump_write(dldump, lg.data(), lg.size() * sizeof(float));
            if (fflush(dldump)) die("GR8_DRAFT_LOGIT_DUMP: flush failed");
        }
    };
#endif
    const int dump_from = (int)knob::logit_dump_from();
    auto main_dump = [&](int s, int M) {
        if (s < dump_from) return;
        if (rdf) {
            const int32_t hdr[] = {s, M, rd_slots, rd_hid};
            dump_write(rdf, hdr, sizeof hdr);
            std::vector<float> h((size_t)M * rd_hid);
            for (int k = 0; k < rd_slots; ++k) {
                G6RT_CK(cudaMemcpy(h.data(), rdump + (size_t)k * MX * rd_hid,
                                  h.size() * sizeof(float), cudaMemcpyDeviceToHost));
                dump_write(rdf, h.data(), h.size() * sizeof(float));
            }
            if (fflush(rdf)) die("GR8_RESID_DUMP: flush failed");
        }
        if (ldump) {
            const int32_t hdr[] = {s, M};
            dump_write(ldump, hdr, sizeof hdr);
            std::vector<float> h((size_t)M * m.vn);
            G6RT_CK(cudaMemcpy(h.data(), m.hlog, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
            dump_write(ldump, h.data(), h.size() * sizeof(float));
            if (fflush(ldump)) die("GR8_LOGIT_DUMP: flush failed");
        }
    };
    cudaGraphExec_t exec[NBK][MX + 1] = {}, exec_d[NBK][2][MX + 1] = {};      // [context bucket][...]
    // DEC_GSPLIT: GR8_DEC_SPLIT=ls (1..nl-1) captures the target step as two graphs, exec = step_in, embed, layers [0, ls) and
    // exec2 = the rest; launched back to back, the second graph's submission overlaps the first one's GPU time (0 = one graph)
    const int gsplit = (int)knob::dec_split();
    if (gsplit >= NL || (gsplit && !graph)) die("GR8_DEC_SPLIT must be 0..layers-1 and needs --graph");
    if (s2mode && !G6_MODEL_FLASH && !G6_MODEL_DS41 && gsplit != 3) die("serving needs GR8_DEC_SPLIT=3 (the production split3 graphs)");
#if G6_MODEL_DS41
    // The Engram side stream forks at step start (graph A) and joins at layers 1 and 14 by an event, so the last join must
    // be in graph A: graph A runs the units of layers [0, split), and the plan puts layer j's Engram in layer j-1's
    // consume unit, so split 0 or split >= j (j = index of the last Engram layer).
    if (gsplit && m.engram)
    {
        int eng_last = -1;
        for (int i = 0; i < NL; ++i) if (geom::is_engram(layers[i])) eng_last = i;
        if (gsplit < eng_last)
            die("GR8_DS_ENGRAM=1 needs GR8_DEC_SPLIT=0 or >= " + std::to_string(eng_last) + " (the last Engram layer's index): the fetch forks and joins inside graph A");
    }
#endif
    cudaGraphExec_t exec2[NBK][MX + 1] = {};
#if G6_MODEL_DS41
    const bool ds_moe_decode_graph = s2mode && ds_moe_decode_topk != ds_moe_main_topk;
    cudaGraphExec_t exec_dec[NBK][MX + 1] = {};
    // SESS > 1: the steps of ns = 2..sessions sessions, one row each (Model::ds_ns = ns at capture: the comp-pool grid and the index loop per session)
    // DSpark: [k][ns][rps] = ns sessions of rps = 1..GR8_DS_MTP + 1 rows each (the verify of k' + 1 rows, for ns <= 8)
    cudaGraphExec_t exec_ms[NBK][33][7] = {}, exec2_ms[NBK][33][7] = {};
    cudaGraphExec_t exec_ms_dec[NBK][33][7] = {};
#endif
#if G6_MODEL_DS41
    std::map<const cudaGraphExec_t*, Ds41SpanSet> span_sets;   // GR8_DS_SPAN: the spans of each captured step graph
    auto replayed = [&](const cudaGraphExec_t* ex, int M) {
        if (!ds41_span_on()) return;
        const auto it = span_sets.find(ex);
        if (it != span_sets.end()) ds41_span_replayed(it->second, m.ds_ns > 1 ? "sessions " : "", M, st);
    };
#endif
    auto launch_main = [&](int k, int M) {
#if G6_MODEL_DS41
        const bool decode_graph = ds41_moe_topk::graph_kind(m.ds_moe_decode_step, ds_moe_main_topk, ds_moe_decode_topk);
        if (m.ds_ns > 1)
        {
            const int ns = m.ds_ns, rps = M / std::max(1, ns);
            g6s2::require(ns <= m.ds_slots && ns <= 32 && rps >= 1 && rps <= 6 && (rps == 1 || ns <= (ds_sess_drafts ? 32 : 8)) && rps * ns == M &&
                          (decode_graph ? exec_ms_dec[k][ns][rps] : exec_ms[k][ns][rps]), "DS41 SESS graph not captured");
            G6RT_CK(cudaGraphLaunch(decode_graph ? exec_ms_dec[k][ns][rps] : exec_ms[k][ns][rps], st));
            if (gsplit) G6RT_CK(cudaGraphLaunch(exec2_ms[k][ns][rps], st));
            replayed(decode_graph ? &exec_ms_dec[k][ns][rps] : &exec_ms[k][ns][rps], M);
            return;
        }
        if (decode_graph) {
            g6s2::require(exec_dec[k][M], "DS41 decode graph not captured");
            G6RT_CK(cudaGraphLaunch(exec_dec[k][M], st));
            replayed(&exec_dec[k][M], M);
            return;
        }
#endif
        G6RT_CK(cudaGraphLaunch(exec[k][M], st));
        if (gsplit) G6RT_CK(cudaGraphLaunch(exec2[k][M], st));
#if G6_MODEL_DS41
        replayed(&exec[k][M], M);
#endif
    };
    auto capture = [&](const std::function<void()>& body, cudaGraphExec_t* ex, const char* what, int M) {
        cudaGraph_t g;
#if G6_MODEL_DS41
        if (ds41_span_on()) ds41_span_begin(span_sets[ex]);
#endif
        G6RT_CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        body();
#if G6_MODEL_DS41
        ds41_span_end(st);
#endif
        G6RT_CK(cudaStreamEndCapture(st, &g));
        size_t nodes = 0;
        G6RT_CK(cudaGraphGetNodes(g, nullptr, &nodes));
        G6RT_CK(cudaGraphInstantiate(ex, g, 0));
        G6RT_CK(cudaGraphDestroy(g));
        fprintf(stderr, "# rank %d: %sgraph for M=%d bucket %d, %zu nodes\n", R, what, M, m.bk[m.cur], nodes);
    };
    if (graph)
    {
        for (int k = 0; k < m.nbk; ++k)
        {
            m.cur = k;
            for (int M : row_counts)
                if (gsplit)
                {
                    capture([&] { launch_step(m, M, 0, st, 0, gsplit, residual_copy); }, &exec[k][M], "split part A ", M);
                    capture([&] { launch_step(m, M, 0, st, 1, gsplit, residual_copy); }, &exec2[k][M], "split part B ", M);
                }
                else capture([&] { launch_step(m, M, 0, st, -1, 0, residual_copy); }, &exec[k][M], "", M);
#if G6_MODEL_DS41
            if (ds_moe_decode_graph) {
                m.ds_moe_decode_step = true;
                for (int M : row_counts)
                    capture([&] { launch_step(m, M, 0, st, -1, 0, residual_copy); }, &exec_dec[k][M], "decode ", M);
                m.ds_moe_decode_step = false;
            }
            for (int ns = 2; ns <= m.ds_slots && ns <= MX; ++ns)
                for (int rps = 1; rps <= ds_ms_rps(ns) && ns * rps <= MX; ++rps)
                {
                    const int M = ns * rps;
                    m.ds_ns = ns;
                    if (gsplit)
                    {
                        capture([&] { launch_step(m, M, 0, st, 0, gsplit, residual_copy); }, &exec_ms[k][ns][rps], "sessions split part A ", M);
                        capture([&] { launch_step(m, M, 0, st, 1, gsplit, residual_copy); }, &exec2_ms[k][ns][rps], "sessions split part B ", M);
                    }
                    else capture([&] { launch_step(m, M, 0, st, -1, 0, residual_copy); }, &exec_ms[k][ns][rps], "sessions ", M);
                    if (ds_moe_decode_graph) {
                        m.ds_moe_decode_step = true;
                        capture([&] { launch_step(m, M, 0, st, -1, 0, residual_copy); }, &exec_ms_dec[k][ns][rps], "sessions decode ", M);
                        m.ds_moe_decode_step = false;
                    }
                    m.ds_ns = 1;
                }
#endif
        }
#if G6_MODEL_DS41
        if (dsk.on())                   // one draft graph per session count: the DS41 layers read the context from their step cells, not from a bucket
        {
            DS.cur = 0;
            for (int S = 1; S <= DS.dsp.nsess; ++S)
            {
                DS.ds_ns = S;
                capture([&] { g6s2::launch_step_ds41_draft(DS, S * g6s2::Ds41Round::BLOCK, st); }, &exec_ds[S], "dspark draft ", S * g6s2::Ds41Round::BLOCK);
            }
            DS.ds_ns = 1;
        }
#endif
        if (mtp && D.d.nbk != m.nbk) die("draft and target context buckets differ");
        for (int k = 0; mtp && k < D.d.nbk; ++k)
        {
            D.d.cur = m.cur = k;
            for (int M : draft_rows) capture([&] { launch_draft_step(D, M, 0, false, st); }, &exec_d[k][0][M], "first draft ", M);
            if (mtp > 1) capture([&] { launch_draft_step(D, 1, 0, true, st); }, &exec_d[k][1][1], "chain draft ", 1);
        }
    }
    FILE* dfl_dump = dfl && R == 0 && knob::dfl_dump() ? fopen(knob::dfl_dump()->c_str(), "wb") : nullptr;   // dflash.cuh dfl_dump_step
    cudaGraphExec_t exec_f[3] = {};     // --dflash: [0] the context part only, [1] the whole draft step, [2] a fill step
    if (graph && dfl)
    {
        capture([&] { dfl_step(DF, st, false); }, &exec_f[0], "dflash context ", dfl::B);
        capture([&] { dfl_step(DF, st, true); }, &exec_f[1], "dflash draft ", dfl::B);
        if (DF.pf_feat) capture([&] { dfl_fill_step(DF, st); }, &exec_f[2], "dflash fill ", dfl::B);
    }
    cudaStream_t st_par[2] = {};        // GR8_DRAFT_PAR: the drafter's streams, [0] context only, [1] drafting
    if (draft_par)
    {
        int lo = 0, hi = 0;
        G6RT_CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        G6RT_CK(cudaStreamCreateWithFlags(&st_par[0], cudaStreamNonBlocking));
        if (draft_par == 2) G6RT_CK(cudaStreamCreateWithPriority(&st_par[1], cudaStreamNonBlocking, hi));
        else st_par[1] = st_par[0];
        if (R == 0) fprintf(stderr, "# rank %d: GR8_DRAFT_PAR=%d: the DFlash2 drafter step runs under the MTP first form "
                            "(drafting stream priority %d, range %d..%d)\n", R, draft_par, draft_par == 2 ? hi : 0, lo, hi);
    }
    fprintf(stderr, "# rank %d: device tensors %.2f GB\n", R, (image ? img.bytes() : store.bytes) / 1e9);
    const bool pf_streams = two && knob::pf_streams() == 2;   // pf_step2s
    if (pf_streams && pf_prof().on && !pf_prof().ev) die("GR8_PF_STREAMS=2 cannot run with GR8_PF_PROF (the marks synchronize one stream)");
    PfLanes lanes2;
    if (pf_streams)
    {
        pf_lanes_init(lanes2, m.nl);
        fprintf(stderr, "# rank %d: prefill lanes on two streams\n", R);
    }
    if (MP && pf_rs) fprintf(stderr, "# rank %d: prefill reduce-scatter form (5 collectives, slice %zu B)%s\n", R, pf_ss(fp),
                             pf_own >= 4 ? ", pair-owner QO attention (GR8_PF_OWN=4)" : pf_own >= 3 ? ", row-owner QO attention (GR8_PF_OWN=3)" : pf_own >= 2 ? ", row-owner projections + index (GR8_PF_OWN=2)" : pf_own ? ", row-owner projections (GR8_PF_OWN)" : "");
    {   // every proxy thread is running: keep all other threads of this process off their CPUs (GR8_PROGRESS_ISOLATE=0 turns it off)
        const std::vector<int> pc = g6::progress_cpus();
        if (progress_isolate && !pc.empty())
        {
            const int moved = g6::isolate_progress_cpus(pc, g6::progress_tids());
            std::string list;
            for (int c : pc) list += (list.empty() ? "" : ",") + std::to_string(c);
            fprintf(stderr, "# rank %d: progress CPUs %s reserved for the proxy threads; %d other thread(s) moved off them\n", R, list.c_str(), moved);
        }
    }
    boot("graphs captured");
    fab.barrier();
    if (R == 0) printf("# m4 layers %s, %zu sequences, chunks of %d, gen %d temp %g seed %u%s\n", layers_s.c_str(), plans.size(), plan.chunk, plan.n_gen, temp, plan.seed,
                      plan.rs ? ", spec-accept rejection" : "");

#if G6_MODEL_FLASH
    if (s2mode)
    {
        g6s2::BootstrapConfig bc;
        auto& sc = bc.scheduler;
        sc.epoch = s2epoch; sc.pool_rows = cap / 64 * 64;
        sc.max_rows = std::min(MX, std::max({s2sess * (mtp + 1), plan.chunk, dfl ? dfl_k + 1 : 0}));   // a round of every session, a prompt chunk, a DFlash2 round
        sc.max_sessions = s2sess; sc.mtp = mtp; sc.prefill_rows = MP; sc.two_lanes = two;
        if (dfl) sc.guard_tokens = 2 * dfl::K;     // a DFlash2 verify writes up to 7 rows past the last token
        // the prefill/decode interleave, as the GLM block reads it: while sessions decode, prefill quanta of at most
        // GR8_S2_DECODE_PF rows on one lane and GR8_S2_DECODE_ROUNDS decode rounds per quantum (scheduler.h Config).
        // Unset, a long prompt runs its quanta with one decode round between them, so the other sessions wait it out
        {
            const int v = (int)knob::s2_decode_pf();
            if (v < 0 || v % 64 || v > MP) die("S2: GR8_S2_DECODE_PF must be 0 or a multiple of 64 no larger than --prefill");
            sc.decode_prefill_rows = (uint32_t)v;
        }
        sc.decode_rounds = (uint32_t)knob::s2_decode_rounds();
        sc.tail_first = knob::s2_tail_first();
        sc.notice_max = (uint32_t)knob::s2_notice_max();
        if (R == 0) fprintf(stderr, "# flash s2 interleave: decode_pf %u rows, decode_rounds %u, tail_first %d, notice_max %u\n",
                            sc.decode_prefill_rows, sc.decode_rounds, sc.tail_first ? 1 : 0, sc.notice_max);
        bc.rank = R; bc.ranks = cfg.ips.size(); bc.head_ip = cfg.ips[0];
        bc.directory = s2dir; bc.port = s2port;
        bc.timeout = (unsigned)knob::s2_timeout();
        g6s2::SigInputs sig;               // s2/signature.h
        sig.model = g6s2::SigModel::Flash;
        sig.cap = cap; sig.layers_s = layers_s; sig.MP = MP; sig.lanes = lanes; sig.mtp = mtp; sig.chunk = plan.chunk; sig.nc = nc;
        sig.set_id = set_id; sig.dfl = dfl; sig.dfl_k = dfl_k; sig.dfl_reuse = dfl_reuse; sig.dfl_reuse_min = dfl_reuse_min;
        sig.s2sess = s2sess; sig.gsplit = gsplit; sig.knobs = knob::signature();
        sig.vis_on = vis_on; sig.vis_id = vis_id;
        // GR8_GLM53F_CACHE: the checkpoint store of this rank. Every rank must keep the same entries, so the mode and
        // the store's blocks join the signature
        std::unique_ptr<g6s2::FlashCheckpoints> ckpt;
        if (flash_cache != g6s2::FlashCache::Off)
        {
            g6s2::FlashStateView v;
            for (g6rt::Model* x : {&m, &D.d})
                for (g6rt::Layer& y : x->layers)
                    if (!y.kda && y.ab.storage)
                    {
                        int cpb = 0;   // a CP binding keeps this rank's pages only (flash_checkpoint.cuh)
#if defined(G6_CP) && G6_CP
                        cpb = y.ab.args.cp_nr ? 1 : 0;
#endif
                        v.bindings.push_back({static_cast<uint4*>(y.ab.args.mla_cache), static_cast<uint4*>(y.ab.args.index_cache),
                                              (unsigned long long)(64 * g6kv::ROW / 16), (unsigned long long)(g6kv::IDX_PAGE / 16), cpb});
                        v.rings.push_back(static_cast<uint8_t*>(y.ab.args.index_raw));
                    }
            v.ring_bytes = (size_t)g6kv::RAW_RING * g6kv::IDX_RAW;
            v.S = m.kda_st.S; v.conv = m.kda_st.conv; v.layers = kda::N_KDA; v.slots = m.kda_st.slots; v.pool_pages = int(sc.pool_rows / 64);
            v.s_slot = m.kda_st.s_layer() / std::max(1, m.kda_st.slots); v.c_slot = m.kda_st.c_layer() / std::max(1, m.kda_st.slots);
            std::string spec = "mem:";
            uint64_t bytes = (uint64_t)flash_budget_mib << 20;
            std::optional<g6s2::FlashGroup> group;
            std::string index_path;
            if (!kv_tiers.empty())
            {
                g6s2::TierSpec t;
                try { t = g6s2::TierSpec::parse(kv_tiers[0], false); } catch (const std::exception& e) { die(std::string("GR8_GLM53F_CACHE: ") + e.what()); }
                if (t.persist && flash_cache == g6s2::FlashCache::Terminal)
                    die("GR8_GLM53F_CACHE=terminal: persist=on keeps cold-exact (aligned) checkpoints only");
                spec = t.store; bytes = t.bytes;
                // group=: the lease shares its KV node with other clusters (s2/flash_checkpoint.h); it starts at size=
                if (!t.group.empty()) group = g6s2::FlashGroup{t.group, (uint32_t)g6nr::NR, (uint32_t)R, t.cap, t.weight};
                if (t.ssd())
                {   // as the pool's stores: the folder starts empty at every boot unless persist=on keeps it and its index
                    const std::string dir = t.store.substr(4) + "/r" + std::to_string(R);
                    if (!t.persist && std::filesystem::exists(dir)) std::filesystem::remove_all(dir);
                    std::filesystem::create_directories(dir);
                    spec = "ssd:" + dir + "/flash-ckpt";
                    if (t.persist) index_path = dir + "/flash-ckpt.index";
                }
                if (t.spare_ram) spec = "ram:" + std::to_string(t.spare_ram_floor) + "+" + spec;
            }
            char host[64] = "host";
            gethostname(host, sizeof host - 1);
            // its own lease name on a KV node: never the pool stash's (g6m4-...)
            const std::string name = std::string("g6m4f-ckpt-") + host + "-" + std::to_string(s2epoch) + "-r" + std::to_string(R);
            static const std::string exe = g6s2::exe_stamp();
            // the cache mode is not in it: the launcher (tools/launch/options.py) and the check above refuse persist=on
            // with terminal, so a persisted entry is always aligned (cold-exact)
            const std::string compat = "flash set " + set_id + " nr " + std::to_string(g6nr::NR) + " rank " + std::to_string(R) +
                " cap " + std::to_string(cap) + " mp " + std::to_string(MP) + " lanes " + std::to_string(lanes) + " bindings " +
                std::to_string(v.bindings.size()) + " state " + std::to_string(v.state_bytes()) + " format " +
                std::to_string(g6s2::FlashManifest::FORMAT) + " engine " + exe;
            try { ckpt.reset(new g6s2::FlashCheckpoints(v, spec, name, bytes, compat, group ? &*group : nullptr, index_path)); }
            catch (const std::exception& e) { die(std::string("GR8_GLM53F_CACHE: ") + e.what()); }
            // persist=on: the entries are checked on a helper thread and agreed by every rank in serve_flash
            if (!index_path.empty()) fprintf(stderr, "# rank %d: flash ckpt persist: %s: %s\n", R, index_path.c_str(), ckpt->loaded.c_str());
            fprintf(stderr, "# rank %d: flash ckpt %s: %s, %llu blocks of %zu B (%zu bindings, KDA state + raw rings %.1f MB = %llu blocks), grid %d rows%s\n",
                    R, g6s2::flash_cache_name(flash_cache), ckpt->describe().c_str(), (unsigned long long)ckpt->blocks(), ckpt->slot_bytes(),
                    v.bindings.size(), v.state_bytes() / 1e6, (unsigned long long)(ckpt->entry_blocks(0) - 1), MP * (two ? 2 : 1),
                    group ? (", group " + group->group + " lease, " + std::to_string(ckpt->segments()) + " segments to start").c_str() : "");
            // persist: every rank joins the agreement on persisted entries (serve_flash), or none does. The index itself is
            // not here: the agreement makes it the same on every rank, and the pool digest compares it at every event
            sig.ckpt = g6s2::flash_cache_name(flash_cache);
            sig.ckpt_blocks = ckpt->blocks();
            sig.ckpt_persist = !index_path.empty();
        }
        bc.runtime_signature = g6s2::runtime_signature(sig);
        // DFlash2 (--dflash-manifest): a session alone drafts with the drafter, as on GLM-5.3 (s2_batch.h Batch::dflash);
        // Dfl::pf_feat holds the prefill taps of the session whose prefill started last (s2/flash_dfl_window.h: with
        // GR8_GLM53F_CACHE, a checkpoint hit inside the window keeps the drafter through GR8_DFL_REUSE)
        g6s2::FlashDflWindow dwin((int)g6s2::MAX_SESSIONS, dfl::WIN, dfl_reuse);
        u64 s2_dstep = 0;
        cudaStream_t dfl_ds = st;        // GR8_DRAFT_PAR: the stream of the drafter step in flight
        auto dfl_install = [&](g6s2::NativeHooks& h, g6s2::FlashExtras& ext) {
            h.dfl_kv = dfl_k; h.dfl_spec = spec;
            h.dsel.on = s2::dsel_env();
            h.dsel.margin = knob::s2_dsel_margin();
            h.dsel.dwell = (int)std::max<int64_t>(1, knob::s2_dsel_dwell());
            h.dsel.probe = (int)std::max<int64_t>(1, knob::s2_dsel_probe());
            h.dsel.d.draft_round = 5.3;         // DFlash2: the drafter step and the MTP first form (the measured gap)
            h.dsel.m.draft_round = 2.6 + 1.0;   // MTP: the first form and the drafter's context-only step
            h.dsel.m.draft_each = 1.25;
            for (int r = 1; r <= SpecLen::KX + 1; ++r) h.dsel.d.table[r] = h.dsel.m.table[r] = spec.table[r];
            h.kstop = kstop;
            h.dfl_reuse.mode = dfl_reuse;
            h.dfl_reuse.min = dfl_reuse_min;
            ext.bind = [&](const g6s2::Grant& g) {   // after serve_flash's bind, which restores a checkpoint hit's rows first
                dwin.bind((int)g.key.slot, g.req.prompt_tokens, g.reuse);
            };
            ext.prefill = [&](const g6s2::Slot& sl) {
                // a request's first prefill quantum: the window its taps fill
                if (DF.pf_feat && dwin.quantum((int)sl.key.slot, sl.req.id, sl.req.prompt_tokens, sl.prefilled))
                {
                    DF.pf_hi = dwin.hi; DF.pf_lo = dwin.lo;
                    if (DF.pf_hi > DF.pf_lo) G6RT_CK(cudaMemsetAsync(DF.pf_feat, 0, (size_t)(DF.pf_hi - DF.pf_lo) * dfl::NT * HID * 2, st));
                }
            };
            // a session alone takes the DFlash2-wide KDA ring at its first round (s2/flash_native.cuh widen)
            ext.dfl_ok = [&](int q) { return dwin.ok(q); };
            ext.dfl_start = [&](int q) {     // the prefill's taps -> the drafter's context ring, 8 rows a fill step
                if (!dwin.fills(q)) return dwin.start(q);
                for (int c0 = dwin.from; c0 < DF.pf_hi; c0 += dfl::B)
                {
                    DflIn di;
                    memset(&di, 0, sizeof di);
                    di.step = ++s2_dstep;
                    di.ctx_rows = std::min(dfl::B, DF.pf_hi - c0); di.ctx_pos0 = c0; di.q0 = c0 + di.ctx_rows; di.lo = DF.pf_lo;
                    *in_f = di;
                    dfl_enqueue(fabf, di.step, dfl_fill_sizes(), true);
                    G6RT_CK(cudaGraphLaunch(exec_f[2], st));
                    G6RT_CK(cudaStreamSynchronize(st));
                    g6s2::require(!*fabf.dev().error, "Flash DFlash2 fabric error (prompt context)");
                }
                return dwin.start(q);
            };
            ext.dfl = [&](const s2::DflReq& r, const g6s2::Sampling& smp) {
                // GR8_DRAFT_PAR: phase 1 launches on the drafter's stream and returns, phase 2 waits for it (s2::DflReq)
                if (r.phase != 2)
                {
                    DflIn di;
                    memset(&di, 0, sizeof di);
                    di.step = ++s2_dstep;
                    di.ctx_rows = r.rows; di.ctx_pos0 = r.pos0; di.q0 = r.pos0 + r.rows; di.lo = r.lo; di.tok0 = r.tok0; di.query = r.query;
                    for (int i = 0; i < dfl::B; ++i) di.srow[i] = glue::SampleRow{smp.inv_temp, smp.seed, 0u, di.q0 + i, 0, smp.top_p};
                    *in_f = di;
                    dfl_enqueue(fabf, di.step, dfl_sizes(DF, di.query), false);
                    dfl_ds = r.phase ? st_par[di.query ? 1 : 0] : st;
                    G6RT_CK(cudaGraphLaunch(exec_f[di.query ? 1 : 0], dfl_ds));
                    if (r.phase == 1) return s2::DflDrafts{};
                }
                G6RT_CK(cudaStreamSynchronize(dfl_ds));
                g6s2::require(!*fabf.dev().error, "Flash DFlash2 fabric error");
                s2::DflDrafts o;
                for (int i = 0; i < dfl::B; ++i) { o.tok[i] = out_f->tok[i]; o.lp[i] = out_f->lp[i]; }
                return o;
            };
            if (R == 0) fprintf(stderr, "# flash serve: DFlash2 for a session alone (k <= %d)%s%s\n", dfl_k,
                                h.dsel.on ? ", drafter choice on" : "",
                                dfl_reuse ? (", checkpoint hits keep the drafter" + std::string(dfl_reuse > 1 ? ", a session alone on MTP takes it" : "") +
                                             " (GR8_DFL_REUSE " + std::to_string(dfl_reuse) + ", warm-up " + std::to_string(dfl_reuse_min) +
                                             " context rows)").c_str() : "");
        };
        // GR8_GLM53F_KV_CHUNKS=1 (borrow-idle): hooks serve_flash adds to its own before the service starts
        auto flash_install = [&](g6s2::NativeHooks& h, g6s2::FlashExtras& ext) {
            if (dfl) dfl_install(h, ext);
            // GR8_GLM53F_KV_CHUNKS=1: a Resize event on tier FLASH_KV_TIER takes every model's chunk map to r.segs chunks
            // online, on every rank at the commit (flash_kv.h; with GR8_GLM53F_KV_LEDGER the bytes move through the box's
            // host ledger). The head announces them under FLASH_CAP_POLICY=borrow-idle (GR8_GLM53F_KV_LEND_MS > 0, s2/flash_lend.h):
            // a lend down to GR8_GLM53F_KV_IDLE_CHUNKS once idle that long, a recall before a request that needs more. Every
            // rank's ACK carries its chunks online, and the head admits only within the least of them
            if (m.fkv.pages)
            {
                if (const auto& l = knob::glm53f_kv_ledger()) for (g6rt::Model* x : {&m, &D.d}) x->fkv.ledger = *l;
                const long lend_ms = (long)knob::glm53f_kv_lend_ms(), keep = (long)knob::glm53f_kv_idle_chunks();   // >= 0, >= 1 (table)
                ext.flash_room = [&] {
                    uint32_t n = UINT32_MAX;
                    for (g6rt::Model* x : {&m, &D.d}) if (x->fkv.pages) n = std::min(n, x->fkv.online);
                    return n;
                };
                h.flash.idle_ms = lend_ms;
                h.flash.keep = (uint32_t)std::min<long>(keep, m.fkv.chunks);
                h.flash.chunk_rows = (uint64_t)m.fkv.chunk_pages * g6kv::PAGE;
                h.flash.chunks = m.fkv.chunks;
                // the bytes this rank may lend (every chunk above keep, all models): the launcher books them as the GPU's
                // in the host ledger (GR8_GLM53F_KV_LEDGER.gpu, one line), so a lend has them to give
                if (lend_ms && !m.fkv.ledger.empty())
                {
                    uint64_t lendable = 0;
                    for (g6rt::Model* x : {&m, &D.d})
                        if (x->fkv.pages) lendable += x->fkv.bytes_at(x->fkv.chunks) - x->fkv.bytes_at(h.flash.keep);
                    const std::string gp = m.fkv.ledger + ".gpu", tmp = gp + ".tmp";
                    if (FILE* f = fopen(tmp.c_str(), "w"))
                    {
                        fprintf(f, "%llu\n", (unsigned long long)lendable);
                        if (fclose(f) || rename(tmp.c_str(), gp.c_str()))
                            fprintf(stderr, "# rank %d: flash kv: could not write %s\n", R, gp.c_str());
                    }
                    else fprintf(stderr, "# rank %d: flash kv: could not write %s\n", R, gp.c_str());
                    fprintf(stderr, "# rank %d: flash kv: borrow-idle, %u of %u chunks lent after %ld ms idle, %llu bytes "
                                    "lendable\n", R, m.fkv.chunks - h.flash.keep, m.fkv.chunks, lend_ms,
                            (unsigned long long)lendable);
                }
                ext.resize = [&](const g6s2::Resize& r) {
                    if (r.tier != g6s2::FLASH_KV_TIER) return false;
                    for (g6rt::Model* x : {&m, &D.d})
                    {
                        if (!x->fkv.pages) continue;
                        if (int e = x->fkv.resize((uint32_t)r.segs, st))
                            fprintf(stderr, "# rank %d: flash kv: to %llu chunks failed (%d): %s; %u online\n", R,
                                    (unsigned long long)r.segs, e, x->fkv.err.c_str(), x->fkv.online);
                        else
                            fprintf(stderr, "# rank %d: flash kv: %u of %u chunks online (%llu tokens) at %lld ms\n", R,
                                    x->fkv.online, x->fkv.chunks, (unsigned long long)x->fkv.tokens(),
                                    (long long)g6s2::s2_wall_ms());
                    }
                    return true;
                };
            }
        };
        g6s2::FlashContext fc{m, D, FP, FP2, fab, fabd, fabp, fabp2, st, in, in_d, out, out_d,
            [&](int k, int rows) {
                g6s2::require(k >= 0 && k < m.nbk && rows > 0 && rows <= MX && exec[k][rows], "Flash main graph missing");
                launch_main(k, rows);
            },
            [&](int k, bool chain, int rows) {
                g6s2::require(k >= 0 && k < D.d.nbk && rows > 0 && rows <= MX && exec_d[k][chain][rows], "Flash draft graph missing");
                G6RT_CK(cudaGraphLaunch(exec_d[k][chain][rows], st));
            }, gsplit, ckpt.get(), flash_cache == g6s2::FlashCache::Terminal, flash_install};
#if defined(G6_CP) && G6_CP
        if (cp_on) { fc.cp_R = cp_R; fc.cp_enqueue = cp_enqueue; fc.cp_pf_error = cp_pf_err; }
#endif
        if (vis_on) fc.vision = &fvis;
        try { g6s2::serve_flash(fc, bc, plan.chunk); }
        catch (const std::exception& e) { fprintf(stderr, "g6m4: rank %d Flash serve failed: %s\n", R, e.what()); return 3; }
    }
#elif G6_MODEL_DS41
    if (s2mode)
    {
        // the DS41 serve (s2/ds41_native.cuh), with optional headless W-row prefill. The graphs above (exec[k][M], M = 1..chunk,
        // and the ns-session graphs) are the ones it launches; parse time refused every setting it cannot serve
        g6s2::BootstrapConfig bc;
        auto& sc = bc.scheduler;
        sc.epoch = s2epoch; sc.pool_rows = s2sess * (cap / 64 * 64);   // SESS: cap rows per session slot (ds41_native.cuh caps one run at cap)
        sc.max_rows = std::min(MX, std::max({s2sess, std::min(s2sess, 8) * (dsk.mtp + 1), plan.chunk}));
        sc.max_sessions = s2sess; sc.mtp = dsk.mtp;
        // drafting stops past the sessions whose verify and 5 draft rows fit one step (at most 8)
        sc.mtp_sessions = (uint32_t)std::min(sc.max_rows / 5, 8u);
        sc.sess_drafts = ds_sess_drafts;    // GR8_DS_SESS_DRAFTS: drafts a session by active sessions (Batch), mtp_sessions bounds limit() only
        // GR8_DS_PARK=1 (s2/scheduler.h Phase::Parked, ds41_native.cuh): a finished chat keeps its slot for a follow-up
        sc.park = knob::ds_park();
        if (sc.park) fprintf(stderr, "# rank %d ds41 GR8_DS_PARK=1: a finished chat parks its slot (%d slots)\n", R, s2sess);
        if (ds_sess_drafts)
            fprintf(stderr, "# rank %d ds41 GR8_DS_SESS_DRAFTS=1: %d sessions, drafts while at most %d are active, draft launches of at most %d sessions\n", R,
                    s2sess, s2::sess_drafts_max(dsk.mtp, (int)sc.max_rows, s2sess), std::min(m.ds_slots, ds_draft_cap));
        // Boot evidence: this line names the logical page count, the page size, the session count and the row caps, so a check
        // can compare them with the head's /status (free_pages == sessions*cap/64, resident == 0).
        fprintf(stderr, "# rank %d: s2 pool %u logical pages of %d rows, %d sessions, cap %d, max_rows %d\n", R,
                sc.pool_rows / 64, 64, s2sess, cap, sc.max_rows);
        sc.c8_pack = c8_pack_env == 1; // GR8_DS_C8PACK opt-in short-prompt pack; parsed before weight load
        sc.tail_first = knob::s2_tail_first();
        sc.notice_max = (uint32_t)knob::s2_notice_max();
        fprintf(stderr, "# rank %d ds41 s2 scheduler tail_first %d notice_max %u\n", R, sc.tail_first ? 1 : 0, sc.notice_max);
        const int pf_min = (int)knob::ds_prefill_min();
        const bool pf_v2 = knob::ds_pf_v2(); // boot-once, agreed signature
        Ds41V2Engine v2engine;
        const int pf_max = pf_v2 ? ds41::v2::MAX_ROWS : ds41moe::ROMAX;
        // Engram on is allowed with W-row prefill (engine max_rows = W, one chunk in flight -- ds41_prefill_single_lane keeps the engine's one slot safe)
        if (pf_min && (pf_min < 2 || m.hc_rows <= MX || m.hc_rows > pf_max || m.qs.on()))
            die("DS41 W-row prefill needs GR8_DS_PREFILL_MIN>=2, MX<W<=" + std::to_string(pf_max) + ", and QA/QS split off");
        if (pf_min && dsk.prefill_keep != 0)
            die("DS41 W-row prefill needs GR8_DS_PREFILL_KEEP=0");
        // GR8_DS_PF_LANES=2 (ds41_lanes.h): two-lane W prefill, quanta of up to 2W rows of one session; unset / 1 = one lane
        const int pf_lanes = ds41::lanes::parse(DS41_TEXT(knob::ds_pf_lanes()));
        if (!pf_lanes) die(ds41::lanes::kParseError);
        if (pf_lanes == 2 && !pf_min) die("GR8_DS_PF_LANES=2 needs W-row prefill (GR8_DS_PREFILL_MIN)");
        bc.prefill_min = (uint32_t)pf_min;
        if (pf_min) g6s2::ds41_prefill_lanes(sc, (uint32_t)m.hc_rows, pf_lanes);
        // Bound each prefill quantum while another session decodes. The default quantum remains
        // unchanged when this knob is zero; the launcher chooses 1024 for DS41 daily serving.
        const int64_t decode_pf = knob::s2_decode_pf();
        if (decode_pf) {
            if (decode_pf < 0 || decode_pf % 64 || uint64_t(decode_pf) > sc.prefill_rows)
                die("DS41: GR8_S2_DECODE_PF must be a multiple of 64 no larger than the prefill width");
            sc.decode_prefill_rows = (uint32_t)decode_pf;
        }
        bc.rank = R; bc.ranks = cfg.ips.size(); bc.head_ip = cfg.ips[0];
        bc.directory = s2dir; bc.port = s2port;
        bc.timeout = (unsigned)knob::s2_timeout();
        g6s2::SigInputs sig;               // s2/signature.h
        sig.model = g6s2::SigModel::Ds41;
        sig.s2sess = s2sess; sig.cap = cap; sig.layers_s = layers_s; sig.ds_mtp = dsk.mtp; sig.chunk = plan.chunk; sig.nc = nc;
        sig.set_id = set_id; sig.dfl = dfl; sig.dfl_k = dfl_k;
        sig.sh_tc = ds41moe::sh_tc_on(); sig.l2hint = ds41moe::ro_l2hint(); sig.fc2stg = ds41moe::ro_fc2stg(); sig.fc1bstg = ds41moe::ro_fc1bstg();
        sig.c8_pack = sc.c8_pack; sig.sess_drafts = ds_sess_drafts; sig.pf_lanes = pf_lanes;
        sig.tk_main = ds_moe_main_topk; sig.tk_decode = ds_moe_decode_topk;
        sig.idx_pipe_b = ds_idx_pipe_b; sig.idx_q_trig = ds_idx_q_trig; sig.idx_pipe_qr = ds_idx_pipe_qr; sig.reuse = reuse_prefix;
        sig.vis_on = vis_on; sig.vis_id = vis_id;
        sig.knobs = knob::signature();
        bc.runtime_signature = g6s2::runtime_signature(sig);
        // DSpark (GR8_DS_MTP): the draft Model, its fabric, the 5-row draft graph (or the eager step) and the compressor commit of the main Model
        g6s2::Ds41Dspark ds{DS, in_ds, out_ds,
            [&](const StepIn& si) {     // 11 collectives: 6 layer exchanges of the 5 block rows, then the 5 markov merges of one row each (x DS.ds_ns sessions)
                std::vector<uint32_t> sizes(ds41::dsp::DRAFT_NCOLL);   // the bytes the kernels publish and wait for; HEAD_NC = 0, not --top-p-cands
                g6s2::require(si.M == DS.ds_ns * g6s2::Ds41Round::BLOCK, "DSpark draft step rows are not 5 per session");
                for (int c = 0; c < ds41::dsp::DRAFT_NCOLL; ++c) sizes[c] = (uint32_t)ds41::dsp::coll_bytes(c, HID, DS.ds_ns);
                fabds.enqueue_step(si.step, sizes);
            },
            [&]() {
                const int S = DS.ds_ns;
                g6s2::require(S >= 1 && S <= DS.dsp.nsess && S <= 8, "DSpark draft step sessions outside the S2 slots");
                if (graph) { g6s2::require(exec_ds[S] != nullptr, "DSpark draft graph not captured"); G6RT_CK(cudaGraphLaunch(exec_ds[S], st)); }
                else g6s2::launch_step_ds41_draft(DS, S * g6s2::Ds41Round::BLOCK, st);
            },
            [&]() -> unsigned { return *fabds.dev().error; },
            [&]() {
                if (m.ds_slots > 1)
                    g6s2::require(ds41_attn_comp_commit_sessions(&m, DS.dsp.q, DS.dsp.acc, DS.dsp.crow0, DS.dsp.cslot, DS.ds_ns, st) == 0,
                                  "DSpark compressor commit of the sessions failed");
                else ds41_attn_comp_commit_all(&m, DS.dsp.q, DS.dsp.acc, st);
            }};
        g6s2::Ds41Context dc{m, st, in, out,
            [&](const StepIn& si) {
                std::vector<uint32_t> sizes(NC, (uint32_t)(si.M * HID * 2));
                sizes[2 * NL] = head_size(si, nc);
                extra_coll_sizes(sizes, NL, si.M);
                fab_enqueue(si.step, sizes);
                cp_enqueue(si.step, si.M, si.context);
                qs_enqueue(si.step, si.M);
            },
            [&]() -> unsigned { return *f.error | cp_error() | ds41_engram_error(m) | (v2engine.v.live ? *v2engine.v.wire.error : 0u); },
            [&](int k, const StepIn& si) {
                if (!graph) { launch_step(m, si.M, si.context, st, -1, 0, residual_copy); return; }   // eager: the offline loop's launch (rm lambda); reads m.ds_ns
                g6s2::require(k >= 0 && k < m.nbk && si.M >= 1 && si.M <= MX &&
                              (m.ds_ns > 1 || (m.ds_moe_decode_step && ds_moe_decode_graph ? exec_dec[k][si.M] : exec[k][si.M])), "DS41 serve graph not captured");
                launch_main(k, si.M);   // m.ds_ns > 1 (set by serve_ds41): the ns-session graph
            },
            dsk.on() ? &ds : nullptr};
        std::unique_ptr<g6s2::Ds41PrefixCache> ds41_cache;
        if (reuse_prefix) {
            size_t budget_mib = 4096;
            if (const auto mib = knob::ds_cache_mib()) budget_mib = (size_t)*mib;
            ds41_cache = std::make_unique<g6s2::Ds41PrefixCache>(m, dsk.on() ? &ds : nullptr, budget_mib);
            dc.ckpt = ds41_cache.get();
            fprintf(stderr, "# rank %d ds41 prefix cache budget %zu MiB\n", R, budget_mib);
        }
        // Pageable H2D sources must outlive the async copies on the non-blocking compute stream.
        // The native prefill hook fences that stream before the next chunk can mutate them.
        std::vector<int> pf_pos(pf_min ? m.hc_rows : 0), pf_tgt(pf_min ? m.hc_rows : 0, -1), pf_sess(pf_min && m.ds_slots > 1 ? m.hc_rows : 0);
        std::vector<int> pf_tok(c8_pack_env ? 512 : 0); // GR8_DS_C8PACK=1 only; pageable source lives through native's post-launch fence
        std::vector<G6AttnStepCell> pf_cells(pf_min ? m.nl : 0);
        // bounded replay (ds41_pf_split.h): GR8_DS_PF_TAIL, read once; 0 = off (the unsplit non-final plan)
        const int pf_tail = ds41::br::parse_tail(DS41_TEXT(knob::ds_pf_tail()));
        if (pf_tail < 0) die(ds41::br::tail_error(DS41_TEXT(knob::ds_pf_tail())));
        ds41::br::l20_tail() = knob::ds_pf_l20_tail();   // GR8_DS_PF_L20_TAIL (L1); the value the fabric plan set
        if (ds41::br::l20_tail() && !pf_tail) die("GR8_DS_PF_L20_TAIL=1 needs GR8_DS_PF_TAIL (layer 20 joins the bounded replay)");
        if (pf_v2 && (!pf_min || !ds41::v2::boot_ok(m.hc_rows, pf_tail, g6nr::NR, pf_lanes,
                                                    ds41::br::late_layer_start())))
            die("GR8_DS_PF_V2 needs W-row prefill, TP4, one lane, tail 128 and the range boundary at layer 21");
        if (pf_v2) {
            v2engine.boot(m,fabv2,st);
            fprintf(stderr, "# rank %d GR8_DS_PF_V2=1 own-row conductor W=%d, rangeA=0..20, stock tail=128, dedicated port=%d\n", R, m.hc_rows,cfg.port+104);
        }
        if (pf_min) fprintf(stderr, "# rank %d DS41 prefill bounded replay tail %d (layers >= %d)\n", R, pf_tail, ds41::br::late_layer_start());
        if (pf_min && ds41::br::l20_tail())
            fprintf(stderr, "# rank %d DS41 GR8_DS_PF_L20_TAIL=1: layer 20 in the bounded replay (ca row on every row, the rest of the layer on the tail rows)\n", R);
        // ds41_pf_split.h late_skip: GR8_DS_PF_LATE_FINAL=1 = a chunk with (N-1) - context >= tail + WINDOW rows after it skips range B. Default off.
        bool pf_late_final = false;
        if (knob::ds_pf_late_final()) {
            if (pf_min && !ds41::br::late_final_valid(pf_tail, m.hc_rows))
                die("GR8_DS_PF_LATE_FINAL=1 needs GR8_DS_PF_TAIL in WINDOW .. W-1 (full chunks compacted)");
            pf_late_final = pf_min != 0;
            if (pf_min) fprintf(stderr, "# rank %d DS41 prefill late tail on the final chunks only: GR8_DS_PF_LATE_FINAL=1 (skip range B when rows after >= %d)\n",
                                R, pf_tail + geom::WINDOW);
        }
        // GR8_DS_PF_LANES=2: lane B's Model view, stream, events and pageable h2d sources (built below; fenced with lane A's by the serve hook)
        std::unique_ptr<Model> mBp;
        cudaStream_t sb = nullptr;
        std::vector<cudaEvent_t> lane_ev;
        std::vector<int> pf_pos2, pf_tgt2, pf_sess2;
        std::vector<G6AttnStepCell> pf_cells2;
        if (pf_min) {
            dc.prefill = [&](const g6s2::Ds41PrefillChunk& c) {
                g6s2::require(c.slot < (uint32_t)m.ds_slots && c.rows <= (uint32_t)m.hc_rows && c.context <= cap,
                              "DS41 prefill chunk outside the session slots / W / capacity");
                const auto nonfinal = ds41::dsp::pf::nonfinal_plan(m.ds41_plan, dsk.on());
                const std::string check = ds41::dsp::pf::check_nonfinal_plan(m.ds41_plan, nonfinal, dsk.on());
                g6s2::require(check.empty(), check.c_str());
                const ds41::br::Split split = pf_tail ? ds41::br::split_plan(nonfinal, layers) : ds41::br::Split{nonfinal, nonfinal.size(), -1};
                if (pf_tail) { const std::string sc2 = ds41::br::check_split(nonfinal, split, layers); g6s2::require(sc2.empty(), sc2.c_str()); }
                const int late = split.on() ? ds41::br::late_rows((int)c.rows, pf_tail) : (int)c.rows;
                g6s2::require(c.prompt_rows > (uint32_t)c.context, "DS41 prefill chunk past the prompt");
                const int rows_after = int(c.prompt_rows - 1 - (uint32_t)c.context);
                const bool skip_late = pf_late_final && split.on() && ds41::br::late_skip(rows_after, pf_tail);   // GR8_DS_PF_LATE_FINAL: no range B in this chunk
                const auto coll = g6s2::ds41_chunk_collectives(layers, m.ecoll, c.rows, (size_t)HID * 2, (size_t)6144 * 2,
                                                               split.on() ? split.i_late : -1, (uint32_t)late, skip_late);
                for (size_t q = 0; q < coll.sizes.size(); ++q)
                    g6s2::require(coll.sizes[q] + (size_t)8 <= g6::fab_slot_stride(f, (int)q), "DS41 prefill fabric slot smaller than chunk payload + tail");
                for (uint32_t r = 0; r < c.rows; ++r) pf_pos[r] = int(c.first + r);
                for (int i = 0; i < m.nl; ++i) {
                    const auto v = g6s2::ds41_chunk_cell(c, false); // DS41 has no S2 shared-index page mapping.
                    pf_cells[i] = G6AttnStepCell{v.context, v.page_count, v.table_size, v.reuse_rows};
                }
                ds41moe::elog_begin(&m, (int)c.first);
                G6RT_CK(cudaStreamSynchronize(0)); // elog_begin uses a synchronous legacy-stream copy; order it before router work on st.
                G6RT_CK(cudaMemcpyAsync(m.tok, c.tok, c.rows * sizeof(int), cudaMemcpyHostToDevice, st));
                G6RT_CK(cudaMemcpyAsync(m.pos, pf_pos.data(), c.rows * sizeof(int), cudaMemcpyHostToDevice, st));
                G6RT_CK(cudaMemcpyAsync(m.tgt, pf_tgt.data(), c.rows * sizeof(int), cudaMemcpyHostToDevice, st));
                G6RT_CK(cudaMemcpyAsync(m.stepp, &c.sequence, sizeof(c.sequence), cudaMemcpyHostToDevice, st));
                G6RT_CK(cudaMemcpyAsync(m.cells, pf_cells.data(), pf_cells.size() * sizeof(pf_cells[0]), cudaMemcpyHostToDevice, st));
                m.ds_ns = 1;   // a chunk is one session's rows
                m.ds_pf_slot = (int)c.slot;
                if (m.ds_slots > 1)   // SESS: every row of the chunk is in its slot's slabs (attention ring / comp, index page, Engram ring)
                {
                    std::fill(pf_sess.begin(), pf_sess.begin() + c.rows, (int)c.slot);
                    G6RT_CK(cudaMemcpyAsync(m.sess, pf_sess.data(), c.rows * sizeof(int), cudaMemcpyHostToDevice, st));
                }
                if (vis_on) {   // DS41 image input: the images whose span overlaps this chunk, encoded into the pool (m.vis_n)
                    int pr = 0;
                    const VisChunk vc{(int)c.slot, c.first, c.rows, c.images};
                    vis.stage(m, 0, &vc, 1, (int)c.rows, pr, st);
                }
                g6s2::require(ds41_idx_set_chunk(&m, c.context) == 0, "DS41 prefill index chunk bound refused");
                if (pf_v2) {
                    const auto& table=ds41_table();
                    for(int i=0;i<split.i_late;++i)
                        if(table.prepare_attn && table.prepare_attn(&m,i,(int)c.rows,st))die("DS41 v2 attention prepare failed");
                    if(!skip_late)prepare_ds41_prefill(m,late,st,split.i_late,m.nl);
                }
                else if (skip_late) prepare_ds41_prefill(m, (int)c.rows, st, 0, split.i_late);
                else if (split.on()) { prepare_ds41_prefill(m, (int)c.rows, st, 0, split.i_late); prepare_ds41_prefill(m, late, st, split.i_late, m.nl); }
                else prepare_ds41_prefill(m, (int)c.rows, st);
                if (dsk.on()) { m.dsp.main_lo = 0; m.dsp.main_hi = late; }
                g6::Fabric::Rs rs; rs.order = coll.order;
                if(pf_v2) {
                    // Range A only rings the dedicated fabric. Main fabric must
                    // never wait on those old doorbells; keep just stock B.
                    if(!skip_late) {
                        auto sizes=coll.sizes; g6::Fabric::Rs late_rs;
                        for(size_t q=0;q<sizes.size();++q) {
                            if(q>=size_t(2*split.i_late) && q<size_t(2*m.nl))late_rs.order.push_back(int(q));
                            else sizes[q]=0;
                        }
                        fab.enqueue_step(c.sequence,sizes,late_rs);
                    }
                    m.ds41_rowshard_chunk=false; m.ds41_rowshard_state=false;
                    m.ds41_rowshard_pending_c=-1; m.ds41_y_quant_ready=false;
                }
                else if (ds41::rs::on(rs_k > 0, 0, rs_k, (int)c.rows, MX, g6nr::NR)) {   // GR8_DS_PF_RS=1: the same rule as the ops' ds41_rs_on
                    const size_t rb = (size_t)HID * 2;
                    if (const char* e = ds41::rs::job_error(coll.sizes, NC, rs_k, (int)c.rows, g6nr::NR, rb)) g6s2::require(false, e);
                    const bool shard_chunk=ds41::rowshard::chunk_on(rowshard,(int)c.rows,late,g6nr::NR);
                    const bool shard_state=shard_chunk && !skip_late;
                    const uint32_t state_bytes=shard_state ? (uint32_t)ds41::rowshard::state_bytes(late,HID,sizeof(ds_mhc::trunk_t)) : 0;
                    ds41::rs::Job j = ds41::rs::job(coll.sizes, coll.order, NC, rs_k, (int)c.rows, g6nr::NR, rb,state_bytes,shard_chunk);
                    if (shard_chunk && y_mxfp8)
                        for (int q = 0; q < rs_k; ++q) j.sizes[(size_t)(NC + q)] = (uint32_t)ds41::rowshard::y_mxfp8_bytes(1024,HID);
                    for (int q : j.rs) {
                        g6s2::require(j.ss * g6nr::NR <= g6::fab_slot_stride(f, q), "DS41 RS: NR slices do not fit an RS collective's slot");
                        rs.mask.set((size_t)q);
                    }
                    for (int q = 0; q < rs_k; ++q)
                        g6s2::require(j.sizes[(size_t)(NC + q)] + (size_t)8 <= g6::fab_slot_stride(f, NC + q), "DS41 RS: the AG payload + tail does not fit its slot");
                    if (shard_state) g6s2::require((size_t)state_bytes + 8 <= g6::fab_slot_stride(f, NC + rs_k), "DS41 row-shard: state payload does not fit its slot");
                    rs.ss = j.ss; rs.bytes = j.bytes; rs.order = j.order;
                    if (pf_direct) {   // GR8_DS_PF_DIRECT: the producers wrote flat [rows][HID]; slice p = Q rows at p Q HID 2, tail words after the NR slices
                        rs.ss = (size_t)ds41::rs::quarter((int)c.rows, g6nr::NR) * HID * 2;
                        rs.flat = rs.mask;
                        for (int q : j.rs) g6s2::require(rs.ss * g6nr::NR + 8 * (size_t)g6nr::NR <= g6::fab_slot_stride(f, q), "DS41 direct: NR flat slices + tail words do not fit an RS slot");
                    }
                    fab.enqueue_step(c.sequence, j.sizes, rs);
                    m.ds41_rowshard_chunk=shard_chunk;
                    m.ds41_rowshard_state=shard_state;
                    m.ds41_y_quant_ready=false;
                }
                else if (coll.order.empty()) { fab.enqueue_step(c.sequence, coll.sizes); m.ds41_rowshard_chunk=false; m.ds41_rowshard_state=false; m.ds41_y_quant_ready=false; }
                else { fab.enqueue_step(c.sequence, coll.sizes, rs); m.ds41_rowshard_chunk=false; m.ds41_rowshard_state=false; m.ds41_y_quant_ready=false; }
                // Engram (GR8_DS_ENGRAM=1): the nonfinal plan has no step_in, and the decode step's call of ds41_engram_begin in launch_step_ds41 sits after step_in, so this
                // callback owns it: the chunk's token window + the ring update (window_wide, or the decode kernel for <= MX rows) and the hash + NVMe prefetch
                // that ST_ENGRAM of layers 1 and 14 consume. After the m.tok / m.pos copies above, same stream, before the plan's embed; enqueue-only.
                if (m.engram) m.engram->plan_next_warm(c.tok - c.first, (int)c.prompt_rows, (int)c.first + (int)c.rows);
                if (m.engram) {
                    ds41_engram_begin(m, (int)c.rows, st, (int)c.slot);   // SESS: the chunk's slot's token ring (slot 0 with one session)
                    fprintf(stderr, "# rank %d request %llu ds41 engram prefill begin chunk %u rows %u\n", R, (unsigned long long)c.request, c.first, c.rows);
                }
                m.ds41_pf_late = late;
                m.ds41_dchunk = pf_direct; m.ds41_dseq = c.sequence;   // GR8_DS_PF_DIRECT: this enqueue's producers write the send slots of step c.sequence
                if (skip_late)
                    fprintf(stderr, "# rank %d request %llu chunk %u rows %u late tail skipped (rows after %d)\n", R, (unsigned long long)c.request, c.first, c.rows, rows_after);
                // skip_late: range A without its last op ST_PF_COMPACT (split.b - 1, check_split), no range B
                Ds41SpanPass span_pass(R, (int)c.rows, st, 0);   // GR8_DS_SPAN: the chunk's ops, a mark each
                const size_t range_a_end = skip_late ? split.b - 1 : split.b;
                ds41::Stop stop;
                if (pf_v2) {
                    v2engine.v.widths.select(c.request,c.slot,R);
                    ds41::v2::timing_begin_chunk(v2engine.v,c.request,c.first,c.rows);
                    v2engine.run(m,(int)c.rows,split.i_late,skip_late);
                    ds41::v2::timing_end_chunk(v2engine.v);
                } else stop = ds41_run_pass(split.ops, 0, range_a_end, &m, (int)c.rows, st, span_pass);
                if (!stop.stopped && split.on() && !skip_late) {   // range B: the layers >= 21 over the chunk's last `late` rows (now rows 0 .. late-1)
                    ds41_attn_set_win_floor(ds41::br::win_floor((int)c.context, (int)c.rows, pf_tail));
                    stop = ds41_run_pass(split.ops, split.b, split.ops.size(), &m, late, st, span_pass);
                    ds41_attn_set_win_floor(0);
                }
                span_pass.close();
                m.ds41_pf_late = 0;
                m.ds_pf_slot = 0;
                m.vis_n = 0;
                m.ds41_dchunk = false; m.ds41_dir_used[0] = m.ds41_dir_used[1] = false;
                if (stop.stopped) die(ds41::stop_message(stop));
                G6RT_CK(cudaPeekAtLastError());
                g6s2::require(ds41_idx_set_chunk(&m, 0) == 0, "DS41 prefill index chunk reset refused");
                if (dsk.on()) { m.dsp.main_lo = 0; m.dsp.main_hi = -1; }
            };
            dc.after_prefill = [&](const g6s2::Ds41PrefillChunk& c) {
                // Native already fenced this chunk. Retire diagnostics here so
                // long requests retain all spans; request/chunk IDs and drops
                // remain in the report. No new fence (default-off diagnostic).
                if(pf_v2)ds41::v2::timing_report(v2engine.v,c.request);
                std::vector<dsr::ExpertLogRec> records;
                const int n = ds41moe::elog_read(&m, records, true);
                g6s2::require(n >= 0, "DS41 prefill distinct-expert counter read failed");
                G6RT_CK(cudaStreamSynchronize(0)); // elog_read's cursor reset must land before the next chunk on st.
                for (const auto& x : records)
                    fprintf(stderr, "# rank %d request %llu chunk %u rows %u moe layer %d distinct %d routes %d\n",
                            R, (unsigned long long)c.request, c.first, c.rows, x.layer, x.distinct, x.routes);
            };
            // GR8_DS_PF_LANES=2 (ds41_lanes.h, the GLM pf_step2s port): one quantum = lane A (chunk a: m, st, the main fabric, Dispatch's id) and
            // lane B (chunk b, the rest of the same session: mB, sb, fabB, lane B's own id). Both walk the same non-final plan; the serial stages
            // (ds41::lanes::serial: attention, indexer, Engram, DSpark main) run as one token chain A(s) < B(s) < A(s') through events, the rest
            // overlaps. Each lane's prologue and runs are dc.prefill's, per Model; lane A joins lane B at the end, so the serve fence on st
            // covers both. dc.prefill (one lane, up to W rows) runs every quantum of W rows or fewer.
            if (pf_lanes == 2) {
                if (pf_lanes_fab != 2) die("GR8_DS_PF_LANES changed between the fabric setup and the serve setup");
                mBp = std::make_unique<Model>();
                const size_t lb = build_ds41_lane_b(*mBp, m, fabB.dev(), st);
                if (vis_on) vis.attach(*mBp);   // lane B's own table and row flags (the pool is shared)
                G6RT_CK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
                pf_pos2.assign(m.hc_rows, 0); pf_tgt2.assign(m.hc_rows, -1); pf_sess2.assign(m.ds_slots > 1 ? m.hc_rows : 0, 0); pf_cells2.resize(m.nl);
                fprintf(stderr, "# rank %d DS41 two-lane prefill: GR8_DS_PF_LANES=2 (quanta up to %d rows, lane B: own stream, fabric, sequence; view %.1f MiB device)\n",
                        R, 2 * m.hc_rows, (double)lb / (1024.0 * 1024.0));
                dc.prefill2 = [&](const g6s2::Ds41PrefillChunk& ca, const g6s2::Ds41PrefillChunk& cb) {
                    Model& mB = *mBp;
                    g6s2::require(ca.rows == (uint32_t)m.hc_rows && cb.rows >= 1 && cb.rows <= (uint32_t)m.hc_rows && cb.first == (uint32_t)ca.context &&
                                  cb.slot == ca.slot && cb.request == ca.request && cb.prompt_rows == ca.prompt_rows && cb.sequence != 0,
                                  "DS41 two-lane quantum: lane B is not the rest of lane A's session after W rows");
                    const g6s2::Ds41PrefillChunk* ch[2] = {&ca, &cb};
                    Model* const mm[2] = {&m, &mB};
                    const cudaStream_t ss[2] = {st, sb};
                    g6::Fabric* const ff[2] = {&fab, &fabB};
                    const g6::FabricDev fd[2] = {f, fabB.dev()};
                    std::vector<int>* const vpos[2] = {&pf_pos, &pf_pos2};
                    std::vector<int>* const vtgt[2] = {&pf_tgt, &pf_tgt2};
                    std::vector<int>* const vsess[2] = {&pf_sess, &pf_sess2};
                    std::vector<G6AttnStepCell>* const vcells[2] = {&pf_cells, &pf_cells2};
                    const auto nonfinal = ds41::dsp::pf::nonfinal_plan(m.ds41_plan, dsk.on());
                    const std::string check = ds41::dsp::pf::check_nonfinal_plan(m.ds41_plan, nonfinal, dsk.on());
                    g6s2::require(check.empty(), check.c_str());
                    const ds41::br::Split split = pf_tail ? ds41::br::split_plan(nonfinal, layers) : ds41::br::Split{nonfinal, nonfinal.size(), -1};
                    if (pf_tail) { const std::string sc2 = ds41::br::check_split(nonfinal, split, layers); g6s2::require(sc2.empty(), sc2.c_str()); }
                    int late[2] = {0, 0}, rows_after[2] = {0, 0};
                    bool skip_late[2] = {false, false};
                    std::vector<ds41::lanes::Run> runs[2];
                    for (int L = 0; L < 2; ++L) {
                        const g6s2::Ds41PrefillChunk& c = *ch[L];
                        g6s2::require(c.slot < (uint32_t)m.ds_slots && c.rows <= (uint32_t)m.hc_rows && c.context <= cap,
                                      "DS41 prefill chunk outside the session slots / W / capacity");
                        late[L] = split.on() ? ds41::br::late_rows((int)c.rows, pf_tail) : (int)c.rows;
                        g6s2::require(c.prompt_rows > (uint32_t)c.context, "DS41 prefill chunk past the prompt");
                        rows_after[L] = int(c.prompt_rows - 1 - (uint32_t)c.context);
                        skip_late[L] = pf_late_final && split.on() && ds41::br::late_skip(rows_after[L], pf_tail);
                        runs[L] = ds41::lanes::lane_runs(split, (int)c.rows, late[L], skip_late[L],
                                                         split.on() ? ds41::br::win_floor((int)c.context, (int)c.rows, pf_tail) : 0);
                    }
                    const ds41::lanes::Schedule sched = ds41::lanes::schedule(split.ops, runs);
                    while ((int)lane_ev.size() < sched.events) {
                        cudaEvent_t e;
                        G6RT_CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                        lane_ev.push_back(e);
                    }
                    // ds41moe::elog_begin is a synchronous legacy-stream copy: both lanes' before any router work (lane B has its own counter)
                    for (int L = 0; L < 2; ++L) ds41moe::elog_begin(mm[L], (int)ch[L]->first);
                    G6RT_CK(cudaStreamSynchronize(0));
                    // lane L's prologue: dc.prefill's, on its Model, stream and fabric (K_PRO, the first link of the token chain)
                    auto prologue = [&](int L) {
                        const g6s2::Ds41PrefillChunk& c = *ch[L];
                        Model& M = *mm[L];
                        const cudaStream_t s = ss[L];
                        const auto coll = g6s2::ds41_chunk_collectives(layers, m.ecoll, c.rows, (size_t)HID * 2, (size_t)6144 * 2,
                                                                       split.on() ? split.i_late : -1, (uint32_t)late[L], skip_late[L]);
                        for (size_t q = 0; q < coll.sizes.size(); ++q)
                            g6s2::require(coll.sizes[q] + (size_t)8 <= g6::fab_slot_stride(fd[L], (int)q), "DS41 prefill fabric slot smaller than chunk payload + tail");
                        for (uint32_t r = 0; r < c.rows; ++r) (*vpos[L])[r] = int(c.first + r);
                        for (int i = 0; i < m.nl; ++i) {
                            const auto v = g6s2::ds41_chunk_cell(c, false);
                            (*vcells[L])[i] = G6AttnStepCell{v.context, v.page_count, v.table_size, v.reuse_rows};
                        }
                        G6RT_CK(cudaMemcpyAsync(M.tok, c.tok, c.rows * sizeof(int), cudaMemcpyHostToDevice, s));
                        G6RT_CK(cudaMemcpyAsync(M.pos, vpos[L]->data(), c.rows * sizeof(int), cudaMemcpyHostToDevice, s));
                        G6RT_CK(cudaMemcpyAsync(M.tgt, vtgt[L]->data(), c.rows * sizeof(int), cudaMemcpyHostToDevice, s));
                        G6RT_CK(cudaMemcpyAsync(M.stepp, &c.sequence, sizeof(c.sequence), cudaMemcpyHostToDevice, s));
                        G6RT_CK(cudaMemcpyAsync(M.cells, vcells[L]->data(), vcells[L]->size() * sizeof(G6AttnStepCell), cudaMemcpyHostToDevice, s));
                        M.ds_ns = 1;
                        M.ds_pf_slot = (int)c.slot;
                        if (m.ds_slots > 1) {
                            std::fill(vsess[L]->begin(), vsess[L]->begin() + c.rows, (int)c.slot);
                            G6RT_CK(cudaMemcpyAsync(M.sess, vsess[L]->data(), c.rows * sizeof(int), cudaMemcpyHostToDevice, s));
                        }
                        g6s2::require(ds41_idx_set_chunk(&M, c.context) == 0, "DS41 prefill index chunk bound refused");
                        if (skip_late[L]) prepare_ds41_prefill(M, (int)c.rows, s, 0, split.i_late);
                        else if (split.on()) { prepare_ds41_prefill(M, (int)c.rows, s, 0, split.i_late); prepare_ds41_prefill(M, late[L], s, split.i_late, m.nl); }
                        else prepare_ds41_prefill(M, (int)c.rows, s);
                        if (dsk.on()) { M.dsp.main_lo = 0; M.dsp.main_hi = late[L]; }
                        g6s2::require(M.ds41_rowshard_pending_c < 0, "DS41 two-lane row shard: prior y all-gather was not published");
                        g6::Fabric::Rs rs; rs.order = coll.order;
                        if (ds41::rs::on(rs_k > 0, 0, rs_k, (int)c.rows, MX, g6nr::NR)) {
                            const size_t rb = (size_t)HID * 2;
                            if (const char* e = ds41::rs::job_error(coll.sizes, NC, rs_k, (int)c.rows, g6nr::NR, rb)) g6s2::require(false, e);
                            const auto p = ds41::lanes::b12_job(coll.sizes, coll.order, NC, rs_k, (int)c.rows, late[L], g6nr::NR, rb,
                                                                 HID, sizeof(ds_mhc::trunk_t), rowshard, y_mxfp8, skip_late[L]);
                            const auto& j = p.job;
                            for (int q : j.rs) {
                                g6s2::require(j.ss * g6nr::NR <= g6::fab_slot_stride(fd[L], q), "DS41 RS: NR slices do not fit an RS collective's slot");
                                rs.mask.set((size_t)q);
                            }
                            for (int q = 0; q < rs_k; ++q)
                                g6s2::require(j.sizes[(size_t)(NC + q)] + (size_t)8 <= g6::fab_slot_stride(fd[L], NC + q), "DS41 RS: the AG payload + tail does not fit its slot");
                            if (p.shard_state)
                                g6s2::require((size_t)p.state_bytes + 8 <= g6::fab_slot_stride(fd[L], NC + rs_k), "DS41 row-shard: state payload does not fit its slot");
                            rs.ss = j.ss; rs.bytes = j.bytes; rs.order = j.order;
                            if (pf_direct) {
                                rs.ss = (size_t)ds41::rs::quarter((int)c.rows, g6nr::NR) * HID * 2;
                                rs.flat = rs.mask;
                                for (int q : j.rs) g6s2::require(rs.ss * g6nr::NR + 8 * (size_t)g6nr::NR <= g6::fab_slot_stride(fd[L], q), "DS41 direct: NR flat slices + tail words do not fit an RS slot");
                            }
                            ff[L]->enqueue_step(c.sequence, j.sizes, rs);
                            M.ds41_rowshard_chunk = p.shard_chunk;
                            M.ds41_rowshard_state = p.shard_state;
                        }
                        else {
                            if (coll.order.empty()) ff[L]->enqueue_step(c.sequence, coll.sizes);
                            else ff[L]->enqueue_step(c.sequence, coll.sizes, rs);
                            M.ds41_rowshard_chunk = false; M.ds41_rowshard_state = false;
                        }
                        M.ds41_y_quant_ready = false;
                        if (m.engram) {   // lane L's per-step set is the engine's active one (use_lane before this call)
                            ds41_engram_begin(M, (int)c.rows, s, (int)c.slot);
                            fprintf(stderr, "# rank %d request %llu ds41 engram prefill begin chunk %u rows %u lane %c\n", R, (unsigned long long)c.request, c.first, c.rows, L ? 'B' : 'A');
                        }
                        M.ds41_pf_late = late[L];
                        M.ds41_dchunk = pf_direct; M.ds41_dseq = c.sequence;
                        if (skip_late[L])
                            fprintf(stderr, "# rank %d request %llu chunk %u rows %u late tail skipped (rows after %d) lane %c\n", R, (unsigned long long)c.request, c.first, c.rows, rows_after[L], L ? 'B' : 'A');
                    };
                    if (m.engram) m.engram->plan_next_warm(ca.tok - ca.first, (int)ca.prompt_rows, (int)cb.first + (int)cb.rows, 2);
                    if (vis_on) {   // DS41 image input: each lane's images on its own stream; the lanes share the pool (one counter)
                        int pr = 0;
                        for (int L = 0; L < 2; ++L) {
                            const VisChunk vc{(int)ch[L]->slot, ch[L]->first, ch[L]->rows, ch[L]->images};
                            vis.stage(*mm[L], L, &vc, 1, (int)ch[L]->rows, pr, ss[L]);
                        }
                    }
                    ds41::Stop stop{};
                    Ds41SpanPass span_pass[2] = {Ds41SpanPass(R, (int)ca.rows, ss[0], 0), Ds41SpanPass(R, (int)cb.rows, ss[1], 1)};   // GR8_DS_SPAN: each lane's ops
                    for (const ds41::lanes::Seg& g : sched.seg) {
                        const int L = g.lane;
                        if (g.wait >= 0) G6RT_CK(cudaStreamWaitEvent(ss[L], lane_ev[(size_t)g.wait], 0));
                        if (m.engram) g6s2::require(m.engram->use_lane(L) == 0, "DS41 two-lane prefill: Engram lane switch refused");
                        if (g.kind == ds41::lanes::K_PRO) prologue(L);
                        else if (g.kind == ds41::lanes::K_OPS || g.kind == ds41::lanes::K_SEC) {
                            ds41_attn_set_win_floor(g.win_floor);   // a host global read at launch: this segment's lane and range
                            stop = ds41_run_pass(split.ops, g.b, g.e, mm[L], g.rows, ss[L], span_pass[L]);
                        }
                        if (stop.stopped) break;
                        if (g.record >= 0) G6RT_CK(cudaEventRecord(lane_ev[(size_t)g.record], ss[L]));
                    }
                    span_pass[0].close(); span_pass[1].close();
                    ds41_attn_set_win_floor(0);
                    if (m.engram) g6s2::require(m.engram->use_lane(0) == 0, "DS41 two-lane prefill: Engram lane reset refused");
                    for (int L = 0; L < 2; ++L) {
                        Model& M = *mm[L];
                        M.ds41_pf_late = 0;
                        M.ds_pf_slot = 0;
                        M.vis_n = 0;
                        M.ds41_dchunk = false; M.ds41_dir_used[0] = M.ds41_dir_used[1] = false;
                        g6s2::require(M.ds41_rowshard_pending_c < 0, "DS41 two-lane row shard: missing y all-gather");
                        M.ds41_rowshard_chunk = false; M.ds41_rowshard_state = false; M.ds41_y_quant_ready = false;
                    }
                    if (stop.stopped) die(ds41::stop_message(stop));
                    G6RT_CK(cudaPeekAtLastError());
                    for (int L = 0; L < 2; ++L) {
                        g6s2::require(ds41_idx_set_chunk(mm[L], 0) == 0, "DS41 prefill index chunk reset refused");
                        if (dsk.on()) { mm[L]->dsp.main_lo = 0; mm[L]->dsp.main_hi = -1; }
                    }
                };
                dc.errors2 = [&]() -> unsigned { return *fabB.dev().error | ds41_engram_error(*mBp); };
                dc.after_prefill2 = [&](const g6s2::Ds41PrefillChunk& c) {
                    std::vector<dsr::ExpertLogRec> records;
                    const int n = ds41moe::elog_read(mBp.get(), records, true);
                    g6s2::require(n >= 0, "DS41 prefill lane B distinct-expert counter read failed");
                    G6RT_CK(cudaStreamSynchronize(0));
                    for (const auto& x : records)
                        fprintf(stderr, "# rank %d request %llu chunk %u rows %u lane B moe layer %d distinct %d routes %d\n",
                                R, (unsigned long long)c.request, c.first, c.rows, x.layer, x.distinct, x.routes);
                };
            }
            // GR8_DS_C8PACK=1: one native W launch over the validated, per-slot ragged table. With the
            // flag off only the single-session callback above runs.
            dc.prefill_pack = [&](const g6s2::Ds41PrefillPack& p) {
                m.ds41_rowshard_chunk=false; m.ds41_rowshard_state=false; m.ds41_y_quant_ready=false; // pack <=512 is never the 4096/1024 row-shard geometry
                const auto& shape = p.layout;
                if (const char* e = ds41::pack::error(shape, m.hc_rows, m.ds_slots)) die(std::string("DS41 ragged prefill: ") + e);
                g6s2::require(shape.rows <= 512 && p.sequence != 0 && m.ds41_pack_enabled, "DS41 ragged prefill boot/sequence/cap");
                const int rows = shape.rows;
                const auto nonfinal = ds41::dsp::pf::nonfinal_plan(m.ds41_plan, dsk.on());
                const std::string check = ds41::dsp::pf::check_nonfinal_plan(m.ds41_plan, nonfinal, dsk.on());
                g6s2::require(check.empty(), check.c_str());
                const ds41::br::Split split = pf_tail ? ds41::br::split_plan(nonfinal, layers) : ds41::br::Split{nonfinal, nonfinal.size(), -1};
                if (pf_tail) { const std::string sc2 = ds41::br::check_split(nonfinal, split, layers); g6s2::require(sc2.empty(), sc2.c_str()); }
                // Every segment is its complete <=64-row nonfinal prefix. GR8_DS_PF_TAIL=128 keeps
                // every row, so no late skip, no compact, and every segment's window floor is 0.
                g6s2::require(!split.on() || pf_tail >= ds41::pack::MAX_SEGMENT_ROWS,
                              "DS41 ragged prefill requires a per-segment late tail of at least 64 rows");
                const auto coll = g6s2::ds41_chunk_collectives(layers, m.ecoll, rows, (size_t)HID * 2, (size_t)6144 * 2,
                                                                 split.on() ? split.i_late : -1, (uint32_t)rows, false);
                for (size_t q = 0; q < coll.sizes.size(); ++q)
                    g6s2::require(coll.sizes[q] + (size_t)8 <= g6::fab_slot_stride(f, int(q)), "DS41 ragged prefill fabric slot too small");
                int max_context = 0;
                for (int j = 0; j < shape.count; ++j) {
                    const auto& s = shape.seg[j];
                    max_context = std::max(max_context, s.context);
                    for (int r = 0; r < s.rows; ++r) {
                        const int row = s.row0 + r;
                        pf_tok[row] = p.chunk[j].tok[r]; pf_pos[row] = s.first + r; pf_sess[row] = s.slot;
                    }
                }
                // Cells are only a decode k_step_in contract; nonfinal DS41 ops use m.pos,
                // m.sess and the pack table. Keep a conservative maximum in the unused cells.
                const int pages = (max_context + 63) / 64;
                for (int i = 0; i < m.nl; ++i) pf_cells[i] = G6AttnStepCell{max_context, pages, pages, 0};
                ds41moe::elog_begin(&m, 0); // every eligible segment starts at position zero; rows map back via shape.seg
                G6RT_CK(cudaStreamSynchronize(0));
                G6RT_CK(cudaMemcpyAsync(m.tok, pf_tok.data(), size_t(rows) * sizeof(int), cudaMemcpyHostToDevice, st));
                G6RT_CK(cudaMemcpyAsync(m.pos, pf_pos.data(), size_t(rows) * sizeof(int), cudaMemcpyHostToDevice, st));
                G6RT_CK(cudaMemcpyAsync(m.tgt, pf_tgt.data(), size_t(rows) * sizeof(int), cudaMemcpyHostToDevice, st));
                G6RT_CK(cudaMemcpyAsync(m.stepp, &p.sequence, sizeof(p.sequence), cudaMemcpyHostToDevice, st));
                G6RT_CK(cudaMemcpyAsync(m.cells, pf_cells.data(), pf_cells.size() * sizeof(pf_cells[0]), cudaMemcpyHostToDevice, st));
                G6RT_CK(cudaMemcpyAsync(m.sess, pf_sess.data(), size_t(rows) * sizeof(int), cudaMemcpyHostToDevice, st));
                m.ds_ns = 1; m.ds41_pack = &shape;
                g6s2::require(ds41_idx_set_chunk(&m, max_context) == 0, "DS41 ragged index bound refused");
                if (vis_on) {   // DS41 image input: every member's images that overlap its rows (the table keys rows by slot)
                    VisChunk vc[ds41::pack::MAX_SEGMENTS];
                    for (int j = 0; j < shape.count; ++j) vc[j] = VisChunk{(int)p.chunk[j].slot, p.chunk[j].first, p.chunk[j].rows, p.chunk[j].images, shape.seg[j].row0};
                    int pr = 0;
                    vis.stage(m, 0, vc, shape.count, rows, pr, st);
                }
                if (split.on()) { prepare_ds41_prefill(m, rows, st, 0, split.i_late); prepare_ds41_prefill(m, rows, st, split.i_late, m.nl); }
                else prepare_ds41_prefill(m, rows, st);
                if (dsk.on()) { m.dsp.main_lo = 0; m.dsp.main_hi = rows; }
                g6::Fabric::Rs rs; rs.order = coll.order;
                if (ds41::rs::on(rs_k > 0, 0, rs_k, rows, MX, g6nr::NR)) {
                    const size_t rb = size_t(HID) * 2;
                    if (const char* e = ds41::rs::job_error(coll.sizes, NC, rs_k, rows, g6nr::NR, rb)) g6s2::require(false, e);
                    const ds41::rs::Job j = ds41::rs::job(coll.sizes, coll.order, NC, rs_k, rows, g6nr::NR, rb);
                    for (int q : j.rs) {
                        g6s2::require(j.ss * g6nr::NR <= g6::fab_slot_stride(f, q), "DS41 ragged RS slices too wide");
                        rs.mask.set(size_t(q));
                    }
                    for (int q = 0; q < rs_k; ++q)
                        g6s2::require(j.sizes[size_t(NC + q)] + size_t(8) <= g6::fab_slot_stride(f, NC + q), "DS41 ragged RS AG slot too small");
                    rs.ss = j.ss; rs.bytes = j.bytes; rs.order = j.order;
                    if (pf_direct) {
                        rs.ss = size_t(ds41::rs::quarter(rows, g6nr::NR)) * HID * 2;
                        rs.flat = rs.mask;
                        for (int q : j.rs) g6s2::require(rs.ss * g6nr::NR + 8 * size_t(g6nr::NR) <= g6::fab_slot_stride(f, q), "DS41 ragged direct slices too wide");
                    }
                    fab.enqueue_step(p.sequence, j.sizes, rs);
                } else if (coll.order.empty()) fab.enqueue_step(p.sequence, coll.sizes);
                else fab.enqueue_step(p.sequence, coll.sizes, rs);
                if (m.engram) ds41_engram_begin_pack(m, shape, st);
                m.ds41_pf_late = rows;
                m.ds41_dchunk = pf_direct; m.ds41_dseq = p.sequence;
                Ds41SpanPass span_pass(R, rows, st, 0);   // GR8_DS_SPAN: the pack's ops, a mark each
                ds41::Stop stop = ds41_run_pass(split.ops, 0, split.b, &m, rows, st, span_pass);
                if (!stop.stopped && split.on()) {
                    ds41_attn_set_win_floor(0);
                    stop = ds41_run_pass(split.ops, split.b, split.ops.size(), &m, rows, st, span_pass);
                }
                span_pass.close();
                m.ds41_pack = nullptr; m.ds41_pf_late = 0; m.vis_n = 0;
                m.ds41_dchunk = false; m.ds41_dir_used[0] = m.ds41_dir_used[1] = false;
                if (stop.stopped) die(ds41::stop_message(stop));
                G6RT_CK(cudaPeekAtLastError());
                g6s2::require(ds41_idx_set_chunk(&m, 0) == 0, "DS41 ragged index reset refused");
                if (dsk.on()) { m.dsp.main_lo = 0; m.dsp.main_hi = -1; }
            };
            dc.after_prefill_pack = [&](const g6s2::Ds41PrefillPack& p) {
                std::vector<dsr::ExpertLogRec> records;
                const int n = ds41moe::elog_read(&m, records, true);
                g6s2::require(n >= 0, "DS41 ragged distinct-expert counter read failed");
                G6RT_CK(cudaStreamSynchronize(0));
                // The expert counter is one aggregate record per layer; expose its packed-row
                // origin map rather than misattributing distinct experts to any one request.
                for (int j = 0; j < p.layout.count; ++j) {
                    const auto& s = p.layout.seg[j];
                    fprintf(stderr, "# rank %d ragged member request %llu slot %d row0 %d rows %d first %d context %d\n",
                            R, (unsigned long long)s.request, s.slot, s.row0, s.rows, s.first, s.context);
                }
                for (const auto& x : records)
                    fprintf(stderr, "# rank %d ragged pack %d rows %d sessions moe layer %d distinct %d routes %d start %d (aggregate)\n",
                            R, p.layout.rows, p.layout.count, x.layer, x.distinct, x.routes, x.start);
            };
        }
        try { g6s2::serve_ds41(dc, bc, plan.chunk); }
        catch (const std::exception& e) { fprintf(stderr, "g6m4: rank %d DS41 serve failed: %s\n", R, e.what()); return 3; }
        if(pf_v2) { v2engine.release(); fabv2.stop_proxy(); }
    }
#else
    if (s2mode)
    {
        // S2 below R served loop. Pool: logical capacity = cap (a past-R request alone at base 0 reads the table as
        // built), runs below R placed in the physical pages every rank and model has.
        g6s2::Config sc;
        sc.epoch = s2epoch;
        sc.pool_rows = (uint32_t)(cap / 64 * 64);
        // one step bounds: a round (every session's mtp + 1 rows), a prompt chunk (sessions.h: chunk <= max_rows) and a
        // DFlash2 verify (dfl_k + 1 rows)
        sc.max_rows = std::min<uint32_t>(MX, std::max<uint32_t>({(uint32_t)s2sess * (uint32_t)(mtp + 1), (uint32_t)plan.chunk,
                                                                  dfl ? (uint32_t)(dfl_k + 1) : 0u}));
        sc.max_sessions = (uint32_t)s2sess;
        sc.mtp = (uint32_t)mtp;
        if (dfl) sc.guard_tokens = 2 * dfl::K;     // a DFlash2 verify writes up to 7 rows past the last token, as the offline loop's cap rule
        sc.prefill_rows = (uint32_t)MP;
        sc.two_lanes = two;
        // GR8_S2_DECODE_PF=<rows>: while a session decodes, prefill quanta of at most this many rows (scheduler.h
        // Config::decode_prefill_rows); unset or 0 = off
        {
            const int v = (int)knob::s2_decode_pf();
            if (v < 0 || v % 64 || v > MP) die("S2: GR8_S2_DECODE_PF must be 0 or a multiple of 64 no larger than --prefill");
            sc.decode_prefill_rows = (uint32_t)v;
        }
        // GR8_S2_DECODE_ROUNDS=<K>: while a prefill waits and sessions decode, K decode rounds per prefill quantum
        // (scheduler.h Config::decode_rounds); unset, 0 or 1 = off
        static_assert(g6s2::MAX_DECODE_ROUNDS == 64, "GR8_S2_DECODE_ROUNDS takes 0..64 (knobs/serving.yaml)");
        sc.decode_rounds = (uint32_t)knob::s2_decode_rounds();
        sc.tail_first = knob::s2_tail_first();
        sc.notice_max = (uint32_t)knob::s2_notice_max();
        sc.steering = steering_boot;
        if (rtt && rtt->local) sc.window_pages = (uint32_t)rtt->local;   // a window run lists its worst case and the window
        sc.r2_rows = (uint32_t)cp_R2;
        // GR8_CP_STEP_ROWS (sized the CP fabric above) and GR8_CP_STEP_SEG=1: Driver::step's rounds past R (cp_rows.h)
        {
            const bool seg = knob::cp_step_seg();
            if (seg && !cp_step) die("GR8_CP_STEP_SEG=1 needs GR8_CP_STEP_ROWS");
            sc.cp_step_rows = (uint32_t)cp_step;
            sc.cp_seg = seg;
            sc.cp_after = g6s2::cp_after(m.bk, m.nbk, m.cp.on ? m.cp.step_R() : -1);
            if (cp_step && R == 0)
                fprintf(stderr, "# s2 GR8_CP_STEP_ROWS=%d: a step past R (context > %llu) carries at most %d rows = %d sessions of %d%s; "
                        "the CP fabric is sized for it\n", cp_step, (unsigned long long)sc.cp_after, cp_step, cp_step / (mtp + 1), mtp + 1,
                        sc.cp_seg ? ", and no session below R (GR8_CP_STEP_SEG)" : "");
        }
        const int place = std::min(m.s2->pool_pages, D.d.s2->pool_pages);
        g6s2::RunPool pool((uint32_t)(cap / 64), (uint32_t)place, (uint32_t)s2_below);
        // the pool's R is the CP map's: a run past R takes its own pages — the prefix and its shard below place, the
        // drafter's tail copy above it — instead of the whole pool, and retained runs keep their pages through a long
        // turn. The stores' tail blocks (per_slot) come with --kv-store below.
        if (m.cp.on && s2_below == m.cp.R) pool.tail((uint32_t)g6nr::NR, 0);
        if (rtt && rtt->local)
        {
            if (!pool.split()) die("GR8_RT_LOCAL needs the pool's page classes (GR8_S2_BELOW = GR8_CP_R)");
            // a tail waits until its records fit every rank's lease, a session past R until the slots cover it
            pool.window((uint32_t)rtt->local, (uint32_t)rtt->V, (uint32_t)rt_runs);
            if (R == 0) fprintf(stderr, "# rank 0: GR8_RT_LOCAL: a run past R holds %d prefix + %d window pages below place %d, its whole tail above "
                                        "(%d pages: the longest tail one run can have, %d rows); %d record ids per rank\n", cp_R / 64, rtt->local, place,
                                cap / 64 - place, (cap / 64 - place) * 64, rtt->V);
        }
        int reuse_grid = 0;
        // the offline loop's rules (prefix_reuse.h), but on by default: the grid rule (exact hits; GR8_REUSE_ALIGN=0 drops it)
        // and NOEND (GR8_REUSE_NOEND=0 restores the prompt >= written_end rule, which refuses replaying a prompt as it is)
        if (reuse_prefix)
        {
            const bool align = knob::reuse_align().value_or(true);
            if (align && !MP) die("GR8_REUSE_ALIGN=1 needs tensor prefill (--prefill N)");
            PrefixReuse rules;
            rules.strict = knob::reuse_strict(); rules.noend = knob::reuse_noend().value_or(true); rules.grid = align ? MP : 0;
            pool.keep(rules);
            reuse_grid = rules.grid;
            if (knob::pool_log()) pool.log(true);   // one stderr line per pool claim: first page, pages, evicted retained runs
            if (R == 0) fprintf(stderr, "# s2 reuse on: retired runs kept until a claim overlaps them; grid %d rows\n", rules.grid);
        }
        // --kv-store: block stores below the pool, in flag order, the same number of blocks on every rank. An SSD store lives in
        // PATH/r<rank>; its system prefixes and the counts survive restarts (their index needs --set-id: the blocks belong to
        // one model). A store's keep=, else --kv-keep, sets its keep rule and policy (kvman-rs stash.rs Keep, Policy)
        std::unique_ptr<g6s2::KvTiers> kvt;
        bool kv_index = false;
        bool kv_whole = false;   // a persist=on ssd store that keeps sessions: the quit stores the pool's runs there
        if (!kv_tiers.empty())
        {
            std::vector<const G6AttnBinding*> bs;
            for (Model* x : {&m, &D.d}) for (G6AttnBinding* b : x->s2->pool) bs.push_back(b);
            size_t nidx = 0;
            for (const G6AttnBinding* b : bs) nidx += b->args.index_cache != nullptr;
            char host[64] = "host";
            gethostname(host, sizeof host - 1);
            // the lease name on a KV node carries the format (its blocks are sized by it)
            const std::string name = std::string("g6m4-") + host + "-" + std::to_string(s2epoch) + "-r" + std::to_string(R) +
                                     (kv_format == "fp8" ? "" : "-" + kv_format);
            try
            {
                kvt.reset(new g6s2::KvTiers(bs));
                pool.stores((uint32_t)kv_system_after);
                for (size_t i = 0; i < kv_tiers.size(); ++i)
                {
                    const g6s2::TierSpec t = g6s2::TierSpec::parse(kv_tiers[i], kv_old);
                    std::string spec = t.store, index;
                    if (t.ssd())
                    {
                        // without persist=on the folder starts empty: a boot that served wrong output (a degenerate boot)
                        // must not hand its blocks or prefix counts to the next one
                        const std::string dir = t.store.substr(4) + "/r" + std::to_string(R);
                        if (!t.persist && std::filesystem::exists(dir))
                        {
                            std::filesystem::remove_all(dir);
                            fprintf(stderr, "# rank %d: s2 store %zu: %s cleared at boot (persist=on keeps it)\n", R, i, dir.c_str());
                        }
                        std::filesystem::create_directories(dir);
                        spec = "ssd:" + dir + "/blocks";
                        if (t.persist && !set_id.empty()) index = dir;
                    }
                    if (t.spare_ram) spec = "ram:" + std::to_string(t.spare_ram_floor) + "+" + spec;
                    // group=: a lease that shares its node with other clusters (s2/kv_share.h); it starts at size=
                    const uint64_t slots = t.group.empty() ? kvt->add(spec, name + "-t" + std::to_string(i), t.bytes)
                                                           : kvt->add_group(spec, name + "-t" + std::to_string(i), t.group, (uint32_t)g6nr::NR,
                                                                            (uint32_t)R, t.bytes, t.cap, t.weight);
                    // what makes stored blocks this engine's: the model, the layout, the block format, and the engine binary
                    // itself (another build may compute other KV bits); an index with another line starts empty
                    static const std::string exe = g6s2::exe_stamp();
                    const std::string compat = "set " + set_id + " nr " + std::to_string(g6nr::NR) + " below " + std::to_string(s2_below) +
                        " place " + std::to_string(place) + " grid " + std::to_string(reuse_grid) + " block " +
                        std::to_string(kvt->blocks().slot_bytes()) + " bindings " + std::to_string(bs.size()) + "/" +
                        std::to_string(nidx) + " slots " + std::to_string(slots) + " format " + std::to_string(g6s2::KV_BLOCK_FORMAT) +
                        " engine " + exe
#if G6_KV4
                        + " kv " + g6kv::NAME   // the latent row's layout (kvfmt.h); the FP8 row keeps the old key
#endif
                        ;
                    const std::string& keep = t.rule.empty() ? kv_keep : t.rule;
                    // persist=on ssd with keep=sessions or all: the index records every entry (+ 4), not only prefixes, so
                    // the runs the quit stores (below, after service.run) come back at the next boot with their sessions
                    const bool whole = t.ssd() && !index.empty() && keep != "system";
                    kv_whole = kv_whole || whole;
                    pool.add_tier(slots, (keep == "system" ? 1 : 2) | (whole ? 4 : 0), index, compat);   // shared prefixes, or those and evicted runs
                    pool.policy((uint32_t)i, 0, keep == "sessions" ? 1 : 0, kvt->blocks().slot_bytes());
                    kv_index = kv_index || !index.empty();
                    fprintf(stderr, "# rank %d: s2 store %zu: %s, %llu blocks of %zu B, keep %s%s\n", R, i, kvt->describe(i).c_str(),
                            (unsigned long long)slots, kvt->blocks().slot_bytes(), keep.c_str(), index.empty() ? "" : ", indexed");
                }
                if (const uint64_t gone = pool.tiers_loaded())
                    fprintf(stderr, "# rank %d: s2 stores: %llu stored records dropped (base came back in no store, a whole index reopened with keep=system, or slots past the store or taken twice)\n", R, (unsigned long long)gone);
                pool.write_limit((uint64_t)std::llround(kv_write_limit * 1000));
                // a run past R keeps its tail: each rank's shard and the replicated parts (only when the pool's R is the CP
                // map's, so the pages past it are the sharded ones)
                if (m.cp.on && s2_below == m.cp.R && kvt->blocks().per_slot()) pool.tail((uint32_t)g6nr::NR, kvt->blocks().per_slot());
            }
            catch (const std::exception& e) { die(std::string("--kv-store: ") + e.what()); }
            if (R == 0)
                fprintf(stderr, "# s2 stores: keep %s where a store sets none, a system prefix is stored after %d conversations%s\n", kv_keep.c_str(), kv_system_after,
                        kv_write_limit > 0 ? (", write limit " + std::to_string(kv_write_limit) + " blocks per page computed").c_str() : "");
        }
        // the copies the pool queued: run them before anything writes those pages, then commit what they wrote
        // Every rank prints its drain with the wall clock (ms since the Unix epoch): the drain runs after the event fence
        // and before the next quantum's fabric waits (G6_SPIN_TIMEOUT_CYC, ~10 s), so the skew between ranks' drains is
        // the number that matters, and only the lines of every rank show it.
        std::atomic<bool> kv_drained{false};   // the head drains a cancelled run on its control thread
        // GR8_KV_DRAIN_DELAY=<rank>:<ms>[:<every>], TEST ONLY: that rank sleeps <ms> inside every <every>th tier drain that
        // copies anything (default every one), as a drain stalled on its store would. Reproduces the rank skew the
        // barrier after a drain (NativeHooks::drained) has to absorb.
        int kv_delay_rank = -1, kv_delay_ms = 0, kv_delay_every = 1;
        long kv_delay_n = 0, kv_drain_n = 0;   // kv_drain_n: this rank's drains that copied anything (the same count on every rank)
        if (const auto& e = knob::kv_drain_delay())
            if (sscanf(e->c_str(), "%d:%d:%d", &kv_delay_rank, &kv_delay_ms, &kv_delay_every) < 2 || kv_delay_ms < 0 || kv_delay_every < 1)
                die("GR8_KV_DRAIN_DELAY=<rank>:<ms>[:<every>]");
        // rank 0, at most once a minute: what the stores did since the last line, once one fills, passes entries on, drops
        // them or refuses them (the store rotates silently otherwise). Counts only: the clock decides nothing else
        std::vector<g6kv_tier_totals> kv_logged;
        auto kv_last_log = std::chrono::steady_clock::now();
        bool kv_logged_once = false;
        auto kv_log = [&] {
            const auto now = std::chrono::steady_clock::now();
            if (R != 0 || (kv_logged_once && now - kv_last_log < std::chrono::minutes(1))) return;
            const auto ts = pool.tiers();
            kv_logged.resize(ts.size(), g6kv_tier_totals{});
            bool any = false;
            for (size_t i = 0; i < ts.size(); ++i)
            {
                const auto &a = kv_logged[i], &b = ts[i];
                if (b.full == a.full && b.moved_out == a.moved_out && b.dropped == a.dropped && b.refused == a.refused) continue;
                const double gb = b.block_bytes / 1e9;
                fprintf(stderr, "# s2 store %zu: %llu of %llu blocks used, %u entries (%u prefixes); in the last %.0f s: %.1f GB spilled, "
                                "%.1f GB restored, full %llu times, %llu entries (%.1f GB) passed on, %llu (%.1f GB) dropped, %llu (%.1f GB) "
                                "refused by the keep policy or write limit\n",
                        i, (unsigned long long)(b.slots - b.free_slots), (unsigned long long)b.slots, b.entries, b.shared,
                        std::chrono::duration<double>(now - kv_last_log).count(), (b.spilled_pages - a.spilled_pages) * gb,
                        (b.restored_pages - a.restored_pages) * gb, (unsigned long long)(b.full - a.full),
                        (unsigned long long)(b.moved_out - a.moved_out), (b.moved_out_pages - a.moved_out_pages) * gb,
                        (unsigned long long)(b.dropped - a.dropped), (b.dropped_pages - a.dropped_pages) * gb,
                        (unsigned long long)(b.refused - a.refused), (b.refused_pages - a.refused_pages) * gb);
                any = true;
            }
            if (any) { kv_logged = ts; kv_last_log = now; kv_logged_once = true; }
        };
        auto kv_drain = [&] {
            if (!kvt) return;
            const auto io = pool.io();
            const long long kv_t0 = g6s2::s2_wall_ms();
            kvt->run(io, st);
            kv_log();
            if (!io.empty())
            {
                kv_drained = true;
                if (R == kv_delay_rank && ++kv_delay_n % kv_delay_every == 0)
                {
                    fprintf(stderr, "# rank %d: GR8_KV_DRAIN_DELAY: sleeping %d ms in tier drain %ld at %lld ms\n", R, kv_delay_ms, kv_delay_n,
                            (long long)g6s2::s2_wall_ms());
                    std::this_thread::sleep_for(std::chrono::milliseconds(kv_delay_ms));
                }
                size_t n[3] = {0, 0, 0};
                for (const auto& x : io) ++n[x.kind];
                fprintf(stderr, "# rank %d: s2 tier io: %zu spilled, %zu moved, %zu restored pages in %.1f ms (device %.1f ms) at %lld ms%s; drain %ld from %lld ms: %s\n", R,
                        n[0], n[1], n[2], kvt->last_ms, kvt->last_device_ms, (long long)g6s2::s2_wall_ms(), kvt->last_ms > 2000 ? " SLOW" : "",
                        ++kv_drain_n, kv_t0, kvt->last_io_text().c_str());
            }
            if (const char* e = pool.io_done()) fprintf(stderr, "# rank %d: s2 tier index: %s\n", R, e);
        };
        // GR8_S2_BIND_DELAY=<rank>:<ms>: that rank sleeps after its bind (after the drain), as a drain that slow would. Shows
        // what a bind skew does to the next quantum on the other ranks (nothing else changes), with or without tiers.
        int bind_delay_rank = -1, bind_delay_ms = 0;
        if (const auto& e = knob::s2_bind_delay())
            if (sscanf(e->c_str(), "%d:%d", &bind_delay_rank, &bind_delay_ms) != 2 || bind_delay_ms < 0) die("GR8_S2_BIND_DELAY=<rank>:<ms>");
        m.s2on = D.d.s2on = true;        // no session: the adapter starts in S2 mode (solo -1)
        G6RT_CK(cudaStreamSynchronize(st));
        // DFlash2: Dfl::pf_feat holds one request's prefill taps, the last WIN prompt positions of the session whose
        // prefill started last (dfl_owner; -1 = none, or a prefix-reuse hit that skipped part of the window, unless
        // GR8_DFL_REUSE: then the rows from dfl_from)
        static_assert(s2::DFL_B == dfl::B, "s2_batch.h DFL_B follows dfl::B");
        static_assert(s2::DFL_WIN == dfl::WIN && s2::DFL_RING == dfl::RING, "s2_batch.h DFL_WIN / DFL_RING follow dfl::WIN / RING");
        int dfl_owner = -1;
        int dfl_from = 0;                // the first window position the owner's prefill taps (GR8_DFL_REUSE: past a hit's rows)
        uint64_t dfl_req = UINT64_MAX;
        std::array<uint32_t, g6s2::MAX_SESSIONS> slot_prompt{};
        std::array<bool, g6s2::MAX_SESSIONS> slot_whole{};   // GR8_DFL_REUSE: a hit holds every prompt row but the last (no prefill)
        u64 s2_dstep = 0;                // the drafter fabric's steps
        cudaStream_t dfl_par_st = st;    // GR8_DRAFT_PAR: the stream of the drafter step in flight
        const bool dfl_tl = R == 0 && knob::s2_roundlog();
        std::chrono::steady_clock::time_point dfl_t0;
        g6s2::NativeContext nx{m, D, P, P2, DQ, fab, fabd, fabp, fabp2, st, pf_streams ? &lanes2 : nullptr, in, in_d, out, out_d, MP, gsplit, two,
            [&](const StepIn& si) {
                std::vector<uint32_t> sizes(NC, (uint32_t)(si.M * HID * 2));
                sizes[2 * NL] = head_size(si, nc);
                extra_coll_sizes(sizes, NL, si.M);
                fab.enqueue_step(si.step, sizes);
                cp_enqueue(si.step, si.M, si.context);
                qs_enqueue(si.step, si.M);
            },
            [&](int k, int M) {
                g6s2::require(k >= 0 && k < m.nbk && M >= 1 && M <= MX && exec[k][M] && exec2[k][M], "S=1 target graph not captured");
                launch_main(k, M);
            },
            [&](int k, bool chain, int M) {
                g6s2::require(k >= 0 && k < D.d.nbk && M >= 1 && M <= MX && exec_d[k][chain][M], "S=1 draft graph not captured");
                G6RT_CK(cudaGraphLaunch(exec_d[k][chain][M], st));
            },
            // the run's pages go into the layers' tables. Staged prefill reads index keys through them too.
            [&](const g6s2::Slot& sl) {
                const bool ex = pool.exclusive_run(sl.run);
                m.cp.pf_R = (int)sl.req.r_rows;   // the pages its prefill stages follow its R
                if (ex)
                {
                    if (pool.split())   // the run's own pages: its CP map over its list
                    {
                        const std::vector<int> run = pool.list(sl.run);
                        const int most = std::max(m.s2->pool_pages, D.d.s2->pool_pages);
                        s2_solo_cp_pages(m, run, most, m.s2->rp[sl.key.slot]);
                        s2_solo_cp_pages(D.d, run, most, m.s2->rp[sl.key.slot]);
                    }
                    else { s2_solo_built(m); s2_solo_built(D.d); }
                }
                else { const std::vector<int> run = pool.list(sl.run); s2_solo_pages(m, run); s2_solo_pages(D.d, run); }
                m.s2_pf_table = D.d.s2_pf_table = !ex;
                // OWN=4: the offline loop's per-request rule (prompt + gen against the QO key table). The CP staging pool holds
                // S2 prefill rows by position, and the run's pages never reach the QO prepass.
                if (pf_qp)
                {
                    const int ctx = (int)(sl.req.prompt_tokens + sl.req.max_tokens);
                    const bool on = pf_qp_select(P, two ? &P2 : nullptr, ctx, (int)sl.req.prompt_tokens);
                    fprintf(stderr, "# rank %d: s2 request %llu prompt %u ctx %d (prompt + gen) -> %s (QP_MIN %d, QP_CTX %d) first %u at %lld ms\n", R,
                            (unsigned long long)sl.req.id, sl.req.prompt_tokens, ctx, on ? "OWN=4 pair-owner" : "OWN=2", pf_qp_min(), P.qp_ctx,
                            (unsigned)sl.prefilled, (long long)g6s2::s2_wall_ms());
                }
                if (dfl && DF.pf_feat && sl.req.id != dfl_req)   // a request's first prefill quantum: as the offline loop per sequence
                {
                    dfl_req = sl.req.id;
                    DF.pf_hi = (int)sl.req.prompt_tokens - 1; DF.pf_lo = std::max(0, DF.pf_hi - dfl::WIN);
                    if (DF.pf_hi > DF.pf_lo) G6RT_CK(cudaMemsetAsync(DF.pf_feat, 0, (size_t)(DF.pf_hi - DF.pf_lo) * dfl::NT * HID * 2, st));
                    dfl_owner = (int)sl.prefilled <= DF.pf_lo || dfl_reuse ? (int)sl.key.slot : -1;
                    dfl_from = std::max(DF.pf_lo, (int)sl.prefilled);
                }
            },
            [&]() { m.s2_pf_table = D.d.s2_pf_table = false; },
            [&]() { cp_pf_in(); }, [&]() { cp_pf_out(); },
            [&]() -> unsigned {   // cp: the CP decode fabric; cp_pf|qp: the CP prefill staging fabric (0x63..), else the QO pair wire (0x65..)
                const unsigned c = cp_error(), p = cp_pf_err();
                if (c || p) fprintf(stderr, "# rank %d: S2 extra error words: cp %08x cp_pf|qp %08x\n", R, c, p);
                return c | p;
            },
            [&](unsigned long long step, int rows) { qsd_enqueue(step, rows); }};
        // GR8_RT_STORE: slot q's table as the attention reads it: the layers' own tables while it runs alone (solo), its
        // slot table in batched steps
        std::vector<int> rt_tab;
        bool rt_told[g6s2::MAX_SESSIONS] = {};   // rank 0 described the session's first spill past L
        auto rt_slot_map = [&](int q) -> const std::vector<int>& {
            const S2Model& s = *m.s2;
            const int* tab = m.s2on ? s.table[q] : m.layers[0].ab.args.block_table;
            rt_tab.resize(m.s2on ? (size_t)s.table_cap : (size_t)m.layers[0].ab.args.table_size);
            G6RT_CK(cudaMemcpy(rt_tab.data(), tab, rt_tab.size() * sizeof(int), cudaMemcpyDeviceToHost));
            return rt_tab;
        };
        auto rt_log = [&](const char* why, int q) {
            if (R == 0) fprintf(stderr, "# rank 0: GR8_RT_STORE %s slot %d: %llu rows spilled, %llu dropped pages, %llu fetched, %llu read locally (no slot)\n",
                                why, q, (unsigned long long)rtt->spilled, (unsigned long long)rtt->dropped,
                                (unsigned long long)rt_fetched(*rtt), (unsigned long long)rt_local(*rtt));
            if (R == 0 && rtt->check) rt_check_print("spilled rows, rank 0", rtt->spill_ck);
        };
        if (rtt)
        {
            nx.before_main = [&](const StepIn& in, const S2In& h) {
                static bool first = true;
                if (R == 0 && first)
                {
                    first = false;
                    fprintf(stderr, "# rank 0: GR8_RT_STORE first main step: %d sessions, slot %d committed %d (%s), L %d\n", h.ns, h.sess[0],
                            in.pos[h.row0[0]], m.s2on ? "batched" : "solo", rtt->L);
                }
                for (int j = 0; j < h.ns; ++j)
                {
                    const int q = h.sess[j], committed = in.pos[h.row0[j]];
                    if (q < 0 || q >= m.s2->nslots || committed / 64 <= rtt->L / 64) continue;
                    G6RT_CK(cudaStreamSynchronize(st));
                    const RtSpillSeen x = rt_spill(m, *rtt, rt_slot_map(q), committed, q, st);
                    if (R == 0 && !rt_told[q])
                    {
                        rt_told[q] = true;
                        fprintf(stderr, "# rank 0: GR8_RT_STORE slot %d first spill (%s, committed %d, exclusive %d split %d): %d pages past L %d: %d spilled, "
                                        "%d another rank's or none, %d past the pool, %d already stored\n", q, m.s2on ? "batched" : "solo", committed,
                                (int)m.s2->exclusive[q], (int)m.s2->ex_split[q], x.pages, rtt->L, x.spilled, x.other, x.range, x.owned);
                    }
                }
            };
            // a later turn's prefill writes and reads the session's KV on the paths that read the local pages
            nx.before_prefill = [&](const g6s2::Slot& sl, uint32_t, uint32_t) {
                const int q = (int)sl.key.slot;
                bool any = false;
                for (int o : rtt->owner) any = any || o == q;
                rt_drop_slot(*rtt, q, st);
                if (any) rt_log("prefill of", q);
            };
        }
        // GR8_RT_LOCAL: slot q's tables with its tail entries (s2_cp_map reads them through S2Model::rt_tail): its slot
        // table always (the batched steps), the layers' tables while they hold its run (its prefill, or it runs alone)
        int rt_layers = -1;                // the slot whose run the layers' tables hold
        std::array<uint32_t, g6s2::MAX_SESSIONS> rt_gen{};   // the generation of the session bound to each slot
        int rt_pf = -1;                    // the slot of the prefill quantum in flight
        int rt_ma = 0, rt_mb = 0;          // its rows on lane A and B
        if (rtt && rtt->local)
        {
            const int most = std::max(m.s2->pool_pages, D.d.s2->pool_pages);
            rtt->write = [&, most](int q) {
                const S2Model& s = *m.s2;
                if (!s.ex_split[q]) return;
                const std::vector<int> t = s2_cp_map(s, s.pool[0]->args, s.run[q], s.rp[q], (size_t)s.table_cap);
                G6RT_CK(g6h2d::copy(s.table[q], t.data(), t.size() * 4));
                if (rt_layers == q) s2_solo_cp_pages(m, s.run[q], most, s.rp[q]);
            };
            nx.before_main = [&](const StepIn& in, const S2In& h) {
                for (int j = 0; j < h.ns; ++j)
                {
                    const int q = h.sess[j];
                    if (q < 0 || q >= m.s2->nslots || !rtt->seq[q].on) continue;
                    if (!m.s2on) rt_layers = q;   // alone: the mode switch wrote its run into the layers' tables
                    int end = 0;
                    for (int r = h.row0[j]; r < h.row0[j] + h.rows[j]; ++r) end = std::max(end, in.pos[r] + 1);
                    rt_spill(m, *rtt, {}, in.pos[h.row0[j]], q, st);
                    rt_ensure(m, *rtt, q, end, st);
                }
            };
            // the quantum writes rows first.. through the layers' tables, which begin_prefill rewrites right after this
            nx.before_prefill = [&](const g6s2::Slot& sl, uint32_t first, uint32_t rows) {
                const int q = (int)sl.key.slot;
                // what the last quantum staged from the store: every quantum with GR8_RT_CHECK, else every 8th that staged
                static uint64_t pf_n = 0, pf_p0 = 0, pf_b0 = 0;
                static double pf_t0 = 0;
                if (R == 0 && rtt->pf_pages > pf_p0 && (rtt->hold || ++pf_n % 8 == 1))
                    fprintf(stderr, "# rank 0: GR8_RT_LOCAL prefill staging since the last line: %llu released pages a layer summed, %.3f GB read, "
                                    "%.1f ms on the host; next quantum rows %u..\n", (unsigned long long)(rtt->pf_pages - pf_p0),
                            (rtt->pf_bytes - pf_b0) / 1e9, rtt->pf_ms - pf_t0, first);
                if (R == 0 && rtt->sel_layers)
                {
                    fprintf(stderr, "# rank 0: GR8_RT_SELSTAT: %llu sampled layer fills selected %llu of %llu released rows (%.1f%%); next quantum rows %u..\n",
                            (unsigned long long)rtt->sel_layers, (unsigned long long)rtt->sel_rows, (unsigned long long)rtt->sel_rel,
                            100.0 * rtt->sel_rows / std::max<uint64_t>(1, rtt->sel_rel), first);
                    rtt->sel_layers = rtt->sel_rows = rtt->sel_rel = 0;
                }
                if (rtt->pf_pages > pf_p0 && (rtt->hold || pf_n % 8 == 1)) { pf_p0 = rtt->pf_pages; pf_b0 = rtt->pf_bytes; pf_t0 = rtt->pf_ms; }
                rt_layers = q;
                rt_pf = rtt->seq[q].on ? q : -1;
                if (rt_pf < 0) return;
                rt_spill(m, *rtt, {}, (int)first, q, st);
                rt_ensure(m, *rtt, q, (int)(first + rows), st);
                rt_ma = std::min<int>((int)rows, MP); rt_mb = (int)rows - rt_ma;
            };
            // the real-fabric fill of a staged layer: the quantum's session's released pages, from the store
            static const bool selstat = knob::rt_selstat();
            m.cp.rt_stage = [&](int i, int start, char* pool_k, size_t ixoff, cudaStream_t s) {
                if (rt_pf < 0) return;
                rt_pf_stage(*rtt, rt_pf, i, start, {{pool_k, ixoff}}, s);
                // GR8_RT_SELSTAT: at each full-index layer's fill, the lanes' latest lists (the previous full-index
                // layer's: GR8_PF_OWN=2/4 keep them in idx_all) against the released rows below the quantum
                if (selstat && m.layers[i].indexer && P.idx_all)
                    rt_selstat(*rtt, {{P.idx_all, rt_ma}, {two ? P2.idx_all : nullptr, rt_mb}}, start / 64 * 64, s);
            };
        }
        try
        {
            g6s2::NativeDispatch nd(nx, sc);
            g6s2::NativeHooks h;
            h.pool.pool = &pool;   // the driver grows, shrinks (workers) and releases runs on it itself
            // GLM's module (ModuleOps) and pool policy (PoolOps): their hooks as members over references to the locals they
            // use. The locals outlive them and the service
#define GLM_CAPTURES(X) X(R) X(m) X(D) X(DF) X(st) X(st_par) X(exec_f) X(fabf) X(in_f) X(out_f) X(plan) X(dfl_reuse) \
    X(rtt) X(pool) X(kvt) X(kv_drained) X(kv_drain) X(bind_delay_rank) X(bind_delay_ms) X(dfl_owner) X(dfl_from) \
    X(slot_prompt) X(slot_whole) X(s2_dstep) X(dfl_par_st) X(dfl_tl) X(dfl_t0) X(rt_told) X(rt_log) X(rt_layers) \
    X(rt_gen) X(nd)
            GLM_CAPTURES(G6S2_CAPTURE_TYPE)
            uint32_t big = 0;   // GR8_CP_R2_BIG, with GR8_CP_R2
#define GLM_POOL_ONLY(X) X(cap) X(rt_runs) X(cp_R2) X(big)
#define GLM_POOL_CAPTURES(X) X(pool) X(m) X(D) GLM_POOL_ONLY(X)
            GLM_POOL_ONLY(G6S2_CAPTURE_TYPE)
            struct GlmPool : g6s2::ModuleCtx
            {
                GLM_POOL_CAPTURES(G6S2_CAPTURE_REF)
                uint64_t on_digest() { return pool.digest(); }
                std::string on_refusal(uint64_t worst, uint32_t r) {
                    const auto v = pool.split_view(worst, r);
                    if (!v) return "";
                    const auto [lo, hi, lofree, hifree] = *v;
                    std::string why = "it takes " + std::to_string(lo) + " pages below place (" + std::to_string(lofree) + " free) and its tail of " +
                                      std::to_string(hi) + " above (" + std::to_string(hifree) + " free)";
                    if (hi > (uint64_t)(cap / 64 - std::min(m.s2->pool_pages, D.d.s2->pool_pages))) why += ": the tail is longer than the drafter class, it never fits";
                    if ((int)pool.window_runs() >= rt_runs)
                        why += "; the slot pool covers " + std::to_string(rt_runs) + " session(s) past R at once (GR8_RT_POOL), all in use";
                    if (const auto iv = pool.ids_view(worst, r))
                    {
                        why += "; its tail needs " + std::to_string((*iv)[0]) + " records a rank, " + std::to_string((*iv)[1]) + " of the lease's " +
                               std::to_string((*iv)[2]) + " are free";
                        if ((*iv)[0] > (*iv)[2]) why += ": more than the lease holds (KV_ROWS size=), it never fits";
                    }
                    return why;
                }
                g6s2::PoolHit on_match(const std::vector<int>& p, uint64_t rows, uint64_t worst, int limit, const CacheNamespace& ns, uint32_t r) {
                    return pool.match(p, rows, worst, limit, ns, r);
                }
                g6s2::Run on_take(const g6s2::PoolHit& x, uint64_t rows, uint64_t worst, uint32_t r) { return pool.take(x, rows, worst, r); }
                uint32_t on_r_rows(const g6s2::Request& r, uint64_t rows, uint64_t worst, const std::vector<int>* prompt) {
                    if (worst <= (uint64_t)cp_R2) return 0;   // below both Rs: one layout
                    if (prompt)
                    {
                        const CacheNamespace ns = g6rt::steering::cache_namespace(r.steering);
                        const int limit = int(r.prompt_tokens) - 1;
                        const bool at_big = pool.match(*prompt, rows, worst, limit, ns, 0).skip > 0;
                        const bool at_small = pool.match(*prompt, rows, worst, limit, ns, (uint32_t)cp_R2).skip > 0;
                        if (at_big != at_small) return at_big ? 0 : (uint32_t)cp_R2;
                    }
                    return pool.window_runs(0) < big && pool.fits(rows, worst, 0) ? 0 : (uint32_t)cp_R2;
                }
                std::optional<g6s2::Shrink> on_make_room(const std::array<g6s2::Slot, g6s2::MAX_SESSIONS>& slots, const g6s2::Request&, uint64_t, uint64_t) {
                    for (const g6s2::Slot& sl : slots)
                        if (sl.phase == g6s2::Phase::Decode && !sl.req.r_rows && pool.shrinkable(sl.run, (uint32_t)cp_R2))
                            return g6s2::Shrink{sl.key, (uint32_t)cp_R2, pool.shrink(sl.run, (uint32_t)cp_R2)};
                    return std::nullopt;
                }
                std::vector<g6kv_tier_totals> on_tiers() { return pool.tiers(); }
            };
            GlmPool gp{{}, GLM_POOL_CAPTURES(G6S2_CAPTURE_ARG)};
#undef GLM_POOL_CAPTURES
#undef GLM_POOL_ONLY
            using GP = GlmPool;
            g6s2_pool_ops& po = h.pool.ops = g6s2::pol::table(gp, pool);
            po.digest = g6s2::pol::digest<GP, &GP::on_digest>;
            if (rtt && rtt->local) po.refusal = g6s2::pol::refusal<GP, &GP::on_refusal>;
            po.status = true;
            // GR8_S2_GROW=0: every run holds its worst case (prompt + max_tokens + guard) from admission, as before growth
            po.grow = knob::s2_grow();
            if (pool.keeping())
            {
                po.match = g6s2::pol::match<GP, &GP::on_match>;
                po.take = g6s2::pol::take<GP, &GP::on_take>;
            }
            // GR8_CP_R2: up to GR8_CP_R2_BIG sessions past it run at GR8_CP_R (their decode reads no released rows below
            // it); the others, and one the pool has no room for at GR8_CP_R, take GR8_CP_R2. A turn that continues a
            // retained run keeps that run's R, since a hit never crosses Rs (kvman-rs Pool::match_prefix_r)
            if (cp_R2)
            {
                big = (uint32_t)knob::cp_r2_big();
                po.r_rows = g6s2::pol::r_rows<GP, &GP::on_r_rows>;
                if (R == 0) fprintf(stderr, "# rank 0: GR8_CP_R2 %d: up to %u sessions past it run at GR8_CP_R %d, the rest at %d\n", cp_R2, big, cp_R, cp_R2);
                // a request that does not fit: a decoding session at GR8_CP_R takes GR8_CP_R2, and its pages between the two Rs
                // below place go back to the pool (s2/pool.h shrink, the engine's half below)
                po.make_room = g6s2::pol::make_room<GP, &GP::on_make_room>;
            }
            if (kvt)
            {
                po.tiers = g6s2::pol::tiers<GP, &GP::on_tiers>;
                po.count = true;
            }
            struct GlmModule : g6s2::ModuleCtx
            {
                GLM_CAPTURES(G6S2_CAPTURE_REF)
                void on_retain(const g6s2::Run& r, const PrefixReuse& s, const CacheNamespace& ns) { pool.retain(r, s, ns); }
                void on_retain_drain(const g6s2::Run& r, const PrefixReuse& s, const CacheNamespace& ns) { pool.retain(r, s, ns); kv_drain(); }   // a promoted prefix
                void on_bind(const g6s2::Grant& g) {     // at the admission fence, every rank, main and draft
                    // the store copies the grant queued: evicted runs out before anything writes their pages, a stored run in
                    kv_drain();
                    if (R == bind_delay_rank && bind_delay_ms)
                    {
                        fprintf(stderr, "# rank %d: GR8_S2_BIND_DELAY: sleeping %d ms after the bind of request %llu at %lld ms\n", R, bind_delay_ms,
                                (unsigned long long)g.req.id, (long long)g6s2::s2_wall_ms());
                        std::this_thread::sleep_for(std::chrono::milliseconds(bind_delay_ms));
                    }
                    const int q = (int)g.key.slot;
                    const int rp = int(g.req.r_rows ? g.req.r_rows : uint32_t(m.s2_below)) / 64;   // the run's prefix pages
                    if (rtt && !rtt->local) { rt_drop_slot(*rtt, q, st); rt_drop(*rtt, pool.list(g.run), st); rt_told[q] = false; }   // a new run's pages are local
                    // release: a split run's tail starts unmapped, cycling through its window (list[rp ..], kvman-rs Pool::window)
                    if (rtt && rtt->local)
                    {
                        if (rt_layers == q) rt_layers = -1;
                        rt_gen[q] = g.key.generation;
                        // a hit on a retained window run takes its window pages, and its records come with them: from its
                        // parked entry, or from its slot when its retirement is at this boundary and not bound yet (the
                        // slot's own included), so it is found before anything ends that slot
                        const bool ex = pool.exclusive_run(g.run);
                        const std::vector<int> l = ex ? pool.list(g.run) : std::vector<int>{};
                        const std::vector<int> win = ex ? std::vector<int>(l.begin() + rp, l.begin() + rp + rtt->local) : std::vector<int>{};
                        // only a hit on a run the pool retained brings records, at the grant's own R: a stored hit restores a
                        // prefix, and a fresh grant can land on the window pages of a parked run the pool already let go
                        int pk = ex && g.reuse && !g.restored ? rt_parked(*rtt, win) : -1;
                        if (pk >= 0 && rtt->seq[pk].L != rp * 64) pk = -1;
                        if (pk != q) rt_end(*rtt, q, st);
                        // the runs the pool let go (evicted, or adopted only in part below R) free their records first. The
                        // head's take already moved this grant's run out of the kept list, so its entry stays.
                        rt_gc(*rtt, [&](const std::vector<int>& kl) { return (pk >= 0 && kl == rtt->seq[pk].list) || pool.is_kept(kl); }, st);
                        if (ex)
                        {
                            if (pk >= 0)
                            {
                                rt_adopt(m, *rtt, pk, q, l, (int)g.reuse, st);
                                if (R == 0) fprintf(stderr, "# rank 0: GR8_RT_LOCAL slot %d adopts a retained run's records at row %u\n", q, g.reuse);
                            }
                            else if ((int)g.reuse > rp * 64)   // every rank holds the same entries, so they all stop here alike
                                die("GR8_RT_LOCAL: a hit past R on a run whose records no slot or parked entry holds");
                            else rt_begin(*rtt, q, win, l, rp * 64, st);
                        }
                    }
                    slot_prompt[q] = g.req.prompt_tokens;
                    slot_whole[q] = dfl_reuse && g.reuse + 1 == g.req.prompt_tokens;
                    if (dfl_owner == q) dfl_owner = -1;     // the slot's previous session owned the taps
                    const bool ex = pool.exclusive_run(g.run);
                    // a prefix-reuse hit keeps the run's pages: rows below g.reuse are the retained session's, and every row
                    // past them is written by this request before any step reads it (prefix_reuse.h)
                    const std::vector<int> run = pool.list(g.run);
                    const int most = std::max(m.s2->pool_pages, D.d.s2->pool_pages);
                    for (Model* x : {&m, &D.d})
                    {
                        if (ex)
                        {
                            if (pool.split()) s2_set_exclusive(*x, q, run, most, rp);   // its own list; the slot view stays unused while solo
                            else s2_set_exclusive(*x, q);
                            // alone: the run's pages clear (the whole pool on the identity map, today's reset)
                            if (!g.reuse)
                            {
                                if (pool.split()) s2_reset_pages(*x, run, st, most);
                                else reset(*x, st);
                            }
                        }
                        else
                        {
                            s2_set_pages(*x, q, run);
                            if (!g.reuse) s2_reset_pages(*x, x->s2->run[q], st, most);
                        }
                        for (int l = 0; l < x->nl; ++l)  // slot q's views (s2_reset, one slot)
                        {
                            const G6AttnBinding* v = x->s2->sab[l][q];
                            G6RT_CK(cudaMemsetAsync(v->args.indices, 0xff, (size_t)v->max_rows * 2048 * 4, st));
                            G6RT_CK(cudaMemsetAsync(v->args.output, 0, (size_t)v->max_rows * v->args.heads * 256 * 2, st));
                            G6RT_CK(cudaMemsetAsync(v->args.error, 0, 4, st));
                        }
                    }
                    G6RT_CK(cudaStreamSynchronize(st));
                }
            // a run grew between rounds (every rank, same order): the new pages, cleared as a new run's are, into slot q's
            // table, and into the layers' own tables when q runs alone on today's path
            // a session takes GR8_CP_R2 (every rank at the commit, before the bind that may take its pages): the drafter's
            // rows of the pages between the two Rs into its fresh pages above place, this rank's main pages of them into the
            // rows store, then the tables
                void on_shrink(const g6s2::Shrink& k, const g6s2::Run&, uint64_t length) {
                    kv_drain();   // retained runs the shrink evicted, out first
                    if (!rtt || !rtt->local) die("a shrink without GR8_RT_LOCAL");
                    rt_gc(*rtt, [&](const std::vector<int>& l) { return pool.is_kept(l); }, st);
                    const int q = (int)k.key.slot, rp = (int)k.r_rows / 64, was_rp = m.s2->rp[q], win = rtt->local;
                    const std::vector<int> was = m.s2->run[q], list = pool.list(k.run);
                    const int most = std::max(m.s2->pool_pages, D.d.s2->pool_pages);
                    std::vector<std::pair<int, int>> moves;
                    for (int lp = rp; lp < was_rp; ++lp) moves.push_back({was[lp], list[rp + win + (lp - rp)]});
                    s2_copy_pages(D.d, moves, most, st);
                    rt_shrink(m, *rtt, q, was, list, (int)k.r_rows, (int)length - 1, st);
                    for (Model* x : {&m, &D.d}) s2_set_exclusive(*x, q, list, most, rp);
                    if (rt_layers == q)   // it runs alone: the layers' tables hold its run
                    {
                        s2_solo_cp_pages(m, list, most, rp);
                        s2_solo_cp_pages(D.d, list, most, rp);
                    }
                    G6RT_CK(cudaStreamSynchronize(st));
                    if (R == 0) fprintf(stderr, "# rank 0: GR8_CP_R2: slot %d takes R %u, %d prefix pages back below place\n", q, k.r_rows, was_rp - rp);
                }
                void on_grown(unsigned q, const g6s2::Run& r) {
                    kv_drain();   // runs the growth evicted, out before the zeroing below
                    if (rtt && rtt->local) rt_gc(*rtt, [&](const std::vector<int>& l) { return pool.is_kept(l); }, st);
                    const std::vector<int> list = pool.list(r);
                    if (rtt && list.size() > m.s2->run[q].size())   // the new pages are cleared: none may stay flagged
                        rt_drop(*rtt, std::vector<int>(list.begin() + m.s2->run[q].size(), list.end()), st);
                    for (Model* x : {&m, &D.d})
                    {
                        const std::vector<int> old = x->s2->run[q];
                        g6s2::require(list.size() > old.size() && std::equal(old.begin(), old.end(), list.begin()), "grown run does not extend the slot's pages");
                        s2_zero_pages(*x, std::vector<int>(list.begin() + old.size(), list.end()), st);
                        s2_set_pages(*x, (int)q, list);
                        if (!x->s2on) s2_solo_pages(*x, list);
                    }
                    G6RT_CK(cudaStreamSynchronize(st));
                }
                void on_retire(const g6s2::Slot& old) {
                    // a grant at the same boundary may have bound the slot again (or adopted its records) first: then the
                    // slot is that session's, and the retirement leaves it alone
                    if (rtt && rtt->local && rt_gen[old.key.slot] != old.key.generation)
                    {
                        if (R == 0) fprintf(stderr, "# rank 0: GR8_RT_LOCAL retire of slot %u after its next grant: nothing to do\n", old.key.slot);
                    }
                    else if (rtt && rtt->local)   // a run with released pages parks its records
                    {
                        const int q = (int)old.key.slot;
                        const RtTier::Seq& sq = rtt->seq[q];
                        if (R == 0 && sq.on)
                            fprintf(stderr, "# rank 0: GR8_RT_LOCAL retire of slot %d: %llu pages released, at most %d of %d window pages in use; all sessions: "
                                            "%llu released, peak %d local tail pages; %llu rows fetched, %zu of %d record ids free\n", q,
                                    (unsigned long long)sq.released, sq.peak, sq.window, (unsigned long long)rtt->released, rtt->peak,
                                    (unsigned long long)rt_fetched(*rtt), rtt->vfree.size(), rtt->V);
                        if (rtt->hold)
                            fprintf(stderr, "# rank %d: GR8_RT_CHECK: %llu released group pages read back, %llu differed\n", R,
                                    (unsigned long long)rtt->readback, (unsigned long long)rtt->readback_bad);
                        // records park for the conversation's next turn whether or not the pool kept the run: on the head a
                        // grant at this boundary has already taken it out of the kept list, on the workers it has not. The
                        // next bind's rt_gc frees them if no grant adopts them.
                        if (sq.on)
                        {
                            if (R == 0) fprintf(stderr, "# rank 0: GR8_RT_LOCAL slot %d parks its records (kept by the pool: %d)\n", q, (int)pool.is_kept(m.s2->run[q]));
                            rt_park(*rtt, q);
                        }
                        if (rtt->hold && rtt->stage_ck.rows)
                        {
                            rt_check_print(("staged prefill rows, rank " + std::to_string(R)).c_str(), rtt->stage_ck);
                            fprintf(stderr, "# rank %d: GR8_RT_CHECK: %llu staged pages under 10 dB or with a rope byte off\n", R, (unsigned long long)rtt->stage_bad);
                        }
                        rt_end(*rtt, q, st);
                        if (rt_layers == q) rt_layers = -1;
                    }
                    else if (rtt) { rt_drop_slot(*rtt, (int)old.key.slot, st); rt_log("retire of", (int)old.key.slot); }   // retained runs stay local
                    nd.retire(old);
                }
                void on_fence() { nd.synchronize(); }
                bool on_drained() { return kv_drained.exchange(false); }
                void on_resize(const g6s2::Resize& r) {
                    if (!g6s2::share_apply(pool, kvt->store(r.tier), r, kv_drain))
                        fprintf(stderr, "# rank %d: s2 share: tier %u to %llu blocks refused (transaction %llu), the lease keeps its size: %s\n", R,
                                r.tier, (unsigned long long)r.slots, (unsigned long long)r.txn, g6kv_tier_error());
                    else
                        fprintf(stderr, "# rank %d: s2 share: tier %u now %llu segments, %llu blocks at %lld ms\n", R, r.tier,
                                (unsigned long long)r.segs, (unsigned long long)r.slots, (long long)g6s2::s2_wall_ms());
                }
                int on_bucket(bool draft, int ctx) { return nd.on_bucket(draft, ctx); }
                void on_mode(int old, int next) { nd.on_mode(old, next); }
                std::vector<int> on_launch(const g6s2::Launch& l) { return nd.on_launch(l); }
                void on_prefill(const g6s2::Slot& s, const g6s2::Payload& p, uint32_t first, uint32_t rows) { nd.on_prefill(s, p, first, rows); }
                g6rt::LpRow on_logprob(int row, bool target, int k) { return nd.on_logprob(row, target, k); }
                float on_draft_lp(int row) { return nd.on_draft_lp(row); }
                bool on_dfl_ok(int q) { return dfl_owner == q || slot_prompt[q] == 1 || slot_whole[q]; }
                int on_dfl_start(int q) {    // the offline loop's fill steps: Dfl::pf_feat -> the ring
                    if (dfl_owner != q || DF.pf_hi <= DF.pf_lo) return (int)slot_prompt[q] - 1;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int c0 = dfl_from; c0 < DF.pf_hi; c0 += dfl::B)
                    {
                        DflIn di;
                        memset(&di, 0, sizeof di);
                        di.step = ++s2_dstep;
                        di.ctx_rows = std::min(dfl::B, DF.pf_hi - c0); di.ctx_pos0 = c0; di.q0 = c0 + di.ctx_rows; di.lo = DF.pf_lo;
                        *in_f = di;
                        dfl_enqueue(fabf, di.step, dfl_fill_sizes(), true);
                        G6RT_CK(cudaGraphLaunch(exec_f[2], st));
                        G6RT_CK(cudaStreamSynchronize(st));
                        g6s2::require(!*fabf.dev().error, "S2 DFlash2 fabric error (prompt context)");
                    }
                    g6s2::s2_tl().cut();
                    if (R == 0) fprintf(stderr, "# s2 dflash slot %d prompt context %d..%d: %.1f ms\n", q, dfl_from, DF.pf_hi - 1,
                                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                    return dfl_from;
                }
                s2::DflDrafts on_dfl(const s2::DflReq& r, const g6s2::Sampling& smp) {
                    // GR8_DRAFT_PAR: phase 1 launches on the drafter's stream and returns, phase 2 waits for it (s2::DflReq).
                    // The target step before phase 1 was fenced on st, and phase 2 comes before the next one.
                    // GR8_S2_TIMELINE: phase 0 is a step in the target stream's sequence; phases 1 + 2 are timed apart (par_*)
                    cudaStream_t& ds = dfl_par_st;
                    g6s2::S2Timeline& tl = g6s2::s2_tl();
                    if (r.phase != 2)
                    {
                        if (r.phase) tl.par_enter(); else tl.enter();
                        DflIn di;
                        memset(&di, 0, sizeof di);
                        di.step = ++s2_dstep;
                        di.ctx_rows = r.rows; di.ctx_pos0 = r.pos0; di.q0 = r.pos0 + r.rows; di.lo = r.lo; di.tok0 = r.tok0; di.query = r.query;
                        const float dtemp = plan.rs ? 0.f : smp.inv_temp;   // rejection sampling: greedy drafts, as dflash_sequence
                        for (int i = 0; i < dfl::B; ++i) di.srow[i] = glue::SampleRow{dtemp, smp.seed, 0u, di.q0 + i, 0, smp.top_p};
                        *in_f = di;
                        dfl_enqueue(fabf, di.step, dfl_sizes(DF, di.query), false);
                        ds = r.phase ? st_par[di.query ? 1 : 0] : st;
                        if (dfl_tl) dfl_t0 = std::chrono::steady_clock::now();
                        if (r.phase) tl.par_pre(ds); else tl.pre(ds);
                        G6RT_CK(cudaGraphLaunch(exec_f[di.query ? 1 : 0], ds));
                        if (r.phase) tl.par_post(ds); else tl.post(ds);
                        if (r.phase == 1) return s2::DflDrafts{};
                    }
                    else tl.par_wait();
                    G6RT_CK(cudaStreamSynchronize(ds));
                    if (r.phase) tl.par_done(r.query ? g6s2::S2Timeline::DQUERY_PAR : g6s2::S2Timeline::DCTX_PAR, r.rows);
                    else tl.done(r.query ? g6s2::S2Timeline::DQUERY : g6s2::S2Timeline::DCTX, r.rows, R);
                    if (dfl_tl)   // GR8_S2_ROUNDLOG: the drafter step's launch to its wait (GR8_DRAFT_PAR: the first form and chain within)
                        fprintf(stderr, "# s2_dfl step %llu query %d par %d ms %.3f\n", (unsigned long long)s2_dstep, (int)r.query, r.phase ? 1 : 0,
                                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dfl_t0).count());
                    g6s2::require(!*fabf.dev().error, "S2 DFlash2 fabric error");
                    s2::DflDrafts o;
                    for (int i = 0; i < dfl::B; ++i) { o.tok[i] = out_f->tok[i]; o.lp[i] = out_f->lp[i]; }
                    return o;
                }
            };
            GlmModule gm{{}, GLM_CAPTURES(G6S2_CAPTURE_ARG)};
#undef GLM_CAPTURES
            using G = GlmModule;
            g6s2_module_ops& o = h.module = g6s2::mod::table(gm);
            if (kvt) o.retain = g6s2::mod::retain<G, &G::on_retain_drain>;
            else if (pool.keeping()) o.retain = g6s2::mod::retain<G, &G::on_retain>;
            o.bind = g6s2::mod::bind<G, &G::on_bind>;
            o.shrink = g6s2::mod::shrink<G, &G::on_shrink>;
            o.grown = g6s2::mod::grown<G, &G::on_grown>;
            o.retire = g6s2::mod::retire<G, &G::on_retire>;
            o.fence = g6s2::mod::call<G, &G::on_fence>;
            if (kvt) o.drained = g6s2::mod::drained<G, &G::on_drained>;
            // a tier on a group lease (--kv-store group=): every rank resizes it at the Resize commit; the head reports
            // to the node, asking for the segments an oversize entry, the pages the tier pushed out since its last
            // report, or the next spill would need (s2/kv_share.h share_want: ask before the next spill drops an entry),
            // and announces what the node grants or asks back
            std::vector<uint32_t> kv_groups;
            for (size_t i = 0; kvt && i < kvt->size(); ++i) if (kvt->group(i)) kv_groups.push_back((uint32_t)i);
            std::vector<g6kv_tier_totals> kv_seen(kv_groups.size(), g6kv_tier_totals{});
            size_t kv_next_group = 0;
            if (!kv_groups.empty())
            {
                o.resize = g6s2::mod::resize<G, &G::on_resize>;
                // GR8_KV_SHARE_WHEN=idle (default): the head offers its blocks to other groups only while the engine is
                // idle, so a give's compaction never pauses a turn; always: whenever the node asks
                const bool share_always = knob::kv_share_when() == "always";
                if (R == 0)
                    h.share = [&, share_always](bool idle) -> std::optional<g6s2::Resize> {
                        // one group tier a report, in turn: one Resize in flight at a time
                        const size_t k = kv_next_group++ % kv_groups.size();
                        const uint32_t t = kv_groups[k];
                        const g6kv_tier_totals now = pool.tiers().at(t);
                        const uint64_t want = g6s2::share_want(kv_seen[k], now, g6kv_store_seg_blocks(kvt->store(t)));
                        kv_seen[k] = now;
                        return g6s2::share_poll(pool, t, kvt->store(t), want, share_always || idle);
                    };
            }
            o.bucket = g6s2::mod::bucket<G, &G::on_bucket>;
            o.mode = g6s2::mod::mode<G, &G::on_mode>;
            o.launch = g6s2::mod::launch<G, &G::on_launch>;
            o.prefill = g6s2::mod::prefill<G, &G::on_prefill>;
            o.logprob = g6s2::mod::logprob<G, &G::on_logprob>;
            o.draft_lp = g6s2::mod::draft_lp<G, &G::on_draft_lp>;
            nd.grammar(o);
            if (dfl)
            {
                h.dfl_kv = dfl_k; h.dfl_spec = spec;
                // drafter choice: drafting ms per round at 2200 MHz. DFlash2: the drafter step and the MTP first form;
                // MTP: the first form, the drafter's context-only step, a chain step per draft after the first
                h.dsel.on = s2::dsel_env();
                // tuning (every rank must see the same values; the launcher's env reaches all of them)
                h.dsel.margin = knob::s2_dsel_margin();
                h.dsel.dwell = (int)std::max<int64_t>(1, knob::s2_dsel_dwell());
                h.dsel.probe = (int)std::max<int64_t>(1, knob::s2_dsel_probe());
                h.dsel.d.draft_round = 5.6 + 2.5;
                h.dsel.m.draft_round = 2.5 + 3.0;
                h.dsel.m.draft_each = 2.1;
                // sweeps (measurement only; every rank must see the same values, as above): GR8_SPEC_TABLE = verify ms for
                // 1..8 rows, GR8_SPEC_DRAFT = DFlash2 round, MTP round, MTP each (ms)
                auto ms_list = [](const std::vector<double>& v) {
                    std::string s;
                    char t[32];
                    for (double x : v) { snprintf(t, sizeof t, "%s%g", s.empty() ? "" : ",", x); s += t; }
                    return s;
                };
                const auto& spec_table = knob::spec_table();
                if (spec_table)
                {
                    const std::vector<double>& t = *spec_table;
                    if (t.size() != (size_t)SpecLen::KX + 1 || std::any_of(t.begin(), t.end(), [](double x) { return !(x > 0); }))
                        die("GR8_SPEC_TABLE: want 8 comma-separated finite positive verify times in ms");
                    for (int r = 1; r <= SpecLen::KX + 1; ++r) h.dfl_spec.table[r] = h.dsel.d.table[r] = h.dsel.m.table[r] = t[r - 1];
                    if (R == 0) fprintf(stderr, "# spec: verify table from GR8_SPEC_TABLE: %s\n", ms_list(t).c_str());
                }
                // GR8_SPEC_DRAFT's default is the three costs above, so a value that differs from them counts as set
                const std::vector<double>& draft_ms = knob::spec_draft();
                if (draft_ms.size() != 3 || std::any_of(draft_ms.begin(), draft_ms.end(), [](double x) { return x < 0; }))
                    die("GR8_SPEC_DRAFT: want 3 comma-separated finite nonnegative drafting times in ms");
                const bool spec_draft = draft_ms[0] != h.dsel.d.draft_round || draft_ms[1] != h.dsel.m.draft_round ||
                                        draft_ms[2] != h.dsel.m.draft_each;
                h.dsel.d.draft_round = draft_ms[0]; h.dsel.m.draft_round = draft_ms[1]; h.dsel.m.draft_each = draft_ms[2];
                if (R == 0 && spec_draft) fprintf(stderr, "# spec: drafting ms from GR8_SPEC_DRAFT: %s\n", ms_list(draft_ms).c_str());
                h.dsel.model = dsel_model;    // the table replaces the drafter choice's verify table and drafting ms above (Batch::dsel)
                h.dsel.tab = dsel_tab;
                if (R == 0 && dsel_model)
                {
                    fprintf(stderr, "# s2 drafter choice cost model %d: table %s\n", dsel_model, dsel_tab.str().c_str());
                    if (spec_table || spec_draft)
                        fprintf(stderr, "# note: the drafter choice reads the table's v1..v8, d, m, e, not GR8_SPEC_TABLE / GR8_SPEC_DRAFT; "
                                        "GR8_SPEC_TABLE still prices the DFlash2 rounds' own draft count\n");
                }
                if (R == 0 && h.dsel.on) fprintf(stderr, "# s2 drafter choice on: DFlash2 or MTP per round (dwell %d, margin %.2f, probe %d)\n",
                                                    h.dsel.dwell, h.dsel.margin, h.dsel.probe);
                h.kstop = kstop;
                if (R == 0 && kstop.mode)
                    fprintf(stderr, "# s2 MTP confidence stop: mode %d, solo MTP rounds verify 1..%d drafts (%s)%s\n", kstop.mode, kstop.kmax,
                            kstop.mode == 1 ? ("tau " + std::to_string(kstop.tau[1]) + "," + std::to_string(kstop.tau[2]) + "," +
                                               std::to_string(kstop.tau[3])).c_str()
                                            : ("MTP estimates, seed3 " + std::to_string(kstop.seed3)).c_str(),
                            h.dsel.on ? "" : dfl_reuse ? "; the drafter choice is off: GR8_DFL_REUSE warm-up rounds only" : "; the drafter choice is off, so no MTP rounds");
                if (R == 0 && kstop.b.mode)
                {
                    std::string t, w;
                    for (int S = 1; S <= s2sess; ++S)
                        t += (S > 1 ? " " : "") + std::to_string(S) + ":" + std::to_string(kstop.b.tau[S][1]).substr(0, 5) + "," +
                             std::to_string(kstop.b.tau[S][2]).substr(0, 5) + "," + std::to_string(kstop.b.tau[S][3]).substr(0, 5);
                    for (int M : s2::kstopb_widths(kstop.b, s2sess, mtp)) w += (w.empty() ? "" : ",") + std::to_string(M);
                    fprintf(stderr, "# s2 MTP shared-round stop: each session verifies %d..%d drafts, tau by sessions %s; main widths %s%s\n",
                            kstop.b.kmin, mtp, t.c_str(), w.c_str(),
                            kstop.b.explore ? (", explore every " + std::to_string(kstop.b.explore) + " rounds").c_str() : "");
                }
                h.dfl_reuse.mode = dfl_reuse;
                h.dfl_reuse.min = dfl_reuse_min;
                h.ngram = ngram;
                if (R == 0 && dfl_reuse) fprintf(stderr, "# s2 dflash reuse %d: reuse hits keep the drafter%s, warm-up %d context rows\n", dfl_reuse,
                                                 dfl_reuse > 1 ? ", a session alone on MTP takes it" : "", dfl_reuse_min);
                // a 1-token prompt has no prefill: the drafter starts from its prompt step (dfl_lo = first, as plain)
                // GR8_DFL_REUSE: also a hit into the window (the fill starts past its rows) and a hit with no prefill at all
                o.dfl_ok = g6s2::mod::dfl_ok<G, &G::on_dfl_ok>;
                o.dfl_start = g6s2::mod::dfl_start<G, &G::on_dfl_start>;
                o.dfl = g6s2::mod::dfl<G, &G::on_dfl>;
            }
            else if (kstop.mode || kstop.b.mode || kstop.log)
            {   // MTP only (TP=6 / TP=3): the confidence stops of round_all (KStop::solo, KStopB) and their log (s2/service.h)
                h.kstop = kstop;
                if (R == 0 && kstop.mode)
                    fprintf(stderr, "# s2 MTP confidence stop (MTP only): mode %d, a session alone verifies 1..%d drafts (tau %s,%s,%s)\n",
                            kstop.mode, kstop.kmax, std::to_string(kstop.tau[1]).c_str(), std::to_string(kstop.tau[2]).c_str(),
                            std::to_string(kstop.tau[3]).c_str());
                if (R == 0 && kstop.b.mode)
                {
                    std::string t;
                    for (int S = 1; S <= s2sess; ++S)
                        t += (S > 1 ? " " : "") + std::to_string(S) + ":" + std::to_string(kstop.b.tau[S][1]).substr(0, 5) + "," +
                             std::to_string(kstop.b.tau[S][2]).substr(0, 5) + "," + std::to_string(kstop.b.tau[S][3]).substr(0, 5);
                    fprintf(stderr, "# s2 MTP shared-round stop (MTP only): each session verifies %d..%d drafts, tau by sessions %s\n",
                            kstop.b.kmin, mtp, t.c_str());
                }
            }
            g6s2::BootstrapConfig bc;
            bc.scheduler = sc;
            bc.rank = (unsigned)R;
            bc.ranks = (unsigned)cfg.ips.size();
            bc.timeout = (unsigned)knob::s2_timeout();
            bc.head_ip = cfg.ips[0];
            bc.directory = s2dir;
            bc.port = (uint16_t)s2port;
            if (g6_attn_ctx_sp() || g6_attn_posidx())   // round 10 E1 attention flags (attn.cu), read once before the bindings
                fprintf(stderr, "# rank %d: decode attention E1: GR8_ATTN_CTX3_SP %d GR8_ATTN_POSIDX %d\n", R, g6_attn_ctx_sp(), g6_attn_posidx());
            g6s2::SigInputs sig;           // s2/signature.h
            sig.model = g6s2::SigModel::Glm;
            sig.s2_below = s2_below; sig.cap = cap; sig.layers_s = layers_s; sig.MP = MP; sig.two = two; sig.mtp = mtp;
            sig.chunk = plan.chunk; sig.nc = nc; sig.pf_own = pf_own; sig.pf_qp = pf_qp; sig.qp_ctx = P.qp_ctx; sig.qp_min = pf_qp_min();
            sig.set_id = set_id; sig.grammar = g6rt::grammar_ready(); sig.dfl_reuse = dfl_reuse; sig.dfl_reuse_min = dfl_reuse_min;
            sig.ngram = s2::ngram_sig(ngram);
            sig.kstop_mode = kstop.mode; sig.kstop_kmax = kstop.kmax; std::copy(kstop.tau, kstop.tau + 4, sig.kstop_tau); sig.kstop_seed3 = kstop.seed3;
            sig.draft_par = draft_par; sig.qs_mode = qs_mode; sig.iq_split = iq_split; sig.mtp_split = mtp_split;
            sig.dhead_on = dhead.on; sig.dhead_tpc = dhead.tpc; sig.dhead_ks = dhead.ks; sig.mtp_last = mtp_last; sig.dhead2 = dhead2;
            sig.dfl_i4 = dfl_i4; sig.dfl_body_i4 = dfl_body_i4_on; sig.dfl_alpha = dfl_alpha;
            sig.dsel_model = dsel_model; sig.dsel_tab = dsel_tab.str(); sig.kstopb = s2::kstopb_sig(kstop.b);
            sig.cp_step = cp_step; sig.cp_seg = sc.cp_seg; sig.ctx_sp = g6_attn_ctx_sp(); sig.posidx = g6_attn_posidx();
            sig.steer_shared = steering_boot.loaded && steering_boot.scope == steering::Scope::Shared;
            sig.knobs = knob::signature();
            bc.runtime_signature = g6s2::runtime_signature(sig);
            fprintf(stderr, "# rank %d: s2 below R %d rows, pool %u logical / %d physical pages, %d sessions, epoch %llu, control %s:%d\n", R,
                    s2_below, sc.pool_rows / 64, place, s2sess, s2epoch, bc.head_ip.c_str(), s2port);
            g6s2::ControlLinks links = g6s2::bootstrap(bc);
            if (kv_index)   // every rank keeps the stored prefixes all of them loaded, in the head's order
            {
                pool.keep_agreed(g6s2::agree_lines(bc, links, pool.stored_summary()));
                size_t n = 0;
                for (const auto& t : pool.tiers()) n += t.entries;
                fprintf(stderr, "# rank %d: s2 stores: %zu stored prefixes from earlier runs\n", R, n);
            }
            // every reachable S2 graph key, captured after the bootstrap (its timeout does not cover the capture) and before
            // the head publishes ready.s2 (Service::run): the all-rank barrier, with no receive timeout, replaces the skew limit
            // the batched graphs carry no DFlash2 taps (their rows outnumber Dfl::feat's 8); the solo graphs keep them
            auto* const tap = m.dfl_tap;
            m.dfl_tap = nullptr;
            // the weights are final once the graphs exist: seal them before the barrier, which absorbs the time it takes
            try
            {
                nd.capture_all(sc.limit(), mtp, plan.chunk, kstop.b, cp_step, kstop.solo ? kstop.kmax : 0);
                m.dfl_tap = tap;
                if (steering_boot.loaded)    // receipt: with shared scope every captured S2 MoE binding is steered
                {
                    int s2moe = 0, s2st = 0;
                    s2_steer_count(m, s2moe, s2st);
                    fprintf(stderr, "# rank %d: steering S2: %d of %d S2 MoE layer bindings carry the shared-scope finalize steering (scope %s)\n",
                            R, s2st, s2moe, steering::scope_name(steering_boot.scope));
                }
                wcheck::seal(R);
                int dev = 0;
                G6RT_CK(cudaGetDevice(&dev));
                wcheck::start_watcher(s2dir, R, dev);
                host_reclaim(R, "serving");     // #4: after the drafter, the S2 pool and the graphs; the barrier absorbs it
                g6s2::capture_barrier(bc, links);
            }
            catch (...) { links.shutdown(); throw; }
            g6s2::Service service(bc, links, std::move(h), plan.chunk);
            service.run();               // returns on DIR/quit
            // a persist=on ssd store that keeps sessions: an orderly quit (an idle engine) stores the runs the pool still
            // holds, the same calls on every rank, so the next boot finds the open sessions and their prefixes. Taking the
            // whole pool evicts every retained run through the stores; the write limit is for serving, not this once-only
            // drain. A SIGKILL or a crash skips it: the index then holds only what was stored while serving
            if (kv_whole)
            {
                try
                {
                    const long long t0 = g6s2::s2_wall_ms();
                    pool.write_limit(0);
                    kv_drain();   // reserve fails while store copies are queued (g6kvman.h): run them first
                    // sweep the pool: reserve at --cap rows (the pool's capacity) until the pool refuses, then halve down
                    // to one page (64 rows, always tried last, also when cap < 128). The halving is the fallback for when a
                    // cap-sized reserve cannot take everything (rows a live run still holds). Each reserve evicts what it needs through the
                    // stores. The calls depend only on the pool's state, which every rank shares, so every rank makes
                    // the same ones; no wall-clock stop (it could end ranks at different points). At most the pool's
                    // pages plus one refusal per size
                    const unsigned before = pool.kept_runs();
                    std::vector<g6s2::Run> held;
                    std::string why;
                    for (uint64_t rows = std::max<uint64_t>(64, (uint64_t)cap);; rows = std::max<uint64_t>(64, rows / 2))
                    {
                        for (;;)
                        {
                            kv_drain();
                            try { held.push_back(pool.reserve(rows, rows)); }
                            catch (const std::exception& e) { if (why.empty()) why = std::to_string(rows) + " rows: " + e.what(); break; }
                        }
                        if (rows == 64) break;
                    }
                    fprintf(stderr, "# rank %d: s2 stores: quit sweep took %zu chunks, %u retained runs before, first refusal: %s\n",
                            R, held.size(), before, why.c_str());
                    kv_drain();
                    for (const auto& r : held) pool.release(r);
                    kv_drain();
                    size_t n = 0;
                    uint64_t pages = 0;
                    uint64_t refused = 0;
                    for (const auto& t : pool.tiers()) { n += t.entries; pages += t.spilled_pages; refused += t.refused; }
                    fprintf(stderr, "# rank %d: s2 stores: quit stored the pool's runs, %zu entries kept for the next engine (%llu pages spilled in all, %llu refused, %u retained runs left in the pool, %lld ms)\n",
                            R, n, (unsigned long long)pages, (unsigned long long)refused, pool.kept_runs(), g6s2::s2_wall_ms() - t0);
                }
                catch (const std::exception& e) { fprintf(stderr, "# rank %d: s2 stores: drain at quit failed: %s\n", R, e.what()); }
            }
            links.shutdown();
            // GR8_QUIT_FAST=1: the stores hold what they keep (their index records are written as each copy commits), so
            // the ranks meet at the end-of-run barrier here and the process ends. Without it the scope ends first: every
            // CUDA graph is destroyed one by one (hundreds, each its own driver calls) before that barrier, and the
            // slowest rank holds the others there. Process exit frees the device memory, the graphs and the context
            if (knob::quit_fast())
            {
                fab.wait_proxy_idle();
                fab.barrier();
                fab.stop_proxy();
                fprintf(stderr, "# rank %d: quit fast: past the barrier, exiting without the teardown\n", R);
                fflush(stdout);
                fflush(stderr);
                _exit(0);
            }
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "g6m4: rank %d S2 epoch failed: %s\n", R, e.what());
            exit(3);
        }
    }
 #endif
    u64 step0 = 0, dstep = 0;
    MtpStats mtp_stats;
    int mtp_steps = 0;
    std::vector<double> t1;             // single-row step times
    PrefixReuse reuse;
    bool reuse_poison_on = false;
    if (reuse_prefix)
    {
        reuse.strict = knob::reuse_strict(); reuse.noend = knob::reuse_noend().value_or(false); reuse_poison_on = knob::reuse_poison();
        const bool align = knob::reuse_align().value_or(false);   // retain only whole cold prefill grid cells (prefix_reuse.h)
        if (align && !MP) die("GR8_REUSE_ALIGN=1 needs tensor prefill (--prefill N)");
        reuse.grid = align ? MP : 0;   // one sub-batch (the lane pairing does not change a row)
        if (R == 0 && reuse.grid) fprintf(stderr, "# reuse grid %d rows (--prefill %d, lanes %d)\n", reuse.grid, MP, two ? 2 : 1);
    }
    HDump hd = hdump_env();             // reuse diagnostics (reuse_hdump.cuh): default off
    {
        hd.zscratch = knob::reuse_zscratch();
        if (hd.zscratch && !reuse_prefix) die("GR8_REUSE_ZSCRATCH=1 needs --reuse-prefix");
    }
    if (R == 0 && (!hd.dir.empty() || hd.zscratch))
        fprintf(stderr, "# reuse diag: hdump %s seqs %zu, zscratch %d\n", hd.dir.empty() ? "off" : hd.dir.c_str(), hd.seqs.size(), (int)hd.zscratch);
    if (!s2mode) host_reclaim(R, "ready");     // offline runs; a served rank trimmed before its capture barrier
    boot("ready");
    for (size_t q = 0; !s2mode && q < plans.size(); ++q)   // offline: each --prompt once
    {
#if G6_MODEL_FLASH
        if (spec_reset_per_request) spec = spec_initial;
#endif
        Plan sp = plans[q];
        // Never skip into a teacher-forced scoring tail (--pf-rows). Without tensor prefill, the decode prompt steps
        // start directly at skip.
        const int old_end = reuse.written_end, old_dirty = reuse.dirty_end;   // dirty: every row written since the last reset
        const int skip = reuse_prefix ? reuse.begin(sp.prompt, MP ? sp.first : (int)sp.prompt.size() - 1, mtp > 0,
                                                    steering::cache_namespace(sp.steering)) : 0;
        if (skip > sp.first) sp.first = skip;
        const Plan& p = sp;
        P.steering_scale = P2.steering_scale = p.steering_scale;
        P.steering_slot = P2.steering_slot = p.steering_slot;
        if (pf_qp)                      // OWN=4 iff prompt + gen fits the QO table (GR8_PF_QP_CTX), else OWN=2 for this request
        {
            const int ctx = (int)p.prompt.size() + p.n_gen;
            const bool on = pf_qp_select(P, two ? &P2 : nullptr, ctx, (int)p.prompt.size());
            fprintf(stderr, "# rank %d: seq %zu prompt %zu ctx %d (prompt + gen) -> %s (QP_MIN %d, QP_CTX %d)\n", R, q, p.prompt.size(), ctx,
                    on ? "OWN=4 pair-owner" : "OWN=2", pf_qp_min(), P.qp_ctx);
        }
        const int PS = prompt_steps(p), NS = total_steps(p);
        if (!skip)
        {
            reset(m, st);
            if (mtp) reset(D.d, st);
        }
        else if (reuse_poison_on && old_dirty > skip)   // gate arm: every stale row since the last reset := NaN (2b G2)
        {
            reuse_poison(m, skip, old_dirty, st);
            if (mtp) reuse_poison(D.d, skip, old_dirty, st);
            if (R == 0) fprintf(stderr, "# reuse seq %zu poisoned rows %d..%d\n", q, skip, old_dirty - 1);
        }
        if (skip && hd.zscratch)            // reuse diag arm: a hit starts from cleared per-seq attention scratch
        {
            zscratch(m, st);
            if (mtp) zscratch(D.d, st);
            if (R == 0) fprintf(stderr, "# reuse seq %zu zscratch layers %zu + %zu\n", q, m.layers.size(), mtp ? D.d.layers.size() : (size_t)0);
        }
        if (dfl && DF.pf_feat)          // prefill taps keep rows at these positions. Each rank fills its owner rows, the rest stay zero
        {
            DF.pf_hi = p.first; DF.pf_lo = std::max(0, p.first - dfl::WIN);
            if (DF.pf_hi > DF.pf_lo) G6RT_CK(cudaMemsetAsync(DF.pf_feat, 0, (size_t)(DF.pf_hi - DF.pf_lo) * dfl::NT * HID * 2, st));
        }
        G6RT_CK(cudaStreamSynchronize(st));
        if (q == 0) boot("first reset");
        printf("seq %zu prompt %zu tokens\n", q, p.prompt.size());
        if (reuse_prefix && R == 0)         // every miss with its reason (first / strict / no-lcp / short), counted by the gate
            fprintf(stderr, "# reuse seq %zu %s retained %d lcp %d history %zu prompt %zu written_end %d dirty_end %d\n", q, reuse.why, skip,
                    reuse.lcp, reuse.history.size(), p.prompt.size(), old_end, old_dirty);
        if (reuse_prefix && R == 0 && reuse.grid)   // the skip before the grid rule (TTFT cost of the exact rule)
            fprintf(stderr, "# reuse seq %zu grid %d raw %d retained %d\n", q, reuse.grid, reuse.raw, skip);
        if (skip) printf("reuse rows 0..%d\n", skip - 1);  // real hit evidence (every rank), NOT a freshly computed hash
        for (int a = skip; a < p.first; a += MP)
        {
#if !G6_MODEL_FLASH
            if (two && p.first - a > MP)         // two sub-batches as two lanes (prefill.cuh pf_step2)
            {
                const int MA = MP, MB = p.first - a - MP < MP ? p.first - a - MP : MP;
                std::vector<int> pos(MA + MB);
                for (int r = 0; r < MA + MB; ++r) pos[r] = a + r;
                pf_inputs(P, MA, p.prompt.data() + a, pos.data());
                pf_inputs(P2, MB, p.prompt.data() + a + MA, pos.data() + MA);
                pf_enqueue(fabp, P, ls0, NL, MA);
                pf_enqueue(fabp2, P2, ls0b, NL, MB);
                const auto t0 = std::chrono::steady_clock::now();
                cp_pf_in();
                if (pf_streams) pf_step2s(m, P, P2, MA, MB, a + MA, a + MA + MB, ls0, ls0b, st, lanes2);
                else pf_step2(m, P, P2, MA, MB, a + MA, a + MA + MB, ls0, ls0b, st);
                cp_pf_out();
                G6RT_CK(cudaStreamSynchronize(st));
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ls0 += NL; ls0b += NL;
                if (mtp)
                {
                    pf_draft_inputs(DQ, MA, p.prompt.data() + a + 1, pos.data());
                    pf_draft(D, DQ, P.x, MA, a + MA, st);
                    G6RT_CK(cudaStreamSynchronize(st));
                    pf_draft_inputs(DQ, MB, p.prompt.data() + a + MA + 1, pos.data() + MA);
                    pf_draft(D, DQ, P2.x, MB, a + MA + MB, st);
                    G6RT_CK(cudaStreamSynchronize(st));
                    if (const unsigned ae = attn_errors(D.d)) { fprintf(stderr, "g6m4: rank %d MTP layer attention error word %08x, seq %zu prefill rows %d..\n", R, ae, q, a); return 3; }
                }
                if (*fp.error || *fp2.error || cp_pf_err()) { fprintf(stderr, "g6m4: rank %d prefill fabric error word %08x / %08x (CP staging / qp wire %08x), seq %zu rows %d..\n", R, *fp.error, *fp2.error, cp_pf_err(), q, a); return 3; }
                if (const unsigned ae = attn_errors(m)) { fprintf(stderr, "g6m4: rank %d attention error word %08x, seq %zu prefill rows %d..\n", R, ae, q, a); return 3; }
                printf("prefill rows %d..%d resid %016llx\n", a, a + MA - 1, (unsigned long long)pf_hash(P, MA));
                printf("prefill rows %d..%d resid %016llx\n", a + MA, a + MA + MB - 1, (unsigned long long)pf_hash(P2, MB));
                fprintf(stderr, "# seq %zu prefill %d + %d rows (two lanes): %.3f ms (%.0f tok/s)\n", q, MA, MB, ms, (MA + MB) * 1000.0 / ms);
                if (reuse_prefix) reuse.prefill_rows(a, MA + MB, mtp > 0);
                if (q > 0 || a > 0) { pf_ms += ms; pf_rows += MA + MB; }
                a += MP;
                continue;
            }
#endif
            const int M = p.first - a < MP ? p.first - a : MP;
            std::vector<int> pos(M);
            for (int r = 0; r < M; ++r) pos[r] = a + r;
#if G6_MODEL_FLASH
            {
                pf_inputs_flash(FP, M, p.prompt.data() + a, pos.data());
                pf_enqueue_flash(fabp, m, FP, ls0, NL, M);
                const int MB = two && p.first - a > MP ? std::min(MP, p.first - a - MP) : 0;
                if (MB)
                {
                    std::vector<int> posB(MB);
                    for (int r = 0; r < MB; ++r) posB[r] = a + M + r;
                    pf_inputs_flash(FP2, MB, p.prompt.data() + a + M, posB.data());
                    pf_enqueue_flash(fabp2, m, FP2, ls0b, NL, MB);
                }
                // GR8_RESID_DUMP: one record per sub-batch (step field 0), every slot copied out as it is reached
                std::function<void(int, int)> pf_copy;
                std::vector<float> pf_h;
                if (rdf)
                {
                    const int32_t hdr[] = {0, M, rd_slots, rd_hid};
                    dump_write(rdf, hdr, sizeof hdr);
                    pf_h.resize((size_t)M * rd_hid);
                    pf_copy = [&](int, int Mr) {
                        G6RT_CK(cudaStreamSynchronize(st));
                        G6RT_CK(cudaMemcpy(pf_h.data(), FP.resid4, (size_t)Mr * rd_hid * sizeof(float), cudaMemcpyDeviceToHost));
                        dump_write(rdf, pf_h.data(), pf_h.size() * sizeof(float));
                    };
                }
                const auto t0 = std::chrono::steady_clock::now();
                // GR8_CP: a sub-batch past R runs on the staged view (prefill.cuh cp_pf_*); below R on the replicated prefix
#if defined(G6_CP) && G6_CP
                const bool staged = cp_pf && a + M + MB > cp_R;
#else
                const bool staged = false;
#endif
                if (staged) cp_pf_in();
                if (MB) pf_step2_flash(m, FP, FP2, M, MB, a + M, a + M + MB, ls0, ls0b, st);
                else pf_step_flash(m, FP, M, a + M, ls0, st, pf_copy);
                if (staged) cp_pf_out();
                G6RT_CK(cudaStreamSynchronize(st));
                if (staged && cp_pf_err()) { fprintf(stderr, "g6m4: rank %d CP staging fabric error word %08x, seq %zu rows %d..\n", R, cp_pf_err(), q, a); return 3; }
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ls0 += NL;
                if (MB) ls0b += NL;
                if (rdf && fflush(rdf)) die("GR8_RESID_DUMP: flush failed");
                if (pool_owner && (*fabix.dev().error || (MB && *fabix2.dev().error))) {
                    fprintf(stderr, "g6m4: rank %d pool-ID fabric error %08x / %08x\n", R,
                            *fabix.dev().error, MB ? *fabix2.dev().error : 0u); return 3;
                }
                if (route_split && (*fabrs.dev().error || (MB && *fabrs2.dev().error))) {
                    fprintf(stderr, "g6m4: rank %d route-table fabric error %08x / %08x\n", R,
                            *fabrs.dev().error, MB ? *fabrs2.dev().error : 0u); return 3;
                }
                if (MB && *fp2.error) { fprintf(stderr, "g6m4: rank %d second prefill lane fabric error %08x\n", R, *fp2.error); return 3; }
                if (*fp.error) { fprintf(stderr, "g6m4: rank %d prefill fabric error word %08x, seq %zu rows %d..\n", R, *fp.error, q, a); return 3; }
                if (const unsigned ae = attn_errors(m)) { fprintf(stderr, "g6m4: rank %d attention error word %08x, seq %zu prefill rows %d..\n", R, ae, q, a); return 3; }
                auto finish_lane = [&](FlashPrefill& FP, int M, int a) {
                if (ldump && knob::pf_head())   // logit dump rows of the prefill (MX-row records, step 0: merge_logits.py needs >= 0)
                    pf_head_flash(m, FP, M, st, [&](int r) {
                        const int32_t hdr[] = {0, r};
                        dump_write(ldump, hdr, sizeof hdr);
                        std::vector<float> h((size_t)r * m.vn);
                        G6RT_CK(cudaMemcpy(h.data(), m.hlog, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
                        dump_write(ldump, h.data(), h.size() * sizeof(float));
                        if (fflush(ldump)) die("GR8_LOGIT_DUMP: flush failed");
                    });
                // GR8_PF_HASH=1 (gate jobs that compare hashes): the resid hash (stream sync + D2H + CPU FNV of M x HID words)
                static const bool pf_hash = knob::pf_hash();
                if (pf_hash) printf("prefill rows %d..%d resid %016llx\n", a, a + M - 1, pf_hash_flash(FP, M, st));
                else printf("prefill rows %d..%d\n", a, a + M - 1);
                if (mtp)
                {
                    const auto td = std::chrono::steady_clock::now();
                    pf_draft_flash(D, FP, M, a, p.prompt.data() + a + 1, dstep, [&](const StepIn& di) {
                        *in_d = di;
                        // A draft-logit dump explicitly requests the otherwise unused prompt outputs.
                        const bool cache_only = !dldump && knob::glm53f_draft_kv();
                        if (cache_only)
                        {
                            pf_draft_cache_flash(D, di.M, di.context, st);
                            G6RT_CK(cudaStreamSynchronize(st));
                            if (const unsigned ae = attn_errors(D.d))
                            {
                                fprintf(stderr, "g6m4: rank %d draft cache error word %08x, seq %zu prefill rows %d..\n", R, ae, q, a);
                                exit(3);
                            }
                            draft_dump(di);
                            return;
                        }
                        std::vector<uint32_t> sizes(3, (uint32_t)(di.M * HID * 2));
                        sizes[2] = (uint32_t)glue::head_bytes(di.M, nc);
                        fabd.enqueue_step(di.step, sizes);
                        const int cur = D.d.cur;            // --graph: the attention replay of the context's bucket (not a captured graph:
                        D.d.cur = bucket_of(D.d, di.context);   // those read the main model's own rows)
                        launch_draft_step(D, di.M, di.context, false, st);
                        D.d.cur = cur;
                        G6RT_CK(cudaStreamSynchronize(st));
                        if (*fd.error) { fprintf(stderr, "g6m4: rank %d draft fabric error word %08x, seq %zu prefill rows %d..\n", R, *fd.error, q, a); exit(3); }
                        if (out_d->attn_err) { fprintf(stderr, "g6m4: rank %d draft attention error word %08x, seq %zu prefill rows %d..\n", R, out_d->attn_err, q, a); exit(3); }
                        draft_dump(di);
                    });
                    fprintf(stderr, "# seq %zu prefill draft %d rows: %.3f ms\n", q, M,
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count());
                }
                };
                finish_lane(FP, M, a);
                if (MB) finish_lane(FP2, MB, a + M);
                fprintf(stderr, "# seq %zu prefill %d rows%s: %.3f ms (%.0f tok/s)\n", q, M + MB,
                        MB ? " (two lanes)" : "", ms, (M + MB) * 1000.0 / ms);
                if (q > 0 || a > 0) { pf_ms += ms; pf_rows += M + MB; }
                if (MB) a += MP;
                continue;
            }
#endif
            pf_inputs(P, M, p.prompt.data() + a, pos.data());
            pf_enqueue(fabp, P, ls0, NL, M);
            const auto t0 = std::chrono::steady_clock::now();
            cp_pf_in();
            pf_step(m, P, M, a + M, ls0, st);
            cp_pf_out();
            G6RT_CK(cudaStreamSynchronize(st));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ls0 += NL;
            if (mtp)
            {
                pf_draft_inputs(DQ, M, p.prompt.data() + a + 1, pos.data());
                pf_draft(D, DQ, P.x, M, a + M, st);
                G6RT_CK(cudaStreamSynchronize(st));
                if (const unsigned ae = attn_errors(D.d)) { fprintf(stderr, "g6m4: rank %d MTP layer attention error word %08x, seq %zu prefill rows %d..\n", R, ae, q, a); return 3; }
            }
            if (*fp.error || cp_pf_err()) { fprintf(stderr, "g6m4: rank %d prefill fabric error word %08x (CP staging / qp wire %08x), seq %zu rows %d..\n", R, *fp.error, cp_pf_err(), q, a); return 3; }
            if (const unsigned ae = attn_errors(m)) { fprintf(stderr, "g6m4: rank %d attention error word %08x, seq %zu prefill rows %d..\n", R, ae, q, a); return 3; }
            printf("prefill rows %d..%d resid %016llx\n", a, a + M - 1, (unsigned long long)pf_hash(P, M));
            fprintf(stderr, "# seq %zu prefill %d rows: %.3f ms (%.0f tok/s)\n", q, M, ms, M * 1000.0 / ms);
            if (reuse_prefix) reuse.prefill_rows(a, M, mtp > 0);
            if (q > 0 || a > 0) { pf_ms += ms; pf_rows += M; }      // the very first sub-batch carries first-launch costs
        }
#if G6_MODEL_FLASH
        // Test-only: draft cache pages before the first live draft. Includes latent, raw key/gate and pooled keys.
        if (const auto& path = knob::glm53f_draft_cache_dump(); path && mtp)
        {
            G6RT_CK(cudaStreamSynchronize(st));
            const std::string name = *path + ".s" + std::to_string(q) + ".r" + std::to_string(R) + ".bin";
            FILE* f = fopen(name.c_str(), "wb");
            if (!f) die("GR8_GLM53F_DRAFT_CACHE_DUMP: cannot open " + name);
            const size_t bytes = hdump_model(f, D.d, 'd', p.first);
            if (fclose(f)) die("GR8_GLM53F_DRAFT_CACHE_DUMP: close failed " + name);
            fprintf(stderr, "# draft cache dump seq %zu rank %d rows %d bytes %zu file %s\n", q, R, p.first, bytes, name.c_str());
        }
#endif
        if (hd.listed(q)) hdump_seq(hd, m, mtp ? &D.d : nullptr, q, R, p.first, st);   // reuse diag: rows [0, p.first)
        int dfl_lo = p.first;           // --dflash: the lowest position the drafter holds context for
        if (dfl && DF.pf_feat && DF.pf_hi > DF.pf_lo)
        {
            const auto t0 = std::chrono::steady_clock::now();
            for (int c0 = DF.pf_lo; c0 < DF.pf_hi; c0 += dfl::B)
            {
                DflIn di;
                memset(&di, 0, sizeof di);
                di.step = ++dstep;
                di.ctx_rows = std::min(dfl::B, DF.pf_hi - c0); di.ctx_pos0 = c0; di.q0 = c0 + di.ctx_rows; di.lo = DF.pf_lo;
                *in_f = di;
                dfl_enqueue(fabf, di.step, dfl_fill_sizes(), true);
                if (graph) G6RT_CK(cudaGraphLaunch(exec_f[2], st));
                else dfl_fill_step(DF, st);
                G6RT_CK(cudaStreamSynchronize(st));
                if (*fabf.dev().error) { fprintf(stderr, "g6m4: rank %d dflash fabric error word %08x at seq %zu prompt context %d\n", R, *fabf.dev().error, q, c0); exit(3); }
                dfl_dump_step(dfl_dump, DF, di, nullptr);
            }
            fprintf(stderr, "# seq %zu dflash prompt context %d..%d: %.1f ms\n", q, DF.pf_lo, DF.pf_hi - 1,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            dfl_lo = DF.pf_lo;
        }
        // MTP / DFlash after the prefill loop (as in m4sim): prompt rows past p.first go through their sequence's prompt steps
        auto rm = [&](const StepIn& si, int s, double& ms) {
            const int M = si.M;
            *in = si;
            std::vector<uint32_t> sizes(NC, (uint32_t)(M * HID * 2));
            sizes[2 * NL] = head_size(si, nc);
            extra_coll_sizes(sizes, NL, si.M);
            fab_enqueue(si.step, sizes);
            cp_enqueue(si.step, M, si.context);
            qs_enqueue(si.step, M);
            const auto t0 = std::chrono::steady_clock::now();
            if (graph) launch_main(bucket_of(m, si.context), M);
            else launch_step(m, M, si.context, st, -1, 0, residual_copy);
            G6RT_CK(cudaStreamSynchronize(st));
            const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            evprof_read(M, st, R);
            ms += dt;
            if (*f.error || cp_error()) { fprintf(stderr, "g6m4: rank %d fabric error word %08x (CP %08x) at seq %zu step %d\n", R, *f.error, cp_error(), q, s); exit(3); }
            const StepOut so = *out;
            if (so.attn_err) { fprintf(stderr, "g6m4: rank %d attention error word %08x at seq %zu step %d\n", R, so.attn_err, q, s); exit(3); }
#if G6_MODEL_FLASH
            main_dump(s, mtp_rows(si, so, (int)p.prompt.size(), mtp > 0 && !p.rs));
#else
            main_dump(s, M);
#endif
            if (reuse_prefix) reuse.main_rows(si.pos[0], M);
            if (rlf) route_log_line(rlf, m, layers, q, s, M, si.pos[0], p.prompt.size());
            std::string line;
            step_line(m, so, s, M, line);
            printf("%s\n", line.c_str());
            if (top > 0) { std::string tl; top_lines(m, s, M, top, tl); fputs(tl.c_str(), stdout); }
            fprintf(stderr, "# seq %zu step %d M %d: %.3f ms\n", q, s, M, dt);
            if (q == 0 && s == 0) boot("first token ready");
            return so;
        };
        if (dfl)
        {
            auto rf = [&](const DflIn& di, int s, double& ms) {
                *in_f = di;
                dfl_enqueue(fabf, di.step, dfl_sizes(DF, di.query), false);
                const auto t0 = std::chrono::steady_clock::now();
                if (graph) G6RT_CK(cudaGraphLaunch(exec_f[di.query ? 1 : 0], st));
                else dfl_step(DF, st, di.query);
                G6RT_CK(cudaStreamSynchronize(st));
                const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ms += dt;
                if (*fabf.dev().error) { fprintf(stderr, "g6m4: rank %d dflash fabric error word %08x at seq %zu step %d\n", R, *fabf.dev().error, q, s); exit(3); }
                const DflOut so = *out_f;
                dfl_dump_step(dfl_dump, DF, di, di.query ? &so : nullptr);
                if (di.query)
                {
                    printf("dflash %d ctx %d @%d q0 %d drafts", s, di.ctx_rows, di.ctx_pos0, di.q0);
                    for (int r = 1; r < dfl::B; ++r) printf(" %d", so.tok[r]);
                    printf("\n");
                }
                fprintf(stderr, "# seq %zu dflash %s %d: %.3f ms\n", q, di.query ? "draft" : "context", s, dt);
                return so;
            };
            const std::vector<int> gen = dflash_sequence(p, step0, dstep, rm, rf, mtp_stats, mtp_steps, dfl_lo, {}, dfl_k, &spec);
            if (dfl_dump) { std::vector<int> all = p.prompt; all.insert(all.end(), gen.begin(), gen.end()); dfl_dump_tokens(dfl_dump, all); }
            printf("generated");
            for (int t : gen) printf(" %d", t);
            printf("\n");
            continue;
        }
        if (mtp)
        {
            auto rd = [&](const StepIn& di, bool chain, int s, double& ms) {
                const int M = di.M;
                *in_d = di;
                std::vector<uint32_t> sizes(3, (uint32_t)(M * HID * 2));
                sizes[2] = (uint32_t)glue::head_bytes(M, nc);
                fabd.enqueue_step(di.step, sizes);
                qsd_enqueue(di.step, M);
                const auto t0 = std::chrono::steady_clock::now();
                if (graph) G6RT_CK(cudaGraphLaunch(exec_d[bucket_of(D.d, di.context)][chain][M], st));
                else launch_draft_step(D, M, di.context, chain, st);
                G6RT_CK(cudaStreamSynchronize(st));
                const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ms += dt;
                if (*fd.error) { fprintf(stderr, "g6m4: rank %d draft fabric error word %08x at seq %zu step %d\n", R, *fd.error, q, s); exit(3); }
                const StepOut so = *out_d;
                if (so.attn_err) { fprintf(stderr, "g6m4: rank %d draft attention error word %08x at seq %zu step %d\n", R, so.attn_err, q, s); exit(3); }
#if G6_MODEL_FLASH
                if (!chain) draft_dump(di);
#endif
                std::string line;
                if (reuse_prefix) reuse.draft_rows(di.pos[0], M, chain);
                const int t = step_line(D.d, so, s, M, line);
                printf("%s%s\n", chain ? "chain" : "draft", line.c_str() + 4);
#if G6_MODEL_FLASH
                if (top > 0)                // --top K: the draft head's top-K per row, as "dtop" (first form) / "ctop" (chain) lines
                {
                    std::string tl;
                    top_lines(D.d, s, M, top, tl);
                    for (size_t i = 0; (i = tl.find("top ", i)) != std::string::npos; i += 5) tl.replace(i, 4, chain ? "ctop " : "dtop ");
                    fputs(tl.c_str(), stdout);
                }
#endif
                fprintf(stderr, "# seq %zu %s %d M %d: %.3f ms\n", q, chain ? "chain" : "draft", s, M, dt);
                return DraftOut{t, so.lp[M - 1]};
            };
            const std::vector<int> gen = mtp_sequence(p, mtp, step0, dstep, rm, rd, mtp_stats, mtp_steps, {}, &spec);
            printf("generated");
            for (int t : gen) printf(" %d", t);
            printf("\n");
            if (reuse_prefix) reuse.finish(p.prompt, gen, (int)gen.size(), true);
            continue;
        }
        int last_tok = 0;
        std::vector<int> gen;
        for (int s = 0; s < NS; ++s)
        {
            const int M = step_rows(p, s);
            *in = step_inputs(p, s, last_tok, step0);
            std::vector<uint32_t> sizes(NC, (uint32_t)(M * HID * 2));
            sizes[2 * NL] = head_size(*in, nc);
            extra_coll_sizes(sizes, NL, M);
            fab_enqueue(in->step, sizes);
            cp_enqueue(in->step, M, in->context);
            qs_enqueue(in->step, M);
            dec_prof().on = knob::dec_prof() && s > 0;      // the first step of a sequence carries first-launch costs
            dec_prof().M = M; dec_prof().last = 0;
            const auto t0 = std::chrono::steady_clock::now();
            if (graph) launch_main(bucket_of(m, in->context), M);
            else launch_step(m, M, in->context, st, -1, 0, residual_copy);
            G6RT_CK(cudaStreamSynchronize(st));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            evprof_read(M, st, R);
            if (dec_prof().on) ++dec_prof().steps[M];
            dec_prof().on = false;
            if (*f.error || cp_error()) { fprintf(stderr, "g6m4: rank %d fabric error word %08x (CP %08x) at seq %zu step %d\n", R, *f.error, cp_error(), q, s); return 3; }
            const StepOut so = *out;
            if (so.attn_err) { fprintf(stderr, "g6m4: rank %d attention error word %08x at seq %zu step %d\n", R, so.attn_err, q, s); return 3; }
            std::string line;
            if (reuse_prefix) reuse.main_rows(in->pos[0], M);
            main_dump(s, M);
            if (rlf) route_log_line(rlf, m, layers, q, s, M, in->pos[0], p.prompt.size());
            last_tok = step_line(m, so, s, M, line);
            printf("%s\n", line.c_str());
            if (top > 0) { std::string tl; top_lines(m, s, M, top, tl); fputs(tl.c_str(), stdout); }
            if (s >= PS - 1) gen.push_back(last_tok);
            fprintf(stderr, "# seq %zu step %d M %d: %.3f ms\n", q, s, M, ms);
            if (q == 0 && s == 0) boot("first token ready");
            if (s >= PS + 2) t1.push_back(ms);
        }
        step0 += NS;
#if G6_MODEL_FLASH
        if (const auto& kd = knob::glm53f_kda_state_dump(); kd && m.kda_st.S)   // gates: KDA S + conv of every layer after the sequence
        {
            const std::string name = *kd + ".rank" + std::to_string(R);
            FILE* kf = fopen(name.c_str(), "wb");
            if (!kf) die("GR8_GLM53F_KDA_STATE_DUMP: cannot open " + name);
            std::vector<float> h(kda::N_KDA * m.kda_st.s_layer());
            G6RT_CK(cudaMemcpy(h.data(), m.kda_st.S, h.size() * 4, cudaMemcpyDeviceToHost));
            dump_write(kf, h.data(), h.size() * 4);
            h.resize(kda::N_KDA * m.kda_st.c_layer());
            G6RT_CK(cudaMemcpy(h.data(), m.kda_st.conv, h.size() * 4, cudaMemcpyDeviceToHost));
            dump_write(kf, h.data(), h.size() * 4);
            fclose(kf);
        }
#endif
        if (reuse_prefix) reuse.finish(p.prompt, gen, (int)gen.size(), false);
        printf("generated");
        for (int t : gen) printf(" %d", t);
        printf("\n");
    }
    if (!t1.empty())
    {
        std::sort(t1.begin(), t1.end());
        fprintf(stderr, "# rank %d: single-row step p50 %.3f ms min %.3f ms p90 %.3f ms over %zu steps, %d layers, %s\n", R,
                t1[t1.size() / 2], t1[0], t1[t1.size() * 9 / 10], t1.size(), NL, graph ? "one graph" : "plain launches");
    }
    if (pf_rows) fprintf(stderr, "# rank %d: prefill %ld rows in %.1f ms = %.0f tok/s (sub-batches of %d, first one excluded), %d layers\n", R,
                         pf_rows, pf_ms, pf_rows * 1000.0 / pf_ms, MP, NL);
#if G6_MODEL_DS41
    if (dsk.on()) { fabds.wait_proxy_idle(); fabds.stop_proxy(); }
#endif
    if (mtp || dfl)
    {
        char tag[32];
        snprintf(tag, sizeof tag, "rank %d", R);
        mtp_stats.print(dfl ? dfl_k : K, tag);
        if (spec.on) fprintf(stderr, "%s\n", spec.summary().c_str());
        if (nc && mtp) fprintf(stderr, "# rank %d: top_p misses %u (target) %u (draft)\n", R, topp_misses(m), topp_misses(D.d));
        if (mtp) { fabd.wait_proxy_idle(); fabd.stop_proxy(); }
        if (dfl) { fabf.wait_proxy_idle(); fabf.stop_proxy(); }
    }
    if (nc && !mtp) fprintf(stderr, "# rank %d: top_p misses %u\n", R, topp_misses(m));
#if G6_MODEL_FLASH
    if (pool_owner) {
        bool ok = fabix.wait_proxy_idle();
        if (two) ok = fabix2.wait_proxy_idle() && ok;
        fabix.stop_proxy();
        if (two) fabix2.stop_proxy();
        if (!ok || *fabix.dev().error || (two && *fabix2.dev().error)) {
            fprintf(stderr, "g6m4: rank %d pool-ID fabric drain failed\n", R); return 3;
        }
        release_flash_pool_transport(FP.pool_transport);
        if (two) release_flash_pool_transport(FP2.pool_transport);
    }
#endif
#if G6_MODEL_FLASH
    if (route_split) {
        bool ok = fabrs.wait_proxy_idle();
        if (two) ok = fabrs2.wait_proxy_idle() && ok;
        fabrs.stop_proxy();
        if (two) fabrs2.stop_proxy();
        if (!ok || *fabrs.dev().error || (two && *fabrs2.dev().error)) {
            fprintf(stderr, "g6m4: rank %d route-table fabric drain failed\n", R); return 3;
        }
        release_flash_route_split(FP.route_split);
        if (two) release_flash_route_split(FP2.route_split);
    }
#endif
    if (two) { fabp2.wait_proxy_idle(); fabp2.stop_proxy(); }
    if (MP) { fabp.wait_proxy_idle(); fabp.stop_proxy(); }
#if G6_NR == 4
    if (qp_roll)
    {
        qp_roll_print(*qp_roll, R);
        if (!fabq.wait_proxy_idle()) { fprintf(stderr, "g6m4: rank %d qp wire proxy drain failed (error word %08x)\n", R, qp_err()); fabq.stop_proxy(); return 3; }
        fabq.stop_proxy();
        if (qp_err()) { fprintf(stderr, "g6m4: rank %d qp wire error word %08x\n", R, qp_err()); return 3; }
        fprintf(stderr, "# rank %d: qp wire clean, %ld builds over the wire\n", R, qp_wire.n);
    }
#endif
#if defined(G6_CP) && G6_CP
    if (cp_pf)
    {
        if (!fabs.wait_proxy_idle()) { fprintf(stderr, "g6m4: rank %d CP staging proxy drain failed (error word %08x)\n", R, *fabs.bulk_dev().error); fabs.stop_proxy(); return 3; }
        fabs.stop_proxy();
        if (cp_pf_net(m).verify) printf("# rank %d: GR8_CP_PF_VERIFY staged-page hash mismatches %llu\n", R, cp_pf_mismatches(m));
    }
#endif
#if G6_MODEL_FLASH
    if (pf_prof().on) flash_pf_prof().print("this rank,");
    pf_xt_dump(FP, R);
    if (two) { fprintf(stderr, "# PXT second lane\n"); pf_xt_dump(FP2, R); }
#else
    if (pf_prof().on) pf_prof().print("this rank,");
#endif
    if (rdf && fclose(rdf)) die("GR8_RESID_DUMP: close failed");
    if (ldump && fclose(ldump)) die("GR8_LOGIT_DUMP: close failed");
#if G6_MODEL_FLASH
    if (didump && fclose(didump)) die("GR8_DRAFT_IN_DUMP: close failed");
    if (dldump && fclose(dldump)) die("GR8_DRAFT_LOGIT_DUMP: close failed");
#endif
    if (rdump) G6RT_CK(cudaFree(rdump));
    if (rlf) fclose(rlf);
    if (knob::dec_prof()) { char tag[32]; snprintf(tag, sizeof tag, "rank %d", R); dp_print(tag, "synchronized", dec_prof().t, dec_prof().steps); }
    if (fabdg_on) { fabdg.wait_proxy_idle(); fabdg.stop_proxy(); }
    if (fabg_on) { fabg.wait_proxy_idle(); fabg.stop_proxy(); }
    fab.wait_proxy_idle();
    fab.barrier();
    fab.stop_proxy();
#if G6_MODEL_DS41
    // C2 dispatch trace: one stop-time line per rank. The counter is host-side in
    // attn/ds41_attn.cuh and stays 0 unless GR8_DS_C2_TRACE=1 is in the env.
    fprintf(stderr, "# rank %d: ds41 C2 trace count=%llu trace=%d\n", R,
            ds41_c2_dispatch_count(), ds41_c2_trace_on() ? 1 : 0);
    // Prefix cache counters: the cache object lives in serve_ds41's scope; it self-registers (Ds41PrefixCache::live) so
    // this stop-time line can report the counters directly instead of deriving them from the HTTP usage field. Absent
    // without --reuse-prefix (no cache was built).
    if (auto* pc = g6s2::Ds41PrefixCache::live())
        fprintf(stderr, "# rank %d: ds41 prefix cache %s\n", R, pc->stats_line().c_str());
    // Count kernel enqueue statements actually reached on the two-stage mHC path, not the option value.
    // The W-row prefill is eager; CUDA graph replays, if any, are not counted here.
    fprintf(stderr, "# rank %d: B12_MHC_KERNEL_LAUNCHES=%llu\n", R, m.ds41_b12_kernel_launches);
#endif
    return 0;
}
