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
//                     prints which form each matrix took. G6_PF_OWN=3 takes its full-head tensors from the same image
//                     (no --qo-manifest / --qo-pack)
//      [--v2-dense-form trellis|rounded]  trellis (default): q_b, o_proj and the dense MLP stored as trellis stay the
//                     release's entries for the dense trellis kernels (kernels/dtr.cuh). rounded: the fallback, which
//                     rounds them to the MXFP8 and int4 forms of a converted pack (not the release's weights)
//      [--layers 0,3,40,77] [--prompt 154822,154824,154826,25062] [--gen 16] [--temp 0] [--seed 1234] [--cap 64] [--chunk 4]
//      [--graph]      one captured graph per row count instead of plain launches (same lines)
//      [--prefill MP] every prompt token but the last goes through the prefill path in sub-batches of up to MP rows,
//                     on a second fabric region (two collectives of MP rows, the layer as its step; prefill.cuh)
//      --prompt may be repeated: each is its own sequence (attention caches reset in between); "@file" reads ids
//      [--reuse-prefix] keep the caches of the longest common prefix of the previous prompt + emitted ids (prefix_reuse.h;
//                     G6_REUSE_STRICT=1: strict append only, G6_REUSE_NOEND=1: no prompt >= written_end rule; default off;
//                     gate arm G6_REUSE_POISON=1: a hit first sets the old sequence's stale rows to NaN, reuse_poison.cuh;
//                     G6_REUSE_ALIGN=1: a hit retains only whole cold prefill grid cells, prefix_reuse.h.
//                     Under S2: every retired session's caches stay in the pool (s2/pool.h), ALIGN and NOEND default 1.
//                     No --dflash-manifest (the drafter's prompt-context taps are written by the prefill a hit skips)
//      G6_RT_STORE=mem:|remote:HOST:PORT (environment; --serve with G6_CP): remote attention rows (rt_rows.cuh). A
//                     session's whole pages past G6_RT_L (default G6_CP_R) go to a lease named after the boot and the
//                     layout, and CP decode reads them through G6_RT_POOL (default 4096) resident slots per indexer group.
//                     A prefill of the session (a later turn) and its retirement drop them back to the local copies.
//                     G6_RT_FMT=int4: the lease holds the rows as INT4 (416 B for 656), turned back into FP8 after each fetch.
//                     G6_RT_LOCAL=N: release: a split run holds its prefix and N window pages per rank, its tail pages
//                     go back to the window once in the store, and the pool grants the drafter's whole tail apart from
//                     them; retained, it keeps its records for its next turn. G6_RT_PLACE=P: the main model's physical pages per rank, so
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
//      [--kv-write-limit F]  the stores write at most F page blocks per page of KV the engine computes (off by default)
//      [--kv-tier STORE,size=N[,keep=evicted|shared|all]] [--kv-shared-after N]  deprecated: --kv-store, with --kv-keep
//                     all when any keep is evicted or all, else system; --kv-system-after N
//      [--top-p F] [--top-p-cands N]  nucleus sampling at T (per request with --serve); N candidates per rank per row
//                     (0 = top_p off; 256 is the served setting; misses printed at the end)
//      [--serve DIR --s2-port N [--s2-sessions S] [--s2-epoch E]]  serve requests through S2 (a G6_CP build); --prompt is
//                     ignored. Up to S sessions share decode steps (runs of at most G6_S2_BELOW rows, default G6_CP_R), a
//                     longer request runs alone at base 0 of the pool (s2/pool.h); requests over the control sockets
//                     (s2/service.h, head = rank 0 on the bootstrap address, port N); DIR is this rank's fresh S2 directory
//                     (cmd.<n> / req.<n> / resp.<n>, ready.s2, status.s2). Needs --graph, --mtp, --prefill, G6_DEC_SPLIT=3.
//                     Without --serve the --prompt sequences run once and the process exits.
//      [--qp-manifest PM --qp-pack PP]  (G6_NR=4) G6_PF_OWN=4 pair weights resident; or
//                     G6_PF_QP_ROLL=S + G6_PF_QP_CTX=N: S rolling slots, the partner's half over RDMA (docs/DESIGN-QO-PAIR.md)
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
//                     Flash: on by default with --mtp K (Flash TP2 verify cost table, spec_len.h flash_tp2).
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
#include "engine_opts.h"
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
#include "s2/cp_rows.h"
#include "s2/pool.h"
#include "s2/kv_tier.cuh"
#include "rt_rows.cuh"
#include "host_reclaim.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <unistd.h>

using namespace g6rt;

static std::vector<std::string> split(const std::string& s, char c)
{
    std::vector<std::string> v;
    size_t a = 0, b;
    while ((b = s.find(c, a)) != std::string::npos) { v.push_back(s.substr(a, b - a)); a = b + 1; }
    if (a < s.size()) v.push_back(s.substr(a));
    return v;
}

int main(int argc, char** argv)
{
    const auto boot_t0 = std::chrono::steady_clock::now();
    int boot_rank = -1;
    auto boot = [&](const char* phase) {
        fprintf(stderr, "# rank %d: boot %.1f s %s\n", boot_rank,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - boot_t0).count(), phase);
    };
    setvbuf(stdout, nullptr, _IOLBF, 0);
    pf_env_check();                    // strict G6_PF_* choices (envparse.h): a bad value exits before any work
#if G6_MODEL_FLASH
    {   // G6_MOE_TOPK (model change, screen arm): unset = the model's top-k; a bad value dies before any load
        std::string e;
        const int tk = env_digit("G6_MOE_TOPK", 1, geom::TOPK, geom::TOPK, &e);
        if (tk < 0) die(e);
        if (tk < geom::TOPK)
            fprintf(stderr, "# G6_MOE_TOPK=%d: routed experts per token cut %d -> %d, weights renormalized (screen arm)\n",
                    tk, geom::TOPK, tk);
    }
#else
    if (getenv("G6_MOE_TOPK")) die("G6_MOE_TOPK is a Flash-model option (this build is the GLM model)");
    if (getenv("G6_PF_ROUTER_SPLIT")) die("G6_PF_ROUTER_SPLIT is a Flash-model option (this build is the GLM model)");
#endif
#if G6_MODEL_FLASH
    const int route_split_mode = flash_route_split_mode();     // G6_PF_ROUTER_SPLIT: strict 0/1, a bad value dies before any load
#endif
    std::string progress_err;          // G6_PROGRESS_ISOLATE: other threads stay off the proxy CPUs (progress_cpus.h)
    const int progress_isolate = env_choice("G6_PROGRESS_ISOLATE", 1, 1, &progress_err);
    if (progress_isolate < 0) die(progress_err);
    g6::FabricConfig cfg;
    std::vector<std::string> pf_ips2;   // --pf-ips2: rail-1 addresses of the prefill fabrics only (G6_PF_STRIPE dual-rail)
    std::string manifest, packfile;
    V2Src v2src;
    g6rt::CommonOpts co;                // every flag m4sim takes too, parsed in engine_opts.h
    std::string qp_man, qp_file;        // --qp-manifest/--qp-pack: this rank's pair weights, G6_PF_OWN=4
    std::string dfl_man, dfl_file;      // --dflash-manifest/--dflash-pack: the DFlash2 drafter (engine/loader-rs/src/dflash2.rs)
    int dfl_sel = -1;                   // --dflash-sel: 1 = selector walk, 0 = argmax, -1 = the pack decides
    int dfl_k = dfl::K;                 // --dflash-k: drafts a round verifies (1..7; the drafter always drafts 7)
#if G6_MODEL_FLASH
    bool spec_reset_per_request = false; // offline evaluation: no adaptive history from earlier requests
    bool spec_fixed = false;            // --spec-fixed: Flash --mtp K verifies all K drafts every round
#endif
    Plan plan;
    bool image = false, hash = true, reuse_prefix = false;
    bool lanes_explicit = false;        // --pf-lanes given (common_arg parses it)
    std::string s2dir;                  // --serve DIR: serve through S2 (below R)
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
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) die("missing value for " + a); return std::string(argv[++i]); };
        std::string oerr;
        if (a == "--pf-lanes") lanes_explicit = true;
        if (g6rt::common_arg(a, next, co, oerr)) { if (!oerr.empty()) die(oerr); continue; }
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
        else if (a == "--s2-port") s2port = atoi(next().c_str());
        else if (a == "--s2-sessions") s2sess = atoi(next().c_str());
        else if (a == "--s2-epoch") s2epoch = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--set-id")
        {
            set_id = next();
            if (set_id.empty() || set_id.size() > 64 || set_id.find_first_not_of("0123456789abcdef") != std::string::npos)
                die("--set-id must be 1..64 lowercase hex digits");
        }
        else if (a == "--steering-profile") steering_dir = next();
        else if (a == "--steering-weight-set") steering_weight_set = next();
        else if (a == "--steering-force") steering_force_s = next();
        else if (a == "--steering-model-revision") steering_model_revision = next();
        else if (a == "--steering-tokenizer-revision") steering_tokenizer_revision = next();
        else if (a == "--steering-chat-template") steering_template_file = next();
        else if (a == "--steering-chat-template-sha256") steering_template_digest = next();
        else die("unknown argument " + a);
    }
    // the shared flags, under the names the rest of this file uses
    const int cap = co.cap, top = co.top, pfrows = co.pfrows, nc = co.nc, mtp = co.mtp, MP = co.MP;
    int lanes = co.lanes;               // Flash: two lanes for a prefill of up to 8192 rows unless --pf-lanes (below)
    const float temp = co.temp;
    const bool graph = co.graph, wide = co.wide, pfv2 = co.pfv2;
    bool adapt = co.adapt;              // Flash: --mtp turns it on unless --spec-fixed (below)
    const std::string &layers_s = co.layers_s, &levels_s = co.levels_s, &qo_man = co.qo_man, &qo_file = co.qo_file;
    std::vector<std::string>& prompts = co.prompts;
    v2src.dir = co.v2_dir; v2src.slices = co.v2_slices; v2src.widths = co.v2_widths; v2src.dense = co.v2_dense;
    v2src.trellis = co.v2_trellis;
    if (co.n_gen) plan.n_gen = *co.n_gen;          // Plan owns these defaults; apply only what the command line set
    if (co.seed) plan.seed = *co.seed;
    if (co.chunk) plan.chunk = *co.chunk;
    if (co.top_p) plan.top_p = *co.top_p;
    if (co.rs) plan.rs = *co.rs;
    const bool release = !v2src.dense.empty();
    if (cfg.rank < 0 || cfg.ips.size() != (size_t)g6nr::NR || (!release && (manifest.empty() || packfile.empty())))
        die("usage: g6m4 --rank R --ips NR,addresses (this build: NR = " + std::to_string(g6nr::NR) + ") [--ips2 ..] --manifest M --pack P|- [--qo-manifest QM --qo-pack QP] [--qp-manifest PM --qp-pack PP] [--layers ..] [--gen N] [--temp T]");
    if (release && (!manifest.empty() || !packfile.empty() || v2src.dir.empty()))
        die("--v2-dense replaces --manifest / --pack and needs --v2 (the image comes from the release alone)");
    { const char* e = getenv("G6_DEC_SHARED_SCRATCH"); if (e && atoi(e) == 1 && !pfv2) die("G6_DEC_SHARED_SCRATCH=1 needs --pfv2"); }
    const std::vector<int> layers = parse_ints(layers_s);
#if G6_MODEL_FLASH
    if (layers_s == "all") {
        if (layers.size() != (size_t)geom::L_MAIN) die("Flash --layers all did not select every main layer");
        for (int i = 0; i < geom::L_MAIN; ++i)
            if (layers[i] != i) die("Flash --layers all selected an unexpected layer");
    }
    fprintf(stderr, "# rank %d: Flash actual main layers %zu first %d last %d\n", cfg.rank, layers.size(),
            layers.empty() ? -1 : layers.front(), layers.empty() ? -1 : layers.back());
#endif
    if (prompts.empty()) prompts.push_back(g6rt::DEFAULT_PROMPT);
    plan.inv_temp = g6rt::inv_temp(temp);
    if (const std::string e = g6rt::common_check(co, MX, plan.chunk); !e.empty()) die(e);
    if (nc < 0 || nc * g6nr::NR > glue::TP_SORT) die("--top-p-cands must be 0.." + std::to_string(glue::TP_SORT / g6nr::NR));
    // Flash lane default: the MP8192 memory gate passed; larger MPs remain opt-in.
#if G6_MODEL_FLASH
    if (!lanes_explicit && MP > 0 && MP <= 8192) lanes = 2;
#endif
    const bool s2mode = !s2dir.empty();
    const bool dfl = !dfl_man.empty();
    if (s2mode && (getenv("G6_RESID_DUMP") || getenv("G6_LOGIT_DUMP")))
        die("G6_RESID_DUMP / G6_LOGIT_DUMP support offline main steps, not --serve");
    // S2 keeps --mtp next to the drafter: a session alone at its prompt tail runs DFlash2, the others MTP (s2_batch.h)
    if (dfl && (dfl_file.empty() || (mtp && !s2mode) || dfl_sel < -1 || dfl_sel > 1 || dfl_k < 1 || dfl_k > dfl::K))
        die("--dflash-manifest needs --dflash-pack, and replaces --mtp (7 drafts per round) except under S2; --dflash-sel 0|1; --dflash-k 1..7");
    if (dfl && !dfl::OK_NR) die("--dflash needs a rank count that divides the drafter's 8 KV heads (TP4 build)");
    if (plan.rs && !mtp && !dfl) die("--spec-accept rejection needs --mtp or --dflash-manifest");
    const int K = dfl ? dfl::K : mtp;   // drafts per round
#if G6_MODEL_FLASH
    if (spec_reset_per_request && s2mode) die("--spec-reset-per-request is offline only (no --serve)");
    // Flash: --mtp K is adaptive speculation (up to K drafts) unless --spec-fixed
    if (spec_fixed && (adapt || !levels_s.empty() || !mtp)) die("--spec-fixed needs --mtp, and not --spec-adaptive / --spec-levels");
    if (mtp && !spec_fixed && !s2mode && !dfl) adapt = true;
#endif
    if ((adapt || !levels_s.empty()) && ((s2mode && !dfl) || (!mtp && !dfl))) die("--spec-adaptive needs --mtp or --dflash-manifest; under S2 it is the DFlash2 sessions' policy");
    if (!levels_s.empty() && !adapt) die("--spec-levels needs --spec-adaptive");
    SpecLen spec;                       // --spec-adaptive. Estimates persist unless the offline reset option is on
    if (adapt)
    {
        const int kmax = dfl ? dfl_k : mtp;
        std::vector<int> lv;
        if (levels_s.empty()) for (int k = 1; k <= kmax; ++k) lv.push_back(k);
        else lv = parse_ints(levels_s);
        for (int k : lv) if (k < 1 || k > kmax) die("--spec-levels: each level must be 1.." + std::to_string(kmax));
        spec.on = true;
        if (!spec.set_levels(lv)) die("--spec-levels: no level");
        spec.draft_round = dfl ? 5.6 : 2.5;      // ms at 2200 MHz: the DFlash2 step, the MTP first-form pass
        spec.draft_each = dfl ? 0 : 2.1;         // an MTP chain step
#if G6_MODEL_FLASH
        if (!dfl) spec.flash_tp2();
#endif
    }
#if G6_MODEL_FLASH
    const SpecLen spec_initial = spec;  // fully configured, before any observations; includes Flash costs and levels
#endif
    if (s2mode && (!graph || !mtp || !MP || s2port < 1 || s2port > 65535 || s2sess < 1 || s2sess > 8 || !s2epoch))
        die("serving (--serve) needs --graph, --mtp, --prefill, --s2-port 1..65535, --s2-sessions 1..8 and --s2-epoch > 0");
    if (getenv("G6_RT_STORE") && !s2mode) die("G6_RT_STORE needs --serve (S2 below R with G6_CP on)");
    if (getenv("G6_RT_POISON")) die("G6_RT_POISON is an m4sim test: g6m4 reads the local copy of a row without a slot");
    // G6_RT_LOCAL (release): each split run's tail cycles through that many window pages per rank; the pool grants the
    // drafter's whole tail apart from them (kvman-rs Pool::window, s2_cp_map). G6_RT_PLACE: the main model's physical
    // pages per rank (m.cp.phys), so --cap (the logical length) may pass what the local pages hold
    const int rt_win = getenv("G6_RT_LOCAL") ? atoi(getenv("G6_RT_LOCAL")) : 0;
    if (getenv("G6_RT_LOCAL") && (rt_win < 1 || !getenv("G6_RT_STORE"))) die("G6_RT_LOCAL needs G6_RT_STORE and a page count >= 1");
    if (getenv("G6_RT_PLACE") && !rt_win) die("G6_RT_PLACE needs G6_RT_LOCAL (the pages past it are reached through the rows tier only)");
    // --kv-store with it: the pool never retains a window run (kvman-rs Pool::window), so the block stores copy at most
    // its prefix, which stays on pool pages; runs below R are as without the tier
    if (rt_win && getenv("G6_RT_L") && getenv("G6_CP_R") && atoi(getenv("G6_RT_L")) != atoi(getenv("G6_CP_R")))
        die("G6_RT_LOCAL needs G6_RT_L = G6_CP_R (the window holds every tail page past R)");
    steering::PreparedBoot steering_prepared;
    try {
        if (!steering_dir.empty() && !hash) die("--steering-profile requires weight hash checks");
        if (!steering_dir.empty() && (set_id.empty() || steering_weight_set.empty()))
            die("--steering-profile needs --set-id and --steering-weight-set");
        const steering::ExpectedModel expected{"glm-5.3", steering_model_revision, steering_weight_set,
            steering_tokenizer_revision, steering_template_digest, HID, 78};
        steering_prepared = steering::prepare_boot(steering_dir, expected, steering_template_file, layers, steering_force_s);
    } catch (const std::exception& e) { die(e.what()); }
    const auto& steering_profile = steering_prepared.profile;
    const auto& steering_boot = steering_prepared.config;
    // G6_MTP_KSTOP=1|2 (s2_batch.h KStop; default 0): a solo DFlash2 session's MTP rounds draft and verify 1..
    // G6_MTP_KSTOP_KMAX (1..3, default --mtp) drafts, chosen from the drafts' log-probabilities: 1 by G6_MTP_KSTOP_TAU=
    // t1,t2,t3 (draft j while p_1 * .. * p_j >= t_j; default 0,0.2,0.5), 2 by the drafter choice's MTP estimates
    // (G6_MTP_KSTOP_SEED3: their prior for a third draft). G6_MTP_LPLOG=1 (rank 0; 2: every rank): one "# kstop" line per
    // solo round, nothing else changes. The policy is in the S2 runtime signature
    s2::KStop kstop;
    {
        std::string ke;
        kstop.mode = env_choice("G6_MTP_KSTOP", 2, (s2mode && dfl && mtp >= 2) ? 2 : 0, &ke);
        const int lplog = env_choice("G6_MTP_LPLOG", 2, 0, &ke);
        if (kstop.mode < 0 || lplog < 0) die(ke);
        kstop.log = lplog == 2 || (lplog == 1 && cfg.rank == 0);
        kstop.kmax = getenv("G6_MTP_KSTOP_KMAX") ? atoi(getenv("G6_MTP_KSTOP_KMAX")) : mtp;
        if (const char* e = getenv("G6_MTP_KSTOP_TAU"))
        {
            std::vector<double> t;
            std::string v = e;
            for (size_t a = 0; a <= v.size();)
            {
                const size_t b = std::min(v.find(',', a), v.size());
                char* end = nullptr;
                const std::string f = v.substr(a, b - a);
                t.push_back(strtod(f.c_str(), &end));
                if (f.empty() || *end || !std::isfinite(t.back())) die("G6_MTP_KSTOP_TAU=t1[,t2[,t3]]: numbers");
                a = b + 1;
            }
            if (t.size() > 3) die("G6_MTP_KSTOP_TAU: at most 3 thresholds");
            for (size_t j = 0; j < t.size(); ++j) kstop.tau[j + 1] = t[j];
        }
        if (const char* e = getenv("G6_MTP_KSTOP_SEED3")) kstop.seed3 = atof(e);
        if (kstop.mode && (!s2mode || !dfl || mtp < 2 || kstop.kmax < 1 || kstop.kmax > 3 || !(kstop.seed3 > 0 && kstop.seed3 < 1) ||
                           kstop.kmax + 1 > std::max(s2sess * (mtp + 1), plan.chunk)))
            die("G6_MTP_KSTOP needs --serve, --dflash-manifest and --mtp 2+; G6_MTP_KSTOP_KMAX 1..3 (verify rows within a step's), "
                "G6_MTP_KSTOP_SEED3 in (0, 1)");
        if (!kstop.mode) kstop.kmax = 0;
        // G6_MTP_KSTOPB=1 (s2_batch.h KStopB; default 0): in shared rounds each session verifies its own k_q of the K
        // drafts: the most j with p_1 * .. * p_i >= t_i for every i <= j, at least G6_MTP_KSTOPB_KMIN (0|1, default 0).
        // G6_MTP_KSTOPB_TAU=t1[,t2[,...]] (up to 7; default 0.6,0.88) at every session count, G6_MTP_KSTOPB_TAU_S<n> at n
        // sessions; past the configured entries there is no stop (the remaining drafts all verify).
        // G6_MTP_KSTOPB_WIDTHS=M,..: the main widths captured besides the multiples of K + 1 (default every width the rule reaches;
        // a width outside them takes the next most likely drafts). G6_MTP_KSTOPB_EXPLORE=n (measurement): every n-th
        // round's shared sessions verify a hashed k_q instead. In the S2 runtime signature
        kstop.b.mode = env_choice("G6_MTP_KSTOPB", 1, 0, &ke);
        kstop.b.kmin = env_choice("G6_MTP_KSTOPB_KMIN", 1, 0, &ke);
        if (kstop.b.mode < 0 || kstop.b.kmin < 0) die(ke);
        auto nums = [](const char* name, size_t most) {
            std::vector<double> t;
            const std::string v = getenv(name);
            for (size_t a = 0; a <= v.size();)
            {
                const size_t b = std::min(v.find(',', a), v.size());
                char* end = nullptr;
                const std::string f = v.substr(a, b - a);
                t.push_back(strtod(f.c_str(), &end));
                if (f.empty() || *end || !std::isfinite(t.back()) || t.size() > most)
                    die(std::string(name) + ": at most " + std::to_string(most) + " comma-separated numbers");
                a = b + 1;
            }
            return t;
        };
        double bt[8] = {0, 0.6, 0.88, 0.95, 0, 0, 0, 0};   // tau 0 past the entries: no stop there
        if (getenv("G6_MTP_KSTOPB_TAU")) { const auto t = nums("G6_MTP_KSTOPB_TAU", 7); std::copy(t.begin(), t.end(), bt + 1); }
        for (int S = 1; S <= s2::S2N_SLOTS; ++S)
        {
            std::copy(bt, bt + 8, kstop.b.tau[S]);
            const std::string n = "G6_MTP_KSTOPB_TAU_S" + std::to_string(S);
            if (getenv(n.c_str())) { const auto t = nums(n.c_str(), 7); std::copy(t.begin(), t.end(), kstop.b.tau[S] + 1); }
        }
        if (getenv("G6_MTP_KSTOPB_WIDTHS"))
            for (double w : nums("G6_MTP_KSTOPB_WIDTHS", 64))
            {
                if (w != (int)w || w < 1 || w > s2sess * (mtp + 1)) die("G6_MTP_KSTOPB_WIDTHS: whole widths 1..sessions x (mtp + 1)");
                kstop.b.widths.push_back((int)w);
            }
        if (getenv("G6_MTP_KSTOPB_EXPLORE")) kstop.b.explore = atoi(getenv("G6_MTP_KSTOPB_EXPLORE"));
        if (kstop.b.mode && (!s2mode || !dfl || kstop.b.explore < 0))
            die("G6_MTP_KSTOPB needs --serve and --dflash-manifest (the draft log-probabilities); G6_MTP_KSTOPB_EXPLORE >= 0");
        if (!kstop.b.mode) kstop.b = s2::KStopB();
    }
#if G6_MODEL_FLASH
    if (s2mode && (s2sess != 1 || reuse_prefix || dfl || plan.rs))
        die("Flash serve supports one session, no prefix reuse, DFlash or rejection sampling");
#endif
    // --reuse-prefix: one conversation's caches (prefix_reuse.h), or under S2 every retired session's run (s2/pool.h).
    // The DFlash2 drafter's prompt-context taps (DF.pf_feat) are written by the prefill rows a hit skips.
    if (!kv_tiers.empty() && !(s2mode && reuse_prefix)) die("--kv-store needs --serve and --reuse-prefix (it keeps the runs prefix reuse retains)");
    if (kv_old)
    {
        if (kv_keep.empty()) kv_keep = kv_old_all ? "all" : "system";
        if (cfg.rank == 0)
            fprintf(stderr, "# warning: --kv-tier and --kv-shared-after are deprecated: --kv-store STORE,size=N, --kv-keep %s, "
                            "--kv-system-after %d\n", kv_keep.c_str(), kv_system_after);
    }
    if (kv_keep.empty()) kv_keep = "system";
    if (kv_keep != "system" && kv_keep != "sessions" && kv_keep != "all") die("--kv-keep system, sessions or all");
    if (kv_system_after < 1) die("--kv-system-after N: N >= 1");
    if (!(kv_write_limit >= 0) || kv_write_limit > 1000) die("--kv-write-limit F: 0 (off) to 1000 blocks per page computed");
    {   // the KV row is fixed at build time (attn/kvfmt.h): the launcher runs bin/g6m4 for fp8 and bin/g6m4-int4 for int4,
        // and each binary refuses the other's format. The int4 build needs the flag, so it never starts by accident.
#if defined(G6_KV4) && G6_KV4
        const std::string built = "int4";
#else
        const std::string built = "fp8";
#endif
        if (kv_format.empty() && built == "int4") die("this engine is the int4 KV build (bin/g6m4-int4): it needs --kv-format int4");
        if (kv_format.empty()) kv_format = built;
        if (kv_format != "fp8" && kv_format != "int4") die("--kv-format fp8 or int4");
        if (kv_format != built)
            die("--kv-format " + kv_format + ": this engine is the " + built + " build (bin/g6m4 is fp8, bin/g6m4-int4 is int4)");
        if (cfg.rank == 0) fprintf(stderr, "# rank 0: KV format %s (latent row %d B, %s)\n", kv_format.c_str(), g6kv::ROW, g6kv::NAME);
    }
    if (reuse_prefix && dfl && !s2mode) die("--reuse-prefix is not supported with --dflash-manifest (the drafter's prompt-context taps come from the prefill)");
    std::vector<Plan> plans;            // one sequence per --prompt
    std::vector<int> row_counts;
    for (const std::string& ps : prompts)
    {
        Plan p = plan;
        if (steering_boot.forced) {
            p.steering = steering_boot.selected(steering_boot.forced_scale_bits);
            p.steering_scale = steering::bits_float(steering_boot.forced_scale_bits);
        }
        p.prompt = parse_ints(ps);
        if (p.prompt.empty()) die("empty prompt");
        if ((int)p.prompt.size() + p.n_gen + 2 * K > cap) die("prompt + gen (+ 2 x drafts) exceeds --cap");
        if (MP) { const int pr = pfrows < 0 ? (int)p.prompt.size() + pfrows : pfrows;   // --pf-rows -N: all rows but the last N
                  p.first = pr > 0 && pr < (int)p.prompt.size() - 1 ? pr : (int)p.prompt.size() - 1; }   // --pf-rows: the rest is teacher-forced through decode steps
        for (int M : plan_rows(p))
            if (std::find(row_counts.begin(), row_counts.end(), M) == row_counts.end()) row_counts.push_back(M);
        plans.push_back(p);
    }
    if (s2mode || reuse_prefix)         // any prompt (reuse changes the offline prompt-step row counts too): 1..chunk
        for (int M = 1; M <= plan.chunk; ++M)
            if (std::find(row_counts.begin(), row_counts.end(), M) == row_counts.end()) row_counts.push_back(M);
    std::vector<int> draft_rows;        // row counts of first-form draft passes: the prompt steps' and 1..K+1
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
    for (int M = 2; M <= kstop.kmax + 1; ++M)   // G6_MTP_KSTOP: solo MTP verifies of 2..kmax + 1 rows, first forms of as many
    {
        if (std::find(row_counts.begin(), row_counts.end(), M) == row_counts.end()) row_counts.push_back(M);
        if (std::find(draft_rows.begin(), draft_rows.end(), M) == draft_rows.end()) draft_rows.push_back(M);
    }
    for (int k : spec.levels)           // --spec-adaptive: a verify graph per level
        if (std::find(row_counts.begin(), row_counts.end(), k + 1) == row_counts.end()) row_counts.push_back(k + 1);
#if G6_MODEL_FLASH
    if (s2mode) for (int n = 1; n <= mtp + 1; ++n)
        if (std::find(row_counts.begin(), row_counts.end(), n) == row_counts.end()) row_counts.push_back(n);
#endif
    const int NL = (int)layers.size(), NC = 2 * NL + 1, R = cfg.rank;
    // G6_PF_OWN=4 (TP4) env refusals before any CUDA / fabric work (qp_wire.h roll_env_refusal)
    const bool own4 = MP && getenv("G6_PF_OWN") && atoi(getenv("G6_PF_OWN")) == 4;
    std::string qre;
    const int qp_roll_s = env_choice("G6_PF_QP_ROLL", 4, 0, &qre);       // rolling pack (pf_env_check made it 0 or 2..4)
    const bool qp_roll_ser = env_choice("G6_PF_QP_ROLL_SER", 1, 0, &qre) == 1;
    const int qp_wire_e = env_choice("G6_PF_QP_WIRE_E", qw::max_entries(), 1, &qre);   // wire ring entries (qp_wire.h)
    {
        if (qp_wire_e < 0) die(qre);
        const std::string why = qw::roll_env_refusal(own4, qp_roll_s, getenv("G6_PF_QP_WIRE_E") != nullptr, qp_wire_e, getenv("G6_PF_QP_CTX"), !qp_file.empty());
        if (!why.empty()) die(why);
        if (own4 && g6nr::NR != 4) die("G6_PF_OWN=4 is TP4 only (G6_NR=4)");
        if (own4 && !qp_roll_s && (qp_man.empty() || qp_file.empty())) die("G6_PF_OWN=4 needs --qp-manifest and --qp-pack (or G6_PF_QP_ROLL)");
        if (!own4 && (!qp_man.empty() || !qp_file.empty())) die("--qp-manifest / --qp-pack need G6_PF_OWN=4 and --prefill");
    }

    G6RT_CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cfg.slot_bytes = (size_t)MX * HID * 2;
    cfg.ncoll = NC;
    cfg.relay_cpu = cfg.proxy_cpu > 7 ? cfg.proxy_cpu - 8 : -1;   // --ring: the relay thread's own core (N-1..N-7 are proxies)
    g6::Fabric fab;
    boot_rank = cfg.rank;
    mem_min_gib();                      // a bad G6_MEM_MIN stops the boot here, not after the weights have loaded
    boot("start");
    cfg.xtrace = 1;                    // G6_XTRACE=1: stage-0 trace on the decode fabric only (xtrace.h); off by default
    fab.init(cfg);
    cfg.xtrace = 0;
    const g6::FabricDev f = fab.dev();
    fab.start_proxy();
    g6::Fabric fabp;                    // prefill region
    const bool pf_rs = getenv("G6_PF_RS") && atoi(getenv("G6_PF_RS")) == 1;     // reduce-scatter form (prefill.cuh)
    const int pf_own = getenv("G6_PF_OWN") ? atoi(getenv("G6_PF_OWN")) : 0;   // 1: row-owner projections, 2: + index (prefill.cuh)
    const bool pf_qo = MP && pf_own == 3;                                        // 3: row-owner QO attention (QO-INTEGRATION.md)
    const bool pf_qp = own4;                                                     // 4: pair-owner QO attention (TP4, qp.cuh)
    if (pf_qo && !release && (qo_man.empty() || qo_file.empty())) die("G6_PF_OWN=3 needs --qo-manifest and --qo-pack");
    if (release && (!qo_man.empty() || !qp_man.empty())) die("--v2-dense: the QO weights come from the release (no --qo- / --qp- packs)");
    v2src.qo = pf_qo;
    // G6_PF_STRIPE=1 (needs --pf-ips2 or --ips2): prefill payload writes >= G6_PF_STRIPE_MIN bytes (default 1 MiB) go
    // out half on each rail (FabricConfig::stripe_min). Unset = on when --pf-ips2 is given, else one rail as before;
    // 0 = one rail.
    auto pf_stripe_cfg = [&](g6::FabricConfig& cp, const std::vector<std::string>& ips2) {
        std::string e;
        const int on = env_choice("G6_PF_STRIPE", 1, ips2.empty() ? 0 : 1, &e);
        if (on < 0) die(e);
        if (!ips2.empty()) cp.ips2 = ips2;
        if (on != 1) return;
        if (cp.ips2.size() != cp.ips.size()) die("G6_PF_STRIPE=1 needs --pf-ips2 (or --ips2) with one address per rank");
        const char* m = getenv("G6_PF_STRIPE_MIN");
        cp.stripe_min = m ? (size_t)atoll(m) : (size_t)1 << 20;
        if (cp.stripe_min < 8192) die("G6_PF_STRIPE_MIN must be >= 8192");
    };
    if (MP)
    {
        g6::FabricConfig cp = cfg;
        cp.slot_bytes = pf_slot_bytes(MP, pf_rs, (int)cfg.ips.size());
        cp.ncoll = pf_rs ? 5 : 2;
        cp.port = cfg.port + 8;
        pf_stripe_cfg(cp, pf_ips2);
        cp.proxy_cpu = cfg.proxy_cpu > 0 ? cfg.proxy_cpu - 1 : -1;
        fabp.init(cp);
        fabp.start_proxy();
    }
    if (lanes != 1 && lanes != 2) die("--pf-lanes must be 1 or 2");
    const bool two = MP && lanes == 2;
    g6::Fabric fabp2;                   // prefill region of the second lane (--pf-lanes 2)
    if (two)
    {
        g6::FabricConfig cp = cfg;
        cp.slot_bytes = pf_slot_bytes(MP, pf_rs, (int)cfg.ips.size());
        cp.ncoll = pf_rs ? 5 : 2;
        cp.port = cfg.port + 24;
        pf_stripe_cfg(cp, pf_ips2);
        cp.proxy_cpu = cfg.proxy_cpu > 1 ? cfg.proxy_cpu - 2 : -1;
        fabp2.init(cp);
        fabp2.start_proxy();
    }
#if G6_MODEL_FLASH
    // Independent pool-ID exchange: default on for TP2 Flash prefill (G6_PF_POOL_OWNER unset = auto, 0 = off,
    // 1 = on and die if unsupported). Existing fabp/fabp2 are unchanged.
    g6::Fabric fabix, fabix2;
    const int pool_mode = flash_pool_owner_mode();
    const bool pool_fits = MP && g6nr::NR == 2 && cfg.ips.size() == 2 && !cfg.ring;
    const char* pool_conflict = nullptr;
    for (const char* key : {"G6_FPF_P1", "G6_FPF_DSA_P1", "G6_FPF_ATTN_P1"})
        if (getenv(key) && atoi(getenv(key)) == 1) pool_conflict = key;
    if (pool_mode == 1 && !pool_fits) die("G6_PF_POOL_OWNER=1 needs TP2 Flash prefill (not ring)");
    if (pool_mode == 1 && pool_conflict) die(std::string("G6_PF_POOL_OWNER=1 conflicts with ") + pool_conflict);
    const bool pool_owner = pool_mode == 1 || (pool_mode == 2 && pool_fits && !pool_conflict);
    if (pool_owner)
    {
        auto init_ids = [&](g6::Fabric& ids, int port, int cpu_offset) {
            g6::FabricConfig cp = cfg;
            cp.ncoll = 1; cp.slot_bytes = flash_pool_owner::payload(MP);
            cp.port = cfg.port + port;
            cp.proxy_cpu = cfg.proxy_cpu > cpu_offset ? cfg.proxy_cpu - cpu_offset : -1;
            pf_stripe_cfg(cp, pf_ips2);
            ids.init(cp); ids.start_proxy();
        };
        init_ids(fabix, 64, 9);
        if (two) init_ids(fabix2, 72, 10);
    }
    // G6_PF_ROUTER_SPLIT=1 (opt-in): one ncoll = 1 route-table fabric per prefill lane, at any rank count and on a
    // switched mesh or the switchless ring (the fabric relays the diagonal, as it does for the prefill fabric).
    // Ports +80 / +88 and proxy CPU offsets 11 / 12 (taken: ports 8 16 24 32 40 48 56 64 72, offsets 1..10).
    g6::Fabric fabrs, fabrs2;
    const bool route_split = route_split_mode == 1;
    if (route_split)
    {
        if (!(MP && g6nr::NR >= 2 && g6nr::NR <= flash_route_split::MAX_NR && cfg.ips.size() == (size_t)g6nr::NR))
            die("G6_PF_ROUTER_SPLIT=1 needs Flash prefill (--prefill) on 2 to 8 ranks");
        auto init_rs = [&](g6::Fabric& rs, int port, int cpu_offset) {
            g6::FabricConfig cp = cfg;
            cp.ncoll = 1; cp.slot_bytes = flash_route_split::payload(MP, (int)cfg.ips.size());
            cp.port = cfg.port + port;
            cp.proxy_cpu = cfg.proxy_cpu > cpu_offset ? cfg.proxy_cpu - cpu_offset : -1;
            pf_stripe_cfg(cp, pf_ips2);
            rs.init(cp); rs.start_proxy();
        };
        init_rs(fabrs, 80, 11);
        if (two) init_rs(fabrs2, 88, 12);
        fprintf(stderr, "# rank %d: G6_PF_ROUTER_SPLIT=1: prefill MoE router row-split, route tables over a dedicated fabric%s\n",
                cfg.rank, two ? " per lane" : "");
    }
#endif
    g6::Fabric fabd;                    // MTP draft region: 3 collectives, decode-sized slots
    if (mtp)
    {
        g6::FabricConfig cd = cfg;
        cd.ncoll = 3;
        cd.port = cfg.port + 16;
        cd.proxy_cpu = cfg.proxy_cpu > 5 ? cfg.proxy_cpu - 6 : -1;   // own core: fabp2 (prefill lane 2) has N-2
        fabd.init(cd);
        fabd.start_proxy();
    }
    g6::Fabric fabf;                    // --dflash: the drafter's region, dfl::NC collectives, decode-sized slots
    if (dfl)
    {
        g6::FabricConfig cd = cfg;
        cd.ncoll = dfl::NC;
        cd.port = cfg.port + 40;
        cd.proxy_cpu = cfg.proxy_cpu > 6 ? cfg.proxy_cpu - 7 : -1;   // own core: not N-2 (fabp2) or N-6 (fabd)
        fabf.init(cd);
        fabf.start_proxy();
    }
#if G6_NR == 4
    g6::Fabric fabq;                    // G6_PF_QP_ROLL: ring of [lo | hi] pair half images, one-peer jobs (qp_wire.h)
    QpWire qp_wire;
    if (qp_roll_s)
    {
        g6::FabricConfig qc = cfg;
        qc.vcap.clear(); qc.vdst_k.clear();
        qc.bulk_bytes = qw::bulk_bytes(qp_wire_e);
        qc.bulk_flags = qw::bulk_flags(qp_wire_e);
        qc.port = cfg.port + 56;
        qc.proxy_cpu = cfg.proxy_cpu > 4 ? cfg.proxy_cpu - 5 : -1;
        fabq.init(qc);
        fabq.start_proxy();
        qp_wire.bd = fabq.bulk_dev(); qp_wire.E = qp_wire_e; qp_wire.peer = qw::partner(cfg.rank);
        qp_wire.enqueue = [&fabq](g6::Fabric::BulkJob j) { fabq.enqueue_bulk(std::move(j)); };
        fprintf(stderr, "# rank %d: G6_PF_QP_ROLL=%d wire fabric: %d entr%s x [lo | hi] %.2f MB = %.3f GB pinned + registered, partner rank %d\n",
                cfg.rank, qp_roll_s, qp_wire_e, qp_wire_e == 1 ? "y" : "ies", qw::HALF / 1e6, qw::bulk_bytes(qp_wire_e) / 1e9, qp_wire.peer);
    }
    auto qp_err = [&]() -> unsigned { return qp_roll_s ? *fabq.bulk_dev().error : 0u; };
#else
    auto qp_err = [&]() -> unsigned { return 0u; };
#endif
    // G6_DFL_REUSE=1|2 (s2_batch.h DflReuse; default 2 when serving DFlash2 with MTP, else 0): 1 = a DFlash2 session whose
    // prefill a prefix-reuse hit cut into the drafter's window keeps the drafter (the rows the prefill taps, and the ring's
    // rows of the last DFlash2 session with the same tokens); 2 = also a session alone on MTP takes it. G6_DFL_REUSE_MIN=
    // <rows> (0..2048, default 256): MTP rounds until the drafter holds that many context rows. Drafts only. In the S2
    // runtime signature
    std::string dre;
    const int dfl_reuse = env_choice("G6_DFL_REUSE", 2, (dfl && s2mode && mtp) ? 2 : 0, &dre);
    if (dfl_reuse < 0) die(dre);
    if (dfl_reuse && !(dfl && s2mode)) die("G6_DFL_REUSE needs --dflash-manifest and --serve");
    int dfl_reuse_min = 256;
    if (const char* e = getenv("G6_DFL_REUSE_MIN"))
    {
        char* end = nullptr;
        const long v = strtol(e, &end, 10);
        if (!*e || *end || v < 0 || v > dfl::WIN) die("G6_DFL_REUSE_MIN: 0.." + std::to_string(dfl::WIN) + " rows");
        dfl_reuse_min = (int)v;
    }
    // G6_DRAFT_PAR=1|2 (s2_batch.h draft_par_env; default 0): a solo DFlash2 round's drafter step runs on its own stream
    // under the MTP first form (and the chain on MTP rounds). 2: a drafting step goes on a high-priority stream (its CTAs
    // are dispatched first while both wait), a context-only step stays at the default priority. Each graph's bytes are as
    // serial. Progress: every kernel that waits for peers runs at most 8 CTAs (k_wait_peers 1, k_walk 1, head merges M),
    // so the other stream always finds SMs. In the S2 runtime signature
    std::string dpe;
    const int draft_par = env_choice("G6_DRAFT_PAR", 2, (dfl && s2mode && mtp) ? 2 : 0, &dpe);
    if (draft_par < 0) die(dpe);
    if (draft_par && !(dfl && s2mode && mtp)) die("G6_DRAFT_PAR needs --dflash-manifest, --mtp and --serve");
    // G6_QA_SPLIT=1|2, G6_IDXQ_SPLIT=1, G6_MTP_SPLIT=1 (rank.cuh QSplit; default 0 = off): replicated GEMVs split by output
    // tiles across the ranks and all-gathered. The main model's gathers run on fabg (port + 64), the MTP draft's on fabdg
    // (port + 72), each created right after its model is built (the plan sets the collective count). Proxy CPU N-9 for
    // both (main and draft steps never run at once; inside the shim / front end's taskset range 0..N-8;
    // G6_QA_SPLIT_CPU moves it). Every rank must run the same values (the S2 runtime signature carries them).
    std::string qse;
    // replicated-GEMV split needs peers to gather across (pointless at TP1); the Flash build plans no gathers
    const bool qs_def = g6nr::NR > 1 && !G6_MODEL_FLASH;
    const int qs_mode = env_choice("G6_QA_SPLIT", 2, qs_def ? 2 : 0, &qse), iq_split = env_choice("G6_IDXQ_SPLIT", 1, qs_def ? 1 : 0, &qse),
              mtp_split = env_choice("G6_MTP_SPLIT", 1, (qs_def && mtp) ? 1 : 0, &qse);
    if (qs_mode < 0 || iq_split < 0 || mtp_split < 0) die(qse);
    if (G6_MODEL_FLASH && (qs_mode || iq_split || mtp_split)) die("G6_QA_SPLIT, G6_IDXQ_SPLIT and G6_MTP_SPLIT are GLM-5.3 paths");
    if (mtp_split && !mtp) die("G6_MTP_SPLIT=1 needs --mtp");
    // G6_S2_DSEL_MODEL=1 + G6_S2_DSEL_TABLE=<key:value list | file> (s2_batch.h DSelTable; default 0): the drafter
    // choice prices both drafters from one frozen cost table and MTP rounds under the confidence stop. Every rank reads
    // the same table; its canonical form is in the S2 runtime signature
    std::string pce;
    const int dsel_model = env_choice("G6_S2_DSEL_MODEL", 1, 0, &pce);
    if (dsel_model < 0) die(pce);
    s2::DSelTable dsel_tab;
    if (dsel_model)
    {
        const char* e = getenv("G6_S2_DSEL_TABLE");
        if (!e || !*e || !dfl || !s2mode) die("G6_S2_DSEL_MODEL needs G6_S2_DSEL_TABLE, --serve and --dflash-manifest");
        std::string text = e, err;
        if (text.find(':') == std::string::npos)    // no key:value: the path of a file holding the table
        {
            std::ifstream f(text);
            if (!f) die("G6_S2_DSEL_TABLE: cannot read " + text);
            text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        if (!dsel_tab.parse(text, &err)) die(err);
    }
    g6::Fabric fabg, fabdg;
    bool fabg_on = false, fabdg_on = false;
    auto qs_fabric = [&](g6::Fabric& fb, const Model& mm, int port_off, const char* what) {
        g6::FabricConfig gc = cfg;
        gc.vcap.clear(); gc.vdst_k.clear();
        gc.ncoll = mm.qs.ncoll();
        gc.slot_bytes = qs_slot_bytes(mm);
        gc.port = cfg.port + port_off;
        gc.proxy_cpu = cfg.proxy_cpu > 8 ? cfg.proxy_cpu - 9 : -1;
        if (const char* e = getenv("G6_QA_SPLIT_CPU")) gc.proxy_cpu = atoi(e);   // its proxy's core (-1 = unpinned); A/B of the placement
        fb.init(gc);
        fb.start_proxy();
        fprintf(stderr, "# rank %d: %s gather fabric: %d collectives of up to %zu B, %.1f MB pinned, proxy CPU %d\n", cfg.rank, what,
                gc.ncoll, gc.slot_bytes, 2.0 * gc.ncoll * (1 + cfg.ips.size()) * ((gc.slot_bytes + 8 + 255) & ~size_t(255)) / 1e6, gc.proxy_cpu);
    };
    auto qs_err = [&]() -> unsigned { return (fabg_on ? *fabg.dev().error : 0u) | (fabdg_on ? *fabdg.dev().error : 0u); };   // the gather proxies' own error words
    // G6_DRAFT_HEAD_I4=1 (rank.cuh build_draft_head_i4; default 0): the MTP draft's head reads INT4 g32 records of the
    // rank's vocab slice; the target's head is unchanged. In the S2 runtime signature.
    const DraftHeadI4 dhead = draft_head_i4_env();
    if (dhead.on && !mtp) die("G6_DRAFT_HEAD_I4=1 needs --mtp");
    // G6_DFL_HEAD_I4=1 (dflash.cuh dfl_head_i4; default 0): the DFlash2 drafter's head reads the same INT4 records (the MTP
    // draft's copy when there is one), with G6_DRAFT_HEAD_KS / _TPC. In the S2 runtime signature.
    std::string dfe;
    const int dfl_i4 = env_choice("G6_DFL_HEAD_I4", 1, 0, &dfe);
    if (dfl_i4 < 0) die(dfe);
    if (dfl_i4 && !dfl) die("G6_DFL_HEAD_I4=1 needs --dflash-manifest");
    // G6_DFL_BODY_I4=1 (dflash.cuh dfl_body_i4; default 0): the DFlash2 drafter's body GEMVs read INT4 g32 copies of their
    // MXFP8 records. Drafts only; +382 MB per rank. In the S2 runtime signature
    const int dfl_body_i4_on = env_choice("G6_DFL_BODY_I4", 1, 0, &dfe);
    if (dfl_body_i4_on < 0) die(dfe);
    if (dfl_body_i4_on && !dfl) die("G6_DFL_BODY_I4=1 needs --dflash-manifest");
    int cp_step = 0;                    // G6_CP_STEP_ROWS (s2/cp_rows.h): the rows a step past R may carry; 0 = off
#if defined(G6_CP) && G6_CP
    // CP: env as m4sim (G6_CP, G6_CP_R, G6_CP_S3, G6_CP_MMAX). Own fabric `fabc` (cplayout.h variable
    // layout: per-collective slots, single-destination partials), port + 32, its own proxy CPU.
    const bool cp_on = !getenv("G6_CP") || atoi(getenv("G6_CP")) != 0;
    const int cp_R = getenv("G6_CP_R") ? atoi(getenv("G6_CP_R")) : 0;
    const int cp_s3 = getenv("G6_CP_S3") ? atoi(getenv("G6_CP_S3")) : 1;
    int cp_mmax = K + 1;
    for (int M : row_counts) cp_mmax = std::max(cp_mmax, M);
    // S2: a split run past R shares batched steps, so any S2 step (sc.max_rows below) may run the CP block.
    // G6_CP_STEP_ROWS=n: S2 steps past R carry at most n rows (Driver::step caps their rounds, cp_rows.h), so the fabric
    // is sized for n: at least the rows before #94 (cp_mmax here), at most the widest S2 step
    if (s2mode)
    {
        const int full = std::min(MX, std::max(s2sess * (mtp + 1), plan.chunk));
        std::string ce;
        cp_step = g6s2::cp_step_rows_env(getenv("G6_CP_STEP_ROWS"), cp_mmax, full, &ce);
        if (cp_step < 0) die(ce);
        cp_mmax = std::max(cp_mmax, cp_step ? cp_step : full);
    }
    else if (getenv("G6_CP_STEP_ROWS") && atoi(getenv("G6_CP_STEP_ROWS"))) die("G6_CP_STEP_ROWS needs --serve");
    if (getenv("G6_CP_MMAX")) cp_mmax = std::max(cp_mmax, atoi(getenv("G6_CP_MMAX")));
    if (cp_R < 0 || cp_R % 64) die("G6_CP_R must be a multiple of 64");
    g6::CpColls cp_colls;
    {
        std::vector<bool> full;
        for (int l : layers) full.push_back(cp_full_index(l));
        cp_colls.make(full, (int)cfg.ips.size(), cp_mmax);
    }
    g6::Fabric fabc;
    if (cp_on)
    {
        g6::FabricConfig cc = cfg;
        cc.vcap = cp_colls.cap;
        cc.vdst_k = cp_colls.dst_k;
        cc.port = cfg.port + 32;
        cc.proxy_cpu = cfg.proxy_cpu > 2 ? cfg.proxy_cpu - 3 : -1;
        fabc.init(cc);
        fabc.start_proxy();
        fprintf(stderr, "# rank %d: G6_CP fabric %d collectives, mmax %d, %.3f GB pinned + registered\n", cfg.rank, cp_colls.ncoll(),
                cp_mmax, (fabc.vlayout().send_bytes() + fabc.vlayout().recv_bytes()) / 1e9);
    }
    // enqueue the CP collectives of a step that runs the CP block (context > R), before its GPU work
    auto cp_enqueue = [&](unsigned long long step, int M, int context) { if (cp_on && context > cp_R) fabc.enqueue_step(step, cp_colls.sizes(M)); };
    auto cp_error = [&]() -> unsigned { return (cp_on ? *fabc.dev().error : 0u) | qs_err(); };   // + the G6_QA_SPLIT proxy
#else
    auto cp_enqueue = [&](unsigned long long, int, int) {};
    auto cp_error = [&]() -> unsigned { return qs_err(); };   // the G6_QA_SPLIT proxy
#endif
    G6RT_CK(bind::init());
    G6RT_CK(l2pf::init_env());     // G6_L2PF (kernels/l2pf.cuh); unset = off
    pf_prof().on = getenv("G6_PF_PROF") != nullptr;
    if (getenv("G6_DEC_PROF") && graph) die("G6_DEC_PROF needs plain launches (no --graph)");
    cudaStream_t st;
    G6RT_CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

    g6::Pack pack;
    DevStore store;
    Model m;
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
        fprintf(stderr, "# rank %d: G6_PF_OWN=3 full-head q_b / o_proj / UK / UV for %zu layers\n", R, qow.size());
    }
#if G6_NR == 4
    g6::Pack qp_pack;
    g6::DevImage qp_img;                // --image: the qp pack streamed into one device allocation (no host copy)
    std::vector<QpW> qpw(layers.size());   // pf_qp: pair weights per model layer (empty outside G6_PF_QP_LAYERS)
    const std::vector<char> qpl = pf_qp ? pf_qp_mask(layers) : std::vector<char>(layers.size(), 0);
    QpRoll* qp_roll = nullptr;
    if (qp_roll_s)                      // S slots, own half from this rank's pack, partner half over fabq (no --qp-pack)
    {
        qp_roll = qp_roll_new(qp_roll_s, qp_roll_ser, std::vector<Find>{find}, R, layers, qpl, qpw, &qp_wire);
        int n = 0;
        for (char c : qpl) n += c != 0;
        fprintf(stderr, "# rank %d: G6_PF_OWN=4 pair q_b / o_proj / UK / UV for %d of %zu layers (G6_PF_QP_LAYERS), rolling: own pack + partner by RDMA\n", R, n, layers.size());
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
        fprintf(stderr, "# rank %d: G6_PF_OWN=4 pair q_b / o_proj / UK / UV for %d of %zu layers (G6_PF_QP_LAYERS)\n", R, n, layers.size());
    }
#endif
    m.wide = wide;
    m.qs.mode = qs_mode;
    m.qs.iq = iq_split == 1;
#if defined(G6_CP) && G6_CP
    m.cp.on = cp_on; m.cp.R = cp_R; m.cp.s3 = cp_s3; m.cp.colls = cp_colls;
    if (const char* e = getenv("G6_RT_PLACE"))
    {
        m.cp.phys = atoi(e);
        if (m.cp.phys < cp_R / 64 + rt_win) die("G6_RT_PLACE must hold the prefix (G6_CP_R / 64 pages) and one window");
        if (R == 0) fprintf(stderr, "# rank 0: G6_RT_PLACE: %d physical pages per rank for the main model (--cap %d is the logical length)\n", m.cp.phys, cap);
    }
#endif
    host_reclaim(R, "after the pack");         // the loaders keep no page cache; this trims the heap they freed
    boot("pack loaded");
#if G6_MODEL_FLASH
    if (mtp) m.kda_ring = mtp + 1;      // KDA state slots for the verify ring
#endif
#if G6_MODEL_FLASH
    // Flash prefill (flash_prefill.cuh): the P1 bodies run the decode blocks in MX-row chunks, so the decode bindings
    // stay decode-sized. Combinations the Flash prefill does not support yet stop here.
    if (MP && (pf_rs || pf_own || !qp_man.empty() || !qo_man.empty() || dfl || reuse_prefix
#if defined(G6_CP) && G6_CP
               || cp_on
#endif
               ))
        die("Flash --prefill: not with G6_PF_RS, G6_PF_OWN, --qo-/--qp- packs, --dflash, --reuse-prefix or CP (P1)");
    if (two && getenv("G6_PF_STREAMS") && atoi(getenv("G6_PF_STREAMS")) == 2)
        die("Flash --pf-lanes 2 uses one compute stream; G6_PF_STREAMS=2 is not supported");
    if (two && getenv("G6_RESID_DUMP"))
        die("Flash --pf-lanes 2: G6_RESID_DUMP is not supported (use G6_PF_HASH / G6_LOGIT_DUMP)");
    build(m, R, layers, find, cap, f, stepp, st, MX);
#else
    build(m, R, layers, find, cap, f, stepp, st, pf_attn_rows(MP));   // bindings sized for the attention chunk
#endif
    if (steering_boot.loaded)
    {
        try { materialize_steering(m, steering_profile, st); }
        catch (const std::exception& e) { die(e.what()); }
        fprintf(stderr, "# rank %d: steering profile %s manifest %s tensor %s scope %s layers %zu forced %s\n", R,
                steering_profile.name.c_str(), steering::digest_hex(steering_profile.manifest_digest).c_str(),
                steering::digest_hex(steering_profile.tensor_digest).c_str(), steering::scope_name(steering_profile.scope),
                steering_profile.layers.size(), steering_boot.forced ? steering_force_s.c_str() : "off");
        if (R == 0) fprintf(stderr, "# steering manifest %s\n", steering_profile.manifest_json.c_str());
    }
    boot("layers bound");
    if (m.qs.on())                      // every rank, at the same point: its TCP bootstrap is a barrier
    {
        qs_fabric(fabg, m, 64, "G6_QA_SPLIT / G6_IDXQ_SPLIT");
        fabg_on = true;
        qs_attach(m, fabg.dev(), f.error);   // GPU-side gather wait timeouts land in the error word the step loop checks
    }
    // enqueue the gathers of a main step (every main step runs all of them), before its GPU work
    auto qs_enqueue = [&](unsigned long long step, int M) { if (fabg_on) fabg.enqueue_step(step, qs_sizes(m, M)); };
    // the same for a draft step (G6_MTP_SPLIT); D is built further down, before any step
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
    g6::Fabric fabs;
    const bool cp_pf = cp_on && MP;
    if (cp_pf)
    {
        g6::FabricConfig sc = cfg;
        sc.vcap.clear(); sc.vdst_k.clear();
        sc.bulk_bytes = 2 * cp_pf_pool_bytes(m);
        sc.bulk_flags = 4;
        sc.port = cfg.port + 48;
        sc.proxy_cpu = cfg.proxy_cpu > 3 ? cfg.proxy_cpu - 4 : -1;
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
    auto cp_pf_err = [&]() -> unsigned { return qp_err(); };   // the qp wire's error word (G6_PF_QP_ROLL)
#endif
    FILE* rlf = R == 0 ? route_log_open(m) : nullptr;     // G6_ROUTE_LOG=<file> (measurement, MTP main steps)
    if (s2mode && R == 0 && getenv("G6_S2_ROUNDLOG") && *getenv("G6_S2_ROUNDLOG") == '1') route_log_alloc(m);   // route copies for the "# s2_round" lines (s2/native.cuh)
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
        if (mtp_split) { D.d.qs.mode = 1; D.d.qs.iq = true; D.d.qs.eh = true; }   // G6_MTP_SPLIT: eh_proj, q_a / kv_a / idx, idx_q
        build_draft(D, m, find, cap, fd, dmalloc<u64>(1), st, MP ? pf_attn_rows(MP) : MX);
        if (D.d.qs.on())
        {
            qs_fabric(fabdg, D.d, 72, "G6_MTP_SPLIT");
            fabdg_on = true;
            qs_attach(D.d, fabdg.dev(), fd.error);   // the draft fabric's error word, which the draft step checks
        }
        build_draft_head_i4(D, dhead, st);
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
        // G6_DFL_PFCTX=0 (measurement): the drafter gets no prompt context from the prefill and starts at the last prompt token
        const bool pfctx = MP && !(getenv("G6_DFL_PFCTX") && atoi(getenv("G6_DFL_PFCTX")) == 0);
        dfl_build(DF, m, dfind, meta, fabf.dev(), dmalloc<u64>(1), dfl_sel < 0 ? meta.sel : dfl_sel == 1, st, pfctx);
        if (dfl_i4)
            dfl_head_i4(DF, D.d.dh_i4 ? D.d.dh_i4 : head_i4_records(m.head_t.lm_head, m.vn, dhead.ks, R, "G6_DFL_HEAD_I4", st),
                        dhead.tpc, dhead.ks);
        if (dfl_body_i4_on) dfl_body_i4(DF, st);
        dfl_attach(m, DF);              // before the target graphs are captured: they carry the residual taps
        if (pfctx) { P.dfl_tap = dfl_pf_tap_launch; P2.dfl_tap = dfl_pf_tap_launch; }
    }
    // S2: below R on the CP binding. Session views on the pool, main and draft; today's graphs below
    // are captured with s2on off (the S = 1 path), the S2 graphs later by s2/native.cuh with s2on on.
    int s2_below = 0;
    std::unique_ptr<RtTier> rtt;       // G6_RT_STORE (rt_rows.cuh); null = off
    int rt_rows = 0, rt_runs = 0;      // release: rows a session past R puts in one step, such sessions the slots cover
    if (s2mode && !G6_MODEL_FLASH)
    {
#if defined(G6_CP) && G6_CP
        if (!cp_on || !cp_R) die("S2: serving below R needs G6_CP on and G6_CP_R");
        s2_below = getenv("G6_S2_BELOW") ? atoi(getenv("G6_S2_BELOW")) : cp_R;
        if (s2_below < 64 || s2_below % 64 || s2_below > cp_R) die("S2: G6_S2_BELOW (default G6_CP_R) must be 64..G6_CP_R, a multiple of 64");
#else
        die("S2: serving needs the CP binding (a G6_CP build)");
#endif
        m.s2_below = D.d.s2_below = s2_below;
        // G6_RT_STORE: on the main model's bindings before s2_init and any prepare copy their arguments
        if (const char* rs = getenv("G6_RT_STORE"))
        {
            const int L = getenv("G6_RT_L") ? atoi(getenv("G6_RT_L")) : cp_R;
            int S = getenv("G6_RT_POOL") ? atoi(getenv("G6_RT_POOL")) : 4096;
            if (s2_below != cp_R) die("G6_RT_STORE needs G6_S2_BELOW = G6_CP_R (runs past R on the pool's page classes)");
            char host[64] = "host";
            gethostname(host, sizeof host - 1);
            const std::string fmt = kv_format.empty() ? g6kv::NAME : kv_format;
            const bool int4 = rt_fmt_int4();
            const std::string tag = "R" + std::to_string(cp_R) + "-L" + std::to_string(L) + "-nr" + std::to_string(g6nr::NR) + "-" + fmt +
                                    rt_fmt_name(int4) + "-" + rt_name_part(set_id);
            const std::string layout = "R=" + std::to_string(cp_R) + " L=" + std::to_string(L) + " nr=" + std::to_string(g6nr::NR) + " row=" +
                                       g6kv::NAME + rt_fmt_layout(int4) + " model=" + rt_name_part(set_id) + " pages=" +
                                       std::to_string(m.layers[0].ab.args.page_count) + " layers=" + std::to_string(m.nl);
            if (set_id.empty() && R == 0) fprintf(stderr, "# rank 0: G6_RT_STORE without --set-id: the lease names no model set\n");
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
                if (rt_win < need) die("G6_RT_LOCAL " + std::to_string(rt_win) + ": a prefill quantum of --prefill " + std::to_string(MP) +
                                         " rows needs at least " + std::to_string(need) + " window pages");
                if (rp + rt_win > place) die("G6_RT_LOCAL: the window does not fit the main model's pages past the prefix");
                // the slot pool: only sessions past R read released rows, each at most rt_rows rows a step (an MTP
                // verify, a DFlash2 verify, or a prompt chunk), and the window layout holds at most rt_runs of them
                // below place. Every row of theirs a step selects needs a slot: 2048 x rt_rows x rt_runs. A smaller
                // G6_RT_POOL lowers rt_runs, and the pool then admits fewer sessions past R at once
                rt_rows = std::min<int>(MX, std::max({mtp + 1, dfl ? dfl::K + 1 : 0, plan.chunk}));   // DFlash2: its widest verify
                rt_runs = std::min(s2sess, place / (rp + rt_win));
                if (getenv("G6_RT_POOL")) rt_runs = std::min(rt_runs, S / (2048 * rt_rows));
                else S = 2048 * rt_rows * rt_runs;
                if (rt_runs < 1) die("G6_RT_POOL " + std::to_string(S) + ": less than one session past R (2048 x " + std::to_string(rt_rows) + " rows)");
                if (R == 0)
                    fprintf(stderr, "# rank 0: G6_RT_LOCAL slot pool %d = 2048 x %d rows a session x %d sessions past R at once (place %d, %d prefix + "
                                    "%d window pages each)\n", S, rt_rows, rt_runs, place, rp, rt_win);
            }
            rtt = rt_build(m, rs, std::string("g6m4-") + rt_name_part(host) + "-" + std::to_string(s2epoch) + "-rt-" + tag + (rt_win ? "-rel" : ""),
                           layout + (rt_win ? " released" : ""), L, S, rt_win, ids, rt_win ? rt_rows : 1, false, int4,
                           getenv("G6_RT_CHECK") && atoi(getenv("G6_RT_CHECK")), st);
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
    u64 ls0 = 1;
    double pf_ms = 0;
    long pf_rows = 0;
    m.nc = nc; D.d.nc = nc;             // before any capture: the head kernels take it by value
    m.rs = plan.rs;                     // the target's verify steps only; the MTP layer's head stays as it is
    if (!pf_qo && !pf_qp)               // QO attention builds the decode RoPE table; without it absorb would take FP64 sincos
        if (int e = g6_attn_rope_init(cap, st)) die("g6_attn_rope_init failed (" + std::to_string(e) + ")");
    boot("prefill and draft built");
    for (int M : row_counts) prepare(m, M, st);
    for (int M : draft_rows) prepare(D.d, M, st);
#if G6_MODEL_FLASH
    if (mtp && MP) for (int M = 1; M <= MX; ++M) prepare(D.d, M, st);   // the draft prefill's MX-row chunks
#endif
    G6RT_CK(cudaStreamSynchronize(st));
    boot("layers prepared");
    // Offline main-step dumps, compatible with m4sim after merging the rank logits.
    // The callback only enqueues D2D copies, so it also works inside split graphs.
    const bool rd_on = R == 0 && getenv("G6_RESID_DUMP");
    const int rd_slots = 2 * NL + 1;
#if G6_MODEL_FLASH
    const int rd_hid = m.resid4 ? geom::HC * HID : HID;
    const float* rd_source = m.resid4 ? m.resid4 : m.resid;
#else
    const int rd_hid = HID;
    const float* rd_source = m.resid;
#endif
    float* rdump = rd_on ? dmalloc<float>((size_t)rd_slots * MX * rd_hid) : nullptr;
    std::function<void(int,int)> residual_copy;
    if (rd_on) residual_copy = [&](int slot, int M) {
        G6RT_CK(cudaMemcpyAsync(rdump + (size_t)slot * MX * rd_hid, rd_source,
                               (size_t)M * rd_hid * sizeof(float), cudaMemcpyDeviceToDevice, st));
    };
    FILE* rdf = rd_on ? fopen(getenv("G6_RESID_DUMP"), "wb") : nullptr;
    if (rd_on && !rdf) die("G6_RESID_DUMP: cannot open the file");
    FILE* ldump = nullptr;
    auto dump_write = [&](FILE* f, const void* data, size_t bytes) {
        if (fwrite(data, 1, bytes, f) != bytes) die("main-step dump: write failed");
    };
    if (const char* path = getenv("G6_LOGIT_DUMP")) {
        const std::string name = std::string(path) + ".rank" + std::to_string(R);
        ldump = fopen(name.c_str(), "wb");
        if (!ldump) die("G6_LOGIT_DUMP: cannot open " + name);
        // Versioned rank header: magic, NR, rank, vocab, first vocab row, vocab rows.
        // bind::vocab_slice is contiguous and unrotated, including uneven TP widths.
        dump_write(ldump, "G6LGR001", 8);
        const int32_t hdr[] = {g6nr::NR, R, geom::VOCAB, m.vlo, m.vn};
        dump_write(ldump, hdr, sizeof hdr);
    }
#if G6_MODEL_FLASH
    // MTP first draft passes only, as m4sim: rank 0 writes G6_DRAFT_IN_DUMP; every
    // rank writes its vocab slice to G6_DRAFT_LOGIT_DUMP.rank<R> in the G6_LOGIT_DUMP rank format (the step field is
    // the first-pass counter), so tools/cp/merge_logits.py gives m4sim's draft logit framing.
    FILE* didump = nullptr;
    FILE* dldump = nullptr;
    int dd_step = 0;
    if ((getenv("G6_DRAFT_IN_DUMP") || getenv("G6_DRAFT_LOGIT_DUMP")) && (!mtp || s2mode))
        die("G6_DRAFT_IN_DUMP / G6_DRAFT_LOGIT_DUMP need --mtp and no --serve");
    if (const char* path = getenv("G6_DRAFT_IN_DUMP"); path && R == 0) {
        didump = fopen(path, "wb");
        if (!didump) die("G6_DRAFT_IN_DUMP: cannot open the file");
    }
    if (const char* path = getenv("G6_DRAFT_LOGIT_DUMP")) {
        const std::string name = std::string(path) + ".rank" + std::to_string(R);
        dldump = fopen(name.c_str(), "wb");
        if (!dldump) die("G6_DRAFT_LOGIT_DUMP: cannot open " + name);
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
            if (fflush(didump)) die("G6_DRAFT_IN_DUMP: flush failed");
        }
        if (dldump) {
            dump_write(dldump, hdr, sizeof hdr);
            std::vector<float> lg((size_t)M * D.d.vn);
            G6RT_CK(cudaMemcpy(lg.data(), D.d.hlog, lg.size() * sizeof(float), cudaMemcpyDeviceToHost));
            dump_write(dldump, lg.data(), lg.size() * sizeof(float));
            if (fflush(dldump)) die("G6_DRAFT_LOGIT_DUMP: flush failed");
        }
    };
#endif
    const int dump_from = getenv("G6_LOGIT_DUMP_FROM") ? atoi(getenv("G6_LOGIT_DUMP_FROM")) : 0;
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
            if (fflush(rdf)) die("G6_RESID_DUMP: flush failed");
        }
        if (ldump) {
            const int32_t hdr[] = {s, M};
            dump_write(ldump, hdr, sizeof hdr);
            std::vector<float> h((size_t)M * m.vn);
            G6RT_CK(cudaMemcpy(h.data(), m.hlog, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
            dump_write(ldump, h.data(), h.size() * sizeof(float));
            if (fflush(ldump)) die("G6_LOGIT_DUMP: flush failed");
        }
    };
    cudaGraphExec_t exec[NBK][MX + 1] = {}, exec_d[NBK][2][MX + 1] = {};      // [context bucket][...]
    // DEC_GSPLIT: G6_DEC_SPLIT=ls (1..nl-1) captures the target step as two graphs, exec = step_in, embed, layers [0, ls) and
    // exec2 = the rest; launched back to back, the second graph's submission overlaps the first one's GPU time (0 = one graph)
    const int gsplit = getenv("G6_DEC_SPLIT") ? atoi(getenv("G6_DEC_SPLIT")) : 0;
    if (gsplit < 0 || gsplit >= NL || (gsplit && !graph)) die("G6_DEC_SPLIT must be 0..layers-1 and needs --graph");
    if (s2mode && !G6_MODEL_FLASH && gsplit != 3) die("serving needs G6_DEC_SPLIT=3 (the production split3 graphs)");
    cudaGraphExec_t exec2[NBK][MX + 1] = {};
    auto launch_main = [&](int k, int M) {
        G6RT_CK(cudaGraphLaunch(exec[k][M], st));
        if (gsplit) G6RT_CK(cudaGraphLaunch(exec2[k][M], st));
    };
    auto capture = [&](const std::function<void()>& body, cudaGraphExec_t* ex, const char* what, int M) {
        cudaGraph_t g;
        G6RT_CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        body();
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
        }
        if (mtp && D.d.nbk != m.nbk) die("draft and target context buckets differ");
        for (int k = 0; mtp && k < D.d.nbk; ++k)
        {
            D.d.cur = m.cur = k;
            for (int M : draft_rows) capture([&] { launch_draft_step(D, M, 0, false, st); }, &exec_d[k][0][M], "first draft ", M);
            if (mtp > 1) capture([&] { launch_draft_step(D, 1, 0, true, st); }, &exec_d[k][1][1], "chain draft ", 1);
        }
    }
    FILE* dfl_dump = dfl && R == 0 && getenv("G6_DFL_DUMP") ? fopen(getenv("G6_DFL_DUMP"), "wb") : nullptr;   // dflash.cuh dfl_dump_step
    cudaGraphExec_t exec_f[3] = {};     // --dflash: [0] the context part only, [1] the whole draft step, [2] a fill step
    if (graph && dfl)
    {
        capture([&] { dfl_step(DF, st, false); }, &exec_f[0], "dflash context ", dfl::B);
        capture([&] { dfl_step(DF, st, true); }, &exec_f[1], "dflash draft ", dfl::B);
        if (DF.pf_feat) capture([&] { dfl_fill_step(DF, st); }, &exec_f[2], "dflash fill ", dfl::B);
    }
    cudaStream_t st_par[2] = {};        // G6_DRAFT_PAR: the drafter's streams, [0] context only, [1] drafting
    if (draft_par)
    {
        int lo = 0, hi = 0;
        G6RT_CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        G6RT_CK(cudaStreamCreateWithFlags(&st_par[0], cudaStreamNonBlocking));
        if (draft_par == 2) G6RT_CK(cudaStreamCreateWithPriority(&st_par[1], cudaStreamNonBlocking, hi));
        else st_par[1] = st_par[0];
        if (R == 0) fprintf(stderr, "# rank %d: G6_DRAFT_PAR=%d: the DFlash2 drafter step runs under the MTP first form "
                            "(drafting stream priority %d, range %d..%d)\n", R, draft_par, draft_par == 2 ? hi : 0, lo, hi);
    }
    fprintf(stderr, "# rank %d: device tensors %.2f GB\n", R, (image ? img.bytes() : store.bytes) / 1e9);
    const bool pf_streams = two && getenv("G6_PF_STREAMS") && atoi(getenv("G6_PF_STREAMS")) == 2;   // pf_step2s
    if (pf_streams && pf_prof().on) die("G6_PF_STREAMS=2 cannot run with G6_PF_PROF (the marks synchronize one stream)");
    PfLanes lanes2;
    if (pf_streams)
    {
        pf_lanes_init(lanes2, m.nl);
        fprintf(stderr, "# rank %d: prefill lanes on two streams\n", R);
    }
    if (MP && pf_rs) fprintf(stderr, "# rank %d: prefill reduce-scatter form (5 collectives, slice %zu B)%s\n", R, pf_ss(fp),
                             pf_own >= 4 ? ", pair-owner QO attention (G6_PF_OWN=4)" : pf_own >= 3 ? ", row-owner QO attention (G6_PF_OWN=3)" : pf_own >= 2 ? ", row-owner projections + index (G6_PF_OWN=2)" : pf_own ? ", row-owner projections (G6_PF_OWN)" : "");
    {   // every proxy thread is running: keep all other threads of this process off their CPUs (G6_PROGRESS_ISOLATE=0 turns it off)
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
        sc.max_rows = std::min(MX, std::max(mtp + 1, plan.chunk));
        sc.max_sessions = 1; sc.mtp = mtp; sc.prefill_rows = MP; sc.two_lanes = two;
        bc.rank = R; bc.ranks = cfg.ips.size(); bc.head_ip = cfg.ips[0];
        bc.directory = s2dir; bc.port = s2port;
        bc.timeout = getenv("G6_S2_TIMEOUT") ? (unsigned)atoi(getenv("G6_S2_TIMEOUT")) : 120;
        bc.runtime_signature = "flash solo cap " + std::to_string(cap) + " layers " + layers_s +
            " mp " + std::to_string(MP) + " lanes " + std::to_string(lanes) + " mtp " + std::to_string(mtp) +
            " chunk " + std::to_string(plan.chunk) + " cands " + std::to_string(nc) + " set " + set_id;
        g6s2::FlashContext fc{m, D, FP, FP2, fab, fabd, fabp, fabp2, st, in, in_d, out, out_d,
            [&](int k, int rows) {
                g6s2::require(k >= 0 && k < m.nbk && rows > 0 && rows <= MX && exec[k][rows], "Flash main graph missing");
                launch_main(k, rows);
            },
            [&](int k, bool chain, int rows) {
                g6s2::require(k >= 0 && k < D.d.nbk && rows > 0 && rows <= MX && exec_d[k][chain][rows], "Flash draft graph missing");
                G6RT_CK(cudaGraphLaunch(exec_d[k][chain][rows], st));
            }};
        try { g6s2::serve_flash(fc, bc, plan.chunk); }
        catch (const std::exception& e) { fprintf(stderr, "g6m4: rank %d Flash serve failed: %s\n", R, e.what()); return 3; }
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
        // G6_S2_DECODE_PF=<rows>: while a session decodes, prefill quanta of at most this many rows (scheduler.h
        // Config::decode_prefill_rows); unset or 0 = off
        if (const char* e = getenv("G6_S2_DECODE_PF")) {
            const int v = atoi(e);
            if (v < 0 || v % 64 || v > MP) die("S2: G6_S2_DECODE_PF must be 0 or a multiple of 64 no larger than --prefill");
            sc.decode_prefill_rows = (uint32_t)v;
        }
        // G6_S2_DECODE_ROUNDS=<K>: while a prefill waits and sessions decode, K decode rounds per prefill quantum
        // (scheduler.h Config::decode_rounds); unset, 0 or 1 = off
        if (const char* e = getenv("G6_S2_DECODE_ROUNDS")) {
            const int v = atoi(e);
            if (v < 0 || v > (int)g6s2::MAX_DECODE_ROUNDS) die("S2: G6_S2_DECODE_ROUNDS must be 0..64");
            sc.decode_rounds = (uint32_t)v;
        }
        sc.steering = steering_boot;
        if (rtt && rtt->local) sc.window_pages = (uint32_t)rtt->local;   // a window run lists its worst case and the window
        // G6_CP_STEP_ROWS (sized the CP fabric above) and G6_CP_STEP_SEG=1: Driver::step's rounds past R (cp_rows.h)
        {
            std::string ce;
            const int seg = env_choice("G6_CP_STEP_SEG", 1, 0, &ce);
            if (seg < 0) die(ce);
            if (seg && !cp_step) die("G6_CP_STEP_SEG=1 needs G6_CP_STEP_ROWS");
            sc.cp_step_rows = (uint32_t)cp_step;
            sc.cp_seg = seg == 1;
            sc.cp_after = g6s2::cp_after(m.bk, m.nbk, m.cp.on ? m.cp.R : -1);
            if (cp_step && R == 0)
                fprintf(stderr, "# s2 G6_CP_STEP_ROWS=%d: a step past R (context > %llu) carries at most %d rows = %d sessions of %d%s; "
                        "the CP fabric is sized for it\n", cp_step, (unsigned long long)sc.cp_after, cp_step, cp_step / (mtp + 1), mtp + 1,
                        sc.cp_seg ? ", and no session below R (G6_CP_STEP_SEG)" : "");
        }
        const int place = std::min(m.s2->pool_pages, D.d.s2->pool_pages);
        g6s2::RunPool pool((uint32_t)(cap / 64), (uint32_t)place, (uint32_t)s2_below);
        // the pool's R is the CP map's: a run past R takes its own pages — the prefix and its shard below place, the
        // drafter's tail copy above it — instead of the whole pool, and retained runs keep their pages through a long
        // turn. The stores' tail blocks (per_slot) come with --kv-store below.
        if (m.cp.on && s2_below == m.cp.R) pool.tail((uint32_t)g6nr::NR, 0);
        if (rtt && rtt->local)
        {
            if (!pool.split()) die("G6_RT_LOCAL needs the pool's page classes (G6_S2_BELOW = G6_CP_R)");
            // a tail waits until its records fit every rank's lease, a session past R until the slots cover it
            pool.window((uint32_t)rtt->local, (uint32_t)rtt->V, (uint32_t)rt_runs);
            if (R == 0) fprintf(stderr, "# rank 0: G6_RT_LOCAL: a run past R holds %d prefix + %d window pages below place %d, its whole tail above "
                                        "(%d pages: the longest tail one run can have, %d rows); %d record ids per rank\n", cp_R / 64, rtt->local, place,
                                cap / 64 - place, (cap / 64 - place) * 64, rtt->V);
        }
        int reuse_grid = 0;
        // the offline loop's rules (prefix_reuse.h), but on by default: the grid rule (exact hits; G6_REUSE_ALIGN=0 drops it)
        // and NOEND (G6_REUSE_NOEND=0 restores the prompt >= written_end rule, which refuses replaying a prompt as it is)
        if (reuse_prefix)
        {
            std::string err;
            const int st_ = env_choice("G6_REUSE_STRICT", 1, 0, &err), ne_ = env_choice("G6_REUSE_NOEND", 1, 1, &err),
                      al_ = env_choice("G6_REUSE_ALIGN", 1, 1, &err);
            if (st_ < 0 || ne_ < 0 || al_ < 0) die(err);
            if (al_ == 1 && !MP) die("G6_REUSE_ALIGN=1 needs tensor prefill (--prefill N)");
            PrefixReuse rules;
            rules.strict = st_ == 1; rules.noend = ne_ == 1; rules.grid = al_ == 1 ? MP : 0;
            pool.keep(rules);
            reuse_grid = rules.grid;
            if (env_choice("G6_POOL_LOG", 1, 0, &err) == 1) pool.log(true);   // one stderr line per pool claim: first page, pages, evicted retained runs
            if (R == 0) fprintf(stderr, "# s2 reuse on: retired runs kept until a claim overlaps them; grid %d rows\n", rules.grid);
        }
        // --kv-store: block stores below the pool, in flag order, the same number of blocks on every rank. An SSD store lives in
        // PATH/r<rank>; its system prefixes and the counts survive restarts (their index needs --set-id: the blocks belong to
        // one model). A store's keep=, else --kv-keep, sets its keep rule and policy (kvman-rs stash.rs Keep, Policy)
        std::unique_ptr<g6s2::KvTiers> kvt;
        bool kv_index = false;
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
                    const uint64_t slots = kvt->add(spec, name + "-t" + std::to_string(i), t.bytes);
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
                    pool.add_tier(slots, keep == "system" ? 1 : 2, index, compat);   // shared prefixes, or those and evicted runs
                    pool.policy((uint32_t)i, 0, keep == "sessions" ? 1 : 0, kvt->blocks().slot_bytes());
                    kv_index = kv_index || !index.empty();
                    fprintf(stderr, "# rank %d: s2 store %zu: %s, %llu blocks of %zu B, keep %s%s\n", R, i, kvt->describe(i).c_str(),
                            (unsigned long long)slots, kvt->blocks().slot_bytes(), keep.c_str(), index.empty() ? "" : ", indexed");
                }
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
        // G6_KV_DRAIN_DELAY=<rank>:<ms>[:<every>], TEST ONLY: that rank sleeps <ms> inside every <every>th tier drain that
        // copies anything (default every one), as a drain stalled on its store would. Reproduces the rank skew the
        // barrier after a drain (NativeHooks::drained) has to absorb.
        int kv_delay_rank = -1, kv_delay_ms = 0, kv_delay_every = 1;
        long kv_delay_n = 0, kv_drain_n = 0;   // kv_drain_n: this rank's drains that copied anything (the same count on every rank)
        if (const char* e = getenv("G6_KV_DRAIN_DELAY"))
            if (sscanf(e, "%d:%d:%d", &kv_delay_rank, &kv_delay_ms, &kv_delay_every) < 2 || kv_delay_ms < 0 || kv_delay_every < 1)
                die("G6_KV_DRAIN_DELAY=<rank>:<ms>[:<every>]");
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
                    fprintf(stderr, "# rank %d: G6_KV_DRAIN_DELAY: sleeping %d ms in tier drain %ld at %lld ms\n", R, kv_delay_ms, kv_delay_n,
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
        // G6_S2_BIND_DELAY=<rank>:<ms>: that rank sleeps after its bind (after the drain), as a drain that slow would. Shows
        // what a bind skew does to the next quantum on the other ranks (nothing else changes), with or without tiers.
        int bind_delay_rank = -1, bind_delay_ms = 0;
        if (const char* e = getenv("G6_S2_BIND_DELAY"))
            if (sscanf(e, "%d:%d", &bind_delay_rank, &bind_delay_ms) != 2 || bind_delay_ms < 0) die("G6_S2_BIND_DELAY=<rank>:<ms>");
        m.s2on = D.d.s2on = true;        // no session: the adapter starts in S2 mode (solo -1)
        G6RT_CK(cudaStreamSynchronize(st));
        // DFlash2: Dfl::pf_feat holds one request's prefill taps, the last WIN prompt positions of the session whose
        // prefill started last (dfl_owner; -1 = none, or a prefix-reuse hit that skipped part of the window, unless
        // G6_DFL_REUSE: then the rows from dfl_from)
        static_assert(s2::DFL_B == dfl::B, "s2_batch.h DFL_B follows dfl::B");
        static_assert(s2::DFL_WIN == dfl::WIN && s2::DFL_RING == dfl::RING, "s2_batch.h DFL_WIN / DFL_RING follow dfl::WIN / RING");
        int dfl_owner = -1;
        int dfl_from = 0;                // the first window position the owner's prefill taps (G6_DFL_REUSE: past a hit's rows)
        uint64_t dfl_req = UINT64_MAX;
        std::array<uint32_t, g6s2::MAX_SESSIONS> slot_prompt{};
        std::array<bool, g6s2::MAX_SESSIONS> slot_whole{};   // G6_DFL_REUSE: a hit holds every prompt row but the last (no prefill)
        u64 s2_dstep = 0;                // the drafter fabric's steps
        cudaStream_t dfl_par_st = st;    // G6_DRAFT_PAR: the stream of the drafter step in flight
        const bool dfl_tl = R == 0 && getenv("G6_S2_ROUNDLOG") && *getenv("G6_S2_ROUNDLOG") == '1';
        std::chrono::steady_clock::time_point dfl_t0;
        g6s2::NativeContext nx{m, D, P, P2, DQ, fab, fabd, fabp, fabp2, st, pf_streams ? &lanes2 : nullptr, in, in_d, out, out_d, MP, gsplit, two,
            [&](const StepIn& si) {
                std::vector<uint32_t> sizes(NC, (uint32_t)(si.M * HID * 2));
                sizes[2 * NL] = head_size(si, nc);
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
                if (ex)
                {
                    if (pool.split())   // the run's own pages: its CP map over its list
                    {
                        const std::vector<int> run = pool.list(sl.run);
                        const int most = std::max(m.s2->pool_pages, D.d.s2->pool_pages);
                        s2_solo_cp_pages(m, run, most);
                        s2_solo_cp_pages(D.d, run, most);
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
        // G6_RT_STORE: slot q's table as the attention reads it: the layers' own tables while it runs alone (solo), its
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
            if (R == 0) fprintf(stderr, "# rank 0: G6_RT_STORE %s slot %d: %llu rows spilled, %llu dropped pages, %llu fetched, %llu read locally (no slot)\n",
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
                    fprintf(stderr, "# rank 0: G6_RT_STORE first main step: %d sessions, slot %d committed %d (%s), L %d\n", h.ns, h.sess[0],
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
                        fprintf(stderr, "# rank 0: G6_RT_STORE slot %d first spill (%s, committed %d, exclusive %d split %d): %d pages past L %d: %d spilled, "
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
        // G6_RT_LOCAL: slot q's tables with its tail entries (s2_cp_map reads them through S2Model::rt_tail): its slot
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
                const std::vector<int> t = s2_cp_map(s, s.pool[0]->args, s.run[q], m.s2_below / 64, (size_t)s.table_cap);
                G6RT_CK(g6h2d::copy(s.table[q], t.data(), t.size() * 4));
                if (rt_layers == q) s2_solo_cp_pages(m, s.run[q], most);
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
                // what the last quantum staged from the store: every quantum with G6_RT_CHECK, else every 8th that staged
                static uint64_t pf_n = 0, pf_p0 = 0, pf_b0 = 0;
                static double pf_t0 = 0;
                if (R == 0 && rtt->pf_pages > pf_p0 && (rtt->hold || ++pf_n % 8 == 1))
                    fprintf(stderr, "# rank 0: G6_RT_LOCAL prefill staging since the last line: %llu released pages a layer summed, %.3f GB read, "
                                    "%.1f ms on the host; next quantum rows %u..\n", (unsigned long long)(rtt->pf_pages - pf_p0),
                            (rtt->pf_bytes - pf_b0) / 1e9, rtt->pf_ms - pf_t0, first);
                if (R == 0 && rtt->sel_layers)
                {
                    fprintf(stderr, "# rank 0: G6_RT_SELSTAT: %llu sampled layer fills selected %llu of %llu released rows (%.1f%%); next quantum rows %u..\n",
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
            static const bool selstat = getenv("G6_RT_SELSTAT") && atoi(getenv("G6_RT_SELSTAT")) == 1;
            m.cp.rt_stage = [&](int i, int start, char* pool_k, size_t ixoff, cudaStream_t s) {
                if (rt_pf < 0) return;
                rt_pf_stage(*rtt, rt_pf, i, start, {{pool_k, ixoff}}, s);
                // G6_RT_SELSTAT: at each full-index layer's fill, the lanes' latest lists (the previous full-index
                // layer's: G6_PF_OWN=2/4 keep them in idx_all) against the released rows below the quantum
                if (selstat && m.layers[i].indexer && P.idx_all)
                    rt_selstat(*rtt, {{P.idx_all, rt_ma}, {two ? P2.idx_all : nullptr, rt_mb}}, start / 64 * 64, s);
            };
        }
        try
        {
            g6s2::NativeDispatch nd(nx, sc);
            g6s2::NativeHooks h;
            h.pool.free_pages = [&] { return pool.free_pages(); };
            h.pool.room = [&] { return pool.room(); };
            h.pool.reserve = [&](uint64_t rows, uint64_t worst) { return pool.reserve(rows, worst); };
            h.pool.digest = [&] { return pool.digest(); };
            h.pool.release = [&](const g6s2::Run& r) { pool.release(r); };
            h.pool.fits = [&](uint64_t rows, uint64_t worst) { return pool.fits(rows, worst); };
            if (rtt && rtt->local)
                h.pool.refusal = [&](uint64_t worst) -> std::string {
                    const auto v = pool.split_view(worst);
                    if (!v) return "";
                    const auto [lo, hi, lofree, hifree] = *v;
                    std::string why = "it takes " + std::to_string(lo) + " pages below place (" + std::to_string(lofree) + " free) and its tail of " +
                                      std::to_string(hi) + " above (" + std::to_string(hifree) + " free)";
                    if (hi > (uint64_t)(cap / 64 - std::min(m.s2->pool_pages, D.d.s2->pool_pages))) why += ": the tail is longer than the drafter class, it never fits";
                    if ((int)pool.window_runs() >= rt_runs)
                        why += "; the slot pool covers " + std::to_string(rt_runs) + " session(s) past R at once (G6_RT_POOL), all in use";
                    if (const auto iv = pool.ids_view(worst))
                    {
                        why += "; its tail needs " + std::to_string((*iv)[0]) + " records a rank, " + std::to_string((*iv)[1]) + " of the lease's " +
                               std::to_string((*iv)[2]) + " are free";
                        if ((*iv)[0] > (*iv)[2]) why += ": more than the lease holds (KV_ROWS size=), it never fits";
                    }
                    return why;
                };
            h.pool.stats = [&] { return pool.stats(); };
            h.pool.runs = [&] { return pool.runs(); };
            // G6_S2_GROW=0: every run holds its worst case (prompt + max_tokens + guard) from admission, as before growth
            {
                std::string err;
                const int grow = env_choice("G6_S2_GROW", 1, 1, &err);
                if (grow < 0) die(err);
                if (grow) h.pool.grow = [&](const g6s2::Run& r, uint64_t rows) { return pool.grow(r, rows); };
            }
            if (pool.keeping())
            {
                h.pool.retain = [&](const g6s2::Run& r, const PrefixReuse& s, const CacheNamespace& ns) { pool.retain(r, s, ns); };
                h.pool.match = [&](const std::vector<int>& p, uint64_t rows, uint64_t worst, int limit, const CacheNamespace& ns) { return pool.match(p, rows, worst, limit, ns); };
                h.pool.take = [&](const g6s2::PoolHit& x, uint64_t rows, uint64_t worst) { return pool.take(x, rows, worst); };
            }
            if (kvt)
            {
                h.pool.tiers = [&] { return pool.tiers(); };
                h.pool.count = [&](const std::vector<int>& p, int64_t span, const CacheNamespace& ns) { return pool.count(p, span, ns); };
                h.pool.retain = [&](const g6s2::Run& r, const PrefixReuse& s, const CacheNamespace& ns) { pool.retain(r, s, ns); kv_drain(); };   // a promoted prefix
            }
            h.bind = [&](const g6s2::Grant& g) {     // at the admission fence, every rank, main and draft
                // the store copies the grant queued: evicted runs out before anything writes their pages, a stored run in
                kv_drain();
                if (R == bind_delay_rank && bind_delay_ms)
                {
                    fprintf(stderr, "# rank %d: G6_S2_BIND_DELAY: sleeping %d ms after the bind of request %llu at %lld ms\n", R, bind_delay_ms,
                            (unsigned long long)g.req.id, (long long)g6s2::s2_wall_ms());
                    std::this_thread::sleep_for(std::chrono::milliseconds(bind_delay_ms));
                }
                const int q = (int)g.key.slot;
                if (rtt && !rtt->local) { rt_drop_slot(*rtt, q, st); rt_drop(*rtt, pool.list(g.run), st); rt_told[q] = false; }   // a new run's pages are local
                // release: a split run's tail starts unmapped, cycling through its window (list[R/64 ..], kvman-rs Pool::window)
                if (rtt && rtt->local)
                {
                    if (rt_layers == q) rt_layers = -1;
                    rt_gen[q] = g.key.generation;
                    // a hit on a retained window run takes its window pages, and its records come with them: from its
                    // parked entry, or from its slot when its retirement is at this boundary and not bound yet (the
                    // slot's own included), so it is found before anything ends that slot
                    const bool ex = pool.exclusive_run(g.run);
                    const std::vector<int> l = ex ? pool.list(g.run) : std::vector<int>{};
                    const int rp = cp_R / 64;
                    const std::vector<int> win = ex ? std::vector<int>(l.begin() + rp, l.begin() + rp + rtt->local) : std::vector<int>{};
                    const int pk = ex && g.reuse ? rt_parked(*rtt, win) : -1;
                    if (pk != q) rt_end(*rtt, q, st);
                    // the runs the pool let go (evicted, or adopted only in part below R) free their records first. The
                    // head's take already moved this grant's run out of the kept list, so its entry stays.
                    rt_gc(*rtt, [&](const std::vector<int>& kl) { return (pk >= 0 && kl == rtt->seq[pk].list) || pool.is_kept(kl); }, st);
                    if (ex)
                    {
                        if (pk >= 0)
                        {
                            rt_adopt(m, *rtt, pk, q, l, (int)g.reuse, st);
                            if (R == 0) fprintf(stderr, "# rank 0: G6_RT_LOCAL slot %d adopts a retained run's records at row %u\n", q, g.reuse);
                        }
                        else if (g.reuse > cp_R)   // every rank holds the same entries, so they all stop here alike
                            die("G6_RT_LOCAL: a hit past R on a run whose records no slot or parked entry holds");
                        else rt_begin(*rtt, q, win, l, st);
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
                        if (pool.split()) s2_set_exclusive(*x, q, run, most);   // its own list; the slot view stays unused while solo
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
            };
            // a run grew between rounds (every rank, same order): the new pages, cleared as a new run's are, into slot q's
            // table, and into the layers' own tables when q runs alone on today's path
            h.grown = [&](unsigned q, const g6s2::Run& r) {
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
            };
            h.retire = [&](const g6s2::Slot& old) {
                // a grant at the same boundary may have bound the slot again (or adopted its records) first: then the
                // slot is that session's, and the retirement leaves it alone
                if (rtt && rtt->local && rt_gen[old.key.slot] != old.key.generation)
                {
                    if (R == 0) fprintf(stderr, "# rank 0: G6_RT_LOCAL retire of slot %u after its next grant: nothing to do\n", old.key.slot);
                }
                else if (rtt && rtt->local)   // a run with released pages parks its records
                {
                    const int q = (int)old.key.slot;
                    const RtTier::Seq& sq = rtt->seq[q];
                    if (R == 0 && sq.on)
                        fprintf(stderr, "# rank 0: G6_RT_LOCAL retire of slot %d: %llu pages released, at most %d of %d window pages in use; all sessions: "
                                        "%llu released, peak %d local tail pages; %llu rows fetched, %zu of %d record ids free\n", q,
                                (unsigned long long)sq.released, sq.peak, sq.window, (unsigned long long)rtt->released, rtt->peak,
                                (unsigned long long)rt_fetched(*rtt), rtt->vfree.size(), rtt->V);
                    if (rtt->hold)
                        fprintf(stderr, "# rank %d: G6_RT_CHECK: %llu released group pages read back, %llu differed\n", R,
                                (unsigned long long)rtt->readback, (unsigned long long)rtt->readback_bad);
                    // records park for the conversation's next turn whether or not the pool kept the run: on the head a
                    // grant at this boundary has already taken it out of the kept list, on the workers it has not. The
                    // next bind's rt_gc frees them if no grant adopts them.
                    if (sq.on)
                    {
                        if (R == 0) fprintf(stderr, "# rank 0: G6_RT_LOCAL slot %d parks its records (kept by the pool: %d)\n", q, (int)pool.is_kept(m.s2->run[q]));
                        rt_park(*rtt, q);
                    }
                    if (rtt->hold && rtt->stage_ck.rows)
                    {
                        rt_check_print(("staged prefill rows, rank " + std::to_string(R)).c_str(), rtt->stage_ck);
                        fprintf(stderr, "# rank %d: G6_RT_CHECK: %llu staged pages under 10 dB or with a rope byte off\n", R, (unsigned long long)rtt->stage_bad);
                    }
                    rt_end(*rtt, q, st);
                    if (rt_layers == q) rt_layers = -1;
                }
                else if (rtt) { rt_drop_slot(*rtt, (int)old.key.slot, st); rt_log("retire of", (int)old.key.slot); }   // retained runs stay local
                nd.retire(old);
            };
            h.fence = [&] { nd.synchronize(); };
            if (kvt) h.drained = [&] { return kv_drained.exchange(false); };
            h.dispatch = nd.hooks();
            if (dfl)
            {
                h.dfl_kv = dfl_k; h.dfl_spec = spec;
                // drafter choice: drafting ms per round at 2200 MHz. DFlash2: the drafter step and the MTP first form;
                // MTP: the first form, the drafter's context-only step, a chain step per draft after the first
                h.dsel.on = s2::dsel_env();
                // tuning (every rank must see the same values; the launcher's env reaches all of them)
                if (const char* e = getenv("G6_S2_DSEL_MARGIN")) h.dsel.margin = atof(e);
                if (const char* e = getenv("G6_S2_DSEL_DWELL")) h.dsel.dwell = std::max(1, atoi(e));
                if (const char* e = getenv("G6_S2_DSEL_PROBE")) h.dsel.probe = std::max(1, atoi(e));
                h.dsel.d.draft_round = 5.6 + 2.5;
                h.dsel.m.draft_round = 2.5 + 3.0;
                h.dsel.m.draft_each = 2.1;
                // sweeps (measurement only; every rank must see the same values, as above): G6_SPEC_TABLE = verify ms for
                // 1..8 rows, G6_SPEC_DRAFT = DFlash2 round, MTP round, MTP each (ms)
                if (const char* e = getenv("G6_SPEC_TABLE"))
                {
                    double t[SpecLen::KX + 1];
                    if (!parse_spec_costs(e, t, SpecLen::KX + 1))
                        die("G6_SPEC_TABLE: want 8 comma-separated finite positive verify times in ms");
                    for (int r = 1; r <= SpecLen::KX + 1; ++r) h.dfl_spec.table[r] = h.dsel.d.table[r] = h.dsel.m.table[r] = t[r - 1];
                    if (R == 0) fprintf(stderr, "# spec: verify table from G6_SPEC_TABLE: %s\n", e);
                }
                if (const char* e = getenv("G6_SPEC_DRAFT"))
                {
                    double d[3];
                    if (!parse_spec_costs(e, d, 3, true))
                        die("G6_SPEC_DRAFT: want 3 comma-separated finite nonnegative drafting times in ms");
                    h.dsel.d.draft_round = d[0]; h.dsel.m.draft_round = d[1]; h.dsel.m.draft_each = d[2];
                    if (R == 0) fprintf(stderr, "# spec: drafting ms from G6_SPEC_DRAFT: %s\n", e);
                }
                h.dsel.model = dsel_model;    // the table replaces the drafter choice's verify table and drafting ms above (Batch::dsel)
                h.dsel.tab = dsel_tab;
                if (R == 0 && dsel_model)
                {
                    fprintf(stderr, "# s2 drafter choice cost model %d: table %s\n", dsel_model, dsel_tab.str().c_str());
                    if (getenv("G6_SPEC_TABLE") || getenv("G6_SPEC_DRAFT"))
                        fprintf(stderr, "# note: the drafter choice reads the table's v1..v8, d, m, e, not G6_SPEC_TABLE / G6_SPEC_DRAFT; "
                                        "G6_SPEC_TABLE still prices the DFlash2 rounds' own draft count\n");
                }
                if (R == 0 && h.dsel.on) fprintf(stderr, "# s2 drafter choice on: DFlash2 or MTP per round (dwell %d, margin %.2f, probe %d)\n",
                                                    h.dsel.dwell, h.dsel.margin, h.dsel.probe);
                h.kstop = kstop;
                if (R == 0 && kstop.mode)
                    fprintf(stderr, "# s2 MTP confidence stop: mode %d, solo MTP rounds verify 1..%d drafts (%s)%s\n", kstop.mode, kstop.kmax,
                            kstop.mode == 1 ? ("tau " + std::to_string(kstop.tau[1]) + "," + std::to_string(kstop.tau[2]) + "," +
                                               std::to_string(kstop.tau[3])).c_str()
                                            : ("MTP estimates, seed3 " + std::to_string(kstop.seed3)).c_str(),
                            h.dsel.on ? "" : dfl_reuse ? "; the drafter choice is off: G6_DFL_REUSE warm-up rounds only" : "; the drafter choice is off, so no MTP rounds");
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
                if (R == 0 && dfl_reuse) fprintf(stderr, "# s2 dflash reuse %d: reuse hits keep the drafter%s, warm-up %d context rows\n", dfl_reuse,
                                                 dfl_reuse > 1 ? ", a session alone on MTP takes it" : "", dfl_reuse_min);
                // a 1-token prompt has no prefill: the drafter starts from its prompt step (dfl_lo = first, as plain)
                // G6_DFL_REUSE: also a hit into the window (the fill starts past its rows) and a hit with no prefill at all
                h.dispatch.dfl_ok = [&](int q) { return dfl_owner == q || slot_prompt[q] == 1 || slot_whole[q]; };
                h.dispatch.dfl_start = [&](int q) {    // the offline loop's fill steps: Dfl::pf_feat -> the ring
                    if (dfl_owner != q || DF.pf_hi <= DF.pf_lo) return (int)slot_prompt[q] - 1;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int c0 = dfl_from; c0 < DF.pf_hi; c0 += dfl::B)
                    {
                        DflIn di;
                        memset(&di, 0, sizeof di);
                        di.step = ++s2_dstep;
                        di.ctx_rows = std::min(dfl::B, DF.pf_hi - c0); di.ctx_pos0 = c0; di.q0 = c0 + di.ctx_rows; di.lo = DF.pf_lo;
                        *in_f = di;
                        fabf.enqueue_step(di.step, dfl_fill_sizes());
                        G6RT_CK(cudaGraphLaunch(exec_f[2], st));
                        G6RT_CK(cudaStreamSynchronize(st));
                        g6s2::require(!*fabf.dev().error, "S2 DFlash2 fabric error (prompt context)");
                    }
                    g6s2::s2_tl().cut();
                    if (R == 0) fprintf(stderr, "# s2 dflash slot %d prompt context %d..%d: %.1f ms\n", q, dfl_from, DF.pf_hi - 1,
                                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                    return dfl_from;
                };
                h.dispatch.dfl = [&](const s2::DflReq& r, const g6s2::Sampling& smp) {
                    // G6_DRAFT_PAR: phase 1 launches on the drafter's stream and returns, phase 2 waits for it (s2::DflReq).
                    // The target step before phase 1 was fenced on st, and phase 2 comes before the next one.
                    // G6_S2_TIMELINE: phase 0 is a step in the target stream's sequence; phases 1 + 2 are timed apart (par_*)
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
                        fabf.enqueue_step(di.step, dfl_sizes(DF, di.query));
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
                    if (dfl_tl)   // G6_S2_ROUNDLOG: the drafter step's launch to its wait (G6_DRAFT_PAR: the first form and chain within)
                        fprintf(stderr, "# s2_dfl step %llu query %d par %d ms %.3f\n", (unsigned long long)s2_dstep, (int)r.query, r.phase ? 1 : 0,
                                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dfl_t0).count());
                    g6s2::require(!*fabf.dev().error, "S2 DFlash2 fabric error");
                    s2::DflDrafts o;
                    for (int i = 0; i < dfl::B; ++i) { o.tok[i] = out_f->tok[i]; o.lp[i] = out_f->lp[i]; }
                    return o;
                };
            }
            g6s2::BootstrapConfig bc;
            bc.scheduler = sc;
            bc.rank = (unsigned)R;
            bc.ranks = (unsigned)cfg.ips.size();
            bc.timeout = getenv("G6_S2_TIMEOUT") ? (unsigned)atoi(getenv("G6_S2_TIMEOUT")) : 120;
            bc.head_ip = cfg.ips[0];
            bc.directory = s2dir;
            bc.port = (uint16_t)s2port;
            bc.runtime_signature = "g6m4 s2 below " + std::to_string(s2_below) + " cap " + std::to_string(cap) + " layers " + layers_s +
                " mp " + std::to_string(MP) + " lanes " + std::to_string(two ? 2 : 1) + " mtp " + std::to_string(mtp) + " chunk " +
                std::to_string(plan.chunk) + " cands " + std::to_string(nc) + " split 3 own " + std::to_string(pf_own) +
                " qpctx " + std::to_string(pf_qp ? P.qp_ctx : 0) + " qpmin " + std::to_string(pf_qp ? pf_qp_min() : 0) +
                (set_id.empty() ? "" : " set " + set_id) +
                (dfl_reuse ? " dflreuse " + std::to_string(dfl_reuse) + "/" + std::to_string(dfl_reuse_min) : "") +
                (kstop.mode ? " kstop " + std::to_string(kstop.mode) + " kmax " + std::to_string(kstop.kmax) + " tau " + std::to_string(kstop.tau[1]) + "," +
                              std::to_string(kstop.tau[2]) + "," + std::to_string(kstop.tau[3]) + " seed3 " + std::to_string(kstop.seed3) : "") +
                (draft_par ? " draftpar " + std::to_string(draft_par) : "") +
                (qs_mode ? " qasplit " + std::to_string(qs_mode) : "") +
                (iq_split ? " idxqsplit 1" : "") + (mtp_split ? " mtpsplit 1" : "") +
                (dhead.on ? " dheadi4 " + std::to_string(dhead.tpc) + "x" + std::to_string(dhead.ks) : "") +
                (dfl_i4 ? " dflheadi4 " + std::to_string(dhead.tpc) + "x" + std::to_string(dhead.ks) : "") +
                (dfl_body_i4_on ? " dflbodyi4 1" : "") +
                (dsel_model ? " dselmodel " + std::to_string(dsel_model) + " " + dsel_tab.str() : "") + s2::kstopb_sig(kstop.b) +
                (cp_step ? " cpsr " + std::to_string(cp_step) + (sc.cp_seg ? " seg" : "") : "");

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
                nd.capture_all(sc.limit(), mtp, plan.chunk, kstop.b, cp_step);
                m.dfl_tap = tap;
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
            links.shutdown();
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
        std::string err;
        const int st_ = env_choice("G6_REUSE_STRICT", 1, 0, &err), ne_ = env_choice("G6_REUSE_NOEND", 1, 0, &err),
                  po_ = env_choice("G6_REUSE_POISON", 1, 0, &err);
        if (st_ < 0 || ne_ < 0 || po_ < 0) die(err);
        reuse.strict = st_ == 1; reuse.noend = ne_ == 1; reuse_poison_on = po_ == 1;
        const int al_ = env_choice("G6_REUSE_ALIGN", 1, 0, &err);   // retain only whole cold prefill grid cells (prefix_reuse.h)
        if (al_ < 0) die(err);
        if (al_ == 1 && !MP) die("G6_REUSE_ALIGN=1 needs tensor prefill (--prefill N)");
        reuse.grid = al_ == 1 ? MP : 0;   // one sub-batch (the lane pairing does not change a row)
        if (R == 0 && reuse.grid) fprintf(stderr, "# reuse grid %d rows (--prefill %d, lanes %d)\n", reuse.grid, MP, two ? 2 : 1);
    }
    HDump hd = hdump_env();             // reuse diagnostics (reuse_hdump.cuh): default off
    {
        std::string err;
        const int zs_ = env_choice("G6_REUSE_ZSCRATCH", 1, 0, &err);
        if (zs_ < 0) die(err);
        if (zs_ == 1 && !reuse_prefix) die("G6_REUSE_ZSCRATCH=1 needs --reuse-prefix");
        hd.zscratch = zs_ == 1;
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
        if (pf_qp)                      // OWN=4 iff prompt + gen fits the QO table (G6_PF_QP_CTX), else OWN=2 for this request
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
                pf_enqueue_flash(fabp, ls0, NL, M);
                const int MB = two && p.first - a > MP ? std::min(MP, p.first - a - MP) : 0;
                if (MB)
                {
                    std::vector<int> posB(MB);
                    for (int r = 0; r < MB; ++r) posB[r] = a + M + r;
                    pf_inputs_flash(FP2, MB, p.prompt.data() + a + M, posB.data());
                    pf_enqueue_flash(fabp2, ls0b, NL, MB);
                }
                // G6_RESID_DUMP: one record per sub-batch (step field 0), every slot copied out as it is reached
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
                if (MB) pf_step2_flash(m, FP, FP2, M, MB, a + M, a + M + MB, ls0, ls0b, st);
                else pf_step_flash(m, FP, M, a + M, ls0, st, pf_copy);
                G6RT_CK(cudaStreamSynchronize(st));
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ls0 += NL;
                if (MB) ls0b += NL;
                if (rdf && fflush(rdf)) die("G6_RESID_DUMP: flush failed");
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
                if (ldump && getenv("G6_PF_HEAD") && atoi(getenv("G6_PF_HEAD")) == 1)   // logit dump rows of the prefill (MX-row records, step 0: merge_logits.py needs >= 0)
                    pf_head_flash(m, FP, M, st, [&](int r) {
                        const int32_t hdr[] = {0, r};
                        dump_write(ldump, hdr, sizeof hdr);
                        std::vector<float> h((size_t)r * m.vn);
                        G6RT_CK(cudaMemcpy(h.data(), m.hlog, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
                        dump_write(ldump, h.data(), h.size() * sizeof(float));
                        if (fflush(ldump)) die("G6_LOGIT_DUMP: flush failed");
                    });
                // G6_PF_HASH=1 (gate jobs that compare hashes): the resid hash (stream sync + D2H + CPU FNV of M x HID words)
                static const bool pf_hash = getenv("G6_PF_HASH") && atoi(getenv("G6_PF_HASH")) == 1;
                if (pf_hash) printf("prefill rows %d..%d resid %016llx\n", a, a + M - 1, pf_hash_flash(FP, M, st));
                else printf("prefill rows %d..%d\n", a, a + M - 1);
                if (mtp)
                {
                    const auto td = std::chrono::steady_clock::now();
                    pf_draft_flash(D, FP, M, a, p.prompt.data() + a + 1, dstep, [&](const StepIn& di) {
                        *in_d = di;
                        // A draft-logit dump explicitly requests the otherwise unused prompt outputs.
                        const bool cache_only = !dldump && (!getenv("G6_FLASH_DRAFT_KV") || atoi(getenv("G6_FLASH_DRAFT_KV")));
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
        if (const char* path = getenv("G6_FLASH_DRAFT_CACHE_DUMP"); path && mtp)
        {
            G6RT_CK(cudaStreamSynchronize(st));
            const std::string name = std::string(path) + ".s" + std::to_string(q) + ".r" + std::to_string(R) + ".bin";
            FILE* f = fopen(name.c_str(), "wb");
            if (!f) die("G6_FLASH_DRAFT_CACHE_DUMP: cannot open " + name);
            const size_t bytes = hdump_model(f, D.d, 'd', p.first);
            if (fclose(f)) die("G6_FLASH_DRAFT_CACHE_DUMP: close failed " + name);
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
                fabf.enqueue_step(di.step, dfl_fill_sizes());
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
            fab.enqueue_step(si.step, sizes);
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
                fabf.enqueue_step(di.step, dfl_sizes(DF, di.query));
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
            fab.enqueue_step(in->step, sizes);
            cp_enqueue(in->step, M, in->context);
            qs_enqueue(in->step, M);
            dec_prof().on = getenv("G6_DEC_PROF") != nullptr && s > 0;      // the first step of a sequence carries first-launch costs
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
        if (const char* kd = getenv("G6_KDA_STATE_DUMP"); kd && m.kda_st.S)   // gates: KDA S + conv of every layer after the sequence
        {
            const std::string name = std::string(kd) + ".rank" + std::to_string(R);
            FILE* kf = fopen(name.c_str(), "wb");
            if (!kf) die("G6_KDA_STATE_DUMP: cannot open " + name);
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
        if (cp_pf_net(m).verify) printf("# rank %d: G6_CP_PF_VERIFY staged-page hash mismatches %llu\n", R, cp_pf_mismatches(m));
    }
#endif
#if G6_MODEL_FLASH
    if (pf_prof().on) flash_pf_prof().print("this rank,");
    pf_xt_dump(FP, R);
    if (two) { fprintf(stderr, "# PXT second lane\n"); pf_xt_dump(FP2, R); }
#else
    if (pf_prof().on) pf_prof().print("this rank,");
#endif
    if (rdf && fclose(rdf)) die("G6_RESID_DUMP: close failed");
    if (ldump && fclose(ldump)) die("G6_LOGIT_DUMP: close failed");
#if G6_MODEL_FLASH
    if (didump && fclose(didump)) die("G6_DRAFT_IN_DUMP: close failed");
    if (dldump && fclose(dldump)) die("G6_DRAFT_LOGIT_DUMP: close failed");
#endif
    if (rdump) G6RT_CK(cudaFree(rdump));
    if (rlf) fclose(rlf);
    if (getenv("G6_DEC_PROF")) { char tag[32]; snprintf(tag, sizeof tag, "rank %d", R); dp_print(tag, "synchronized", dec_prof().t, dec_prof().steps); }
    if (fabdg_on) { fabdg.wait_proxy_idle(); fabdg.stop_proxy(); }
    if (fabg_on) { fabg.wait_proxy_idle(); fabg.stop_proxy(); }
    fab.wait_proxy_idle();
    fab.barrier();
    fab.stop_proxy();
    return 0;
}
