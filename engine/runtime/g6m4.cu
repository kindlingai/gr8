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
//      [--kv-store STORE,size=N(G|T)]...  under S2 with --reuse-prefix: block stores below the pool, tried in flag order
//                     (s2/kv_tier.cuh, kvman-rs src/stash.rs). A request that continues a stored entry reads its pages
//                     back and skips their prefill. STORE: ssd:PATH (a folder on this box: PATH/r<rank>; with --set-id
//                     its system prefixes and their counts survive restarts) or remote:HOST:PORT (a lease on a KV node,
//                     ./gr8 kv up; RDMA where both ends can, else TCP)
//      [--kv-keep system|sessions|all]  what the stores keep (default system). system: the system prompts and tool
//                     definitions (the span the front end sends, else every grid cell) that --kv-system-after N
//                     conversations started with (default 2), once each. sessions: those, plus the runs the pool evicts
//                     that continued their conversation at least once, stored on the system prefix they start with.
//                     all: every run the pool evicts (benchmarks)
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
//      [--qp-manifest PM --qp-pack PP]  (G6_NR=4) G6_PF_OWN=4 pair weights resident from tools/convert/pairpack.py; or
//                     G6_PF_QP_ROLL=S + G6_PF_QP_CTX=N: S rolling slots, the partner's half over RDMA (docs/DESIGN-QO-PAIR.md)
//      [--dflash-manifest DM --dflash-pack DP [--dflash-sel 0|1] [--dflash-k K]]  replaces MTP with the DFlash2 block
//                     drafter (dflash.cuh): 7 drafts per round in one forward; DP from tools/convert/dflash2.py. --dflash-sel 1 =
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
//      [--set-id HEX]  the model set id from ./gr8 glm53 up (tools/launch/modelset.py). Under S2 it joins the bootstrap
//                     agreement, so ranks loaded from different model sets refuse to start. Otherwise unused.
#ifndef DEC_S2
#error "g6m4 needs a DEC_S2 build (scripts/build.sh defines it at every TP)"
#endif
#include <algorithm>
#include <chrono>
#include "packfind.h"
#include "weight_check.cuh"
#include "prefill.cuh"
#include "dflash.cuh"
#include "prefix_reuse.h"
#include "reuse_poison.cuh"
#include "reuse_hdump.cuh"
#include "../comms/fabric.h"
#include "../comms/progress_cpus.h"
#include "s2/native.cuh"
#include "s2/pool.h"
#include "s2/kv_tier.cuh"
#include "host_reclaim.h"
#include <filesystem>
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
    std::string progress_err;          // G6_PROGRESS_ISOLATE: other threads stay off the proxy CPUs (progress_cpus.h)
    const int progress_isolate = env_choice("G6_PROGRESS_ISOLATE", 1, 1, &progress_err);
    if (progress_isolate < 0) die(progress_err);
    g6::FabricConfig cfg;
    std::string manifest, packfile, layers_s = "0,3,40,77";
    V2Src v2src;
    std::string qo_man, qo_file;        // --qo-manifest/--qo-pack: full-head QO weights (tools/convert/qopack.py), G6_PF_OWN=3
    std::string qp_man, qp_file;        // --qp-manifest/--qp-pack: this rank's pair weights (tools/convert/pairpack.py), G6_PF_OWN=4
    std::string dfl_man, dfl_file;      // --dflash-manifest/--dflash-pack: the DFlash2 drafter (tools/convert/dflash2.py)
    int dfl_sel = -1;                   // --dflash-sel: 1 = selector walk, 0 = argmax, -1 = the pack decides
    int dfl_k = dfl::K;                 // --dflash-k: drafts a round verifies (1..7; the drafter always drafts 7)
    bool adapt = false;                 // --spec-adaptive
    std::string levels_s;               // --spec-levels
    std::vector<std::string> prompts;
    Plan plan;
    int cap = 64;
    float temp = 0.f;
    bool graph = false, image = false, hash = true, reuse_prefix = false;
    int MP = 0, top = 0;
    bool wide = false, pfv2 = false;
    int pfrows = 0, mtp = 0, lanes = 1;
    int nc = 0;                         // --top-p-cands N: top_p candidates per rank per row (0 = top_p off)
    std::string s2dir;                  // --serve DIR: serve through S2 (below R)
    std::vector<std::string> kv_tiers;  // --kv-store (and the deprecated --kv-tier), in flag order
    std::string kv_keep;                // --kv-keep system|sessions|all
    int kv_system_after = 2;            // --kv-system-after
    double kv_write_limit = 0;          // --kv-write-limit F
    std::string kv_format;              // --kv-format fp8|int4: which latent row this binary was built for (refused otherwise)
    bool kv_old = false, kv_old_all = false;   // a deprecated --kv-tier, and whether one kept evicted runs
    std::string set_id;                 // --set-id HEX: joins the S2 bootstrap agreement
    int s2port = 0, s2sess = 8;
    unsigned long long s2epoch = 1;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) die("missing value for " + a); return std::string(argv[++i]); };
        if (a == "--rank") cfg.rank = atoi(next().c_str());
        else if (a == "--ips") cfg.ips = split(next(), ',');
        else if (a == "--ips2") cfg.ips2 = split(next(), ',');
        else if (a == "--ring") cfg.ring = true;
        else if (a == "--ring-devs") cfg.ring_devs = split(next(), ',');
        else if (a == "--proxy-cpu") cfg.proxy_cpu = atoi(next().c_str());
        else if (a == "--manifest") manifest = next();
        else if (a == "--pack") packfile = next();
        else if (a == "--v2") v2src.dir = next();
        else if (a == "--v2-slices") v2src.slices = next();
        else if (a == "--v2-widths") v2src.widths = next();
        else if (a == "--v2-dense") v2src.dense = next();
        else if (a == "--v2-dense-form")
        {
            const std::string f = next();
            if (f != "trellis" && f != "rounded") die("--v2-dense-form: trellis or rounded");
            v2src.trellis = f == "trellis";
        }
        else if (a == "--qo-manifest") qo_man = next();
        else if (a == "--qo-pack") qo_file = next();
        else if (a == "--qp-manifest") qp_man = next();
        else if (a == "--qp-pack") qp_file = next();
        else if (a == "--dflash-manifest") dfl_man = next();
        else if (a == "--dflash-pack") dfl_file = next();
        else if (a == "--dflash-sel") dfl_sel = atoi(next().c_str());
        else if (a == "--dflash-k") dfl_k = atoi(next().c_str());
        else if (a == "--spec-adaptive") adapt = true;
        else if (a == "--spec-levels") levels_s = next();
        else if (a == "--layers") layers_s = next();
        else if (a == "--prompt") prompts.push_back(next());
        else if (a == "--gen") plan.n_gen = atoi(next().c_str());
        else if (a == "--temp") temp = (float)atof(next().c_str());
        else if (a == "--seed") plan.seed = (uint32_t)strtoul(next().c_str(), nullptr, 10);
        else if (a == "--cap") cap = atoi(next().c_str());
        else if (a == "--chunk") plan.chunk = atoi(next().c_str());
        else if (a == "--graph") graph = true;
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
        else if (a == "--prefill") MP = atoi(next().c_str());
        else if (a == "--top") top = atoi(next().c_str());
        else if (a == "--wide") wide = true;
        else if (a == "--pfv2") pfv2 = true;
        else if (a == "--pf-rows") pfrows = atoi(next().c_str());
        else if (a == "--mtp") mtp = atoi(next().c_str());
        else if (a == "--pf-lanes") lanes = atoi(next().c_str());
        else if (a == "--serve") s2dir = next();
        else if (a == "--top-p") plan.top_p = (float)atof(next().c_str());
        else if (a == "--top-p-cands") nc = atoi(next().c_str());
        else if (a == "--spec-accept")
        {
            const std::string v = next();
            if (v == "coupled") plan.rs = false;
            else if (v == "rejection") plan.rs = true;
            else die("--spec-accept must be coupled or rejection");
        }
        else if (a == "--s2-port") s2port = atoi(next().c_str());
        else if (a == "--s2-sessions") s2sess = atoi(next().c_str());
        else if (a == "--s2-epoch") s2epoch = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--set-id")
        {
            set_id = next();
            if (set_id.empty() || set_id.size() > 64 || set_id.find_first_not_of("0123456789abcdef") != std::string::npos)
                die("--set-id must be 1..64 lowercase hex digits");
        }
        else die("unknown argument " + a);
    }
    const bool release = !v2src.dense.empty();
    if (cfg.rank < 0 || cfg.ips.size() != (size_t)g6nr::NR || (!release && (manifest.empty() || packfile.empty())))
        die("usage: g6m4 --rank R --ips NR,addresses (this build: NR = " + std::to_string(g6nr::NR) + ") [--ips2 ..] --manifest M --pack P|- [--qo-manifest QM --qo-pack QP] [--qp-manifest PM --qp-pack PP] [--layers ..] [--gen N] [--temp T]");
    if (release && (!manifest.empty() || !packfile.empty() || v2src.dir.empty()))
        die("--v2-dense replaces --manifest / --pack and needs --v2 (the image comes from the release alone)");
    { const char* e = getenv("G6_DEC_SHARED_SCRATCH"); if (e && atoi(e) == 1 && !pfv2) die("G6_DEC_SHARED_SCRATCH=1 needs --pfv2"); }
    const std::vector<int> layers = parse_ints(layers_s);
    if (prompts.empty()) prompts.push_back("154822,154824,154826,25062");
    plan.inv_temp = temp > 0.f ? 1.f / temp : 0.f;
    if (plan.chunk < 1 || plan.chunk > MX) die("--chunk must be 1.." + std::to_string(MX));
    if (mtp < 0 || mtp > std::min<int>(MX - 1, 7)) die("--mtp must be 1..7 drafts");   // the scheduler caps k at 7 whatever MX is
    if (nc < 0 || nc * g6nr::NR > glue::TP_SORT) die("--top-p-cands must be 0.." + std::to_string(glue::TP_SORT / g6nr::NR));
    const bool s2mode = !s2dir.empty();
    const bool dfl = !dfl_man.empty();
    // S2 keeps --mtp next to the drafter: a session alone at its prompt tail runs DFlash2, the others MTP (s2_batch.h)
    if (dfl && (dfl_file.empty() || (mtp && !s2mode) || dfl_sel < -1 || dfl_sel > 1 || dfl_k < 1 || dfl_k > dfl::K))
        die("--dflash-manifest needs --dflash-pack, and replaces --mtp (7 drafts per round) except under S2; --dflash-sel 0|1; --dflash-k 1..7");
    if (dfl && !dfl::OK_NR) die("--dflash needs a rank count that divides the drafter's 8 KV heads (TP4 build)");
    if (plan.rs && !mtp && !dfl) die("--spec-accept rejection needs --mtp or --dflash-manifest");
    const int K = dfl ? dfl::K : mtp;   // drafts per round
    if ((adapt || !levels_s.empty()) && ((s2mode && !dfl) || (!mtp && !dfl))) die("--spec-adaptive needs --mtp or --dflash-manifest; under S2 it is the DFlash2 sessions' policy");
    if (!levels_s.empty() && !adapt) die("--spec-levels needs --spec-adaptive");
    SpecLen spec;                       // --spec-adaptive. Its estimates persist across requests
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
    }
    if (s2mode && (!graph || !mtp || !MP || s2port < 1 || s2port > 65535 || s2sess < 1 || s2sess > 8 || !s2epoch))
        die("serving (--serve) needs --graph, --mtp, --prefill, --s2-port 1..65535, --s2-sessions 1..8 and --s2-epoch > 0");
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
    for (int k : spec.levels)           // --spec-adaptive: a verify graph per level
        if (std::find(row_counts.begin(), row_counts.end(), k + 1) == row_counts.end()) row_counts.push_back(k + 1);
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
    if (MP)
    {
        g6::FabricConfig cp = cfg;
        cp.slot_bytes = pf_slot_bytes(MP, pf_rs, (int)cfg.ips.size());
        cp.ncoll = pf_rs ? 5 : 2;
        cp.port = cfg.port + 8;
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
        cp.proxy_cpu = cfg.proxy_cpu > 1 ? cfg.proxy_cpu - 2 : -1;
        fabp2.init(cp);
        fabp2.start_proxy();
    }
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
#if defined(G6_CP) && G6_CP
    // CP: env as m4sim (G6_CP, G6_CP_R, G6_CP_S3, G6_CP_MMAX). Own fabric `fabc` (cplayout.h variable
    // layout: per-collective slots, single-destination partials), port + 32, its own proxy CPU.
    const bool cp_on = !getenv("G6_CP") || atoi(getenv("G6_CP")) != 0;
    const int cp_R = getenv("G6_CP_R") ? atoi(getenv("G6_CP_R")) : 0;
    const int cp_s3 = getenv("G6_CP_S3") ? atoi(getenv("G6_CP_S3")) : 1;
    int cp_mmax = K + 1;
    for (int M : row_counts) cp_mmax = std::max(cp_mmax, M);
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
    auto cp_error = [&]() -> unsigned { return cp_on ? *fabc.dev().error : 0u; };
#else
    auto cp_enqueue = [&](unsigned long long, int, int) {};
    auto cp_error = [&]() -> unsigned { return 0u; };
#endif
    G6RT_CK(bind::init());
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
#if defined(G6_CP) && G6_CP
    m.cp.on = cp_on; m.cp.R = cp_R; m.cp.s3 = cp_s3; m.cp.colls = cp_colls;
#endif
    host_reclaim(R, "after the pack");         // the loaders keep no page cache; this trims the heap they freed
    boot("pack loaded");
    build(m, R, layers, find, cap, f, stepp, st, pf_attn_rows(MP));   // bindings sized for the attention chunk
    boot("layers bound");
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
        build_prefill(P, m, find, MP, fp, dmalloc<u64>(1), st);
    }
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
        build_prefill(P2, m, find, MP, fp2, dmalloc<u64>(1), st, &P);
    }
    u64 ls0b = 1;
    Draft D;
    DraftPrefill DQ;
    g6::FabricDev fd{};
    StepIn* in_d = nullptr;
    StepOut* out_d = nullptr;
    if (mtp)
    {
        fd = fabd.dev();
        G6RT_CK(cudaHostAlloc((void**)&in_d, sizeof(StepIn), cudaHostAllocMapped));
        G6RT_CK(cudaHostAlloc((void**)&out_d, sizeof(StepOut), cudaHostAllocMapped));
        build_draft(D, m, find, cap, fd, dmalloc<u64>(1), st, MP ? pf_attn_rows(MP) : MX);
        D.d.graph = graph; D.d.in = in_d; D.d.out = out_d;
        DQ.P.attn_rows = pf_attn_rows(MP);
        if (MP) build_draft_prefill(DQ, D, find, MP, fp, pfv2, st, &P);
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
        dfl_attach(m, DF);              // before the target graphs are captured: they carry the residual taps
        if (pfctx) { P.dfl_tap = dfl_pf_tap_launch; P2.dfl_tap = dfl_pf_tap_launch; }
    }
    // S2: below R on the CP binding. Session views on the pool, main and draft; today's graphs below
    // are captured with s2on off (the S = 1 path), the S2 graphs later by s2/native.cuh with s2on on.
    int s2_below = 0;
    if (s2mode)
    {
#if defined(G6_CP) && G6_CP
        if (!cp_on || !cp_R) die("S2: serving below R needs G6_CP on and G6_CP_R");
        s2_below = getenv("G6_S2_BELOW") ? atoi(getenv("G6_S2_BELOW")) : cp_R;
        if (s2_below < 64 || s2_below % 64 || s2_below > cp_R) die("S2: G6_S2_BELOW (default G6_CP_R) must be 64..G6_CP_R, a multiple of 64");
#else
        die("S2: serving needs the CP binding (a G6_CP build)");
#endif
        m.s2_below = D.d.s2_below = s2_below;
        s2_init(m, s2sess, st);
        s2_init(D.d, s2sess, st);
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
    G6RT_CK(cudaStreamSynchronize(st));
    cudaGraphExec_t exec[NBK][MX + 1] = {}, exec_d[NBK][2][MX + 1] = {};      // [context bucket][...]
    // DEC_GSPLIT: G6_DEC_SPLIT=ls (1..nl-1) captures the target step as two graphs, exec = step_in, embed, layers [0, ls) and
    // exec2 = the rest; launched back to back, the second graph's submission overlaps the first one's GPU time (0 = one graph)
    const int gsplit = getenv("G6_DEC_SPLIT") ? atoi(getenv("G6_DEC_SPLIT")) : 0;
    if (gsplit < 0 || gsplit >= NL || (gsplit && !graph)) die("G6_DEC_SPLIT must be 0..layers-1 and needs --graph");
    if (s2mode && gsplit != 3) die("serving needs G6_DEC_SPLIT=3 (the production split3 graphs)");
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
                    capture([&] { launch_step(m, M, 0, st, 0, gsplit); }, &exec[k][M], "split part A ", M);
                    capture([&] { launch_step(m, M, 0, st, 1, gsplit); }, &exec2[k][M], "split part B ", M);
                }
                else capture([&] { launch_step(m, M, 0, st); }, &exec[k][M], "", M);
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

    if (s2mode)
    {
        // S2 below R served loop. Pool: logical capacity = cap (a past-R request alone at base 0 reads the table as
        // built), runs below R placed in the physical pages every rank and model has.
        g6s2::Config sc;
        sc.epoch = s2epoch;
        sc.pool_rows = (uint32_t)(cap / 64 * 64);
        // one step bounds: a round (every session's mtp + 1 rows) and a prompt chunk (sessions.h: chunk <= max_rows)
        sc.max_rows = std::min<uint32_t>(MX, std::max<uint32_t>((uint32_t)s2sess * (uint32_t)(mtp + 1), (uint32_t)plan.chunk));
        sc.max_sessions = (uint32_t)s2sess;
        sc.mtp = (uint32_t)mtp;
        if (dfl) sc.guard_tokens = 2 * dfl::K;     // a DFlash2 verify writes up to 7 rows past the last token, as the offline loop's cap rule
        sc.prefill_rows = (uint32_t)MP;
        sc.two_lanes = two;
        const int place = std::min(m.s2->pool_pages, D.d.s2->pool_pages);
        g6s2::RunPool pool((uint32_t)(cap / 64), (uint32_t)place, (uint32_t)s2_below);
        // the pool's R is the CP map's: a run past R takes its own pages — the prefix and its shard below place, the
        // drafter's tail copy above it — instead of the whole pool, and retained runs keep their pages through a long
        // turn. The stores' tail blocks (per_slot) come with --kv-store below.
        if (m.cp.on && s2_below == m.cp.R) pool.tail((uint32_t)g6nr::NR, 0);
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
        // one model). --kv-keep sets each store's keep rule and policy (kvman-rs stash.rs Keep, Policy)
        std::unique_ptr<g6s2::KvTiers> kvt;
        bool kv_index = false;
        const int kv_keep_rule = kv_keep == "system" ? 1 : 2;           // shared prefixes, or those and evicted runs
        const uint32_t kv_min_hits = kv_keep == "sessions" ? 1 : 0;
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
                    pool.add_tier(slots, kv_keep_rule, index, compat);
                    pool.policy((uint32_t)i, 0, kv_min_hits, kvt->blocks().slot_bytes());
                    kv_index = kv_index || !index.empty();
                    fprintf(stderr, "# rank %d: s2 store %zu: %s, %llu blocks of %zu B%s\n", R, i, kvt->describe(i).c_str(),
                            (unsigned long long)slots, kvt->blocks().slot_bytes(), index.empty() ? "" : ", indexed");
                }
                pool.write_limit((uint64_t)std::llround(kv_write_limit * 1000));
                // a run past R keeps its tail: each rank's shard and the replicated parts (only when the pool's R is the CP
                // map's, so the pages past it are the sharded ones)
                if (m.cp.on && s2_below == m.cp.R && kvt->blocks().per_slot()) pool.tail((uint32_t)g6nr::NR, kvt->blocks().per_slot());
            }
            catch (const std::exception& e) { die(std::string("--kv-store: ") + e.what()); }
            if (R == 0)
                fprintf(stderr, "# s2 stores keep %s: a system prefix is stored after %d conversations%s\n", kv_keep.c_str(), kv_system_after,
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
        long kv_delay_n = 0;
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
                fprintf(stderr, "# rank %d: s2 tier io: %zu spilled, %zu moved, %zu restored pages in %.1f ms (device %.1f ms) at %lld ms%s\n", R,
                        n[0], n[1], n[2], kvt->last_ms, kvt->last_device_ms, (long long)g6s2::s2_wall_ms(), kvt->last_ms > 2000 ? " SLOW" : "");
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
        // prefill started last (dfl_owner; -1 = none, or a prefix-reuse hit that skipped part of the window)
        static_assert(s2::DFL_B == dfl::B, "s2_batch.h DFL_B follows dfl::B");
        int dfl_owner = -1;
        uint64_t dfl_req = UINT64_MAX;
        std::array<uint32_t, g6s2::MAX_SESSIONS> slot_prompt{};
        u64 s2_dstep = 0;                // the drafter fabric's steps
        g6s2::NativeContext nx{m, D, P, P2, DQ, fab, fabd, fabp, fabp2, st, pf_streams ? &lanes2 : nullptr, in, in_d, out, out_d, MP, gsplit, two,
            [&](const StepIn& si) {
                std::vector<uint32_t> sizes(NC, (uint32_t)(si.M * HID * 2));
                sizes[2 * NL] = head_size(si, nc);
                fab.enqueue_step(si.step, sizes);
                cp_enqueue(si.step, si.M, si.context);
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
                    dfl_owner = (int)sl.prefilled <= DF.pf_lo ? (int)sl.key.slot : -1;
                }
            },
            [&]() { m.s2_pf_table = D.d.s2_pf_table = false; },
            [&]() { cp_pf_in(); }, [&]() { cp_pf_out(); },
            [&]() -> unsigned {   // cp: the CP decode fabric; cp_pf|qp: the CP prefill staging fabric (0x63..), else the QO pair wire (0x65..)
                const unsigned c = cp_error(), p = cp_pf_err();
                if (c || p) fprintf(stderr, "# rank %d: S2 extra error words: cp %08x cp_pf|qp %08x\n", R, c, p);
                return c | p;
            }};
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
                h.pool.retain = [&](const g6s2::Run& r, const PrefixReuse& s) { pool.retain(r, s); };
                h.pool.match = [&](const std::vector<int>& p, uint64_t rows, uint64_t worst, int limit) { return pool.match(p, rows, worst, limit); };
                h.pool.take = [&](const g6s2::PoolHit& x, uint64_t rows, uint64_t worst) { return pool.take(x, rows, worst); };
            }
            if (kvt)
            {
                h.pool.tiers = [&] { return pool.tiers(); };
                h.pool.count = [&](const std::vector<int>& p, int64_t span) { return pool.count(p, span); };
                h.pool.retain = [&](const g6s2::Run& r, const PrefixReuse& s) { pool.retain(r, s); kv_drain(); };   // a promoted prefix
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
                slot_prompt[q] = g.req.prompt_tokens;
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
                const std::vector<int> list = pool.list(r);
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
            h.retire = [&](const g6s2::Slot& old) { nd.retire(old); };
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
                if (R == 0 && h.dsel.on) fprintf(stderr, "# s2 drafter choice on: DFlash2 or MTP per round (dwell %d, margin %.2f, probe %d)\n",
                                                    h.dsel.dwell, h.dsel.margin, h.dsel.probe);
                // a 1-token prompt has no prefill: the drafter starts from its prompt step (dfl_lo = first, as plain)
                h.dispatch.dfl_ok = [&](int q) { return dfl_owner == q || slot_prompt[q] == 1; };
                h.dispatch.dfl_start = [&](int q) {    // the offline loop's fill steps: Dfl::pf_feat -> the ring
                    if (dfl_owner != q || DF.pf_hi <= DF.pf_lo) return (int)slot_prompt[q] - 1;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int c0 = DF.pf_lo; c0 < DF.pf_hi; c0 += dfl::B)
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
                    if (R == 0) fprintf(stderr, "# s2 dflash slot %d prompt context %d..%d: %.1f ms\n", q, DF.pf_lo, DF.pf_hi - 1,
                                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                    return DF.pf_lo;
                };
                h.dispatch.dfl = [&](const s2::DflReq& r, const g6s2::Sampling& smp) {
                    DflIn di;
                    memset(&di, 0, sizeof di);
                    di.step = ++s2_dstep;
                    di.ctx_rows = r.rows; di.ctx_pos0 = r.pos0; di.q0 = r.pos0 + r.rows; di.lo = r.lo; di.tok0 = r.tok0; di.query = r.query;
                    const float dtemp = plan.rs ? 0.f : smp.inv_temp;   // rejection sampling: greedy drafts, as dflash_sequence
                    for (int i = 0; i < dfl::B; ++i) di.srow[i] = glue::SampleRow{dtemp, smp.seed, 0u, di.q0 + i, 0, smp.top_p};
                    *in_f = di;
                    fabf.enqueue_step(di.step, dfl_sizes(DF, di.query));
                    G6RT_CK(cudaGraphLaunch(exec_f[di.query ? 1 : 0], st));
                    G6RT_CK(cudaStreamSynchronize(st));
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
                (set_id.empty() ? "" : " set " + set_id);
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
                nd.capture_all(sc.limit(), mtp, plan.chunk);
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
        Plan sp = plans[q];
        // Never skip into a teacher-forced scoring tail (--pf-rows). Without tensor prefill, the decode prompt steps
        // start directly at skip.
        const int old_end = reuse.written_end, old_dirty = reuse.dirty_end;   // dirty: every row written since the last reset
        const int skip = reuse_prefix ? reuse.begin(sp.prompt, MP ? sp.first : (int)sp.prompt.size() - 1, mtp > 0) : 0;
        if (skip > sp.first) sp.first = skip;
        const Plan& p = sp;
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
        printf("seq %zu prompt %zu tokens\n", q, p.prompt.size());
        if (reuse_prefix && R == 0)         // every miss with its reason (first / strict / no-lcp / short), counted by the gate
            fprintf(stderr, "# reuse seq %zu %s retained %d lcp %d history %zu prompt %zu written_end %d dirty_end %d\n", q, reuse.why, skip,
                    reuse.lcp, reuse.history.size(), p.prompt.size(), old_end, old_dirty);
        if (reuse_prefix && R == 0 && reuse.grid)   // the skip before the grid rule (TTFT cost of the exact rule)
            fprintf(stderr, "# reuse seq %zu grid %d raw %d retained %d\n", q, reuse.grid, reuse.raw, skip);
        if (skip) printf("reuse rows 0..%d\n", skip - 1);  // real hit evidence (every rank), NOT a freshly computed hash
        for (int a = skip; a < p.first; a += MP)
        {
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
            const int M = p.first - a < MP ? p.first - a : MP;
            std::vector<int> pos(M);
            for (int r = 0; r < M; ++r) pos[r] = a + r;
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
            const auto t0 = std::chrono::steady_clock::now();
            if (graph) launch_main(bucket_of(m, si.context), M);
            else launch_step(m, M, si.context, st);
            G6RT_CK(cudaStreamSynchronize(st));
            const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            evprof_read(M, st, R);
            ms += dt;
            if (*f.error || cp_error()) { fprintf(stderr, "g6m4: rank %d fabric error word %08x (CP %08x) at seq %zu step %d\n", R, *f.error, cp_error(), q, s); exit(3); }
            const StepOut so = *out;
            if (so.attn_err) { fprintf(stderr, "g6m4: rank %d attention error word %08x at seq %zu step %d\n", R, so.attn_err, q, s); exit(3); }
            if (reuse_prefix) reuse.main_rows(si.pos[0], M);
            if (rlf) route_log_line(rlf, m, layers, q, s, M, si.pos[0], p.prompt.size());
            std::string line;
            step_line(m, so, s, M, line);
            printf("%s\n", line.c_str());
            if (top > 0) { std::string tl; top_lines(m, s, M, top, tl); fputs(tl.c_str(), stdout); }
            fprintf(stderr, "# seq %zu step %d M %d: %.3f ms\n", q, s, M, dt);
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
                const auto t0 = std::chrono::steady_clock::now();
                if (graph) G6RT_CK(cudaGraphLaunch(exec_d[bucket_of(D.d, di.context)][chain][M], st));
                else launch_draft_step(D, M, di.context, chain, st);
                G6RT_CK(cudaStreamSynchronize(st));
                const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ms += dt;
                if (*fd.error) { fprintf(stderr, "g6m4: rank %d draft fabric error word %08x at seq %zu step %d\n", R, *fd.error, q, s); exit(3); }
                const StepOut so = *out_d;
                if (so.attn_err) { fprintf(stderr, "g6m4: rank %d draft attention error word %08x at seq %zu step %d\n", R, so.attn_err, q, s); exit(3); }
                std::string line;
                if (reuse_prefix) reuse.draft_rows(di.pos[0], M, chain);
                const int t = step_line(D.d, so, s, M, line);
                printf("%s%s\n", chain ? "chain" : "draft", line.c_str() + 4);
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
            dec_prof().on = getenv("G6_DEC_PROF") != nullptr && s > 0;      // the first step of a sequence carries first-launch costs
            dec_prof().M = M; dec_prof().last = 0;
            const auto t0 = std::chrono::steady_clock::now();
            if (graph) launch_main(bucket_of(m, in->context), M);
            else launch_step(m, M, in->context, st);
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
            last_tok = step_line(m, so, s, M, line);
            printf("%s\n", line.c_str());
            if (top > 0) { std::string tl; top_lines(m, s, M, top, tl); fputs(tl.c_str(), stdout); }
            if (s >= PS - 1) gen.push_back(last_tok);
            fprintf(stderr, "# seq %zu step %d M %d: %.3f ms\n", q, s, M, ms);
            if (s >= PS + 2) t1.push_back(ms);
        }
        step0 += NS;
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
    if (pf_prof().on) pf_prof().print("this rank,");
    if (rlf) fclose(rlf);
    if (getenv("G6_DEC_PROF")) { char tag[32]; snprintf(tag, sizeof tag, "rank %d", R); dp_print(tag, "synchronized", dec_prof().t, dec_prof().steps); }
    fab.wait_proxy_idle();
    fab.barrier();
    fab.stop_proxy();
    return 0;
}
