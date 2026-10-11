// M4 reference driver: the mini model (a list of real layers + the real head) on ONE GPU with six simulated
// ranks, through the same rank step as the six-process runtime (rank.cuh). A loopback fabric stands in for the NIC:
// the ranks run a stage one after another, then deliver() copies every send slot of that collective into the shared
// recv slots and writes the tails, so the wait / consume / head-merge kernels run unchanged. deliver() is a kernel,
// so a whole six-rank step is one stream of launches: run as they are, or (--graph) captured once per row count and
// replayed, with the attention through its replay entry. Both forms must print the same lines.
//
// Test sequence (rank.cuh Plan): step 0 = the prompt rows at positions 0..n-1, then --gen single-row steps, each fed
// the token sampled from the previous step's last row. After every step all six ranks must hold the same sampled
// ids, logprobs and residual; the driver prints rank 0's line ("step ..."), which g6m4 must reproduce on six nodes.
//
// Shared-index layers reuse the index list of the nearest full-index layer BEFORE them in --layers (in the full model
// that is the layer's own group leader; at these short contexts every list is "all causal keys", so it is the same).
//
// Every buffer a kernel touches is device memory; the host uses cudaMemcpy only (WSL: no managed memory).
//
// m4sim --packs DIR | --v2 DIR --v2-slices DIR --v2-widths FILE --v2-dense DIR   [--layers 0,3,40,77] [--prompt 154822,154824,154826,25062] [--gen 16] [--temp 0] [--seed 1234]
//       [--cap 64] [--chunk 4] [--graph] [--prefill MP]
// --prefill MP: every prompt token but the last goes through the prefill path in sub-batches of up to MP rows
// (prefill.cuh; its own loopback fabric with two collectives and the layer as step); the last prompt token and the
// generated ones go through decode steps as before.
// --prompt may be repeated: each is its own sequence (attention caches reset in between); "@file" reads ids
// --v2-dense (Rust loader build): each simulated rank's image is planned from the exlr8 release as g6m4 --v2-dense plans
//        it, and each tensor is read from the release when a rank first binds it (only the --layers' tensors are read).
//        --v2-dense-form trellis|rounded as in g6m4
// [--serve DIR]  requests from DIR/req.<n> until DIR/quit, rank 0 streams DIR/resp.<n> (serve.h); --prompt is ignored
// [--top-p F] [--top-p-cands N]  nucleus sampling at T (per request with --serve); N candidates per rank per row
//                (0 = top_p off; 256 is the served setting; misses printed at the end)
// [--s2] (DEC_S2 build, needs --mtp, no --prefill / --serve): every --prompt is a session (<= 8) and their rows share
//        steps (s2_batch.h mtp_batch, s2.cuh). Lines per session: "q<j> " + the solo line (step / draft / chain /
//        round / generated), so session j's lines compare with the solo run of its prompt after the prefix is stripped
// [--spec-accept coupled|rejection]  as g6m4 (needs --mtp, no --s2): rejection = greedy drafts, accepted by rejection sampling
// [--spec-adaptive [--spec-levels 1,2,4]]  as g6m4 (needs --mtp, no --s2): the drafts each round verifies (spec_len.h)
#include <algorithm>
#include <chrono>
#include "packfind.h"
#include "prefill.cuh"
#include "serve.h"
#ifdef DEC_S2
#include <map>
#include <set>
#include "s2_batch.h"
#endif

using namespace g6rt;

constexpr int NR = g6nr::NR;           // simulated ranks: six, or four in a -DG6_NR=4 build

// the NIC: every rank's send slot of collective c -> the shared recv slot [c][rank], then the tail. grid NR, block 128.
struct Loop
{
    const char* send[NR];
    char* recv;
    const u64* stepp;
    int c, ncoll, n16;                // n16 = payload / 16
    size_t stride;
    int mode = 0, M = 0, row_bytes = 0;   // prefill RS form: 1 = reduce-scatter (slices of M rows at p * ss, kept at their
                                          // offsets), 2 = all-gather (pf_rows(M, sender) x row_bytes)
};
__global__ void __launch_bounds__(128) k_deliver(Loop a)
{
    const u64 step = *a.stepp;
    const size_t slot = (step & 1) * a.ncoll + a.c;
    const char* src = a.send[blockIdx.x] + slot * a.stride;
    char* dst = a.recv + (slot * NR + blockIdx.x) * a.stride;
    const u64 seq = g6::coll_seq(step, a.c, a.ncoll);
    auto copy = [&](size_t off, size_t n16) {
        const uint4* s16 = reinterpret_cast<const uint4*>(src + off);
        uint4* d16 = reinterpret_cast<uint4*>(dst + off);
        for (size_t i = threadIdx.x; i < n16; i += 128) d16[i] = s16[i];
        __syncthreads();
        if (threadIdx.x == 0)
        {
            __threadfence();
            *reinterpret_cast<u64*>(d16 + n16) = seq;
        }
        __syncthreads();
    };
    if (a.mode == 1)
    {
        const size_t ss = ((a.stride / NR) & ~size_t(255));
        for (int p = 0; p < NR; ++p) copy(p * ss, (size_t)pf_rows(a.M, p, NR) * HID * 2 / 16);
    }
    else if (a.mode == 2) copy(0, (size_t)pf_rows(a.M, blockIdx.x, NR) * a.row_bytes / 16);
    else copy(0, a.n16);
}

// noise floor (G6_PERTURB): resid[row][j] *= 1 + eps * u(pos, j), u uniform in [-1, 1) from an integer hash
__global__ void k_perturb(float* resid, const int* pos, float eps)
{
    const int row = blockIdx.x;
    const unsigned p = (unsigned)pos[row];
    for (int j = threadIdx.x; j < HID; j += blockDim.x)
    {
        unsigned h = p * 0x9E3779B1u ^ (unsigned)j * 0x85EBCA77u;
        h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
        resid[(size_t)row * HID + j] *= 1.f + eps * ((float)(h >> 8) * (2.f / 16777216.f) - 1.f);
    }
}

#if defined(G6_CP) && G6_CP
// the NIC for the CP fabric (comms/cplayout.h): collective c of every sender -> the receivers' slots (all peers or one),
// payload then tail. grid NR (one CTA per sender), block 128. Pointers for both parities, picked by the device step.
struct VLoop
{
    const char* src[NR][2];
    char* dst[NR][NR - 1][2];
    int ndst[NR];
    const u64* stepp;
    int c, ncoll;
    size_t bytes;
};
__global__ void __launch_bounds__(128) k_vdeliver(VLoop a)
{
    const u64 step = *a.stepp, seq = g6::coll_seq(step, a.c, a.ncoll);
    const int s = blockIdx.x, par = step & 1;
    const uint4* src = reinterpret_cast<const uint4*>(a.src[s][par]);
    for (int j = 0; j < a.ndst[s]; ++j)
    {
        uint4* dst = reinterpret_cast<uint4*>(a.dst[s][j][par]);
        for (size_t i = threadIdx.x; i < a.bytes / 16; i += 128) dst[i] = src[i];
        __syncthreads();
        if (threadIdx.x == 0) { __threadfence(); *reinterpret_cast<u64*>(a.dst[s][j][par] + a.bytes) = seq; }
        __syncthreads();
    }
}
#endif

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    pf_env_check();                    // strict G6_PF_* choices (envparse.h): a bad value exits before any work
    std::string packs, layers_s = "0,3,40,77";
    V2Src v2src;                        // --v2-dense: every rank's image from the exlr8 release (no packs)
    std::vector<std::string> prompts;
    Plan plan;
    int cap = 64;
    float temp = 0.f;
    bool graph = false;
    int MP = 0, top = 0;
    bool wide = false, pfv2 = false;
    int pfrows = 0, mtp = 0, lanes = 1;
    bool reuse = false;                 // --reuse-prefix: keep the caches of the rows a prompt shares with the previous one
    std::string serve;
    int nc = 0;                         // --top-p-cands N: top_p candidates per rank per row (0 = top_p off)
    std::string qo_man, qo_file;       // --qo-manifest/--qo-pack: full-head QO weights (tools/convert/qopack.py) for G6_PF_OWN=3
    std::string mtp_dump;               // --mtp-dump PREFIX: rank 0 draft buffers of seq 0 (tools/m6/check_draft.py)
    bool s2mode = false;                // --s2: the prompts as sessions of one batched decode (DEC_S2)
    bool adapt = false;                 // --spec-adaptive
    std::string levels_s;               // --spec-levels
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) die("missing value for " + a); return std::string(argv[++i]); };
        if (a == "--packs") packs = next();
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
        else if (a == "--layers") layers_s = next();
        else if (a == "--prompt") prompts.push_back(next());
        else if (a == "--gen") plan.n_gen = atoi(next().c_str());
        else if (a == "--temp") temp = (float)atof(next().c_str());
        else if (a == "--seed") plan.seed = (uint32_t)strtoul(next().c_str(), nullptr, 10);
        else if (a == "--cap") cap = atoi(next().c_str());
        else if (a == "--chunk") plan.chunk = atoi(next().c_str());
        else if (a == "--graph") graph = true;
        else if (a == "--prefill") MP = atoi(next().c_str());
        else if (a == "--top") top = atoi(next().c_str());
        else if (a == "--wide") wide = true;
        else if (a == "--pfv2") pfv2 = true;
        else if (a == "--pf-rows") pfrows = atoi(next().c_str());
        else if (a == "--mtp") mtp = atoi(next().c_str());
        else if (a == "--pf-lanes") lanes = atoi(next().c_str());
        else if (a == "--reuse-prefix") reuse = true;
        else if (a == "--mtp-dump") mtp_dump = next();
        else if (a == "--serve") serve = next();
        else if (a == "--top-p") plan.top_p = (float)atof(next().c_str());
        else if (a == "--top-p-cands") nc = atoi(next().c_str());
        else if (a == "--s2") s2mode = true;
        else if (a == "--spec-adaptive") adapt = true;
        else if (a == "--spec-levels") levels_s = next();
        else if (a == "--spec-accept")
        {
            const std::string v = next();
            if (v == "coupled") plan.rs = false;
            else if (v == "rejection") plan.rs = true;
            else die("--spec-accept must be coupled or rejection");
        }
        else die("unknown argument " + a);
    }
    const bool release = !v2src.dense.empty();
    if (packs.empty() == !release || (release && (v2src.dir.empty() || v2src.slices.empty() || v2src.widths.empty())))
        die("usage: m4sim --packs DIR | --v2 DIR --v2-slices DIR --v2-widths FILE --v2-dense DIR [--layers 0,3,40,77] [--prompt ids] [--gen 16] [--temp 0] [--seed 1234] [--cap 64]");
#ifndef G6_RS_LOADER
    if (release) die("--v2-dense needs the Rust loader (scripts/build.sh links it when cargo is installed)");
#endif
    { const char* e = getenv("G6_DEC_SHARED_SCRATCH"); if (e && atoi(e) == 1 && !pfv2) die("G6_DEC_SHARED_SCRATCH=1 needs --pfv2"); }
    const std::vector<int> layers = parse_ints(layers_s);
    if (prompts.empty()) prompts.push_back("154822,154824,154826,25062");
    plan.inv_temp = temp > 0.f ? 1.f / temp : 0.f;
    if (plan.chunk < 1 || plan.chunk > MX) die("--chunk must be 1.." + std::to_string(MX));
    if (mtp < 0 || mtp > std::min<int>(MX - 1, 7)) die("--mtp must be 1..7 drafts");   // the scheduler caps k at 7 whatever MX is
    if (plan.rs && (s2mode || !mtp)) die("--spec-accept rejection needs --mtp, and not --s2");
    if ((adapt || !levels_s.empty()) && (s2mode || !mtp || (!levels_s.empty() && !adapt))) die("--spec-adaptive needs --mtp, and not --s2; --spec-levels needs it");
    SpecLen spec;                       // --spec-adaptive (graphs are captured per row count on first use here)
    if (adapt)
    {
        std::vector<int> lv;
        if (levels_s.empty()) for (int k = 1; k <= mtp; ++k) lv.push_back(k);
        else lv = parse_ints(levels_s);
        for (int k : lv) if (k < 1 || k > mtp) die("--spec-levels: each level must be 1.." + std::to_string(mtp));
        spec.on = true;
        if (!spec.set_levels(lv)) die("--spec-levels: no level");
        spec.draft_round = 2.5;
        spec.draft_each = 2.1;
    }
    if (nc < 0 || nc * NR > glue::TP_SORT) die("--top-p-cands must be 0.." + std::to_string(glue::TP_SORT / NR));
#ifdef DEC_S2
    if (s2mode && (!mtp || MP || !serve.empty() || prompts.size() > (size_t)S2N)) die("--s2 needs --mtp, no --prefill / --serve, at most 8 --prompt");
#else
    if (s2mode) die("--s2 needs a DEC_S2 build");
#endif
    std::vector<Plan> plans;            // one sequence per --prompt
    std::vector<int> row_counts;
    for (const std::string& ps : prompts)
    {
        Plan p = plan;
        p.prompt = parse_ints(ps);
        if (p.prompt.empty()) die("empty prompt");
        if ((int)p.prompt.size() + p.n_gen + 2 * mtp > cap) die("prompt + gen (+ 2 x drafts) exceeds --cap");
        if (MP) { const int pr = pfrows < 0 ? (int)p.prompt.size() + pfrows : pfrows;   // --pf-rows -N: all rows but the last N
                  p.first = pr > 0 && pr < (int)p.prompt.size() - 1 ? pr : (int)p.prompt.size() - 1; }   // --pf-rows: the rest is teacher-forced through decode steps
        for (int M : plan_rows(p))
            if (std::find(row_counts.begin(), row_counts.end(), M) == row_counts.end()) row_counts.push_back(M);
        plans.push_back(p);
    }
    const int NL = (int)layers.size(), NC = 2 * NL + 1;
#if defined(G6_CP) && G6_CP
    // CP: G6_CP=0|1 (default 1 in a CP build), G6_CP_R (replicated prefix tokens, multiple of 64, default 0),
    // G6_CP_S3=0|1 (default 1), G6_CP_MMAX (CP fabric rows, default = the most rows of any step)
    const bool cp_on = !getenv("G6_CP") || atoi(getenv("G6_CP")) != 0;
    const int cp_R = getenv("G6_CP_R") ? atoi(getenv("G6_CP_R")) : 0;
    const int cp_s3 = getenv("G6_CP_S3") ? atoi(getenv("G6_CP_S3")) : 1;
    int cp_mmax = mtp + 1;
    for (int M : row_counts) cp_mmax = std::max(cp_mmax, M);
    if (getenv("G6_CP_MMAX")) cp_mmax = std::max(cp_mmax, atoi(getenv("G6_CP_MMAX")));
    if (cp_R < 0 || cp_R % 64) die("G6_CP_R must be a multiple of 64");
    g6::CpColls cp_colls;
    {
        std::vector<bool> full;
        for (int l : layers) full.push_back(cp_full_index(l));
        cp_colls.make(full, NR, cp_mmax);
    }
    g6::VLayout cp_L;
    cp_L.make(NR, cp_colls.cap, cp_colls.dst_k);
    std::vector<char*> cp_send(NR), cp_recv(NR);
    std::vector<unsigned long long*> cp_bell(NR);
    if (cp_on)
        for (int r = 0; r < NR; ++r)
        {
            cp_send[r] = dmalloc<char>(cp_L.send_bytes());
            cp_recv[r] = dmalloc<char>(cp_L.recv_bytes());
            cp_bell[r] = dmalloc<unsigned long long>(cp_colls.ncoll());
        }
    if (cp_on) fprintf(stderr, "# G6_CP loopback: %d collectives, mmax %d, %.3f GB per rank\n", cp_colls.ncoll(), cp_mmax,
                       (cp_L.send_bytes() + cp_L.recv_bytes()) / 1e9);
#endif
    // --s2 on a CP build (G6_CP on): S2 below R. Every session's run <= G6_S2_BELOW rows (default G6_CP_R,
    // multiple of 64, <= G6_CP_R), on the CP binding as built; runs are granted by G6_S2_RUNS or the default layout
    int s2_below = 0;
#if defined(G6_CP) && G6_CP
    if (s2mode && cp_on)
    {
        s2_below = getenv("G6_S2_BELOW") ? atoi(getenv("G6_S2_BELOW")) : cp_R;
        if (s2_below < 64 || s2_below % 64 || s2_below > cp_R) die("--s2 with G6_CP: G6_S2_BELOW (default G6_CP_R) must be 64..G6_CP_R, a multiple of 64");
        fprintf(stderr, "# s2 below R: runs of at most %d rows, G6_CP_R %d\n", s2_below, cp_R);
    }
#endif
    G6RT_CK(bind::init());
    pf_prof().on = getenv("G6_PF_PROF") != nullptr;

    // ---- loopback fabric
    const size_t stride = ((size_t)MX * HID * 2 + 8 + 255) / 256 * 256;
    char* recv = dmalloc<char>(2 * (size_t)NC * NR * stride);
    u64* stepp = dmalloc<u64>(1);
    unsigned* err = dmalloc<unsigned>(1);
    unsigned long long* doorbell = dmalloc<unsigned long long>(NC);

    // ---- packs -> device tensors (shared across ranks by sha256) -> models
    cudaStream_t st;
    G6RT_CK(cudaStreamCreate(&st));
    StepIn* in = dmalloc<StepIn>(1);
    StepOut* out = dmalloc<StepOut>(NR);
    std::vector<g6::Pack> pack(NR);
    std::vector<Model> R(NR);           // sized once: the attention bindings must not move
    DevStore store;
#ifdef G6_RS_LOADER
    PlanStore pstore;
    pstore.plans.resize(NR);
#endif
    // rank r's pack (or, with --v2-dense, its planned image)
    auto rload = [&](int r) {
#ifdef G6_RS_LOADER
        if (release) { pstore.plans[r] = rs_plan_release(pack[r], r, v2src); return; }
#endif
        load_pack(pack[r], packs + "/rank" + std::to_string(r) + ".manifest.json", packs + "/rank" + std::to_string(r) + ".pack", r);
    };
    // rank r's tensors, with the FP8 copies of its dense trellis matrices for prefill (dtr_find.h), made once
    std::vector<Find> rfinds(NR);
    auto rfind = [&](int r) -> const Find& {
        if (rfinds[r]) return rfinds[r];
        Find base = pack_find(pack[r], store);
#ifdef G6_RS_LOADER
        if (release) base = plan_find(pack[r], pstore, r);
#endif
        rfinds[r] = dtr_overlay(base, MP > 0, st, r);
        return rfinds[r];
    };
    Loop loop{}, loop_pf{};
    unsigned long long* xt_dev[NR] = {};   // G6_XTRACE GPU records per simulated rank (null = off)
    std::vector<Prefill> PF(NR), PF2(NR);                // PF2: the second lane (--pf-lanes 2), its own loopback region
    const bool pf_rs = getenv("G6_PF_RS") && atoi(getenv("G6_PF_RS")) == 1;     // reduce-scatter form (prefill.cuh)
    const int pf_own = getenv("G6_PF_OWN") ? atoi(getenv("G6_PF_OWN")) : 0;   // 1: row-owner projections, 2: + index (prefill.cuh)
    const bool pf_qo = MP && pf_own == 3;                                        // 3: row-owner QO attention (QO-INTEGRATION.md)
    const bool pf_qp = MP && pf_own == 4;                                        // 4: pair-owner QO attention (TP4, qp.cuh)
    if (pf_qp && NR != 4) die("G6_PF_OWN=4 is TP4 only (G6_NR=4)");
#if G6_NR == 4
    std::vector<std::vector<QpW>> qpw(NR, std::vector<QpW>(layers.size()));   // pf_qp: per rank, assembled from the packs
    const std::vector<char> qpl = pf_qp ? pf_qp_mask(layers) : std::vector<char>(layers.size(), 0);
    std::string qre;
    const int qp_roll_s = env_choice("G6_PF_QP_ROLL", 4, 0, &qre);       // rolling pack (pf_env_check made it 0 or 2..4)
    const bool qp_roll_ser = env_choice("G6_PF_QP_ROLL_SER", 1, 0, &qre) == 1;
    if (qp_roll_s && !pf_qp) die("G6_PF_QP_ROLL needs G6_PF_OWN=4 and --prefill");
    std::vector<QpRoll*> qp_roll(NR, nullptr);
#endif
    std::vector<QoW> qow;                                                       // pf_qo: full-head weights per model layer
    g6::Pack qo_pack;                                                           // --qo-pack (else assembled from the six packs)
    if ((qo_man.empty()) != (qo_file.empty())) die("--qo-manifest and --qo-pack go together");
    const int NCP = pf_rs ? 5 : 2;
    const size_t stride_pf = (pf_slot_bytes(MP, pf_rs, NR) + 8 + 255) / 256 * 256;
    char* recv_pf = MP ? dmalloc<char>(2 * (size_t)NCP * NR * stride_pf) : nullptr;
    u64* stepp_pf = dmalloc<u64>(1);
    unsigned long long* doorbell_pf = dmalloc<unsigned long long>(NCP);
    if (lanes != 1 && lanes != 2) die("--pf-lanes must be 1 or 2");
    const bool two = MP && lanes == 2;
    char* recv_pf2 = two ? dmalloc<char>(2 * (size_t)NCP * NR * stride_pf) : nullptr;
    u64* stepp_pf2 = dmalloc<u64>(1);
    unsigned long long* doorbell_pf2 = dmalloc<unsigned long long>(NCP);
    Loop loop_pf2{};
    // MTP draft pass: a third loopback region (3 collectives, decode-sized slots), its own step cell and step blocks
    const int NCD = 3;
    char* recv_d = mtp ? dmalloc<char>(2 * (size_t)NCD * NR * stride) : nullptr;
    u64* stepp_d = dmalloc<u64>(1);
    unsigned long long* doorbell_d = dmalloc<unsigned long long>(NCD);
    StepIn* in_d = dmalloc<StepIn>(1);
    StepOut* out_d = dmalloc<StepOut>(NR);
    std::vector<Draft> DR(mtp ? NR : 0);
    std::vector<DraftPrefill> DQ(mtp && MP ? NR : 0);
    Loop loop_d{};
    if (pf_qo)                          // the full-head weights need every rank's slices: load the six packs first
    {
        for (int r = 0; r < NR; ++r) rload(r);
        std::vector<Find> finds;
        for (int r = 0; r < NR; ++r) finds.push_back(rfind(r));
        if (!qo_man.empty()) load_pack(qo_pack, qo_man, qo_file, 0);
        const Find qfind = pack_find(qo_pack, store);
        for (int l : layers)
        {
            QoW w;
            if (!qo_man.empty()) { if (!qo_from_pack(qfind, l, w)) die("--qo-pack has no full-head tensors for layer " + std::to_string(l)); }
            else w = qo_from_ranks(finds, l);
            qow.push_back(w);
        }
        fprintf(stderr, "# G6_PF_OWN=3: full-head q_b / o_proj / UK / UV for %zu layers\n", qow.size());
    }
#if G6_NR == 4
    if (pf_qp)                          // the pair weights need both pair members' slices: load the four packs first
    {
        for (int r = 0; r < NR; ++r) rload(r);
        std::vector<Find> finds;
        for (int r = 0; r < NR; ++r) finds.push_back(rfind(r));
        int n = 0;
        for (size_t k = 0; k < layers.size(); ++k)
        {
            if (!qpl[k]) continue;
            if (!qp_roll_s) for (int r = 0; r < NR; ++r) qpw[r][k] = qp_from_ranks(finds, r, layers[k]);
            ++n;
        }
        if (qp_roll_s)                  // S slots per rank, built from the pair members' slices during the prefill
            for (int r = 0; r < NR; ++r) qp_roll[r] = qp_roll_new(qp_roll_s, qp_roll_ser, finds, r, layers, qpl, qpw[r]);
        fprintf(stderr, "# G6_PF_OWN=4: pair q_b / o_proj / UK / UV for %d of %zu layers (G6_PF_QP_LAYERS), assembled from the packs%s\n",
                n, layers.size(), qp_roll_s ? " (rolling)" : "");
    }
#endif
    for (int r = 0; r < NR; ++r)
    {
        if (!pf_qo && !pf_qp) rload(r);
        g6::FabricDev f{};
        f.send = dmalloc<char>(2 * (size_t)NC * stride);
        f.recv = recv; f.doorbell = doorbell; f.error = err;
        f.rank = r; f.nr = NR; f.ncoll = NC; f.stride = stride;
        f.xt = g6::xt_alloc(NC);        // G6_XTRACE=1: stage-0 GPU stamps on the decode fabric (xtrace.h); null = off
        xt_dev[r] = f.xt;
        R[r].wide = wide;
#if defined(G6_CP) && G6_CP
        R[r].cp.on = cp_on; R[r].cp.R = cp_R; R[r].cp.s3 = cp_s3; R[r].cp.colls = cp_colls;
#endif
#ifdef DEC_S2
        if (s2mode && s2_below) R[r].s2_below = s2_below;       // CP: S2 below R on the CP binding as built
        else if (s2mode) R[r].s2_pool_slots = (int)prompts.size();       // the layers' own bindings hold the S2 pool
        if (s2mode && !s2_below && getenv("G6_S2_POOL")) R[r].s2_pool_pages = atoi(getenv("G6_S2_POOL"));   // pool pages (variable runs)
#endif
        build(R[r], r, layers, rfind(r), cap, f, stepp, st, pf_attn_rows(MP));
#if defined(G6_CP) && G6_CP
        if (cp_on)
            for (int c = 0; c < cp_colls.ncoll(); ++c) R[r].cp.vs.push_back(cp_L.vslot(c, r, cp_send[r], cp_recv[r], cp_bell[r], err));
        if (cp_on && MP) cp_pf_alloc(R[r], st);     // step 4: prefill staging pools
#endif
        R[r].graph = graph; R[r].in = in; R[r].out = out + r;
        loop.send[r] = f.send;
        if (MP)
        {
            g6::FabricDev fp = f;
            fp.xt = nullptr;
            fp.send = dmalloc<char>(2 * (size_t)NCP * stride_pf);
            fp.recv = recv_pf; fp.doorbell = doorbell_pf; fp.ncoll = NCP; fp.stride = stride_pf;
            PF[r].v2 = pfv2;
            PF[r].loop = PF2[r].loop = true;
            PF[r].own = PF2[r].own = pf_own >= 1;
            PF[r].ownix = PF2[r].ownix = pf_own == 2 || pf_qp;
            PF[r].qo = PF2[r].qo = pf_qo;
            PF[r].qow = PF2[r].qow = &qow;
#if G6_NR == 4
            PF[r].qp = PF2[r].qp = pf_qp;
            PF[r].qpw = PF2[r].qpw = &qpw[r];
            PF[r].qpl = PF2[r].qpl = qpl;
            PF[r].roll = PF2[r].roll = qp_roll[r];
#endif
            PF[r].attn_rows = pf_attn_rows(MP);
            build_prefill(PF[r], R[r], rfind(r), MP, fp, stepp_pf, st);
            loop_pf.send[r] = fp.send;
            if (two)
            {
                g6::FabricDev f2 = fp;
                f2.send = dmalloc<char>(2 * (size_t)NCP * stride_pf);
                f2.recv = recv_pf2; f2.doorbell = doorbell_pf2;
                PF2[r].attn_rows = pf_attn_rows(MP);
                build_prefill(PF2[r], R[r], rfind(r), MP, f2, stepp_pf2, st, &PF[r]);
                loop_pf2.send[r] = f2.send;
            }
        }
        if (mtp)
        {
            g6::FabricDev fd = f;
            fd.xt = nullptr;
            fd.send = dmalloc<char>(2 * (size_t)NCD * stride);
            fd.recv = recv_d; fd.doorbell = doorbell_d; fd.ncoll = NCD;
#ifdef DEC_S2
            if (s2mode && s2_below) DR[r].d.s2_below = s2_below;
            else if (s2mode) DR[r].d.s2_pool_slots = (int)prompts.size();
            if (s2mode && !s2_below && getenv("G6_S2_POOL")) DR[r].d.s2_pool_pages = atoi(getenv("G6_S2_POOL"));
#endif
            build_draft(DR[r], R[r], rfind(r), cap, fd, stepp_d, st, MP ? pf_attn_rows(MP) : MX);
            DR[r].d.graph = graph; DR[r].d.in = in_d; DR[r].d.out = out_d + r;
            loop_d.send[r] = fd.send;
            if (MP) DQ[r].P.attn_rows = pf_attn_rows(MP);   // DQ is empty without --prefill (was an out-of-bounds write)
            if (MP) build_draft_prefill(DQ[r], DR[r], rfind(r), MP, PF[r].f, pfv2, st, &PF[r]);
        }
        G6RT_CK(cudaStreamSynchronize(st));
    }
    for (Model& m : R) { m.nc = nc; m.rs = plan.rs; }   // before any capture: the head kernels take them by value
    for (Draft& D : DR) D.d.nc = nc;
#ifdef DEC_S2
    S2In* in_s2 = dmalloc<S2In>(1);     // --s2: the step's session layout, target and draft steps
    S2In* in_s2d = dmalloc<S2In>(1);
    if (s2mode)
    {
        for (Model& m : R) { s2_init(m, (int)plans.size(), st); m.s2->in = in_s2; }
        for (Draft& D : DR) { s2_init(D.d, (int)plans.size(), st); D.d.s2->in = in_s2d; }
        // G6_S2_RUNS=b:p,b:p,..: slot q's run (base page, pages) on every model (variable runs; default: q x run, full run)
        if (const char* e = getenv("G6_S2_RUNS"))
        {
            std::string r = e;
            for (char& c : r) if (c == ':') c = ',';
            const std::vector<int> v = parse_ints(r);
            if (v.size() % 2 || (int)v.size() / 2 > (int)plans.size()) die("G6_S2_RUNS: base:pages per slot");
            for (size_t k = 0; k + 1 < v.size(); k += 2)
            {
                for (Model& m : R) s2_set_pages(m, (int)k / 2, s2_run_list(v[k], v[k + 1]));
                for (Draft& D : DR) s2_set_pages(D.d, (int)k / 2, s2_run_list(v[k], v[k + 1]));
            }
        }
    }
#endif
    // Gate: DEC_S2 CP build, --prefill, no --s2: G6_S2_PFBASE=B[,pages] runs every sequence as one session
    // at base page B below R: tables B + i (s2_solo_pages, main and draft), staged prefill index keys through them
    // (Model::s2_pf_table), StepIn::page_bound = the pool's pages, and s2_reset_pages clears the run. Stdout must equal
    // the run at B = 0.
    // G6_S2_PFPOISON=1: fill every cache with 0x3c bytes first, so a read outside the run changes the output.
    int pf_base = 0;
#if defined(DEC_S2) && defined(G6_CP) && G6_CP
    int pf_pages = 0;
    if (const char* e = getenv("G6_S2_PFBASE"))
    {
        const std::vector<int> v = parse_ints(e);
        if (s2mode || !cp_on || !MP || v.empty() || v.size() > 2 || v[0] < 0) die("G6_S2_PFBASE=B[,pages]: needs G6_CP on, --prefill, no --s2");
        pf_base = v[0];
        const int below = getenv("G6_S2_BELOW") ? atoi(getenv("G6_S2_BELOW")) : cp_R;
        if (below < 64 || below % 64 || below > cp_R) die("G6_S2_PFBASE: G6_S2_BELOW (default G6_CP_R) must be 64..G6_CP_R, a multiple of 64");
        for (Model& m : R) { m.s2_below = below; s2_init(m, 1, st); }
        for (Draft& D : DR) { D.d.s2_below = below; s2_init(D.d, 1, st); }
        pf_pages = v.size() > 1 ? v[1] : std::min(R[0].s2->run_pages, R[0].s2->pool_pages - pf_base);
        for (const Draft& D : DR) if (v.size() < 2) pf_pages = std::min(pf_pages, D.d.s2->pool_pages - pf_base);
        if (getenv("G6_S2_PFPOISON"))
        {
            auto poison = [&](Model& m) {
                for (const Layer& y : m.layers)
                {
                    const G6Attn& a = y.ab.args;
                    G6RT_CK(cudaMemsetAsync(a.mla_cache, 0x3c, (size_t)a.page_count * 64 * 656, st));
                    if (a.index_cache) G6RT_CK(cudaMemsetAsync(a.index_cache, 0x3c, (size_t)std::max(a.page_count, a.table_size) * 64 * 132, st));
                }
            };
            for (Model& m : R) poison(m);
            for (Draft& D : DR) poison(D.d);
        }
        // G6_S2_PFNOFIX=1 (negative control): tables at B but staged index keys at logical pages, from page 0; must differ
        const bool ixt = !getenv("G6_S2_PFNOFIX");
        for (Model& m : R) { s2_solo_pages(m, s2_run_list(pf_base, pf_pages)); m.s2on = false; m.s2_pf_table = ixt; }
        for (Draft& D : DR) { s2_solo_pages(D.d, s2_run_list(pf_base, pf_pages)); D.d.s2on = false; D.d.s2_pf_table = ixt; }
        G6RT_CK(cudaStreamSynchronize(st));
        fprintf(stderr, "# s2 base-page gate: every sequence at base page %d, run of %d pages (main pool %d pages)%s%s\n", pf_base, pf_pages,
                R[0].s2->pool_pages, getenv("G6_S2_PFPOISON") ? ", caches poisoned 0x3c" : "", ixt ? "" : ", NOFIX");
    }
#endif
    // StepIn to the device; G6_S2_PFBASE: the pool's page bound
    auto put_in = [&](StepIn* dst, const StepIn& si) {
        StepIn x = si;
#ifdef DEC_S2
        if (pf_base) x.page_bound = (dst == in ? R[0] : DR[0].d).s2->pool_pages;
#endif
        G6RT_CK(g6h2d::copy(dst, &x, sizeof x));
    };
    loop_d.recv = recv_d; loop_d.stepp = stepp_d; loop_d.ncoll = NCD; loop_d.stride = stride;
    loop.recv = recv; loop.stepp = stepp; loop.ncoll = NC; loop.stride = stride;
    loop_pf.recv = recv_pf; loop_pf.stepp = stepp_pf; loop_pf.ncoll = NCP; loop_pf.stride = stride_pf;
    loop_pf2.recv = recv_pf2; loop_pf2.stepp = stepp_pf2; loop_pf2.ncoll = NCP; loop_pf2.stride = stride_pf;
    if (MP && pf_rs) fprintf(stderr, "# prefill reduce-scatter form (5 collectives, slice %zu B)%s\n", pf_ss(PF[0].f),
                             pf_own >= 4 ? ", pair-owner QO attention (G6_PF_OWN=4)" : pf_own >= 3 ? ", row-owner QO attention (G6_PF_OWN=3)" : pf_own >= 2 ? ", row-owner projections + index (G6_PF_OWN=2)" : pf_own ? ", row-owner projections (G6_PF_OWN)" : "");
#ifdef G6_RS_LOADER
    if (release) { store.bytes = pstore.bytes; store.shared_bytes = pstore.shared_bytes; }
#endif
    fprintf(stderr, "# device tensors: %.2f GB held, %.2f GB shared between ranks\n", store.bytes / 1e9, store.shared_bytes / 1e9);
    printf("# m4 layers %s, %zu sequences, chunks of %d, gen %d temp %g seed %u\n", layers_s.c_str(), plans.size(), plan.chunk, plan.n_gen, temp, plan.seed);

    auto deliver = [&](int c, size_t payload) {
        Loop a = loop;
        a.c = c; a.n16 = (int)(payload / 16);
        k_deliver<<<NR, 128, 0, st>>>(a);
    };
#if defined(G6_CP) && G6_CP
    auto vdeliver = [&](int c, int M) {
        VLoop a{};
        a.stepp = stepp; a.c = c; a.ncoll = cp_colls.ncoll(); a.bytes = cp_colls.sizes(M)[c];
        for (int s = 0; s < NR; ++s)
        {
            for (int par = 0; par < 2; ++par) a.src[s][par] = cp_send[s] + cp_L.send_at(c, par);
            for (int p = 0; p < NR; ++p)
            {
                const int j = cp_L.slot_of(c, p, s);
                if (j < 0) continue;
                for (int par = 0; par < 2; ++par) a.dst[s][a.ndst[s]][par] = cp_recv[p] + cp_L.recv_at(c, par, j);
                a.ndst[s]++;
            }
        }
        k_vdeliver<<<NR, 128, 0, st>>>(a);
    };
    // layer i's attention block on every rank: the CP block in its three phases with the loopback between them
    auto attn_all = [&](int i, int M, int context) {
        if (!cp_active(R[0], context)) { for (Model& m : R) launch_attn_block(m, i, M, context, st); return; }
        for (Model& m : R) launch_attn_cp(m, i, M, context, 0, st);
        vdeliver(cp_colls.q[i], M);
        if (cp_colls.cand[i] >= 0) vdeliver(cp_colls.cand[i], M);
        for (Model& m : R) launch_attn_cp(m, i, M, context, 1, st);
        for (int k = 1; k < NR; ++k) vdeliver(cp_colls.part[(size_t)i * (NR - 1) + k - 1], M);
        for (Model& m : R) launch_attn_cp(m, i, M, context, 2, st);
    };
#else
    auto attn_all = [&](int i, int M, int context) { for (Model& m : R) launch_attn_block(m, i, M, context, st); };
#endif
    // triage: G6_RESID_DUMP=FILE — rank 0's residual stream after the embedding and after every consume c
    // (slot c + 1: 2i = after layer i's attention, 2i + 1 = after its MLP), per main step: int32 s, M, slots, HID, then
    // slots x M x HID float32. tools/cp/gateB_layers.py compares two dumps per sublayer.
    const bool rd_on = getenv("G6_RESID_DUMP") != nullptr;
    const int rd_slots = 2 * NL + 1;
    float* rdump = rd_on ? dmalloc<float>((size_t)rd_slots * MX * HID) : nullptr;
    auto rd_copy = [&](int slot, int M) {
        if (rdump) G6RT_CK(cudaMemcpyAsync(rdump + (size_t)slot * MX * HID, R[0].resid, (size_t)M * HID * 4, cudaMemcpyDeviceToDevice, st));
    };
    // noise floor: G6_PERTURB=eps (replicated runs): after layer 0's attention consume, every rank's residual
    // x *= 1 + eps * u, u in [-1, 1) from a hash of (position row, element) — the same on every rank, deterministic.
    const float perturb = getenv("G6_PERTURB") ? (float)atof(getenv("G6_PERTURB")) : 0.f;
    // one step of the six ranks, M rows, in stream order
    // DEC_GSPLIT (G6_DEC_SPLIT=ls): pc 0 = step_in, embed, layers [0, ls); pc 1 = layers [ls, NL), head, step_out (-1 = all)
    auto enqueue = [&](int M, int context, int pc = -1, int ls = 0) {
        const size_t part = (size_t)M * HID * 2;
        if (pc != 1) { for (Model& m : R) { launch_step_in(m, st); launch_embed(m, M, st); } rd_copy(0, M); }
        for (int i = pc == 1 ? ls : 0; i < (pc == 0 ? ls : NL); ++i)
        {
            attn_all(i, M, context);
            deliver(2 * i, part);
            for (Model& m : R) launch_consume(m, 2 * i, M, m.layers[i].mlp_norm, m.x2, st);
            if (perturb != 0.f && i == 0) for (Model& m : R) k_perturb<<<M, 256, 0, st>>>(m.resid, m.pos, perturb);
            rd_copy(2 * i + 1, M);
            for (Model& m : R) launch_mlp_block(m, i, M, st);
            deliver(2 * i + 1, part);
            for (Model& m : R) launch_consume(m, 2 * i + 1, M, norm_after(m, i), m.x, st);
            rd_copy(2 * i + 2, M);
        }
        if (pc == 0) return;
        for (Model& m : R) launch_head_local(m, M, st);
        deliver(2 * NL, glue::head_bytes(M, nc));
        for (Model& m : R) { launch_head_merge(m, M, st); launch_step_out(m, M, st); }
    };

    // one prefill sub-batch of the six ranks, M rows, layer i as layer-step ls0 + i
    // loopback delivery of prefill partial c (0 attention, 1 MLP) and the consume of every rank; RS form: RS, owner
    // rows, AG delivery, gather
    auto deliver_l = [&](const Loop& l, int c, int M, int mode, int row_bytes) {
        Loop a = l;
        a.c = c; a.n16 = M * (HID * 2 / 16); a.mode = mode; a.M = M; a.row_bytes = row_bytes;
        k_deliver<<<NR, 128, 0, st>>>(a);
    };
    auto pf_deliver_consume = [&](std::vector<Prefill>& L, const Loop& l, int c, int M, const std::function<const __nv_bfloat16*(Model&)>& w,
                                  __half* Prefill::*out) {
        if (!pf_rs)
        {
            for (Model& m : R) pf_consume(L[m.rank], c, M, w(m), L[m.rank].*out, st);
            return;
        }
        if (pf_qo && c == 0)            // the owners normed and published c1 in their attention block
        {
            deliver_l(l, 1, M, 2, HID * 2);
            for (Model& m : R) pf_consume_ag(L[m.rank], 0, M, L[m.rank].*out, st);
            return;
        }
        for (Model& m : R) pf_consume_rs(L[m.rank], c, M, w(m), L[m.rank].*out, st);
        deliver_l(l, 2 * c + 1, M, 2, HID * 2);
        for (Model& m : R) pf_consume_ag(L[m.rank], c, M, L[m.rank].*out, st);
    };
    // the consume of layer i's MLP partial (c2 / c3 in the RS form; G6_PF_OWN: c3 carries the owner records)
    // (G6_PF_OWN=2: then the owner IDs of full-index layer i + 1 on c4)
    auto pf_deliver_consume_x = [&](std::vector<Prefill>& L, const Loop& l, int i, int M, int ctx) {
        if (!pf_rs)
        {
            for (Model& m : R) pf_consume(L[m.rank], 1, M, norm_after(m, i), L[m.rank].x, st);
            return;
        }
        for (Model& m : R) pf_consume_x_rs(m, L[m.rank], i, M, st);
        deliver_l(l, 3, M, 2, pf_x_row_bytes(R[0], L[0], i));
        for (Model& m : R) pf_consume_x_ag(m, L[m.rank], i, M, st);
        if (!pf_ix_next(R[0], L[0], i)) return;
        for (Model& m : R) pf_ix_score(m, L[m.rank], i, M, ctx, st);
        deliver_l(l, 4, M, 2, IX_ROW);
        for (Model& m : R) pf_ix_gather(m, L[m.rank], i, M, st);
    };
    auto pf_deliver_part = [&](const Loop& l, int c, int M) { if (!(pf_qo && c == 0)) deliver_l(l, pf_rs ? 2 * c : c, M, pf_rs ? 1 : 0, 0); };
    auto pf_fin_rs = [&](std::vector<Prefill>& L, const Loop& l, int M) {
        if (!pf_rs) return;
        for (Model& m : R) pf_finish_put(L[m.rank], M, st);
        deliver_l(l, 4, M, 2, HID * 4);
        for (Model& m : R) pf_finish_gather(L[m.rank], M, st);
    };
#if defined(G6_CP) && G6_CP
    // step 4 (prefill.cuh cp_pf_*): staged view around a prefill sub-batch; gather layer i before its first attention
    // block, copy the rows back after its last
    const bool cp_pf = cp_on && MP;
    // negative control (gate4.sh): G6_CP_PF_BREAK=gather|back skips that half of the staging (the gate must then FAIL)
    const std::string cp_break = getenv("G6_CP_PF_BREAK") ? getenv("G6_CP_PF_BREAK") : "";
    auto cp_enter = [&]() { if (cp_pf) for (Model& m : R) cp_pf_enter(m); };
    auto cp_exit = [&]() { if (cp_pf) for (Model& m : R) cp_pf_exit(m); };
    auto cp_gather = [&](int i, int start) {
        if (!cp_pf || cp_break == "gather") return;
        const CpPfSrc s = cp_pf_src(R, i);
        for (Model& m : R) cp_pf_gather(m, s, i, start, st);
    };
    auto cp_back = [&](int i, int a0, int n) { if (cp_pf && cp_break != "back") for (Model& m : R) cp_pf_back(m, i, a0, n, st); };
#else
    auto cp_enter = [&]() {};
    auto cp_exit = [&]() {};
    auto cp_gather = [&](int, int) {};
    auto cp_back = [&](int, int, int) {};
#endif
    auto enqueue_pf = [&](int M, int context, u64 ls0) {
        cp_enter();
        for (Model& m : R) pf_embed(m, PF[m.rank], M, st);
        for (int i = 0; i < NL; ++i)
        {
            pf_set_step(PF[0], ls0 + i, st);
            cp_gather(i, context - M);
            for (Model& m : R) pf_attn_block(m, PF[m.rank], i, M, context, st);
            cp_back(i, context - M, M);
            pf_deliver_part(loop_pf, 0, M);
            pf_deliver_consume(PF, loop_pf, 0, M, [&](Model& m) { return m.layers[i].mlp_norm; }, &Prefill::x2);
            for (Model& m : R) pf_mlp_block(m, PF[m.rank], i, M, st);
            pf_deliver_part(loop_pf, 1, M);
            pf_deliver_consume_x(PF, loop_pf, i, M, context);
        }
        pf_fin_rs(PF, loop_pf, M);
        cp_exit();
    };
    // two sub-batches as two lanes, in pf_step2's order (prefill.cuh), the six ranks stage by stage
    auto enqueue_pf2 = [&](int MA, int MB, int ctxA, int ctxB, u64 lsA, u64 lsB) {
        auto attn = [&](std::vector<Prefill>& L, const Loop& l, int i, int M, int ctx, u64 ls) {
            pf_set_step(L[0], ls, st);
            for (Model& m : R) pf_attn_block(m, L[m.rank], i, M, ctx, st);
            pf_deliver_part(l, 0, M);
        };
        auto mlp = [&](std::vector<Prefill>& L, const Loop& l, int i, int M) {
            pf_deliver_consume(L, l, 0, M, [&](Model& m) { return m.layers[i].mlp_norm; }, &Prefill::x2);
            for (Model& m : R) pf_mlp_block(m, L[m.rank], i, M, st);
            pf_deliver_part(l, 1, M);
        };
        auto fin = [&](std::vector<Prefill>& L, const Loop& l, int i, int M, int ctx) {
            pf_deliver_consume_x(L, l, i, M, ctx);
        };
        cp_enter();
        for (Model& m : R) pf_embed(m, PF[m.rank], MA, st);
        pf_stagger(st);                  // G6_PF_STAGGER: one stream here, so only proves the delay changes no output
        for (Model& m : R) pf_embed(m, PF2[m.rank], MB, st);
        for (int i = 0; i < NL; ++i)
        {
            cp_gather(i, std::min(ctxA - MA, ctxB - MB));
            attn(PF, loop_pf, i, MA, ctxA, lsA + i);
            if (i > 0) fin(PF2, loop_pf2, i - 1, MB, ctxB);
            attn(PF2, loop_pf2, i, MB, ctxB, lsB + i);
            cp_back(i, ctxA - MA, MA);
            cp_back(i, ctxB - MB, MB);
            mlp(PF, loop_pf, i, MA);
            mlp(PF2, loop_pf2, i, MB);
            fin(PF, loop_pf, i, MA, ctxA);
        }
        fin(PF2, loop_pf2, NL - 1, MB, ctxB);
        pf_fin_rs(PF, loop_pf, MA);
        pf_fin_rs(PF2, loop_pf2, MB);
        cp_exit();
    };
    u64 ls0 = 1, ls0b = 1;

    cudaGraphExec_t exec[NBK][MX + 1] = {};  // [context bucket][M] (rank.cuh Model::bk)
    bool ready[NBK][MX + 1] = {};
    // DEC_GSPLIT: G6_DEC_SPLIT=ls (1..NL-1): exec = step_in, embed, layers [0, ls), exec2 = the rest, launched back to back
    const int gsplit = getenv("G6_DEC_SPLIT") ? atoi(getenv("G6_DEC_SPLIT")) : 0;
    if (gsplit < 0 || gsplit >= NL || (gsplit && !graph)) die("G6_DEC_SPLIT must be 0..layers-1 and needs --graph");
    cudaGraphExec_t exec2[NBK][MX + 1] = {};
    auto launch_main = [&](int k, int M) {
        G6RT_CK(cudaGraphLaunch(exec[k][M], st));
        if (gsplit) G6RT_CK(cudaGraphLaunch(exec2[k][M], st));
    };
    // the context bucket of a step; set as the capturing bucket of every rank (and draft) model
    auto bucket = [&](int context, bool draft) {
        const int k = draft ? bucket_of(DR[0].d, context) : bucket_of(R[0], context);
        for (Model& m : R) m.cur = k;
        for (Draft& D : DR) D.d.cur = k;
        return k;
    };
    int bad = 0, steps = 0;
    u64 step0 = 0;
    std::vector<double> t1;             // single-row step times

    // ---- MTP (--mtp K): one draft step of the six ranks (rank.cuh "MTP draft pass"), and the step runners
    auto enqueue_draft = [&](int M, int context, bool chain) {
        auto deliver_d = [&](int c, size_t payload) {
            Loop a = loop_d;
            a.c = c; a.n16 = (int)(payload / 16);
            k_deliver<<<NR, 128, 0, st>>>(a);
        };
        const size_t part = (size_t)M * HID * 2;
        for (Draft& D : DR) { launch_step_in(D.d, st); launch_draft_embed(D, M, chain, st); }
        for (Draft& D : DR) launch_attn_block(D.d, 0, M, context, st);
        deliver_d(0, part);
        for (Draft& D : DR) launch_consume(D.d, 0, M, D.d.layers[0].mlp_norm, D.d.x2, st);
        for (Draft& D : DR) launch_mlp_block(D.d, 0, M, st);
        deliver_d(1, part);
        for (Draft& D : DR) launch_consume(D.d, 1, M, D.d.final_norm, D.d.x, st);
        for (Draft& D : DR) launch_head_local(D.d, M, st);
        deliver_d(2, glue::head_bytes(M, nc));
        for (Draft& D : DR) { launch_head_merge(D.d, M, st); launch_step_out(D.d, M, st); launch_draft_keep(D, M, st); }
    };
    cudaGraphExec_t exec_d[NBK][2][MX + 1] = {};
    bool ready_d[NBK][2][MX + 1] = {};
    auto capture = [&](const std::function<void()>& body, cudaGraphExec_t* ex, const char* what, int M) {
        cudaGraph_t g;
        G6RT_CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        body();
        G6RT_CK(cudaStreamEndCapture(st, &g));
        size_t nodes = 0;
        G6RT_CK(cudaGraphGetNodes(g, nullptr, &nodes));
        G6RT_CK(cudaGraphInstantiate(ex, g, 0));
        G6RT_CK(cudaGraphDestroy(g));
        fprintf(stderr, "# %s graph for M=%d: %zu nodes (six ranks + loopback)\n", what, M, nodes);
    };
    auto check_fabric = [&](const char* what) {
        unsigned e;
        G6RT_CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
        if (e) { fprintf(stderr, "m4sim: fabric error word %08x in a %s step\n", e, what); exit(3); }
    };
    // runs one target step; so = the six StepOut blocks. Returns the wall time in ms.
    FILE* rlf = route_log_open(R[0]);     // G6_ROUTE_LOG=<file> (measurement, MTP main steps, rank 0)
    auto run_main = [&](const StepIn& si, StepOut* so) {
        const int M = si.M, k = bucket(si.context, false);
        if (!ready[k][M])
        {
            for (Model& m : R) prepare(m, M, st);
            G6RT_CK(cudaStreamSynchronize(st));
            if (graph && gsplit)
            {
                capture([&] { enqueue(M, si.context, 0, gsplit); }, &exec[k][M], "target split part A", M);
                capture([&] { enqueue(M, si.context, 1, gsplit); }, &exec2[k][M], "target split part B", M);
            }
            else if (graph) capture([&] { enqueue(M, si.context); }, &exec[k][M], "target", M);
            ready[k][M] = true;
        }
        put_in(in, si);
        const auto t0 = std::chrono::steady_clock::now();
        if (graph) launch_main(k, M);
        else enqueue(M, si.context);
        G6RT_CK(cudaStreamSynchronize(st));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        check_fabric("target");
        G6RT_CK(cudaMemcpy(so, out, NR * sizeof(StepOut), cudaMemcpyDeviceToHost));
        return ms;
    };
    auto run_draft = [&](const StepIn& si, bool chain, StepOut* so) {
        const int M = si.M, k = bucket(si.context, true);
        if (!ready_d[k][chain][M])
        {
            for (Draft& D : DR) prepare(D.d, M, st);
            G6RT_CK(cudaStreamSynchronize(st));
            if (graph) capture([&] { enqueue_draft(M, si.context, chain); }, &exec_d[k][chain][M], chain ? "chain draft" : "first draft", M);
            ready_d[k][chain][M] = true;
        }
        put_in(in_d, si);
        const auto t0 = std::chrono::steady_clock::now();
        if (graph) G6RT_CK(cudaGraphLaunch(exec_d[k][chain][M], st));
        else enqueue_draft(M, si.context, chain);
        G6RT_CK(cudaStreamSynchronize(st));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        check_fabric("draft");
        G6RT_CK(cudaMemcpy(so, out_d, NR * sizeof(StepOut), cudaMemcpyDeviceToHost));
        return ms;
    };
    // prints rank 0's line of a step (prefix replaces "step"), counts ranks that differ; returns the last row's token
    // G6_LOGIT_DUMP=FILE appends, per main step s >= G6_LOGIT_DUMP_FROM (default 0): int32 s, int32 M,
    // then per rank int32 vn + M x vn float32 head logits (its vocab slice). tools/cp/gateB.py compares two dumps.
    FILE* ldump = getenv("G6_LOGIT_DUMP") ? fopen(getenv("G6_LOGIT_DUMP"), "wb") : nullptr;
    if (getenv("G6_LOGIT_DUMP") && !ldump) die("G6_LOGIT_DUMP: cannot open the file");
    const int ldump_from = getenv("G6_LOGIT_DUMP_FROM") ? atoi(getenv("G6_LOGIT_DUMP_FROM")) : 0;
    FILE* rdf = rd_on ? fopen(getenv("G6_RESID_DUMP"), "wb") : nullptr;
    if (rd_on && !rdf) die("G6_RESID_DUMP: cannot open the file");
    FILE* idf = getenv("G6_IDX_DUMP") ? fopen(getenv("G6_IDX_DUMP"), "wb") : nullptr;
    auto logit_dump = [&](int s, int M) {
        if (rdf && s >= ldump_from)
        {
            std::vector<float> h((size_t)rd_slots * M * HID);
            for (int k = 0; k < rd_slots; ++k)
                G6RT_CK(cudaMemcpy(h.data() + (size_t)k * M * HID, rdump + (size_t)k * MX * HID, (size_t)M * HID * 4, cudaMemcpyDeviceToHost));
            const int hdr[4] = {s, M, rd_slots, HID};
            fwrite(hdr, 4, 4, rdf); fwrite(h.data(), 4, h.size(), rdf);
        }
        if (idf && s >= ldump_from)   // G6_IDX_DUMP: rank 0's selected ids of the first --layers layer, int32 s, M, M x 2048
        {
            std::vector<int> h((size_t)M * 2048);
            G6RT_CK(cudaMemcpy(h.data(), R[0].layers[0].ab.args.indices, h.size() * 4, cudaMemcpyDeviceToHost));
            fwrite(&s, 4, 1, idf); fwrite(&M, 4, 1, idf); fwrite(h.data(), 4, h.size(), idf);
        }
        if (!ldump || s < ldump_from) return;
        fwrite(&s, 4, 1, ldump); fwrite(&M, 4, 1, ldump);
        for (Model& m : R)
        {
            std::vector<float> lg((size_t)M * m.vn);
            G6RT_CK(cudaMemcpy(lg.data(), m.hlog, lg.size() * 4, cudaMemcpyDeviceToHost));
            fwrite(&m.vn, 4, 1, ldump); fwrite(lg.data(), 4, lg.size(), ldump);
        }
    };
    auto lines = [&](const char* prefix, bool draft, const StepOut* so, int s, int M) {
        std::string line0;
        int last = 0;
        for (int r = 0; r < NR; ++r)
        {
            Model& m = draft ? DR[r].d : R[r];
            if (so[r].attn_err) { fprintf(stderr, "m4sim: attention error word %08x, rank %d, %s %d\n", so[r].attn_err, r, prefix, s); exit(3); }
            std::string line;
            const int t = step_line(m, so[r], s, M, line);
            line = prefix + line.substr(4);
            if (r == 0) { line0 = line; last = t; }
            else if (line != line0) { ++bad; printf("RANK %d DIFFERS: %s\n", r, line.c_str()); }
        }
        printf("%s\n", line0.c_str());
        return last;
    };
    u64 dstep = 0;
    auto dumpf = [&](const char* name, const void* dev, size_t bytes) {
        std::vector<char> h(bytes);
        G6RT_CK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
        FILE* fo = fopen((mtp_dump + "." + name).c_str(), "wb");
        if (!fo || fwrite(h.data(), 1, bytes, fo) != bytes) die("cannot write the --mtp-dump file " + std::string(name));
        fclose(fo);
    };
    MtpStats mtp_stats;
    std::vector<int> hist;              // --reuse-prefix: the tokens of the previous sequence, prompt + generated
    Emitter em;
    PfTimes pt;                         // G6_PF_TIMES=1 (serve.h)
#ifdef DEC_S2
    if (s2mode)                         // --s2: all prompts as sessions of one batched decode
    {
        for (Model& m : R) { reset(m, st); s2_reset(m, st); }
        for (Draft& D : DR) { reset(D.d, st); s2_reset(D.d, st); }
        G6RT_CK(cudaStreamSynchronize(st));
        const int NSS = (int)plans.size();
        std::vector<s2::Session> SS;
        for (const Plan& p : plans)
        {
            s2::Session x;
            x.prompt = p.prompt; x.chunk = p.chunk; x.n_gen = p.n_gen;
            SS.push_back(x);
            printf("q%zu seq 0 prompt %zu tokens\n", SS.size() - 1, p.prompt.size());
        }
        int solo = -1, solo_base = 0;    // S=1 on today's path: the session's slot (-1 = S2 path), its run's base page
        std::map<std::string, cudaGraphExec_t> gx;   // graph form: per step layout (s2_batch.h Step::key + buckets + sources)
        std::set<std::string> prepped;
        std::vector<int> ms(NSS, 0), cj(NSS, 0), mrow0(NSS, 0);
        u64 mstep = 0, dstep2 = 0;
        size_t free0 = 0, total = 0;
        G6RT_CK(cudaMemGetInfo(&free0, &total));
        std::map<std::pair<int, int>, std::vector<double>> tk;   // (kind, M) -> step wall ms
        // B4 probe only (measurement): G6_S2_CLS=c runs every step at k1 row class c; G6_S2_ROWHASH=1 adds one line per
        // target row: "q<j> row <pos> resid <fnv of the row> lpt <bits>" (rank 0)
        static const int force_cls = getenv("G6_S2_CLS") ? atoi(getenv("G6_S2_CLS")) : 0;
        static const bool rowhash = getenv("G6_S2_ROWHASH") != nullptr;
        if (force_cls) fprintf(stderr, "# G6_S2_CLS=%d: every step at k1 row class %d (probe)\n", force_cls, force_cls);
        s2::Hooks hk;
        hk.run = [&](const s2::Step& sp) {
            const bool draft = sp.kind != s2::Step::MAIN, chain = sp.kind == s2::Step::CHAIN;
            const int M = sp.M();
            StepIn si;
            memset(&si, 0, sizeof si);
            si.step = draft ? ++dstep2 : ++mstep;
            si.M = M;
            for (int r = 0; r < M; ++r)
            {
                si.tok[r] = sp.r[r].tok; si.pos[r] = sp.r[r].pos; si.tgt[r] = sp.r[r].tgt;
                si.srow[r] = glue::SampleRow{plan.inv_temp, plan.seed, 0u, sp.r[r].spos, 0, plan.top_p};
            }
            S2In h;
            memset(&h, 0, sizeof h);
            h.ns = (int)sp.sess.size(); h.cls = force_cls ? force_cls : sp.cls;
            for (int j = 0; j < h.ns; ++j)
            {
                h.sess[j] = sp.sess[j]; h.row0[j] = sp.row0[j]; h.rows[j] = sp.rows[j]; h.ctx[j] = sp.ctx[j];
                si.context = std::max(si.context, sp.ctx[j]);
            }
            for (int j = 0; j < h.ns; ++j)
            {
                const S2Model& s2m = *(draft ? DR[0].d : R[0]).s2;
                h.base[j] = s2m.base[h.sess[j]];
                h.src0[j] = mrow0[h.sess[j]];
                if (h.ctx[j] > s2m.pages[h.sess[j]] * 64) die("s2: session context past its run (G6_S2_RUNS grant)");   // caller's grant check
            }
            std::string key = sp.key();   // kind + M; until attention is layout-free the per-session layout stays in this driver's key
            for (int j = 0; j < h.ns; ++j) key += ":" + std::to_string(h.sess[j]) + "x" + std::to_string(h.rows[j]);
            auto set = [&](Model& m, bool first) {
                m.s2->h = h;
                for (int j = 0; j < h.ns; ++j)
                {
                    m.s2->bk[h.sess[j]] = bucket_of(m, h.ctx[j]);
                    if (first) key += ":b" + std::to_string(m.s2->bk[h.sess[j]]) + "p" + std::to_string(h.base[j]) +
                                      (draft && !chain ? "s" + std::to_string(h.src0[j]) : "");
                }
            };
            if (draft) for (Draft& D : DR) set(D.d, D.d.rank == 0);
            else for (Model& m : R) set(m, m.rank == 0);
            StepOut so[NR];
            if (solo >= 0)                  // S=1: today's path and graphs (run_main / run_draft), the run at solo_base
            {
                si.page_bound = (draft ? DR[0].d : R[0]).s2->pool_pages;
                const double w = draft ? run_draft(si, chain, so) : run_main(si, so);
                tk[{(int)sp.kind + 3, M}].push_back(w);
            }
            else
            {
                if (!prepped.count(key))
                {
                    if (draft) for (Draft& D : DR) s2_prepare(D.d, M, st);
                    else for (Model& m : R) s2_prepare(m, M, st);
                    G6RT_CK(cudaStreamSynchronize(st));
                    if (graph)
                    {
                        cudaGraphExec_t ex;
                        if (draft) capture([&] { enqueue_draft(M, si.context, chain); }, &ex, ("s2 " + key).c_str(), M);
                        else if (gsplit)            // DEC_GSPLIT as the solo target step: parts A (step in .. layers < ls), B
                        {
                            cudaGraphExec_t ex2;
                            capture([&] { enqueue(M, si.context, 0, gsplit); }, &ex, ("s2 split part A " + key).c_str(), M);
                            capture([&] { enqueue(M, si.context, 1, gsplit); }, &ex2, ("s2 split part B " + key).c_str(), M);
                            gx[key + "|B"] = ex2;
                        }
                        else capture([&] { enqueue(M, si.context); }, &ex, ("s2 " + key).c_str(), M);
                        gx[key] = ex;
                    }
                    prepped.insert(key);
                }
                G6RT_CK(g6h2d::copy(draft ? in_d : in, &si, sizeof si));
                G6RT_CK(g6h2d::copy(draft ? in_s2d : in_s2, &h, sizeof h));
                const auto t0 = std::chrono::steady_clock::now();
                if (graph)
                {
                    G6RT_CK(cudaGraphLaunch(gx[key], st));
                    if (gsplit && !draft) G6RT_CK(cudaGraphLaunch(gx[key + "|B"], st));
                }
                else if (draft) enqueue_draft(M, si.context, chain);
                else enqueue(M, si.context);
                G6RT_CK(cudaStreamSynchronize(st));
                tk[{(int)sp.kind, M}].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                check_fabric(draft ? "s2 draft" : "s2 target");
                G6RT_CK(cudaMemcpy(so, draft ? out_d : out, NR * sizeof(StepOut), cudaMemcpyDeviceToHost));
            }
            for (int r = 0; r < NR; ++r)
                if (so[r].attn_err) { fprintf(stderr, "m4sim: attention error word %08x, rank %d, s2 step %s\n", so[r].attn_err, r, key.c_str()); exit(3); }
            for (int j = 0; j < h.ns; ++j)
            {
                const int q = h.sess[j];
                int s;
                if (!draft) { s = ms[q]++; mrow0[q] = h.row0[j]; }
                else if (!chain) { s = ms[q] - 1; cj[q] = 0; }
                else s = ++cj[q];
                const std::string prefix = "q" + std::to_string(q) + " " + (chain ? "chain" : draft ? "draft" : "step");
                std::string line0;
                for (int r = 0; r < NR; ++r)
                {
                    std::string line;
                    s2_step_line(draft ? DR[r].d : R[r], so[r], s, h.row0[j], h.rows[j], line);
                    line = prefix + line.substr(4);
                    if (r == 0) line0 = line;
                    else if (line != line0) { ++bad; printf("RANK %d DIFFERS: %s\n", r, line.c_str()); }
                }
                printf("%s\n", line0.c_str());
                if (rowhash && !draft)
                    for (int r = h.row0[j]; r < h.row0[j] + h.rows[j]; ++r)
                    {
                        std::vector<float> v(HID);
                        G6RT_CK(cudaMemcpy(v.data(), R[0].resid + (size_t)r * HID, HID * 4, cudaMemcpyDeviceToHost));
                        uint32_t u;
                        memcpy(&u, &so[0].lpt[r], 4);
                        printf("q%d row %d resid %016llx lpt %08x\n", q, si.pos[r], (unsigned long long)fnv(0xcbf29ce484222325ull, v.data(), HID * 4), u);
                    }
            }
            if (!draft) ++steps;
            return std::vector<int>(so[0].tok, so[0].tok + M);
        };
        hk.round = [&](int q, long rd, int base, int a, int K) { printf("q%d round %ld base %d accepted %d of %d\n", q, rd, base, a, K); };
        // S=1 transition contract (joint TP6 + TP4 ruling, TP4 option (a), provisional): a session alone (fresh, or the
        // survivor of 2 -> 1) runs today's path (s2on off on every model, the layers' own bindings, D.hprev, today's graphs)
        // with its run where it is: the layers' tables = its pages (s2_solo_pages), StepIn::page_bound = the pool's pages.
        // G6_S2_SOLO1=0: stay on the S2 path at S=1 (screen). G6_S2_RELOC=1: the fallback, relocate the survivor to base 0.
        // G6_S2_GEN=a,b,..: per-session --gen (so the sessions finish at different rounds and 2 -> 1 happens).
        const bool solo1 = !getenv("G6_S2_SOLO1") || atoi(getenv("G6_S2_SOLO1")) != 0;
        const bool reloc = getenv("G6_S2_RELOC") && atoi(getenv("G6_S2_RELOC")) != 0;
        if (const char* e = getenv("G6_S2_GEN"))
        {
            const std::vector<int> g = parse_ints(e);
            for (int q = 0; q < NSS && q < (int)g.size(); ++q) SS[q].n_gen = g[q];
        }
        auto go_solo = [&](int q) {
            const auto t0 = std::chrono::steady_clock::now();
            const int B = R[0].s2->base[q];
            int pages = 0;
            if (reloc && B)
            {
                for (int k = 0; k < NSS; ++k) pages = std::max(pages, (int)SS[k].prompt.size() + SS[k].n_gen + 2 * mtp + 63);
                pages = std::min(pages / 64, R[0].s2->pages[q]);
                if (const char* e = getenv("G6_S2_RELOC_PAGES")) pages = std::min(atoi(e), R[0].s2->pages[q]);   // stall timing: copy N pages
                for (Model& m : R) s2_relocate(m, q, pages, st);
                for (Draft& D : DR) s2_relocate(D.d, q, pages, st);
            }
            solo_base = R[0].s2->base[q];
            const int solo_pages = R[0].s2->pages[q];
            const std::vector<int> run = R[0].s2->run[q];
            for (Model& m : R) { s2_solo_pages(m, run); m.s2on = false; }
            for (Draft& D : DR) { s2_solo_pages(D.d, run); s2_hprev_to_solo(D, q, st); D.d.s2on = false; }
            G6RT_CK(cudaStreamSynchronize(st));
            solo = q;
            fprintf(stderr, "# s2 -> S=1: session %d alone, run at base page %d (was %d), %s %d pages; %.3f ms (tables%s, hprev)\n", q, solo_base, B,
                    reloc && B ? "relocated" : "no copy,", pages, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                    reloc && B ? " + relocation" : "");
        };
        std::vector<s2::Stats> sst(NSS);
        std::vector<std::vector<int>> gen(NSS);
        {
            s2::Batch b(mtp, MX, hk);
            std::vector<int> qs(NSS);
            for (int q = 0; q < NSS; ++q) qs[q] = q;
            if (solo1 && NSS == 1) go_solo(0);
            b.admit(qs, SS);
            for (;;)
            {
                std::vector<int> act;
                for (int q : qs) if (b.active(q)) act.push_back(q);
                if (act.empty()) break;
                if (solo1 && solo < 0 && act.size() == 1) go_solo(act[0]);
                b.round(qs);
            }
            for (int q = 0; q < NSS; ++q) { gen[q] = b.result(q); sst[q] = b.stats(q); }
        }
        for (int q = 0; q < NSS; ++q)
        {
            printf("q%d generated", q);
            for (int t : gen[q]) printf(" %d", t);
            printf("\n");
            long tok = 0, rounds = sst[q].rounds;
            tok = sst[q].tokens;
            fprintf(stderr, "# s2 session %d: %ld rounds, %ld tokens, accepted", q, rounds, tok);
            for (size_t a = 0; a < sst[q].acc.size(); ++a) fprintf(stderr, " %zu:%ld", a, sst[q].acc[a]);
            fprintf(stderr, "\n");
        }
        size_t free1 = 0;
        G6RT_CK(cudaMemGetInfo(&free1, &total));
        fprintf(stderr, "# s2: %zu step layouts (%s), device free memory %.1f MiB lower after them (graphs, replay entries, bound layers)\n",
                prepped.size(), graph ? "graphs" : "plain launches", ((double)free0 - (double)free1) / 1048576.0);
        static const char* kn[6] = {"verify/prompt", "first draft", "chain", "S=1 today's path verify/prompt", "S=1 today's path first draft",
                                    "S=1 today's path chain"};
        for (auto& kv : tk)
        {
            std::vector<double> v = kv.second;
            std::sort(v.begin(), v.end());
            fprintf(stderr, "# s2 %s steps M=%d: %zu, p50 %.3f ms min %.3f ms (six ranks in turn)\n", kn[kv.first.first], kv.first.second, v.size(),
                    v[v.size() / 2], v[0]);
        }
        printf("# m4sim: %d steps, %d rank lines differ from rank 0\n", steps, bad);
        return bad ? 1 : 0;
    }
#endif
    if (!serve.empty()) serve_ready(serve);
    for (size_t q = 0;; ++q)
    {
        Plan p;
        if (serve.empty())
        {
            if (q >= plans.size()) break;
            p = plans[q];
        }
        else
        {
            ServeReq rq;
            if (!serve_next(serve, q, glue::VOCAB, rq)) break;
            pt.start(serve + "/req." + std::to_string(q));
            p = plan;
            p.prompt = rq.prompt; p.n_gen = rq.n_gen; p.seed = rq.seed;
            p.inv_temp = rq.temp > 0.f ? 1.f / rq.temp : 0.f;
            p.top_p = rq.top_p;
            em.open(serve, q, true, rq);
            if (p.prompt.empty() || p.n_gen < 1 || (int)p.prompt.size() + p.n_gen + 2 * mtp > cap)
            {
                em.end("error", 0, 0, 0, nullptr);
                continue;
            }
            if (MP) p.first = (int)p.prompt.size() - 1;
        }
        // prefix reuse: rows 0..skip-1 hold the same tokens as the previous sequence (prompt + generated), so their cache rows stay
        // (every layer's KV and index keys, and the MTP layer's); processing restarts at row skip and overwrites the rest
        int skip = 0;
        if (reuse && q > 0)
        {
            // hist = the previous prompt + what it generated; its last token was never fed, so it has no cache row
            while (skip < (int)hist.size() - 1 && skip < (int)p.prompt.size() - 1 && hist[skip] == p.prompt[skip]) ++skip;
        }
        if (!skip)
        {
#if defined(DEC_S2) && defined(G6_CP) && G6_CP
            if (getenv("G6_S2_PFBASE"))         // base-page gate: clear the run only (s2_reset_pages)
            {
                for (Model& m : R) s2_reset_pages(m, s2_run_list(pf_base, pf_pages), st);
                for (Draft& D : DR) s2_reset_pages(D.d, s2_run_list(pf_base, pf_pages), st);
            }
            else
#endif
            {
                for (Model& m : R) reset(m, st);
                for (Draft& D : DR) reset(D.d, st);
            }
        }
        if (p.first < skip) p.first = skip;
        if (serve.empty()) pt.start("");
        else pt.lap("open");
        G6RT_CK(cudaStreamSynchronize(st));
        pt.lap("reset");
        if (skip) printf("seq %zu prompt %zu tokens, %d reused\n", q, p.prompt.size(), skip);
        else printf("seq %zu prompt %zu tokens\n", q, p.prompt.size());
        const int npf = p.first;
        for (int a = skip; a < npf; a += MP)
        {
            if (two && npf - a > MP)             // two sub-batches as two lanes
            {
                const int MA = MP, MB = npf - a - MP < MP ? npf - a - MP : MP;
                std::vector<int> pos(MA + MB);
                for (int r = 0; r < MA + MB; ++r) pos[r] = a + r;
                for (Model& m : R)
                {
                    pf_inputs(PF[m.rank], MA, p.prompt.data() + a, pos.data());
                    pf_inputs(PF2[m.rank], MB, p.prompt.data() + a + MA, pos.data() + MA);
                }
                pt.lap("inputs_h2d");
                const auto t0 = std::chrono::steady_clock::now();
                enqueue_pf2(MA, MB, a + MA, a + MA + MB, ls0, ls0b);
                G6RT_CK(cudaStreamSynchronize(st));
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                pt.lap("prefill");
                ls0 += NL; ls0b += NL;
                if (mtp)
                    for (int lane = 0; lane < 2; ++lane)
                    {
                        const int M = lane ? MB : MA, a0 = lane ? a + MA : a;
                        for (Model& m : R) pf_draft_inputs(DQ[m.rank], M, p.prompt.data() + a0 + 1, pos.data() + (a0 - a));
                        for (Model& m : R) pf_draft(DR[m.rank], DQ[m.rank], (lane ? PF2 : PF)[m.rank].x, M, a0 + M, st);
                        G6RT_CK(cudaStreamSynchronize(st));
                        for (Draft& D : DR) if (const unsigned ae = attn_errors(D.d)) { fprintf(stderr, "m4sim: MTP layer attention error word %08x, prefill rows %d..\n", ae, a0); return 3; }
                    }
                pt.lap("draft_prefill");
                if (const char* dd = mtp ? getenv("G6_PF_DRAFTDUMP") : nullptr)   // measurement only (PF_DRAFTKV screen): the MTP
                    for (Model& m : R)                                           // layer's caches -> <dir>/r<rank>.s<seq>.{kv,idx}
                    {
                        const G6Attn& c = DR[m.rank].d.layers[0].ab.args;
                        const size_t slots = (size_t)c.page_size * c.page_count;
                        const std::string b = std::string(dd) + "/r" + std::to_string(m.rank) + ".s" + std::to_string(q);
                        for (int k = 0; k < 2; ++k)
                        {
                            const void* src = k ? c.index_cache : c.mla_cache;
                            if (!src) continue;
                            std::vector<uint8_t> h(slots * (k ? 132 : 656));
                            G6RT_CK(cudaMemcpy(h.data(), src, h.size(), cudaMemcpyDeviceToHost));
                            FILE* f = fopen((b + (k ? ".idx" : ".kv")).c_str(), "wb");
                            if (!f) die("G6_PF_DRAFTDUMP: cannot write " + b);
                            fwrite(h.data(), 1, h.size(), f); fclose(f);
                        }
                    }
                unsigned e;
                G6RT_CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
                if (e) { fprintf(stderr, "m4sim: fabric error word %08x in prefill rows %d.. of seq %zu\n", e, a, q); return 3; }
                double hash_ms = 0;
                for (int lane = 0; lane < 2; ++lane)
                {
                    std::vector<Prefill>& L = lane ? PF2 : PF;
                    const int M = lane ? MB : MA, a0 = lane ? a + MA : a;
                    // test hook G6_PF_HASH_POKE=rank,row: +1 ulp on the first word of that lane-0 row on that rank before
                    // the cross-rank check (must print RANK n DIFFERS)
                    if (const char* pk = getenv("G6_PF_HASH_POKE"); pk && lane == 0)
                    {
                        int pr = 0, row = 0;
                        if (sscanf(pk, "%d,%d", &pr, &row) == 2 && pr >= 0 && pr < NR && row >= 0 && row < M)
                        {
                            uint32_t w;
                            uint32_t* at = reinterpret_cast<uint32_t*>(L[pr].resid) + static_cast<size_t>(row) * HID;
                            G6RT_CK(cudaMemcpy(&w, at, 4, cudaMemcpyDeviceToHost));
                            ++w;
                            G6RT_CK(g6h2d::copy(at, &w, 4));
                            fprintf(stderr, "# poke: rank %d row %d +1 ulp\n", pr, a0 + row);
                        }
                    }
                    const auto th = std::chrono::steady_clock::now();
                    const uint64_t h0 = pf_hash(L[0], M);
                    hash_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th).count();
                    for (Model& m : R)
                    {
                        if (const unsigned ae = attn_errors(m)) { fprintf(stderr, "m4sim: attention error word %08x, rank %d, prefill rows %d..\n", ae, m.rank, a0); return 3; }
                        if (pf_hash(L[m.rank], M) != h0) { ++bad; printf("RANK %d DIFFERS in prefill rows %d..%d\n", m.rank, a0, a0 + M - 1); }
                    }
                    printf("prefill rows %d..%d resid %016llx\n", a0, a0 + M - 1, (unsigned long long)h0);
                    pf_resdump(L[0], M, q, a0);
                }
                fprintf(stderr, "# seq %zu prefill %d + %d rows (two lanes): %.3f ms for the six ranks in turn\n", q, MA, MB, ms);
                fprintf(stderr, "# seq %zu pf_hash rank 0 both lanes (what one g6m4 rank does): %.3f ms\n", q, hash_ms);
                pt.lap("checks_hash_x6");
                a += MP;
                continue;
            }
            const int M = npf - a < MP ? npf - a : MP;
            std::vector<int> pos(M);
            for (int r = 0; r < M; ++r) pos[r] = a + r;
            for (Model& m : R) pf_inputs(PF[m.rank], M, p.prompt.data() + a, pos.data());
            const auto t0 = std::chrono::steady_clock::now();
            enqueue_pf(M, a + M, ls0);
            G6RT_CK(cudaStreamSynchronize(st));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ls0 += NL;
            if (mtp)
            {
                for (Model& m : R) pf_draft_inputs(DQ[m.rank], M, p.prompt.data() + a + 1, pos.data());
                for (Model& m : R) pf_draft(DR[m.rank], DQ[m.rank], PF[m.rank].x, M, a + M, st);
                G6RT_CK(cudaStreamSynchronize(st));
                for (Draft& D : DR) if (const unsigned ae = attn_errors(D.d)) { fprintf(stderr, "m4sim: MTP layer attention error word %08x, prefill rows %d..\n", ae, a); return 3; }
            }
            unsigned e;
            G6RT_CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
            if (e) { fprintf(stderr, "m4sim: fabric error word %08x in prefill rows %d.. of seq %zu\n", e, a, q); return 3; }
            const uint64_t h0 = pf_hash(PF[0], M);
            for (Model& m : R)
            {
                if (const unsigned ae = attn_errors(m)) { fprintf(stderr, "m4sim: attention error word %08x, rank %d, prefill rows %d..\n", ae, m.rank, a); return 3; }
                if (pf_hash(PF[m.rank], M) != h0) { ++bad; printf("RANK %d DIFFERS in prefill rows %d..%d\n", m.rank, a, a + M - 1); }
            }
            printf("prefill rows %d..%d resid %016llx\n", a, a + M - 1, (unsigned long long)h0);
            pf_resdump(PF[0], M, q, a);
            fprintf(stderr, "# seq %zu prefill %d rows: %.3f ms for the six ranks in turn\n", q, M, ms);
        }
        // G6_KV_HASH=1 (measurement, after the prefill): per layer, FNV of the stored latent record of
        // every prompt position < npf in logical order, read from every rank that stores it (all ranks in a replicated
        // build; the owner, plus every rank below R, under CP); holders must agree byte for byte
        if (getenv("G6_KV_HASH") && npf > 0)
        {
            G6RT_CK(cudaStreamSynchronize(st));
            for (int i = 0; i < NL; ++i)
            {
                uint64_t h = 0xcbf29ce484222325ull;
                int mism = 0;
                std::vector<std::vector<uint8_t>> kv(NR);
                std::vector<std::vector<int>> tb(NR);
                for (Model& m : R)
                {
                    const G6Attn& c = m.layers[i].ab.args;
                    kv[m.rank].resize((size_t)c.page_count * c.page_size * 656);
                    tb[m.rank].resize(c.table_size);
                    G6RT_CK(cudaMemcpy(kv[m.rank].data(), c.mla_cache, kv[m.rank].size(), cudaMemcpyDeviceToHost));
                    G6RT_CK(cudaMemcpy(tb[m.rank].data(), c.block_table, tb[m.rank].size() * 4, cudaMemcpyDeviceToHost));
                }
                const int ps = R[0].layers[i].ab.args.page_size;
                for (int x = 0; x < npf; ++x)
                {
                    const uint8_t* ref = nullptr;
                    for (int r = 0; r < NR; ++r)
                    {
                        const int pg = tb[r][x / ps];
                        if (pg < 0) continue;
                        const uint8_t* rec = kv[r].data() + ((size_t)pg * ps + x % ps) * 656;
                        if (!ref) ref = rec;
                        else if (memcmp(ref, rec, 656)) ++mism;
                    }
                    if (!ref) die("G6_KV_HASH: position " + std::to_string(x) + " stored by no rank");
                    h = fnv(h, ref, 656);
                }
                printf("kvhash seq %zu layer %d positions %d %016llx%s\n", q, layers[i], npf, (unsigned long long)h,
                       mism ? " HOLDERS DIFFER" : "");
                if (mism) ++bad;
            }
        }
        if (mtp)
        {
            StepOut so[NR], sd[NR];
            auto rm = [&](const StepIn& si, int s, double& ms) {
                ms += run_main(si, so);
                if (rlf) route_log_line(rlf, R[0], layers, q, s, si.M, si.pos[0], p.prompt.size());
                lines("step", false, so, s, si.M);
                if (top > 0) for (Model& m : R) { std::string tl; top_lines(m, s, si.M, top, tl); fputs(tl.c_str(), stdout); }
                logit_dump(s, si.M);
                return so[0];
            };
            auto rd = [&](const StepIn& di, bool chain, int s, double& ms) {
                ms += run_draft(di, chain, sd);
                const int t = lines(chain ? "chain" : "draft", true, sd, s, di.M);
                if (!mtp_dump.empty() && di.step == 1)               // the first prompt step's draft pass
                {
                    const int M = di.M;
                    dumpf("a.tok.i32", DR[0].d.tok, M * 4); dumpf("a.pos.i32", DR[0].d.pos, M * 4);
                    dumpf("a.main_x.f16", R[0].x, (size_t)M * HID * 2);
                    dumpf("a.eh_in.f16", DR[0].d.eh_in, (size_t)M * 2 * HID * 2); dumpf("a.eh_out.f16", DR[0].d.eh_out, (size_t)M * HID * 2);
                    dumpf("a.resid.f32", DR[0].d.resid, (size_t)M * HID * 4); dumpf("a.x.f16", DR[0].d.x, (size_t)M * HID * 2);
                    dumpf("a.hprev.f16", DR[0].hprev, HID * 2);
                }
                if (!mtp_dump.empty() && di.step == 2 && chain)      // ... and the chain step after it (one prompt step only)
                {
                    dumpf("b.tok.i32", DR[0].d.tok, 4); dumpf("b.pos.i32", DR[0].d.pos, 4);
                    dumpf("b.eh_in.f16", DR[0].d.eh_in, 2 * HID * 2); dumpf("b.eh_out.f16", DR[0].d.eh_out, HID * 2);
                }
                return DraftOut{t, sd[0].lp[di.M - 1]};
            };
            std::function<bool(const int*, int, int)> emit;
            if (em.on) emit = [&](const int* t, int k, int) { pt.first_token(q); return em.push(t, k); };
            pt.lap("pre_decode");
            const std::vector<int> gen = mtp_sequence(p, mtp, step0, dstep, rm, rd, mtp_stats, steps, emit, &spec);
            hist = p.prompt; hist.insert(hist.end(), gen.begin(), gen.end());
            printf("generated");
            for (int t : gen) printf(" %d", t);
            printf("\n");
            em.end(nullptr, mtp_stats.rounds, mtp, mtp_stats.tokens, mtp_stats.acc, mtp_stats.drafted);
            continue;
        }
        const int PS = prompt_steps(p), NS = total_steps(p);
        int last_tok = 0, ns_run = NS;
        std::vector<int> gen;
        for (int s = 0; s < NS; ++s)
        {
            const int M = step_rows(p, s);
            const StepIn si = step_inputs(p, s, last_tok, step0);
            const int k = bucket(si.context, false);
            if (!ready[k][M])
            {
                for (Model& m : R) prepare(m, M, st);
                G6RT_CK(cudaStreamSynchronize(st));
                if (graph)
                    for (int pc = gsplit ? 0 : -1; pc < (gsplit ? 2 : 0); ++pc)       // DEC_GSPLIT: parts A, B
                    {
                        cudaGraph_t g;
                        G6RT_CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                        enqueue(M, si.context, pc, gsplit);
                        G6RT_CK(cudaStreamEndCapture(st, &g));
                        size_t nodes = 0;
                        G6RT_CK(cudaGraphGetNodes(g, nullptr, &nodes));
                        G6RT_CK(cudaGraphInstantiate(pc == 1 ? &exec2[k][M] : &exec[k][M], g, 0));
                        G6RT_CK(cudaGraphDestroy(g));
                        fprintf(stderr, "# graph%s for M=%d bucket %d: %zu nodes (six ranks + loopback)\n", pc < 0 ? "" : pc ? " split part B" : " split part A",
                                M, R[0].bk[k], nodes);
                    }
                ready[k][M] = true;
            }
            put_in(in, si);
            const auto t0 = std::chrono::steady_clock::now();
            if (graph) launch_main(k, M);
            else enqueue(M, si.context);
            G6RT_CK(cudaStreamSynchronize(st));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            unsigned e;
            G6RT_CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
            if (e) { fprintf(stderr, "m4sim: fabric error word %08x at seq %zu step %d\n", e, q, s); return 3; }
            StepOut so[NR];
            G6RT_CK(cudaMemcpy(so, out, sizeof so, cudaMemcpyDeviceToHost));

            std::string line0;
            for (Model& m : R)
            {
                if (so[m.rank].attn_err) { fprintf(stderr, "m4sim: attention error word %08x, rank %d seq %zu step %d\n", so[m.rank].attn_err, m.rank, q, s); return 3; }
                std::string line;
                const int t = step_line(m, so[m.rank], s, M, line);
                if (m.rank == 0) { line0 = line; last_tok = t; }
                else if (line != line0) { ++bad; printf("RANK %d DIFFERS: %s\n", m.rank, line.c_str()); }
            }
            printf("%s\n", line0.c_str());
            if (top > 0) for (Model& m : R) { std::string tl; top_lines(m, s, M, top, tl); fputs(tl.c_str(), stdout); }
            logit_dump(s, M);
            if (s >= PS - 1) gen.push_back(last_tok);
            fprintf(stderr, "# seq %zu step %d M %d: %.3f ms for the six ranks in turn\n", q, s, M, ms);
            if (s >= PS + 2) t1.push_back(ms);
            if (s >= PS - 1 && em.on) pt.first_token(q);
            if (s >= PS - 1 && em.on && em.push(&last_tok, 1)) { ns_run = s + 1; break; }
        }
        step0 += ns_run;
        steps += ns_run;
        em.end(nullptr, 0, 0, 0, nullptr);
        hist = p.prompt; hist.insert(hist.end(), gen.begin(), gen.end());
        printf("generated");
        for (int t : gen) printf(" %d", t);
        printf("\n");
    }
    if (pf_prof().on) { pf_prof().print("six ranks in turn,"); qo_prof().print(); }
    if (!t1.empty())
    {
        std::sort(t1.begin(), t1.end());
        fprintf(stderr, "# single-row step, six ranks in turn, %s: p50 %.3f ms min %.3f ms (%.1f us per rank per layer at p50), %d layers\n",
                graph ? "one graph" : "plain launches", t1[t1.size() / 2], t1[0], t1[t1.size() / 2] * 1000 / NR / NL, NL);
    }
    if (mtp) mtp_stats.print(mtp, "six ranks in turn");
    if (spec.on) fprintf(stderr, "%s\n", spec.summary().c_str());
    if (rlf) fclose(rlf);
    if (ldump) fclose(ldump);
    if (rdf) fclose(rdf);
    if (idf) fclose(idf);
    if (nc) fprintf(stderr, "# top_p misses %u (target, rank 0) %u (draft, rank 0)\n", topp_misses(R[0]), mtp ? topp_misses(DR[0].d) : 0u);
#if G6_NR == 4
    for (int r = 0; r < NR; ++r) if (qp_roll[r]) qp_roll_print(*qp_roll[r], r);
#endif
    printf("# m4sim: %d steps, %d rank lines differ from rank 0\n", steps, bad);
    if (xt_dev[0])                      // G6_XTRACE: GPU records of every simulated rank, after the run
    {
        G6RT_CK(cudaDeviceSynchronize());
        const char* d = getenv("G6_XTRACE_OUT");
        const std::string pth = std::string(d ? d : "/tmp") + "/g6xtrace.m4sim.txt";
        if (FILE* o = fopen(pth.c_str(), "w"))
        {
            fprintf(o, "# g6xtrace v1 m4sim nr=%d ncoll=%d steps=%d (loopback: no proxy records)\n", NR, NC, g6::XT_STEPS);
            for (int r = 0; r < NR; ++r) g6::xt_dump_gpu(o, xt_dev[r], NC, NR, r);
            fclose(o);
            fprintf(stderr, "m4sim: G6_XTRACE dump -> %s\n", pth.c_str());
        }
    }
    return bad ? 1 : 0;
}
