// The S2 runtime signatures (the string every rank's bootstrap compares) against transcripts recorded from g6m4.cu's
// three builders before R6a moves them to s2/signature.h. Each model's section draws its inputs from a seeded sweep of
// every field's domain (empty set_id, dfl_alpha != 1, random doubles through std::to_string) and hashes the strings.
// Each hash must equal the recorded one. Rerecord only for a change meant to alter a signature, and say so in the commit.
//   --dump MODEL   print that model's strings (glm, flash, ds41)
// The generator is mt19937_64 with modulo ranges. No call or + takes two draws as operands: their order is unspecified,
// and GCC on x86_64 takes them right to left. Host only.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include "../../ds41_vision_id.h"

namespace {
// R6a freeze: the inputs signature.h will take, and g6m4.cu's three expressions with its names bound to them
enum class SigModel { Glm, Flash, Ds41 };
struct SigInputs
{
    SigModel model = SigModel::Glm;
    int cap = 0;
    std::string layers_s;
    int chunk = 0, nc = 0;
    std::string set_id;
    bool dfl = false;
    int dfl_k = 0, s2sess = 1;
    std::string knobs;                  // knob::signature()
    int MP = 0, mtp = 0;                // GLM and Flash
    int dfl_reuse = 0, dfl_reuse_min = 0;
    int lanes = 0, gsplit = 0;          // Flash
    std::string ckpt;                   // Flash: g6s2::flash_cache_name, empty when the cache is off
    uint64_t ckpt_blocks = 0;
    bool ckpt_persist = false;
    int s2_below = 0;                   // GLM
    bool two = false;
    int pf_own = 0;
    bool pf_qp = false;
    int qp_ctx = 0, qp_min = 0;
    bool grammar = false;               // g6rt::grammar_ready()
    std::string ngram;                  // s2::ngram_sig
    int kstop_mode = 0, kstop_kmax = 0;
    double kstop_tau[4] = {};
    double kstop_seed3 = 0;
    int draft_par = 0, qs_mode = 0;
    bool iq_split = false, mtp_split = false;
    int dhead_on = 0, dhead_tpc = 0, dhead_ks = 0;
    bool mtp_last = false;
    int dhead2 = 0;
    bool dfl_i4 = false, dfl_body_i4 = false;
    float dfl_alpha = 1.f;
    int dsel_model = 0;
    std::string dsel_tab;               // s2::DSelTable::str
    std::string kstopb;                 // s2::kstopb_sig
    int cp_step = 0;
    bool cp_seg = false;
    int ctx_sp = 0, posidx = 0;         // g6_attn_ctx_sp(), g6_attn_posidx()
    bool steer_shared = false;          // a loaded steering profile of shared scope
    int ds_mtp = 0;                     // DS41: GR8_DS_MTP
    bool sh_tc = false;
    int l2hint = 0, fc2stg = 0, fc1bstg = 0;
    bool c8_pack = false, sess_drafts = false;
    int pf_lanes = 1;
    int tk_main = 0, tk_decode = 0;
    int idx_pipe_b = 0, idx_q_trig = 0, idx_pipe_qr = 0;
    bool reuse = false;
    bool vis_on = false;
    std::string vis_id;
};

std::string legacy_flash(const SigInputs& in)
{
    const int cap = in.cap, MP = in.MP, lanes = in.lanes, mtp = in.mtp, nc = in.nc, dfl_k = in.dfl_k, dfl_reuse = in.dfl_reuse,
              dfl_reuse_min = in.dfl_reuse_min, s2sess = in.s2sess, gsplit = in.gsplit;
    const std::string &layers_s = in.layers_s, &set_id = in.set_id;
    const bool dfl = in.dfl;
    struct { int chunk; } plan{in.chunk};
    struct { std::string signature() const { return s; } std::string s; } knob{in.knobs};
    std::string runtime_signature = "flash solo cap " + std::to_string(cap) + " layers " + layers_s +
            " mp " + std::to_string(MP) + " lanes " + std::to_string(lanes) + " mtp " + std::to_string(mtp) +
            " chunk " + std::to_string(plan.chunk) + " cands " + std::to_string(nc) + " set " + set_id +
            (dfl ? " dflash k " + std::to_string(dfl_k) : std::string()) +
            (dfl_reuse ? " dflreuse " + std::to_string(dfl_reuse) + "/" + std::to_string(dfl_reuse_min) : std::string()) +
            (s2sess > 1 ? " sessions " + std::to_string(s2sess) + " split " + std::to_string(gsplit) : std::string()) +
            knob.signature();
    if (!in.ckpt.empty())
    {
        const bool index_path_empty = !in.ckpt_persist;
        runtime_signature += std::string(" ckpt ") + in.ckpt + " blocks " + std::to_string(in.ckpt_blocks) +
                (index_path_empty ? "" : " persist");
    }
    return runtime_signature;
}

std::string legacy_ds41(const SigInputs& in)
{
    const int s2sess = in.s2sess, cap = in.cap, nc = in.nc, dfl_k = in.dfl_k, pf_lanes = in.pf_lanes;
    const int ds_moe_main_topk = in.tk_main, ds_moe_decode_topk = in.tk_decode;
    const int ds_idx_pipe_b = in.idx_pipe_b, ds_idx_q_trig = in.idx_q_trig, ds_idx_pipe_qr = in.idx_pipe_qr;
    const std::string &layers_s = in.layers_s, &set_id = in.set_id;
    const bool dfl = in.dfl, ds_sess_drafts = in.sess_drafts, reuse_prefix = in.reuse, vis_on = in.vis_on;
    const std::string& vis_id = in.vis_id;
    struct { int mtp; } dsk{in.ds_mtp};
    struct { int chunk; } plan{in.chunk};
    struct { bool c8_pack; } sc{in.c8_pack};
    struct M {
        bool sh_tc; int l2, stg, bstg;
        bool sh_tc_on() const { return sh_tc; } int ro_l2hint() const { return l2; }
        int ro_fc2stg() const { return stg; } int ro_fc1bstg() const { return bstg; }
    } ds41moe{in.sh_tc, in.l2hint, in.fc2stg, in.fc1bstg};
    struct { std::string signature() const { return s; } std::string s; } knob{in.knobs};
    std::string runtime_signature;
    runtime_signature = std::string(s2sess > 1 ? "ds41 sessions " + std::to_string(s2sess) : std::string("ds41 solo")) + " cap " + std::to_string(cap) + " layers " + layers_s + " mtp " + std::to_string(dsk.mtp) +
            " chunk " + std::to_string(plan.chunk) + " cands " + std::to_string(nc) + " set " + set_id +
            (dfl ? " dflash k " + std::to_string(dfl_k) : std::string()) +
            " shtc " + std::to_string(ds41moe.sh_tc_on() ? 1 : 0) +
            (ds41moe.ro_l2hint() ? " l2h " + std::to_string(ds41moe.ro_l2hint()) : std::string()) +
            (ds41moe.ro_fc2stg() ? std::string(" stg 1") : std::string()) +
            (ds41moe.ro_fc1bstg() ? std::string(" fc1bstg 1") : std::string());
    if (sc.c8_pack) runtime_signature += " c8pack 1";
    if (ds_sess_drafts) runtime_signature += " sessdrafts 1";
    if (pf_lanes == 2) runtime_signature += " pflanes 2";
    if (ds_moe_decode_topk != ds_moe_main_topk)
        runtime_signature += " tkmain " + std::to_string(ds_moe_main_topk) +
                                " tkdecode " + std::to_string(ds_moe_decode_topk);
    if (ds_idx_pipe_b) runtime_signature += " idxpipeb 1";
    if (ds_idx_q_trig) runtime_signature += " idxqtrig 1";
    if (ds_idx_pipe_qr) runtime_signature += " idxpipeqr 1";
    if (reuse_prefix) runtime_signature += " reuse 1";
    runtime_signature += ds41v::signature(vis_on, vis_id);
    runtime_signature += knob.signature();
    return runtime_signature;
}

std::string legacy_glm(const SigInputs& in)
{
    const int s2_below = in.s2_below, cap = in.cap, MP = in.MP, mtp = in.mtp, nc = in.nc, pf_own = in.pf_own;
    const int dfl_reuse = in.dfl_reuse, dfl_reuse_min = in.dfl_reuse_min, draft_par = in.draft_par, qs_mode = in.qs_mode;
    const int dhead2 = in.dhead2, dsel_model = in.dsel_model, cp_step = in.cp_step;
    const bool two = in.two, pf_qp = in.pf_qp, iq_split = in.iq_split, mtp_split = in.mtp_split, mtp_last = in.mtp_last;
    const bool dfl_i4 = in.dfl_i4, dfl_body_i4_on = in.dfl_body_i4;
    const float dfl_alpha = in.dfl_alpha;
    const std::string &layers_s = in.layers_s, &set_id = in.set_id;
    struct { int chunk; } plan{in.chunk};
    struct { int qp_ctx; } P{in.qp_ctx};
    struct { bool cp_seg; } sc{in.cp_seg};
    struct { int on, tpc, ks; } dhead{in.dhead_on, in.dhead_tpc, in.dhead_ks};
    struct { std::string str() const { return s; } std::string s; } dsel_tab{in.dsel_tab};
    struct { int mode, kmax; double tau[4]; double seed3; } kstop{in.kstop_mode, in.kstop_kmax,
        {in.kstop_tau[0], in.kstop_tau[1], in.kstop_tau[2], in.kstop_tau[3]}, in.kstop_seed3};
    struct { bool loaded; bool shared; } steering_boot{in.steer_shared, in.steer_shared};
    struct { std::string signature() const { return s; } std::string s; } knob{in.knobs};
    auto pf_qp_min = [&] { return in.qp_min; };
    auto grammar_ready = [&] { return in.grammar; };
    auto ngram_sig = [&] { return in.ngram; };
    auto kstopb_sig = [&] { return in.kstopb; };
    auto g6_attn_ctx_sp = [&] { return in.ctx_sp; };
    auto g6_attn_posidx = [&] { return in.posidx; };
    return "g6m4 s2 below " + std::to_string(s2_below) + " cap " + std::to_string(cap) + " layers " + layers_s +
                " mp " + std::to_string(MP) + " lanes " + std::to_string(two ? 2 : 1) + " mtp " + std::to_string(mtp) + " chunk " +
                std::to_string(plan.chunk) + " cands " + std::to_string(nc) + " split 3 own " + std::to_string(pf_own) +
                " qpctx " + std::to_string(pf_qp ? P.qp_ctx : 0) + " qpmin " + std::to_string(pf_qp ? pf_qp_min() : 0) +
                (set_id.empty() ? "" : " set " + set_id) + (grammar_ready() ? " grammar" : "") +
                (dfl_reuse ? " dflreuse " + std::to_string(dfl_reuse) + "/" + std::to_string(dfl_reuse_min) : "") + ngram_sig() +
                (kstop.mode ? " kstop " + std::to_string(kstop.mode) + " kmax " + std::to_string(kstop.kmax) + " tau " + std::to_string(kstop.tau[1]) + "," +
                              std::to_string(kstop.tau[2]) + "," + std::to_string(kstop.tau[3]) + " seed3 " + std::to_string(kstop.seed3) : "") +
                (draft_par ? " draftpar " + std::to_string(draft_par) : "") +
                (qs_mode ? " qasplit " + std::to_string(qs_mode) : "") +
                (iq_split ? " idxqsplit 1" : "") + (mtp_split ? " mtpsplit 1" : "") +
                (dhead.on ? " dheadi4 " + std::to_string(dhead.tpc) + "x" + std::to_string(dhead.ks) : "") +
                (mtp_last ? " mtplast 1" : "") + (dhead2 ? " dhead2 " + std::to_string(dhead2) : "") +
                (dfl_i4 ? " dflheadi4 " + std::to_string(dhead.tpc) + "x" + std::to_string(dhead.ks) : "") +
                (dfl_body_i4_on ? " dflbodyi4 1" : "") +
                (dfl_alpha != 1.f ? " dflalpha " + std::to_string(dfl_alpha) : "") +
                (dsel_model ? " dselmodel " + std::to_string(dsel_model) + " " + dsel_tab.str() : "") + kstopb_sig() +
                (cp_step ? " cpsr " + std::to_string(cp_step) + (sc.cp_seg ? " seg" : "") : "") +
                (g6_attn_ctx_sp() ? " ctxsp " + std::to_string(g6_attn_ctx_sp()) : "") + (g6_attn_posidx() ? " posidx 1" : "") +
                (steering_boot.loaded && steering_boot.shared ? " s2steer 1" : "") +
                knob.signature();
}

std::string runtime_signature(const SigInputs& in)
{
    return in.model == SigModel::Flash ? legacy_flash(in) : in.model == SigModel::Ds41 ? legacy_ds41(in) : legacy_glm(in);
}

struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    uint64_t U(uint64_t lo, uint64_t hi) { return hi - lo == UINT64_MAX ? g() : lo + g() % (hi - lo + 1); }
    bool coin(unsigned pct) { return U(0, 99) < pct; }
    int I(int lo, int hi) { return int(U(0, uint64_t(hi - lo)) + uint64_t(lo)); }
    // a double over many magnitudes, negatives and std::to_string's rounding edges included
    double D() {
        const uint64_t k = U(0, 5);
        if (k == 0) return double(U(0, 1000)) / 1000.0;
        if (k == 1) return double(U(0, UINT64_MAX) >> 11) * 0x1.0p-53;
        if (k == 2) return (double(U(0, 2000000)) - 1000000.0) / 7.0;
        if (k == 3) return 0.0000005 * double(U(0, 4));
        if (k == 4) { uint64_t b = U(0, UINT64_MAX); double x; std::memcpy(&x, &b, sizeof x); return x == x && x - x == 0 ? x : 1e300; }
        return double(U(0, 3)) * 0.25;
    }
};
struct Tx {
    uint64_t h = 14695981039346656037ull;
    bool dump = false;
    void operator()(const std::string& s) {
        for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
        h = (h ^ '\n') * 1099511628211ull;
        if (dump) printf("%s\n", s.c_str());
    }
};
std::string hex(Rng& r, int n) {
    std::string s;
    for (int i = 0; i < n; ++i) s += "0123456789abcdef"[r.U(0, 15)];
    return s;
}
std::string pick(Rng& r, std::initializer_list<const char*> v) { return *(v.begin() + r.U(0, v.size() - 1)); }
std::string words(Rng& r) {
    const uint64_t k = r.U(0, 3);
    if (k == 0) return "";
    if (k == 1) return " knobs GR8_DS_COMP_FP4=1";
    if (k == 2) return " knobs GR8_ATTN_SC2=4 GR8_ROUTE_OVL=0";
    return " " + hex(r, 12);
}

void common(Rng& r, SigInputs& in) {
    in.cap = r.coin(30) ? 262144 : r.I(0, 1 << 22);
    in.layers_s = pick(r, {"all", "0,3,40,77", "", "1,14", "@layers.txt"});
    in.chunk = r.I(1, 32);
    in.nc = r.coin(50) ? 256 : r.I(0, 1024);
    const uint64_t s = r.U(0, 3);
    in.set_id = s == 0 ? "" : s == 1 ? "ab12" : s == 2 ? hex(r, 64) : hex(r, 1);
    in.dfl = r.coin(50);
    in.dfl_k = r.I(1, 7);
    in.s2sess = r.I(1, 32);
    in.knobs = words(r);
}

void glm(Tx& tx, int n) {
    Rng r(0x676c6d);
    for (int i = 0; i < n; ++i) {
        SigInputs in;
        in.model = SigModel::Glm;
        common(r, in);
        in.MP = r.coin(70) ? 4096 : r.I(0, 8192);
        in.mtp = r.I(0, 7);
        in.dfl_reuse = r.I(0, 2);
        in.dfl_reuse_min = r.I(0, 2048);
        in.s2_below = r.I(0, 65536);
        in.two = r.coin(50);
        in.pf_own = r.I(0, 4);
        in.pf_qp = r.coin(50);
        in.qp_ctx = r.I(0, 1 << 20);
        in.qp_min = r.I(0, 65536);
        in.grammar = r.coin(50);
        in.ngram = r.coin(70) ? "" : " ngram " + hex(r, 16) + " kmax 7 kmin 2 margin 0.10000000000000001";
        in.kstop_mode = r.I(0, 2);
        in.kstop_kmax = r.I(0, 3);
        for (int j = 0; j < 4; ++j) in.kstop_tau[j] = r.D();
        in.kstop_seed3 = r.D();
        in.draft_par = r.I(0, 2);
        in.qs_mode = r.I(0, 2);
        in.iq_split = r.coin(50);
        in.mtp_split = r.coin(50);
        in.dhead_on = r.I(0, 1);
        in.dhead_tpc = r.I(1, 8);
        in.dhead_ks = r.I(1, 64);
        in.mtp_last = r.coin(50);
        in.dhead2 = r.I(0, 2);
        in.dfl_i4 = r.coin(50);
        in.dfl_body_i4 = r.coin(50);
        in.dfl_alpha = r.coin(50) ? 1.f : float(r.D());
        in.dsel_model = r.I(0, 1);
        in.dsel_tab = r.coin(50) ? "" : "tab " + hex(r, 8);
        in.kstopb = r.coin(60) ? "" : " kstopb 1 kmin 0 tau 0.59999999999999998,0.88 widths 2,3 explore " + std::to_string(r.I(0, 9));
        in.cp_step = r.coin(50) ? 0 : r.I(1, 64);
        in.cp_seg = r.coin(50);
        in.ctx_sp = r.I(0, 3);
        in.posidx = r.I(0, 1);
        in.steer_shared = r.coin(30);
        tx(runtime_signature(in));
    }
}

void flash(Tx& tx, int n) {
    Rng r(0x666c617368);
    for (int i = 0; i < n; ++i) {
        SigInputs in;
        in.model = SigModel::Flash;
        common(r, in);
        in.MP = r.coin(70) ? 4096 : r.I(0, 8192);
        in.mtp = r.I(0, 7);
        in.dfl_reuse = r.I(0, 2);
        in.dfl_reuse_min = r.I(0, 2048);
        in.lanes = r.I(1, 2);
        in.gsplit = r.I(0, 3);
        in.ckpt = pick(r, {"", "", "aligned", "terminal"});
        in.ckpt_blocks = r.coin(50) ? r.U(0, 1 << 20) : r.U(0, UINT64_MAX);
        in.ckpt_persist = r.coin(50);
        tx(runtime_signature(in));
    }
}

void ds41(Tx& tx, int n) {
    Rng r(0x64733431);
    for (int i = 0; i < n; ++i) {
        SigInputs in;
        in.model = SigModel::Ds41;
        common(r, in);
        in.ds_mtp = r.I(0, 5);
        in.sh_tc = r.coin(50);
        in.l2hint = r.I(0, 2);
        in.fc2stg = r.I(0, 1);
        in.fc1bstg = r.I(0, 1);
        in.c8_pack = r.coin(50);
        in.sess_drafts = r.coin(50);
        in.pf_lanes = r.I(0, 2);
        in.tk_main = r.I(1, 6);
        in.tk_decode = r.coin(60) ? in.tk_main : r.I(1, 6);
        in.idx_pipe_b = r.I(0, 1);
        in.idx_q_trig = r.I(0, 1);
        in.idx_pipe_qr = r.I(0, 1);
        in.reuse = r.coin(50);
        in.vis_on = r.coin(50);
        in.vis_id = r.coin(20) ? "" : hex(r, 40);
        tx(runtime_signature(in));
    }
}
}   // namespace

int main(int argc, char** argv)
{
    const char* only = argc == 3 && !std::strcmp(argv[1], "--dump") ? argv[2] : nullptr;
    struct Section { const char* name; void (*run)(Tx&, int); int n; uint64_t want; };
    const Section sections[] = {
        {"glm", glm, 20000, 0xc41b350c8f1fb75cull},
        {"flash", flash, 20000, 0x4e2f07741a5d469bull},
        {"ds41", ds41, 20000, 0xeaec6922c35a6de2ull},
    };
    int bad = 0;
    for (const Section& s : sections) {
        if (only && std::strcmp(only, s.name)) continue;
        Tx tx;
        tx.dump = only != nullptr;
        s.run(tx, s.n);
        if (only) continue;
        printf("signature %s: %d strings, hash %016" PRIx64 "%s\n", s.name, s.n, tx.h, tx.h == s.want ? "" : " (MISMATCH)");
        bad += tx.h != s.want;
    }
    return bad ? 1 : 0;
}
