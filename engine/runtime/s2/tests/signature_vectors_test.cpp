// The S2 runtime signatures (s2/signature.h: the string every rank's bootstrap compares) against transcripts recorded
// from g6m4.cu's three builders before R6a moved them there. Each model's section draws its inputs from a seeded sweep of
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
#include "../signature.h"

namespace {
using g6s2::SigInputs;
using g6s2::SigModel;
using g6s2::runtime_signature;

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
