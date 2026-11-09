// The fabric plan (fabric_plan.h) over a seeded sweep of its inputs in this build's shape: the options the bring-up reads
// (rank, mesh or ring, rails, --pf-ips2, --proxy-cpu -1..13, --prefill 0 / 4096 / 8192, lanes, MTP 0..3, DFlash2, --serve,
// sessions, GR8_PF_QP_ROLL and its wire entries, DSpark, the C8 pack, Engram, the router split), every knob it reads
// (GR8_PF_STRIPE unset / 0 / 1, GR8_XTRACE_PF, GR8_PF_RS, the CP knobs, GR8_QA_SPLIT_CPU, the Flash P1 conflicts, every
// DS41 prefill knob, valid and refused values) and the build values main passes in. Each case's inputs and describe() of
// its plan (every FabricConfig field in creation order, the scalars, or the refusal) make its transcript; each shape's
// hash must equal the recorded one. g6m4's GR8_FABRIC_DUMP showed the plan builds what the inline bring-up built, over
// the launcher's lists on a GB10 (docs/HOST-CORE.md, R6c). Rerecord only for a change meant to alter a fabric, and say
// so in the commit.
//   fabric_plan_vectors_test [--dump]   --dump prints every transcript
// The generator is mt19937_64 with modulo ranges, one draw per statement. Host only, Linux only (clearenv: the knob
// defaults come from the table with an empty environment).
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include "../../fabric_plan.h"

namespace {
using g6rt::EngineOptions;
using g6rt::FabricPlan;
using g6rt::PlanInputs;
using g6rt::PlanKnobs;

struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    uint64_t U(uint64_t lo, uint64_t hi) { return lo + g() % (hi - lo + 1); }
    bool coin(unsigned pct) { return U(0, 99) < pct; }
    int I(int lo, int hi) { return int(U(0, uint64_t(hi - lo)) + uint64_t(lo)); }
    template <class T> T pick(std::initializer_list<T> v) { return *(v.begin() + U(0, v.size() - 1)); }
};
struct Tx {
    uint64_t h = 14695981039346656037ull;
    bool dump = false;
    void operator()(const std::string& s) {
        for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
        if (dump) fputs(s.c_str(), stdout);
    }
};

std::vector<std::string> addrs(const char* net, int n)
{
    std::vector<std::string> v;
    for (int i = 0; i < n; ++i) v.push_back(net + std::to_string(i + 1));
    return v;
}
std::optional<std::string> text(Rng& r, std::initializer_list<const char*> v)
{
    const char* s = r.pick(v);
    if (!s) return std::nullopt;
    return std::string(s);
}
template <class T> std::string opt(const std::optional<T>& v)
{
    if (!v) return "-";
    if constexpr (std::is_same_v<T, std::string>) return *v;
    else return std::to_string(*v);
}
std::string ints(const std::vector<int>& v)
{
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

// --layers: all main layers, or a subset (DS41's with and without layers past the bounded-replay split)
std::vector<int> layers(Rng& r)
{
    const int k = r.I(0, 3);
    std::vector<int> v;
    if (k == 0) for (int l = 0; l < geom::L_MAIN; ++l) v.push_back(l);
    else if (k == 1) v = {0, 3, geom::L_MAIN / 2, geom::L_MAIN - 1};
    else if (k == 2) v = {0, 1, 2};
    else v = {geom::L_MAIN - 3, geom::L_MAIN - 2, geom::L_MAIN - 1};
    return v;
}

void one(Rng& r, const PlanKnobs& k0, Tx& tx, std::map<std::string, int>& kinds, int& refused)
{
    const int NR = g6nr::NR;
    EngineOptions o;
    g6::FabricConfig& c = o.cfg;
    c.rank = r.I(0, NR - 1);
    c.ips = addrs("10.0.0.", NR);
    c.ring = r.coin(25);
    if (c.ring && r.coin(30)) c.ring_devs = {"mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"};
    if (r.coin(40)) c.ips2 = addrs("10.1.0.", NR);
    c.proxy_cpu = r.I(-1, 13);
    if (r.coin(30)) o.pf_ips2 = addrs("10.2.0.", NR);
    o.MP = r.pick({0, 4096, 8192});
    o.lanes = r.coin(1) ? 3 : r.I(1, 2);
    o.mtp = r.I(0, 3);
    o.dfl = r.coin(50);
    o.s2mode = r.coin(60);
    o.K = r.I(0, 7);
    o.plan.chunk = r.pick({1, 4, 8, 16});
    o.s2sess = r.I(1, G6_MODEL_DS41 ? 32 : 8);
    o.rt_win = r.coin(30) ? 2 : 0;
    o.row_counts = {1, 2, 3};
    const int extra = r.I(0, 3);
    for (int i = 0; i < extra; ++i) o.row_counts.push_back(r.I(4, g6mx::MX));
    o.layers = layers(r);
    o.NL = (int)o.layers.size();
    o.NX = G6_MODEL_DS41 && r.coin(50) ? 2 : 0;
    o.NC = 2 * o.NL + 1 + o.NX;
    o.qp_roll_s = r.coin(20) ? r.I(2, 4) : 0;
    o.qp_wire_e = r.I(1, 3);
    int c8 = 0, dsmtp = 0, dscap = 0, rsplit = 0;
#if G6_MODEL_DS41
    c8 = o.c8_pack_env = r.I(0, 1);
    dsmtp = o.dsk.mtp = r.coin(50) ? r.I(1, 5) : 0;
    dscap = o.ds_draft_cap = r.I(1, 8);
#endif
    if (dsmtp) o.mtp = 0;               // the parse refuses --mtp with DSpark: fabd and fabds share port + 16
#if G6_MODEL_FLASH
    rsplit = o.route_split_mode = r.coin(25) ? 1 : 0;
#endif

    PlanInputs in;
    PlanKnobs& k = in.k;
    k = k0;
    k.xtrace_pf = r.coin(20);
    k.pf_stripe = r.pick({0, 1, 2}) == 0 ? std::nullopt : std::optional<bool>(r.coin(50));
    if (r.coin(20)) k.pf_stripe_min = 65536;
    k.pf_rs = r.coin(50);
    k.fpf_attn_p1 = r.coin(8);
    k.fpf_dsa_p1 = r.coin(8);
    k.fpf_p1 = r.coin(8);
    if (r.coin(20)) k.qa_split_cpu = r.I(-1, 12);
    const int cpk = r.I(0, 2);
    if (cpk) k.cp = cpk == 2;
    k.cp_r = r.coin(4) ? r.pick<int64_t>({100, -64}) : r.pick<int64_t>({k0.cp_r, k0.cp_r, 0, 64, 32768, 131072});
    k.cp_s3 = r.coin(50);
    k.cp_step_rows = r.coin(3) ? 1 : r.pick<int64_t>({0, 0, 0, 0, 8, 24, 256});
    if (r.coin(20)) k.cp_mmax = r.pick<int64_t>({16, 512});
    if (r.coin(15)) k.cp_r2 = r.coin(15) ? 70 : r.pick<int64_t>({0, 1024, 65536});
    k.ds_prefill_w = r.coin(5) ? text(r, {"16", "5000"}) : text(r, {nullptr, nullptr, "1024", "2048", "4096", "4096"});
    k.ds_pf_tail = r.coin(4) ? text(r, {"5"}) : text(r, {nullptr, "0", "128", "128", "256"});
    k.ds_pf_l20_tail = r.coin(30);
    k.ds_pf_rs = r.coin(50);
    k.ds_mhc_row_shard = r.coin(5);
    k.ds_mhc_fuse = r.coin(60);
    k.ds_mhc_y_mxfp8 = r.coin(5);
    if (r.coin(15)) {                  // the row-shard recipe: W 4096, tail 128, RS and the fused mHC
        k.ds_prefill_w = "4096";
        k.ds_pf_tail = "128";
        k.ds_pf_rs = true;
        k.ds_mhc_fuse = true;
        k.ds_mhc_row_shard = true;
        k.ds_mhc_y_mxfp8 = r.coin(50);
    }
    k.ds_pf_pool = r.coin(4) ? text(r, {"999", "x"}) : text(r, {nullptr, nullptr, "0", "2", "4"});
    k.ds_pf_lanes = r.coin(3) ? text(r, {"3"}) : text(r, {nullptr, "1", "2", "2"});
    k.ds_pf_direct = r.coin(30);
    k.ds_subdump = r.coin(3);
    k.ds_stripe = r.coin(10);
    if (r.coin(30)) k.ds_stripe_min = 262144;
    k.ds_xtrace = r.coin(20);
    in.pf_sp = r.coin(30);
    in.pool_owner_mode = r.pick({0, 1, 2, 2, 2});
    in.dfl_ncoll = r.pick({12, 14});
    in.head_payload = r.pick<size_t>({0, 4096, 262144, 1 << 21});
    in.trunk_bytes = r.pick<size_t>({2, 4});
    for (int l : o.layers) in.full_index.push_back(geom::full_index(l));
    in.cp_build = r.coin(90);
    in.gather = r.coin(30);
    in.draft_gather = o.mtp && r.coin(30);

    char b[1024];
    snprintf(b, sizeof b,
             "case rank=%d ring=%d devs=%zu ips2=%zu pf_ips2=%zu proxy=%d MP=%d lanes=%d mtp=%d dfl=%d s2=%d K=%d chunk=%d sess=%d "
             "rt_win=%d rows=%s layers=%s NX=%d qp=%d,%d c8=%d dsmtp=%d dscap=%d rsplit=%d | xpf=%d stripe=%s,%lld pf_rs=%d "
             "fpf=%d%d%d qa_cpu=%s cp=%s,%lld,%d,%lld,%s,%s ds=%s,%s,%d,%d,%d,%d,%d,%s,%s,%d,%d,%d,%lld,%d | sp=%d pool=%d dncoll=%d "
             "head=%zu trunk=%zu cp_build=%d gather=%d,%d\n",
             c.rank, c.ring, c.ring_devs.size(), c.ips2.size(), o.pf_ips2.size(), c.proxy_cpu, o.MP, o.lanes, o.mtp, o.dfl,
             o.s2mode, o.K, o.plan.chunk, o.s2sess, o.rt_win, ints(o.row_counts).c_str(), ints(o.layers).c_str(), o.NX,
             o.qp_roll_s, o.qp_wire_e, c8, dsmtp, dscap, rsplit,
             k.xtrace_pf, opt(k.pf_stripe).c_str(), (long long)k.pf_stripe_min, k.pf_rs, k.fpf_attn_p1, k.fpf_dsa_p1, k.fpf_p1,
             opt(k.qa_split_cpu).c_str(), opt(k.cp).c_str(), (long long)k.cp_r, k.cp_s3, (long long)k.cp_step_rows,
             opt(k.cp_mmax).c_str(), opt(k.cp_r2).c_str(), opt(k.ds_prefill_w).c_str(), opt(k.ds_pf_tail).c_str(),
             k.ds_pf_l20_tail, k.ds_pf_rs, k.ds_mhc_row_shard, k.ds_mhc_fuse, k.ds_mhc_y_mxfp8, opt(k.ds_pf_pool).c_str(),
             opt(k.ds_pf_lanes).c_str(), k.ds_pf_direct, k.ds_subdump, k.ds_stripe, (long long)k.ds_stripe_min, k.ds_xtrace,
             in.pf_sp, in.pool_owner_mode, in.dfl_ncoll, in.head_payload, in.trunk_bytes, in.cp_build, in.gather,
             in.draft_gather);
    tx(b);
    const FabricPlan p = g6rt::fabric_plan(o, in);
    tx(g6rt::describe(p));
    if (!p.refusal.empty()) { ++refused; ++kinds["refusal: " + p.refusal.substr(0, 60)]; }
    for (const auto& f : p.fabrics) ++kinds[g6rt::fab_name(f.kind)];
}
}   // namespace

int main(int argc, char** argv)
{
    const bool dump = argc == 2 && !std::strcmp(argv[1], "--dump");
    ::clearenv();
    if (knob::init()) return 2;
    const PlanKnobs k0 = g6rt::plan_knobs();
#if G6_MODEL_FLASH
    const char* model = "flash";
#elif G6_MODEL_DS41
    const char* model = "ds41";
#else
    const char* model = "glm";
#endif
    const std::string shape = std::string(model) + " nr" + std::to_string(g6nr::NR) + " mx" + std::to_string(g6mx::MX);
    const int n = 20000;
    Rng r(0x6fab51c0 + g6nr::NR);
    Tx tx;
    tx.dump = dump;
    std::map<std::string, int> kinds;
    int refused = 0;
    for (int i = 0; i < n; ++i) one(r, k0, tx, kinds, refused);
    struct Want { const char* shape; uint64_t hash; };
    const Want wants[] = {
        {"glm nr6 mx8", 0xee0cd5ef43dd2635ull},
        {"glm nr4 mx32", 0x8f1da8b7421ae74bull},
        {"glm nr3 mx32", 0xc845c7c8c70e40b6ull},
        {"flash nr4 mx32", 0x623b99a841ce800aull},
        {"flash nr3 mx32", 0xf9dc44042fc2de31ull},
        {"flash nr2 mx32", 0x4cf96175c8f81c0eull},
        {"ds41 nr4 mx48", 0xc406a4de0972ea85ull},
    };
    const Want* w = nullptr;
    for (const Want& x : wants) if (shape == x.shape) w = &x;
    const bool ok = w && w->hash == tx.h;
    if (dump) return ok ? 0 : 1;
    printf("fabric plan %s: %d cases (%d refused), hash %016" PRIx64 "%s\n", shape.c_str(), n, refused, tx.h,
           ok ? "" : w ? " (MISMATCH)" : " (no recorded hash for this shape)");
    std::string cov = "  fabrics:";
    for (int i = 0; i < g6rt::kFabKinds; ++i) cov += " " + std::string(g6rt::fab_name(g6rt::FabKind(i))) + "=" + std::to_string(kinds[g6rt::fab_name(g6rt::FabKind(i))]);
    printf("%s\n", cov.c_str());
    int reasons = 0;
    for (const auto& [key, cnt] : kinds) reasons += key.rfind("refusal: ", 0) == 0;
    printf("  distinct refusals: %d\n", reasons);
    return ok ? 0 : 1;
}
