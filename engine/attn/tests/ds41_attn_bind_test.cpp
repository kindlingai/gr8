// Host test (no GPU, no CUDA calls) for the attention weight binding of ds41_attn_plan.h: it runs plan_attn on the loader's tensor
// lists for one rank of the release (loader-rs ds41.rs, attention tensors only, tests/data/attn_plan_tp*.txt) for every
// layer in the list, plus mutated lists that must be refused. This is the check that would have caught o_a being asked for [1024][4096]
// when the pack has [G*1024][4096], without a Spark.
//
//   g++ -std=c++17 -O1 -I /usr/local/cuda/include -I engine -o /tmp/ds41_attn_bind_test engine/attn/tests/ds41_attn_bind_test.cpp
//   /tmp/ds41_attn_bind_test engine/attn/tests/data/attn_plan_tp*.txt
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "../ds41_attn_plan.h"

struct Rec { std::string dt; std::vector<long long> shape; };
using List = std::map<std::string, Rec>;

static int g_fail = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++g_fail; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static bool load(const char* path, List& L)
{
    std::ifstream f(path);
    if (!f) return false;
    std::string name, dt, sh;
    while (f >> name >> dt >> sh)
    {
        Rec r;
        r.dt = dt;
        std::string s = sh.substr(1, sh.size() - 2);
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) r.shape.push_back(atoll(tok.c_str()));
        L[name] = r;
    }
    return !L.empty();
}

static ds41a::PlanTensor finder(const List& L, const std::string& n)
{
    ds41a::PlanTensor t;
    auto it = L.find(n);
    if (it == L.end()) return t;
    t.p = (const void*)(uintptr_t)(0x1000 + std::hash<std::string>()(n) % 0xFFFF0000u);
    t.nd = (int)it->second.shape.size();
    for (int k = 0; k < t.nd && k < 4; ++k) t.shape[k] = it->second.shape[k];
    t.bf16 = it->second.dt == "bf16";
    t.fp32 = it->second.dt == "f32";
    return t;
}

static std::string run(const List& L, int l, ds41a::Plan& plan)
{
    auto find = [&](const std::string& n) { return finder(L, n); };
    const bool comp = L.count(ds41a::plan_name(l, "cmp.wgate")) != 0;
    return ds41a::plan_attn(find, l, comp, plan);
}

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s LIST...\n", argv[0]); return 2; }
    for (int a = 1; a < argc; ++a)
    {
        List L;
        if (!load(argv[a], L)) { fprintf(stderr, "cannot read %s\n", argv[a]); return 2; }
        std::set<int> layers;
        for (auto& kv : L) layers.insert(atoi(kv.first.c_str() + 1));
        printf("%s: %zu tensors, %zu layers\n", argv[a], L.size(), layers.size());
        for (int l : layers)
        {
            ds41a::Plan P;
            const std::string err = run(L, l, P);
            EXPECT(err.empty(), "layer %d: %s", l, err.c_str());
            if (!err.empty()) continue;
            const long long sink = L[ds41a::plan_name(l, "attn_sink")].shape[0];
            const bool comp = L.count(ds41a::plan_name(l, "cmp.wgate")) != 0;
            EXPECT(P.heads == sink && P.heads % 16 == 0, "layer %d: heads %d vs attn_sink %lld", l, P.heads, sink);
            EXPECT(P.G == P.heads / ds41a::HPG && P.G >= 1, "layer %d: G %d", l, P.G);
            const ds41a::Weights& W = P.W;
            EXPECT(W.wo_a.N == ds41a::OLORA && W.wo_a.K == ds41a::GROUP_IN, "layer %d: o_a N,K = %d,%d (one group's: 1024,4096)", l, W.wo_a.N, W.wo_a.K);
            EXPECT(W.wo_b.N == ds41a::HID && W.wo_b.K == P.G * ds41a::OLORA, "layer %d: o_b N,K = %d,%d", l, W.wo_b.N, W.wo_b.K);
            EXPECT(W.wq_b.N == P.heads * ds41a::HD && W.wq_b.K == ds41a::QA, "layer %d: q_b N,K = %d,%d", l, W.wq_b.N, W.wq_b.K);
            const ds41a::Fp8Mat* ms[5] = {&W.wq_a, &W.wq_b, &W.wkv, &W.wo_a, &W.wo_b};
            for (const ds41a::Fp8Mat* m : ms) EXPECT(m->w && m->s && m->gv, "layer %d: a matrix without codes/scales/gv", l);
            EXPECT(W.q_norm && W.kv_norm && W.sink, "layer %d: a norm/sink pointer is null", l);
            EXPECT((W.c_wkv != nullptr) == comp && (W.c_wgate != nullptr) == comp && (W.c_norm != nullptr) == comp, "layer %d: compressor pointers vs gate", l);
        }
        // refusals: each mutated list must fail with a message naming the tensor
        const int lz = *layers.begin();
        struct Mut { const char* what; const char* tensor; int dim; long long v; const char* dt; const char* want; };
        const Rec oa = L[ds41a::plan_name(lz, "o_a.mx.codes")];
        const long long rows = oa.shape[0];
        const Mut muts[] = {
            {"o_a codes with ONE group's rows", "o_a.mx.codes", 0, 1024, nullptr, "o_a.mx.codes"},
            {"o_a scales with one group's rows", "o_a.mx.scales", 0, 1024, nullptr, "o_a.mx.scales"},
            {"o_a gv with one group's tiles", "o_a.mx.gv", 0, 64, nullptr, "o_a.mx.gv"},
            {"q_b with 24 heads", "q_b.mx.codes", 0, 24 * 512, nullptr, "q_b"},
            {"q_b gv records of the wrong K", "q_b.mx.gv", 1, 39, nullptr, "q_b.mx.gv"},
            {"attn_sink as bf16", "attn_sink", -1, 0, "bf16", "attn_sink"},
            {"o_b with one group's K", "o_b.mx.codes", 1, 1024, nullptr, "o_b.mx.codes"},
        };
        for (const Mut& m : muts)
        {
            if (rows <= 1024 && std::strstr(m.what, "o_a")) continue;   // one group: the mutation is the pack's own shape
            List M = L;
            Rec& r = M[ds41a::plan_name(lz, m.tensor)];
            if (m.dim >= 0) r.shape[m.dim] = m.v;
            if (m.dt) r.dt = m.dt;
            ds41a::Plan P;
            const std::string err = run(M, lz, P);
            EXPECT(!err.empty() && err.find(m.want) != std::string::npos, "refusal '%s': got '%s'", m.what, err.c_str());
        }
        {
            List M = L;
            M.erase(ds41a::plan_name(lz, "q_a_norm"));
            ds41a::Plan P;
            const std::string err = run(M, lz, P);
            EXPECT(err.find("q_a_norm") != std::string::npos, "refusal 'q_a_norm missing': got '%s'", err.c_str());
        }
    }
    printf("%s (%d failure(s))\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
