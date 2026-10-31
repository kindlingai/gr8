// DeepSeek-V4.1-Flash-1017 attention A: the shape checks and the pointer binding of one layer's attention weights, with no CUDA
// calls. ds41_attn_ops.cuh's ds41_attn_bind runs it through the Model's tensor finder and then allocates the device state; the host
// test tests/ds41_attn_bind_test.cpp runs the same code on the loader's tensor list, so a shape rule that does not
// match the pack is caught without a GPU.
#pragma once
#include <cstdio>
#include <string>
#include "ds41_attn_types.h"

namespace ds41a
{
struct PlanTensor
{
    const void* p = nullptr;   // null = the pack has no such tensor
    int nd = 0;
    long long shape[4] = {0, 0, 0, 0};
    bool bf16 = false, fp32 = false;
};

struct Plan
{
    Weights W{};   // inv (the rope table) is the caller's: it is a device allocation
    int heads = 0, G = 0;
};

inline std::string plan_name(int l, const char* nm)
{
    char b[96];
    snprintf(b, sizeof b, "L%03d.%s", l, nm);
    return b;
}

// Find: PlanTensor(const std::string&). err keeps the first failure; after it every call answers null.
template <class Find>
struct Planner
{
    Find& find;
    int l;
    std::string err;

    void fail(const std::string& what) { if (err.empty()) err = "DS41 attention bind: " + what; }

    const uint8_t* u8(const std::string& name, long long n0, long long n1)
    {
        const PlanTensor t = find(name);
        if (!t.p || t.nd != 2 || t.shape[0] != n0 || t.shape[1] != n1) { fail("bad tensor " + name); return nullptr; }
        return (const uint8_t*)t.p;
    }
    // N, K: one group's matrix. rows: the tensor's first dim (default N; G * N for the grouped o_a, whose groups are stacked in the rows).
    Fp8Mat fp8(const char* nm, long long N, long long K, long long rows = 0)
    {
        if (!rows) rows = N;
        const std::string p = plan_name(l, nm);
        Fp8Mat w{};
        w.w = u8(p + ".mx.codes", rows, K);
        w.s = u8(p + ".mx.scales", rows, K / 32);   // the pack's form: [rows][K/32], the block-row scale repeated per row
        w.N = (int)N;
        w.K = (int)K;
        // the pack's GEMV records [rows/16][K/32][528] (densev.cuh Mxfp8), when the pack has them: the decode projections use them
        const PlanTensor g = find(p + ".mx.gv");
        if (g.p)
        {
            if (g.nd != 3 || g.shape[0] != rows / 16 || g.shape[1] != K / 32 || g.shape[2] != 528) fail("bad tensor " + p + ".mx.gv");
            else w.gv = (const uint8_t*)g.p;
        }
        return w;
    }
    const bf16* vec(const char* nm, long long n, bool want_f32 = false)
    {
        const std::string p = plan_name(l, nm);
        const PlanTensor t = find(p);
        if (!t.p || t.nd != 1 || t.shape[0] != n || (want_f32 ? !t.fp32 : !t.bf16)) { fail("bad tensor " + p); return nullptr; }
        return (const bf16*)t.p;
    }
    const bf16* mat_bf16(const char* nm, long long n0, long long n1)
    {
        const std::string p = plan_name(l, nm);
        const PlanTensor t = find(p);
        if (!t.p || !t.bf16 || t.nd != 2 || t.shape[0] != n0 || t.shape[1] != n1) { fail("bad compressor of layer " + std::to_string(l)); return nullptr; }
        return (const bf16*)t.p;
    }
};

// Binds layer l's attention weights into out.W. compressor: the layer has the compressor gate (ratio-2 KV sources). "" = ok, else the message
// to die with. The rank's heads come from q_b (the pack slices q_b, attn_sink, o_a and o_b by o group); heads % 16 == 0 is q_b's GEMV tile rule.
template <class Find>
std::string plan_attn(Find&& find, int l, bool compressor, Plan& out)
{
    Planner<Find> P{find, l, std::string()};
    const PlanTensor qb = find(plan_name(l, "q_b.mx.codes"));
    if (!qb.p || qb.nd != 2 || qb.shape[1] != QA || qb.shape[0] % (16 * HD)) return "DS41 attention bind: bad q_b of layer " + std::to_string(l);
    out.heads = (int)(qb.shape[0] / HD);
    out.G = out.heads / HPG;
    Weights& W = out.W;
    W.wq_a = P.fp8("q_a", QA, HID);
    W.wq_b = P.fp8("q_b", (long long)out.heads * HD, QA);
    W.wkv = P.fp8("kv", HD, HID);
    W.wo_a = P.fp8("o_a", OLORA, GROUP_IN, (long long)out.G * OLORA);   // the pointers cover all G groups of this rank
    W.wo_b = P.fp8("o_b", HID, (long long)out.G * OLORA);
    W.q_norm = P.vec("q_a_norm", QA);
    W.kv_norm = P.vec("kv_norm", HD);
    W.sink = (const float*)P.vec("attn_sink", out.heads, true);
    if (compressor)
    {
        W.c_wkv = P.mat_bf16("cmp.wkv", HD, HID);
        W.c_wgate = P.mat_bf16("cmp.wgate", HD, HID);
        W.c_norm = P.vec("cmp.norm", HD);
    }
    return P.err;
}
}   // namespace ds41a
