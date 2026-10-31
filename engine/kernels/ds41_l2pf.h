// GR8_L2PF targets of a DS41 layer. MoE: the gate (l2pf::T_ROUTER) and the shared expert's three records (l2pf::T_SHARED),
// in the order ST_ROUTER and ST_SHARED read them (gate, fc1 = sh_w1 + sh_w3, fc2 = sh_w2). Attention (add_attn): the projection records
// the attention decode GEMV reads first, q_a + kv (l2pf::T_XIN) and q_b (l2pf::T_QB), for the MLP-collective window of the layer before. Global scope, host only: the
// sizes are the shapes bind_moe checks. The queue, the pacing,
// the windows and the env (GR8_L2PF*) are kernels/l2pf.cuh's; ds41_step.cuh places the windows (ds41_l2pf_window).
// Nothing here launches a kernel or reads a weight.
#pragma once
#include <cstddef>
#include "l2pf.cuh"

namespace ds41l2pf
{
constexpr size_t GV_REC = 528;   // one MXFP8 gv record: 16 rows x 32 k (bind_moe: sh.w1 / w3 [IW/16][HID/32][528], sh.w2 [HID/16][IW/32][528])

// the attention decode projection records (Fp8Mat::gv, MXFP8 GEMV records [N/16][K/32][528]); a null pointer adds nothing
inline void add_gv(l2pf::Plan& P, int t, const void* gv, int N, int K)
{
    if (gv && N > 0 && K > 0) P.add(t, gv, static_cast<size_t>(N / 16) * (K / 32) * GV_REC);
}
// layer attention's first reads: q_a [1280][5120] and kv [512][5120] (T_XIN, in the order the projection issues them), q_b [heads*512][1280] (T_QB).
// The o_a / o_b records are not added: they are read after the attention core, too late for the MLP window of the layer before.
inline void add_attn(l2pf::Plan& P, const void* qa, int qa_n, int qa_k, const void* kv, int kv_n, int kv_k, const void* qb, int qb_n, int qb_k)
{
    add_gv(P, l2pf::T_XIN, qa, qa_n, qa_k);
    add_gv(P, l2pf::T_XIN, kv, kv_n, kv_k);
    add_gv(P, l2pf::T_QB, qb, qb_n, qb_k);
}

// gate: bf16 [ne][hid]. shared: w1, w3 = [iw/16][hid/32] records, w2 = [hid/16][iw/32] records. A null pointer adds nothing.
inline void add_moe(l2pf::Plan& P, int ne, int hid, int iw, const void* gate, const void* sh_w1, const void* sh_w3, const void* sh_w2)
{
    const size_t fc1 = static_cast<size_t>(iw / 16) * (hid / 32) * GV_REC, fc2 = static_cast<size_t>(hid / 16) * (iw / 32) * GV_REC;
    if (gate) P.add(l2pf::T_ROUTER, gate, static_cast<size_t>(ne) * hid * 2);
    if (sh_w1) P.add(l2pf::T_SHARED, sh_w1, fc1);
    if (sh_w3) P.add(l2pf::T_SHARED, sh_w3, fc1);
    if (sh_w2) P.add(l2pf::T_SHARED, sh_w2, fc2);
}
}   // namespace ds41l2pf
