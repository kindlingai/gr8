// DFlash2 block drafter: one rank's draft step, as launch helpers in fabric phases.
//
// The drafter (a separate pack per rank, tools/convert/dflash2.py) drafts a block of K = 7 tokens in one forward:
//   context  for every accepted row of the last target step: the target residual after the layers in
//            target_layer_ids (6), concatenated -> fc (this rank's 6144/NR columns) -> all-gather (collective 0) ->
//            hidden_norm -> per drafter layer j: k_norm(k_proj) with RoPE and v_proj of this rank's KV heads, stored in
//            the ring at the row's position
//   block    8 rows at positions q0..q0+7: the bonus token (row 0) and 7 mask tokens, through 6 layers. Per layer:
//            in_norm -> conv prepare -> q, k, v of this rank's heads (q_norm / k_norm, NeoX RoPE; the block's k, v go
//            into the ring) -> attention (a query sees every row of the block, and the context keys k with
//            q - k < window) -> o_proj partial, all-reduce (collective 1 + 2j) -> conv finish -> residual ->
//            post_norm -> conv prepare -> gate/up, silu, down partial, all-reduce (collective 2 + 2j) -> conv finish ->
//            residual -> the next in_norm (or the final norm)
//   head     the target's lm_head on this rank's vocab slice. Argmax: gr8's head sampler (collective 2L + 1), row r
//            keyed to position q0 + r like the target's sample of that position. Selector mode (DFlash2): this rank's
//            top-16 per row -> all-gather (collective 2L + 1) -> global top-16 -> the walk with the selector codebooks
//   conv     two taps over the block rows r and r-1 (none for r = 0), per channel c: (base[side][tap][c] +
//            delta_r[side][tap][c / 16]); delta = kernel_projection of the prepare input, side 0 for the prepare conv,
//            side 1 for the finish conv. The finish conv is linear, so it runs after the all-reduce on the summed rows.
// KV: a ring of RING positions per layer and local KV head (fp16), indexed by position & (RING - 1). The window is 2048,
// so a step reads at most 2055 positions; DflIn::lo is the lowest position written for this sequence.
// The draft step has its own fabric region (NC collectives, decode-sized slots), step counter and DflIn / DflOut, and
// runs as one captured graph (dfl_step), or phase by phase with a loopback driver delivering between phases (dfl_phase).
// The target step writes its residual rows after the tapped layers into Dfl::feat (Model::dfl_tap, inside its graph).
// Prompt context: the prefill sub-batches write the owner rows of the last WIN prompt positions into Dfl::pf_feat
// (Prefill::dfl_tap); after the prefill, fill steps gather them 8 rows at a time (collectives 0..NT-1: every rank's rows,
// zero where it is not the owner, summed: exact) and run the context part on them (fc slices on collective NT).
#pragma once
#include "prefill.cuh"

namespace g6rt
{
namespace dfl
{
constexpr int B = 8, K = B - 1;               // block rows (bonus + drafts) = verify rows
constexpr int L = 6, NT = 6;                  // drafter layers, tapped target layers
constexpr int HD = 128, NH = 64, NKV = 8, INTER = 12288;
constexpr bool OK_NR = NKV % g6nr::NR == 0;   // TP4 (and TP1/2/8); a TP6 build compiles but refuses --dflash
constexpr int NHR = OK_NR ? NH / g6nr::NR : 1, NKR = OK_NR ? NKV / g6nr::NR : 1;
constexpr int NIR = OK_NR ? INTER / g6nr::NR : 16, NFR = OK_NR ? HID / g6nr::NR : 16;
constexpr int RING = 4096, WIN = 2048;
constexpr int GS = 16, NG = HID / GS, CPO = 4 * NG;    // conv group size, groups, kernel_projection outputs (2 sides x 2 taps)
constexpr int TOPK = 16, SELR = 256;
constexpr int NC = 2 * L + 2;                 // 0 fc slices, 1 + 2j / 2 + 2j layer j attention / MLP partial, 2L + 1 head
constexpr int NPH = NC + 1;                   // phases: phase p ends with the publish of collective p (the last: StepOut)
constexpr int SPLIT = 16;                     // attention key splits per head
constexpr float FEAT_SCALE = 1.f / 64.f;      // target residual -> fp16 features. fc is linear, and k_ctx_norm undoes it
constexpr unsigned ERR = 0x64000000u;         // drafter fabric wait error word
enum { C_ROWS, C_POS0, C_Q0, C_LO, C_N };     // device cells
static_assert(B <= MX, "the verify step of a block carries B rows, within the row budget (MX = 8, or 32 under DEC_S2)");
}

// host -> GPU once per draft step (pinned)
struct DflIn
{
    u64 step;
    int ctx_rows;                     // context rows: rows 0..ctx_rows-1 of Dfl::feat, at positions ctx_pos0..
    int ctx_pos0;
    int q0;                           // position of the bonus row. Row r predicts q0 + r
    int lo;                           // lowest position written for this sequence (keys below are never read)
    int tok0;                         // bonus token
    int query;                        // read by the host only: 0 = the context-only graph (no block, no head, no DflOut)
    glue::SampleRow srow[dfl::B];     // row r's sampling key (pos = q0 + r); row 0 unused
};
struct DflOut
{
    int tok[dfl::B];                  // tok[r] = draft for position q0 + r (r = 1..7)
    float lp[dfl::B];                 // lp[r]: the draft's log-probability (with the selector: among the walk's top 16)
};

struct DLayer
{
    bind::Gv kvc, ac, q, kv, o, mc, gu, down;
    const __nv_bfloat16 *in_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    const __nv_bfloat16 *abase = nullptr, *mbase = nullptr;   // [2 sides][2 taps][6144]
};

struct Dfl
{
    int rank = -1;
    bool sel = false;                 // DFlash2 selector walk (else argmax / gr8's sampler)
    int mask_id = 0;
    float theta = 1e6f, eps = 1e-5f;
    int tlayer[dfl::NT] = {};         // target_layer_ids
    g6::FabricDev f{};
    u64* stepp = nullptr;
    unsigned* tickets = nullptr;      // [NC]
    const DflIn* in = nullptr;        // pinned (real fabric) or device (loopback)
    DflOut* out = nullptr;
    int* cell = nullptr;              // [C_N] device
    int *tok = nullptr, *qpos = nullptr;    // [B]
    glue::SampleRow* srow = nullptr;  // [B]
    // weights
    bind::Gv fc, hproj;
    DLayer ly[dfl::L];
    const __nv_bfloat16 *hnorm = nullptr, *norm = nullptr, *pred = nullptr, *succ = nullptr, *emb = nullptr;
    const int* emb_rows = nullptr;    // dfltest: block row r reads row r of its small embedding table
    // buffers (device)
    __half* feat = nullptr;           // [B][NT][6144] fp16, FEAT_SCALE x the target residual
    __half* pf_feat = nullptr;        // [WIN][NT][6144] prompt context rows written by the prefill (this rank's owner rows)
    int pf_lo = 0, pf_hi = 0;         // the prompt positions the prefill taps keep, this request (pf_lo = row 0 of pf_feat)
    u32 *xf_feat = nullptr, *xb_feat = nullptr;
    float* fco = nullptr;             // [B][NFR] this rank's fc columns
    __half* ctxn = nullptr;           // [B][6144]
    u32 *xf_ctxn = nullptr, *xb_ctxn = nullptr;
    __half* kvc = nullptr;            // [L][B][2 NKR HD]
    __half *kring = nullptr, *vring = nullptr;   // [L][NKR][RING][HD]
    float* resid = nullptr;           // [B][6144]
    __half *x = nullptr, *xc = nullptr;
    u32 *xf_x = nullptr, *xb_x = nullptr, *xf_xc = nullptr, *xb_xc = nullptr;
    float *dlt_a = nullptr, *dlt_m = nullptr;   // [B][CPO]
    __half *q = nullptr, *kvq = nullptr;        // [B][NHR HD], [B][2 NKR HD]
    float* qf = nullptr;              // [B][NHR][HD] normed + rotated
    float* apart = nullptr;           // [NHR][SPLIT][B][HD + 4]
    __half* ao = nullptr;             // [B][NHR HD]
    u32 *xf_ao = nullptr, *xb_ao = nullptr;
    uint16_t* part = nullptr;         // [B][6144] BF16 partial staging
    __half *guo = nullptr, *act = nullptr;      // [B][2 NIR], [B][NIR]
    u32 *xf_act = nullptr, *xb_act = nullptr;
    // head
    bind::Head head;
    u32* xf_head = nullptr;
    float* hlog = nullptr;            // [B][vn]
    float* hp = nullptr;              // [B][SELR]
    int vlo = 0, vn = 0, nc = 0;
    int* tok_out = nullptr;
    float* lp = nullptr;
    unsigned* topp_miss = nullptr;
};

namespace dflk
{
using namespace dfl;

struct InArgs { const uint32_t* in; u64* stepp; int* cell; int* tok; int* qpos; glue::SampleRow* srow; int mask_id; };
// grid 1, block 128
__global__ void __launch_bounds__(128) k_in(InArgs a)
{
    __shared__ DflIn sh;
    const int t = threadIdx.x;
    for (int i = t; i < (int)(sizeof(DflIn) / 4); i += 128) reinterpret_cast<uint32_t*>(&sh)[i] = __ldcv(a.in + i);
    __syncthreads();
    if (t == 0)
    {
        *a.stepp = sh.step;
        a.cell[C_ROWS] = sh.ctx_rows; a.cell[C_POS0] = sh.ctx_pos0; a.cell[C_Q0] = sh.q0; a.cell[C_LO] = sh.lo;
    }
    if (t < B)
    {
        a.tok[t] = t ? a.mask_id : sh.tok0;
        a.qpos[t] = sh.q0 + t;
        a.srow[t] = sh.srow[t];
    }
}

// target residual rows -> fp16 features of tap slot s. grid (HID / 256, M), block 256
__global__ void __launch_bounds__(256) k_tap(const float* resid, __half* feat, int s)
{
    const int m = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    feat[((size_t)m * NT + s) * HID + c] = __float2half_rn(resid[(size_t)m * HID + c] * FEAT_SCALE);
}

// collective 0: every rank's [B][NFR] fp32 fc columns -> the full rows -> hidden_norm -> fp16. grid (RN_CL, B), block RN_T
struct CtxArgs { g6::FabricDev f; const u64* stepp; int c; const __nv_bfloat16* w; __half* out; float eps; };
__global__ void __cluster_dims__(glue::RN_CL, 1, 1) __launch_bounds__(glue::RN_T) k_ctx_norm(CtxArgs a)
{
    const u64 step = *a.stepp;
    const int m = blockIdx.y;
    const int col0 = blockIdx.x * glue::RN_COLS + threadIdx.x * 8;
    const int p = col0 / NFR;
    const float* src = reinterpret_cast<const float*>(glue::peer_slot(a.f, step, a.c, p)) + (size_t)m * NFR + (col0 - p * NFR);
    float x[8];
    const float4 v0 = __ldcv(reinterpret_cast<const float4*>(src)), v1 = __ldcv(reinterpret_cast<const float4*>(src) + 1);
    constexpr float u = 1.f / FEAT_SCALE;          // a power of two: exact
    x[0] = v0.x * u; x[1] = v0.y * u; x[2] = v0.z * u; x[3] = v0.w * u; x[4] = v1.x * u; x[5] = v1.y * u; x[6] = v1.z * u; x[7] = v1.w * u;
    glue::cluster_rms_store(x, a.w, a.eps, a.out + (size_t)m * HID, col0);
}

__device__ __forceinline__ float block_sum128(float v, float* sh)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    return sh[0] + sh[1] + sh[2] + sh[3];
}
// cos / sin of the NeoX pair of dimension d at pos: angle pos * theta^(-2 (d % 64) / 128), formed in double
__device__ __forceinline__ void rope_cs(int pos, int d, float theta, float& c, float& s)
{
    const double inv = exp(-(2.0 * (d & 63) / HD) * log((double)theta));
    double sd, cd;
    sincos((double)pos * inv, &sd, &cd);
    c = (float)cd;
    s = (float)sd;
}
// one head row of 128 (thread = dimension): RMSNorm with w, then NeoX RoPE at pos (rot = false: norm only)
__device__ __forceinline__ float norm_rope(float v, const __nv_bfloat16* w, float eps, float c, float s, float* sh, float* rot)
{
    const float ss = block_sum128(v * v, sh);
    v = v * rsqrtf(ss * (1.f / HD) + eps) * __bfloat162float(w[threadIdx.x]);
    rot[threadIdx.x] = v;
    __syncthreads();
    const float o = rot[threadIdx.x ^ 64];
    __syncthreads();
    return threadIdx.x < 64 ? v * c - o * s : v * c + o * s;
}

// k, v rows [rows][2 NKR HD] (k heads, then v heads) -> k_norm + RoPE -> ring at their positions.
// ctx = 1: rows 0..cell[C_ROWS]-1 at cell[C_POS0] + r, one layer per blockIdx.y. ctx = 0: the block rows at qpos.
struct KvArgs
{
    const __half* src[L];             // per layer
    const __nv_bfloat16* knorm[L];
    __half *kr[L], *vr[L];            // the layer's ring [NKR][RING][HD]
    const int* cell;
    const int* qpos;
    float theta, eps;
    int ctx;
};
// grid (B, layers), block 128
__global__ void __launch_bounds__(128) k_kv(KvArgs a)
{
    __shared__ float sh[4], rot[HD];
    const int r = blockIdx.x, j = blockIdx.y, d = threadIdx.x;
    if (a.ctx && r >= a.cell[C_ROWS]) return;
    const int pos = a.ctx ? a.cell[C_POS0] + r : a.qpos[r];
    float c, s;
    rope_cs(pos, d, a.theta, c, s);
    const __half* row = a.src[j] + (size_t)r * 2 * NKR * HD;
    const size_t slot = (size_t)(pos & (RING - 1)) * HD + d;
    for (int h = 0; h < NKR; ++h)
    {
        const float k = norm_rope(__half2float(row[h * HD + d]), a.knorm[j], a.eps, c, s, sh, rot);
        a.kr[j][(size_t)h * RING * HD + slot] = __float2half_rn(k);
        a.vr[j][(size_t)h * RING * HD + slot] = row[(NKR + h) * HD + d];
    }
}
// q rows [B][NHR HD] -> q_norm + RoPE -> fp32 [B][NHR][HD]. grid B, block 128
__global__ void __launch_bounds__(128) k_q(const __half* q, const __nv_bfloat16* qnorm, const int* qpos, float theta, float eps, float* out)
{
    __shared__ float sh[4], rot[HD];
    const int r = blockIdx.x, d = threadIdx.x;
    float c, s;
    rope_cs(qpos[r], d, theta, c, s);
    for (int h = 0; h < NHR; ++h)
        out[((size_t)r * NHR + h) * HD + d] = norm_rope(__half2float(q[((size_t)r * NHR + h) * HD + d]), qnorm, eps, c, s, sh, rot);
}

// attention partials: CTA = (local q head h, key split s), warp w = block row w. Keys: positions kmin..q0+7 where
// kmin = max(lo, q0 - WIN + 1), row 0's window. Row w starts at max(lo, q0 + w - WIN + 1). grid (NHR, SPLIT), block 256
struct AttnArgs { const float* q; const __half *kr, *vr; const int* cell; float* part; };
__global__ void __launch_bounds__(256) k_attn(AttnArgs a)
{
    const int h = blockIdx.x, s = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int q0 = a.cell[C_Q0], lo = a.cell[C_LO];
    const int kmin = max(lo, q0 - (WIN - 1)), kmax = q0 + B - 1;
    const int n = kmax - kmin + 1, per = (n + SPLIT - 1) / SPLIT;
    const int k0 = max(kmin + s * per, max(lo, q0 + w - (WIN - 1))), k1 = min(kmax + 1, kmin + (s + 1) * per);
    const int hk = h / (NHR / NKR);
    const float4 qv = *reinterpret_cast<const float4*>(a.q + ((size_t)w * NHR + h) * HD + l * 4);
    const float sc = rsqrtf((float)HD);
    const __half* kb = a.kr + (size_t)hk * RING * HD + l * 4;
    const __half* vb = a.vr + (size_t)hk * RING * HD + l * 4;
    float m = -INFINITY, sum = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = k0; k < k1; ++k)
    {
        const size_t o = (size_t)(k & (RING - 1)) * HD;
        const uint2 kk = *reinterpret_cast<const uint2*>(kb + o);
        const uint2 vv = *reinterpret_cast<const uint2*>(vb + o);
        const float2 k01 = __half22float2(as_h2(kk.x)), k23 = __half22float2(as_h2(kk.y));
        float d = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
        #pragma unroll
        for (int x = 16; x > 0; x >>= 1) d += __shfl_xor_sync(0xffffffffu, d, x);
        d *= sc;
        const float mn = fmaxf(m, d), cf = __expf(m - mn), p = __expf(d - mn);
        const float2 v01 = __half22float2(as_h2(vv.x)), v23 = __half22float2(as_h2(vv.y));
        sum = sum * cf + p;
        acc[0] = acc[0] * cf + p * v01.x; acc[1] = acc[1] * cf + p * v01.y;
        acc[2] = acc[2] * cf + p * v23.x; acc[3] = acc[3] * cf + p * v23.y;
        m = mn;
    }
    float* o = a.part + (((size_t)h * SPLIT + s) * B + w) * (HD + 4);
    *reinterpret_cast<float4*>(o + l * 4) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    if (l == 0) { o[HD] = m; o[HD + 1] = sum; }
}
// combine the splits -> fp16 [B][NHR HD]. grid (NHR, B), block 128
__global__ void __launch_bounds__(128) k_attn_comb(const float* part, __half* out)
{
    const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    float mx = -INFINITY;
    for (int s = 0; s < SPLIT; ++s) mx = fmaxf(mx, part[(((size_t)h * SPLIT + s) * B + r) * (HD + 4) + HD]);
    float num = 0.f, den = 0.f;
    for (int s = 0; s < SPLIT; ++s)
    {
        const float* o = part + (((size_t)h * SPLIT + s) * B + r) * (HD + 4);
        if (o[HD] == -INFINITY) continue;
        const float e = __expf(o[HD] - mx);
        num += e * o[d];
        den += e * o[HD + 1];
    }
    out[((size_t)r * NHR + h) * HD + d] = __float2half_rn(num / den);
}

// conv prepare: xc[r][c] = (b0[c] + dl[r][g]) x[r][c] + (r > 0) (b1[c] + dl[r][NG + g]) x[r-1][c], g = c / GS, dl = side 0 of
// the kernel projection, b0 / b1 = base[0][0] / base[0][1]. grid (HID / 256, B), block 256
__global__ void __launch_bounds__(256) k_conv(const __half* x, const float* dlt, const __nv_bfloat16* base, __half* xc)
{
    const int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x, g = c / GS;
    const float* dl = dlt + (size_t)r * CPO;
    float v = (__bfloat162float(base[c]) + dl[g]) * __half2float(x[(size_t)r * HID + c]);
    if (r) v += (__bfloat162float(base[HID + c]) + dl[NG + g]) * __half2float(x[(size_t)(r - 1) * HID + c]);
    xc[(size_t)r * HID + c] = __float2half_rn(v);
}

// all-reduce consume of a partial, conv finish (side 1), residual add, next RMSNorm. grid (RN_CL, B), block RN_T
struct ArArgs
{
    g6::FabricDev f;
    const u64* stepp;
    int c;
    const float* dlt;                 // [B][CPO]; side 1 at + 2 NG
    const __nv_bfloat16* base;        // [2][2][6144]; side 1 at + 2 HID
    float* resid;
    const __nv_bfloat16* w;
    __half* out;
    float eps;
};
__global__ void __cluster_dims__(glue::RN_CL, 1, 1) __launch_bounds__(glue::RN_T) k_ar_conv_norm(ArArgs a)
{
    const u64 step = *a.stepp;
    const int m = blockIdx.y;
    const int col0 = blockIdx.x * glue::RN_COLS + threadIdx.x * 8, g = col0 / GS;
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, yp[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int p = 0; p < a.f.nr; ++p)
    {
        const char* sl = glue::peer_slot(a.f, step, a.c, p);
        float b[8];
        glue::bf16x8_to_f(__ldcv(reinterpret_cast<const uint4*>(sl + ((size_t)m * HID + col0) * 2)), b);
        #pragma unroll
        for (int k = 0; k < 8; ++k) y[k] += b[k];
        if (m)
        {
            glue::bf16x8_to_f(__ldcv(reinterpret_cast<const uint4*>(sl + ((size_t)(m - 1) * HID + col0) * 2)), b);
            #pragma unroll
            for (int k = 0; k < 8; ++k) yp[k] += b[k];
        }
    }
    const float* dl = a.dlt + (size_t)m * CPO + 2 * NG;
    const __nv_bfloat16* b0 = a.base + 2 * HID + col0;
    const __nv_bfloat16* b1 = a.base + 3 * HID + col0;
    float4* r = reinterpret_cast<float4*>(a.resid + (size_t)m * HID + col0);
    const float4 r0 = r[0], r1 = r[1];
    float x[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
    #pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        float v = (__bfloat162float(b0[k]) + dl[g]) * y[k];
        if (m) v += (__bfloat162float(b1[k]) + dl[NG + g]) * yp[k];
        x[k] += v;
    }
    r[0] = make_float4(x[0], x[1], x[2], x[3]);
    r[1] = make_float4(x[4], x[5], x[6], x[7]);
    glue::cluster_rms_store(x, a.w, a.eps, a.out + (size_t)m * HID, col0);
}

// act[r][i] = silu(gate[r][i]) up[r][i]. grid (NIR / 256, B), block 256
__global__ void __launch_bounds__(256) k_silu(const __half* gu, __half* act)
{
    const int r = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    const float g = __half2float(gu[(size_t)r * 2 * NIR + i]), u = __half2float(gu[(size_t)r * 2 * NIR + NIR + i]);
    act[(size_t)r * NIR + i] = __float2half_rn(silu_f(g) * u);
}

// ---------------------------------------------------------------- selector head
struct Cand { float logit; int id; };
__device__ __forceinline__ bool cand_before(float la, int ia, float lb, int ib) { return la > lb || (la == lb && ia < ib); }
// this rank's top-16 (logit desc, id asc) of rows 1..7 -> send slot of collective c ([K][TOPK] Cand), then publish.
// grid K, block 1024: 16 rounds of a block argmax over the slice, excluding the ones taken
struct TopArgs { g6::FabricDev f; const u64* stepp; int c; const float* logits; int ldl, vlo, vn; unsigned* ticket; };
__global__ void __launch_bounds__(1024) k_top_local(TopArgs a)
{
    __shared__ float s_l[32];
    __shared__ int s_i[32];
    __shared__ int s_taken[TOPK];
    const int r = blockIdx.x + 1, t = threadIdx.x, w = t >> 5, l = t & 31;
    const float* row = a.logits + (size_t)r * a.ldl;
    Cand* dst = reinterpret_cast<Cand*>(glue::send_slot(a.f, *a.stepp, a.c)) + (size_t)(r - 1) * TOPK;
    for (int k = 0; k < TOPK; ++k)
    {
        float bl = -INFINITY;
        int bi = 0x7fffffff;
        for (int i = t; i < a.vn; i += 1024)
        {
            bool taken = false;
            for (int q = 0; q < k; ++q) taken |= s_taken[q] == i;
            if (!taken && cand_before(row[i], i, bl, bi)) { bl = row[i]; bi = i; }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            const float ol = __shfl_xor_sync(0xffffffffu, bl, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (cand_before(ol, oi, bl, bi)) { bl = ol; bi = oi; }
        }
        if (l == 0) { s_l[w] = bl; s_i[w] = bi; }
        __syncthreads();
        if (t == 0)
        {
            for (int i = 1; i < 32; ++i) if (cand_before(s_l[i], s_i[i], bl, bi)) { bl = s_l[i]; bi = s_i[i]; }
            s_taken[k] = bi;
            dst[k] = Cand{bl, a.vlo + bi};
        }
        __syncthreads();
    }
    glue::publish(a.f, a.c, g6::coll_seq(*a.stepp, a.c, a.f.ncoll), a.ticket);
}
// merge every rank's candidates -> global top-16 per row, then the walk, with prev = the bonus token at s = 1:
//   score(c) = logit(c) + sum_i pred[prev][i] hp[s][i] succ[c][i]
//   pick = argmax of score * inv_temp + gumbel(row s's key, c), or of score when greedy
//   prev = pick, s = s + 1, up to s = 7. lp_out = log-probability of the pick in softmax(score) over the 16
// grid 1, block 256. Warp w < 7 merges row w + 1, then warp w scores candidates w and w + 8.
struct WalkArgs
{
    g6::FabricDev f; const u64* stepp; int c;
    const float* hp;                  // [B][SELR]
    const __nv_bfloat16 *pred, *succ; // [vocab][SELR]
    const int* tok;                   // tok[0] = bonus
    const glue::SampleRow* srow;
    int* tok_out;
    float* lp_out;
};
__global__ void __launch_bounds__(256) k_walk(WalkArgs a)
{
    __shared__ float s_l[K][TOPK];
    __shared__ int s_i[K][TOPK];
    __shared__ float s_sc[TOPK];
    __shared__ int s_prev;
    const u64 step = *a.stepp;
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    glue::wait_peers(a.f, step, a.c, g6::coll_seq(step, a.c, a.f.ncoll), (size_t)K * TOPK * sizeof(Cand), ERR);
    if (w < K && l == 0)
    {
        float bl[TOPK];
        int bi[TOPK];
        int n = 0;
        for (int p = 0; p < a.f.nr; ++p)
        {
            const Cand* src = reinterpret_cast<const Cand*>(glue::peer_slot(a.f, step, a.c, p)) + (size_t)w * TOPK;
            for (int k = 0; k < TOPK; ++k)
            {
                const Cand x = src[k];
                if (n == TOPK && !cand_before(x.logit, x.id, bl[TOPK - 1], bi[TOPK - 1])) break;   // a rank's list is sorted
                int at = n < TOPK ? n++ : TOPK - 1;
                while (at > 0 && cand_before(x.logit, x.id, bl[at - 1], bi[at - 1])) { bl[at] = bl[at - 1]; bi[at] = bi[at - 1]; --at; }
                bl[at] = x.logit; bi[at] = x.id;
            }
        }
        for (int k = 0; k < TOPK; ++k) { s_l[w][k] = bl[k]; s_i[w][k] = bi[k]; }
    }
    if (t == 0) s_prev = a.tok[0];
    __syncthreads();
    for (int s = 0; s < K; ++s)
    {
        const int prev = s_prev;
        const float* hp = a.hp + (size_t)(s + 1) * SELR;
        for (int cc = w; cc < TOPK; cc += 8)
        {
            const int id = s_i[s][cc];
            float acc = 0.f;
            for (int i = l; i < SELR; i += 32)
                acc += __bfloat162float(a.pred[(size_t)prev * SELR + i]) * hp[i] * __bfloat162float(a.succ[(size_t)id * SELR + i]);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (l == 0) s_sc[cc] = s_l[s][cc] + acc;
        }
        __syncthreads();
        if (t == 0)
        {
            const glue::SampleRow sr = a.srow[s + 1];
            float best = -INFINITY;
            int bk = 0;
            for (int k = 0; k < TOPK; ++k)
            {
                const float v = sr.inv_temp == 0.f ? s_sc[k] : s_sc[k] * sr.inv_temp + glue::gumbel(sr, s_i[s][k]);
                if (v > best) { best = v; bk = k; }
            }
            float mx = -INFINITY, z = 0.f;
            for (int k = 0; k < TOPK; ++k) mx = fmaxf(mx, s_sc[k]);
            for (int k = 0; k < TOPK; ++k) z += expf(s_sc[k] - mx);
            s_prev = s_i[s][bk];
            a.tok_out[s + 1] = s_i[s][bk];
            a.lp_out[s + 1] = s_sc[bk] - mx - logf(z);   // log-probability of the pick among the TOPK (scores at T = 1)
        }
        __syncthreads();
    }
}

// prefill tap: rows r0.. of a sub-batch whose position is in [lo, hi) -> pf_feat[pos - lo][s]. grid (HID / 256, rows), block 256
__global__ void __launch_bounds__(256) k_pf_tap(const float* resid, const int* pos, int r0, int lo, int hi, __half* pf, int s)
{
    const int j = r0 + blockIdx.y, c = blockIdx.x * 256 + threadIdx.x, p = pos[j];
    if (p < lo || p >= hi) return;
    pf[((size_t)(p - lo) * NT + s) * HID + c] = __float2half_rn(resid[(size_t)j * HID + c] * FEAT_SCALE);
}
// fill: pf_feat rows ctx_pos0 - lo .. (+ B, zero past ctx_rows) of tap s -> send slot of collective s, then publish.
// grid 1, block 128
__global__ void __launch_bounds__(128) k_fill_put(const __half* pf, const int* cell, int s, g6::FabricDev f, const u64* stepp, unsigned* ticket)
{
    const u64 step = *stepp;
    const int row0 = cell[C_POS0] - cell[C_LO], rows = cell[C_ROWS];
    constexpr int R16 = HID * 2 / 16;
    uint4* dst = reinterpret_cast<uint4*>(glue::send_slot(f, step, s));
    for (int i = threadIdx.x; i < B * R16; i += 128)
    {
        const int r = i / R16;
        dst[i] = r < rows ? reinterpret_cast<const uint4*>(pf + ((size_t)(row0 + r) * NT + s) * HID)[i % R16] : make_uint4(0u, 0u, 0u, 0u);
    }
    glue::publish(f, s, g6::coll_seq(step, s, f.ncoll), ticket);
}
// fill: collectives 0..NT-1 summed over the ranks (one owner per row, zeros elsewhere) -> feat [B][NT][6144].
// grid (HID / 256, B, NT), block 256; after the NT waits
__global__ void __launch_bounds__(256) k_fill_get(g6::FabricDev f, const u64* stepp, __half* feat)
{
    const u64 step = *stepp;
    const int r = blockIdx.y, s = blockIdx.z, c = blockIdx.x * 256 + threadIdx.x;
    float v = 0.f;
    for (int p = 0; p < f.nr; ++p)
        v += __half2float(__ushort_as_half(__ldcv(reinterpret_cast<const unsigned short*>(glue::peer_slot(f, step, s, p)) + (size_t)r * HID + c)));
    feat[((size_t)r * NT + s) * HID + c] = __float2half_rn(v);
}

struct OutArgs { const int* tok; const float* lp; DflOut* out; };
// grid 1, block 32
__global__ void __launch_bounds__(32) k_out(OutArgs a)
{
    const int t = threadIdx.x;
    if (t < B) { a.out->tok[t] = a.tok[t]; a.out->lp[t] = a.lp[t]; }
}
}   // namespace dflk

// ---------------------------------------------------------------- build
// find: the drafter pack (tools/convert/dflash2.py); main: the built target model (embedding, lm_head, vocab slice).
// meta: manifest values (mask id, target layers, theta, eps). f: a fabric region with dfl::NC collectives and slots of at
// least B x 6144 BF16. sel: the DFlash2 selector (needs D.sel.* in the pack).
struct DflMeta { int mask_id = 0, nr = 0, block = 0, layers = 0, window = 0; int tl[dfl::NT] = {}; float theta = 0, eps = 0; bool sel = false; };
// D.meta (tools/convert/dflash2.py META): int32 [17]
inline DflMeta dfl_read_meta(const Find& find)
{
    const TensorRef t = find("D.meta");
    if (!t.p || t.nd != 1 || t.shape[0] != 17) die("dflash pack has no D.meta [17] (tools/convert/dflash2.py)");
    int v[17];
    G6RT_CK(cudaMemcpy(v, t.p, sizeof v, cudaMemcpyDeviceToHost));
    DflMeta m;
    m.mask_id = v[0]; m.nr = v[1]; m.block = v[2]; m.layers = v[3]; m.window = v[4];
    for (int i = 0; i < dfl::NT; ++i) m.tl[i] = v[5 + i];
    m.theta = (float)v[11];
    memcpy(&m.eps, &v[12], 4);
    m.sel = v[13] != 0;
    if (m.sel && (v[13] != dfl::SELR || v[14] != dfl::TOPK)) die("dflash pack: selector rank / top-k " + std::to_string(v[13]) + " / " +
                                                                std::to_string(v[14]) + ", this engine has 256 / 16");
    if (v[15] != dfl::GS || v[16] != 2) die("dflash pack: conv group / taps " + std::to_string(v[15]) + " / " + std::to_string(v[16]) + ", want 16 / 2");
    if (m.mask_id < 0 || m.mask_id >= glue::VOCAB) die("dflash pack: mask token id out of the vocabulary");
    return m;
}
// pf: the prompt-context taps (Dfl::pf_feat, WIN x 73.7 KB) for a driver that prefills
inline void dfl_build(Dfl& D, const Model& main, const Find& find, const DflMeta& meta, const g6::FabricDev& f, u64* stepp, bool sel,
                      cudaStream_t st, bool pf = false)
{
    using namespace dfl;
    if (meta.nr != g6nr::NR || meta.block != B || meta.layers != L || meta.window != WIN)
        die("dflash pack: built for " + std::to_string(meta.nr) + " ranks, block " + std::to_string(meta.block) + ", " +
            std::to_string(meta.layers) + " layers, window " + std::to_string(meta.window) + "; this engine needs " +
            std::to_string(g6nr::NR) + " ranks, block 8, 6 layers, window 2048");
    if (f.ncoll != NC || f.stride < (size_t)B * HID * 2 + 8) die("dflash fabric: wrong collective count or slot size");
    D.rank = main.rank; D.f = f; D.stepp = stepp; D.sel = sel;
    D.mask_id = meta.mask_id; D.theta = meta.theta; D.eps = meta.eps;
    for (int i = 0; i < NT; ++i) D.tlayer[i] = meta.tl[i];
    D.emb = main.emb;
    D.tickets = dmalloc<unsigned>(NC);
    D.cell = dmalloc<int>(C_N);
    D.tok = dmalloc<int>(B);
    D.qpos = dmalloc<int>(B);
    D.srow = dmalloc<glue::SampleRow>(B);
    auto frag = [&](int k) { return dmalloc<u32>((size_t)k / 16 * 64); };
    D.feat = dmalloc<__half>((size_t)B * NT * HID);
    if (pf) D.pf_feat = dmalloc<__half>((size_t)WIN * NT * HID);
    D.xf_feat = frag(NT * HID); D.xb_feat = frag(NT * HID);
    D.fco = dmalloc<float>((size_t)B * NFR);
    D.ctxn = dmalloc<__half>((size_t)B * HID);
    D.xf_ctxn = frag(HID); D.xb_ctxn = frag(HID);
    D.kvc = dmalloc<__half>((size_t)L * B * 2 * NKR * HD);
    D.kring = dmalloc<__half>((size_t)L * NKR * RING * HD);
    D.vring = dmalloc<__half>((size_t)L * NKR * RING * HD);
    D.resid = dmalloc<float>((size_t)B * HID);
    D.x = dmalloc<__half>((size_t)B * HID);
    D.xc = dmalloc<__half>((size_t)B * HID);
    D.xf_x = frag(HID); D.xb_x = frag(HID); D.xf_xc = frag(HID); D.xb_xc = frag(HID);
    D.dlt_a = dmalloc<float>((size_t)B * CPO);
    D.dlt_m = dmalloc<float>((size_t)B * CPO);
    D.q = dmalloc<__half>((size_t)B * NHR * HD);
    D.kvq = dmalloc<__half>((size_t)B * 2 * NKR * HD);
    D.qf = dmalloc<float>((size_t)B * NHR * HD);
    D.apart = dmalloc<float>((size_t)NHR * SPLIT * B * (HD + 4));
    D.ao = dmalloc<__half>((size_t)B * NHR * HD);
    D.xf_ao = frag(NHR * HD); D.xb_ao = frag(NHR * HD);
    D.part = dmalloc<uint16_t>((size_t)B * HID);
    D.guo = dmalloc<__half>((size_t)B * 2 * NIR);
    D.act = dmalloc<__half>((size_t)B * NIR);
    D.xf_act = frag(NIR); D.xb_act = frag(NIR);
    D.xf_head = frag(HID);
    D.vlo = main.vlo; D.vn = main.vn;
    D.hlog = dmalloc<float>((size_t)B * D.vn);
    D.hp = dmalloc<float>((size_t)B * SELR);
    D.tok_out = dmalloc<int>(B);
    D.lp = dmalloc<float>(B);
    D.topp_miss = dmalloc<unsigned>(1);
    // GEMV bindings: the codec follows the record (528 B groups = MXFP8 with fp16 fragments, 512 B = BF16 with BF16 fragments)
    auto gv = [&](const std::string& name, int tpc, int ks, const u32* xf, const u32* xb, void* y, int odt, int K_, int O_, bind::Gv& g) {
        const TensorRef t = find(name);
        if (!t.p) die("dflash pack has no " + name);
        if (t.nd != 3 || (t.shape[2] != 528 && t.shape[2] != 512)) die("dflash pack: " + name + " is not a GEMV record");
        const int codec = t.shape[2] == 528 ? bind::MXFP8 : bind::BF16W;
        if (const char* e = bind::make_gv(as_bind(t), codec, tpc, ks, B, codec == bind::BF16W ? xb : xf, y, odt, g)) die(name + ": " + e);
        if (g.K != K_ || g.O != O_) die("dflash pack: " + name + " is [" + std::to_string(g.O) + "][" + std::to_string(g.K) + "], want [" +
                                        std::to_string(O_) + "][" + std::to_string(K_) + "]");
    };
    auto vec = [&](const std::string& name, long long n) {
        const TensorRef t = find(name);
        if (!t.p || !t.bf16) die("dflash pack has no BF16 " + name);
        long long c = 1;
        for (int i = 0; i < t.nd; ++i) c *= t.shape[i];
        if (c != n) die("dflash pack: " + name + " has " + std::to_string(c) + " values, want " + std::to_string(n));
        return static_cast<const __nv_bfloat16*>(t.p);
    };
    gv("D.fc.gv", 1, 32, D.xf_feat, D.xb_feat, D.fco, gv::O_F32, NT * HID, NFR, D.fc);
    D.hnorm = vec("D.hidden_norm", HID);
    D.norm = vec("D.norm", HID);
    for (int j = 0; j < L; ++j)
    {
        DLayer& y = D.ly[j];
        const std::string p = "D" + std::to_string(j) + ".";
        gv(p + "kv.gv", 1, 32, D.xf_ctxn, D.xb_ctxn, D.kvc + (size_t)j * B * 2 * NKR * HD, gv::O_F16, HID, 2 * NKR * HD, y.kvc);
        gv(p + "aconv.gv", 1, 16, D.xf_x, D.xb_x, D.dlt_a, gv::O_F32, HID, CPO, y.ac);
        gv(p + "q.gv", 1, 16, D.xf_xc, D.xb_xc, D.q, gv::O_F16, HID, NHR * HD, y.q);
        gv(p + "kv.gv", 1, 32, D.xf_xc, D.xb_xc, D.kvq, gv::O_F16, HID, 2 * NKR * HD, y.kv);
        gv(p + "o.gv", 2, 8, D.xf_ao, D.xb_ao, D.part, gv::O_BF16, NHR * HD, HID, y.o);
        gv(p + "mconv.gv", 1, 16, D.xf_x, D.xb_x, D.dlt_m, gv::O_F32, HID, CPO, y.mc);
        gv(p + "gu.gv", 2, 16, D.xf_xc, D.xb_xc, D.guo, gv::O_F16, HID, 2 * NIR, y.gu);
        gv(p + "down.gv", 2, 8, D.xf_act, D.xb_act, D.part, gv::O_BF16, NIR, HID, y.down);
        y.in_norm = vec(p + "in_norm", HID);
        y.post_norm = vec(p + "post_norm", HID);
        y.q_norm = vec(p + "q_norm", HD);
        y.k_norm = vec(p + "k_norm", HD);
        y.abase = vec(p + "aconv.base", 4 * HID);
        y.mbase = vec(p + "mconv.base", 4 * HID);
    }
    if (sel)
    {
        gv("D.sel.hproj.gv", 1, 32, D.xf_x, D.xb_x, D.hp, gv::O_F32, HID, SELR, D.hproj);
        D.pred = vec("D.sel.pred", (long long)glue::VOCAB * SELR);
        D.succ = vec("D.sel.succ", (long long)glue::VOCAB * SELR);
    }
    // the target's lm_head on the drafter's final-norm rows (no MTP eh_proj)
    bind::HeadTensors ht = main.head_t;
    ht.eh = bind::Tensor();
    bind::Scratch sc{};
    sc.xf_head = D.xf_head;
    bind::HeadIo io;
    io.h = D.x; io.logits = D.hlog;
    if (const char* e = bind::make_head(ht, D.rank, B, sc, io, D.head)) die(std::string("dflash head: ") + e);
    G6RT_CK(cudaStreamSynchronize(st));
    size_t wb = 0;
    auto gvb = [&](const bind::Gv& g) { wb += bind::gv_bytes(g); };
    gvb(D.fc);
    for (const DLayer& y : D.ly) { gvb(y.kvc); gvb(y.ac); gvb(y.q); gvb(y.o); gvb(y.mc); gvb(y.gu); gvb(y.down); }
    fprintf(stderr, "# rank %d: dflash drafter %s, records %.3f GB, ring %.1f MB, prompt taps %.1f MB, %s\n", D.rank,
            sel ? "DFlash2 selector" : "argmax", wb / 1e9, 2.0 * L * NKR * RING * HD * 2 / 1e6, pf ? (double)WIN * NT * HID * 2 / 1e6 : 0.0,
            D.fc.codec == bind::MXFP8 ? "MXFP8" : "BF16");
}

// ---------------------------------------------------------------- the step, in phases
inline void dfl_xfrag(const __half* src, int K_, u32* f16, u32* b16, cudaStream_t st)
{
    bind::Xfrag a;
    a.on = 1; a.src = src; a.ld = K_; a.K = K_; a.M = dfl::B; a.f16 = f16; a.bf16 = b16;
    bind::launch(a, st);
}
inline void dfl_put(Dfl& D, int c, cudaStream_t st)
{
    k_put_publish<<<1, 128, 0, st>>>(reinterpret_cast<const uint4*>(D.part), D.f, D.stepp, c, dfl::B * (HID * 2 / 16), D.tickets + c);
}
inline void dfl_wait(Dfl& D, int c, size_t bytes, cudaStream_t st)
{
    glue::k_wait_peers<<<1, 32, 0, st>>>(D.f, D.stepp, c, bytes, dfl::ERR);
}
inline size_t dfl_payload(const Dfl& D, int c)
{
    using namespace dfl;
    if (c == 0) return (size_t)B * NFR * 4;
    if (c == NC - 1) return D.sel ? (size_t)K * TOPK * sizeof(dflk::Cand) : glue::head_bytes(B, D.nc);
    return (size_t)B * HID * 2;
}
inline std::vector<uint32_t> dfl_sizes(const Dfl& D, bool query)
{
    std::vector<uint32_t> v;
    for (int c = 0; c < (query ? dfl::NC : 1); ++c) v.push_back((uint32_t)dfl_payload(D, c));
    return v;
}
// a fill step: the NT feature gathers, then the fc slices
inline std::vector<uint32_t> dfl_fill_sizes()
{
    std::vector<uint32_t> v(dfl::NT, (uint32_t)(dfl::B * HID * 2));
    v.push_back((uint32_t)(dfl::B * dfl::NFR * 4));
    return v;
}
inline glue::HeadArgs dfl_head_args(Dfl& D)
{
    glue::HeadArgs a{D.f, D.stepp, dfl::NC - 1, dfl::B, D.hlog, D.vn, D.vlo, D.vn, D.srow, D.tickets + dfl::NC - 1, D.tok_out, D.lp,
                     nullptr, nullptr, D.nc, D.topp_miss};
    bind::head_args(D.head, a);
    return a;
}
// attention block of layer j (from x = in_norm output) up to the publish of collective 1 + 2j
inline void dfl_attn(Dfl& D, int j, cudaStream_t st)
{
    using namespace dfl;
    DLayer& y = D.ly[j];
    dfl_xfrag(D.x, HID, D.xf_x, D.xb_x, st);
    bind::launch(y.ac, st);
    dflk::k_conv<<<dim3(HID / 256, B), 256, 0, st>>>(D.x, D.dlt_a, y.abase, D.xc);
    dfl_xfrag(D.xc, HID, D.xf_xc, D.xb_xc, st);
    bind::launch(y.q, st);
    bind::launch(y.kv, st);
    dflk::k_q<<<B, 128, 0, st>>>(D.q, y.q_norm, D.qpos, D.theta, D.eps, D.qf);
    dflk::KvArgs ka{};
    ka.src[0] = D.kvq; ka.knorm[0] = y.k_norm;
    ka.kr[0] = D.kring + (size_t)j * NKR * RING * HD; ka.vr[0] = D.vring + (size_t)j * NKR * RING * HD;
    ka.cell = D.cell; ka.qpos = D.qpos; ka.theta = D.theta; ka.eps = D.eps; ka.ctx = 0;
    dflk::k_kv<<<dim3(B, 1), 128, 0, st>>>(ka);
    dflk::k_attn<<<dim3(NHR, SPLIT), 256, 0, st>>>(dflk::AttnArgs{D.qf, ka.kr[0], ka.vr[0], D.cell, D.apart});
    dflk::k_attn_comb<<<dim3(NHR, B), 128, 0, st>>>(D.apart, D.ao);
    dfl_xfrag(D.ao, NHR * HD, D.xf_ao, D.xb_ao, st);
    bind::launch(y.o, st);
    dfl_put(D, 1 + 2 * j, st);
}
// MLP block of layer j (from x = post_norm output) up to the publish of collective 2 + 2j
inline void dfl_mlp(Dfl& D, int j, cudaStream_t st)
{
    using namespace dfl;
    DLayer& y = D.ly[j];
    dfl_xfrag(D.x, HID, D.xf_x, D.xb_x, st);
    bind::launch(y.mc, st);
    dflk::k_conv<<<dim3(HID / 256, B), 256, 0, st>>>(D.x, D.dlt_m, y.mbase, D.xc);
    dfl_xfrag(D.xc, HID, D.xf_xc, D.xb_xc, st);
    bind::launch(y.gu, st);
    dflk::k_silu<<<dim3(NIR / 256, B), 256, 0, st>>>(D.guo, D.act);
    dfl_xfrag(D.act, NIR, D.xf_act, D.xb_act, st);
    bind::launch(y.down, st);
    dfl_put(D, 2 + 2 * j, st);
}
// consume collective c (a layer partial): conv finish with dlt / base side 1, residual, norm w -> x
inline void dfl_consume(Dfl& D, int c, const float* dlt, const __nv_bfloat16* base, const __nv_bfloat16* w, cudaStream_t st)
{
    dfl_wait(D, c, dfl_payload(D, c), st);
    dflk::k_ar_conv_norm<<<dim3(glue::RN_CL, dfl::B), glue::RN_T, 0, st>>>(dflk::ArArgs{D.f, D.stepp, c, dlt, base, D.resid, w, D.x, D.eps});
}
// the context part from collective c (fc slices) on: hidden_norm, this rank's K/V of every layer -> the ring
inline void dfl_ctx_store(Dfl& D, int c, cudaStream_t st)
{
    using namespace dfl;
    dfl_wait(D, c, (size_t)B * NFR * 4, st);
    dflk::k_ctx_norm<<<dim3(glue::RN_CL, B), glue::RN_T, 0, st>>>(dflk::CtxArgs{D.f, D.stepp, c, D.hnorm, D.ctxn, D.eps});
    dfl_xfrag(D.ctxn, HID, D.xf_ctxn, D.xb_ctxn, st);
    dflk::KvArgs ka{};
    for (int j = 0; j < L; ++j)
    {
        bind::launch(D.ly[j].kvc, st);
        ka.src[j] = D.kvc + (size_t)j * B * 2 * NKR * HD; ka.knorm[j] = D.ly[j].k_norm;
        ka.kr[j] = D.kring + (size_t)j * NKR * RING * HD; ka.vr[j] = D.vring + (size_t)j * NKR * RING * HD;
    }
    ka.cell = D.cell; ka.qpos = D.qpos; ka.theta = D.theta; ka.eps = D.eps; ka.ctx = 1;
    dflk::k_kv<<<dim3(B, L), 128, 0, st>>>(ka);
}
inline void dfl_in(Dfl& D, cudaStream_t st)
{
    dflk::k_in<<<1, 128, 0, st>>>(dflk::InArgs{reinterpret_cast<const uint32_t*>(D.in), D.stepp, D.cell, D.tok, D.qpos, D.srow, D.mask_id});
}
// phase p (0..NPH-1). query = false: the context part only (phases 0 and 1, phase 1 without the block)
inline void dfl_phase(Dfl& D, int p, cudaStream_t st, bool query = true)
{
    using namespace dfl;
    if (p == 0)
    {
        dfl_in(D, st);
        dfl_xfrag(D.feat, NT * HID, D.xf_feat, D.xb_feat, st);
        bind::launch(D.fc, st);
        k_put_publish<<<1, 128, 0, st>>>(reinterpret_cast<const uint4*>(D.fco), D.f, D.stepp, 0, (int)(dfl_payload(D, 0) / 16), D.tickets);
        return;
    }
    if (p == 1)
    {
        dfl_ctx_store(D, 0, st);
        if (!query) return;
        glue::k_embed_norm<<<dim3(glue::RN_CL, B), glue::RN_T, 0, st>>>(glue::EmbedArgs{D.emb_rows ? D.emb_rows : D.tok, D.emb, D.ly[0].in_norm, D.resid, D.x, D.eps});
        dfl_attn(D, 0, st);
        return;
    }
    if (p < NC - 1)                   // even p = 2 + 2j: layer j's attention partial in, its MLP out. Odd p = 3 + 2j: layer j's MLP in, layer j + 1's attention out
    {
        const int j = (p - 2) / 2;
        if (p % 2 == 0)
        {
            dfl_consume(D, p - 1, D.dlt_a, D.ly[j].abase, D.ly[j].post_norm, st);
            dfl_mlp(D, j, st);
        }
        else
        {
            dfl_consume(D, p - 1, D.dlt_m, D.ly[j].mbase, D.ly[j + 1].in_norm, st);
            dfl_attn(D, j + 1, st);
        }
        return;
    }
    if (p == NC - 1)                  // MLP of the last layer in, final norm, head out
    {
        dfl_consume(D, p - 1, D.dlt_m, D.ly[L - 1].mbase, D.norm, st);
        bind::launch(D.head.xf_h, st);
        bind::launch(D.head.lm_head, st);
        if (D.sel)
        {
            dfl_xfrag(D.x, HID, D.xf_x, D.xb_x, st);
            bind::launch(D.hproj, st);
            dflk::k_top_local<<<K, 1024, 0, st>>>(dflk::TopArgs{D.f, D.stepp, NC - 1, D.hlog, D.vn, D.vlo, D.vn, D.tickets + NC - 1});
        }
        else glue::k_head_local<<<B, 1024, 0, st>>>(dfl_head_args(D));
        return;
    }
    // p == NC: the merge (gr8's head merge in argmax mode, the walk in selector mode), StepOut
    if (D.sel)
        dflk::k_walk<<<1, 256, 0, st>>>(dflk::WalkArgs{D.f, D.stepp, NC - 1, D.hp, D.pred, D.succ, D.tok, D.srow, D.tok_out, D.lp});
    else if (D.nc > 0) glue::k_head_merge_tp<<<B, 1024, 0, st>>>(dfl_head_args(D));
    else glue::k_head_merge<<<1, 32, 0, st>>>(dfl_head_args(D));
    dflk::k_out<<<1, 32, 0, st>>>(dflk::OutArgs{D.tok_out, D.lp, D.out});
}
// the whole step on one stream (real fabric). query = false: the context part only
inline void dfl_step(Dfl& D, cudaStream_t st, bool query = true)
{
    for (int p = 0; p < (query ? dfl::NPH : 2); ++p) dfl_phase(D, p, st, query);
}
// a fill step (prompt context from Dfl::pf_feat), phases 0..2: 0 = the NT feature gathers out; 1 = in, fc slices out
// (collective NT); 2 = the context part. DflIn: ctx_rows, ctx_pos0 (a position in [pf_lo, pf_hi)), lo = pf_lo.
constexpr int DFL_FILL_PH = 3;
inline void dfl_fill_phase(Dfl& D, int p, cudaStream_t st)
{
    using namespace dfl;
    if (p == 0)
    {
        dfl_in(D, st);
        for (int s = 0; s < NT; ++s) dflk::k_fill_put<<<1, 128, 0, st>>>(D.pf_feat, D.cell, s, D.f, D.stepp, D.tickets + s);
        return;
    }
    if (p == 1)
    {
        for (int s = 0; s < NT; ++s) dfl_wait(D, s, (size_t)B * HID * 2, st);
        dflk::k_fill_get<<<dim3(HID / 256, B, NT), 256, 0, st>>>(D.f, D.stepp, D.feat);
        dfl_xfrag(D.feat, NT * HID, D.xf_feat, D.xb_feat, st);
        bind::launch(D.fc, st);
        k_put_publish<<<1, 128, 0, st>>>(reinterpret_cast<const uint4*>(D.fco), D.f, D.stepp, NT, (int)((size_t)B * NFR * 4 / 16), D.tickets + NT);
        return;
    }
    dfl_ctx_store(D, NT, st);
}
inline void dfl_fill_step(Dfl& D, cudaStream_t st)
{
    for (int p = 0; p < DFL_FILL_PH; ++p) dfl_fill_phase(D, p, st);
}

// G6_DFL_DUMP=<file> (measurement, rank 0): after a draft or fill step, one record: int32 kind (0 = context rows only,
// 1 = context rows + drafts), ctx_pos0, ctx_rows, q0, tok0, drafts[7]; then the context rows' features, fp16
// [ctx_rows][NT][6144] (FEAT_SCALE x the target residual). kind 2 = the committed tokens of the request: int32 n, then n ids.
inline void dfl_dump_step(FILE* f, const Dfl& D, const DflIn& di, const DflOut* o)
{
    if (!f) return;
    int h[12] = {o ? 1 : 0, di.ctx_pos0, di.ctx_rows, di.q0, di.tok0};
    for (int r = 1; r < dfl::B; ++r) h[4 + r] = o ? o->tok[r] : -1;
    fwrite(h, 4, 12, f);
    std::vector<__half> v((size_t)di.ctx_rows * dfl::NT * HID);
    G6RT_CK(cudaMemcpy(v.data(), D.feat, v.size() * 2, cudaMemcpyDeviceToHost));
    fwrite(v.data(), 2, v.size(), f);
}
inline void dfl_dump_tokens(FILE* f, const std::vector<int>& t)
{
    if (!f) return;
    int h[12] = {2, (int)t.size()};
    fwrite(h, 4, 12, f);
    fwrite(t.data(), 4, t.size(), f);
    fflush(f);
}

// ---------------------------------------------------------------- the target side
// Model::dfl_tap hook: after layer i's second consume, the residual rows of a tapped layer -> Dfl::feat
inline Dfl*& dfl_of(const Model& m)
{
    static Dfl* by_rank[8] = {};
    return by_rank[m.rank & 7];
}
inline void dfl_tap_launch(Model& m, int i, int M, cudaStream_t st)
{
    Dfl* D = dfl_of(m);
    for (int s = 0; s < dfl::NT; ++s)
        if (D->tlayer[s] == m.layers[i].l) dflk::k_tap<<<dim3(HID / 256, M), 256, 0, st>>>(m.resid, D->feat, s);
}
// Prefill::dfl_tap hook: after layer i of a prefill sub-batch, this rank's up-to-date rows (its owner rows in the RS form;
// rank 0 alone in the all-reduce form, so a gathered row is never summed twice) -> Dfl::pf_feat
inline void dfl_pf_tap_launch(Model& m, Prefill& P, int i, int M, cudaStream_t st)
{
    Dfl* D = dfl_of(m);
    if (!D->pf_feat || D->pf_hi <= D->pf_lo) return;
    for (int s = 0; s < dfl::NT; ++s)
    {
        if (D->tlayer[s] != m.layers[i].l) continue;
        const int r0 = P.rs ? pf_r0(M, P.f.rank, P.f.nr) : 0, rows = P.rs ? pf_rows(M, P.f.rank, P.f.nr) : P.f.rank == 0 ? M : 0;
        if (rows > 0) dflk::k_pf_tap<<<dim3(HID / 256, rows), 256, 0, st>>>(P.resid, P.pos, r0, D->pf_lo, D->pf_hi, D->pf_feat, s);
    }
}
inline void dfl_attach(Model& m, Dfl& D)
{
    for (int s = 0; s < dfl::NT; ++s)
    {
        bool found = false;
        for (const Layer& y : m.layers) found |= y.l == D.tlayer[s];
        if (!found) die("--dflash: target layer " + std::to_string(D.tlayer[s]) + " is not in --layers");
    }
    dfl_of(m) = &D;
    m.dfl_tap = dfl_tap_launch;
}

// ---------------------------------------------------------------- the sequence
// Prompt steps (rows past p.first, `chunk` rows per step) each feed the drafter their rows as context. The last one also
// drafts. Then rounds until n_gen tokens are committed:
//   verify   one target step over [last committed token, d1..d7] at positions base..base+7
//   accept   a = leading drafts equal to the target's samples (sampling is keyed by position and the drafts were
//            sampled with the same keys); the a + 1 sampled tokens are committed
//   draft    context = verify rows 0..a (positions base..base+a), block at base + a + 1 (bonus = the last sample)
// run_main(si, s, ms) -> StepOut and emit(tokens, count, first row in the last run_main step), as in mtp_sequence.
// run_draft(di, s, ms) -> DflOut. kv = the drafts a round verifies (1..K): a verify step carries kv + 1 rows, and the
// drafter still drafts all 7. sl (--spec-adaptive): each round verifies sl->choose() of the drafts (at most kv), from
// the drafter's own probabilities (DflOut::lp).
template <class RunMain, class RunDraft>
inline std::vector<int> dflash_sequence(const Plan& p, u64& main_step, u64& draft_step, RunMain&& run_main, RunDraft&& run_draft,
                                        MtpStats& S, int& steps, int lo, const std::function<bool(const int*, int, int)>& emit = {},
                                        int kv = dfl::K, SpecLen* sl = nullptr)
{
    using namespace dfl;
    const bool ad = sl && sl->on;
    float dl[B];                      // dl[r]: draft r's log-probability under the drafter (DflOut::lp)
    for (float& x : dl) x = NAN;
    const int n = (int)p.prompt.size(), PS = prompt_steps(p);
    const u64 step0 = main_step;
    std::vector<int> tokens = p.prompt;
    int d[B] = {};
    const float dtemp = p.rs ? 0.f : p.inv_temp;   // rejection sampling: greedy drafts (they must not use the target's noise)
    auto draft = [&](int rows, int pos0, int bonus, bool query, int s, double& ms) {
        DflIn di;
        memset(&di, 0, sizeof di);
        di.step = ++draft_step;
        di.ctx_rows = rows; di.ctx_pos0 = pos0; di.q0 = pos0 + rows; di.lo = lo; di.tok0 = bonus; di.query = query;
        for (int r = 0; r < B; ++r) di.srow[r] = glue::SampleRow{dtemp, p.seed, 0u, di.q0 + r, 0, p.top_p};
        const DflOut o = run_draft(di, s, ms);
        if (query) for (int r = 1; r < B; ++r) { d[r] = o.tok[r]; dl[r] = o.lp[r]; }
    };
    int s = 0;
    for (; s < PS; ++s)
    {
        const StepIn si = step_inputs(p, s, 0, step0);
        main_step = si.step;
        double ms = 0;
        const StepOut so = run_main(si, s, ms);
        const bool last = s == PS - 1;
        if (last)
        {
            tokens.push_back(so.tok[si.M - 1]);
            if (emit && emit(&so.tok[si.M - 1], 1, si.M - 1)) { steps += s + 1; return std::vector<int>(tokens.begin() + n, tokens.end()); }
        }
        draft(si.M, si.pos[0], last ? so.tok[si.M - 1] : 0, last, s, ms);
    }
    while ((int)tokens.size() - n < p.n_gen)
    {
        const int base = (int)tokens.size() - 1;
        double pe = 0;                // the decision's expected tokens (log line)
        const int k = ad ? sl->choose(dl, kv, kv, &pe) : kv;
        StepIn si;
        memset(&si, 0, sizeof si);
        si.step = ++main_step; si.M = k + 1;
        for (int r = 0; r <= k; ++r)
        {
            si.tok[r] = r ? d[r] : tokens[base];
            si.pos[r] = base + r;
            si.srow[r] = glue::SampleRow{p.inv_temp, p.seed, 0u, base + r + 1, 0, p.top_p, p.lp > 0 ? 1 : 0};
            si.tgt[r] = -1;
            si.steering_scale[r] = p.steering_scale;
        }
        si.context = base + k + 1;
        double tv = 0, tf = 0;
        const StepOut so = run_main(si, s, tv);
        int a = 0;
        while (a < k && d[a + 1] == so.tok[a]) ++a;
        for (int r = 0; r <= a; ++r) tokens.push_back(so.tok[r]);
        if (ad)
        {
            sl->observe(k, a, dl, pe);
            printf("round %ld base %d accepted %d of %d (expected %.2f tokens, hash %08x; p", S.rounds, base, a, k, pe, SpecLen::hash(dl, kv, k));
            for (int r = 1; r <= kv; ++r) printf(" %.3f", std::exp(dl[r]));
            printf(")\n");
        }
        else printf("round %ld base %d accepted %d of %d\n", S.rounds, base, a, kv);
        ++S.rounds; S.tokens += a + 1; ++S.acc[a]; S.drafted += k;
        if (emit && emit(so.tok, a + 1, 0))
        {
            S.t_verify.push_back(tv); S.t_first.push_back(0); S.t_chain.push_back(0);
            ++s;
            steps += s;
            const size_t end = std::min(tokens.size(), (size_t)n + p.n_gen);
            return std::vector<int>(tokens.begin() + n, tokens.begin() + end);
        }
        draft(a + 1, base, so.tok[a], true, s, tf);
        S.t_verify.push_back(tv); S.t_first.push_back(tf); S.t_chain.push_back(0);
        ++s;
    }
    steps += s;
    return std::vector<int>(tokens.begin() + n, tokens.begin() + n + p.n_gen);
}

}   // namespace g6rt
