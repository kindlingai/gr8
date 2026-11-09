// DSpark drafter attention (layers 40..42 of DeepSeek-V4.1-Flash-1017): a fork of the kernels in ds41_attn.cuh. Same maths,
// two differences, both from the release's own DSparkAttention (model.py:1033-1075):
//
//   1. Positions come from an explicit device array `pos` [rows] instead of start + row. The draft block's rows sit at q+1 .. q+5 of
//      each session and a verify step's rows at P .. P+k, so rows of different sessions do not share one start.
//   2. Block mode: the 5 rows of a session attend to the same 133 keys and there is no causal mask. Keys = the main window (positions
//      q-127 .. q, or 0 .. q while q < 127) plus the block's own 5 keys (q+1 .. q+5), which the layer wrote into the ring just before.
//      The release concatenates the block keys after the 128-slot window cache; here they sit in the 136-row ring (docs/DEEPSEEK-DSPARK.md
//      section 2), the key set is the same.
//
// Everything else (kv_norm eps, plain rope theta 10000 through Weights::inv, fp8 quantize-dequantize of the key, learned sink, online
// softmax with bf16 P) is ds41a's code, reused unchanged: the helpers, the gemm, the rmsnorm and the weights/scratch structs come from
// ds41_attn.cuh. Header-only; no gr8 runtime dependency.
//
// Ring: one layer's ring is `base + session * sess_stride` elements, ring_rows (136) rows of 512 bf16 each, slot = position % ring_rows.
// Rejected verify-row keys cannot destroy a key a later query needs: the slot of position p holds p - 136 before, and p - 136 < q - 127.
#pragma once
#include <cstdio>
#include <cstdlib>
#include "ds41_attn.cuh"
#include "../knobs-rs/gr8_knobs.h"
#include "../runtime/ds41_dspark_api.h"

namespace ds41dsp
{
using ds41a::bf16;
using ds41a::HID;      // beside ::HID (kernels/common.cuh) in the full engine build, a bare HID under "using namespace ds41a" is ambiguous
using ds41a::HD;
using ds41a::NOPE;
using ds41a::ROPE;
using ds41a::WIN;

constexpr int BLOCK_MAX = 8;                       // ring_rows - WIN: block rows the ring can hold next to a full window

// ------------------------------------------------------------------------------------------------ projection route
// GR8_DS_DSP_GV=1: the DSpark projections (the draft layers' q_a, q_b, kv, o_a, o_b and ds_main's main_proj + 3 kv) run through ds41a::proj_mm,
// i.e. the MXFP8 GEMV for 1..8 rows (DSpark runs one session: 5 draft / 6 verify rows), instead of the scalar-FMA k_gemm_mxfp8. A matrix that
// cannot take it (no gv records, hooks unbound) is a hard error with the knob on, never a silent GEMM. Default off: every site calls
// ds41a::launch_gemm. The GEMV changes the fp32 summation order only (same fp8 products, same activations).
inline int g_dsp_gv = -1;                          // -1: knob not read yet (a test sets it to 0 / 1)
inline bool dsp_gv_on()
{
    if (g_dsp_gv < 0) g_dsp_gv = knob::ds_dsp_gv() ? 1 : 0;
    return g_dsp_gv != 0;
}
// the one rule: knob on and ds41a's GEMV rule (g_use_gv, 1..8 rows, gv records, hooks bound)
inline bool dsp_use_gv(const ds41a::Fp8Mat& W, int rows) { return dsp_gv_on() && ds41a::use_gv(W, rows); }
// Knob on and the GEMV rule fails (more than 8 rows, no gv records, hooks unbound): never hide it behind the GEMM. Tests replace g_dsp_fail.
inline void dsp_die(const ds41a::Fp8Mat& W, int rows)
{
    fprintf(stderr, "ds41dsp: GR8_DS_DSP_GV=1 but the projection N=%d K=%d at %d rows cannot take the GEMV (gv records %s, g_use_gv %d)\n", W.N, W.K, rows,
            W.gv ? "present" : "MISSING", (int)ds41a::g_use_gv);
    abort();
}
inline void (*g_dsp_fail)(const ds41a::Fp8Mat&, int) = &dsp_die;
// GR8_DS_DSP_TC=1: a prefill launch (wide, more than 8 rows, one group) calls the attention block's tensor-core MXFP8 GEMM, ds41_gemm_mx_tc
// (ds41_attn.cu, 74 TFLOP/s at M = 1024 against the scalar k_gemm_mxfp8), instead of ds41a::launch_gemm. Default off. The entry declines a shape or a
// pointer it cannot take (rc 1 / 2) and launches nothing then: the scalar GEMM runs. It is declared weak so this TU links without that entry (null = scalar).
// The shapes DS_MAIN gives it: main_proj N = 5120 K = 15360 lda = 15360 lsa = 480 ldy = 5120; each wkv N = 512 K = 5120 lda = 5120 lsa = 160 ldy = 512; all inside
// K % 128, N % 64, lda % 16, lsa % 4, even ldy; MainS pieces are 256-aligned (MainS::carve), the weights are the Fp8Mat the scalar GEMM reads.
extern "C" int ds41_gemm_mx_tc(const uint8_t* A, int lda, const uint8_t* SA, int lsa, const uint8_t* W, const uint8_t* SW, void* Y, int ldy, int M, int N, int K,
                               void* st) __attribute__((weak));
inline int g_dsp_tc = -1;                          // -1: knob not read yet (a test sets it to 0 / 1)
inline int g_dsp_tc_rc[3] = {0, 0, 0};             // launches by ds41_gemm_mx_tc: [0] taken (rc 0), [1] shape declined, [2] pointer declined (boot line / test)
inline bool dsp_tc_on()
{
    if (g_dsp_tc < 0) g_dsp_tc = knob::ds_dsp_tc() ? 1 : 0;
    return g_dsp_tc != 0;
}
// true = the tensor-core GEMM was launched on `st`; false = nothing was launched (knob off, entry absent, shape or pointer declined): the caller runs launch_gemm.
inline bool dsp_try_tc(const ds41a::Fp8Mat& W, const uint8_t* A, int lda, const uint8_t* SA, int lsa, bf16* Y, int ldy, int M, cudaStream_t st)
{
    if (!dsp_tc_on() || !ds41_gemm_mx_tc) return false;
    const int rc = ds41_gemm_mx_tc(A, lda, SA, lsa, W.w, W.s, Y, ldy, M, W.N, W.K, st);
    if (rc >= 0 && rc <= 2) ++g_dsp_tc_rc[rc];
    return rc == 0;
}

// ds41a::proj_mm's arguments. gv_ready: xf already holds A's table (the caller passed dsp_use_gv of the matrix that built it).
// wide: a prefill launch (ds_main_rows). With the knob on, M > 8 rows falls to the GEMM instead of the loud failure; 1..8 rows still take the GEMV.
// SESS > 1 (several sessions in one DSpark step: up to 8 x 6 verify / 8 x 5 draft rows): M > 8 also takes the GEMM (the GEMV is 1..8 rows);
// build() still checks with the knob on that every projection takes the GEMV at the one-session shapes.
// With GR8_DS_DSP_TC=1 that wide GEMM is the tensor-core entry when it takes the shape (dsp_try_tc), else the scalar one.
inline void dsp_mm(const ds41a::Fp8Mat& W, const uint8_t* A, int lda, const uint8_t* SA, int lsa, uint32_t* xf, bool gv_ready, bf16* Y, int ldy, int M,
                   int groups, size_t a_gs, size_t sa_gs, size_t y_gs, cudaStream_t st, bool wide = false)
{
    if (dsp_use_gv(W, M))
    {
        if (!gv_ready) ds41a::launch_xfrag(A, lda, SA, lsa, M, W.K, groups, a_gs, sa_gs, xf, st);
        ds41a::launch_gemv(xf, W, Y, ldy, M, groups, y_gs, st);
    }
    else if (dsp_gv_on() && M <= 8) g_dsp_fail(W, M);
    else if (wide && M > 8 && groups == 1 && dsp_try_tc(W, A, lda, SA, lsa, Y, ldy, M, st)) {}
    else ds41a::launch_gemm(A, lda, SA, lsa, W, Y, ldy, M, groups, a_gs, sa_gs, y_gs, st);
}


// ------------------------------------------------------------------------------------------------ rope on the last 64 dims
// ds41a::k_rope_tail with the position of row r read from pos[r].
__global__ void k_rope_tail_pos(bf16* x, long ld, int rows, int heads, const int* __restrict__ pos, const float* __restrict__ inv, int inverse)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)rows * heads * (ROPE / 2)) return;
    const int i = (int)(t % (ROPE / 2));
    const long rh = t / (ROPE / 2);
    const int h = (int)(rh % heads), r = (int)(rh / heads);
    bf16* p = x + (size_t)r * ld + (size_t)h * HD + NOPE + 2 * i;
    float ra, rb;
    ds41a::rot_pair(__bfloat162float(p[0]), __bfloat162float(p[1]), (float)pos[r] * inv[i], inverse != 0, ra, rb);
    p[0] = __float2bfloat16_rn(ra);
    p[1] = __float2bfloat16_rn(rb);
}

// ------------------------------------------------------------------------------------------------ window KV prep
// ds41a::k_kv_prep with the ring slot of row r = pos[r] % ring_rows of session r / rows_per_sess. kv0 [rows][512] (wkv out) ->
// kv_norm -> rope tail (angle pos * inv) -> fp8 quantize-dequantize over 512 -> the ring row. One block of 512 per row.
// Used for both the verify rows' main keys (rows_per_sess = k + 1, ST_DS_MAIN) and the draft block's own keys (rows_per_sess = 5).
__global__ void __launch_bounds__(512) k_kv_prep_pos(const bf16* __restrict__ kv0, const bf16* __restrict__ nw, const float* __restrict__ inv,
                                                     const int* __restrict__ pos, int rows_per_sess, bf16* ring, size_t sess_stride,
                                                     int ring_rows, const int* __restrict__ dsess = nullptr)
{
    __shared__ float sh[32];
    const int row = blockIdx.x, c = threadIdx.x;
    const int p = pos[row];
    if (p < 0) return;                                      // block-uniform; before the first barrier
    const float v = __bfloat162float(kv0[(size_t)row * HD + c]);
    const float ss = ds41a::block_sum(v * v, sh);
    const float r = rsqrtf(ss / (float)HD + ds41a::EPS);
    float x = ds41a::bf16r(__bfloat162float(nw[c]) * (v * r));
    const float partner = __shfl_xor_sync(0xFFFFFFFFu, x, 1);
    if (c >= NOPE)
    {
        const int i = (c - NOPE) >> 1;
        const float ang = (float)p * inv[i];
        float ra, rb;
        if ((c & 1) == 0) { ds41a::rot_pair(x, partner, ang, false, ra, rb); x = ds41a::bf16r(ra); }
        else { ds41a::rot_pair(partner, x, ang, false, ra, rb); x = ds41a::bf16r(rb); }
    }
    uint8_t code, e;
    const bf16 o = __float2bfloat16_rn(ds41a::act_quant_lane(x, code, e));
    ring[(size_t)(dsess ? dsess[row] : row / rows_per_sess) * sess_stride + (size_t)(p % ring_rows) * HD + c] = o;   // SESS > 1: the row's S2 slot
}

// ------------------------------------------------------------------------------------------------ o prep (inverse rope + quantize)
// ds41a::k_o_prep with the position of row r read from pos[r]. One block of 512 per (row, head).
__global__ void __launch_bounds__(512) k_o_prep_pos(const bf16* __restrict__ o, int heads, const int* __restrict__ pos, const float* __restrict__ inv,
                                                    uint8_t* codes, uint8_t* sc)
{
    const int row = blockIdx.x / heads, h = blockIdx.x % heads, c = threadIdx.x;
    const size_t base = ((size_t)row * heads + h) * HD;
    float x = __bfloat162float(o[base + c]);
    const float partner = __shfl_xor_sync(0xFFFFFFFFu, x, 1);
    if (c >= NOPE)
    {
        const int i = (c - NOPE) >> 1;
        const float ang = (float)pos[row] * inv[i];
        float ra, rb;
        if ((c & 1) == 0) { ds41a::rot_pair(x, partner, ang, true, ra, rb); x = ds41a::bf16r(ra); }
        else { ds41a::rot_pair(partner, x, ang, true, ra, rb); x = ds41a::bf16r(rb); }
    }
    uint8_t code, e;
    ds41a::act_quant_lane(x, code, e);
    codes[base + c] = code;
    if ((c & 31) == 0) sc[((size_t)row * heads + h) * (HD / 32) + (c >> 5)] = e;
}

// ------------------------------------------------------------------------------------------------ block attention core
// One CTA = one row x 16 heads (the layout of ds41a::k_attn). Row t belongs to session s = t / blk; q = qpos[s] is the last accepted
// position. Key list of every row of the session: WLs = min(q + 1, 128) window keys (list slot i < WLs holds position q - WLs + 1 + i),
// then the block's `blk` keys (list slot WLs + j holds position q + 1 + j). All are read from the session's ring at position %
// ring_rows. No causal mask: row r sees block keys of rows after it too (the drafter is bidirectional inside its block).
// Scores by tensor core (bf16 in, fp32 accumulate), running max / rescale per 64-key block exactly like ds41a::k_attn.
struct BlkP
{
    const bf16* q;           // [rows][heads][512], rope applied
    const bf16* ring;        // layer ring base; session s at ring + s * sess_stride
    size_t sess_stride;
    int ring_rows;
    const int* qpos;         // [sessions] last accepted position
    const int* sess;         // SESS > 1: [rows] S2 slot of each row (its ring slab); null: slab = session s
    const float* sink;       // [heads]
    bf16* out;               // [rows][heads][512]
    int rows, heads, blk;
    float scale;             // 512^-0.5
};

__global__ void __launch_bounds__(ds41a::AT_THREADS, 1) k_attn_blk(BlkP p)
{
    using namespace ds41a;
    extern __shared__ __align__(16) uint8_t smem[];
    bf16* Qs = (bf16*)smem;
    bf16* Ks = Qs + 16 * AT_QLD;
    float* Ss = (float*)(Ks + 64 * AT_KLD);
    bf16* Ps = (bf16*)(Ss + 16 * AT_SLD);
    const bf16** rowp = (const bf16**)(Ps + 16 * AT_PLD);
    float* sc_s = (float*)(rowp + 64);
    float* fin_s = sc_s + 16;

    const int t = blockIdx.x, hg = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int sh = tid >> 4, sl = tid & 15;   // softmax role: head sh, slots sl*4 .. sl*4+3
    const int s = t / p.blk;
    const int q = p.qpos[s];
    const int WLs = min(q + 1, WIN);
    const int L = WLs + p.blk;
    const bf16* ring = p.ring + (size_t)(p.sess ? p.sess[t] : s) * p.sess_stride;
    {
        const uint4* src = (const uint4*)(p.q + ((size_t)t * p.heads + (size_t)hg * 16) * HD);
        for (int c = tid; c < 16 * HD / 8; c += AT_THREADS)
            *(uint4*)(Qs + (c >> 6) * AT_QLD + (c & 63) * 8) = src[c];
    }
    float m_run = -1e30f, sum_run = 0.0f;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    const int nblk = (L + 63) >> 6;
    for (int blk = 0; blk < nblk; blk++)
    {
        if (tid < 64)
        {
            const int i = blk * 64 + tid;
            const bf16* rp = nullptr;
            if (i < L)
            {
                const int pos = i < WLs ? q - WLs + 1 + i : q + 1 + (i - WLs);
                rp = ring + (size_t)(pos % p.ring_rows) * HD;
            }
            rowp[tid] = rp;
        }
        __syncthreads();
        for (int c = tid; c < 64 * 64; c += AT_THREADS)
        {
            const bf16* rp = rowp[c >> 6];
            uint4 v = make_uint4(0, 0, 0, 0);
            if (rp) v = ((const uint4*)rp)[c & 63];
            *(uint4*)(Ks + (c >> 6) * AT_KLD + (c & 63) * 8) = v;
        }
        __syncthreads();
        {   // S[16 heads][8 slots of this warp] = Q . K^T over 512
            float cc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const bf16* krow = Ks + (w * 8 + (lane >> 2)) * AT_KLD + 2 * (lane & 3);
#pragma unroll 4
            for (int ks = 0; ks < HD / 16; ks++)
            {
                uint32_t a[4];
                ldsm_x4(a, Qs + (lane & 15) * AT_QLD + ks * 16 + (lane >> 4) * 8);
                const uint32_t b0 = *(const uint32_t*)(krow + ks * 16);
                const uint32_t b1 = *(const uint32_t*)(krow + ks * 16 + 8);
                mma16816(cc, a, b0, b1);
            }
            const int r0 = lane >> 2, col = w * 8 + 2 * (lane & 3);
            const bool v0 = rowp[col] != nullptr, v1 = rowp[col + 1] != nullptr;
            Ss[r0 * AT_SLD + col] = v0 ? cc[0] * p.scale : -INFINITY;
            Ss[r0 * AT_SLD + col + 1] = v1 ? cc[1] * p.scale : -INFINITY;
            Ss[(r0 + 8) * AT_SLD + col] = v0 ? cc[2] * p.scale : -INFINITY;
            Ss[(r0 + 8) * AT_SLD + col + 1] = v1 ? cc[3] * p.scale : -INFINITY;
        }
        __syncthreads();
        {   // online softmax, head sh, 16 threads x 4 slots
            const float4 s4 = *(const float4*)(Ss + sh * AT_SLD + sl * 4);
            float mb = fmaxf(fmaxf(s4.x, s4.y), fmaxf(s4.z, s4.w));
            for (int o = 8; o; o >>= 1) mb = fmaxf(mb, __shfl_xor_sync(0xFFFFFFFFu, mb, o));
            const float m_new = fmaxf(m_run, mb);
            const float scl = expf(m_run - m_new);
            const float p0 = expf(s4.x - m_new), p1 = expf(s4.y - m_new), p2 = expf(s4.z - m_new), p3 = expf(s4.w - m_new);
            float ps = (p0 + p1) + (p2 + p3);
            for (int o = 8; o; o >>= 1) ps += __shfl_xor_sync(0xFFFFFFFFu, ps, o);
            sum_run = sum_run * scl + ps;
            m_run = m_new;
            bf16* pr = Ps + sh * AT_PLD + sl * 4;
            pr[0] = __float2bfloat16_rn(p0);
            pr[1] = __float2bfloat16_rn(p1);
            pr[2] = __float2bfloat16_rn(p2);
            pr[3] = __float2bfloat16_rn(p3);
            if (sl == 0) sc_s[sh] = scl;
        }
        __syncthreads();
        {   // acc[16 heads][64 dims of this warp] = acc * scale + P . V   (V = the same K rows)
            const float sc0 = sc_s[lane >> 2], sc1 = sc_s[(lane >> 2) + 8];
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                acc[nt][0] *= sc0;
                acc[nt][1] *= sc0;
                acc[nt][2] *= sc1;
                acc[nt][3] *= sc1;
            }
#pragma unroll
            for (int kj = 0; kj < 4; kj++)
            {
                uint32_t a[4];
                ldsm_x4(a, Ps + (lane & 15) * AT_PLD + kj * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int np = 0; np < 4; np++)
                {
                    uint32_t b[4];
                    const int mi = lane >> 3, r = lane & 7;
                    ldsm_x4_t(b, Ks + (kj * 16 + (mi & 1) * 8 + r) * AT_KLD + w * 64 + (np * 2 + (mi >> 1)) * 8);
                    mma16816(acc[np * 2], a, b[0], b[1]);
                    mma16816(acc[np * 2 + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
    if (sl == 0) fin_s[sh] = sum_run + expf(p.sink[hg * 16 + sh] - m_run);
    __syncthreads();
    {
        const int r0 = lane >> 2;
        const float d0 = fin_s[r0], d1 = fin_s[r0 + 8];
        bf16* o0 = p.out + ((size_t)t * p.heads + hg * 16 + r0) * HD;
        bf16* o1 = p.out + ((size_t)t * p.heads + hg * 16 + r0 + 8) * HD;
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            const int d = w * 64 + nt * 8 + 2 * (lane & 3);
            *(__nv_bfloat162*)(o0 + d) = __floats2bfloat162_rn(acc[nt][0] / d0, acc[nt][1] / d0);
            *(__nv_bfloat162*)(o1 + d) = __floats2bfloat162_rn(acc[nt][2] / d1, acc[nt][3] / d1);
        }
    }
}

// ------------------------------------------------------------------------------------------------ host side
inline void launch_attn_blk(const BlkP& p, cudaStream_t st)
{
    static bool init = false;
    if (!init)
    {
        cudaFuncSetAttribute(k_attn_blk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ds41a::AT_SMEM);
        init = true;
    }
    k_attn_blk<<<dim3(p.rows, p.heads / 16), ds41a::AT_THREADS, ds41a::AT_SMEM, st>>>(p);
}

// ---- the three stages of a DSpark layer's attention (ST_ATTN_PROJ / ST_ATTN_CORE / ST_ATTN_OUT), same inputs and outputs as ds41a's.
// rows = sessions * rows_per_sess; pos [rows] = absolute position of every row (q+1+r in the draft block); y = bf16 [rows][5120], the
// sub-layer input already RMS-normed.
//
// attn_proj_blk: S.qa, S.qr, S.q (rope applied at pos), and the block's own keys written into the ring.
void attn_proj_blk(const ds41a::Weights& W, const Ring& R, const ds41a::Scratch& S, const bf16* y, int rows, int rows_per_sess,
                          const int* pos, int heads, cudaStream_t st)
{
    using namespace ds41a;
    launch_act_quant(y, HID, S.yq, HID, S.ys, HID / 32, rows, HID, st);
    dsp_mm(W.wq_a, S.yq, HID, S.ys, HID / 32, S.xf_y, false, S.qa, QA, rows, 1, 0, 0, 0, st);
    k_rmsnorm<<<rows, 256, 0, st>>>(S.qa, QA, W.q_norm, S.qr, QA, QA, EPS);
    launch_act_quant(S.qr, QA, S.qrq, QA, S.qrs, QA / 32, rows, QA, st);
    dsp_mm(W.wq_b, S.qrq, QA, S.qrs, QA / 32, S.xf_qr, false, S.q, heads * HD, rows, 1, 0, 0, 0, st);
    k_rope_tail_pos<<<(unsigned)(((long)rows * heads * 32 + 255) / 256), 256, 0, st>>>(S.q, (long)heads * HD, rows, heads, pos, W.inv, 0);
    dsp_mm(W.wkv, S.yq, HID, S.ys, HID / 32, S.xf_y, ds41a::use_gv(W.wq_a, rows), S.kv0, HD, rows, 1, 0, 0, 0, st);   // wq_a built y's table
    k_kv_prep_pos<<<rows, 512, 0, st>>>(S.kv0, W.kv_norm, W.inv, pos, rows_per_sess, R.base, R.sess_stride, R.ring_rows, R.sess);
}
// attn_core_blk: S.o = block attention of S.q over the session's ring (window + the block's keys). qpos [sessions] = dsp_q.
void attn_core_blk(const ds41a::Weights& W, const Ring& R, const ds41a::Scratch& S, int rows, int rows_per_sess, const int* qpos,
                          int heads, cudaStream_t st)
{
    if (rows_per_sess < 1 || rows_per_sess > BLOCK_MAX || R.ring_rows < WIN + rows_per_sess || rows % rows_per_sess || heads % 16)
    {
        fprintf(stderr, "ds41dsp::attn_core_blk: bad geometry (rows %d, block %d, ring_rows %d, heads %d)\n", rows, rows_per_sess, R.ring_rows, heads);
        abort();
    }
    BlkP a;
    a.q = S.q; a.ring = R.base; a.sess_stride = R.sess_stride; a.ring_rows = R.ring_rows; a.qpos = qpos; a.sess = R.sess; a.sink = W.sink; a.out = S.o;
    a.rows = rows; a.heads = heads; a.blk = rows_per_sess; a.scale = ds41a::SOFTMAX_SCALE;
    launch_attn_blk(a, st);
}
// attn_out_blk: out [rows][5120] bf16 = this rank's o_b partial (summed across ranks at TP > 1) from S.o; inverse rope at pos.
void attn_out_blk(const ds41a::Weights& W, const ds41a::Scratch& S, int rows, const int* pos, int heads, bf16* out, cudaStream_t st)
{
    using namespace ds41a;
    const int G = heads / HPG;
    k_o_prep_pos<<<rows * heads, 512, 0, st>>>(S.o, heads, pos, W.inv, S.oq, S.os);
    dsp_mm(W.wo_a, S.oq, heads * HD, S.os, heads * HD / 32, S.xf_o, false, S.oa, G * OLORA, rows, G, GROUP_IN, GROUP_IN / 32, OLORA, st);
    launch_act_quant(S.oa, G * OLORA, S.oaq, G * OLORA, S.oas, G * OLORA / 32, rows, G * OLORA, st);
    dsp_mm(W.wo_b, S.oaq, G * OLORA, S.oas, G * OLORA / 32, S.xf_oa, false, out, HID, rows, 1, 0, 0, 0, st);
}
}   // namespace ds41dsp
