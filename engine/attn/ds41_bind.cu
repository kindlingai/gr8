// DeepSeek-V4.1-Flash-1017 attention B, the indexer's kernel TU: the C entry points of ds41_idx_api.h over the header-only kernels of
// ds41_idx.cuh (namespace ds41i). The step ops (ds41_idx_ops.cuh) live in the rank TU and reach these through the api header only.
#include "ds41_idx.cuh"
#include "ds41_idx_api.h"
#include "../knobs-rs/gr8_knobs.h"
#include <cstdlib>
#include <cstring>

using namespace ds41i;

// Perf paths of the dense index stage. Both are bit-identical to the reference kernels, so they are the default; the knobs switch
// back for debugging:
//   GR8_DS41_IDX_LOGITS=mma (default, tensor cores, bit-identical to dp4a) | dp4a (reference kernel) | mmafast (tensor cores, own fp32 rounding)
//   GR8_DS41_IDX_TOPK=mc (default, multi-CTA top-k for long rows) | 1cta (single-CTA reference kernel)
static inline int knob_logits()
{
    const std::string& v = knob::ds41_idx_logits();
    if (v == "dp4a") return 0;
    if (v == "mmafast") return 2;
    return 1;
}
static inline int knob_topk() { return knob::ds41_idx_topk() == "1cta" ? 0 : 1; }

static inline cudaStream_t S_(void* s) { return (cudaStream_t)s; }
static inline PagedRows ca_of(void* ca, const int* pages, int rpp) { return ca_map(ca, pages, rpp, (size_t)rpp * CA_ROW); }
static inline IcaMap ica_of(void* ica, const int* pages, int rpp) { return ica_map(ica, pages, rpp, (size_t)rpp * (IK_CODES + IK_SF)); }

extern "C"
{
size_t ds41_idx_scratch_bytes(int Rcap, int max_rows) { return Scratch::bytes(Rcap, max_rows); }
int ds41_idx_pipe_qr_prepare()
{
    int dev = -1, limit = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return (int)e;
    e = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (e != cudaSuccess) return (int)e;
    cudaFuncAttributes attr{};
    e = cudaFuncGetAttributes(&attr, k_logits_pipe_qr);
    if (e != cudaSuccess) return (int)e;
    // Opt-in limit covers both the compiler's static bytes and our explicit
    // dynamic workspace. The static part also stays within 48 KiB.
    if (limit <= 0 || attr.sharedSizeBytes > 48 * 1024 ||
        (size_t)attr.sharedSizeBytes + PIPE_QR_SMEM_BYTES > (size_t)limit)
        return (int)cudaErrorInvalidConfiguration;
    e = cudaFuncSetAttribute(k_logits_pipe_qr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PIPE_QR_SMEM_BYTES);
    return (int)e;
}

void ds41_idx_ca_row_r1(const void* y512, const void* norm_w, const int* dpos, void* ca, const int* pages, int rpp, void* pre, void* shadow,
                        int R, void* stream)
{
    CaR1P p{};
    p.y = (const bf16*)y512; p.norm_w = (const bf16*)norm_w; p.inv = rope_inv_ds41(); p.pos0 = 0;
    p.ca = ca_of(ca, pages, rpp); p.pre = (bf16*)pre; p.roped = nullptr; p.dpos = dpos; p.shadow = (bf16*)shadow;
    launch_ca_row_r1(p, R, S_(stream));
}
void ds41_idx_ica_row(const void* pre, const void* wk, const void* knorm, const int* dpos, int ratio, void* ica, const int* pages, int rpp,
                      int R, void* stream)
{
    IcaP p{};
    p.pre = (const bf16*)pre; p.wk = (const bf16*)wk; p.knorm_w = (const bf16*)knorm; p.inv = rope_inv_ds41();
    p.row0 = 0; p.ratio = ratio; p.ica = ica_of(ica, pages, rpp); p.kdbg = nullptr; p.dpos = dpos;
    launch_ica_row(p, R, S_(stream));
}
void ds41_idx_gemv(const void* X, int ldx, const void* W, int N, int K, void* Y, int ldy, int R, void* stream)
{
    launch_gemv_bf16((const bf16*)X, ldx, (const bf16*)W, N, K, (bf16*)Y, ldy, R, S_(stream));
}
void ds41_idx_gemm_mx(const void* A, int lda, const void* SA, int lsa, const void* W, const void* SW, int N, int K, void* Y, int ldy, int M,
                      void* stream)
{
    GemmMxP p{};
    p.A = (const uint8_t*)A; p.SA = (const uint8_t*)SA; p.W = (const uint8_t*)W; p.SW = (const uint8_t*)SW; p.Y = (bf16*)Y;
    p.M = M; p.N = N; p.K = K; p.lda = lda; p.lsa = lsa; p.ldy = ldy;
    launch_gemm_mx(p, S_(stream));
}
// The attention block's tensor-core block-scaled MXFP8 GEMM, defined in the attention TU (ds41_attn.cu). 0 = launched, 1 = shape refused,
// 2 = pointer misaligned (nothing launched). Weak: a binary without it still links and the wrapper reports -1.
extern "C" int ds41_gemm_mx_tc(const uint8_t* A, int lda, const uint8_t* SA, int lsa, const uint8_t* W, const uint8_t* SW, void* Y, int ldy, int M, int N, int K,
                               void* st) __attribute__((weak));
int ds41_idx_gemm_mx_tc(const void* A, int lda, const void* SA, int lsa, const void* W, const void* SW, int N, int K, void* Y, int ldy, int M,
                        void* stream)
{
    if (!ds41_gemm_mx_tc) return -1;
    return ds41_gemm_mx_tc((const uint8_t*)A, lda, (const uint8_t*)SA, lsa, (const uint8_t*)W, (const uint8_t*)SW, Y, ldy, M, N, K, stream);
}
void ds41_idx_klen(const int* dpos, int ratio, int* klen, int R, void* stream) { launch_klen(dpos, ratio, klen, R, S_(stream)); }

static inline IndexIn in_of(const Ds41IdxStage* s)
{
    IndexIn in{};
    in.q_b = (const bf16*)s->q_b; in.wproj = (const bf16*)s->wproj; in.inv = rope_inv_ds41(); in.pos0 = 0; in.R = s->R;
    in.klen = s->klen; in.max_klen = s->max_klen; in.k = ica_of((void*)s->ica, s->pages, s->rpp); in.dpos = s->dpos;
    static const int mma = knob_logits(), mc = knob_topk();
    in.logits_mma = mma;
    in.pipe_b = s->pipe_b != 0;
    in.q_trig = s->q_trig != 0;
    in.pipe_qr = s->pipe_qr != 0;
    in.topk_mc = mc;
    return in;
}
void ds41_idx_topk(const Ds41IdxStage* s, void* stream)
{
    const Scratch sc = Scratch::carve(s->scratch, s->Rcap, s->max_rows);
    const IndexIn in = in_of(s);
    if (s->use_cand) stage_index_cand(in, sc, sc.cand, s->out, s->ldo, S_(stream));
    else stage_index_dense(in, sc, s->out, s->ldo, S_(stream));
}
void ds41_idx_cand_blocks(const Ds41IdxStage* s, void* stream)
{
    const Scratch sc = Scratch::carve(s->scratch, s->Rcap, s->max_rows);
    stage_cand_blocks(in_of(s), sc, S_(stream));
}
// the candidate block list [Rcap][CAND_BLOCKS] int32 inside a scratch carved for (Rcap, max_rows): where ds41_idx_cand_blocks leaves it
void* ds41_idx_cand_list(void* scratch, int Rcap, int max_rows)
{
    return Scratch::carve(scratch, Rcap, max_rows).cand;
}
}
