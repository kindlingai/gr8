// DeepSeek-V4.1-Flash-1017 attention B (the indexer): the kernel-free C interface between the step driver's ops (ds41_idx_ops.cuh, compiled into the
// rank TU) and the indexer kernels (ds41_idx.cuh, compiled once into ds41_bind.cu). Plain POD arguments and void* device pointers, so
// the rank TU never sees the kernels (it would define them a second time). Rows are positions dpos[r] read on the device (graph-safe;
// dpos = Model::pos). Row storage is flat: one "page" of rpp rows, page table = a device int32 holding 0 (there is no paged pool yet).
#pragma once
#include <cstddef>
#include <cstdint>

extern "C"
{
// bytes of the shared stage scratch (logits, block scores, candidate lists, ...) for Rcap query rows over max_rows compressed rows
size_t ds41_idx_scratch_bytes(int Rcap, int max_rows);
// Called once at indexer bind when PIPE_QR=1. Checks the actual device's opt-in
// shared-memory ceiling and configures the QR kernel before any on launch.
int ds41_idx_pipe_qr_prepare();

// layer 20 ca row: y512 = bf16 [R][512] (wkv(x)), norm_w bf16 [512]; writes the 288 B fp4 row of every token to ca (flat [rpp][288]),
// the post-norm PRE-rope latent to pre [R][512] (the ica row's input) and the bf16 value of the stored row to shadow [rpp][512] (row = position).
void ds41_idx_ca_row_r1(const void* y512, const void* norm_w, const int* dpos, void* ca, const int* pages, int rpp, void* pre, void* shadow,
                        int R, void* stream);
// ica (index key) row: pre = post-norm pre-rope latent rows (ratio 1: [R][512], slot = token; ratio 2: the compressor's cnorm, slot k = k-th
// completed pair of the call), wk bf16 [128][512], knorm bf16 [128]. ica = flat [rpp][68] (64 B codes plane, then 4 B scales plane).
void ds41_idx_ica_row(const void* pre, const void* wk, const void* knorm, const int* dpos, int ratio, void* ica, const int* pages, int rpp,
                      int R, void* stream);
// Y[r][n] = bf16(sum_k X[r][k] W[n][k]); W bf16 [N][K], K % 256 == 0
void ds41_idx_gemv(const void* X, int ldx, const void* W, int N, int K, void* Y, int ldy, int R, void* stream);
// block-scaled fp8 gemm (attention A's arithmetic): A codes [M][lda] + E8M0 bytes [M][lsa], W codes [N][K] + E8M0 bytes [N][K/32]
void ds41_idx_gemm_mx(const void* A, int lda, const void* SA, int lsa, const void* W, const void* SW, int N, int K, void* Y, int ldy, int M,
                      void* stream);
// The same product on attention A's tensor-core GEMM (ds41_gemm_mx_tc in ds41_attn.cu, resolved at link time). Not bit-exact with the scalar gemm above
// (rel-L2 about 1e-4). Returns 0 when it ran; nonzero when it did not (-1: the entry is not linked in; else the entry refused the shape) and nothing was
// launched, so the caller falls back to ds41_idx_gemm_mx.
int ds41_idx_gemm_mx_tc(const void* A, int lda, const void* SA, int lsa, const void* W, const void* SW, int N, int K, void* Y, int ldy, int M,
                        void* stream);
// klen[r] = (dpos[r] + 1) / ratio
void ds41_idx_klen(const int* dpos, int ratio, int* klen, int R, void* stream);

struct Ds41IdxStage
{
    const void* q_b;       // bf16 [R][4096]
    const void* wproj;     // bf16 [R][32]
    const int* dpos;       // device [R]
    const int* klen;       // device [R]
    int max_klen;          // host bound of klen (grid size)
    int R;
    const void* ica;       // index-K rows (flat, rpp rows) of the kv-source layer
    const int* pages;
    int rpp;
    void* scratch;         // ds41_idx_scratch_bytes(Rcap, max_rows) bytes
    int Rcap, max_rows;
    int use_cand;          // 1: layers 24/28/32/36 (logits over the layer-20 candidate blocks), 0: dense (layers 2, 8, 14, 20)
    int pipe_b;            // boot-bound arm B for the four dense index sources only; 0 is the plain dispatch
    int q_trig;            // default-off once-per-row fp64 query rope trigonometry
    int pipe_qr;           // default-off Q-stationary 16-row cp.async dense logits
    int* out;              // int32 [R][ldo] top-512, ascending, -1 tail
    int ldo;
};
void ds41_idx_topk(const Ds41IdxStage* s, void* stream);
// layer 20 only, after its dense ds41_idx_topk: block scores of the logits it left in the scratch, top-2048 blocks -> the scratch's candidates
void ds41_idx_cand_blocks(const Ds41IdxStage* s, void* stream);
// the candidate block list ([Rcap][2048] int32, one row per query row) inside a scratch carved for (Rcap, max_rows); bounded replay moves its rows
void* ds41_idx_cand_list(void* scratch, int Rcap, int max_rows);
}
