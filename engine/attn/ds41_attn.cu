// DeepSeek-V4.1-Flash-1017 attention TU: the kernels and stage functions live in ds41_attn.cuh (header-only, namespace ds41a;
// ds41a::attn_proj / comp_row / attn_core / attn_out are the four plan stages, layer_forward chains them). This file holds the
// one symbol the decode profiler hook needs (attn.cu:22 defines it for the other models) and makes the stage functions
// link-visible through plain C entry points a step glue can bind without including the header.
#include "ds41_attn.cuh"
#include "ds41_sh_tc.cuh"   // the shared expert's tensor-core adapters (C entries ds41_sh_* below)
#include "attn.h"

extern "C" { void (*g6_attn_mark)(int stage, void* stream) = nullptr; }

extern "C" {
size_t ds41_attn_scratch_bytes(int rows, int heads) { return ds41a::Scratch::bytes(rows, heads); }
// dstart: device word with the first position (the engine's m.pos), or null to use the host start. All launches are async on st.
void ds41_attn_proj(const ds41a::Weights* W, ds41a::Cache* C, const ds41a::Scratch* S, const void* y, int n, int start, int heads, void* st,
                    const int* dstart)
{
    ds41a::attn_proj(*W, *C, *S, (const ds41a::bf16*)y, n, start, heads, (cudaStream_t)st, dstart);
}
void ds41_attn_proj_q(const ds41a::Weights* W, ds41a::Cache* C, const ds41a::Scratch* S, const void* y,
                      const uint8_t* codes, const uint8_t* scales, int n, int start, int heads, void* st, const int* dstart)
{
    ds41a::attn_proj(*W, *C, *S, (const ds41a::bf16*)y, n, start, heads, (cudaStream_t)st, dstart, codes, scales);
}
void ds41_attn_comp_row(const ds41a::Weights* W, ds41a::Cache* C, const ds41a::Scratch* S, const void* y, int n, int start, void* st,
                        const int* dstart)
{
    ds41a::comp_row(*W, *C, *S, (const ds41a::bf16*)y, n, start, (cudaStream_t)st, dstart);
}
// DSpark verify chunk, after the accept: re-park the compressor's pending half (kvs / scs slot 0) of this source layer for the accepted last
// position *dq (a = *dacc accepted drafts); a no-op when *dq is odd. Async on st; needs the chunk's S.ckv / S.csc intact.
void ds41_attn_comp_commit(const ds41a::Cache* C, const ds41a::Scratch* S, const int* dq, const int* dacc, void* st)
{
    ds41a::comp_commit(*C, *S, dq, dacc, (cudaStream_t)st);
}
// SESS > 1: ns sessions of a draft step; session j: dq[j], dacc[j], its verify rows from row drow0[j], its slot dslot[j] (device words)
void ds41_attn_comp_commit_ns(const ds41a::Cache* C, const ds41a::Scratch* S, const int* dq, const int* dacc, const int* drow0, const int* dslot, int ns,
                              void* st)
{
    ds41a::comp_commit(*C, *S, dq, dacc, (cudaStream_t)st, drow0, dslot, ns);
}
void ds41_attn_core(const ds41a::Weights* W, const ds41a::Cache* C, const void* comp_src, const ds41a::Scratch* S, int n, int start, int heads,
                    const int* topk, int ldtk, int TK, void* st, const int* dstart)
{
    ds41a::attn_core(*W, *C, (const ds41a::bf16*)comp_src, *S, n, start, heads, topk, ldtk, TK, (cudaStream_t)st, dstart);
}
void ds41_attn_out(const ds41a::Weights* W, const ds41a::Scratch* S, int n, int start, int heads, void* out, void* st, const int* dstart, int direct_c2)
{
    ds41a::attn_out(*W, *S, n, start, heads, (ds41a::bf16*)out, (cudaStream_t)st, dstart, direct_c2 != 0);
}
// Dense MXFP8 GEMV with the official act-quant: y[rows][ldy] = fp8gemm(act_quant(x), W), W->gv set. scratch: codes u8 [8][K], scales u8 [8][K/32],
// xf u32 [4K]. 0 = ok. (The Engram wkv, ds41_engram_project in ds41_attn_ops.cuh.)
int ds41_attn_project(const ds41a::Fp8Mat* W, const void* x, int rows, void* y, int ldy, uint8_t* codes, uint8_t* scales, uint32_t* xf, void* st)
{
    return ds41a::project_rows(*W, (const ds41a::bf16*)x, rows, (ds41a::bf16*)y, ldy, codes, scales, xf, (cudaStream_t)st);
}
// Decode GEMV switch (a process-wide setting): on = projections at <= 8 rows use the MXFP8 GEMV where the weight has records; tpc / ks = 0 -> defaults.
// fn_xfrag / fn_gemv: the launchers of ds41_attn_gv.cuh from the rank TU (this TU does not hold the GEMV kernels); also needed by ds41_attn_project.
void ds41_attn_set_gv(int on, int tpc, int ks, ds41a::XfragFn fn_xfrag, ds41a::GemvFn fn_gemv)
{
    ds41a::g_fn_xfrag = fn_xfrag;
    ds41a::g_fn_gemv = fn_gemv;
    ds41a::g_use_gv = on != 0 && fn_xfrag && fn_gemv;
    if (tpc > 0) ds41a::g_gv_tpc = tpc;
    ds41a::g_gv_ks = ks > 0 ? ks : 0;
}
// GR8_DS_SRC_FAST bits (a process-wide setting): 1 = the ratio-2 compressor projections at <= 8 rows use k_f32gemv2 (bit-identical to k_f32gemm2); 2 is read by the rank TU
// (idx wq_b through the MXFP8 GEMV). 0 = the generic launches.
void ds41_attn_set_src_fast(int bits)
{
    ds41a::g_src_fast = bits;
}
// Tensor-core GEMM for M >= 9 (k_gemm_mx8t, ds41_attn.cuh): a process-wide setting, default off.
void ds41_attn_set_tc(int on) { ds41a::g_use_tc = on != 0; }
void ds41_attn_set_comp_fp8(int on) { ds41a::g_comp_fp8 = on != 0; }
void ds41_attn_comp_prepack(const ds41a::bf16* src, uint8_t* codes, uint8_t* scales, void* st)
{
    ds41a::launch_act_quant(src, ds41a::HID, codes, ds41a::HID, scales, ds41a::HID / 32, ds41a::HD, ds41a::HID, (cudaStream_t)st);
}

// Launch order of that GEMM (default 0 = N tile fastest): 1 = M tile fastest for every launch; 2 = auto, M tile fastest only for a weight of >= 100 MB (ds41a::TC_MF_AUTO_BYTES).
// Only matters when the weights are much bigger than L2 and M spans several tiles (Engram wkv). Any other nonzero value means 1.
void ds41_attn_set_tc_mfast(int mode) { ds41a::g_tc_mfast = mode == 2 ? 2 : mode != 0 ? 1 : 0; }
// Explicit tensor-core MXFP8 GEMM for other TUs (the indexer's wq_b), one group, independent of ds41_attn_set_tc:
//   Y[M][N] (bf16, row stride ldy elements) = fp8gemm(A codes [M][K] (row stride lda bytes) + E8M0 scales SA (row stride lsa bytes),
//                                                      W codes [N][K] + E8M0 scales SW [N][K/32]), all device pointers, async on st.
// 0 = launched. 1 = shape not supported (needs K % 128, N % 64, lda % 16, lsa % 4, M >= 1, ldy >= N and even), 2 = a pointer is misaligned
// (A, W 16 B; SA, SW, Y 4 B). On nonzero nothing is launched: the caller runs its scalar GEMM.
int ds41_gemm_mx_tc(const uint8_t* A, int lda, const uint8_t* SA, int lsa, const uint8_t* W, const uint8_t* SW, void* Y, int ldy, int M, int N, int K, void* st)
{
    ds41a::GemmP p;
    p.A = A; p.SA = SA; p.W = W; p.SW = SW; p.Y = (ds41a::bf16*)Y;
    p.M = M; p.N = N; p.K = K;
    p.lda = lda; p.lsa = lsa; p.ldy = ldy;
    p.a_gs = p.sa_gs = p.w_gs = p.sw_gs = p.y_gs = 0;
    if (K <= 0 || N <= 0 || ldy < N || (ldy & 1) || !ds41a::tc_ok(p)) return 1;
    if (((uintptr_t)A | (uintptr_t)W) & 15 || ((uintptr_t)SA | (uintptr_t)SW | (uintptr_t)Y) & 3) return 2;
    ds41a::launch_gemm_tc(p, 1, (cudaStream_t)st);
    return 0;
}
// GR8_DS_ATTN_CPA (a process-wide setting): 1 = the attention core gathers its K rows with cp.async (k_attn_cpa, bit-identical output). 0 = k_attn.
void ds41_attn_set_cpasync(int on)
{
    ds41a::g_attn_cpa = on != 0;
}
void ds41_attn_set_split(int on)
{
    ds41a::g_attn_split = on != 0;
}
// Bounded replay (runtime/ds41_pf_split.h): window positions below `pos` are masked in every attention core launched after this call (decode
// form, the engine's). The prefill hook sets it to the chunk tail's first position around the layers >= 21 and back to 0 (none) after them;
// eager launches only (a captured graph keeps the value it was captured with: 0).
void ds41_attn_set_win_floor(int pos)
{
    ds41a::g_win_floor = pos > 0 ? pos : 0;
}
// 1 = this layer's wq_b would take the GEMV at `rows` rows (the same switch attn_proj uses), so the idx wq_b can reuse its q_b activation table (S.xf_qr)
int ds41_attn_gv_ready(const ds41a::Weights* W, int rows)
{
    return ds41a::use_gv(W->wq_b, rows) ? 1 : 0;
}
// The shared expert on tensor cores (ds41_sh_tc.cuh). The rank TU's shared expert (kernels/ds41_moe_ops.cuh, GR8_DS_SH_TC=1)
// reaches these and ds41_gemm_mx_tc above; nothing here runs unless that flag is 1. All async on st.
// x bf16 [M][K] (row stride ldx) -> E4M3 codes [M][K] (stride ldc) + E8M0 bytes [M][K/32] (stride lds): k_act_quant, the activation quantizer of the
// attention TC path, unchanged (launch_act_quant). 0 = launched, 1 = K % 32 or M < 1 (nothing launched).
int ds41_sh_act_quant(const void* x, int ldx, uint8_t* codes, int ldc, uint8_t* scales, int lds, int M, int K, void* st)
{
    if (M < 1 || K <= 0 || K % 32) return 1;
    ds41a::launch_act_quant((const ds41a::bf16*)x, ldx, codes, ldc, scales, lds, M, K, (cudaStream_t)st);
    return 0;
}
// A1 repack: MXFP8 records [T][K/32][528] -> plain codes [16T][ldc] + E8M0 bytes [16T][lds] (k_mx_unrecord, the inverse of k_kda_mx_record).
// Columns >= K of the target are not written (the caller zeroes them for a pad). 0 = launched, 1 = bad shape.
int ds41_sh_unrecord(const uint8_t* rec, int T, int K, uint8_t* codes, int ldc, uint8_t* scales, int lds, void* st)
{
    if (T < 1 || K <= 0 || K % 32 || ldc < K || lds < K / 32) return 1;
    ds41a::k_mx_unrecord<<<(unsigned)((size_t)T * (K / 32)), 256, 0, (cudaStream_t)st>>>(rec, K / 32, codes, ldc, scales, lds);
    return 0;
}
// A3: Y13 bf16 [M][ldy] (gate cols 0..I-1, up cols I..2I-1) -> act codes [M][KP] (stride ldc) + E8M0 bytes [M][KP/32] (stride lds),
// act = swiglu_lim(gate, up, lim) quantized per 32 like k_act_quant; blocks from I up to KP are code 0, byte 0. 0 = launched, 1 = bad shape.
int ds41_sh_swiglu_q(const void* y13, int ldy, int I, float lim, uint8_t* codes, int ldc, uint8_t* scales, int lds, int M, int KP, void* st)
{
    if (M < 1 || I <= 0 || I % 32 || KP < I || KP % 32 || ldy < 2 * I || ldc < KP || lds < KP / 32) return 1;
    const long warps = (long)M * (KP / 32);
    ds41a::k_sh_swiglu_q<<<(unsigned)((warps + 7) / 8), 256, 0, (cudaStream_t)st>>>((const ds41a::bf16*)y13, ldy, I, lim, codes, ldc, scales, lds, M, KP);
    return 0;
}
// this layer's rope table: compressed = 0 -> theta 10000 plain (layers 0, 1); 1 -> theta 160000 + YaRN (orig 65536, factor 16, beta 32 / 1)
void ds41_attn_inv_freqs(int compressed, float* out32)
{
    if (compressed) ds41a::inv_freqs_host(160000.0, 65536, out32);
    else ds41a::inv_freqs_host(10000.0, 0, out32);
}
}
// Stop print: non-inline accessors so the engine TU can link to the c2 dispatch counter. ds41_attn.cuh is not
// includable there (its device kernels would be defined twice in one binary), so these two are the link-visible names.
unsigned long long ds41_c2_dispatch_count() { return ds41a::c2_dispatch_count(); }
bool ds41_c2_trace_on() { return ds41a::c2_trace_on(); }
