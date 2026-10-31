// DSpark device API of DeepSeek-V4.1-Flash-1017 (B12): plain types + declarations, NO kernels. The step-driver TU (rank.cuh -> ds41_dspark_ops.cuh)
// includes only this header; the kernels and the definitions live in ds41_dspark.cuh / ds41_attn_dsp.cuh, compiled ONCE inside the attention TU
// (engine/attn/ds41_attn_dspark.cu = ds41_attn.cu + the DSpark code), because they reuse B4's ds41_attn.cuh kernels and a second copy would be a
// duplicate-symbol link error (R3, whole-engine MODEL=ds41 build). Never include ds41_attn.cuh / ds41_dspark.cuh from rank.cuh or the ops header.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>
#include "../attn/ds41_attn_types.h"
#include "ds41_dspark_dev.h"

namespace ds41dsp
{
using ds41a::bf16;
constexpr int MARKOV_MAXS = 8;                    // sessions per markov step (smem tile of w1 rows)

struct Ring
{
    bf16* base = nullptr;        // session s of this layer: base + s * sess_stride
    size_t sess_stride = 0;      // elements between sessions (ring_rows * 512 for a packed layer ring)
    int ring_rows = 136;
    const int* sess = nullptr;   // SESS > 1: device [rows] S2 slot of each row of the step (Model::sess); null: session = row / rows_per_sess
};

struct MainW
{
    ds41a::Fp8Mat main_proj;                      // mtp.0.main_proj fp8 [5120][15360]
    const bf16* main_norm = nullptr;              // mtp.0.main_norm [5120]
    ds41a::Fp8Mat wkv[3];                         // mtp.j.attn.wkv, j = layer 40 + j
    const bf16* kv_norm[3] = {};                  // mtp.j.attn.kv_norm [512]
    const float* inv = nullptr;                   // [32] plain rope theta 10000 (no YaRN): the table of layers 0 and 1
    Ring ring[3];
};

struct MainS                                      // scratch for M rows
{
    uint8_t *tq, *ts, *hq, *hs;                   // act-quant codes / scales of the tap [M][15360] and of main_x [M][5120]
    bf16 *mh0, *mh, *mk;                          // main_proj out, main_x (normed), wkv out [M][512]
    uint32_t *xf_tap, *xf_h;                      // GR8_DS_DSP_GV: the GEMV's fp16 fragment tables (8 tokens) of the tap [15360] and of main_x [5120], K * 16 B each
    static size_t al(size_t b) { return (b + 255) & ~(size_t)255; }
    static size_t bytes(int M)
    {
        return al((size_t)M * 15360) + al((size_t)M * 480) + al((size_t)M * 5120) + al((size_t)M * 160) + 2 * al((size_t)M * 5120 * 2) + al((size_t)M * 512 * 2) +
               al((size_t)15360 * 16) + al((size_t)5120 * 16);
    }
    void carve(void* base, int M)
    {
        uint8_t* p = (uint8_t*)base;
        auto take = [&](size_t b) { uint8_t* r = p; p += al(b); return r; };
        tq = take((size_t)M * 15360);
        ts = take((size_t)M * 480);
        hq = take((size_t)M * 5120);
        hs = take((size_t)M * 160);
        mh0 = (bf16*)take((size_t)M * 5120 * 2);
        mh = (bf16*)take((size_t)M * 5120 * 2);
        mk = (bf16*)take((size_t)M * 512 * 2);
        xf_tap = (uint32_t*)take((size_t)15360 * 16);
        xf_h = (uint32_t*)take((size_t)5120 * 16);
    }
};

struct Dsp
{
    int nsess = 0;                                // sessions the arrays below are sized for (the S2 slots; 1 = one session). The sessions OF A STEP are
                                                  // Model::ds_ns of the Model the op runs on (set before the launch or capture; 1 with one session)
    int k = 5;                                    // drafts per round; the verify step has k + 1 rows per session
    bool draft_model = false;                     // true in the DRAFT Model's copy (ds41dsp_ops::build): its ST_FINISH writes xpre and stores nothing
    bool host_accept = false;                     // draft Model: q / golden / acc are written by the host (s2/ds41_dspark_round.cuh) before the step; ST_DS_DRAFT_IN skips the device accept
    // main Model: ST_DS_TAP, ST_DS_MAIN
    bf16* tap = nullptr;                          // [rows][3][5120] BF16, written at the entry of main layers 37 / 38 / 39
    // PREFILL (card section 9; ds41_dspark_prefill.h): a non-final chunk of W rows > MX. tap_rows = capacity of `tap` in rows (0 = MX, the decode
    // step); main_lo / main_hi = the chunk rows [lo, hi) that go through ST_DS_TAP / ST_DS_MAIN (hi = -1: all rows; the default). The prefill
    // driver sets them per chunk (pf::main_rows) before the plan runs. Host state: a captured graph bakes them in (eager or per-shape graphs only).
    int tap_rows = 0;
    int ms_rows = 0;                              // rows MainS is carved for (0 = MX; a prefill-capable build: PREFILL_MS_ROWS = 128). ds_main_rows launches <= min(ms_rows, ring_rows)
    int main_lo = 0, main_hi = -1;
    MainW mw;                                     // main_proj, main_norm, wkv x3, kv_norm x3, plain rope table, the 3 key rings (mw.ring[j])
    MainS ms;                                     // scratch of ds_main, carved for the largest verify step
    // the verify step's own arrays (the main Model's m.tok, m.tok_out, m.pos), read by ST_DS_DRAFT_IN
    const int *vtok = nullptr, *vtgt = nullptr, *vpos = nullptr;
    // accept result, device-resident: [nsess]
    int *q = nullptr, *golden = nullptr, *acc = nullptr;
    // SESS > 1, the compressor commit of a draft step (ds41_attn_comp_commit_sessions): session j's first row in the main chunk and its S2 slot [nsess]
    int *crow0 = nullptr, *cslot = nullptr;
    // draft Model: ST_DS_DRAFT_IN .. ST_DS_MARKOV
    bf16* x = nullptr;                            // [nsess * 5][5120] pre-norm head input of the last draft layer (ds_mhc::finish xpre)
    const bf16* membed = nullptr;                 // markov_head.embed [VOCAB][256] (replicated)
    const bf16* mhead = nullptr;                  // markov_head.head, THIS rank's vocab shard [vn][256]
    const bf16* conf_proj = nullptr;              // confidence_head.proj [5376]
    int* dtok = nullptr;                          // [5][nsess] merge output, i-major
    int* draft = nullptr;                         // [nsess][5] the drafts, session-major (the next verify step's rows 1..k)
    float* conf = nullptr;                        // [nsess][5] raw FP32 confidence score
};

// ---- definitions: ds41_attn_dsp.cuh (attn_*_blk) and ds41_dspark.cuh (the rest), in the attention TU
bool dsp_gv_enabled();                                                       // GR8_DS_DSP_GV=1 (the rank TU cannot see ds41_attn_dsp.cuh's inline helpers)
bool dsp_gv_ok(const ds41a::Fp8Mat& W, int rows);                            // enabled and the GEMV rule holds for W at `rows` rows
void attn_proj_blk(const ds41a::Weights& W, const Ring& R, const ds41a::Scratch& S, const bf16* y, int rows, int rows_per_sess, const int* pos, int heads, cudaStream_t st);
void attn_core_blk(const ds41a::Weights& W, const Ring& R, const ds41a::Scratch& S, int rows, int rows_per_sess, const int* qpos, int heads, cudaStream_t st);
void attn_out_blk(const ds41a::Weights& W, const ds41a::Scratch& S, int rows, const int* pos, int heads, bf16* out, cudaStream_t st);
void ds_tap(const float* streams, __nv_bfloat16* tap, int slot, int rows, cudaStream_t st);
void ds_tap(const __nv_bfloat16* streams, __nv_bfloat16* tap, int slot, int rows, cudaStream_t st);   // BF16 trunk (-DDS_MHC_TRUNK_BF16)
void ds_main(const MainW& W, const MainS& S, const bf16* tap, int M, int rows_per_sess, const int* pos, cudaStream_t st, bool wide = false,
             const int* dsess = nullptr);   // dsess: SESS > 1, the S2 slot of each row (its ring slab); null = row / rows_per_sess
void ds_main_rows(const MainW& W, const MainS& S, const bf16* tap, int rows, const int* pos, int cap, cudaStream_t st, const int* dsess = nullptr);   // prefill: <= cap rows per launch, ascending
void ds_draft_in(const int* vtok, const int* vtgt, const int* vpos, int rows_per_sess, int nsess, int* q, int* golden, int* acc, int* tok, int* pos, cudaStream_t st);
void markov_add(const bf16* membed, const bf16* mhead, const int* golden, const int* dtok, int i, int nsess, float* hlog, long ld_row, int vn, cudaStream_t st);
void conf_head(const bf16* x, const bf16* membed, const bf16* proj, const int* golden, const int* dtok, int nsess, float* conf, int* draft, cudaStream_t st);
void ds_verify_in(const int* golden, const int* draft, const int* q, int k, int nsess, int* tok, int* pos, cudaStream_t st);
}   // namespace ds41dsp
