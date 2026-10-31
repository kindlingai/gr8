// The KV cache row layouts, the one definition every kernel, copy and store shares.
//
// MLA latent row (one token, one layer), selected at compile time by G6_KV4 (default 0):
//   G6_KV4=0  656 B: [512 E4M3 codes][4 FP32 scales, one per 128 dims][64 BF16 rope]
//   G6_KV4=1  416 B: [256 B INT4 codes][16 FP16 scales, one per 32 dims][64 BF16 rope]
//             The latent is rotated by H, the 512-point Sylvester Hadamard scaled by 1/sqrt(512) (H = H^T = H^-1), before
//             it is quantized. Byte i holds dims 2i (low nibble) and 2i+1 (high nibble) as code q + 8, q in -7..7; the
//             value is q * scale. Decode rotates the absorbed query by H and the attention context back by H; prefill
//             rotates each key back when it builds its FP16 key table. See docs/KV4.md.
// Index key row (22 full-index caches), unchanged by G6_KV4: per 64-row page, 64 x 128 E4M3 codes, then 64 FP32 scales.
#pragma once
#include "../model_geom.h"
#ifndef G6_KV4
#define G6_KV4 0
#endif

namespace g6kv
{
#if G6_MODEL_DS41
// DeepSeek-V4.1-Flash-1017 (docs/DEEPSEEK.md, docs/DEEPSEEK-ATTN-B.md; the rows are the release's own, engine/attn/ds41_idx.cuh):
//   ca row (compressed KV, 512 wide, rope tail included) 288 B = [256 B FP4 e2m1 codes, element 2i in the LOW nibble][32 B E4M3 scales, one per 16 dims]
//   ica row (index key, 128 wide)                         68 B = [64 B FP4 codes][4 B E8M0 scales, one per 32 dims], stored PLANAR per page
//                                                              (all the page's codes, then all its scale words) so both planes stay aligned
// Only the four KV-source layers 2, 8, 14, 20 own rows (ratio 2: 32 rows per 64-token page; ratio 1: 64 rows). Per 64-token page:
// L2/8/14 each 32 x 288 = 9,216 B ca + 32 x 68 = 2,176 B ica; L20 64 x 288 = 18,432 B + 64 x 68 = 4,352 B; 56,960 B in all = 890 B/token.
// Every layer also keeps a WINDOW = 128 token ring of its own wkv rows (fp8-valued rows), outside this pool.
constexpr int ROW = 288, IDX_ROW = 68, PAGE = 64;   // bytes of one ca row / one ica row; the kvman page is 64 tokens
constexpr int CA_CODES = 256, CA_SF_OFF = 256, ICA_CODES = 64, ICA_SF = 4;
constexpr int CA_PAGE_R2 = 32 * ROW, ICA_PAGE_R2 = 32 * IDX_ROW, CA_PAGE_R1 = 64 * ROW, ICA_PAGE_R1 = 64 * IDX_ROW;   // 9216, 2176, 18432, 4352
constexpr const char* NAME = "ds41-fp4";
#elif G6_MODEL_FLASH
// GLM 5.3 Flash (attn/flash_attn.cu): NoPE MLA, the 512-d latent only (kv_a_layernorm applied), selected by
// G6_FLASH_KV_FP8 (default 0):
//   0  1024 B: [512 BF16]
//   1   528 B: [512 E4M3 codes][4 FP32 scales, one per 128 dims, amax / 448]
// Index cache (kpool indexer, every DSA layer), per 64-row page (16 pools of KPOOL = 4 consecutive tokens):
//   [64 tokens][IDX_DIM FP16 raw key = wk(x), before k_norm][IDX_DIM FP16 raw pool gate = x @ compress_gate^T]  32 KB
//   [16 pools][IDX_DIM FP32 pooled key]                                                                         8 KB
// A row writes only its own raw slot; the row that completes a pool (pos % 4 == 3) then forms the pool's key from the
// 4 raw slots: k_j = LayerNorm(raw_k_j) * w + b (eps 1e-6), p_j = softmax_j(gate_j + ape_j) per channel,
// pooled = sum_j p_j k_j. A re-appended row (speculation rollback) rewrites its slot and, at pos % 4 == 3, the pool.
#ifndef G6_FLASH_KV_FP8
#define G6_FLASH_KV_FP8 0
#endif
#if G6_FLASH_KV_FP8
constexpr int ROW = 528, SCALE_OFF = 512;
constexpr const char* NAME = "fp8nope";
#else
constexpr int ROW = 1024, SCALE_OFF = 0;
constexpr const char* NAME = "bf16nope";
#endif
constexpr int IDX_HEADS = 32, IDX_DIM = 128, KPOOL = 4, IDX_TOPK = 2048, TOP_POOLS = IDX_TOPK / KPOOL;
constexpr int PAGE = 64, PAGE_POOLS = PAGE / KPOOL;
constexpr int IDX_RAW = 2 * IDX_DIM * 2;                    // bytes of one token's raw key + gate
constexpr int IDX_POOL_OFF = PAGE * IDX_RAW;                // pooled keys after the 64 raw slots
constexpr int IDX_PAGE = IDX_POOL_OFF + PAGE_POOLS * IDX_DIM * 4;   // 40960
constexpr int IDX_ROW = IDX_PAGE / PAGE;                    // 640 per token (sizing)
constexpr int SEL = 2304;   // selected positions per row: TOP_POOLS * KPOOL + the tail (<= 3), padded to 9 x 256
#elif G6_KV4
constexpr int ROW = 288 + 2 * geom::QK_ROPE, SCALE_OFF = 256, ROPE_OFF = 288;
constexpr const char* NAME = "int4g32h";
#else
constexpr int ROW = 528 + 2 * geom::QK_ROPE, SCALE_OFF = 512, ROPE_OFF = 528;
constexpr const char* NAME = "fp8";
#endif
#if !G6_MODEL_FLASH && !G6_MODEL_DS41
constexpr int IDX_ROW = 132;
#endif
// both latent rows by name, whatever the build's row is, for code that converts one to the other (the remote rows
// tier's GR8_RT_FMT=int4 records, runtime/rt_rows.cuh)
namespace fp8row { constexpr int ROW = 656, SCALE_OFF = 512, ROPE_OFF = 528; }
namespace int4row { constexpr int ROW = 416, SCALE_OFF = 256, ROPE_OFF = 288; }
static_assert(ROW % 16 == 0, "rows are gathered in 16-byte pieces");
}
