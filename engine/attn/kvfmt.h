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
#ifndef G6_KV4
#define G6_KV4 0
#endif

namespace g6kv
{
#if G6_KV4
constexpr int ROW = 416, SCALE_OFF = 256, ROPE_OFF = 288;
constexpr const char* NAME = "int4g32h";
#else
constexpr int ROW = 656, SCALE_OFF = 512, ROPE_OFF = 528;
constexpr const char* NAME = "fp8";
#endif
constexpr int IDX_ROW = 132;
static_assert(ROW % 16 == 0, "rows are gathered in 16-byte pieces");
}
