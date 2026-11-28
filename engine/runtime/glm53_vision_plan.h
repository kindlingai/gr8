// GLM-5.3 image geometry (Glm5Next ViT + 6144-dim merger + learned row_break_embed trained by exlr8-vproj), host and
// device. No CUDA types. The front end plans the canvas with the same smart_resize + zero pad as glm5v (patch 14,
// merge 2, 16..8000 merged visual tokens); GLM-5.3 appends one learned row_break_embed after each merged row, so the
// spliced prompt span is (vit_h / 2) * (vit_w / 2 + 1) copies of <|image|> (154854).
#pragma once

namespace glm53v {

constexpr int PATCH = 14;           // vision_config.patch_size
constexpr int MERGE = 2;            // vision_config.spatial_merge_size
constexpr int MIN_TOKENS = 16;      // min merged visual tokens before row-break insertion
constexpr int MAX_TOKENS = 8000;    // max merged visual tokens before row-break insertion

// Merged visual tokens of a patch grid (one per 2 x 2 block of patches, before row breaks).
constexpr int tokens(int vit_h, int vit_w) { return (vit_h / MERGE) * (vit_w / MERGE); }

// Spliced prompt span of a patch grid: H_p * (W_p + 1) copies of <|image|> (row_break_embed at column W_p of each
// merged row), or 0 for a grid the processor cannot produce (both sides even, MIN_TOKENS..MAX_TOKENS merged tokens).
constexpr int span_of(int vit_h, int vit_w)
{
    if (vit_h < MERGE || vit_w < MERGE || vit_h % MERGE || vit_w % MERGE) return 0;
    if (vit_h > MERGE * MAX_TOKENS || vit_w > MERGE * MAX_TOKENS) return 0;
    const int hp = vit_h / MERGE, wp = vit_w / MERGE;
    const int t = hp * wp;
    return t >= MIN_TOKENS && t <= MAX_TOKENS ? hp * (wp + 1) : 0;
}

constexpr int MAX_PATCHES = MAX_TOKENS * MERGE * MERGE;   // 32000
// Worst-case spliced span occurs when W_p = 1 and H_p = MAX_TOKENS: H_p * (W_p + 1) = 2 * MAX_TOKENS = 16000.
constexpr int MAX_SPAN = MAX_TOKENS * 2;                  // 16000

static_assert(span_of(8, 8) == 20 && span_of(16, 16) == 72 && span_of(178, 178) == 8010 &&
              span_of(2, 16000) == 8001 && span_of(16000, 2) == 16000 && !span_of(2, 16002) &&
              !span_of(6, 6) && !span_of(3, 8), "the GLM-5.3 vision grids");

}  // namespace glm53v
