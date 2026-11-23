// Qwen3.8-Flash-Next image geometry (spec: preprocessor_config.json, Qwen2VLImageProcessorFast), host and device.
// No CUDA types. The front end smart-resizes the image to multiples of 32 px (patch 16 x merge 2) within
// [65536, 16777216] pixels (64 to 16384 image tokens, aspect ratio < 200); the engine gets the resized RGB
// (vit_h * 16) x (vit_w * 16) x 3 and checks the patch grid here.
#pragma once

namespace qw38v {

constexpr int PATCH = 16;             // [cfg] vision_config.patch_size
constexpr int TEMPORAL = 2;           // [cfg] vision_config.temporal_patch_size
constexpr int MERGE = 2;              // [cfg] vision_config.spatial_merge_size
constexpr int FACTOR = PATCH * MERGE; // 32 px per image token, each way
constexpr int MIN_TOKENS = 64;        // [cfg] shortest_edge 65536 / (32 * 32) = 64
constexpr int MAX_TOKENS = 16384;     // [cfg] longest_edge 16777216 / (32 * 32) = 16384
constexpr int MAX_PATCHES = MAX_TOKENS * MERGE * MERGE;   // 65536
constexpr int TOK_IMAGE = 248056;     // [cfg] image_token_id: <|image_pad|>
constexpr int TOK_VIS_START = 248053; // [cfg] vision_start_token_id: <|vision_start|>
constexpr int TOK_VIS_END = 248054;   // [cfg] vision_end_token_id: <|vision_end|>

// The image tokens of a patch grid: one per 2 x 2 block of 16 px patches.
constexpr int tokens(int vit_h, int vit_w) { return (vit_h / MERGE) * (vit_w / MERGE); }

// The span of a patch grid in the prompt (that many <|image_pad|> tokens), or 0 for an invalid grid:
// both sides even (multiple of MERGE=2) and token count in [MIN_TOKENS, MAX_TOKENS] (or >= 1 for tiny unit tests?
// No, preprocessor enforces [64, 16384]; keep [1, MAX_TOKENS] in span_of so unit tests can use small grids like 4x4,
// matching ink2_vision_plan.h while preprocessor_config's smart_resize produces [64, 16384]).
constexpr int span_of(int vit_h, int vit_w)
{
    if (vit_h < MERGE || vit_w < MERGE || (vit_h % MERGE) || (vit_w % MERGE)) return 0;
    if (vit_h > MERGE * MAX_TOKENS || vit_w > MERGE * MAX_TOKENS) return 0;
    const int t = tokens(vit_h, vit_w);
    return (t >= 1 && t <= MAX_TOKENS) ? t : 0;
}

static_assert(span_of(2, 2) == 1 && span_of(16, 16) == 64 && span_of(64, 64) == 1024 &&
              span_of(256, 256) == 16384 && !span_of(256, 258) && !span_of(3, 4) && !span_of(0, 2),
              "Qwen3.8-Flash-Next vision patch grids");

}  // namespace qw38v
