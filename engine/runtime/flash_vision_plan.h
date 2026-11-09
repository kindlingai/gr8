// GLM-5.3-Flash image geometry (transformers `Glm5NextImageProcessor`, docs/FLASH-VISION.md), host and device. No CUDA
// types. The front end plans the canvas (smart_resize and the zero pad); the engine gets the patch grid and checks it here.
#pragma once

namespace glm5v {

constexpr int PATCH = 14;           // [cfg] vision_config.patch_size
constexpr int MERGE = 2;            // [cfg] vision_config.spatial_merge_size
constexpr int MIN_TOKENS = 16;      // [cfg] processor_config.json min_image_tokens
constexpr int MAX_TOKENS = 8000;    // [cfg] processor_config.json max_image_tokens

// The image tokens of a patch grid: one per 2 x 2 block of patches.
constexpr int tokens(int vit_h, int vit_w) { return (vit_h / MERGE) * (vit_w / MERGE); }

// The span of a patch grid in the prompt (that many image tokens), or 0 for a grid the processor cannot produce: both
// sides even, between MIN_TOKENS and MAX_TOKENS tokens.
constexpr int span_of(int vit_h, int vit_w)
{
    if (vit_h < MERGE || vit_w < MERGE || vit_h % MERGE || vit_w % MERGE) return 0;
    if (vit_h > MERGE * MAX_TOKENS || vit_w > MERGE * MAX_TOKENS) return 0;
    const int t = tokens(vit_h, vit_w);
    return t >= MIN_TOKENS && t <= MAX_TOKENS ? t : 0;
}

constexpr int MAX_PATCHES = MAX_TOKENS * MERGE * MERGE;   // 32000
static_assert(span_of(8, 8) == 16 && span_of(178, 178) == 7921 && span_of(2, 16000) == 8000 && !span_of(2, 16002) &&
              !span_of(6, 6) && !span_of(3, 8), "the processor's grids");

}  // namespace glm5v
