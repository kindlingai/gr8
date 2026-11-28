// Tencent Hy4-flash image geometry (spec: processor_config.json, HYV4VLImageProcessor, HYV4FlashVisionPatchMerger),
// host and device. No CUDA types.
//
// The front end smart-resizes the image to multiples of 32 px (patch 16 x merge 2) within [4096, 4194304] pixels
// (up to 2048x2048 = 16,384 ViT patches, aspect ratio <= 200); the engine gets the resized RGB (vit_h * 16) x
// (vit_w * 16) x 3 and emits (vit_h / 2) * (vit_w / 2 + 1) image embeddings (one extra image_newline per merged row).
#pragma once

namespace hy4v {

constexpr int PATCH = 16;              // [cfg] vision_config.patch_size
constexpr int MERGE = 2;               // [cfg] vision_config.spatial_merge_size
constexpr int FACTOR = PATCH * MERGE;  // 32 px per merged spatial cell
constexpr int MIN_PIXELS = 4096;       // [cfg] processor_config shortest_edge
constexpr int MAX_PIXELS = 4194304;    // [cfg] processor_config longest_edge (2048 x 2048)
constexpr int MAX_PATCHES = 16384;     // [cfg] vision_config.max_vit_seq_len (128 x 128)
constexpr int TOK_IMG_START = 120118;  // [cfg] image_start_token_id
constexpr int TOK_IMG_END = 120119;    // [cfg] image_end_token_id
constexpr int TOK_IMAGE = 120120;      // [cfg] image_token_id (<|hy_image_pad:opensource|>)
constexpr int TOK_IMG_NL = 120121;     // [cfg] image_newline_token_id
constexpr int TOK_VIDEO = 120124;      // [cfg] video_token_id

// Merged image tokens for a patch grid: (vit_h / 2) rows, each with (vit_w / 2) merged tokens + 1 image_newline.
constexpr int tokens(int vit_h, int vit_w) { return (vit_h / MERGE) * (vit_w / MERGE + 1); }

// Valid patch grid span in prompt tokens, or 0 for an invalid grid.
constexpr int span_of(int vit_h, int vit_w) {
    if (vit_h < MERGE || vit_w < MERGE || (vit_h % MERGE) || (vit_w % MERGE)) return 0;
    if (vit_h > MAX_PATCHES || vit_w > MAX_PATCHES) return 0;
    const long long patches = static_cast<long long>(vit_h) * static_cast<long long>(vit_w);
    if (patches < 4 || patches > MAX_PATCHES) return 0;
    return tokens(vit_h, vit_w);
}

static_assert(span_of(2, 2) == 2 && span_of(4, 4) == 6 && span_of(64, 64) == 1056 &&
              span_of(128, 128) == 4160 && !span_of(130, 128) && !span_of(3, 4) && !span_of(0, 2),
              "Hy4-flash vision patch grids");

}  // namespace hy4v
