// Inkling-2-Small image geometry (spec: the checkpoint's image processor), host and device. No CUDA types. The front end
// composites and pads the image; the engine gets the patch grid and checks it here. The processor does not resize: an
// image of height h and width w becomes ceil(h / 40) x ceil(w / 40) image tokens, each a 2 x 2 block of 20 px patches.
#pragma once

namespace ink2v {

constexpr int PATCH = 20;           // [cfg] vision_config native patch size
constexpr int MERGE = 2;            // [cfg] vision_config adapter merge size
constexpr int FACTOR = PATCH * MERGE;   // 40 px of image per image token, each way
constexpr int MIN_TOKENS = 1;
constexpr int MAX_TOKENS = 6400;    // the engine's budget: 3200 x 3200 px, or 3840 x 2160 (5184 tokens)
constexpr int MAX_PATCHES = MAX_TOKENS * MERGE * MERGE;   // 25600

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

// The grid of an image of height x width pixels (tokens == 0: outside the budget, or an empty image).
struct Plan {
    int rows = 0, cols = 0;         // image tokens down and across
    int vit_h = 0, vit_w = 0;       // patches down and across
    int pad_h = 0, pad_w = 0;       // the padded canvas, pixels
    int tokens = 0;
};
constexpr Plan plan(int height, int width)
{
    Plan p;
    if (height < 1 || width < 1 || height > FACTOR * MAX_TOKENS || width > FACTOR * MAX_TOKENS) return p;
    p.rows = (height + FACTOR - 1) / FACTOR;
    p.cols = (width + FACTOR - 1) / FACTOR;
    p.vit_h = p.rows * MERGE;
    p.vit_w = p.cols * MERGE;
    p.pad_h = p.vit_h * PATCH;
    p.pad_w = p.vit_w * PATCH;
    p.tokens = span_of(p.vit_h, p.vit_w);
    return p;
}

static_assert(span_of(2, 2) == 1 && span_of(4, 6) == 6 && span_of(160, 160) == 6400 && span_of(108, 192) == 5184 &&
              !span_of(162, 160) && !span_of(3, 4) && !span_of(4, 1) && !span_of(0, 2) && span_of(2, 12800) == 6400 &&
              !span_of(2, 12802), "the processor's grids");
static_assert(plan(1, 1).tokens == 1 && plan(1, 1).pad_h == 40 && plan(41, 40).rows == 2 && plan(41, 40).cols == 1 &&
              plan(2160, 3840).tokens == 5184 && plan(3200, 3200).tokens == 6400 && plan(3201, 3200).tokens == 0 &&
              plan(0, 5).tokens == 0, "the processor's plan");

}  // namespace ink2v
