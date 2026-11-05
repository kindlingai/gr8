// DS41 image span geometry (DeepSeek-V4.1-Flash-1017 `inference/image_processor.py`), host and device. No CUDA types.
// The front end plans the resize (plan_image_grid); the engine gets the patch grid and checks it here.
#pragma once
#include <cstdint>

namespace ds41v {

constexpr int PATCH = 14;        // [cfg] vision_config.patch_size
constexpr int DOWN = 3;          // [cfg] vision_config.downsample_ratio
constexpr int MAX_TOKENS = 1024; // [cfg] vision_config.max_image_tokens: the span, delimiters included

// image_processor.num_image_tokens: [IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h + [IMAGE_END]
constexpr int span_tokens(int llm_h, int llm_w) { return llm_h * (llm_w + 1) + 2; }

struct Grid {
    int vit_h = 0, vit_w = 0, llm_h = 0, llm_w = 0;
    bool ok = false;
    constexpr int span() const { return span_tokens(llm_h, llm_w); }
    constexpr int patches() const { return vit_h * vit_w; }
    // the type of span row s: 0 IMAGE_START, 1 IMAGE, 2 IMAGE_NEW_LINE, 3 IMAGE_END (image_processor's order)
    constexpr int type(int s) const { return s == 0 ? 0 : s == span() - 1 ? 3 : ((s - 1) % (llm_w + 1) == llm_w ? 2 : 1); }
};

// The llm grid of a patch grid (llm_grid: ceil(patches / 3) each way); ok = a grid plan_image_grid can produce
// (at least one patch each way, a span within MAX_TOKENS).
constexpr Grid grid_of(int vit_h, int vit_w)
{
    Grid g;
    if (vit_h < 1 || vit_w < 1 || vit_h > DOWN * MAX_TOKENS || vit_w > DOWN * MAX_TOKENS) return g;
    g.vit_h = vit_h; g.vit_w = vit_w;
    g.llm_h = (vit_h + DOWN - 1) / DOWN;
    g.llm_w = (vit_w + DOWN - 1) / DOWN;
    g.ok = g.span() <= MAX_TOKENS;
    return g;
}

// The most aligner rows (llm_h * llm_w) any valid grid has, and so the most patches (each row covers 3 x 3 patches).
constexpr int max_llm_rows()
{
    int best = 0;
    for (int h = 1; h <= MAX_TOKENS; ++h)
        for (int w = 1; span_tokens(h, w) <= MAX_TOKENS; ++w)
            if (h * w > best) best = h * w;
    return best;
}
constexpr int MAX_LLM = max_llm_rows();
constexpr int MAX_PATCHES = MAX_LLM * DOWN * DOWN;
static_assert(MAX_LLM == 1021 && MAX_PATCHES == 9189, "one 1 x 1021 row: 1022 + 2 = 1024 span tokens");

}  // namespace ds41v
