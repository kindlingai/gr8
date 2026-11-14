// Image input, host side (DS41, docs/DS41-VISION.md; GLM-5.3-Flash, docs/FLASH-VISION.md; Inkling-2-Small,
// ink2_vision_plan.h): a request's images
// (PROTOCOL.md `images`, the shim's `img` header fields), their checks against the prompt, and the digest that keeps
// image prompts apart in the prefix cache. No CUDA types.
//
// `img <pos>:<vit_h>:<vit_w>:<key>:<rgb base64>`: the span starts at prompt index pos and is image_model().span(vit_h,
// vit_w) image tokens long. rgb is the padded image, (vit_h * patch) x (vit_w * patch) RGB bytes (image_model().patch:
// 14 px for DS41 and GLM-5.3-Flash, 20 px for Inkling-2-Small). key is the front end's (logs only): the engine digests
// the pixels itself (image_digest).
#pragma once
#include "ds41_vision_plan.h"
#include "flash_vision_plan.h"
#include "ink2_vision_plan.h"
#include "../engram/sha256.h"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace g6rt {

constexpr int TOK_IMAGE = 129264;    // [cfg] DS41 image_token_id: <｜deepseek_image｜>, and every row of an image span
constexpr int TOK_IMAGE_GLM5V = 154854;   // [cfg] GLM-5.3-Flash image_token_id: <|image|>, every row of an image span
constexpr int TOK_IMAGE_INK2 = 200054;    // [cfg] Inkling-2-Small image_token_id: <|unused_200054|>, every row of an image span
constexpr int IMAGES_MAX = 16;       // per request (the front end's limit)
constexpr int IMAGE_PATCH = 14;      // the DS41 and GLM-5.3-Flash patch size: their rgb is (vit_h * 14) x (vit_w * 14)

// The image geometry of the engine's model: its image token, its patch size in pixels and the span of a patch grid (0:
// a grid its processor cannot produce).
struct ImageModel {
    const char* name;
    int token;
    int patch;
    int (*span)(int vit_h, int vit_w);
};
inline int ds41_span(int vit_h, int vit_w) { const ds41v::Grid g = ds41v::grid_of(vit_h, vit_w); return g.ok ? g.span() : 0; }
inline int glm5v_span(int vit_h, int vit_w) { return glm5v::span_of(vit_h, vit_w); }
inline int ink2_span(int vit_h, int vit_w) { return ink2v::span_of(vit_h, vit_w); }
constexpr ImageModel DS41_IMAGES{"ds41", TOK_IMAGE, IMAGE_PATCH, ds41_span};
constexpr ImageModel GLM5V_IMAGES{"glm5v", TOK_IMAGE_GLM5V, IMAGE_PATCH, glm5v_span};
constexpr ImageModel INK2_IMAGES{"ink2", TOK_IMAGE_INK2, ink2v::PATCH, ink2_span};
// the build's model (a host test that runs both models sets it)
#if defined(G6_MODEL_FLASH) && G6_MODEL_FLASH
inline ImageModel& image_model() { static ImageModel m = GLM5V_IMAGES; return m; }
#elif defined(G6_MODEL_INK2) && G6_MODEL_INK2
inline ImageModel& image_model() { static ImageModel m = INK2_IMAGES; return m; }
#else
inline ImageModel& image_model() { static ImageModel m = DS41_IMAGES; return m; }
#endif

struct ImageIn {
    int pos = 0, vit_h = 0, vit_w = 0, span = 0;
    std::string key;
    std::shared_ptr<const std::vector<uint8_t>> rgb;   // vit_h * patch * vit_w * patch * 3 bytes
};

// The engine runs images only with the vision tower loaded (GR8_DS41_VISION / GR8_GLM53F_VISION / GR8_INK2_VISION, the driver sets
// this at boot).
inline bool& vision_flag() { static bool on = false; return on; }
inline bool vision_ready() { return vision_flag(); }

namespace img {

inline void fail(const char* what) { throw std::runtime_error(std::string("img: ") + what); }

// Standard base64, padded (the front end's encoding); anything else is refused.
inline std::vector<uint8_t> b64_decode(const std::string& s) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    if (s.size() % 4) fail("rgb is not padded base64");
    std::vector<uint8_t> out;
    out.reserve(s.size() / 4 * 3);
    for (size_t i = 0; i < s.size(); i += 4) {
        const bool last = i + 4 == s.size();
        const int pad = last ? (s[i + 3] == '=') + (s[i + 2] == '=') : 0;
        if (pad == 1 && s[i + 2] == '=') fail("rgb base64 padding");
        int v[4];
        for (int k = 0; k < 4 - pad; ++k) if ((v[k] = val(s[i + k])) < 0) fail("rgb is not base64");
        for (int k = 4 - pad; k < 4; ++k) v[k] = 0;
        const uint32_t w = uint32_t(v[0]) << 18 | uint32_t(v[1]) << 12 | uint32_t(v[2]) << 6 | uint32_t(v[3]);
        out.push_back(uint8_t(w >> 16));
        if (pad < 2) out.push_back(uint8_t(w >> 8));
        if (pad < 1) out.push_back(uint8_t(w));
    }
    return out;
}

inline int parse_int(const std::string& s) {
    if (s.empty() || s.size() > 9) fail("bad integer");
    int v = 0;
    for (char c : s) { if (c < '0' || c > '9') fail("bad integer"); v = v * 10 + (c - '0'); }
    return v;
}

// One `img` header value.
inline ImageIn parse_field(const std::string& v) {
    size_t c[4], at = 0;
    for (auto& x : c) { x = v.find(':', at); if (x == std::string::npos) fail("field needs pos:vit_h:vit_w:key:rgb"); at = x + 1; }
    ImageIn im;
    im.pos = parse_int(v.substr(0, c[0]));
    im.vit_h = parse_int(v.substr(c[0] + 1, c[1] - c[0] - 1));
    im.vit_w = parse_int(v.substr(c[1] + 1, c[2] - c[1] - 1));
    im.key = v.substr(c[2] + 1, c[3] - c[2] - 1);
    if (im.key.size() != 16) fail("key must be 16 hex digits");
    for (char h : im.key) if (!((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f'))) fail("key must be 16 hex digits");
    im.span = image_model().span(im.vit_h, im.vit_w);
    if (!im.span) fail("bad patch grid");
    auto rgb = std::make_shared<std::vector<uint8_t>>(b64_decode(v.substr(c[3] + 1)));
    const size_t patch = size_t(image_model().patch);
    if (rgb->size() != size_t(im.vit_h) * im.vit_w * patch * patch * 3) fail("rgb size does not match the grid");
    im.rgb = std::move(rgb);
    return im;
}

// Spans ascend, do not overlap, lie inside the prompt and are all image tokens.
inline void check(const std::vector<int>& prompt, const std::vector<ImageIn>& images) {
    if (images.size() > size_t(IMAGES_MAX)) fail("too many images");
    const int tok = image_model().token;
    int64_t end = 0;
    for (const auto& im : images) {
        if (im.pos < end || int64_t(im.pos) + im.span > int64_t(prompt.size())) fail("image spans must ascend inside the prompt");
        for (int i = im.pos; i < im.pos + im.span; ++i) if (prompt[size_t(i)] != tok) fail("an image span holds a non-image token");
        end = int64_t(im.pos) + im.span;
    }
}

// sha256 over every image's (pos, vit_h, vit_w as u32 LE, rgb), in order: the prefix-cache namespace of an image prompt.
inline std::array<uint8_t, 32> digest(const std::vector<ImageIn>& images) {
    ds41::engram::Sha256 h;
    auto u32 = [&](uint32_t x) { uint8_t b[4] = {uint8_t(x), uint8_t(x >> 8), uint8_t(x >> 16), uint8_t(x >> 24)}; h.update(b, 4); };
    u32(uint32_t(images.size()));
    for (const auto& im : images) {
        u32(uint32_t(im.pos)); u32(uint32_t(im.vit_h)); u32(uint32_t(im.vit_w));
        h.update(im.rgb->data(), im.rgb->size());
    }
    const std::string hex = h.hex();
    std::array<uint8_t, 32> out{};
    auto nib = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
    for (size_t i = 0; i < 32; ++i) out[i] = uint8_t(nib(hex[2 * i]) << 4 | nib(hex[2 * i + 1]));
    return out;
}

// Why the engine will not run this image request ("" = it will). The rows of an image span reach the model only through
// prefill chunks (the embed scatter), so the prompt must prefill (no prompt logprobs, no steering, a long enough prompt)
// and its last token, a MAIN row, must not be an image row.
inline const char* refusal(const std::vector<int>& prompt, const std::vector<ImageIn>& images, bool decode_prompt) {
    if (images.empty()) return "";
    if (!vision_ready()) return "the engine was started without the vision tower";
    if (decode_prompt) return "an image prompt must prefill (no prompt_logprobs, no steering; DS41: GR8_DS_PREFILL_MIN on)";
    const auto& last = images.back();
    if (size_t(last.pos) + size_t(last.span) >= prompt.size()) return "the prompt ends inside an image span";
    return "";
}

}  // namespace img
}  // namespace g6rt
