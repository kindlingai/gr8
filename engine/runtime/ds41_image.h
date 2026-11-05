// DS41 image input, host side: a request's images (PROTOCOL.md `images`, the shim's `img` header fields), their checks
// against the prompt, and the digest that keeps image prompts apart in the prefix cache. No CUDA types.
//
// `img <pos>:<vit_h>:<vit_w>:<key>:<rgb base64>`: the span starts at prompt index pos and is ds41v::grid_of(vit_h, vit_w)
// .span() image tokens long; rgb is the padded image, (vit_h * 14) x (vit_w * 14) RGB bytes. key is the front end's
// (logs only): the engine digests the pixels itself (image_digest).
#pragma once
#include "ds41_vision_plan.h"
#include "../engram/sha256.h"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace g6rt {

constexpr int TOK_IMAGE = 129264;    // [cfg] image_token_id: <｜deepseek_image｜>, and every row of an image span
constexpr int IMAGES_MAX = 16;       // per request (the front end's limit)

struct ImageIn {
    int pos = 0, vit_h = 0, vit_w = 0, span = 0;
    std::string key;
    std::shared_ptr<const std::vector<uint8_t>> rgb;   // vit_h * 14 * vit_w * 14 * 3 bytes
};

// The engine runs images only with the vision tower loaded (GR8_DS_VISION, g6m4.cu sets this at boot).
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
    const ds41v::Grid g = ds41v::grid_of(im.vit_h, im.vit_w);
    if (!g.ok) fail("bad patch grid");
    im.span = g.span();
    auto rgb = std::make_shared<std::vector<uint8_t>>(b64_decode(v.substr(c[3] + 1)));
    if (rgb->size() != size_t(im.vit_h) * im.vit_w * ds41v::PATCH * ds41v::PATCH * 3) fail("rgb size does not match the grid");
    im.rgb = std::move(rgb);
    return im;
}

// Spans ascend, do not overlap, lie inside the prompt and are all image tokens.
inline void check(const std::vector<int>& prompt, const std::vector<ImageIn>& images) {
    if (images.size() > size_t(IMAGES_MAX)) fail("too many images");
    int64_t end = 0;
    for (const auto& im : images) {
        if (im.pos < end || int64_t(im.pos) + im.span > int64_t(prompt.size())) fail("image spans must ascend inside the prompt");
        for (int i = im.pos; i < im.pos + im.span; ++i) if (prompt[size_t(i)] != TOK_IMAGE) fail("an image span holds a non-image token");
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
    if (decode_prompt) return "an image prompt must prefill (no prompt_logprobs, no steering, GR8_DS_PREFILL_MIN on)";
    const auto& last = images.back();
    if (size_t(last.pos) + size_t(last.span) >= prompt.size()) return "the prompt ends inside an image span";
    return "";
}

}  // namespace img
}  // namespace g6rt
