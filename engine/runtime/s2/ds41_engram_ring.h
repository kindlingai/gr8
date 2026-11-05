// DS41 Engram history of a stored or parked chat, host side. The live steps leave each slot's Engram ring holding the
// tokens of positions [pos - kRing, pos) with -1 at image rows (ds41_engram_ops.cuh k_engram_dead_tok: an image token is
// dead in every n-gram). A prefix-cache entry (Ds41PrefixCache::capture) and a GR8_DS_PARK resume (ds41_native.cuh) rebuild
// that ring from the chat's tokens, so they take the request's image rows too: an image that ends within 3 tokens of the
// snapshot position gives the same n-grams after the reuse as in the live run. No CUDA types.
#pragma once
#include "../ds41_image.h"
#include "../../engram/engram_ring.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace g6s2 {

// [first, first + rows) of each image span of a request, ascending (ServeReq::images is ascending)
using Ds41ImageRows = std::vector<std::pair<int, int>>;

inline Ds41ImageRows ds41_image_rows(const std::vector<g6rt::ImageIn>& images) {
    Ds41ImageRows r;
    r.reserve(images.size());
    for (const auto& im : images) r.emplace_back(im.pos, im.span);
    return r;
}

inline bool ds41_image_row(const Ds41ImageRows& rows, int t) {
    for (const auto& s : rows)
        if (t >= s.first && t < s.first + s.second) return true;
    return false;
}

// ring = kRing ints: the ring of tokens [pos - kRing, pos), -1 before position 0 and at image rows (the live ring)
inline void ds41_engram_ring(int* ring, const std::vector<int>& tokens, uint32_t pos, const Ds41ImageRows& images) {
    constexpr int K = ds41::engram::kRing;
    std::fill_n(ring, K, -1);
    const int end = int(std::min<size_t>(pos, tokens.size()));
    for (int t = std::max(0, end - K); t < end; ++t)
        ds41::engram::ring_put(ring, t, ds41_image_row(images, t) ? -1 : tokens[size_t(t)]);
}

} // namespace g6s2
