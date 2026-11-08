// Host-only synthetic index fixtures. These are NOT model decision rows.
#pragma once
#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace sweep_topk {
constexpr int width = 512;
// Floyd sampling: exactly min(512, visible) distinct positions, no rejection
// loop near the 512/513 boundary. Keep the small selected set sorted.
template<class RNG> inline void uniform(int32_t* out, int visible, RNG& rng) {
    if (visible < 0) throw std::runtime_error("negative visible rows");
    const int count = std::min(width, visible);
    int size = 0;
    for (int j = visible - count; j < visible; ++j) {
        const int draw = std::uniform_int_distribution<int>(0, j)(rng);
        auto at = std::lower_bound(out, out + size, draw);
        const int value = at != out + size && *at == draw ? j : draw;
        at = std::lower_bound(out, out + size, value);
        std::move_backward(at, out + size, out + size + 1);
        *at = value;
        ++size;
    }
    std::fill(out + count, out + width, -1);
}
// Shared: one random anchor set visible to every row, 100% shared indices.
// Local: contiguous most recent keys; overlap is governed by position steps.
// Uniform: independent, without replacement. All preserve input row order.
inline std::vector<int32_t> make(const std::vector<int32_t>& positions, int ratio,
                                 int key_rows, const std::string& pattern, uint64_t seed) {
    if (positions.empty() || ratio < 1 || ratio > 2 || key_rows < 1)
        throw std::runtime_error("invalid top-k shape");
    if (pattern != "uniform" && pattern != "shared" && pattern != "local")
        throw std::runtime_error("unknown top-k pattern");
    std::vector<int32_t> result(positions.size() * width, -1), anchor(width);
    std::mt19937_64 rng(seed);
    int common = key_rows;
    for (auto p : positions) {
        if (p < 0 || p > 1048576) throw std::runtime_error("invalid position");
        common = std::min(common, (p + 1) / ratio);
    }
    if (pattern == "shared") uniform(anchor.data(), common, rng);
    for (size_t row = 0; row < positions.size(); ++row) {
        auto* out = result.data() + row * width;
        const int visible = std::min(key_rows, (positions[row] + 1) / ratio);
        if (pattern == "uniform") uniform(out, visible, rng);
        else if (pattern == "shared") std::copy(anchor.begin(), anchor.end(), out);
        else for (int i = 0, count = std::min(width, visible); i < count; ++i)
            out[i] = visible - count + i;
    }
    return result;
}
struct Metrics {
    size_t rows = 0, active = 0, unique = 0, adjacent_pairs = 0;
    double within_row_unique_fraction = 1, adjacent_overlap_fraction = 0;
};
// Adjacent overlap = intersection / min(active counts), excluding empty pairs.
// Only compare rows in the same cache/session slot. This is not all-pairs overlap.
inline Metrics metrics(const std::vector<int32_t>& topk, const std::vector<int32_t>& slots = {}) {
    if (topk.size() % width || (!slots.empty() && slots.size() != topk.size() / width))
        throw std::runtime_error("invalid metrics shape");
    Metrics m; m.rows = topk.size() / width;
    std::vector<int32_t> prev;
    for (size_t r = 0; r < m.rows; ++r) {
        std::vector<int32_t> current;
        for (int i = 0; i < width; ++i) if (topk[r * width + i] >= 0) current.push_back(topk[r * width + i]);
        m.active += current.size();
        std::sort(current.begin(), current.end());
        current.erase(std::unique(current.begin(), current.end()), current.end());
        m.unique += current.size();
        if (!current.empty() && !prev.empty() && (slots.empty() || slots[r] == slots[r-1])) {
            size_t a = 0, b = 0, intersection = 0;
            while (a < prev.size() && b < current.size()) {
                if (prev[a] < current[b]) ++a;
                else if (prev[a] > current[b]) ++b;
                else { ++a; ++b; ++intersection; }
            }
            m.adjacent_overlap_fraction += double(intersection) / std::min(prev.size(), current.size());
            ++m.adjacent_pairs;
        }
        prev = std::move(current);
    }
    if (m.active) m.within_row_unique_fraction = double(m.unique) / m.active;
    if (m.adjacent_pairs) m.adjacent_overlap_fraction /= m.adjacent_pairs;
    return m;
}
} // namespace sweep_topk
