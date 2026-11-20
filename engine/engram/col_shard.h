// Column ownership for Engram: keep global hash ids, pack only owned scales.
#pragma once
#include "hash.h"

namespace ds41 { namespace engram {
struct ColShard
{
    bool enabled = false;
    uint32_t first[kIds] = {}, count[kIds] = {};
    int32_t local[kIds] = {};  // -1 for a foreign column; otherwise its packed scale-row base
    int64_t rows = 0;

    static ColShard make(const HashParams& hp, int e, int rank, int nr)
    {
        ColShard s;
        s.enabled = true;
        for (int j = 0; j < kIds; ++j)
        {
            s.first[j] = hp.offset[e][j]; s.count[j] = hp.prime[e][j];
            s.local[j] = j % nr == rank ? (int32_t)s.rows : -1;
            if (s.local[j] >= 0) s.rows += s.count[j];
        }
        return s;
    }
    // Known column at the device call sites; no search and no runtime mode branch.
    ENGRAM_HD int64_t at(int j, uint32_t id) const
    {
        const uint32_t d = id - first[j];
        return local[j] >= 0 && d < count[j] ? int64_t(local[j]) + d : -1;
    }
    // CPU fetches (including lookahead) contain ids without their column indices.
    int64_t at(uint32_t id) const
    {
        for (int j = 0; j < kIds; ++j)
            if (id >= first[j] && id - first[j] < count[j]) return at(j, id);
        return -1;
    }
    // Runs must tile the packed space, in column order, without crossing a gap.
    bool run(int64_t first_row, int64_t n, int64_t packed) const
    {
        if (first_row < 0 || n <= 0 || first_row + n >= (1ll << 30) - 1) return false;
        for (int j = 0; j < kIds; ++j)
            if (local[j] >= 0 && first_row >= first[j] && first_row + n <= int64_t(first[j]) + count[j])
                return int64_t(local[j]) + first_row - first[j] == packed;
        return false;
    }
};
}} // namespace ds41::engram
