// Host-only canonical session KV staging, compact-page remapping, shard assembly, and session metadata (#159).
// Used by `canon_mover.cuh` (CUDA device<->host movers), `service.h` (import header validation + rank-0 export
// metadata), and host-only CPU tests (scripts/host_tests.sh).
#pragma once

#include "../../kvman-rs/g6kvman.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace g6s2 {

// Parsed header of a `"g6canon1"` canonical KV blob (kvman-rs src/canon.rs).
struct CanonBlobInfo {
    uint32_t mla_row = 0;
    uint32_t nb = 0;
    uint64_t first = 0;
    uint64_t rows = 0;
    std::vector<uint8_t> index; // 0 or 1 per binding
};

inline uint32_t canon_le32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline uint64_t canon_le64(const uint8_t* p) {
    return uint64_t(canon_le32(p)) | (uint64_t(canon_le32(p + 4)) << 32);
}
inline void canon_put_le32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(uint8_t((x >> (8 * i)) & 0xff));
}
inline void canon_put_le64(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back(uint8_t((x >> (8 * i)) & 0xff));
}

// Validates a `"g6canon1"` blob header and total byte length without allocating row storage.
inline bool canon_inspect_blob(const uint8_t* data, size_t n, CanonBlobInfo& out, std::string& err) {
    out = CanonBlobInfo{};
    if (n < 32 || std::memcmp(data, "g6canon1", 8) != 0) {
        err = "not a g6canon1 blob";
        return false;
    }
    const uint32_t mla_row = canon_le32(data + 8);
    const uint32_t nb = canon_le32(data + 12);
    const uint64_t first = canon_le64(data + 16);
    const uint64_t rows = canon_le64(data + 24);
    if (mla_row == 0 || (mla_row % 16) != 0 || nb == 0 || size_t(nb) > n - 32 || first + rows < first) {
        err = "invalid g6canon1 header";
        return false;
    }
    uint64_t row_bytes = 0;
    out.index.resize(nb);
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t f = data[32 + b];
        if (f > 1) {
            err = "invalid g6canon1 binding flag";
            return false;
        }
        out.index[b] = f;
        row_bytes += uint64_t(mla_row) + (f ? 132ull : 0ull);
    }
    const uint64_t left = uint64_t(n - 32 - nb);
    if (row_bytes == 0 || (rows > 0 && left / row_bytes != rows) || rows * row_bytes != left) {
        err = "g6canon1 byte length mismatch";
        return false;
    }
    out.mla_row = mla_row;
    out.nb = nb;
    out.first = first;
    out.rows = rows;
    return true;
}

inline bool canon_read_file_bytes(const std::string& path, std::vector<uint8_t>& out, std::string& err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    const std::streamoff sz = f.tellg();
    if (sz < 0) {
        err = "cannot size " + path;
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.resize(size_t(sz));
    if (sz > 0 && !f.read(reinterpret_cast<char*>(out.data()), sz)) {
        err = "read failed on " + path;
        return false;
    }
    return true;
}

inline void canon_write_file_atomic(const std::string& path, const void* data, size_t n) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("canon: cannot create " + tmp);
        if (n > 0) f.write(reinterpret_cast<const char*>(data), std::streamsize(n));
        if (!f) throw std::runtime_error("canon: write failed on " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("canon: rename failed for " + path);
}

inline bool canon_inspect_file(const std::string& path, CanonBlobInfo& out, std::string& err) {
    std::vector<uint8_t> bytes;
    if (!canon_read_file_bytes(path, bytes, err)) return false;
    return canon_inspect_blob(bytes.data(), bytes.size(), out, err);
}

// Compact owned-page plan for one rank of one session run.
// Maps the sparse physical pages this rank actually holds for [first_row, first_row + rows) into dense indices
// [0 .. owned_pages), preserving the `< place` vs `>= place` partition so `g6kv_canon_export` / `g6kv_canon_import`
// see the exact same logical-to-compact mapping without allocating host memory for the whole KV pool.
struct CanonPlan {
    std::vector<uint32_t> phys_pages;    // dense index k -> real physical page id in the GPU pool
    std::vector<uint32_t> compact_list;  // run's page list rewritten into dense indices (0 for pages not owned)
    uint32_t compact_place = 0;          // number of owned physical pages < real place (or UINT32_MAX if place == UINT32_MAX)
    uint32_t first_page = 0;
    uint32_t end_page = 0;
};

inline CanonPlan canon_plan_owned(const uint8_t* flags, uint32_t nb, uint32_t nr, uint32_t below_rows,
                                  const uint32_t* list, size_t n_list, uint32_t place, uint32_t rank,
                                  uint64_t first_row, uint64_t rows) {
    CanonPlan p;
    if (rows == 0) {
        p.compact_place = (place == UINT32_MAX) ? UINT32_MAX : 0;
        return p;
    }
    if ((below_rows % 64u) != 0) throw std::runtime_error("canon_plan_owned: below_rows must be a multiple of 64");
    const uint64_t end_row = first_row + rows;
    if (end_row < first_row) throw std::runtime_error("canon_plan_owned: row overflow");
    p.first_page = uint32_t(first_row / 64u);
    p.end_page = uint32_t((end_row + 63u) / 64u);
    if (p.end_page > n_list) throw std::runtime_error("canon_plan_owned: rows past the run's page list");

    bool any_non_cp = false, any_cp = false;
    for (uint32_t b = 0; b < nb; ++b) {
        if ((flags[b] & 2u) && nr > 0) any_cp = true;
        else any_non_cp = true;
    }

    // Collect every physical page in `list` that this rank holds for [first_page, end_page), partitioned into
    // low (< place) and high (>= place) so compact_list preserves s2_list_place's `< place` filter order.
    std::vector<uint32_t> low_all;
    low_all.reserve(n_list);
    for (size_t i = 0; i < n_list; ++i) {
        if (list[i] < place) low_all.push_back(list[i]);
    }
    const uint32_t rp = below_rows / 64u;

    std::vector<bool> need_list_idx(n_list, false);
    if (any_non_cp) {
        for (uint32_t lp = p.first_page; lp < p.end_page; ++lp) need_list_idx[lp] = true;
    }
    // Track which low_all indices are owned by CP bindings on this rank.
    std::vector<uint32_t> cp_phys_owned;
    if (any_cp) {
        uint32_t j = 0;
        for (uint32_t lp = 0; lp < p.end_page; ++lp) {
            if (lp < rp) {
                if (lp >= p.first_page) need_list_idx[lp] = true;
            } else if (nr > 0 && (lp % nr) == rank) {
                const uint32_t at = rp + j++;
                if (at >= low_all.size())
                    throw std::runtime_error("canon_plan_owned: CP tail page missing below place");
                if (lp >= p.first_page) cp_phys_owned.push_back(low_all[at]);
            }
        }
    }

    // Gather all unique needed physical pages split into < place and >= place.
    std::vector<uint32_t> low_phys, high_phys;
    std::unordered_map<uint32_t, uint32_t> seen;
    auto add_phys = [&](uint32_t phys) {
        if (seen.count(phys)) return;
        seen[phys] = 1;
        if (phys < place) low_phys.push_back(phys);
        else high_phys.push_back(phys);
    };
    for (size_t i = 0; i < n_list; ++i) {
        if (need_list_idx[i]) add_phys(list[i]);
    }
    for (uint32_t phys : cp_phys_owned) add_phys(phys);

    // Preserve ascending physical order inside low and high partitions (deterministic, cache-friendly).
    std::sort(low_phys.begin(), low_phys.end());
    std::sort(high_phys.begin(), high_phys.end());

    // Important: for CP bindings, `canon::table` indexes `low_all[rp + j]` where `low_all` is ALL entries of
    // `compact_list` with `< compact_place`, in list order! So every entry of `list` that is `< place` (even for
    // lp < first_page, e.g. tail-only migration where first_page > 0!) MUST still satisfy `< compact_place` in
    // `compact_list` so `rp + j` lands on the exact same slot!
    // We guarantee this by mapping un-needed `< place` entries to a valid dummy `< compact_place` index (0, or a
    // dedicated slot if low_phys is empty), and un-needed `>= place` entries to `compact_place` (which is `>= compact_place`).
    std::unordered_map<uint32_t, uint32_t> phys_to_compact;
    p.phys_pages.reserve(low_phys.size() + high_phys.size());
    for (uint32_t phys : low_phys) {
        phys_to_compact[phys] = uint32_t(p.phys_pages.size());
        p.phys_pages.push_back(phys);
    }
    const uint32_t n_low = uint32_t(p.phys_pages.size());
    // If a CP binding has un-owned low entries before first_page and n_low == 0 (this rank owns 0 low pages for this
    // range), reserve compact_place = 1 dummy low threshold (never read since this rank owns none of them).
    const uint32_t low_sentinel = 0;
    const uint32_t cplace = (place == UINT32_MAX) ? UINT32_MAX : (n_low > 0 ? n_low : (any_cp ? 1u : 0u));
    // If we bumped cplace to 1 while n_low == 0, high_phys must start at index >= cplace; however if high_phys is
    // non-empty and n_low == 0 with any_cp, we can avoid shifting phys_pages by only bumping when high_phys is empty,
    // or simply mapping high indices via compact_place = n_low (wait: if n_low == 0 and any_cp, this rank owns NO
    // pages in any CP binding in [first_page, end_page), so `canon::table` for CP only needs low_all.size() >= rp+j
    // for lp >= first_page where it IS owned — which never happens when n_low == 0!).
    // Therefore cplace = n_low works cleanly whenever n_low > 0, and if n_low == 0 && any_cp, no CP page is owned!
    (void)cplace;
    for (uint32_t phys : high_phys) {
        phys_to_compact[phys] = uint32_t(p.phys_pages.size());
        p.phys_pages.push_back(phys);
    }
    p.compact_place = (place == UINT32_MAX) ? UINT32_MAX : n_low;

    p.compact_list.resize(n_list, 0);
    for (size_t i = 0; i < n_list; ++i) {
        const auto it = phys_to_compact.find(list[i]);
        if (it != phys_to_compact.end()) {
            p.compact_list[i] = it->second;
        } else if (list[i] < place) {
            p.compact_list[i] = low_sentinel; // < compact_place (only matters if n_low > 0)
        } else {
            p.compact_list[i] = std::max(p.compact_place, uint32_t(p.phys_pages.size())); // >= compact_place
        }
    }
    return p;
}

// Per-rank compact shard produced by exporting one rank's owned GPU pages.
struct CanonShard {
    uint32_t mla_row = 0;
    uint32_t nb = 0;
    uint32_t nr = 0;
    uint32_t rank = 0;
    uint32_t below_rows = 0;
    uint32_t compact_place = 0;
    uint64_t first_row = 0;
    uint64_t rows = 0;
    uint32_t owned_pages = 0;
    std::vector<uint8_t> flags;
    std::vector<uint32_t> compact_list;
    std::vector<std::vector<uint8_t>> mla; // [nb][owned_pages * 64 * mla_row]
    std::vector<std::vector<uint8_t>> idx; // [nb][owned_pages * 64 * 132] (empty if !(flags[b] & 1))
};

inline std::vector<uint8_t> canon_shard_encode(const CanonShard& s) {
    std::vector<uint8_t> out;
    out.insert(out.end(), {'g', '6', 'c', 's', 'h', 'r', '1', 0});
    canon_put_le32(out, s.mla_row);
    canon_put_le32(out, s.nb);
    canon_put_le32(out, s.nr);
    canon_put_le32(out, s.rank);
    canon_put_le32(out, s.below_rows);
    canon_put_le32(out, s.compact_place);
    canon_put_le64(out, s.first_row);
    canon_put_le64(out, s.rows);
    canon_put_le32(out, s.owned_pages);
    canon_put_le32(out, uint32_t(s.compact_list.size()));
    out.insert(out.end(), s.flags.begin(), s.flags.end());
    for (uint32_t x : s.compact_list) canon_put_le32(out, x);
    for (uint32_t b = 0; b < s.nb; ++b) {
        out.insert(out.end(), s.mla[b].begin(), s.mla[b].end());
        if (s.flags[b] & 1u) out.insert(out.end(), s.idx[b].begin(), s.idx[b].end());
    }
    return out;
}

inline CanonShard canon_shard_decode(const uint8_t* data, size_t n) {
    if (n < 56 || std::memcmp(data, "g6cshr1\0", 8) != 0)
        throw std::runtime_error("canon_shard_decode: invalid magic or truncated header");
    CanonShard s;
    s.mla_row = canon_le32(data + 8);
    s.nb = canon_le32(data + 12);
    s.nr = canon_le32(data + 16);
    s.rank = canon_le32(data + 20);
    s.below_rows = canon_le32(data + 24);
    s.compact_place = canon_le32(data + 28);
    s.first_row = canon_le64(data + 32);
    s.rows = canon_le64(data + 40);
    s.owned_pages = canon_le32(data + 48);
    const uint32_t n_list = canon_le32(data + 52);
    size_t off = 56;
    if (off + s.nb + size_t(n_list) * 4 > n) throw std::runtime_error("canon_shard_decode: truncated metadata");
    s.flags.assign(data + off, data + off + s.nb);
    off += s.nb;
    s.compact_list.resize(n_list);
    for (uint32_t i = 0; i < n_list; ++i) {
        s.compact_list[i] = canon_le32(data + off);
        off += 4;
    }
    const size_t mla_bytes = size_t(s.owned_pages) * 64u * size_t(s.mla_row);
    const size_t idx_bytes = size_t(s.owned_pages) * 64u * 132u;
    s.mla.resize(s.nb);
    s.idx.resize(s.nb);
    for (uint32_t b = 0; b < s.nb; ++b) {
        if (off + mla_bytes > n) throw std::runtime_error("canon_shard_decode: truncated MLA data");
        s.mla[b].assign(data + off, data + off + mla_bytes);
        off += mla_bytes;
        if (s.flags[b] & 1u) {
            if (off + idx_bytes > n) throw std::runtime_error("canon_shard_decode: truncated IDX data");
            s.idx[b].assign(data + off, data + off + idx_bytes);
            off += idx_bytes;
        }
    }
    if (off != n) throw std::runtime_error("canon_shard_decode: trailing bytes");
    return s;
}

inline void canon_shard_write(const std::string& path, const CanonShard& s) {
    const auto bytes = canon_shard_encode(s);
    canon_write_file_atomic(path, bytes.data(), bytes.size());
}

inline CanonShard canon_shard_read(const std::string& path) {
    std::vector<uint8_t> bytes;
    std::string err;
    if (!canon_read_file_bytes(path, bytes, err)) throw std::runtime_error("canon_shard_read: " + err);
    return canon_shard_decode(bytes.data(), bytes.size());
}

// Assembles 1 (replicated / nr == 0) or nr (CP) rank shards into a single layout-independent `"g6canon1"` blob
// via `g6kv_canon_export`. When CP ranks have different owned_pages / compact_lists (each rank holds only its own
// pages!), we project all ranks onto one unified compact page space so one `list` + `place` serves `g6kv_canon_export`.
inline std::vector<uint8_t> canon_assemble_shards(const std::vector<CanonShard>& shards, bool check = true) {
    if (shards.empty()) throw std::runtime_error("canon_assemble_shards: empty shard list");
    const CanonShard& s0 = shards[0];
    const uint32_t nr = s0.nr;
    const size_t ranks = shards.size();
    if (nr > 0 && ranks != size_t(nr))
        throw std::runtime_error("canon_assemble_shards: shard count does not match nr");
    if (nr == 0 && ranks != 1)
        throw std::runtime_error("canon_assemble_shards: non-CP assemble expects 1 shard");

    for (size_t r = 0; r < ranks; ++r) {
        const CanonShard& sr = shards[r];
        if (sr.mla_row != s0.mla_row || sr.nb != s0.nb || sr.nr != s0.nr || sr.below_rows != s0.below_rows ||
            sr.first_row != s0.first_row || sr.rows != s0.rows || sr.flags != s0.flags ||
            sr.compact_list.size() != s0.compact_list.size()) {
            throw std::runtime_error("canon_assemble_shards: shard metadata mismatch across ranks");
        }
        if (nr > 0 && sr.rank != uint32_t(r)) {
            throw std::runtime_error("canon_assemble_shards: shards must be ordered by rank 0..nr-1");
        }
    }

    // Fast path: 1 rank (nr == 0) — pass s0's compact buffers directly to g6kv_canon_export.
    const uint32_t nb = s0.nb;
    if (ranks == 1) {
        std::vector<const uint8_t*> m(nb), x(nb);
        std::vector<size_t> mb(nb), xb(nb);
        for (uint32_t b = 0; b < nb; ++b) {
            m[b] = s0.mla[b].empty() ? nullptr : s0.mla[b].data();
            mb[b] = s0.mla[b].size();
            x[b] = s0.idx[b].empty() ? nullptr : s0.idx[b].data();
            xb[b] = s0.idx[b].size();
        }
        size_t need = 0;
        int rc = g6kv_canon_export(s0.mla_row, nb, s0.flags.data(), 0, s0.below_rows,
                                   s0.compact_list.data(), s0.compact_list.size(), s0.compact_place,
                                   m.data(), mb.data(), x.data(), xb.data(),
                                   s0.first_row, s0.rows, check ? 1 : 0, nullptr, 0, &need);
        if (rc != 1 && rc != 0) throw std::runtime_error(std::string("g6kv_canon_export size: ") + g6kv_canon_error());
        std::vector<uint8_t> blob(need);
        rc = g6kv_canon_export(s0.mla_row, nb, s0.flags.data(), 0, s0.below_rows,
                               s0.compact_list.data(), s0.compact_list.size(), s0.compact_place,
                               m.data(), mb.data(), x.data(), xb.data(),
                               s0.first_row, s0.rows, check ? 1 : 0, blob.data(), blob.size(), &need);
        if (rc != 0) throw std::runtime_error(std::string("g6kv_canon_export: ") + g6kv_canon_error());
        return blob;
    }

    // Multi-rank CP path: each rank's shard may have a different `owned_pages` count (e.g. rank 0 holds one more
    // page past R than rank nr-1). Build a canonical identity logical list `0..end_page-1` (`place = UINT32_MAX`) and
    // scatter each rank's owned pages into its logical page slots before calling `g6kv_canon_export`!
    const uint32_t end_page = s0.rows == 0 ? 0u : uint32_t((s0.first_row + s0.rows + 63u) / 64u);
    const uint32_t first_page = s0.rows == 0 ? 0u : uint32_t(s0.first_row / 64u);
    const uint32_t rp = s0.below_rows / 64u;
    std::vector<uint32_t> ident_list(end_page);
    for (uint32_t i = 0; i < end_page; ++i) ident_list[i] = i;

    const size_t mla_page = 64u * size_t(s0.mla_row);
    const size_t idx_page = 64u * 132u;

    // Under ident_list (place = UINT32_MAX), canon::table puts:
    // - non-CP binding: lp -> lp
    // - CP binding: lp < rp -> lp; lp >= rp && lp % nr == r -> rp + j (where j = (lp - rp) / nr)
    // So the max page index any rank r reads is end_page!
    std::vector<std::vector<std::vector<uint8_t>>> um(ranks, std::vector<std::vector<uint8_t>>(nb));
    std::vector<std::vector<std::vector<uint8_t>>> ux(ranks, std::vector<std::vector<uint8_t>>(nb));
    for (size_t r = 0; r < ranks; ++r) {
        const CanonShard& sr = shards[r];
        std::vector<uint32_t> low_src;
        for (uint32_t p : sr.compact_list) if (p < sr.compact_place) low_src.push_back(p);
        for (uint32_t b = 0; b < nb; ++b) {
            const bool is_cp = (sr.flags[b] & 2u) != 0;
            const bool has_idx = (sr.flags[b] & 1u) != 0;
            um[r][b].assign(size_t(end_page) * mla_page, 0);
            if (has_idx) ux[r][b].assign(size_t(end_page) * idx_page, 0);
            uint32_t j = 0;
            for (uint32_t lp = 0; lp < end_page; ++lp) {
                int64_t src_p = -1, dst_p = -1;
                if (!is_cp) {
                    if (lp >= first_page) { src_p = sr.compact_list[lp]; dst_p = lp; }
                } else if (lp < rp) {
                    if (lp >= first_page) { src_p = sr.compact_list[lp]; dst_p = lp; }
                } else if ((lp % nr) == uint32_t(r)) {
                    const uint32_t at = rp + j++;
                    if (lp >= first_page) {
                        if (at >= low_src.size()) throw std::runtime_error("canon_assemble_shards: missing CP page");
                        src_p = low_src[at];
                        dst_p = at;
                    }
                }
                if (src_p >= 0 && dst_p >= 0) {
                    std::memcpy(um[r][b].data() + size_t(dst_p) * mla_page,
                                sr.mla[b].data() + size_t(src_p) * mla_page, mla_page);
                    if (has_idx) {
                        std::memcpy(ux[r][b].data() + size_t(dst_p) * idx_page,
                                    sr.idx[b].data() + size_t(src_p) * idx_page, idx_page);
                    }
                }
            }
        }
    }

    std::vector<const uint8_t*> m(ranks * nb), x(ranks * nb);
    std::vector<size_t> mb(ranks * nb), xb(ranks * nb);
    for (size_t r = 0; r < ranks; ++r) {
        for (uint32_t b = 0; b < nb; ++b) {
            m[r * nb + b] = um[r][b].data();
            mb[r * nb + b] = um[r][b].size();
            x[r * nb + b] = ux[r][b].empty() ? nullptr : ux[r][b].data();
            xb[r * nb + b] = ux[r][b].size();
        }
    }
    size_t need = 0;
    int rc = g6kv_canon_export(s0.mla_row, nb, s0.flags.data(), nr, s0.below_rows,
                               ident_list.data(), ident_list.size(), UINT32_MAX,
                               m.data(), mb.data(), x.data(), xb.data(),
                               s0.first_row, s0.rows, check ? 1 : 0, nullptr, 0, &need);
    if (rc != 1 && rc != 0) throw std::runtime_error(std::string("g6kv_canon_export size: ") + g6kv_canon_error());
    std::vector<uint8_t> blob(need);
    rc = g6kv_canon_export(s0.mla_row, nb, s0.flags.data(), nr, s0.below_rows,
                           ident_list.data(), ident_list.size(), UINT32_MAX,
                           m.data(), mb.data(), x.data(), xb.data(),
                           s0.first_row, s0.rows, check ? 1 : 0, blob.data(), blob.size(), &need);
    if (rc != 0) throw std::runtime_error(std::string("g6kv_canon_export: ") + g6kv_canon_error());
    return blob;
}

// Unpacks a `"g6canon1"` blob for one destination rank into compact owned-page buffers ready for GPU upload.
struct CanonUnpacked {
    CanonBlobInfo info;
    CanonPlan plan;
    std::vector<std::vector<uint8_t>> mla; // [nb][owned_pages * 64 * mla_row]
    std::vector<std::vector<uint8_t>> idx; // [nb][owned_pages * 64 * 132]
    uint64_t written = 0;
};

inline CanonUnpacked canon_unpack_for_rank(const uint8_t* blob, size_t blob_n, uint32_t mla_row, uint32_t nb,
                                           const uint8_t* flags, uint32_t nr, uint32_t below_rows,
                                           const uint32_t* list, size_t n_list, uint32_t place, uint32_t rank) {
    CanonUnpacked u;
    std::string err;
    if (!canon_inspect_blob(blob, blob_n, u.info, err))
        throw std::runtime_error("canon_unpack_for_rank: " + err);
    u.plan = canon_plan_owned(flags, nb, nr, below_rows, list, n_list, place, rank, u.info.first, u.info.rows);
    const size_t n_owned = u.plan.phys_pages.size();
    u.mla.resize(nb);
    u.idx.resize(nb);
    std::vector<uint8_t*> mp(nb), xp(nb);
    std::vector<size_t> mb(nb), xb(nb);
    for (uint32_t b = 0; b < nb; ++b) {
        u.mla[b].assign(n_owned * 64u * size_t(mla_row), 0);
        mp[b] = u.mla[b].empty() ? nullptr : u.mla[b].data();
        mb[b] = u.mla[b].size();
        if (flags[b] & 1u) {
            u.idx[b].assign(n_owned * 64u * 132u, 0);
            xp[b] = u.idx[b].empty() ? nullptr : u.idx[b].data();
            xb[b] = u.idx[b].size();
        } else {
            xp[b] = nullptr;
            xb[b] = 0;
        }
    }
    if (g6kv_canon_import(blob, blob_n, mla_row, nb, flags, nr, below_rows,
                          u.plan.compact_list.data(), u.plan.compact_list.size(), u.plan.compact_place, rank,
                          mp.data(), mb.data(), xp.data(), xb.data(), &u.written) != 0) {
        throw std::runtime_error(std::string("g6kv_canon_import: ") + g6kv_canon_error());
    }
    return u;
}

// Host session metadata accompanying an exported KV blob (`export.<export_id>.meta`), written by rank 0 at the
// export event fence. Lets an operator or coordinator construct `req.<new_id>` for importing onto another engine.
struct CanonSessionMeta {
    uint64_t request_id = 0;
    uint64_t export_id = 0;
    uint32_t prompt_tokens = 0;
    uint32_t emitted_tokens = 0;
    uint32_t committed_rows = 0;
    uint32_t remaining_max_tokens = 0;
    std::string req_header;        // original req.<id> header line (n_gen ... stop ...)
    std::vector<int32_t> tokens;   // prompt ++ emitted
};

inline void canon_meta_write(const std::string& path, const CanonSessionMeta& m) {
    std::ostringstream o;
    o << "g6s2_export_meta 1\n"
      << "request " << m.request_id << " export " << m.export_id
      << " prompt " << m.prompt_tokens << " emitted " << m.emitted_tokens
      << " rows " << m.committed_rows << " remaining " << m.remaining_max_tokens << "\n"
      << m.req_header << "\n";
    for (size_t i = 0; i < m.tokens.size(); ++i) {
        if (i) o << ' ';
        o << m.tokens[i];
    }
    o << "\n";
    const std::string s = o.str();
    canon_write_file_atomic(path, s.data(), s.size());
}

inline CanonSessionMeta canon_meta_read(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("canon_meta_read: cannot open " + path);
    std::string magic, line2, hdr, tokline;
    if (!std::getline(f, magic) || magic != "g6s2_export_meta 1")
        throw std::runtime_error("canon_meta_read: bad header");
    if (!std::getline(f, line2) || !std::getline(f, hdr) || !std::getline(f, tokline))
        throw std::runtime_error("canon_meta_read: truncated file");
    CanonSessionMeta m;
    std::istringstream is(line2);
    std::string k1, k2, k3, k4, k5, k6;
    if (!(is >> k1 >> m.request_id >> k2 >> m.export_id >> k3 >> m.prompt_tokens >> k4 >> m.emitted_tokens
             >> k5 >> m.committed_rows >> k6 >> m.remaining_max_tokens) ||
        k1 != "request" || k2 != "export" || k3 != "prompt" || k4 != "emitted" || k5 != "rows" || k6 != "remaining") {
        throw std::runtime_error("canon_meta_read: bad metadata line");
    }
    m.req_header = hdr;
    std::istringstream ts(tokline);
    int32_t tok = 0;
    while (ts >> tok) m.tokens.push_back(tok);
    return m;
}

// Rewrites `n_gen <N>` in the original request header to `remaining_max_tokens` and appends the full committed
// token sequence (`prompt ++ emitted`) as the prompt line of a `req.<new_id>` file ready for `IMPORT`.
inline std::string canon_make_import_req(const CanonSessionMeta& m) {
    if (m.remaining_max_tokens == 0) throw std::runtime_error("canon_make_import_req: session has 0 remaining tokens");
    if (m.tokens.size() <= m.committed_rows)
        throw std::runtime_error("canon_make_import_req: need at least one tail token past committed_rows");
    std::string hdr = m.req_header;
    // Replace leading "n_gen <old>" with "n_gen <remaining>"
    if (hdr.rfind("n_gen ", 0) == 0) {
        const size_t sp = hdr.find(' ', 6);
        const std::string rest = sp == std::string::npos ? "" : hdr.substr(sp);
        hdr = "n_gen " + std::to_string(m.remaining_max_tokens) + rest;
    } else if (hdr.empty()) {
        hdr = "n_gen " + std::to_string(m.remaining_max_tokens) + " temp 0 seed 1 min 0 stop -";
    }
    std::ostringstream o;
    o << hdr << "\n";
    for (size_t i = 0; i < m.tokens.size(); ++i) {
        if (i) o << ',';
        o << m.tokens[i];
    }
    o << "\n";
    return o.str();
}

} // namespace g6s2
