// GR8_KV_SHARE_PAGES: the rows [0, rows) of KV page `from` -> page `to` (kvman-rs PageCopy, g6kvman.h g6kv_copy), for
// one cache binding: its latent cache (64 rows of mla_row B a page, row-major) and, when it has one, its index cache
// (64 x idx_codes B of codes, then 64 FP32 scales: PLANAR within the page, attn/attn.cu append_index_row). Rows [rows, 64)
// of `to` and all of `from` stay. One routine for the device kernel (s2.cuh k_kv_copy_page_rows, W = uint4) and the
// host layout test (tests/test_kv_copy_rows.cpp, W = uint32_t): thread t of nt copies every nt-th word.
#pragma once
#include <cstddef>
#include <cstdint>
#ifdef __CUDACC__
#define G6KV_COPY_HD __host__ __device__
#else
#define G6KV_COPY_HD
#endif

namespace g6kvcopy
{
struct Copy
{
    uint32_t from, to, rows;   // = g6kv_copy
};

// W: the copy word. rows * mla_row and rows * idx_codes are multiples of sizeof(W) (416, 656, 128: any rows for 16 B)
template <class W>
G6KV_COPY_HD inline void copy_page_rows(W* mla, W* idx, unsigned mla_row, unsigned idx_row, Copy c, unsigned t,
                                        unsigned nt)
{
    const size_t wb = sizeof(W);
    const size_t mp = 64 * (size_t)mla_row / wb, mn = (size_t)c.rows * mla_row / wb;
    for (size_t k = t; k < mn; k += nt) mla[(size_t)c.to * mp + k] = mla[(size_t)c.from * mp + k];
    if (!idx) return;
    const size_t codes = (size_t)idx_row - 4;   // the row's codes; its FP32 scale is in the scale plane
    const size_t ip = 64 * (size_t)idx_row / wb, in = (size_t)c.rows * codes / wb;
    for (size_t k = t; k < in; k += nt) idx[(size_t)c.to * ip + k] = idx[(size_t)c.from * ip + k];
    uint32_t* s = reinterpret_cast<uint32_t*>(idx);
    const size_t sp = 64 * (size_t)idx_row / 4, so = 64 * codes / 4;
    for (size_t k = t; k < c.rows; k += nt) s[(size_t)c.to * sp + so + k] = s[(size_t)c.from * sp + so + k];
}
}   // namespace g6kvcopy
