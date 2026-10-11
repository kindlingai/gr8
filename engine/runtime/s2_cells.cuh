// S2 step layout (S2In) and the kernel that turns it into attention step cells. Own header so the attention unit test
// (tests/s2_attn_test.cu) runs the same kernel as the engine. Included by s2.cuh; needs -DDEC_S2.
//
// Two forms of the cells of a step (the same S2In, one kernel):
//  - per session: cells[session slot][layer] = the session's bounds (context, page_count = the pool's pages, table_size =
//    pages, reuse_rows = rows of a shared-index layer). One attention call per session and layer (G6_S2_ATTN_PERSESSION=1).
//    page_count only bounds a table entry. The slot's pages may sit anywhere in the pool, and its table holds -1 past them.
//  - batched: one cell per layer for ALL rows of the step (bcells[layer], its `rows` pointer fixed at init) and, per layer,
//    one G6AttnRow per step row (rows[layer][row]): the bounds and the block table of the session the row belongs to. Cell
//    fields context / table_size are the largest of the sessions' (validation bound), reuse_rows = M or 0.
//    The attention call takes the step's rows in step order, so its outputs are the concatenated rows and a shared-index
//    layer reads its source layer's index row by step row.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "../attn/replay.h"

namespace g6rt
{

constexpr int S2N = 8;                // session slots

// per-step session layout. ns sessions in the step; j-th: slot sess[j], rows row0[j]..row0[j]+rows[j]-1, context ctx[j]
// (sessions in ascending slot order, rows contiguous, row0[0] = 0)
struct S2In
{
    int ns, cls;                      // cls: the step's k1 row class (1 / 3 / 5)
    int sess[S2N], row0[S2N], rows[S2N], ctx[S2N];
    int src0[S2N];                    // draft model, first form: the j-th session's first row in the main model's x
};

struct S2Cells
{
    const uint32_t* in;
    G6AttnStepCell* cells;            // [S2N][nl]
    const int* shared;                // [nl]
    int nl;
    G6AttnStepCell* bcells;           // [nl] batched cells (rows preset); nullptr = per-session cells only
    G6AttnRow* rows;                  // [nl][rows_stride] per-row bounds, written for rows 0..M-1
    const int* const* tabs;           // [S2N * nl] block table of (slot, layer), fixed addresses
    int rows_stride;                  // G6_MX
    int page_bound;                   // the pool's pages: every session's page_count
};

// grid 1, block 128
__global__ void __launch_bounds__(128) k_s2_cells(S2Cells c)
{
    __shared__ S2In sh;
    const int t = threadIdx.x;
    if (t < (int)(sizeof(S2In) / 4)) reinterpret_cast<uint32_t*>(&sh)[t] = __ldcv(c.in + t);
    __syncthreads();
    for (int k = t; k < sh.ns * c.nl; k += 128)
    {
        const int j = k / c.nl, l = k % c.nl;
        G6AttnStepCell& x = c.cells[sh.sess[j] * c.nl + l];
        const int pages = (sh.ctx[j] + 63) / 64;
        x.context = sh.ctx[j];
        x.page_count = c.page_bound;
        x.table_size = pages;
        x.reuse_rows = c.shared[l] ? sh.rows[j] : 0;
    }
    if (!c.bcells || sh.ns <= 0) return;
    const int M = sh.row0[sh.ns - 1] + sh.rows[sh.ns - 1];
    for (int k = t; k < c.nl * M; k += 128)
    {
        const int l = k / M, r = k % M;
        int j = 0;
        while (j + 1 < sh.ns && r >= sh.row0[j + 1]) ++j;
        const int pages = (sh.ctx[j] + 63) / 64;
        G6AttnRow& x = c.rows[(size_t)l * c.rows_stride + r];
        x.table = c.tabs[sh.sess[j] * c.nl + l];
        x.context = sh.ctx[j];
        x.page_count = c.page_bound;
        x.table_size = pages;
        x.ibase = 0;                         // CP index-run base: S2 views run replicated (cp_nr = 0), so unused here
        x.sess = sh.sess[j];
    }
    int ctx = 0, tab = 0;
    for (int j = 0; j < sh.ns; ++j)
    {
        ctx = max(ctx, sh.ctx[j]);
        tab = max(tab, (sh.ctx[j] + 63) / 64);
    }
    for (int l = t; l < c.nl; l += 128)
    {
        G6AttnStepCell& x = c.bcells[l];
        x.context = ctx;
        x.page_count = c.page_bound;
        x.table_size = tab;
        x.reuse_rows = c.shared[l] ? M : 0;
    }
}

}   // namespace g6rt
