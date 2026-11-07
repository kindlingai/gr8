// ds41_rs.h's quarter and slice_stride stay C++ for the kernels, and ds41_pf_split.h keeps TAIL_MAX for its static
// checks; engine/scheduler-rs src/plan_sizes.rs has copies for the fabric plan. They agree over every row count of a
// prefill chunk, each width and the row sizes of the three models.
#include <cstdio>
#include "../../ds41_pf_split.h"
#include "../../ds41_rs.h"

int main()
{
    int bad = 0;
    if (g6plan_ds41_tail_max() != ds41::br::TAIL_MAX) { printf("TAIL_MAX differs\n"); ++bad; }
    long n = 0;
    for (int nr = 1; nr <= 8; ++nr)
        for (int rows = 0; rows <= 8192; ++rows) {
            const int q = ds41::rs::quarter(rows, nr);
            if (g6plan_ds41_rs_quarter(rows, nr) != q) { if (!bad++) printf("quarter(%d, %d) differs\n", rows, nr); }
            for (size_t row_bytes : {size_t(2), size_t(6144 * 2), size_t(7168 * 2), size_t(5120 * 2)}) {
                ++n;
                if (g6plan_ds41_rs_slice_stride(q, row_bytes) != ds41::rs::slice_stride(q, row_bytes))
                    if (!bad++) printf("slice_stride(%d, %zu) differs\n", q, row_bytes);
            }
        }
    printf("plan twins: %ld cases, %d differ\n", n, bad);
    return bad ? 1 : 0;
}
