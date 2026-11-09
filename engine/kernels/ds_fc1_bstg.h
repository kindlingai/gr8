// FC1 B-ring word addresses, shared by the CUDA copy and the host reference test.
// One cp.async copies four u32: both register words of two adjacent MMA lanes.
#pragma once
#include "ds_plan_mt.h"

namespace dsr
{
struct Fc1BstgCopy { int part, tile, step, pair; };
template <int NTG>
DSM_HD Fc1BstgCopy fc1_bstg_copy(int i)
{
    const int part = i / (NTG * 4 * 16), rem = i % (NTG * 4 * 16);
    return {part, rem / (4 * 16), (rem / 16) & 3, rem & 15};
}
template <int K1, int NGP>
DSM_HD int fc1_bstg_src_word(int tok, int lane0, int part, int rec, int step)
{
    return mt_xoff(tok, lane0, K1 * 4) + (part * NGP * 4 + rec * 4 + step) * 64;
}
template <int NTG>
DSM_HD int fc1_bstg_dst_word(int part, int tile, int step, int lane)
{
    return ((part * NTG + tile) * 4 + step) * 64 + lane * 2;
}
} // namespace dsr
