// The DFlash2 selector's top-16 ids (dflash.cuh k_top_local -> k_walk), host and device: the "none" id and the guard.
// k_top_local starts each round at (logit -inf, id TOP_NONE). A round over a slice whose untaken logits are all NaN
// (no logit orders before it: cand_before is false for NaN) ends there, and the rank sends (-inf, TOP_NONE). -inf
// logits still order (by id), so only NaN leaves TOP_NONE. TOP_NONE sorts after every real id among the -inf entries,
// so the merge keeps a real entry over it. k_walk reads pred / succ rows by id, so a merged id outside the vocabulary
// is an out-of-bounds read: walk_id swaps in id 0 and the walk sets ERR_TOPK in the draft fabric's error word, which
// every host path checks after the draft step (the request fails). Every rank merges the same lists, so every rank
// sets it. For a finite row, every id is a vocabulary id and both functions return it unchanged.
#pragma once

namespace g6rt
{
namespace dfl
{
constexpr int TOP_NONE = 0x7fffffff;           // k_top_local: no orderable logit left in this rank's slice of the row
constexpr unsigned ERR_TOPK = 0x65000000u;     // | (row << 8): a merged top-16 id of the row outside the vocabulary
// the id k_top_local sends for its slice index bi (vlo = the slice's first vocabulary id)
__host__ __device__ __forceinline__ int top_id(int vlo, int bi) { return bi == TOP_NONE ? TOP_NONE : vlo + bi; }
// the id k_walk keeps for a merged id; bad |= the id was outside [0, vocab)
__host__ __device__ __forceinline__ int walk_id(int id, int vocab, bool& bad)
{
    const bool out = (unsigned)id >= (unsigned)vocab;
    bad |= out;
    return out ? 0 : id;
}
}
}
