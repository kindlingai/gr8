// TR3 K3 fc1 in the "wide" order (RESULTS-M1 item 1c): the same bytes regrouped so that a warp reads its 8 k-steps
// of tiles with three 64-bit loads (256 B per instruction, contiguous per warp) instead of eight 32-bit tile loads
// (96 B each; every load touches two 64 B lines and half of them are shared with the neighbour tile).
//
// Pack order of one fc1 matrix (gate or up of one block): [KSH k16 steps][8 tiles j][96 B].
// Wide order:                                            [KSH/8 groups][3 loads n][8 tiles j][32 lanes][8 B].
// The 6144 B of a group (8 k-steps x 8 tiles) stay where they are, so the reorder runs in place, one CTA per group
// (k_tr3_widen, once after the pack is on the device; K4 experts are left alone). Inside a group, lane l of warp j
// owns 192 bits: its chunk of tile (k-step 8*group + p, j) at bits 24p .. 24p+23, little endian over the three u64.
// A lane's chunk of a tile = stream bits [24l, 24l+24), MSB first = the new bits of its 8 weights. The 13 stream
// bits before them (the rest of its first 16-bit state) are the low 13 bits of lane l-1's chunk of the same tile
// (lane 31 for lane 0: the stream is tail-biting), fetched with one shuffle:
//   m = (chunk[l-1] << 24) | chunk[l];   state j = (m >> (21 - 3j)) & 0xffff,  j = 0..7.
#pragma once
#include <cstring>
#include <cuda_runtime.h>
#include "common.cuh"

constexpr int TR3W_GROUP = 8 * 8 * 96;          // bytes of one group of a K3 fc1 matrix
constexpr int TR3W_MAT = KSH * 8 * 96;          // bytes of one K3 fc1 matrix

// chunk p (0..7) of a lane out of its three u64 of a group
__host__ __device__ __forceinline__ u32 tr3w_chunk(const u64 (&w)[3], int p)
{
    const int bit = 24 * p, n = bit >> 6, sh = bit & 63;
    u64 v = w[n] >> sh;
    if (sh > 40) v |= w[n + 1] << (64 - sh);
    return static_cast<u32>(v) & 0xffffffu;
}

// the lane's chunk of one tile in pack order (words are LE u32, stream bits MSB first)
__host__ __device__ __forceinline__ u32 tr3w_tile_chunk(const uint8_t* tile, int lane)
{
    const int wi = lane * 3 / 4, off = (lane * 24) & 31;          // first word, first bit inside it (0, 24, 16, 8)
    const u32* t = reinterpret_cast<const u32*>(tile);
    const u64 m = (static_cast<u64>(t[wi]) << 32) | t[wi < 23 ? wi + 1 : 23];
    return static_cast<u32>(m >> (40 - off)) & 0xffffffu;
}

__host__ __device__ __forceinline__ void tr3w_pack(u64 (&v)[3], int p, u32 chunk)
{
    const int bit = 24 * p, n = bit >> 6, sh = bit & 63;
    v[n] |= static_cast<u64>(chunk) << sh;
    if (sh > 40) v[n + 1] |= static_cast<u64>(chunk) >> (64 - sh);
}

// ---------------------------------------------------------------------------------------------- K2 and K3 (#65)
// The 2-bit width (exlr8 v2) uses the same order with BITS = 2: a lane's chunk of a tile is the new stream bits of its
// 8 weights, 8 * BITS of them (24 at K3, 16 at K2), and a group of 8 tiles is BITS u64 per lane (three at K3, two at
// K2; 256 B per load instruction either way). The 16 - BITS bits before a chunk are the low bits of lane l - 1's:
//   m = (chunk[l-1] << 8 BITS) | chunk[l];   state j = (m >> (7 BITS - BITS j)) & 0xffff.
// For BITS = 3 these are the tr3w_* functions above, constant for constant.
template <int BITS> __host__ __device__ constexpr bool trw_ok() { return BITS == 2 || BITS == 3; }   // widths that have a wide order
template <int BITS> __host__ __device__ constexpr int trw_nw() { return BITS == 2 ? 2 : 3; }        // u64 per lane per group (array bound 3)

template <int BITS>
__host__ __device__ __forceinline__ u32 trw_chunk(const u64 (&w)[3], int p)
{
    constexpr int CB = 8 * BITS;
    const int bit = CB * p, n = bit >> 6, sh = bit & 63;
    u64 v = w[n] >> sh;
    if (CB > 16 && sh > 64 - CB) v |= w[n + 1] << (64 - sh);
    return static_cast<u32>(v) & static_cast<u32>((1ull << CB) - 1);
}

template <int BITS>
__host__ __device__ __forceinline__ u32 trw_tile_chunk(const uint8_t* tile, int lane)
{
    constexpr int CB = 8 * BITS, NW = 8 * BITS;
    const int bit = lane * CB, wi = bit >> 5, off = bit & 31;
    const u32* t = reinterpret_cast<const u32*>(tile);
    const u64 m = (static_cast<u64>(t[wi]) << 32) | t[wi + 1 < NW ? wi + 1 : NW - 1];
    return static_cast<u32>(m >> (64 - off - CB)) & static_cast<u32>((1ull << CB) - 1);
}

template <int BITS>
__host__ __device__ __forceinline__ void trw_pack(u64 (&v)[3], int p, u32 chunk)
{
    constexpr int CB = 8 * BITS;
    const int bit = CB * p, n = bit >> 6, sh = bit & 63;
    v[n] |= static_cast<u64>(chunk) << sh;
    if (CB > 16 && sh > 64 - CB) v[n + 1] |= static_cast<u64>(chunk) >> (64 - sh);
}

// one fc1 group (8 k-steps x 8 tiles, 2048 * BITS bytes) of warp j in place; every thread of the CTA calls it
template <int BITS>
__device__ __forceinline__ void trw_widen_group(uint8_t* grp, int j, int lane)
{
    u64 v[3] = {0, 0, 0};
    #pragma unroll
    for (int p = 0; p < 8; ++p) trw_pack<BITS>(v, p, trw_tile_chunk<BITS>(grp + (p * 8 + j) * 32 * BITS, lane));
    __syncthreads();                                              // every read of the group before its first write
    #pragma unroll
    for (int n = 0; n < BITS; ++n) *reinterpret_cast<u64*>(grp + (n * 8 + j) * 256 + lane * 8) = v[n];
}

// one fc2 n-tile (8 k-tiles, 256 * BITS bytes) of lane `lane` in place; the warp calls it
template <int BITS>
__device__ __forceinline__ void trw_widen_ntile(uint8_t* nt, int lane)
{
    u64 v[3] = {0, 0, 0};
    #pragma unroll
    for (int p = 0; p < 8; ++p) trw_pack<BITS>(v, p, trw_tile_chunk<BITS>(nt + p * 32 * BITS, lane));
    __syncwarp();
    #pragma unroll
    for (int n = 0; n < BITS; ++n) *reinterpret_cast<u64*>(nt + n * 256 + lane * 8) = v[n];
}

// In-place pack order -> wide order for every K3 (and K2) fc1 matrix of the rank's tr3.fc1. tab = tr3.table [E][6] (fc1 byte
// offset, -, -, n blocks, bits, -); kmix = tr3.kmix [E] or nullptr (exlr8 v2 packs, #65: the width of each (block,
// matrix), 4 bits at bit 4 * (2b + m); 0 = all at bits, as without kmix). grid = E * nb * 2 * (KSH / 8), nb = max
// blocks per expert (g6nr::EB: 3 at TP6, 4 at TP4), block = 256.
// Run it exactly once per tensor: a second run would scramble the bits.
__global__ void __launch_bounds__(256) k_tr3_widen(uint8_t* fc1, const long long* tab, const long long* kmix, int nb)
{
    const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
    const int g = blockIdx.x % (KSH / 8);
    const int bm = (blockIdx.x / (KSH / 8)) % (2 * nb);           // block * 2 + matrix
    const int e = blockIdx.x / (KSH / 8 * 2 * nb);
    const long long* tr = tab + 6 * e;
    if (bm >= 2 * static_cast<int>(tr[3])) return;
    const long long km = kmix ? kmix[e] : 0;
    int k = static_cast<int>(tr[4]);
    size_t off = static_cast<size_t>(bm) * k * (TR3W_MAT / 3);
    if (km)
    {
        k = static_cast<int>((km >> (4 * bm)) & 15);
        off = 0;
        for (int i = 0; i < bm; ++i) off += static_cast<size_t>((km >> (4 * i)) & 15) * (TR3W_MAT / 3);
    }
    if (k == 3) trw_widen_group<3>(fc1 + tr[0] + off + static_cast<size_t>(g) * TR3W_GROUP, j, lane);
    else if (k == 2) trw_widen_group<2>(fc1 + tr[0] + off + static_cast<size_t>(g) * 4096, j, lane);
}

inline void launch_tr3_widen(void* fc1, const long long* tab, const long long* kmix, int n_experts, int nb, cudaStream_t st)
{
    k_tr3_widen<<<n_experts * nb * 2 * (KSH / 8), 256, 0, st>>>(static_cast<uint8_t*>(fc1), tab, kmix, nb);
}

// fc2 (down) of a K3 (or K2) expert, same idea. Pack order of one block: [KSH n-tiles][8 k-tiles p][96 B]; a warp of the
// decode fc2 reads the 8 k-tiles of one n-tile (768 B, contiguous) with eight 32-bit tile loads.
// Wide order: [KSH n-tiles][3 loads n][32 lanes][8 B]: the 768 B of an n-tile stay where they are, the lane's chunk of
// k-tile p sits at bits 24p .. 24p+23 of its three u64. grid = E * nb * (KSH / 8), block = 256 (warp j = n-tile
// 8 * group + j). tab [E][6]: fc2 byte offset at [1]. Run it exactly once per tensor.
constexpr int TR3W_NT = 8 * 96;                 // bytes of one n-tile of a K3 fc2 block
__global__ void __launch_bounds__(256) k_tr3_widen2(uint8_t* fc2, const long long* tab, int nb)
{
    const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
    const int g = blockIdx.x % (KSH / 8);
    const int b = (blockIdx.x / (KSH / 8)) % nb;
    const long long* tr = tab + 6 * (blockIdx.x / (KSH / 8 * nb));
    const int k = static_cast<int>(tr[4]);
    if ((k != 3 && k != 2) || b >= static_cast<int>(tr[3])) return;
    uint8_t* grp = fc2 + tr[1] + static_cast<size_t>(b) * k * (TR3W_MAT / 3) + static_cast<size_t>(g * 8 + j) * 8 * 32 * k;
    if (k == 3) trw_widen_ntile<3>(grp, lane);
    else trw_widen_ntile<2>(grp, lane);
}

inline void launch_tr3_widen2(void* fc2, const long long* tab, int n_experts, int nb, cudaStream_t st)
{
    k_tr3_widen2<<<n_experts * nb * (KSH / 8), 256, 0, st>>>(static_cast<uint8_t*>(fc2), tab, nb);
}

// host reference of the fc2 reorder, one K3 block (TR3W_MAT bytes), in place
inline void tr3w_widen2_host(uint8_t* mat)
{
    uint8_t tmp[TR3W_NT];
    for (int nt = 0; nt < KSH; ++nt)
    {
        uint8_t* grp = mat + static_cast<size_t>(nt) * TR3W_NT;
        for (int lane = 0; lane < 32; ++lane)
        {
            u64 v[3] = {0, 0, 0};
            for (int p = 0; p < 8; ++p) tr3w_pack(v, p, tr3w_tile_chunk(grp + p * 96, lane));
            for (int n = 0; n < 3; ++n) memcpy(tmp + n * 256 + lane * 8, &v[n], 8);
        }
        memcpy(grp, tmp, TR3W_NT);
    }
}

// host reference of the same reorder, one K3 matrix (TR3W_MAT bytes), in place
inline void tr3w_widen_host(uint8_t* mat)
{
    uint8_t tmp[TR3W_GROUP];
    for (int g = 0; g < KSH / 8; ++g)
    {
        uint8_t* grp = mat + static_cast<size_t>(g) * TR3W_GROUP;
        for (int j = 0; j < 8; ++j)
            for (int lane = 0; lane < 32; ++lane)
            {
                u64 v[3] = {0, 0, 0};
                for (int p = 0; p < 8; ++p) tr3w_pack(v, p, tr3w_tile_chunk(grp + (p * 8 + j) * 96, lane));
                for (int n = 0; n < 3; ++n) memcpy(tmp + (n * 8 + j) * 256 + lane * 8, &v[n], 8);
            }
        memcpy(grp, tmp, TR3W_GROUP);
    }
}
