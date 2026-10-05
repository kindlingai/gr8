// K1 (decode expert stream) argument block, shared by the TR3 and NVFP4 builds.
#pragma once
#include "common.cuh"
#ifdef DEC_SLOT
#include "decslot.cuh"
#endif

constexpr int SMAX = 72;       // max distinct experts per step (slot SMAX = shared expert)
constexpr int ACTW = g6nr::EB * 128;   // max intermediate columns per rank, routed experts: 384 (TP6) / 512 (TP4)
constexpr int SACTW = g6nr::DENSE_CH;  // same for the shared expert / dense MLP slot (dense layers 0..2): 2048 / 3072
constexpr int GPMAX = 16;      // max k-split partials per (block, matrix) in TR3 fc1

struct K1Args
{
    // step inputs
    const __half* x;           // [8][6144] fp16 (rows >= M unused)
    const u32* xfrag;          // [384][32][2] x in mma-B fragment layout (tokens as columns)
    int* ctl;                  // [3] plan index cells: fc1 reads [0], fc2 reads [1], fin reads [2]
    int n_plans;
    const int* slot_expert;    // [NP][SMAX]
    const int* slot_mask;      // [NP][SMAX] bit m set = token m routed to this slot
    const float* slot_w;       // [NP][SMAX][8] routing weight of token m for this slot
    const int* tok_slot;       // [NP][8][8] slot of route k of token m, -1 = none
    const float* tok_w;        // [NP][8][8]
    int n_slots, M;            // n_slots = slots the grids are launched for (>= n_slots_dev of every plan)
    const int* n_slots_dev;    // [NP] distinct routed experts of the plan (RoutePlan::n_slots); CTAs of slots beyond it exit

    // routed experts, TR3 block records (converter pack layout); an expert has 2 or 3 blocks (TP6) or 4 (TP4) of K2, K3
    // or K4 on a rank
    const long long* t_tab;    // [E][6] fc1 byte offset, fc2 byte offset, first block row (v1/v2/v3), n blocks, bits, first block
    const long long* t_kmix;   // [E] fc1 width per (block, matrix) (tr3.kmix, exlr8 v2 packs, #65), nullptr = all at bits
    const uint8_t* t_fc1;      // at tab[0]: [n blocks][2][384][8][TB], TB = 32*bits
    int t_wide;                // K3 experts of t_fc1 are in the wide order (tr3_wide.cuh, k_tr3_widen ran on it)
    const uint8_t* t_fc2;      // at tab[1]: [n blocks][384][8][TB]
    int t_wide2;               // K3 experts of t_fc2 are in the wide order (k_tr3_widen2 ran on it)
    const __half* t_v1;        // [block row][2][6144]  suh gate, suh up
    const __half* t_v2;        // [block row][3][128]   svh gate, svh up, suh down
    const __half* t_v3;        // [block row][6144]     svh down
    int B, KS1, WG, NS2;       // max 128-blocks per expert (grid and scratch stride), fc1 k-splits per CTA grid, warp groups, fc2 n-splits
    const __half *t_sg, *t_v1b, *t_v3b;   // block-scale form (#65 prototype, SH layers): layer signs [2][6144] (in, out) as
                               // fp16 +-1, v1 as [block row][2][48], v3 as [block row][48]; nullptr = t_v1 / t_v3
    int sh_e, sh_mask;         // trellis shared expert (exlr8 v2, #65; the SH kernels): its tr3.table row (256) and the
                               // token mask (all M); it runs as slot n_slots with weight 1, in place of the s_* records
    float* gpart;              // [SMAX][B][2][GPMAX][8][128]
    __half* z;                 // [SMAX][B][8][6144]

    // routed experts, NVFP4 row-tile records
    const uint8_t* n_fc1;      // [E][NCH][384][2*144]
    const uint8_t* n_fc2;      // [E][384][NCH][144]
    const float* n_alpha;      // [E][3]
    int n_nch, n_cw, n_ns2;    // 16-col chunks per rank, chunks per fc1 CTA, fc2 out splits
    int n_ks;                  // fc1 k parts per chunk (one warp each)

    // shared expert, int4 g32 row-tile records; or MXFP8 gv records (densev.cuh) when s_mx
    const uint8_t* s_fc1;      // [NCHs][192][2*288], NCHs*16 <= SACTW        s_mx: gate [NCHs][192][528]
    const uint8_t* s_fc2;      // [384][NCHs/2][288]                         s_mx: down [384][NCHs/2][528]
    const uint8_t* s_up;       //                                            s_mx: up   [NCHs][192][528]
    int s_nch, s_cw, s_ns2, s_ks;
    int s_mx;

    __half* act;               // [SMAX][8][ACTW] routed slots, then [8][SACTW] shared / dense slot
    __half* ypart;             // [SMAX+1][8][6144]
    __half* out;               // [8][6144] fp16, or BF16 bits when out_bf16 (a fabric send slot: fp32 sum, one rounding)
    int out_bf16;
    int s_direct;              // dense layer (n_slots = 0, no finalize launch): the shared/dense MLP fc2 writes out itself
    int* fctl;                 // [FCTL_N] counters of the fused fc1+fc2 launch (k1.cuh k1_tr3_fused), zero between steps
#ifdef DEC_SLOT
    dslot::Slot slot;          // send = nullptr: out as above; else out is this rank's send slot of slot.c (decslot.cuh)
#endif
};
// where finalize / the dense fc2 write the partial
#ifdef DEC_SLOT
#define K1_OUT(A) ((A).slot.send ? reinterpret_cast<__half*>(dslot::ptr((A).slot)) : (A).out)
#else
#define K1_OUT(A) ((A).out)
#endif
// [0] work item counter, [1 + slot*B + b] fc1 units done, [1 + SMAX*EB] shared fc1 CTAs done, [2 + SMAX*EB] CTAs left
constexpr int FCTL_N = 3 + SMAX * g6nr::EB;
