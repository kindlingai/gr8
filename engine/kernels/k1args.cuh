// K1 (decode expert stream) argument block, shared by the TR3 and NVFP4 builds.
#pragma once
#include "common.cuh"
#ifdef DEC_SLOT
#include "decslot.cuh"
#endif

constexpr int SMAX = g6mx::SLOTS;   // max routed slots per step (72 at 8 rows, 288 at 32); slot SMAX = shared expert
constexpr int ACTW = g6nr::EB * 128;   // max intermediate columns per rank, routed experts: 384 (TP6) / 512 (TP4)
constexpr int SACTW = g6nr::DENSE_CH;  // same for the shared expert / dense MLP slot (dense layers 0..2): 2048 / 3072
constexpr int GPMAX = 16;      // max k-split partials per (block, matrix) in TR3 fc1

struct K1Args
{
    // step inputs
    const __half* x;           // [M][6144] fp16 (rows >= M unused), M <= 8 (G6_MX under DEC_S2)
    const u32* xfrag;          // [ceil(M/8)][384][32][2] x in mma-B fragment layout (tokens as columns), one table per 8 tokens
    int* ctl;                  // [3] plan index cells: fc1 reads [0], fc2 reads [1], fin reads [2]
    int n_plans;
    const int* slot_expert;    // [NP][SMAX]
    const int* slot_mask;      // [NP][SMAX] bit c set = column c of the slot's n=8 tile is used
    const float* slot_w;       // [NP][SMAX][8] routing weight of the token in column c of the slot (0 if unused)
    const int* slot_tok;       // [NP][SMAX][8] token in column c of the slot, -1 = unused (M <= 8: column c = token c)
    const int* tok_slot;       // [NP][G6_MX][8] slot of route k of token m, -1 = none
    const int* tok_col;        // [NP][G6_MX][8] column of token m in that slot
    const float* tok_w;        // [NP][G6_MX][8]
    int n_slots, M;            // n_slots = slots the grids are launched for (>= n_slots_dev of every plan)
    int smax;                  // routed-slot capacity of the scratch (act / z / gpart / ypart): SMAX at 8 rows, 8 * rows above (<= SMAX)
    int ysh;                   // rows of ypart ahead of the shared expert's rows: SMAX * 8 at 8 rows (as before), 0 above (TR3 keeps no routed ypart)
    int s_ntp;                 // shared fc2: row tiles of 8 per pass over the weight stream (smem holds s_ntp B tables), 1 .. ceil(M/8)
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
    int xc;                    // TR3_XCOMP builds: fc1 x table compact (separate fc1 only, M <= 3; bind.cuh make_k1)
    int B, KS1, WG, NS2;       // max 128-blocks per expert (grid and scratch stride), fc1 k-splits per CTA grid, warp groups, fc2 n-splits
    const __half *t_sg, *t_v1b, *t_v3b;   // block-scale form (exlr8 v2 layers, the SH kernels, #65): layer signs [2][6144]
                               // (in, out) as fp16 +-1, tr3.v1 as [block row][2][48] scales, tr3.v3 as [block row][48]
    int sh_e;                  // trellis shared expert (exlr8 v2, #65; the SH kernels): its tr3.table row (256). It runs
                               // with weight 1 in place of the s_* records, as scratch slots n_slots .. n_slots + ceil(M/8) - 1
                               // (one per 8 rows, k1.cuh k1_slot)
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

    __half* act;               // [smax][8][ACTW] routed slots, then [8 ceil(M/8)][SACTW] shared / dense slot
    __half* ypart;             // [ysh/8 + ...] ysh rows, then the shared expert's [MX][6144] rows (M <= 8: [SMAX+1][8][6144])
    __half* out;               // [M][6144] fp16, or BF16 bits when out_bf16 (a fabric send slot: fp32 sum, one rounding)
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
