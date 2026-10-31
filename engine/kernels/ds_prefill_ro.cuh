// gr8 DeepSeek-V4.1-Flash read-once prefill MoE. Our own code.
//
// W rows (up to dsr::RO_MAX = 4096) per call. Two parts, new kernels only (the decode and chunked-path kernels are not edited):
//   planner  k_route_ro -> k_scan_ro -> k_place_ro (+ a memset of emask): the plan_ref plan (ds_plan_mt.h), slot = one distinct expert, tiles of 8 columns,
//            rank-r token of expert e at global tile tile0_e + r / 8, column r % 8. Integer atomics only (atomicOr of token bits), rank = popcount of the
//            mask words below the token, so the plan is deterministic. Contract: fields past n_slots / n_tiles, and tok_* for k >= TK, are not written.
//   experts  k_fc1_ro / k_fc2_ro: CTA = (expert, chunk group). The CTA walks the expert's tiles in groups of NTG (ds_ro.h) over weight records that live in
//            L2 after the first group; every record is owned by exactly one CTA. Each group runs the unchanged fc1_part_mt / run_mt inner loops and the
//            unchanged epilogue on its tiles, so a column's bits equal the M <= 32 kernels' (same k split, same fp32 reduce order).
//   The combine is dsh::k_combine_mt (token-major, reads ypart [tile][8][NOUT]); it works for MM = 1024 / 2048 / 4096 as is
//   (MM is only the plan type; every offset is size_t; GPU test: tests/ds_ro_w4096_test.cu).
// Plain launches only (no PDL).
#pragma once
#include "ds_router.cuh"
#include "mxfp4.cuh"
#include "ds_ro.h"
#include "ds_fc1_bstg.h"

namespace dsr
{
// grid = ceil(M / 8), block 256. Warp w of CTA b scores token 8 b + w (k_plan_pw's loop body, copied), writes the token's routes and sets the token's bit in the
// emask row of every selected expert. emask [NE][MM / 32] words, zeroed before (cudaMemsetAsync). logits [M][NE].
// KC = routes kept per token (the top-k cut, GR8_DS_MOE_TOPK; KC == TK is today's code): the first KC picks in selection order, weights renormalized over them;
// a route k in [KC, TK) is written as tok_expert -1, tok_w 0 and is in no emask row (k_place_ro skips it, k_expert_log does not count it).
template <int NE = NEXP, int TK = TOPK, int MM = 1024, int KC = TK>
__global__ void __launch_bounds__(256) k_route_ro(const float* __restrict__ logits, const float* __restrict__ bias, int M, PlanMT<TK, MM>* __restrict__ P,
                                                 unsigned* __restrict__ emask)
{
    constexpr int RW = NE / 32, MW = MM / 32;
    static_assert(NE % 32 == 0 && MM % 32 == 0 && MM <= RO_MAX && TK <= 8 && TK <= NE, "k_route_ro tiling");
    static_assert(KC >= 1 && KC <= TK, "k_route_ro: kept routes");
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, m = blockIdx.x * 8 + w;
    if (m >= M) return;
    float s[RW], key[RW];
    #pragma unroll
    for (int i = 0; i < RW; ++i)
    {
        const float sc = sqrt_softplus(logits[m * NE + l + 32 * i]);
        s[i] = sc;
        key[i] = sc + bias[l + 32 * i];
    }
    unsigned taken = 0;
    float my_k = key[0];
    int my_e = l;
    #pragma unroll
    for (int i = 1; i < RW; ++i)
        if (key[i] > my_k) { my_k = key[i]; my_e = l + 32 * i; }           // strict: the lowest id wins a tie on this lane
    float picked[KC];
    int pe[KC];
    #pragma unroll
    for (int k = 0; k < KC; ++k)
    {
        const unsigned kb = key_bits(my_k);
        const unsigned best = __reduce_max_sync(0xffffffffu, kb);
        const int be = static_cast<int>(__reduce_min_sync(0xffffffffu, kb == best ? static_cast<unsigned>(my_e) : 0xffffffffu));
        float sv = 0.f;
        if ((be & 31) == l)
        {
            #pragma unroll
            for (int i = 0; i < RW; ++i) if (i == (be >> 5)) sv = s[i];
            taken |= 1u << (be >> 5);
            my_k = -INFINITY;
            my_e = 0x7fffffff;
            #pragma unroll
            for (int i = 0; i < RW; ++i)
                if (!(taken >> i & 1) && key[i] > my_k) { my_k = key[i]; my_e = l + 32 * i; }
        }
        picked[k] = __shfl_sync(0xffffffffu, sv, be & 31);
        pe[k] = be;
    }
    float sum = 0.f;
    #pragma unroll
    for (int k = 0; k < KC; ++k) sum += picked[k];                          // selection order, fixed
    if (l < KC)
    {
        float wv = 0.f; int ev = 0;
        #pragma unroll
        for (int k = 0; k < KC; ++k) if (k == l) { wv = picked[k]; ev = pe[k]; }
        P->tok_expert[m][l] = ev;
        P->tok_w[m][l] = wv / (sum + 1e-20f) * ROUTE_SCALE;
        atomicOr(emask + ev * MW + (m >> 5), 1u << (m & 31));
    }
    else if constexpr (KC < TK)
    {
        if (l < TK) { P->tok_expert[m][l] = -1; P->tok_w[m][l] = 0.f; }      // the cut routes: k_place_ro and k_expert_log read tok_expert
    }
}

// grid = 1, block NE. Thread e = expert e: n = tokens of the expert (popcount of its mask row), slot = rank among the experts with n > 0, tile0 = tiles of the
// experts below. Writes slot_expert / slot_tile0 / slot_nt / tile_mask, the dead columns of each expert's last tile (tile_tok -1, tile_w 0), tile0_of[e]
// (0 for an expert with no token), n_slots, n_tiles. The live columns are k_place_ro's.
template <int NE = NEXP, int TK = TOPK, int MM = 1024>
__global__ void __launch_bounds__(NE) k_scan_ro(PlanMT<TK, MM>* __restrict__ P, const unsigned* __restrict__ emask, int* __restrict__ tile0_of)
{
    constexpr int RW = NE / 32, MW = MM / 32;
    static_assert(NE % 32 == 0 && NE <= 1024 && MM % 32 == 0 && MM <= RO_MAX, "k_scan_ro tiling");
    __shared__ int wcount[RW], wtiles[RW];
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    int n = 0;
    #pragma unroll 8
    for (int i = 0; i < MW; ++i) n += __popc(emask[t * MW + i]);
    const int nt = mt_ntiles(n);
    int in_s = n ? 1 : 0, in_t = nt;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)                                        // warp scans (NE % 32 == 0: every warp is full)
    {
        const int vs = __shfl_up_sync(0xffffffffu, in_s, o), vt = __shfl_up_sync(0xffffffffu, in_t, o);
        if (l >= o) { in_s += vs; in_t += vt; }
    }
    if (l == 31) { wcount[w] = in_s; wtiles[w] = in_t; }
    __syncthreads();
    int sb = 0, tb = 0;
    #pragma unroll
    for (int i = 0; i < RW; ++i) if (i < w) { sb += wcount[i]; tb += wtiles[i]; }
    int tile0 = 0;
    if (n)
    {
        const int slot = sb + in_s - 1;
        tile0 = tb + in_t - nt;
        P->slot_expert[slot] = t;
        P->slot_tile0[slot] = tile0;
        P->slot_nt[slot] = nt;
        for (int j = 0; j < nt; ++j) P->tile_mask[tile0 + j] = mt_tile_mask(n, j);
        const int last = tile0 + nt - 1;
        for (int c = ro_last_live(n); c < 8; ++c) { P->tile_tok[last][c] = -1; P->tile_w[last][c] = 0.f; }
    }
    tile0_of[t] = tile0;
    if (t == NE - 1) { P->n_slots = sb + in_s; P->n_tiles = tb + in_t; }     // lane 31 of the last warp: in_s / in_t are its warp's totals
}

// GR8_DS_RO_ORDER=1: the launch order of the expert loop (ds_ro.h ro_ord_*): grid 1, block NE. Reads slot_nt / n_slots (k_scan_ro) and writes order[pos] = slot for
// pos < n_slots, a permutation of the slots. Slot t's key is computed by thread t (O(n_slots) each), its position by counting the slots that run before it. A new
// kernel: k_scan_ro and the planner are unchanged; only launched when the flag is on.
template <int NE = NEXP, int TK = TOPK, int MM = 1024>
__global__ void __launch_bounds__(NE) k_order_ro(const PlanMT<TK, MM>* __restrict__ P, int* __restrict__ order)
{
    static_assert(NE % 32 == 0 && NE <= 1024, "k_order_ro tiling");
    __shared__ int nt_s[NE], num_s[NE], den_s[NE], cls_s[NE];
    const int t = threadIdx.x, ns = P->n_slots;
    if (t < ns) nt_s[t] = P->slot_nt[t];
    __syncthreads();
    if (t < ns) ro_ord_key(nt_s, ns, t, num_s[t], den_s[t], cls_s[t]);
    __syncthreads();
    if (t < ns)
    {
        int pos = 0;
        for (int j = 0; j < ns; ++j) pos += (j != t) && ro_ord_before(num_s[j], den_s[j], cls_s[j], j, num_s[t], den_s[t], cls_s[t], t);
        order[pos] = t;
    }
}

// grid = ceil(M * TK / 256), block 256. Route r = (m, k): rank of token m in the expert's mask -> tile, column; fills the combine's tok_slot / tok_col and the
// live columns of tile_w / tile_tok. A cut route (k >= KC) gets tok_slot -1, tok_col 0 (what k_combine skips) and places nothing.
template <int NE = NEXP, int TK = TOPK, int MM = 1024, int KC = TK>
__global__ void __launch_bounds__(256) k_place_ro(PlanMT<TK, MM>* __restrict__ P, const unsigned* __restrict__ emask, const int* __restrict__ tile0_of, int M)
{
    constexpr int MW = MM / 32;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M * TK) return;
    const int m = r / TK, k = r % TK;
    if constexpr (KC < TK)
        if (k >= KC) { P->tok_slot[m][k] = -1; P->tok_col[m][k] = 0; return; }
    const int e = P->tok_expert[m][k];
    int tile, col;
    mt_route_pos_w(tile0_of[e], emask + e * MW, m, tile, col);
    P->tok_slot[m][k] = tile;
    P->tok_col[m][k] = col;
    P->tile_w[tile][col] = P->tok_w[m][k];
    P->tile_tok[tile][col] = m;
}

// Distinct-expert counter: grid 1, block 256. Counts the routes of each expert in smem (atomicAdd of integers: exact), reduces distinct / heaviest, and
// appends one record to the device ring (ExpertLog, ds_ro.h). Reads P->tok_expert [m][k] (k < TK, m < M), written by k_plan_pw / k_route_ro. Plain launch
// after the planner; only the prefill paths launch it (rows > 32), so the shipped decode kernels are untouched.
template <int NE = NEXP, int TK = TOPK, int MM = 1024, int KC = TK>
__global__ void __launch_bounds__(256) k_expert_log(const PlanMT<TK, MM>* __restrict__ P, int M, int layer, ExpertLog* __restrict__ L)
{
    __shared__ int cnt[NE];
    __shared__ int rd[8], rx[8];
    for (int e = threadIdx.x; e < NE; e += 256) cnt[e] = 0;
    __syncthreads();
    for (int r = threadIdx.x; r < M * TK; r += 256)
    {
        if constexpr (KC < TK)
            if (r % TK >= KC) continue;                                      // a cut route is in no expert's count
        atomicAdd(&cnt[P->tok_expert[r / TK][r % TK]], 1);
    }
    __syncthreads();
    int d = 0, mx = 0;
    for (int e = threadIdx.x; e < NE; e += 256) { d += cnt[e] > 0; mx = cnt[e] > mx ? cnt[e] : mx; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { d += __shfl_xor_sync(0xffffffffu, d, o); const int v = __shfl_xor_sync(0xffffffffu, mx, o); mx = v > mx ? v : mx; }
    if ((threadIdx.x & 31) == 0) { rd[threadIdx.x >> 5] = d; rx[threadIdx.x >> 5] = mx; }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int D = 0, X = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) { D += rd[w]; X = rx[w] > X ? rx[w] : X; }
        const unsigned seq = atomicAdd(&L->cursor, 1u);
        ExpertLogRec& q = L->rec[seq % ELOG_N];
        q.seq = static_cast<int>(seq); q.layer = layer; q.rows = M; q.distinct = D; q.maxn = X; q.routes = M * KC; q.start = L->win_start;
    }
}
}   // namespace dsr

namespace mx4
{
// GR8_DS_RO_L2LA: warp 0 prefetches the byte range [p, p + bytes) into L2, one bulk prefetch (ds_ro.h ro_pf_chunk) per lane. A hint: no register, no wait, no value.
__device__ __forceinline__ void ro_l2_prefetch(const uint8_t* p, unsigned bytes, int lane)
{
    unsigned off, sz;
    if (dsr::ro_pf_chunk(bytes, lane, off, sz))
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + off), "r"(sz));
#endif
    }
}
// ---------------------------------------------------------------------------------------------------- fc1, read-once expert loop
// CTA = (expert slot, CW chunks) x KS k parts (a warp each), loops over the expert's tile groups of NTG. grid = n_slots * (I / 16 / CW) (ro_fc1_work).
// The k-split partials of a group are summed in rounds of PST tiles through one <= 32 KB smem buffer, fixed order q = 1..KS-1 as k_fc1_mt. Barriers are
// CTA-uniform (rb < ntg depends on the slot only); the part-1.. warps skip the store, not the barriers.
// L2H (GR8_DS_MOE_L2HINT >= 1): the weight loads carry the L2 evict_last hint (Grp::loadh<true>); no other load or store, and no value, changes.
template <int K1, int I, int KS, int CW, int DEPTH, int NTG, int MINB = 0, bool ORD = false, bool LAON = false, bool L2H = false>
__global__ void __launch_bounds__(KS * CW * 32, MINB) k_fc1_ro(ArgsMT A)
{
    constexpr int NCH = I / 16, NG = K1 / KG, NGP = NG / KS, UPS = NCH / CW, NW = KS * CW;
    constexpr int PMAX = 32768 / (NW * 1024), PST = KS == 1 ? 1 : (NTG < PMAX ? NTG : (PMAX > 0 ? PMAX : 1));
    static_assert(K1 % KG == 0 && NG % KS == 0 && NCH % CW == 0, "fc1 tiling");
    static_assert(NTG % PST == 0 && (KS == 1 || NW * 1024 * PST <= 32768), "fc1 reduce rounds");
    static_assert(!LAON || static_cast<size_t>(CW) * NG * 2 * GR <= static_cast<size_t>(dsr::RO_PF_LANES) * dsr::RO_PF_CHUNK, "fc1 L2 lookahead: the range must fit one chunk per lane");
    __shared__ float4 ps[KS > 1 ? PST : 1][NW * 32 * 2];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int slot, chunk, part;
    dsr::ro_fc1_work(blockIdx.x, warp, UPS, KS, CW, slot, chunk, part);
    if (A.n_slots_dev && slot >= *A.n_slots_dev) return;
    if constexpr (ORD) slot = dsr::ro_pos_slot(A.order, slot);              // GR8_DS_RO_ORDER: CTA group `slot` (position) works slot order[position]
    if constexpr (LAON)                                                      // GR8_DS_RO_L2LA: warp 0 prefetches the weights of CTA blockIdx.x + la (a hint)
        if (warp == 0)
        {
            int bp;
            if (dsr::ro_pf_target(blockIdx.x, A.la, UPS, A.n_slots_dev ? *A.n_slots_dev : A.n_slots, bp))
            {
                const int sp = dsr::ro_pos_slot(ORD ? A.order : nullptr, bp / UPS);
                ro_l2_prefetch(A.fc1 + dsr::ro_fc1_pf_rec(A.slot_expert[sp], bp % UPS, CW, NCH, NG) * GR, static_cast<unsigned>(dsr::ro_fc1_pf_recs(CW, NG) * GR), lane);
            }
        }
    const int nt = A.slot_nt[slot], tile0 = A.slot_tile0[slot], ng = dsr::ro_ngroups(nt, NTG);
    const uint8_t* s = A.fc1 + dsr::ro_fc1_rec(A.slot_expert[slot], chunk, part * NGP, 0, NCH, NG) * GR;
    #pragma unroll 1
    for (int g = 0; g < ng; ++g)
    {
        const int ntg = dsr::ro_group_nt(nt, NTG, g), t0 = dsr::ro_tile(tile0, g, NTG, 0);
        int boff[NTG];
        #pragma unroll
        for (int t = 0; t < NTG; ++t)
            boff[t] = t < ntg ? dsr::mt_xoff(A.tile_tok[(t0 + t) * 8 + (lane >> 2)], lane, K1 * 4) : 0;
        float ag[NTG][4], au[NTG][4];
        #pragma unroll
        for (int t = 0; t < NTG; ++t) { ag[t][0] = ag[t][1] = ag[t][2] = ag[t][3] = 0.f; au[t][0] = au[t][1] = au[t][2] = au[t][3] = 0.f; }
        fc1_part_mt<NGP, DEPTH, NTG, L2H>(s, A.xfrag, boff, ntg, lane, part * NGP * 4, ag, au);
        if constexpr (KS > 1)
        {
            #pragma unroll
            for (int rb = 0; rb < NTG; rb += PST)
                if (rb < ntg)
                {
                    #pragma unroll
                    for (int tt = 0; tt < PST; ++tt)
                        if (rb + tt < ntg)
                        {
                            float4* p = ps[tt] + (warp * 32 + lane) * 2;
                            p[0] = make_float4(ag[rb + tt][0], ag[rb + tt][1], ag[rb + tt][2], ag[rb + tt][3]);
                            p[1] = make_float4(au[rb + tt][0], au[rb + tt][1], au[rb + tt][2], au[rb + tt][3]);
                        }
                    __syncthreads();
                    if (part == 0)
                        #pragma unroll
                        for (int tt = 0; tt < PST; ++tt)
                            if (rb + tt < ntg)
                                for (int q = 1; q < KS; ++q)      // fixed order
                                {
                                    const float4 gq = ps[tt][((warp + q) * 32 + lane) * 2], uq = ps[tt][((warp + q) * 32 + lane) * 2 + 1];
                                    ag[rb + tt][0] += gq.x; ag[rb + tt][1] += gq.y; ag[rb + tt][2] += gq.z; ag[rb + tt][3] += gq.w;
                                    au[rb + tt][0] += uq.x; au[rb + tt][1] += uq.y; au[rb + tt][2] += uq.z; au[rb + tt][3] += uq.w;
                                }
                    __syncthreads();                              // ps is rewritten by the next round / group
                }
        }
        if (part == 0)
        {
            const int r = lane >> 2, c = lane & 3;
            #pragma unroll
            for (int t = 0; t < NTG; ++t)
                if (t < ntg)
                {
                    __half* a0 = A.act + (static_cast<size_t>(t0 + t) * 8 + 2 * c) * I + chunk * 16 + r;
                    a0[0] = __float2half_rn(swiglu_lim(ag[t][0], au[t][0], A.limit));
                    a0[I] = __float2half_rn(swiglu_lim(ag[t][1], au[t][1], A.limit));
                    a0[8] = __float2half_rn(swiglu_lim(ag[t][2], au[t][2], A.limit));
                    a0[I + 8] = __float2half_rn(swiglu_lim(ag[t][3], au[t][3], A.limit));
                }
        }
    }
}

// FC1BSTG: copy exactly the 16-byte B fragments used by two adjacent lanes into
// one of two shared record stages. A part owns K1 / KS of the reduction; a
// record is four k16 steps. The two CW warps of that part share the copy.
template <int K1, int KS, int NGP, int NTG, int NW>
__device__ __forceinline__ void fc1_stage_b_record(u32* dst, const ArgsMT& A, int t0, int ntg, int rec)
{
    constexpr int COPIES = KS * NTG * 4 * 16;
    for (int i = threadIdx.x; i < COPIES; i += NW * 32)
    {
        const dsr::Fc1BstgCopy c = dsr::fc1_bstg_copy<NTG>(i);
        if (c.tile < ntg)
        {
            const int lane0 = c.pair * 2;
            const int tok = A.tile_tok[(t0 + c.tile) * 8 + (lane0 >> 2)];
            const u32* src = A.xfrag + dsr::fc1_bstg_src_word<K1, NGP>(tok, lane0, c.part, rec, c.step);
            const unsigned sm = static_cast<unsigned>(__cvta_generic_to_shared(dst + i * 4));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sm), "l"(src) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Staged-B twin of k_fc1_ro. The original kernel remains byte-for-byte on the
// default-off path. Weight record order, MMA order, KS reduction and epilogue
// are the same; only the B load address changes from global to shared.
template <int K1, int I, int KS, int CW, int DEPTH, int NTG, int MINB = 0, bool ORD = false, bool LAON = false, bool L2H = false>
__global__ void __launch_bounds__(KS * CW * 32, MINB) k_fc1_ro_bstg(ArgsMT A)
{
    constexpr int NCH = I / 16, NG = K1 / KG, NGP = NG / KS, UPS = NCH / CW, NW = KS * CW;
    constexpr int PMAX = 32768 / (NW * 1024), PST = KS == 1 ? 1 : (NTG < PMAX ? NTG : (PMAX > 0 ? PMAX : 1));
    constexpr int STAGE_WORDS = KS * NTG * 4 * 64;
    static_assert(K1 % KG == 0 && NG % KS == 0 && NCH % CW == 0, "fc1 staged tiling");
    static_assert(NTG % PST == 0 && (KS == 1 || NW * 1024 * PST <= 32768), "fc1 staged reduce rounds");
    static_assert(!LAON || static_cast<size_t>(CW) * NG * 2 * GR <= static_cast<size_t>(dsr::RO_PF_LANES) * dsr::RO_PF_CHUNK, "fc1 staged L2 lookahead");
    extern __shared__ __align__(16) u32 sb[];   // two record stages, KS * NTG * 4 k16 slices each
    __shared__ float4 ps[KS > 1 ? PST : 1][NW * 32 * 2];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int slot, chunk, part;
    dsr::ro_fc1_work(blockIdx.x, warp, UPS, KS, CW, slot, chunk, part);
    if (A.n_slots_dev && slot >= *A.n_slots_dev) return;
    if constexpr (ORD) slot = dsr::ro_pos_slot(A.order, slot);
    if constexpr (LAON)
        if (warp == 0)
        {
            int bp;
            if (dsr::ro_pf_target(blockIdx.x, A.la, UPS, A.n_slots_dev ? *A.n_slots_dev : A.n_slots, bp))
            {
                const int sp = dsr::ro_pos_slot(ORD ? A.order : nullptr, bp / UPS);
                ro_l2_prefetch(A.fc1 + dsr::ro_fc1_pf_rec(A.slot_expert[sp], bp % UPS, CW, NCH, NG) * GR, static_cast<unsigned>(dsr::ro_fc1_pf_recs(CW, NG) * GR), lane);
            }
        }
    const int nt = A.slot_nt[slot], tile0 = A.slot_tile0[slot], ng = dsr::ro_ngroups(nt, NTG);
    const uint8_t* s = A.fc1 + dsr::ro_fc1_rec(A.slot_expert[slot], chunk, part * NGP, 0, NCH, NG) * GR;
    #pragma unroll 1
    for (int group = 0; group < ng; ++group)
    {
        const int ntg = dsr::ro_group_nt(nt, NTG, group), t0 = dsr::ro_tile(tile0, group, NTG, 0);
        int boff[NTG];
        #pragma unroll
        for (int t = 0; t < NTG; ++t) boff[t] = dsr::fc1_bstg_dst_word<NTG>(part, t, 0, lane);
        float ag[NTG][4], au[NTG][4];
        #pragma unroll
        for (int t = 0; t < NTG; ++t) { ag[t][0] = ag[t][1] = ag[t][2] = ag[t][3] = 0.f; au[t][0] = au[t][1] = au[t][2] = au[t][3] = 0.f; }
        fc1_stage_b_record<K1, KS, NGP, NTG, NW>(sb, A, t0, ntg, 0);
        if constexpr (NGP > 1) fc1_stage_b_record<K1, KS, NGP, NTG, NW>(sb + STAGE_WORDS, A, t0, ntg, 1);
        Grp next_g, next_u;
        next_g.template loadh<L2H>(s, lane); next_u.template loadh<L2H>(s + GR, lane);
        #pragma unroll 1
        for (int rec = 0; rec < NGP; ++rec)
        {
            if (rec + 1 < NGP) asm volatile("cp.async.wait_group 1;" ::: "memory");
            else asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
            const Grp cur_g = next_g, cur_u = next_u;
            if (rec + 1 < NGP)
            {
                next_g.template loadh<L2H>(s + (rec + 1) * 2 * GR, lane);
                next_u.template loadh<L2H>(s + (rec + 1) * 2 * GR + GR, lane);
            }
            const u32* xb = sb + (rec & 1) * STAGE_WORDS;
            cur_g.template run_mt<NTG>(xb, boff, 0, ntg, ag);
            cur_u.template run_mt<NTG>(xb, boff, 0, ntg, au);
            __syncthreads();
            if (rec + 2 < NGP)
                fc1_stage_b_record<K1, KS, NGP, NTG, NW>(sb + (rec & 1) * STAGE_WORDS, A, t0, ntg, rec + 2);
        }
        if constexpr (KS > 1)
        {
            #pragma unroll
            for (int rb = 0; rb < NTG; rb += PST)
                if (rb < ntg)
                {
                    #pragma unroll
                    for (int tt = 0; tt < PST; ++tt)
                        if (rb + tt < ntg)
                        {
                            float4* p = ps[tt] + (warp * 32 + lane) * 2;
                            p[0] = make_float4(ag[rb + tt][0], ag[rb + tt][1], ag[rb + tt][2], ag[rb + tt][3]);
                            p[1] = make_float4(au[rb + tt][0], au[rb + tt][1], au[rb + tt][2], au[rb + tt][3]);
                        }
                    __syncthreads();
                    if (part == 0)
                        #pragma unroll
                        for (int tt = 0; tt < PST; ++tt)
                            if (rb + tt < ntg)
                                for (int q = 1; q < KS; ++q)
                                {
                                    const float4 gq = ps[tt][((warp + q) * 32 + lane) * 2], uq = ps[tt][((warp + q) * 32 + lane) * 2 + 1];
                                    ag[rb + tt][0] += gq.x; ag[rb + tt][1] += gq.y; ag[rb + tt][2] += gq.z; ag[rb + tt][3] += gq.w;
                                    au[rb + tt][0] += uq.x; au[rb + tt][1] += uq.y; au[rb + tt][2] += uq.z; au[rb + tt][3] += uq.w;
                                }
                    __syncthreads();
                }
        }
        if (part == 0)
        {
            const int r = lane >> 2, c = lane & 3;
            #pragma unroll
            for (int t = 0; t < NTG; ++t)
                if (t < ntg)
                {
                    __half* a0 = A.act + (static_cast<size_t>(t0 + t) * 8 + 2 * c) * I + chunk * 16 + r;
                    a0[0] = __float2half_rn(swiglu_lim(ag[t][0], au[t][0], A.limit));
                    a0[I] = __float2half_rn(swiglu_lim(ag[t][1], au[t][1], A.limit));
                    a0[8] = __float2half_rn(swiglu_lim(ag[t][2], au[t][2], A.limit));
                    a0[I + 8] = __float2half_rn(swiglu_lim(ag[t][3], au[t][3], A.limit));
                }
        }
    }
}

// ---------------------------------------------------------------------------------------------------- fc2, read-once expert loop
// CTA = (expert slot, WARPS output tiles), a warp per output tile (all k), loops over the expert's tile groups of NTG; the group's act tiles are staged in
// DYNAMIC smem (NTG * (I / 16) * 64 words; launch with that many bytes, and cudaFuncSetAttribute MaxDynamicSharedMemorySize above 48 KB).
// grid = n_slots * (NOUT / 16 / WARPS) (ro_fc2_work). L2H (GR8_DS_MOE_L2HINT = 2): the weight loads carry the L2 evict_last hint, as k_fc1_ro.
// STG (GR8_DS_MOE_FC2STG = 1): the act tile of a group is read into registers in one batch (KSI * 64 / (WARPS * 32) u32 per thread, the same address of each word), then stored to smem;
// STG = false is the loop that loads and stores one word per trip. Every value and address is the same; only the order of the loads and the stores changes.
template <int I, int NOUT, int WARPS, int DEPTH, int NTG, bool ORD = false, bool LAON = false, bool L2H = false, bool STG = false>
__global__ void __launch_bounds__(WARPS * 32) k_fc2_ro(ArgsMT A)
{
    constexpr int NG = I / KG, OT = NOUT / 16, UPS = OT / WARPS, KSI = I / 16;
    static_assert(I % KG == 0 && OT % WARPS == 0, "fc2 tiling");
    static_assert(!LAON || static_cast<size_t>(WARPS) * NG * GR <= static_cast<size_t>(dsr::RO_PF_LANES) * dsr::RO_PF_CHUNK, "fc2 L2 lookahead: the range must fit one chunk per lane");
    static_assert(!STG || (KSI * 64) % (WARPS * 32) == 0, "fc2 staging batch: whole trips only");
    extern __shared__ __align__(16) u32 sb[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int slot, ot;
    dsr::ro_fc2_work(blockIdx.x, warp, UPS, WARPS, slot, ot);
    if (A.n_slots_dev && slot >= *A.n_slots_dev) return;
    if constexpr (ORD) slot = dsr::ro_pos_slot(A.order, slot);              // GR8_DS_RO_ORDER, as k_fc1_ro
    if constexpr (LAON)                                                      // GR8_DS_RO_L2LA, as k_fc1_ro
        if (warp == 0)
        {
            int bp;
            if (dsr::ro_pf_target(blockIdx.x, A.la, UPS, A.n_slots_dev ? *A.n_slots_dev : A.n_slots, bp))
            {
                const int sp = dsr::ro_pos_slot(ORD ? A.order : nullptr, bp / UPS);
                ro_l2_prefetch(A.fc2 + dsr::ro_fc2_pf_rec(A.slot_expert[sp], bp % UPS, WARPS, OT, NG) * GR, static_cast<unsigned>(dsr::ro_fc2_pf_recs(WARPS, NG) * GR), lane);
            }
        }
    const int nt = A.slot_nt[slot], tile0 = A.slot_tile0[slot], ng = dsr::ro_ngroups(nt, NTG);
    const uint8_t* s = A.fc2 + dsr::ro_fc2_rec(A.slot_expert[slot], ot, 0, OT, NG) * GR;
    const u32* xb = sb;
    int boff[NTG];
    #pragma unroll
    for (int t = 0; t < NTG; ++t) boff[t] = t * KSI * 64 + lane * 2;
    #pragma unroll 1
    for (int gr = 0; gr < ng; ++gr)
    {
        const int ntg = dsr::ro_group_nt(nt, NTG, gr), t0 = dsr::ro_tile(tile0, gr, NTG, 0);
        Grp nx[DEPTH];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (p < NG) nx[p].template loadh<L2H>(s + p * GR, lane);   // the first records are in flight while the act tiles are staged
        #pragma unroll
        for (int t = 0; t < NTG; ++t)
            if (t < ntg)
            {
                if constexpr (STG)
                {
                    constexpr int NI = KSI * 64 / (WARPS * 32);              // trips of the loop below (9 at I 576, 8 warps)
                    u32 v[NI];
                    #pragma unroll
                    for (int k = 0; k < NI; ++k)
                    {
                        const int i = threadIdx.x + k * (WARPS * 32);
                        const int ks = i >> 6, l = (i >> 1) & 31, reg = i & 1;
                        v[k] = __ldcg(reinterpret_cast<const u32*>(A.act + (static_cast<size_t>(t0 + t) * 8 + (l >> 2)) * I + ks * 16 + reg * 8 + 2 * (l & 3)));
                    }
                    #pragma unroll
                    for (int k = 0; k < NI; ++k) sb[t * KSI * 64 + threadIdx.x + k * (WARPS * 32)] = v[k];
                }
                else
                    for (int i = threadIdx.x; i < KSI * 64; i += WARPS * 32)
                    {
                        const int ks = i >> 6, l = (i >> 1) & 31, reg = i & 1;
                        sb[t * KSI * 64 + i] = __ldcg(reinterpret_cast<const u32*>(A.act + (static_cast<size_t>(t0 + t) * 8 + (l >> 2)) * I + ks * 16 + reg * 8 + 2 * (l & 3)));
                    }
            }
        __syncthreads();
        float acc[NTG][4];
        #pragma unroll
        for (int t = 0; t < NTG; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
        for (int g = 0; g < NG; g += DEPTH)
        {
            Grp cur[DEPTH];
            #pragma unroll
            for (int p = 0; p < DEPTH; ++p) cur[p] = nx[p];
            #pragma unroll
            for (int p = 0; p < DEPTH; ++p)
                if (g + DEPTH + p < NG) nx[p].template loadh<L2H>(s + (g + DEPTH + p) * GR, lane);
            #pragma unroll
            for (int p = 0; p < DEPTH; ++p)
                if (g + p < NG) cur[p].template run_mt<NTG>(xb, boff, (g + p) * 4, ntg, acc);
        }
        const int r = lane >> 2, c = lane & 3, o = ot * 16 + r;
        #pragma unroll
        for (int t = 0; t < NTG; ++t)
            if (t < ntg)
            {
                const int mask = A.tile_mask[t0 + t];
                const float* w = A.tile_w + (t0 + t) * 8;
                __half* y = A.ypart + static_cast<size_t>(t0 + t) * 8 * NOUT;
                if ((mask >> (2 * c)) & 1)
                {
                    y[(2 * c) * NOUT + o] = __float2half_rn(w[2 * c] * acc[t][0]);
                    y[(2 * c) * NOUT + o + 8] = __float2half_rn(w[2 * c] * acc[t][2]);
                }
                if ((mask >> (2 * c + 1)) & 1)
                {
                    y[(2 * c + 1) * NOUT + o] = __float2half_rn(w[2 * c + 1] * acc[t][1]);
                    y[(2 * c + 1) * NOUT + o + 8] = __float2half_rn(w[2 * c + 1] * acc[t][3]);
                }
            }
        __syncthreads();                                              // sb is restaged by the next group
    }
}
}   // namespace mx4
