// F10 DS41 prefill big combine as reduce-scatter + all-gather (GR8_DS_PF_RS=1, default off). Host rules and the slot index math,
// shared by the kernels (ds41_rs.cuh), the host wiring (g6m4.cu, ds41_step.cuh) and the host test (engine/tests/test_ds41_rs.cpp).
// No CUDA, no fabric types. Design: reports/ds-in-gr8-build/f7/F7-SCOPE.md (SESS8), with one change: every rank owns Q = ceil(M / NR)
// rows (rank q: rows [q Q, min(M, (q + 1) Q)), the last ranks may own fewer or none), so every RS slice and every AG payload is Q rows
// (Q x H x 2 bytes; a peer's tail sits at the same offset for every peer and glue::wait_peers works unchanged). The rows past M in a
// slice are stale bytes that no rank reads.
//
// One big collective c < K (a layer exchange of a W-row prefill chunk, before the bounded-replay split) runs as:
//  1. publish: row m goes to my send slot of c at owner(m) * SS + local(m) * H * 2 (pub_off); the doorbell of c rings.
//  2. proxy, RS form (Fabric::Rs mask bit c): peer p gets slice p (send + p SS, Q H 2 bytes) into its recv slot (c, me) at offset 0.
//  3. k_rs_reduce: after the 3 peers' slices are in, my Q rows: sum p = 0..NR-1 in FP32 from 0.f (own slice from my send slot, the
//     others from recv (c, p)), round to BF16, store to the device sum (Model::ds41_gsum) and to my send slot of c_ag = NC + c; ring c_ag.
//  4. proxy: c_ag goes to every peer (a plain collective of Q H 2 bytes), right after c in the order (exec_order).
//  5. k_rs_gather: after the 3 peers' AG payloads are in, copy their rows into the device sum. The consumer reads it (Ds41DevBlock).
// Bit-equal to the all-reduce: the owner adds the same values in the same order (p = 0..NR-1 from 0.f) and every consumer only uses
// rn16 of the sum (ds_mhc.cuh), and rn16 of a BF16 value is that value.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __CUDACC__
#define DS41_RS_HD __host__ __device__
#else
#define DS41_RS_HD
#endif

namespace ds41 { namespace rs {

// rows each rank owns (and the rows of every RS slice / AG payload)
DS41_RS_HD inline int quarter(int rows, int nr) { return (rows + nr - 1) / nr; }
// the RS slice stride in the send slot: Q rows + the 8-byte tail, 256-aligned (fabric slot alignment)
DS41_RS_HD inline size_t slice_stride(int q, size_t row_bytes) { return ((size_t)q * row_bytes + 8 + 255) & ~size_t(255); }
// the rows rank p owns: [p Q, p Q + owned(rows, nr, p))
DS41_RS_HD inline int owned(int rows, int nr, int p) { const int q = quarter(rows, nr), n = rows - p * q; return n < 0 ? 0 : n > q ? q : n; }
// byte offset of row m in my RS send slot (publish)
DS41_RS_HD inline size_t pub_off(int m, int q, size_t ss, size_t row_bytes) { return (size_t)(m / q) * ss + (size_t)(m % q) * row_bytes; }
// the AG collective of big collective c
DS41_RS_HD inline int c_ag(int nc, int c) { return nc + c; }
// Main fabric: decode collectives + k prefill AG collectives + the optional B12 row-shard
// tail-state collective. The lossy MXFP8 y wire form changes payload bytes, not this count.
inline int main_ncoll(int nc, int k, bool rowshard = false) { return nc + k + (rowshard ? 1 : 0); }
// build_ds41's fabric check: f_ncoll = the fabric's count, model_ncoll = Model::ncoll() (2 nl + 1 + Engram 2, or the draft's DRAFT_NCOLL),
// ag = the AG collectives (main Model only), rowshard = one tail-state collective.
// nullptr = fine; else the engine dies with the text.
inline const char* fabric_ncoll_error(int f_ncoll, int model_ncoll, int ag, bool draft, bool rowshard = false)
{
    if (ag < 0 || (draft && (ag || rowshard)) || (rowshard && !ag))
        return "DS41 build: AG/state collectives only on the main Model with GR8_DS_PF_RS=1";
    if (f_ncoll != main_ncoll(model_ncoll, ag, rowshard))
        return draft ? "DS41 draft fabric ncoll must be DRAFT_NCOLL (11)" : "DS41 fabric ncoll must be 2 * layers + 1 (+ 2 with GR8_DS_ENGRAM=1, + 2 i_late with GR8_DS_PF_RS=1, + 1 with GR8_DS_MHC_ROW_SHARD=1)";
    return nullptr;
}

// Is collective c of a `rows`-row step an RS one? enabled = GR8_DS_PF_RS=1 on this Model; k = the RS collectives 0..k-1 (2 i_late).
// rows > mx: a prefill chunk, never a decode step (the decode graphs are untouched); rows >= nr: every rank owns a row of a full slice.
DS41_RS_HD inline bool on(bool enabled, int c, int k, int rows, int mx, int nr) { return enabled && c >= 0 && c < k && rows > mx && rows >= nr; }

// The proxy job of one RS chunk. In: the chunk's sizes / order (s2/ds41_prefill.h ds41_chunk_collectives; empty order = 0..n-1),
// nc = the decode collectives (the AG ones start there), k, rows, nr, row_bytes. Out: rs = the RS collectives (mask bits), ss and the
// per-peer slice bytes, sizes extended to nc + k with the AG payloads, order = the base order with c_ag right after each RS c.
struct Job {
    std::vector<int> rs;
    size_t ss = 0;
    std::vector<uint32_t> bytes;
    std::vector<uint32_t> sizes;
    std::vector<int> order;
};
// empty error = fine (the caller dies / requires with the text)
inline const char* job_error(const std::vector<uint32_t>& sizes, int nc, int k, int rows, int nr, size_t row_bytes)
{
    if (nr < 2 || rows < nr || k < 1 || nc < k) return "DS41 RS: needs nr >= 2, rows >= nr, 1 <= k <= nc";
    if ((int)sizes.size() < k || (int)sizes.size() > nc) return "DS41 RS: the chunk has fewer than k or more than nc collectives";
    for (int c = 0; c < k; ++c)
        if (sizes[(size_t)c] != (uint64_t)rows * row_bytes) return "DS41 RS: an RS collective (c < k) does not carry the chunk's rows";
    if ((uint64_t)quarter(rows, nr) * row_bytes > UINT32_MAX) return "DS41 RS: slice past 32 bits";
    return nullptr;
}
inline Job job(const std::vector<uint32_t>& sizes, const std::vector<int>& order, int nc, int k, int rows, int nr, size_t row_bytes,
               uint32_t state_bytes = 0, bool y_after_engram = false)
{
    Job j;
    const int q = quarter(rows, nr);
    j.ss = slice_stride(q, row_bytes);
    j.bytes.assign((size_t)nr, (uint32_t)((size_t)q * row_bytes));
    j.sizes = sizes;
    j.sizes.resize((size_t)(nc + k + (state_bytes ? 1 : 0)), 0);
    std::vector<int> base = order;
    if (base.empty()) for (int c = 0; c < (int)sizes.size(); ++c) base.push_back(c);
    int pending_ag = -1;
    for (size_t i=0; i<base.size(); ++i) {
        const int c=base[i];
        j.order.push_back(c);
        if (pending_ag >= 0) { j.order.push_back(pending_ag); pending_ag=-1; }
        if (c >= 0 && c < k) {
            j.rs.push_back(c); j.sizes[(size_t)c_ag(nc, c)] = j.bytes[0];
            // With row-sharded mHC, Engram between c=1/27 and the next layer's
            // prepare must run before AG(y); otherwise the proxy waits for y
            // while Engram waits for its own collective (a deadlock).
            const bool defer=y_after_engram && i+1<base.size() && c%2==1 &&
                             base[i+1]>=nc-2 && base[i+1]<nc;
            if(defer) pending_ag=c_ag(nc,c);
            else j.order.push_back(c_ag(nc,c));
            if (state_bytes && c == k-1) { j.order.push_back(nc+k); j.sizes[(size_t)(nc+k)] = state_bytes; }
        }
    }
    return j;
}

}}  // namespace ds41::rs
