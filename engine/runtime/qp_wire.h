// G6_PF_OWN=4 with G6_PF_QP_ROLL (docs/DESIGN-QO-PAIR.md section 6): the wire layout and protocol of the rolling pair
// pack in g6m4. Host-only (no CUDA, no verbs).
//
// A bulk-mode g6::Fabric (fabric.h bulk mode) per rank. Its bulk area is a ring of E entries; entry e is
// [lo half image | hi half image], HALF bytes each. A half image is one pair member's 16-head slice of a layer, contiguous:
// q_b codes [4096][2048] | q_b scales [4096][64] | o_proj codes [6144][4096] | o_proj scales [6144][128] | UK BF16 [16][192][512]
// | UV BF16 [16][256][512] | q_b svh f16 [4096] | o_proj suh f16 [4096]. The last two carry the member's slices of a dense
// trellis q_b / o_proj (kernels/dtr.cuh, whose codes are then its exact FP8 tile copy). MXFP8 layers leave them unused. Position h of the entry (0 = lo, 1 = hi) belongs to member qp_order(r, l)[h], the same order as
// the slot halves (qo_weights.cuh), so placement is identity: rank r stages ITS half at ITS position and RDMA-writes that
// range to the partner p = r ^ 1 at the same offset; p's half arrives at the other position. No extra copy on either side.
//
// Flags (two per entry) and doorbells use the same indices: ready = 2e, consumed = 2e + 1. Build n (0, 1, 2, ... counted per
// rank; both pair members run the same build sequence) uses entry e = n % E and posts / waits values n + 1:
//   GPU (stage stream): wait flags[consumed e][p] >= n + 1 - E (p is done with my previous write into entry e; skipped for
//     n < E); my half -> my position of entry e (device copies into the pinned area); doorbell[ready e] = n + 1.
//   proxy job TX(n): db ready e, seq n + 1; wait consumed e of p >= n + 1 - E; write my position to p; post ready e = n + 1
//     to p (fabric.h BulkJob::peer = p).
//   GPU: own half pack -> slot; wait flags[ready e][p] >= n + 1; partner position -> slot (2D copies for o_proj); doorbell
//     [consumed e] = n + 1; then g6_attn_qo_load.
//   proxy job CN(n): db consumed e, seq n + 1; post consumed e = n + 1 to p.
// Proxy queue order per rank: TX(0), CN(0), TX(1), CN(1), ... TX(n) needs my GPU past p's CN(n - E), which needs p's
// TX(n - E) earlier in p's queue: no cycle.
// Overwrite safety: my position of entry e is rewritten for build n only after p consumed build n - E (so my RDMA read of it
// is complete); p writes into my partner position of entry e only after I consumed build n - E (its TX waits my CN).
// On a wait timeout k_qp_wait sets the wire error word and the build still copies the (stale) half and rings consumed, on
// purpose, so the partner does not hang; the host sees the error word and exits with code 3 at the end of the sub-batch.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace g6rt
{
namespace qw
{
constexpr size_t QC = (size_t)4096 * 2048, QS = (size_t)4096 * 64, OC = (size_t)6144 * 4096, OS = (size_t)6144 * 128,
                 UK = (size_t)16 * 192 * 512 * 2, UV = (size_t)16 * 256 * 512 * 2, QV = (size_t)4096 * 2, OV = (size_t)4096 * 2;
constexpr size_t OFF_QC = 0, OFF_QS = OFF_QC + QC, OFF_OC = OFF_QS + QS, OFF_OS = OFF_OC + OC, OFF_UK = OFF_OS + OS,
                 OFF_UV = OFF_UK + UK, OFF_QV = OFF_UV + UV, OFF_OV = OFF_QV + QV, HALF = OFF_OV + OV;
static_assert(HALF == 41943040 + 16384, "half image: 40 MiB + 16 KiB per pair member and layer");
static_assert(OFF_QS % 8 == 0 && OFF_OC % 8 == 0 && OFF_OS % 8 == 0 && OFF_UK % 8 == 0 && OFF_UV % 8 == 0 && OFF_QV % 8 == 0 &&
              OFF_OV % 8 == 0, "8-B ranges");
// error words of the wire (its fabric's BulkDev::error); the bulk proxy's own codes are bulk_proxy.h 0x71-0x74
constexpr unsigned ERR_WAIT_CONSUMED = 0x65000100u;  // | entry << 4 | peer: GPU timed out waiting p's consumed flag
constexpr unsigned ERR_WAIT_READY = 0x65000200u;     // | entry << 4 | peer: GPU timed out waiting p's ready flag
constexpr unsigned ERR_TAG = 0x65000300u;            // | entry << 4 | peer: p's ready aux is not my tag of build n

inline int max_entries() { return 4; }
inline size_t entry_bytes() { return 2 * HALF; }
inline size_t bulk_bytes(int E) { return (size_t)E * entry_bytes(); }
inline int bulk_flags(int E) { return 2 * E; }
inline int entry(long n, int E) { return (int)(n % E); }
inline int f_ready(int e) { return 2 * e; }
inline int f_consumed(int e) { return 2 * e + 1; }
inline size_t pos_off(int e, int h) { return (size_t)e * entry_bytes() + (size_t)h * HALF; }
inline unsigned long long val(long n) { return (unsigned long long)n + 1; }
// consumed value p must have posted before build n may write entry e (0 = no wait)
inline unsigned long long consumed_min(long n, int E) { return n >= E ? (unsigned long long)(n + 1 - E) : 0ull; }
inline int partner(int r) { return r ^ 1; }
// ready aux tag: p posts it with ready(n) (BulkJob post_flag aux = aux_out[ready e]); I compare it to mine for the same
// build. Equal only if both members build the same model layer with the same S, E, QO table (G6_PF_QP_CTX) and weight
// form (trellis: the layer's q_b / o_proj are dense trellis): a pair whose G6_PF_QP_LAYERS / ROLL / WIRE_E / QP_CTX or
// weights differ fails closed (ERR_TAG) and reads no other layer's half.
inline unsigned long long tag(int layer, int S, int E, int qp_ctx, bool trellis = false)
{
    return 0x4c32000000000000ull | ((unsigned long long)trellis << 48) | ((unsigned long long)(layer & 0xff) << 40) |
           ((unsigned long long)(S & 0xf) << 36) | ((unsigned long long)(E & 0xf) << 32) | (unsigned long long)(unsigned)qp_ctx;
}
// g6m4 boot refusals of the rolling-pack env, before any fabric / CUDA work. "" = ok, else the die message.
// pf_qp: G6_PF_OWN=4 with --prefill; roll / wire_e: env_choice results (-1 = bad value, err already set by the caller);
// qp_ctx: getenv("G6_PF_QP_CTX"); wire_e_set: G6_PF_QP_WIRE_E present in the env; qp_pack: --qp-pack given.
inline std::string roll_env_refusal(bool pf_qp, int roll, bool wire_e_set, int wire_e, const char* qp_ctx, bool qp_pack)
{
    if (wire_e_set && roll <= 0) return "G6_PF_QP_WIRE_E is set but G6_PF_QP_ROLL is not (the wire ring exists only with ROLL)";
    if (roll <= 0) return "";
    if (!pf_qp) return "G6_PF_QP_ROLL needs G6_PF_OWN=4 and --prefill";
    if (wire_e < 1 || wire_e > max_entries()) return "G6_PF_QP_WIRE_E: 1.." + std::to_string(max_entries()) + " (unset = 1)";
    if (qp_pack) return "G6_PF_QP_ROLL: no --qp-pack (the rolling pack is built from this rank's pack + the partner's half by RDMA)";
    // without it the QO key table would be sized at the cap, and every request would run OWN=4
    if (!qp_ctx || !*qp_ctx) return "G6_PF_QP_ROLL needs G6_PF_QP_CTX (the QO key / RoPE table size, in positions)";
    char* end = nullptr;
    const long v = std::strtol(qp_ctx, &end, 10);
    if (*end || qp_ctx[0] < '1' || qp_ctx[0] > '9' || v <= 0 || v > 0x7fffffffL)
        return std::string("G6_PF_QP_CTX='") + qp_ctx + "': a positive decimal position count";
    return "";
}
}   // namespace qw
}   // namespace g6rt
