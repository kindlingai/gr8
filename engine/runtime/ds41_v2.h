// DS41 sequence-parallel prefill v2: shared host contract and CUDA entry points.
// This header is CPU-testable; implementations live in ds41_v2.cuh. No stock
// fabric slots, sequence counters, or decode buffers belong to this interface.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "../comms/fabric.h"
#include "ds41_v2_width.h"
#include "ds41_v2_attn_pipe.h"
#include "ds41_v2_l20_tail.h"
#include "ds41_v2_send_ffn_opt.h"
#include "ds41_v2_bf16_tc.h"

namespace ds41::v2 {
constexpr int MAX_ROWS = 8192;
constexpr int MAX_ROWS_W16K = 16384;
inline int max_rows(bool w16k = false) { return w16k ? MAX_ROWS_W16K : MAX_ROWS; }
constexpr int FIRST_LATE = 21;
constexpr int COLUMN_GROUPS = 4;
constexpr int MAX_SEGMENTS = 8;
constexpr int COMM_BLOCKS = 8;
constexpr int COMM_THREADS = 256;

struct Geometry {
    int M = 0, Q = 0, row0 = 0, owned = 0, rank = 0, nr = 0;
    int padded() const { return Q * nr; }
};
inline Geometry geometry(int M, int nr, int rank, int max_m = MAX_ROWS_W16K) {
    if (M < 1 || M > max_m || nr < 1 || rank < 0 || rank >= nr) return {};
    const int Q = (M + nr - 1) / nr;
    return {M, Q, rank * Q, std::max(0, std::min(Q, M - rank * Q)), rank, nr};
}
inline bool boot_ok(int width, int tail, int nr, int lanes, int late_layer, bool w16k = false) {
    return nr == 4 && width >= 256 && width <= max_rows(w16k) && width % 256 == 0 &&
           tail == 128 && lanes == 1 && late_layer == FIRST_LATE;
}
// GR8_DS_PF_V2=1 asks for v2; the boot runs it only where these hold. Anything else is not an error:
// layers 0..20 then run the old range-A path (the stock ops), and the boot logs one line with the reason.
// Knob-time inputs (the fabric plan sees them); the release's experts are checked after the weights bind.
struct BootIn {
    bool w16k = false;
    bool serve = false;       // g6m4 --serve (offline g6m4 and m4sim are not)
    bool prefill = false;     // W-row prefill admitted (GR8_DS_PREFILL_MIN > 0)
    int width = 0, tail = 0, nr = 0, lanes = 1, late_layer = FIRST_LATE;
    bool sh_tc = false;       // GR8_DS_SH_TC=1: v2's FFN reads the tensor-core shared weights
    bool row_shard = false;   // GR8_DS_MHC_ROW_SHARD=1, the 1M-context setup
};
// nullptr = v2 can run. Otherwise the failed condition, for the boot's one log line.
inline const char* fallback_reason(const BootIn& b) {
    if (!b.serve) return "not a serve run (offline or simulator)";
    if (!b.prefill) return "no W-row prefill (GR8_DS_PREFILL_MIN=0)";
    if (b.nr != 4) return "TP is not 4";
    if (b.width < 256 || b.width > max_rows(b.w16k) || b.width % 256)
        return b.w16k ? "GR8_DS_PREFILL_W is not 256..16384 in steps of 256" : "GR8_DS_PREFILL_W is not 256..8192 in steps of 256";
    if (b.tail != 128) return "GR8_DS_PF_TAIL is not 128";
    if (b.lanes != 1) return "GR8_DS_PF_LANES is not 1";
    if (b.late_layer != FIRST_LATE) return "GR8_DS_PF_L20_TAIL=1 moves the late range to layer 20";
    if (!b.sh_tc) return "GR8_DS_SH_TC=0 (exact output)";
    if (b.row_shard) return "GR8_DS_MHC_ROW_SHARD=1 (the 1M-context setup)";
    return nullptr;
}
// The rows a W-row chunk may take on the old path (the read-once MoE plan, ds41moe::ROMAX). A fallback from a
// larger v2 width runs chunks of this many rows; the buffers sized for the larger width stay.
constexpr int OLD_MAX_ROWS = 4096;
inline int fallback_width(int width) { return width > OLD_MAX_ROWS ? OLD_MAX_ROWS : width; }
inline size_t aligned(size_t n, size_t a = 16) { return (n + a - 1) / a * a; }

// Physical ids group similar capacities into the fabric's three slot runs.
// One layer = one fabric epoch, NOT one slot per layer. This stays below the
// fabric's 128-bit masks while allowing any number of chunks/layers.
enum class Coll : int {
    AttnIn = 0, Index = 1, Candidates = 2, FfnIn = 3, Tail = 4, Boot = 5,
    Engram = 6, AttnOut = 7,
    Ffn0 = 8, Ffn1 = 9, Ffn2 = 10, Ffn3 = 11, Count = 12
};
inline Coll ffn_group(int group) { return Coll(int(Coll::Ffn0) + group); }
inline std::vector<int> layer_order(bool engram, bool index, bool tail, bool candidates = false) {
    std::vector<int> o;
    if (engram) o.push_back(int(Coll::Engram));
    o.push_back(int(Coll::AttnIn));
    if (index) o.push_back(int(Coll::Index));
    if (candidates) o.push_back(int(Coll::Candidates));
    o.push_back(int(Coll::AttnOut));
    o.push_back(int(Coll::FfnIn));
    for (int c = 0; c < COLUMN_GROUPS; ++c) o.push_back(int(ffn_group(c)));
    if (tail) o.push_back(int(Coll::Tail));
    return o;
}

inline std::vector<int> l20_tail_order(bool tail) {
    return tail ? std::vector<int>{int(Coll::AttnIn), int(Coll::Tail)}
                : std::vector<int>{int(Coll::AttnIn)};
}

// AG: src [owned][row_bytes] -> dst [M][row_bytes]. No reader consumes padding.
// A2A: src [M][row_bytes] -> dst [nr][Q][row_bytes], source-rank major.
// The caller allocates nr*Q rows for A2A, Q for owned outputs. Segment source
// and destination arrays are separate, contiguous and device resident.
// Each segment's wire extent is align16(Q * row_bytes); all segments travel
// in ONE collective. Gather/scatter strips those wire-only alignment bytes.
struct Segment { const void* src = nullptr; void* dst = nullptr; size_t row_bytes = 0; };
struct WireLayout {
    size_t offsets[MAX_SEGMENTS] = {};
    size_t payload = 0, stride = 0;
    int count = 0;
};
inline WireLayout wire_layout(int Q, const Segment* segments, int count) {
    WireLayout w;
    if (Q < 1 || count < 1 || count > MAX_SEGMENTS) return w;
    for (int s = 0; s < count; ++s) {
        if (!segments[s].row_bytes || segments[s].row_bytes > UINT32_MAX / size_t(Q)) return {};
        w.offsets[s] = w.payload;
        w.payload += aligned(size_t(Q) * segments[s].row_bytes);
        if (w.payload > UINT32_MAX) return {};
    }
    w.count = count;
    w.stride = aligned(w.payload + 8, 256);  // per-destination RS/A2A slice + tail
    return w;
}
// Which part of the final tail belongs to each owner, including ragged chunks
// where the last 128 rows cross rank boundaries. Never assume rank 3 owns it.
struct TailSlice { int local = 0, count = 0, destination = 0; };
inline TailSlice tail_slice(const Geometry& g, int tail = 128) {
    const int first = std::max(0, g.M - tail), lo = std::max(first, g.row0);
    const int hi = std::min(g.M, g.row0 + g.owned);
    return hi > lo ? TailSlice{lo - g.row0, hi - lo, lo - first} : TailSlice{};
}

// A separate uniform-layout fabric retains the existing ring/doorbell/proxy
// implementation. All three runs retain two epoch parities. Capacity inputs
// are maxima across layers, not a requirement that their actual wires match.
inline g6::FabricConfig fabric_config(g6::FabricConfig base,int width,int hidden,
                                      size_t tail_row_bytes,size_t engram_row_bytes) {
    const int Q=(width+3)/4;
    base.ncoll=int(Coll::Count); base.compact=true; base.pool=0;
    base.bulk_bytes=0; base.bulk_flags=0; base.vcap.clear(); base.vdst_k.clear();
    base.run1=int(Coll::Engram); base.run2=int(Coll::Ffn0); base.run3=0;
    base.slot_bytes=std::max({size_t(Q)*8192, size_t(128)*tail_row_bytes, size_t(1)<<20});
    const size_t a2a=std::max(size_t(Q)*2112,size_t(Q)*engram_row_bytes);
    base.slot_bytes1=4*aligned(a2a+8,256)-8;
    base.slot_bytes2=4*aligned(size_t(Q)*(hidden/COLUMN_GROUPS)*2+8,256)-8;
    base.slot_bytes3=0; base.sge2=false;
    return base;
}

// Each enqueue is one collective job with its own Rs::ss/bytes (the old fabric
// Rs descriptor has one stride for a job). The proxy consumes jobs in enqueue
// order. begin_layer + order cursor enforce identical dependency order on all
// ranks; skip optional collectives by the same layer metadata, never rank.
struct Order {
    std::vector<int> expected;
    size_t next = 0;
    bool take(Coll c) {
        if (next == expected.size() || expected[next] != int(c)) return false;
        ++next; return true;
    }
    bool complete() const { return next == expected.size(); }
};
} // namespace ds41::v2

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_bf16.h>
namespace ds41::v2 {
// Per-model ownership: conductor owns streams/events/fabric and own mHC state;
// A/B own their numerical scratch, referenced through attn/ffn. No two-lane
// aliases. st is borrowed; cs and the events are created/destroyed here.
struct Timing;
struct State {
    Timing* timing = nullptr;
    CommWidths widths;
    AttnPipe attn_pipe;
    L20Tail l20_tail;
    SendFfnOpt send_ffn_opt;
    Bf16Tc bf16_tc;
    cudaStream_t st = nullptr, cs = nullptr;
    cudaEvent_t produced[int(Coll::Count)] = {}, received[int(Coll::Count)] = {};
    g6::Fabric* fabric = nullptr;
    g6::FabricDev wire = {};
    unsigned* tickets = nullptr;
    Geometry g;
    uint64_t epoch = 0;
    Order order;
    bool live = false;
    void* attn = nullptr;
    void* ffn = nullptr;
    // Type-erased trunk_t; the conductor supplies its own-row mHC allocations.
    void* streams = nullptr;
    float *mix = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
    __nv_bfloat16 *y = nullptr, *attn_out = nullptr, *ffn_out = nullptr;
};

// init does not create/bootstrap the fabric. The boot owner supplies a live
// dedicated Fabric (with config from fabric_config), starts its proxy, then
// binds it. Call release only after draining the streams and proxy.
void init(State&, g6::Fabric&, cudaStream_t compute);
void release(State&);
void begin_layer(State&, Geometry, bool engram, bool index, bool tail = false, bool candidates = false);
void begin_l20_tail(State&, Geometry, bool tail);
void end_layer(State&);
void fork(State&, Coll);               // st producer -> cs
void record_received(State&, Coll);    // after cs remote work, before local join
void join(State&, Coll);               // cs -> st at first remote consumer

// enqueue-only APIs; no host synchronization. AG normally uses st. For A2A
// and RS the caller does fork(), operation(..., cs), record_received(), then
// keeps independent compute on st before join(). RS accumulates bf16 partials
// in FP32 and stores bf16, matching the current mHC input representation.
void all_gather(State&, Coll, const Segment*, int count, cudaStream_t);
void all_to_all(State&, Coll, const Segment*, int count, cudaStream_t);
void attn_out_a2a_opt(State&, const uint8_t* oaq, const uint8_t* oas, int K, void* output_recv);
void all_gather_ffn_in(State&, const Segment*, int count, uint8_t* record_q, int H, cudaStream_t);
void reduce_scatter(State&, int group, const __nv_bfloat16* all_rows,
                    __nv_bfloat16* own_rows, int columns, cudaStream_t);

// Boot-only bounded transfers: equal-size shards -> rank-major full weight.
// Caller restores logical output-column/scale tiling (not necessarily a flat
// concatenation). No per-request weight exchange. Call on every rank in the
// same tensor order; source/destination are device pointers.
void gather_weight(State&, const void* shard, size_t shard_bytes, void* rank_major);
} // namespace ds41::v2
#endif
