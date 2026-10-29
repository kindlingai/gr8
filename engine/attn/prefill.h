#pragma once
#include "bind.h"
// Opt-in state: private converted weights; private or shared scratch/cuBLAS handle.
// The original G6Attn/G6AttnBinding ABIs and all decode entrypoints are unchanged.
struct G6AttnPrefill;
struct G6AttnTrace {
    float *q, *scores, *probs, *context, *index_scores;
    // Optional device destinations: [M,H,512], [M,H,2048], [M,H,2048],
    // [M,H,512], [M,T]. Scores include the 1/16 scale. Null fields are skipped.
};
// *out must be null. Create outside capture, with a successfully created binding.
// Read once at create: GR8_PF_TILE=64/128/256 (default256), GR8_PF_DENSE>=0
// (default8192; zero always sparse), GR8_PF_EVENTS=0|1 (default0).
// Missing/empty variables use defaults;
// malformed/out-of-range values return cudaErrorInvalidValue. Dense limits are
// logical context thresholds; allocation rounds them up to32 within capacity.
// Config stays frozen for all calls/graphs on this state; changing env later has
// no effect. Use matching settings when creating a shared root and borrowers.
// Pack/binding lifetime exceeds state lifetime; release state before binding.
extern "C" int g6a_bind_prefill_create(G6AttnPrefill** out,G6AttnBinding*,void* stream);
// A null storage_owner creates one H11/full-index scratch allocation sized for
// b->capacity/max_rows (even when b is H10 or a reuse layer). A non-null owner
// borrows and retains its scratch and cuBLAS handle; incompatible device/shape/size is rejected.
// UK/UV remain private to each state. Serialize ALL uses of a shared workspace
// on one stream, or order streams externally; overlapping calls are unsafe.
// Complete outstanding GPU uses before release. States may be released in any
// order: borrowers retain scratch after the original owner is destroyed.
extern "C" int g6a_bind_prefill_create_shared(G6AttnPrefill** out,G6AttnBinding*,void* stream,G6AttnPrefill* storage_owner);
extern "C" int g6a_bind_prefill_run(G6AttnPrefill*,G6AttnBinding*,const G6AttnStep*,void* stream);
extern "C" int g6a_bind_prefill_release(G6AttnPrefill*);
// Accounting: private weight bytes plus scratch only for the allocating root.
// Sum once over the states at creation to count shared storage once. Not peak
// memory: excludes cuBLAS private allocations and binding/cache storage. After
// destroying the root, its accounting charge disappears but retained scratch
// remains live until the final borrower is released.
extern "C" size_t g6a_bind_prefill_bytes(const G6AttnPrefill*);
// Query effective tile=min(configured tile,max_rows) and frozen dense threshold.
extern "C" int g6a_bind_prefill_get_config(const G6AttnPrefill*,int* query_tile,int* dense_limit);
// Diagnostic runners: same computation with optional FP32 stage copies.
extern "C" int g6a_bind_prefill_trace(G6AttnPrefill*,G6AttnBinding*,const G6AttnStep*,const G6AttnTrace*,void* stream);
extern "C" int g6a_bind_baseline_trace(G6AttnBinding*,const G6AttnStep*,const G6AttnTrace*,void* stream);

// Optional diagnostic event profile, independent per state (also with shared
// scratch). GR8_PF_EVENTS is frozen at create; OFF allocates/records/queries no
// profiling events. ON allocates 1+5*ceil(max_rows/query_tile) events once.
// No profiling events are recorded during stream capture; graph replay is NOT
// profiled. Preserve the existing external stream ordering/lifetime contract.
enum G6AttnPrefillProfileStatus {
    G6_PF_PROFILE_OFF=0, G6_PF_PROFILE_EMPTY=1, G6_PF_PROFILE_READY=2,
    G6_PF_PROFILE_CONSUMED=3, G6_PF_PROFILE_CAPTURE_SKIPPED=4,
    G6_PF_PROFILE_FAILED=5
};
struct G6AttnPrefillProfile {
    uint64_t sequence; // increments on every run/trace attempt on this state
    int status, rows, context, heads, query_tile, tiles, full_index, traced;
    double index_ms, mla_ms, glue_ms, other_ms, total_ms;
};
// Returns a CUDA status; timing fields are valid ONLY with status READY.
// Read once BEFORE the next run/trace on this state, including alternating
// lanes, and before an external graph replay (no host hook runs on replay).
// Every attempt invalidates the previous record; this is not a queue.
// READY explicitly synchronizes the last event, then consumes the record.
// OFF/EMPTY/CONSUMED/CAPTURE_SKIPPED/FAILED return metadata and zero times,
// without waiting. A failed read consumes the record as FAILED.
// Total spans pre-append to last tile end. Index=prep/score/select (full only);
// MLA=QK/gather/softmax/PV/map; glue=append/expand/UK/UV/reuse-ID copy;
// other=inter-tile gaps. Groups partition that span (event rounding applies).
// Trace-copy overhead is included when traced=1. Event timing perturbs the
// workload: use OFF whole-call measurements for performance acceptance.
extern "C" int g6a_bind_prefill_profile_read(G6AttnPrefill*,G6AttnPrefillProfile* out);

// Row-owner phase1b. All pointers are device addresses; calls enqueue on stream
// without allocating or synchronizing. Binding/state must be successfully
// created and unmodified. Same shared-scratch serialization/lifetime as run.
// rows==0 is a no-op (other step fields and output pointers may be null/zero).
// Otherwise 0<rows<=max_rows, rows<=context<=capacity, context>0, positions
// must be in [0,context); appended positions must be unique. Pointer lengths,
// page validity, and external stream ordering remain the caller's responsibility.
//
// Append ONLY index keys for all rows: consumes rows/context/positions/index_k,
// ignores all other step fields. Requires full-index binding. Uses exactly the
// normal normalize/RoPE/FP8 append; does not read/write latent KV, IDs or output.
// Clears the binding error word before append; errors are asynchronous as usual.
extern "C" int g6a_bind_index_append(G6AttnBinding*,const G6AttnStep* full_step,void* stream);
// Score/select ONLY owner rows from the already appended index-key cache.
// For contiguous ownership [r0,r1), caller offsets positions/index_q/index_weight
// to r0, sets rows=r1-r0, context>=max(owner positions)+1. Other activation fields
// are unused. out_ids is required int32[rows,2048]; optional out_scores is
// FP32[rows,context]. If context is below the full call's context, omitted future
// score columns reconstruct as -INF. No implicit gather, offset or row map.
// Reuse mode instead copies reuse_indices through optional reuse_row_map using
// existing semantics; allowed on non-full layers, leaves out_scores untouched.
// reuse_rows bounds the source; with a map, source and out_ids must not alias;
// without a map exact in-place reuse is allowed. No partial source/output alias.
// Output buffers must not overlap caches, activations or scratch. Binding-owned
// IDs may be supplied explicitly. No cache/MLA/output writes, no binding args
// mutation. Preserves prior error bits. Read them before a subsequent regular
// prefill_run, which clears them. Every attempt invalidates any old whole-call
// profile (EMPTY when enabled, OFF otherwise); no subset timing is reported.
// Intended sequence: append(all rows) -> score(owner) -> caller all-gather IDs ->
// normal full-row prefill_run with reuse_indices (appends latent KV exactly once).
extern "C" int g6a_bind_prefill_index_rows(G6AttnPrefill*,G6AttnBinding*,
    const G6AttnStep* owner_step,int* out_ids,float* out_scores,void* stream);

#if G6_MODEL_FLASH
// Optional TP2 Flash selection owner. Both callbacks enqueue work on the supplied
// stream; no allocations/synchronization. publish packs the owned half after
// selection; receive fills the peer half before ANY MLA selected-ID consumer.
// Buffers and callback state must survive stream completion. Caller supplies
// lane-private transport and unchanged per-rank projections/cache inputs.
struct G6FlashPoolOwner {
    int rank;
    void* opaque;
    int (*publish)(void*, const int* indices, const int* positions, int rows, void* stream);
    int (*receive)(void*, int* indices, const int* positions, int rows, void* stream);
};
extern "C" int g6a_bind_prefill_run_owned(G6AttnPrefill*, G6AttnBinding*,
    const G6AttnStep*, void* stream, const G6FlashPoolOwner*);
#endif
