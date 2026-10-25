#pragma once
#include "attn.h"
#include "replay.h"

// Every weight below is contiguous BF16 device storage, except UK / UV when uks / uvs are set (#50: E4M3 bytes + f32
// scales). Shape is copied from the manifest, NOT inferred from a pointer. No dependency on the pack loader.
struct G6AttnTensor {
    const void* data;
    int ndim;
    int64_t shape[3];
};
struct G6AttnPack {
    G6AttnTensor uk;          // Lnnn.kv_b.uk [H,192,512]; transposed once by binder
    G6AttnTensor uv;          // Lnnn.kv_b.uv [H,256,512]; borrowed unchanged
    G6AttnTensor kv_norm;     // Lnnn.kv_a_norm [512]
    G6AttnTensor index_norm;  // Lnnn.idx.k_norm.w [128], only full-index layers
    G6AttnTensor index_bias;  // Lnnn.idx.k_norm.b [128], only full-index layers
#if G6_MODEL_FLASH
    G6AttnTensor index_ape;   // Lnnn.self_attn.indexer.index_kpool_compress_ape [4,128]; uk is [H,256,512], not transposed
#endif
    // #50 (G6_KVB=fp8): uk / uv are u8 E4M3 [H,192,512] / [H,256,512] and these their f32 scales, the release's 128x128
    // block scale of every weight row and 128-column block. Both null: uk / uv are BF16.
    G6AttnTensor uks;         // Lnnn.kv_b.uk.scale [H,192,4]; transposed once by binder to [H,4,192]
    G6AttnTensor uvs;         // Lnnn.kv_b.uv.scale [H,256,4]; borrowed unchanged
};
struct G6AttnStep {
    int rows, context;        // current logical context, NOT allocated capacity
    const void *q_b, *kv_a;   // FP16 [rows,H,256], [rows,576]
    const void *index_q, *index_k, *index_weight; // FP16, full-index steps only
    const int *positions;    // device int32 [rows], absolute positions
    const int *reuse_indices, *reuse_row_map; // device [reuse_rows,2048], [rows]
    int reuse_rows;
#if G6_MODEL_FLASH
    const void* index_gate;   // FP16 [rows,128] x @ index_kpool_compress_gate^T, full-index steps only
#endif
#ifdef DEC_S2
    int s2_batched;           // the step is replayed with a batched cell (G6AttnStepCell::rows): rows of several sessions
#endif
};
struct G6AttnBinding {
    G6Attn args;              // populated ABI; output/indices/error owned below
    void *storage, *scratch;
    size_t storage_bytes, scratch_bytes;
    int rank, layer, head_begin, capacity, max_rows, full_index;
    // Flash, G6_FLASH_KV_CHUNKS: the MLA and index caches' regions (G6KvChunks*, kvchunks.h; caller-owned), else null
    void* kv_chunks[2];
};

// Binding must initially be zero-initialized and must NOT be copied after create.
// Returned statuses are CUDA error codes. Pack pointer lifetime >= binding lifetime.
// Create/reset/prepare enqueue work on stream (void* cudaStream_t). Order later
// work on that stream, or synchronize it first. Create/release are load-time only.
extern "C" int g6a_bind_create(G6AttnBinding*, const G6AttnPack*, int rank,
    int layer, int capacity, int max_rows, int query_tile, void* stream);
// Same, but the decode scratch is ext_scratch (>= scratch need, caller-owned and zeroed, never freed by release)
// instead of a slice of storage. Bindings may share one ext_scratch only if their runs never overlap (one stream).
extern "C" int g6a_bind_create_ext(G6AttnBinding*, const G6AttnPack*, int rank,
    int layer, int capacity, int max_rows, int query_tile, void* stream, void* ext_scratch, size_t ext_bytes);
#if G6_MODEL_FLASH
struct G6KvChunks;
// Flash (G6_FLASH_KV_CHUNKS): the MLA and index caches are the two regions (kvchunks.h, each at least the capacity's
// pages; the caller frees them after release), the rest is storage as g6_attn_bind_create. Kernels see the same
// pointers; no step may reach a page whose chunk is offline. Reset zeroes only what the regions back.
extern "C" int g6_attn_bind_create_chunked(G6AttnBinding*, const G6AttnPack*, int rank, int layer, int capacity,
    int max_rows, int query_tile, void* stream, G6KvChunks* mla, G6KvChunks* idx);
// Pages of the smallest chunk that gives back all its bytes: its MLA and index bytes are whole granules (0: none).
extern "C" uint32_t g6_flash_kv_chunk_pages(size_t granule);
#endif
// Decode scratch bytes a binding with these heads / capacity / max_rows / query_tile needs (0: invalid).
extern "C" size_t g6a_bind_scratch_need(int heads, int capacity, int max_rows, int query_tile);
#if defined(G6_CP) && G6_CP
// CP binding: storage for the replicated prefix (< cp_R tokens, multiple of 64) + this rank's tail pages
// (page % cp_nr == rank); args.cp_nr / cp_rank set. rank = this rank (its head group per layer, as above).
// ext_scratch / ext_bytes as g6_attn_bind_create_ext (nullptr, 0 = own scratch).
extern "C" int g6a_bind_create_cp(G6AttnBinding*, const G6AttnPack*, int rank, int layer, int capacity,
    int max_rows, int query_tile, int cp_R, int cp_nr, void* stream, void* ext_scratch, size_t ext_bytes);
// cp_phys > 0: at most cp_phys physical pages (>= cp_R / 64); this rank's tail pages past them have -1 in the table as
// built, and the driver maps them (remote rows release, runtime/rt_rows.cuh). 0 = every tail page, as above.
extern "C" int g6a_bind_create_cp_phys(G6AttnBinding*, const G6AttnPack*, int rank, int layer, int capacity,
    int max_rows, int query_tile, int cp_R, int cp_nr, int cp_phys, void* stream, void* ext_scratch, size_t ext_bytes);
#endif
// Validate step, populate args and clear only its error word; preserve KV history.
// Full-index layers may also reuse (MTP later iterations). Shared layers MUST reuse.
extern "C" int g6a_bind_prepare(G6AttnBinding*, const G6AttnStep*, void* stream);
extern "C" int g6a_bind_run(G6AttnBinding*, const G6AttnStep*, void* stream);
// Call ONCE outside capture with fixed-address inputs, fixed rows/reuse mode,
// and a cudaMalloc step cell. Binding/cell/input storage must outlive the graph.
// For reused indices, step.reuse_rows is the allocated source-row upper bound.
// captured_capacity <= binding capacity; use separate power-of-two bucket graphs
// sharing the allocation to avoid scanning the full allocation on short contexts.
extern "C" int g6a_bind_replay_init(G6AttnBinding*, const G6AttnStep*,
    G6AttnStepCell*, int captured_capacity, G6AttnReplay*, void* stream);
// Explicit new-sequence reset: clear caches/output/error and set indices to -1.
extern "C" int g6a_bind_reset(G6AttnBinding*, void* stream);
// #50: rows x 512 E4M3 weights with f32 scales [rows][4] (one per row and 128-column block) -> BF16 [rows][512], each
// BF16(e4m3 x scale): the bytes the BF16 loader stores for the same weights (engine/loader-rs/src/release.rs).
extern "C" int g6a_kvb_bf16(const void* fp8, const float* scale, void* bf16, int rows, void* stream);
// Complete outstanding uses before release; frees ONLY binder-owned storage.
extern "C" int g6a_bind_release(G6AttnBinding*);
