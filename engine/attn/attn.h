#pragma once
#include <stddef.h>
#include <stdint.h>
// C ABI for the rank driver and independent Python correctness harness.
// All tensor pointers are device addresses. Scalar metadata is host-readable.
struct G6Attn {
    int rows, heads, context, page_size, page_count, table_size, query_tile;
    const void *q_b, *kv_a, *kv_norm;
    const void *index_q, *index_k, *index_weight, *index_norm, *index_bias;
    const void *uk, *uv;
    const int *positions, *block_table;
    void *mla_cache, *index_cache;
    int *indices;                 // [rows,2048], logical positions, -1 padding
    const int *reuse_indices, *reuse_row_map;
    int reuse_rows;
    void *output;                 // fp16 [rows,heads,256]
    unsigned *error;              // initialize to zero before a step; asynchronous device diagnostics
#if defined(G6_CP) && G6_CP
    // J4 context-parallel KV (BUILD-CPE.md). cp_nr = 0: replicated (today's semantics). cp_nr > 0: this rank SCORES only
    // keys whose page (pos / page_size) % cp_nr == cp_rank, block_table holds G6_CP_REMOTE for pages stored elsewhere,
    // page_count = physical pages of this rank (the replay cell's page_count is then ignored).
    int cp_nr, cp_rank;
#endif
#ifdef DEC_S2
    // S2 batched step (the replay cell carries per-row sessions, replay.h G6AttnRow): set by g6_attn_bind_prepare from
    // G6AttnStep::s2_batched. Host-side dispatch only (the cell is device data): picks the fixed-K decode kernels.
    int s2_batched;
#endif
};
extern "C" size_t g6_attn_scratch_bytes(const G6Attn*);
extern "C" int g6_attn_run(const G6Attn*,void* scratch,size_t bytes,void* stream);
// Test/loader entrypoint: append only, same equations as the full path.
extern "C" int g6_attn_append(const G6Attn*,void* stream);
// Tests: materialize indexer scores (FP32 [rows,context]), independently of top-k.
extern "C" int g6_attn_index_scores(const G6Attn*,void* scratch,size_t bytes,float* scores,void* stream);
#if defined(G6_CP) && G6_CP
#define G6_CP_REMOTE (-2)
struct G6AttnStepCell;
// CP decode stages (one query tile: rows <= query_tile). cell = replay cell (graph form) or nullptr (plain form: a holds the
// live context). Buffers (device, caller-owned; H = a->heads, M = a->rows):
//   cand_all  u64 [cp_nr][M][2048] candidate keys (ordered_score << 32 | ~pos, 0 = none), descending; own list at [cp_rank]
//   q, rope   fp32 [M][H][512], [M][H][64]: absorbed query of the head group being computed (g6_attn_cp_qabs: my own)
//   part      fp32 partial of one head group: c [M][H][512], m [M][H], l [M][H]   (M * H * 514 floats)
//   part_all  fp32 [cp_nr] partials of MY head group in shard order 0..cp_nr-1
// front: validate (graph form), append owned/replicated pages, absorb my heads (-> g6_attn_cp_qabs)
extern "C" int g6_attn_cp_front(const G6Attn*,const G6AttnStepCell* cell,void* scratch,size_t bytes,void* stream);
// index: full-index layers: local top-2048 over my score-owned keys -> cand_all[cp_rank]; shared layers: copy the reused
//        list (the merged list of its source layer; cand_all unused)
extern "C" int g6_attn_cp_index(const G6Attn*,const G6AttnStepCell* cell,void* scratch,size_t bytes,uint64_t* cand_all,void* stream);
// full-index layers: exact merge of cand_all (after every rank's list landed) -> a->indices (same set and order as replicated)
extern "C" int g6_attn_cp_merge_index(const G6Attn*,const G6AttnStepCell* cell,const uint64_t* cand_all,void* stream);
// split-K partial of one head group over my score-owned keys of the merged list: unnormalized (m, l, c)
extern "C" int g6_attn_cp_group(const G6Attn*,const G6AttnStepCell* cell,void* scratch,size_t bytes,const float* q,const float* rope,float* part,void* stream);
// LSE merge of part_all in shard order 0..cp_nr-1 -> context -> value_project -> a->output
extern "C" int g6_attn_cp_finish(const G6Attn*,const G6AttnStepCell* cell,void* scratch,size_t bytes,const float* part_all,void* stream);
// my absorbed query in the scratch workspace (valid after g6_attn_cp_front)
extern "C" int g6_attn_cp_qabs(const G6Attn*,void* scratch,const float** q,const float** rope);
#endif
