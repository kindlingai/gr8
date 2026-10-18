#pragma once
#include <stddef.h>
#include <stdint.h>
#include "../model_geom.h"
// C ABI for the rank driver and independent Python correctness harness.
// All tensor pointers are device addresses. Scalar metadata is host-readable.
// Remote attention rows (runtime/rt_rows.cuh): a physical page flagged in G6Attn::rt_page has its latent rows in the
// resident slots of the layer's indexer group too; the local copy stays valid, so a row without a slot reads it.
// A released page (table entry page_count + v, v < rt_vpages) has no local copy: its rows read only from the slots,
// and its index keys sit in rt_ix.
// One indexer group's resident slots (device memory). The group's full-index layer runs the resolve after its own
// selection: hits stamp their slot, misses take slots (clock order from hand, never one used this step) and go to the
// request ring, which the host helper serves (kvman-rs src/rows.rs). Its wait sits before the first scores.
struct G6RtGroup {
    int *pos2slot;                  // [rows] row (table page * 64 + pos % 64) -> slot, -1 absent, -2 claimed in a resolve
    int *slot2pos;                  // [slots] -> row index, -1 free
    unsigned long long *last;       // [slots] the step that last used the slot
    int *miss;                      // [cap] scratch: the row indices one resolve missed
    const unsigned long long *step; // the driver's step counter (device)
    void *req;                      // mapped pinned: g6kv_rows_req, then cap g6kv_rows_ent (kvman-rs g6kvman.h)
    const unsigned long long *done; // mapped pinned: the helper's last finished seq (~0: a read failed)
    unsigned long long seq;         // requests rung so far
    unsigned long long base, rec;   // the group's first record in the store, record bytes
    int slots, rows, cap, hand, n;  // n: misses of the last resolve (the wait skips at 0)
    int kbase;                      // the row of the store's first record (released pages only: page_count * 64)
    unsigned long long local;       // misses the pool had no room for: read from the local copy
    // G6_RT_FMT=int4: the store's records hold INT4 rows (kvfmt.h int4row), and the helper reads each into the tail of
    // its FP8 slot (out + slot * nl * 656 + soff); after the wait, rt_unpack turns it into that slot's FP8 rows in place.
    // out nullptr: FP8 records, read straight into the slots
    unsigned char *out;
    int nl, soff;                   // layers in the group, the record's place in its slot
};
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
#if G6_MODEL_FLASH
    // kpool indexer (attn/flash_attn.cu): the pool gate projection of the rows, FP16 [rows,128] = x @ compress_gate^T,
    // and the pool position bias, BF16 [4,128] (index_kpool_compress_ape). indices is [rows,g6kv::SEL] (kvfmt.h).
    const void *index_gate, *index_ape;
#endif
#if defined(G6_CP) && G6_CP
    // context-parallel KV. cp_nr = 0: replicated (today's semantics). cp_nr > 0: this rank SCORES only
    // keys whose page (pos / page_size) % cp_nr == cp_rank, block_table holds G6_CP_REMOTE for pages stored elsewhere,
    // page_count = physical pages of this rank (the replay cell's page_count is then ignored).
    int cp_nr, cp_rank;
    // cp_nr = 0 only: the index cache's page of each logical page (table_size entries). nullptr = block_table. The staged
    // prefill view of an S2 run below R sets it: its block_table addresses the staging pool, while the run's index keys
    // stay on its own pages. Other staged views stage the index keys in the pool too (runtime/prefill.cuh).
    const int *index_table;
#endif
#ifdef DEC_S2
    // S2 batched step (the replay cell carries per-row sessions, replay.h G6AttnRow): set by g6_attn_bind_prepare from
    // G6AttnStep::s2_batched. Host-side dispatch only (the cell is device data): picks the fixed-K decode kernels.
    int s2_batched;
#endif
    // Remote rows: page p with rt_page[p] != 0 reads row p * 64 + pos % 64 from its slot rt_table[row] (>= 0),
    // at rt_rows + slot * rt_stride; -1 = the local row, or for a released page (p >= page_count) error 0x40000.
    // rt_page nullptr = off. rt_group: set on the group's full-index layer, which runs the resolve and the wait (CP
    // decode). Released pages: rt_vpages of them after the physical ones, their index keys at rt_ix (full-index layers).
    const unsigned char *rt_page;
    const int *rt_table;
    const unsigned char *rt_rows;
    G6RtGroup *rt_group;
    int rt_stride;
    int rt_int4;                    // on rt_group's layer: its G6RtGroup has a stage (host-side dispatch of rt_unpack)
    int rt_vpages;
    const unsigned char *rt_ix;
};
// Remote rows, G6_RT_FMT=int4 (runtime/rt_rows.cuh): n FP8 rows -> INT4 rows (spill) and back (tests); warp per row,
// strides in bytes. cudaErrorNotSupported in a G6_KV4 build, whose rows are INT4 already.
extern "C" int g6_rt_pack_int4(const void* src,size_t src_stride,void* dst,size_t dst_stride,int n,void* stream);
extern "C" int g6_rt_unpack_int4(const void* src,size_t src_stride,void* dst,size_t dst_stride,int n,void* stream);
extern "C" size_t g6_attn_scratch_bytes(const G6Attn*);
extern "C" void (*g6_attn_mark)(int stage,void* stream);
extern "C" int g6_attn_rope_init(int cap,void* stream);   // decode RoPE table when no QO attention builds it   // decode profile hook (attn.cu AMARK); null = off
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
// G6_CP_ONECHAIN: g6_attn_cp_group of G head groups of gh heads in ONE call (a->heads = G * gh): q / rope [M][G gh][512] /
// [M][G gh][64] (group g's heads at g * gh), group g's partial into part + g * gblk floats (gblk >= M * gh * 514). The same bits
// as G calls when g6_attn_cp_groups_ok(gh). buf: device, g6_attn_cp_groups_bytes(G * gh, M) bytes (its own scratch)
extern "C" int g6_attn_cp_groups_ok(int gh);
extern "C" size_t g6_attn_cp_groups_bytes(int heads,int rows);
extern "C" int g6_attn_cp_groups(const G6Attn*,const G6AttnStepCell* cell,void* buf,size_t bytes,const float* q,const float* rope,float* part,
                                 int gh,int64_t gblk,void* stream);
// LSE merge of part_all in shard order 0..cp_nr-1 -> context -> value_project -> a->output
extern "C" int g6_attn_cp_finish(const G6Attn*,const G6AttnStepCell* cell,void* scratch,size_t bytes,const float* part_all,void* stream);
// my absorbed query in the scratch workspace (valid after g6_attn_cp_front)
extern "C" int g6_attn_cp_qabs(const G6Attn*,void* scratch,const float** q,const float** rope);
#endif
