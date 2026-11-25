// C interface of the Rust pack loader (engine/loader-rs). Built as libg6load.a with cargo; g6m4 links it when built
// with -DG6_RS_LOADER (scripts/build.sh does when cargo is available). See src/lib.rs for the environment knobs.
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {

struct g6l_extent
{
    uint64_t offset, nbytes;
};

// Staging buffers come from alloc_staging (pinned host memory); copy moves n bytes at src to image offset dst_off and
// returns 0 on success. Each worker passes its own index, so copy may keep one stream per worker.
struct g6l_sink
{
    void* ctx;
    void* (*alloc_staging)(void* ctx, size_t n);
    void (*free_staging)(void* ctx, void* p);
    int (*copy)(void* ctx, int worker, uint64_t dst_off, const void* src, size_t n);
};

struct g6l_stats
{
    double seconds;
    uint64_t image_bytes, read_bytes, ranges;
    uint32_t threads, mode;   // mode: 0 mmap, 1 read
};

// Fills a pack_bytes image through sink: tensor bytes from the pack file, overlaid by the ranges of sources (NULL or ""
// for a full pack), resolved against root (NULL or "" = the root the sources file names). Returns 0 on success,
// otherwise 1 with a message in err (errlen bytes).
int g6l_load(const char* pack, uint64_t pack_bytes, const g6l_extent* tensors, size_t ntensors, const char* sources,
             const char* root, const g6l_sink* sink, g6l_stats* stats, char* err, size_t errlen);

// Plans a rank's image without a v2 pack: the base pack's tensors, with the routed experts of every layer the width
// manifest lists read from the raw v2 width files (v2_dir or v2_dir/experts) at the offsets of the index slices
// (slices_dir/tp<N>/k<K>/rank<r>.safetensors; N and r from the base manifest). Writes the image's manifest to
// out_manifest (its sha256 fields are zero). Returns a handle, or NULL with a message in err.
void* g6l_v2_plan(const char* base_manifest, const char* base_pack, const char* v2_dir, const char* slices_dir,
                  const char* widths, const char* out_manifest, char* err, size_t errlen);
// Plans rank `rank`'s image at TP `nr` from the exlr8 v2 release alone, no base pack (src/release.rs): every non-expert
// tensor from the dense and vocab files under dense_dir (or dense_dir/dense), the experts as g6l_v2_plan reads them.
// qo != 0 adds every layer's full-head q_b / o_proj / UK / UV (L%03d.qo.*, the QO pack's tensors, GR8_PF_OWN=3).
// trellis != 0 gives the trellis matrices as the release's entries (engine/kernels/dtr.cuh), 0 rounds them to the
// MXFP8 and int4 forms of a converted pack. layout: a layout file (src/layout.rs) that says which pieces the rank
// holds, or NULL / "" for the formulas. Writes
// the image's manifest to out_manifest and, on rank 0, which form each matrix took to stderr. Returns a handle, or NULL
// with a message in err.
void* g6l_release_plan(const char* dense_dir, const char* v2_dir, const char* slices_dir, const char* widths, uint32_t nr,
                       uint32_t rank, int qo, int trellis, const char* layout, const char* out_manifest, char* err,
                       size_t errlen);
// Plans rank `rank` of `nr` of an Inkling-2-Small checkpoint directory (config.json + its safetensors shards) as a
// g6pack-v1 image and writes that image's manifest to out_manifest. NULL with a message in err when the plan fails.
void* g6l_ink2_plan(const char* model_dir, uint32_t nr, uint32_t rank, const char* out_manifest, char* err, size_t errlen);
void* g6l_ink2_plan_ex(const char* model_dir, uint32_t nr, uint32_t rank, const char* variant, const char* out_manifest, char* err, size_t errlen);
void* g6l_ink2_plan_ex2(const char* model_dir, uint32_t nr, uint32_t rank, const char* variant, int with_mtp, const char* out_manifest, char* err, size_t errlen);
// Plans rank `rank` of `nr` of a Qwen3.8-Flash-Next exlr8 v2 release directory (dense/exlr8-{dense,vocab,mtp},
// experts/, index-slices/, width manifest `widths`, optional `layout`). Writes the g6pack-v1 manifest to out_manifest.
void* g6l_qw38_plan(const char* release_dir, const char* widths, uint32_t nr, uint32_t rank, int with_mtp,
                    const char* layout, const char* out_manifest, char* err, size_t errlen);
// Fills dst with image bytes off .. off + n of a planned image, without a sink (one tensor at a time, for m4sim).
// Returns 0, or 1 with a message in err.
int g6l_plan_read(void* plan, uint64_t off, uint64_t n, void* dst, char* err, size_t errlen);
// Loads a planned image through sink, like g6l_load. Returns 0, or 1 with a message in err.
int g6l_plan_load(void* plan, const g6l_sink* sink, g6l_stats* stats, char* err, size_t errlen);
// Loads only the image byte ranges in keep[0..nkeep) from a planned image through sink (EL3 #151 delta piece-prefetch:
// skips chunks and source ranges outside keep so resident expert slabs are never re-read). Returns 0, or 1 + err.
int g6l_plan_load_ranges(void* plan, const g6l_extent* keep, size_t nkeep, const g6l_sink* sink, g6l_stats* stats,
                         char* err, size_t errlen);
void g6l_plan_free(void* plan);

// Layout-to-layout piece-delta resolver (EL3 #151 Stage 3). old_layout / new_layout may be NULL or "" to use the
// width's formula ownership; old_nr == 0 denotes a joiner box with no resident pieces.
void* g6l_delta_new(const char* old_layout, uint32_t old_nr, uint32_t old_rank,
                    const char* new_layout, uint32_t new_nr, uint32_t new_rank, char* err, size_t errlen);
// Queries one (layer, expert) row's piece delta. kept_old/kept_new receive *n_kept pairs (each <= 16), added receives
// *n_added target block indices, dropped receives *n_dropped old block indices. Returns 0, or 1 + err.
int g6l_delta_halves(const void* delta, uint32_t layer, uint32_t expert,
                     uint8_t* kept_old, uint8_t* kept_new, uint32_t* n_kept,
                     uint8_t* added, uint32_t* n_added,
                     uint8_t* dropped, uint32_t* n_dropped, char* err, size_t errlen);
void g6l_delta_free(void* delta);
}
