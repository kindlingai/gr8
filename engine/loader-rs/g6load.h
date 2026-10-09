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
// Loads a planned image through sink, like g6l_load. Returns 0, or 1 with a message in err.
int g6l_plan_load(void* plan, const g6l_sink* sink, g6l_stats* stats, char* err, size_t errlen);
void g6l_plan_free(void* plan);
}
