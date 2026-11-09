// C interface of engine/loader-rs/src/ds41.rs: the DeepSeek-V4.1-Flash-1017 rank image, planned from the exlr8 native
// release (kindlingai/deepseek-v4.1-flash-exlr8-native). Tensor names, forms and ownership: docs/DEEPSEEK-WEIGHTS.md.
// Use with g6load.h (g6l_plan_load, g6l_plan_read, g6l_plan_free).
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Plans rank `rank` of `nr` (TP width 1, 2, 3, 4 or 6) of the release under `release_dir` (read only; headers and
// index files). Writes the image manifest (g6pack-v1 fields; sha256 are zero) to `out_manifest` and the Engram tables
// the rank serves (g6engram-v2: per table its split-file runs) to `out_engram`. Returns a handle for g6l_plan_load /
// g6l_plan_read / g6l_plan_free, or NULL with a message in `err`.
void* g6l_ds41_plan(const char* release_dir, uint32_t nr, uint32_t rank, const char* out_manifest, const char* out_engram,
                    char* err, size_t errlen);

#ifdef __cplusplus
}
#endif
