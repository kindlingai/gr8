// C interface of engine/loader-rs/src/ds41.rs: the DeepSeek-V4.1-Flash-1017 rank image, planned from the checkpoint.
// Tensor names, forms and ownership: docs/DEEPSEEK-WEIGHTS.md. Use with g6load.h (g6l_plan_load, g6l_plan_read, g6l_plan_free).
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Plans rank `rank` of `nr` (TP width 1, 2, 3, 4 or 6) of the checkpoint under `ckpt_dir` (48 safetensors shards, read
// only). Reads the shard headers only (about 0.4 s). Writes the image manifest (g6pack-v1 fields; sha256 are zero) to
// `out_manifest`. Returns a handle for g6l_plan_load / g6l_plan_read / g6l_plan_free, or NULL with a message in `err`.
void* g6l_ds41_plan(const char* ckpt_dir, uint32_t nr, uint32_t rank, const char* out_manifest, char* err, size_t errlen);

#ifdef __cplusplus
}
#endif
