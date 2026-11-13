// C interface of the Inkling-2-Small in-place planner (src/ink2.rs). Link libg6load.a. Our own code.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Plans rank `rank`'s device image at TP `nr` straight from the checkpoint directory `model_dir` (config.json,
// optional hf_quant_config.json, model.safetensors.index.json + shards or model.safetensors, optional mtp.safetensors).
// Writes the image's manifest (g6pack-v1, sha256 fields zero) to out_manifest. Returns a handle for g6l_plan_load,
// g6l_plan_read and g6l_plan_free (g6load.h), or NULL with a message in err.
void* g6l_ink2_plan(const char* model_dir, uint32_t nr, uint32_t rank, const char* out_manifest, char* err, size_t errlen);

#ifdef __cplusplus
}
#endif
