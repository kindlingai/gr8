// C interface of the Tencent Hy4-flash exlr8 v2 release planner (src/hy4.rs). Link libg6load.a.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Plans rank `rank`'s device image at TP `nr` (1, 2 or 4) from an assembled Hy4-flash exlr8 v2 release directory
// and width manifest `widths`. Writes the image's g6pack-v1 manifest to `out_manifest` and returns a handle for
// g6l_plan_load / g6l_plan_read / g6l_plan_free (g6load.h), or NULL with a message in `err`.
void* g6l_hy4_plan(const char* release_dir, const char* widths, uint32_t nr, uint32_t rank,
                   const char* out_manifest, char* err, size_t errlen);

#ifdef __cplusplus
}
#endif
