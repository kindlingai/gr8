// C interface of the ERNIE-5.5-Flash-Preview exlr8 v2/v4 release planner (src/ernie55.rs). Link libg6load.a.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Plans rank `rank`'s full device image at TP `nr` (1, 2 or 4) from an assembled ERNIE 5.5 Flash exlr8 release.
void* g6l_ernie55_plan(const char* release_dir, const char* widths, uint32_t nr, uint32_t rank,
                       const char* out_manifest, char* err, size_t errlen);

// Plans one split part (0=full, 1=perm+experts+embed, 2=dense linears, 3=hybrid VRAM core, 4=hybrid pageable host).
void* g6l_ernie55_plan_part(const char* release_dir, const char* widths, uint32_t nr, uint32_t rank, uint32_t part,
                            const char* out_manifest, char* err, size_t errlen);

#ifdef __cplusplus
}
#endif
