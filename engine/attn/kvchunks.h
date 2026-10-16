// KV memory that shrinks and grows at runtime (#153): one region of KV pages (a binding's MLA cache, or its index
// cache) in reserved virtual memory, backed granule by granule (cuMemCreate / cuMemMap, 2 MiB on GB10). The pool's
// chunks (engine/kvman-rs Pool::chunks) decide which pages must be backed: g6_kvc_sync backs every granule that holds a
// byte of an online chunk's pages or of a page past place, and releases every other granule. A granule two chunks
// share stays while either is online, so a shrink returns slightly less than the chunks' bytes.
// The address never moves, so block tables, kernels and captured CUDA graphs keep working across a resize, as long as
// no kernel reads an offline page (the pool hands none out). Pages that come back read as zero.
// Host only; calls the CUDA driver API (link -lcuda). Not thread safe. Order: run the pool's queued spills (io())
// before a sync that releases their pages, and sync before the pool's grow_pool() hands the pages out.
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {

struct G6KvChunks;

// A region of `bytes` on `device`, every granule backed (zeroed). 0 or a CUDA driver error code.
int g6_kvc_create(G6KvChunks** out, size_t bytes, int device);
void g6_kvc_free(G6KvChunks* c);
void* g6_kvc_ptr(const G6KvChunks* c);
size_t g6_kvc_granule(const G6KvChunks* c);
// bytes backed now
size_t g6_kvc_backed(const G6KvChunks* c);
// Pages of page_bytes from the region's start; chunk k is pages [k * chunk_pages, min((k + 1) * chunk_pages, place)),
// online[k] != 0 when it is online. Pages from place on are always backed. Backs and releases granules to match;
// granules it backs are zeroed on `stream` (synchronized before the call returns). *mapped / *released: granule bytes
// changed (either may be NULL).
int g6_kvc_sync(G6KvChunks* c, size_t page_bytes, uint32_t chunk_pages, uint32_t place, const uint8_t* online,
                uint32_t chunks, void* stream, size_t* mapped, size_t* released);
}
