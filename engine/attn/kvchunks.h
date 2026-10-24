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
// The same region with only the granules g6_kvc_sync(page_bytes, chunk_pages, place, online, chunks) would back
// (zeroed): a region larger than what is online at boot needs no more memory than that.
int g6_kvc_create_masked(G6KvChunks** out, size_t bytes, int device, size_t page_bytes, uint32_t chunk_pages,
                         uint32_t place, const uint8_t* online, uint32_t chunks);
void g6_kvc_free(G6KvChunks* c);
// Zeroes every backed granule on `stream` (asynchronous); a granule not backed is left alone.
int g6_kvc_zero(G6KvChunks* c, void* stream);
void* g6_kvc_ptr(const G6KvChunks* c);
// the bytes it was created with
size_t g6_kvc_bytes(const G6KvChunks* c);
size_t g6_kvc_granule(const G6KvChunks* c);
// the granule a region on `device` would have (0 on an error)
size_t g6_kvc_device_granule(int device);
// bytes backed now
size_t g6_kvc_backed(const G6KvChunks* c);
// Pages of page_bytes from the region's start; chunk k is pages [k * chunk_pages, min((k + 1) * chunk_pages, place)),
// online[k] != 0 when it is online; chunks = ceil(place / chunk_pages), the pool's g6kv_chunk_count (anything else is
// CUDA_ERROR_INVALID_VALUE). Pages from place on are always backed. Backs and
// releases granules to match. It synchronizes the whole context first (no stream may still use a granule that goes),
// and zeroes the granules it backs on `stream`, synchronized before it returns. *mapped / *released: granule bytes
// changed (either may be NULL). On an error the granules changed so far stay changed (*mapped / *released count them,
// g6_kvc_backed says what is backed), and
// the zeroing of granules it backed may still be in flight on `stream`.
int g6_kvc_sync(G6KvChunks* c, size_t page_bytes, uint32_t chunk_pages, uint32_t place, const uint8_t* online,
                uint32_t chunks, void* stream, size_t* mapped, size_t* released);
}
