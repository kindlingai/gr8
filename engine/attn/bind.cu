#include "bind.h"
#include "../g6nr.h"
#include "../h2d.h"
#include <cuda_runtime.h>
#include <climits>
#include <initializer_list>
#include <vector>

namespace {
bool shape(const G6AttnTensor& t, std::initializer_list<int64_t> dims) {
    if (!t.data || t.ndim != (int)dims.size()) return false;
    int j = 0;
    for (auto d : dims) if (t.shape[j++] != d) return false;
    return true;
}
__global__ void transpose_uk(const uint16_t* src, uint16_t* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int h = i / (512 * 192), k = (i / 192) % 512, d = i % 192;
        dst[i] = src[(h * 192 + d) * 512 + k];
    }
}
__global__ void identity_pages(int* table, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) table[i] = i;
}
constexpr int invalid = (int)cudaErrorInvalidValue;
}

#if defined(G6_CP) && G6_CP
static int create(G6AttnBinding* out, const G6AttnPack* p, int rank, int layer, int capacity, int max_rows,
    int query_tile, void* s, void* ext_scratch, size_t ext_bytes, int cp_R, int cp_nr);
#endif
extern "C" size_t g6_attn_bind_scratch_need(int heads, int capacity, int max_rows, int query_tile) {
    if (capacity <= 0 || max_rows <= 0 || query_tile <= 0 || query_tile > 64 || capacity > INT_MAX / query_tile) return 0;
    G6Attn a{};
    a.rows = max_rows; a.context = capacity; a.heads = heads; a.page_size = 64;
    a.page_count = a.table_size = (int)(((int64_t)capacity + 63) / 64);
    a.query_tile = query_tile;
    return g6_attn_scratch_bytes(&a);
}
extern "C" int g6_attn_bind_create(G6AttnBinding* out, const G6AttnPack* p,
    int rank, int layer, int capacity, int max_rows, int query_tile, void* s) {
    return g6_attn_bind_create_ext(out, p, rank, layer, capacity, max_rows, query_tile, s, nullptr, 0);
}
#if defined(G6_CP) && G6_CP
extern "C" int g6_attn_bind_create_ext(G6AttnBinding* out, const G6AttnPack* p,
    int rank, int layer, int capacity, int max_rows, int query_tile, void* s, void* ext_scratch, size_t ext_bytes) {
    return create(out, p, rank, layer, capacity, max_rows, query_tile, s, ext_scratch, ext_bytes, 0, 0);
}
extern "C" int g6_attn_bind_create_cp(G6AttnBinding* out, const G6AttnPack* p, int rank, int layer, int capacity,
    int max_rows, int query_tile, int cp_R, int cp_nr, void* s, void* ext_scratch, size_t ext_bytes) {
    if (cp_nr < 2 || cp_nr > 8 || rank >= cp_nr || cp_R < 0 || cp_R % 64) return invalid;
    return create(out, p, rank, layer, capacity, max_rows, query_tile, s, ext_scratch, ext_bytes, cp_R, cp_nr);
}
static int create(G6AttnBinding* out, const G6AttnPack* p, int rank, int layer, int capacity, int max_rows,
    int query_tile, void* s, void* ext_scratch, size_t ext_bytes, int cp_R, int cp_nr) {
#else
extern "C" int g6_attn_bind_create_ext(G6AttnBinding* out, const G6AttnPack* p,
    int rank, int layer, int capacity, int max_rows, int query_tile, void* s, void* ext_scratch, size_t ext_bytes) {
#endif
    if (!out || out->storage || !p || rank < 0 || rank >= g6nr::NR || layer < 0 || layer > 78 ||
        capacity <= 0 || max_rows <= 0 || max_rows > capacity || query_tile <= 0 ||
        query_tile > 64 || capacity > INT_MAX / query_tile) return invalid;
    // head group of this rank at this layer: (rank - layer) % NR, the larger groups first (TP6 edges
    // 0,11,22,33,44,54,64; TP4 16 heads each)
    const int group = (rank - layer % g6nr::NR + g6nr::NR) % g6nr::NR;
    const int h = g6nr::width(g6nr::HEADS, group);
    const bool full = layer == 78 || layer <= 2 || (layer - 2) % 4 == 0;
    if (!shape(p->uk, {h,192,512}) || !shape(p->uv, {h,256,512}) ||
        !shape(p->kv_norm, {512}) ||
        (full && (!shape(p->index_norm, {128}) || !shape(p->index_bias, {128})))) return invalid;

    G6AttnBinding b{};
    b.rank = rank; b.layer = layer; b.head_begin = g6nr::start(g6nr::HEADS, group);
    b.capacity = capacity; b.max_rows = max_rows; b.full_index = full;
    G6Attn& a = b.args;
    a.rows = max_rows; a.context = capacity; a.heads = h; a.page_size = 64;
    a.page_count = a.table_size = (int)(((int64_t)capacity + 63) / 64);
#if defined(G6_CP) && G6_CP
    // CP page map: logical pages < R/64 replicated, the rest stored by page % cp_nr. Physical
    // pages: the replicated prefix first, then my tail pages in logical order. The table is per binding (one sequence
    // today); a per-session table is the same map with its own physical pages.
    std::vector<int> cp_table;
    a.cp_nr = cp_nr; a.cp_rank = cp_nr ? rank : 0;
    if (cp_nr) {
        cp_table.resize(a.table_size);
        int phys = 0;
        for (int lp = 0; lp < a.table_size; ++lp)
            cp_table[lp] = (lp < cp_R / 64 || lp % cp_nr == rank) ? phys++ : G6_CP_REMOTE;
        a.page_count = phys;
    }
#endif
    a.query_tile = query_tile;
    a.kv_norm = p->kv_norm.data; a.uv = p->uv.data;
    a.index_norm = full ? p->index_norm.data : nullptr;
    a.index_bias = full ? p->index_bias.data : nullptr;
#if defined(G6_CP) && G6_CP
    {   // CP: cp_group also runs the other ranks' head groups on this scratch (NR=6: 10 or 11 heads) -> size for the widest
        G6Attn w = a; if (cp_nr) w.heads = (64 + cp_nr - 1) / cp_nr;
        b.scratch_bytes = g6_attn_scratch_bytes(&w);
    }
#else
    b.scratch_bytes = g6_attn_scratch_bytes(&a);
#endif
    if (!b.scratch_bytes || (ext_scratch && ext_bytes < b.scratch_bytes)) return invalid;
    size_t used = 0;
    auto take = [&](size_t n) { size_t at = used; used = (used + n + 255) & ~size_t(255); return at; };
    const size_t uk = take((size_t)h * 512 * 192 * 2);
    const size_t kv = take((size_t)a.page_count * 64 * 656);
    // Shared layers never append/index their own indexer cache. Index keys sit on the same pages as the latent (under
    // CP: the replicated prefix and this rank's pages past R), so the index cache has the latent's page count.
    const int ix_pages = a.page_count;
    const size_t ix = take(full ? (size_t)ix_pages * 64 * 132 : 0);
    const size_t table = take((size_t)a.table_size * sizeof(int));
    const size_t ids = take((size_t)max_rows * 2048 * sizeof(int));
    const size_t output = take((size_t)max_rows * h * 256 * 2);
    const size_t error = take(sizeof(unsigned));
    const size_t scratch = take(ext_scratch ? 0 : b.scratch_bytes);   // ext: caller-owned, shared, not freed here
    b.storage_bytes = used;
    cudaError_t e = cudaMalloc(&b.storage, used);
    if (e != cudaSuccess) return (int)e;
    char* base = (char*)b.storage;
    a.uk = base + uk; a.mla_cache = base + kv;
    a.index_cache = full ? base + ix : nullptr;
    a.block_table = (int*)(base + table); a.indices = (int*)(base + ids);
    a.output = base + output; a.error = (unsigned*)(base + error);
    b.scratch = ext_scratch ? ext_scratch : base + scratch;
    cudaStream_t stream = (cudaStream_t)s;
    e = cudaMemsetAsync(b.storage, 0, used, stream);
    if (e == cudaSuccess) {
        transpose_uk<<<(h * 512 * 192 + 255) / 256,256,0,stream>>>(
            (const uint16_t*)p->uk.data, (uint16_t*)const_cast<void*>(a.uk), h * 512 * 192);
#if defined(G6_CP) && G6_CP
        if (cp_nr) {   // after the storage memset on the same stream; synchronize: cp_table is a host temporary
            e = g6h2d::copy((void*)a.block_table, cp_table.data(), cp_table.size() * sizeof(int), stream);
        }
        else
#endif
        identity_pages<<<(a.page_count + 255) / 256,256,0,stream>>>(
            (int*)const_cast<int*>(a.block_table), a.page_count);
        if (e == cudaSuccess) e = cudaGetLastError();
    }
    if (e == cudaSuccess) e = cudaMemsetAsync(a.indices, 0xff, (size_t)max_rows * 2048 * 4, stream);
    if (e != cudaSuccess) { cudaFree(b.storage); return (int)e; }
    *out = b;
    return (int)cudaSuccess;
}

extern "C" int g6_attn_bind_prepare(G6AttnBinding* b, const G6AttnStep* step, void* s) {
    if (!b || !b->storage || !step || step->rows <= 0 || step->rows > b->max_rows ||
        step->context <= 0 || step->context > b->capacity || step->rows > step->context ||
        !step->q_b || !step->kv_a || !step->positions) return invalid;
    if (step->reuse_indices) {
        if (step->reuse_rows <= 0 || (!step->reuse_row_map && step->reuse_rows < step->rows) ||
            (step->reuse_row_map && step->reuse_indices == b->args.indices)) return invalid;
    } else if (!b->full_index || !step->index_q || !step->index_k || !step->index_weight ||
               step->reuse_row_map || step->reuse_rows) return invalid;
    G6Attn& a = b->args;
    a.rows = step->rows; a.context = step->context;
    a.q_b = step->q_b; a.kv_a = step->kv_a; a.positions = step->positions;
    a.index_q = step->index_q; a.index_k = step->index_k; a.index_weight = step->index_weight;
    a.reuse_indices = step->reuse_indices; a.reuse_row_map = step->reuse_row_map;
    a.reuse_rows = step->reuse_rows;
#ifdef DEC_S2
    a.s2_batched = step->s2_batched;
#endif
    return (int)cudaMemsetAsync(a.error, 0, sizeof(unsigned), (cudaStream_t)s);
}
extern "C" int g6_attn_bind_run(G6AttnBinding* b, const G6AttnStep* step, void* s) {
    int e = g6_attn_bind_prepare(b, step, s);
    return e ? e : g6_attn_run(&b->args, b->scratch, b->scratch_bytes, s);
}
extern "C" int g6_attn_bind_replay_init(G6AttnBinding* b, const G6AttnStep* step,
    G6AttnStepCell* cell, int captured_capacity, G6AttnReplay* out, void* s) {
    if (!b || !step || !cell || !out || captured_capacity < step->context ||
        captured_capacity <= 0 || captured_capacity > b->capacity) return invalid;
    cudaPointerAttributes attr{};
    cudaError_t e = cudaPointerGetAttributes(&attr, cell);
    if (e != cudaSuccess) return (int)e;
    if (attr.type != cudaMemoryTypeDevice) return invalid;
    int code = g6_attn_bind_prepare(b, step, s);
    if (code) return code;
    G6AttnReplay r{};
    r.fixed = b->args;
    r.fixed.context = captured_capacity;
    r.cell = cell; r.scratch = b->scratch; r.scratch_bytes = b->scratch_bytes;
    *out = r;
    return 0;
}
extern "C" int g6_attn_bind_reset(G6AttnBinding* b, void* s) {
    if (!b || !b->storage) return invalid;
    const G6Attn& a = b->args;
    cudaStream_t stream = (cudaStream_t)s;
    cudaError_t e = cudaMemsetAsync(a.mla_cache, 0, (size_t)a.page_count * 64 * 656, stream);
    const int ix_pages = a.page_count;
    if (e == cudaSuccess && a.index_cache) e = cudaMemsetAsync(a.index_cache, 0, (size_t)ix_pages * 64 * 132, stream);
    if (e == cudaSuccess) e = cudaMemsetAsync(a.indices, 0xff, (size_t)b->max_rows * 2048 * 4, stream);
    if (e == cudaSuccess) e = cudaMemsetAsync(a.output, 0, (size_t)b->max_rows * a.heads * 256 * 2, stream);
    if (e == cudaSuccess) e = cudaMemsetAsync(a.error, 0, sizeof(unsigned), stream);
    return (int)e;
}
extern "C" int g6_attn_bind_release(G6AttnBinding* b) {
    if (!b) return invalid;
    cudaError_t e = b->storage ? cudaFree(b->storage) : cudaSuccess;
    if (e == cudaSuccess) *b = G6AttnBinding{};
    return (int)e;
}
