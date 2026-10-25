// The attention binder (bind.h) for the G6_MODEL_FLASH build: flash_attn.cu's caches and scratch. Same contract as
// bind.cu; the differences: every DSA layer has its own kpool indexer (geom::full_index), UK is bound as stored
// [H,256,512] (no transpose), the pack adds index_ape, a step adds index_gate, the index cache uses the kvfmt.h kpool
// page (IDX_PAGE per 64 rows) and indices are [max_rows][g6kv::SEL]. No context-parallel binding.
#include "bind.h"
#include "../g6nr.h"
#include "../model_geom.h"
#include "kvfmt.h"
#include <cuda_runtime.h>
#include <climits>
#include <initializer_list>
#if !G6_MODEL_FLASH
#error "flash_bind.cu is the G6_MODEL_FLASH binder; the GLM build compiles bind.cu"
#endif

namespace {
bool shape(const G6AttnTensor& t, std::initializer_list<int64_t> dims) {
    if (!t.data || t.ndim != (int)dims.size()) return false;
    int j = 0;
    for (auto d : dims) if (t.shape[j++] != d) return false;
    return true;
}
__global__ void identity_pages(int* table, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) table[i] = i;
}
constexpr int invalid = (int)cudaErrorInvalidValue;
constexpr int SEL = g6kv::SEL, PAGE = g6kv::PAGE;
}

extern "C" size_t g6a_bind_scratch_need(int heads, int capacity, int max_rows, int query_tile) {
    if (capacity <= 0 || max_rows <= 0 || query_tile <= 0 || query_tile > 64 || capacity > INT_MAX / query_tile) return 0;
    G6Attn a{};
    a.rows = max_rows; a.context = capacity; a.heads = heads; a.page_size = PAGE;
    a.page_count = a.table_size = (int)(((int64_t)capacity + PAGE - 1) / PAGE);
    a.query_tile = query_tile;
    return g6a_scratch_bytes(&a);
}
extern "C" int g6a_bind_create(G6AttnBinding* out, const G6AttnPack* p,
    int rank, int layer, int capacity, int max_rows, int query_tile, void* s) {
    return g6a_bind_create_ext(out, p, rank, layer, capacity, max_rows, query_tile, s, nullptr, 0);
}
#if defined(G6_CP) && G6_CP
extern "C" int g6a_bind_create_cp(G6AttnBinding*, const G6AttnPack*, int, int, int, int, int, int, int, void*, void*, size_t) {
    return (int)cudaErrorNotSupported;
}
#endif
extern "C" int g6a_bind_create_ext(G6AttnBinding* out, const G6AttnPack* p,
    int rank, int layer, int capacity, int max_rows, int query_tile, void* s, void* ext_scratch, size_t ext_bytes) {
    if (!out || out->storage || !p || rank < 0 || rank >= g6nr::NR || layer < 0 || layer > geom::MTP_LAYER ||
        !geom::full_index(layer) || capacity <= 0 || max_rows <= 0 || max_rows > capacity || query_tile <= 0 ||
        query_tile > 64 || capacity > INT_MAX / query_tile) return invalid;
    // head group of this rank at this layer: (rank - layer) % NR (as bind.cu and loader-rs Shard::heads)
    const int group = (rank - layer % g6nr::NR + g6nr::NR) % g6nr::NR;
    const int h = g6nr::width(g6nr::HEADS, group);
    if (!shape(p->uk, {h, geom::QK_NOPE, geom::KV_LORA}) || !shape(p->uv, {h, geom::V_HEAD, geom::KV_LORA}) ||
        !shape(p->kv_norm, {geom::KV_LORA}) || !shape(p->index_norm, {g6kv::IDX_DIM}) ||
        !shape(p->index_bias, {g6kv::IDX_DIM}) || !shape(p->index_ape, {g6kv::KPOOL, g6kv::IDX_DIM})) return invalid;

    G6AttnBinding b{};
    b.rank = rank; b.layer = layer; b.head_begin = g6nr::start(g6nr::HEADS, group);
    b.capacity = capacity; b.max_rows = max_rows; b.full_index = 1;
    G6Attn& a = b.args;
    a.rows = max_rows; a.context = capacity; a.heads = h; a.page_size = PAGE;
    a.page_count = a.table_size = (int)(((int64_t)capacity + PAGE - 1) / PAGE);
    a.query_tile = query_tile;
    a.kv_norm = p->kv_norm.data; a.uk = p->uk.data; a.uv = p->uv.data;
    a.index_norm = p->index_norm.data; a.index_bias = p->index_bias.data; a.index_ape = p->index_ape.data;
    b.scratch_bytes = g6a_scratch_bytes(&a);
    if (!b.scratch_bytes || (ext_scratch && ext_bytes < b.scratch_bytes)) return invalid;
    size_t used = 0;
    auto take = [&](size_t n) { size_t at = used; used = (used + n + 255) & ~size_t(255); return at; };
    const size_t kv = take((size_t)a.page_count * PAGE * g6kv::ROW);
    const size_t ix = take((size_t)a.page_count * g6kv::IDX_PAGE);
    const size_t table = take((size_t)a.table_size * sizeof(int));
    const size_t ids = take((size_t)max_rows * SEL * sizeof(int));
    const size_t output = take((size_t)max_rows * h * geom::V_HEAD * 2);
    const size_t error = take(sizeof(unsigned));
    const size_t scratch = take(ext_scratch ? 0 : b.scratch_bytes);
    b.storage_bytes = used;
    cudaError_t e = cudaMalloc(&b.storage, used);
    if (e != cudaSuccess) return (int)e;
    char* base = (char*)b.storage;
    a.mla_cache = base + kv; a.index_cache = base + ix;
    a.block_table = (int*)(base + table); a.indices = (int*)(base + ids);
    a.output = base + output; a.error = (unsigned*)(base + error);
    b.scratch = ext_scratch ? ext_scratch : base + scratch;
    cudaStream_t stream = (cudaStream_t)s;
    e = cudaMemsetAsync(b.storage, 0, used, stream);
    if (e == cudaSuccess) {
        identity_pages<<<(a.page_count + 255) / 256, 256, 0, stream>>>((int*)const_cast<int*>(a.block_table), a.page_count);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess) e = cudaMemsetAsync(a.indices, 0xff, (size_t)max_rows * SEL * 4, stream);
    if (e != cudaSuccess) { cudaFree(b.storage); return (int)e; }
    *out = b;
    return (int)cudaSuccess;
}

extern "C" int g6a_bind_prepare(G6AttnBinding* b, const G6AttnStep* step, void* s) {
    if (!b || !b->storage || !step || step->rows <= 0 || step->rows > b->max_rows ||
        step->context <= 0 || step->context > b->capacity || step->rows > step->context ||
        !step->q_b || !step->kv_a || !step->positions) return invalid;
    if (step->reuse_indices) {
        if (step->reuse_rows <= 0 || (!step->reuse_row_map && step->reuse_rows < step->rows) ||
            (step->reuse_row_map && step->reuse_indices == b->args.indices)) return invalid;
    } else if (!step->index_q || !step->index_k || !step->index_weight || !step->index_gate ||
               step->reuse_row_map || step->reuse_rows) return invalid;
    G6Attn& a = b->args;
    a.rows = step->rows; a.context = step->context;
    a.q_b = step->q_b; a.kv_a = step->kv_a; a.positions = step->positions;
    a.index_q = step->index_q; a.index_k = step->index_k; a.index_weight = step->index_weight;
    a.index_gate = step->index_gate;
    a.reuse_indices = step->reuse_indices; a.reuse_row_map = step->reuse_row_map;
    a.reuse_rows = step->reuse_rows;
#ifdef DEC_S2
    a.s2_batched = step->s2_batched;
#endif
    return (int)cudaMemsetAsync(a.error, 0, sizeof(unsigned), (cudaStream_t)s);
}
extern "C" int g6a_bind_run(G6AttnBinding* b, const G6AttnStep* step, void* s) {
    int e = g6a_bind_prepare(b, step, s);
    return e ? e : g6a_run(&b->args, b->scratch, b->scratch_bytes, s);
}
extern "C" int g6a_bind_replay_init(G6AttnBinding* b, const G6AttnStep* step,
    G6AttnStepCell* cell, int captured_capacity, G6AttnReplay* out, void* s) {
    if (!b || !step || !cell || !out || captured_capacity < step->context ||
        captured_capacity <= 0 || captured_capacity > b->capacity) return invalid;
    cudaPointerAttributes attr{};
    cudaError_t e = cudaPointerGetAttributes(&attr, cell);
    if (e != cudaSuccess) return (int)e;
    if (attr.type != cudaMemoryTypeDevice) return invalid;
    int code = g6a_bind_prepare(b, step, s);
    if (code) return code;
    G6AttnReplay r{};
    r.fixed = b->args;
    r.fixed.context = captured_capacity;
    r.cell = cell; r.scratch = b->scratch; r.scratch_bytes = b->scratch_bytes;
    *out = r;
    return 0;
}
extern "C" int g6a_bind_reset(G6AttnBinding* b, void* s) {
    if (!b || !b->storage) return invalid;
    const G6Attn& a = b->args;
    cudaStream_t stream = (cudaStream_t)s;
    cudaError_t e = cudaMemsetAsync(a.mla_cache, 0, (size_t)a.page_count * PAGE * g6kv::ROW, stream);
    if (e == cudaSuccess) e = cudaMemsetAsync(a.index_cache, 0, (size_t)a.page_count * g6kv::IDX_PAGE, stream);
    if (e == cudaSuccess) e = cudaMemsetAsync(a.indices, 0xff, (size_t)b->max_rows * SEL * 4, stream);
    if (e == cudaSuccess) e = cudaMemsetAsync(a.output, 0, (size_t)b->max_rows * a.heads * geom::V_HEAD * 2, stream);
    if (e == cudaSuccess) e = cudaMemsetAsync(a.error, 0, sizeof(unsigned), stream);
    return (int)e;
}
extern "C" int g6a_bind_release(G6AttnBinding* b) {
    if (!b) return invalid;
    cudaError_t e = b->storage ? cudaFree(b->storage) : cudaSuccess;
    if (e == cudaSuccess) *b = G6AttnBinding{};
    return (int)e;
}
