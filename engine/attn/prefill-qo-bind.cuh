// Row-owner QO prefill attention entry (included at the end of attn.cu, after prefill.cuh). See prefill-qo-bind.h.
// Per call: absorb (cuBLAS strided batched UK, FP32) -> pack (RN16 + RoPE, pf v2 pack_q math, row-major [row][h][576])
// -> QO core (B: prepass to an FP16 key table + core_b<true>; v1: prototype core) -> RN16 -> UV (cuBLAS, FP16 out
// written straight into the o_proj input layout [row][h*256]).
#include "prefill-qo-bind.h"
#if PF_ROPE2 && !PF_QOROPE
#error "PF_ROPE2 reads the PF_QOROPE table: build with -DPF_QOROPE=1"
#endif
#if PF_QOROPE && !PF_QOABS16
#undef PF_QOABS16
#define PF_QOABS16 1                        // PF_QOROPE implies the QOABS16 absorb/pack change
#endif
#include <cstring>
#include <ctime>
#include "prefill-qo-proto.cuh"
#include "prefill-qo-b.cuh"
#if __has_include("prefill-qo-r1.cuh")
#include "prefill-qo-r1.cuh"
#define G6_QO_HAVE_R1 1
#endif

struct G6AttnQoWork {
    void* data = nullptr; size_t bytes = 0; int cap = 0, max_rows = 0, device = 0;
    cublasHandle_t blas = nullptr; std::atomic<unsigned> refs{1};
};
struct G6AttnQo {
    int kernel = 0;                         // 0 = B, 1 = v1, 2 = R1
    int cap = 0, max_rows = 0;
    int heads = 64;                         // 64 (G6_PF_OWN=3), or 32 (G6_PF_OWN=4 pair-owner, TP4 builds only): uk->shape[0]
    void* storage = nullptr; size_t weight_bytes = 0, bytes = 0;
    __half *uk = nullptr, *uv = nullptr;    // [heads][512][192], [heads][256][512] FP16
    G6AttnQoWork* w = nullptr;
    __half *table = nullptr, *q = nullptr, *ctxh = nullptr; uint8_t* bad = nullptr; float* ctx = nullptr;
#if PF_QOROPE
    float2* rope = nullptr;                 // [pos < cap][j < 32] {cos, sin} = ga::cs(pos, j); in the shared work, built by the owner
#endif
};
namespace gqi {
constexpr int H = 64;
__global__ void uk_half_t(const void* src, __half* dst, int n) {      // [h][192][512] BF16 -> [h][512][192] FP16
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { int h = i / (512 * 192), k = (i / 192) % 512, d = i % 192; dst[i] = __float2half_rn(ga::bfloat(src, ((int64_t)h * 192 + d) * 512 + k)); }
}
// The pack kernels take the head count as a template parameter of an inlined body: the 64-head kernels keep their names
// and code, and a TP4 build adds the 32-head ones (G6_PF_OWN=4) that a TP6 build never compiles.
// q[row][h][576] from qf[row][h][512] FP32 and the RoPE half of q_b (same equations as gp::pack_q)
template <int H>
__device__ __forceinline__ void pack_q_rows_body(const G6AttnQoStep& s, const float* qf, __half* q, unsigned* error) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)s.rows * H * 576) return;
    int d = i % 576, h = (i / 576) % H, row = i / (576 * H);
    float x;
    if (d < 512) x = qf[((int64_t)row * H + h) * 512 + d];
    else {
        int j = (d - 512) / 2; float c, sn; ga::cs(s.positions[row], j, c, sn);
        const __half* qb = (const __half*)s.q_b + (int64_t)row * s.ld_qb + h * 256 + 192 + 2 * j;
        float u = __half2float(qb[0]), v = __half2float(qb[1]);
        x = (d & 1) ? v * c + u * sn : u * c - v * sn;
    }
    __half z = __float2half_rn(x);
    if (!isfinite(__half2float(z))) atomicOr(error, 4u);
    q[i] = z;
}
__global__ void pack_q_rows(G6AttnQoStep s, const float* qf, __half* q, unsigned* error) { pack_q_rows_body<H>(s, qf, q, error); }
#if G6_NR == 4
__global__ void pack_q_rows32(G6AttnQoStep s, const float* qf, __half* q, unsigned* error) { pack_q_rows_body<32>(s, qf, q, error); }
#endif
#if PF_QOABS16
// PF_QOABS16: the absorb GEMM writes q[row][h][0..511] in FP16 directly; this fills d 512..575 (pack_q_rows' RoPE branch)
#if PF_QOROPE
// PF_QOROPE: {cos, sin} per (pos, j) from ga::cs itself, once per device at create (owner) for pos < cap
__global__ void rope_table(float2* t, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float c, sn; ga::cs(i / 32, i % 32, c, sn); t[i] = make_float2(c, sn); }
}
#endif
template <int H>
__device__ __forceinline__ void pack_q_rope_body(const G6AttnQoStep& s, __half* q, unsigned* error
#if PF_QOROPE
                            , const float2* rope, int cap
#endif
                            ) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)s.rows * H * 64) return;
    int d = 512 + i % 64, h = (i / 64) % H, row = i / (64 * H);
    int j = (d - 512) / 2; float c, sn;
#if PF_QOROPE
    const int pos = s.positions[row];
    if (pos >= 0 && pos < cap) { float2 t = rope[pos * 32 + j]; c = t.x; sn = t.y; } else ga::cs(pos, j, c, sn);
#else
    ga::cs(s.positions[row], j, c, sn);
#endif
    const __half* qb = (const __half*)s.q_b + (int64_t)row * s.ld_qb + h * 256 + 192 + 2 * j;
    float u = __half2float(qb[0]), v = __half2float(qb[1]);
    float x = (d & 1) ? v * c + u * sn : u * c - v * sn;
    __half z = __float2half_rn(x);
    if (!isfinite(__half2float(z))) atomicOr(error, 4u);
    q[((int64_t)row * H + h) * 576 + d] = z;
}
#if PF_QOROPE
#define G6_QO_ROPE_PARAMS , const float2* rope, int cap
#define G6_QO_ROPE_ARGS , rope, cap
#else
#define G6_QO_ROPE_PARAMS
#define G6_QO_ROPE_ARGS
#endif
__global__ void pack_q_rope(G6AttnQoStep s, __half* q, unsigned* error G6_QO_ROPE_PARAMS) {
    pack_q_rope_body<H>(s, q, error G6_QO_ROPE_ARGS);
}
#if G6_NR == 4
__global__ void pack_q_rope32(G6AttnQoStep s, __half* q, unsigned* error G6_QO_ROPE_PARAMS) {
    pack_q_rope_body<32>(s, q, error G6_QO_ROPE_ARGS);
}
#endif
#undef G6_QO_ROPE_PARAMS
#undef G6_QO_ROPE_ARGS
#endif
#if defined(PF_QODUMP) && PF_QOROPE
__global__ void rope_check_k(const float2* t, int n, unsigned* bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c, sn; ga::cs(i / 32, i % 32, c, sn);
    if (__float_as_uint(c) != __float_as_uint(t[i].x) || __float_as_uint(sn) != __float_as_uint(t[i].y)) atomicAdd(bad, 1u);
}
inline void rope_check(const float2* t, int cap, cudaStream_t st) {
    unsigned* d = nullptr; unsigned h = 0; cudaMalloc(&d, 4); cudaMemsetAsync(d, 0, 4, st);
    rope_check_k<<<(cap * 32 + 255) / 256, 256, 0, st>>>(t, cap * 32, d);
    cudaMemcpyAsync(&h, d, 4, cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); cudaFree(d);
    int dev = 0; cudaGetDevice(&dev);
    fprintf(stderr, "ropechk dev=%d cap=%d entries=%d mismatches=%u\n", dev, cap, cap * 32, h);
}
#endif
#ifdef PF_QODUMP
// screen hooks: G6_QO_DUMP=1 prints an FNV64 of q per call; G6_QO_NANINJ=1 writes FP16 NaN into q[0] on every call
inline void qo_dump(const __half* q, int rows, int H, int context, cudaStream_t st) {
    static const bool dump = std::getenv("G6_QO_DUMP") != nullptr, inj = std::getenv("G6_QO_NANINJ") != nullptr;
    static std::atomic<int> injected{0};
    if (inj) {
        static const unsigned short nan = 0x7E00; g6h2d::copy((void*)q, &nan, 2, st);   // pageable source: pinned staging (h2d.h)
        if (injected.exchange(1) == 0) fprintf(stderr, "qodump NaN injected into q[0] (every call)\n");
    }
    if (!dump) return;
    cudaStreamCaptureStatus cs; cudaStreamIsCapturing(st, &cs);
    if (cs != cudaStreamCaptureStatusNone) { fprintf(stderr, "qodump capturing, skipped\n"); return; }
    const size_t n = (size_t)rows * H * 576 * 2;
    unsigned char* h = (unsigned char*)malloc(n);
    cudaStreamSynchronize(st); cudaMemcpy(h, q, n, cudaMemcpyDeviceToHost);
    unsigned long long f = 1469598103934665603ull;
    for (size_t k = 0; k < n; k++) { f ^= h[k]; f *= 1099511628211ull; }
    free(h);
    int dev = 0; cudaGetDevice(&dev);
    fprintf(stderr, "qodump dev=%d rows=%d ctx=%d hash=%016llx\n", dev, rows, context, f);
}
#endif
static size_t layout(G6AttnQo& p, void* base) {
    size_t used = 0; auto take = [&](size_t n) { size_t at = used; used = (used + n + 255) & ~size_t(255); return (void*)((uintptr_t)base + at); };
    size_t C = (size_t)p.cap, R = (size_t)p.max_rows, H = (size_t)p.heads;
#if PF_QOROPE
    p.rope = (float2*)take(C * 32 * sizeof(float2));   // first, so a non-owner (cap <= owner cap) sees the owner's table
#endif
    p.table = (__half*)take(C * 576 * 2); p.bad = (uint8_t*)take(C);
    p.q = (__half*)take(R * H * 576 * 2); p.ctx = (float*)take(R * H * 512 * 4); p.ctxh = (__half*)take(R * H * 512 * 2);
    return used;
}
static cudaError_t release_work(G6AttnQoWork* w) {
    if (!w || w->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return cudaSuccess;
    cublasDestroy(w->blas); auto e = cudaFree(w->data); delete w; return e;
}
inline int cap_limit = 0;   // g6_attn_qo_cap
// heads a state may hold in this build: 64, and 32 (pair-owner) in a TP4 build
inline bool heads_ok(int h) { return h == 64 || (G6_NR == 4 && h == 32); }
}   // namespace gqi

extern "C" int g6_attn_qo_create(G6AttnQo** out, const G6AttnBinding* b, const G6AttnTensor* uk, const G6AttnTensor* uv,
                                 int max_rows, void* s, G6AttnQo* owner) {
    if (!out || *out || !b || !b->storage || !uk || !uv || max_rows <= 0) return int(cudaErrorInvalidValue);
    auto shape = [](const G6AttnTensor* t, int64_t a, int64_t c, int64_t d) { return t->data && t->ndim == 3 && t->shape[0] == a && t->shape[1] == c && t->shape[2] == d; };
    const int heads = uk->ndim == 3 ? (int)uk->shape[0] : 0;
    if (!gqi::heads_ok(heads) || !shape(uk, heads, 192, 512) || !shape(uv, heads, 256, 512)) return int(cudaErrorInvalidValue);
    const char* k = std::getenv("G6_PF_QO");
    int kernel = (!k || !*k || !strcmp(k, "b")) ? 0 : !strcmp(k, "v1") ? 1 : !strcmp(k, "r1") ? 2 : -1;
#ifndef G6_QO_HAVE_R1
    if (kernel == 2) kernel = -1;
#endif
    if (kernel < 0) return int(cudaErrorInvalidValue);
    auto p = new G6AttnQo; p->kernel = kernel; p->cap = (b->capacity + 31) & ~31; p->max_rows = max_rows; p->heads = heads;
    if (gqi::cap_limit > 0 && p->cap > ((gqi::cap_limit + 31) & ~31)) p->cap = (gqi::cap_limit + 31) & ~31;
    int device = 0; cudaError_t e = cudaGetDevice(&device);
    if (e != cudaSuccess) { delete p; return int(e); }
    const size_t need = gqi::layout(*p, nullptr);
    if (owner) {
        auto w = owner->w;
        if (!w || w->device != device || p->cap > w->cap || max_rows > w->max_rows || need > w->bytes) { delete p; return int(cudaErrorInvalidValue); }
        p->w = w; w->refs.fetch_add(1, std::memory_order_relaxed);
    } else {
        auto w = new G6AttnQoWork; w->bytes = need; w->cap = p->cap; w->max_rows = max_rows; w->device = device;
        e = cudaMalloc(&w->data, need);
        if (e != cudaSuccess) { delete w; delete p; return int(e); }
        int rc = gp::blas_status(cublasCreate(&w->blas));
        if (!rc) rc = gp::blas_status(cublasSetMathMode(w->blas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        if (rc) { cudaFree(w->data); delete w; delete p; return rc; }
        p->w = w;
    }
    gqi::layout(*p, p->w->data);
    p->weight_bytes = (size_t)heads * (512 * 192 + 256 * 512) * 2;
    p->bytes = p->weight_bytes + (owner ? 0 : p->w->bytes);
    e = cudaMalloc(&p->storage, p->weight_bytes);
    if (e != cudaSuccess) { gqi::release_work(p->w); delete p; return int(e); }
    p->uk = (__half*)p->storage; p->uv = p->uk + (size_t)heads * 512 * 192;
    cudaStream_t st = (cudaStream_t)s;
    gqi::uk_half_t<<<(heads * 512 * 192 + 255) / 256, 256, 0, st>>>(uk->data, p->uk, heads * 512 * 192);
    gp::weight_half<<<(heads * 256 * 512 + 255) / 256, 256, 0, st>>>(uv->data, p->uv, heads * 256 * 512);
#if PF_QOROPE
    if (!owner) gqi::rope_table<<<(p->cap * 32 + 255) / 256, 256, 0, st>>>(p->rope, p->cap * 32);
#if PF_ROPE2
    if (!owner) {                                   // attn.cu decode/prefill kernels read the same table (ga::cs_t)
        const float2* t = p->rope; const int c = p->cap;
        void *at = nullptr, *ac = nullptr;     // pageable sources: pinned staging (h2d.h)
        e = cudaGetSymbolAddress(&at, ga::g_rope);
        if (e == cudaSuccess) e = cudaGetSymbolAddress(&ac, ga::g_rope_cap);
        if (e == cudaSuccess) e = g6h2d::copy(at, &t, sizeof t, st);
        if (e == cudaSuccess) e = g6h2d::copy(ac, &c, sizeof c, st);
        if (e != cudaSuccess) { cudaFree(p->storage); gqi::release_work(p->w); delete p; return int(e); }
    }
#endif
#ifdef PF_QODUMP
    if (!owner && std::getenv("G6_QO_ROPECHK")) gqi::rope_check(p->rope, p->cap, st);
#endif
#endif
    static bool configured = false;
    if (!configured)
    {
        gqob::configure<true>(); gqo::configure();
#if PF_QOBH
        gqob::configure_half<true>();
#endif
#ifdef G6_QO_HAVE_R1
        gqo_r1::configure();          // 100,016 B dynamic smem
#endif
        configured = true;
    }
    e = cudaGetLastError();
    if (e != cudaSuccess) { cudaFree(p->storage); gqi::release_work(p->w); delete p; return int(e); }
    *out = p;
    return 0;
}

namespace gqi {
inline double* prof_ms = nullptr;   // g6_attn_qo_prof
inline double prof_last = 0;
inline void prof_tick(cudaStream_t st, int k) {
    if (!prof_ms) return;
    cudaStreamSynchronize(st);
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    const double n = t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
    if (k >= 0) prof_ms[k] += n - prof_last;
    prof_last = n;
}
}   // namespace gqi
extern "C" void g6_attn_qo_prof(int on, double* ms) { gqi::prof_ms = on ? ms : nullptr; }
extern "C" int g6_attn_qo_run(G6AttnQo* p, G6AttnBinding* b, const G6AttnQoStep* s, void* stream) {
    if (!p || !b || !b->storage || !s) return int(cudaErrorInvalidValue);
    if (s->rows == 0) return 0;
    if (s->rows < 0 || s->rows > p->max_rows || s->context <= 0 || s->context > b->capacity || s->context > p->cap || !s->q_b ||
        s->ld_qb < p->heads * 256 ||
        !s->positions || !s->ids || !s->out) return int(cudaErrorInvalidValue);
    if ((uintptr_t)b->args.mla_cache % 16) return int(cudaErrorInvalidValue);   // 656-B records, 16-B cp.async gathers (B, R1)
    cudaStream_t st = (cudaStream_t)stream;
    const int R = s->rows, H = p->heads;
    int rc = gp::blas_status(cublasSetStream(p->w->blas, st));
    if (rc) return rc;
    const float one = 1.f, zero = 0.f;
    gqi::prof_tick(st, -1);
#if PF_QOABS16
    // absorb: q[row][h][0..511] = RN16(UK_h (512 x 192) x q_b_nope[row][h]); pack_q_rope fills the RoPE tail 512..575
    rc = gp::blas_status(cublasGemmStridedBatchedEx(p->w->blas, CUBLAS_OP_T, CUBLAS_OP_N, 512, R, 192, &one,
        p->uk, CUDA_R_16F, 192, 512 * 192, s->q_b, CUDA_R_16F, s->ld_qb, 256, &zero, p->q, CUDA_R_16F, H * 576, 576, H,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if (rc) return rc;
    unsigned* err = b->args.error;
#if G6_NR == 4
    if (H == 32)
        gqi::pack_q_rope32<<<(unsigned)(((int64_t)R * H * 64 + 255) / 256), 256, 0, st>>>(*s, p->q, err
#if PF_QOROPE
            , p->rope, p->cap
#endif
            );
    else
#endif
    gqi::pack_q_rope<<<(unsigned)(((int64_t)R * H * 64 + 255) / 256), 256, 0, st>>>(*s, p->q, err
#if PF_QOROPE
        , p->rope, p->cap
#endif
        );
#else
    // absorb: qf[row][h][512] = UK_h (512 x 192) x q_b_nope[row][h] (192); ctx doubles as the FP32 Q buffer
    rc = gp::blas_status(cublasGemmStridedBatchedEx(p->w->blas, CUBLAS_OP_T, CUBLAS_OP_N, 512, R, 192, &one,
        p->uk, CUDA_R_16F, 192, 512 * 192, s->q_b, CUDA_R_16F, s->ld_qb, 256, &zero, p->ctx, CUDA_R_32F, H * 512, 512, H,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if (rc) return rc;
    unsigned* err = b->args.error;
#if G6_NR == 4
    if (H == 32) gqi::pack_q_rows32<<<(unsigned)(((int64_t)R * H * 576 + 255) / 256), 256, 0, st>>>(*s, p->ctx, p->q, err);
    else
#endif
    gqi::pack_q_rows<<<(unsigned)(((int64_t)R * H * 576 + 255) / 256), 256, 0, st>>>(*s, p->ctx, p->q, err);
#endif
#ifdef PF_QODUMP
    gqi::qo_dump(p->q, R, H, s->context, st);
#endif
    cudaError_t e;
    gqi::prof_tick(st, 0);
    if (p->kernel == 0) {
        e = gqob::prepass((const uint8_t*)b->args.mla_cache, p->table, p->bad, s->context, st);
        if (e != cudaSuccess) return int(e);
        gqi::prof_tick(st, 1);
#if PF_QOBH
        gqob::HalfArgs a{p->q, p->table, p->bad, s->ids, s->positions, p->ctxh, err, R, H, s->context};
        e = gqob::run_half<true>(a, st);
#else
        gqob::Args a{p->q, p->table, p->bad, s->ids, s->positions, p->ctx, err, R, H, s->context};
        e = gqob::run<true>(a, st);
#endif
    } else {
        gqo::Args a{p->q, (const uint8_t*)b->args.mla_cache, s->ids, s->positions, p->ctx, err, R, H, s->context};
#ifdef G6_QO_HAVE_R1
        e = p->kernel == 2 ? gqo_r1::run(a, st) : gqo::run(a, st);
#else
        e = gqo::run(a, st);
#endif
    }
    if (e != cudaSuccess) return int(e);
    gqi::prof_tick(st, 2);
#if PF_QOBH
    if (p->kernel != 0)   // v1/R1 still produce FP32; preserve their cast.
#endif
    gp::cast_half<<<(R * H * 512 + 255) / 256, 256, 0, st>>>(p->ctx, p->ctxh, R * H * 512);
    // UV: out[row][h*256 + 0..255] = UV_h (256 x 512) x ctxh[row][h]
    rc = gp::blas_status(cublasGemmStridedBatchedEx(p->w->blas, CUBLAS_OP_T, CUBLAS_OP_N, 256, R, 512, &one,
        p->uv, CUDA_R_16F, 512, 256 * 512, p->ctxh, CUDA_R_16F, H * 512, 512, &zero, s->out, CUDA_R_16F, H * 256, 256, H,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if (rc) return rc;
    gqi::prof_tick(st, 3);
    return int(cudaGetLastError());
}
extern "C" int g6_attn_qo_warm(G6AttnQo* p, int rows, void* stream) {
    if (!p || rows <= 0 || rows > p->max_rows) return int(cudaErrorInvalidValue);
    cudaStream_t st = (cudaStream_t)stream;
    const int R = rows, H = p->heads;
    int rc = gp::blas_status(cublasSetStream(p->w->blas, st));
    if (rc) return rc;
    const float one = 1.f, zero = 0.f;
    // the same calls as run: absorb B = ctxh read as [R][16384] FP16 (run: q_b), UV C = q read as [R][16384] (run: out)
#if PF_QOABS16
    rc = gp::blas_status(cublasGemmStridedBatchedEx(p->w->blas, CUBLAS_OP_T, CUBLAS_OP_N, 512, R, 192, &one,
        p->uk, CUDA_R_16F, 192, 512 * 192, p->ctxh, CUDA_R_16F, H * 256, 256, &zero, p->q, CUDA_R_16F, H * 576, 576, H,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
    rc = gp::blas_status(cublasGemmStridedBatchedEx(p->w->blas, CUBLAS_OP_T, CUBLAS_OP_N, 512, R, 192, &one,
        p->uk, CUDA_R_16F, 192, 512 * 192, p->ctxh, CUDA_R_16F, H * 256, 256, &zero, p->ctx, CUDA_R_32F, H * 512, 512, H,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
    if (rc) return rc;
    rc = gp::blas_status(cublasGemmStridedBatchedEx(p->w->blas, CUBLAS_OP_T, CUBLAS_OP_N, 256, R, 512, &one,
        p->uv, CUDA_R_16F, 512, 256 * 512, p->ctxh, CUDA_R_16F, H * 512, 512, &zero, p->q, CUDA_R_16F, H * 256, 256, H,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if (rc) return rc;
    return int(cudaStreamSynchronize(st));
}
extern "C" size_t g6_attn_qo_bytes(const G6AttnQo* p) { return p ? p->bytes : 0; }
extern "C" void g6_attn_qo_cap(int cap) { gqi::cap_limit = cap > 0 ? cap : 0; }
extern "C" int g6_attn_qo_load(G6AttnQo* p, const G6AttnTensor* uk, const G6AttnTensor* uv, void* stream) {
    if (!p || !p->storage || !uk || !uv) return int(cudaErrorInvalidValue);
    auto shape = [](const G6AttnTensor* t, int64_t a, int64_t c, int64_t d) { return t->data && t->ndim == 3 && t->shape[0] == a && t->shape[1] == c && t->shape[2] == d; };
    const int heads = p->heads;
    if (!shape(uk, heads, 192, 512) || !shape(uv, heads, 256, 512)) return int(cudaErrorInvalidValue);
    cudaStream_t st = (cudaStream_t)stream;
    // create's two conversions, so the FP16 bytes equal a state created from these tensors
    gqi::uk_half_t<<<(heads * 512 * 192 + 255) / 256, 256, 0, st>>>(uk->data, p->uk, heads * 512 * 192);
    gp::weight_half<<<(heads * 256 * 512 + 255) / 256, 256, 0, st>>>(uv->data, p->uv, heads * 256 * 512);
    return int(cudaGetLastError());
}
extern "C" int g6_attn_qo_release(G6AttnQo* p) {
    if (!p) return 0;
    auto e = cudaFree(p->storage); auto w = gqi::release_work(p->w); delete p;
    return e != cudaSuccess ? int(e) : int(w);
}
