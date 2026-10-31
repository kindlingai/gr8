// Engram on the device (DeepSeek-V4.1-Flash-1017 layers 1 and 14): hash kernel, NVMe prefetch on a side stream, dequant-gather, gate.
// Contract: docs/DEEPSEEK-ENGRAM.md. The element maths is engram_math.h (host-tested against the official model.py); the row store is
// rowstore.h; the boot checks are engram_boot.h. Two things come from outside and are hooks: the rank-sum of the dequantized rows
// (the step driver's collective) and the wkv projection (the dense MXFP8 GEMV for L001/L014.engram.wkv, one weight stream for all rows).
//
//   step start  begin_step(rows, win_d, main):   side stream: hash(win) -> ids, D2H, one host fetch per layer (NVMe -> pinned staging),
//                                                an event per layer. Enqueue-only: capturable into a CUDA graph.
//   layer entry op(model, e, rows, T, main):     main waits the layer's event, dequant-gather (this rank's rows, zeros elsewhere),
//                                                hooks.reduce (sum over ranks, in place), hooks.project (kv = wkv(emb)), gate kernel
//                                                (T <- bf16(T + gate * value), in place on the BF16-valued FP32 trunk).
#pragma once
#include <atomic>
#include <cstdio>
#include <deque>
#include <string>
#include <utility>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#include "engram_boot.h"
#include "engram_math.h"
#include "engram_ring.h"
#include "act_quant.h"

// The attention code's tensor-core MXFP8 GEMM (engine/attn/ds41_attn.cu): Y[M][N] bf16 = fp8gemm(A codes [M][K] + E8M0 scales SA, W codes [N][K] + SW [N][K/32]).
// 0 = launched, 1 = shape (K % 128, N % 64, lda % 16, lsa % 4, ldy >= N and even), 2 = a pointer misaligned (A, W 16 B; SA, SW, Y 4 B); nothing launched
// on nonzero. Weak, so a build without that file still links: the null entry keeps the scalar hook (Config::project) and init says so.
extern "C" int ds41_gemm_mx_tc(const uint8_t* A, int lda, const uint8_t* SA, int lsa, const uint8_t* W, const uint8_t* SW, void* Y, int ldy,
                               int M, int N, int K, void* st) __attribute__((weak));

namespace ds41 { namespace engram {

constexpr int kWideMaxRows = 4096;    // W-row prefill: GR8_DS_PREFILL_W is MX..4096
constexpr int kDecodeRows = 32;   // MX: up to this many rows begin_step hashes with the decode launch <<<1, rows>>>
constexpr int kNotSup = 801;   // ds41::NOTSUP (cudaErrorNotSupported): the hook is not wired

// device flavour of the math policy of gate_from_sums: rsqrt as 1/sqrt (two correctly rounded ops, as torch on the CPU), expf
struct DevMath
{
    __host__ __device__ static float rsqrt(float x)
    {
#ifdef __CUDA_ARCH__
        return __fdiv_rn(1.f, __fsqrt_rn(x));
#else
        return 1.f / sqrtf(x);
#endif
    }
    __host__ __device__ static float sqrt(float x)
    {
#ifdef __CUDA_ARCH__
        return __fsqrt_rn(x);
#else
        return sqrtf(x);
#endif
    }
    __host__ __device__ static float exp(float x) { return expf(x); }
};

// ---- kernels --------------------------------------------------------------------------------------------------------------------

// one thread per row: the 2 x 24 table row ids of the window win[row][0..3] (token at pos, pos-1, pos-2, pos-3; -1 before the start).
// ids layout [2][rows][24]: one layer's ids are contiguous for the fetch.
__global__ void k_hash(HashParams hp, const uint32_t* cmap, int rows, const int32_t* win, uint32_t* ids, unsigned* bad)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    uint32_t loc[kNE][kIds];
    if (hash_row(hp, cmap, kVocab, win + 4 * r, loc)) atomicAdd(bad, 1u);
    hash_store_ids(rows, ids, r, loc);
}

// W-row chunks (prefill, rows > MX): the window without the ring's "rows + 3 <= kRing" limit. One thread per row, grid ceil(rows/256) x 256.
// k_win_wide reads tok / pos / ring and writes win[r][0..3]; k_ring_put_tail then writes the last kRing rows into the ring. Stream order:
// the get kernel first (the put kernel overwrites slots the get kernel still reads, engram_ring.h). `bad` counts rows whose position is not
// pos[0] + r (a chunk that is not one run of consecutive positions): the engine fails closed on it (status -2).
__global__ void k_win_wide(const int* tok, const int* pos, int rows, const int* ring, int32_t* win, unsigned* bad)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    if (!win_wide_row(tok, pos, ring, r, win + 4 * r)) atomicAdd(bad, 1u);
}
__global__ void k_ring_put_tail(int* ring, const int* tok, const int* pos, int rows)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) ring_put_tail(ring, tok, pos, rows, r);
}

// emb[row][g*256 + i] = bf16(float(e4m3 w) * float(e8m0 s)) for the ids this rank owns, 0 for the others.
// grid (rows, 3), block 256 = 8 ids x 32 chunks of 8 elements (a chunk lies inside one 32-element scale group).
__global__ void k_dequant(const uint32_t* ids, const uint8_t* stage, const uint8_t* s8, int64_t first, int64_t valid, int rows, __nv_bfloat16* emb)
{
    const int r = blockIdx.x, g = blockIdx.y * 8 + (threadIdx.x >> 5), c = threadIdx.x & 31;
    const int64_t loc = (int64_t)ids[(size_t)r * kIds + g] - first;
    uint32_t o[4] = {0, 0, 0, 0};
    if (loc >= 0 && loc < valid)
    {
        const uint2 w = *(const uint2*)(stage + ((size_t)r * kIds + g) * kRowBytes + c * 8);
        const uint8_t s = s8[(size_t)loc * kScaleBytes + (c >> 2)];
        const uint8_t* wb = (const uint8_t*)&w;
#pragma unroll
        for (int k = 0; k < 4; ++k) o[k] = (uint32_t)dequant1(wb[2 * k], s) | (uint32_t)dequant1(wb[2 * k + 1], s) << 16;
    }
    *(uint4*)(emb + (size_t)r * kEmbDim + (size_t)g * kRowBytes + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
}

// The W-row wkv's activation quantizer (act_quant.h): emb bf16 [rows][6144] -> codes u8 [rows][6144], E8M0 bytes [rows][192]. One warp per
// block of 32 (a lane per element), 8 warps per CTA: grid ceil(rows * 192 / 8).
__global__ void __launch_bounds__(256) k_act_quant_rows(const __nv_bfloat16* __restrict__ x, int rows, uint8_t* __restrict__ codes, uint8_t* __restrict__ sc)
{
    constexpr int kb = kEmbDim / 32;
    const int lane = threadIdx.x & 31;
    const long w = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= (long)rows * kb) return;
    const int r = (int)(w / kb), b = (int)(w % kb);
    const float v = __bfloat162float(x[(size_t)r * kEmbDim + b * 32 + lane]);
    float a = fabsf(v);
#pragma unroll
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, o));
    const int e = aq_exp(a);
    codes[(size_t)r * kEmbDim + b * 32 + lane] = aq_code(v, e);
    if (lane == 0) sc[(size_t)r * kb + b] = aq_scale_byte(e);
}

// w[i] = float(q[i]) * float(k[i]): the reference only ever uses the product of the two weights
__global__ void k_wprod(const __nv_bfloat16* q, const __nv_bfloat16* k, float* w, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = __fmul_rn(__bfloat162float(q[i]), __bfloat162float(k[i]));
}

// The gate and the write, one block per (row, stream). kv = bf16 [rows][25600]: stream j's key at [j*5120, +5120), the value at [4*5120, +5120).
//   T[row][j][i] <- bf16( T + gate * value[i] ), the product and the sum rounded separately (torch: h + gate.unsqueeze(-1) * value)
constexpr int kGateThreads = 512, kGatePer = kHid / kGateThreads;
static_assert(kHid % kGateThreads == 0, "5120 elements per block-row");
// The trunk is float, or __nv_bfloat16 when the engine is built with -DDS_MHC_TRUNK_BF16 (it only ever holds BF16-valued numbers, so the BF16 trunk is
// bit-identical): the load widens to FP32 (exact), the store keeps the 16 bits f32_to_bf16 makes (round_bf16 already did, NaN payload included).
// The body is shared; k_gate(float*) and op(float*) keep their own (non-template) symbols, k_gate / op for __nv_bfloat16* are added beside them.
__device__ __forceinline__ float eg_ld(const float* p) { return *p; }
__device__ __forceinline__ float eg_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void eg_st(float* p, float v) { *p = v; }
__device__ __forceinline__ void eg_st(__nv_bfloat16* p, float v) { *p = __ushort_as_bfloat16(f32_to_bf16(v)); }
template <class TT>
__device__ __forceinline__ void gate_body(TT* T, const __nv_bfloat16* kv, const float* wgt, int rows)
{
    __shared__ float red[kGateThreads / 32][3];
    __shared__ float gate_s;
    const int r = blockIdx.x, j = blockIdx.y, tid = threadIdx.x;
    TT* t = T + ((size_t)r * kStreams + j) * kHid;
    const __nv_bfloat16* key = kv + (size_t)r * kKvDim + (size_t)j * kHid;
    const __nv_bfloat16* val = kv + (size_t)r * kKvDim + (size_t)kStreams * kHid;
    const float* w = wgt + (size_t)j * kHid;
    float hv[kGatePer];
    float sh = 0.f, sk = 0.f, shk = 0.f;
#pragma unroll
    for (int q = 0; q < kGatePer; ++q)
    {
        const int i = tid + q * kGateThreads;
        hv[q] = eg_ld(t + i);
        const float kk = __bfloat162float(key[i]);
        sh += __fmul_rn(hv[q], hv[q]);
        sk += __fmul_rn(kk, kk);
        shk += __fmul_rn(__fmul_rn(hv[q], w[i]), kk);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        sh += __shfl_xor_sync(0xffffffffu, sh, o);
        sk += __shfl_xor_sync(0xffffffffu, sk, o);
        shk += __shfl_xor_sync(0xffffffffu, shk, o);
    }
    if ((tid & 31) == 0) { red[tid >> 5][0] = sh; red[tid >> 5][1] = sk; red[tid >> 5][2] = shk; }
    __syncthreads();
    if (tid == 0)
    {
        float a = 0.f, b = 0.f, c = 0.f;
        for (int k = 0; k < kGateThreads / 32; ++k) { a += red[k][0]; b += red[k][1]; c += red[k][2]; }
        gate_s = gate_from_sums<DevMath>(a, b, c, 1e-20f);
    }
    __syncthreads();
    const float gate = gate_s;
#pragma unroll
    for (int q = 0; q < kGatePer; ++q)
    {
        const int i = tid + q * kGateThreads;
        eg_st(t + i, round_bf16(__fadd_rn(hv[q], __fmul_rn(gate, __bfloat162float(val[i])))));
    }
}
__global__ void __launch_bounds__(kGateThreads) k_gate(float* T, const __nv_bfloat16* kv, const float* wgt, int rows) { gate_body<float>(T, kv, wgt, rows); }
__global__ void __launch_bounds__(kGateThreads) k_gate(__nv_bfloat16* T, const __nv_bfloat16* kv, const float* wgt, int rows) { gate_body<__nv_bfloat16>(T, kv, wgt, rows); }

#define EG_CK(call, err) do { const cudaError_t e_ = (call); if (e_ != cudaSuccess) { *(err) = std::string("engram: " #call ": ") + cudaGetErrorString(e_); return false; } } while (0)

// ---- the engine-side object --------------------------------------------------------------------------------------------------------

// Hooks come from the step driver (reduce) and the dense GEMV wrapper (project). 0 = ok, else an error code (kNotSup if unwired).
struct EngramHooks
{
    // sums emb (BF16 [rows][6144], this rank's partial: zeros for rows owned by other ranks) over all ranks in place, fp32 in rank order
    int (*reduce)(void* model, int e, int rows, void* stream, __nv_bfloat16* emb) = nullptr;
    // kv = wkv(emb): MXFP8 [25600][6144] weights of layer e, input BF16 [rows][6144], output BF16 [rows][25600]; one weight stream for all rows
    int (*project)(void* model, int e, int rows, void* stream, const __nv_bfloat16* emb, __nv_bfloat16* kv) = nullptr;
};

struct Config
{
    std::string prefix;              // weights PREFIX (PREFIX.engram.json, .cmap, .L###.w8/.s8)
    int rank = 0, nr = 1;
    int max_rows = 8;                // MX; a prefill build (GR8_DS_PREFILL_W) sets its W, up to 4096
    Knobs knobs;                     // read_knobs()
    const __nv_bfloat16* q_w[kNE] = {};   // device [4][5120] bf16: layer e's engram.q_weight / engram.k_weight (the image)
    const __nv_bfloat16* k_w[kNE] = {};
    EngramHooks hooks;
    // optional, for chunks of more than kDecodeRows rows: layer e's engram.wkv as the image's MXFP8 pair (device): codes u8 [25600][6144], E8M0 scales u8
    // [25600][192]. Set (and ds41_gemm_mx_tc linked) = those chunks use the tensor-core GEMM; else they use hooks.project.
    const uint8_t* wkv_codes[kNE] = {};
    const uint8_t* wkv_scales[kNE] = {};
};

class Engram
{
public:
    ~Engram() { release(); }

    // Fail-closed init (the caller stops the boot on false): sidecar and cmap checks, s8 + cmap + q*k to the device, row store open.
    bool init(const Config& c, std::string* err)
    {
        cfg_ = c;
        if (c.max_rows < 1 || c.max_rows > kWideMaxRows) { *err = "engram: max_rows outside 1.." + std::to_string(kWideMaxRows); return false; }
        if (!load_sidecar(c.prefix, c.rank, c.nr, &sc_, err)) return false;
        for (int e = 0; e < kNE; ++e)
            if (!c.q_w[e] || !c.k_w[e]) { *err = "engram: q_weight / k_weight pointers of layer " + std::to_string(kLayerIds[e]) + " missing"; return false; }

        EG_CK(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking), err);
        EG_CK(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming), err);
        for (int e = 0; e < kNE; ++e) EG_CK(cudaEventCreateWithFlags(&done_[e], cudaEventDisableTiming), err);
        const size_t mr = (size_t)c.max_rows;
        EG_CK(cudaMalloc(&cmap_d_, sizeof(uint32_t) * kVocab), err);
        EG_CK(cudaMemcpy(cmap_d_, sc_.cmap.data(), sizeof(uint32_t) * kVocab, cudaMemcpyHostToDevice), err);
        EG_CK(cudaMalloc(&ids_d_, kNE * mr * kIds * sizeof(uint32_t)), err);
        EG_CK(cudaHostAlloc(&ids_h_, kNE * mr * kIds * sizeof(uint32_t), cudaHostAllocDefault), err);
        EG_CK(cudaHostAlloc(&bad_h_, 2 * sizeof(unsigned), cudaHostAllocMapped), err);
        EG_CK(cudaHostGetDevicePointer(&bad_d_, bad_h_, 0), err);
        bad_h_[0] = bad_h_[1] = 0;
        EG_CK(cudaMalloc(&emb_d_[0], kNE * mr * kEmbDim * sizeof(__nv_bfloat16)), err);
        emb_d_[1] = emb_d_[0] + mr * kEmbDim;
        EG_CK(cudaMalloc(&kv_d_, mr * kKvDim * sizeof(__nv_bfloat16)), err);
        tc_ = ds41_gemm_mx_tc != nullptr && c.max_rows > kDecodeRows && c.wkv_codes[0] && c.wkv_scales[0] && c.wkv_codes[1] && c.wkv_scales[1];
        if (c.max_rows > kDecodeRows)
            fprintf(stderr, "engram: wkv of W-row chunks (> %d rows): %s\n", kDecodeRows,
                    tc_ ? "tensor-core GEMM (ds41_gemm_mx_tc)" : ds41_gemm_mx_tc ? "scalar hook (the wkv tensors were not given)" : "scalar hook (ds41_gemm_mx_tc is not linked)");
        if (tc_)
        {
            EG_CK(cudaMalloc(&aq_codes_, mr * kEmbDim), err);
            EG_CK(cudaMalloc(&aq_scales_, mr * (kEmbDim / 32)), err);
        }
        EG_CK(cudaMalloc(&wgt_d_[0], kNE * kStreams * kHid * sizeof(float)), err);
        wgt_d_[1] = wgt_d_[0] + kStreams * kHid;
        for (int e = 0; e < kNE; ++e)
        {
            EG_CK(cudaHostAlloc(&stage_h_[e], mr * kIds * kRowBytes, cudaHostAllocMapped), err);
            EG_CK(cudaHostGetDevicePointer(&stage_d_[e], stage_h_[e], 0), err);
            const TableInfo& t = sc_.t[e];
            EG_CK(cudaMalloc(&s8_d_[e], (size_t)t.part * kScaleBytes), err);
            if (!load_file_to_device(t.s8, (uint64_t)t.part * kScaleBytes, s8_d_[e], err)) return false;
            k_wprod<<<(kStreams * kHid + 255) / 256, 256>>>(c.q_w[e], c.k_w[e], wgt_d_[e], kStreams * kHid);
        }
        EG_CK(cudaGetLastError(), err);
        EG_CK(cudaDeviceSynchronize(), err);

        std::vector<TableFile> files;
        for (int e = 0; e < kNE; ++e) files.push_back({sc_.t[e].w8, sc_.t[e].first, sc_.t[e].valid});
        if (!store_.open(files, c.knobs.store, err)) return false;
        for (int e = 0; e < kNE; ++e)
            fprintf(stderr, "# rank %d: engram rowstore layer %03d file %s %s block %zu\n",
                    c.rank, kLayerIds[e], sc_.t[e].w8.c_str(),
                    store_.direct(e) ? "direct=1" : "fallback=buffered", store_.block());
        hooks_ = c.hooks;
        for (int e = 0; e < kNE; ++e) { wkv_codes_[e] = c.wkv_codes[e]; wkv_scales_[e] = c.wkv_scales[e]; }
        verbose_ = c.knobs.verbose;
        if (c.knobs.lookahead)
        {
            warm_win_in_.assign(mr * kNGram, -1);
            warm_win_work_.assign(mr * kNGram, -1);
            warm_ids_.assign(mr * kIds, 0);
            warm_stop_ = false;
            warm_planned_ = false;
            warm_armed_ = false;
            warm_e1_left_ = 0;
            warm_pending_.store(false, std::memory_order_relaxed);
            pthread_attr_t attr;
            int rc = pthread_attr_init(&attr);
            if (rc) { *err = std::string("engram: warm pthread_attr_init: ") + strerror(rc); return false; }
            rc = pthread_attr_setstacksize(&attr, kWarmStackBytes);
            if (!rc) rc = pthread_create(&warm_th_, &attr, &Engram::warm_entry, this);
            pthread_attr_destroy(&attr);
            if (rc) { *err = std::string("engram: warm pthread_create: ") + strerror(rc); return false; }
            warm_on_ = true;
            fprintf(stderr, "# rank %d: engram lookahead=1 (shared rowstore L001 cache warm, max_rows %d)\n", c.rank, c.max_rows);
        }
        ready_ = true;
        return true;
    }

    void set_hooks(const EngramHooks& h) { hooks_ = h; }
    bool ready() const { return ready_; }

    // Solo W-row prefill only (GR8_DS_ENGRAM_LOOKAHEAD=1): copy the next chunk's 4-gram token windows
    // before begin_step. After this step's exact L014 fetch(es) complete on side_ (e1_fetches = 1 on
    // single-lane prefill, 2 on two-lane prefill after lane B's L014), the background worker hashes
    // L001 ids and calls store_.fetch(0, ..., nullptr) to warm the shared RowCache while the later layers run.
    void plan_next_warm(const int* prompt, int prompt_rows, int next_first, int e1_fetches = 1)
    {
        if (!ready_ || !warm_on_ || !prompt || prompt_rows < 2 || next_first < 0 || next_first >= prompt_rows - 1 || e1_fetches < 1) return;
        const int rows = std::min(cfg_.max_rows, prompt_rows - 1 - next_first);
        if (rows < 1) return;
        std::lock_guard<std::mutex> lk(warm_mu_);
        warm_rows_ = rows;
        for (int r = 0; r < rows; ++r)
            for (int k = 0; k < kNGram; ++k)
                warm_win_in_[(size_t)r * kNGram + k] = (next_first + r - k >= 0) ? prompt[next_first + r - k] : -1;
        warm_planned_ = true;
        warm_armed_ = false;
        warm_e1_left_ = e1_fetches;
        warm_pending_.store(true, std::memory_order_release);
    }

    // The window of a W-row chunk (rows > MX; consecutive positions of one sequence) and the ring update, on `main`, before begin_step.
    // tok / pos: device int [rows] (the chunk's tokens and absolute positions); ring: the kRing-slot token ring; win: int32 [rows][4].
    // Two kernels in the order get, put (engram_ring.h). Chunks of up to MX rows use the decode kernel (ds41_engram_begin), not this one.
    int window_wide(const int* tok, const int* pos, int rows, int* ring, int32_t* win, cudaStream_t main)
    {
        if (!ready_ || rows < 1 || rows > cfg_.max_rows) return cudaErrorInvalidValue;
        const int blocks = (rows + 255) / 256;
        k_win_wide<<<blocks, 256, 0, main>>>(tok, pos, rows, ring, win, bad_d_ + 1);
        if (const cudaError_t r = cudaGetLastError()) return r;
        k_ring_put_tail<<<blocks, 256, 0, main>>>(ring, tok, pos, rows);
        return cudaGetLastError();
    }

    // Enqueue the step's hash + prefetch on the side stream (after everything already on `main`). win_d: int32 [rows][4], the tokens at
    // positions pos, pos-1, pos-2, pos-3 of each row (-1 before the start of the sequence). Capturable; the Job of (layer, rows) is stable.
    int begin_step(int rows, const int32_t* win_d, cudaStream_t main)
    {
        if (!ready_ || rows < 1 || rows > cfg_.max_rows) return cudaErrorInvalidValue;
        cudaError_t r;
        if ((r = cudaEventRecord(fork_, main)) != cudaSuccess) return r;
        if ((r = cudaStreamWaitEvent(side_, fork_, 0)) != cudaSuccess) return r;
        if (rows <= kDecodeRows) k_hash<<<1, rows, 0, side_>>>(kHashParams, cmap_d_, rows, win_d, ids_d_, bad_d_);   // the decode launch
        else k_hash<<<(rows + 127) / 128, 128, 0, side_>>>(kHashParams, cmap_d_, rows, win_d, ids_d_, bad_d_);
        if ((r = cudaGetLastError()) != cudaSuccess) return r;
        if ((r = cudaMemcpyAsync(ids_h_, ids_d_, kNE * (size_t)rows * kIds * sizeof(uint32_t), cudaMemcpyDeviceToHost, side_)) != cudaSuccess) return r;
        for (int e = 0; e < kNE; ++e)
        {
            if ((r = cudaLaunchHostFunc(side_, &Engram::fetch_cb, job(e, rows))) != cudaSuccess) return r;
            if ((r = cudaEventRecord(done_[e], side_)) != cudaSuccess) return r;
        }
        return 0;
    }

    // The ST_ENGRAM op of layer slot e (0 = layer 1, 1 = layer 14). T = the trunk [rows][4][5120], BF16-valued, edited in place: FP32, or BF16 under
    // DS_MHC_TRUNK_BF16 (the overload below; same body).
    int op(void* model, int e, int rows, float* T, cudaStream_t main) { return op_t(model, e, rows, T, main); }
    int op(void* model, int e, int rows, __nv_bfloat16* T, cudaStream_t main) { return op_t(model, e, rows, T, main); }
    template <class TT>
    int op_t(void* model, int e, int rows, TT* T, cudaStream_t main)
    {
        if (!hooks_.reduce || !hooks_.project) return kNotSup;
        if (!ready_ || e < 0 || e >= kNE || rows < 1 || rows > cfg_.max_rows) return cudaErrorInvalidValue;
        cudaError_t r;
        if ((r = cudaStreamWaitEvent(main, done_[e], 0)) != cudaSuccess) return r;
        k_dequant<<<dim3(rows, kIds / 8), 256, 0, main>>>(ids_d_ + (size_t)e * rows * kIds, stage_d_[e], s8_d_[e], sc_.t[e].first, sc_.t[e].valid, rows, emb_d_[e]);
        if ((r = cudaGetLastError()) != cudaSuccess) return r;
        if (int x = hooks_.reduce(model, e, rows, main, emb_d_[e])) return x;
        if (!(tc_ && rows > kDecodeRows && project_wide(e, rows, main)))   // chunks of <= kDecodeRows rows: the hook, as in the decode build
            if (int x = hooks_.project(model, e, rows, main, emb_d_[e], kv_d_)) return x;
        k_gate<<<dim3(rows, kStreams), kGateThreads, 0, main>>>(T, kv_d_, wgt_d_[e], rows);
        return cudaGetLastError();
    }

    // kv = wkv(emb) for a W-row chunk: the activation quantizer, then the tensor-core GEMM. True = launched; false = the GEMM refused the shape
    // or an alignment (nothing launched): the caller runs the hook. The refusal is a bug of the constants, so it says so once.
    bool project_wide(int e, int rows, cudaStream_t main)
    {
        k_act_quant_rows<<<(unsigned)(((long)rows * (kEmbDim / 32) + 7) / 8), 256, 0, main>>>(emb_d_[e], rows, aq_codes_, aq_scales_);
        if (cudaGetLastError() != cudaSuccess) return false;
        const int rc = ds41_gemm_mx_tc(aq_codes_, kEmbDim, aq_scales_, kEmbDim / 32, wkv_codes_[e], wkv_scales_[e], kv_d_, kKvDim, rows, kKvDim, kEmbDim, main);
        if (rc && !warned_) { warned_ = true; fprintf(stderr, "engram: ds41_gemm_mx_tc refused the wkv GEMM (code %d, %d rows): the scalar hook runs instead\n", rc, rows); }
        return rc == 0;
    }

    // The engine fails closed: 0 = fine, else the errno of the first failed NVMe read (> 0), -1 for a window token outside the
    // vocabulary, -2 for a W-row chunk whose rows are not consecutive positions. Read it after the step's stream sync.
    int status() const
    {
        const int f = fail_.load();
        if (f) return f;
        if (!ready_) return 0;
        if (((volatile unsigned*)bad_h_)[0]) return -1;
        return ((volatile unsigned*)bad_h_)[1] ? -2 : 0;
    }
    static std::string status_text(int es)
    {
        if (es > 0) return std::string("NVMe read failed: ") + strerror(es);
        return es == -2 ? "the rows of a W-row chunk are not consecutive positions" : "a window token is outside the vocabulary";
    }
    std::string stats_line() const
    {
        const FetchStats& s = store_.totals();
        char b[320];
        snprintf(b, sizeof b, "engram: %llu fetches, %llu rows asked (%llu foreign), %llu hits, %llu miss (%llu distinct), %llu reads, %.1f MiB read, io %.1f ms total",
                 (unsigned long long)store_.fetches(), (unsigned long long)s.req, (unsigned long long)s.foreign, (unsigned long long)s.hits,
                 (unsigned long long)s.miss_req, (unsigned long long)s.miss_uniq, (unsigned long long)s.blocks, s.bytes / 1048576.0, s.io_us / 1000.0);
        return b;
    }
    const RowStore& store() const { return store_; }
    const Sidecar& sidecar() const { return sc_; }
    // the device buffers, for tests
    const uint32_t* ids_device() const { return ids_d_; }

    // Two-lane W prefill (GR8_DS_PF_LANES=2): a second per-step set (ids, the staging rows, the fork / done events) for lane B's chunk, so lane
    // A's begin_step -> op of the same quantum reads its own ids while lane B's begin_step runs in between. use_lane(L) makes set L the one
    // begin_step / op use (host only; a queued fetch keeps the buffers of its own set). The side stream, the scratch and the tables stay one.
    // Never called = one set.
    bool add_lane(std::string* err)
    {
        if (!ready_ || alt_.ids_d) { *err = "engram: add_lane before init or twice"; return false; }
        const size_t mr = (size_t)cfg_.max_rows;
        EG_CK(cudaEventCreateWithFlags(&alt_.fork, cudaEventDisableTiming), err);
        for (int e = 0; e < kNE; ++e) EG_CK(cudaEventCreateWithFlags(&alt_.done[e], cudaEventDisableTiming), err);
        EG_CK(cudaMalloc(&alt_.ids_d, kNE * mr * kIds * sizeof(uint32_t)), err);
        EG_CK(cudaHostAlloc(&alt_.ids_h, kNE * mr * kIds * sizeof(uint32_t), cudaHostAllocDefault), err);
        for (int e = 0; e < kNE; ++e)
        {
            EG_CK(cudaHostAlloc(&alt_.stage_h[e], mr * kIds * kRowBytes, cudaHostAllocMapped), err);
            EG_CK(cudaHostGetDevicePointer(&alt_.stage_d[e], alt_.stage_h[e], 0), err);
        }
        return true;
    }
    int use_lane(int L)
    {
        if (L != 0 && L != 1) return cudaErrorInvalidValue;
        if (L == lane_) return 0;
        if (!alt_.ids_d) return cudaErrorInvalidValue;
        std::swap(fork_, alt_.fork); std::swap(ids_d_, alt_.ids_d); std::swap(ids_h_, alt_.ids_h);
        for (int e = 0; e < kNE; ++e) { std::swap(done_[e], alt_.done[e]); std::swap(stage_h_[e], alt_.stage_h[e]); std::swap(stage_d_[e], alt_.stage_d[e]); }
        lane_ = L;
        return 0;
    }
    const __nv_bfloat16* kv_device() const { return kv_d_; }

private:
    struct Job { Engram* self; int e, rows; uint32_t* ids_h; uint8_t* stage_h; };   // the buffers of the set the job was made for

    static void CUDART_CB fetch_cb(void* p) { Job* j = (Job*)p; j->self->do_fetch(j->e, j->rows, j->ids_h, j->stage_h); }
    void do_fetch(int e, int rows, uint32_t* ids_h, uint8_t* stage_h)
    {
        FetchStats st;
        const int r = store_.fetch(e, ids_h + (size_t)e * rows * kIds, (size_t)rows * kIds, stage_h, &st);
        if (r) { int z = 0; fail_.compare_exchange_strong(z, r); }
        if (verbose_) fprintf(stderr, "[L%03d] %s\n", kLayerIds[e], RowStore::line(st).c_str());
        if (e == 1 && warm_on_ && warm_pending_.load(std::memory_order_acquire))
        {
            bool notify = false;
            {
                std::lock_guard<std::mutex> lk(warm_mu_);
                if (warm_planned_ && warm_e1_left_ > 0 && --warm_e1_left_ == 0) { warm_armed_ = true; notify = true; }
            }
            if (notify) warm_cv_.notify_one();
        }
    }
    static constexpr size_t kWarmStackBytes = 1u << 20;   // 1 MiB worker stack
    static void* warm_entry(void* p) { static_cast<Engram*>(p)->warm_loop(); return nullptr; }
    void warm_loop()
    {
        for (;;)
        {
            int rows = 0;
            {
                std::unique_lock<std::mutex> lk(warm_mu_);
                warm_cv_.wait(lk, [this] { return warm_stop_ || (warm_armed_ && warm_planned_); });
                if (warm_stop_) return;
                rows = warm_rows_;
                warm_win_in_.swap(warm_win_work_);
                warm_planned_ = false;
                warm_armed_ = false;
                warm_e1_left_ = 0;
                warm_pending_.store(false, std::memory_order_relaxed);
            }
            bool bad = false;
            for (int r = 0; r < rows; ++r)
            {
                uint32_t loc[kNE][kIds];
                if (hash_row(kHashParams, sc_.cmap.data(), kVocab, warm_win_work_.data() + (size_t)r * kNGram, loc)) { bad = true; break; }
                memcpy(warm_ids_.data() + (size_t)r * kIds, loc[0], kIds * sizeof(uint32_t));
            }
            if (bad) continue;
            FetchStats st;
            const int rc = store_.fetch(0, warm_ids_.data(), (size_t)rows * kIds, nullptr, &st);
            (void)rc;   // speculative cache warm never sets exact fail_; exact do_fetch is authoritative
            if (verbose_) fprintf(stderr, "[L001-warm] %s\n", RowStore::line(st).c_str());
        }
    }
    void stop_warm()
    {
        if (!warm_on_) return;
        {
            std::lock_guard<std::mutex> lk(warm_mu_);
            warm_stop_ = true;
            warm_planned_ = false;
            warm_armed_ = false;
            warm_e1_left_ = 0;
            warm_pending_.store(false, std::memory_order_relaxed);
        }
        warm_cv_.notify_all();
        pthread_join(warm_th_, nullptr);
        warm_on_ = false;
    }
    Job* job(int e, int rows)
    {
        for (Job& j : jobs_) if (j.e == e && j.rows == rows && j.ids_h == ids_h_) return &j;
        jobs_.push_back({this, e, rows, ids_h_, stage_h_[e]});
        return &jobs_.back();
    }

    bool load_file_to_device(const std::string& path, uint64_t bytes, void* dev, std::string* err)
    {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) { *err = path + ": open failed"; return false; }
        constexpr size_t kChunk = 16u << 20;
        void* buf = nullptr;
        if (cudaHostAlloc(&buf, kChunk, cudaHostAllocDefault) != cudaSuccess) { ::close(fd); *err = "engram: pinned buffer for " + path; return false; }
        bool ok = true;
        for (uint64_t x = 0; x < bytes && ok;)
        {
            const size_t n = (size_t)std::min<uint64_t>(kChunk, bytes - x);
            const ssize_t got = ::pread(fd, buf, n, (off_t)x);
            if (got != (ssize_t)n) { *err = path + ": short read at byte " + std::to_string(x); ok = false; break; }
            if (cudaMemcpy((char*)dev + x, buf, n, cudaMemcpyHostToDevice) != cudaSuccess) { *err = "engram: copy of " + path + " to the device failed"; ok = false; }
            x += n;
        }
        cudaFreeHost(buf);
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // the scales live on the device now: do not keep a second copy in the page cache
        ::close(fd);
        return ok;
    }

    void release()
    {
        if (side_) cudaStreamSynchronize(side_);
        stop_warm();
        use_lane(0);
        if (alt_.fork) cudaEventDestroy(alt_.fork);
        if (alt_.ids_d) cudaFree(alt_.ids_d);
        if (alt_.ids_h) cudaFreeHost(alt_.ids_h);
        for (int e = 0; e < kNE; ++e)
        {
            if (alt_.done[e]) cudaEventDestroy(alt_.done[e]);
            if (alt_.stage_h[e]) cudaFreeHost(alt_.stage_h[e]);
        }
        alt_ = LaneSet{};
        for (int e = 0; e < kNE; ++e)
        {
            if (stage_h_[e]) cudaFreeHost(stage_h_[e]);
            if (s8_d_[e]) cudaFree(s8_d_[e]);
            if (done_[e]) cudaEventDestroy(done_[e]);
            stage_h_[e] = nullptr; s8_d_[e] = nullptr; done_[e] = nullptr;
        }
        if (fork_) cudaEventDestroy(fork_);
        if (side_) cudaStreamDestroy(side_);
        if (cmap_d_) cudaFree(cmap_d_);
        if (ids_d_) cudaFree(ids_d_);
        if (ids_h_) cudaFreeHost(ids_h_);
        if (bad_h_) cudaFreeHost(bad_h_);
        if (aq_codes_) cudaFree(aq_codes_);
        if (aq_scales_) cudaFree(aq_scales_);
        aq_codes_ = aq_scales_ = nullptr;
        if (emb_d_[0]) cudaFree(emb_d_[0]);
        if (kv_d_) cudaFree(kv_d_);
        if (wgt_d_[0]) cudaFree(wgt_d_[0]);
        fork_ = nullptr; side_ = nullptr; cmap_d_ = nullptr; ids_d_ = nullptr; ids_h_ = nullptr; bad_h_ = nullptr;
        emb_d_[0] = emb_d_[1] = nullptr; kv_d_ = nullptr; wgt_d_[0] = wgt_d_[1] = nullptr;
        ready_ = false;
    }

    Config cfg_;
    Sidecar sc_;
    RowStore store_;
    EngramHooks hooks_;
    bool ready_ = false, verbose_ = false, tc_ = false, warned_ = false;
    uint8_t *aq_codes_ = nullptr, *aq_scales_ = nullptr;    // the quantized embedding of a W-row chunk (tc_ only)
    const uint8_t* wkv_codes_[kNE] = {};
    const uint8_t* wkv_scales_[kNE] = {};
    cudaStream_t side_ = nullptr;
    cudaEvent_t fork_ = nullptr, done_[kNE] = {};
    struct LaneSet { cudaEvent_t fork = nullptr, done[kNE] = {}; uint32_t *ids_d = nullptr, *ids_h = nullptr; uint8_t *stage_h[kNE] = {}, *stage_d[kNE] = {}; };
    LaneSet alt_;   // add_lane: the set of the lane not in use (use_lane swaps it with the members above)
    int lane_ = 0;
    uint32_t* cmap_d_ = nullptr;
    uint32_t* ids_d_ = nullptr;
    uint32_t* ids_h_ = nullptr;
    unsigned *bad_h_ = nullptr, *bad_d_ = nullptr;
    uint8_t* stage_h_[kNE] = {};
    uint8_t* stage_d_[kNE] = {};
    uint8_t* s8_d_[kNE] = {};
    __nv_bfloat16* emb_d_[kNE] = {};
    __nv_bfloat16* kv_d_ = nullptr;
    float* wgt_d_[kNE] = {};
    std::deque<Job> jobs_;
    std::mutex warm_mu_;
    std::condition_variable warm_cv_;
    pthread_t warm_th_{};
    std::vector<int32_t> warm_win_in_, warm_win_work_;
    std::vector<uint32_t> warm_ids_;
    int warm_rows_ = 0, warm_e1_left_ = 0;
    bool warm_on_ = false, warm_planned_ = false, warm_armed_ = false, warm_stop_ = false;
    std::atomic<bool> warm_pending_{false};
    std::atomic<int> fail_{0};
};
#undef EG_CK

}}  // namespace ds41::engram
