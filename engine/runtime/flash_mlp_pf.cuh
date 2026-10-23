// Flash prefill MLP & MoE block (INTERFACES.md §8.2-8.3, PREFILL-PLAN.md §2 P2(a)/(b)):
//   bind_mlp_prefill(m, P, i, find, st) / bind_mlp_prefill(m, P, st)
//   pf_mlp_block_gemm(m, P, i, M, st)
//
// Input:  P.xn2 [M][4096] FP32 (router input, no BF16 rounding before top-k), P.x2 [M][4096] FP16
// Output: P.part[1] [M][4096] BF16 (this rank's MLP partial; caller publishes & sums)
//
// Dense layers (0..2): W8A8 MXFP8 GEMMs (bindp::launch_pair(gate, up), k_pf_dtr_mid, down) on exact FP8 tile copies
//   (dtr::make_fp8) or raw K4 trellis tiles (k_gemm_tr); gate and up share mlp_in (rotation 3), so k_xq8_rot runs once.
// MoE layers (3..45): FP32 router GEMM (launch_router: k_router_tile, 0-ULP bit-identical to flash_mlp::k_linear), trellis shared
//   expert at index NEXP=288 (L.shx FP8 GEMM copy or L.exs trellis pass), and bindp::launch_experts<g6nr::EB> (top-8
//   sigmoid+bias routing with ROUTE_SCALE=2.5, grouped segments <= 64 rows, TR3->E4M3 fc1/fc2, SwiGLU clamp 10.0).
#pragma once
#if G6_MODEL_FLASH
#include <map>
#include <memory>
#include <vector>
#include "rank.cuh"
#include "dtr_find.h"
#include "../kernels/bind_prefill.cuh"
#include "flash_route_split.cuh"

namespace g6rt
{

static_assert(flash_route_split::SLOTS == geom::TOPK, "G6_PF_ROUTER_SPLIT record carries TOPK slots");

namespace flash_mlp_pf
{

// Multi-row FP32 x BF16 -> FP32 linear kernel for router.w [NEXP=288][HID=4096] (and small BF16 projections).
// Each warp computes one output channel n across MR rows simultaneously, using the exact same per-row FMA order and
// warp_sum reduction as flash_mlp::k_linear (0-ULP bit-identical to flash_mlp::k_linear at any M, 4x fewer weight reads).
template <int MR = 4>
__global__ void __launch_bounds__(256) k_linear_mr(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                                   float* __restrict__ y, int K, int N, int M)
{
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), m0 = blockIdx.y * MR, lane = threadIdx.x & 31;
    if (n >= N || m0 >= M) return;
    const uint16_t* wn = w + static_cast<size_t>(n) * K;
    float v[MR];
    #pragma unroll
    for (int r = 0; r < MR; ++r) v[r] = 0.f;
    for (int k = lane; k < K; k += 32)
    {
        const float wv = flash_mlp::bf(__ldg(wn + k));
        #pragma unroll
        for (int r = 0; r < MR; ++r)
            if (m0 + r < M)
                v[r] = fmaf(__ldg(x + static_cast<size_t>(m0 + r) * K + k), wv, v[r]);
    }
    #pragma unroll
    for (int r = 0; r < MR; ++r)
    {
        v[r] = flash_mlp::warp_sum(v[r]);
        if (!lane && m0 + r < M) y[static_cast<size_t>(m0 + r) * N + n] = v[r];
    }
}

// Register-tiled router GEMM, 0-ULP bit-identical to flash_mlp::k_linear: lane l of a warp owns the lane class
// k = l + 32 j of every (row, channel) output in the warp's R x C tile and runs the same sequential fmaf chain over
// j = 0 .. K/32-1, then the same warp_sum per output. A CTA (WR x WC warps) stages TR = R WR rows of x and TC = C WC
// channels of w for KC values of j in shared memory, so each x / w element read from L2 serves C / R FMAs.
// Needs K % (32 KC) == 0 and K % 8 == 0 (launch_router falls back to k_linear_mr otherwise).
template <int R, int C, int WR, int WC, int KC>
__global__ void __launch_bounds__(WR * WC * 32) k_router_tile(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                                              float* __restrict__ y, int K, int N, int M)
{
    constexpr int NT = WR * WC * 32, TR = R * WR, TC = C * WC, KS = 32 * KC;
    __shared__ __align__(16) float xs[TR][KS];
    __shared__ __align__(16) uint16_t ws[TC][KS];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wr = warp / WC, wc = warp % WC;
    const int m0 = blockIdx.y * TR, n0 = blockIdx.x * TC;
    float acc[R][C];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
    for (int k0 = 0; k0 < K; k0 += KS)
    {
        for (int i = threadIdx.x; i < TR * KS / 4; i += NT)
        {
            const int r = i / (KS / 4), q = i % (KS / 4);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + r < M) v = __ldg(reinterpret_cast<const float4*>(x + static_cast<size_t>(m0 + r) * K + k0) + q);
            *reinterpret_cast<float4*>(&xs[r][q * 4]) = v;
        }
        for (int i = threadIdx.x; i < TC * KS / 8; i += NT)
        {
            const int c = i / (KS / 8), q = i % (KS / 8);
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (n0 + c < N) v = __ldg(reinterpret_cast<const uint4*>(w + static_cast<size_t>(n0 + c) * K + k0) + q);
            *reinterpret_cast<uint4*>(&ws[c][q * 8]) = v;
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < KC; ++j)
        {
            float xv[R], wv[C];
            #pragma unroll
            for (int r = 0; r < R; ++r) xv[r] = xs[wr * R + r][lane + 32 * j];
            #pragma unroll
            for (int c = 0; c < C; ++c) wv[c] = flash_mlp::bf(ws[wc * C + c][lane + 32 * j]);
            #pragma unroll
            for (int r = 0; r < R; ++r)
                #pragma unroll
                for (int c = 0; c < C; ++c) acc[r][c] = fmaf(xv[r], wv[c], acc[r][c]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int c = 0; c < C; ++c)
        {
            const float v = flash_mlp::warp_sum(acc[r][c]);
            const int m = m0 + wr * R + r, n = n0 + wc * C + c;
            if (!lane && m < M && n < N) y[static_cast<size_t>(m) * N + n] = v;
        }
}

// Router logits y [M][N] = x [M][K] FP32 . w [N][K] BF16 (0-ULP vs flash_mlp::k_linear); k_linear_mr below 64 rows or
// with G6_PF_ROUTER1=1. 9216 rows x 288 x 4096 on one GB10: k_linear_mr 16.3 ms, k_router_tile<8,16,2,2,8> 2.5 ms.
template <int R = 8, int C = 16, int WR = 2, int WC = 2, int KC = 8>
inline void launch_router(const float* x, const uint16_t* w, float* y, int K, int N, int M, cudaStream_t st)
{
    static const bool old = getenv("G6_PF_ROUTER1") && atoi(getenv("G6_PF_ROUTER1")) == 1;
    if (old || M < 64 || K % (32 * KC) || K % 8)   // small M: one CTA row tile per 16 rows idles most lanes
    {
        constexpr int MR = 4;
        k_linear_mr<MR><<<dim3((N + 7) / 8, (M + MR - 1) / MR), 256, 0, st>>>(x, w, y, K, N, M);
        return;
    }
    k_router_tile<R, C, WR, WC, KC><<<dim3((N + C * WC - 1) / (C * WC), (M + R * WR - 1) / (R * WR)), WR * WC * 32, 0, st>>>(
        x, w, y, K, N, M);
}

__global__ void __launch_bounds__(256) k_swiglu_bf16(const float* __restrict__ g, const float* __restrict__ u,
                                                     uint16_t* __restrict__ out, long long n)
{
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if (i < n) out[i] = f2bf(swiglu_f(g[i], u[i]));
}

struct State
{
    int MP = 0;
    char* sc_mem = nullptr;
    bindp::Scratch sc{};
    float* logits = nullptr;                  // [MP][NEXP] fp32 router logits
    float* bf16_act = nullptr;                // [MP][DENSE_CH] fp32 gate/act scratch for lossless BF16 dense layers
    float* bf16_up = nullptr;                 // [MP][DENSE_CH] fp32 up scratch for lossless BF16 dense layers
    uint16_t* bf16_mid = nullptr;             // [MP][DENSE_CH] bf16 swiglu scratch for lossless BF16 dense layers
    float* bf16_out = nullptr;                // [MP][HID] fp32 fallback scratch for lossless BF16 dense layers
    cublasHandle_t blas = nullptr;
    std::vector<bindp::Tensors> T;            // [m.nl]
    std::vector<bool> bound;                  // [m.nl]

    ~State()
    {
        if (sc_mem) cudaFree(sc_mem);
        if (logits) cudaFree(logits);
        if (bf16_act) cudaFree(bf16_act);
        if (bf16_up) cudaFree(bf16_up);
        if (bf16_mid) cudaFree(bf16_mid);
        if (bf16_out) cudaFree(bf16_out);
        if (blas) cublasDestroy(blas);
    }
};

inline std::map<std::pair<const Model*, const void*>, std::shared_ptr<State>>& states()
{
    static std::map<std::pair<const Model*, const void*>, std::shared_ptr<State>> s;
    return s;
}

inline bind::Tensor as_bind_tensor(const TensorRef& r)
{
    bind::Tensor t;
    t.p = r.p; t.nd = r.nd; t.fmt = r.fmt;
    for (int i = 0; i < 4; ++i) t.shape[i] = r.shape[i];
    return t;
}

// Populate bindp::Tensors for layer i from m.layers[i].flash_mlp_w (and optional Find).
// Makes exact FP8 tile copies for dense gate/up/down (unless G6_DTR_PF_DENSE_COPY=0, which uses k_gemm_tr directly)
// and optional shared-expert FP8 copies (G6_PF_SHFP8=1; default 0 on the fp16 expert body, 1 with G6_PF_MOE8=1).
inline void bind_layer_tensors(Model& m, int i, const Find* find, cudaStream_t st, bindp::Tensors& bt)
{
    Layer& y = m.layers[i];
    const flash_mlp::Weights& w = y.flash_mlp_w;
    if (find)
        bt = bindp::layer_tensors(y.l, [&](const char* name) { return as_bind_tensor((*find)(name)); });
    else
        bt = bindp::Tensors();

    if (w.moe)
    {
        // Copy expert tensor refs from y.T (already bound and width-scanned by flash_mlp::load)
        bt.t[bindp::T_TR3_FC1]  = y.T.t[bind::T_TR3_FC1];
        bt.t[bindp::T_TR3_FC2]  = y.T.t[bind::T_TR3_FC2];
        bt.t[bindp::T_TR3_V1]   = y.T.t[bind::T_TR3_V1];
        bt.t[bindp::T_TR3_V2]   = y.T.t[bind::T_TR3_V2];
        bt.t[bindp::T_TR3_V3]   = y.T.t[bind::T_TR3_V3];
        bt.t[bindp::T_TR3_TAB]  = y.T.t[bind::T_TR3_TAB];
        bt.t[bindp::T_TR3_KMIX] = y.T.t[bind::T_TR3_KMIX];
        bt.t[bindp::T_TR3_SG]   = y.T.t[bind::T_TR3_SG];
        bt.t[bindp::T_TR3_V1B]  = y.T.t[bind::T_TR3_V1B];
        bt.t[bindp::T_TR3_V3B]  = y.T.t[bind::T_TR3_V3B];
        if (!bt.t[bindp::T_ROUTER].p && w.router)
        {
            bt.t[bindp::T_ROUTER].p = w.router;
            bt.t[bindp::T_ROUTER].nd = 2;
            bt.t[bindp::T_ROUTER].shape[0] = geom::NEXP;
            bt.t[bindp::T_ROUTER].shape[1] = HID;
        }
        if (!bt.t[bindp::T_ROUTER_BIAS].p && w.bias)
        {
            bt.t[bindp::T_ROUTER_BIAS].p = w.bias;
            bt.t[bindp::T_ROUTER_BIAS].nd = 1;
            bt.t[bindp::T_ROUTER_BIAS].shape[0] = geom::NEXP;
        }
        // default off: the FP8 copy quantizes the shared expert's activations, the fp16 trellis pass matches decode
        static const bool shfp8 = getenv("G6_PF_SHFP8") ? atoi(getenv("G6_PF_SHFP8")) != 0
                                                        : getenv("G6_PF_MOE8") && atoi(getenv("G6_PF_MOE8")) == 1;
        if (shfp8 && !y.T.shfp8.codes[0] && !y.T.fc1_wide && !y.T.fc2_wide)
            if (const char* e = bind::make_shared_fp8(y.T, st))
                die("layer " + std::to_string(y.l) + ": " + e);
        static const bool wide = !getenv("G6_PF_NOWIDE") || atoi(getenv("G6_PF_NOWIDE")) == 0;
        if (wide && !y.T.fc1_wide)
        {
            bind::widen_fc1(y.T, st);
            if (!getenv("G6_NOWIDE2")) bind::widen_fc2(y.T, st);
            y.flash_mlp_w.experts = y.T;
        }
        bt.fc1_wide = y.T.fc1_wide;
        bt.e4m3 = y.T.e4m3;
        bt.fc2_wide = y.T.fc2_wide;
        bt.k2 = y.T.k2;
        bt.k3 = y.T.k3;
        bt.sh_trellis = y.T.sh_trellis;
        if (shfp8 && y.T.shfp8.codes[0])
        {
            const bind::LayerTensors::ShFp8& F = y.T.shfp8;
            const int id[3] = {bindp::T_S_GATE, bindp::T_S_UP, bindp::T_S_DOWN};
            for (int j = 0; j < 3; ++j)
            {
                const long long n = j < 2 ? F.I : HID, k = j < 2 ? HID : F.I;
                bt.t[id[j]] = bind::Tensor(); bt.t[id[j]].p = F.codes[j]; bt.t[id[j]].nd = 2; bt.t[id[j]].shape[0] = n; bt.t[id[j]].shape[1] = k;
                bt.s[id[j]] = bind::Tensor(); bt.s[id[j]].p = F.scales[j]; bt.s[id[j]].nd = 2; bt.t[id[j]].shape[0] = n; bt.s[id[j]].shape[0] = n; bt.s[id[j]].shape[1] = k / 32;
            }
            bt.sh_fp8 = 1;
            for (int j = 0; j < 6; ++j) bt.sh_tr[j] = y.T.sh_tr[j];
        }
    }
    else if (w.gate_t.w)
    {
        bt.tm.gate = w.gate_t;
        bt.tm.up   = w.up_t;
        bt.tm.down = w.down_t;
        bt.tsuh[2] = w.gate_t.suh; bt.tsvh[2] = w.gate_t.svh;
        bt.tsuh[3] = w.up_t.suh;   bt.tsvh[3] = w.up_t.svh;
        bt.tsuh[4] = w.down_t.suh; bt.tsvh[4] = w.down_t.svh;
        static const bool dcopy = getenv("G6_PF_DTR8") && atoi(getenv("G6_PF_DTR8")) == 1;
        if (dcopy && (!bt.t[bindp::T_D_GATE].p || !bt.t[bindp::T_D_UP].p || !bt.t[bindp::T_D_DOWN].p))
        {
            const dtr::Mat* mats[3] = {&w.gate_t, &w.up_t, &w.down_t};
            const int ids[3] = {bindp::T_D_GATE, bindp::T_D_UP, bindp::T_D_DOWN};
            for (int j = 0; j < 3; ++j)
            {
                if (bt.t[ids[j]].p) continue;
                dtr::Fp8 F;
                if (const char* e = dtr::make_fp8(*mats[j], st, F)) die("layer " + std::to_string(y.l) + " dense make_fp8: " + e);
                bt.t[ids[j]].p = F.codes; bt.t[ids[j]].nd = 2; bt.t[ids[j]].shape[0] = F.N; bt.t[ids[j]].shape[1] = F.K;
                bt.s[ids[j]].p = F.scales; bt.s[ids[j]].nd = 2; bt.s[ids[j]].shape[0] = F.N; bt.s[ids[j]].shape[1] = F.K / 32;
            }
        }
    }
}

template <class Prefill>
inline State& ensure_state(Model& m, Prefill& P, const Find* find, cudaStream_t st)
{
    auto& ptr = states()[{&m, static_cast<const void*>(&P)}];
    if (!ptr || ptr->MP < P.MP)
    {
        ptr = std::make_shared<State>();
        ptr->MP = P.MP;
        const size_t sb = bindp::scratch_bytes(P.MP, g6nr::EB);
        ptr->sc_mem = dmalloc<char>(sb);
        ptr->sc = bindp::scratch_at(ptr->sc_mem, P.MP, g6nr::EB);
        ptr->logits = dmalloc<float>(static_cast<size_t>(P.MP) * geom::NEXP);
        ptr->T.resize(m.nl);
        ptr->bound.assign(m.nl, false);
        G6RT_CK(bind::init());
    }
    return *ptr;
}

}   // namespace flash_mlp_pf

template <class Prefill>
inline void bind_mlp_prefill(Model& m, Prefill& P, int i, const Find& find, cudaStream_t st)
{
    flash_mlp_pf::State& S = flash_mlp_pf::ensure_state(m, P, &find, st);
    if (i < 0 || i >= m.nl) die("bind_mlp_prefill: layer index out of range");
    if (!S.bound[i])
    {
        flash_mlp_pf::bind_layer_tensors(m, i, &find, st, S.T[i]);
        S.bound[i] = true;
    }
}

template <class Prefill>
inline void bind_mlp_prefill(Model& m, Prefill& P, cudaStream_t st)
{
    flash_mlp_pf::State& S = flash_mlp_pf::ensure_state(m, P, nullptr, st);
    for (int i = 0; i < m.nl; ++i)
        if (!S.bound[i])
        {
            flash_mlp_pf::bind_layer_tensors(m, i, nullptr, st, S.T[i]);
            S.bound[i] = true;
        }
}

template <class Prefill>
inline void pf_mlp_block_gemm(Model& m, Prefill& P, int i, int M, cudaStream_t st)
{
    if (M < 1 || M > P.MP) die("pf_mlp_block_gemm: M outside 1..MP");
    flash_mlp_pf::State& S = flash_mlp_pf::ensure_state(m, P, nullptr, st);
    if (!S.bound[i])
    {
        flash_mlp_pf::bind_layer_tensors(m, i, nullptr, st, S.T[i]);
        S.bound[i] = true;
    }
    Layer& y = m.layers[i];
    const flash_mlp::Weights& w = y.flash_mlp_w;
    // Lossless BF16 dense fallback (when dense trellis weights are absent)
    if (!w.moe && !w.gate_t.w && w.gate)
    {
        if (!S.bf16_act)
        {
            S.bf16_act = dmalloc<float>(static_cast<size_t>(P.MP) * g6nr::DENSE_CH);
            S.bf16_up  = dmalloc<float>(static_cast<size_t>(P.MP) * g6nr::DENSE_CH);
            S.bf16_mid = dmalloc<uint16_t>(static_cast<size_t>(P.MP) * g6nr::DENSE_CH);
            S.bf16_out = dmalloc<float>(static_cast<size_t>(P.MP) * HID);
            if (cublasCreate(&S.blas) != CUBLAS_STATUS_SUCCESS) die("flash_mlp_pf: cublasCreate failed");
        }
        static const bool exact = getenv("G6_PF_BF16_EXACT") && atoi(getenv("G6_PF_BF16_EXACT")) == 1;
        if (exact)
        {
            flash_mlp::k_dense_gate<<<dim3((w.I + 7) / 8, M), 256, 0, st>>>(P.xn2, w.gate, w.up, S.bf16_act, w.I);
            flash_mlp::k_linear<<<dim3((HID + 7) / 8, M), 256, 0, st>>>(S.bf16_act, w.down, S.bf16_out, w.I, HID);
            flash_mlp::k_to_bf16<<<(static_cast<long long>(M) * HID + 255) / 256, 256, 0, st>>>(
                S.bf16_out, reinterpret_cast<uint16_t*>(P.part[1]), M * HID);
            return;
        }
        flash_mlp::k_to_bf16<<<(static_cast<long long>(M) * HID + 255) / 256, 256, 0, st>>>(P.xn2, S.sc.xb, M * HID);
        const float one = 1.f, zero = 0.f;
        if (cublasSetStream(S.blas, st) != CUBLAS_STATUS_SUCCESS) die("flash_mlp_pf: cublasSetStream failed");
        if (cublasGemmEx(S.blas, CUBLAS_OP_T, CUBLAS_OP_N, w.I, M, HID, &one, w.gate, CUDA_R_16BF, HID,
                         S.sc.xb, CUDA_R_16BF, HID, &zero, S.bf16_act, CUDA_R_32F, w.I, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            die("flash_mlp_pf: cuBLAS dense gate failed");
        if (cublasGemmEx(S.blas, CUBLAS_OP_T, CUBLAS_OP_N, w.I, M, HID, &one, w.up, CUDA_R_16BF, HID,
                         S.sc.xb, CUDA_R_16BF, HID, &zero, S.bf16_up, CUDA_R_32F, w.I, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            die("flash_mlp_pf: cuBLAS dense up failed");
        const long long nmid = static_cast<long long>(M) * w.I;
        flash_mlp_pf::k_swiglu_bf16<<<static_cast<unsigned>((nmid + 255) / 256), 256, 0, st>>>(
            S.bf16_act, S.bf16_up, S.bf16_mid, nmid);
        if (cublasGemmEx(S.blas, CUBLAS_OP_T, CUBLAS_OP_N, HID, M, w.I, &one, w.down, CUDA_R_16BF, w.I,
                         S.bf16_mid, CUDA_R_16BF, w.I, &zero, S.bf16_out, CUDA_R_32F, HID, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            die("flash_mlp_pf: cuBLAS dense down failed");
        flash_mlp::k_to_bf16<<<(static_cast<long long>(M) * HID + 255) / 256, 256, 0, st>>>(
            S.bf16_out, reinterpret_cast<uint16_t*>(P.part[1]), M * HID);
        return;
    }

    bindp::Io io;
    io.xn2 = P.xn2;
    io.x2 = P.x2;
    io.logits = S.logits;
    io.mlp_part = reinterpret_cast<uint16_t*>(P.part[1]);
    bindp::Layer L;
    if (const char* e = bindp::make_layer(S.T[i], M, S.sc, io, nullptr, L))
        die("layer " + std::to_string(y.l) + " pf_mlp_block_gemm: " + e);

    if (L.moe)
    {
        // FP32 router input (no BF16 rounding before top-k), 0-ULP bit-identical to flash_mlp::k_linear
        // G6_PF_ROUTER_SPLIT (opt-in): this rank routes its own rows only, publishes them, and takes the peers' rows after the
        // hx / shared launches; launch_route then skips k_pf_route (Expert::pre_routed). Off: the replicated path below.
        const bool split = P.route_split.fabric && flash_route_split::active(M, P.route_split.f.nr);
        if (split)
        {
            const int r0 = flash_route_split::r0(M, P.route_split.f.rank, P.route_split.f.nr);
            const int rows = flash_route_split::rows(M, P.route_split.f.rank, P.route_split.f.nr);
            flash_mlp_pf::launch_router(P.xn2 + static_cast<size_t>(r0) * HID, w.router,
                                        S.logits + static_cast<size_t>(r0) * geom::NEXP, HID, geom::NEXP, rows, st);
            bindp::launch_route_slice(L.ex, r0, rows, st);
            flash_route_publish(P.route_split, L.ex.tok_e, L.ex.tok_w, M, st);
        }
        else flash_mlp_pf::launch_router(P.xn2, w.router, S.logits, HID, geom::NEXP, M, st);
        bindp::launch_hx(L.ex, st);           // G6_PF_HX: pre-rotated fc1 rows for the shared and the routed passes
        bindp::launch_shared<g6nr::EB>(L, st);
        if (split)
        {
            flash_route_receive(P.route_split, L.ex.tok_e, L.ex.tok_w, M, st);
            L.ex.pre_routed = 1;
        }
        bindp::launch_experts<g6nr::EB>(L.ex, st);
    }
    else
    {
        bindp::launch(L.xq_x2, st);
        bindp::launch_pair(L.gate, L.up, st);
        bindp::launch(L.act, st);
        bindp::launch(L.xq_act, st);
        bindp::launch(L.down, st);
    }
}

}   // namespace g6rt
#endif
