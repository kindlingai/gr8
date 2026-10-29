// Flash decode MLP, correctness-first BF16 dense weights and existing exlr8 K1 experts.
// W1 names/layouts: loader-rs/FLASH.md. No attention dependency and no weight conversion.
#pragma once
#include <string>
#include "bind.cuh"
#include "glue.cuh"
#include "../knobs-rs/gr8_knobs.h"
#include "../runtime/flash_mark.h"
#if G6_MODEL_FLASH
namespace flash_mlp
{
// A warp owns an output channel. FP32 input, BF16 row-major weights, FP32 accumulation.
// This intentionally avoids the legacy router's BF16 input rounding before top-k.
__device__ __forceinline__ float bf(uint16_t x) { return __uint_as_float(static_cast<unsigned>(x) << 16); }
__device__ __forceinline__ float warp_sum(float v)
{
    #pragma unroll
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
    return v;
}
template <int NW, int MF, bool HAS_XHALF = false>
__global__ void __launch_bounds__(NW * 32) k_router_linear(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                                           float* __restrict__ y, int M_rt, __half* __restrict__ xhalf = nullptr)
{
    constexpr int R = MF > 0 ? MF : 1;
    __shared__ float sx[R][512];
    __shared__ __align__(16) uint16_t sw[NW][512];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n = blockIdx.x * NW + warp, m0 = MF > 0 ? 0 : blockIdx.y;
    const uint4* wn4 = (n < geom::NEXP) ? reinterpret_cast<const uint4*>(w + static_cast<size_t>(n) * HID) : nullptr;
    uint4* sw4 = reinterpret_cast<uint4*>(sw[warp]);
    float v[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) v[r] = 0.f;
    #pragma unroll
    for (int c = 0; c < HID / 256; c += 2)
    {
        const int k0 = c * 256;
        uint4 w0, w1;
        if (wn4)
        {
            w0 = __ldg(wn4 + (c + 0) * 32 + lane);
            w1 = __ldg(wn4 + (c + 1) * 32 + lane);
        }
        #pragma unroll
        for (int r = 0; r < R; ++r)
        {
            const float* xm = x + static_cast<size_t>(m0 + r) * HID + k0;
            #pragma unroll
            for (int i = tid; i < 512 / 4; i += NW * 32)
            {
                const float4 xv = __ldg(reinterpret_cast<const float4*>(xm) + i);
                reinterpret_cast<float4*>(sx[r])[i] = xv;
                if constexpr (HAS_XHALF)
                {
                    if (xhalf && blockIdx.x == 0)
                    {
                        __half* xhm = xhalf + static_cast<size_t>(m0 + r) * HID + k0 + i * 4;
                        xhm[0] = __float2half_rn(xv.x); xhm[1] = __float2half_rn(xv.y);
                        xhm[2] = __float2half_rn(xv.z); xhm[3] = __float2half_rn(xv.w);
                    }
                }
            }
        }
        if (wn4)
        {
            sw4[0 * 32 + lane] = w0;
            sw4[1 * 32 + lane] = w1;
        }
        __syncthreads();
        if (wn4)
        {
            #pragma unroll
            for (int s = 0; s < 16; ++s)
            {
                const float wf = bf(sw[warp][s * 32 + lane]);
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    v[r] = fmaf(sx[r][s * 32 + lane], wf, v[r]);
            }
        }
        __syncthreads();
    }
    if (wn4)
    {
        #pragma unroll
        for (int r = 0; r < R; ++r)
        {
            const float sum = warp_sum(v[r]);
            if (!lane) y[static_cast<size_t>(m0 + r) * geom::NEXP + n] = sum;
        }
    }
}
__global__ void __launch_bounds__(256) k_linear(const float* x, const uint16_t* w, float* y, int K, int N, __half* xhalf = nullptr)
{
    const int n = blockIdx.x * 8 + threadIdx.x / 32, m = blockIdx.y, lane = threadIdx.x % 32;
    if (xhalf)
    {
        const int i = blockIdx.x * 256 + threadIdx.x;
        if (i < K) xhalf[static_cast<size_t>(m) * K + i] = __float2half_rn(x[static_cast<size_t>(m) * K + i]);
    }
    if (n >= N) return;
    const float* xm = x + static_cast<size_t>(m) * K;
    const uint16_t* wn = w + static_cast<size_t>(n) * K;
    float v = 0.f;
    #pragma unroll 8
    for (int k = lane; k < K; k += 32) v = fmaf(__ldg(xm + k), bf(__ldg(wn + k)), v);
    v = warp_sum(v);
    if (!lane) y[static_cast<size_t>(m) * N + n] = v;
}
__global__ void __launch_bounds__(256) k_dense_gate(const float* x, const uint16_t* gate, const uint16_t* up,
                                                   float* act, int I)
{
    const int n = blockIdx.x * 8 + threadIdx.x / 32, m = blockIdx.y, lane = threadIdx.x % 32;
    if (n >= I) return;
    float g = 0.f, u = 0.f;
    for (int k = lane; k < HID; k += 32)
    {
        const float xv = x[static_cast<size_t>(m) * HID + k];
        const size_t ix = static_cast<size_t>(n) * HID + k;
        g = fmaf(xv, bf(gate[ix]), g); u = fmaf(xv, bf(up[ix]), u);
    }
    g = warp_sum(g); u = warp_sum(u);
    if (!lane) act[static_cast<size_t>(m) * I + n] = swiglu_f(g, u);
}
__global__ void __launch_bounds__(256) k_to_half(const float* x, __half* y, int n)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}
__global__ void __launch_bounds__(256) k_to_bf16(const float* x, uint16_t* y, int n)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = f2bf(x[i]);
}

struct Weights
{
    const uint16_t *gate = nullptr, *up = nullptr, *down = nullptr, *router = nullptr;
    const float* bias = nullptr;
    int I = 0;
    bool moe = false;
    dtr::Mat gate_t, up_t, down_t;
    bind::LayerTensors experts;  // scan_widths/widen as for the legacy binder, before make()
};

// find(name) returns the runtime TensorRef (p, nd, shape, bf16, fp32, fmt).
template <class F> inline const char* load(int layer, F find, Weights& out)
{
    out = Weights();
    char prefix[16]; snprintf(prefix, sizeof prefix, "L%03d.", layer);
    auto get = [&](const char* suffix) { return find(std::string(prefix) + suffix); };
    auto cvt = [&](const char* name) {
        const auto t = find(std::string(name));
        bind::Tensor b; b.p = t.p; b.nd = t.nd; b.fmt = t.fmt;
        for (int i = 0; i < 4; ++i) b.shape[i] = t.shape[i];
        return b;
    };
    out.experts = bind::layer_tensors(layer, cvt);
    out.moe = out.experts.t[bind::T_TR3_TAB].p != nullptr;
    if (out.moe)
    {
        const auto r = get("router.w"), b = get("router.bias");
        if (!r.p || !r.bf16 || r.nd != 2 || r.shape[0] != geom::NEXP || r.shape[1] != HID)
            return "Flash router.w must be BF16 [NEXP,HID]";
        if (!b.p || !b.fp32 || b.nd != 1 || b.shape[0] != geom::NEXP)
            return "Flash router.bias must be F32 [NEXP]";
        out.router = static_cast<const uint16_t*>(r.p); out.bias = static_cast<const float*>(b.p);
        if (const char* e = bind::scan_widths(out.experts)) return e;
        if (!out.experts.sh_trellis) return "Flash MoE needs the W1 trellis shared expert at index NEXP";
        return nullptr;
    }
    if (get("dense.gate.trellis").p)
    {
        if (const char* e = dtr::make_mat(get("dense.gate.trellis"), get("dense.gate.suh"), get("dense.gate.svh"), dtr::SIDE_OUT, out.gate_t)) return e;
        if (const char* e = dtr::make_mat(get("dense.up.trellis"), get("dense.up.suh"), get("dense.up.svh"), dtr::SIDE_OUT, out.up_t)) return e;
        if (const char* e = dtr::make_mat(get("dense.down.trellis"), get("dense.down.suh"), get("dense.down.svh"), dtr::SIDE_IN, out.down_t)) return e;
        if (!out.gate_t.w || !out.up_t.w || !out.down_t.w || out.gate_t.K != HID || out.up_t.K != HID ||
            out.gate_t.N != g6nr::DENSE_CH || out.up_t.N != g6nr::DENSE_CH || out.down_t.K != g6nr::DENSE_CH || out.down_t.N != HID)
            return "Flash dense trellis gate/up/down shapes do not match [DENSE_CH,HID], [DENSE_CH,HID], [HID,DENSE_CH]";
        out.I = out.gate_t.N;
        return nullptr;
    }
    const auto g = get("dense.gate"), u = get("dense.up"), d = get("dense.down");
    if (!g.p || !u.p || !d.p || !g.bf16 || !u.bf16 || !d.bf16 || g.nd != 2 || u.nd != 2 || d.nd != 2 ||
        g.shape[1] != HID || u.shape[1] != HID || g.shape[0] != u.shape[0] || d.shape[0] != HID ||
        d.shape[1] != g.shape[0] || g.shape[0] != g6nr::DENSE_CH)
        return "Flash dense gate/up/down must be BF16 [DENSE_CH,HID], [DENSE_CH,HID], [HID,DENSE_CH]";
    out.I = static_cast<int>(g.shape[0]);
    out.gate = static_cast<const uint16_t*>(g.p); out.up = static_cast<const uint16_t*>(u.p); out.down = static_cast<const uint16_t*>(d.p);
    return nullptr;
}

// Additional workspace, owned by caller, zero not required. K1 Scratch is separate and must be zeroed once.
struct Workspace { float *act, *dense_out, *logits; __half* xhalf; u32* xfrag; int rows; };
inline size_t workspace_layout(Workspace* out, char* base, int rows)
{
    size_t offset = 0;
    auto take = [&](size_t n) { char* p = base ? base + offset : nullptr; offset += (n + 255) / 256 * 256; return p; };
    Workspace w; w.rows = rows;
    w.act = reinterpret_cast<float*>(take(static_cast<size_t>(rows) * g6nr::DENSE_CH * 4));
    w.dense_out = reinterpret_cast<float*>(take(static_cast<size_t>(rows) * HID * 4));
    w.logits = reinterpret_cast<float*>(take(static_cast<size_t>(rows) * geom::NEXP * 4));
    w.xhalf = reinterpret_cast<__half*>(take(static_cast<size_t>(rows) * HID * 2));
    w.xfrag = reinterpret_cast<u32*>(take(static_cast<size_t>(g6mx::tiles(rows)) * KSH * 64 * sizeof(u32)));
    if (out) *out = w;
    return offset;
}
inline size_t workspace_bytes(int rows = G6_MX) { return workspace_layout(nullptr, nullptr, rows); }
inline Workspace workspace_at(void* base, int rows = G6_MX)
{
    Workspace w; workspace_layout(&w, static_cast<char*>(base), rows); return w;
}
struct Block
{
    Weights w;
    Workspace scratch;
    int M = 0;
    const float* x = nullptr;
    const __half* xh = nullptr;
    uint16_t* out = nullptr;
    glue::RoutePlan* plan = nullptr;
    bind::Xfrag xf;
    bind::K1 k1;
    dtr::Gemv glu_t, down_t;
    int topk = geom::TOPK;            // routed experts kept per token: the GR8_MOE_TOPK cut is set by the main-model callers only
};                                    // (the MTP drafter keeps the model's top-k)
inline const char* make(const Weights& w, int M, const float* x, uint16_t* out, glue::RoutePlan* plan,
                        const Workspace& ws, const bind::Scratch& sc, Block& b, int fuse = -1,
                        const __half* xh = nullptr)
{
    if (fuse < 0) fuse = static_cast<int>(knob::fuse());
    b = Block();
    if (M < 1 || M > sc.rows || M > ws.rows || M > G6_MX) return "Flash MLP M exceeds scratch rows";
    if (!x || !out || !ws.act || !ws.dense_out || !ws.logits || !ws.xhalf || !ws.xfrag) return "Flash MLP needs input/output/workspace";
    b.w = w; b.scratch = ws; b.M = M; b.x = x; b.xh = xh; b.out = out; b.plan = plan;
    const __half* x_half = xh ? xh : ws.xhalf;
    if (!w.moe)
    {
        if (!w.gate_t.w) return nullptr;
        if (const char* e = dtr::make_glu(w.gate_t, w.up_t, w.down_t, M, x_half, HID, sc.dxd, sc.dt, 128, 0, 1, b.glu_t)) return e;
        return dtr::make_out(w.down_t, M, nullptr, 0, sc.dxd, out, HID, dtr::O_BF16, sc.dt, 192, 0, 1, b.down_t);
    }
    if (!plan) return "Flash MoE needs a RoutePlan";
    b.xf.on = !w.experts.sh_trellis; b.xf.src = x_half; b.xf.ld = b.xf.K = HID; b.xf.M = M; b.xf.f16 = ws.xfrag;
    bind::Tensor unused[3];
    return bind::make_k1(w.experts, M, sc, x_half, ws.xfrag, bind::plan_ptrs(plan), out, 1, b.k1, unused, fuse);
}
// out = this rank's BF16 partial. Caller publishes/all-reduces it, then applies mHC.
inline void launch(const Block& b, cudaStream_t st)
{
    if (b.w.moe)
    {
        const int64_t r_nw = knob::router_nw();
        const bool r_mf = knob::router_mf();
        __half* xh_out = b.xh ? nullptr : b.scratch.xhalf;
        auto dispatch_nw = [&](auto nw_tag) {
            constexpr int NW = decltype(nw_tag)::value;
            const unsigned gx = (geom::NEXP + NW - 1) / NW;
            if (!xh_out)
            {
                if (b.M == 1) k_router_linear<NW, 1, false><<<gx, NW * 32, 0, st>>>(b.x, b.w.router, b.scratch.logits, 1);
                else if (b.M == 4 && r_mf) k_router_linear<NW, 4, false><<<gx, NW * 32, 0, st>>>(b.x, b.w.router, b.scratch.logits, 4);
                else k_router_linear<NW, 0, false><<<dim3(gx, b.M), NW * 32, 0, st>>>(b.x, b.w.router, b.scratch.logits, b.M);
            }
            else
            {
                if (b.M == 1) k_router_linear<NW, 1, true><<<gx, NW * 32, 0, st>>>(b.x, b.w.router, b.scratch.logits, 1, xh_out);
                else if (b.M == 4 && r_mf) k_router_linear<NW, 4, true><<<gx, NW * 32, 0, st>>>(b.x, b.w.router, b.scratch.logits, 4, xh_out);
                else k_router_linear<NW, 0, true><<<dim3(gx, b.M), NW * 32, 0, st>>>(b.x, b.w.router, b.scratch.logits, b.M, xh_out);
            }
        };
        if (r_nw == 4) dispatch_nw(std::integral_constant<int, 4>{});
        else if (r_nw == 8) dispatch_nw(std::integral_constant<int, 8>{});
        else dispatch_nw(std::integral_constant<int, 2>{});
        FMARK(FM_MLP_ROUTER, st);
        switch (b.topk)             // one instance per K; K = TOPK is the pre-cut kernel
        {
            case 1: glue::k_router_plan<1><<<1, glue::NEXP, 0, st>>>(b.scratch.logits, b.w.bias, b.M, b.plan); break;
            case 2: glue::k_router_plan<2><<<1, glue::NEXP, 0, st>>>(b.scratch.logits, b.w.bias, b.M, b.plan); break;
            case 3: glue::k_router_plan<3><<<1, glue::NEXP, 0, st>>>(b.scratch.logits, b.w.bias, b.M, b.plan); break;
            case 4: glue::k_router_plan<4><<<1, glue::NEXP, 0, st>>>(b.scratch.logits, b.w.bias, b.M, b.plan); break;
            case 5: glue::k_router_plan<5><<<1, glue::NEXP, 0, st>>>(b.scratch.logits, b.w.bias, b.M, b.plan); break;
            case 6: glue::k_router_plan<6><<<1, glue::NEXP, 0, st>>>(b.scratch.logits, b.w.bias, b.M, b.plan); break;
            case 7: glue::k_router_plan<7><<<1, glue::NEXP, 0, st>>>(b.scratch.logits, b.w.bias, b.M, b.plan); break;
            default: glue::k_router_plan<geom::TOPK><<<1, glue::NEXP, 0, st>>>(b.scratch.logits, b.w.bias, b.M, b.plan); break;
        }
        FMARK(FM_MLP_PLAN, st);
        FMARK(FM_MLP_XHALF, st);
        bind::launch(b.xf, st);
        FMARK(FM_MLP_XFRAG, st);
        // launch_k1's dispatch, with a mark between the stages (keep in sync with launch_k1 in bind.cuh; the fused
        // fc1+fc2 launch lands in the FM_K1_FC1 stage)
        if (b.k1.fuse)
        {
            bind::launch_k1_fused(b.k1, st);
            FMARK(FM_K1_FC1, st);
            FMARK(FM_K1_FC2, st);
        }
        else
        {
            bind::launch_k1_fc1(b.k1, st);
            FMARK(FM_K1_FC1, st);
            bind::launch_k1_fc2(b.k1, st);
            FMARK(FM_K1_FC2, st);
        }
        bind::launch_k1_fin(b.k1, st);
        FMARK(FM_K1_FIN, st);
    }
    else if (b.glu_t.on)
    {
        if (!b.xh) k_to_half<<<(b.M * HID + 255) / 256, 256, 0, st>>>(b.x, b.scratch.xhalf, b.M * HID);
        dtr::launch(b.glu_t, st);
        FMARK(FM_D_GATE, st);
        dtr::launch(b.down_t, st);
        FMARK(FM_D_DOWN, st);
    }
    else
    {
        k_dense_gate<<<dim3((b.w.I + 7) / 8, b.M), 256, 0, st>>>(b.x, b.w.gate, b.w.up, b.scratch.act, b.w.I);
        FMARK(FM_D_GATE, st);
        k_linear<<<dim3((HID + 7) / 8, b.M), 256, 0, st>>>(b.scratch.act, b.w.down, b.scratch.dense_out, b.w.I, HID);
        FMARK(FM_D_DOWN, st);
        k_to_bf16<<<(b.M * HID + 255) / 256, 256, 0, st>>>(b.scratch.dense_out, b.out, b.M * HID);
        FMARK(FM_D_OUT, st);
    }
}
}
#endif
