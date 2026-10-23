// W4: the GLM 5.3 Flash KDA (linear attention) block, decode. INTERFACES.md 4.2. Our own code.
//
//   kda::bind_kda(w, l, find, rank, nr)   build: this rank's view of KDA layer l (heads h0 .. h0 + nh, loader rule)
//   kda::build_state / build_scratch      build: fp32 state S [N_KDA][slots][nhmax][128][128] + conv history
//                                         [N_KDA][slots][3][3][nhmax*128]; scratch for `rows` rows
//   kda::reset(state, slot, st)           zero one slot's state, every KDA layer (session start)
//   kda::launch(w, state, scratch, xb, row_slot, M, partial, st)
//       xb: BF16 fragment table of the fp32 block input (K = HID); row_slot: device [M]; partial: BF16 [M][HID], this
//       rank's heads through its o_proj shard. Rows of one slot must be consecutive and in position order.
//       Five launches: projections (q, k, v own rows; f_a, g_a, b_proj whole) -> prep (conv + silu, gates) ->
//       step (fp32 state recurrence) -> out (o_norm * sigmoid gate, BF16 fragments) -> o_proj.
// Graph-capturable: no host sync, no allocation in reset/launch.
#pragma once
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "math.h"
#include "../kernels/densev.cuh"
#include "../kernels/dtr.cuh"
#include "../runtime/flash_mark.h"

namespace kda
{
// rank r's heads of layer l at TP nr (loader-rs release.rs Shard::heads: chunk(widths(64, nr), l, r))
inline void heads(int l, int rank, int nr, int& h0, int& nh)
{
    std::vector<int> w(nr);
    for (int j = 0; j < nr; ++j) w[j] = HEADS / nr + (j < HEADS % nr);
    const int j = (rank + nr - l % nr) % nr;
    h0 = 0;
    for (int i = 0; i < j; ++i) h0 += w[i];
    nh = w[j];
}
inline int nhmax(int nr) { return (HEADS + nr - 1) / nr; }

struct KdaLayer
{
    int l = -1, li = -1, h0 = 0, nh = 0;
    const uint8_t *q = nullptr, *k = nullptr, *v = nullptr;   // .gv records at the own rows [nh*8][HID/16][512]
    const uint8_t *fa = nullptr, *ga = nullptr, *b = nullptr; // .gv records, whole: [8][HID/16][512], [4][HID/16][512]
    const uint8_t *fa_mx = nullptr, *ga_mx = nullptr;         // .mx.gv records, whole: [8][HID/32][528]
    const uint8_t *o = nullptr;                               // o_proj shard .gv [HID/16][nh*8][512]
    dtr::Mat qt, kt, vt, ot;                                  // W8 trellis q, k, v (SIDE_OUT) and o_proj (SIDE_IN)
    const uint8_t *mxc[3] = {}, *mxs[3] = {};                 // q, k, v own rows as MXFP8 (prefill GEMM): [nh*128][HID], [..][HID/32]
    const uint16_t *conv[3] = {};                             // [own ch][4]
    const uint16_t *fb = nullptr, *gb = nullptr;              // [own ch][128]
    const float *A_log = nullptr, *dt_bias = nullptr;         // [own heads], [own ch]
    const uint16_t *onorm = nullptr;                          // [128]
};

struct KdaState
{
    float *S = nullptr, *conv = nullptr;
    int slots = 0, nhmax = 0;
    const int* src = nullptr;           // device [1]: carry-mode source slot (Args::src), nullptr = off
    size_t s_layer() const { return static_cast<size_t>(slots) * nhmax * DK * DV; }
    size_t c_layer() const { return static_cast<size_t>(slots) * NHIST * 3 * nhmax * DK; }
};

struct KdaScratch
{
    float *P = nullptr, *C = nullptr, *G = nullptr, *Z = nullptr, *O = nullptr, *OG = nullptr;
    uint16_t *OF = nullptr, *OH = nullptr;
    int rows = 0, dlmax = 0;
};

// ---------------------------------------------------------------------------------------------- build
template <class T> inline const char* kda_get(const T& r, const char* what, std::initializer_list<long long> shape, const void*& p)
{
    if (!r.p) return what;
    long long n = 1, want = 1;
    for (int i = 0; i < r.nd; ++i) n *= r.shape[i];
    for (long long s : shape) want *= s;
    if (n != want) return what;
    p = r.p;
    return nullptr;
}

// find(name) -> {p, nd, shape[]} (rank.cuh TensorRef). Names: loader-rs FLASH.md. Returns nullptr or the missing name.
template <class FindT> inline const char* bind_kda(KdaLayer& w, int l, const FindT& find, int rank, int nr)
{
    static thread_local std::string err;
    w = KdaLayer();
    w.l = l;
    w.li = kda_slot(l);
    heads(l, rank, nr, w.h0, w.nh);
    char pre[16];
    snprintf(pre, sizeof pre, "L%03d.", l);
    const std::string P = pre, A = P + "self_attn.";
    const long long G = HID / 16, dl = static_cast<long long>(w.nh) * DK;
    const size_t row16 = static_cast<size_t>(G) * 512;   // bytes of one 16-row tile of a K = HID .gv record
    const void* p = nullptr;
    auto get = [&](const std::string& n, std::initializer_list<long long> s) -> const void* {
        if (!kda_get(find(n), "x", s, p)) return p;
        if (err.empty()) err = "KDA: no tensor " + n + " of the expected size";
        return nullptr;
    };
    auto tmat = [&](const std::string& base, int side, long long wantK, long long wantN, dtr::Mat& m) {
        if (const char* e = dtr::make_mat(find(base + ".trellis"), find(base + ".suh"), find(base + ".svh"), side, m))
            { if (err.empty()) err = std::string("KDA ") + base + ": " + e; return; }
        if (!m.w || m.K != wantK || m.N != wantN)
            if (err.empty()) err = "KDA: bad trellis shape for " + base;
    };
    err.clear();
    if (find(A + "q_proj.weight.trellis").p)
    {
        tmat(A + "q_proj.weight", dtr::SIDE_OUT, HID, dl, w.qt);
        tmat(A + "k_proj.weight", dtr::SIDE_OUT, HID, dl, w.kt);
        tmat(A + "v_proj.weight", dtr::SIDE_OUT, HID, dl, w.vt);
    }
    else
    {
        const uint8_t* q = static_cast<const uint8_t*>(get(A + "q_proj.weight.gv", {D / 16, G, 512}));
        const uint8_t* k = static_cast<const uint8_t*>(get(A + "k_proj.weight.gv", {D / 16, G, 512}));
        const uint8_t* v = static_cast<const uint8_t*>(get(A + "v_proj.weight.gv", {D / 16, G, 512}));
        const size_t off = static_cast<size_t>(w.h0) * (DK / 16) * row16;
        w.q = q ? q + off : nullptr; w.k = k ? k + off : nullptr; w.v = v ? v + off : nullptr;
    }
    const char* qkv[3] = {"q_proj", "k_proj", "v_proj"};
    for (int i = 0; i < 3; ++i)
        if (find(A + qkv[i] + ".weight.mx.codes").p)
        {
            w.mxc[i] = static_cast<const uint8_t*>(get(A + qkv[i] + ".weight.mx.codes", {dl, HID}));
            w.mxs[i] = static_cast<const uint8_t*>(get(A + qkv[i] + ".weight.mx.scales", {dl, HID / 32}));
        }
    if (find(A + "f_a_proj.weight.mx.gv").p)
        w.fa_mx = static_cast<const uint8_t*>(get(A + "f_a_proj.weight.mx.gv", {DK / 16, HID / 32, 528}));
    else
        w.fa = static_cast<const uint8_t*>(get(A + "f_a_proj.weight.gv", {DK / 16, G, 512}));
    if (find(A + "g_a_proj.weight.mx.gv").p)
        w.ga_mx = static_cast<const uint8_t*>(get(A + "g_a_proj.weight.mx.gv", {DK / 16, HID / 32, 528}));
    else
        w.ga = static_cast<const uint8_t*>(get(A + "g_a_proj.weight.gv", {DK / 16, G, 512}));
    w.b = static_cast<const uint8_t*>(get(A + "b_proj.weight.gv", {HEADS / 16, G, 512}));
    if (find(P + "o_proj.trellis").p)
        tmat(P + "o_proj", dtr::SIDE_IN, dl, HID, w.ot);
    else
        w.o = static_cast<const uint8_t*>(get(P + "o_proj.gv", {G, dl / 16, 512}));
    const char* cn[3] = {"q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"};
    const size_t c0 = static_cast<size_t>(w.h0) * DK;
    for (int i = 0; i < 3; ++i)
    {
        const uint16_t* c = static_cast<const uint16_t*>(get(A + cn[i], {D, CONV}));
        w.conv[i] = c ? c + c0 * CONV : nullptr;
    }
    const uint16_t* fb = static_cast<const uint16_t*>(get(A + "f_b_proj.weight", {D, DK}));
    const uint16_t* gb = static_cast<const uint16_t*>(get(A + "g_b_proj.weight", {D, DK}));
    w.fb = fb ? fb + c0 * DK : nullptr; w.gb = gb ? gb + c0 * DK : nullptr;
    const float* al = static_cast<const float*>(get(A + "A_log", {HEADS}));
    const float* dt = static_cast<const float*>(get(A + "dt_bias", {D}));
    w.A_log = al ? al + w.h0 : nullptr; w.dt_bias = dt ? dt + c0 : nullptr;
    w.onorm = static_cast<const uint16_t*>(get(A + "o_norm.weight", {DK}));
    return err.empty() ? nullptr : err.c_str();
}

inline const char* build_state(KdaState& s, int slots, int nr)
{
    s = KdaState();
    s.slots = slots;
    s.nhmax = nhmax(nr);
    if (cudaMalloc(&s.S, N_KDA * s.s_layer() * sizeof(float)) != cudaSuccess) return "KDA state: cudaMalloc S";
    if (cudaMalloc(&s.conv, N_KDA * s.c_layer() * sizeof(float)) != cudaSuccess) return "KDA state: cudaMalloc conv";
    cudaMemset(s.S, 0, N_KDA * s.s_layer() * sizeof(float));
    cudaMemset(s.conv, 0, N_KDA * s.c_layer() * sizeof(float));
    return cudaDeviceSynchronize() == cudaSuccess ? nullptr : "KDA state: zero";
}

inline const char* build_scratch(KdaScratch& s, int rows, int nr)
{
    s = KdaScratch();
    s.rows = rows;
    s.dlmax = nhmax(nr) * DK;
    PLayout pl;
    pl.dl = s.dlmax;
    const size_t r = static_cast<size_t>(rows), dl = s.dlmax, tiles = (rows + 7) / 8;
    float** f[] = {&s.P, &s.C, &s.G, &s.Z, &s.O, &s.OG};
    const size_t n[] = {r * pl.ld(), r * 3 * dl, r * dl, r * dl, r * dl, r * dl};
    for (int i = 0; i < 6; ++i)
        if (cudaMalloc(f[i], n[i] * sizeof(float)) != cudaSuccess) return "KDA scratch: cudaMalloc";
    if (cudaMalloc(&s.OF, tiles * dl * 8 * sizeof(uint16_t)) != cudaSuccess) return "KDA scratch: cudaMalloc";
    if (cudaMalloc(&s.OH, r * dl * sizeof(uint16_t)) != cudaSuccess) return "KDA scratch: cudaMalloc";
    cudaMemset(s.OF, 0, tiles * dl * 8 * sizeof(uint16_t));
    cudaMemset(s.OH, 0, r * dl * sizeof(uint16_t));
    return cudaDeviceSynchronize() == cudaSuccess ? nullptr : "KDA scratch: zero";
}

// ---------------------------------------------------------------------------------------------- kernels
struct ProjPart
{
    const uint8_t* w = nullptr;
    const u32* xfrag = nullptr;
    void* y = nullptr;
    int codec = 2, K = 0, ot = 0, tpc = 1, ks = 1, cta0 = 0;
};
struct ProjArgs
{
    static constexpr int MAXP = 6;
    ProjPart p[MAXP];
    const u32* xfrag = nullptr;
    int n = 0, ctas = 0, M = 0, ld = 0, odt = gv::O_F32;
    void add(const uint8_t* w, void* y, int K, int O, int tpc, int ks, const u32* xf = nullptr, int codec = 2)
    {
        if (!w) return;
        const int ot = O / 16;
        p[n++] = ProjPart{w, xf ? xf : xfrag, y, codec, K, ot, tpc, ks, ctas};
        ctas += (ot + tpc - 1) / tpc;
    }
};
// several BF16 / MXFP8 GEMVs on one input, each part as gv_gemv would run it (gv_multi with up to 6 parts and NT row tiles)
template <int NT>
__global__ void __launch_bounds__(1024) k_kda_proj(ProjArgs a)
{
    int i = 0;
    while (i + 1 < a.n && static_cast<int>(blockIdx.x) >= a.p[i + 1].cta0) ++i;
    const ProjPart& p = a.p[i];
    if (p.codec == 1) gv_gemv_body<Mxfp8, NT>(p.w, p.xfrag, p.K, p.ot, p.tpc, p.ks, a.M, p.y, a.ld, a.odt, blockIdx.x - p.cta0);
    else gv_gemv_body<Bf16w, NT>(p.w, p.xfrag, p.K, p.ot, p.tpc, p.ks, a.M, p.y, a.ld, a.odt, blockIdx.x - p.cta0);
}
inline void launch_proj(const ProjArgs& a, cudaStream_t st)
{
    if (!a.n || !a.ctas) return;
    g6mx::nt_dispatch(g6mx::tiles(a.M), [&](auto N) {
        constexpr int NTv = decltype(N)::value;
        k_kda_proj<NTv><<<a.ctas, 1024, 0, st>>>(a);
    });
}

__global__ void __launch_bounds__(PREP_T) k_kda_prep(Args a)
{
    __shared__ PrepSmem s;
    PrepThread th;
    const int h = blockIdx.x >> 1, hf = blockIdx.x & 1, t = threadIdx.x;
    prep_stage(a, s, h, hf, t);
    __syncthreads();
    for (int m = 0; m < a.M; ++m)
    {
        prep_row_a(a, s, th, m, h, hf, t);
        __syncthreads();
        prep_row_b(a, s, m, h, hf, t);
        __syncthreads();
    }
    prep_end(a, th, h, hf, t);
}

__global__ void __launch_bounds__(STEP_T) k_kda_step(Args a)
{
    __shared__ StepSmem s;
    StepThread th;
    const int h = blockIdx.x, vb = blockIdx.y, t = threadIdx.x;
    for (int m = 0; m < a.M; ++m)
    {
        step_row_0(a, s, th, m, h, vb, t);
        __syncthreads();
        step_row_1(a, s, m, h, t);
        __syncthreads();
        step_row_2(s, th, t);
        __syncthreads();
        const float o = step_row_3(a, s, th, m, h, vb, t);
        __syncthreads();
        step_row_3b(s, t, o);
        __syncthreads();
        step_row_4(a, s, m, h, vb, t);
    }
    step_end(a, th, h, vb, t);
}

__global__ void __launch_bounds__(OUT_T) k_kda_out(Args a)
{
    __shared__ OutSmem s;
    const int h = blockIdx.x, m = blockIdx.y, t = threadIdx.x;
    out_0(a, s, m, h, t);
    __syncthreads();
    out_1(s, t);
    __syncthreads();
    out_2(a, s, m, h, t);
}

template <bool WRITE_DBG, int M_MODE = 0>
__global__ void __launch_bounds__(CORE_T, 1) k_kda_core(Args a)
{
    __shared__ CoreSmem s;
    CoreThread th;
    const int h = blockIdx.x, t = threadIdx.x;
    if constexpr (M_MODE == 1)
    {
        core_row_0<WRITE_DBG, 1>(a, s, th, 0, h, t);
        __syncthreads();
        core_row_1<WRITE_DBG, 1>(a, s, th, 0, h, t);
        __syncthreads();
        core_row_23<1>(a, s, th, 0, h, t);
        __syncthreads();
        core_row_4<1>(a, s, th, 0, h, t);
        __syncthreads();
        core_row_5<WRITE_DBG, 1>(a, s, 0, h, t);
    }
    else if constexpr (M_MODE == 4)
    {
        core_row_0<WRITE_DBG, 4>(a, s, th, 0, h, t);
        __syncthreads();
        core_row_1<WRITE_DBG, 4>(a, s, th, 0, h, t);
        __syncthreads();
        #pragma unroll 1
        for (int m = 0; m < a.M; ++m)
        {
            core_row_23<4>(a, s, th, m, h, t);
            __syncthreads();
            core_row_4<4>(a, s, th, m, h, t);
            __syncthreads();
            core_row_5<WRITE_DBG, 4>(a, s, m, h, t);
            if (m + 1 < a.M) __syncthreads();
        }
    }
    else
    {
        #pragma unroll 1
        for (int m = 0; m < a.M; ++m)
        {
            core_row_0<WRITE_DBG, 0>(a, s, th, m, h, t);
            __syncthreads();
            core_row_1<WRITE_DBG, 0>(a, s, th, m, h, t);
            __syncthreads();
            core_row_23<0>(a, s, th, m, h, t);
            __syncthreads();
            core_row_4<0>(a, s, th, m, h, t);
            __syncthreads();
            core_row_5<WRITE_DBG, 0>(a, s, m, h, t);
            if (m + 1 < a.M) __syncthreads();
        }
    }
}

__global__ void k_kda_reset(float* S, float* conv, size_t s_layer, size_t c_layer, size_t s_slot, size_t c_slot, int slot)
{
    const int li = blockIdx.y;
    float* s = S + li * s_layer + slot * s_slot;
    float* c = conv + li * c_layer + slot * c_slot;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < s_slot; i += static_cast<size_t>(gridDim.x) * blockDim.x) s[i] = 0.f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < c_slot; i += static_cast<size_t>(gridDim.x) * blockDim.x) c[i] = 0.f;
}

// ---------------------------------------------------------------------------------------------- launch
inline void reset(const KdaState& s, int slot, cudaStream_t st)
{
    k_kda_reset<<<dim3(32, N_KDA), 256, 0, st>>>(s.S, s.conv, s.s_layer(), s.c_layer(), s.s_layer() / s.slots, s.c_layer() / s.slots, slot);
}

inline Args make_args(const KdaLayer& w, const KdaState& s, const KdaScratch& sc, const int* row_slot, int M)
{
    Args a;
    a.nh = w.nh; a.h0 = w.h0; a.M = M;
    for (int i = 0; i < 3; ++i) a.conv_w[i] = w.conv[i];
    a.fb = w.fb; a.gb = w.gb; a.A_log = w.A_log; a.dt_bias = w.dt_bias; a.onorm = w.onorm;
    a.S = s.S + w.li * s.s_layer();
    a.cst = s.conv + w.li * s.c_layer();
    a.nhmax = s.nhmax;
    a.row_slot = row_slot;
    a.src = s.src;
    a.P = sc.P; a.C = sc.C; a.G = sc.G; a.Z = sc.Z; a.O = sc.O; a.OG = sc.OG;
    a.OF = w.ot.w ? nullptr : sc.OF;
    a.OH = w.ot.w ? sc.OH : nullptr;
    return a;
}

// prep, step, out on projections already in a.P (launch, and the tests from golden projections)
inline void launch_core(const Args& a, cudaStream_t st, bool write_dbg = true)
{
    if (a.M == 1)
    {
        if (write_dbg) k_kda_core<true, 1><<<a.nh, CORE_T, 0, st>>>(a);
        else           k_kda_core<false, 1><<<a.nh, CORE_T, 0, st>>>(a);
    }
    else if (a.M <= 4)
    {
        if (write_dbg) k_kda_core<true, 4><<<a.nh, CORE_T, 0, st>>>(a);
        else           k_kda_core<false, 4><<<a.nh, CORE_T, 0, st>>>(a);
    }
    else
    {
        if (write_dbg) k_kda_core<true, 0><<<a.nh, CORE_T, 0, st>>>(a);
        else           k_kda_core<false, 0><<<a.nh, CORE_T, 0, st>>>(a);
    }
    FMARK(FM_K_PREP, st);
    FMARK(FM_K_STEP, st);
    FMARK(FM_K_OUT, st);
}
// o_proj of M <= g6mx::MX rows from the fragment table OF (K = nh * 128) or OH fp16 into partial [M][HID]
inline void launch_oproj(const KdaLayer& w, const uint16_t* OF, int M, void* partial, int odt, cudaStream_t st,
                         const uint16_t* OH = nullptr, const dtr::Scratch* dt = nullptr
#ifdef DEC_SLOT
                         , const dslot::Slot* slot = nullptr
#endif
                         )
{
    if (w.ot.w)
    {
        dtr::Gemv g;
        const int dodt = odt == gv::O_BF16 ? dtr::O_BF16 : dtr::O_F32;
        dtr::make_out(w.ot, M, reinterpret_cast<const __half*>(OH), w.nh * DK, nullptr, partial, HID, dodt, *dt, 192, 0, 1, g);
#ifdef DEC_SLOT
        if (slot) g.a.slot = *slot;
#endif
        dtr::launch(g, st);
        return;
    }
    ProjArgs po;
    po.xfrag = reinterpret_cast<const u32*>(OF); po.M = M; po.ld = HID; po.odt = odt;
    po.add(w.o, partial, w.nh * DK, HID, 4, 8);
    launch_proj(po, st);
}
inline void launch_oproj(const KdaLayer& w, const KdaScratch& sc, int M, void* partial, int odt, cudaStream_t st,
                         const dtr::Scratch* dt = nullptr
#ifdef DEC_SLOT
                         , const dslot::Slot* slot = nullptr
#endif
                         )
{
    launch_oproj(w, sc.OF, M, partial, odt, st, sc.OH, dt
#ifdef DEC_SLOT
                 , slot
#endif
                 );
}

struct ProjTrellisArgs
{
    const uint8_t* w[3] = {};
    const __half* suh[3] = {};
    const __half* svh[3] = {};
    long long sB = 0, sG = 0;
    int K = 0, N = 0, ks = 16, M = 0;
    const __half* x = nullptr;
    int ldx = 0;
    u32* xa = nullptr;
    float* part = nullptr;
    int* cnt = nullptr;
    float* y[3] = {};
    int ldy = 0;
    int dtr_ctas = 0;
    ProjArgs pa;
};

template <class C, int NT>
__device__ __forceinline__ void gv_gemv_256_ks32(const uint8_t* stream, const u32* xfrag, int K, int out_tiles,
                                                 int M, void* y, int ystride, int cta, float4 (&red)[32][32])
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ot = cta;
    if (ot >= out_tiles) return;
    const int NG2 = K / (16 * C::G), NGP = NG2 / 32;
    float acc[4][NT][4];
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub)
    {
        #pragma unroll
        for (int t = 0; t < NT; ++t) acc[sub][t][0] = acc[sub][t][1] = acc[sub][t][2] = acc[sub][t][3] = 0.f;
        gv_warp_part<C, NT>(stream, xfrag, K, ot, NG2, NGP, sub * 8 + warp, acc[sub]);
    }
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub)
            red[sub * 8 + warp][lane] = make_float4(acc[sub][t][0], acc[sub][t][1], acc[sub][t][2], acc[sub][t][3]);
        __syncthreads();
        if (warp == 0)
            for (int q = 1; q < 32; ++q)
            {
                const float4 v = red[q][lane];
                acc[0][t][0] += v.x; acc[0][t][1] += v.y; acc[0][t][2] += v.z; acc[0][t][3] += v.w;
            }
        if (t + 1 < NT) __syncthreads();
    }
    if (warp != 0) return;
    const int r = lane >> 2, c = lane & 3;
    const size_t o = static_cast<size_t>(ot) * 16 + r;
    float* yf = static_cast<float*>(y);
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        const int m0 = t * 8 + 2 * c;
        if (m0 < M)     { yf[static_cast<size_t>(m0) * ystride + o] = acc[0][t][0]; yf[static_cast<size_t>(m0) * ystride + o + 8] = acc[0][t][2]; }
        if (m0 + 1 < M) { yf[static_cast<size_t>(m0 + 1) * ystride + o] = acc[0][t][1]; yf[static_cast<size_t>(m0 + 1) * ystride + o + 8] = acc[0][t][3]; }
    }
}

// Pre-rotate x * suh[0..2] (12 CTAs of 8 warps = 96 warps for 3 * 32 blocks of 128 channels) concurrently with
// the 20 f_a / g_a / b_proj CTAs in a single 32-CTA wave (< 48 SMs).
template <int NT>
__global__ void __launch_bounds__(256) k_kda_rot_pa(ProjTrellisArgs a)
{
    __shared__ __align__(16) float4 red[32][32];
    const int bidx = static_cast<int>(blockIdx.x);
    if (bidx < 12)
    {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int mat = bidx >> 2, blk = (bidx & 3) * 8 + warp;
        const int K16 = a.K >> 4, k = blk * 128 + lane * 4;
        float s[4];
        dtr::ld4h(a.suh[mat] + k, s);
        u32* xa_mat = a.xa + static_cast<size_t>(mat) * NT * K16 * 64;
        for (int m = 0; m < a.M; ++m)
        {
            float xv[4];
            dtr::ld4h(a.x + static_cast<size_t>(m) * a.ldx + k, xv);
            float v[4] = {xv[0] * s[0], xv[1] * s[1], xv[2] * s[2], xv[3] * s[3]};
            had128(v, lane);
            store_frag_row(xa_mat + static_cast<size_t>(m >> 3) * K16 * 64, blk * 8, m & 7, lane, v);
        }
        return;
    }
    const int gcta = bidx - 12;
    int i = 0;
    while (i + 1 < a.pa.n && gcta >= a.pa.p[i + 1].cta0) ++i;
    const ProjPart& p = a.pa.p[i];
    if (p.codec == 1) gv_gemv_256_ks32<Mxfp8, NT>(p.w, p.xfrag, p.K, p.ot, a.pa.M, p.y, a.pa.ld, gcta - p.cta0, red);
    else              gv_gemv_256_ks32<Bf16w, NT>(p.w, p.xfrag, p.K, p.ot, a.pa.M, p.y, a.pa.ld, gcta - p.cta0, red);
}

// Pre-rotated KDA trellis q/k/v GEMVs + 20 tail pa CTAs.
template <int NT>
__global__ void __launch_bounds__(256, NT <= 2 ? 4 : 2) k_kda_proj_trellis_norot(ProjTrellisArgs a)
{
    constexpr int BITS = 4, TB = 32 * BITS, NW = 8, KDA_UNR = 8;
    __shared__ __align__(16) u32 sm[4096];
    if (static_cast<int>(blockIdx.x) >= a.dtr_ctas)
    {
        const int gcta = static_cast<int>(blockIdx.x) - a.dtr_ctas;
        int i = 0;
        while (i + 1 < a.pa.n && gcta >= a.pa.p[i + 1].cta0) ++i;
        const ProjPart& p = a.pa.p[i];
        auto& red = *reinterpret_cast<float4 (*)[32][32]>(sm);
        if (p.codec == 1) gv_gemv_256_ks32<Mxfp8, NT>(p.w, p.xfrag, p.K, p.ot, a.pa.M, p.y, a.pa.ld, gcta - p.cta0, red);
        else              gv_gemv_256_ks32<Bf16w, NT>(p.w, p.xfrag, p.K, p.ot, a.pa.M, p.y, a.pa.ld, gcta - p.cta0, red);
        return;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, j = warp & 7;
    const int p    = (a.ks == 16) ? (blockIdx.x & 15) : ((a.ks == 8) ? (blockIdx.x & 7) : (blockIdx.x % a.ks));
    const int cidx = (a.ks == 16) ? (blockIdx.x >> 4) : ((a.ks == 8) ? (blockIdx.x >> 3) : (blockIdx.x / a.ks));
    const int mat  = cidx % 3, nb = cidx / 3;
    const int K16 = a.K >> 4, sc = K16 / a.ks, s0 = p * sc;
    const int g0 = s0 >> 3, ng = sc >> 3;
    const uint8_t* wb = a.w[mat] + nb * a.sB + j * TB;
    const uint8_t* gb = wb + g0 * a.sG;
    Tr3Lane<BITS, true> ld;
    ld.init(lane);
    u32 na[KDA_UNR];
    #pragma unroll
    for (int u = 0; u < KDA_UNR; ++u) na[u] = ld.loadw(gb + u * (8 * TB), lane);

    const int xts = K16 * 64;
    const u32* xb = a.xa + (static_cast<size_t>(mat) * NT * K16 + s0) * 64 + lane * 2;
    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    if (ng == 1)
    {
        #pragma unroll
        for (int u = 0; u < KDA_UNR; ++u)
        {
            u32 f0[2], f1[2];
            ld.decodew(na[u], f0, f1);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const uint2 b = __ldg(reinterpret_cast<const uint2*>(xb + t * xts + u * 64));
                mma16816(acc[t], f0[0], f1[0], f0[1], f1[1], b.x, b.y);
            }
        }
    }
    else
    {
        for (int g = 0; g < ng; ++g)
        {
            u32 wa[KDA_UNR];
            #pragma unroll
            for (int u = 0; u < KDA_UNR; ++u) wa[u] = na[u];
            if (g + 1 < ng)
            {
                gb += a.sG;
                #pragma unroll
                for (int u = 0; u < KDA_UNR; ++u) na[u] = ld.loadw(gb + u * (8 * TB), lane);
            }
            #pragma unroll
            for (int u = 0; u < KDA_UNR; ++u)
            {
                u32 f0[2], f1[2];
                ld.decodew(wa[u], f0, f1);
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(xb + t * xts + u * 64));
                    mma16816(acc[t], f0[0], f1[0], f0[1], f1[1], b.x, b.y);
                }
            }
            xb += KDA_UNR * 64;
        }
    }
    const int M = a.M;
    float* pp = a.part + (static_cast<size_t>(cidx) * a.ks + p) * M * 128;
    const int col = j * 16 + (lane >> 2), c = lane & 3;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        const int m0 = t * 8 + 2 * c;
        if (m0 < M)     { pp[m0 * 128 + col] = acc[t][0]; pp[m0 * 128 + col + 8] = acc[t][2]; }
        if (m0 + 1 < M) { pp[(m0 + 1) * 128 + col] = acc[t][1]; pp[(m0 + 1) * 128 + col + 8] = acc[t][3]; }
    }
    if (a.ks > 1) __threadfence();
    __syncthreads();
    int* last_p = reinterpret_cast<int*>(sm);
    if (threadIdx.x == 0) *last_p = a.ks == 1 || atomicAdd(a.cnt + cidx, 1) == a.ks - 1;
    __syncthreads();
    if (!*last_p) return;
    if (a.ks > 1) __threadfence();

    const int n0 = nb * 128 + lane * 4;
    for (int m = warp; m < M; m += NW)
    {
        const float* pb = a.part + static_cast<size_t>(cidx) * a.ks * M * 128 + m * 128 + lane * 4;
        float v[4] = {0.f, 0.f, 0.f, 0.f}, s[4];
        dtr::ld4h(a.svh[mat] + n0, s);
        #pragma unroll 4
        for (int r = 0; r < a.ks; ++r)
        {
            const float4 w = __ldcg(reinterpret_cast<const float4*>(pb + static_cast<size_t>(r) * M * 128));
            v[0] += w.x; v[1] += w.y; v[2] += w.z; v[3] += w.w;
        }
        had128(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] *= s[e];
        *reinterpret_cast<float4*>(a.y[mat] + static_cast<size_t>(m) * a.ldy + n0) = make_float4(v[0], v[1], v[2], v[3]);
    }
    if (a.ks > 1 && threadIdx.x == 0) a.cnt[cidx] = 0;
}

// Fused KDA input projections: 3 trellis K4 GEMVs (q_proj, k_proj, v_proj) + 3 .mx.gv/.gv GEMVs (f_a, g_a, b_proj)
// in one launch. Preserves exact per-matrix accumulation order (0-ULP bit-identical to the 4 separate launches).
template <int NT>
__global__ void __launch_bounds__(256) k_kda_proj_trellis(ProjTrellisArgs a)
{
    constexpr int BITS = 4, TB = 32 * BITS, NW = 8, KDA_UNR = 8;
    __shared__ __align__(16) u32 sm[4096];
    if (static_cast<int>(blockIdx.x) >= a.dtr_ctas)
    {
        const int gcta = static_cast<int>(blockIdx.x) - a.dtr_ctas;
        int i = 0;
        while (i + 1 < a.pa.n && gcta >= a.pa.p[i + 1].cta0) ++i;
        const ProjPart& p = a.pa.p[i];
        auto& red = *reinterpret_cast<float4 (*)[32][32]>(sm);
        if (p.codec == 1) gv_gemv_256_ks32<Mxfp8, NT>(p.w, p.xfrag, p.K, p.ot, a.pa.M, p.y, a.pa.ld, gcta - p.cta0, red);
        else              gv_gemv_256_ks32<Bf16w, NT>(p.w, p.xfrag, p.K, p.ot, a.pa.M, p.y, a.pa.ld, gcta - p.cta0, red);
        return;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, j = warp & 7;
    const int p    = (a.ks == 16) ? (blockIdx.x & 15) : ((a.ks == 8) ? (blockIdx.x & 7) : (blockIdx.x % a.ks));
    const int cidx = (a.ks == 16) ? (blockIdx.x >> 4) : ((a.ks == 8) ? (blockIdx.x >> 3) : (blockIdx.x / a.ks));
    const int mat  = cidx % 3, nb = cidx / 3;
    const int K16 = a.K >> 4, sc = K16 / a.ks, sw = sc, s0c = p * sc, s0 = s0c;
    const int g0 = s0 >> 3, ng = sw >> 3;
    const uint8_t* wb = a.w[mat] + nb * a.sB + j * TB;
    const uint8_t* gb = wb + g0 * a.sG;
    Tr3Lane<BITS, true> ld;
    ld.init(lane);
    u32 na0[KDA_UNR], na1[KDA_UNR];
    #pragma unroll
    for (int u = 0; u < KDA_UNR; ++u) na0[u] = ld.loadw(gb + u * (8 * TB), lane);
    if (ng >= 2)
    {
        const uint8_t* gb1 = gb + a.sG;
        #pragma unroll
        for (int u = 0; u < KDA_UNR; ++u) na1[u] = ld.loadw(gb1 + u * (8 * TB), lane);
    }

    const __half* suh = a.suh[mat];
    const int nblk_k = sc >> 3;
    const int ntask = a.M * nblk_k;
    for (int t = warp; t < ntask; t += NW)
    {
        const int m = (a.M == 1) ? 0 : ((a.M == 4) ? (t & 3) : (t % a.M));
        const int blk = (a.M == 1) ? t : ((a.M == 4) ? (t >> 2) : (t / a.M));
        const int k = s0c * 16 + blk * 128 + lane * 4;
        float x[4], s[4];
        dtr::ld4h(a.x + static_cast<size_t>(m) * a.ldx + k, x);
        dtr::ld4h(suh + k, s);
        float v[4] = {x[0] * s[0], x[1] * s[1], x[2] * s[2], x[3] * s[3]};
        had128(v, lane);
        store_frag_row(sm + (m >> 3) * sc * 64, blk * 8, m & 7, lane, v);
    }
    __syncthreads();
    const u32* xb = sm + lane * 2;
    const int xts = sc * 64;

    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    if (ng == 2)
    {
        #pragma unroll
        for (int u = 0; u < KDA_UNR; ++u)
        {
            u32 f0[2], f1[2];
            ld.decodew(na0[u], f0, f1);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const uint2 b = *reinterpret_cast<const uint2*>(xb + t * xts + u * 64);
                mma16816(acc[t], f0[0], f1[0], f0[1], f1[1], b.x, b.y);
            }
        }
        #pragma unroll
        for (int u = 0; u < KDA_UNR; ++u)
        {
            u32 f0[2], f1[2];
            ld.decodew(na1[u], f0, f1);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const uint2 b = *reinterpret_cast<const uint2*>(xb + t * xts + (KDA_UNR + u) * 64);
                mma16816(acc[t], f0[0], f1[0], f0[1], f1[1], b.x, b.y);
            }
        }
    }
    else
    {
        for (int g = 0; g < ng; ++g)
        {
            u32 wa[KDA_UNR];
            #pragma unroll
            for (int u = 0; u < KDA_UNR; ++u) wa[u] = na0[u];
            if (g + 1 < ng)
            {
                gb += a.sG;
                #pragma unroll
                for (int u = 0; u < KDA_UNR; ++u) na0[u] = ld.loadw(gb + u * (8 * TB), lane);
            }
            #pragma unroll
            for (int u = 0; u < KDA_UNR; ++u)
            {
                u32 f0[2], f1[2];
                ld.decodew(wa[u], f0, f1);
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    const uint2 b = *reinterpret_cast<const uint2*>(xb + t * xts + u * 64);
                    mma16816(acc[t], f0[0], f1[0], f0[1], f1[1], b.x, b.y);
                }
            }
            xb += KDA_UNR * 64;
        }
    }
    const int M = a.M;
    float* pp = a.part + (static_cast<size_t>(cidx) * a.ks + p) * M * 128;
    const int col = j * 16 + (lane >> 2), c = lane & 3;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        const int m0 = t * 8 + 2 * c;
        if (m0 < M)     { pp[m0 * 128 + col] = acc[t][0]; pp[m0 * 128 + col + 8] = acc[t][2]; }
        if (m0 + 1 < M) { pp[(m0 + 1) * 128 + col] = acc[t][1]; pp[(m0 + 1) * 128 + col + 8] = acc[t][3]; }
    }
    if (a.ks > 1) __threadfence();
    __syncthreads();
    int* last_p = reinterpret_cast<int*>(sm);
    if (threadIdx.x == 0) *last_p = a.ks == 1 || atomicAdd(a.cnt + cidx, 1) == a.ks - 1;
    __syncthreads();
    if (!*last_p) return;
    if (a.ks > 1) __threadfence();

    const int n0 = nb * 128 + lane * 4;
    for (int m = warp; m < M; m += NW)
    {
        const float* pb = a.part + static_cast<size_t>(cidx) * a.ks * M * 128 + m * 128 + lane * 4;
        float v[4] = {0.f, 0.f, 0.f, 0.f}, s[4];
        if (a.ks == 16)
        {
            #pragma unroll
            for (int r = 0; r < 16; ++r)
            {
                const float4 w = __ldcg(reinterpret_cast<const float4*>(pb + static_cast<size_t>(r) * M * 128));
                v[0] += w.x; v[1] += w.y; v[2] += w.z; v[3] += w.w;
            }
        }
        else
        {
            for (int r = 0; r < a.ks; ++r)
            {
                const float4 w = __ldcg(reinterpret_cast<const float4*>(pb + static_cast<size_t>(r) * M * 128));
                v[0] += w.x; v[1] += w.y; v[2] += w.z; v[3] += w.w;
            }
        }
        had128(v, lane);
        dtr::ld4h(a.svh[mat] + n0, s);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] *= s[e];
        *reinterpret_cast<float4*>(a.y[mat] + static_cast<size_t>(m) * a.ldy + n0) = make_float4(v[0], v[1], v[2], v[3]);
    }
    if (a.ks > 1 && threadIdx.x == 0) a.cnt[cidx] = 0;
}

// partial: BF16 [M][HID] (odt O_BF16), or fp32 with odt = O_F32 (tests)
// q, k, v (own rows), f_a, g_a, b_proj projections of M <= g6mx::MX rows of xb / x / xf into P [M][a.pl().ld()]
inline void launch_proj_in(const KdaLayer& w, const Args& a, float* P, const u32* xb, int M, cudaStream_t st,
                           const __half* x = nullptr, const u32* xf = nullptr, const dtr::Scratch* dt = nullptr,
                           u32* xa = nullptr)
{
    const PLayout pl = a.pl();
    const int dl = a.dl();
    ProjArgs pa;
    pa.xfrag = xb; pa.M = M; pa.ld = pl.ld(); pa.odt = gv::O_F32;
    if (!w.qt.w)
    {
        pa.add(w.q, P + pl.q(), HID, dl, 4, 8);
        pa.add(w.k, P + pl.q() + dl, HID, dl, 4, 8);
        pa.add(w.v, P + pl.q() + 2 * dl, HID, dl, 4, 8);
    }
    if (w.fa_mx) pa.add(w.fa_mx, P + pl.fa(), HID, DK, 1, 32, xf, 1);
    else         pa.add(w.fa,    P + pl.fa(), HID, DK, 1, 32);
    if (w.ga_mx) pa.add(w.ga_mx, P + pl.ga(), HID, DK, 1, 32, xf, 1);
    else         pa.add(w.ga,    P + pl.ga(), HID, DK, 1, 32);
    pa.add(w.b, P + pl.b(), HID, HEADS, 1, 32);
    if (w.qt.w)
    {
        static const int ks_env = getenv("G6_KDA_PROJ_KS") ? atoi(getenv("G6_KDA_PROJ_KS")) : 16;
        static const int mode_env = !getenv("G6_KDA_PROJ_MODE") ? 0 : atoi(getenv("G6_KDA_PROJ_MODE"));
        const int KS = (ks_env == 8 || ks_env == 16 || ks_env == 32) ? ks_env : 16;
        const int nblk = dl / 128;
        if (dt && dtr::part_floats(dl, 3, KS, M) <= dt->part_n && 3 * nblk <= dt->cnt_n)
        {
            ProjTrellisArgs ta;
            const dtr::Mat* m3[3] = {&w.qt, &w.kt, &w.vt};
            for (int s = 0; s < 3; ++s)
            {
                ta.w[s] = m3[s]->w;
                ta.suh[s] = m3[s]->suh;
                ta.svh[s] = m3[s]->svh;
                ta.y[s] = P + pl.q() + s * dl;
            }
            ta.sB = w.qt.sB; ta.sG = w.qt.sG;
            ta.K = HID; ta.N = dl; ta.ks = KS; ta.M = M;
            ta.x = x; ta.ldx = HID; ta.xa = xa;
            ta.part = dt->part; ta.cnt = dt->cnt; ta.ldy = pl.ld();
            ta.dtr_ctas = nblk * 3 * KS;
            ta.pa = pa;
            g6mx::nt_dispatch(g6mx::tiles(M), [&](auto N) {
                constexpr int NTv = decltype(N)::value;
                if (xa && mode_env == 1)
                {
                    k_kda_rot_pa<NTv><<<12 + ta.pa.ctas, 256, 0, st>>>(ta);
                    k_kda_proj_trellis_norot<NTv><<<ta.dtr_ctas, 256, 0, st>>>(ta);
                }
                else if (xa && mode_env == 2)
                {
                    k_kda_rot_pa<NTv><<<12, 256, 0, st>>>(ta);
                    k_kda_proj_trellis_norot<NTv><<<ta.dtr_ctas + ta.pa.ctas, 256, 0, st>>>(ta);
                }
                else
                {
                    k_kda_proj_trellis<NTv><<<ta.dtr_ctas + ta.pa.ctas, 256, 0, st>>>(ta);
                }
            });
            return;
        }
        const dtr::Mat* m3[3] = {&w.qt, &w.kt, &w.vt};
        for (int s = 0; s < 3; ++s)
        {
            dtr::Gemv g;
            dtr::make_out(*m3[s], M, x, HID, nullptr, P + pl.q() + s * dl, pl.ld(), dtr::O_F32, *dt, s * 64, 0, 1, g);
            dtr::launch(g, st);
        }
    }
    launch_proj(pa, st);
}

inline void launch(const KdaLayer& w, const KdaState& s, const KdaScratch& sc, const u32* xb, const int* row_slot, int M,
                   void* partial, cudaStream_t st, int odt = gv::O_BF16,
                   const __half* x = nullptr, const u32* xf = nullptr, const dtr::Scratch* dt = nullptr
#ifdef DEC_SLOT
                   , const dslot::Slot* slot = nullptr
#endif
                   )
{
    const Args a = make_args(w, s, sc, row_slot, M);
    u32* xa = (dt != nullptr && sc.C != nullptr) ? reinterpret_cast<u32*>(sc.C) : nullptr;
    launch_proj_in(w, a, sc.P, xb, M, st, x, xf, dt, xa);
    FMARK(FM_K_PROJ, st);
    launch_core(a, st, dt == nullptr);
    launch_oproj(w, sc, M, partial, odt, st, dt
#ifdef DEC_SLOT
                 , slot
#endif
                 );
    FMARK(FM_K_OPROJ, st);
}
}   // namespace kda
