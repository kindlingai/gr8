// G6_L2PF: a paced L2 prefetch of the weights the decode step reads next, issued in the step's windows where DRAM is
// idle (the collective waits and the latency-bound attention). Modelled on blaesus/dsgx10 (ds41f kernels/l2pf.cu,
// docs/engine.md "A paced L2 prefetch in the step's windows"; TensorFold's paced one-block prefetch): one warp on a
// high-priority side stream issues cp.async.bulk.prefetch.L2 of 8 KiB chunks from a token bucket (G6_L2PF_GBPS, in
// waves of at most G6_L2PF_WAVE chunks), up to a byte budget per window, and a collective window ends once its
// collective has arrived (a second warp polls the peers' tails, as k_wait_peers does). Not the burst our G6_OPROJ_PF
// tried (8 CTAs issuing at once: dependent loads 427 -> 2049 ns in dsgx10's measurement, our attention +12..43 us):
// small waves leave the attention's dependent loads alone (436 ns at 175 GB/s in waves of 8).
// cp.async.bulk.prefetch.L2 (sm_90+, so sm_121a): one instruction per chunk handled by the bulk-copy unit, no register,
// no shared memory, no completion to wait for; it writes nothing and changes no load's value, so every output byte is
// unchanged. (prefetch.global.L2 would take 128 instructions per 8 KiB; ld + discard ties up registers and load slots.)
//
// A layer's targets in the order the step reads them, one queue per layer (plan_layer): x_in (q_a, kv_a, idx_k, idx_w;
// the rank's tiles under G6_QA_SPLIT), idx_q, q_b, UK, UV, o_proj (capped at G6_L2PF_OPMB), router, the shared expert's
// fc1 (the dense MLP's gate + up on layers 0-2). Windows (G6_L2PF bits), each from its first target not yet read, and
// continuing where the layer's earlier windows ended (State, device memory):
//   1 = W_M  the wait for the MLP partials of layer i-1: layer i's queue from x_in
//   2 = W_Q  the q_a gather wait (G6_QA_SPLIT) from idx_q, the idx_q gather wait (G6_IDXQ_SPLIT) from UK
//   4 = W_T  the attention (no collective: its budget only) from UK
//   8 = W_A  the wait for the attention partials from the router
// Each window issues at most G6_L2PF_US[w] us at the pace and G6_L2PF_MB MiB. The side stream is forked at each window
// and joined once at the end of the step (part), so the step's stream never waits for a prefetch. Off (G6_L2PF unset or
// 0): no stream, no launch, no fork; the graphs are unchanged.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "bind.cuh"
#include "glue.cuh"

namespace l2pf
{

constexpr uint32_t CHUNK = 8192;      // bytes per bulk prefetch
constexpr int MAXR = 20;              // ranges in a queue (16 + kv_b FP8 scales, #50, with headroom)
enum { W_M, W_Q, W_T, W_A, NW };
enum { T_XIN, T_IDXQ, T_QB, T_UK, T_UV, T_OPROJ, T_ROUTER, T_SHARED, NTG };

// a layer's targets: up to 4 byte ranges each, in read order
struct Plan
{
    const void* p[NTG][4] = {};
    size_t n[NTG][4] = {};
    void add(int t, const void* q, size_t b)
    {
        for (int k = 0; k < 4; ++k)
            if (!n[t][k]) { p[t][k] = q; n[t][k] = b; return; }
    }
};
struct Queue
{
    const uint8_t* p[MAXR];
    uint32_t n[MAXR];                 // bytes, multiples of 16
    uint32_t end[MAXR];               // chunks up to and including range i
    uint32_t from[NTG];               // first chunk of target t
    int count;
};
struct State { unsigned long long tag; uint32_t chunk; };   // tag = step << 8 | layer + 1 of the last window
struct Stop                           // a collective window ends when every peer's tail of collective c holds its seq
{
    g6::FabricDev f{};                // f.nr = 0: no collective (the attention window)
    const u64* stepp = nullptr;
    int c = 0;
    size_t payload = 0;
};

__device__ __forceinline__ unsigned long long gtimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// block 64: warp 0 issues chunks [max(first, where the layer's last window ended), + budget) at one per ns_chunk ns
// (a poll issues those due, at most wave <= 32); warp 1 polls the collective's tails and stops warp 0 once all arrived.
__global__ void __launch_bounds__(64) k_l2pf(const __grid_constant__ Queue q, int layer, uint32_t first, uint32_t budget,
                                             uint32_t ns_chunk, int wave, Stop s, State* st)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __shared__ volatile int stop, fin;
    const int lane = threadIdx.x & 31;
    if (threadIdx.x == 0) { stop = 0; fin = 0; }
    __syncthreads();
    const u64 step = *s.stepp;
    if (threadIdx.x >= 32)
    {
        if (!s.f.nr) return;
        const u64 seq = g6::coll_seq(step, s.c, s.f.ncoll);
        const bool peer = lane < s.f.nr && lane != s.f.rank;
        const u64* tail = peer ? reinterpret_cast<const u64*>(glue::peer_slot(s.f, step, s.c, lane) + s.payload) : nullptr;
        while (!fin)
        {
            if (__all_sync(~0u, !peer || glue::ld_acquire_sys(tail) == seq)) { stop = 1; break; }
            __nanosleep(128);
        }
        return;
    }
    const unsigned long long tag = step << 8 | static_cast<unsigned>(layer + 1);
    uint32_t c0 = 0;
    if (lane == 0 && st->tag == tag) c0 = st->chunk;
    c0 = max(__shfl_sync(~0u, c0, 0), first);
    const uint32_t total = q.count ? q.end[q.count - 1] : 0, last = min(total, c0 + budget);
    const unsigned long long t0 = __shfl_sync(~0u, gtimer(), 0);
    uint32_t done = min(c0, last);
    while (done < last && !stop)
    {
        const unsigned long long t = __shfl_sync(~0u, gtimer(), 0), by_now = c0 + (t - t0) / ns_chunk + 1;
        const uint32_t cap = min(last, done + static_cast<uint32_t>(wave)), due = by_now < cap ? static_cast<uint32_t>(by_now) : cap;
        const uint32_t c = done + lane;
        if (c < due)
        {
            int i = 0;
            while (c >= q.end[i]) ++i;
            const uint32_t off = (c - (i ? q.end[i - 1] : 0)) * CHUNK, left = q.n[i] - off;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(q.p[i] + off), "r"(min(left, CHUNK)) : "memory");
        }
        if (due > done) done = due;
        else __nanosleep(64);
    }
    if (lane == 0) { st->tag = tag; st->chunk = max(done, c0); }
    fin = 1;
#endif
}

struct Cfg
{
    int win = 0;                      // G6_L2PF: window mask (0 = off)
    unsigned tmask = (1u << NTG) - 1; // G6_L2PF_T: targets (bit = T_*)
    float gbps = 175;                 // G6_L2PF_GBPS: the pace
    int wave = 8;                     // G6_L2PF_WAVE: chunks per poll at most (1..32)
    float us[NW] = {30, 30, 100, 30}; // G6_L2PF_US: budget per window, us at the pace (W_M, W_Q, W_T, W_A)
    float mb = 10;                    // G6_L2PF_MB: MiB per window at most
    float opmb = 8;                   // G6_L2PF_OPMB: o_proj's entries prefetched per layer, MB (0 = all)
    cudaStream_t side = nullptr;
    cudaEvent_t fork = nullptr, join = nullptr;
    State* state = nullptr;
    int prio = 0;
    bool armed = false, pending = false;
};
inline Cfg& cfg() { static Cfg c; return c; }

// the side stream, events and state (once, outside any capture); win = 0 makes nothing
inline cudaError_t setup(const Cfg& in)
{
    Cfg& c = cfg();
    const cudaStream_t side = c.side;
    const cudaEvent_t fork = c.fork, join = c.join;
    State* state = c.state;
    const int prio = c.prio;
    c = in;
    c.side = side; c.fork = fork; c.join = join; c.state = state; c.prio = prio;
    c.armed = c.pending = false;
    c.wave = c.wave < 1 ? 1 : c.wave > 32 ? 32 : c.wave;
    if (!(c.gbps > 0)) c.gbps = 175;
    if (!c.win || c.side) return cudaSuccess;
    int lo = 0, hi = 0;
    cudaError_t e = cudaDeviceGetStreamPriorityRange(&lo, &hi);
    c.prio = hi;
    if (e == cudaSuccess) e = cudaStreamCreateWithPriority(&c.side, cudaStreamNonBlocking, hi);
    if (e == cudaSuccess) e = cudaEventCreateWithFlags(&c.fork, cudaEventDisableTiming);
    if (e == cudaSuccess) e = cudaEventCreateWithFlags(&c.join, cudaEventDisableTiming);
    if (e == cudaSuccess) e = cudaMalloc(&c.state, sizeof(State));
    if (e == cudaSuccess) e = cudaMemset(c.state, 0, sizeof(State));
    return e;
}
// G6_L2PF, G6_L2PF_T, G6_L2PF_GBPS, G6_L2PF_WAVE, G6_L2PF_US=wm,wq,wt,wa, G6_L2PF_MB, G6_L2PF_OPMB (bind::init)
inline cudaError_t init_env()
{
    Cfg c;
    const char* e = getenv("G6_L2PF");
    c.win = e ? atoi(e) & 15 : 0;
    if (!c.win) return cudaSuccess;
    if ((e = getenv("G6_L2PF_T")) && *e) c.tmask = static_cast<unsigned>(strtoul(e, nullptr, 0));
    if ((e = getenv("G6_L2PF_GBPS")) && *e) c.gbps = static_cast<float>(atof(e));
    if ((e = getenv("G6_L2PF_WAVE")) && *e) c.wave = atoi(e);
    if ((e = getenv("G6_L2PF_US")) && *e) sscanf(e, "%f,%f,%f,%f", &c.us[0], &c.us[1], &c.us[2], &c.us[3]);
    if ((e = getenv("G6_L2PF_MB")) && *e) c.mb = static_cast<float>(atof(e));
    if ((e = getenv("G6_L2PF_OPMB")) && *e) c.opmb = static_cast<float>(atof(e));
    fprintf(stderr, "# G6_L2PF=%d: targets 0x%x, %.0f GB/s in waves of %d x 8 KiB, budgets %.0f/%.0f/%.0f/%.0f us, <= %.1f MiB a window, "
            "o_proj <= %.1f MB\n", c.win, c.tmask, c.gbps, c.wave, c.us[0], c.us[1], c.us[2], c.us[3], c.mb, c.opmb);
    return setup(c);
}

// a layer's targets from its binding and its attention weights (UK [H][512][192], UV [H][256][512] BF16)
inline Plan plan_layer(const bind::Layer& b, const void* uk, const void* uv, const void* uks = nullptr, const void* uvs = nullptr)
{
    Plan P;
    for (const bind::Gv* g : {&b.q_a, &b.kv_a, &b.idx_k, &b.idx_w})
        if (g->on) P.add(T_XIN, g->w, bind::gv_bytes(*g));
    if (b.idx_q.on) P.add(T_IDXQ, b.idx_q.w, bind::gv_bytes(b.idx_q));
    auto dtr_add = [&](int t, const dtr::Gemv& g, size_t cap) {
        P.add(t, g.a.suh[0], static_cast<size_t>(g.a.K) * 2);
        P.add(t, g.a.svh[0], static_cast<size_t>(g.a.N) * 2);
        const size_t n = static_cast<size_t>(g.a.K) * g.a.N * dtr::BITS / 8;
        P.add(t, g.a.w[0], cap && cap < n ? cap : n);
    };
    if (b.q_b_t.on) dtr_add(T_QB, b.q_b_t, 0);
    else if (b.q_b.on) P.add(T_QB, b.q_b.w, bind::gv_bytes(b.q_b));
    const int H = b.heads;
    // kv_b BF16, or E4M3 with f32 block scales (#50, G6_KVB=fp8: uks / uvs set): one byte a weight plus the scales
    const size_t ukw = uks ? 1 : 2, uvw = uvs ? 1 : 2;
    if (uk) P.add(T_UK, uk, static_cast<size_t>(H) * 512 * 192 * ukw);
    if (uk && uks) P.add(T_UK, uks, static_cast<size_t>(H) * 4 * 192 * 4);
    if (uv) P.add(T_UV, uv, static_cast<size_t>(H) * 256 * 512 * uvw);
    if (uv && uvs) P.add(T_UV, uvs, static_cast<size_t>(H) * 256 * 4 * 4);
    const size_t opcap = static_cast<size_t>(cfg().opmb * 1e6);
    if (b.o_proj_t.on) dtr_add(T_OPROJ, b.o_proj_t, opcap);
    else if (b.o_proj.on) P.add(T_OPROJ, b.o_proj.w, opcap && opcap < bind::gv_bytes(b.o_proj) ? opcap : bind::gv_bytes(b.o_proj));
    if (b.router.on) P.add(T_ROUTER, b.router.w, bind::gv_bytes(b.router));
    if (b.shw) P.add(T_SHARED, b.shw, b.shw_bytes);
    else if (b.glu_t.on) { for (int k = 0; k < 2; ++k) P.add(T_SHARED, b.glu_t.a.w[k], static_cast<size_t>(b.glu_t.a.K) * b.glu_t.a.N * dtr::BITS / 8); }
    else { for (int k = 0; k < 2; ++k) P.add(T_SHARED, b.sh[k].p, bind::rec_bytes(b.sh[k])); }
    return P;
}

inline Queue queue(const Plan& P, unsigned tmask)
{
    Queue q{};
    uint32_t c = 0;
    for (int t = 0; t < NTG; ++t)
    {
        q.from[t] = c;
        if (!(tmask >> t & 1)) continue;
        for (int k = 0; k < 4 && P.n[t][k]; ++k)
        {
            if (q.count == MAXR) { fprintf(stderr, "l2pf: queue full (%d ranges), target %d range %d dropped\n", MAXR, t, k); break; }
            const uintptr_t b = reinterpret_cast<uintptr_t>(P.p[t][k]), lo = (b + 15) & ~static_cast<uintptr_t>(15),
                            hi = (b + P.n[t][k]) & ~static_cast<uintptr_t>(15);
            if (hi <= lo) continue;
            q.p[q.count] = reinterpret_cast<const uint8_t*>(lo);
            q.n[q.count] = static_cast<uint32_t>(hi - lo);
            c += (q.n[q.count] + CHUNK - 1) / CHUNK;
            q.end[q.count++] = c;
        }
    }
    return q;
}

// step (part) begin / end: windows only between them (the main model's launch_step), the side stream joined at the end
inline void arm() { Cfg& c = cfg(); c.armed = c.win != 0 && c.side; }
inline void join(cudaStream_t st)
{
    Cfg& c = cfg();
    c.armed = false;
    if (!c.pending) return;
    cudaEventRecord(c.join, c.side);
    cudaStreamWaitEvent(st, c.join, 0);
    c.pending = false;
}
// window w of layer `layer` (the step's layer index), its first target `from`, on st where the window starts
inline void window(int w, int from, int layer, const Plan& P, const Stop& s, cudaStream_t st)
{
    Cfg& c = cfg();
    if (!c.armed || !(c.win >> w & 1)) return;
    const Queue q = queue(P, c.tmask);
    const uint32_t total = q.count ? q.end[q.count - 1] : 0;
    if (q.from[from] >= total) return;
    const double ns = CHUNK / c.gbps;
    const uint32_t ns_chunk = static_cast<uint32_t>(ns + 0.5) ? static_cast<uint32_t>(ns + 0.5) : 1;
    const double bu = c.us[w] * 1000.0 / ns, bm = c.mb * 1048576.0 / CHUNK;
    const uint32_t budget = static_cast<uint32_t>(bu < bm ? bu : bm);
    if (!budget) return;
    cudaEventRecord(c.fork, st);
    cudaStreamWaitEvent(c.side, c.fork, 0);
    cudaLaunchConfig_t lc = {};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributePriority;
    at[0].val.priority = c.prio;
    lc.gridDim = dim3(1); lc.blockDim = dim3(64); lc.stream = c.side; lc.attrs = at; lc.numAttrs = 1;
    cudaLaunchKernelEx(&lc, k_l2pf, q, layer, q.from[from], budget, ns_chunk, c.wave, s, c.state);
    c.pending = true;
}

}  // namespace l2pf
