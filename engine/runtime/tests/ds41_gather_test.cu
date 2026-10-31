// F7 bit-equality gate of the DS41 prefill gather (GR8_DS_PF_GATHER=1, runtime/ds41_gather.cuh). For each rank r and row count M: the same fabric
// slots (pinned host memory, the real layout: own partial in the send slot, peer p's in recv slot (c, p)) are consumed twice from the same state:
//   ref:    the consume kernels with RefBlock = the Ds41Block slot reads (ds41_step.cuh: 4 x 2-byte __ldcv, FP32 sum in rank order from 0.f)
//   gather: k_ds41_gather -> device BF16 sum -> the consume kernels with Ds41DevBlock
// and every output buffer must match byte for byte: the trunk, y, pre, post, comb, mix (k_step FUSE = consume_prepare), and the trunk (consume).
// Partials: random BF16 incl. +-0, subnormals, large magnitudes, and rows where the 4 partials cancel. Exit 0 = every line identical=yes.
// Build (dev 5090): nvcc -O2 -std=c++17 -DG6_MODEL_DS41=1 -DDS_MHC_TRUNK_BF16 -arch=sm_120 -I engine -o ds41_gather_test engine/runtime/tests/ds41_gather_test.cu
// RefBlock copies Ds41Block::init / operator() (ds41_step.cuh:16-39); keep the two in step.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "../../kernels/glue.cuh"
#include "../../mhc/ds_mhc.cuh"
#include "../ds41_gather.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

struct RefBlock {
    g6::FabricDev f;
    const u64* stepp;
    int c;
    const unsigned short* base[g6nr::NR] = {};
    __device__ __forceinline__ u64 init() {
        const u64 step = *stepp;
        #pragma unroll
        for (int p = 0; p < g6nr::NR; ++p) base[p] = (const unsigned short*)glue::peer_slot(f, step, c, p);
        return step;
    }
    __device__ __forceinline__ void wait(u64) const {}
    __device__ __forceinline__ float operator()(size_t k) const {
        unsigned short v[g6nr::NR];
        #pragma unroll
        for (int p = 0; p < g6nr::NR; ++p) v[p] = __ldcv(base[p] + k);
        float sum = 0.f;
        #pragma unroll
        for (int p = 0; p < g6nr::NR; ++p) sum += __uint_as_float((unsigned)v[p] << 16);
        return sum;
    }
};

using TT = ds_mhc::trunk_t;
static const int H = HID, NC = 3, C = 1;   // collective 1 of 3
static int g_bad = 0;

static unsigned short bf(float v) { unsigned u; memcpy(&u, &v, 4); return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float fl(unsigned short b) { unsigned u = (unsigned)b << 16; float v; memcpy(&v, &u, 4); return v; }

struct State { TT* streams; float *mix, *pre, *post, *comb, *part; unsigned* done; __nv_bfloat16* y; };
static State alloc_state(int M) {
    State s;
    CK(cudaMalloc(&s.streams, (size_t)M * 4 * H * sizeof(TT))); CK(cudaMalloc(&s.mix, (size_t)M * ds_mhc::MIX * 4));
    CK(cudaMalloc(&s.pre, (size_t)M * 16)); CK(cudaMalloc(&s.post, (size_t)M * 16)); CK(cudaMalloc(&s.comb, (size_t)M * 64));
    CK(cudaMalloc(&s.part, (size_t)M * ds_mhc::NV * (H / ds_mhc::PW) * 4));
    const int dw = ds_mhc::done_words(M); CK(cudaMalloc(&s.done, dw * 4)); CK(cudaMemset(s.done, 0, dw * 4));
    CK(cudaMalloc(&s.y, (size_t)M * H * 2));
    return s;
}
static ds_mhc::BuffersT<TT> buf(const State& s) {
    ds_mhc::BuffersT<TT> b{}; b.streams = s.streams; b.mix = s.mix; b.pre = s.pre; b.post = s.post; b.comb = s.comb; b.done = s.done; b.part = s.part;
    return b;
}
template <class T> static std::vector<char> get(const T* d, size_t n) { std::vector<char> h(n * sizeof(T)); CK(cudaMemcpy(h.data(), d, h.size(), cudaMemcpyDeviceToHost)); return h; }
static void copy_state(const State& d, const State& s, int M) {
    CK(cudaMemcpy(d.streams, s.streams, (size_t)M * 4 * H * sizeof(TT), cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(d.pre, s.pre, (size_t)M * 16, cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(d.post, s.post, (size_t)M * 16, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(d.comb, s.comb, (size_t)M * 64, cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(d.mix, s.mix, (size_t)M * ds_mhc::MIX * 4, cudaMemcpyDeviceToDevice));
    CK(cudaMemset(d.y, 0, (size_t)M * H * 2));
}
static void cmp(const char* what, const std::vector<char>& a, const std::vector<char>& b, int r, int M) {
    size_t n = 0; for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
    printf("%s rank=%d M=%d bytes=%zu identical=%s%s\n", what, r, M, a.size(), n ? "no" : "yes", n ? (" (" + std::to_string(n) + " bytes differ)").c_str() : "");
    if (n) ++g_bad;
}

int main() {
    std::mt19937 rng(20261026);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    const int Ms[] = {2048, 846, 128, 49};
    // weights: fn F32 [24][4H], base [24], scale [3], norm [H]
    std::vector<float> fn((size_t)24 * 4 * H), base(24), scale(3); std::vector<unsigned short> norm(H);
    for (auto& v : fn) v = u(rng) * 0.02f; for (auto& v : base) v = u(rng); scale = {0.5f, 0.7f, 0.9f};
    for (auto& v : norm) v = bf(1.f + 0.1f * u(rng));
    float *dfn, *dbase, *dscale; __nv_bfloat16* dnorm;
    CK(cudaMalloc(&dfn, fn.size() * 4)); CK(cudaMemcpy(dfn, fn.data(), fn.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dbase, 96)); CK(cudaMemcpy(dbase, base.data(), 96, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dscale, 12)); CK(cudaMemcpy(dscale, scale.data(), 12, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dnorm, H * 2)); CK(cudaMemcpy(dnorm, norm.data(), H * 2, cudaMemcpyHostToDevice));
    const ds_mhc::Weights<float> W{dfn, dbase, dscale, dnorm};
    for (int M : Ms)
        for (int r : {0, 3}) {
            // fabric: uniform layout, slot = M rows of H BF16 + tail
            g6::FabricConfig cfg; cfg.slot_bytes = (size_t)M * H * 2;
            g6::FabricDev f{}; f.nr = g6nr::NR; f.rank = r;
            if (!g6::fab_slot_layout(cfg, NC, &f)) { printf("layout refused\n"); return 1; }
            const size_t half = g6::fab_slot_base(f, NC), send_b = 2 * half, recv_b = 2 * half * g6nr::NR;
            char* region; CK(cudaHostAlloc((void**)&region, send_b + recv_b, cudaHostAllocMapped | cudaHostAllocPortable));
            f.send = region; f.recv = region + send_b;
            const u64 step = 7;   // parity 1
            u64* dstep; CK(cudaMalloc(&dstep, 8)); CK(cudaMemcpy(dstep, &step, 8, cudaMemcpyHostToDevice));
            // partials: rank p's at peer_slot(step, C, p) (own = send slot)
            std::vector<std::vector<unsigned short>> part(g6nr::NR, std::vector<unsigned short>((size_t)M * H));
            for (size_t k = 0; k < (size_t)M * H; ++k) {
                const int kind = rng() % 16;
                for (int p = 0; p < g6nr::NR; ++p) {
                    float v = u(rng) * (kind == 0 ? 3e4f : kind == 1 ? 1e-39f : 4.f);
                    if (kind == 2) v = p == 0 ? 0.f : -0.f;
                    part[p][k] = bf(v);
                }
                if (kind == 3) { part[1][k] = part[0][k] ^ 0x8000; part[3][k] = part[2][k] ^ 0x8000; }   // the 4 partials cancel
            }
            for (int p = 0; p < g6nr::NR; ++p) {
                char* dst = p == r ? f.send + g6::fab_send_off(f, 1, C) : f.recv + g6::fab_recv_off(f, 1, C, p);
                memcpy(dst, part[p].data(), (size_t)M * H * 2);
            }
            // initial state: random BF16-valued trunk, pre / post / comb in range
            State s0 = alloc_state(M), sa = alloc_state(M), sb = alloc_state(M);
            {
                std::vector<TT> t((size_t)M * 4 * H); for (auto& v : t) v = (TT)fl(bf(u(rng)));
                std::vector<float> pre((size_t)M * 4), post((size_t)M * 4), comb((size_t)M * 16);
                for (auto& v : pre) v = 0.5f + 0.5f * u(rng); for (auto& v : post) v = 1.f + u(rng); for (auto& v : comb) v = 0.25f + 0.2f * u(rng);
                CK(cudaMemcpy(s0.streams, t.data(), t.size() * sizeof(TT), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(s0.pre, pre.data(), pre.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(s0.post, post.data(), post.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(s0.comb, comb.data(), comb.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemset(s0.mix, 0, (size_t)M * ds_mhc::MIX * 4));
            }
            __nv_bfloat16* gsum; CK(cudaMalloc(&gsum, (size_t)M * H * 2));
            const RefBlock ref{f, dstep, C};
            auto gather = [&] {
                k_ds41_gather<<<dim3(H / 8 / DS41_GATHER_T, M), DS41_GATHER_T>>>(f, dstep, C, H, gsum);
                return Ds41DevBlock{gsum, dstep};
            };
            // 1. consume_prepare (k_step FUSE)
            copy_state(sa, s0, M); copy_state(sb, s0, M);
            ds_mhc::consume_prepare<float>(buf(sa), ref, W, M, H, 1e-6f, sa.y, 0);
            ds_mhc::consume_prepare<float>(buf(sb), gather(), W, M, H, 1e-6f, sb.y, 0);
            CK(cudaDeviceSynchronize());
            cmp("consume_prepare trunk", get(sa.streams, (size_t)M * 4 * H), get(sb.streams, (size_t)M * 4 * H), r, M);
            cmp("consume_prepare y", get(sa.y, (size_t)M * H), get(sb.y, (size_t)M * H), r, M);
            cmp("consume_prepare pre/post/comb/mix", [&] { auto a = get(sa.pre, M * 4); auto b = get(sa.post, M * 4); auto c = get(sa.comb, M * 16); auto d = get(sa.mix, (size_t)M * ds_mhc::MIX);
                     a.insert(a.end(), b.begin(), b.end()); a.insert(a.end(), c.begin(), c.end()); a.insert(a.end(), d.begin(), d.end()); return a; }(),
                [&] { auto a = get(sb.pre, M * 4); auto b = get(sb.post, M * 4); auto c = get(sb.comb, M * 16); auto d = get(sb.mix, (size_t)M * ds_mhc::MIX);
                     a.insert(a.end(), b.begin(), b.end()); a.insert(a.end(), c.begin(), c.end()); a.insert(a.end(), d.begin(), d.end()); return a; }(), r, M);
            // 2. consume (plain)
            copy_state(sa, s0, M); copy_state(sb, s0, M);
            ds_mhc::consume(buf(sa), ref, M, H, 0);
            ds_mhc::consume(buf(sb), gather(), M, H, 0);
            CK(cudaDeviceSynchronize());
            cmp("consume trunk", get(sa.streams, (size_t)M * 4 * H), get(sb.streams, (size_t)M * 4 * H), r, M);
            // 3. the gathered sum itself = rn16(the slot sum) on the host (FP32 adds in rank order from 0.f)
            {
                auto g = get(gsum, (size_t)M * H);
                size_t n = 0;
                for (size_t k = 0; k < (size_t)M * H; ++k) {
                    float sum = 0.f; for (int p = 0; p < g6nr::NR; ++p) sum += fl(part[p][k]);
                    unsigned short want = bf(sum), have; memcpy(&have, g.data() + 2 * k, 2);
                    if (sum != sum) { if ((have & 0x7f80) != 0x7f80 || !(have & 0x7f)) ++n; } else if (have != want) ++n;
                }
                printf("gather sum vs host rank=%d M=%d identical=%s%s\n", r, M, n ? "no" : "yes", n ? (" (" + std::to_string(n) + " elements)").c_str() : "");
                if (n) ++g_bad;
            }
            CK(cudaFreeHost(region)); CK(cudaFree(gsum)); CK(cudaFree(dstep));
            for (State* s : {&s0, &sa, &sb}) { CK(cudaFree(s->streams)); CK(cudaFree(s->mix)); CK(cudaFree(s->pre)); CK(cudaFree(s->post)); CK(cudaFree(s->comb)); CK(cudaFree(s->part)); CK(cudaFree(s->done)); CK(cudaFree(s->y)); }
        }
    printf("ds41 gather: %s\n", g_bad ? "FAIL" : "ALL IDENTICAL");
    return g_bad ? 1 : 0;
}
