// F7 k16 bit-equality gate of the staged slot reads (GR8_DS_DEC_STAGE=1; runtime/ds41_stage.cuh + the has_stage_v branches of ds_mhc.cuh k_step / k_consume / k_finish).
//   (no argument)  HOST mode. Runs on any machine, makes NO CUDA call. It executes the kernels' staging code on the CPU thread by thread: the same tile maps
//                  (ds_mhc::stage_kstep / stage_consume / stage_finish, which the kernels call), each 16-byte load (alignment, bounds, each needed element
//                  loaded and no other row), the rank-order FP32 sum from smem, and the finish loop's per-thread column order; every (row, column) of every shape must
//                  equal the Ds41Block reference (4 x 2-byte loads, FP32 sum in rank order from 0.f) bit for bit.
//   gpu            GPU mode (a spark; refuses anything but sm_121). For each rank r and row count M the same fabric slots (pinned host memory, the real layout)
//                  are consumed twice from the same state: ref = RefBlock (a copy of Ds41Block's slot reads), staged = Ds41Staged<RefBlock>. Every output must match byte
//                  for byte: k_step FUSE (trunk, y, pre, post, comb, mix), k_consume (trunk), k_finish (trunk, out, xpre; also grid 1 with row0 = M-1).
//                  Exit 0 = every line identical=yes.
// Build (compile-only on the dev box): nvcc -O2 -std=c++17 -DG6_MODEL_DS41=1 -DDS_MHC_TRUNK_BF16 -gencode=arch=compute_121a,code=sm_121a -Xcompiler -Wall -I engine -o ds41_stage_test engine/runtime/tests/ds41_stage_test.cu
// RefBlock copies Ds41Block::init / operator() (ds41_step.cuh); keep the two in step.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "../../kernels/glue.cuh"
#include "../../mhc/ds_mhc.cuh"
#include "../ds41_stage.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

struct RefBlock {
    g6::FabricDev f;
    const u64* stepp;
    int c;
    size_t wait_payload = 0;
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
using SB = Ds41Staged<RefBlock>;

using TT = ds_mhc::trunk_t;
static const int H = HID, NC = 3, C = 1;   // collective 1 of 3
static int g_bad = 0;

static unsigned short bf(float v) { unsigned u; memcpy(&u, &v, 4); return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float fl(unsigned short b) { unsigned u = (unsigned)b << 16; float v; memcpy(&v, &u, 4); return v; }
static unsigned bits(float v) { unsigned u; memcpy(&u, &v, 4); return u; }

// ---------------------------------------------------------------------------------------------------------------------------------- HOST mode
// random BF16 slot partials incl. +-0, subnormals, large magnitudes, cancelling rows; part[p][k], k = row * H + h
static std::vector<std::vector<unsigned short>> make_parts(int M, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<std::vector<unsigned short>> part(g6nr::NR, std::vector<unsigned short>((size_t)M * H));
    for (size_t k = 0; k < (size_t)M * H; ++k) {
        const int kind = rng() % 16;
        for (int p = 0; p < g6nr::NR; ++p) {
            float v = u(rng) * (kind == 0 ? 3e4f : kind == 1 ? 1e-39f : 4.f);
            if (kind == 2) v = p == 0 ? 0.f : -0.f;
            part[p][k] = bf(v);
        }
        if (kind == 3) { part[1][k] = part[0][k] ^ 0x8000; part[3][k] = part[2][k] ^ 0x8000; }
    }
    return part;
}
static float ref_sum(const std::vector<std::vector<unsigned short>>& part, size_t k) {   // Ds41Block::operator()
    float sum = 0.f;
    for (int p = 0; p < g6nr::NR; ++p) sum += fl(part[p][k]);
    return sum;
}
// one 16-byte load of Ds41Staged::ld16: slot p, elem a multiple of 8; checks alignment and bounds (a load past the slot would read the next slot / tail)
static bool ld16_host(const std::vector<std::vector<unsigned short>>& part, int M, int p, size_t elem, unsigned short (&out)[8]) {
    if (elem % 8 != 0 || elem + 8 > (size_t)M * H) return false;
    memcpy(out, part[p].data() + elem, 16);
    return true;
}
static void host_bad(const char* what, int M, const std::string& why) { printf("%s M=%d FAIL: %s\n", what, M, why.c_str()); ++g_bad; }

static void host_kstep(int M, const std::vector<std::vector<unsigned short>>& part) {
    using namespace ds_mhc;
    constexpr int NS = g6nr::NR;
    static_assert(NS * RT * (PW / 8) <= THREADS, "one 16-byte load per thread");
    size_t checked = 0; int bad = 0;
    for (int tile = 0; tile < (M + RT - 1) / RT; ++tile)
        for (int c = 0; c < H / PW; ++c) {
            const int m0 = tile * RT, nr = std::min(RT, M - m0);
            unsigned short stg[NS][RT][PW]; unsigned char wr[NS][RT][PW]; memset(wr, 0, sizeof wr); memset(stg, 0xAB, sizeof stg);
            for (int t = 0; t < NS * RT * (PW / 8); ++t) {   // the loading threads
                const StageMap q = stage_kstep(t);
                if (q.rr >= nr) continue;
                unsigned short v[8];
                if (!ld16_host(part, M, q.p, (size_t)(m0 + q.rr) * H + (size_t)c * PW + q.seg * 8, v)) { ++bad; continue; }
                for (int e = 0; e < 8; ++e) { stg[q.p][q.rr][q.seg * 8 + e] = v[e]; ++wr[q.p][q.rr][q.seg * 8 + e]; }
            }
            for (int t = 0; t < THREADS; ++t) {   // the summing threads
                const int r = t / PW, hh = t % PW;
                if (r >= nr) continue;
                float ys = 0.f;
                for (int p = 0; p < NS; ++p) { if (wr[p][r][hh] != 1) ++bad; ys += fl(stg[p][r][hh]); }
                if (bits(ys) != bits(ref_sum(part, (size_t)(m0 + r) * H + c * PW + hh))) ++bad;
                ++checked;
            }
            for (int p = 0; p < NS; ++p) for (int rr = nr; rr < RT; ++rr) for (int e = 0; e < PW; ++e) if (wr[p][rr][e]) ++bad;   // no load for rows >= nr
        }
    printf("host k_step    M=%3d (row, column) cells=%zu bad=%d\n", M, checked, bad);
    if (bad) host_bad("k_step", M, "mismatch");
}
static void host_consume(int M, const std::vector<std::vector<unsigned short>>& part) {
    using namespace ds_mhc;
    constexpr int NS = g6nr::NR;
    static_assert(NS * (THREADS / 8) <= THREADS, "one 16-byte load per thread");
    size_t checked = 0; int bad = 0;
    for (int m = 0; m < M; ++m)
        for (int bx = 0; bx < (H + THREADS - 1) / THREADS; ++bx) {
            unsigned short stg[NS][THREADS]; unsigned char wr[NS][THREADS]; memset(wr, 0, sizeof wr); memset(stg, 0xAB, sizeof stg);
            for (int t = 0; t < NS * (THREADS / 8); ++t) {
                const StageMap q = stage_consume(t);
                const int col = bx * THREADS + q.seg * 8;
                if (col >= H) continue;
                unsigned short v[8];
                if (!ld16_host(part, M, q.p, (size_t)m * H + col, v)) { ++bad; continue; }
                for (int e = 0; e < 8; ++e) { stg[q.p][q.seg * 8 + e] = v[e]; ++wr[q.p][q.seg * 8 + e]; }
            }
            for (int t = 0; t < THREADS; ++t) {
                const int h = bx * THREADS + t;
                if (h >= H) continue;
                float ys = 0.f;
                for (int p = 0; p < NS; ++p) { if (wr[p][t] != 1) ++bad; ys += fl(stg[p][t]); }
                if (bits(ys) != bits(ref_sum(part, (size_t)m * H + h))) ++bad;
                ++checked;
            }
        }
    printf("host k_consume M=%3d (row, column) cells=%zu bad=%d\n", M, checked, bad);
    if (bad) host_bad("k_consume", M, "mismatch");
}
static void host_finish(int M, const std::vector<std::vector<unsigned short>>& part) {
    using namespace ds_mhc;
    constexpr int NS = g6nr::NR;
    if (H % FCH) { host_bad("k_finish", M, "H % FCH != 0"); return; }
    size_t checked = 0; int bad = 0;
    for (int mi = 0; mi < M; ++mi) {   // one CTA per row
        std::vector<std::vector<int>> order(THREADS);   // each thread's column sequence in the staged loop
        for (int h0 = 0; h0 < H; h0 += FCH) {
            unsigned short stg[NS][FCH]; unsigned char wr[NS][FCH]; memset(wr, 0, sizeof wr); memset(stg, 0xAB, sizeof stg);
            for (int t = 0; t < THREADS; ++t)
                for (int u = t; u < NS * (FCH / 8); u += THREADS) {
                    const StageMap q = stage_finish(u);
                    unsigned short v[8];
                    if (!ld16_host(part, M, q.p, (size_t)mi * H + h0 + q.seg * 8, v)) { ++bad; continue; }
                    for (int e = 0; e < 8; ++e) { stg[q.p][q.seg * 8 + e] = v[e]; ++wr[q.p][q.seg * 8 + e]; }
                }
            for (int t = 0; t < THREADS; ++t)
                for (int h = h0 + t; h < h0 + FCH; h += THREADS) {
                    float ys = 0.f;
                    for (int p = 0; p < NS; ++p) { if (wr[p][h - h0] != 1) ++bad; ys += fl(stg[p][h - h0]); }
                    if (bits(ys) != bits(ref_sum(part, (size_t)mi * H + h))) ++bad;
                    order[t].push_back(h);
                    ++checked;
                }
        }
        for (int t = 0; t < THREADS; ++t) {   // the plain loop is h = t; h < H; h += THREADS: same columns in the same order => sq adds in the same order
            size_t i = 0;
            for (int h = t; h < H; h += THREADS, ++i) if (i >= order[t].size() || order[t][i] != h) { ++bad; break; }
            if (i != order[t].size()) ++bad;
        }
    }
    printf("host k_finish  M=%3d (row, column) cells=%zu bad=%d\n", M, checked, bad);
    if (bad) host_bad("k_finish", M, "mismatch");
}
static int host_main() {
    std::mt19937 rng(20261026);
    if (H % ds_mhc::PW || H % 8 || H % ds_mhc::FCH) { printf("H=%d is not a multiple of PW / 8 / FCH\n", H); return 1; }
    for (int M : {1, 2, 5, 6, 7, 12, 33, 47, 48}) {   // M % RT != 0 (partial tiles) and the decode shapes c1 (6) / c8 (48)
        const auto part = make_parts(M, rng);
        host_kstep(M, part); host_consume(M, part); host_finish(M, part);
    }
    printf("ds41 stage host: %s\n", g_bad ? "FAIL" : "ALL IDENTICAL");
    return g_bad ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------------------------------- GPU mode
struct State { TT* streams; float *mix, *pre, *post, *comb, *part; unsigned* done; __nv_bfloat16 *y, *out, *xpre; };
static State alloc_state(int M) {
    State s;
    CK(cudaMalloc(&s.streams, (size_t)M * 4 * H * sizeof(TT))); CK(cudaMalloc(&s.mix, (size_t)M * ds_mhc::MIX * 4));
    CK(cudaMalloc(&s.pre, (size_t)M * 16)); CK(cudaMalloc(&s.post, (size_t)M * 16)); CK(cudaMalloc(&s.comb, (size_t)M * 64));
    CK(cudaMalloc(&s.part, (size_t)M * ds_mhc::NV * (H / ds_mhc::PW) * 4));
    const int dw = ds_mhc::done_words(M); CK(cudaMalloc(&s.done, dw * 4)); CK(cudaMemset(s.done, 0, dw * 4));
    CK(cudaMalloc(&s.y, (size_t)M * H * 2)); CK(cudaMalloc(&s.out, (size_t)M * H * 2)); CK(cudaMalloc(&s.xpre, (size_t)M * H * 2));
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
    CK(cudaMemset(d.y, 0, (size_t)M * H * 2)); CK(cudaMemset(d.out, 0, (size_t)M * H * 2)); CK(cudaMemset(d.xpre, 0, (size_t)M * H * 2));
}
static void cmp(const char* what, const std::vector<char>& a, const std::vector<char>& b, int r, int M) {
    size_t n = 0; for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
    printf("%s rank=%d M=%d bytes=%zu identical=%s%s\n", what, r, M, a.size(), n ? "no" : "yes", n ? (" (" + std::to_string(n) + " bytes differ)").c_str() : "");
    if (n) ++g_bad;
}
static std::vector<char> mixall(const State& s, int M) {
    auto a = get(s.pre, M * 4); auto b = get(s.post, M * 4); auto c = get(s.comb, M * 16); auto d = get(s.mix, (size_t)M * ds_mhc::MIX);
    a.insert(a.end(), b.begin(), b.end()); a.insert(a.end(), c.begin(), c.end()); a.insert(a.end(), d.begin(), d.end());
    return a;
}

static int gpu_main() {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    if (pr.major != 12 || pr.minor != 1) { printf("refusing to run: sm_%d%d is not sm_121\n", pr.major, pr.minor); return 3; }
    std::mt19937 rng(20261026);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    const int Ms[] = {1, 5, 6, 7, 33, 47, 48};
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
            g6::FabricConfig cfg; cfg.slot_bytes = (size_t)M * H * 2;
            g6::FabricDev f{}; f.nr = g6nr::NR; f.rank = r;
            if (!g6::fab_slot_layout(cfg, NC, &f)) { printf("layout refused\n"); return 1; }
            const size_t half = g6::fab_slot_base(f, NC), send_b = 2 * half, recv_b = 2 * half * g6nr::NR;
            char* region; CK(cudaHostAlloc((void**)&region, send_b + recv_b, cudaHostAllocMapped | cudaHostAllocPortable));
            f.send = region; f.recv = region + send_b;
            const u64 step = 7;   // parity 1
            u64* dstep; CK(cudaMalloc(&dstep, 8)); CK(cudaMemcpy(dstep, &step, 8, cudaMemcpyHostToDevice));
            const auto part = make_parts(M, rng);
            for (int p = 0; p < g6nr::NR; ++p) {
                char* dst = p == r ? f.send + g6::fab_send_off(f, 1, C) : f.recv + g6::fab_recv_off(f, 1, C, p);
                memcpy(dst, part[p].data(), (size_t)M * H * 2);
            }
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
            const RefBlock ref{f, dstep, C};
            const SB stg{ref};
            // 1. consume_prepare (k_step FUSE)
            copy_state(sa, s0, M); copy_state(sb, s0, M);
            ds_mhc::consume_prepare<float>(buf(sa), ref, W, M, H, 1e-6f, sa.y, 0);
            ds_mhc::consume_prepare<float>(buf(sb), stg, W, M, H, 1e-6f, sb.y, 0);
            CK(cudaDeviceSynchronize());
            cmp("consume_prepare trunk", get(sa.streams, (size_t)M * 4 * H), get(sb.streams, (size_t)M * 4 * H), r, M);
            cmp("consume_prepare y", get(sa.y, (size_t)M * H), get(sb.y, (size_t)M * H), r, M);
            cmp("consume_prepare pre/post/comb/mix", mixall(sa, M), mixall(sb, M), r, M);
            // 2. consume (plain, k_consume)
            copy_state(sa, s0, M); copy_state(sb, s0, M);
            ds_mhc::consume(buf(sa), ref, M, H, 0);
            ds_mhc::consume(buf(sb), stg, M, H, 0);
            CK(cudaDeviceSynchronize());
            cmp("consume trunk", get(sa.streams, (size_t)M * 4 * H), get(sb.streams, (size_t)M * 4 * H), r, M);
            // 3. finish (k_finish): store = true, out + xpre; then one row (grid 1, row0 = M - 1)
            copy_state(sa, s0, M); copy_state(sb, s0, M);
            ds_mhc::finish(buf(sa), ref, dnorm, M, H, 1e-6f, true, sa.out, 0, nullptr, nullptr, sa.xpre);
            ds_mhc::finish(buf(sb), stg, dnorm, M, H, 1e-6f, true, sb.out, 0, nullptr, nullptr, sb.xpre);
            CK(cudaDeviceSynchronize());
            cmp("finish trunk", get(sa.streams, (size_t)M * 4 * H), get(sb.streams, (size_t)M * 4 * H), r, M);
            cmp("finish out", get(sa.out, (size_t)M * H), get(sb.out, (size_t)M * H), r, M);
            cmp("finish xpre", get(sa.xpre, (size_t)M * H), get(sb.xpre, (size_t)M * H), r, M);
            copy_state(sa, s0, M); copy_state(sb, s0, M);
            ds_mhc::finish(buf(sa), ref, dnorm, 1, H, 1e-6f, true, sa.out, 0, nullptr, nullptr, sa.xpre, M - 1);
            ds_mhc::finish(buf(sb), stg, dnorm, 1, H, 1e-6f, true, sb.out, 0, nullptr, nullptr, sb.xpre, M - 1);
            CK(cudaDeviceSynchronize());
            cmp("finish row0 trunk", get(sa.streams, (size_t)M * 4 * H), get(sb.streams, (size_t)M * 4 * H), r, M);
            cmp("finish row0 out", get(sa.out, (size_t)H), get(sb.out, (size_t)H), r, M);
            CK(cudaFreeHost(region)); CK(cudaFree(dstep));
            for (State* s : {&s0, &sa, &sb}) { CK(cudaFree(s->streams)); CK(cudaFree(s->mix)); CK(cudaFree(s->pre)); CK(cudaFree(s->post)); CK(cudaFree(s->comb)); CK(cudaFree(s->part)); CK(cudaFree(s->done)); CK(cudaFree(s->y)); CK(cudaFree(s->out)); CK(cudaFree(s->xpre)); }
        }
    printf("ds41 stage gpu: %s\n", g_bad ? "FAIL" : "ALL IDENTICAL");
    return g_bad ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && !strcmp(argv[1], "gpu")) return gpu_main();
    if (argc > 1) { printf("usage: ds41_stage_test [gpu]\n"); return 2; }
    return host_main();
}
