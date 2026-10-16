// Ready-data mHC bench: C(M) of flash_mhc::k_mhc_out_inv_prepare (the fused prefill consume kernel) with the
// peer tail already satisfied. Same kernel, launch shape (<<<M, 256>>>) and outputs as pf_consume_flash; the
// fabric block is replaced by one that reads the own partial from device memory and the peer partial from
// pinned mapped host memory (like the fabric slot), after one volatile poll of a pre-set host flag.
//   mhc_ready_bench [REPS] M...      (default M = 128 256 512 1024 2048 4096 8192)
// Rows per M: ready (peer in pinned host), devpeer (peer on device), nowait (no flag poll), hostldg (peer in pinned
// host read with __ldg instead of __ldcv: timing only, not a valid production load), h2dcopy (copy engine, peer
// partial pinned host -> device, alone). Byte model per row:
// streams 64 KiB read + 64 KiB write, partials 2 x 8 KiB, out f16 8 KiB, normalized f32 16 KiB = 168 KiB.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <random>
#include <vector>
#include "mhc/mhc.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

template <bool CV = true> struct ReadyBlockT {
    const volatile unsigned* flag;  // pinned host, pre-set to 1 (nullptr = no poll)
    const unsigned short* own;      // device BF16 partial [M][4096]
    const unsigned short* peer;     // pinned host or device BF16 partial [M][4096]
    __device__ __forceinline__ unsigned long long init() { return 0; }
    __device__ __forceinline__ void wait(unsigned long long) const {
        if (!flag) return;
        __shared__ unsigned ok;
        if (!threadIdx.x) { unsigned v; do v = *flag; while (!v); ok = v; }
        __syncthreads();
        (void)ok;
    }
    __device__ __forceinline__ float operator()(size_t k) const {
        const unsigned short a = __ldg(own + k), b = CV ? __ldcv(peer + k) : __ldg(peer + k);
        return __uint_as_float((unsigned)a << 16) + __uint_as_float((unsigned)b << 16);
    }
};
using ReadyBlock = ReadyBlockT<true>;

// Byte floor: the same DRAM bytes as one consume row (streams read + write, 2 partials, out f16 + normalized f32)
// with no fn work, one row per CTA, default occupancy.
__global__ void k_floor(float* streams, const unsigned short* own, const unsigned short* peer, __half* out, float* xn)
{
    const int m = blockIdx.x;
    float* s = streams + (size_t)m * 4 * 4096;
    #pragma unroll 4
    for (int idx = 0; idx < 16; ++idx) {
        const int h = threadIdx.x + idx * 256;
        const float y = __uint_as_float((unsigned)__ldg(own + (size_t)m * 4096 + h) << 16) +
                        __uint_as_float((unsigned)__ldg(peer + (size_t)m * 4096 + h) << 16);
        float a = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) { const float v = __ldg(s + j * 4096 + h) + y; s[j * 4096 + h] = v; a += v; }
        out[(size_t)m * 4096 + h] = __float2half_rn(a);
        xn[(size_t)m * 4096 + h] = a;
    }
}

static std::mt19937 rng(1234);
template <class T> T* dev_fill(size_t n, float lo, float hi, bool bf16 = false)
{
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<unsigned char> h(n * sizeof(T));
    for (size_t i = 0; i < n; ++i) {
        const float f = u(rng);
        if (bf16) { unsigned w; memcpy(&w, &f, 4); unsigned short s = (unsigned short)(w >> 16); memcpy(&h[i * sizeof(T)], &s, 2); }
        else memcpy(&h[i * sizeof(T)], &f, sizeof(T));
    }
    T* d; CK(cudaMalloc(&d, h.size())); CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 10;
    { cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); printf("SMs %d L2 %d KiB smem/SM %zu KiB\n", p.multiProcessorCount, p.l2CacheSize / 1024, p.sharedMemPerMultiprocessor / 1024); }
    std::vector<int> Ms;
    for (int i = 2; i < argc; ++i) Ms.push_back(atoi(argv[i]));
    if (Ms.empty()) Ms = {128, 256, 512, 1024, 2048, 4096, 8192};
    int MX = 0;
    for (int m : Ms) MX = m > MX ? m : MX;
    const int H = 4096;
    const float eps = 1e-6f;
    flash_mhc::Weights w;
    w.fn = reinterpret_cast<const __nv_bfloat16*>(dev_fill<unsigned short>((size_t)24 * 4 * H, -0.02f, 0.02f, true));
    w.base = dev_fill<float>(24, -0.5f, 0.5f);
    w.scale = dev_fill<float>(3, 0.5f, 1.5f);
    w.norm = reinterpret_cast<const __nv_bfloat16*>(dev_fill<unsigned short>(H, 0.5f, 1.5f, true));
    const size_t ns = (size_t)MX * 4 * H;
    float* streams0 = dev_fill<float>(ns, -1.f, 1.f);
    float* post0 = dev_fill<float>((size_t)MX * 4, 0.5f, 1.5f);
    float* comb0 = dev_fill<float>((size_t)MX * 16, 0.2f, 0.3f);
    flash_mhc::Buffers b{};
    CK(cudaMalloc(&b.streams, ns * 4)); CK(cudaMalloc(&b.inv, (size_t)MX * 4)); CK(cudaMalloc(&b.mix, (size_t)MX * 24 * 4));
    CK(cudaMalloc(&b.pre, (size_t)MX * 16)); CK(cudaMalloc(&b.post, (size_t)MX * 16)); CK(cudaMalloc(&b.comb, (size_t)MX * 64));
    unsigned short* own = dev_fill<unsigned short>((size_t)MX * H, -1.f, 1.f, true);
    unsigned short* peer_dev = dev_fill<unsigned short>((size_t)MX * H, -1.f, 1.f, true);
    unsigned short *peer_host, *peer_host_d;
    CK(cudaHostAlloc(&peer_host, (size_t)MX * H * 2, cudaHostAllocMapped));
    CK(cudaMemcpy(peer_host, peer_dev, (size_t)MX * H * 2, cudaMemcpyDeviceToHost));
    CK(cudaHostGetDevicePointer(&peer_host_d, peer_host, 0));
    unsigned *flag, *flag_d;
    CK(cudaHostAlloc(&flag, 4, cudaHostAllocMapped)); *flag = 1;
    CK(cudaHostGetDevicePointer(&flag_d, flag, 0));
    __half* out; float* xn;
    CK(cudaMalloc(&out, (size_t)MX * H * 2)); CK(cudaMalloc(&xn, (size_t)MX * H * 4));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto reset = [&](int M) {
        CK(cudaMemcpyAsync(b.streams, streams0, (size_t)M * 4 * H * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(b.post, post0, (size_t)M * 16, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(b.comb, comb0, (size_t)M * 64, cudaMemcpyDeviceToDevice));
    };
    uint4* fnp;
    CK(cudaMalloc(&fnp, (size_t)3 * 4 * 16 * flash_mhc::THREADS * 16));
    flash_mhc::k_mhc_pack_fn<<<3 * 4 * 16, flash_mhc::THREADS>>>(w.fn, fnp);
    CK(cudaDeviceSynchronize());
    // kernel kind: 0 = k_mhc_out_inv_prepare, R > 0 = k_mhc_out_inv_prepare_pk<.., R>
    auto launch = [&](int kind, int M, const auto& blk) {
        using B = std::decay_t<decltype(blk)>;
        if (kind == 0) flash_mhc::k_mhc_out_inv_prepare<<<M, flash_mhc::THREADS>>>(b, blk, w, H, eps, out, xn);
        else if (kind == 1) flash_mhc::k_mhc_out_inv_prepare_pk<B, 1><<<M, flash_mhc::THREADS>>>(b, blk, w, fnp, M, eps, out, xn);
        else if (kind == 2) flash_mhc::k_mhc_out_inv_prepare_pk<B, 2><<<(M + 1) / 2, flash_mhc::THREADS>>>(b, blk, w, fnp, M, eps, out, xn);
        else if (kind == 4) flash_mhc::k_mhc_out_inv_prepare_pk<B, 4><<<(M + 3) / 4, flash_mhc::THREADS>>>(b, blk, w, fnp, M, eps, out, xn);
        // re-read kernels: kind 1RM = k_mhc_out_inv_prepare_rr<.., R, MINB M>
        else if (kind == 112) flash_mhc::k_mhc_out_inv_prepare_rr<B, 1, 2><<<M, flash_mhc::THREADS>>>(b, blk, w, fnp, M, eps, out, xn);
        else if (kind == 113) flash_mhc::k_mhc_out_inv_prepare_rr<B, 1, 3><<<M, flash_mhc::THREADS>>>(b, blk, w, fnp, M, eps, out, xn);
        else if (kind == 114) flash_mhc::k_mhc_out_inv_prepare_rr<B, 1, 4><<<M, flash_mhc::THREADS>>>(b, blk, w, fnp, M, eps, out, xn);
        else if (kind == 121) flash_mhc::k_mhc_out_inv_prepare_rr<B, 2, 1><<<(M + 1) / 2, flash_mhc::THREADS>>>(b, blk, w, fnp, M, eps, out, xn);
        else if (kind == 122) flash_mhc::k_mhc_out_inv_prepare_rr<B, 2, 2><<<(M + 1) / 2, flash_mhc::THREADS>>>(b, blk, w, fnp, M, eps, out, xn);
        else if (kind == 123) flash_mhc::k_mhc_out_inv_prepare_rr<B, 2, 3><<<(M + 1) / 2, flash_mhc::THREADS>>>(b, blk, w, fnp, M, eps, out, xn);
        else if (kind == 99) k_floor<<<M, 256>>>(b.streams, blk.own, blk.peer, out, xn);
    };
    // all outputs of one launch from reset buffers, as bytes
    auto snap = [&](int M) {
        std::vector<unsigned char> v;
        auto add = [&](const void* d, size_t n) { size_t o = v.size(); v.resize(o + n); CK(cudaMemcpy(v.data() + o, d, n, cudaMemcpyDeviceToHost)); };
        add(b.streams, (size_t)M * 4 * H * 4); add(b.inv, (size_t)M * 4); add(b.mix, (size_t)M * 24 * 4);
        add(b.pre, (size_t)M * 16); add(b.post, (size_t)M * 16); add(b.comb, (size_t)M * 64);
        add(out, (size_t)M * H * 2); add(xn, (size_t)M * H * 4);
        return v;
    };
    auto run = [&](int M, const auto& blk, int kind = 0) {   // median of reps, each from reset buffers
        std::vector<float> t;
        for (int r = 0; r <= reps; ++r) {
            reset(M);
            cudaEventRecord(e0);
            launch(kind, M, blk);
            cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1));
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (r) t.push_back(ms);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };
    printf("%6s %10s %10s %10s %10s %10s %9s %9s\n", "M", "ready_ms", "devpeer_ms", "nowait_ms", "hostldg_ms", "h2dcopy_ms", "GB/s", "us/row");
    for (int M : Ms) {
        const float tr = run(M, ReadyBlock{flag_d, own, peer_host_d});
        const float td = run(M, ReadyBlock{flag_d, own, peer_dev});
        const float tn = run(M, ReadyBlock{nullptr, own, peer_host_d});
        const float tg = run(M, ReadyBlockT<false>{flag_d, own, peer_host_d});
        std::vector<float> tc;
        for (int r = 0; r <= reps; ++r) {
            cudaEventRecord(e0);
            CK(cudaMemcpyAsync(peer_dev, peer_host, (size_t)M * H * 2, cudaMemcpyHostToDevice));
            cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1));
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (r) tc.push_back(ms);
        }
        std::sort(tc.begin(), tc.end());
        const double bytes = (double)M * 168 * 1024;
        printf("%6d %10.3f %10.3f %10.3f %10.3f %10.3f %9.1f %9.3f\n", M, tr, td, tn, tg, tc[tc.size() / 2], bytes / (tr * 1e-3) / 1e9, tr * 1e3 / M);
        // packed-fn kernels: bytes vs the old kernel (ready block), then time (ready and device peer)
        reset(M); CK(cudaMemset(out, 0, (size_t)M * H * 2)); CK(cudaMemset(xn, 0, (size_t)M * H * 4));
        launch(0, M, ReadyBlock{flag_d, own, peer_host_d}); CK(cudaDeviceSynchronize());
        const std::vector<unsigned char> ref = snap(M);
        if (getenv("MB_RR")) {
            const float fl = run(M, ReadyBlock{flag_d, own, peer_dev}, 99);
            printf("        floor (same bytes, no fn): %8.3f ms = %.1f GB/s\n", fl, (double)M * 168 * 1024 / (fl * 1e-3) / 1e9);
        }
        const std::vector<int> kinds = getenv("MB_RR") ? std::vector<int>{2, 112, 113, 114, 121, 122, 123} : std::vector<int>{1, 2, 4};
        for (int kind : kinds) {
            reset(M); CK(cudaMemset(out, 0, (size_t)M * H * 2)); CK(cudaMemset(xn, 0, (size_t)M * H * 4));
            launch(kind, M, ReadyBlock{flag_d, own, peer_host_d}); CK(cudaDeviceSynchronize());
            const std::vector<unsigned char> v = snap(M);
            size_t diff = 0;
            for (size_t i = 0; i < v.size(); ++i) diff += v[i] != ref[i];
            const float pr = run(M, ReadyBlock{flag_d, own, peer_host_d}, kind), pd = run(M, ReadyBlock{flag_d, own, peer_dev}, kind);
            printf("        kind %3d: ready %8.3f ms devpeer %8.3f ms  %s (%zu bytes differ)\n", kind, pr, pd, diff ? "NOT EQUAL" : "BYTE-EQUAL", diff);
        }
        if (getenv("MB_RR")) {   // option X: rr writes the fragment tables instead of FP32 rows
            const size_t fw = (size_t)(M + 7) / 8 * 8 * H / 2;
            unsigned *f16a, *bf16a, *f16b, *bf16b;
            CK(cudaMalloc(&f16a, fw * 4)); CK(cudaMalloc(&bf16a, fw * 4)); CK(cudaMalloc(&f16b, fw * 4)); CK(cudaMalloc(&bf16b, fw * 4));
            CK(cudaMemset(f16a, 0xff, fw * 4)); CK(cudaMemset(bf16a, 0xff, fw * 4)); CK(cudaMemset(f16b, 0x7f, fw * 4)); CK(cudaMemset(bf16b, 0x7f, fw * 4));
            reset(M); launch(2, M, ReadyBlock{flag_d, own, peer_dev}); CK(cudaDeviceSynchronize());
            flash_mhc::fragments(xn, M, H, f16a, bf16a, 0); CK(cudaDeviceSynchronize());
            const std::vector<unsigned char> ref2 = snap(M);
            using B = ReadyBlock;
            auto lf = [&]() { flash_mhc::k_mhc_out_inv_prepare_rr<B, 2, 1><<<(M + 1) / 2, flash_mhc::THREADS>>>(b, B{flag_d, own, peer_dev}, w, fnp, M, eps, out, nullptr, f16b, bf16b); };
            reset(M); CK(cudaMemset(xn, 0, (size_t)M * H * 4)); lf(); CK(cudaDeviceSynchronize());
            std::vector<unsigned> ha(fw), hb(fw);
            size_t d16 = 0, db = 0;
            CK(cudaMemcpy(ha.data(), f16a, fw * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb.data(), f16b, fw * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < fw; ++i) d16 += ha[i] != hb[i];
            CK(cudaMemcpy(ha.data(), bf16a, fw * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb.data(), bf16b, fw * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < fw; ++i) db += ha[i] != hb[i];
            const std::vector<unsigned char> v2 = snap(M);   // streams..out equal; xn differs by design (not written)
            size_t dother = 0;
            const size_t xn_off = v2.size() - (size_t)M * H * 4;
            for (size_t i = 0; i < xn_off; ++i) dother += v2[i] != ref2[i];
            std::vector<float> t;
            for (int r = 0; r <= reps; ++r) {
                reset(M); cudaEventRecord(e0); lf(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
                float ms; cudaEventElapsedTime(&ms, e0, e1); if (r) t.push_back(ms);
            }
            std::sort(t.begin(), t.end());
            std::vector<float> tf;
            for (int r = 0; r <= reps; ++r) {
                cudaEventRecord(e0); flash_mhc::fragments(xn, M, H, f16a, bf16a, 0); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
                float ms; cudaEventElapsedTime(&ms, e0, e1); if (r) tf.push_back(ms);
            }
            std::sort(tf.begin(), tf.end());
            printf("        rr+frags: devpeer %8.3f ms  frag words differ f16 %zu bf16 %zu, other outputs differ %zu bytes  %s | fragments() alone %.3f ms\n",
                   t[t.size() / 2], d16, db, dother, (d16 || db || dother) ? "NOT EQUAL" : "BYTE-EQUAL", tf[tf.size() / 2]);
            cudaFree(f16a); cudaFree(bf16a); cudaFree(f16b); cudaFree(bf16b);
        }
    }
    return 0;
}
