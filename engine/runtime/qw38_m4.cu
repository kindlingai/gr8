// g6m4_qw38: one rank of Qwen3.8-Flash-Next on the real RDMA fabric (TP = 1, 2, 3, 4).
// Supports both standalone cluster smoke runs (--tokens ... --gen ...) and S2 multi-session serving (--serve DIR).
// Uses device-driven collectives (ink2::ar_exchange, dual-rail striped when --ips2 is given) so multi-box TP=2..4
// has minimal collective latency and stays byte-identical to m4sim_qw38.
#include "qw38_step.cuh"
#include "vision_id.h"
#include "../comms/fabric.h"
#include "../kernels/ink2_fabric.cuh"
#include "s2/qw38_native.cuh"
#include <chrono>

#ifndef G6_NR
#define G6_NR 2
#endif

namespace {
using qw38rt::die;

// Graph-safe device-driven AR ring/wait: reads step counter from device memory so captured CUDA graphs
// work across all steps of the same parity (par = 0 or 1)!
struct Qw38ArWait {
    g6::FabricDev f;
    size_t bytes;
    const unsigned long long* d_step;
    unsigned code;
    int par, c;
};
__global__ void k_qw38_ar_ring_wait(Qw38ArWait a) {
    const int p = threadIdx.x;
    const unsigned long long seq = g6::coll_seq(*a.d_step, a.c, a.f.ncoll);
    if (p == 0) {
        __threadfence_system();
        ink2::ar_st_release_sys(a.f.doorbell + a.c, seq);
    }
    if (p < a.f.nr && p != a.f.rank) {
        const unsigned long long* tail = reinterpret_cast<const unsigned long long*>(
            a.f.recv + g6::fab_recv_off(a.f, a.par, a.c, p) + a.bytes);
        const long long t0 = clock64();
        unsigned it = 0;
        while (ink2::ar_ld_acquire_sys(tail) != seq) {
            if ((++it & 1023u) == 0 && (*reinterpret_cast<volatile unsigned*>(a.f.error) != 0 || clock64() - t0 > INK2_SPIN_TIMEOUT_CYC)) {
                atomicCAS(a.f.error, 0u, a.code | static_cast<unsigned>(p));
                break;
            }
        }
    }
}

inline void qw38_ar_exchange(const g6::FabricDev& f, const unsigned long long* d_step, int par, int c,
                             char* parts, size_t bytes, cudaStream_t st) {
    const size_t n16 = bytes / 16;
    const unsigned grid = static_cast<unsigned>((n16 + 255) / 256);
    ink2::k_ink2_ar_copy<<<grid, 256, 0, st>>>(
        reinterpret_cast<const uint4*>(parts + static_cast<size_t>(f.rank) * bytes),
        reinterpret_cast<uint4*>(f.send + g6::fab_send_off(f, par, c)), n16);
    Qw38ArWait a{f, bytes, d_step, ink2::INK2_ERR_AR | (static_cast<unsigned>(c) << 8), par, c};
    k_qw38_ar_ring_wait<<<1, 32, 0, st>>>(a);
    for (int p = 0; p < f.nr; ++p) {
        if (p != f.rank) {
            ink2::k_ink2_ar_copy<true><<<grid, 256, 0, st>>>(
                reinterpret_cast<const uint4*>(f.recv + g6::fab_recv_off(f, par, c, p)),
                reinterpret_cast<uint4*>(parts + static_cast<size_t>(p) * bytes), n16);
        }
    }
}

struct Qw38ArInject {
    g6::FabricDev f;
    const unsigned long long* d_step;
    __nv_bfloat16* hyper;
    const __nv_bfloat16* parts;
    const float* inj;
    size_t bytes;
    int M, par, c;
    unsigned code;
};

// Single-kernel fused RDMA AllReduce + HC residual injection for small decode batches (M <= 8).
// Replaces copy-out + ring-wait + (NR-1) copy-in + k_hc_inject (5..6 launches -> 1 launch!),
// reducing partials in exact rank order 0..NR-1 for bit-exact parity with k_hc_inject.
// Grid: 1, Block: 320 (320 * 8 = 2560 hidden dims exact).
template <int NR>
__global__ void __launch_bounds__(320) k_qw38_ar_fused_inject_small(Qw38ArInject a) {
    const int d8 = threadIdx.x; // 0..319 (each uint4 = 8 BF16s)
    constexpr size_t ROW_U4 = geom::HID / 8; // 320
    const size_t part_stride_u4 = static_cast<size_t>(a.M) * ROW_U4;
    const uint4* local_p = reinterpret_cast<const uint4*>(a.parts) + static_cast<size_t>(a.f.rank) * part_stride_u4;
    uint4* send_p = reinterpret_cast<uint4*>(a.f.send + g6::fab_send_off(a.f, a.par, a.c));

    uint4 my_row0{};
    for (int m = 0; m < a.M; ++m) {
        const size_t off = static_cast<size_t>(m) * ROW_U4 + d8;
        const uint4 v = local_p[off];
        send_p[off] = v;
        if (m == 0) my_row0 = v;
    }
    __syncthreads();

    const unsigned long long seq = g6::coll_seq(*a.d_step, a.c, a.f.ncoll);
    if (d8 == 0) {
        __threadfence_system();
        ink2::ar_st_release_sys(a.f.doorbell + a.c, seq);
    }
    if (d8 < NR && d8 != a.f.rank) {
        const unsigned long long* tail = reinterpret_cast<const unsigned long long*>(
            a.f.recv + g6::fab_recv_off(a.f, a.par, a.c, d8) + a.bytes);
        const long long t0 = clock64();
        unsigned it = 0;
        while (ink2::ar_ld_acquire_sys(tail) != seq) {
            if ((++it & 1023u) == 0 && (*reinterpret_cast<volatile unsigned*>(a.f.error) != 0 || clock64() - t0 > INK2_SPIN_TIMEOUT_CYC)) {
                atomicCAS(a.f.error, 0u, a.code | static_cast<unsigned>(d8));
                break;
            }
        }
    }
    __syncthreads();

    const uint4* rptr[NR];
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        rptr[r] = (r == a.f.rank)
                      ? local_p
                      : reinterpret_cast<const uint4*>(a.f.recv + g6::fab_recv_off(a.f, a.par, a.c, r));
    }

    for (int m = 0; m < a.M; ++m) {
        float w[geom::HC];
        #pragma unroll
        for (int s = 0; s < geom::HC; ++s) w[s] = a.inj[m * geom::HC + s];

        float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const size_t roff = static_cast<size_t>(m) * ROW_U4 + d8;
        #pragma unroll
        for (int r = 0; r < NR; ++r) {
            const uint4 pv = (r == a.f.rank) ? ((m == 0) ? my_row0 : local_p[roff]) : __ldcv(rptr[r] + roff);
            const __nv_bfloat162* pp = reinterpret_cast<const __nv_bfloat162*>(&pv);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 fv = __bfloat1622float2(pp[k]);
                y[2 * k] += fv.x;
                y[2 * k + 1] += fv.y;
            }
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) y[k] = __bfloat162float(__float2bfloat16(y[k]));

        __nv_bfloat16* h = a.hyper + static_cast<size_t>(m) * geom::HC_D;
        #pragma unroll
        for (int s = 0; s < geom::HC; ++s) {
            uint4* hp = reinterpret_cast<uint4*>(h + s * geom::HID) + d8;
            const uint4 prev = *hp;
            const __nv_bfloat162* pr = reinterpret_cast<const __nv_bfloat162*>(&prev);
            uint4 out;
            __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&out);
            const float ws = w[s];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 fp = __bfloat1622float2(pr[k]);
                const float a0 = __bfloat162float(__float2bfloat16(y[2 * k] * ws));
                const float a1 = __bfloat162float(__float2bfloat16(y[2 * k + 1] * ws));
                op[k] = __floats2bfloat162_rn(fp.x + a0, fp.y + a1);
            }
            *hp = out;
        }
    }
}

// Multi-CTA HC residual injection reading peers directly from RDMA recv slots via __ldcv (M > 8).
// Eliminates all (NR-1) intermediate HBM copy-in kernels and their HBM write+read round-trips!
// Grid: (M * 5), Block: 64 (5 * 64 * 8 = 2560 hidden dims exact).
template <int NR>
__global__ void k_qw38_hc_inject_from_recv(Qw38ArInject a) {
    const int m = blockIdx.x / 5;
    const int blk = blockIdx.x % 5;
    if (m >= a.M) return;
    const int d8 = blk * 64 + threadIdx.x; // 0..319
    constexpr size_t ROW_U4 = geom::HID / 8;
    const size_t part_stride_u4 = static_cast<size_t>(a.M) * ROW_U4;
    const uint4* local_p = reinterpret_cast<const uint4*>(a.parts) + static_cast<size_t>(a.f.rank) * part_stride_u4;

    float w[geom::HC];
    #pragma unroll
    for (int s = 0; s < geom::HC; ++s) w[s] = a.inj[m * geom::HC + s];

    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const size_t roff = static_cast<size_t>(m) * ROW_U4 + d8;
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        const uint4 pv = (r == a.f.rank)
                             ? local_p[roff]
                             : __ldcv(reinterpret_cast<const uint4*>(a.f.recv + g6::fab_recv_off(a.f, a.par, a.c, r)) + roff);
        const __nv_bfloat162* pp = reinterpret_cast<const __nv_bfloat162*>(&pv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 fv = __bfloat1622float2(pp[k]);
            y[2 * k] += fv.x;
            y[2 * k + 1] += fv.y;
        }
    }
    #pragma unroll
    for (int k = 0; k < 8; ++k) y[k] = __bfloat162float(__float2bfloat16(y[k]));

    __nv_bfloat16* h = a.hyper + static_cast<size_t>(m) * geom::HC_D;
    #pragma unroll
    for (int s = 0; s < geom::HC; ++s) {
        uint4* hp = reinterpret_cast<uint4*>(h + s * geom::HID) + d8;
        const uint4 prev = *hp;
        const __nv_bfloat162* pr = reinterpret_cast<const __nv_bfloat162*>(&prev);
        uint4 out;
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&out);
        const float ws = w[s];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 fp = __bfloat1622float2(pr[k]);
            const float a0 = __bfloat162float(__float2bfloat16(y[2 * k] * ws));
            const float a1 = __bfloat162float(__float2bfloat16(y[2 * k + 1] * ws));
            op[k] = __floats2bfloat162_rn(fp.x + a0, fp.y + a1);
        }
        *hp = out;
    }
}

inline void qw38_ar_exchange_inject(const g6::FabricDev& f, const unsigned long long* d_step, int par, int c,
                                    __nv_bfloat16* hyper, const __nv_bfloat16* parts, const float* inj,
                                    int M, cudaStream_t st) {
    const size_t bytes = static_cast<size_t>(M) * geom::HID * sizeof(__nv_bfloat16);
    const unsigned code = ink2::INK2_ERR_AR | (static_cast<unsigned>(c) << 8);
    Qw38ArInject a{f, d_step, hyper, parts, inj, bytes, M, par, c, code};
    if constexpr (G6_NR > 1) {
        if (M <= 8) {
            k_qw38_ar_fused_inject_small<G6_NR><<<1, 320, 0, st>>>(a);
            return;
        }
        const size_t n16 = bytes / 16;
        const unsigned grid = static_cast<unsigned>((n16 + 255) / 256);
        ink2::k_ink2_ar_copy<<<grid, 256, 0, st>>>(
            reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(parts) + static_cast<size_t>(f.rank) * bytes),
            reinterpret_cast<uint4*>(f.send + g6::fab_send_off(f, par, c)), n16);
        Qw38ArWait w{f, bytes, d_step, code, par, c};
        k_qw38_ar_ring_wait<<<1, 32, 0, st>>>(w);
        k_qw38_hc_inject_from_recv<G6_NR><<<M * 5, 64, 0, st>>>(a);
    }
}

struct Wire {
    g6::Fabric fab;
    g6::FabricDev f{};
    unsigned long long step = 0;
    unsigned long long* d_step = nullptr;
    int rank = 0, nr = 1;
    int n_layers = 0;
    size_t head_rb = 0;
    bool fast = true;

    ~Wire() { if (d_step) cudaFree(d_step); }

    void publish() { __atomic_store_n(f.doorbell, g6::coll_seq(step, 0, f.ncoll), __ATOMIC_RELEASE); }

    void wait_host(size_t bytes) {
        const unsigned long long seq = g6::coll_seq(step, 0, f.ncoll);
        const int par = static_cast<int>(step & 1);
        const auto t0 = std::chrono::steady_clock::now();
        for (int p = 0; p < nr; ++p) {
            if (p == rank) continue;
            const auto* tail = reinterpret_cast<const unsigned long long*>(f.recv + g6::fab_recv_off(f, par, 0, p) + bytes);
            unsigned it = 0;
            while (__atomic_load_n(tail, __ATOMIC_ACQUIRE) != seq) {
                if ((++it & 0xFFFFu) == 0 && (*f.error || std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)))
                    die("rank " + std::to_string(rank) + ": fabric timeout from rank " + std::to_string(p));
            }
        }
    }

    // Multi-collective single-step device-driven AR (ncoll = 2*L + 1, ZERO syncs across all layer collectives!)
    void fast_begin(int M, bool skip_head, int n_layers_override, cudaStream_t st) {
        const int nl = (n_layers_override > 0) ? n_layers_override : n_layers;
        const uint32_t wire_part = static_cast<uint32_t>((size_t(M) * geom::HID * 2 + 15) / 16 * 16);
        const uint32_t wire_head = static_cast<uint32_t>((size_t(M) * head_rb + 7) / 8 * 8);
        std::vector<uint32_t> sizes(static_cast<size_t>(2 * nl + (skip_head ? 0 : 1)), wire_part);
        if (!skip_head) sizes.back() = wire_head;
        fab.enqueue_step(step, sizes);
        QW38_CK(cudaMemcpyAsync(d_step, &step, sizeof(step), cudaMemcpyHostToDevice, st));
    }

    void fast_dev(int c, __nv_bfloat16* parts, size_t bytes, cudaStream_t st) {
        const size_t wire = (bytes + 15) / 16 * 16;
        qw38_ar_exchange(f, d_step, static_cast<int>(step & 1), c, reinterpret_cast<char*>(parts), wire, st);
    }

    void fast_dev_inject(int c, __nv_bfloat16* hyper, const __nv_bfloat16* parts, const float* inj,
                         int M, cudaStream_t st) {
        qw38_ar_exchange_inject(f, d_step, static_cast<int>(step & 1), c, hyper, parts, inj, M, st);
    }

    void fast_host(int c, uint8_t* parts, size_t bytes) {
        const int par = static_cast<int>(step & 1);
        const size_t wire = (bytes + 7) / 8 * 8;
        const unsigned long long seq = g6::coll_seq(step, c, f.ncoll);
        char* dst = f.send + g6::fab_send_off(f, par, c);
        std::memcpy(dst, parts + size_t(rank) * bytes, bytes);
        if (wire > bytes) std::memset(dst + bytes, 0, wire - bytes);
        __atomic_store_n(f.doorbell + c, seq, __ATOMIC_RELEASE);
        const auto t0 = std::chrono::steady_clock::now();
        for (int p = 0; p < nr; ++p) {
            if (p == rank) continue;
            const char* slot = f.recv + g6::fab_recv_off(f, par, c, p);
            const auto* tail = reinterpret_cast<const unsigned long long*>(slot + wire);
            unsigned it = 0;
            while (__atomic_load_n(tail, __ATOMIC_ACQUIRE) != seq) {
                if ((++it & 0xFFFFu) == 0 && (*f.error || std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)))
                    die("rank " + std::to_string(rank) + ": fabric timeout at head collective from rank " + std::to_string(p));
            }
            std::memcpy(parts + size_t(p) * bytes, slot, bytes);
        }
    }

    void fast_end(cudaStream_t st) {
        QW38_CK(cudaStreamSynchronize(st));
        if (!fab.wait_proxy_idle(60.0)) die("fabric proxy timeout at step " + std::to_string(step));
        if (*f.error) die("fabric device error " + std::to_string(*f.error));
        ++step;
    }

    void exchange_dev(int, __nv_bfloat16* parts, size_t bytes) {
        const int par = static_cast<int>(step & 1);
        const size_t wire = (bytes + 15) / 16 * 16;
        fab.enqueue_step(step, {static_cast<uint32_t>(wire)});
        char* d = reinterpret_cast<char*>(parts);
        QW38_CK(cudaMemcpy(f.send + g6::fab_send_off(f, par, 0), d + size_t(rank) * bytes, bytes, cudaMemcpyDeviceToHost));
        publish();
        wait_host(wire);
        if (!fab.wait_proxy_idle(60.0)) die("fabric proxy timeout");
        for (int p = 0; p < nr; ++p)
            if (p != rank) QW38_CK(cudaMemcpy(d + size_t(p) * bytes, f.recv + g6::fab_recv_off(f, par, 0, p), bytes, cudaMemcpyHostToDevice));
        ++step;
    }

    void exchange_host(int, uint8_t* parts, size_t bytes) {
        const size_t wire = (bytes + 7) / 8 * 8;
        const int par = static_cast<int>(step & 1);
        fab.enqueue_step(step, {static_cast<uint32_t>(wire)});
        char* dst = f.send + g6::fab_send_off(f, par, 0);
        std::memcpy(dst, parts + size_t(rank) * bytes, bytes);
        if (wire > bytes) std::memset(dst + bytes, 0, wire - bytes);
        publish();
        wait_host(wire);
        if (!fab.wait_proxy_idle(60.0)) die("fabric proxy timeout");
        for (int p = 0; p < nr; ++p)
            if (p != rank) std::memcpy(parts + size_t(p) * bytes, f.recv + g6::fab_recv_off(f, par, 0, p), bytes);
        ++step;
    }
};

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) v.push_back(item);
    return v;
}
} // namespace

int main(int argc, char** argv) {
    using namespace qw38rt;
    std::string dir, widths, layout, layers = "all", tokens, s2dir, set_id, ips, ips2, kv_keep = "all";
    std::vector<std::string> kv_tiers;
    int rank = -1, proxy_cpu = -1, gen = 16, chunk = 8, prefill = 256, nc = 256, cap = 65536, mtp = 0;
    int s2port = 7341, s2sess = 1, s2epoch = 0, kv_sys_after = 2;
    double kv_wlim = 0.0;
    bool reuse = false;
    float temp = 0.f, top_p = 1.f;
    uint32_t seed = 1234;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--rank") rank = std::atoi(val().c_str());
        else if (a == "--ips") ips = val();
        else if (a == "--ips2") ips2 = val();
        else if (a == "--proxy-cpu") proxy_cpu = std::atoi(val().c_str());
        else if (a == "--packs" || a == "--v2-dense" || a == "--model" || a == "--pack") dir = val();
        else if (a == "--manifest") (void)val();
        else if (a == "--widths") widths = val();
        else if (a == "--layout") layout = val();
        else if (a == "--layers") layers = val();
        else if (a == "--tokens") tokens = val();
        else if (a == "--gen") gen = std::atoi(val().c_str());
        else if (a == "--chunk") chunk = std::atoi(val().c_str());
        else if (a == "--prefill") prefill = std::atoi(val().c_str());
        else if (a == "--top-p-cands") nc = std::atoi(val().c_str());
        else if (a == "--cap") cap = std::atoi(val().c_str());
        else if (a == "--mtp") mtp = std::atoi(val().c_str());
        else if (a == "--serve") s2dir = val();
        else if (a == "--s2-port") s2port = std::atoi(val().c_str());
        else if (a == "--s2-sessions" || a == "--sessions") s2sess = std::atoi(val().c_str());
        else if (a == "--s2-epoch") s2epoch = std::atoi(val().c_str());
        else if (a == "--set-id") set_id = val();
        else if (a == "--reuse-prefix") reuse = true;
        else if (a == "--kv-store") { kv_tiers.push_back(val()); reuse = true; }
        else if (a == "--kv-keep") kv_keep = val();
        else if (a == "--kv-system-after") kv_sys_after = std::atoi(val().c_str());
        else if (a == "--kv-write-limit") kv_wlim = std::atof(val().c_str());
        else if (a == "--temp") temp = static_cast<float>(std::atof(val().c_str()));
        else if (a == "--top-p") top_p = static_cast<float>(std::atof(val().c_str()));
        else if (a == "--seed") seed = static_cast<uint32_t>(std::strtoul(val().c_str(), nullptr, 10));
        else if (a == "--graph" || a == "--no-graph") {}
        else if (a == "--grammar-dir") (void)val();
        else die("unknown option " + a);
    }
    if (nc < 0 || nc > 2048 / G6_NR) die("--top-p-cands must be 0.." + std::to_string(2048 / G6_NR));
    const auto peers = split_csv(ips);
    if (static_cast<int>(peers.size()) != G6_NR) die("--ips needs " + std::to_string(G6_NR) + " addresses");
    if (rank < 0 || rank >= G6_NR) die("--rank must be 0.." + std::to_string(G6_NR - 1));
    if (dir.empty()) die("name the release directory: --packs DIR");
    knob::init();

    Options o;
    o.nr = G6_NR;
    o.rank0 = rank;
    o.nloc = 1;
    o.slots = s2sess;
    o.max_rows = std::max(prefill, std::max(chunk, o.slots * (mtp + 1)));
    o.cap = cap;
    o.nc = std::max(nc, 1);
    o.mtp = mtp;
    o.layers = layers;
    o.widths_path = widths;
    o.layout_path = layout;

    Engine e;
    e.init(dir, o);
    if (!s2dir.empty() && knob::qw38_vision()) {
        const std::string vp = dir + "/dense/vision-bf16.safetensors";
        if (std::filesystem::exists(vp)) {
            e.load_vision(vp);
            e.vision.log = true;
            g6rt::vision_flag() = true;
            std::fprintf(stderr, "# rank %d: qw38 vision: %.1f MiB, id %s\n",
                         rank, e.vision_device_bytes() / (1024.0 * 1024.0), e.vision_id().c_str());
            if (rank == 0) {
                if (FILE* fv = std::fopen((s2dir + "/vision.s2").c_str(), "w")) {
                    std::fputs("qw38\n", fv);
                    std::fclose(fv);
                }
            }
        } else {
            std::fprintf(stderr, "# rank %d: qw38 vision off: %s not found\n", rank, vp.c_str());
        }
    }
    if (!s2dir.empty() && rank == 0 && !e.vision_ready())
        std::remove((s2dir + "/vision.s2").c_str());

    Wire wire;
    if (G6_NR > 1) {
        wire.fast = knob::qw38_fast_ar();
        const size_t max_part_b = (size_t(o.max_rows) * geom::HID * 2 + 15) / 16 * 16;
        const size_t max_head_b = (size_t(o.max_rows) * head_row_bytes(o.nc) + 15) / 16 * 16;
        const size_t slot_b = std::max(max_part_b, max_head_b);
        g6::FabricConfig cfg;
        cfg.rank = rank;
        cfg.ips = peers;
        if (!ips2.empty()) cfg.ips2 = split_csv(ips2);
        cfg.proxy_cpu = proxy_cpu;
        cfg.ncoll = wire.fast ? static_cast<int>(2 * e.L.size() + 1) : 1;
        cfg.stripe_min = cfg.ips2.empty() ? 0 : 65536;
        cfg.compact = true;
        cfg.slot_bytes = (slot_b + 7) / 8 * 8;
        wire.fab.init(cfg);
        wire.f = wire.fab.dev();
        wire.fab.start_proxy();
        wire.fab.barrier();
        wire.rank = rank;
        wire.nr = G6_NR;
        wire.n_layers = static_cast<int>(e.L.size());
        wire.head_rb = head_row_bytes(o.nc);
        if (wire.fast) {
            wire.d_step = qw38rt::dalloc<unsigned long long>(1);
            e.step_begin = [&](int M, bool skip_head, int nl) { wire.fast_begin(M, skip_head, nl, e.stream); };
            e.gather_dev = [&](int c, __nv_bfloat16* p, size_t b) { wire.fast_dev(c, p, b, e.stream); };
            e.gather_inject = [&](int c, __nv_bfloat16* h, const __nv_bfloat16* p, const float* inj, int M) {
                wire.fast_dev_inject(c, h, p, inj, M, e.stream);
            };
            e.gather_host = [&](int c, uint8_t* p, size_t b) { wire.fast_host(c, p, b); };
            e.step_end = [&]() { wire.fast_end(e.stream); };
            e.step_parity = [&]() -> int { return static_cast<int>(wire.step & 1); };
        } else {
            e.gather_dev = [&](int c, __nv_bfloat16* p, size_t b) { wire.exchange_dev(c, p, b); };
            e.gather_host = [&](int c, uint8_t* p, size_t b) { wire.exchange_host(c, p, b); };
        }
    }

    if (!s2dir.empty()) {
        g6s2::BootstrapConfig bc;
        auto& sc = bc.scheduler;
        sc.epoch = static_cast<uint64_t>(s2epoch);
        sc.pool_rows = static_cast<uint32_t>(s2sess) * static_cast<uint32_t>(e.max_pages_per_slot()) * geom::KV_PAGE;
        sc.max_rows = static_cast<uint32_t>(std::max(prefill, std::max(chunk, s2sess)));
        sc.max_sessions = static_cast<uint32_t>(s2sess);
        sc.prefill_rows = static_cast<uint32_t>(prefill);
        sc.mtp = static_cast<uint32_t>(mtp);
        sc.two_lanes = false;
        bc.rank = static_cast<unsigned>(rank);
        bc.ranks = static_cast<unsigned>(G6_NR);
        bc.head_ip = peers[0];
        bc.directory = s2dir;
        bc.port = static_cast<uint16_t>(s2port);
        bc.timeout = static_cast<unsigned>(knob::s2_timeout());
        bc.runtime_signature = "qw38|tp" + std::to_string(G6_NR) + "|cap" + std::to_string(o.cap) +
                               "|sess" + std::to_string(s2sess) + "|layers " + layers +
                               "|chunk" + std::to_string(chunk) + "|pf" + std::to_string(prefill) +
                               "|reuse" + std::to_string(reuse ? 1 : 0) + "|nc" + std::to_string(o.nc) +
                               "|set " + set_id + "|" + knob::signature() +
                               g6vis::signature(e.vision_ready(), e.vision_id());

        g6s2::Qw38Context ctx{e, rank, reuse, kv_tiers, kv_keep, kv_sys_after, kv_wlim, set_id};
        g6s2::serve_qw38(ctx, bc, chunk);
        if (G6_NR > 1) wire.fab.stop_proxy();
        return 0;
    }

    // Standalone cluster smoke / benchmark run (supports --sessions 1..8)
    std::vector<int> prompt;
    if (!tokens.empty()) {
        std::stringstream ss(tokens);
        std::string item;
        while (std::getline(ss, item, ',')) prompt.push_back(std::atoi(item.c_str()));
    }
    if (prompt.empty()) prompt.push_back(geom::TOK_BOS);
    const int P = static_cast<int>(prompt.size());
    const int S = std::max(1, s2sess);
    const float inv_temp = temp > 0.f ? 1.f / temp : 0.f;
    std::vector<RowOut> out;
    std::vector<int> next(static_cast<size_t>(S), 0);
    std::vector<std::vector<int>> said(static_cast<size_t>(S));

    const auto t_pf0 = std::chrono::steady_clock::now();
    for (int s = 0; s < S; ++s) {
        e.reset_slot(s);
        for (int p = 0; p < P; p += prefill) {
            const int n = std::min(prefill, P - p);
            const bool last = (p + n == P);
            std::vector<qw38::StepSeg> segs = {{s, 0, p, n}};
            std::vector<RowSample> sm(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) sm[size_t(i)] = RowSample{inv_temp, top_p, seed, uint32_t(p + i + 1)};
            e.forward(segs, &prompt[size_t(p)], sm.data(), out, !last);
            if (e.has_mtp) {
                const int sync_n = std::min(n, P - 1 - p);
                if (sync_n > 0) e.mtp_sync_kv(s, 0, p, sync_n, &prompt[size_t(p + 1)]);
                if (last) e.mtp_save_last_hyper(s, n - 1);
            }
            if (last) next[size_t(s)] = out[size_t(n - 1)].tok;
        }
    }
    QW38_CK(cudaStreamSynchronize(e.stream));
    const auto t_pf1 = std::chrono::steady_clock::now();

    const auto t_dec0 = std::chrono::steady_clock::now();
    int total_drafted = 0, total_accepted = 0, mtp_rounds = 0;
    if (gen > 0) {
        for (int s = 0; s < S; ++s) said[size_t(s)].push_back(next[size_t(s)]);
        if (e.has_mtp && mtp > 0) {
            std::vector<int> pos(static_cast<size_t>(S), P);
            auto all_done = [&]() {
                for (int s = 0; s < S; ++s)
                    if (static_cast<int>(said[size_t(s)].size()) < gen) return false;
                return true;
            };
            while (!all_done()) {
                e.mtp_round(S, mtp, next, pos, inv_temp, top_p, seed, said, total_drafted, total_accepted);
                ++mtp_rounds;
            }
            for (int s = 0; s < S; ++s)
                if (static_cast<int>(said[size_t(s)].size()) > gen) said[size_t(s)].resize(size_t(gen));
        } else {
            for (int g = 1; g < gen; ++g) {
                std::vector<qw38::StepSeg> segs;
                std::vector<RowSample> sm;
                for (int s = 0; s < S; ++s) {
                    segs.push_back({s, s, P + g - 1, 1});
                    sm.push_back(RowSample{inv_temp, top_p, seed, uint32_t(P + g)});
                }
                e.forward(segs, next.data(), sm.data(), out, false);
                for (int s = 0; s < S; ++s) {
                    next[size_t(s)] = out[size_t(s)].tok;
                    said[size_t(s)].push_back(next[size_t(s)]);
                }
            }
        }
    }
    QW38_CK(cudaStreamSynchronize(e.stream));
    const auto t_dec1 = std::chrono::steady_clock::now();

    const double pf_ms = std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count();
    const double dec_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
    if (rank == 0) {
        const double tps_s = gen > 1 ? ((gen - 1) / (dec_ms / 1e3 + 1e-9)) : 0.0;
        std::fprintf(stderr, "# g6m4_qw38 TP=%d P=%d gen=%d sess=%d mtp=%d: prefill=%.1f ms (%.1f tok/s), decode=%.1f ms (%.2f tok/s/sess, %.2f agg tok/s)",
                     G6_NR, P, gen, S, mtp, pf_ms, (P * S) / (pf_ms / 1e3 + 1e-9), dec_ms, tps_s, tps_s * S);
        if (mtp_rounds > 0) {
            const double acc_rate = total_drafted > 0 ? (100.0 * total_accepted / total_drafted) : 0.0;
            const double tok_per_round = double(S * mtp_rounds + total_accepted) / double(S * mtp_rounds);
            std::fprintf(stderr, " [mtp acc=%.1f%%, %.2f tok/round, %d rounds]", acc_rate, tok_per_round, mtp_rounds);
        }
        std::fprintf(stderr, "\n");
        if (e.o.profile) e.prof.print(rank);
        for (int s = 0; s < S; ++s) {
            std::printf("session %d:", s);
            for (int t : said[size_t(s)]) std::printf(" %d", t);
            std::printf("\n");
        }
    }
    if (G6_NR > 1) wire.fab.stop_proxy();
    return 0;
}
