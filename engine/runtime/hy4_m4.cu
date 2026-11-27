// g6m4_hy4: one rank of Tencent Hy4-flash on the real RDMA fabric (TP = 1 or TP = 2).
// Supports standalone smoke/bench runs (with optional built-in DFly2) and S2 multi-session serving (--serve DIR).
#include "hy4_step.cuh"
#include "../comms/fabric.h"
#include "../kernels/ink2_fabric.cuh"
#include "../knobs-rs/gr8_knobs.h"
#include "s2/hy4_native.cuh"
#include <chrono>
#include <sstream>

#ifndef G6_NR
#define G6_NR 1
#endif

namespace {
using hy4rt::die;

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) v.push_back(item);
    return v;
}

struct Hy4ArWait {
    g6::FabricDev f;
    size_t bytes;
    const unsigned long long* d_step;
    unsigned code;
    int par, c;
};

__global__ void k_hy4_ar_ring_wait(Hy4ArWait a) {
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

struct Hy4Wire {
    g6::Fabric fab;
    g6::FabricDev f{};
    unsigned long long step = 0;
    unsigned long long* d_step = nullptr;
    int rank = 0, nr = 1;
    size_t head_rb = 0;

    ~Hy4Wire() { if (d_step) cudaFree(d_step); }

    void init(int r, int n_ranks, int max_M, size_t rb, int ncoll,
              const std::vector<std::string>& peers, int port, int proxy_cpu,
              const std::vector<std::string>& peers2) {
        rank = r; nr = n_ranks; head_rb = rb;
        if (nr <= 1) return;
        d_step = hy4rt::dalloc<unsigned long long>(1);
        g6::FabricConfig cfg{};
        cfg.rank = r;
        cfg.ips = peers;
        cfg.ips2 = peers2;
        cfg.proxy_cpu = proxy_cpu;
        cfg.port = port;
        cfg.ncoll = ncoll;
        const size_t max_part_b = ((size_t(max_M) * geom::HID * 2 + 15) / 16) * 16;
        const size_t max_head_b = ((size_t(max_M) * rb + 15) / 16) * 16;
        cfg.slot_bytes = ((std::max(max_part_b, max_head_b) + 7) / 8) * 8;
        cfg.stripe_min = peers2.empty() ? 0 : 65536;
        cfg.compact = false;
        fab.init(cfg);
        f = fab.dev();
        fab.start_proxy();
        fab.barrier();
    }

    void begin(int M, bool skip_head, cudaStream_t st) {
        if (nr <= 1) return;
        const uint32_t wire_part = static_cast<uint32_t>(((size_t(M) * geom::HID * 2 + 15) / 16) * 16);
        const uint32_t wire_head = static_cast<uint32_t>(((size_t(M) * head_rb + 7) / 8) * 8);
        std::vector<uint32_t> sizes(static_cast<size_t>(2 * geom::L_MAIN + (skip_head ? 0 : 1)), wire_part);
        if (!skip_head) sizes.back() = wire_head;
        fab.enqueue_step(step, sizes);
        HY4_CK(cudaMemcpyAsync(d_step, &step, sizeof(step), cudaMemcpyHostToDevice, st));
    }

    void begin_dfly(int W, int K, cudaStream_t st) {
        if (nr <= 1) return;
        const uint32_t wire_part = static_cast<uint32_t>(((size_t(W) * geom::HID * 2 + 15) / 16) * 16);
        const uint32_t wire_head = static_cast<uint32_t>(((head_rb + 7) / 8) * 8);
        std::vector<uint32_t> sizes(static_cast<size_t>(2 * hy4::dfly::NUM_LAYERS + K), wire_part);
        for (int k = 0; k < K; ++k) sizes[static_cast<size_t>(2 * hy4::dfly::NUM_LAYERS + k)] = wire_head;
        fab.enqueue_step(step, sizes);
        HY4_CK(cudaMemcpyAsync(d_step, &step, sizeof(step), cudaMemcpyHostToDevice, st));
    }

    void exchange_parts(int c, int M, __nv_bfloat16* parts, cudaStream_t st) const {
        if (nr <= 1) return;
        const size_t bytes = ((size_t(M) * geom::HID * 2 + 15) / 16) * 16;
        const size_t n16 = bytes / 16;
        const unsigned grid = static_cast<unsigned>((n16 + 255) / 256);
        const int par = static_cast<int>(step & 1);
        ink2::k_ink2_ar_copy<<<grid, 256, 0, st>>>(
            reinterpret_cast<const uint4*>(parts + static_cast<size_t>(rank) * M * geom::HID),
            reinterpret_cast<uint4*>(f.send + g6::fab_send_off(f, par, c)), n16);
        Hy4ArWait a{f, bytes, d_step, ink2::INK2_ERR_AR | (static_cast<unsigned>(c) << 8), par, c};
        k_hy4_ar_ring_wait<<<1, 32, 0, st>>>(a);
        for (int p = 0; p < nr; ++p) {
            if (p != rank) {
                ink2::k_ink2_ar_copy<true><<<grid, 256, 0, st>>>(
                    reinterpret_cast<const uint4*>(f.recv + g6::fab_recv_off(f, par, c, p)),
                    reinterpret_cast<uint4*>(parts + static_cast<size_t>(p) * M * geom::HID), n16);
            }
        }
    }

    void gather_head(int c, int M, uint8_t* host_parts, size_t rb) const {
        if (nr <= 1) return;
        const size_t bytes = size_t(M) * rb;
        const size_t wire = ((bytes + 7) / 8) * 8;
        const int par = static_cast<int>(step & 1);
        const unsigned long long seq = g6::coll_seq(step, c, f.ncoll);
        char* dst = f.send + g6::fab_send_off(f, par, c);
        std::memcpy(dst, host_parts + static_cast<size_t>(rank) * bytes, bytes);
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
                    die("hy4 head gather timeout from peer " + std::to_string(p));
            }
            std::memcpy(host_parts + static_cast<size_t>(p) * bytes, slot, bytes);
        }
    }

    void end(cudaStream_t st) {
        if (nr <= 1) return;
        HY4_CK(cudaStreamSynchronize(st));
        if (!fab.wait_proxy_idle(60.0)) die("fabric proxy timeout at step " + std::to_string(step));
        if (*f.error) die("fabric device error " + std::to_string(*f.error));
        ++step;
    }
};

} // namespace

int main(int argc, char** argv) {
    std::string packs, widths, engram_dir, tok_str, serve_dir, ips, ips2, kv_fmt_str;
    int rank = 0, port = 18600, proxy_cpu = -1, s2_port = 7341, s2_sessions = 4, s2_epoch = 1;
    int bench_ctx = 0, bench_kv_pos = 0, dfly_tokens = 5;
    size_t engram_cache_mb = 0;
    bool engram_cache_set = false;
    int cap = 1048576, pool_rows = 0, chunk = 8, prefill = 1024, gen = 16, nc = 256;
    float temp = 0.0f, top_p = 1.0f;
    uint32_t seed = 42;
    bool profile = false, no_engram = false;
    bool mx8_set = false, mx8_val = true;
    bool dfly_set = false, dfly_val = true, dfly_tokens_set = false;
    bool graphs_set = false, graphs_val = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* f) { if (i + 1 >= argc) die(std::string("missing value for ") + f); return std::string(argv[++i]); };
        if (a == "--packs" || a == "--v2-dense" || a == "--model" || a == "--pack") packs = need(a.c_str());
        else if (a == "--widths")         widths = need("--widths");
        else if (a == "--engram-cache-mb"){ engram_cache_mb = std::stoull(need("--engram-cache-mb")); engram_cache_set = true; }
        else if (a == "--engram")         engram_dir = need("--engram");
        else if (a == "--kv-format")      kv_fmt_str = need("--kv-format");
        else if (a == "--mx8")            { mx8_set = true; mx8_val = true; }
        else if (a == "--no-mx8")         { mx8_set = true; mx8_val = false; }
        else if (a == "--dfly")           { dfly_set = true; dfly_val = true; }
        else if (a == "--no-dfly")        { dfly_set = true; dfly_val = false; }
        else if (a == "--dfly-tokens")    { dfly_tokens = std::stoi(need("--dfly-tokens")); dfly_tokens_set = true; }
        else if (a == "--graph")          { graphs_set = true; graphs_val = true; }
        else if (a == "--no-graph")       { graphs_set = true; graphs_val = false; }
        else if (a == "--rank")           rank = std::stoi(need("--rank"));
        else if (a == "--ips")            ips = need("--ips");
        else if (a == "--ips2")           ips2 = need("--ips2");
        else if (a == "--port")           port = std::stoi(need("--port"));
        else if (a == "--proxy-cpu")      proxy_cpu = std::stoi(need("--proxy-cpu"));
        else if (a == "--serve")          serve_dir = need("--serve");
        else if (a == "--s2-port")        s2_port = std::stoi(need("--s2-port"));
        else if (a == "--s2-sessions" || a == "--sessions") s2_sessions = std::stoi(need(a.c_str()));
        else if (a == "--s2-epoch")       s2_epoch = std::stoi(need("--s2-epoch"));
        else if (a == "--cap")            cap = std::stoi(need("--cap"));
        else if (a == "--pool-rows")      pool_rows = std::stoi(need("--pool-rows"));
        else if (a == "--chunk")          chunk = std::stoi(need("--chunk"));
        else if (a == "--prefill")        prefill = std::stoi(need("--prefill"));
        else if (a == "--top-p-cands")    nc = std::stoi(need("--top-p-cands"));
        else if (a == "--tokens")         tok_str = need("--tokens");
        else if (a == "--gen")            gen = std::stoi(need("--gen"));
        else if (a == "--bench-ctx")      bench_ctx = std::stoi(need("--bench-ctx"));
        else if (a == "--bench-kv-pos")   bench_kv_pos = std::stoi(need("--bench-kv-pos"));
        else if (a == "--temp")           temp = std::stof(need("--temp"));
        else if (a == "--top-p")          top_p = std::stof(need("--top-p"));
        else if (a == "--seed")           seed = static_cast<uint32_t>(std::stoul(need("--seed")));
        else if (a == "--profile")        profile = true;
        else if (a == "--no-engram")      no_engram = true;
        else if (a == "--layers" || a == "--mtp" || a == "--layout" || a == "--manifest" ||
                 a == "--set-id" || a == "--grammar-dir") (void)need(a.c_str());
        else if (a == "--reuse-prefix") {}
        else die("unknown arg: " + a);
    }
    if (nc < 0 || nc > 2048 / G6_NR)
        die("--top-p-cands must be 0.." + std::to_string(2048 / G6_NR));
    if (packs.empty()) die("need --packs <release_dir>");
    knob::init();

    std::vector<std::string> peers = split_csv(ips);
    if (peers.empty()) peers.push_back("127.0.0.1");
    if (G6_NR > 1 && static_cast<int>(peers.size()) != G6_NR)
        die("--ips needs " + std::to_string(G6_NR) + " addresses");
    if (rank < 0 || rank >= G6_NR) die("--rank out of range");

    hy4rt::Options o;
    o.nr = G6_NR;
    o.rank0 = rank;
    o.nloc = 1;
    const int eff_dfly_k = std::max(1, dfly_tokens_set ? dfly_tokens : static_cast<int>(knob::hy4_dfly_tokens()));
    o.slots = s2_sessions;
    o.max_rows = std::max({chunk, prefill, s2_sessions * (eff_dfly_k + 1), 16});
    o.cap = cap;
    o.pool_rows = pool_rows;
    o.nc = std::max(nc, 1);
    o.widths_path = widths;
    o.engram_dir = engram_dir;
    o.engram_cache_mib = engram_cache_mb;
    o.engram_cache_set = engram_cache_set;
    o.profile = profile;
    o.no_engram = no_engram;
    if (!kv_fmt_str.empty()) { o.kv_fmt = hy4rt::parse_kv_fmt(kv_fmt_str); o.kv_fmt_set = true; }
    if (mx8_set)         { o.mx8_dense = mx8_val; o.mx8_set = true; }
    if (dfly_set)        { o.use_dfly = dfly_val; o.dfly_set = true; }
    if (dfly_tokens_set) { o.dfly_tokens = dfly_tokens; o.dfly_tokens_set = true; }
    if (graphs_set)      { o.use_graphs = graphs_val; o.graphs_set = true; }

    hy4rt::Engine e;
    e.init(packs, o);

    Hy4Wire wire;
    const int ncoll = std::max(2 * geom::L_MAIN + 1, 2 * hy4::dfly::NUM_LAYERS + e.o.dfly_tokens);
    wire.init(rank, G6_NR, o.max_rows, hy4rt::head_row_bytes(o.nc), ncoll, peers, port, proxy_cpu, split_csv(ips2));
    if (G6_NR > 1) {
        e.on_step_begin = [&](int M, bool skip_head) { wire.begin(M, skip_head, e.stream); };
        e.on_ar_norm_inject = [&](int c, int M, const __nv_bfloat16* post_ln) {
            wire.exchange_parts(c, M, e.parts, e.stream);
            hy4::k_hy4_sum_norm_inject<<<M, 256, 0, e.stream>>>(
                e.parts, post_ln, e.hc_sc.post, e.hyper, e.o.nr, M, M);
        };
        e.on_head_gather = [&](int c, int M, uint8_t* hp, size_t rb) { wire.gather_head(c, M, hp, rb); };
        e.on_step_end = [&]() { wire.end(e.stream); };

        e.on_dfly_begin = [&](int W) { wire.begin_dfly(W, W - 1, e.stream); };
        e.on_dfly_ar_add = [&](int c, int W, const __nv_bfloat16* post_ln) {
            wire.exchange_parts(c, W, e.parts, e.stream);
            hy4::k_hy4_sum_norm_add1<<<W, 256, 0, e.stream>>>(e.parts, post_ln, e.dfly_sc.h, e.o.nr, W, W);
        };
        e.on_dfly_head_gather = [&](int c, int M, uint8_t* hp, size_t rb) { wire.gather_head(c, M, hp, rb); };
        e.on_dfly_end = [&]() { wire.end(e.stream); };
    }

    if (!serve_dir.empty()) {
        g6s2::BootstrapConfig bc;
        auto& sc = bc.scheduler;
        sc.epoch = static_cast<uint64_t>(s2_epoch);
        sc.pool_rows = static_cast<uint32_t>(e.pool_pages) * geom::KV_PAGE;
        sc.max_rows = static_cast<uint32_t>(o.max_rows);
        sc.max_sessions = static_cast<uint32_t>(s2_sessions);
        sc.prefill_rows = static_cast<uint32_t>(prefill);
        sc.mtp = 0;
        sc.two_lanes = false;
        bc.rank = static_cast<unsigned>(rank);
        bc.ranks = static_cast<unsigned>(G6_NR);
        bc.head_ip = peers[0];
        bc.directory = serve_dir;
        bc.port = static_cast<uint16_t>(s2_port);
        bc.timeout = static_cast<unsigned>(knob::s2_timeout());
        bc.runtime_signature = "hy4|tp" + std::to_string(G6_NR) + "|kv" + hy4rt::kv_fmt_name(e.o.kv_fmt) +
                               "|mx8" + std::to_string(e.o.mx8_dense ? 1 : 0) +
                               "|dfly" + std::to_string(e.has_dfly ? e.o.dfly_tokens : 0) +
                               "|cap" + std::to_string(o.cap) + "|pool" + std::to_string(e.pool_pages) +
                               "|sess" + std::to_string(s2_sessions) + "|chunk" + std::to_string(chunk) +
                               "|pf" + std::to_string(prefill) + "|nc" + std::to_string(o.nc) + "|" + knob::signature();
        g6s2::serve_hy4(g6s2::Hy4Context{e, rank}, bc, chunk);
        if (G6_NR > 1) wire.fab.stop_proxy();
        return 0;
    }

    if (bench_kv_pos > 0) {
        const int S = std::max(1, s2_sessions);
        if (bench_kv_pos + gen + e.o.dfly_tokens + 2 > cap) die("--bench-kv-pos + --gen exceeds --cap");
        e.seed_bench_kv(bench_kv_pos, S);
        std::vector<hy4rt::RowOut> out;
        std::vector<int> cur(size_t(S), geom::TOK_BOS);
        const float inv_temp = temp > 0.f ? 1.f / temp : 0.f;
        const int warm = 4;
        for (int g = 0; g < warm; ++g) {
            std::vector<hy4::StepSeg> segs;
            std::vector<hy4rt::RowSample> sm;
            for (int s = 0; s < S; ++s) {
                segs.push_back({s, s, bench_kv_pos - warm + g, 1});
                sm.push_back(hy4rt::RowSample{inv_temp, top_p, seed, uint32_t(g + 1)});
            }
            e.forward(segs, cur.data(), sm.data(), out, false);
            for (int s = 0; s < S; ++s) cur[size_t(s)] = out[size_t(s)].tok;
        }
        HY4_CK(cudaStreamSynchronize(e.stream));
        if (e.o.profile) e.prof = hy4rt::ProfileAccum{};
        const auto td0 = std::chrono::steady_clock::now();
        hy4rt::SpecStats st{};
        if (S == 1 && e.has_dfly) {
            std::vector<int> said_b;
            hy4rt::run_dfly_gen_s1(e, 0, bench_kv_pos, cur[0], 0, gen + 1, inv_temp, top_p, seed, said_b, st);
        } else {
            for (int g = 0; g < gen; ++g) {
                std::vector<hy4::StepSeg> segs;
                std::vector<hy4rt::RowSample> sm;
                for (int s = 0; s < S; ++s) {
                    segs.push_back({s, s, bench_kv_pos + g, 1});
                    sm.push_back(hy4rt::RowSample{inv_temp, top_p, seed, uint32_t(bench_kv_pos + g + 1)});
                }
                e.forward(segs, cur.data(), sm.data(), out, false);
                for (int s = 0; s < S; ++s) cur[size_t(s)] = out[size_t(s)].tok;
            }
        }
        HY4_CK(cudaStreamSynchronize(e.stream));
        const auto td1 = std::chrono::steady_clock::now();
        if (rank == 0) {
            const double dec_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();
            const double tps = gen / (dec_ms / 1e3 + 1e-9);
            std::fprintf(stderr, "# g6m4_hy4 BENCH_KV TP=%d kv=%s mx8=%d dfly=%d ctx=%d gen=%d sess=%d: decode=%.1f ms (%.2f tok/s/sess, %.2f agg tok/s)",
                         G6_NR, hy4rt::kv_fmt_name(e.o.kv_fmt), e.o.mx8_dense ? 1 : 0, (S == 1 && e.has_dfly) ? 1 : 0,
                         bench_kv_pos, gen, S, dec_ms, tps, tps * S);
            if (st.rounds > 0) {
                std::fprintf(stderr, " | dfly: %.2f tok/round (acc=%.1f%%, %d/%d)",
                             double(st.total_emitted) / st.rounds, 100.0 * double(st.total_accepted) / std::max(1, st.total_drafted),
                             st.total_accepted, st.total_drafted);
            }
            std::fprintf(stderr, "\n");
            if (e.o.profile) e.prof.print(rank);
        }
        if (G6_NR > 1) wire.fab.stop_proxy();
        return 0;
    }

    // Standalone cluster smoke / bench mode
    std::vector<int> prompt;
    if (bench_ctx > 0) {
        prompt.resize(size_t(bench_ctx));
        prompt[0] = geom::TOK_BOS;
        for (int i = 1; i < bench_ctx; ++i) prompt[size_t(i)] = 100 + ((i * 131 + 17) % 30000);
    } else if (!tok_str.empty()) {
        std::string raw_toks = tok_str;
        for (char& ch : raw_toks) if (ch == ',') ch = ' ';
        std::stringstream ss(raw_toks);
        int tid = 0;
        while (ss >> tid) prompt.push_back(tid);
    } else {
        prompt.push_back(geom::TOK_BOS);
    }
    const int P = static_cast<int>(prompt.size());
    const int S = std::max(1, s2_sessions);
    std::vector<hy4rt::RowOut> out;
    std::vector<int> next(size_t(S), 0);
    std::vector<int> last_pf_snap_off(size_t(S), 0);
    std::vector<std::vector<int>> said(static_cast<size_t>(S));
    const float inv_temp = temp > 0.f ? 1.f / temp : 0.f;

    const auto t_pf0 = std::chrono::steady_clock::now();
    for (int s = 0; s < S; ++s) {
        e.reset_slot(s);
        for (int p = 0; p < P; p += prefill) {
            const int n = std::min(prefill, P - p);
            const bool last = (p + n == P);
            std::vector<hy4::StepSeg> segs = {{s, 0, p, n}};
            std::vector<hy4rt::RowSample> sm(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) sm[size_t(i)] = hy4rt::RowSample{inv_temp, top_p, seed, uint32_t(p + i + 1)};
            e.forward(segs, &prompt[size_t(p)], sm.data(), out, !last);
            if (last) {
                next[size_t(s)] = out[size_t(n - 1)].tok;
                last_pf_snap_off[size_t(s)] = e.snap_off_for_seg_len(n);
            }
        }
    }
    HY4_CK(cudaStreamSynchronize(e.stream));
    const auto t_pf1 = std::chrono::steady_clock::now();

    const auto t_dec0 = std::chrono::steady_clock::now();
    hy4rt::SpecStats st{};
    if (gen > 0) {
        if (S == 1 && e.has_dfly) {
            hy4rt::run_dfly_gen_s1(e, 0, P, next[0], last_pf_snap_off[0], gen, inv_temp, top_p, seed, said[0], st);
        } else {
            for (int s = 0; s < S; ++s) said[size_t(s)].push_back(next[size_t(s)]);
            for (int g = 1; g < gen; ++g) {
                std::vector<hy4::StepSeg> segs;
                std::vector<hy4rt::RowSample> sm;
                for (int s = 0; s < S; ++s) {
                    segs.push_back({s, s, P + g - 1, 1});
                    sm.push_back(hy4rt::RowSample{inv_temp, top_p, seed, uint32_t(P + g)});
                }
                e.forward(segs, next.data(), sm.data(), out, false);
                for (int s = 0; s < S; ++s) {
                    next[size_t(s)] = out[size_t(s)].tok;
                    said[size_t(s)].push_back(next[size_t(s)]);
                }
            }
        }
    }
    HY4_CK(cudaStreamSynchronize(e.stream));
    const auto t_dec1 = std::chrono::steady_clock::now();

    if (rank == 0) {
        const double pf_ms  = std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count();
        const double dec_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
        const double tps = gen > 1 ? ((gen - 1) / (dec_ms / 1e3 + 1e-9)) : 0.0;
        std::fprintf(stderr, "# g6m4_hy4 TP=%d kv=%s mx8=%d dfly=%d P=%d gen=%d sess=%d: prefill=%.1f ms (%.1f tok/s), decode=%.1f ms (%.2f tok/s/sess, %.2f agg tok/s)",
                     G6_NR, hy4rt::kv_fmt_name(e.o.kv_fmt), e.o.mx8_dense ? 1 : 0, (S == 1 && e.has_dfly) ? 1 : 0,
                     P, gen, S, pf_ms, (double(P) * S) / (pf_ms / 1e3 + 1e-9), dec_ms, tps, tps * S);
        if (st.rounds > 0) {
            std::fprintf(stderr, " | dfly: %.2f tok/round (acc=%.1f%%, %d/%d)",
                         double(st.total_emitted) / st.rounds, 100.0 * double(st.total_accepted) / std::max(1, st.total_drafted),
                         st.total_accepted, st.total_drafted);
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
