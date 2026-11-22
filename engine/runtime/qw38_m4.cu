// g6m4_qw38: one rank of Qwen3.8-Flash-Next on the real RDMA fabric (TP = 1, 2, 3, 4).
// Supports both standalone cluster smoke runs (--tokens ... --gen ...) and S2 multi-session serving (--serve DIR).
// Uses device-driven collectives (ink2::ar_exchange, dual-rail striped when --ips2 is given) so multi-box TP=2..4
// has minimal collective latency and stays byte-identical to m4sim_qw38.
#include "qw38_step.cuh"
#include "../comms/fabric.h"
#include "../kernels/ink2_fabric.cuh"
#include "s2/qw38_native.cuh"
#include <chrono>

#ifndef G6_NR
#define G6_NR 2
#endif

namespace {
using qw38rt::die;

struct Wire {
    g6::Fabric fab;
    g6::FabricDev f{};
    unsigned long long step = 0;
    int rank = 0, nr = 1;
    bool fast = true;

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

    void exchange_dev(int, __nv_bfloat16* parts, size_t bytes) {
        const int par = static_cast<int>(step & 1);
        const size_t wire = (bytes + 15) / 16 * 16;
        fab.enqueue_step(step, {static_cast<uint32_t>(wire)});
        if (fast) {
            ink2::ar_exchange(f, step, 0, reinterpret_cast<char*>(parts), wire, 0);
            QW38_CK(cudaStreamSynchronize(0));
            if (!fab.wait_proxy_idle(60.0)) die("fabric proxy timeout");
            if (*f.error) die("fabric device error " + std::to_string(*f.error));
        } else {
            char* d = reinterpret_cast<char*>(parts);
            QW38_CK(cudaMemcpy(f.send + g6::fab_send_off(f, par, 0), d + size_t(rank) * bytes, bytes, cudaMemcpyDeviceToHost));
            publish();
            wait_host(wire);
            if (!fab.wait_proxy_idle(60.0)) die("fabric proxy timeout");
            for (int p = 0; p < nr; ++p)
                if (p != rank) QW38_CK(cudaMemcpy(d + size_t(p) * bytes, f.recv + g6::fab_recv_off(f, par, 0, p), bytes, cudaMemcpyHostToDevice));
        }
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
        else if (a == "--s2-sessions") s2sess = std::atoi(val().c_str());
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
    o.slots = s2dir.empty() ? 1 : s2sess;
    o.max_rows = std::max(prefill, std::max(chunk, o.slots));
    o.cap = cap;
    o.nc = std::max(nc, 1);
    o.mtp = mtp;
    o.layers = layers;
    o.widths_path = widths;
    o.layout_path = layout;

    Engine e;
    e.init(dir, o);

    Wire wire;
    if (G6_NR > 1) {
        const size_t max_part_b = (size_t(o.max_rows) * geom::HID * 2 + 15) / 16 * 16;
        const size_t max_head_b = (size_t(o.max_rows) * head_row_bytes(o.nc) + 15) / 16 * 16;
        const size_t slot_b = std::max(max_part_b, max_head_b);
        g6::FabricConfig cfg;
        cfg.rank = rank;
        cfg.ips = peers;
        if (!ips2.empty()) cfg.ips2 = split_csv(ips2);
        cfg.proxy_cpu = proxy_cpu;
        cfg.ncoll = 1;
        cfg.stripe_min = cfg.ips2.empty() ? 0 : 65536;
        cfg.slot_bytes = (slot_b + 7) / 8 * 8;
        wire.fab.init(cfg);
        wire.f = wire.fab.dev();
        wire.fab.start_proxy();
        wire.fab.barrier();
        wire.rank = rank;
        wire.nr = G6_NR;
        if (const char* fe = std::getenv("GR8_QW38_FAST_AR")) wire.fast = (std::atoi(fe) != 0);
        e.gather_dev = [&](int c, __nv_bfloat16* p, size_t b) { wire.exchange_dev(c, p, b); };
        e.gather_host = [&](int c, uint8_t* p, size_t b) { wire.exchange_host(c, p, b); };
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
                               "|set " + set_id + "|" + knob::signature();

        g6s2::Qw38Context ctx{e, rank, reuse, kv_tiers, kv_keep, kv_sys_after, kv_wlim, set_id};
        g6s2::serve_qw38(ctx, bc, chunk);
        if (G6_NR > 1) wire.fab.stop_proxy();
        return 0;
    }

    // Standalone cluster smoke run
    std::vector<int> prompt;
    if (!tokens.empty()) {
        std::stringstream ss(tokens);
        std::string item;
        while (std::getline(ss, item, ',')) prompt.push_back(std::atoi(item.c_str()));
    }
    if (prompt.empty()) prompt.push_back(geom::TOK_BOS);
    const int P = static_cast<int>(prompt.size());
    const float inv_temp = temp > 0.f ? 1.f / temp : 0.f;
    std::vector<RowOut> out;
    std::vector<int> said;
    int next = 0;

    e.reset_slot(0);
    const auto t_pf0 = std::chrono::steady_clock::now();
    for (int p = 0; p < P; p += prefill) {
        const int n = std::min(prefill, P - p);
        const bool last = (p + n == P);
        std::vector<qw38::StepSeg> segs = {{0, 0, p, n}};
        std::vector<RowSample> sm(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) sm[size_t(i)] = RowSample{inv_temp, top_p, seed, uint32_t(p + i + 1)};
        e.forward(segs, &prompt[size_t(p)], sm.data(), out, !last);
        if (last) next = out[size_t(n - 1)].tok;
    }
    QW38_CK(cudaStreamSynchronize(e.stream));
    const auto t_pf1 = std::chrono::steady_clock::now();

    const auto t_dec0 = std::chrono::steady_clock::now();
    if (gen > 0) {
        said.push_back(next);
        for (int g = 1; g < gen; ++g) {
            std::vector<qw38::StepSeg> segs = {{0, 0, P + g - 1, 1}};
            RowSample sm{inv_temp, top_p, seed, uint32_t(P + g)};
            e.forward(segs, &next, &sm, out, false);
            next = out[0].tok;
            said.push_back(next);
        }
    }
    QW38_CK(cudaStreamSynchronize(e.stream));
    const auto t_dec1 = std::chrono::steady_clock::now();

    const double pf_ms = std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count();
    const double dec_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
    if (rank == 0) {
        std::fprintf(stderr, "# g6m4_qw38 TP=%d P=%d gen=%d: prefill=%.1f ms (%.1f tok/s), decode=%.1f ms (%.2f tok/s)\n",
                     G6_NR, P, gen, pf_ms, P / (pf_ms / 1e3 + 1e-9), dec_ms,
                     gen > 1 ? ((gen - 1) / (dec_ms / 1e3 + 1e-9)) : 0.0);
        if (o.profile) e.prof.print(rank);
        std::printf("session 0:");
        for (int t : said) std::printf(" %d", t);
        std::printf("\n");
    }
    return 0;
}
