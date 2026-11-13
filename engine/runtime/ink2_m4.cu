// g6m4_ink2: one rank of Inkling-2-Small (ink2_step.cuh). The checkpoint (a directory with config.json and its
// safetensors files, or pre-rendered tp<nr>.rank<rank>.{manifest.json,pack}) is planned and loaded in place by the
// Rust loader; the rank's partial sums cross the RDMA fabric (comms/fabric.h) and are summed in rank order, as
// m4sim_ink2 (ink2_sim.cu) sums them on one GPU.
//
//   g6m4_ink2 --rank R --ips a,b,.. [--ips2 a,b,..] [--proxy-cpu N] --v2-dense DIR [--layers all|0,5,..]
//             [--top-p-cands 256] [--chunk 8] [--prefill 512] [--cap 262144] [--graph|--no-graph] [--mtp 0]
//             [--reuse-prefix] [--kv-store SPEC] [--kv-keep S] [--kv-system-after N] [--kv-write-limit F]
//             --serve DIR --s2-port PORT --s2-sessions N --s2-epoch E [--set-id ID]
//   without --serve: --tokens 1,2,3 --gen 16 [--temp 0 --top-p 1 --seed 1234] runs one session, the same on every
//   rank, and prints its tokens (a cluster check without the service).
//
// Each collective runs as one fabric step on a single-slot alternating-parity fabric (dual-rail striped when --ips2 is
// given), so both 2*L-collective prefill steps (head skipped) and (2*L + 1)-collective decode steps share one slot.
#include "ink2_step.cuh"
#include "../comms/fabric.h"
#include "s2/ink2_native.cuh"
#include <chrono>

#ifndef G6_NR
#define G6_NR 6
#endif

namespace
{
using ink2rt::die;

struct Wire
{
    g6::Fabric fab;
    g6::FabricDev f{};
    unsigned long long step = 0;
    int rank = 0, nr = 1;

    void publish() { __atomic_store_n(f.doorbell, g6::coll_seq(step, 0, f.ncoll), __ATOMIC_RELEASE); }
    void wait(size_t bytes)
    {
        const unsigned long long seq = g6::coll_seq(step, 0, f.ncoll);
        const int par = static_cast<int>(step & 1);
        const auto t0 = std::chrono::steady_clock::now();
        for (int p = 0; p < nr; ++p)
        {
            if (p == rank) continue;
            const unsigned long long* tail = reinterpret_cast<const unsigned long long*>(f.recv + g6::fab_recv_off(f, par, 0, p) + bytes);
            unsigned it = 0;
            while (__atomic_load_n(tail, __ATOMIC_ACQUIRE) != seq)
            {
                if ((++it & 0xFFFFu) == 0 && (*f.error || std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)))
                    die("rank " + std::to_string(rank) + ": fabric: no share from rank " + std::to_string(p) + " at exchange " + std::to_string(step) +
                        " (error word " + std::to_string(*f.error) + ")");
            }
        }
    }
    void exchange_host(int, uint8_t* parts, size_t bytes)
    {
        const size_t wire = (bytes + 7) / 8 * 8;
        const int par = static_cast<int>(step & 1);
        fab.enqueue_step(step, {static_cast<uint32_t>(wire)});
        char* dst = f.send + g6::fab_send_off(f, par, 0);
        memcpy(dst, parts + static_cast<size_t>(rank) * bytes, bytes);
        if (wire > bytes) memset(dst + bytes, 0, wire - bytes);
        publish();
        wait(wire);
        if (!fab.wait_proxy_idle(60.0)) die("rank " + std::to_string(rank) + ": fabric: proxy did not finish exchange " + std::to_string(step));
        for (int p = 0; p < nr; ++p)
            if (p != rank) memcpy(parts + static_cast<size_t>(p) * bytes, f.recv + g6::fab_recv_off(f, par, 0, p), bytes);
        ++step;
    }
    void exchange_dev(int, float* parts, size_t bytes)
    {
        const int par = static_cast<int>(step & 1);
        fab.enqueue_step(step, {static_cast<uint32_t>(bytes)});
        char* d = reinterpret_cast<char*>(parts);
        INK2RT_CK(cudaMemcpy(f.send + g6::fab_send_off(f, par, 0), d + static_cast<size_t>(rank) * bytes, bytes, cudaMemcpyDeviceToHost));
        publish();
        wait(bytes);
        if (!fab.wait_proxy_idle(60.0)) die("rank " + std::to_string(rank) + ": fabric: proxy did not finish exchange " + std::to_string(step));
        for (int p = 0; p < nr; ++p)
            if (p != rank) INK2RT_CK(cudaMemcpy(d + static_cast<size_t>(p) * bytes, f.recv + g6::fab_recv_off(f, par, 0, p), bytes, cudaMemcpyHostToDevice));
        ++step;
    }
};

std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) v.push_back(item);
    return v;
}
}   // namespace

int main(int argc, char** argv)
{
    using namespace ink2rt;
    std::string dir, layers = "all", tokens, s2dir, set_id, ips, ips2, kv_keep = "all";
    std::vector<std::string> kv_tiers;
    int rank = -1, proxy_cpu = -1, gen = 16, chunk = 8, prefill = 512, nc = 256, cap = 262144, mtp = 0, s2port = 7341, s2sess = 1, s2epoch = 0, kv_sys_after = 2;
    double kv_wlim = 0.0;
    bool reuse = false;
    float temp = 0.f, top_p = 1.f;
    uint32_t seed = 1234;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto val = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--rank") rank = atoi(val().c_str());
        else if (a == "--ips") ips = val();
        else if (a == "--ips2") ips2 = val();
        else if (a == "--proxy-cpu") proxy_cpu = atoi(val().c_str());
        else if (a == "--v2-dense" || a == "--model") dir = val();
        else if (a == "--manifest" || a == "--pack") val();   // a pack build's flags: the checkpoint is the model here
        else if (a == "--layers") layers = val();
        else if (a == "--tokens") tokens = val();
        else if (a == "--gen") gen = atoi(val().c_str());
        else if (a == "--chunk") chunk = atoi(val().c_str());
        else if (a == "--prefill") prefill = atoi(val().c_str());
        else if (a == "--top-p-cands") nc = atoi(val().c_str());
        else if (a == "--cap") cap = atoi(val().c_str());
        else if (a == "--temp") temp = static_cast<float>(atof(val().c_str()));   // the run without --serve; a served request carries its own
        else if (a == "--top-p") top_p = static_cast<float>(atof(val().c_str()));
        else if (a == "--seed") seed = static_cast<uint32_t>(strtoul(val().c_str(), nullptr, 10));
        else if (a == "--mtp") mtp = atoi(val().c_str());
        else if (a == "--serve") s2dir = val();
        else if (a == "--s2-port") s2port = atoi(val().c_str());
        else if (a == "--s2-sessions") s2sess = atoi(val().c_str());
        else if (a == "--s2-epoch") s2epoch = atoi(val().c_str());
        else if (a == "--set-id") set_id = val();
        else if (a == "--graph" || a == "--no-graph") {}   // the step is eager
        else if (a == "--reuse-prefix") reuse = true;
        else if (a == "--kv-store") kv_tiers.push_back(val());
        else if (a == "--kv-keep") kv_keep = val();
        else if (a == "--kv-system-after") kv_sys_after = atoi(val().c_str());
        else if (a == "--kv-write-limit") kv_wlim = atof(val().c_str());
        else die("unknown option " + a);
    }
    if (!kv_tiers.empty()) reuse = true;
    const std::vector<std::string> ip = split(ips), ip2 = split(ips2);
    const int nr = static_cast<int>(ip.size());
    if (nr != G6_NR) die("--ips names " + std::to_string(nr) + " ranks; this instance is TP " + std::to_string(G6_NR));
    if (nc < 0 || nc > 2048 / G6_NR) die("--top-p-cands must be 0.." + std::to_string(2048 / G6_NR));
    if (rank < 0 || rank >= nr) die("--rank must be 0.." + std::to_string(nr - 1));
    if (dir.empty()) die("name the checkpoint: --v2-dense DIR");
    if (mtp != 0) die("--mtp " + std::to_string(mtp) + ": Inkling-2-Small serves at MTP 0");
    const int max_chunk = s2dir.empty() ? 256 : 8;
    if (s2sess < 1 || s2sess > 8 || chunk < 1 || chunk > max_chunk || prefill < 1 || prefill > 512 || cap < 64)
        die("--s2-sessions 1..8, --chunk 1.." + std::to_string(max_chunk) + ", --prefill 1..512, --cap >= 64");
    if (kv_keep != "system" && kv_keep != "sessions" && kv_keep != "all") die("--kv-keep must be system, sessions or all");
    if (kv_sys_after < 1 || kv_wlim < 0.0) die("--kv-system-after >= 1, --kv-write-limit >= 0");
    if (!ip2.empty() && ip2.size() != ip.size()) die("--ips2 must name as many addresses as --ips");
    knob::init();
    INK2RT_CK(cudaSetDeviceFlags(cudaDeviceMapHost));

    Options o;
    o.nr = nr;
    o.rank0 = rank;
    o.nloc = 1;
    o.slots = s2dir.empty() ? 1 : s2sess;
    o.max_rows = std::max(s2dir.empty() ? chunk : std::max(prefill, chunk), o.slots);
    o.cap = cap / 64 * 64;
    o.nc = std::max(nc, 1);
    o.layers = layers;
    o.whole_image = true;
    Engine e;
    e.init(dir, o);

    Wire w;
    w.rank = rank;
    w.nr = nr;
    if (nr > 1)
    {
        g6::FabricConfig cfg;
        cfg.rank = rank;
        cfg.ips = ip;
        cfg.ips2 = ip2;
        cfg.proxy_cpu = proxy_cpu;
        cfg.ncoll = 1;
        cfg.stripe_min = ip2.empty() ? 0 : 65536;
        cfg.slot_bytes = (e.max_bytes() + 7) / 8 * 8;
        w.fab.init(cfg);
        w.f = w.fab.dev();
        w.fab.start_proxy();
        w.fab.barrier();
        fprintf(stderr, "# rank %d: fabric of %d slot(s), %zu bytes, stripe_min %zu\n", rank, cfg.ncoll, cfg.slot_bytes, cfg.stripe_min);
        e.gather = [&w](int c, float* parts, size_t bytes) { w.exchange_dev(c, parts, bytes); };
        e.gather_head = [&w](int c, uint8_t* parts, size_t bytes) { w.exchange_host(c, parts, bytes); };
    }
    const auto step = [&](const ink2::RowsHost& rh, const int* tk, const RowSample* sm, std::vector<RowOut>& out) {
        e.forward(rh, tk, sm, out);
        e.commit();
    };

    if (!s2dir.empty())
    {
        g6s2::BootstrapConfig bc;
        auto& sc = bc.scheduler;
        sc.epoch = static_cast<uint64_t>(s2epoch);
        sc.pool_rows = static_cast<uint32_t>(s2sess) * static_cast<uint32_t>(o.cap);
        sc.max_rows = static_cast<uint32_t>(std::max(chunk, s2sess));
        sc.max_sessions = static_cast<uint32_t>(s2sess);
        sc.mtp = 0;
        sc.two_lanes = false;
        bc.rank = static_cast<unsigned>(rank);
        bc.ranks = static_cast<unsigned>(nr);
        bc.head_ip = ip[0];
        bc.directory = s2dir;
        bc.port = static_cast<uint16_t>(s2port);
        bc.timeout = static_cast<unsigned>(knob::s2_timeout());
        bc.runtime_signature = "ink2|tp" + std::to_string(nr) + "|cap" + std::to_string(o.cap) + "|sess" + std::to_string(s2sess) + "|layers " + layers + "|chunk" + std::to_string(chunk) +
                               "|pf" + std::to_string(prefill) + "|reuse" + std::to_string(reuse ? 1 : 0) + "|nc" + std::to_string(o.nc) + "|set " + set_id + "|" + knob::signature();
        fprintf(stderr, "# rank %d: s2 pool %u logical pages of %d rows, %d sessions, cap %d, max_rows %u, prefill %d\n",
                rank, sc.pool_rows / 64, 64, s2sess, o.cap, sc.max_rows, o.max_rows);
        try
        {
            g6s2::serve_ink2(g6s2::Ink2Context{e, step, rank, reuse, kv_tiers, kv_keep, kv_sys_after, kv_wlim, set_id}, bc, chunk);
        }
        catch (const std::exception& x)
        {
            die(x.what());
        }
        if (nr > 1) w.fab.stop_proxy();
        return 0;
    }

    std::vector<int> prompt;
    for (const std::string& t : split(tokens)) prompt.push_back(atoi(t.c_str()));
    if (prompt.empty()) prompt.push_back(geom::TOK_BOS < e.c.vocab ? static_cast<int>(geom::TOK_BOS) : 1);
    const int P = static_cast<int>(prompt.size());
    if (P + gen > o.cap) die("the prompt and --gen pass --cap");
    const float inv_temp = temp > 0.f ? 1.f / temp : 0.f;
    std::vector<RowOut> out;
    std::vector<int> said;
    int next = 0;
    e.reset_slot(0);
    for (int p = 0; p < P; p += chunk)
    {
        const int n = std::min(chunk, P - p);
        ink2::RowsHost rh;
        rh.add(0, p, n);
        const bool last = p + n == P;
        std::vector<RowSample> sm;
        if (last)
        {
            sm.resize(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) sm[i] = RowSample{inv_temp, top_p, seed, static_cast<uint32_t>(p + i + 1)};
        }
        step(rh, &prompt[static_cast<size_t>(p)], last ? sm.data() : nullptr, out);
        if (last) next = out[static_cast<size_t>(n) - 1].tok;
    }
    for (int g = 0; g < gen; ++g)
    {
        said.push_back(next);
        if (g + 1 == gen) break;
        ink2::RowsHost rh;
        rh.add(0, P + g, 1);
        const RowSample sm{inv_temp, top_p, seed, static_cast<uint32_t>(P + g + 1)};
        step(rh, &next, &sm, out);
        next = out[0].tok;
    }
    printf("rank %d tokens:", rank);
    for (int t : said) printf(" %d", t);
    printf("\n");
    if (nr > 1) w.fab.stop_proxy();
    return 0;
}
