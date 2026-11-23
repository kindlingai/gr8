// m4sim_qw38: Qwen3.8-Flash-Next with all NR ranks of a TP width (1..4) simulated on one GPU.
// Partial sums from all ranks meet in rank order (0..nr-1), matching multi-box g6m4_qw38 byte-for-byte.
//
//   m4sim_qw38 --tp 1|2|3|4 --packs DIR [--widths FILE] [--layout FILE] [--layers all|0,1,3]
//              [--tokens 248044,785,...] [--gen 16] [--sessions 1] [--chunk 64]
//              [--temp 0 --top-p 1 --seed 1234] [--top-p-cands 256] [--cap 4096]
#include "qw38_step.cuh"

#ifndef G6_NR
#define G6_NR 1
#endif

int main(int argc, char** argv) {
    using namespace qw38rt;
    std::string dir, widths, layout, layers = "all", tokens;
    int gen = 16, sessions = 1, chunk = 64, nc = 256, cap = 4096, mtp = 0;
    float temp = 0.f, top_p = 1.f;
    uint32_t seed = 1234;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--packs" || a == "--model" || a == "--v2-dense") dir = val();
        else if (a == "--widths") widths = val();
        else if (a == "--layout") layout = val();
        else if (a == "--layers") layers = val();
        else if (a == "--tokens") tokens = val();
        else if (a == "--gen") gen = std::atoi(val().c_str());
        else if (a == "--sessions" || a == "--s2-sessions") sessions = std::atoi(val().c_str());
        else if (a == "--chunk" || a == "--prefill") chunk = std::atoi(val().c_str());
        else if (a == "--top-p-cands") nc = std::atoi(val().c_str());
        else if (a == "--cap") cap = std::atoi(val().c_str());
        else if (a == "--temp") temp = static_cast<float>(std::atof(val().c_str()));
        else if (a == "--top-p") top_p = static_cast<float>(std::atof(val().c_str()));
        else if (a == "--seed") seed = static_cast<uint32_t>(std::strtoul(val().c_str(), nullptr, 10));
        else if (a == "--mtp") mtp = std::atoi(val().c_str());
        else if (a == "--graph" || a == "--no-graph") {}
        else die("unknown option " + a);
    }
    if (nc < 0 || nc > 2048 / G6_NR) die("--top-p-cands must be 0.." + std::to_string(2048 / G6_NR));
    if (dir.empty()) die("name the release directory: --packs DIR");
    if (sessions < 1 || sessions > 8 || chunk < 1 || chunk > 1024 || gen < 0 || cap < 64)
        die("--sessions 1..8, --chunk 1..1024, --gen >= 0, --cap >= 64");
    knob::init();

    Options o;
    o.nr = G6_NR;
    o.rank0 = 0;
    o.nloc = G6_NR;
    o.slots = sessions;
    o.max_rows = std::max(chunk, sessions * (mtp + 1));
    o.cap = cap;
    o.nc = std::max(nc, 1);
    o.mtp = mtp;
    o.layers = layers;
    o.widths_path = widths;
    o.layout_path = layout;

    Engine e;
    e.init(dir, o);

    std::vector<int> prompt;
    if (!tokens.empty()) {
        std::stringstream ss(tokens);
        std::string item;
        while (std::getline(ss, item, ',')) prompt.push_back(std::atoi(item.c_str()));
    }
    if (prompt.empty()) prompt.push_back(geom::TOK_BOS);
    for (int t : prompt)
        if (t < 0 || t >= geom::VOCAB) die("--tokens: invalid token id " + std::to_string(t));

    const int P = static_cast<int>(prompt.size());
    if (P + gen > cap) die("prompt + --gen exceeds --cap");

    std::vector<RowOut> out;
    std::vector<int> next(static_cast<size_t>(sessions), 0);
    std::vector<std::vector<int>> said(static_cast<size_t>(sessions));
    const float inv_temp = temp > 0.f ? 1.f / temp : 0.f;

    const auto t_pf0 = std::chrono::steady_clock::now();
    for (int s = 0; s < sessions; ++s) {
        e.reset_slot(s);
        for (int p = 0; p < P; p += chunk) {
            const int n = std::min(chunk, P - p);
            const bool last_chunk = (p + n == P);
            std::vector<qw38::StepSeg> segs = {{s, 0, p, n}};
            std::vector<RowSample> sm(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) sm[size_t(i)] = RowSample{inv_temp, top_p, seed, uint32_t(p + i + 1)};
            e.forward(segs, &prompt[size_t(p)], sm.data(), out, !last_chunk);
            if (e.has_mtp) {
                const int sync_n = std::min(n, P - 1 - p);
                if (sync_n > 0) e.mtp_sync_kv(s, 0, p, sync_n, &prompt[size_t(p + 1)]);
                if (last_chunk) e.mtp_save_last_hyper(s, n - 1);
            }
            if (last_chunk) next[size_t(s)] = out[size_t(n - 1)].tok;
        }
    }
    QW38_CK(cudaStreamSynchronize(e.stream));
    const auto t_pf1 = std::chrono::steady_clock::now();

    const auto t_dec0 = std::chrono::steady_clock::now();
    int total_drafted = 0, total_accepted = 0, mtp_rounds = 0;
    if (gen > 0) {
        for (int s = 0; s < sessions; ++s) said[size_t(s)].push_back(next[size_t(s)]);
        if (e.has_mtp && mtp > 0) {
            std::vector<int> pos(size_t(sessions), P);
            auto all_done = [&]() {
                for (int s = 0; s < sessions; ++s)
                    if (static_cast<int>(said[size_t(s)].size()) < gen) return false;
                return true;
            };
            while (!all_done()) {
                e.mtp_round(sessions, mtp, next, pos, inv_temp, top_p, seed, said, total_drafted, total_accepted);
                ++mtp_rounds;
            }
            for (int s = 0; s < sessions; ++s)
                if (static_cast<int>(said[size_t(s)].size()) > gen) said[size_t(s)].resize(size_t(gen));
        } else {
            for (int g = 1; g < gen; ++g) {
                std::vector<qw38::StepSeg> segs;
                std::vector<int> tk;
                std::vector<RowSample> sm;
                for (int s = 0; s < sessions; ++s) {
                    segs.push_back({s, s, P + g - 1, 1});
                    tk.push_back(next[size_t(s)]);
                    sm.push_back(RowSample{inv_temp, top_p, seed, uint32_t(P + g)});
                }
                e.forward(segs, tk.data(), sm.data(), out, false);
                for (int s = 0; s < sessions; ++s) {
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
    const double tps = gen > 1 ? ((gen - 1) / (dec_ms / 1e3 + 1e-9)) : 0.0;
    std::fprintf(stderr, "# m4sim_qw38 TP=%d P=%d gen=%d sess=%d mtp=%d: prefill=%.1f ms (%.1f tok/s), decode=%.1f ms (%.2f tok/s/sess, %.2f agg tok/s)",
                 G6_NR, P, gen, sessions, mtp, pf_ms, (P * sessions) / (pf_ms / 1e3 + 1e-9), dec_ms, tps, tps * sessions);
    if (mtp_rounds > 0) {
        const double acc_rate = total_drafted > 0 ? (100.0 * total_accepted / total_drafted) : 0.0;
        const double tok_per_round = double(sessions * mtp_rounds + total_accepted) / double(sessions * mtp_rounds);
        std::fprintf(stderr, " [mtp acc=%.1f%%, %.2f tok/round, %d rounds]", acc_rate, tok_per_round, mtp_rounds);
    }
    std::fprintf(stderr, "\n");
    if (e.o.profile) e.prof.print(0);

    for (int s = 0; s < sessions; ++s) {
        std::printf("session %d:", s);
        for (int t : said[size_t(s)]) std::printf(" %d", t);
        std::printf("\n");
    }
    return 0;
}
