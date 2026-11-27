// m4sim_ernie55: ERNIE-5.5-Flash-Preview single-GPU simulator (TP=1 resident, TP=1 hybrid 5090 offload, or simulated TP=2).
// Supports:
//   1. Prompt + generation + per-layer parity dump (--packs DIR --widths JSON --tokens ... --gen N --dump DIR)
//      with --hybrid / --no-hybrid, --kv-format bf16|fp8|int4, --mx8 / --no-mx8, --graph / --no-graph
//   2. Fast exact KLD/NLL evaluation over held-out numpy token rows (--eval-npy PATH --eval-start 977 --eval-count 64
//      --eval-out PATH --eval-ref-logp PATH [--eval-dump-full])
//   3. Long-context & multi-session concurrency benchmark (--bench-ctx N / --bench-kv-pos N --sessions S --gen G)
#include "ernie55_step.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef G6_NR
#define G6_NR 1
#endif

namespace {
using ernie55rt::die;

void read_npy_2d_i32(const std::string& path, std::vector<int32_t>& data, int& rows, int& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open npy: " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) die("not a npy file: " + path);
    uint8_t ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) { uint16_t h16; f.read(reinterpret_cast<char*>(&h16), 2); hlen = h16; }
    else             { f.read(reinterpret_cast<char*>(&hlen), 4); }
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), hlen);
    const bool is_i64 = (hdr.find("<i8") != std::string::npos || hdr.find("<u8") != std::string::npos);
    const bool is_i32 = (hdr.find("<i4") != std::string::npos || hdr.find("<u4") != std::string::npos);
    if (!is_i64 && !is_i32) die("unsupported npy dtype in " + hdr);
    const size_t p0 = hdr.find('(');
    const size_t p1 = hdr.find(')', p0);
    if (p0 == std::string::npos || p1 == std::string::npos) die("cannot parse shape in " + hdr);
    std::string shp = hdr.substr(p0 + 1, p1 - p0 - 1);
    const size_t cm = shp.find(',');
    rows = std::stoi(shp.substr(0, cm));
    cols = std::stoi(shp.substr(cm + 1));
    const size_t n = size_t(rows) * size_t(cols);
    data.resize(n);
    if (is_i32) {
        f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * 4));
    } else {
        std::vector<int64_t> tmp(n);
        f.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(n * 8));
        for (size_t i = 0; i < n; ++i) data[i] = static_cast<int32_t>(tmp[i]);
    }
}

void eval_heldout(ernie55rt::Engine& e, const std::string& npy_path, int start_row, int n_rows, int chunk,
                  const std::string& dump_top_path, const std::string& ref_logp_path, bool dump_full) {
    std::vector<int32_t> toks_all;
    int R = 0, C = 0;
    read_npy_2d_i32(npy_path, toks_all, R, C);
    if (start_row < 0 || start_row + n_rows > R) die("eval row range out of bounds");
    std::fprintf(stderr, "# ernie55 eval: %s [%d x %d], rows [%d..%d)\n",
                 npy_path.c_str(), R, C, start_row, start_row + n_rows);

    FILE* fp = nullptr;
    if (!dump_top_path.empty()) {
        fp = std::fopen(dump_top_path.c_str(), "wb");
        if (!fp) die("cannot open --eval-out " + dump_top_path);
        const int32_t hdr[4] = {0x4535354B, n_rows, C - 1, geom::VOCAB};
        std::fwrite(hdr, sizeof(int32_t), 4, fp);
    }

    FILE* ref_fp = nullptr;
    if (!ref_logp_path.empty()) {
        ref_fp = std::fopen(ref_logp_path.c_str(), "rb");
        if (!ref_fp) die("cannot open --eval-ref-logp " + ref_logp_path);
        int32_t rh[4] = {};
        if (std::fread(rh, sizeof(int32_t), 4, ref_fp) != 4 || rh[0] != 0x45353552 ||
            rh[1] != n_rows || rh[2] != C - 1 || rh[3] != geom::VOCAB)
            die("bad header in --eval-ref-logp " + ref_logp_path);
    }
    std::vector<uint16_t> ref_buf(ref_fp ? geom::VOCAB : 0);

    double sum_nll = 0.0, sum_ref_nll = 0.0, sum_kld = 0.0;
    int64_t n_tok = 0, top1_eq = 0, top5_eq = 0;
    std::vector<ernie55rt::RowOut> out;
    std::vector<float> logits;
    const auto t0 = std::chrono::steady_clock::now();

    for (int r = 0; r < n_rows; ++r) {
        const int32_t* row = toks_all.data() + size_t(start_row + r) * C;
        e.reset_slot(0);
        for (int p = 0; p < C - 1; p += chunk) {
            const int n = std::min(chunk, (C - 1) - p);
            std::vector<ernie55::StepSeg> segs = {{0, 0, p, n}};
            e.forward(segs, row + p, nullptr, out, false, &logits);
            for (int i = 0; i < n; ++i) {
                const float* lg = logits.data() + size_t(i) * geom::VOCAB;
                float mx = -INFINITY;
                for (int v = 0; v < geom::VOCAB; ++v) if (lg[v] > mx) mx = lg[v];
                double ds = 0.0;
                for (int v = 0; v < geom::VOCAB; ++v) ds += std::exp(double(lg[v] - mx));
                const float lse = mx + static_cast<float>(std::log(ds));
                const int tgt = row[p + i + 1];
                sum_nll += double(lse - lg[tgt]);
                if (ref_fp) {
                    if (std::fread(ref_buf.data(), 2, geom::VOCAB, ref_fp) != size_t(geom::VOCAB))
                        die("short read on --eval-ref-logp");
                    double kl = 0.0;
                    float ref_mx = -INFINITY;
                    int ref_top1 = 0;
                    for (int v = 0; v < geom::VOCAB; ++v) {
                        uint32_t u = uint32_t(ref_buf[size_t(v)]) << 16;
                        float lpr;
                        std::memcpy(&lpr, &u, 4);
                        if (lpr > ref_mx) { ref_mx = lpr; ref_top1 = v; }
                        const double pr = std::exp(double(lpr));
                        const double lpq = double(lg[v] - lse);
                        kl += pr * (double(lpr) - lpq);
                    }
                    uint32_t ut = uint32_t(ref_buf[size_t(tgt)]) << 16;
                    float lpr_t; std::memcpy(&lpr_t, &ut, 4);
                    sum_ref_nll += -double(lpr_t);
                    sum_kld += kl;
                    if (out[size_t(i)].tok == ref_top1) ++top1_eq;
                    for (size_t kk = 0; kk < std::min<size_t>(5, out[size_t(i)].top.size()); ++kk) {
                        if (out[size_t(i)].top[kk].id == ref_top1) { ++top5_eq; break; }
                    }
                }
                ++n_tok;
                if (fp) {
                    if (dump_full) {
                        std::vector<uint16_t> lp16(geom::VOCAB);
                        for (int v = 0; v < geom::VOCAB; ++v) {
                            const __nv_bfloat16 b = __float2bfloat16(lg[v] - lse);
                            std::memcpy(&lp16[size_t(v)], &b, 2);
                        }
                        std::fwrite(lp16.data(), 2, geom::VOCAB, fp);
                    } else {
                        const float rec[3] = {lse, lg[tgt] - lse, lg[out[size_t(i)].tok] - lse};
                        const int32_t top1 = out[size_t(i)].tok;
                        std::fwrite(rec, 4, 3, fp);
                        std::fwrite(&top1, 4, 1, fp);
                    }
                }
            }
        }
        if ((r + 1) % 8 == 0 || r + 1 == n_rows) {
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "# eval row %d/%d: mean_nll=%.6f ppl=%.4f (%.1f tok/s)\n",
                         r + 1, n_rows, sum_nll / double(n_tok), std::exp(sum_nll / double(n_tok)),
                         double(n_tok) / (sec + 1e-9));
        }
    }
    if (fp) std::fclose(fp);
    if (ref_fp) {
        std::fclose(ref_fp);
        std::printf("EVAL_SUMMARY rows=%d tokens=%lld mean_nll=%.6f ppl=%.6f ref_nll=%.6f ref_ppl=%.6f kld=%.6f top1_agree=%.4f top5_agree=%.4f\n",
                    n_rows, static_cast<long long>(n_tok), sum_nll / double(n_tok), std::exp(sum_nll / double(n_tok)),
                    sum_ref_nll / double(n_tok), std::exp(sum_ref_nll / double(n_tok)),
                    sum_kld / double(n_tok), 100.0 * double(top1_eq) / double(n_tok),
                    100.0 * double(top5_eq) / double(n_tok));
    } else {
        std::printf("EVAL_SUMMARY rows=%d tokens=%lld mean_nll=%.6f ppl=%.6f\n",
                    n_rows, static_cast<long long>(n_tok), sum_nll / double(n_tok), std::exp(sum_nll / double(n_tok)));
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string packs, widths, tok_str, dump_dir, eval_npy, eval_out, eval_ref_logp, kv_fmt_str;
    int cap = 16384, pool_rows = 0, chunk = 512, gen = 16, sessions = 1, nc = 256;
    int eval_start = 977, eval_count = 64, bench_ctx = 0, bench_kv_pos = 0;
    float temp = 0.0f, top_p = 1.0f;
    uint32_t seed = 42;
    bool profile = false, dump_logits = false, eval_dump_full = false;
    bool mx8_set = false, mx8_val = true;
    bool mx8_head_set = false, mx8_head_val = false;
    bool graphs_set = false, graphs_val = true;
    bool hybrid_set = false, hybrid_val = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* f) { if (i + 1 >= argc) die(std::string("missing value for ") + f); return std::string(argv[++i]); };
        if (a == "--packs")               packs = need("--packs");
        else if (a == "--widths")         widths = need("--widths");
        else if (a == "--kv-format")      kv_fmt_str = need("--kv-format");
        else if (a == "--mx8")            { mx8_set = true; mx8_val = true; }
        else if (a == "--no-mx8")         { mx8_set = true; mx8_val = false; }
        else if (a == "--mx8-head")       { mx8_head_set = true; mx8_head_val = true; }
        else if (a == "--no-mx8-head")    { mx8_head_set = true; mx8_head_val = false; }
        else if (a == "--graph")          { graphs_set = true; graphs_val = true; }
        else if (a == "--no-graph")       { graphs_set = true; graphs_val = false; }
        else if (a == "--hybrid")         { hybrid_set = true; hybrid_val = true; }
        else if (a == "--no-hybrid")      { hybrid_set = true; hybrid_val = false; }
        else if (a == "--tokens")         tok_str = need("--tokens");
        else if (a == "--cap")            cap = std::stoi(need("--cap"));
        else if (a == "--pool-rows")      pool_rows = std::stoi(need("--pool-rows"));
        else if (a == "--chunk")          chunk = std::stoi(need("--chunk"));
        else if (a == "--gen")            gen = std::stoi(need("--gen"));
        else if (a == "--sessions")       sessions = std::stoi(need("--sessions"));
        else if (a == "--temp")           temp = std::stof(need("--temp"));
        else if (a == "--top-p")          top_p = std::stof(need("--top-p"));
        else if (a == "--seed")           seed = static_cast<uint32_t>(std::stoul(need("--seed")));
        else if (a == "--top-p-cands")    nc = std::stoi(need("--top-p-cands"));
        else if (a == "--dump")           { dump_dir = need("--dump"); dump_logits = true; }
        else if (a == "--eval-npy")       eval_npy = need("--eval-npy");
        else if (a == "--eval-start")     eval_start = std::stoi(need("--eval-start"));
        else if (a == "--eval-count")     eval_count = std::stoi(need("--eval-count"));
        else if (a == "--eval-out")       eval_out = need("--eval-out");
        else if (a == "--eval-ref-logp")  eval_ref_logp = need("--eval-ref-logp");
        else if (a == "--eval-dump-full") eval_dump_full = true;
        else if (a == "--bench-ctx")      bench_ctx = std::stoi(need("--bench-ctx"));
        else if (a == "--bench-kv-pos")   bench_kv_pos = std::stoi(need("--bench-kv-pos"));
        else if (a == "--profile")        profile = true;
        else if (a == "--layers")         (void)need("--layers");
        else die("unknown arg: " + a);
    }
    if (nc < 0 || nc > 2048 / G6_NR)
        die("--top-p-cands must be 0.." + std::to_string(2048 / G6_NR));
    if (packs.empty()) die("need --packs <release_dir>");
    knob::init();

    ernie55rt::Options o;
    o.nr = G6_NR;
    o.rank0 = 0;
    o.nloc = G6_NR;
    o.slots = sessions;
    o.max_rows = std::max({chunk, sessions, 16});
    o.cap = cap;
    o.pool_rows = pool_rows;
    o.nc = nc;
    o.widths_path = widths;
    o.profile = profile;
    o.dump_dir = dump_dir;
    if (!kv_fmt_str.empty()) { o.kv_fmt = ernie55rt::parse_kv_fmt(kv_fmt_str); o.kv_fmt_set = true; }
    if (mx8_set)             { o.mx8_dense = mx8_val; o.mx8_set = true; }
    if (mx8_head_set)        { o.mx8_head = mx8_head_val; o.mx8_head_set = true; }
    if (graphs_set)          { o.use_graphs = graphs_val; o.graphs_set = true; }
    if (hybrid_set)          { o.use_hybrid = hybrid_val; o.hybrid_set = true; }

    ernie55rt::Engine e;
    e.init(packs, o);

    if (!eval_npy.empty()) {
        eval_heldout(e, eval_npy, eval_start, eval_count, chunk, eval_out, eval_ref_logp, eval_dump_full);
        if (e.o.profile) e.prof.print(0);
        return 0;
    }

    auto run_prompt = [&](const std::vector<int>& prompt, bool print_sessions, bool allow_dump) {
        const int P = static_cast<int>(prompt.size());
        if (P + gen + 2 > cap) die("prompt + --gen exceeds --cap");

        std::vector<ernie55rt::RowOut> out;
        std::vector<float> full_logits;
        std::vector<int> next(size_t(sessions), 0);
        std::vector<std::vector<int>> said(static_cast<size_t>(sessions));
        const float inv_temp = temp > 0.f ? 1.f / temp : 0.f;

        const auto t_pf0 = std::chrono::steady_clock::now();
        for (int s = 0; s < sessions; ++s) {
            e.reset_slot(s);
            for (int p = 0; p < P; p += chunk) {
                const int n = std::min(chunk, P - p);
                const bool last = (p + n == P);
                std::vector<ernie55::StepSeg> segs = {{s, 0, p, n}};
                std::vector<ernie55rt::RowSample> sm(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) sm[size_t(i)] = ernie55rt::RowSample{inv_temp, top_p, seed, uint32_t(p + i + 1)};
                e.forward(segs, &prompt[size_t(p)], sm.data(), out, !last, (last && allow_dump && dump_logits) ? &full_logits : nullptr);
                if (last) next[size_t(s)] = out[size_t(n - 1)].tok;
            }
        }
        ERNIE55_CK(cudaStreamSynchronize(e.stream));
        const auto t_pf1 = std::chrono::steady_clock::now();

        const auto t_dec0 = std::chrono::steady_clock::now();
        if (gen > 0) {
            for (int s = 0; s < sessions; ++s) said[size_t(s)].push_back(next[size_t(s)]);
            for (int g = 1; g < gen; ++g) {
                std::vector<ernie55::StepSeg> segs;
                std::vector<int> tk;
                std::vector<ernie55rt::RowSample> sm;
                for (int s = 0; s < sessions; ++s) {
                    segs.push_back({s, s, P + g - 1, 1});
                    tk.push_back(next[size_t(s)]);
                    sm.push_back(ernie55rt::RowSample{inv_temp, top_p, seed, uint32_t(P + g)});
                }
                e.forward(segs, tk.data(), sm.data(), out, false);
                for (int s = 0; s < sessions; ++s) {
                    next[size_t(s)] = out[size_t(s)].tok;
                    said[size_t(s)].push_back(next[size_t(s)]);
                }
            }
        }
        ERNIE55_CK(cudaStreamSynchronize(e.stream));
        const auto t_dec1 = std::chrono::steady_clock::now();

        const double pf_ms  = std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count();
        const double dec_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
        const double tps = gen > 1 ? ((gen - 1) / (dec_ms / 1e3 + 1e-9)) : 0.0;
        std::fprintf(stderr, "# m4sim_ernie55 TP=%d kv=%s mx8=%d hybrid=%d P=%d gen=%d sess=%d: prefill=%.1f ms (%.1f tok/s), decode=%.1f ms (%.2f tok/s/sess, %.2f agg tok/s)\n",
                     G6_NR, ernie55rt::kv_fmt_name(e.o.kv_fmt), e.o.mx8_dense ? 1 : 0, e.o.use_hybrid ? 1 : 0,
                     P, gen, sessions, pf_ms, (double(P) * sessions) / (pf_ms / 1e3 + 1e-9), dec_ms, tps, tps * sessions);
        if (e.o.profile) e.prof.print(0);
        if (print_sessions) {
            for (int s = 0; s < sessions; ++s) {
                std::printf("session %d:", s);
                for (int t : said[size_t(s)]) std::printf(" %d", t);
                std::printf("\n");
            }
        }
    };

    if (!tok_str.empty()) {
        std::vector<int> prompt;
        std::string raw_toks = tok_str;
        for (char& ch : raw_toks) if (ch == ',') ch = ' ';
        std::stringstream ss(raw_toks);
        int tid = 0;
        while (ss >> tid) prompt.push_back(tid);
        if (prompt.empty()) prompt.push_back(geom::TOK_BOS);
        run_prompt(prompt, true, true);
        if (e.o.profile && (bench_kv_pos > 0 || bench_ctx > 0)) e.prof.reset();
    }

    if (bench_kv_pos > 0) {
        if (bench_kv_pos + gen + 2 > cap) die("--bench-kv-pos + --gen exceeds --cap");
        for (int s = 0; s < sessions; ++s) e.reset_slot(s);
        e.seed_bench_kv(bench_kv_pos, sessions);
        std::vector<ernie55rt::RowOut> out;
        std::vector<int> cur(size_t(sessions), geom::TOK_BOS);
        const float inv_temp = temp > 0.f ? 1.f / temp : 0.f;
        const int warm = 4;
        for (int g = 0; g < warm; ++g) {
            std::vector<ernie55::StepSeg> segs;
            std::vector<ernie55rt::RowSample> sm;
            for (int s = 0; s < sessions; ++s) {
                segs.push_back({s, s, bench_kv_pos - warm + g, 1});
                sm.push_back(ernie55rt::RowSample{inv_temp, top_p, seed, uint32_t(g + 1)});
            }
            e.forward(segs, cur.data(), sm.data(), out, false);
            for (int s = 0; s < sessions; ++s) cur[size_t(s)] = out[size_t(s)].tok;
        }
        ERNIE55_CK(cudaStreamSynchronize(e.stream));
        if (e.o.profile) e.prof.reset();
        const auto td0 = std::chrono::steady_clock::now();
        for (int g = 0; g < gen; ++g) {
            std::vector<ernie55::StepSeg> segs;
            std::vector<ernie55rt::RowSample> sm;
            for (int s = 0; s < sessions; ++s) {
                segs.push_back({s, s, bench_kv_pos + g, 1});
                sm.push_back(ernie55rt::RowSample{inv_temp, top_p, seed, uint32_t(bench_kv_pos + g + 1)});
            }
            e.forward(segs, cur.data(), sm.data(), out, false);
            for (int s = 0; s < sessions; ++s) cur[size_t(s)] = out[size_t(s)].tok;
        }
        ERNIE55_CK(cudaStreamSynchronize(e.stream));
        const auto td1 = std::chrono::steady_clock::now();
        const double dec_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();
        const double tps = gen / (dec_ms / 1e3 + 1e-9);
        std::fprintf(stderr, "# m4sim_ernie55 BENCH_KV TP=%d kv=%s mx8=%d hybrid=%d ctx=%d gen=%d sess=%d: decode=%.1f ms (%.2f tok/s/sess, %.2f agg tok/s)\n",
                     G6_NR, ernie55rt::kv_fmt_name(e.o.kv_fmt), e.o.mx8_dense ? 1 : 0, e.o.use_hybrid ? 1 : 0,
                     bench_kv_pos, gen, sessions, dec_ms, tps, tps * sessions);
        if (e.o.profile) e.prof.print(0);
        if (e.o.profile && bench_ctx > 0) e.prof.reset();
    }

    if (bench_ctx > 0) {
        std::vector<int> prompt(static_cast<size_t>(bench_ctx));
        prompt[0] = geom::TOK_BOS;
        for (int i = 1; i < bench_ctx; ++i) prompt[size_t(i)] = 100 + ((i * 131 + 17) % 30000);
        run_prompt(prompt, tok_str.empty(), tok_str.empty());
    } else if (tok_str.empty() && bench_kv_pos <= 0) {
        run_prompt({geom::TOK_BOS}, true, true);
    }
    return 0;
}
