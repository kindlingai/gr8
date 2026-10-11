// The request file and the response lines. m4sim --serve DIR reads requests one at a time with serve_next. S2
// (g6m4 --serve, s2/sessions.h) parses the same request format itself and writes the same response lines through
// Emitter, plus its own `a` line (frontend/g6s2shim.py describes the S2 files).
//   DIR/req.<n>  (written atomically: req.<n>.tmp then rename), two lines:
//       n_gen <max_tokens> temp <float> seed <uint32> min <min_tokens> stop <id,id,..|-> [top_p <float>] [lp <k>] [plp <k>]
//       (lp: logprobs of the generated tokens with their top k, 0..20; plp: the same for the prompt tokens; absent = off.
//        A request with plp is run as decode steps from its first token, no prefill path: the shim limits its length.)
//       <prompt ids, comma-separated>
//   DIR/quit     ends the loop
//   DIR/ready    written once the model is loaded, before the first request is read
//   DIR/resp.<n> (rank 0, one flushed line per event):
//       l <lp> <rank> <n> <id>:<lp> ..                   with lp: one line per token of the next t line, before it
//                                                        (lp: logprob of the token; rank: 1 = best, -1 = beyond the top 20;
//                                                        then n top entries, best first)
//       p <j> <lp> <rank> <n> <id>:<lp> ..               with plp: prompt token j (1 <= j < prompt length), before the first t
//       t <id> <id> ..                                   newly committed tokens
//       m <rounds> <draft_tokens> <accepted> <acc_pos0> ..   cumulative MTP counters since start
//       end <stop|length|error> <stop_id|-1> <completion_tokens>
#pragma once
#include <sys/stat.h>
#include <chrono>
#include <ctime>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace g6rt {

struct ServeReq
{
    std::vector<int> prompt, stop;
    int n_gen = 0, min_tokens = 0;
    float temp = 0.f, top_p = 0.f;
    uint32_t seed = 1234u;
    int lp = -1, plp = -1;        // logprobs / prompt_logprobs: k (0..LP_MAX) or -1 = off
};

constexpr int LP_MAX = 20;       // = glue::LP_K (rank.cuh checks)
// one logprob entry of the response: the token's logprob and rank, then its position's top list
struct LpRow
{
    float lp = 0.f;
    int rank = -1;               // -1 = not within the top LP_MAX
    int n = 0;                   // entries of the top list (0 = none asked)
    int id[LP_MAX];
    float tlp[LP_MAX];
};

inline bool serve_exists(const std::string& p) { struct stat s; return stat(p.c_str(), &s) == 0; }

// DIR/ready: the rank has loaded and waits for requests
inline void serve_ready(const std::string& dir)
{
    if (FILE* f = fopen((dir + "/ready").c_str(), "w")) fclose(f);
}

// A request file's text -> r. Line 1 holds the parameters, at any length; line 2 the prompt ids. Any malformed field, or
// a stop or prompt id outside 0..vocab-1 (the kernels index the embedding table with prompt ids unchecked), leaves
// r.n_gen = 0, which every rank answers with "end error".
inline void serve_parse(const std::string& text, int vocab, ServeReq& r)
{
    r = ServeReq();
    const size_t nl = text.find('\n');
    const std::string head = text.substr(0, nl);
    const std::string ids = nl == std::string::npos ? "" : text.substr(nl + 1, text.find('\n', nl + 1) - nl - 1);
    std::vector<std::string> w;                                    // the parameter line's words
    for (size_t i = 0; i < head.size();)
    {
        const size_t j = head.find(' ', i);
        if (j != i) w.push_back(head.substr(i, j - i));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    auto num = [](const std::string& s, long lo, long hi, long* out) {   // the whole word, in [lo, hi]
        char* e;
        errno = 0;
        const long x = strtol(s.c_str(), &e, 10);
        if (s.empty() || *e || errno || x < lo || x > hi) return false;
        *out = x;
        return true;
    };
    auto real = [](const std::string& s, float* out) {
        char* e;
        *out = strtof(s.c_str(), &e);
        return !s.empty() && !*e;
    };
    auto list = [&](const std::string& s, std::vector<int>& v) {      // id,id,..: every one in the vocabulary
        for (size_t i = 0; i <= s.size();)
        {
            const size_t j = std::min(s.find(',', i), s.size());
            long x;
            if (!num(s.substr(i, j - i), 0, vocab - 1, &x)) return false;
            v.push_back((int)x);
            i = j + 1;
        }
        return true;
    };
    long n_gen, seed, min_tokens;
    const bool ok = w.size() >= 10 && w[0] == "n_gen" && num(w[1], 1, 1 << 30, &n_gen) && w[2] == "temp" &&
                    real(w[3], &r.temp) && w[4] == "seed" && num(w[5], 0, 0xFFFFFFFFL, &seed) && w[6] == "min" &&
                    num(w[7], 0, 1 << 30, &min_tokens) && w[8] == "stop" && (w[9] == "-" || list(w[9], r.stop)) &&
                    !ids.empty() && list(ids, r.prompt);
    if (!ok) { r = ServeReq(); return; }
    r.n_gen = (int)n_gen; r.seed = (uint32_t)seed; r.min_tokens = (int)min_tokens;
    for (size_t i = 10; i + 1 < w.size(); i += 2)                  // optional (older shims: absent)
    {
        long k;
        if (w[i] == "top_p") { if (!real(w[i + 1], &r.top_p)) r.top_p = 0.f; }
        // the shim never sends k > LP_MAX; clamping is a backstop
        else if (w[i] == "lp") r.lp = num(w[i + 1], 0, 1 << 30, &k) ? (int)std::min<long>(k, LP_MAX) : -1;
        else if (w[i] == "plp") r.plp = num(w[i + 1], 0, 1 << 30, &k) ? (int)std::min<long>(k, LP_MAX) : -1;
        else break;
    }
}

// blocks until DIR/req.<n> or DIR/quit exists; false on quit
inline bool serve_next(const std::string& dir, size_t n, int vocab, ServeReq& r)
{
    const std::string f = dir + "/req." + std::to_string(n);
    for (;;)
    {
        if (serve_exists(f)) break;
        if (serve_exists(dir + "/quit")) return false;
        usleep(300);
    }
    FILE* fp = fopen(f.c_str(), "r");
    if (!fp) return false;
    std::string text;
    int c;
    while ((c = fgetc(fp)) != EOF) text.push_back((char)c);
    fclose(fp);
    serve_parse(text, vocab, r);
    return true;
}

// G6_PF_TIMES=1 (measurement only, default off; host clock only, no extra syncs): wall-clock laps of one request on this
// process, from the request file's mtime to the first committed token; one stderr line per request:
//   # times seq <n> <item>=<ms> .. total=<ms>
struct PfTimes
{
    bool on = getenv("G6_PF_TIMES") && atoi(getenv("G6_PF_TIMES")) == 1;
    std::chrono::steady_clock::time_point t;
    std::string s;
    double tot = 0;
    bool first = false;
    // start of a request; file: the request file (its mtime -> the 'poll' item), or empty
    void start(const std::string& file)
    {
        if (!on) return;
        t = std::chrono::steady_clock::now(); s.clear(); tot = 0; first = false;
        struct stat st;
        if (!file.empty() && stat(file.c_str(), &st) == 0)
        {
            timespec now;
            clock_gettime(CLOCK_REALTIME, &now);
            const double ms = (now.tv_sec - st.st_mtim.tv_sec) * 1e3 + (now.tv_nsec - st.st_mtim.tv_nsec) * 1e-6;
            add("poll", ms);
        }
    }
    void add(const char* k, double ms) { char b[64]; snprintf(b, sizeof b, " %s=%.3f", k, ms); s += b; tot += ms; }
    void lap(const char* k)
    {
        if (!on) return;
        const auto n = std::chrono::steady_clock::now();
        add(k, std::chrono::duration<double, std::milli>(n - t).count());
        t = n;
    }
    void first_token(size_t q)                       // call when the first token is committed
    {
        if (!on || first) return;
        first = true;
        lap("to_first_token");
        fprintf(stderr, "# times seq %zu%s total=%.3f\n", q, s.c_str(), tot);
    }
};

// counts committed tokens against n_gen, stop ids and min_tokens; rank 0 writes them to DIR/resp.<n>
struct Emitter
{
    FILE* f = nullptr;
    bool on = false;
    int n_gen = 0, min_tokens = 0, emitted = 0, stop_id = -1;
    bool done = false;
    std::vector<int> stop;
    void open(const std::string& dir, size_t n, bool rank0, const ServeReq& r)
    {
        on = true; n_gen = r.n_gen; min_tokens = r.min_tokens; stop = r.stop; emitted = 0; stop_id = -1; done = false;
        f = rank0 ? fopen((dir + "/resp." + std::to_string(n)).c_str(), "w") : nullptr;
    }
    static void lp_text(std::string& s, const LpRow& r)
    {
        char b[64];
        snprintf(b, sizeof b, " %.9g %d %d", r.lp, r.rank, r.n);
        s += b;
        for (int i = 0; i < r.n; ++i)
        {
            snprintf(b, sizeof b, " %d:%.9g", r.id[i], r.tlp[i]);
            s += b;
        }
    }
    // prompt token j's logprob entry (--serve request with plp)
    void push_prompt(int j, const LpRow& r)
    {
        if (!on || !f) return;
        std::string line = "p " + std::to_string(j);
        lp_text(line, r);
        fprintf(f, "%s\n", line.c_str());
        fflush(f);
    }
    // returns true when the request is finished (a stop id was emitted or n_gen reached). lps: k entries (the logprobs
    // of the tokens, request with lp) or null
    bool push(const int* t, int k, const LpRow* lps = nullptr)
    {
        if (!on) return false;
        if (done) return true;
        std::string line = "t", lpl;
        for (int i = 0; i < k && !done; ++i)
        {
            line += " " + std::to_string(t[i]);
            if (lps) { lpl += "l"; lp_text(lpl, lps[i]); lpl += "\n"; }
            ++emitted;
            for (int s : stop)
                if (t[i] == s && emitted > min_tokens) { stop_id = s; done = true; }
            if (emitted >= n_gen) done = true;
        }
        if (f) { fprintf(f, "%s%s\n", lpl.c_str(), line.c_str()); fflush(f); }
        return done;
    }
    // drafted: drafts verified in all rounds (-1: rounds * K, a fixed draft length)
    void end(const char* reason_if_error, long rounds, int K, long tokens, const long* acc, long drafted = -1)
    {
        if (!on) return;
        if (f)
        {
            if (K > 0)
            {
                fprintf(f, "m %ld %ld %ld", rounds, drafted < 0 ? rounds * K : drafted, tokens - rounds);
                for (int i = 0; i < K; ++i)
                {
                    long c = 0;
                    for (int a = i + 1; a <= K; ++a) c += acc[a];
                    fprintf(f, " %ld", c);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "end %s %d %d\n", reason_if_error ? reason_if_error : stop_id >= 0 ? "stop" : "length", stop_id, emitted);
            fclose(f);
            f = nullptr;
        }
        on = false;
    }
};

}   // namespace g6rt
