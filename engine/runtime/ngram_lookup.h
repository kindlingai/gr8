// ngram_lookup.h: the GLM-distilled n-gram table of S5 (#56) as a host-side draft source. Host code only.
//
// tools/ngram/ngram.py export writes the table from GLM-5.3's own traces: for each frequent context (the last n tokens,
// several n with backoff) the token GLM most often chose next and its mean probability there, plus a calibration of
// those probabilities measured on held-out traces (the raw means overestimate acceptance by 30-60%). A lookup chains:
// the proposed token extends the context for the next lookup, so one hit can propose a run. The target verifies every
// proposed token as it verifies a drafter's, so the table can only change speed, never output.
//
// File (text, one record a line; '#' lines are comments):
//   g6ngram 1
//   ns 8 6 4 3                       context lengths, longest first (lookups back off in this order)
//   cal c0 c1 .. c7 | cal none       acceptance per probability bucket (SpecLen::bucket's edges); none: the raw means
//   e n t1 .. tn tok p               context t1..tn (oldest first) -> next token tok at mean probability p
// Contexts are keyed by a 64-bit hash of (n, tokens). A collision proposes a wrong token, which the verify rejects: it
// costs speed only. Every rank must load the same file: digest() (FNV-1a over its bytes) goes into the S2 runtime
// signature, because the chain decides each round's verify width and every rank must choose the same one.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace g6rt {

struct NgramTable
{
    static constexpr int NB = 8;                  // calibration buckets (edges as SpecLen::bucket)
    static constexpr int NMAX = 32;               // longest context length accepted
    std::vector<int> ns;                          // context lengths, descending
    bool has_cal = false;
    double cal[NB] = {};
    std::unordered_map<uint64_t, std::pair<int, float>> t;
    uint64_t digest = 0;

    static int bucket(double p)
    {
        static const double edge[NB - 1] = {0.2, 0.4, 0.6, 0.75, 0.88, 0.95, 0.99};
        int b = 0;
        while (b < NB - 1 && p >= edge[b]) ++b;
        return b;
    }
    static uint64_t key(const int* ctx, int n)
    {
        uint64_t h = 1469598103934665603ull ^ (uint64_t)n;
        for (int i = 0; i < n; ++i)
        {
            uint32_t v = (uint32_t)ctx[i];
            for (int b = 0; b < 4; ++b) { h ^= (v >> (8 * b)) & 0xff; h *= 1099511628211ull; }
        }
        return h;
    }
    static uint64_t fnv(const std::string& s)
    {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
        return h;
    }
    // the calibrated chance that a proposed token at stored probability p is right, given its prefix was
    double conf(double p) const { return has_cal ? cal[bucket(p)] : p; }
    int nmax() const { return ns.empty() ? 0 : ns[0]; }

    // the file's text; false with *err on a malformed one
    bool parse(const std::string& text, std::string* err)
    {
        *this = NgramTable();
        digest = fnv(text);
        auto fail = [&](int ln, const std::string& w) { *err = "ngram table line " + std::to_string(ln) + ": " + w; return false; };
        std::istringstream in(text);
        std::string line;
        int ln = 0;
        bool head = false, have_ns = false, have_cal = false;
        while (std::getline(in, line))
        {
            ++ln;
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            std::string tag;
            ls >> tag;
            if (!head)
            {
                int v = 0;
                if (tag != "g6ngram" || !(ls >> v) || v != 1) return fail(ln, "want 'g6ngram 1' first");
                head = true;
                continue;
            }
            if (tag == "ns")
            {
                int n;
                while (ls >> n)
                {
                    if (n < 1 || n > NMAX) return fail(ln, "context length 1.." + std::to_string(NMAX));
                    ns.push_back(n);
                }
                if (ns.empty()) return fail(ln, "no context length");
                std::sort(ns.rbegin(), ns.rend());
                ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
                have_ns = true;
            }
            else if (tag == "cal")
            {
                std::string a;
                if (!(ls >> a)) return fail(ln, "cal: none or 8 numbers");
                if (a != "none")
                {
                    has_cal = true;
                    char* end = nullptr;
                    cal[0] = strtod(a.c_str(), &end);
                    if (*end) return fail(ln, "cal: none or 8 numbers");
                    for (int b = 1; b < NB; ++b) if (!(ls >> cal[b])) return fail(ln, "cal: none or 8 numbers");
                    if (ls >> a) return fail(ln, "cal: none or 8 numbers");
                    for (int b = 0; b < NB; ++b) if (!(cal[b] >= 0 && cal[b] <= 1)) return fail(ln, "cal: each in [0, 1]");
                }
                have_cal = true;
            }
            else if (tag == "e")
            {
                if (!have_ns) return fail(ln, "an entry before 'ns'");
                int n;
                if (!(ls >> n) || std::find(ns.begin(), ns.end(), n) == ns.end()) return fail(ln, "entry length not in ns");
                int ctx[NMAX];
                for (int i = 0; i < n; ++i) if (!(ls >> ctx[i])) return fail(ln, "short entry");
                int tok;
                double p;
                if (!(ls >> tok >> p) || tok < 0 || !(p > 0 && p <= 1)) return fail(ln, "entry: token >= 0 and p in (0, 1]");
                t[key(ctx, n)] = {tok, (float)p};
            }
            else return fail(ln, "unknown record '" + tag + "'");
        }
        if (!head || !have_ns || !have_cal) { *err = "ngram table: needs 'g6ngram 1', 'ns' and 'cal'"; return false; }
        return true;
    }
    // the next token after ctx[0..len) at the longest context length with an entry; false: none
    bool next(const int* ctx, int len, int* tok, float* p) const
    {
        for (int n : ns)
        {
            if (len < n) continue;
            const auto it = t.find(key(ctx + len - n, n));
            if (it != t.end()) { *tok = it->second.first; *p = it->second.second; return true; }
        }
        return false;
    }
    // a chain of at most kmax tokens after tokens[0..len): out[1..k], their calibrated conditionals c[1..k]; returns k
    int chain(const int* tokens, int len, int kmax, int* out, double* c) const
    {
        const int nm = nmax();
        if (!nm || kmax < 1) return 0;
        std::vector<int> w(tokens + std::max(0, len - nm), tokens + len);
        int k = 0;
        while (k < kmax)
        {
            int tok;
            float p;
            if (!next(w.data(), (int)w.size(), &tok, &p)) break;
            ++k;
            out[k] = tok;
            c[k] = conf(p);
            w.push_back(tok);
        }
        return k;
    }
};

}   // namespace g6rt
