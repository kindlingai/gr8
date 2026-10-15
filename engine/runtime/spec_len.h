// spec_len.h: the draft length of each speculative round (--spec-adaptive). Host code only: g++ builds its
// test, engine/tests/test_spec_len.cpp. It makes one decision per round; s2_batch.h's drafter choice builds on it.
//
// A round with k drafts costs verify(k + 1 rows) plus the drafting, and is expected to commit
//   E(k) = 1 + sum over j = 1..k of c_1 * ... * c_j
// tokens, where c_j is the chance that draft j is accepted given that drafts 1..j-1 were. choose() returns the allowed
// k with the most E(k) per millisecond.
//   c_j     per (position j, confidence bucket of the draft's own probability) from decayed counts of past rounds,
//           shrunk toward the bucket over all positions, then toward the position's recent rate, then toward a seed. The drafter's probability is
//           exp(DflOut::lp) for DFlash2 (top-16 softmax of the selector scores, or the head's log-probability) and
//           exp(StepOut::lp) of an MTP draft step.
//   verify  the measured step times by row count at 2200 MHz (median launch-to-sync of DFlash2 verify steps), fixed.
// Every rank runs choose() and more() on its own, and a rank that picks another k waits forever on the others' fabric
// collectives. So every input must be bit-identical on every rank: the drafts' log-probabilities, the accept counts and
// the fixed table. This is stricter than for the drafted tokens. A 1-ULP difference in a draft's probability leaves the
// token the same but can move it across a bucket edge. The probabilities are identical today because every merge
// and reduce that feeds them runs in rank order (k_walk, k_ar_conv_norm, head_merge). A reduce that starts at the local
// rank would break adaptive k while fixed k keeps working. Step times differ per rank and must not feed the decision (a
// refit on them did, and deadlocked the fabric). hash() goes into each round's log line, so a hang can be traced to the
// first round whose hash differs between the rank logs.
// Rounds only teach c_j for the positions they offered (k >= j) after an accepted prefix.
#pragma once
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace g6rt {

// Boot-time cost overrides: an exact comma-separated list of finite milliseconds.
// Verify steps must cost > 0; drafting may be free in a tuning experiment.
inline bool parse_spec_costs(const char* text, double* values, int count, bool allow_zero = false)
{
    if (!text) return false;
    const char* p = text;
    for (int i = 0; i < count; ++i)
    {
        while (std::isspace((unsigned char)*p)) ++p;
        char* end;
        errno = 0;
        const double v = std::strtod(p, &end);
        if (end == p || errno == ERANGE || !std::isfinite(v) || (allow_zero ? v < 0 : v <= 0)) return false;
        values[i] = v;
        p = end;
        while (std::isspace((unsigned char)*p)) ++p;
        if (i + 1 == count) return *p == '\0';
        if (*p != ',') return false;
        ++p;
    }
    return false;
}

struct SpecLen
{
    static constexpr int KX = 7;      // most drafts in a round
    static constexpr int NB = 8;      // confidence buckets
    bool on = false;
    std::vector<int> levels;          // allowed draft counts, ascending, each <= KX
    double draft_round = 0;           // drafting cost paid once per round (DFlash2 step, MTP first-form pass), ms
    double draft_each = 0;            // drafting cost per draft after the first (MTP chain step), ms
    double table[KX + 2] = {0, 43.0, 52.4, 60.4, 68.4, 76.1, 83.7, 91.1, 97.4};   // verify ms by rows 1..8 (TP4 v2 home, DFlash2 steps)
    double seed[KX + 1] = {0, 0.75, 0.72, 0.70, 0.70, 0.70, 0.68, 0.68};
    double pn[KX + 1] = {}, pk[KX + 1] = {};             // per position: decayed rounds offered / accepted
    double bn[KX + 1][NB] = {}, bk[KX + 1][NB] = {};     // per (position, bucket)
    double pos_decay = 0.9;           // per round: the position rate follows the text within a few dozen rounds
    double bucket_decay = 0.998;      // the calibration of the drafter's probability changes slowly
    double prior = 4;                 // pseudo-rounds of the fallback in each estimate
    long rounds = 0, chosen[KX + 1] = {};
    double pred_sum = 0, got_sum = 0; // predicted and actual tokens per round, summed (log line)

    static int bucket(float lp)
    {
        if (!std::isfinite(lp)) return 0;
        static const double edge[NB - 1] = {0.2, 0.4, 0.6, 0.75, 0.88, 0.95, 0.99};
        const double p = std::exp((double)lp);
        int b = 0;
        while (b < NB - 1 && p >= edge[b]) ++b;
        return b;
    }
    double rate(int j) const { return (pk[j] + prior * seed[j]) / (pn[j] + prior); }
    // c_j for a draft with log-probability lp (NaN: unknown, the position's rate). The (position, bucket) count shrinks
    // toward the bucket's count over all positions, which shrinks toward the position's rate: positions that small
    // levels never offer still learn from the drafter's probability at the positions they do offer.
    double cond(int j, float lp) const
    {
        const double r = rate(j);
        if (std::isnan(lp)) return r;
        const int b = bucket(lp);
        double n = 0, k = 0;
        for (int i = 1; i <= KX; ++i) { n += bn[i][b]; k += bk[i][b]; }
        const double pool = (k + prior * r) / (n + prior);
        return (bk[j][b] + prior * pool) / (bn[j][b] + prior);
    }
    double verify_ms(int rows) const { return table[std::min(std::max(rows, 1), KX + 1)]; }
    double draft_ms(int drafted) const { return draft_round + draft_each * std::max(0, drafted - 1); }

    // E(k) / ms for every k <= n with lp[1..n] (lp may be null); drafted = drafts already paid for (cost of the round)
    double ratio(int k, const float* lp, int drafted, double* e_out = nullptr) const
    {
        double e = 1, s = 1;
        for (int j = 1; j <= k; ++j) { s *= cond(j, lp ? lp[j] : NAN); e += s; }
        if (e_out) *e_out = e;
        return e / (verify_ms(k + 1) + draft_ms(std::max(drafted, k)));
    }
    // the draft count to verify: the allowed level <= n with the most expected tokens per ms (ties: fewer drafts). With
    // no level <= n the caller drafted too few, and a non-level could ask for a verify graph that was never captured.
    int choose(const float* lp, int n, int drafted, double* e_out = nullptr) const
    {
        int best = 0;
        double br = -1, be = 1;
        for (int k : levels)
        {
            if (k > n) break;
            double e;
            const double r = ratio(k, lp, drafted, &e);
            if (r > br * (1 + 1e-9)) { br = r; best = k; be = e; }
        }
        if (!best) { fprintf(stderr, "g6: spec_len: %d drafts, below every level\n", n); abort(); }
        if (e_out) *e_out = be;
        return best;
    }
    // MTP: after drafting n drafts (lp[1..n] known), is one more chain step worth it? It is when drafting up to the next
    // level above n (the drafts after n at their positions' rates) gives a better ratio than the best level <= n. A
    // level below the smallest one has ratio 0, so the chain always reaches the smallest level.
    bool more(const float* lp, int n) const
    {
        const auto up = std::upper_bound(levels.begin(), levels.end(), n);
        if (up == levels.end()) return false;
        const int next = *up;
        double now = 0;
        for (int k : levels)
            if (k <= n) now = std::max(now, ratio(k, lp, n));
        double e = 1, s = 1;
        for (int j = 1; j <= next; ++j) { s *= j <= n ? cond(j, lp[j]) : rate(j); e += s; }
        return e / (verify_ms(next + 1) + draft_ms(next)) > now;
    }
    // FNV-1a over the bytes of lp[1..n] and k (the round's log line; see the header)
    static uint32_t hash(const float* lp, int n, int k)
    {
        uint32_t h = 2166136261u;
        auto add = [&](const void* p, size_t len) { for (size_t i = 0; i < len; ++i) h = (h ^ ((const uint8_t*)p)[i]) * 16777619u; };
        if (lp) add(lp + 1, sizeof(float) * n);
        add(&k, sizeof k);
        return h;
    }
    // a round verified k drafts and accepted a of them. lp[1..k]: the drafts' log-probabilities (null: none)
    void observe(int k, int a, const float* lp, double predicted)
    {
        for (int j = 1; j <= KX; ++j)
        {
            pn[j] *= pos_decay; pk[j] *= pos_decay;
            for (int b = 0; b < NB; ++b) { bn[j][b] *= bucket_decay; bk[j][b] *= bucket_decay; }
        }
        for (int j = 1; j <= k && j <= a + 1; ++j)
        {
            const double acc = j <= a ? 1 : 0;
            pn[j] += 1; pk[j] += acc;
            if (lp) { const int b = bucket(lp[j]); bn[j][b] += 1; bk[j][b] += acc; }
        }
        ++rounds; ++chosen[k];
        pred_sum += predicted; got_sum += a + 1;
    }
    // false when no level is in 1..KX
    bool set_levels(const std::vector<int>& v)
    {
        levels.clear();
        for (int k : v) if (k >= 1 && k <= KX) levels.push_back(k);
        std::sort(levels.begin(), levels.end());
        levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
        return !levels.empty();
    }
    std::string summary() const
    {
        std::string s = "# spec-adaptive: " + std::to_string(rounds) + " rounds, chosen k";
        char b[128];
        for (int k = 1; k <= KX; ++k) if (chosen[k]) { snprintf(b, sizeof b, " %d:%ld", k, chosen[k]); s += b; }
        snprintf(b, sizeof b, "; tokens per round predicted %.3f, got %.3f", rounds ? pred_sum / rounds : 0., rounds ? got_sum / rounds : 0.);
        return s + b;
    }
};

}   // namespace g6rt
