// Host-only, bounded per-request DSpark calibration. No device reads or stream fences.
#pragma once
#include "../ds41_dspark_dev.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
namespace g6s2 {
inline double ds41_draft_logp(float dot, double temperature) {
    const double z = ds41::dspd::conf_clamped(dot, float(temperature));
    // Stable log(sigmoid(z)); invalid confidence is conservatively zero survival.
    return std::isfinite(z) ? -std::log1p(std::exp(-z)) : -INFINITY;
}
struct Ds41DraftStats {
    uint64_t request = 0, rounds = 0, emitted = 0, verified = 0, accepted = 0;
    uint64_t hash = 14695981039346656037ull;
    std::array<uint64_t, 6> hist{};
    std::array<uint64_t, 5> observed{}, realized{};
    std::array<double, 5> predicted{};
    std::array<std::array<uint64_t, 6>, 32> step_count{};
    std::array<std::array<double, 6>, 32> verify_us{};
    uint64_t other_verify_steps = 0;
    double other_verify_us = 0;
    uint64_t draft_steps = 0;
    double draft_us = 0;
    bool finished = false;
    void verify_time(int ns, int rps, double us) {
        if (ns >= 1 && ns <= 32 && rps >= 1 && rps <= 6) {
            ++step_count[size_t(ns - 1)][size_t(rps - 1)];
            verify_us[size_t(ns - 1)][size_t(rps - 1)] += us;
        } else {
            ++other_verify_steps; other_verify_us += us; // forced inserts can exceed a draft block
        }
    }
    void draft_time(double us) { ++draft_steps; draft_us += us; }
    void observe(int ns, int k, int a, const std::array<double, 5>& lp) {
        ++rounds; verified += uint64_t(k); accepted += uint64_t(a); ++hist[size_t(k)];
        // Canonical little-endian uint32 pairs, independent of native endianness.
        for (uint32_t v : {uint32_t(ns), uint32_t(k + 1)})
            for (unsigned shift = 0; shift < 32; shift += 8) { hash ^= (v >> shift) & 255u; hash *= 1099511628211ull; }
        double p = 1;
        for (int j = 0; j < k; ++j) {
            p *= std::exp(lp[size_t(j)]);
            ++observed[size_t(j)]; predicted[size_t(j)] += p; realized[size_t(j)] += j < a;
        }
    }
    std::string finish(int rank, unsigned slot) {
        if (finished) return {};
        finished = true;
        return line(rank, slot);
    }
    std::string line(int rank, unsigned slot) const {
        char b[512];
        std::snprintf(b, sizeof b, "# ds41 draft rank=%d request=%llu slot=%u rounds=%llu emitted_tokens=%llu drafts_verified=%llu drafts_accepted=%llu rps_seq_hash=%016llx rps_hist=",
            rank, (unsigned long long)request, slot, (unsigned long long)rounds, (unsigned long long)emitted,
            (unsigned long long)verified, (unsigned long long)accepted, (unsigned long long)hash);
        std::string s = b;
        for (size_t j = 0; j < hist.size(); ++j) { if (j) s += ','; s += std::to_string(hist[j]); }
        s += " calibration_j_n_predsum_accepted=";
        for (size_t j = 0; j < observed.size(); ++j) {
            std::snprintf(b, sizeof b, "%s%zu:%llu:%.9g:%llu", j ? "," : "", j + 1,
                (unsigned long long)observed[j], predicted[j], (unsigned long long)realized[j]);
            s += b;
        }
        std::snprintf(b, sizeof b, " draft_steps=%llu draft_us_sum=%.9g verify_ns_rps_n_us=",
            (unsigned long long)draft_steps, draft_us);
        s += b;
        bool first = true;
        for (size_t ns = 0; ns < step_count.size(); ++ns) for (size_t r = 0; r < 6; ++r) {
            if (!step_count[ns][r]) continue;
            std::snprintf(b, sizeof b, "%s%zu:%zu:%llu:%.9g", first ? "" : ",", ns + 1, r + 1,
                (unsigned long long)step_count[ns][r], verify_us[ns][r]);
            s += b; first = false;
        }
        if (first) s += "none";
        std::snprintf(b, sizeof b, " other_verify_steps=%llu other_verify_us=%.9g",
            (unsigned long long)other_verify_steps, other_verify_us);
        s += b;
        return s;
    }
};
} // namespace g6s2
