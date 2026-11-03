// ctx_lookup.h: the request's own tokens as a draft source (GR8_LOOKUP). Host code only.
//
// The committed tokens (prompt and reply) are indexed by their bigrams. A lookup after tokens [0, len) takes the latest
// SCAN earlier places where the last two tokens occur, extends each match backwards up to LONG tokens, and proposes the
// tokens that followed the longest match (the latest one on a tie). Text the model repeats, such as a file it rewrites
// with a small change or a paragraph it quotes, then drafts itself. The target verifies every proposed token, so the
// lookup can change speed only, never output.
//
// A proposed token is taken as right with chance p(b) given its prefix was, b the bucket of the match length. p(b) is
// the bucket's rate over the chains of the request's rounds (verified or only scored), starting from seed[b] weighted as
// `prior` positions.
#pragma once
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace g6rt {

struct CtxLookup
{
    static constexpr int NB = 5;                  // match lengths 2, 3, 4..7, 8..15, 16 and more
    static constexpr int SCAN = 16, LONG = 64;
    static constexpr double seed[NB] = {0.3, 0.45, 0.6, 0.75, 0.85};
    static constexpr double prior = 4;
    double n[NB] = {}, k[NB] = {};                // positions offered and accepted in this request, by bucket
    std::unordered_map<uint64_t, std::vector<int>> at;   // bigram -> positions of its second token, ascending
    int indexed = 0;                              // tokens [0, indexed) are in `at`

    static int bucket(int len) { return len <= 2 ? 0 : len == 3 ? 1 : len < 8 ? 2 : len < 16 ? 3 : 4; }
    static uint64_t key(int a, int b) { return (uint64_t)(uint32_t)a << 32 | (uint32_t)b; }
    double p(int b) const { return (k[b] + prior * seed[b]) / (n[b] + prior); }

    // After t[0, len), which must extend the tokens of earlier calls: the chain in out[1..m] (m <= kmax), each token's
    // chance in c[1..m], the match length in *mlen. Returns m, 0 when the last two tokens have no earlier place.
    // Indexing the new tokens costs O(len - indexed); the first call on a long prompt indexes all of it.
    int chain(const int* t, int len, int kmax, int* out, double* c, int* mlen)
    {
        for (int i = std::max(1, indexed); i < len; ++i) at[key(t[i - 1], t[i])].push_back(i);
        indexed = std::max(indexed, len);
        *mlen = 0;
        if (len < 3) return 0;
        const auto it = at.find(key(t[len - 2], t[len - 1]));
        if (it == at.end()) return 0;
        const std::vector<int>& v = it->second;
        int best = -1, bl = 0;
        for (int i = (int)v.size() - 1, seen = 0; i >= 0 && seen < SCAN; --i)
        {
            const int j = v[i];
            if (j >= len - 1) continue;           // the tail itself
            ++seen;
            int l = 2;
            while (l < LONG && j - l >= 0 && t[j - l] == t[len - 1 - l]) ++l;
            if (l > bl) { bl = l; best = j; }
        }
        if (best < 0) return 0;
        const int m = std::min(kmax, len - 1 - best);
        const double pb = p(bucket(bl));
        for (int i = 1; i <= m; ++i) { out[i] = t[best + i]; c[i] = pb; }
        *mlen = bl;
        return m;
    }
    // a chain with match length mlen had `offered` positions checked, of which the first `accepted` were right
    void observe(int mlen, int offered, int accepted)
    {
        const int b = bucket(mlen);
        k[b] += accepted;
        n[b] += accepted + (accepted < offered ? 1 : 0);
    }
};

}   // namespace g6rt
