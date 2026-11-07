// A request's tokens indexed by bigram, and the longest earlier occurrence of a sequence's tail: the host side of the
// sequence penalty (penalty.h, s2/dispatch.h). The same search serves a context-lookup draft source (what followed the
// match). Host only.
//
// The sequence is the committed tokens, t[0, n), followed by a short tail the caller has not committed (a step's
// drafts). `extend` indexes the committed tokens as they come; `longest` finds, among the latest SCAN places where the
// tail's last two tokens occur earlier (in the index, or in the tail itself), the one the longest run of tokens before
// them also matches, up to LONG, the latest on a tie. The caller penalizes or drafts the token that followed it.
#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace g6rt
{

struct CtxMatch
{
    int len = 0;      // tokens matched (0: none)
    int next = -1;    // the position after the earlier occurrence: the token that continued it
};

struct CtxIndex
{
    static constexpr int SCAN = 16, LONG = 64;
    std::unordered_map<uint64_t, std::vector<int>> at;   // bigram -> positions of its second token, ascending
    int indexed = 0;                                     // tokens [0, indexed) are in `at`

    static uint64_t key(int a, int b) { return (uint64_t)(uint32_t)a << 32 | (uint32_t)b; }
    void clear() { at.clear(); indexed = 0; }

    // index t(i) for i in [indexed, n); t(i) for i < indexed must be what it was
    template <class T> void extend(const T& t, int n)
    {
        for (int i = indexed > 0 ? indexed : 1; i < n; ++i) at[key(t(i - 1), t(i))].push_back(i);
        if (n > indexed) indexed = n;
    }
    // the sequence t(0 .. len): [0, indexed) through the index, [indexed, len) by a scan. len >= 2 for a match.
    template <class T> CtxMatch longest(const T& t, int len) const
    {
        CtxMatch m;
        if (len < 2) return m;
        const int a = t(len - 2), b = t(len - 1);
        std::vector<int> ends;                           // candidate second-token positions, latest first
        for (int j = len - 2; j >= indexed && j >= 1 && (int)ends.size() < SCAN; --j)
            if (t(j) == b && t(j - 1) == a) ends.push_back(j);
        if (const auto it = at.find(key(a, b)); it != at.end())
            for (int i = (int)it->second.size() - 1; i >= 0 && (int)ends.size() < SCAN; --i)
                if (it->second[i] < len - 1) ends.push_back(it->second[i]);
        for (int j : ends)
        {
            int l = 2;
            while (l < LONG && j - l >= 0 && t(j - l) == t(len - 1 - l)) ++l;
            if (l > m.len) { m.len = l; m.next = j + 1; }
        }
        return m;
    }
};

// The longest run ending the sequence s(0 .. len) that occurs in the reference r(0 .. n) with a token after it: the
// guard scope of the sequence penalty (the retry's reply against the call it must not repeat). Brute force: the
// reference is one tool call.
template <class S, class R> CtxMatch longest_in(const S& s, int len, const R& r, int n)
{
    CtxMatch m;
    if (len < 1) return m;
    for (int j = n - 2; j >= 0; --j)
    {
        if (r(j) != s(len - 1)) continue;
        int l = 1;
        while (l < CtxIndex::LONG && j - l >= 0 && len - 1 - l >= 0 && r(j - l) == s(len - 1 - l)) ++l;
        if (l > m.len) { m.len = l; m.next = j + 1; }
    }
    return m;
}

}   // namespace g6rt
