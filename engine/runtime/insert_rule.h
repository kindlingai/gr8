// A request's insert rule (the `ins` field of an S2 request, PROTOCOL.md `insert`): once the reply holds `after` tokens
// and none of them is `unless`, the reply's next token is replaced by `tokens`. The engine applies it in s2_batch.h, from
// the committed tokens only, so every rank applies it at the same token.
#pragma once
#include <vector>

namespace g6rt
{
struct InsertRule
{
    int after = -1;                   // -1: no rule
    int unless = -1;                  // -1: none
    std::vector<int> tokens;
    bool on() const { return after >= 0; }
};
constexpr int INSERT_MAX = 64;        // tokens one rule may insert
}   // namespace g6rt
