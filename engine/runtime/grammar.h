// A request's grammar (the `gram` field of an S2 request): from its arming token on, the reply's tokens follow an
// XGrammar grammar (JSON, or a JSON schema). Every rank keeps the same matcher state from the committed tokens
// (s2/sessions.h Sessions::emit), so every rank masks the same tokens of its vocabulary slice and the ranks never send
// a mask to each other. grammar.cc holds the XGrammar side, compiled by the host compiler: nvcc never sees its headers.
// Builds without G6_GRAMMAR have the spec and its parsing only, and refuse a request that carries one.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace g6rt
{
struct GrammarSpec
{
    enum Kind { OFF = 0, JSON = 1, SCHEMA = 2 } kind = OFF;
    bool strict = false;              // SCHEMA: properties and items the schema does not name are refused
    int arm = -1;                     // the token after which the grammar applies (the think close); -1: from the start
    std::string schema;               // SCHEMA: the JSON schema
    bool on() const { return kind != OFF; }
};

struct GrammarState;
struct GrammarDeleter { void operator()(GrammarState*) const; };
using Grammar = std::unique_ptr<GrammarState, GrammarDeleter>;
using GrammarRef = std::shared_ptr<GrammarState>;    // a session's grammar (s2/sessions.h), shared with its dispatch

#if G6_GRAMMAR
// Boot: the tokenizer (DIR/tokenizer.json) and stop ids (DIR/generation_config.json eos_token_id) the grammars compile
// against, for a vocabulary of `vocab` ids. Empty on success, else the reason.
std::string grammar_init(const std::string& dir, int vocab);
bool grammar_ready();
// 32-bit words of one row's mask: bit v of word v / 32 set = token v allowed
int grammar_words();
// A session's grammar, compiled on every rank at bind. Throws std::runtime_error with the compiler's reason.
Grammar grammar_open(const GrammarSpec& spec);
// The reply's committed tokens, in order: every one, from the first (those up to the arming token pass unchecked)
void grammar_commit(GrammarState& g, const int* ids, int n);
// The masks for a step's rows that sample this session's next reply tokens: row 0 samples after the committed reply,
// row i after drafts[0 .. i-1]. on[i] = 1 and out[i * grammar_words() ...] filled where the grammar applies, else 0
// (before arming, after a draft the grammar refuses, or once the grammar has broken). The drafts are rolled back.
void grammar_masks(GrammarState& g, const int* drafts, int nd, int* on, uint32_t* out);
#else
inline void GrammarDeleter::operator()(GrammarState*) const {}
inline std::string grammar_init(const std::string&, int) { return "this engine was built without XGrammar (G6_XGRAMMAR)"; }
inline bool grammar_ready() { return false; }
inline int grammar_words() { return 0; }
inline Grammar grammar_open(const GrammarSpec&) { return nullptr; }
inline void grammar_commit(GrammarState&, const int*, int) {}
inline void grammar_masks(GrammarState&, const int*, int nd, int* on, uint32_t*) { for (int i = 0; i <= nd; ++i) on[i] = 0; }
#endif
}   // namespace g6rt
