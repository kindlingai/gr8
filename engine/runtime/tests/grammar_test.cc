// Request grammars through XGrammar (grammar.h / grammar.cc) on a small byte-level tokenizer this test writes: the
// grammar applies only after its arming token, the masks of a step's rows follow its drafts and leave the matcher where
// it was, a draft the grammar refuses turns the rows after it off, and the committed reply moves the state on.
// build_m4.sh builds and runs it when G6_XGRAMMAR is set.
#include "grammar.h"
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace g6rt;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// ids: 0..13 single characters (byte-level: Ġ is a space, Ċ a newline), 14 = "<think>", 15 = "</think>", 16 = "<eos>"
static const std::vector<std::string> TOK = {"{", "}", "\"", ":", ",", "a", "b", "1", "2", "Ġ", "x", "[", "]", "Ċ"};
constexpr int THINK = 14, ARM = 15, EOS = 16, V = 17;
static int id(char c) {
    const std::string t = c == ' ' ? "Ġ" : c == '\n' ? "Ċ" : std::string(1, c);    // byte-level: U+0120, U+010A
    for (size_t i = 0; i < TOK.size(); ++i) if (TOK[i] == t) return int(i);
    throw std::runtime_error("no token");
}
static bool allows(const uint32_t* m, int v) { return (m[v >> 5] >> (v & 31)) & 1u; }

int main() {
    char dir[] = "/tmp/grammar_testXXXXXX";
    if (!mkdtemp(dir)) return 1;
    {
        std::ofstream t(std::string(dir) + "/tokenizer.json");
        t << R"({"model": {"type": "BPE", "vocab": {)";
        for (size_t i = 0; i < TOK.size(); ++i) {
            std::string k = TOK[i] == "\"" ? "\\\"" : TOK[i];
            t << (i ? ", " : "") << '"' << k << "\": " << i;
        }
        t << R"(}, "merges": []}, "added_tokens": [{"id": 14, "content": "<think>"}, {"id": 15, "content": "</think>"},
               {"id": 16, "content": "<eos>"}], "pre_tokenizer": {"type": "ByteLevel"}, "decoder": {"type": "ByteLevel"}})";
        std::ofstream(std::string(dir) + "/generation_config.json") << R"({"eos_token_id": 16})";
    }
    const std::string err = grammar_init(dir, V);
    if (!err.empty()) { std::printf("grammar_init: %s\n", err.c_str()); return 1; }
    const int W = grammar_words();
    CHECK(grammar_ready() && W == 1);

    GrammarSpec s;
    s.kind = GrammarSpec::SCHEMA;
    s.arm = ARM;
    s.schema = R"({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})";
    Grammar g = grammar_open(s);
    std::vector<int> on(8);
    std::vector<uint32_t> bits(8 * W);

    // before the think close the grammar does not apply; a draft that is the close arms the rows after it
    const int drafts1[] = {ARM, id('{')};
    grammar_masks(*g, drafts1, 2, on.data(), bits.data());
    CHECK(on[0] == 0 && on[1] == 1 && on[2] == 1);
    CHECK(allows(&bits[1 * W], id('{')) && !allows(&bits[1 * W], id('x')) && !allows(&bits[1 * W], id('[')));
    // a 2-space pretty print: a newline after the brace, and "a" is required, so no empty object
    CHECK(allows(&bits[2 * W], id('\n')) && !allows(&bits[2 * W], id('"')) && !allows(&bits[2 * W], id('}')));

    // the reply so far: thinking, then the close
    const int think[] = {THINK, id('x'), ARM};
    grammar_commit(*g, think, 3);
    // row 0 samples after the committed reply; the drafts {\n  "a" are good, then 'x' where only ':' fits turns the rest off
    std::vector<int> drafts2;
    for (char c : std::string("{\n  \"a\"x")) drafts2.push_back(id(c));
    grammar_masks(*g, drafts2.data(), int(drafts2.size()), on.data(), bits.data());
    for (int i = 0; i <= 7; ++i) CHECK(on[i] == 1);
    CHECK(on[8] == 0);
    CHECK(allows(&bits[0], id('{')) && !allows(&bits[0], EOS));
    CHECK(allows(&bits[7 * W], id(':')) && !allows(&bits[7 * W], id('x')));
    // the masks left the matcher where it was
    grammar_masks(*g, nullptr, 0, on.data(), bits.data());
    CHECK(on[0] == 1 && allows(&bits[0], id('{')) && !allows(&bits[0], id('"')));

    // commit a whole valid object, pretty-printed: then only the stop id
    const std::string obj = "{\n  \"a\": 12\n}";
    std::vector<int> ids;
    for (char c : obj) ids.push_back(id(c));
    grammar_commit(*g, ids.data(), int(ids.size()));
    grammar_masks(*g, nullptr, 0, on.data(), bits.data());
    CHECK(on[0] == 1 && allows(&bits[0], EOS) && !allows(&bits[0], id(',')));
    const int eos = EOS;
    grammar_commit(*g, &eos, 1);
    grammar_masks(*g, nullptr, 0, on.data(), bits.data());
    CHECK(on[0] == 0);                    // terminated: nothing left to mask

    // a grammar from the first token (no arming), and json_object: any object, so no bare array
    GrammarSpec j;
    j.kind = GrammarSpec::JSON;
    Grammar h = grammar_open(j);
    grammar_masks(*h, nullptr, 0, on.data(), bits.data());
    CHECK(on[0] == 1 && allows(&bits[0], id('{')) && !allows(&bits[0], id('[')));
    // a committed token the grammar refuses breaks it: the reply goes on unmasked
    const int bad = id('x');
    grammar_commit(*h, &bad, 1);
    grammar_masks(*h, nullptr, 0, on.data(), bits.data());
    CHECK(on[0] == 0);

    std::printf(fails ? "grammar_test: %d FAIL\n" : "grammar_test: PASS\n", fails);
    return fails != 0;
}
