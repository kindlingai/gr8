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

// ids: 0..12 single characters, 13 = "<think>", 14 = "</think>", 15 = "<eos>"
static const std::vector<std::string> TOK = {"{", "}", "\"", ":", ",", "a", "b", "1", "2", "Ġ", "x", "[", "]"};
constexpr int THINK = 13, ARM = 14, EOS = 15, V = 16;
static int id(char c) {
    const std::string t = c == ' ' ? "Ġ" : std::string(1, c);    // byte-level: a space is U+0120
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
        t << R"(}, "merges": []}, "added_tokens": [{"id": 13, "content": "<think>"}, {"id": 14, "content": "</think>"},
               {"id": 15, "content": "<eos>"}], "pre_tokenizer": {"type": "ByteLevel"}, "decoder": {"type": "ByteLevel"}})";
        std::ofstream(std::string(dir) + "/generation_config.json") << R"({"eos_token_id": 15})";
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
    CHECK(allows(&bits[2 * W], id('"')) && !allows(&bits[2 * W], id('}')));   // "a" is required: no empty object

    // the reply so far: thinking, then the close
    const int think[] = {THINK, id('x'), ARM};
    grammar_commit(*g, think, 3);
    // row 0 samples after the committed reply; drafts {"a" are good, then 'x' where only ':' fits turns the rest off
    const int drafts2[] = {id('{'), id('"'), id('a'), id('"'), id('x')};
    grammar_masks(*g, drafts2, 5, on.data(), bits.data());
    CHECK(on[0] == 1 && on[1] == 1 && on[2] == 1 && on[3] == 1 && on[4] == 1 && on[5] == 0);
    CHECK(allows(&bits[0], id('{')) && !allows(&bits[0], EOS));
    CHECK(allows(&bits[4 * W], id(':')) && !allows(&bits[4 * W], id('x')));
    // the masks left the matcher where it was
    grammar_masks(*g, nullptr, 0, on.data(), bits.data());
    CHECK(on[0] == 1 && allows(&bits[0], id('{')) && !allows(&bits[0], id('"')));

    // commit a whole valid object (JSON's usual separators): then only the stop id
    const std::string obj = "{\"a\": 12}";
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
